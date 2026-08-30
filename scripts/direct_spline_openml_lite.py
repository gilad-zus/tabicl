"""Run the TFM-Retouche-style DirectSpline experiment with only OpenML.

Only the small OpenML client is required.  By default it downloads the public
TabArena v0.1 suite (OpenML suite 457); ``--task-id-file`` instead runs an
explicit frozen OpenML task bank.  In either case it uses each task's published split
``repeat=0, fold=0, sample=0`` as the untouched outer test set, and performs
up to eight guarded, stratified training bags locally.  A classification task
with fewer than eight rows in its rarest training class uses fewer bags so
every fitting fold still contains every class.

The output directory is self-contained: it records the task IDs, split hashes,
frozen configuration draws, per-bag validation/test predictions, model choices,
and a paired-Elo report.  It never imports TabArena, AutoGluon, or Ray.

Examples
--------
Install the only additional dependency in the existing server environment::

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -m pip install openml

First run a one-task smoke test (the normal TabICLv2 baseline plus the
predeclared DirectSpline default)::

    python scripts/direct_spline_openml_lite.py \
      --output-dir results/openml_direct_spline/smoke --task-id 363621 --device cuda

Then run the full default arm, followed by the shared 10-config tuning arm::

    python scripts/direct_spline_openml_lite.py \
      --output-dir results/openml_direct_spline/default --device cuda

    python scripts/direct_spline_openml_lite.py \
      --output-dir results/openml_direct_spline/tuned --device cuda \
      --n-random-configs 10 --tuning-seed 20260813

Do not alter the default configuration after inspecting a smoke-test outer-test
score.  Its purpose is only to verify data access, memory, and artifact layout.
Use a new output directory for each changed command; ``--resume`` accepts only
the exact immutable fingerprint recorded in its manifest.

The sibling ``direct_spline_openml_standard.py`` launcher selects the corrected
full-pipeline arm.  It keeps TabICLv2's ordinary preprocessing and eight-view
ensemble in both its identity and spline paths, so its paired Elo directly
answers whether the spline itself helps.  The default uses every row of each
inner-bag fit partition as context.  ``--context-cap`` is available for a
memory-constrained diagnostic, but is explicitly labelled as capped rather
than public-estimator parity.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This needs to be set before the first CUDA allocation.  It reduces allocator
# fragmentation without changing model computations; an explicit user setting
# still takes precedence.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from tabicl._experiments.direct_spline_openml import (
    CLASSIFIER_CHECKPOINT,
    REGRESSOR_CHECKPOINT,
    STANDARD_TABICL_CONFIG,
    TABARENA_V0PT1_OPENML_SUITE_ID,
    _json_dump,
    _resolve_checkpoint,
    _resolve_device,
    _safe_name,
    effective_inner_bag_count,
    load_tabarena_openml_task,
    run_task_config,
    run_standard_tabarena_baseline,
    summarize_experiment,
    summarize_task_tuning,
    tabarena_v0pt1_task_ids,
)
from tabicl._experiments.direct_spline_protocol import (
    DEFAULT_DIRECT_SPLINE_CONFIG,
    shared_random_direct_spline_configs,
)
from tabicl._experiments.direct_spline_openml_standard import (
    ValidationSplitInfeasibleError,
    run_task_full_context_refit_checkpoint_audit_standard,
    run_task_unconditional_full_context_refit_standard,
    run_task_validation_selected_full_refit_standard,
    run_task_config_standard,
    shared_standard_direct_spline_configs,
    standard_direct_spline_config,
    summarize_full_context_refit_checkpoint_audit_experiment,
    summarize_full_context_refit_checkpoint_audit_task,
    summarize_full_context_refit_experiment,
    summarize_full_context_refit_task,
    summarize_validation_selected_full_refit_experiment,
    summarize_validation_selected_full_refit_task,
)


# Increment this whenever a code change can alter predictions, validation
# selection, or persisted artifact semantics.  Unlike the source hashes, this
# value is deliberately *not* ignored by --allow-compatible-code-resume.
EXPERIMENT_SEMANTICS_VERSION = 9
_RESUME_LAUNCHER_SOURCE = "scripts/direct_spline_openml_lite.py"
_RETOUCHE_EFFICIENCY_RESUME_SOURCE_PATHS = frozenset(
    {
        "scripts/direct_spline_openml_adaptive_retouche.py",
        "scripts/direct_spline_openml_lite.py",
        "src/tabicl/_experiments/direct_spline_openml_standard.py",
    }
)


def parse_args(
    *,
    default_pipeline: str = "lite",
    required_pipeline: str | None = None,
    checkpoint_audit: bool = False,
    validation_selected_refit: bool = False,
    adaptive_phase1: bool = False,
    adaptive_retouche: bool = False,
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    if checkpoint_audit or validation_selected_refit:
        parser.set_defaults(oof_source_dir=None)
    else:
        parser.add_argument(
            "--oof-source-dir",
            type=Path,
            default=None,
            help=(
                "Full-context refit launcher only: optionally read completed standard-pipeline OOF bags from "
                "this earlier run after each unconditional full-context spline has been fitted and frozen. "
                "The old bags are used only for a post-hoc validation/test correlation diagnostic; they never "
                "select a configuration, step count, guard, or prediction."
            ),
        )
    parser.add_argument(
        "--task-id", action="append", type=int,
        help="OpenML task ID. Repeat for a pilot; omit to run the public 51-task suite.",
    )
    parser.add_argument(
        "--task-id-file",
        type=Path,
        default=None,
        help=(
            "A frozen JSON task bank (with selected_task_ids or task_ids), an earlier experiment_manifest.json, "
            "or a text file of positive OpenML task IDs, one per line. Cannot be combined with --task-id."
        ),
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Take the first N selected tasks; for smoke tests only.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help=(
            "Skip tasks with more input columns than this cap. Omit to keep the full suite; "
            "all exclusions are recorded in the progress and final summary."
        ),
    )
    parser.add_argument("--outer-repeat", type=int, default=0)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--outer-sample", type=int, default=0)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--n-random-configs", type=int, default=0)
    parser.add_argument("--tuning-seed", type=int, default=20_260_813)
    parser.add_argument("--ensemble-rounds", type=int, default=None, help="Defaults to twice the number of configs.")
    parser.add_argument("--protocol-seed", type=int, default=0, help="Controls shared bags and support contexts, not HPO draws.")
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument(
        "--pipeline",
        choices=("lite", "standard"),
        default=default_pipeline,
        help=(
            "'lite' reproduces the existing raw one-context headroom path. "
            "'standard' uses the matched normal TabICLv2 preprocessing/ensemble path."
        ),
    )
    parser.add_argument(
        "--context-cap",
        type=int,
        default=0,
        help=(
            "Standard pipeline only: maximum inner-bag fit rows used as context. "
            "0 (the default) means every fit row and enables exact public-estimator identity parity."
        ),
    )
    parser.add_argument(
        "--train-context-rows",
        type=int,
        default=0,
        help=(
            "Standard pipeline only: labelled non-query context rows per adapter training episode. "
            "0 (the default) matches the deployment context scale: every available non-query row "
            "when uncapped, or --context-cap rows in a capped diagnostic."
        ),
    )
    parser.add_argument(
        "--adapter-steps",
        type=int,
        default=None,
        help=(
            "Standard pipeline only: maximum DirectSpline optimisation steps per bag. "
            "Omit to use the configuration default (150 for D)."
        ),
    )
    parser.add_argument(
        "--adapter-patience",
        type=int,
        default=None,
        help=(
            "Standard pipeline only: consecutive validation checks without improvement before early stopping. "
            "Omit to use the configuration default (10 for D)."
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=None,
        help=(
            "Standard pipeline only: evaluate held-out bag validation every N optimisation steps. "
            "Omit to use the configuration default (10 for D)."
        ),
    )
    if checkpoint_audit:
        parser.add_argument(
            "--checkpoint-steps",
            type=_parse_checkpoint_steps,
            default=(0, 25, 50, 100, 200, 300, 500),
            metavar="STEP[,STEP,...]",
            help=(
                "Development-only full-refit audit: sorted DirectSpline checkpoints to freeze and score. "
                "The default is 0,25,50,100,200,300,500. When --adapter-steps is omitted, its value is set "
                "to the largest requested checkpoint."
            ),
        )
    if validation_selected_refit:
        # Although uncapped refits use every row, retain a new protocol seed
        # as well so any deterministic support/preprocessing randomness is not
        # silently inherited from the seed-0 audit.
        parser.set_defaults(protocol_seed=20_260_826)
        parser.add_argument(
            "--validation-fraction",
            type=float,
            default=0.20,
            help=(
                "Validation-selected refit only: fraction of each published outer-training split held out "
                "for the one deterministic checkpoint-selection split."
            ),
        )
        parser.add_argument(
            "--split-seed",
            type=int,
            default=20_260_826,
            help=(
                "Validation-selected refit only: root seed for the deterministic inner train/validation split. "
                "It is intentionally independent of --protocol-seed."
            ),
        )
        parser.add_argument(
            "--adapter-seed",
            type=int,
            default=20_260_826,
            help=(
                "Validation-selected refit only: DirectSpline initialization and episode-sampling seed. "
                "It is intentionally independent of --protocol-seed and differs from the earlier seed-0 audit."
            ),
        )
        parser.add_argument(
            "--selection-checkpoint-interval",
            type=int,
            default=25,
            help=(
                "Validation-selected refit only: evaluate the inner validation split every N adapter steps "
                "(and always at the final step)."
            ),
        )
        parser.add_argument(
            "--cosine-min-lr-ratio",
            type=float,
            default=0.01,
            help=(
                "Validation-selected refit only: final/base learning-rate ratio of the frozen cosine schedule."
            ),
        )
        parser.add_argument(
            "--selection-relative-improvement",
            type=float,
            default=0.005,
            help=(
                "Validation-selected refit only: validation improvement required for a spline checkpoint to "
                "beat step-0 identity."
            ),
        )
        parser.add_argument(
            "--identity-regularization",
            type=float,
            default=0.0 if adaptive_phase1 else 0.01,
            help=(
                "Validation-selected refit only: function-space identity-deformation penalty weight for the "
                "regularized arm; Phase 1 fixes this at zero."
            ),
        )
    elif adaptive_retouche:
        # This protocol is deliberately a new seed from the one-split
        # Phase-1 development run.  The user may still make it explicit on a
        # command line, but every source of optimization randomness is
        # frozen in the manifest.
        parser.set_defaults(protocol_seed=20_260_828)
        parser.add_argument(
            "--adapter-seed",
            type=int,
            default=20_260_828,
            help=(
                "Retouche-style adaptive experiment only: DirectSpline initialization and episode-sampling "
                "seed. It is independent of the deterministic eight-fold split seed."
            ),
        )
        parser.add_argument(
            "--cosine-min-lr-ratio",
            type=float,
            default=0.01,
            help=(
                "Retouche-style adaptive experiment only: final/base learning-rate ratio of the "
                "500-step maximum cosine schedule."
            ),
        )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--skip-standard-baseline",
        action="store_true",
        help="Skip normal eight-estimator TabICLv2 inference; use only for a fast DirectSpline smoke test.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed bag/config artifacts from this exact output directory.")
    parser.add_argument(
        "--allow-compatible-code-resume",
        action="store_true",
        help=(
            "With --resume, reuse artifacts only when every source hash and experimental setting is "
            "identical and merely the Git revision metadata changed. The revision transition is recorded."
        ),
    )
    parser.add_argument(
        "--allow-equivalent-hardware-resume",
        action="store_true",
        help=(
            "With --resume, explicitly allow a new Slurm/CUDA allocation only when all model, "
            "data, numerical-software, and equivalent-GPU properties match. The physical GPU UUID, "
            "CUDA index, and visible-device count may differ and are recorded in resume provenance."
        ),
    )
    parser.add_argument(
        "--allow-retouche-efficiency-resume",
        action="store_true",
        help=(
            "With --resume, explicitly preserve completed Retouche arms from semantics version 8 "
            "(full 500-step checkpoint search) while running unfinished arms under semantics version 9 "
            "(12-check early stopping and no repeated checkpoint identity inference). All baseline, split, "
            "seed, model, guard, and evaluation settings must still match; the migration is recorded."
        ),
    )
    parser.add_argument(
        "--retry-cuda-oom-skips",
        action="store_true",
        help=(
            "With --resume, retry tasks previously recorded as CUDA out-of-memory instead of "
            "treating those hardware-specific skips as permanent."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a frozen-manifest preview without writing an output directory or running fits.",
    )
    args = parser.parse_args()
    if checkpoint_audit and args.adapter_steps is None:
        args.adapter_steps = max(args.checkpoint_steps)
    if validation_selected_refit and args.adapter_steps is None:
        args.adapter_steps = 500
    if adaptive_retouche:
        if args.adapter_steps is None:
            args.adapter_steps = 500
        if args.validation_interval is None:
            args.validation_interval = 25
    if required_pipeline is not None and args.pipeline != required_pipeline:
        parser.error(
            f"this launcher requires --pipeline {required_pipeline}; "
            f"received --pipeline {args.pipeline}"
        )
    return args


def _parse_checkpoint_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint steps must be comma-separated integers") from error
    if not steps:
        raise argparse.ArgumentTypeError("checkpoint steps must not be empty")
    if steps != tuple(sorted(set(steps))):
        raise argparse.ArgumentTypeError("checkpoint steps must be sorted and unique")
    if steps[0] != 0 or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("checkpoint steps must start at 0 and be non-negative")
    return steps


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("tabicl", "torch", "numpy", "pandas", "scikit-learn", "huggingface-hub", "openml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _experiment_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        Path(__file__).resolve(),
        root / "scripts" / "direct_spline_openml_standard.py",
        root / "scripts" / "direct_spline_openml_full_refit.py",
        root / "scripts" / "direct_spline_openml_validation_selected_refit.py",
        root / "scripts" / "direct_spline_openml_adaptive_phase1.py",
        root / "scripts" / "direct_spline_openml_adaptive_retouche.py",
        root / "scripts" / "direct_spline_openml_adaptive_retouche_d_tabarena.py",
        root / "src" / "tabicl" / "__init__.py",
    }
    experiment_dir = root / "src" / "tabicl" / "_experiments"
    paths.update(experiment_dir.glob("direct_spline*.py"))
    paths.add(experiment_dir / "tabarena_direct_spline_protocol.py")
    # The standard path reconstructs the public estimator from these complete
    # packages.  Hashing only the two experiment runners allowed changes in
    # preprocessing, aggregation, inference, or the spline itself to go
    # unnoticed by the immutable run fingerprint.
    for package in ("_sklearn", "_model", "_hyperspline"):
        paths.update((root / "src" / "tabicl" / package).glob("*.py"))
    ordered_paths = sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in ordered_paths
    }


def _precision_configuration() -> dict[str, Any]:
    """Capture process-wide numeric settings which can change GPU results."""
    warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return {
        "torch_default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": None if warn_only is None else bool(warn_only()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _selected_cuda_hardware(device: torch.device) -> dict[str, Any]:
    """Describe the selected CUDA device without relying on unstable reprs."""
    properties = torch.cuda.get_device_properties(device)
    return {
        "index": int(device.index),
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "multi_processor_count": int(properties.multi_processor_count),
        "uuid": None if getattr(properties, "uuid", None) is None else str(properties.uuid),
    }


def _resolve_execution_environment(
    requested_device: str | torch.device,
    *,
    dry_run: bool,
) -> tuple[torch.device | None, dict[str, Any]]:
    """Resolve the runtime device and create a reproducibility record.

    A dry run remains usable on a CPU-only machine even when the requested
    device is CUDA: the manifest records that CUDA could not be resolved, while
    a real run retains the normal hard failure from ``_resolve_device``.
    """
    requested = str(requested_device)
    parsed = torch.device(requested_device)
    cuda_available = bool(torch.cuda.is_available())
    cuda_record: dict[str, Any] = {
        "available": cuda_available,
        "torch_cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "visible_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "selected_hardware": None,
    }
    resolution_status = "resolved"
    resolution_error: str | None = None
    resolved: torch.device | None
    if parsed.type != "cuda":
        resolved = _resolve_device(parsed)
    elif dry_run and not cuda_available:
        resolved = None
        resolution_status = "cuda_unavailable_dry_run"
    else:
        try:
            resolved = parsed if dry_run else _resolve_device(parsed)
            if resolved.index is None:
                resolved = torch.device("cuda", torch.cuda.current_device())
            cuda_record["selected_hardware"] = _selected_cuda_hardware(resolved)
        except (RuntimeError, AssertionError, ValueError) as error:
            if not dry_run:
                raise
            resolved = None
            resolution_status = "cuda_resolution_failed_dry_run"
            resolution_error = f"{type(error).__name__}: {error}"
    record = {
        "requested_device": requested,
        "resolved_device": None if resolved is None else str(resolved),
        "resolution_status": resolution_status,
        "resolution_error": resolution_error,
        "cuda": cuda_record,
        "python_hash_seed": {
            "value": os.environ.get("PYTHONHASHSEED"),
            "fixed_before_process_start": "PYTHONHASHSEED" in os.environ,
        },
        "numeric_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "precision": _precision_configuration(),
    }
    return resolved, record


def _checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}


def _required_checkpoint_kinds(task_ids: list[int]) -> set[str]:
    """Read task metadata only, avoiding an unnecessary regression download for a classification smoke test."""
    try:
        import openml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Install `openml` before starting a non-dry experiment run.") from error
    kinds: set[str] = set()
    for task_id in task_ids:
        task_type = str(openml.tasks.get_task(task_id).task_type).lower()
        kinds.add("regressor" if "regression" in task_type else "classifier")
    return kinds


def _resolve_run_checkpoints(args: argparse.Namespace, task_ids: list[int]) -> dict[str, dict[str, Any] | None]:
    """Resolve and hash the actual weights before creating a runnable manifest."""
    required = _required_checkpoint_kinds(task_ids)
    fingerprints: dict[str, dict[str, Any] | None] = {"classifier": None, "regressor": None}
    if "classifier" in required:
        classifier = _resolve_checkpoint(args.classifier_checkpoint, CLASSIFIER_CHECKPOINT)
        args.classifier_checkpoint = classifier
        fingerprints["classifier"] = _checkpoint_fingerprint(classifier)
    if "regressor" in required:
        regressor = _resolve_checkpoint(args.regressor_checkpoint, REGRESSOR_CHECKPOINT)
        args.regressor_checkpoint = regressor
        fingerprints["regressor"] = _checkpoint_fingerprint(regressor)
    return fingerprints


def _restore_run_checkpoints(
    args: argparse.Namespace,
    previous_manifest: dict[str, Any],
) -> dict[str, dict[str, Any] | None]:
    """Restore and verify checkpoint identities without querying OpenML.

    The initial run freezes which checkpoint kinds are needed.  A resume can
    therefore recover those paths directly from the immutable manifest instead
    of issuing one OpenML metadata request per task before fingerprint checks.
    """
    try:
        previous_run = previous_manifest["immutable_run"]
        stored_fingerprints = previous_run["checkpoint_fingerprints"]
    except (KeyError, TypeError) as error:
        raise ValueError("cannot safely resume: manifest has no checkpoint fingerprints") from error
    if not isinstance(stored_fingerprints, dict):
        raise ValueError("cannot safely resume: manifest checkpoint fingerprints are invalid")

    restored: dict[str, dict[str, Any] | None] = {}
    for kind, default_version in (
        ("classifier", CLASSIFIER_CHECKPOINT),
        ("regressor", REGRESSOR_CHECKPOINT),
    ):
        expected = stored_fingerprints.get(kind)
        argument_name = f"{kind}_checkpoint"
        stored_argument = previous_run.get(f"{kind}_checkpoint_argument")
        requested_argument = getattr(args, argument_name)
        if expected is None:
            # An explicitly supplied but unused checkpoint argument is still
            # part of the immutable command. Restore it so the resume can be
            # invoked without repeating irrelevant arguments.
            if requested_argument is None and stored_argument is not None:
                setattr(args, argument_name, Path(stored_argument))
            restored[kind] = None
            continue
        if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
            raise ValueError(f"cannot safely resume: invalid {kind} checkpoint fingerprint")
        checkpoint = _resolve_checkpoint(
            requested_argument if requested_argument is not None else Path(expected["path"]),
            default_version,
        )
        actual = _checkpoint_fingerprint(checkpoint)
        if actual != expected:
            raise ValueError(
                f"refusing to resume: {kind} checkpoint no longer matches the immutable manifest"
            )
        setattr(args, argument_name, checkpoint)
        restored[kind] = actual
    return restored


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _same_experimental_semantics(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow only a revision-metadata change, never a source-code change."""
    if (
        previous.get("experiment_semantics_version")
        != current.get("experiment_semantics_version")
    ):
        return False
    ignored = {"repository_revision"}
    return (
        {key: value for key, value in previous.items() if key not in ignored}
        == {key: value for key, value in current.items() if key not in ignored}
    )


def _normalise_equivalent_hardware_environment(environment: Any) -> Any:
    """Remove Slurm allocation identities while retaining numerical hardware identity."""

    if not isinstance(environment, dict):
        return environment
    normalised = copy.deepcopy(environment)
    cuda = normalised.get("cuda")
    if not isinstance(cuda, dict):
        return normalised
    # These identify a scheduler allocation, not its numerical execution
    # capability. A new Slurm allocation routinely changes all three.
    cuda.pop("visible_device_count", None)
    hardware = cuda.get("selected_hardware")
    if isinstance(hardware, dict):
        hardware.pop("index", None)
        hardware.pop("uuid", None)
    return normalised


def _same_sources_except_resume_launcher(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Permit only the explicit resume-control change in this launcher file."""

    previous_hashes = previous.get("source_sha256")
    current_hashes = current.get("source_sha256")
    if not isinstance(previous_hashes, dict) or not isinstance(current_hashes, dict):
        return False
    if set(previous_hashes) != set(current_hashes):
        return False
    return all(
        previous_hashes[path] == current_hashes[path]
        for path in previous_hashes
        if path != _RESUME_LAUNCHER_SOURCE
    )


def _difference_paths(previous: Any, current: Any, *, prefix: str) -> list[str]:
    """Return compact paths for unequal leaves in two JSON-like values."""

    if isinstance(previous, dict) and isinstance(current, dict):
        paths: list[str] = []
        for key in sorted(set(previous) | set(current)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                paths.append(child)
            else:
                paths.extend(_difference_paths(previous[key], current[key], prefix=child))
        return paths
    if isinstance(previous, list) and isinstance(current, list):
        paths = []
        if len(previous) != len(current):
            paths.append(f"{prefix}.length")
        for index, (old_item, new_item) in enumerate(zip(previous, current)):
            paths.extend(
                _difference_paths(old_item, new_item, prefix=f"{prefix}[{index}]")
            )
        return paths
    return [] if previous == current else [prefix]


def _equivalent_hardware_resume_mismatches(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Explain every field that prevents an equivalent-hardware resume."""

    mismatches: list[str] = []
    if previous.get("experiment_semantics_version") != current.get(
        "experiment_semantics_version"
    ):
        mismatches.append("experiment_semantics_version")

    previous_hashes = previous.get("source_sha256")
    current_hashes = current.get("source_sha256")
    if not isinstance(previous_hashes, dict) or not isinstance(current_hashes, dict):
        mismatches.append("source_sha256")
    else:
        for path in sorted(set(previous_hashes) | set(current_hashes)):
            if path == _RESUME_LAUNCHER_SOURCE:
                continue
            if previous_hashes.get(path) != current_hashes.get(path):
                mismatches.append(f"source_sha256.{path}")

    ignored = {"repository_revision", "source_sha256", "execution_environment"}
    previous_fields = {key: value for key, value in previous.items() if key not in ignored}
    current_fields = {key: value for key, value in current.items() if key not in ignored}
    mismatches.extend(
        _difference_paths(previous_fields, current_fields, prefix="immutable_run")
    )
    mismatches.extend(
        _difference_paths(
            _normalise_equivalent_hardware_environment(previous.get("execution_environment")),
            _normalise_equivalent_hardware_environment(current.get("execution_environment")),
            prefix="execution_environment",
        )
    )
    return sorted(set(mismatches))


def _same_equivalent_hardware_resume_semantics(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Check a Slurm-safe resume without silently accepting model/code changes.

    This is deliberately narrower than a broad fingerprint override: every
    non-launcher source and every experiment field must match. Only the
    scheduler allocation identity inside the execution environment may change.
    """

    return not _equivalent_hardware_resume_mismatches(previous, current)


def _retouche_efficiency_resume_mismatches(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_equivalent_hardware: bool,
) -> list[str]:
    """Validate the one permitted mixed-budget Retouche resume migration.

    Version-8 completed artifacts searched every checkpoint through step 500.
    Version 9 may stop an unfinished bag after 12 stale validation checks and
    avoids repeatedly evaluating the invariant identity arm. Reusing the old
    artifacts is valid for the headline comparison because the baseline,
    splits, model, seeds, guard, and held-out evaluation are unchanged. This
    helper deliberately permits no other experimental transition.
    """

    mismatches: list[str] = []
    if previous.get("experiment_semantics_version") != 8:
        mismatches.append("experiment_semantics_version.previous_not_8")
    if current.get("experiment_semantics_version") != 9:
        mismatches.append("experiment_semantics_version.current_not_9")
    if previous.get("adaptive_retouche") is not True or current.get("adaptive_retouche") is not True:
        mismatches.append("adaptive_retouche")
    if mismatches:
        return sorted(set(mismatches))

    old = copy.deepcopy(previous)
    new = copy.deepcopy(current)
    old["experiment_semantics_version"] = new["experiment_semantics_version"]
    old["repository_revision"] = new.get("repository_revision")

    old_hashes = old.get("source_sha256")
    new_hashes = new.get("source_sha256")
    if not isinstance(old_hashes, dict) or not isinstance(new_hashes, dict):
        mismatches.append("source_sha256")
    elif set(old_hashes) != set(new_hashes):
        mismatches.append("source_sha256.keys")
    else:
        for path in _RETOUCHE_EFFICIENCY_RESUME_SOURCE_PATHS:
            if path not in old_hashes:
                mismatches.append(f"source_sha256.{path}")
            else:
                old_hashes[path] = new_hashes[path]

    old_configs = old.get("configs")
    new_configs = new.get("configs")
    if not isinstance(old_configs, list) or not isinstance(new_configs, list) or len(old_configs) != len(new_configs):
        mismatches.append("configs")
    else:
        for index, (old_config, new_config) in enumerate(zip(old_configs, new_configs, strict=True)):
            if not isinstance(old_config, dict) or not isinstance(new_config, dict):
                mismatches.append(f"configs[{index}]")
                continue
            if old_config.get("adapter_patience") is not None or new_config.get("adapter_patience") != 12:
                mismatches.append(f"configs[{index}].adapter_patience")
                continue
            old_config["adapter_patience"] = 12

    for field in ("adaptive_retouche_settings", "adaptive_retouche_contract"):
        old_record = old.get(field)
        new_record = new.get(field)
        expected_early_stop = {"stale_validation_checks": 12}
        if not isinstance(old_record, dict) or not isinstance(new_record, dict):
            mismatches.append(field)
        elif old_record.get("early_stopping") is not None or new_record.get("early_stopping") != expected_early_stop:
            mismatches.append(f"{field}.early_stopping")
        else:
            old_record["early_stopping"] = expected_early_stop

    if allow_equivalent_hardware:
        old["execution_environment"] = _normalise_equivalent_hardware_environment(
            old.get("execution_environment")
        )
        new["execution_environment"] = _normalise_equivalent_hardware_environment(
            new.get("execution_environment")
        )

    mismatches.extend(_difference_paths(old, new, prefix="immutable_run"))
    return sorted(set(mismatches))


def _slurm_allocation_provenance() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")
    }


def _event_reporter(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def report(event: dict[str, Any]) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        event_name = event["event"]
        task_prefix = f"task={event.get('task_id', '?')}"
        if event_name == "bag_started":
            pipeline = " standard" if event.get("pipeline") == "standard_ensemble" else ""
            print(
                f"[{task_prefix} bag={event['bag']}{pipeline}] fitting: train={event['fit_rows']} "
                f"validation={event['validation_rows']} support={event['support_rows']}",
                flush=True,
            )
        elif event_name == "adapter_validation":
            print(
                f"[{task_prefix} bag={event['bag']} step={event['step']}] "
                f"validation={event['validation_error']:.6g} best={event['best_validation_error']:.6g} "
                f"stale={event['stale_validations']} elapsed={event['elapsed_seconds']:.1f}s",
                flush=True,
            )
        elif event_name == "full_refit_started":
            print(
                f"[{task_prefix} full-refit config={event['config_label']}] "
                f"fitting on every outer-training row: steps={event['refit_steps']} "
                "unconditional; no OOF selection or guard",
                flush=True,
            )
        elif event_name == "full_refit_step":
            print(
                f"[{task_prefix} full-refit step={event['step']}] "
                f"objective={event['objective']:.6g} elapsed={event['elapsed_seconds']:.1f}s",
                flush=True,
            )
        elif event_name == "full_refit_completed":
            print(
                f"[{task_prefix} full-refit] complete: "
                "unconditional "
                f"steps={event['executed_steps']}/{event['refit_steps']} "
                f"time={event['train_seconds']:.1f}s peak={event['peak_allocated_gib']:.2f}GiB",
                flush=True,
            )
        elif event_name == "full_refit_checkpoint_audit_started":
            print(
                f"[{task_prefix} checkpoint-audit config={event['config_label']}] "
                f"freezing development-only checkpoints={event['checkpoint_steps']}",
                flush=True,
            )
        elif event_name == "full_refit_checkpoint_frozen":
            objective = event.get("objective")
            objective_text = "identity" if objective is None else f"objective={objective:.6g}"
            print(
                f"[{task_prefix} checkpoint-audit step={event['step']}] {objective_text} "
                f"deformation={event['mean_grid_deformation']:.3g} "
                f"elapsed={event['elapsed_seconds']:.1f}s",
                flush=True,
            )
        elif event_name == "full_refit_checkpoint_audit_completed":
            print(
                f"[{task_prefix} checkpoint-audit] complete: "
                f"frozen={event['checkpoint_steps_frozen']} steps={event['executed_steps']} "
                f"time={event['train_seconds']:.1f}s peak={event['peak_allocated_gib']:.2f}GiB",
                flush=True,
            )
        elif event_name == "validation_selected_refit_started":
            print(
                f"[{task_prefix} validation-refit config={event['config_label']}] "
                f"selecting on inner train={event['inner_train_rows']}, validation={event['validation_rows']}; "
                "outer test remains withheld",
                flush=True,
            )
        elif event_name == "validation_selection_checkpoint":
            print(
                f"[{task_prefix} validation-refit step={event['step']}] "
                f"validation={event['validation_error']:.6g}/identity={event['identity_validation_error']:.6g} "
                f"deformation={event['mean_grid_deformation']:.3g} "
                f"elapsed={event['elapsed_seconds']:.1f}s",
                flush=True,
            )
        elif event_name == "validation_selected_refit_completed":
            selected = "spline" if event["selected_use_adapted"] else "identity"
            print(
                f"[{task_prefix} validation-refit config={event['config_label']}] complete: "
                f"selected={selected}@{event['selected_step']} "
                f"selection={event['selection_seconds']:.1f}s refit={event['refit_seconds']:.1f}s",
                flush=True,
            )
        elif event_name == "validation_selected_refit_rebuilding_incomplete":
            print(
                f"[{task_prefix} validation-refit config={event['config_label']}] "
                "rebuilding an incomplete or corrupt resume artifact",
                flush=True,
            )
        elif event_name == "bag_completed":
            parity = ""
            if "identity_parity_max_abs_validation" in event:
                parity = f" parity≤{event['identity_parity_max_abs_validation']:.2g}"
            print(
                f"[{task_prefix} bag={event['bag']}] complete: guard={'adapted' if event['guard_selected_adapted'] else 'identity'} "
                f"validation={event['adapted_error']:.6g}/{event['identity_error']:.6g} "
                f"steps={event['adapter_steps_executed']} time={event['train_seconds']:.1f}s "
                f"peak={event['peak_allocated_gib']:.2f}GiB{parity}",
                flush=True,
            )
        elif event_name == "identity_view_parity_failed":
            print(
                f"[{task_prefix} bag={event['bag']}] {event['split']} identity input-view parity failed "
                f"on {event['query_rows']} query rows; diagnostics saved to {path}",
                flush=True,
            )
        elif event_name == "config_started":
            detail = (
                f"using {event['effective_bags']}/{event['requested_bags']} stratified bags"
                if event["effective_bags"] != event["requested_bags"]
                else f"using {event['effective_bags']} bags"
            )
            print(f"[{task_prefix} config={event['config_label']}] {detail}", flush=True)
        elif event_name in {"standard_baseline_started", "standard_baseline_completed"}:
            suffix = "starting normal 8-estimator TabICLv2 baseline"
            if event_name.endswith("completed"):
                suffix = (
                    "complete: outer-test score withheld until validation-only selection is complete "
                    f"time={event['elapsed_seconds']:.1f}s"
                )
            print(f"[{task_prefix}] standard baseline: {suffix}", flush=True)
        elif event_name == "standard_baseline_skipped":
            print(
                f"[{task_prefix} dataset={event['dataset_name']}] standard baseline skipped after CUDA OOM; "
                "continuing with the paired DirectSpline task",
                flush=True,
            )
        elif event_name == "task_skipped":
            if event.get("reason") == "cuda_out_of_memory":
                print(
                    f"[{task_prefix} dataset={event['dataset_name']}] skipped after CUDA OOM "
                    f"during {event['stage']}; continuing with the next task",
                    flush=True,
                )
            elif event.get("reason") == "validation_split_infeasible":
                print(
                    f"[{task_prefix} dataset={event['dataset_name']}] skipped: no valid deterministic "
                    f"inner validation split ({event['error']})",
                    flush=True,
                )
            else:
                print(
                    f"[{task_prefix} dataset={event['dataset_name']}] skipped: "
                    f"features={event['n_features']} exceeds max_features={event['max_features']}",
                    flush=True,
                )
        elif event_name.endswith("reused"):
            print(f"[{task_prefix}] reused {event_name.replace('_', ' ')}", flush=True)

    return report


def _interrupted_artifact_path(path: Path) -> Path:
    """Choose a non-destructive quarantine name beside an interrupted artifact."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for suffix in range(10_000):
        disambiguator = "" if suffix == 0 else f"-{suffix}"
        candidate = path.with_name(f"{path.name}.interrupted-{timestamp}{disambiguator}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not choose a quarantine path for {path}")


def _repair_interrupted_config_summaries(output_dir: Path) -> list[dict[str, Any]]:
    """Quarantine malformed config summaries so --resume can rebuild them.

    Completed bag files are independent checkpoints.  ``config_summary.json``
    is only the small final aggregation written after every bag finishes.  If
    an interruption leaves that write malformed, retaining the file makes the
    normal resume shortcut fail.  Move only malformed summaries aside (never
    delete them); resume then reuses valid bags and regenerates the aggregate.
    """

    raw_dir = output_dir / "raw"
    if not raw_dir.is_dir():
        return []
    recovered: list[dict[str, Any]] = []
    for summary_path in sorted(raw_dir.rglob("config_summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            quarantine_path = _interrupted_artifact_path(summary_path)
            summary_path.replace(quarantine_path)
            recovered.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "artifact": str(summary_path),
                    "quarantine": str(quarantine_path),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    if recovered:
        recovery_log = output_dir / "artifact_recoveries.jsonl"
        with recovery_log.open("a", encoding="utf-8") as handle:
            for item in recovered:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"Recovered {len(recovered)} interrupted config summary artifact(s); "
            "resume will rebuild them from the completed bag checkpoints.",
            flush=True,
        )
    return recovered


def _configs(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    if args.pipeline == "standard":
        context_cap = None if args.context_cap == 0 else args.context_cap
        train_context_rows = None if args.train_context_rows == 0 else args.train_context_rows
        adapter_steps = getattr(args, "adapter_steps", None)
        adapter_patience = getattr(args, "adapter_patience", None)
        validation_interval = getattr(args, "validation_interval", None)
        default = standard_direct_spline_config(
            context_cap=context_cap,
            train_context_rows=train_context_rows,
            adapter_steps=adapter_steps,
            adapter_patience=adapter_patience,
            validation_interval=validation_interval,
        )
        random = shared_standard_direct_spline_configs(
            args.n_random_configs,
            seed=args.tuning_seed,
            context_cap=context_cap,
            adapter_steps=adapter_steps,
            adapter_patience=adapter_patience,
            validation_interval=validation_interval,
        )
        if train_context_rows is not None:
            for config in random:
                config["train_context_rows"] = train_context_rows
        return ["D", *(f"R{index + 1}" for index in range(len(random)))], [default, *random]
    default = dict(DEFAULT_DIRECT_SPLINE_CONFIG)
    random = shared_random_direct_spline_configs(args.n_random_configs, seed=args.tuning_seed)
    return ["D", *(f"R{index + 1}" for index in range(len(random)))], [default, *random]


def _validation_selected_refit_configs(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    """Freeze the scheduler-only and identity-regularised arms.

    The two arms deliberately share every initialization, split, episode, and
    scheduling setting.  The explicit function-space penalty is their only
    optimisation difference.
    """

    base = standard_direct_spline_config(
        context_cap=None,
        train_context_rows=None if args.train_context_rows == 0 else args.train_context_rows,
        adapter_steps=args.adapter_steps,
        validation_interval=args.selection_checkpoint_interval,
    )
    base.update(
        {
            "random_state": int(args.adapter_seed),
            "adapter_patience": None,
            "selection_checkpoint_interval": int(args.selection_checkpoint_interval),
            "cosine_schedule_steps": int(args.adapter_steps),
            "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
            "selection_relative_improvement": float(args.selection_relative_improvement),
            "guard_relative_improvement": float(args.selection_relative_improvement),
            "identity_regularization": 0.0,
        }
    )
    regularized = dict(base)
    regularized["identity_regularization"] = float(args.identity_regularization)
    return ["cosine", "cosine_identity_regularized"], [base, regularized]


def _adaptive_phase1_validation_selected_refit_configs(
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Freeze the three Phase-1 architecture arms.

    Every arm shares the exact same split, seed, cosine schedule, checkpoint
    interval, and selection threshold.  The only predeclared difference is
    the numerical adapter architecture.  The old identity penalty is omitted
    deliberately: this experiment tests heterogeneous basis capacity and
    conditional interactions, not another regularisation setting.
    """

    labels, configs = _validation_selected_refit_configs(args)
    baseline = dict(configs[0])
    baseline["adapter_architecture"] = "fixed_cubic"
    # Keep the in-memory value in its canonical JSON form.  Tuples serialize
    # as lists in the immutable manifest; reconstructing tuples on resume made
    # an unchanged Phase-1 run fail direct manifest equality checks.
    expert_specs = [[1, 4], [2, 8], [3, 20]]
    adaptive = dict(baseline)
    adaptive.update(
        {
            "adapter_architecture": "adaptive_columns",
            "adaptive_expert_specs": expert_specs,
            "adaptive_routing_temperature": 1.0,
        }
    )
    conditional = dict(adaptive)
    conditional.update(
        {
            "adapter_architecture": "conditional_adaptive_columns",
            "conditional_interaction_rank": 4,
            "conditional_interaction_bound": 0.25,
        }
    )
    return ["fixed_cubic20", "adaptive_columns", "conditional_adaptive_columns"], [
        baseline,
        adaptive,
        conditional,
    ]


def _adaptive_retouche_configs(
    args: argparse.Namespace, *, d_only: bool = False
) -> tuple[list[str], list[dict[str, Any]]]:
    """Freeze the final three-arm, preserved-fold DirectSpline bank.

    Each arm is trained independently inside every normal TabICLv2 bag.  The
    fold's best trained checkpoint is retained for the identity guard and for
    its outer-test prediction: there is deliberately no all-row refit.  The
    only arm difference is the spline basis/conditional architecture; the
    optimization schedule and all regularization controls are shared.
    """

    base = standard_direct_spline_config(
        context_cap=None,
        train_context_rows=None,
        adapter_steps=int(args.adapter_steps),
        validation_interval=int(args.validation_interval),
    )
    base.update(
        {
            "random_state": int(args.adapter_seed),
            # A conservative 12-check patience (300 stale optimisation steps)
            # is fixed from Phase-1 validation trajectories before this run.
            # The validation-best trained member is still preserved directly;
            # no all-row refit follows early stopping.
            "adapter_patience": 12,
            "cosine_schedule_steps": int(args.adapter_steps),
            "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
            "guard_relative_improvement": 0.005,
            # Phase 1's architecture comparison intentionally had no
            # function-space identity penalty. Keep that fixed here so this
            # final protocol changes selection/deployment, not a second axis.
            "identity_regularization": 0.0,
            "adapter_architecture": "fixed_cubic",
        }
    )
    adaptive = dict(base)
    adaptive.update(
        {
            "adapter_architecture": "adaptive_columns",
            "adaptive_expert_specs": [[1, 4], [2, 8], [3, 20]],
            "adaptive_routing_temperature": 1.0,
        }
    )
    conditional = dict(adaptive)
    conditional.update(
        {
            "adapter_architecture": "conditional_adaptive_columns",
            "conditional_interaction_rank": 4,
            "conditional_interaction_bound": 0.25,
        }
    )
    if d_only:
        # This is a deliberately separate final evaluation: it tests the
        # strongest fixed arm on the complete benchmark without spending the
        # additional compute on architecture selection or ensembling.
        return ["D"], [base]
    # D denotes the fixed cubic-20 default. T and T+E are created later from
    # guarded OOF predictions of all three predeclared configurations.
    return ["D", "adaptive_columns", "conditional_adaptive_columns"], [base, adaptive, conditional]


def _validate(
    args: argparse.Namespace,
    *,
    full_context_refit: bool = False,
    checkpoint_audit: bool = False,
    validation_selected_refit: bool = False,
    adaptive_phase1: bool = False,
    adaptive_retouche: bool = False,
) -> None:
    if args.bags < 2:
        raise ValueError("--bags must be at least two")
    if args.n_random_configs < 0 or args.bootstrap_rounds <= 0:
        raise ValueError("random-config and bootstrap counts must be non-negative/positive")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")
    if getattr(args, "task_id", None) and getattr(args, "task_id_file", None) is not None:
        raise ValueError("--task-id and --task-id-file cannot be combined")
    if args.max_features is not None and args.max_features <= 0:
        raise ValueError("--max-features must be positive")
    if args.context_cap < 0:
        raise ValueError("--context-cap must be zero (all rows) or positive")
    if args.train_context_rows < 0:
        raise ValueError("--train-context-rows must be zero (all available rows) or positive")
    standard_schedule_options = {
        "--adapter-steps": getattr(args, "adapter_steps", None),
        "--adapter-patience": getattr(args, "adapter_patience", None),
        "--validation-interval": getattr(args, "validation_interval", None),
    }
    for option, value in standard_schedule_options.items():
        if value is not None and value <= 0:
            raise ValueError(f"{option} must be positive when provided")
    if getattr(args, "pipeline", "lite") != "standard" and any(
        value is not None for value in standard_schedule_options.values()
    ):
        raise ValueError("--adapter-steps, --adapter-patience, and --validation-interval require --pipeline standard")
    if min(args.outer_repeat, args.outer_fold, args.outer_sample) < 0:
        raise ValueError("outer repeat/fold/sample values must be non-negative")
    if args.allow_compatible_code_resume and not args.resume:
        raise ValueError("--allow-compatible-code-resume requires --resume")
    if getattr(args, "allow_equivalent_hardware_resume", False) and not args.resume:
        raise ValueError("--allow-equivalent-hardware-resume requires --resume")
    if getattr(args, "allow_retouche_efficiency_resume", False) and not args.resume:
        raise ValueError("--allow-retouche-efficiency-resume requires --resume")
    if getattr(args, "allow_retouche_efficiency_resume", False) and not adaptive_retouche:
        raise ValueError("--allow-retouche-efficiency-resume is only valid for the Retouche-style adaptive experiment")
    if args.retry_cuda_oom_skips and not args.resume:
        raise ValueError("--retry-cuda-oom-skips requires --resume")
    if getattr(args, "dry_run", False) and args.resume:
        raise ValueError("--dry-run cannot be combined with --resume; it is a non-persistent preview")
    if checkpoint_audit and validation_selected_refit:
        raise ValueError("checkpoint audit and validation-selected refit are mutually exclusive")
    if adaptive_retouche and (checkpoint_audit or validation_selected_refit or adaptive_phase1 or full_context_refit):
        raise ValueError(
            "the Retouche-style adaptive experiment is incompatible with refit, checkpoint-audit, and Phase-1 modes"
        )
    if adaptive_retouche:
        if args.pipeline != "standard":
            raise ValueError("the Retouche-style adaptive experiment requires --pipeline standard")
        if args.n_random_configs != 0:
            raise ValueError(
                "the Retouche-style adaptive experiment has exactly three predeclared arms; use --n-random-configs 0"
            )
        if args.context_cap != 0 or args.train_context_rows != 0:
            raise ValueError(
                "the Retouche-style adaptive experiment requires all inner-bag fit rows as deployment and training context"
            )
        if args.adapter_steps != 500:
            raise ValueError("the Retouche-style adaptive experiment fixes --adapter-steps to 500")
        if args.validation_interval != 25:
            raise ValueError("the Retouche-style adaptive experiment fixes --validation-interval to 25")
        if args.adapter_patience is not None:
            raise ValueError(
                "the Retouche-style adaptive experiment fixes a 12-check validation early-stop patience; "
                "do not pass --adapter-patience"
            )
        if not 0.0 < args.cosine_min_lr_ratio <= 1.0:
            raise ValueError("--cosine-min-lr-ratio must lie in (0, 1]")
        if args.oof_source_dir is not None:
            raise ValueError("--oof-source-dir is not used by the Retouche-style adaptive experiment")
    if validation_selected_refit:
        if args.pipeline != "standard":
            raise ValueError("the validation-selected refit experiment requires --pipeline standard")
        if args.n_random_configs != 0:
            raise ValueError(
                "the validation-selected refit experiment has only predeclared arms; "
                "use --n-random-configs 0"
            )
        if adaptive_phase1 and float(args.identity_regularization) != 0.0:
            raise ValueError(
                "the adaptive Phase-1 experiment fixes identity_regularization=0; "
                "do not pass --identity-regularization"
            )
        if args.context_cap != 0:
            raise ValueError(
                "the validation-selected refit experiment requires --context-cap 0 so each identity control "
                "uses the exact full available context"
            )
        if args.adapter_patience is not None:
            raise ValueError(
                "the validation-selected refit experiment runs through the fixed horizon; "
                "do not pass --adapter-patience"
            )
        if args.adapter_steps is None or args.adapter_steps <= 0:
            raise ValueError("the validation-selected refit experiment requires a positive --adapter-steps")
        if not 0.0 < args.validation_fraction < 0.5:
            raise ValueError("--validation-fraction must lie in (0, 0.5)")
        if args.selection_checkpoint_interval <= 0:
            raise ValueError("--selection-checkpoint-interval must be positive")
        if not 0.0 < args.cosine_min_lr_ratio <= 1.0:
            raise ValueError("--cosine-min-lr-ratio must lie in (0, 1]")
        if not 0.0 <= args.selection_relative_improvement < 1.0:
            raise ValueError("--selection-relative-improvement must lie in [0, 1)")
        if args.identity_regularization < 0.0:
            raise ValueError("--identity-regularization must be non-negative")
        if args.oof_source_dir is not None:
            raise ValueError("--oof-source-dir is not used by the validation-selected refit experiment")
    elif full_context_refit:
        if args.pipeline != "standard":
            raise ValueError("the full-context refit experiment requires --pipeline standard")
        if args.n_random_configs != 0:
            raise ValueError(
                "the unconditional full-context refit experiment requires --n-random-configs 0; "
                "its sole predeclared configuration must not be selected using old OOF results"
            )
        if args.context_cap != 0:
            raise ValueError(
                "the full-context refit experiment requires --context-cap 0 so its deployment context exactly "
                "matches ordinary full-outer-training TabICLv2"
            )
        if checkpoint_audit:
            steps = getattr(args, "checkpoint_steps", None)
            if not isinstance(steps, tuple) or not steps:
                raise ValueError("the checkpoint audit requires a non-empty --checkpoint-steps schedule")
            if steps != tuple(sorted(set(steps))) or steps[0] != 0 or any(step < 0 for step in steps):
                raise ValueError("--checkpoint-steps must be sorted, unique, non-negative, and start at zero")
            if args.adapter_steps is None or int(args.adapter_steps) < steps[-1]:
                raise ValueError("--adapter-steps must be at least the largest --checkpoint-steps value")
            if args.oof_source_dir is not None:
                raise ValueError("--oof-source-dir is not used by the development-only checkpoint audit")
    elif args.oof_source_dir is not None:
        raise ValueError("--oof-source-dir is available only in the full-context refit launcher")


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    """Recognize both modern typed and older text-only PyTorch CUDA OOMs."""
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "CUDA out of memory" in str(error)
    )


def _write_run_progress(
    output_dir: Path,
    task_summaries: list[dict[str, Any]],
    skipped_tasks: list[dict[str, Any]],
) -> None:
    _json_dump(
        output_dir / "run_progress.json",
        {
            "completed_task_ids": [item["task_id"] for item in task_summaries],
            "skipped_tasks": skipped_tasks,
        },
    )


def _write_all_skipped_summary(
    output_dir: Path,
    *,
    skipped_tasks: list[dict[str, Any]],
    task_eligibility: dict[str, Any],
) -> dict[str, Any]:
    """Write a valid terminal report when resource/eligibility skips leave no pairs."""
    summary = {
        "status": "no_completed_tasks",
        "n_tasks": 0,
        "n_skipped_tasks": len(skipped_tasks),
        "task_eligibility": task_eligibility,
        "skipped_tasks": skipped_tasks,
        "paired_elo_note": "No paired Elo can be computed because no task completed both arms.",
        "paired_results": {},
        "distinct_paired_result_keys": [],
        "paired_result_aliases": {},
        "standard_tabarena": {
            "available": False,
            "n_tasks_available": 0,
            "n_tasks_missing": 0,
        },
        "end_to_end_vs_standard_tabarena": {
            "available": False,
            "n_tasks_available": 0,
            "results": {},
        },
        "task_results_csv": None,
    }
    _json_dump(output_dir / "summary.json", summary)
    return summary


def _persisted_task_skips(output_dir: Path) -> dict[int, dict[str, Any]]:
    """Recover every recorded skip so an interrupted resume cannot erase it."""
    path = output_dir / "run_progress.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_skips = payload.get("skipped_tasks", [])
    if not isinstance(raw_skips, list):
        raise ValueError(f"{path} has an invalid skipped_tasks record")
    recovered: dict[int, dict[str, Any]] = {}
    for item in raw_skips:
        if isinstance(item, dict) and "task_id" in item:
            recovered[int(item["task_id"])] = item
    return recovered


def _persisted_cuda_oom_skips(output_dir: Path) -> dict[int, dict[str, Any]]:
    """Recover hardware-specific skips so resume does not repeat a known OOM."""
    return {
        task_id: item
        for task_id, item in _persisted_task_skips(output_dir).items()
        if item.get("reason") == "cuda_out_of_memory"
    }


def _frozen_manifest_task_ids(manifest: dict[str, Any], manifest_path: Path) -> list[int]:
    """Read the canonical task order from an existing immutable manifest."""
    try:
        raw_task_ids = manifest["immutable_run"]["data_source"]["task_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"cannot safely resume: {manifest_path} has no immutable OpenML task list"
        ) from error
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        raise ValueError(f"cannot safely resume: {manifest_path} has an invalid immutable OpenML task list")
    try:
        task_ids = [int(task_id) for task_id in raw_task_ids]
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot safely resume: {manifest_path} has a non-integer OpenML task ID") from error
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"cannot safely resume: {manifest_path} has duplicate OpenML task IDs")
    return task_ids


def _task_ids_from_file(path: Path) -> list[int]:
    """Read an explicit, reviewable task bank without silently accepting malformed input."""

    if not path.is_file():
        raise FileNotFoundError(f"--task-id-file does not exist or is not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"--task-id-file must be UTF-8 text: {path}") from error
    if not text.strip():
        raise ValueError(f"--task-id-file is empty: {path}")

    raw_ids: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Plain text is intentionally narrow: one positive integer per line.
        # This makes a pasted command or a CSV header fail rather than changing
        # the benchmark membership silently.
        raw_ids = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    else:
        if isinstance(payload, list):
            raw_ids = payload
        elif isinstance(payload, dict):
            if payload.get("is_complete") is False:
                raise ValueError(
                    f"--task-id-file is an incomplete task bank; rebuild it with the requested number of eligible tasks: {path}"
                )
            raw_ids = payload.get("selected_task_ids", payload.get("task_ids"))
            if raw_ids is None:
                try:
                    raw_ids = payload["immutable_run"]["data_source"]["task_ids"]
                except (KeyError, TypeError):
                    raw_ids = None
        else:
            raw_ids = None
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(
            f"--task-id-file must be a JSON list, a JSON object with selected_task_ids/task_ids, an "
            f"experiment_manifest.json, or one task ID per line: {path}"
        )
    try:
        task_ids = [int(task_id) for task_id in raw_ids]
    except (TypeError, ValueError) as error:
        raise ValueError(f"--task-id-file contains a non-integer OpenML task ID: {path}") from error
    if any(task_id <= 0 for task_id in task_ids):
        raise ValueError(f"--task-id-file task IDs must be positive: {path}")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"--task-id-file has duplicate OpenML task IDs: {path}")
    return task_ids


def _task_file_provenance(path: Path) -> dict[str, Any]:
    """Record the reviewed task-bank bytes outside the prediction fingerprint."""

    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_oof_source_manifest(source_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Load an earlier completed standard DirectSpline run read-only."""

    resolved_dir = source_dir.resolve()
    manifest_path = resolved_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "--oof-source-dir must contain a completed DirectSpline experiment_manifest.json: "
            f"{manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read OOF source manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("immutable_run"), dict):
        raise ValueError(f"OOF source manifest is missing immutable_run: {manifest_path}")
    if not isinstance(payload.get("run_fingerprint_sha256"), str):
        raise ValueError(f"OOF source manifest is missing its run fingerprint: {manifest_path}")
    return resolved_dir, payload


def _source_manifest_task_ids(source_manifest: dict[str, Any], source_dir: Path) -> list[int]:
    try:
        raw_ids = source_manifest["immutable_run"]["data_source"]["task_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"OOF source has no frozen task IDs: {source_dir}") from error
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"OOF source task list is empty or malformed: {source_dir}")
    try:
        task_ids = [int(task_id) for task_id in raw_ids]
    except (TypeError, ValueError) as error:
        raise ValueError(f"OOF source task list contains a non-integer ID: {source_dir}") from error
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"OOF source task list has duplicate IDs: {source_dir}")
    return task_ids


def _ordered_subsequence(selected: list[int], source: list[int]) -> bool:
    """Return whether a pilot selection preserves the frozen source order."""

    cursor = 0
    for task_id in source:
        if cursor < len(selected) and selected[cursor] == task_id:
            cursor += 1
    return cursor == len(selected)


def _oof_source_provenance(source_dir: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Store immutable read-only provenance for artifacts owned by another run."""

    manifest_path = source_dir / "experiment_manifest.json"
    return {
        "path": str(source_dir),
        "manifest_sha256": _sha256_file(manifest_path),
        "run_fingerprint_sha256": source_manifest["run_fingerprint_sha256"],
        "task_ids": _source_manifest_task_ids(source_manifest, source_dir),
    }


def _same_checkpoint_identities(left: Any, right: Any) -> bool:
    """Compare immutable checkpoint bytes without pinning a cache location."""

    if not isinstance(left, dict) or not isinstance(right, dict):
        return left == right
    if set(left) != set(right):
        return False
    for kind in left:
        left_item = left[kind]
        right_item = right[kind]
        if left_item is None or right_item is None:
            if left_item is not None or right_item is not None:
                return False
            continue
        if not isinstance(left_item, dict) or not isinstance(right_item, dict):
            return False
        if (
            left_item.get("bytes") != right_item.get("bytes")
            or left_item.get("sha256") != right_item.get("sha256")
        ):
            return False
    return True


def _validate_oof_source_compatibility(
    *,
    source_dir: Path,
    source_manifest: dict[str, Any],
    current_immutable_run: dict[str, Any],
    selected_task_ids: list[int],
) -> None:
    """Reject a source whose OOF evidence cannot answer this refit question.

    Source-code hashes intentionally differ: this launcher adds a *new* final
    all-row refit after the already completed OOF bags.  Everything that could
    change the OOF split, adapter optimisation, or ordinary baseline must
    nevertheless match exactly.
    """

    source = source_manifest["immutable_run"]
    source_ids = _source_manifest_task_ids(source_manifest, source_dir)
    differences: list[str] = []
    if source.get("pipeline") != "standard":
        differences.append("pipeline is not standard")
    if source.get("full_context_refit") is True:
        differences.append("source is already a full-context refit rather than an OOF bag source")
    if not _ordered_subsequence(selected_task_ids, source_ids):
        differences.append("selected task IDs are not an ordered subset of the source task bank")
    source_data = source.get("data_source")
    current_data = current_immutable_run.get("data_source")
    if not isinstance(source_data, dict) or not isinstance(current_data, dict):
        differences.append("missing frozen OpenML data-source metadata")
    elif source_data.get("outer_split") != current_data.get("outer_split"):
        differences.append("outer OpenML split differs")
    comparisons = {
        "inner_bags": "inner bag count",
        "inner_bag_policy": "inner bag policy",
        "protocol_seed": "protocol seed",
        "config_labels": "configuration labels",
        "configs": "DirectSpline configuration",
        "task_eligibility": "task eligibility",
        "standard_tabarena_baseline": "normal TabICLv2 baseline configuration",
    }
    for key, label in comparisons.items():
        if source.get(key) != current_immutable_run.get(key):
            differences.append(f"{label} differs")
    # A dry run deliberately does not resolve checkpoints. A real refit must
    # prove the weights are identical to the OOF source before it starts.
    if (
        current_immutable_run.get("checkpoint_fingerprints") is not None
        and not _same_checkpoint_identities(
            source.get("checkpoint_fingerprints"),
            current_immutable_run.get("checkpoint_fingerprints"),
        )
    ):
        differences.append("checkpoint fingerprint differs")
    source_standard = source.get("standard_pipeline")
    current_standard = current_immutable_run.get("standard_pipeline")
    if not isinstance(source_standard, dict) or not isinstance(current_standard, dict):
        differences.append("missing standard-pipeline metadata")
    else:
        for key in (
            "context_cap",
            "requested_train_context_rows",
            "resolved_train_context_rows",
            "normal_tabarena_config",
        ):
            if source_standard.get(key) != current_standard.get(key):
                differences.append(f"standard-pipeline {key} differs")
    if differences:
        raise ValueError(
            "refusing to reuse OOF artifacts from "
            f"{source_dir}: " + "; ".join(differences)
        )


def _validate_oof_source_task_artifacts(
    *,
    source_dir: Path,
    source_manifest: dict[str, Any],
    task: Any,
    config_labels: list[str],
) -> None:
    """Confirm all selected source bags and their ordinary baseline are complete."""

    source_fingerprint = source_manifest["run_fingerprint_sha256"]
    missing: list[str] = []
    for label in config_labels:
        config_dir = source_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}" / f"config_{label}"
        summary_path = config_dir / "config_summary.json"
        predictions_path = config_dir / "config_predictions.npz"
        if not summary_path.is_file() or not predictions_path.is_file():
            missing.append(f"config={label} aggregation")
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"OOF source has unreadable {summary_path}: {error}") from error
        if not isinstance(summary, dict) or summary.get("run_fingerprint_hash") != source_fingerprint:
            raise ValueError(f"OOF source configuration has another run fingerprint: {summary_path}")
        if summary.get("outer_split_hash") != task.outer_split_hash:
            raise ValueError(f"OOF source outer split does not match task {task.task_id}: {summary_path}")
        effective_bags = summary.get("effective_bags")
        bag_metadata = summary.get("bag_metadata")
        if not isinstance(effective_bags, int) or effective_bags < 1 or not isinstance(bag_metadata, list):
            raise ValueError(f"OOF source configuration is missing complete bag metadata: {summary_path}")
        if len(bag_metadata) != effective_bags:
            raise ValueError(f"OOF source configuration has incomplete bag metadata: {summary_path}")
        for bag in range(effective_bags):
            if not (config_dir / f"bag_{bag}.npz").is_file():
                missing.append(f"config={label} bag={bag}")
    baseline_dir = source_dir / "standard_tabarena_baseline" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"
    baseline_summary = baseline_dir / "summary.json"
    baseline_predictions = baseline_dir / "predictions.npz"
    if not baseline_summary.is_file() or not baseline_predictions.is_file():
        missing.append("normal TabICLv2 baseline")
    else:
        try:
            baseline = json.loads(baseline_summary.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"OOF source has unreadable {baseline_summary}: {error}") from error
        if not isinstance(baseline, dict) or baseline.get("run_fingerprint_hash") != source_fingerprint:
            raise ValueError(f"OOF source baseline has another run fingerprint: {baseline_summary}")
        if baseline.get("outer_split_hash") != task.outer_split_hash:
            raise ValueError(f"OOF source baseline outer split does not match task {task.task_id}")
    if missing:
        raise FileNotFoundError(
            f"OOF source task {task.task_id} is incomplete; refusing to mix old and new bags: "
            + ", ".join(missing)
        )


def _posthoc_oof_source_for_task(
    *,
    source_dir: Path | None,
    source_manifest: dict[str, Any] | None,
    current_immutable_run: dict[str, Any],
    selected_task_ids: list[int],
    task: Any,
    config_labels: list[str],
) -> tuple[Path | None, str | None]:
    """Resolve optional old bags after refit without invalidating the primary arm."""

    if source_dir is None or source_manifest is None:
        return None, None
    try:
        _validate_oof_source_compatibility(
            source_dir=source_dir,
            source_manifest=source_manifest,
            current_immutable_run=current_immutable_run,
            selected_task_ids=selected_task_ids,
        )
        _validate_oof_source_task_artifacts(
            source_dir=source_dir,
            source_manifest=source_manifest,
            task=task,
            config_labels=config_labels,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    return source_dir, None


def main(
    *,
    default_pipeline: str = "lite",
    required_pipeline: str | None = None,
    full_context_refit: bool = False,
    checkpoint_audit: bool = False,
    validation_selected_refit: bool = False,
    adaptive_phase1: bool = False,
    adaptive_retouche: bool = False,
    adaptive_retouche_d_only: bool = False,
    description: str | None = None,
) -> None:
    if adaptive_phase1 and not validation_selected_refit:
        raise ValueError("adaptive_phase1 requires validation_selected_refit=True")
    if adaptive_retouche and adaptive_phase1:
        raise ValueError("adaptive_retouche and adaptive_phase1 are mutually exclusive")
    if adaptive_retouche_d_only and not adaptive_retouche:
        raise ValueError("adaptive_retouche_d_only requires adaptive_retouche=True")
    args = parse_args(
        default_pipeline=default_pipeline,
        required_pipeline=required_pipeline,
        checkpoint_audit=checkpoint_audit,
        validation_selected_refit=validation_selected_refit,
        adaptive_phase1=adaptive_phase1,
        adaptive_retouche=adaptive_retouche,
        description=description,
    )
    _validate(
        args,
        full_context_refit=full_context_refit,
        checkpoint_audit=checkpoint_audit,
        validation_selected_refit=validation_selected_refit,
        adaptive_phase1=adaptive_phase1,
        adaptive_retouche=adaptive_retouche,
    )
    labels, configs = (
        _adaptive_retouche_configs(args, d_only=adaptive_retouche_d_only)
        if adaptive_retouche
        else _adaptive_phase1_validation_selected_refit_configs(args)
        if adaptive_phase1
        else _validation_selected_refit_configs(args)
        if validation_selected_refit
        else _configs(args)
    )
    manifest_path = args.output_dir / "experiment_manifest.json"
    previous: dict[str, Any] | None = None
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} already exists. Use --resume only for the exact immutable run, "
                "or choose a new --output-dir."
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif args.resume:
        raise FileNotFoundError(f"cannot safely --resume without {manifest_path}")

    if previous is not None and args.oof_source_dir is None:
        prior_source = previous.get("immutable_run", {}).get("oof_source")
        if isinstance(prior_source, dict) and isinstance(prior_source.get("path"), str):
            args.oof_source_dir = Path(prior_source["path"])
            print(
                f"Resuming with frozen post-hoc OOF diagnostic source {args.oof_source_dir}; "
                "it will not affect full-context training",
                flush=True,
            )
    oof_source_dir: Path | None = None
    oof_source_manifest: dict[str, Any] | None = None
    oof_source_provenance: dict[str, Any] | None = None
    if args.oof_source_dir is not None:
        oof_source_dir, oof_source_manifest = _load_oof_source_manifest(args.oof_source_dir)
        oof_source_provenance = _oof_source_provenance(oof_source_dir, oof_source_manifest)

    if args.task_id:
        task_ids = list(args.task_id)
        task_file_provenance = None
    elif args.task_id_file is not None:
        task_ids = _task_ids_from_file(args.task_id_file)
        task_file_provenance = _task_file_provenance(args.task_id_file)
    elif previous is not None:
        # A resume must be reproducible even when OpenML is temporarily down.
        task_ids = _frozen_manifest_task_ids(previous, manifest_path)
        task_file_provenance = None
        print(f"Resuming with {len(task_ids)} task ID(s) frozen in {manifest_path}; skipping OpenML suite lookup", flush=True)
    else:
        task_ids = tabarena_v0pt1_task_ids()
        task_file_provenance = None
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    if args.dry_run:
        checkpoint_fingerprints = None
    elif previous is not None:
        checkpoint_fingerprints = _restore_run_checkpoints(args, previous)
    else:
        checkpoint_fingerprints = _resolve_run_checkpoints(args, task_ids)
    device, execution_environment = _resolve_execution_environment(args.device, dry_run=args.dry_run)
    if previous is not None:
        # Preserve the prior value exactly on resume.  In particular, an older
        # explicit-task run recorded the historical suite ID even though it did
        # not consume the full suite.
        suite_id = previous.get("immutable_run", {}).get("data_source", {}).get(
            "suite_id", TABARENA_V0PT1_OPENML_SUITE_ID
        )
    else:
        suite_id = (
            None
            if args.task_id_file is not None
            else TABARENA_V0PT1_OPENML_SUITE_ID
        )
    immutable_run = {
        "schema_version": 6,
        "experiment_semantics_version": EXPERIMENT_SEMANTICS_VERSION,
        "repository_revision": _git_revision(),
        "source_sha256": _experiment_source_hashes(),
        "python": sys.version,
        "dependencies": _dependency_versions(),
        "execution_environment": execution_environment,
        "data_source": {
            "provider": "OpenML",
            "suite_id": suite_id,
            "task_ids": task_ids,
            "outer_split": {
                "repeat": args.outer_repeat,
                "fold": args.outer_fold,
                "sample": args.outer_sample,
            },
        },
        "task_eligibility": {
            "max_features": args.max_features,
            "rule": (
                "All input columns are eligible."
                if args.max_features is None
                else "Skip a task when its published outer-training table has more input columns than max_features."
            ),
        },
        "inner_bags": args.bags,
        "inner_bags_role": (
            "one deterministic inner train/validation split per task; bags are otherwise unused"
            if validation_selected_refit
            else
            "not used by the full-context checkpoint audit"
            if checkpoint_audit
            else "optional historical OOF diagnostic compatibility only"
            if full_context_refit
            else "training and validation"
        ),
        "inner_bag_policy": (
            "A task-specific deterministic stratified ShuffleSplit for classification (or ShuffleSplit for "
            "regression) is persisted with both index arrays; infeasible rare-class splits are explicitly skipped."
            if validation_selected_refit
            else
            "Use the requested count unless a classification task's rarest outer-training class is smaller; "
            "then use that smaller valid stratified count so every fitting fold retains at least two rows/class."
        ),
        "protocol_seed": args.protocol_seed,
        "bootstrap_rounds": args.bootstrap_rounds,
        "ensemble_rounds": args.ensemble_rounds or max(1, 2 * len(configs)),
        "pipeline": args.pipeline,
        "full_context_refit": bool(full_context_refit),
        "checkpoint_audit": bool(checkpoint_audit),
        "validation_selected_refit": bool(validation_selected_refit),
        "adaptive_phase1": bool(adaptive_phase1),
        "adaptive_retouche": bool(adaptive_retouche),
        "checkpoint_steps": list(args.checkpoint_steps) if checkpoint_audit else None,
        "validation_selected_refit_settings": (
            None
            if not validation_selected_refit
            else {
                "validation_fraction": float(args.validation_fraction),
                "split_seed": int(args.split_seed),
                "adapter_seed": int(args.adapter_seed),
                "selection_checkpoint_interval": int(args.selection_checkpoint_interval),
                "cosine_schedule_steps": int(args.adapter_steps),
                "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
                "selection_relative_improvement": float(args.selection_relative_improvement),
                "identity_regularization": 0.0 if adaptive_phase1 else float(args.identity_regularization),
            }
        ),
        "adaptive_retouche_settings": (
            None
            if not adaptive_retouche
            else {
                "adapter_seed": int(args.adapter_seed),
                "adapter_steps": int(args.adapter_steps),
                "checkpoint_interval": int(args.validation_interval),
                "scheduler": {
                    "kind": "cosine",
                    "horizon_steps": int(args.adapter_steps),
                    "min_lr_ratio": float(args.cosine_min_lr_ratio),
                },
                "early_stopping": {"stale_validation_checks": 12},
                "identity_regularization": 0.0,
                "guard_relative_improvement": 0.005,
            }
        ),
        "oof_source": oof_source_provenance,
        "standard_pipeline": (
            None
            if args.pipeline == "lite"
            else {
                "identity_context": (
                    "one inner-train context for checkpoint selection, then all outer-training rows for refit"
                    if validation_selected_refit
                    else
                    "all outer-training rows"
                    if full_context_refit
                    else "all inner-bag fit rows"
                    if args.context_cap == 0
                    else "stratified capped rows"
                ),
                "context_cap": None if args.context_cap == 0 else args.context_cap,
                "requested_train_context_rows": args.train_context_rows,
                "resolved_train_context_rows": configs[0]["train_context_rows"],
                "normal_tabarena_config": STANDARD_TABICL_CONFIG,
                "final_deployment": (
                    {
                        "role": "eight-fold preserved-checkpoint guarded ensemble",
                        "fit_context": "all rows of each inner-bag fitting partition",
                        "validation": "the held-out partition of each bag",
                        "checkpoint_selection": "best trained spline checkpoint by the per-bag deployment metric",
                        "identity_guard": "per-bag post-training validation guard with a 0.5% relative-improvement threshold",
                        "test_prediction": "mean of the eight guarded bag predictions; no all-row refit",
                        "configuration_selection": "pooled guarded OOF validation predictions only",
                        "outer_test_selection": "forbidden",
                        "variants": labels,
                    }
                    if adaptive_retouche
                    else None
                    if not full_context_refit and not validation_selected_refit
                    else {
                        "role": "validation-selected checkpoint then fresh all-row refit",
                        "inner_selection_context": "the persisted inner-training subset",
                        "validation": "one deterministic held-out subset of the published outer-training rows",
                        "full_refit_context": "all outer-training rows",
                        "schedule": "fixed cosine horizon; refit uses the selected checkpoint duration and same LR prefix",
                        "checkpoint_selection": "best observed validation checkpoint or step-0 identity only",
                        "identity_reference": "fresh identity prediction from the matching inner or full-context estimator",
                        "outer_test_selection": "forbidden",
                        "variants": labels,
                    }
                    if validation_selected_refit
                    else {
                        "role": "development-only full-context checkpoint learning curve",
                        "context": "all outer-training rows",
                        "configuration": "sole predeclared DirectSpline configuration",
                        "checkpoint_steps": list(args.checkpoint_steps),
                        "checkpoint_selection": "none; outer test is diagnostic only",
                        "identity_reference": "the fresh all-row estimator's exact pre-optimisation identity prediction",
                    }
                    if checkpoint_audit
                    else {
                        "context": "all outer-training rows",
                        "configuration": "sole predeclared DirectSpline configuration",
                        "schedule": "fixed predeclared adapter_steps; no early stopping or OOF checkpoint selection",
                        "guard": "none; the spline is evaluated unconditionally on every completed task",
                        "identity_reference": "the fresh all-row estimator's exact pre-optimisation identity prediction",
                        "oof_source_role": "optional post-hoc validation/test correlation diagnostic only",
                    }
                ),
            }
        ),
        "guard": {
            "binary": "1 - ROC-AUC",
            "multiclass": "log loss",
            "regression": "MSE",
            "required_relative_improvement": (
                configs[0]["selection_relative_improvement"]
                if validation_selected_refit
                else configs[0]["guard_relative_improvement"]
            ),
            "scope": (
                "single_inner_validation_checkpoint_selection_then_full_context_refit"
                if validation_selected_refit
                else
                "development_only_checkpoint_curve_no_selection"
                if checkpoint_audit
                else
                "posthoc_oof_correlation_only_no_deployment_guard"
                if full_context_refit
                else "retouche_per_bag_validation_guard_then_test_ensemble"
            ),
        },
        "leaderboard_metric": {"binary": "1 - ROC-AUC", "multiclass": "log loss", "regression": "RMSE"},
        "config_labels": labels,
        "configs": configs,
        "classifier_checkpoint_argument": None if args.classifier_checkpoint is None else str(args.classifier_checkpoint.resolve()),
        "regressor_checkpoint_argument": None if args.regressor_checkpoint is None else str(args.regressor_checkpoint.resolve()),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "standard_tabarena_baseline": (
            STANDARD_TABICL_CONFIG
            if full_context_refit or (not validation_selected_refit and not args.skip_standard_baseline)
            else None
        ),
        "full_context_refit_contract": (
            None
            if not full_context_refit
            else {
                "oof_used_for_training_or_deployment": False,
                "separate_standard_baseline_run": False,
                "identity_prediction_source": "same fresh full-context estimator used by the adapted path",
                "spline_evaluated_for_every_task": not checkpoint_audit,
                "checkpoint_audit_development_only": bool(checkpoint_audit),
                "outer_test_checkpoint_selection_permitted": False if checkpoint_audit else None,
            }
        ),
        "validation_selected_refit_contract": (
            None
            if not validation_selected_refit
            else {
                "outer_test_used_for_selection": False,
                "split_indices_persisted": True,
                "adapter_seed_independent_of_protocol_seed": True,
                "identity_is_a_selectable_step_zero_checkpoint": True,
                "full_refit_uses_selected_duration": True,
                "full_refit_preserves_inner_cosine_schedule_horizon": True,
                "selection_uses_patience": False,
                "predeclared_variants": labels,
            }
        ),
        "adaptive_phase1_contract": (
            None
            if not adaptive_phase1
            else {
                "purpose": "development comparison of fixed, adaptive-column, and conditional adaptive DirectSpline bases",
                "outer_test_used_for_architecture_selection": False,
                "train_validation_test_protocol": "one persisted inner split selects identity or checkpoint; a fresh all-row refit uses the selected duration",
                "predeclared_arms": labels,
                "adaptive_expert_specs": [
                    {"degree": 1, "n_control_points": 4},
                    {"degree": 2, "n_control_points": 8},
                    {"degree": 3, "n_control_points": 20},
                ],
                "conditional_interaction": {
                    "rank": 4,
                    "residual_amplitude_bound": 0.25,
                    "enabled_only_for": "conditional_adaptive_columns",
                },
                "identity_regularization": 0.0,
            }
        ),
        "adaptive_retouche_contract": (
            None
            if not adaptive_retouche
            else {
                "purpose": (
                    "final TabArena evaluation of the fixed cubic-20 DirectSpline adapter against TabICLv2"
                    if adaptive_retouche_d_only
                    else "final matched TabICLv2 comparison of fixed, adaptive-column, and conditional DirectSpline adapters"
                ),
                "outer_test_used_for_training_or_selection": False,
                "predeclared_arms": labels,
                "train_validation_test_protocol": "each of eight bags trains on its fit rows, retains its validation-best spline checkpoint, applies a per-bag identity guard, and contributes that guarded member to the test ensemble",
                "identity_is_a_checkpoint_candidate": False,
                "identity_guard_scope": "per bag after selecting the best trained checkpoint",
                "all_row_refit": False,
                "adapter_steps": 500,
                "checkpoint_interval": 25,
                "scheduler": "single-cycle cosine from base LR to 1% of base LR",
                "early_stopping": {"stale_validation_checks": 12},
                "identity_regularization": 0.0,
                "adaptive_expert_specs": None if adaptive_retouche_d_only else [
                    {"degree": 1, "n_control_points": 4},
                    {"degree": 2, "n_control_points": 8},
                    {"degree": 3, "n_control_points": 20},
                ],
                "conditional_interaction": None if adaptive_retouche_d_only else {
                    "rank": 4,
                    "residual_amplitude_bound": 0.25,
                    "enabled_only_for": "conditional_adaptive_columns",
                },
                "reporting": {
                    "D": "fixed cubic-20 default",
                    "T": (
                        "alias of D; architecture selection is intentionally disabled for this D-only evaluation"
                        if adaptive_retouche_d_only
                        else "single best predeclared arm chosen by guarded OOF validation error"
                    ),
                    "T+E": (
                        "alias of D; architecture ensembling is intentionally disabled for this D-only evaluation"
                        if adaptive_retouche_d_only
                        else "greedy ensemble of guarded arms chosen by OOF validation only"
                    ),
                },
            }
        ),
    }
    run_fingerprint_hash = _fingerprint(immutable_run)
    if previous is not None:
        if previous.get("run_fingerprint_sha256") != run_fingerprint_hash or previous.get("immutable_run") != immutable_run:
            previous_run = previous.get("immutable_run")
            hardware_resume_mismatches = (
                _equivalent_hardware_resume_mismatches(previous_run, immutable_run)
                if args.allow_equivalent_hardware_resume and isinstance(previous_run, dict)
                else []
            )
            retouche_efficiency_resume_mismatches = (
                _retouche_efficiency_resume_mismatches(
                    previous_run,
                    immutable_run,
                    allow_equivalent_hardware=bool(args.allow_equivalent_hardware_resume),
                )
                if args.allow_retouche_efficiency_resume and isinstance(previous_run, dict)
                else []
            )
            compatible_code_resume = (
                args.allow_compatible_code_resume
                and isinstance(previous_run, dict)
                and _same_experimental_semantics(previous_run, immutable_run)
            )
            compatible_hardware_resume = (
                args.allow_equivalent_hardware_resume
                and isinstance(previous_run, dict)
                and _same_equivalent_hardware_resume_semantics(previous_run, immutable_run)
            )
            compatible_retouche_efficiency_resume = (
                args.allow_retouche_efficiency_resume
                and isinstance(previous_run, dict)
                and not retouche_efficiency_resume_mismatches
            )
            if not (
                compatible_code_resume
                or compatible_hardware_resume
                or compatible_retouche_efficiency_resume
            ):
                if args.allow_retouche_efficiency_resume:
                    mismatch_summary = ", ".join(retouche_efficiency_resume_mismatches) or "unknown"
                    raise ValueError(
                        "refusing Retouche efficiency resume: differences extend beyond the explicit "
                        "full-horizon-to-patience-12 migration. "
                        f"Mismatched fields: {mismatch_summary}."
                    )
                if args.allow_equivalent_hardware_resume:
                    mismatch_summary = ", ".join(hardware_resume_mismatches) or "unknown"
                    raise ValueError(
                        "refusing equivalent-hardware resume: the new allocation differs in model/training "
                        "semantics or stable GPU/software/precision properties. "
                        f"Mismatched fields: {mismatch_summary}. Choose a new --output-dir."
                    )
                raise ValueError(
                    "refusing to resume: the existing output directory has a different immutable run fingerprint; "
                    "choose a new --output-dir."
                )
            prior_hash = str(previous["run_fingerprint_sha256"])
            if compatible_code_resume:
                provenance_path = args.output_dir / "compatible_code_resumes.jsonl"
                provenance = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "previous_run_fingerprint_sha256": prior_hash,
                    "previous_source_sha256": previous_run.get("source_sha256"),
                    "resumed_source_sha256": immutable_run["source_sha256"],
                    "reason": "explicit revision-metadata-only resume with identical sources",
                }
                message = (
                    "Resuming with identical source hashes and an explicitly recorded Git revision change; "
                    f"retaining immutable run fingerprint {prior_hash[:12]}"
                )
            elif compatible_retouche_efficiency_resume:
                provenance_path = args.output_dir / "retouche_efficiency_resumes.jsonl"
                provenance = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "previous_run_fingerprint_sha256": prior_hash,
                    "proposed_resume_fingerprint_sha256": run_fingerprint_hash,
                    "previous_experiment_semantics_version": previous_run.get(
                        "experiment_semantics_version"
                    ),
                    "resumed_experiment_semantics_version": immutable_run.get(
                        "experiment_semantics_version"
                    ),
                    "previous_source_sha256": previous_run.get("source_sha256"),
                    "resumed_source_sha256": immutable_run["source_sha256"],
                    "previous_execution_environment": previous_run.get("execution_environment"),
                    "resumed_execution_environment": immutable_run["execution_environment"],
                    "completed_artifact_policy": (
                        "reuse version-8 completed configs/bags with their full 500-step checkpoint search"
                    ),
                    "unfinished_artifact_policy": (
                        "run version-9 configs/bags with a 500-step maximum and 12 stale-check patience"
                    ),
                    "unchanged_comparison_contract": [
                        "TabICLv2 baseline",
                        "OpenML tasks and outer splits",
                        "eight deterministic inner bags",
                        "adapter and protocol seeds",
                        "spline architectures and optimizer",
                        "cosine schedule and checkpoint interval",
                        "validation identity guard",
                        "held-out test evaluation",
                    ],
                    "ignored_allocation_identity_fields": (
                        [
                            "cuda.visible_device_count",
                            "cuda.selected_hardware.index",
                            "cuda.selected_hardware.uuid",
                        ]
                        if args.allow_equivalent_hardware_resume
                        else []
                    ),
                    "slurm_allocation": _slurm_allocation_provenance(),
                    "reason": (
                        "explicit mixed-compute Retouche resume: completed full-horizon searches are "
                        "retained, while only unfinished work receives the conservative efficiency stop"
                    ),
                }
                message = (
                    "Resuming the Retouche comparison with completed full-horizon artifacts preserved and "
                    "patience-12 applied only to unfinished work; "
                    f"retaining immutable run fingerprint {prior_hash[:12]}"
                )
                migrations = previous.setdefault("resume_protocol_migrations", [])
                if not isinstance(migrations, list):
                    raise ValueError("existing manifest has an invalid resume_protocol_migrations record")
                if not any(
                    isinstance(item, dict)
                    and item.get("proposed_resume_fingerprint_sha256") == run_fingerprint_hash
                    for item in migrations
                ):
                    migrations.append(provenance)
                    _json_dump(manifest_path, previous)
            else:
                provenance_path = args.output_dir / "equivalent_hardware_resumes.jsonl"
                provenance = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "previous_run_fingerprint_sha256": prior_hash,
                    "proposed_resume_fingerprint_sha256": run_fingerprint_hash,
                    "previous_repository_revision": previous_run.get("repository_revision"),
                    "resumed_repository_revision": immutable_run.get("repository_revision"),
                    "previous_source_sha256": previous_run.get("source_sha256"),
                    "resumed_source_sha256": immutable_run["source_sha256"],
                    "previous_execution_environment": previous_run.get("execution_environment"),
                    "resumed_execution_environment": immutable_run["execution_environment"],
                    "ignored_allocation_identity_fields": [
                        "cuda.visible_device_count",
                        "cuda.selected_hardware.index",
                        "cuda.selected_hardware.uuid",
                    ],
                    "slurm_allocation": _slurm_allocation_provenance(),
                    "reason": "explicit equivalent-hardware resume across a scheduler allocation",
                }
                message = (
                    "Resuming across an explicitly recorded equivalent Slurm/CUDA allocation; "
                    f"retaining immutable run fingerprint {prior_hash[:12]}"
                )
            with provenance_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(provenance, sort_keys=True) + "\n")
            print(message, flush=True)
            run_fingerprint_hash = prior_hash
        manifest = previous
    else:
        manifest = {
            "experiment": (
                "DirectSpline OpenML standard-ensemble validation-selected checkpoint and full-context refit experiment"
                if validation_selected_refit
                else "DirectSpline OpenML standard-ensemble development-only full-context checkpoint audit"
                if checkpoint_audit
                else
                "DirectSpline OpenML standard-ensemble full-context refit experiment"
                if full_context_refit
                else "DirectSpline OpenML standard-ensemble experiment"
                if args.pipeline == "standard"
                else "DirectSpline OpenML Lite experiment"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "immutable_run": immutable_run,
            "run_fingerprint_sha256": run_fingerprint_hash,
            "outer_test_policy": (
                "read only after both variants' validation-selected inner-model and full-context-refit predictions "
                "are frozen; never used for splitting, fitting, checkpoint selection, regularisation selection, "
                "or identity selection"
                if validation_selected_refit
                else "read only after all requested checkpoint predictions are frozen, to diagnose this development "
                "learning curve; never used for optimisation, checkpoint selection, configuration selection, "
                "regularisation selection, guarding, HPO, or ensembling"
                if checkpoint_audit
                else "never read by preprocessing fitting, adapter optimisation, guard, HPO, or ensembling"
            ),
            "absolute_elo_note": (
                "This run computes paired Elo deltas versus its own matched TabICL identity baseline. "
                "Those deltas cannot be compared numerically with absolute ELO on TabArena's large published method pool."
            ),
        }
        if task_file_provenance is not None:
            # The immutable item is the exact ordered task-ID list above.  The
            # file digest is retained as human-auditable provenance without
            # making an absolute path part of the resume fingerprint.
            manifest["task_selection_file"] = task_file_provenance
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _json_dump(manifest_path, manifest)
    if args.dry_run:
        print(
            f"Previewing immutable run fingerprint {run_fingerprint_hash[:12]} for {len(task_ids)} task(s); "
            f"no manifest or output directory was written ({manifest_path})",
            flush=True,
        )
        print(json.dumps(manifest, indent=2, default=str), flush=True)
        return
    print(f"Using immutable run fingerprint {run_fingerprint_hash[:12]} for {len(task_ids)} task(s): {manifest_path}", flush=True)
    if device is None:  # pragma: no cover - real runs fail during resolution above
        raise RuntimeError(f"could not resolve requested execution device {args.device!r}")
    if args.resume:
        _repair_interrupted_config_summaries(args.output_dir)
    progress = _event_reporter(args.output_dir / "progress.jsonl")
    task_summaries: list[dict[str, Any]] = []
    persisted_task_skips = _persisted_task_skips(args.output_dir) if args.resume else {}
    # Keep every prior skip in each progress snapshot until that exact task is
    # revisited. If this resume is interrupted early, later OOM records remain
    # recoverable instead of disappearing from run_progress.json.
    skipped_tasks: list[dict[str, Any]] = list(persisted_task_skips.values())
    persisted_cuda_oom_skips = {
        task_id: item
        for task_id, item in persisted_task_skips.items()
        if item.get("reason") == "cuda_out_of_memory"
    }
    if args.retry_cuda_oom_skips and persisted_cuda_oom_skips:
        retry_ids = ", ".join(str(task_id) for task_id in sorted(persisted_cuda_oom_skips))
        print(f"Retrying previously recorded CUDA-OOM task(s): {retry_ids}", flush=True)
    for task_id in task_ids:
        if task_id in persisted_cuda_oom_skips and not args.retry_cuda_oom_skips:
            skipped = persisted_cuda_oom_skips[task_id]
            print(
                f"[task={task_id} dataset={skipped['dataset_name']}] reusing recorded CUDA-OOM skip; "
                "pass --retry-cuda-oom-skips to retry it",
                flush=True,
            )
            _write_run_progress(args.output_dir, task_summaries, skipped_tasks)
            continue
        if task_id in persisted_task_skips:
            # The on-disk record remains intact until the next progress write.
            # In memory, remove it now so either a new skip or a successful task
            # replaces the stale outcome exactly once.
            skipped_tasks = [
                item for item in skipped_tasks if int(item["task_id"]) != int(task_id)
            ]
        task = load_tabarena_openml_task(
            task_id,
            outer_repeat=args.outer_repeat,
            outer_fold=args.outer_fold,
            outer_sample=args.outer_sample,
        )
        print(
            f"[task={task.task_id} dataset={task.dataset_name}] "
            f"{task.problem_type}; outer train={len(task.y_train)}, test={len(task.y_test)}, "
            f"features={task.x_train.shape[1]}",
            flush=True,
        )
        n_features = int(task.x_train.shape[1])
        if args.max_features is not None and n_features > args.max_features:
            skipped = {
                "task_id": task.task_id,
                "dataset_id": task.dataset_id,
                "dataset_name": task.dataset_name,
                "problem_type": task.problem_type,
                "n_features": n_features,
                "max_features": args.max_features,
                "reason": "n_features_exceeds_max_features",
            }
            skipped_tasks.append(skipped)
            progress({"event": "task_skipped", **skipped})
            _write_run_progress(args.output_dir, task_summaries, skipped_tasks)
            continue
        standard_tabarena = None
        baseline_cuda_oom = False
        if not full_context_refit and not validation_selected_refit and not args.skip_standard_baseline:
            try:
                standard_tabarena = run_standard_tabarena_baseline(
                    task=task,
                    output_dir=args.output_dir,
                    device=device,
                    classifier_checkpoint=args.classifier_checkpoint,
                    regressor_checkpoint=args.regressor_checkpoint,
                    resume=args.resume,
                    run_fingerprint_hash=run_fingerprint_hash,
                    progress=progress,
                )
            except RuntimeError as error:
                if not _is_cuda_out_of_memory(error):
                    raise
                allocated_gib = reserved_gib = total_gib = None
                if device.type == "cuda":
                    allocated_gib = float(torch.cuda.memory_allocated(device) / 2**30)
                    reserved_gib = float(torch.cuda.memory_reserved(device) / 2**30)
                    total_gib = float(torch.cuda.get_device_properties(device).total_memory / 2**30)
                progress(
                    {
                        "event": "standard_baseline_skipped",
                        "task_id": task.task_id,
                        "dataset_id": task.dataset_id,
                        "dataset_name": task.dataset_name,
                        "problem_type": task.problem_type,
                        "reason": "cuda_out_of_memory",
                        "device": str(device),
                        "cuda_allocated_gib": allocated_gib,
                        "cuda_reserved_gib": reserved_gib,
                        "cuda_total_gib": total_gib,
                        "error": str(error).splitlines()[0],
                    }
                )
                baseline_cuda_oom = True
        if baseline_cuda_oom:
            # Release traceback-owned tensors before a normal bag-only run
            # attempts smaller inner fits.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if full_context_refit:
                # This experiment's primary comparison is intentionally
                # exact full-row DirectSpline versus exact full-row TabICLv2.
                # Without the latter there is no valid endpoint, and a final
                # full-row spline is unlikely to fit where the normal model
                # itself did not. Record a recoverable hardware skip rather
                # than silently reporting a different comparison.
                skipped = {
                    "task_id": task.task_id,
                    "dataset_id": task.dataset_id,
                    "dataset_name": task.dataset_name,
                    "problem_type": task.problem_type,
                    "n_features": n_features,
                    "outer_train_rows": int(len(task.y_train)),
                    "outer_test_rows": int(len(task.y_test)),
                    "reason": "cuda_out_of_memory",
                    "stage": "standard_baseline_required_for_full_context_refit",
                    "device": str(device),
                }
                skipped_tasks.append(skipped)
                progress({"event": "task_skipped", **skipped})
                _write_run_progress(args.output_dir, task_summaries, skipped_tasks)
                continue

        stage = (
            "validation_selected_refit"
            if validation_selected_refit
            else "full_context_checkpoint_audit"
            if checkpoint_audit
            else "full_context_refit"
            if full_context_refit
            else "config_D"
        )
        cuda_oom_skipped = False
        try:
            run_config = run_task_config_standard if args.pipeline == "standard" else run_task_config
            if not full_context_refit and not validation_selected_refit:
                for label, config in zip(labels, configs, strict=True):
                    stage = f"config_{label}"
                    effective_bags = effective_inner_bag_count(task, requested_bags=args.bags)
                    print(
                        f"[task={task.task_id} config={label}] starting/recovering "
                        f"{effective_bags}/{args.bags} valid stratified bags",
                        flush=True,
                    )
                    result = run_config(
                        task=task,
                        label=label,
                        config=config,
                        output_dir=args.output_dir,
                        bags=args.bags,
                        protocol_seed=args.protocol_seed,
                        device=device,
                        classifier_checkpoint=args.classifier_checkpoint,
                        regressor_checkpoint=args.regressor_checkpoint,
                        resume=args.resume,
                        run_fingerprint_hash=run_fingerprint_hash,
                        progress=progress,
                    )
                    print(
                        f"[task={task.task_id} config={label}] "
                        f"guarded validation={result['validation']['guarded']['deployment_error']:.6g}; "
                        "outer-test score withheld until validation-only selection is complete",
                        flush=True,
                    )
            stage = "task_summary"
            try:
                if validation_selected_refit:
                    validation_refit_result = run_task_validation_selected_full_refit_standard(
                        task=task,
                        config_labels=labels,
                        configs=configs,
                        validation_fraction=args.validation_fraction,
                        validation_seed=args.split_seed,
                        output_dir=args.output_dir,
                        protocol_seed=args.protocol_seed,
                        device=device,
                        classifier_checkpoint=args.classifier_checkpoint,
                        regressor_checkpoint=args.regressor_checkpoint,
                        resume=args.resume,
                        run_fingerprint_hash=run_fingerprint_hash,
                        progress=progress,
                    )
                    task_summary = summarize_validation_selected_full_refit_task(
                        task=task,
                        output_dir=args.output_dir,
                        task_result=validation_refit_result,
                    )
                elif full_context_refit:
                    if checkpoint_audit:
                        audit_result = run_task_full_context_refit_checkpoint_audit_standard(
                            task=task,
                            config_labels=labels,
                            configs=configs,
                            checkpoint_steps=args.checkpoint_steps,
                            output_dir=args.output_dir,
                            protocol_seed=args.protocol_seed,
                            device=device,
                            classifier_checkpoint=args.classifier_checkpoint,
                            regressor_checkpoint=args.regressor_checkpoint,
                            resume=args.resume,
                            run_fingerprint_hash=run_fingerprint_hash,
                            progress=progress,
                        )
                        task_summary = summarize_full_context_refit_checkpoint_audit_task(
                            task=task,
                            output_dir=args.output_dir,
                            audit_result=audit_result,
                        )
                    else:
                        refit_result = run_task_unconditional_full_context_refit_standard(
                            task=task,
                            config_labels=labels,
                            configs=configs,
                            output_dir=args.output_dir,
                            protocol_seed=args.protocol_seed,
                            device=device,
                            classifier_checkpoint=args.classifier_checkpoint,
                            regressor_checkpoint=args.regressor_checkpoint,
                            resume=args.resume,
                            run_fingerprint_hash=run_fingerprint_hash,
                            progress=progress,
                        )
                        # The primary full-context prediction is now frozen. Only
                        # at this point may historical OOF bags be opened, and
                        # only to test whether their validation signal predicted
                        # the unconditional full-context outcome.
                        task_oof_source, task_oof_error = _posthoc_oof_source_for_task(
                            source_dir=oof_source_dir,
                            source_manifest=oof_source_manifest,
                            current_immutable_run=immutable_run,
                            selected_task_ids=task_ids,
                            task=task,
                            config_labels=labels,
                        )
                        if task_oof_source is not None:
                            print(
                                f"[task={task.task_id}] loading old OOF bags only for post-hoc correlation",
                                flush=True,
                            )
                        elif task_oof_error is not None:
                            print(
                                f"[task={task.task_id}] old OOF diagnostic unavailable; "
                                "keeping the completed unconditional full-refit result",
                                flush=True,
                            )
                        task_summary = summarize_full_context_refit_task(
                            task=task,
                            output_dir=args.output_dir,
                            refit_result=refit_result,
                            oof_source_dir=task_oof_source,
                            oof_diagnostic_unavailable_reason=task_oof_error,
                        )
                else:
                    task_summary = summarize_task_tuning(
                        task=task,
                        config_labels=labels,
                        output_dir=args.output_dir,
                        ensemble_rounds=immutable_run["ensemble_rounds"],
                        standard_tabarena=standard_tabarena,
                    )
            except json.JSONDecodeError:
                if checkpoint_audit or validation_selected_refit:
                    raise
                if oof_source_dir is not None:
                    raise RuntimeError(
                        "OOF source contains a malformed aggregation; it is read-only and cannot be repaired "
                        "by this refit run"
                    ) from None
                # A filesystem interruption can surface immediately after the
                # configuration loop. Reuse the normal resume path: complete
                # bags are retained and only the final aggregation is rebuilt.
                recovered = _repair_interrupted_config_summaries(args.output_dir)
                if not recovered:
                    raise
                print(
                    f"[task={task.task_id}] rebuilding interrupted configuration aggregation from bag checkpoints",
                    flush=True,
                )
                for label, config in zip(labels, configs, strict=True):
                    run_config(
                        task=task,
                        label=label,
                        config=config,
                        output_dir=args.output_dir,
                        bags=args.bags,
                        protocol_seed=args.protocol_seed,
                        device=device,
                        classifier_checkpoint=args.classifier_checkpoint,
                        regressor_checkpoint=args.regressor_checkpoint,
                        resume=True,
                        run_fingerprint_hash=run_fingerprint_hash,
                        progress=progress,
                    )
                if full_context_refit:
                    refit_result = run_task_unconditional_full_context_refit_standard(
                        task=task,
                        config_labels=labels,
                        configs=configs,
                        output_dir=args.output_dir,
                        protocol_seed=args.protocol_seed,
                        device=device,
                        classifier_checkpoint=args.classifier_checkpoint,
                        regressor_checkpoint=args.regressor_checkpoint,
                        resume=True,
                        run_fingerprint_hash=run_fingerprint_hash,
                        progress=progress,
                    )
                    task_oof_source, task_oof_error = _posthoc_oof_source_for_task(
                        source_dir=oof_source_dir,
                        source_manifest=oof_source_manifest,
                        current_immutable_run=immutable_run,
                        selected_task_ids=task_ids,
                        task=task,
                        config_labels=labels,
                    )
                    task_summary = summarize_full_context_refit_task(
                        task=task,
                        output_dir=args.output_dir,
                        refit_result=refit_result,
                        oof_source_dir=task_oof_source,
                        oof_diagnostic_unavailable_reason=task_oof_error,
                    )
                else:
                    task_summary = summarize_task_tuning(
                        task=task,
                        config_labels=labels,
                        output_dir=args.output_dir,
                        ensemble_rounds=immutable_run["ensemble_rounds"],
                        standard_tabarena=standard_tabarena,
                    )
        except ValidationSplitInfeasibleError as error:
            skipped = {
                "task_id": task.task_id,
                "dataset_id": task.dataset_id,
                "dataset_name": task.dataset_name,
                "problem_type": task.problem_type,
                "n_features": n_features,
                "outer_train_rows": int(len(task.y_train)),
                "outer_test_rows": int(len(task.y_test)),
                "reason": "validation_split_infeasible",
                "stage": stage,
                "error": str(error),
            }
            skipped_tasks.append(skipped)
            progress({"event": "task_skipped", **skipped})
            _write_run_progress(args.output_dir, task_summaries, skipped_tasks)
            continue
        except RuntimeError as error:
            if not _is_cuda_out_of_memory(error):
                raise
            allocated_gib = reserved_gib = total_gib = None
            if device.type == "cuda":
                allocated_gib = float(torch.cuda.memory_allocated(device) / 2**30)
                reserved_gib = float(torch.cuda.memory_reserved(device) / 2**30)
                total_gib = float(torch.cuda.get_device_properties(device).total_memory / 2**30)
            skipped = {
                "task_id": task.task_id,
                "dataset_id": task.dataset_id,
                "dataset_name": task.dataset_name,
                "problem_type": task.problem_type,
                "n_features": n_features,
                "outer_train_rows": int(len(task.y_train)),
                "outer_test_rows": int(len(task.y_test)),
                "reason": "cuda_out_of_memory",
                "stage": stage,
                "device": str(device),
                "cuda_allocated_gib": allocated_gib,
                "cuda_reserved_gib": reserved_gib,
                "cuda_total_gib": total_gib,
                "error": str(error).splitlines()[0],
            }
            skipped_tasks.append(skipped)
            progress({"event": "task_skipped", **skipped})
            _write_run_progress(args.output_dir, task_summaries, skipped_tasks)
            cuda_oom_skipped = True
        if cuda_oom_skipped:
            # This runs after the exception handler has released its traceback
            # and the tensors referenced by the failed backward frame.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        task_summaries.append(task_summary)
        _write_run_progress(args.output_dir, task_summaries, skipped_tasks)
    if task_summaries:
        if validation_selected_refit:
            summary = summarize_validation_selected_full_refit_experiment(
                task_summaries=task_summaries,
                output_dir=args.output_dir,
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.protocol_seed,
                skipped_tasks=skipped_tasks,
                task_eligibility=immutable_run["task_eligibility"],
            )
        elif full_context_refit:
            if checkpoint_audit:
                summary = summarize_full_context_refit_checkpoint_audit_experiment(
                    task_summaries=task_summaries,
                    output_dir=args.output_dir,
                    bootstrap_rounds=args.bootstrap_rounds,
                    bootstrap_seed=args.protocol_seed,
                    skipped_tasks=skipped_tasks,
                    task_eligibility=immutable_run["task_eligibility"],
                )
            else:
                summary = summarize_full_context_refit_experiment(
                    task_summaries=task_summaries,
                    output_dir=args.output_dir,
                    bootstrap_rounds=args.bootstrap_rounds,
                    bootstrap_seed=args.protocol_seed,
                    skipped_tasks=skipped_tasks,
                    task_eligibility=immutable_run["task_eligibility"],
                )
        else:
            summary = summarize_experiment(
                task_summaries=task_summaries,
                output_dir=args.output_dir,
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.protocol_seed,
                skipped_tasks=skipped_tasks,
                task_eligibility=immutable_run["task_eligibility"],
            )
    else:
        summary = _write_all_skipped_summary(
            args.output_dir,
            skipped_tasks=skipped_tasks,
            task_eligibility=immutable_run["task_eligibility"],
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
