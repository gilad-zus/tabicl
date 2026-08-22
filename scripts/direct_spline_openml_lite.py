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
    run_task_config_standard,
    shared_standard_direct_spline_configs,
    standard_direct_spline_config,
)


# Increment this whenever a code change can alter predictions, validation
# selection, or persisted artifact semantics.  Unlike the source hashes, this
# value is deliberately *not* ignored by --allow-compatible-code-resume.
EXPERIMENT_SEMANTICS_VERSION = 4
_RESUME_LAUNCHER_SOURCE = "scripts/direct_spline_openml_lite.py"


def parse_args(
    *,
    default_pipeline: str = "lite",
    required_pipeline: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-id", action="append", type=int,
        help="OpenML task ID. Repeat for a pilot; omit to run the public 51-task suite.",
    )
    parser.add_argument(
        "--task-id-file",
        type=Path,
        default=None,
        help=(
            "A frozen JSON task bank (with selected_task_ids or task_ids) or a text file of positive "
            "OpenML task IDs, one per line. Cannot be combined with --task-id."
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
    if required_pipeline is not None and args.pipeline != required_pipeline:
        parser.error(
            f"this launcher requires --pipeline {required_pipeline}; "
            f"received --pipeline {args.pipeline}"
        )
    return args


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


def _same_equivalent_hardware_resume_semantics(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Check a Slurm-safe resume without silently accepting model/code changes.

    This is deliberately narrower than a broad fingerprint override: every
    non-launcher source and every experiment field must match. Only the
    scheduler allocation identity inside the execution environment may change.
    """

    if previous.get("experiment_semantics_version") != current.get("experiment_semantics_version"):
        return False
    if not _same_sources_except_resume_launcher(previous, current):
        return False
    ignored = {"repository_revision", "source_sha256", "execution_environment"}
    if {key: value for key, value in previous.items() if key not in ignored} != {
        key: value for key, value in current.items() if key not in ignored
    }:
        return False
    return _normalise_equivalent_hardware_environment(previous.get("execution_environment")) == (
        _normalise_equivalent_hardware_environment(current.get("execution_environment"))
    )


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


def _validate(args: argparse.Namespace) -> None:
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
    if args.retry_cuda_oom_skips and not args.resume:
        raise ValueError("--retry-cuda-oom-skips requires --resume")
    if getattr(args, "dry_run", False) and args.resume:
        raise ValueError("--dry-run cannot be combined with --resume; it is a non-persistent preview")


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
        else:
            raw_ids = None
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(
            f"--task-id-file must be a JSON list, a JSON object with selected_task_ids/task_ids, "
            f"or one task ID per line: {path}"
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


def main(
    *,
    default_pipeline: str = "lite",
    required_pipeline: str | None = None,
) -> None:
    args = parse_args(default_pipeline=default_pipeline, required_pipeline=required_pipeline)
    _validate(args)
    labels, configs = _configs(args)
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
        suite_id = None if args.task_id_file is not None else TABARENA_V0PT1_OPENML_SUITE_ID
    immutable_run = {
        "schema_version": 5,
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
        "inner_bag_policy": (
            "Use the requested count unless a classification task's rarest outer-training class is smaller; "
            "then use that smaller valid stratified count so every fitting fold retains at least two rows/class."
        ),
        "protocol_seed": args.protocol_seed,
        "bootstrap_rounds": args.bootstrap_rounds,
        "ensemble_rounds": args.ensemble_rounds or max(1, 2 * len(configs)),
        "pipeline": args.pipeline,
        "standard_pipeline": (
            None
            if args.pipeline == "lite"
            else {
                "identity_context": "all inner-bag fit rows" if args.context_cap == 0 else "stratified capped rows",
                "context_cap": None if args.context_cap == 0 else args.context_cap,
                "requested_train_context_rows": args.train_context_rows,
                "resolved_train_context_rows": configs[0]["train_context_rows"],
                "normal_tabarena_config": STANDARD_TABICL_CONFIG,
            }
        ),
        "guard": {
            "binary": "1 - ROC-AUC",
            "multiclass": "log loss",
            "regression": "MSE",
            "required_relative_improvement": configs[0]["guard_relative_improvement"],
            "scope": "retouche_per_bag_validation_guard_then_test_ensemble",
        },
        "leaderboard_metric": {"binary": "1 - ROC-AUC", "multiclass": "log loss", "regression": "RMSE"},
        "config_labels": labels,
        "configs": configs,
        "classifier_checkpoint_argument": None if args.classifier_checkpoint is None else str(args.classifier_checkpoint.resolve()),
        "regressor_checkpoint_argument": None if args.regressor_checkpoint is None else str(args.regressor_checkpoint.resolve()),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "standard_tabarena_baseline": None if args.skip_standard_baseline else STANDARD_TABICL_CONFIG,
    }
    run_fingerprint_hash = _fingerprint(immutable_run)
    if previous is not None:
        if previous.get("run_fingerprint_sha256") != run_fingerprint_hash or previous.get("immutable_run") != immutable_run:
            previous_run = previous.get("immutable_run")
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
            if not (compatible_code_resume or compatible_hardware_resume):
                if args.allow_equivalent_hardware_resume:
                    raise ValueError(
                        "refusing equivalent-hardware resume: the new allocation differs in model/training "
                        "semantics or stable GPU/software/precision properties; choose a new --output-dir."
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
                "DirectSpline OpenML standard-ensemble experiment"
                if args.pipeline == "standard"
                else "DirectSpline OpenML Lite experiment"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "immutable_run": immutable_run,
            "run_fingerprint_sha256": run_fingerprint_hash,
            "outer_test_policy": "never read by preprocessing fitting, adapter optimisation, guard, HPO, or ensembling",
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
        if not args.skip_standard_baseline:
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
            # Release traceback-owned tensors before attempting the smaller
            # inner-bag fits.  A full-outer baseline OOM must not discard an
            # otherwise viable paired DirectSpline task.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        stage = "config_D"
        cuda_oom_skipped = False
        try:
            for label, config in zip(labels, configs, strict=True):
                stage = f"config_{label}"
                effective_bags = effective_inner_bag_count(task, requested_bags=args.bags)
                print(
                    f"[task={task.task_id} config={label}] starting/recovering "
                    f"{effective_bags}/{args.bags} valid stratified bags",
                    flush=True,
                )
                run_config = run_task_config_standard if args.pipeline == "standard" else run_task_config
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
                task_summary = summarize_task_tuning(
                    task=task,
                    config_labels=labels,
                    output_dir=args.output_dir,
                    ensemble_rounds=immutable_run["ensemble_rounds"],
                    standard_tabarena=standard_tabarena,
                )
            except json.JSONDecodeError:
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
                task_summary = summarize_task_tuning(
                    task=task,
                    config_labels=labels,
                    output_dir=args.output_dir,
                    ensemble_rounds=immutable_run["ensemble_rounds"],
                    standard_tabarena=standard_tabarena,
                )
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
