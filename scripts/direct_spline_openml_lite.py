"""Run the TFM-Retouche-style DirectSpline experiment without TabArena.

Only the small OpenML client is required.  It downloads the public TabArena
v0.1 suite (OpenML suite 457), uses each task's published split
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


def parse_args(*, default_pipeline: str = "lite") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-id", action="append", type=int,
        help="OpenML task ID. Repeat for a pilot; omit to run the public 51-task suite.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Take the first N suite tasks; for smoke tests only.")
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
        default=1024,
        help="Standard pipeline only: sampled labelled-context rows per adapter training episode.",
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
            "With --resume, reuse artifacts after a code-only runtime/metric fix when every experimental "
            "setting is identical. The source transition is recorded separately."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the frozen manifest but do not download task data or run fits.")
    return parser.parse_args()


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
        "launcher": Path(__file__).resolve(),
        "openml_runner": root / "src" / "tabicl" / "_experiments" / "direct_spline_openml.py",
        "standard_openml_runner": root / "src" / "tabicl" / "_experiments" / "direct_spline_openml_standard.py",
        "protocol": root / "src" / "tabicl" / "_experiments" / "tabarena_direct_spline_protocol.py",
        "direct_spline": root / "src" / "tabicl" / "_hyperspline" / "module.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


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


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _same_experimental_semantics(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow an explicit resume across a recorded code-only runtime fix."""
    ignored = {"repository_revision", "source_sha256"}
    return (
        {key: value for key, value in previous.items() if key not in ignored}
        == {key: value for key, value in current.items() if key not in ignored}
    )


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
        elif event_name == "config_started":
            detail = (
                f"using {event['effective_bags']}/{event['requested_bags']} stratified bags"
                if event["effective_bags"] != event["requested_bags"]
                else f"using {event['effective_bags']} bags"
            )
            print(f"[{task_prefix} config={event['config_label']}] {detail}", flush=True)
        elif event_name in {"standard_baseline_started", "standard_baseline_completed"}:
            suffix = (
                "starting normal 8-estimator TabICLv2 baseline"
                if event_name.endswith("started")
                else f"complete: test={event['test']['benchmark_error']:.6g} time={event['elapsed_seconds']:.1f}s"
            )
            print(f"[{task_prefix}] standard baseline: {suffix}", flush=True)
        elif event_name == "task_skipped":
            print(
                f"[{task_prefix} dataset={event['dataset_name']}] skipped: "
                f"features={event['n_features']} exceeds max_features={event['max_features']}",
                flush=True,
            )
        elif event_name.endswith("reused"):
            print(f"[{task_prefix}] reused {event_name.replace('_', ' ')}", flush=True)

    return report


def _configs(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    if args.pipeline == "standard":
        context_cap = None if args.context_cap == 0 else args.context_cap
        default = standard_direct_spline_config(
            context_cap=context_cap,
            train_context_rows=args.train_context_rows,
        )
        random = shared_standard_direct_spline_configs(
            args.n_random_configs,
            seed=args.tuning_seed,
            context_cap=context_cap,
        )
        for config in random:
            config["train_context_rows"] = args.train_context_rows
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
    if args.max_features is not None and args.max_features <= 0:
        raise ValueError("--max-features must be positive")
    if args.context_cap < 0:
        raise ValueError("--context-cap must be zero (all rows) or positive")
    if args.train_context_rows <= 0:
        raise ValueError("--train-context-rows must be positive")
    if min(args.outer_repeat, args.outer_fold, args.outer_sample) < 0:
        raise ValueError("outer repeat/fold/sample values must be non-negative")
    if args.allow_compatible_code_resume and not args.resume:
        raise ValueError("--allow-compatible-code-resume requires --resume")


def main(*, default_pipeline: str = "lite") -> None:
    args = parse_args(default_pipeline=default_pipeline)
    _validate(args)
    labels, configs = _configs(args)
    task_ids = list(args.task_id) if args.task_id else tabarena_v0pt1_task_ids()
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    checkpoint_fingerprints = None if args.dry_run else _resolve_run_checkpoints(args, task_ids)
    manifest_path = args.output_dir / "experiment_manifest.json"
    immutable_run = {
        "schema_version": 3,
        "repository_revision": _git_revision(),
        "source_sha256": _experiment_source_hashes(),
        "python": sys.version,
        "dependencies": _dependency_versions(),
        "data_source": {
            "provider": "OpenML",
            "suite_id": TABARENA_V0PT1_OPENML_SUITE_ID,
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
                "train_context_rows": args.train_context_rows,
                "normal_tabarena_config": STANDARD_TABICL_CONFIG,
            }
        ),
        "guard": {
            "binary": "1 - ROC-AUC",
            "multiclass": "log loss",
            "regression": "MSE",
            "required_relative_improvement": configs[0]["guard_relative_improvement"],
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
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} already exists. Use --resume only for the exact immutable run, "
                "or choose a new --output-dir."
            )
        if previous.get("run_fingerprint_sha256") != run_fingerprint_hash or previous.get("immutable_run") != immutable_run:
            previous_run = previous.get("immutable_run")
            if not (
                args.allow_compatible_code_resume
                and isinstance(previous_run, dict)
                and _same_experimental_semantics(previous_run, immutable_run)
            ):
                raise ValueError(
                    "refusing to resume: the existing output directory has a different immutable run fingerprint; "
                    "choose a new --output-dir."
                )
            prior_hash = str(previous["run_fingerprint_sha256"])
            provenance_path = args.output_dir / "compatible_code_resumes.jsonl"
            with provenance_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "previous_run_fingerprint_sha256": prior_hash,
                            "previous_source_sha256": previous_run.get("source_sha256"),
                            "resumed_source_sha256": immutable_run["source_sha256"],
                            "reason": "explicit compatible code-only resume",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                "Resuming with explicitly recorded compatible code-only changes; "
                f"retaining immutable run fingerprint {prior_hash[:12]}",
                flush=True,
            )
            run_fingerprint_hash = prior_hash
        manifest = previous
    elif args.resume:
        raise FileNotFoundError(f"cannot safely --resume without {manifest_path}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment": (
                "DirectSpline OpenML TabArena-v0.1 standard-ensemble experiment"
                if args.pipeline == "standard"
                else "DirectSpline OpenML TabArena-v0.1 Lite reproduction"
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
        _json_dump(manifest_path, manifest)
    print(f"Using immutable run fingerprint {run_fingerprint_hash[:12]} for {len(task_ids)} task(s): {manifest_path}", flush=True)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, default=str), flush=True)
        return
    device = _resolve_device(args.device)
    progress = _event_reporter(args.output_dir / "progress.jsonl")
    task_summaries: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
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
            _json_dump(
                args.output_dir / "run_progress.json",
                {
                    "completed_task_ids": [item["task_id"] for item in task_summaries],
                    "skipped_tasks": skipped_tasks,
                },
            )
            continue
        standard_tabarena = None
        if not args.skip_standard_baseline:
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
        for label, config in zip(labels, configs, strict=True):
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
        task_summary = summarize_task_tuning(
            task=task,
            config_labels=labels,
            output_dir=args.output_dir,
            ensemble_rounds=immutable_run["ensemble_rounds"],
            standard_tabarena=standard_tabarena,
        )
        task_summaries.append(task_summary)
        _json_dump(
            args.output_dir / "run_progress.json",
            {
                "completed_task_ids": [item["task_id"] for item in task_summaries],
                "skipped_tasks": skipped_tasks,
            },
        )
    summary = summarize_experiment(
        task_summaries=task_summaries,
        output_dir=args.output_dir,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.protocol_seed,
        skipped_tasks=skipped_tasks,
        task_eligibility=immutable_run["task_eligibility"],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
