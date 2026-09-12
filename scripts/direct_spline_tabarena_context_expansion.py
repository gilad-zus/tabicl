"""Evaluate the frozen DirectSpline context-expansion method on TabArena-Lite.

This is the publication-facing comparison run for the problem types on which
the method was frozen: multiclass classification and regression.  It uses the
official OpenML suite-457 outer split, ordinary full-outer-training TabICLv2
as the end-to-end reference, and the same eight-bag A/B cross-fitted context
expansion that was evaluated on the held-out confirmation banks.

Unlike ``direct_spline_openml_crossfit_context_expansion.py``, this launcher
does not require a completed ordinary DirectSpline source run.  That source
run was needed while developing the diagnostic comparison, but its learned
states are not reused: context expansion deliberately replays the trajectory
to retain the two independently selected A/B checkpoints.  Here we therefore
compute only the full TabICLv2 reference and run the frozen expansion once.

The default task bank is the 21 supported TabArena-Lite tasks (8 multiclass,
13 regression). Existing full-training TabICLv2 predictions can be supplied
as reusable baseline sources, avoiding repeated reference inference. Training
episodes have a fixed context ceiling for large tables; deployment contexts
remain uncapped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from direct_spline_openml_crossfit_context_expansion import (
    CONTEXT_EXPANSION_SCHEMA_VERSION,
    _comparison,
    _run_task,
    _write_csv,
)
from direct_spline_openml_support_audit import SourceCase, _canonical_json, _sha256, _write_json
from tabicl._experiments.direct_spline_openml import (
    _standard_baseline_dir,
    load_tabarena_openml_task,
    run_standard_tabarena_baseline,
)
from tabicl._experiments.direct_spline_openml_standard import standard_direct_spline_config


TABARENA_LITE_MULTICLASS_TASK_IDS = (363614, 363677, 363685, 363699, 363702, 363704, 363707, 363711)
TABARENA_LITE_REGRESSION_TASK_IDS = (
    363612,
    363615,
    363625,
    363631,
    363672,
    363675,
    363678,
    363686,
    363693,
    363697,
    363698,
    363705,
    363708,
)
TABARENA_LITE_SUPPORTED_TASK_IDS = TABARENA_LITE_MULTICLASS_TASK_IDS + TABARENA_LITE_REGRESSION_TASK_IDS


def frozen_config() -> dict[str, Any]:
    """Return the configuration frozen before the held-out 10+10 evaluation."""

    config = standard_direct_spline_config(
        context_cap=None,
        train_context_rows=None,
        adapter_steps=500,
        validation_interval=10,
    )
    config.update(
        {
            "random_state": 0,
            "adapter_patience": 10,
            "guard_relative_improvement": 0.005,
            "identity_regularization": 0.0,
            "adapter_architecture": "fixed_cubic",
        }
    )
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", help="Optional TabArena-Lite task ID; repeatable.")
    parser.add_argument("--protocol-seed", type=int, default=20260910)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument(
        "--train-context-cap",
        type=int,
        default=16_384,
        help="Maximum labelled context rows per training episode; deployment still uses every fold row.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--baseline-source-dir",
        type=Path,
        action="append",
        default=[],
        help="Earlier run containing reusable standard_tabarena_baseline predictions; repeatable.",
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.bags < 2:
        raise ValueError("--bags must be at least two")
    if args.train_context_cap < 1:
        raise ValueError("--train-context-cap must be positive")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    task_ids = list(TABARENA_LITE_SUPPORTED_TASK_IDS if args.task_id is None else args.task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("--task-id values must be unique")
    unsupported = sorted(set(task_ids) - set(TABARENA_LITE_SUPPORTED_TASK_IDS))
    if unsupported:
        raise ValueError(f"tasks are not multiclass/regression members of TabArena-Lite: {unsupported}")
    args.task_id = task_ids
    # ``_run_task`` shares its implementation with the optional diagnostic
    # replay launcher.  This publication run deliberately has no old replay.
    args.reference_crossfit_dir = None
    args.reference_atol = 1e-8
    return args


def _manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    imported_script = Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")
    return {
        "context_expansion_schema_version": CONTEXT_EXPANSION_SCHEMA_VERSION,
        "experiment": "Frozen DirectSpline A/B context expansion on TabArena-Lite multiclass and regression",
        "task_ids": sorted(args.task_id),
        "suite_id": 457,
        "outer_split": {"repeat": 0, "fold": 0, "sample": 0},
        "problem_type_scope": ["multiclass", "regression"],
        "baseline_source_dirs": [str(path.resolve()) for path in args.baseline_source_dir],
        "protocol_seed": int(args.protocol_seed),
        "requested_bags": int(args.bags),
        "training_context_policy": {
            "maximum_rows_per_episode": int(args.train_context_cap),
            "deployment_context_cap": None,
            "purpose": "bound differentiable-backbone memory on large TabArena tables",
        },
        "config_label": "D",
        "frozen_source_config": config,
        "effective_training_note": "The A/B experiment disables early termination and independently selects two checkpoints from the full frozen 500-step trajectory, exactly as in the held-out confirmation run.",
        "full_reference": "ordinary public TabICLv2 fit on every outer-training row",
        "selection": "Each context condition selects its blend alpha from cross-fitted OOF predictions; outer-test labels are report-only.",
        "script_sha256": _sha256(Path(__file__)),
        "context_expansion_script_sha256": _sha256(imported_script),
    }


def _prepare_output(output_dir: Path, manifest: dict[str, Any], *, resume: bool) -> str:
    fingerprint = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    payload = {**manifest, "run_fingerprint_sha256": fingerprint}
    path = output_dir / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical_json(existing) != _canonical_json(payload):
            raise ValueError("output directory belongs to a different immutable experiment")
        if not resume:
            raise ValueError("output directory exists; pass --resume to reuse matching artifacts")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return fingerprint


def _source_case(*, task: Any, source_dir: Path, config: dict[str, Any], bags: int) -> SourceCase:
    placeholder = source_dir / "standalone_source_metadata"
    return SourceCase(
        source_dir=source_dir,
        config_dir=placeholder,
        config_summary_path=placeholder / "config_summary.json",
        config_predictions_path=placeholder / "config_predictions.npz",
        task_id=int(task.task_id),
        dataset_id=int(task.dataset_id),
        dataset_name=str(task.dataset_name),
        problem_type=str(task.problem_type),
        n_classes=task.n_classes,
        outer_split_hash=str(task.outer_split_hash),
        config_label="D",
        config=dict(config),
        requested_bags=int(bags),
        effective_bags=int(bags),
    )


def _summary(task_summaries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    methods = ("identity", "raw_spline", "selected_blend")
    return {
        "context_expansion_schema_version": CONTEXT_EXPANSION_SCHEMA_VERSION,
        "n_tasks": len(task_summaries),
        "task_summaries": task_summaries,
        "context_effect_vs_original": {
            method: _comparison(
                task_summaries=task_summaries,
                candidate_condition="expanded",
                candidate_method=method,
                reference_condition="original",
                reference_method=method,
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.bootstrap_seed + index,
            )
            for index, method in enumerate(methods)
        },
        "end_to_end_vs_full_training_tabiclv2": {
            f"{condition}_{method}": _comparison(
                task_summaries=task_summaries,
                candidate_condition=condition,
                candidate_method=method,
                reference_condition=None,
                reference_method=None,
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.bootstrap_seed + 100 + index,
            )
            for index, (condition, method) in enumerate(
                (condition, method) for condition in ("original", "expanded") for method in methods
            )
        },
        "retouche_comparison_note": "These are the TabArena-Lite multiclass/regression tasks, outer split, and task metrics used by TFM-Retouche. Binary fallback is a scoring-time composition with TabICLv2 and requires no rerun. Absolute published-pool Elo still requires the TabArena evaluator.",
    }


def _existing_baseline_source(task: Any, source_dirs: list[Path]) -> Path | None:
    for source_dir in source_dirs:
        if (_standard_baseline_dir(source_dir, task) / "predictions.npz").is_file():
            return source_dir
    return None


def _effective_config(config: dict[str, Any], *, train_context_cap: int) -> dict[str, Any]:
    effective = dict(config)
    effective["train_context_rows"] = int(train_context_cap)
    return effective


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    frozen_source_config = frozen_config()
    config = _effective_config(frozen_source_config, train_context_cap=args.train_context_cap)
    args.baseline_source_dir = [path.resolve() for path in args.baseline_source_dir]
    fingerprint = _prepare_output(args.output_dir, _manifest(args, config), resume=bool(args.resume))
    generated_source_dir = args.output_dir / "full_training_reference"
    task_summaries: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for position, task_id in enumerate(args.task_id, start=1):
        print(f"[{position}/{len(args.task_id)}] loading TabArena-Lite task {task_id}", flush=True)
        task = load_tabarena_openml_task(task_id, outer_repeat=0, outer_fold=0, outer_sample=0)
        if task.problem_type not in {"multiclass", "regression"}:
            raise ValueError(f"task {task_id} resolved to unsupported problem type {task.problem_type!r}")
        source_dir = _existing_baseline_source(task, args.baseline_source_dir)
        reused_baseline = source_dir is not None
        if source_dir is None:
            source_dir = generated_source_dir
            run_standard_tabarena_baseline(
                task=task,
                output_dir=source_dir,
                device=device,
                classifier_checkpoint=args.classifier_checkpoint,
                regressor_checkpoint=args.regressor_checkpoint,
                resume=bool(args.resume),
                run_fingerprint_hash=fingerprint,
            )
        case = _source_case(task=task, source_dir=source_dir, config=config, bags=args.bags)
        task_summary = _run_task(
            case=case, task=task, output_dir=args.output_dir, args=args, fingerprint=fingerprint
        )
        baseline_prediction = _standard_baseline_dir(source_dir, task) / "predictions.npz"
        task_summary["full_tabiclv2_prediction_source"] = {
            "directory": str(source_dir),
            "reused": reused_baseline,
            "prediction_sha256": _sha256(baseline_prediction),
        }
        task_summaries.append(task_summary)
        task_summaries.sort(key=lambda item: int(item["task_id"]))
        _write_json(args.output_dir / "task_summaries.json", task_summaries)
        _write_json(args.output_dir / "summary.json", _summary(task_summaries, args))

    rows = []
    for item in task_summaries:
        row = {
            "task_id": item["task_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "full_tabiclv2_benchmark_error": item["source_full_outer_training_tabiclv2"]["benchmark_error"],
        }
        for condition in ("original", "expanded"):
            row[f"{condition}_selected_alpha"] = item[f"{condition}_context"]["blend_selection"]["selected_alpha"]
            for method in ("identity", "raw_spline", "selected_blend"):
                row[f"{condition}_{method}_benchmark_error"] = item[f"{condition}_context"]["outer_test"][method]["benchmark_error"]
        rows.append(row)
    for row in rows:
        row["composite_directspline_benchmark_error"] = row["expanded_selected_blend_benchmark_error"]
        row["deployment"] = "expanded_directspline_selected_blend"
    rows.sort(key=lambda item: int(item["task_id"]))
    _write_csv(args.output_dir / "task_results.csv", rows)
    print(
        json.dumps(
            {
                "n_tasks": len(task_summaries),
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
