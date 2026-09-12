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

The default task bank is all 51 tasks.  On the 30 binary tasks the declared
deployment rule is an exact fallback to ordinary TabICLv2; the spline adapter
is evaluated only on the 21 supported tasks (8 multiclass, 13 regression).
This yields a complete benchmark row without imputing failed spline runs or
claiming that DirectSpline improves binary classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from direct_spline_openml_crossfit_context_expansion import (
    CONTEXT_EXPANSION_SCHEMA_VERSION,
    _comparison,
    _run_task,
    _write_csv,
)
from direct_spline_openml_support_audit import SourceCase, _canonical_json, _sha256, _write_json
from tabicl._experiments.direct_spline_openml import (
    _metric_bundle,
    _safe_name,
    _standard_baseline_dir,
    load_tabarena_openml_task,
    run_standard_tabarena_baseline,
)
from tabicl._experiments.direct_spline_openml import _paired_comparison_summary
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
TABARENA_LITE_BINARY_TASK_IDS = (
    363613,
    363616,
    363618,
    363619,
    363620,
    363621,
    363623,
    363624,
    363626,
    363627,
    363628,
    363629,
    363630,
    363632,
    363671,
    363673,
    363674,
    363676,
    363679,
    363681,
    363682,
    363683,
    363684,
    363689,
    363691,
    363694,
    363696,
    363700,
    363706,
    363712,
)
TABARENA_LITE_ALL_TASK_IDS = tuple(sorted(TABARENA_LITE_SUPPORTED_TASK_IDS + TABARENA_LITE_BINARY_TASK_IDS))


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.bags < 2:
        raise ValueError("--bags must be at least two")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    task_ids = list(TABARENA_LITE_ALL_TASK_IDS if args.task_id is None else args.task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("--task-id values must be unique")
    unsupported = sorted(set(task_ids) - set(TABARENA_LITE_ALL_TASK_IDS))
    if unsupported:
        raise ValueError(f"tasks are not members of TabArena-Lite suite 457: {unsupported}")
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
        "adapter_problem_type_scope": ["multiclass", "regression"],
        "binary_deployment_rule": "exact ordinary full-outer-training TabICLv2 fallback; no spline is trained",
        "protocol_seed": int(args.protocol_seed),
        "requested_bags": int(args.bags),
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


def _summary(
    task_summaries: list[dict[str, Any]], binary_summaries: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    methods = ("identity", "raw_spline", "selected_blend")
    return {
        "context_expansion_schema_version": CONTEXT_EXPANSION_SCHEMA_VERSION,
        "n_tasks": len(task_summaries) + len(binary_summaries),
        "n_adapted_tasks": len(task_summaries),
        "n_binary_identity_fallback_tasks": len(binary_summaries),
        "task_summaries": task_summaries,
        "binary_identity_fallback_summaries": binary_summaries,
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
        "full_suite_composite_vs_full_training_tabiclv2": _full_suite_comparison(
            task_summaries, binary_summaries, args
        ),
        "retouche_comparison_note": "These are the same 51 TabArena-Lite tasks, outer split, and task metrics used by TFM-Retouche. Absolute published-pool Elo is computed after exporting these per-task errors to the TabArena evaluator; it is not a two-method win/loss Elo.",
    }


def _full_suite_comparison(
    task_summaries: list[dict[str, Any]], binary_summaries: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any] | None:
    records = [
        {
            "problem_type": item["problem_type"],
            "reference": item["source_full_outer_training_tabiclv2"]["benchmark_error"],
            "candidate": item["expanded_context"]["outer_test"]["selected_blend"]["benchmark_error"],
        }
        for item in task_summaries
    ] + [
        {
            "problem_type": "binary",
            "reference": item["full_tabiclv2_benchmark_error"],
            "candidate": item["full_tabiclv2_benchmark_error"],
        }
        for item in binary_summaries
    ]
    if not records:
        return None
    return _paired_comparison_summary(
        reference=np.asarray([item["reference"] for item in records], dtype=float),
        candidate=np.asarray([item["candidate"] for item in records], dtype=float),
        problem_types=np.asarray([item["problem_type"] for item in records], dtype=object),
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed + 500,
        reference_label="full_outer_training_tabiclv2",
        candidate_label="binary_identity_else_expanded_directspline",
    )


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    config = frozen_config()
    fingerprint = _prepare_output(args.output_dir, _manifest(args, config), resume=bool(args.resume))
    source_dir = args.output_dir / "full_training_reference"
    task_summaries: list[dict[str, Any]] = []
    binary_summaries: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for position, task_id in enumerate(args.task_id, start=1):
        print(f"[{position}/{len(args.task_id)}] loading TabArena-Lite task {task_id}", flush=True)
        task = load_tabarena_openml_task(task_id, outer_repeat=0, outer_fold=0, outer_sample=0)
        run_standard_tabarena_baseline(
            task=task,
            output_dir=source_dir,
            device=device,
            classifier_checkpoint=args.classifier_checkpoint,
            regressor_checkpoint=args.regressor_checkpoint,
            resume=bool(args.resume),
            run_fingerprint_hash=fingerprint,
        )
        if task.problem_type == "binary":
            prediction_path = _standard_baseline_dir(source_dir, task) / "predictions.npz"
            with np.load(prediction_path, allow_pickle=False) as payload:
                prediction = np.asarray(payload["prediction"], dtype=float)
            binary_summaries.append(
                {
                    "task_id": task.task_id,
                    "dataset_id": task.dataset_id,
                    "dataset_name": task.dataset_name,
                    "problem_type": "binary",
                    "outer_split_hash": task.outer_split_hash,
                    "deployment": "exact_full_training_tabiclv2_identity_fallback",
                    "full_tabiclv2_benchmark_error": _metric_bundle(
                        "binary", task.y_test, prediction, task.n_classes
                    )["benchmark_error"],
                }
            )
            binary_summaries.sort(key=lambda item: int(item["task_id"]))
            _write_json(args.output_dir / "binary_identity_fallback_summaries.json", binary_summaries)
            _write_json(args.output_dir / "summary.json", _summary(task_summaries, binary_summaries, args))
            continue
        if task.problem_type not in {"multiclass", "regression"}:
            raise ValueError(f"task {task_id} resolved to unknown problem type {task.problem_type!r}")
        case = _source_case(task=task, source_dir=source_dir, config=config, bags=args.bags)
        task_summaries.append(
            _run_task(case=case, task=task, output_dir=args.output_dir, args=args, fingerprint=fingerprint)
        )
        task_summaries.sort(key=lambda item: int(item["task_id"]))
        _write_json(args.output_dir / "task_summaries.json", task_summaries)
        _write_json(args.output_dir / "summary.json", _summary(task_summaries, binary_summaries, args))

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
    for item in binary_summaries:
        rows.append(
            {
                "task_id": item["task_id"],
                "dataset_name": item["dataset_name"],
                "problem_type": "binary",
                "full_tabiclv2_benchmark_error": item["full_tabiclv2_benchmark_error"],
                "composite_directspline_benchmark_error": item["full_tabiclv2_benchmark_error"],
                "deployment": item["deployment"],
            }
        )
    for row in rows:
        if row.get("problem_type") != "binary":
            row["composite_directspline_benchmark_error"] = row["expanded_selected_blend_benchmark_error"]
            row["deployment"] = "expanded_directspline_selected_blend"
    rows.sort(key=lambda item: int(item["task_id"]))
    _write_csv(args.output_dir / "task_results.csv", rows)
    print(
        json.dumps(
            {
                "n_tasks": len(task_summaries) + len(binary_summaries),
                "n_adapted_tasks": len(task_summaries),
                "n_binary_identity_fallback_tasks": len(binary_summaries),
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
