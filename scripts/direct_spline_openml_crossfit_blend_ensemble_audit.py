"""Measure how test-time averaging changes a completed DirectSpline result.

This CPU-only diagnostic uses the archived ``bag_*.npz`` prediction arrays.
For each of the eight bags, it compares both independently selected spline
states with that bag's identity prediction.  It then compares those individual
outcomes with the deployed prediction, which averages the sixteen spline
states and the eight identity bag predictions.

The question is diagnostic rather than deployable: does averaging turn a set
of individually worse spline predictions into a better final spline ensemble?
Outer-test labels are used only after all archived predictions are read.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_crossfit_blend_ensemble_audit.py \
      --crossfit-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_crossfit_blend/multiclass_D_260904 \
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_crossfit_blend_ensemble_audit/multiclass_D_260904
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from direct_spline_openml_crossfit_blend import (
    CROSSFIT_BLEND_SCHEMA_VERSION,
    _assemble_task_predictions,
    _load_bag,
    _load_source_task,
)
from direct_spline_openml_support_audit import (
    SourceCase,
    _canonical_json,
    _find_source_cases,
    _load_json,
    _sha256,
    _write_json,
)
from tabicl._experiments.direct_spline_openml import (
    OpenMLTaskData,
    _metric_bundle,
    _safe_name,
)


ENSEMBLE_AUDIT_SCHEMA_VERSION = 1
_TIE_ATOL = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crossfit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _task_dir(crossfit_dir: Path, task: OpenMLTaskData) -> Path:
    return crossfit_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _outcome(reference: float, candidate: float) -> str:
    difference = float(candidate) - float(reference)
    if difference < -_TIE_ATOL:
        return "win"
    if difference > _TIE_ATOL:
        return "loss"
    return "tie"


def _mean_metrics(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("need at least one metric bundle")
    return {
        key: float(np.mean([float(item[key]) for item in metrics]))
        for key in ("deployment_error", "benchmark_error")
    }


def _audit_task(*, case: SourceCase, task: OpenMLTaskData, crossfit_dir: Path) -> dict[str, Any]:
    task_dir = _task_dir(crossfit_dir, task)
    stored = _load_json(task_dir / "task_summary.json", label="crossfit task summary")
    effective_bags = int(stored["effective_bags"])
    paths = [task_dir / f"bag_{bag}.npz" for bag in range(effective_bags)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bags for task {task.task_id}: {missing}")
    bags = [_load_bag(path) for path in paths]
    identity_oof, spline_oof, _units, identity_ensemble, spline_ensemble = _assemble_task_predictions(
        task=task, bag_results=bags
    )

    member_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for bag in bags:
        member_pairs.append((bag.identity_test, bag.spline_selected_on_a_test))
        member_pairs.append((bag.identity_test, bag.spline_selected_on_b_test))
    identity_members = [_metric_bundle(task.problem_type, task.y_test, identity, task.n_classes) for identity, _ in member_pairs]
    spline_members = [_metric_bundle(task.problem_type, task.y_test, spline, task.n_classes) for _, spline in member_pairs]
    identity_member_mean = _mean_metrics(identity_members)
    spline_member_mean = _mean_metrics(spline_members)
    identity_ensemble_metrics = _metric_bundle(task.problem_type, task.y_test, identity_ensemble, task.n_classes)
    spline_ensemble_metrics = _metric_bundle(task.problem_type, task.y_test, spline_ensemble, task.n_classes)
    oof_identity = _metric_bundle(task.problem_type, task.y_train, identity_oof, task.n_classes)
    oof_spline = _metric_bundle(task.problem_type, task.y_train, spline_oof, task.n_classes)

    return {
        "task_id": int(task.task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "effective_bags": effective_bags,
        "recorded_blend_alpha": float(stored["blend_selection"]["selected_alpha"]),
        "oof_raw": {
            "identity": oof_identity,
            "spline": oof_spline,
            "outcome": _outcome(oof_identity["deployment_error"], oof_spline["deployment_error"]),
        },
        "test_mean_individual": {
            "identity": identity_member_mean,
            "spline": spline_member_mean,
            "outcome": _outcome(identity_member_mean["deployment_error"], spline_member_mean["deployment_error"]),
        },
        "test_ensemble": {
            "identity": identity_ensemble_metrics,
            "spline": spline_ensemble_metrics,
            "outcome": _outcome(
                identity_ensemble_metrics["deployment_error"], spline_ensemble_metrics["deployment_error"]
            ),
        },
        "test_ensemble_minus_mean_individual": {
            "identity_deployment_error": float(identity_ensemble_metrics["deployment_error"] - identity_member_mean["deployment_error"]),
            "spline_deployment_error": float(spline_ensemble_metrics["deployment_error"] - spline_member_mean["deployment_error"]),
            "relative_spline_advantage": float(
                (spline_ensemble_metrics["deployment_error"] - identity_ensemble_metrics["deployment_error"])
                - (spline_member_mean["deployment_error"] - identity_member_mean["deployment_error"])
            ),
        },
    }


def _transition_counts(items: Sequence[Mapping[str, Any]], *, left: str, right: str) -> dict[str, int]:
    counts = {f"{first}_to_{second}": 0 for first in ("win", "tie", "loss") for second in ("win", "tie", "loss")}
    for item in items:
        first = str(item[left]["outcome"])
        second = str(item[right]["outcome"])
        counts[f"{first}_to_{second}"] += 1
    return counts


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(*, crossfit_dir: Path, crossfit_manifest: Mapping[str, Any], source_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ensemble_audit_schema_version": ENSEMBLE_AUDIT_SCHEMA_VERSION,
        "crossfit_dir": str(crossfit_dir),
        "crossfit_manifest_sha256": _sha256(crossfit_dir / "experiment_manifest.json"),
        "crossfit_blend_schema_version": crossfit_manifest.get("crossfit_blend_schema_version"),
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "config_label": str(args.config_label),
        "task_ids": None if args.task_id is None else sorted(set(int(item) for item in args.task_id)),
        "outer_test_label_policy": "diagnostic only: outer test labels score archived individual and ensemble predictions",
        "script_sha256": _sha256(Path(__file__)),
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing ensemble audit manifest")
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different ensemble audit")
        if not resume:
            raise ValueError("ensemble audit output directory already exists; pass --resume to rewrite it")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def main() -> None:
    args = _parse_args()
    crossfit_dir = args.crossfit_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    crossfit_manifest = _load_json(crossfit_dir / "experiment_manifest.json", label="crossfit manifest")
    if crossfit_manifest.get("crossfit_blend_schema_version") != CROSSFIT_BLEND_SCHEMA_VERSION:
        raise ValueError("unsupported crossfit blend artifact schema")
    source_value = crossfit_manifest.get("source_dir")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("crossfit manifest has no source_dir")
    source_dir = Path(source_value).resolve()
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=None if args.task_id is None else set(int(item) for item in args.task_id),
    )
    manifest = _manifest(crossfit_dir=crossfit_dir, crossfit_manifest=crossfit_manifest, source_dir=source_dir, args=args)
    _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))

    task_summaries: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        task = _load_source_task(case=case, immutable_run=immutable_run)
        item = _audit_task(case=case, task=task, crossfit_dir=crossfit_dir)
        task_summaries.append(item)
        print(
            f"[{index}/{len(cases)}] task {case.task_id}: "
            f"OOF={item['oof_raw']['outcome']} individual={item['test_mean_individual']['outcome']} "
            f"ensemble={item['test_ensemble']['outcome']}",
            flush=True,
        )

    rows = []
    for item in task_summaries:
        rows.append({
            "task_id": item["task_id"],
            "dataset_id": item["dataset_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "recorded_blend_alpha": item["recorded_blend_alpha"],
            "oof_raw_outcome": item["oof_raw"]["outcome"],
            "individual_outcome": item["test_mean_individual"]["outcome"],
            "ensemble_outcome": item["test_ensemble"]["outcome"],
            "individual_identity_deployment_error": item["test_mean_individual"]["identity"]["deployment_error"],
            "individual_spline_deployment_error": item["test_mean_individual"]["spline"]["deployment_error"],
            "ensemble_identity_deployment_error": item["test_ensemble"]["identity"]["deployment_error"],
            "ensemble_spline_deployment_error": item["test_ensemble"]["spline"]["deployment_error"],
            "relative_spline_ensemble_advantage": item["test_ensemble_minus_mean_individual"]["relative_spline_advantage"],
        })
    _write_csv(args.output_dir / "task_results.csv", rows)
    summary = {
        "ensemble_audit_schema_version": ENSEMBLE_AUDIT_SCHEMA_VERSION,
        "exploratory": True,
        "n_tasks": len(task_summaries),
        "all_tasks": {
            "oof_to_ensemble": _transition_counts(task_summaries, left="oof_raw", right="test_ensemble"),
            "individual_to_ensemble": _transition_counts(task_summaries, left="test_mean_individual", right="test_ensemble"),
        },
        "recorded_identity_selection": {
            "n_tasks": int(sum(item["recorded_blend_alpha"] == 0.0 for item in task_summaries)),
            "oof_to_ensemble": _transition_counts(
                [item for item in task_summaries if item["recorded_blend_alpha"] == 0.0],
                left="oof_raw",
                right="test_ensemble",
            ),
            "individual_to_ensemble": _transition_counts(
                [item for item in task_summaries if item["recorded_blend_alpha"] == 0.0],
                left="test_mean_individual",
                right="test_ensemble",
            ),
        },
        "task_summaries": task_summaries,
    }
    _write_json(args.output_dir / "task_summaries.json", task_summaries)
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(task_summaries), "all_tasks": summary["all_tasks"]}), flush=True)


if __name__ == "__main__":
    main()
