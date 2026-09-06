"""Audit validation rules for a completed cross-fitted DirectSpline blend run.

This is a CPU-only, retrospective analysis.  It never trains an adapter or
alters a saved prediction.  It reconstructs the independent A/B validation
predictions from the completed ``bag_*.npz`` files and compares three rules
for choosing the table-level blend weight from ``{0, .25, .5, .75, 1}``:

* the recorded mean-relative-loss one-standard-error rule;
* the minimum mean-relative-loss rule; and
* the minimum pooled OOF loss rule.

The final rule gives each validation row the same weight in the objective:

    alpha = argmin_a loss(y_oof, (1 - a) identity_oof + a spline_oof).

Outer-test labels are used only after all three rule decisions are fixed.
Because the completed run's outer-test results are already known, this audit
is exploratory: it can identify a candidate selection policy, which must be
confirmed on fresh outer splits before it becomes a reported method.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_crossfit_blend_selection_audit.py \
      --crossfit-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_crossfit_blend/multiclass_D_260904 \
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_crossfit_blend_selection_audit/multiclass_D_260904
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
    _ALPHA_GRID,
    _alpha_selection,
    _assemble_task_predictions,
    _blend_prediction,
    _load_bag,
    _load_source_task,
    _source_standard_prediction,
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
    _paired_comparison_summary,
    _safe_name,
)
from tabicl._experiments.direct_spline_protocol import deployment_error


SELECTION_AUDIT_SCHEMA_VERSION = 1
_TIE_ATOL = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crossfit-dir", type=Path, required=True, help="Completed cross-fitted blend result directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--task-id", type=int, action="append", help="Audit only these task IDs. Repeatable.")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    return args


def _argmin_record(records: Sequence[Mapping[str, Any]], *, key: str) -> dict[str, Any]:
    if not records:
        raise ValueError("need at least one alpha record")
    return dict(min(records, key=lambda item: (float(item[key]), float(item["alpha"]))))


def _pooled_oof_selection(
    *,
    task: OpenMLTaskData,
    identity_oof: np.ndarray,
    spline_oof: np.ndarray,
    alphas: Sequence[float] = _ALPHA_GRID,
) -> dict[str, Any]:
    """Choose alpha by the same full-row loss used for validation reporting."""
    records: list[dict[str, float]] = []
    for alpha in alphas:
        candidate = _blend_prediction(identity_oof, spline_oof, float(alpha))
        error = deployment_error(task.problem_type, task.y_train, candidate, n_classes=task.n_classes)
        records.append({"alpha": float(alpha), "pooled_oof_deployment_error": float(error)})
    best = _argmin_record(records, key="pooled_oof_deployment_error")
    return {
        "selection_rule": "smallest alpha at the minimum pooled cross-fitted OOF deployment loss",
        "candidate_alphas": records,
        "selected_alpha": float(best["alpha"]),
    }


def _task_dir(crossfit_dir: Path, task: OpenMLTaskData) -> Path:
    return crossfit_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _load_crossfit_predictions(
    *,
    crossfit_dir: Path,
    task: OpenMLTaskData,
    effective_bags: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    task_dir = _task_dir(crossfit_dir, task)
    paths = [task_dir / f"bag_{bag}.npz" for bag in range(effective_bags)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing crossfit bag artifacts for task {task.task_id}: {missing}")
    result = _assemble_task_predictions(task=task, bag_results=[_load_bag(path) for path in paths])
    saved_path = task_dir / "task_predictions.npz"
    if saved_path.is_file():
        with np.load(saved_path, allow_pickle=False) as saved:
            for key, actual in (("identity_oof", result[0]), ("spline_oof", result[1]), ("identity_test", result[3]), ("spline_test", result[4])):
                expected = np.asarray(saved[key], dtype=float)
                if expected.shape != actual.shape or not np.allclose(expected, actual, atol=1e-12, rtol=0.0):
                    raise ValueError(f"saved {key} does not match reconstructed bag predictions for task {task.task_id}")
    return result


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing selection audit manifest")
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different immutable selection audit; choose a new --output-dir")
        if not resume:
            raise ValueError("selection audit output directory already exists; pass --resume to rewrite it")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def _selection_audit_manifest(
    *, crossfit_dir: Path, crossfit_manifest: Mapping[str, Any], source_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "selection_audit_schema_version": SELECTION_AUDIT_SCHEMA_VERSION,
        "crossfit_dir": str(crossfit_dir),
        "crossfit_manifest_sha256": _sha256(crossfit_dir / "experiment_manifest.json"),
        "crossfit_blend_schema_version": crossfit_manifest.get("crossfit_blend_schema_version"),
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "config_label": str(args.config_label),
        "task_ids": None if args.task_id is None else sorted(set(int(item) for item in args.task_id)),
        "alpha_grid": list(_ALPHA_GRID),
        "policies": {
            "recorded_one_se_mean_relative": "smallest alpha within one SE of best mean half-relative loss",
            "minimum_mean_relative": "smallest alpha at minimum mean half-relative loss",
            "minimum_pooled_oof_loss": "smallest alpha at minimum full-row pooled OOF log loss or MSE",
        },
        "outer_test_label_policy": "exploratory only: test labels score already-known completed predictions after all three policy alphas are selected",
        "script_sha256": _sha256(Path(__file__)),
    }


def _comparison(
    *,
    task_summaries: Sequence[Mapping[str, Any]],
    method: str,
    reference: str,
    bootstrap_rounds: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = [item for item in task_summaries if item["errors"].get(reference) is not None]
    return _paired_comparison_summary(
        reference=np.asarray([item["errors"][reference] for item in selected], dtype=float),
        candidate=np.asarray([item["errors"][method] for item in selected], dtype=float),
        problem_types=np.asarray([item["problem_type"] for item in selected], dtype=object),
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=bootstrap_seed,
        reference_label=reference,
        candidate_label=method,
    )


def _run_task(*, case: SourceCase, task: OpenMLTaskData, crossfit_dir: Path) -> dict[str, Any]:
    task_summary_path = _task_dir(crossfit_dir, task) / "task_summary.json"
    stored = _load_json(task_summary_path, label="crossfit task summary")
    if int(stored["effective_bags"]) != case.effective_bags:
        raise ValueError(f"crossfit/source effective bag count mismatch for task {task.task_id}")
    identity_oof, spline_oof, units, identity_test, spline_test = _load_crossfit_predictions(
        crossfit_dir=crossfit_dir,
        task=task,
        effective_bags=int(stored["effective_bags"]),
    )

    current = _alpha_selection(task=task, units=units, alphas=_ALPHA_GRID)
    stored_current = float(stored["blend_selection"]["selected_alpha"])
    if float(current["selected_alpha"]) != stored_current:
        raise ValueError(f"cannot reproduce recorded alpha for task {task.task_id}")
    min_relative = _argmin_record(current["candidate_alphas"], key="mean_relative_error_change")
    pooled = _pooled_oof_selection(task=task, identity_oof=identity_oof, spline_oof=spline_oof)
    policy_alphas = {
        "recorded_one_se_mean_relative": float(current["selected_alpha"]),
        "minimum_mean_relative": float(min_relative["alpha"]),
        "minimum_pooled_oof_loss": float(pooled["selected_alpha"]),
        "raw_spline": 1.0,
    }

    # From this point onward the fixed outer-test labels are used only to score
    # the policies selected above.
    predictions = {
        "matched_inner_bag_identity": identity_test,
        **{name: _blend_prediction(identity_test, spline_test, alpha) for name, alpha in policy_alphas.items()},
    }
    full_standard = _source_standard_prediction(source_dir=case.source_dir, task=task)
    if full_standard is not None:
        predictions["source_full_outer_training_tabiclv2"] = full_standard
    errors = {
        name: float(_metric_bundle(task.problem_type, task.y_test, prediction, task.n_classes)["benchmark_error"])
        for name, prediction in predictions.items()
    }
    return {
        "task_id": int(task.task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "effective_bags": int(stored["effective_bags"]),
        "selection": {
            "recorded_one_se_mean_relative": current,
            "minimum_mean_relative": {
                "selection_rule": "smallest alpha at the minimum mean relative held-out loss",
                "selected_alpha": float(min_relative["alpha"]),
                "candidate_alphas": current["candidate_alphas"],
            },
            "minimum_pooled_oof_loss": pooled,
        },
        "policy_alphas": policy_alphas,
        "oof": {
            "identity": _metric_bundle(task.problem_type, task.y_train, identity_oof, task.n_classes),
            "spline": _metric_bundle(task.problem_type, task.y_train, spline_oof, task.n_classes),
        },
        "errors": errors,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty task table")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    requested_ids = None if args.task_id is None else set(int(item) for item in args.task_id)
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=requested_ids,
    )
    manifest = _selection_audit_manifest(
        crossfit_dir=crossfit_dir, crossfit_manifest=crossfit_manifest, source_dir=source_dir, args=args
    )
    _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))

    task_summaries: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        task = _load_source_task(case=case, immutable_run=immutable_run)
        result = _run_task(case=case, task=task, crossfit_dir=crossfit_dir)
        task_summaries.append(result)
        print(
            f"[{index}/{len(cases)}] task {case.task_id} {case.dataset_name}: "
            f"current={result['policy_alphas']['recorded_one_se_mean_relative']}, "
            f"relative={result['policy_alphas']['minimum_mean_relative']}, "
            f"pooled={result['policy_alphas']['minimum_pooled_oof_loss']}",
            flush=True,
        )

    methods = (
        "recorded_one_se_mean_relative",
        "minimum_mean_relative",
        "minimum_pooled_oof_loss",
        "raw_spline",
    )
    rows = []
    for item in task_summaries:
        row: dict[str, Any] = {
            "task_id": item["task_id"],
            "dataset_id": item["dataset_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "effective_bags": item["effective_bags"],
            "identity_benchmark_error": item["errors"]["matched_inner_bag_identity"],
            "full_tabiclv2_benchmark_error": item["errors"].get("source_full_outer_training_tabiclv2"),
        }
        for method in methods:
            row[f"{method}_alpha"] = item["policy_alphas"][method]
            row[f"{method}_benchmark_error"] = item["errors"][method]
        rows.append(row)
    _write_csv(args.output_dir / "task_results.csv", rows)
    _write_json(args.output_dir / "task_summaries.json", task_summaries)

    paired_matched = {
        method: _comparison(
            task_summaries=task_summaries,
            method=method,
            reference="matched_inner_bag_identity",
            bootstrap_rounds=args.bootstrap_rounds,
            bootstrap_seed=args.bootstrap_seed + offset,
        )
        for offset, method in enumerate(methods)
    }
    paired_full = {
        method: _comparison(
            task_summaries=task_summaries,
            method=method,
            reference="source_full_outer_training_tabiclv2",
            bootstrap_rounds=args.bootstrap_rounds,
            bootstrap_seed=args.bootstrap_seed + 100 + offset,
        )
        for offset, method in enumerate(methods)
    }
    summary = {
        "selection_audit_schema_version": SELECTION_AUDIT_SCHEMA_VERSION,
        "exploratory": True,
        "n_tasks": len(task_summaries),
        "policy_alpha_counts": {
            method: {
                str(alpha): int(sum(item["policy_alphas"][method] == alpha for item in task_summaries))
                for alpha in _ALPHA_GRID
            }
            for method in methods
        },
        "paired_vs_matched_inner_bag_identity": paired_matched,
        "end_to_end_vs_source_full_outer_training_tabiclv2": paired_full,
        "task_summaries": task_summaries,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(task_summaries), "policy_alpha_counts": summary["policy_alpha_counts"]}), flush=True)


if __name__ == "__main__":
    main()
