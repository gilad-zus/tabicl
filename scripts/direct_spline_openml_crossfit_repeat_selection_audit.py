"""Audit whether two-repeat OOF prediction averaging improves blend selection.

This script combines two completed DirectSpline cross-fit runs with the same
outer split and adapter configuration but different ``--protocol-seed``
values.  Each outer-training row consequently has two predictions from models
whose fitting and checkpoint selection excluded that row.  It compares two
predeclared table-level blend rules:

* ``mean_member_pooled_oof_loss`` averages the loss of the two held-out
  predictions.  This is the natural extension of the existing pooled-OOF
  policy when predictions are judged independently.
* ``mean_prediction_pooled_oof_loss`` averages the two held-out predictions
  for each row first, then scores that prediction ensemble.  It can reward
  cancellation of split-specific DirectSpline errors.

Every selected alpha is applied to the same test prediction: the mean of the
two repeat-level identity ensembles and the mean of their DirectSpline
ensembles.  Thus a difference in reported test performance is caused by the
selection decision, not by one policy receiving more test predictors.

The outer-test labels are used only after all OOF alphas and the common test
prediction have been constructed.  This is a mechanistic pilot, not a fresh
benchmark: its task subset must be fixed without inspecting these results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from direct_spline_openml_support_audit import (
    SourceCase,
    _as_int,
    _canonical_json,
    _find_source_cases,
    _load_json,
    _sha256,
    _write_json,
)
from direct_spline_openml_crossfit_blend import (
    CROSSFIT_BLEND_SCHEMA_VERSION,
    _ALPHA_GRID,
    _assemble_task_predictions,
    _blend_prediction,
    _load_bag,
    _load_source_task,
    _source_standard_prediction,
)
from tabicl._experiments.direct_spline_openml import (
    OpenMLTaskData,
    _metric_bundle,
    _paired_comparison_summary,
    _safe_name,
)
from tabicl._experiments.direct_spline_protocol import deployment_error


REPEAT_SELECTION_AUDIT_SCHEMA_VERSION = 1
_POLICIES = (
    "first_repeat_pooled_oof_loss",
    "second_repeat_pooled_oof_loss",
    "mean_member_pooled_oof_loss",
    "mean_prediction_pooled_oof_loss",
    "raw_spline",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-crossfit-dir", type=Path, required=True, help="Completed original cross-fit run.")
    parser.add_argument("--second-crossfit-dir", type=Path, required=True, help="Completed new-partition cross-fit run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--task-id", type=int, action="append", required=True, help="Pilot task ID. Repeatable.")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    if len(set(args.task_id)) != len(args.task_id):
        raise ValueError("--task-id values must be unique")
    return args


def _task_dir(crossfit_dir: Path, task: OpenMLTaskData) -> Path:
    return crossfit_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _load_repeat_predictions(
    *, crossfit_dir: Path, task: OpenMLTaskData, effective_bags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct and verify one repeat's OOF and deployed prediction pair."""
    task_dir = _task_dir(crossfit_dir, task)
    bag_paths = [task_dir / f"bag_{bag}.npz" for bag in range(effective_bags)]
    missing = [str(path) for path in bag_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing repeat artifacts for task {task.task_id}: {missing}")
    identity_oof, spline_oof, _units, identity_test, spline_test = _assemble_task_predictions(
        task=task,
        bag_results=[_load_bag(path) for path in bag_paths],
    )
    saved_path = task_dir / "task_predictions.npz"
    if not saved_path.is_file():
        raise FileNotFoundError(f"missing saved task predictions for task {task.task_id}: {saved_path}")
    with np.load(saved_path, allow_pickle=False) as saved:
        for key, actual in (
            ("identity_oof", identity_oof),
            ("spline_oof", spline_oof),
            ("identity_test", identity_test),
            ("spline_test", spline_test),
        ):
            expected = np.asarray(saved[key], dtype=float)
            if expected.shape != actual.shape or not np.allclose(expected, actual, atol=1e-12, rtol=0.0):
                raise ValueError(f"saved {key} does not match reconstructed repeat predictions for task {task.task_id}")
    return identity_oof, spline_oof, identity_test, spline_test


def _select_minimum_loss(*, task: OpenMLTaskData, candidates: Mapping[float, float], rule: str) -> dict[str, Any]:
    records = [
        {"alpha": float(alpha), "pooled_oof_deployment_error": float(error)}
        for alpha, error in sorted(candidates.items())
    ]
    best = min(records, key=lambda item: (float(item["pooled_oof_deployment_error"]), float(item["alpha"])))
    return {"selection_rule": rule, "candidate_alphas": records, "selected_alpha": float(best["alpha"])}


def _single_repeat_selection(
    *, task: OpenMLTaskData, identity_oof: np.ndarray, spline_oof: np.ndarray, rule: str
) -> dict[str, Any]:
    return _select_minimum_loss(
        task=task,
        candidates={
            float(alpha): float(
                deployment_error(
                    task.problem_type,
                    task.y_train,
                    _blend_prediction(identity_oof, spline_oof, float(alpha)),
                    n_classes=task.n_classes,
                )
            )
            for alpha in _ALPHA_GRID
        },
        rule=rule,
    )


def _two_repeat_selections(
    *,
    task: OpenMLTaskData,
    first_identity_oof: np.ndarray,
    first_spline_oof: np.ndarray,
    second_identity_oof: np.ndarray,
    second_spline_oof: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return selection records before any outer-test labels are examined."""
    first = _single_repeat_selection(
        task=task,
        identity_oof=first_identity_oof,
        spline_oof=first_spline_oof,
        rule="smallest alpha at the minimum pooled OOF loss from the original partition",
    )
    second = _single_repeat_selection(
        task=task,
        identity_oof=second_identity_oof,
        spline_oof=second_spline_oof,
        rule="smallest alpha at the minimum pooled OOF loss from the independent second partition",
    )
    mean_member: dict[float, float] = {}
    mean_prediction: dict[float, float] = {}
    identity_mean = 0.5 * (first_identity_oof + second_identity_oof)
    spline_mean = 0.5 * (first_spline_oof + second_spline_oof)
    for alpha in _ALPHA_GRID:
        first_prediction = _blend_prediction(first_identity_oof, first_spline_oof, float(alpha))
        second_prediction = _blend_prediction(second_identity_oof, second_spline_oof, float(alpha))
        mean_member[float(alpha)] = 0.5 * (
            deployment_error(task.problem_type, task.y_train, first_prediction, n_classes=task.n_classes)
            + deployment_error(task.problem_type, task.y_train, second_prediction, n_classes=task.n_classes)
        )
        mean_prediction[float(alpha)] = deployment_error(
            task.problem_type,
            task.y_train,
            _blend_prediction(identity_mean, spline_mean, float(alpha)),
            n_classes=task.n_classes,
        )
    return {
        "first_repeat_pooled_oof_loss": first,
        "second_repeat_pooled_oof_loss": second,
        "mean_member_pooled_oof_loss": _select_minimum_loss(
            task=task,
            candidates=mean_member,
            rule="smallest alpha at the minimum mean loss of two independently held-out predictions",
        ),
        "mean_prediction_pooled_oof_loss": _select_minimum_loss(
            task=task,
            candidates=mean_prediction,
            rule="smallest alpha at the minimum pooled OOF loss after averaging two independently held-out predictions per row",
        ),
    }


def _load_crossfit_manifest(path: Path, *, label: str) -> Mapping[str, Any]:
    manifest = _load_json(path / "experiment_manifest.json", label=label)
    if manifest.get("crossfit_blend_schema_version") != CROSSFIT_BLEND_SCHEMA_VERSION:
        raise ValueError(f"{label} has unsupported cross-fit artifact schema")
    return manifest


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing repeat-selection audit manifest")
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different repeat-selection protocol; choose a new --output-dir")
        if not resume:
            raise ValueError("repeat-selection audit output directory already exists; pass --resume to rewrite it")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, dict(manifest))


def _audit_manifest(
    *, first_dir: Path, second_dir: Path, first_manifest: Mapping[str, Any], second_manifest: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "repeat_selection_audit_schema_version": REPEAT_SELECTION_AUDIT_SCHEMA_VERSION,
        "first_crossfit_dir": str(first_dir),
        "second_crossfit_dir": str(second_dir),
        "first_crossfit_manifest_sha256": _sha256(first_dir / "experiment_manifest.json"),
        "second_crossfit_manifest_sha256": _sha256(second_dir / "experiment_manifest.json"),
        "source_dir": first_manifest["source_dir"],
        "source_manifest_sha256": first_manifest["source_manifest_sha256"],
        "config_label": str(args.config_label),
        "task_ids": sorted(int(task_id) for task_id in args.task_id),
        "alpha_grid": list(_ALPHA_GRID),
        "selection_policies": {
            "mean_member_pooled_oof_loss": "average two independently held-out prediction losses before choosing alpha",
            "mean_prediction_pooled_oof_loss": "average two independently held-out predictions per row before choosing alpha",
        },
        "common_test_ensemble": {
            "identity": "mean of the two 8-bag identity ensembles",
            "spline": "mean of the two 16-selected-state spline ensembles",
            "policy_application": "each alpha is applied to the same 16-bag identity / 32-selected-state spline pair",
        },
        "outer_test_label_policy": "outer-test labels are only used after all policy alphas and common test predictions are constructed",
        "script_sha256": _sha256(Path(__file__)),
    }


def _comparison(
    *, task_summaries: Sequence[Mapping[str, Any]], method: str, reference: str, bootstrap_rounds: int, bootstrap_seed: int
) -> dict[str, Any]:
    usable = [item for item in task_summaries if item["errors"].get(reference) is not None]
    return _paired_comparison_summary(
        reference=np.asarray([item["errors"][reference] for item in usable], dtype=float),
        candidate=np.asarray([item["errors"][method] for item in usable], dtype=float),
        problem_types=np.asarray([item["problem_type"] for item in usable], dtype=object),
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=bootstrap_seed,
        reference_label=reference,
        candidate_label=method,
    )


def _run_task(
    *, case: SourceCase, task: OpenMLTaskData, first_dir: Path, second_dir: Path
) -> dict[str, Any]:
    first_summary = _load_json(_task_dir(first_dir, task) / "task_summary.json", label="first-repeat task summary")
    second_summary = _load_json(_task_dir(second_dir, task) / "task_summary.json", label="second-repeat task summary")
    first_bags = int(first_summary["effective_bags"])
    second_bags = int(second_summary["effective_bags"])
    if first_bags != second_bags:
        raise ValueError(f"repeat bag-count mismatch for task {task.task_id}: {first_bags} != {second_bags}")
    first_identity_oof, first_spline_oof, first_identity_test, first_spline_test = _load_repeat_predictions(
        crossfit_dir=first_dir, task=task, effective_bags=first_bags
    )
    second_identity_oof, second_spline_oof, second_identity_test, second_spline_test = _load_repeat_predictions(
        crossfit_dir=second_dir, task=task, effective_bags=second_bags
    )
    selection = _two_repeat_selections(
        task=task,
        first_identity_oof=first_identity_oof,
        first_spline_oof=first_spline_oof,
        second_identity_oof=second_identity_oof,
        second_spline_oof=second_spline_oof,
    )
    policy_alphas = {name: float(record["selected_alpha"]) for name, record in selection.items()}
    policy_alphas["raw_spline"] = 1.0

    # All validation decisions above are now locked.  From here onwards the
    # fixed outer-test labels score common, already-constructed predictions.
    identity_test = 0.5 * (first_identity_test + second_identity_test)
    spline_test = 0.5 * (first_spline_test + second_spline_test)
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
        "effective_bags_per_repeat": first_bags,
        "selection": selection,
        "policy_alphas": policy_alphas,
        "oof": {
            "first_repeat": {
                "identity": _metric_bundle(task.problem_type, task.y_train, first_identity_oof, task.n_classes),
                "spline": _metric_bundle(task.problem_type, task.y_train, first_spline_oof, task.n_classes),
            },
            "second_repeat": {
                "identity": _metric_bundle(task.problem_type, task.y_train, second_identity_oof, task.n_classes),
                "spline": _metric_bundle(task.problem_type, task.y_train, second_spline_oof, task.n_classes),
            },
            "mean_prediction": {
                "identity": _metric_bundle(
                    task.problem_type, task.y_train, 0.5 * (first_identity_oof + second_identity_oof), task.n_classes
                ),
                "spline": _metric_bundle(
                    task.problem_type, task.y_train, 0.5 * (first_spline_oof + second_spline_oof), task.n_classes
                ),
            },
        },
        "errors": errors,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    first_dir = args.first_crossfit_dir.resolve()
    second_dir = args.second_crossfit_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    first_manifest = _load_crossfit_manifest(first_dir, label="first cross-fit manifest")
    second_manifest = _load_crossfit_manifest(second_dir, label="second cross-fit manifest")
    for key in ("source_dir", "source_manifest_sha256", "config_label", "requested_bags"):
        if first_manifest.get(key) != second_manifest.get(key):
            raise ValueError(f"cross-fit repeats differ in {key!r}; they cannot be combined")
    if int(first_manifest["protocol_seed"]) == int(second_manifest["protocol_seed"]):
        raise ValueError("cross-fit repeats must use different protocol seeds")
    source_dir = Path(str(first_manifest["source_dir"])).resolve()
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    requested_ids = set(int(task_id) for task_id in args.task_id)
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=requested_ids,
    )
    found_ids = {int(case.task_id) for case in cases}
    if found_ids != requested_ids:
        raise ValueError(f"requested/source task IDs differ: requested={sorted(requested_ids)}, found={sorted(found_ids)}")
    for case in cases:
        if case.problem_type not in {"multiclass", "regression"}:
            raise ValueError(f"task {case.task_id} has unsupported problem type {case.problem_type!r}")
    manifest = _audit_manifest(
        first_dir=first_dir,
        second_dir=second_dir,
        first_manifest=first_manifest,
        second_manifest=second_manifest,
        args=args,
    )
    _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))

    task_summaries = []
    for index, case in enumerate(cases, start=1):
        task = _load_source_task(case=case, immutable_run=immutable_run)
        result = _run_task(case=case, task=task, first_dir=first_dir, second_dir=second_dir)
        task_summaries.append(result)
        print(
            f"[{index}/{len(cases)}] task {case.task_id} {case.dataset_name}: "
            f"member={result['policy_alphas']['mean_member_pooled_oof_loss']}, "
            f"prediction={result['policy_alphas']['mean_prediction_pooled_oof_loss']}",
            flush=True,
        )
    task_summaries.sort(key=lambda item: int(item["task_id"]))
    rows = []
    for item in task_summaries:
        row: dict[str, Any] = {
            "task_id": item["task_id"],
            "dataset_id": item["dataset_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "effective_bags_per_repeat": item["effective_bags_per_repeat"],
            "identity_benchmark_error": item["errors"]["matched_inner_bag_identity"],
            "full_tabiclv2_benchmark_error": item["errors"].get("source_full_outer_training_tabiclv2"),
        }
        for method in _POLICIES:
            row[f"{method}_alpha"] = item["policy_alphas"][method]
            row[f"{method}_benchmark_error"] = item["errors"][method]
        rows.append(row)
    _write_csv(args.output_dir / "task_results.csv", rows)
    _write_json(args.output_dir / "task_summaries.json", task_summaries)
    summary = {
        "repeat_selection_audit_schema_version": REPEAT_SELECTION_AUDIT_SCHEMA_VERSION,
        "exploratory": True,
        "n_tasks": len(task_summaries),
        "policy_alpha_counts": {
            method: {
                str(alpha): int(sum(item["policy_alphas"][method] == alpha for item in task_summaries))
                for alpha in _ALPHA_GRID
            }
            for method in _POLICIES
        },
        "paired_vs_matched_inner_bag_identity": {
            method: _comparison(
                task_summaries=task_summaries,
                method=method,
                reference="matched_inner_bag_identity",
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.bootstrap_seed + index,
            )
            for index, method in enumerate(_POLICIES)
        },
        "end_to_end_vs_source_full_outer_training_tabiclv2": {
            method: _comparison(
                task_summaries=task_summaries,
                method=method,
                reference="source_full_outer_training_tabiclv2",
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.bootstrap_seed + 100 + index,
            )
            for index, method in enumerate(_POLICIES)
        },
        "task_summaries": task_summaries,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(task_summaries), "policy_alpha_counts": summary["policy_alpha_counts"]}), flush=True)


if __name__ == "__main__":
    main()
