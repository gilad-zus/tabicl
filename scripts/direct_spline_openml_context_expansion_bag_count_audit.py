"""Audit how much the frozen context-expansion result depends on bag count.

This CPU-only diagnostic re-scores archived expanded-context predictions from
the completed eight-bag DirectSpline experiment.  It considers every possible
subset of one, two, four, and eight bags.  No spline, preprocessor, or
TabICLv2 prediction is retrained.

For each subset it reports two policies:

* ``fixed_eight_bag_alpha`` keeps the alpha selected from the completed
  eight-bag OOF evidence.  It isolates the effect of reducing test-time
  averaging while holding the final identity/spline decision fixed.
* ``subset_oof_alpha`` reselects alpha from only the selected bags' OOF rows.
  It estimates the complete effect of using fewer existing bags, including the
  smaller amount of validation evidence available to the table-level blend.

The latter remains a diagnostic rather than a replacement for a fresh
one-trajectory train/validation/test experiment: its bag checkpoints were
still selected with the original A/B protocol.  Outer-test labels are used
only to score the already archived subset predictions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from direct_spline_openml_crossfit_blend import (
    _ALPHA_GRID,
    _blend_prediction,
    _load_source_task,
    _source_standard_prediction,
    _validate_case,
)
from direct_spline_openml_crossfit_context_expansion import (
    CONTEXT_EXPANSION_SCHEMA_VERSION,
    ContextExpansionBagPredictions,
    _load_bag,
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
    _prediction_shape,
    _safe_name,
)
from tabicl._experiments.direct_spline_protocol import deployment_error


BAG_COUNT_AUDIT_SCHEMA_VERSION = 1
_DEFAULT_BAG_COUNTS = (1, 2, 4, 8)
_POLICIES = ("fixed_eight_bag_alpha", "subset_oof_alpha")
_TIE_ATOL = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crossfit-dir", type=Path, required=True, help="Completed context-expansion result directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--task-id", type=int, action="append", help="Audit only these task IDs. Repeatable.")
    parser.add_argument(
        "--bag-count",
        type=int,
        action="append",
        help="Subset size to audit. Repeatable; defaults to 1, 2, 4, and 8.",
    )
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    counts = _DEFAULT_BAG_COUNTS if args.bag_count is None else tuple(sorted(set(int(value) for value in args.bag_count)))
    if not counts or any(count < 1 for count in counts):
        raise ValueError("--bag-count values must be positive")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    args.bag_count = counts
    return args


def _task_dir(crossfit_dir: Path, task: OpenMLTaskData) -> Path:
    return crossfit_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _outcome(reference: float, candidate: float) -> str:
    difference = float(candidate) - float(reference)
    if difference < -_TIE_ATOL:
        return "win"
    if difference > _TIE_ATOL:
        return "loss"
    return "tie"


def _prediction_and_oof_for_subset(
    *, task: OpenMLTaskData, bags: Sequence[ContextExpansionBagPredictions]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return OOF labels/predictions and test identity/spline means for bags."""

    if not bags:
        raise ValueError("a subset must contain at least one bag")
    oof_indices = np.concatenate(
        [np.concatenate((bag.selection_a_indices, bag.selection_b_indices)) for bag in bags]
    )
    if oof_indices.ndim != 1 or np.unique(oof_indices).size != oof_indices.size:
        raise ValueError("subset bags must have disjoint OOF rows")
    expected_shape = _prediction_shape(oof_indices.size, task.problem_type, task.n_classes)
    identity_oof = np.empty(expected_shape, dtype=float)
    spline_oof = np.empty(expected_shape, dtype=float)
    offset = 0
    for bag in bags:
        count_a = int(bag.selection_a_indices.size)
        count_b = int(bag.selection_b_indices.size)
        identity_oof[offset : offset + count_a] = bag.expanded_identity_selection_a
        spline_oof[offset : offset + count_a] = bag.expanded_spline_selected_on_b_selection_a
        offset += count_a
        identity_oof[offset : offset + count_b] = bag.expanded_identity_selection_b
        spline_oof[offset : offset + count_b] = bag.expanded_spline_selected_on_a_selection_b
        offset += count_b
    if offset != oof_indices.size or not np.isfinite(identity_oof).all() or not np.isfinite(spline_oof).all():
        raise RuntimeError("subset has incomplete or invalid archived OOF predictions")
    identity_test = np.mean([bag.expanded_identity_test for bag in bags], axis=0)
    spline_test = np.mean(
        [0.5 * (bag.expanded_spline_selected_on_a_test + bag.expanded_spline_selected_on_b_test) for bag in bags],
        axis=0,
    )
    return np.asarray(task.y_train[oof_indices]), identity_oof, spline_oof, identity_test, spline_test


def _select_alpha(
    *, task: OpenMLTaskData, labels: np.ndarray, identity_prediction: np.ndarray, spline_prediction: np.ndarray
) -> dict[str, Any]:
    candidates = []
    for alpha in _ALPHA_GRID:
        prediction = _blend_prediction(identity_prediction, spline_prediction, float(alpha))
        candidates.append(
            {
                "alpha": float(alpha),
                "deployment_error": float(
                    deployment_error(task.problem_type, labels, prediction, n_classes=task.n_classes)
                ),
            }
        )
    selected = min(candidates, key=lambda item: (item["deployment_error"], item["alpha"]))
    return {"selected_alpha": float(selected["alpha"]), "candidate_alphas": candidates}


def _score_subset(
    *,
    task: OpenMLTaskData,
    bags: Sequence[ContextExpansionBagPredictions],
    fixed_alpha: float,
    subset: tuple[int, ...],
) -> dict[str, Any]:
    labels, identity_oof, spline_oof, identity_test, spline_test = _prediction_and_oof_for_subset(task=task, bags=bags)
    selection = _select_alpha(
        task=task,
        labels=labels,
        identity_prediction=identity_oof,
        spline_prediction=spline_oof,
    )
    identity_metrics = _metric_bundle(task.problem_type, task.y_test, identity_test, task.n_classes)
    spline_metrics = _metric_bundle(task.problem_type, task.y_test, spline_test, task.n_classes)
    policies = {}
    for policy, alpha in (
        ("fixed_eight_bag_alpha", fixed_alpha),
        ("subset_oof_alpha", float(selection["selected_alpha"])),
    ):
        metrics = _metric_bundle(
            task.problem_type,
            task.y_test,
            _blend_prediction(identity_test, spline_test, alpha),
            task.n_classes,
        )
        policies[policy] = {"alpha": float(alpha), "metrics": metrics}
    return {
        "subset": list(subset),
        "subset_oof_rows": int(labels.shape[0]),
        "subset_selection": selection,
        "identity": identity_metrics,
        "raw_spline": spline_metrics,
        "policies": policies,
    }


def _summarize_subsets(
    *, subset_records: Sequence[Mapping[str, Any]], policy: str, full_reference: Mapping[str, float] | None
) -> dict[str, Any]:
    if not subset_records:
        raise ValueError("cannot summarise zero subsets")
    candidate_benchmark = np.asarray(
        [float(item["policies"][policy]["metrics"]["benchmark_error"]) for item in subset_records], dtype=float
    )
    identity_benchmark = np.asarray([float(item["identity"]["benchmark_error"]) for item in subset_records], dtype=float)
    alphas = np.asarray([float(item["policies"][policy]["alpha"]) for item in subset_records], dtype=float)
    outcomes_vs_identity = [_outcome(reference, candidate) for reference, candidate in zip(identity_benchmark, candidate_benchmark)]
    result: dict[str, Any] = {
        "n_subsets": len(subset_records),
        "mean_benchmark_error": float(np.mean(candidate_benchmark)),
        "minimum_benchmark_error": float(np.min(candidate_benchmark)),
        "maximum_benchmark_error": float(np.max(candidate_benchmark)),
        "mean_identity_benchmark_error": float(np.mean(identity_benchmark)),
        "mean_relative_change_vs_subset_identity": float(
            np.mean((candidate_benchmark - identity_benchmark) / np.maximum(np.abs(identity_benchmark), 1e-12))
        ),
        "subset_outcomes_vs_identity": {name: int(outcomes_vs_identity.count(name)) for name in ("win", "tie", "loss")},
        "alpha_counts": {str(alpha): int(np.sum(alphas == alpha)) for alpha in _ALPHA_GRID},
    }
    if full_reference is not None:
        reference_benchmark = float(full_reference["benchmark_error"])
        outcomes_vs_full = [_outcome(reference_benchmark, candidate) for candidate in candidate_benchmark]
        result["mean_relative_change_vs_full_tabiclv2"] = float(
            (result["mean_benchmark_error"] - reference_benchmark) / max(abs(reference_benchmark), 1e-12)
        )
        result["mean_subset_outcomes_vs_full_tabiclv2"] = {
            name: int(outcomes_vs_full.count(name))
            for name in ("win", "tie", "loss")
        }
    return result


def _audit_task(
    *, case: SourceCase, task: OpenMLTaskData, crossfit_dir: Path, bag_counts: Sequence[int]
) -> dict[str, Any]:
    task_dir = _task_dir(crossfit_dir, task)
    stored = _load_json(task_dir / "task_summary.json", label="context-expansion task summary")
    effective_bags = int(stored["effective_bags"])
    paths = [task_dir / f"bag_{bag}.npz" for bag in range(effective_bags)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bag artifacts for task {task.task_id}: {missing}")
    bags = [_load_bag(path) for path in paths]
    fixed_alpha = float(stored["expanded_context"]["blend_selection"]["selected_alpha"])
    source_prediction = _source_standard_prediction(source_dir=case.source_dir, task=task)
    full_reference = None if source_prediction is None else _metric_bundle(task.problem_type, task.y_test, source_prediction, task.n_classes)
    results_by_count: dict[str, Any] = {}
    for count in bag_counts:
        if count > effective_bags:
            raise ValueError(f"task {task.task_id} has {effective_bags} bags, cannot audit subset size {count}")
        subset_records = [
            _score_subset(
                task=task,
                bags=[bags[index] for index in subset],
                fixed_alpha=fixed_alpha,
                subset=tuple(subset),
            )
            for subset in itertools.combinations(range(effective_bags), count)
        ]
        results_by_count[str(count)] = {
            "n_subsets": len(subset_records),
            "policies": {
                policy: _summarize_subsets(
                    subset_records=subset_records,
                    policy=policy,
                    full_reference=full_reference,
                )
                for policy in _POLICIES
            },
            "subsets": subset_records,
        }
    full_record = results_by_count.get(str(effective_bags), {}).get("subsets")
    if full_record is None:
        full_record = [
            _score_subset(
                task=task,
                bags=bags,
                fixed_alpha=fixed_alpha,
                subset=tuple(range(effective_bags)),
            )
        ]
    if len(full_record) != 1:
        raise RuntimeError("the complete bag subset must be unique")
    expected = float(stored["expanded_context"]["outer_test"]["selected_blend"]["benchmark_error"])
    actual = float(full_record[0]["policies"]["fixed_eight_bag_alpha"]["metrics"]["benchmark_error"])
    if not np.isclose(expected, actual, rtol=0.0, atol=1e-12):
        raise ValueError(f"eight-bag reproduction mismatch for task {task.task_id}: {actual} != {expected}")
    return {
        "task_id": int(task.task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "effective_bags": effective_bags,
        "fixed_eight_bag_alpha": fixed_alpha,
        "source_full_outer_training_tabiclv2": full_reference,
        "eight_bag_reproduction_benchmark_error": actual,
        "bag_counts": results_by_count,
    }


def _comparison(
    *, task_summaries: Sequence[Mapping[str, Any]], bag_count: int, policy: str, bootstrap_rounds: int, bootstrap_seed: int
) -> dict[str, Any] | None:
    usable = [item for item in task_summaries if item["source_full_outer_training_tabiclv2"] is not None]
    if not usable:
        return None
    reference = np.asarray(
        [float(item["source_full_outer_training_tabiclv2"]["benchmark_error"]) for item in usable], dtype=float
    )
    candidate = np.asarray(
        [float(item["bag_counts"][str(bag_count)]["policies"][policy]["mean_benchmark_error"]) for item in usable],
        dtype=float,
    )
    problem_types = np.asarray([str(item["problem_type"]) for item in usable], dtype=object)
    return _paired_comparison_summary(
        reference=reference,
        candidate=candidate,
        problem_types=problem_types,
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=bootstrap_seed,
        reference_label="source_full_outer_training_tabiclv2",
        candidate_label=f"{policy}_mean_over_all_{bag_count}_bag_subsets",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(
    *, crossfit_dir: Path, crossfit_manifest: Mapping[str, Any], source_dir: Path, cases: Sequence[SourceCase], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "bag_count_audit_schema_version": BAG_COUNT_AUDIT_SCHEMA_VERSION,
        "experiment": "DirectSpline archived expanded-context bag-count audit",
        "crossfit_dir": str(crossfit_dir.resolve()),
        "crossfit_manifest_sha256": _sha256(crossfit_dir / "experiment_manifest.json"),
        "context_expansion_schema_version": crossfit_manifest.get("context_expansion_schema_version"),
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "config_label": str(args.config_label),
        "task_ids": [int(case.task_id) for case in cases],
        "bag_counts": [int(count) for count in args.bag_count],
        "policies": {
            "fixed_eight_bag_alpha": "Keep the alpha from all eight OOF bags; isolate test-time bag averaging.",
            "subset_oof_alpha": "Select alpha from only the chosen bags' cross-fitted OOF rows; estimates the complete reduced-bag pipeline.",
        },
        "outer_test_label_policy": "Diagnostic only: outer-test labels score archived subset predictions after every subset and alpha is constructed.",
        "script_sha256": _sha256(Path(__file__)),
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing bag-count audit manifest")
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different bag-count audit")
        if not resume:
            raise ValueError("bag-count audit output directory already exists; pass --resume to rewrite it")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def main() -> None:
    args = _parse_args()
    crossfit_dir = args.crossfit_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    crossfit_manifest = _load_json(crossfit_dir / "experiment_manifest.json", label="context-expansion manifest")
    if crossfit_manifest.get("context_expansion_schema_version") != CONTEXT_EXPANSION_SCHEMA_VERSION:
        raise ValueError("unsupported context-expansion artifact schema")
    source_value = crossfit_manifest.get("source_dir")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("context-expansion manifest has no source_dir")
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
    cases = [case for case in cases if case.problem_type in {"multiclass", "regression"}]
    if not cases:
        raise ValueError("no requested multiclass or regression D source tasks")
    for case in cases:
        _validate_case(case)
    manifest = _manifest(
        crossfit_dir=crossfit_dir,
        crossfit_manifest=crossfit_manifest,
        source_dir=source_dir,
        cases=cases,
        args=args,
    )
    _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))

    task_summaries: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        task = _load_source_task(case=case, immutable_run=immutable_run)
        item = _audit_task(case=case, task=task, crossfit_dir=crossfit_dir, bag_counts=args.bag_count)
        task_summaries.append(item)
        for bag_count, count_record in item["bag_counts"].items():
            for subset_record in count_record["subsets"]:
                for policy in _POLICIES:
                    metrics = subset_record["policies"][policy]["metrics"]
                    subset_rows.append(
                        {
                            "task_id": item["task_id"],
                            "dataset_id": item["dataset_id"],
                            "dataset_name": item["dataset_name"],
                            "problem_type": item["problem_type"],
                            "bag_count": int(bag_count),
                            "subset": ",".join(str(index) for index in subset_record["subset"]),
                            "policy": policy,
                            "alpha": subset_record["policies"][policy]["alpha"],
                            "subset_oof_rows": subset_record["subset_oof_rows"],
                            "identity_benchmark_error": subset_record["identity"]["benchmark_error"],
                            "raw_spline_benchmark_error": subset_record["raw_spline"]["benchmark_error"],
                            "selected_benchmark_error": metrics["benchmark_error"],
                            "selected_deployment_error": metrics["deployment_error"],
                        }
                    )
        print(f"[{position}/{len(cases)}] task {case.task_id}: reproduced all {item['effective_bags']} bags", flush=True)

    task_rows = []
    for item in task_summaries:
        for bag_count, count_record in item["bag_counts"].items():
            for policy, policy_record in count_record["policies"].items():
                task_rows.append(
                    {
                        "task_id": item["task_id"],
                        "dataset_id": item["dataset_id"],
                        "dataset_name": item["dataset_name"],
                        "problem_type": item["problem_type"],
                        "bag_count": int(bag_count),
                        "policy": policy,
                        "n_subsets": policy_record["n_subsets"],
                        "mean_selected_benchmark_error": policy_record["mean_benchmark_error"],
                        "minimum_selected_benchmark_error": policy_record["minimum_benchmark_error"],
                        "maximum_selected_benchmark_error": policy_record["maximum_benchmark_error"],
                        "mean_identity_benchmark_error": policy_record["mean_identity_benchmark_error"],
                        "mean_relative_change_vs_subset_identity": policy_record["mean_relative_change_vs_subset_identity"],
                        "mean_relative_change_vs_full_tabiclv2": policy_record.get("mean_relative_change_vs_full_tabiclv2"),
                    }
                )
    _write_csv(args.output_dir / "subset_results.csv", subset_rows)
    _write_csv(args.output_dir / "task_results.csv", task_rows)
    task_summaries.sort(key=lambda item: int(item["task_id"]))
    _write_json(args.output_dir / "task_summaries.json", task_summaries)
    summary = {
        "bag_count_audit_schema_version": BAG_COUNT_AUDIT_SCHEMA_VERSION,
        "exploratory": True,
        "n_tasks": len(task_summaries),
        "bag_counts": list(args.bag_count),
        "policies": list(_POLICIES),
        "paired_vs_source_full_outer_training_tabiclv2": {
            str(bag_count): {
                policy: _comparison(
                    task_summaries=task_summaries,
                    bag_count=bag_count,
                    policy=policy,
                    bootstrap_rounds=args.bootstrap_rounds,
                    bootstrap_seed=args.bootstrap_seed + 100 * position + policy_index,
                )
                for policy_index, policy in enumerate(_POLICIES)
            }
            for position, bag_count in enumerate(args.bag_count)
        },
        "outer_test_label_policy": manifest["outer_test_label_policy"],
        "interpretation": {
            "fixed_eight_bag_alpha": "Holds the original eight-bag decision fixed and measures only the effect of fewer averaged prediction bags.",
            "subset_oof_alpha": "Uses fewer OOF rows and fewer prediction bags, but retains A/B-selected checkpoints; it does not replace a fresh one-trajectory validation selection experiment.",
        },
        "task_summaries": task_summaries,
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(task_summaries), "bag_counts": list(args.bag_count)}), flush=True)


if __name__ == "__main__":
    main()
