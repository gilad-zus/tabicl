"""Localise DirectSpline harm to its nominal uniform-K20 spline regions.

This is a retrospective CPU-only audit of a completed *standard-pipeline*
DirectSpline run.  It replays the exact public TabICL numerical preprocessing
for each saved bag, maps every validation and outer-test row to the active
uniform cubic K20 basis, and joins those support records to the already frozen
identity/adapted predictions.

The primary hypothesis is deliberately narrower than the earlier task-level
support audit:

    DirectSpline harm is concentrated in query rows whose active spline basis
    functions had little fitting-context activation.  A further manifestation
    is a feature/knot region where the spline helps validation rows but hurts
    outer-test rows.

For a row coordinate x, its effective basis support is

    sum_n <B(x), B(x_n)>,

where B is the cubic K20 B-spline design and x_n ranges over that bag's fitting
context.  This respects the overlapping four-basis-function structure of a
cubic spline instead of treating knot spans as independent hard bins.  The
ordinary knot-span occupancy is retained as an interpretable companion
quantity.

The source artifacts do not preserve trained adapter states.  Consequently
this audit cannot recover the learned deformation |f_j(x)-x| or derivative;
it tests support localisation using the actual frozen DirectSpline predictions.
It is the required first check before an instrumented GPU replay is justified.

Test labels are never used for preprocessing, support measurement, splitting,
or any decision.  They are read only after every bag's support records are
constructed, solely to score frozen prediction arrays post hoc.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \\
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_row_support_audit.py \\
      --source-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828 \\
      --task-id 75158 --task-id 167111 \\
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_row_support_audit/multiclass_pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from direct_spline_openml_support_audit import (
    SourceCase,
    _as_int,
    _assert_summary_validation_metrics,
    _bag_splits,
    _canonical_json,
    _find_source_cases,
    _fit_preprocessing_views,
    _load_bag,
    _load_json,
    _load_predictions,
    _seed,
    _sha256,
    _source_normal_config,
    _validate_aggregated_predictions,
    _write_csv,
    _write_json,
    load_tabarena_openml_task,
)


AUDIT_SCHEMA_VERSION = 1
_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class _CellSupport:
    """One feature/normalisation branch's fixed support information."""

    normalization_method: str
    feature_index: int
    feature_name: str
    context_interval_counts: np.ndarray
    validation_interval_indices: np.ndarray
    test_interval_indices: np.ndarray


@dataclass(frozen=True)
class _SplitSupport:
    """Row-level support summaries fixed before any target is scored."""

    row_metrics: Mapping[str, np.ndarray]
    cells: tuple[_CellSupport, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Completed standard-pipeline DirectSpline run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D", help="Frozen source configuration label (default: D).")
    parser.add_argument("--task-id", type=int, action="append", help="Audit only these completed tasks. Repeatable.")
    parser.add_argument("--openml-cache-dir", type=Path, default=None, help="Optional OpenML cache used only for replay.")
    parser.add_argument("--sparse-min-count", type=int, default=5)
    parser.add_argument("--sparse-min-fraction", type=float, default=0.001)
    parser.add_argument(
        "--minimum-cell-query-rows",
        type=int,
        default=5,
        help="Minimum rows in both validation and test before a cell can be tested for a sign flip (default: 5).",
    )
    parser.add_argument("--row-strata", type=int, default=4, help="Rank-balanced support strata per bag/split (default: 4).")
    parser.add_argument("--resume", action="store_true", help="Rewrite only a matching immutable audit output directory.")
    args = parser.parse_args()
    if args.sparse_min_count < 1:
        raise ValueError("--sparse-min-count must be positive")
    if not 0.0 < args.sparse_min_fraction <= 1.0:
        raise ValueError("--sparse-min-fraction must lie in (0, 1]")
    if args.minimum_cell_query_rows < 1:
        raise ValueError("--minimum-cell-query-rows must be positive")
    if args.row_strata < 2:
        raise ValueError("--row-strata must be at least two")
    return args


def _uniform_knots(*, n_control_points: int, degree: int) -> np.ndarray:
    """Return DirectSpline's clamped open-uniform knot vector on [-1, 1]."""

    if degree < 1 or n_control_points <= degree:
        raise ValueError("require n_control_points > degree >= 1")
    internal = np.linspace(-1.0, 1.0, n_control_points - degree + 1, dtype=float)[1:-1]
    return np.concatenate((np.full(degree + 1, -1.0), internal, np.full(degree + 1, 1.0)))


def _bspline_design(
    *, values: np.ndarray, knots: np.ndarray, degree: int, n_control_points: int, standardized_range: float
) -> np.ndarray:
    """Evaluate the same clamped local Cox--de Boor basis as DirectSpline."""

    x = np.asarray(values, dtype=float).reshape(-1)
    knot_vector = np.asarray(knots, dtype=float).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("basis values must be finite and nonempty")
    if standardized_range <= 0.0:
        raise ValueError("standardized_range must be positive")
    if knot_vector.size != n_control_points + degree + 1:
        raise ValueError("knot vector does not match spline dimensions")
    u = np.clip(x / standardized_range, -1.0, 1.0)
    spans = np.searchsorted(knot_vector, u, side="right") - 1
    spans = np.clip(spans, degree, n_control_points - 1).astype(int, copy=False)
    local = np.ones((u.size, degree + 1), dtype=float)
    left = np.zeros_like(local)
    right = np.zeros_like(local)
    for order in range(1, degree + 1):
        left[:, order] = u - knot_vector[spans + 1 - order]
        right[:, order] = knot_vector[spans + order] - u
        saved = np.zeros(u.size, dtype=float)
        for index in range(order):
            denominator = right[:, index + 1] + left[:, order - index]
            temporary = np.divide(local[:, index], denominator, out=np.zeros_like(denominator), where=np.abs(denominator) > _EPS)
            local[:, index] = saved + right[:, index + 1] * temporary
            saved = left[:, order - index] * temporary
        local[:, order] = saved
    indices = spans[:, None] - degree + np.arange(degree + 1, dtype=int)[None, :]
    design = np.zeros((u.size, n_control_points), dtype=float)
    np.add.at(design, (np.arange(u.size)[:, None], indices), local)
    if not np.allclose(design.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
        raise RuntimeError("B-spline basis does not form a partition of unity")
    return design


def _span_indices(values: np.ndarray, *, standardized_range: float, n_intervals: int) -> np.ndarray:
    if n_intervals < 1:
        raise ValueError("n_intervals must be positive")
    clipped = np.clip(np.asarray(values, dtype=float), -standardized_range, standardized_range)
    edges = np.linspace(-standardized_range, standardized_range, n_intervals + 1, dtype=float)
    return np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, n_intervals - 1)


def _basis_mass(*, context: np.ndarray, query: np.ndarray, knots: np.ndarray, degree: int, n_control_points: int, standardized_range: float) -> np.ndarray:
    """Return sum_n <B(query), B(context_n)> for every query coordinate."""

    context_design = _bspline_design(
        values=context,
        knots=knots,
        degree=degree,
        n_control_points=n_control_points,
        standardized_range=standardized_range,
    )
    query_design = _bspline_design(
        values=query,
        knots=knots,
        degree=degree,
        n_control_points=n_control_points,
        standardized_range=standardized_range,
    )
    result = query_design @ context_design.sum(axis=0)
    if not np.isfinite(result).all() or np.any(result < -1e-10):
        raise RuntimeError("effective basis support must be finite and non-negative")
    return np.maximum(result, 0.0)


def _row_support(
    *,
    views: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_names: Sequence[str],
    degree: int,
    n_control_points: int,
    standardized_range: float,
    sparse_min_count: int,
    sparse_min_fraction: float,
) -> tuple[_SplitSupport, _SplitSupport]:
    """Build all validation/test row support before any labels are consulted."""

    knots = _uniform_knots(n_control_points=n_control_points, degree=degree)
    n_intervals = n_control_points - degree
    validation_basis: list[np.ndarray] = []
    test_basis: list[np.ndarray] = []
    validation_counts: list[np.ndarray] = []
    test_counts: list[np.ndarray] = []
    validation_sparse: list[np.ndarray] = []
    test_sparse: list[np.ndarray] = []
    validation_outside: list[np.ndarray] = []
    test_outside: list[np.ndarray] = []
    validation_cells: list[_CellSupport] = []
    test_cells: list[_CellSupport] = []

    validation_rows: int | None = None
    test_rows: int | None = None
    for method, (context, validation, test) in views.items():
        if context.shape[1] != len(feature_names):
            raise RuntimeError("numerical preprocessing feature names do not match replayed views")
        if validation_rows is None:
            validation_rows, test_rows = validation.shape[0], test.shape[0]
        elif validation.shape[0] != validation_rows or test.shape[0] != test_rows:
            raise RuntimeError("normalisation branches disagree on query row count")
        for feature_index, feature_name in enumerate(feature_names):
            context_values = np.asarray(context[:, feature_index], dtype=float)
            validation_values = np.asarray(validation[:, feature_index], dtype=float)
            test_values = np.asarray(test[:, feature_index], dtype=float)
            if not (np.isfinite(context_values).all() and np.isfinite(validation_values).all() and np.isfinite(test_values).all()):
                raise ValueError("preprocessing produced non-finite numerical values")
            interval_context = _span_indices(context_values, standardized_range=standardized_range, n_intervals=n_intervals)
            interval_validation = _span_indices(validation_values, standardized_range=standardized_range, n_intervals=n_intervals)
            interval_test = _span_indices(test_values, standardized_range=standardized_range, n_intervals=n_intervals)
            context_counts = np.bincount(interval_context, minlength=n_intervals).astype(int, copy=False)
            sparse_threshold = max(sparse_min_count, int(math.ceil(sparse_min_fraction * context_values.size)))
            validation_basis.append(
                _basis_mass(
                    context=context_values,
                    query=validation_values,
                    knots=knots,
                    degree=degree,
                    n_control_points=n_control_points,
                    standardized_range=standardized_range,
                )
            )
            test_basis.append(
                _basis_mass(
                    context=context_values,
                    query=test_values,
                    knots=knots,
                    degree=degree,
                    n_control_points=n_control_points,
                    standardized_range=standardized_range,
                )
            )
            validation_counts.append(context_counts[interval_validation])
            test_counts.append(context_counts[interval_test])
            validation_sparse.append(context_counts[interval_validation] < sparse_threshold)
            test_sparse.append(context_counts[interval_test] < sparse_threshold)
            validation_outside.append((validation_values < context_values.min()) | (validation_values > context_values.max()))
            test_outside.append((test_values < context_values.min()) | (test_values > context_values.max()))
            cell = _CellSupport(
                normalization_method=str(method),
                feature_index=feature_index,
                feature_name=str(feature_name),
                context_interval_counts=context_counts,
                validation_interval_indices=interval_validation,
                test_interval_indices=interval_test,
            )
            validation_cells.append(cell)
            test_cells.append(cell)

    if validation_rows is None or test_rows is None or not validation_basis:
        raise RuntimeError("public TabICL preprocessing produced no numerical support coordinates")

    def summarise(
        basis: list[np.ndarray], counts: list[np.ndarray], sparse: list[np.ndarray], outside: list[np.ndarray], cells: list[_CellSupport]
    ) -> _SplitSupport:
        basis_matrix = np.column_stack(basis)
        count_matrix = np.column_stack(counts)
        sparse_matrix = np.column_stack(sparse)
        outside_matrix = np.column_stack(outside)
        return _SplitSupport(
            row_metrics={
                "mean_log1p_basis_support": np.log1p(basis_matrix).mean(axis=1),
                "minimum_basis_support": basis_matrix.min(axis=1),
                "mean_interval_count": count_matrix.mean(axis=1),
                "minimum_interval_count": count_matrix.min(axis=1),
                "fraction_sparse_coordinates": sparse_matrix.mean(axis=1),
                "fraction_outside_context_range": outside_matrix.mean(axis=1),
                # Higher values mean weaker support and are used for rank strata.
                "support_risk": -np.log1p(basis_matrix).mean(axis=1),
                "n_support_coordinates": np.full(basis_matrix.shape[0], basis_matrix.shape[1], dtype=int),
            },
            cells=tuple(cells),
        )

    return (
        summarise(validation_basis, validation_counts, validation_sparse, validation_outside, validation_cells),
        summarise(test_basis, test_counts, test_sparse, test_outside, test_cells),
    )


def _encoded_labels(labels: np.ndarray, *, n_classes: int) -> np.ndarray:
    values = np.asarray(labels)
    unique = np.unique(values)
    if np.array_equal(unique, np.arange(n_classes)):
        encoded = values.astype(int, copy=False)
    else:
        encoded = np.searchsorted(unique, values).astype(int, copy=False)
    if encoded.ndim != 1 or np.any(encoded < 0) or np.any(encoded >= n_classes):
        raise ValueError("cannot align classification labels with frozen prediction columns")
    return encoded


def _per_row_loss(*, problem_type: str, labels: np.ndarray, prediction: np.ndarray, n_classes: int | None) -> np.ndarray:
    """Return a decomposable row loss matching the cohort's deployed metric."""

    target = np.asarray(labels)
    values = np.asarray(prediction, dtype=float)
    if problem_type == "regression":
        if values.ndim != 1 or values.shape != target.shape:
            raise ValueError("regression prediction shape does not match labels")
        return np.square(values - target.astype(float))
    if problem_type != "multiclass" or n_classes is None:
        raise ValueError("row-local audit currently supports only regression and multiclass cohorts")
    if values.ndim != 2 or values.shape[0] != target.size or values.shape[1] != n_classes:
        raise ValueError("multiclass prediction shape does not match labels/classes")
    encoded = _encoded_labels(target, n_classes=n_classes)
    probability = values[np.arange(target.size), encoded]
    return -np.log(np.clip(probability, _EPS, 1.0))


def _rank_strata(risk: np.ndarray, *, n_strata: int) -> np.ndarray:
    """Assign deterministic, balanced low-to-high-risk strata despite ties."""

    values = np.asarray(risk, dtype=float)
    if values.ndim != 1 or values.size < n_strata or not np.isfinite(values).all():
        raise ValueError("risk must be finite one-dimensional with at least n_strata rows")
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=int)
    result[order] = np.minimum(n_strata - 1, (np.arange(values.size) * n_strata) // values.size)
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    centered_x, centered_y = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)))
    return None if denominator <= _EPS else float(np.dot(centered_x, centered_y) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.size < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    return _pearson(_average_ranks(x), _average_ranks(y))


def _row_records(
    *,
    case: SourceCase,
    bag: int,
    split: str,
    source_indices: np.ndarray,
    support: _SplitSupport,
    identity_prediction: np.ndarray,
    adapted_prediction: np.ndarray,
    labels: np.ndarray,
    selected_adapted: bool,
    n_strata: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    """Score a support-frozen split and return row records plus bag association."""

    identity_loss = _per_row_loss(
        problem_type=case.problem_type, labels=labels, prediction=identity_prediction, n_classes=case.n_classes
    )
    adapted_loss = _per_row_loss(
        problem_type=case.problem_type, labels=labels, prediction=adapted_prediction, n_classes=case.n_classes
    )
    delta = adapted_loss - identity_loss
    risk = np.asarray(support.row_metrics["support_risk"], dtype=float)
    if not (risk.shape == delta.shape == np.asarray(source_indices).shape):
        raise RuntimeError("row support, labels, and frozen prediction rows do not align")
    strata = _rank_strata(risk, n_strata=n_strata)
    records: list[dict[str, Any]] = []
    for position in range(delta.size):
        record: dict[str, Any] = {
            "task_id": case.task_id,
            "dataset_id": case.dataset_id,
            "dataset_name": case.dataset_name,
            "problem_type": case.problem_type,
            "config_label": case.config_label,
            "bag": bag,
            "split": split,
            "source_row_index": int(source_indices[position]),
            "guard_selected_adapted": selected_adapted,
            "identity_row_loss": float(identity_loss[position]),
            "adapted_row_loss": float(adapted_loss[position]),
            "adapted_minus_identity_row_loss": float(delta[position]),
            "support_stratum": int(strata[position]),
        }
        for key, values in support.row_metrics.items():
            value = values[position]
            record[key] = int(value) if np.issubdtype(np.asarray(value).dtype, np.integer) else float(value)
        records.append(record)
    per_stratum = [float(delta[strata == level].mean()) for level in range(n_strata)]
    summary = {
        "task_id": case.task_id,
        "dataset_name": case.dataset_name,
        "problem_type": case.problem_type,
        "config_label": case.config_label,
        "bag": bag,
        "split": split,
        "guard_selected_adapted": selected_adapted,
        "n_rows": int(delta.size),
        "mean_adapted_minus_identity_row_loss": float(delta.mean()),
        "support_risk_vs_harm_pearson": _pearson(risk, delta),
        "support_risk_vs_harm_spearman": _spearman(risk, delta),
        "lowest_support_minus_highest_support_mean_harm": float(per_stratum[-1] - per_stratum[0]),
        **{f"mean_harm_support_stratum_{level}": value for level, value in enumerate(per_stratum)},
    }
    return records, summary, delta


def _cell_records(
    *,
    case: SourceCase,
    bag: int,
    split: str,
    support: _SplitSupport,
    delta_loss: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in support.cells:
        interval_indices = (
            cell.validation_interval_indices if split == "validation" else cell.test_interval_indices
        )
        for interval, context_count in enumerate(cell.context_interval_counts):
            query_mask = interval_indices == interval
            n_query = int(query_mask.sum())
            if n_query == 0:
                continue
            local_delta = delta_loss[query_mask]
            records.append(
                {
                    "task_id": case.task_id,
                    "dataset_id": case.dataset_id,
                    "dataset_name": case.dataset_name,
                    "problem_type": case.problem_type,
                    "config_label": case.config_label,
                    "bag": bag,
                    "split": split,
                    "normalization_method": cell.normalization_method,
                    "numerical_feature_index": cell.feature_index,
                    "numerical_feature_name": cell.feature_name,
                    "knot_interval": interval,
                    "context_interval_count": int(context_count),
                    "n_query_rows": n_query,
                    "mean_adapted_minus_identity_row_loss": float(local_delta.mean()),
                    "fraction_rows_harmed": float(np.mean(local_delta > 0.0)),
                }
            )
    return records


def _cell_flip_summary(*, cell_records: Sequence[Mapping[str, Any]], minimum_query_rows: int) -> list[dict[str, Any]]:
    """Compare validation/test benefit in the same feature/span after scoring."""

    grouped: dict[tuple[int, str, int, str, int], dict[str, Mapping[str, Any]]] = {}
    for record in cell_records:
        key = (
            int(record["task_id"]),
            str(record["normalization_method"]),
            int(record["numerical_feature_index"]),
            str(record["numerical_feature_name"]),
            int(record["knot_interval"]),
        )
        grouped.setdefault(key, {})[str(record["split"])] = record
    per_task: dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for pair in grouped.values():
        validation, test = pair.get("validation"), pair.get("test")
        if validation is None or test is None:
            continue
        if int(validation["n_query_rows"]) < minimum_query_rows or int(test["n_query_rows"]) < minimum_query_rows:
            continue
        per_task.setdefault(int(validation["task_id"]), []).append((validation, test))
    results: list[dict[str, Any]] = []
    for task_id, pairs in sorted(per_task.items()):
        validation, test = pairs[0]
        flips = [
            pair
            for pair in pairs
            if float(pair[0]["mean_adapted_minus_identity_row_loss"]) < 0.0
            and float(pair[1]["mean_adapted_minus_identity_row_loss"]) > 0.0
        ]
        results.append(
            {
                "task_id": task_id,
                "dataset_name": validation["dataset_name"],
                "problem_type": validation["problem_type"],
                "eligible_validation_test_cells": len(pairs),
                "validation_help_test_harm_cells": len(flips),
                "validation_help_test_harm_fraction": float(len(flips) / len(pairs)),
                "mean_context_count_flipped_cells": (
                    None if not flips else float(np.mean([int(pair[0]["context_interval_count"]) for pair in flips]))
                ),
                "mean_context_count_nonflipped_cells": (
                    None
                    if len(flips) == len(pairs)
                    else float(
                        np.mean(
                            [
                                int(pair[0]["context_interval_count"])
                                for pair in pairs
                                if pair not in flips
                            ]
                        )
                    )
                ),
            }
        )
    return results


def _task_records(*, bag_records: Sequence[Mapping[str, Any]], cell_flip_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flips = {int(record["task_id"]): record for record in cell_flip_records}
    by_task: dict[int, list[Mapping[str, Any]]] = {}
    for record in bag_records:
        by_task.setdefault(int(record["task_id"]), []).append(record)
    results: list[dict[str, Any]] = []
    for task_id, records in sorted(by_task.items()):
        sample = records[0]
        result: dict[str, Any] = {
            "task_id": task_id,
            "dataset_name": sample["dataset_name"],
            "problem_type": sample["problem_type"],
            "n_bag_split_records": len(records),
        }
        for split in ("validation", "test"):
            split_records = [record for record in records if record["split"] == split]
            for key in (
                "mean_adapted_minus_identity_row_loss",
                "support_risk_vs_harm_pearson",
                "support_risk_vs_harm_spearman",
                "lowest_support_minus_highest_support_mean_harm",
            ):
                values = [float(record[key]) for record in split_records if record.get(key) is not None]
                result[f"{split}_mean_{key}"] = None if not values else float(np.mean(values))
        result.update(flips.get(task_id, {}))
        results.append(result)
    return results


def _validate_task_against_source(*, case: SourceCase, task: Any) -> None:
    for field, source_value, actual_value in (
        ("dataset_id", case.dataset_id, task.dataset_id),
        ("dataset_name", case.dataset_name, task.dataset_name),
        ("problem_type", case.problem_type, task.problem_type),
        ("outer_split_hash", case.outer_split_hash, task.outer_split_hash),
    ):
        if source_value != actual_value:
            raise ValueError(f"OpenML {field} changed for task {case.task_id}: source={source_value!r}, current={actual_value!r}")


def _audit_case(
    *,
    case: SourceCase,
    immutable_run: Mapping[str, Any],
    normal_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if case.problem_type not in {"multiclass", "regression"}:
        raise ValueError(
            f"row-local audit excludes {case.problem_type!r} task {case.task_id}: its deployed metric is not row-decomposable"
        )
    task = load_tabarena_openml_task(case.task_id, cache_dir=args.openml_cache_dir)
    _validate_task_against_source(case=case, task=task)
    config_summary = _load_json(case.config_summary_path, label="source config summary")
    predictions = _load_predictions(case.config_predictions_path)
    protocol_seed = _as_int(immutable_run.get("protocol_seed"), name="source protocol_seed")
    expected_splits = list(_bag_splits(task, requested_bags=case.requested_bags, seed=_seed(protocol_seed, task.task_id, 0)))
    if len(expected_splits) != case.effective_bags:
        raise RuntimeError("source effective bags differ from deterministic split replay")
    degree = int(case.config.get("degree", 3))
    n_control_points = _as_int(case.config.get("n_control_points"), name="n_control_points")
    standardized_range = float(case.config.get("standardized_range", 4.0))
    if degree != 3 or n_control_points != 20:
        raise ValueError("row-local audit currently verifies the intended uniform cubic K20 configuration only")

    pending: list[tuple[int, Any, np.ndarray, _SplitSupport, _SplitSupport]] = []
    bag_results: list[Any] = []
    # No labels are used in this loop.  It fixes every row/cell support record
    # before the later post-hoc scoring block reads y_test.
    for bag_index, (fit_indices, validation_indices) in enumerate(expected_splits):
        bag_result = _load_bag(case.config_dir / f"bag_{bag_index}.npz")
        if int(bag_result.metadata.get("bag", -1)) != bag_index:
            raise RuntimeError("source bag metadata has wrong bag index")
        if not np.array_equal(np.asarray(bag_result.validation_indices, dtype=int), np.asarray(validation_indices, dtype=int)):
            raise RuntimeError("source bag validation split differs from deterministic replay")
        fit_x = task.x_train.iloc[fit_indices].reset_index(drop=True)
        validation_x = task.x_train.iloc[validation_indices].reset_index(drop=True)
        views, feature_names = _fit_preprocessing_views(
            fit_x=fit_x,
            validation_x=validation_x,
            test_x=task.x_test,
            fit_labels=task.y_train[fit_indices],
            problem_type=case.problem_type,
            normal_config=dict(normal_config),
        )
        validation_support, test_support = _row_support(
            views=views,
            feature_names=feature_names,
            degree=degree,
            n_control_points=n_control_points,
            standardized_range=standardized_range,
            sparse_min_count=args.sparse_min_count,
            sparse_min_fraction=args.sparse_min_fraction,
        )
        pending.append((bag_index, bag_result, np.asarray(validation_indices, dtype=int), validation_support, test_support))
        bag_results.append(bag_result)
        print(f"task {case.task_id} bag {bag_index + 1}/{case.effective_bags}: support fixed", flush=True)

    # All support information is fixed.  This post-hoc block only scores
    # frozen bag predictions; it cannot influence splitting, preprocessing, or
    # an experiment decision.
    row_records: list[dict[str, Any]] = []
    bag_records: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    for bag_index, bag_result, validation_indices, validation_support, test_support in pending:
        selected_adapted = bool(bag_result.metadata.get("guard_selected_adapted"))
        validation_rows, validation_summary, validation_delta = _row_records(
            case=case,
            bag=bag_index,
            split="validation",
            source_indices=validation_indices,
            support=validation_support,
            identity_prediction=np.asarray(bag_result.identity_validation),
            adapted_prediction=np.asarray(bag_result.adapted_validation),
            labels=np.asarray(task.y_train)[validation_indices],
            selected_adapted=selected_adapted,
            n_strata=args.row_strata,
        )
        test_rows, test_summary, test_delta = _row_records(
            case=case,
            bag=bag_index,
            split="test",
            source_indices=np.arange(task.y_test.size, dtype=int),
            support=test_support,
            identity_prediction=np.asarray(bag_result.identity_test),
            adapted_prediction=np.asarray(bag_result.adapted_test),
            labels=np.asarray(task.y_test),
            selected_adapted=selected_adapted,
            n_strata=args.row_strata,
        )
        row_records.extend(validation_rows)
        row_records.extend(test_rows)
        bag_records.extend((validation_summary, test_summary))
        cell_records.extend(_cell_records(case=case, bag=bag_index, split="validation", support=validation_support, delta_loss=validation_delta))
        cell_records.extend(_cell_records(case=case, bag=bag_index, split="test", support=test_support, delta_loss=test_delta))
    _validate_aggregated_predictions(case=case, bag_results=bag_results, predictions=predictions)
    _assert_summary_validation_metrics(case=case, config_summary=config_summary, task=task, predictions=predictions)
    return row_records, bag_records, cell_records


def _manifest(*, source_dir: Path, source_manifest: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "DirectSpline row-local nominal support versus frozen-loss audit",
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
        "source_repository_revision": source_manifest.get("immutable_run", {}).get("repository_revision"),
        "config_label": args.config_label,
        "task_ids": None if args.task_id is None else sorted(set(args.task_id)),
        "support_definition": {
            "basis_support": "sum_n dot(B(query), B(fit_row_n)) using actual uniform cubic K20 active basis functions",
            "span_occupancy": "fitting-row count in the query coordinate's nominal K20 knot interval",
            "learned_deformation_available": False,
            "limitation": "source Retouche bags do not persist learned adapter state; this audit cannot recover f_j(x)-x or derivatives",
        },
        "parameters": {
            "sparse_min_count": args.sparse_min_count,
            "sparse_min_fraction": args.sparse_min_fraction,
            "minimum_cell_query_rows": args.minimum_cell_query_rows,
            "row_strata": args.row_strata,
        },
        "test_label_role": "outer-test labels score frozen predictions only after all row/cell support records are fixed",
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing row-local audit manifest")
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different row-local audit; choose a new --output-dir")
        if not resume:
            raise ValueError("row-local audit output directory already exists; pass --resume to rewrite matching results")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def main() -> None:
    args = _parse_args()
    source_dir = args.source_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    normal_config = _source_normal_config(immutable_run)
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=None if args.task_id is None else set(args.task_id),
    )
    manifest = _manifest(source_dir=source_dir, source_manifest=source_manifest, args=args)
    _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))

    rows: list[dict[str, Any]] = []
    bags: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}: replaying row support", flush=True)
        task_rows, task_bags, task_cells = _audit_case(
            case=case,
            immutable_run=immutable_run,
            normal_config=normal_config,
            args=args,
        )
        rows.extend(task_rows)
        bags.extend(task_bags)
        cells.extend(task_cells)
    rows.sort(key=lambda record: (int(record["task_id"]), int(record["bag"]), str(record["split"]), int(record["source_row_index"])))
    bags.sort(key=lambda record: (int(record["task_id"]), int(record["bag"]), str(record["split"])))
    cells.sort(key=lambda record: (int(record["task_id"]), int(record["bag"]), str(record["split"]), str(record["normalization_method"]), int(record["numerical_feature_index"]), int(record["knot_interval"])))
    flips = _cell_flip_summary(cell_records=cells, minimum_query_rows=args.minimum_cell_query_rows)
    tasks = _task_records(bag_records=bags, cell_flip_records=flips)
    paths = {
        "row_records_csv": args.output_dir / "row_records.csv",
        "bag_records_csv": args.output_dir / "bag_records.csv",
        "cell_records_csv": args.output_dir / "cell_records.csv",
        "cell_flip_records_csv": args.output_dir / "cell_flip_records.csv",
        "task_records_csv": args.output_dir / "task_records.csv",
        "task_records_json": args.output_dir / "task_records.json",
    }
    _write_csv(paths["row_records_csv"], rows)
    _write_csv(paths["bag_records_csv"], bags)
    _write_csv(paths["cell_records_csv"], cells)
    _write_csv(paths["cell_flip_records_csv"], flips)
    _write_csv(paths["task_records_csv"], tasks)
    _write_json(paths["task_records_json"], tasks)
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "source": {
            "source_dir": str(source_dir),
            "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
            "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
            "source_repository_revision": immutable_run.get("repository_revision"),
            "config_label": args.config_label,
        },
        "scope": {
            "n_tasks": len(tasks),
            "n_bag_split_records": len(bags),
            "n_row_records": len(rows),
            "n_cell_records": len(cells),
            "test_label_role": manifest["test_label_role"],
        },
        "support_definition": manifest["support_definition"],
        "parameters": manifest["parameters"],
        "interpretation": {
            "positive_row_harm": "adapted per-row loss minus identity per-row loss is positive",
            "positive_support_risk_association": "weak nominal basis support is associated with greater DirectSpline harm within a bag/split",
            "positive_lowest_support_minus_highest_support_harm": "the least-supported row stratum has more DirectSpline harm",
            "validation_help_test_harm_cell": "a feature/span has negative mean adapted-minus-identity validation loss and positive test loss; descriptive only, never a selection rule",
        },
        "task_records": tasks,
        "files": {key: str(path) for key, path in paths.items()},
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(tasks), "n_rows": len(rows), "n_cells": len(cells)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
