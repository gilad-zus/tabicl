"""Audit whether a DirectSpline knot configuration is statistically supported.

This is a CPU-only, label-free-with-respect-to-test diagnostic for completed
standard-pipeline DirectSpline runs.  It replays the exact public TabICL
numerical preprocessing for every saved inner bag and measures the leverage of
several hypothetical B-spline bases fitted on that bag's fitting rows.

For a column's fitting design matrix ``B`` and a query basis vector ``b(x)``,
the audit reports the ridge-stabilised basis leverage

``b(x)^T (B^T B + lambda I)^-1 b(x)``.

Large leverage means that the local basis coordinates used at ``x`` are weakly
determined by the fitting context.  The value is normalised by mean fitting-row
leverage before validation/test comparisons, so candidate bases with different
numbers of control points can be compared as *support shifts*, rather than
only by their different parameter counts.

The audit does not train a spline, inspect an adapter checkpoint, alter a
prediction, or use outer-test labels.  Frozen DirectSpline outcomes are read
from a completed support-audit task-record file only after all leverage records
have been constructed.  It therefore answers a structural question:

    Would a coarser, finer, or context-quantile B-spline basis provide better
    support for the validation and test coordinates than the current K=20
    uniform cubic basis?

It does *not* establish that a candidate will predict better.  That requires a
subsequent train--validation--test configuration-bank experiment.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_basis_leverage_audit.py \
      --source-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828 \
      --outcome-records /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_support_audit/multiclass_D/task_records.json \
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_basis_leverage_audit/multiclass_D
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from direct_spline_openml_support_audit import (
    SourceCase,
    _as_int,
    _bag_splits,
    _bootstrap_correlation,
    _canonical_json,
    _correlation,
    _find_source_cases,
    _fit_preprocessing_views,
    _load_bag,
    _load_json,
    _seed,
    _sha256,
    _source_normal_config,
    _write_csv,
    _write_json,
    load_tabarena_openml_task,
)


AUDIT_SCHEMA_VERSION = 1
_MIN_KNOT_INTERVAL = 0.01
_OUTCOME_FIELDS = (
    "test_adapted_regret",
    "test_adapted_relative_improvement",
    "test_adapted_outcome",
    "validation_adapted_relative_improvement",
)


@dataclass(frozen=True)
class BasisCandidate:
    """One hypothetical spline basis; no candidate is fitted or trained."""

    name: str
    placement: str
    degree: int
    n_control_points: int


def _parse_control_points(value: str, *, option: str, degree: int) -> tuple[int, ...]:
    try:
        points = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{option} must be comma-separated integer control-point counts") from error
    if not points:
        raise ValueError(f"{option} must provide at least one control-point count")
    if len(points) != len(set(points)):
        raise ValueError(f"{option} must not repeat a control-point count")
    if any(point <= degree for point in points):
        raise ValueError(f"every {option} count must be greater than --degree")
    if any(_MIN_KNOT_INTERVAL * (point - degree) >= 2.0 for point in points):
        raise ValueError(f"{option} requests too many intervals for the DirectSpline knot floor")
    return points


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Completed standard-pipeline DirectSpline source run.")
    parser.add_argument(
        "--outcome-records",
        type=Path,
        required=True,
        help="Completed support-audit task_records.json from the same source run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D", help="Frozen source configuration label (default: D).")
    parser.add_argument("--task-id", type=int, action="append", help="Audit only these completed task IDs. Repeatable.")
    parser.add_argument("--openml-cache-dir", type=Path, default=None, help="Optional cache used only for task/split replay.")
    parser.add_argument("--degree", type=int, default=3, help="Candidate B-spline degree (default: 3).")
    parser.add_argument(
        "--uniform-control-points",
        default="8,10,20,40",
        help="Comma-separated uniform-knot control-point counts (default: 8,10,20,40).",
    )
    parser.add_argument(
        "--quantile-control-points",
        default="20",
        help="Comma-separated context-quantile-knot control-point counts (default: 20).",
    )
    parser.add_argument(
        "--ridge-relative",
        type=float,
        default=1e-6,
        help="Ridge as a fraction of mean Gram diagonal (default: 1e-6).",
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--resume", action="store_true", help="Rewrite only a matching immutable audit output directory.")
    args = parser.parse_args()
    if args.degree < 1:
        raise ValueError("--degree must be positive")
    args.uniform_control_points = _parse_control_points(
        args.uniform_control_points, option="--uniform-control-points", degree=args.degree
    )
    args.quantile_control_points = _parse_control_points(
        args.quantile_control_points, option="--quantile-control-points", degree=args.degree
    )
    if args.ridge_relative <= 0.0 or not math.isfinite(args.ridge_relative):
        raise ValueError("--ridge-relative must be finite and positive")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    return args


def _candidates(args: argparse.Namespace) -> tuple[BasisCandidate, ...]:
    candidates = tuple(
        [BasisCandidate(f"uniform_k{points}", "uniform", args.degree, points) for points in args.uniform_control_points]
        + [BasisCandidate(f"quantile_k{points}", "quantile", args.degree, points) for points in args.quantile_control_points]
    )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate basis names must be unique")
    return candidates


def _uniform_knots(*, n_control_points: int, degree: int) -> np.ndarray:
    """Return the exact open-uniform, clamped DirectSpline knot vector."""

    n_internal = n_control_points - degree - 1
    internal = np.linspace(-1.0, 1.0, n_internal + 2, dtype=float)[1:-1]
    return np.concatenate((np.full(degree + 1, -1.0), internal, np.full(degree + 1, 1.0)))


def _strict_widths(widths: np.ndarray) -> np.ndarray:
    """Match DirectSpline's non-collapsing quantile-knot projection."""

    values = np.asarray(widths, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("knot widths must be a finite nonempty one-dimensional array")
    if _MIN_KNOT_INTERVAL * values.size >= 2.0:
        raise ValueError("too many knot intervals for the DirectSpline minimum interval")
    relative = np.clip(values, np.finfo(float).eps, None)
    relative = relative / relative.sum()
    return _MIN_KNOT_INTERVAL + (2.0 - _MIN_KNOT_INTERVAL * values.size) * relative


def _quantile_knots(*, context: np.ndarray, n_control_points: int, degree: int, standardized_range: float) -> np.ndarray:
    """Return per-column context-quantile knots with DirectSpline's width floor."""

    values = np.asarray(context, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("context coordinates must be finite and nonempty")
    if standardized_range <= 0.0:
        raise ValueError("standardized_range must be positive")
    intervals = n_control_points - degree
    u = np.clip(values / standardized_range, -1.0, 1.0)
    if intervals == 1:
        widths = np.array([2.0], dtype=float)
    else:
        quantiles = np.quantile(u, np.linspace(0.0, 1.0, intervals + 1, dtype=float)[1:-1])
        boundaries = np.concatenate((np.array([-1.0]), np.asarray(quantiles, dtype=float), np.array([1.0])))
        widths = np.diff(boundaries)
    widths = _strict_widths(widths)
    internal = -1.0 + np.cumsum(widths)[:-1]
    return np.concatenate((np.full(degree + 1, -1.0), internal, np.full(degree + 1, 1.0)))


def _bspline_design(*, values: np.ndarray, knots: np.ndarray, degree: int, n_control_points: int, standardized_range: float) -> np.ndarray:
    """Evaluate the local Cox--de Boor design used by DirectSpline on CPU."""

    x = np.asarray(values, dtype=float).reshape(-1)
    knot_vector = np.asarray(knots, dtype=float).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("basis values must be finite and nonempty")
    if standardized_range <= 0.0:
        raise ValueError("standardized_range must be positive")
    if knot_vector.size != n_control_points + degree + 1:
        raise ValueError("knot vector does not match spline dimensions")
    if not np.all(np.diff(knot_vector) >= 0.0):
        raise ValueError("knot vector must be ordered")
    u = np.clip(x / standardized_range, -1.0, 1.0)
    spans = np.searchsorted(knot_vector, u, side="right") - 1
    spans = np.clip(spans, degree, n_control_points - 1).astype(int, copy=False)
    local = np.ones((u.size, degree + 1), dtype=float)
    left = np.zeros_like(local)
    right = np.zeros_like(local)
    eps = np.finfo(float).eps
    for order in range(1, degree + 1):
        left[:, order] = u - knot_vector[spans + 1 - order]
        right[:, order] = knot_vector[spans + order] - u
        saved = np.zeros(u.size, dtype=float)
        for index in range(order):
            denominator = right[:, index + 1] + left[:, order - index]
            temporary = np.divide(local[:, index], denominator, out=np.zeros_like(denominator), where=np.abs(denominator) > eps)
            local[:, index] = saved + right[:, index + 1] * temporary
            saved = left[:, order - index] * temporary
        local[:, order] = saved
    indices = spans[:, None] - degree + np.arange(degree + 1, dtype=int)[None, :]
    design = np.zeros((u.size, n_control_points), dtype=float)
    np.add.at(design, (np.arange(u.size)[:, None], indices), local)
    if not np.allclose(design.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10):
        raise RuntimeError("B-spline basis does not form a partition of unity")
    return design


def _split_leverage(*, context: np.ndarray, validation: np.ndarray, test: np.ndarray, candidate: BasisCandidate, standardized_range: float, ridge_relative: float) -> dict[str, float]:
    """Return raw and context-normalised leverage for one feature/view/basis."""

    knots = (
        _uniform_knots(n_control_points=candidate.n_control_points, degree=candidate.degree)
        if candidate.placement == "uniform"
        else _quantile_knots(
            context=context,
            n_control_points=candidate.n_control_points,
            degree=candidate.degree,
            standardized_range=standardized_range,
        )
    )
    design_context = _bspline_design(
        values=context,
        knots=knots,
        degree=candidate.degree,
        n_control_points=candidate.n_control_points,
        standardized_range=standardized_range,
    )
    gram = design_context.T @ design_context
    mean_diagonal = float(np.trace(gram) / candidate.n_control_points)
    ridge = ridge_relative * max(mean_diagonal, np.finfo(float).eps)
    regularised = gram + ridge * np.eye(candidate.n_control_points, dtype=float)
    eigenvalues = np.linalg.eigvalsh(regularised)
    condition = float(eigenvalues[-1] / max(float(eigenvalues[0]), np.finfo(float).eps))

    def leverage(values: np.ndarray) -> np.ndarray:
        design = _bspline_design(
            values=values,
            knots=knots,
            degree=candidate.degree,
            n_control_points=candidate.n_control_points,
            standardized_range=standardized_range,
        )
        solved = np.linalg.solve(regularised, design.T).T
        result = np.einsum("ij,ij->i", design, solved)
        if not np.isfinite(result).all() or np.any(result < -1e-12):
            raise RuntimeError("basis leverage must be finite and nonnegative")
        return np.maximum(result, 0.0)

    context_values = leverage(context)
    validation_values = leverage(validation)
    test_values = leverage(test)
    context_mean = float(np.mean(context_values))
    if not context_mean > 0.0:
        raise RuntimeError("mean fitting-context leverage must be positive")

    def metrics(prefix: str, values: np.ndarray) -> dict[str, float]:
        mean = float(np.mean(values))
        return {
            f"{prefix}_mean_leverage": mean,
            f"{prefix}_median_leverage": float(np.median(values)),
            f"{prefix}_p95_leverage": float(np.quantile(values, 0.95)),
            f"{prefix}_max_leverage": float(np.max(values)),
            f"{prefix}_mean_relative_leverage": float(mean / context_mean),
        }

    result = {
        "ridge": ridge,
        "regularised_gram_condition_number": condition,
        "effective_parameter_count": float(np.trace(np.linalg.solve(regularised, gram))),
        **metrics("context", context_values),
        **metrics("validation", validation_values),
        **metrics("test", test_values),
    }
    for suffix in ("mean_leverage", "median_leverage", "p95_leverage", "max_leverage", "mean_relative_leverage"):
        result[f"test_minus_validation_{suffix}"] = float(result[f"test_{suffix}"] - result[f"validation_{suffix}"])
    return result


def _feature_records(*, case: SourceCase, bag: int, views: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], feature_names: list[str], candidates: tuple[BasisCandidate, ...], standardized_range: float, ridge_relative: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method, (context, validation, test) in views.items():
        if context.shape[1] != len(feature_names):
            raise RuntimeError("numerical preprocessing feature names do not match replayed views")
        for feature_index, name in enumerate(feature_names):
            for candidate in candidates:
                metrics = _split_leverage(
                    context=context[:, feature_index],
                    validation=validation[:, feature_index],
                    test=test[:, feature_index],
                    candidate=candidate,
                    standardized_range=standardized_range,
                    ridge_relative=ridge_relative,
                )
                records.append(
                    {
                        "task_id": case.task_id,
                        "dataset_id": case.dataset_id,
                        "dataset_name": case.dataset_name,
                        "problem_type": case.problem_type,
                        "config_label": case.config_label,
                        "bag": bag,
                        "normalization_method": method,
                        "numerical_feature_index": feature_index,
                        "numerical_feature_name": name,
                        "candidate": candidate.name,
                        "knot_placement": candidate.placement,
                        "spline_degree": candidate.degree,
                        "spline_control_points": candidate.n_control_points,
                        "spline_knot_intervals": candidate.n_control_points - candidate.degree,
                        "nominal_standardized_range": standardized_range,
                        **metrics,
                    }
                )
    return records


def _load_outcomes(path: Path) -> dict[int, dict[str, Any]]:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read frozen outcome records {path}: {type(error).__name__}: {error}") from error
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("--outcome-records must be a JSON list of support-audit task records")
    outcomes: dict[int, dict[str, Any]] = {}
    for row in payload:
        task_id = _as_int(row.get("task_id"), name="outcome task_id")
        if task_id in outcomes:
            raise ValueError(f"duplicate task {task_id} in --outcome-records")
        missing = [field for field in _OUTCOME_FIELDS if field not in row]
        if missing:
            raise ValueError(f"outcome record {task_id} lacks {missing}")
        outcomes[task_id] = row
    return outcomes


def _audit_manifest(*, source_dir: Path, source_manifest: Mapping[str, Any], args: argparse.Namespace, candidates: tuple[BasisCandidate, ...]) -> dict[str, Any]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "DirectSpline nominal B-spline basis-leverage audit",
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
        "source_repository_revision": source_manifest.get("immutable_run", {}).get("repository_revision"),
        "outcome_records": str(args.outcome_records.resolve()),
        "outcome_records_sha256": _sha256(args.outcome_records),
        "config_label": args.config_label,
        "task_ids": None if args.task_id is None else sorted(set(args.task_id)),
        "candidates": [candidate.__dict__ for candidate in candidates],
        "parameters": {
            "ridge_relative": args.ridge_relative,
            "minimum_knot_interval": _MIN_KNOT_INTERVAL,
            "bootstrap_rounds": args.bootstrap_rounds,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "test_label_role": "none; frozen test outcome fields come from --outcome-records after basis leverage is computed",
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing basis-leverage manifest")
        if _canonical_json(existing) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different immutable basis-leverage audit; choose a new --output-dir")
        if not resume:
            raise ValueError("basis-leverage output directory already exists; pass --resume to rewrite matching results")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def _validate_task_against_source(*, case: SourceCase, task: Any) -> None:
    for field, source_value, actual_value in (
        ("dataset_id", case.dataset_id, task.dataset_id),
        ("dataset_name", case.dataset_name, task.dataset_name),
        ("problem_type", case.problem_type, task.problem_type),
        ("outer_split_hash", case.outer_split_hash, task.outer_split_hash),
    ):
        if source_value != actual_value:
            raise ValueError(f"OpenML {field} changed for task {case.task_id}: source={source_value!r}, current={actual_value!r}")


def _audit_case(*, case: SourceCase, immutable_run: Mapping[str, Any], normal_config: Mapping[str, Any], candidates: tuple[BasisCandidate, ...], ridge_relative: float) -> list[dict[str, Any]]:
    outer_split = immutable_run.get("data_source", {}).get("outer_split")
    if not isinstance(outer_split, Mapping):
        raise ValueError("source manifest has no immutable OpenML outer split")
    task = load_tabarena_openml_task(
        case.task_id,
        outer_repeat=_as_int(outer_split.get("repeat"), name="outer repeat"),
        outer_fold=_as_int(outer_split.get("fold"), name="outer fold"),
        outer_sample=_as_int(outer_split.get("sample"), name="outer sample"),
    )
    _validate_task_against_source(case=case, task=task)
    protocol_seed = _as_int(immutable_run.get("protocol_seed"), name="source protocol_seed")
    expected_splits = list(_bag_splits(task, requested_bags=case.requested_bags, seed=_seed(protocol_seed, task.task_id, 0)))
    if len(expected_splits) != case.effective_bags:
        raise RuntimeError(f"source effective bags differ from deterministic replay for task {case.task_id}")
    standardized_range = float(case.config.get("standardized_range", 4.0))
    if standardized_range <= 0.0:
        raise ValueError("source DirectSpline standardized_range must be positive")
    records: list[dict[str, Any]] = []
    for bag_index, (fit_indices, validation_indices) in enumerate(expected_splits):
        bag_path = case.config_dir / f"bag_{bag_index}.npz"
        bag_result = _load_bag(bag_path)
        if int(bag_result.metadata.get("bag", -1)) != bag_index:
            raise RuntimeError(f"source bag metadata has wrong bag index: {bag_path}")
        if not np.array_equal(np.asarray(bag_result.validation_indices, dtype=int), np.asarray(validation_indices, dtype=int)):
            raise RuntimeError(f"source bag validation split differs from deterministic replay: {bag_path}")
        fit_x = task.x_train.iloc[fit_indices].reset_index(drop=True)
        validation_x = task.x_train.iloc[validation_indices].reset_index(drop=True)
        views, feature_names = _fit_preprocessing_views(
            fit_x=fit_x,
            validation_x=validation_x,
            test_x=task.x_test,
            fit_labels=task.y_train[fit_indices],
            problem_type=case.problem_type,
            normal_config=normal_config,
        )
        records.extend(
            _feature_records(
                case=case,
                bag=bag_index,
                views=views,
                feature_names=feature_names,
                candidates=candidates,
                standardized_range=standardized_range,
                ridge_relative=ridge_relative,
            )
        )
        print(f"task {case.task_id} bag {bag_index + 1}/{case.effective_bags}: {len(records)} cumulative feature-basis records", flush=True)
    return records


def _mean(records: list[dict[str, Any]], field: str) -> float:
    values = np.asarray([float(record[field]) for record in records], dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"cannot aggregate non-finite {field}")
    return float(np.mean(values))


def _aggregate_records(*, feature_records: list[dict[str, Any]], outcomes: Mapping[int, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_bag: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    by_task: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in feature_records:
        candidate = str(record["candidate"])
        by_bag.setdefault((int(record["task_id"]), int(record["bag"]), candidate), []).append(record)
        by_task.setdefault((int(record["task_id"]), candidate), []).append(record)
    bag_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    aggregate_fields = (
        "context_mean_leverage",
        "validation_mean_leverage",
        "test_mean_leverage",
        "context_mean_relative_leverage",
        "validation_mean_relative_leverage",
        "test_mean_relative_leverage",
        "test_minus_validation_mean_leverage",
        "test_minus_validation_mean_relative_leverage",
        "regularised_gram_condition_number",
        "effective_parameter_count",
    )
    for (task_id, bag, candidate), records in sorted(by_bag.items()):
        exemplar = records[0]
        bag_records.append(
            {
                "task_id": task_id,
                "dataset_name": exemplar["dataset_name"],
                "problem_type": exemplar["problem_type"],
                "bag": bag,
                "candidate": candidate,
                "knot_placement": exemplar["knot_placement"],
                "spline_degree": exemplar["spline_degree"],
                "spline_control_points": exemplar["spline_control_points"],
                "n_feature_view_records": len(records),
                **{field: _mean(records, field) for field in aggregate_fields},
            }
        )
    for (task_id, candidate), records in sorted(by_task.items()):
        if task_id not in outcomes:
            raise ValueError(f"source task {task_id} has no frozen outcome record")
        exemplar = records[0]
        outcome = outcomes[task_id]
        relative_improvement = outcome["test_adapted_relative_improvement"]
        if relative_improvement is None or not math.isfinite(float(relative_improvement)):
            relative_regret = None
        else:
            relative_regret = -float(relative_improvement)
        task_records.append(
            {
                "task_id": task_id,
                "dataset_id": exemplar["dataset_id"],
                "dataset_name": exemplar["dataset_name"],
                "problem_type": exemplar["problem_type"],
                "candidate": candidate,
                "knot_placement": exemplar["knot_placement"],
                "spline_degree": exemplar["spline_degree"],
                "spline_control_points": exemplar["spline_control_points"],
                "n_feature_view_records": len(records),
                **{field: _mean(records, field) for field in aggregate_fields},
                "test_adapted_regret": outcome["test_adapted_regret"],
                "test_adapted_relative_improvement": relative_improvement,
                "test_adapted_relative_regret": relative_regret,
                "test_adapted_outcome": outcome["test_adapted_outcome"],
                "validation_adapted_relative_improvement": outcome["validation_adapted_relative_improvement"],
            }
        )
    return bag_records, task_records


def _candidate_summary(task_records: list[dict[str, Any]], *, candidate: str, bootstrap_rounds: int, bootstrap_seed: int) -> dict[str, Any]:
    rows = [record for record in task_records if record["candidate"] == candidate]
    predictor = "test_minus_validation_mean_relative_leverage"
    usable = [record for record in rows if record["test_adapted_relative_regret"] is not None]
    x = np.asarray([float(record[predictor]) for record in usable], dtype=float)
    y = np.asarray([float(record["test_adapted_relative_regret"]) for record in usable], dtype=float)
    outcome_groups: dict[str, dict[str, Any]] = {}
    for outcome in sorted({str(record["test_adapted_outcome"]) for record in rows}):
        group = [record for record in rows if str(record["test_adapted_outcome"]) == outcome]
        outcome_groups[outcome] = {
            "n_tasks": len(group),
            "mean_support_shift": _mean(group, predictor),
            "median_support_shift": float(np.median([float(record[predictor]) for record in group])),
        }
    return {
        "candidate": candidate,
        "n_tasks": len(rows),
        "predictor": predictor,
        "outcome": "negative test_adapted_relative_improvement; positive means DirectSpline harmed",
        "correlation": _correlation(x, y),
        "task_bootstrap": _bootstrap_correlation(x, y, rounds=bootstrap_rounds, seed=bootstrap_seed),
        "by_frozen_directspline_outcome": outcome_groups,
    }


def _summary(*, task_records: list[dict[str, Any]], candidates: tuple[BasisCandidate, ...], bootstrap_rounds: int, bootstrap_seed: int) -> dict[str, Any]:
    names = [candidate.name for candidate in candidates]
    candidate_summaries = {
        name: _candidate_summary(
            task_records,
            candidate=name,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed,
        )
        for name in names
    }
    baseline = "uniform_k20" if "uniform_k20" in names else names[0]
    baseline_rows = {int(row["task_id"]): row for row in task_records if row["candidate"] == baseline}
    pairwise: dict[str, Any] = {}
    for candidate in names:
        rows = {int(row["task_id"]): row for row in task_records if row["candidate"] == candidate}
        common = sorted(set(baseline_rows) & set(rows))
        deltas = np.asarray(
            [
                float(rows[task_id]["test_minus_validation_mean_relative_leverage"])
                - float(baseline_rows[task_id]["test_minus_validation_mean_relative_leverage"])
                for task_id in common
            ],
            dtype=float,
        )
        pairwise[candidate] = {
            "reference_candidate": baseline,
            "n_tasks": len(common),
            "mean_support_shift_delta": float(np.mean(deltas)),
            "median_support_shift_delta": float(np.median(deltas)),
            "fraction_with_lower_support_shift": float(np.mean(deltas < 0.0)),
        }
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "DirectSpline nominal B-spline basis-leverage audit",
        "n_tasks": len({int(record["task_id"]) for record in task_records}),
        "candidate_summaries": candidate_summaries,
        "candidate_support_shift_vs_uniform_k20": pairwise,
        "interpretation": (
            "The audit compares only the information geometry of hypothetical bases. "
            "Lower leverage shift means validation/test coordinates are better supported relative to fitting rows, "
            "not that the candidate would necessarily lower prediction error."
        ),
    }


def main() -> None:
    args = _parse_args()
    source_dir = args.source_dir.resolve()
    if args.openml_cache_dir is not None:
        import os

        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    candidates = _candidates(args)
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    normal_config = _source_normal_config(immutable_run)
    requested = None if args.task_id is None else set(args.task_id)
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=requested,
    )
    outcomes = _load_outcomes(args.outcome_records.resolve())
    missing_outcomes = sorted(case.task_id for case in cases if case.task_id not in outcomes)
    if missing_outcomes:
        raise ValueError(f"--outcome-records lacks source tasks {missing_outcomes}")
    manifest = _audit_manifest(source_dir=source_dir, source_manifest=source_manifest, args=args, candidates=candidates)
    _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))
    feature_records: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}: replaying basis support", flush=True)
        feature_records.extend(
            _audit_case(
                case=case,
                immutable_run=immutable_run,
                normal_config=normal_config,
                candidates=candidates,
                ridge_relative=args.ridge_relative,
            )
        )
    bag_records, task_records = _aggregate_records(feature_records=feature_records, outcomes=outcomes)
    summary = _summary(
        task_records=task_records,
        candidates=candidates,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_csv(args.output_dir / "feature_records.csv", feature_records)
    _write_csv(args.output_dir / "bag_records.csv", bag_records)
    _write_csv(args.output_dir / "task_records.csv", task_records)
    _write_json(args.output_dir / "task_records.json", task_records)
    summary["files"] = {
        "feature_records_csv": str(args.output_dir / "feature_records.csv"),
        "bag_records_csv": str(args.output_dir / "bag_records.csv"),
        "task_records_csv": str(args.output_dir / "task_records.csv"),
        "task_records_json": str(args.output_dir / "task_records.json"),
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(
        {
            "n_tasks": summary["n_tasks"],
            "candidates": [candidate.name for candidate in candidates],
            "summary": str(args.output_dir / "summary.json"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
