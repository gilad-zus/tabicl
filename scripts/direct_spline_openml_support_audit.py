"""Audit whether DirectSpline failures coincide with unsupported query regions.

This is a retrospective, CPU-only diagnostic for a completed standard-pipeline
DirectSpline result.  For each completed inner bag it replays *only* the public
TabICL numerical preprocessing fitted on that bag's fitting rows; it never
loads a TabICL checkpoint or trains an adapter.  It then measures how much of
the validation and outer-test feature mass lies in numerically unsupported
regions relative to that fitting context, and joins those features to the
already-frozen identity/adapted prediction artifacts.

"Unsupported" is deliberately a property of the fitted numerical coordinate
system, not an assertion about labels: a query value is unsupported when it is
outside the empirical fitting-context range or falls in a fixed spline interval
whose fitting-context occupancy is below the declared threshold.  Tail and
boundary mass are also reported separately.  The fixed D arm has a cubic
20-control-point spline, hence 17 non-degenerate knot intervals on ``[-4, 4]``
in the TabICL-preprocessed coordinates.

The saved standard-pipeline bags do not contain learned adapter state.  Thus
this audit cannot recover learned gates, learned location/scale, derivatives,
or the actual post-training active spline coordinate.  It is intentionally a
nominal-support audit.  A positive or ambiguous result motivates an
instrumented replay that saves those learned-state quantities.

Outer-test targets never enter preprocessing, support calculation, splitting,
or any audit decision.  They are used solely to score frozen prediction arrays
from the source experiment after all support features have been calculated.

Examples
--------

Run one cohort on a CPU host::

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \\
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_support_audit.py \\
      --source-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828 \\
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_support_audit/multiclass_D

After running several cohorts, combine their task-level evidence::

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \\
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_support_audit.py \\
      --input-summary .../multiclass_D/summary.json \\
      --input-summary .../regression_D/summary.json \\
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_support_audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from tabicl._experiments.direct_spline_openml import (
    STANDARD_TABICL_CONFIG,
    _bag_splits,
    _load_bag,
    _seed,
    _safe_name,
    deployment_error,
    load_tabarena_openml_task,
)
from tabicl._sklearn.preprocessing import EnsembleGenerator, TransformToNumerical


AUDIT_SCHEMA_VERSION = 1
_TIE_ATOL = 1e-12
_PREDICTION_ATOL = 1e-7


@dataclass(frozen=True)
class SourceCase:
    """One completed source task/configuration pair."""

    source_dir: Path
    config_dir: Path
    config_summary_path: Path
    config_predictions_path: Path
    task_id: int
    dataset_id: int
    dataset_name: str
    problem_type: str
    n_classes: int | None
    outer_split_hash: str
    config_label: str
    config: dict[str, Any]
    requested_bags: int
    effective_bags: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source-dir",
        type=Path,
        help="Completed standard-pipeline DirectSpline result directory to audit.",
    )
    mode.add_argument(
        "--input-summary",
        type=Path,
        action="append",
        help="Existing support-audit summary. Repeat this option to create a combined summary.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D", help="Frozen source configuration label (default: D).")
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        help="Audit only these completed task IDs. Repeat for a small pilot.",
    )
    parser.add_argument(
        "--openml-cache-dir",
        type=Path,
        default=None,
        help="Optional OpenML cache directory. It is used only for source-task replay.",
    )
    parser.add_argument("--sparse-min-count", type=int, default=5)
    parser.add_argument("--sparse-min-fraction", type=float, default=0.001)
    parser.add_argument("--boundary-inner", type=float, default=3.5)
    parser.add_argument("--boundary-outer", type=float, default=4.5)
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow rewriting an output directory only when its immutable audit manifest matches.",
    )
    args = parser.parse_args()
    if args.input_summary is not None:
        if len(args.input_summary) < 2:
            raise ValueError("combined mode needs at least two --input-summary paths")
        prohibited = ("task_id", "openml_cache_dir")
        if any(getattr(args, name) is not None for name in prohibited):
            raise ValueError("--task-id and --openml-cache-dir apply only with --source-dir")
        return args
    if args.sparse_min_count < 1:
        raise ValueError("--sparse-min-count must be positive")
    if not 0.0 < args.sparse_min_fraction <= 1.0:
        raise ValueError("--sparse-min-fraction must lie in (0, 1]")
    if not 0.0 <= args.boundary_inner <= args.boundary_outer:
        raise ValueError("require 0 <= --boundary-inner <= --boundary-outer")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    return args


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path} ({type(error).__name__}: {error})") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _as_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error
    return result


def _source_normal_config(immutable_run: Mapping[str, Any]) -> dict[str, Any]:
    standard = immutable_run.get("standard_tabarena_baseline")
    if not isinstance(standard, Mapping):
        standard_pipeline = immutable_run.get("standard_pipeline")
        standard = standard_pipeline.get("normal_tabarena_config") if isinstance(standard_pipeline, Mapping) else None
    if not isinstance(standard, Mapping):
        raise ValueError("source manifest has no standard TabICL preprocessing configuration")
    required = ("n_estimators", "norm_methods", "feat_shuffle_method", "outlier_threshold", "random_state")
    missing = [name for name in required if name not in standard]
    if missing:
        raise ValueError(f"source standard TabICL configuration lacks {missing}")
    if str(standard.get("numerical_preprocessing", "existing")) != "existing":
        raise ValueError("support audit only reproduces the existing TabICL numerical preprocessing path")
    return dict(standard)


def _expected_config_from_manifest(
    immutable_run: Mapping[str, Any], *, config_label: str
) -> dict[str, Any]:
    labels = immutable_run.get("config_labels")
    configs = immutable_run.get("configs")
    if not isinstance(labels, list) or not isinstance(configs, list) or len(labels) != len(configs):
        raise ValueError("source manifest has inconsistent config_labels/configs")
    try:
        index = [str(label) for label in labels].index(config_label)
    except ValueError as error:
        raise ValueError(f"source manifest has no config label {config_label!r}") from error
    config = configs[index]
    if not isinstance(config, dict):
        raise ValueError(f"source manifest config {config_label!r} is not an object")
    return dict(config)


def _find_source_cases(
    *,
    source_dir: Path,
    manifest: Mapping[str, Any],
    config_label: str,
    requested_task_ids: set[int] | None,
) -> list[SourceCase]:
    immutable = manifest.get("immutable_run")
    if not isinstance(immutable, Mapping):
        raise ValueError("source manifest has no immutable_run")
    expected_config = _expected_config_from_manifest(immutable, config_label=config_label)
    expected_fingerprint = manifest.get("run_fingerprint_sha256")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("source manifest has no run_fingerprint_sha256")

    cases: list[SourceCase] = []
    for summary_path in sorted(source_dir.glob(f"raw/task_*/config_{config_label}/config_summary.json")):
        summary = _load_json(summary_path, label="source config summary")
        task_id = _as_int(summary.get("task_id"), name=f"task_id in {summary_path}")
        if requested_task_ids is not None and task_id not in requested_task_ids:
            continue
        if summary.get("pipeline") != "standard_ensemble":
            raise ValueError(f"source summary is not a standard-ensemble run: {summary_path}")
        if str(summary.get("config_label")) != config_label:
            raise ValueError(f"source summary label differs from requested {config_label!r}: {summary_path}")
        config = summary.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"source summary has no config object: {summary_path}")
        if _canonical_json(config) != _canonical_json(expected_config):
            raise ValueError(f"source config differs from frozen manifest for {summary_path}")
        if summary.get("run_fingerprint_hash") != expected_fingerprint:
            raise ValueError(f"source config fingerprint differs from manifest: {summary_path}")
        predictions_path = summary_path.parent / "config_predictions.npz"
        if not predictions_path.is_file():
            raise FileNotFoundError(f"missing frozen source prediction artifact: {predictions_path}")
        requested_bags = _as_int(summary.get("requested_bags"), name=f"requested_bags in {summary_path}")
        effective_bags = _as_int(summary.get("effective_bags"), name=f"effective_bags in {summary_path}")
        if requested_bags < 2 or effective_bags < 2:
            raise ValueError(f"invalid bag count in {summary_path}")
        bag_paths = sorted(summary_path.parent.glob("bag_*.npz"))
        expected_names = {f"bag_{index}.npz" for index in range(effective_bags)}
        observed_names = {path.name for path in bag_paths}
        if observed_names != expected_names:
            raise FileNotFoundError(
                f"source needs exactly {sorted(expected_names)} for support audit, found {sorted(observed_names)} in {summary_path.parent}"
            )
        cases.append(
            SourceCase(
                source_dir=source_dir,
                config_dir=summary_path.parent,
                config_summary_path=summary_path,
                config_predictions_path=predictions_path,
                task_id=task_id,
                dataset_id=_as_int(summary.get("dataset_id"), name=f"dataset_id in {summary_path}"),
                dataset_name=str(summary.get("dataset_name")),
                problem_type=str(summary.get("problem_type")),
                n_classes=None if summary.get("n_classes") is None else _as_int(summary.get("n_classes"), name="n_classes"),
                outer_split_hash=str(summary.get("outer_split_hash")),
                config_label=config_label,
                config=config,
                requested_bags=requested_bags,
                effective_bags=effective_bags,
            )
        )
    if not cases:
        requested = " requested task IDs" if requested_task_ids is not None else " completed configs"
        raise FileNotFoundError(f"source directory has no{requested} for config {config_label!r}: {source_dir}")

    observed_ids = {case.task_id for case in cases}
    if requested_task_ids is not None:
        missing = sorted(requested_task_ids - observed_ids)
        if missing:
            raise FileNotFoundError(f"requested task IDs have no completed {config_label!r} source config: {missing}")
    return cases


def _normalised_fit_labels(problem_type: str, labels: np.ndarray) -> np.ndarray:
    """Return the labels needed solely to recreate EnsembleGenerator branches."""

    values = np.asarray(labels)
    if problem_type == "regression":
        # The ensemble generator uses the regression labels only when it builds
        # view metadata; their numerical scale does not affect preprocessing.
        # StandardScaler is nevertheless reproduced to match the public fit.
        mean = float(np.mean(values))
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        return ((values - mean) / scale).astype(np.float64)
    unique = np.unique(values)
    expected = np.arange(unique.size)
    if not np.array_equal(unique, expected):
        # This is the same LabelEncoder convention used by the public
        # classifier, expressed without loading a model checkpoint.
        return np.searchsorted(unique, values).astype(int)
    return values.astype(int)


def _feature_names_after_filter(
    *,
    fit_x: Any,
    encoder: TransformToNumerical,
    generator: EnsembleGenerator,
    numerical_indices: np.ndarray,
) -> list[str]:
    if not hasattr(fit_x, "columns"):
        return [f"feature_{index}" for index in numerical_indices]
    input_names = [str(name) for name in fit_x.columns]
    encoded_numeric_names = {
        int(output): input_names[int(input_index)]
        for input_index, output in zip(
            np.asarray(encoder.numeric_input_positions_, dtype=int),
            np.asarray(encoder.numeric_output_positions_, dtype=int),
            strict=True,
        )
    }
    kept_encoded_positions = np.flatnonzero(generator.unique_filter_.features_to_keep_)
    return [
        encoded_numeric_names.get(int(kept_encoded_positions[index]), f"feature_{index}")
        for index in numerical_indices
    ]


def _fit_preprocessing_views(
    *,
    fit_x: Any,
    validation_x: Any,
    test_x: Any,
    fit_labels: np.ndarray,
    problem_type: str,
    normal_config: Mapping[str, Any],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    """Replay the exact public numerical preprocessing without model inference."""

    encoder = TransformToNumerical(verbose=False)
    encoded_fit = np.asarray(encoder.fit_transform(fit_x))
    encoded_validation = np.asarray(encoder.transform(validation_x))
    encoded_test = np.asarray(encoder.transform(test_x))
    generator = EnsembleGenerator(
        classification=problem_type != "regression",
        n_estimators=_as_int(normal_config["n_estimators"], name="normal n_estimators"),
        norm_methods=list(normal_config["norm_methods"]),
        feat_shuffle_method=str(normal_config["feat_shuffle_method"]),
        class_shuffle_method=str(normal_config.get("class_shuffle_method", "shift")),
        outlier_threshold=float(normal_config["outlier_threshold"]),
        random_state=_as_int(normal_config["random_state"], name="normal random_state"),
    ).fit(encoded_fit, _normalised_fit_labels(problem_type, fit_labels))
    filtered_validation = generator.unique_filter_.transform(encoded_validation)
    filtered_test = generator.unique_filter_.transform(encoded_test)
    kept_encoded_positions = np.flatnonzero(generator.unique_filter_.features_to_keep_)
    raw_numerical_positions = np.asarray(encoder.numeric_output_positions_, dtype=int)
    numerical_indices = np.flatnonzero(np.isin(kept_encoded_positions, raw_numerical_positions)).astype(int)
    feature_names = _feature_names_after_filter(
        fit_x=fit_x,
        encoder=encoder,
        generator=generator,
        numerical_indices=numerical_indices,
    )
    views: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method, preprocessor in generator.preprocessors_.items():
        views[str(method)] = (
            np.asarray(preprocessor.X_transformed_, dtype=np.float64)[:, numerical_indices],
            np.asarray(preprocessor.transform(filtered_validation), dtype=np.float64)[:, numerical_indices],
            np.asarray(preprocessor.transform(filtered_test), dtype=np.float64)[:, numerical_indices],
        )
    if not views:
        raise RuntimeError("public TabICL ensemble preprocessing produced no normalization views")
    return views, feature_names


def _fraction(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    return 0.0 if values.size == 0 else float(np.mean(values))


def _support_split_metrics(
    *,
    context: np.ndarray,
    query: np.ndarray,
    interval_edges: np.ndarray,
    standardized_range: float,
    boundary_inner: float,
    boundary_outer: float,
    sparse_min_count: int,
    sparse_min_fraction: float,
) -> dict[str, float | int]:
    """Measure one feature's query support against its fitting context."""

    context_values = np.asarray(context, dtype=float).reshape(-1)
    query_values = np.asarray(query, dtype=float).reshape(-1)
    finite_context = context_values[np.isfinite(context_values)]
    finite_query = query_values[np.isfinite(query_values)]
    if finite_context.size == 0:
        raise ValueError("preprocessing produced a numerical feature with no finite fitting-context values")
    if finite_query.size != query_values.size:
        raise ValueError("preprocessing produced non-finite validation/test numerical values")

    n_intervals = interval_edges.size - 1

    def interval_index(values: np.ndarray) -> np.ndarray:
        # The DirectSpline evaluates u=clip(z / range, -1, 1).  Values
        # outside the nominal range therefore land at the outer polynomial
        # piece for occupancy purposes; their separate tail/outside flags
        # retain the fact that their original coordinate was unsupported.
        clipped = np.clip(values, -standardized_range, standardized_range)
        return np.clip(np.searchsorted(interval_edges, clipped, side="right") - 1, 0, n_intervals - 1)

    context_counts = np.bincount(interval_index(finite_context), minlength=n_intervals)
    sparse_threshold = max(sparse_min_count, int(np.ceil(sparse_min_fraction * finite_context.size)))
    query_counts = context_counts[interval_index(finite_query)]
    outside = (finite_query < float(np.min(finite_context))) | (finite_query > float(np.max(finite_context)))
    sparse = query_counts < sparse_threshold
    tail = np.abs(finite_query) > standardized_range
    boundary = (np.abs(finite_query) >= boundary_inner) & (np.abs(finite_query) <= boundary_outer)
    return {
        "finite_rows": int(finite_query.size),
        "tail_rate": _fraction(tail),
        "boundary_rate": _fraction(boundary),
        "outside_context_range_rate": _fraction(outside),
        "sparse_interval_rate": _fraction(sparse),
        "unsupported_rate": _fraction(outside | sparse),
        "sparse_interval_count_threshold": int(sparse_threshold),
        "empty_context_interval_fraction": _fraction(context_counts == 0),
        "minimum_context_interval_count": int(np.min(context_counts)),
        "maximum_context_interval_count": int(np.max(context_counts)),
    }


def _feature_support_records(
    *,
    case: SourceCase,
    bag: int,
    views: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_names: list[str],
    degree: int,
    n_control_points: int,
    standardized_range: float,
    boundary_inner: float,
    boundary_outer: float,
    sparse_min_count: int,
    sparse_min_fraction: float,
) -> list[dict[str, Any]]:
    n_intervals = n_control_points - degree
    if n_intervals < 1:
        raise ValueError("DirectSpline configuration has no non-degenerate knot intervals")
    interval_edges = np.linspace(-standardized_range, standardized_range, n_intervals + 1)
    records: list[dict[str, Any]] = []
    for method, (context, validation, test) in views.items():
        if context.shape[1] != len(feature_names):
            raise RuntimeError("numerical preprocessing feature names do not match the replayed views")
        for feature_index, name in enumerate(feature_names):
            context_metrics = _support_split_metrics(
                context=context[:, feature_index],
                query=context[:, feature_index],
                interval_edges=interval_edges,
                standardized_range=standardized_range,
                boundary_inner=boundary_inner,
                boundary_outer=boundary_outer,
                sparse_min_count=sparse_min_count,
                sparse_min_fraction=sparse_min_fraction,
            )
            validation_metrics = _support_split_metrics(
                context=context[:, feature_index],
                query=validation[:, feature_index],
                interval_edges=interval_edges,
                standardized_range=standardized_range,
                boundary_inner=boundary_inner,
                boundary_outer=boundary_outer,
                sparse_min_count=sparse_min_count,
                sparse_min_fraction=sparse_min_fraction,
            )
            test_metrics = _support_split_metrics(
                context=context[:, feature_index],
                query=test[:, feature_index],
                interval_edges=interval_edges,
                standardized_range=standardized_range,
                boundary_inner=boundary_inner,
                boundary_outer=boundary_outer,
                sparse_min_count=sparse_min_count,
                sparse_min_fraction=sparse_min_fraction,
            )
            record: dict[str, Any] = {
                "task_id": case.task_id,
                "dataset_id": case.dataset_id,
                "dataset_name": case.dataset_name,
                "problem_type": case.problem_type,
                "config_label": case.config_label,
                "bag": bag,
                "normalization_method": method,
                "numerical_feature_index": feature_index,
                "numerical_feature_name": name,
                "spline_degree": degree,
                "spline_control_points": n_control_points,
                "spline_knot_intervals": n_intervals,
                "nominal_standardized_range": standardized_range,
            }
            for split, metrics in (("context", context_metrics), ("validation", validation_metrics), ("test", test_metrics)):
                for key, value in metrics.items():
                    record[f"{split}_{key}"] = value
            for key in (
                "tail_rate",
                "boundary_rate",
                "outside_context_range_rate",
                "sparse_interval_rate",
                "unsupported_rate",
            ):
                record[f"test_minus_validation_{key}"] = float(test_metrics[key]) - float(validation_metrics[key])
            records.append(record)
    return records


def _safe_error(
    *, problem_type: str, labels: np.ndarray, prediction: np.ndarray, n_classes: int | None
) -> tuple[float, bool]:
    try:
        value = float(deployment_error(problem_type, labels, prediction, n_classes=n_classes))
    except (TypeError, ValueError, FloatingPointError):
        return float("inf"), False
    return (value, bool(np.isfinite(value)))


def _relative_improvement(identity_error: float, candidate_error: float) -> float | None:
    if not (np.isfinite(identity_error) and np.isfinite(candidate_error)) or identity_error <= 0.0:
        return None
    return float((identity_error - candidate_error) / identity_error)


def _outcome(candidate_error: float, identity_error: float) -> str:
    if not (np.isfinite(candidate_error) and np.isfinite(identity_error)):
        return "invalid"
    if candidate_error < identity_error - _TIE_ATOL:
        return "win"
    if candidate_error > identity_error + _TIE_ATOL:
        return "loss"
    return "tie"


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return None if not values else float(np.mean(values))


def _bag_outcome_record(
    *,
    case: SourceCase,
    bag_index: int,
    bag_result: Any,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    feature_records: list[dict[str, Any]],
) -> dict[str, Any]:
    identity_validation_error, identity_validation_valid = _safe_error(
        problem_type=case.problem_type,
        labels=validation_labels,
        prediction=bag_result.identity_validation,
        n_classes=case.n_classes,
    )
    adapted_validation_error, adapted_validation_valid = _safe_error(
        problem_type=case.problem_type,
        labels=validation_labels,
        prediction=bag_result.adapted_validation,
        n_classes=case.n_classes,
    )
    identity_test_error, identity_test_valid = _safe_error(
        problem_type=case.problem_type,
        labels=test_labels,
        prediction=bag_result.identity_test,
        n_classes=case.n_classes,
    )
    adapted_test_error, adapted_test_valid = _safe_error(
        problem_type=case.problem_type,
        labels=test_labels,
        prediction=bag_result.adapted_test,
        n_classes=case.n_classes,
    )
    guarded_test_error, guarded_test_valid = _safe_error(
        problem_type=case.problem_type,
        labels=test_labels,
        prediction=bag_result.guarded_test,
        n_classes=case.n_classes,
    )
    selected_adapted = bool(bag_result.metadata.get("guard_selected_adapted"))
    support_fields = (
        "context_tail_rate",
        "validation_tail_rate",
        "test_tail_rate",
        "test_minus_validation_tail_rate",
        "context_boundary_rate",
        "validation_boundary_rate",
        "test_boundary_rate",
        "test_minus_validation_boundary_rate",
        "context_outside_context_range_rate",
        "validation_outside_context_range_rate",
        "test_outside_context_range_rate",
        "test_minus_validation_outside_context_range_rate",
        "context_sparse_interval_rate",
        "validation_sparse_interval_rate",
        "test_sparse_interval_rate",
        "test_minus_validation_sparse_interval_rate",
        "context_unsupported_rate",
        "validation_unsupported_rate",
        "test_unsupported_rate",
        "test_minus_validation_unsupported_rate",
    )
    record: dict[str, Any] = {
        "task_id": case.task_id,
        "dataset_id": case.dataset_id,
        "dataset_name": case.dataset_name,
        "problem_type": case.problem_type,
        "config_label": case.config_label,
        "bag": bag_index,
        "support_available": bool(feature_records),
        "n_feature_branch_records": len(feature_records),
        "guard_selected_adapted": selected_adapted,
        "identity_validation_error": identity_validation_error,
        "identity_validation_valid": identity_validation_valid,
        "adapted_validation_error": adapted_validation_error,
        "adapted_validation_valid": adapted_validation_valid,
        "adapted_validation_relative_improvement": _relative_improvement(
            identity_validation_error, adapted_validation_error
        ),
        "identity_test_error": identity_test_error,
        "identity_test_valid": identity_test_valid,
        "adapted_test_error": adapted_test_error,
        "adapted_test_valid": adapted_test_valid,
        "adapted_test_regret": (
            None
            if not (identity_test_valid and adapted_test_valid)
            else float(adapted_test_error - identity_test_error)
        ),
        "adapted_test_relative_improvement": _relative_improvement(identity_test_error, adapted_test_error),
        "adapted_test_outcome": _outcome(adapted_test_error, identity_test_error),
        "guarded_test_error": guarded_test_error,
        "guarded_test_valid": guarded_test_valid,
        "guard_false_positive": bool(
            selected_adapted
            and identity_test_valid
            and adapted_test_valid
            and adapted_test_error > identity_test_error + _TIE_ATOL
        ),
    }
    for field in support_fields:
        record[field] = _mean(feature_records, field)
    return record


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    required = (
        "identity_validation",
        "adapted_validation",
        "guarded_validation",
        "identity_test",
        "adapted_test",
        "guarded_test",
    )
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in required if key not in payload]
            if missing:
                raise ValueError(f"source prediction artifact lacks {missing}: {path}")
            return {key: np.asarray(payload[key]) for key in required}
    except OSError as error:
        raise ValueError(f"cannot load source prediction artifact: {path}") from error


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    difference = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    return float(np.max(difference)) if difference.size else 0.0


def _validate_aggregated_predictions(
    *,
    case: SourceCase,
    bag_results: list[Any],
    predictions: Mapping[str, np.ndarray],
) -> None:
    for key in ("identity_test", "adapted_test", "guarded_test"):
        replayed = np.mean([np.asarray(result.__dict__[key]) for result in bag_results], axis=0)
        maximum = _max_abs_difference(replayed, np.asarray(predictions[key]))
        if not np.isfinite(maximum) or maximum > _PREDICTION_ATOL:
            raise RuntimeError(
                f"source {key} ensemble does not reproduce its bag artifacts for task {case.task_id}: max_abs={maximum}"
            )
    shape = predictions["identity_validation"].shape
    for key in ("identity_validation", "adapted_validation", "guarded_validation"):
        replayed = np.full(shape, np.nan, dtype=np.float64)
        for result in bag_results:
            values = np.asarray(result.__dict__[key])
            replayed[result.validation_indices] = values
        maximum = _max_abs_difference(replayed, np.asarray(predictions[key]))
        if not np.isfinite(maximum) or maximum > _PREDICTION_ATOL:
            raise RuntimeError(
                f"source {key} OOF predictions do not reproduce its bag artifacts for task {case.task_id}: max_abs={maximum}"
            )


def _assert_summary_validation_metrics(
    *, case: SourceCase, config_summary: Mapping[str, Any], task: Any, predictions: Mapping[str, np.ndarray]
) -> None:
    validation = config_summary.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"source summary has no validation metrics: {case.config_summary_path}")
    for arm in ("identity", "adapted", "guarded"):
        reported = validation.get(arm)
        if not isinstance(reported, Mapping):
            raise ValueError(f"source summary validation has no {arm} arm: {case.config_summary_path}")
        replayed, valid = _safe_error(
            problem_type=case.problem_type,
            labels=task.y_train,
            prediction=np.asarray(predictions[f"{arm}_validation"]),
            n_classes=case.n_classes,
        )
        if not valid or not np.isclose(replayed, float(reported["deployment_error"]), rtol=1e-7, atol=1e-10):
            raise RuntimeError(
                f"source validation metric mismatch for task {case.task_id}, arm {arm}: "
                f"replayed={replayed}, reported={reported.get('deployment_error')}"
            )


def _task_record(
    *,
    case: SourceCase,
    task: Any,
    config_summary: Mapping[str, Any],
    feature_records: list[dict[str, Any]],
    bag_records: list[dict[str, Any]],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    errors: dict[str, tuple[float, bool]] = {}
    for split, labels in (("validation", task.y_train), ("test", task.y_test)):
        for arm in ("identity", "adapted", "guarded"):
            errors[f"{arm}_{split}"] = _safe_error(
                problem_type=case.problem_type,
                labels=labels,
                prediction=np.asarray(predictions[f"{arm}_{split}"]),
                n_classes=case.n_classes,
            )
    identity_test, identity_test_valid = errors["identity_test"]
    adapted_test, adapted_test_valid = errors["adapted_test"]
    guarded_test, guarded_test_valid = errors["guarded_test"]
    record: dict[str, Any] = {
        "task_id": case.task_id,
        "dataset_id": case.dataset_id,
        "dataset_name": case.dataset_name,
        "problem_type": case.problem_type,
        "n_classes": case.n_classes,
        "config_label": case.config_label,
        "outer_split_hash": case.outer_split_hash,
        "requested_bags": case.requested_bags,
        "effective_bags": case.effective_bags,
        "support_available": bool(feature_records),
        "n_feature_branch_records": len(feature_records),
        "n_bag_records": len(bag_records),
        "validation_identity_error": errors["identity_validation"][0],
        "validation_adapted_error": errors["adapted_validation"][0],
        "validation_guarded_error": errors["guarded_validation"][0],
        "validation_adapted_relative_improvement": _relative_improvement(
            errors["identity_validation"][0], errors["adapted_validation"][0]
        ),
        "test_identity_error": identity_test,
        "test_identity_valid": identity_test_valid,
        "test_adapted_error": adapted_test,
        "test_adapted_valid": adapted_test_valid,
        "test_adapted_regret": (
            None if not (identity_test_valid and adapted_test_valid) else float(adapted_test - identity_test)
        ),
        "test_adapted_relative_improvement": _relative_improvement(identity_test, adapted_test),
        "test_adapted_outcome": _outcome(adapted_test, identity_test),
        "test_guarded_error": guarded_test,
        "test_guarded_valid": guarded_test_valid,
        "test_guarded_regret": (
            None if not (identity_test_valid and guarded_test_valid) else float(guarded_test - identity_test)
        ),
        "test_guarded_relative_improvement": _relative_improvement(identity_test, guarded_test),
        "test_guarded_outcome": _outcome(guarded_test, identity_test),
        "guard_selected_adapted_bags": int(sum(bool(row["guard_selected_adapted"]) for row in bag_records)),
        "guard_false_positive_bags": int(sum(bool(row["guard_false_positive"]) for row in bag_records)),
        "guard_false_positive_bag_fraction": float(np.mean([bool(row["guard_false_positive"]) for row in bag_records])),
        "source_config_summary_sha256": _sha256(case.config_summary_path),
        "source_config_predictions_sha256": _sha256(case.config_predictions_path),
    }
    support_fields = (
        "context_tail_rate",
        "validation_tail_rate",
        "test_tail_rate",
        "test_minus_validation_tail_rate",
        "context_boundary_rate",
        "validation_boundary_rate",
        "test_boundary_rate",
        "test_minus_validation_boundary_rate",
        "context_outside_context_range_rate",
        "validation_outside_context_range_rate",
        "test_outside_context_range_rate",
        "test_minus_validation_outside_context_range_rate",
        "context_sparse_interval_rate",
        "validation_sparse_interval_rate",
        "test_sparse_interval_rate",
        "test_minus_validation_sparse_interval_rate",
        "context_unsupported_rate",
        "validation_unsupported_rate",
        "test_unsupported_rate",
        "test_minus_validation_unsupported_rate",
    )
    for field in support_fields:
        record[field] = _mean(feature_records, field)
    reported_fraction = config_summary.get("guard_selected_adapted_fraction")
    if reported_fraction is not None and not np.isclose(
        float(reported_fraction), record["guard_selected_adapted_bags"] / len(bag_records), rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError(f"source guard fraction disagrees with bag artifacts for task {case.task_id}")
    return record


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = left[valid], right[valid]
    result: dict[str, Any] = {"n": int(x.size), "pearson": None, "spearman": None}
    if x.size < 2:
        return result
    if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        result["pearson"] = float(np.corrcoef(x, y)[0, 1])
    ranked_x, ranked_y = _average_ranks(x), _average_ranks(y)
    if float(np.std(ranked_x)) > 0.0 and float(np.std(ranked_y)) > 0.0:
        result["spearman"] = float(np.corrcoef(ranked_x, ranked_y)[0, 1])
    return result


def _bootstrap_correlation(
    left: np.ndarray, right: np.ndarray, *, rounds: int, seed: int
) -> dict[str, Any]:
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = left[valid], right[valid]
    result: dict[str, Any] = {"rounds": rounds, "n_tasks": int(x.size), "pearson_95": None, "spearman_95": None}
    if x.size < 3:
        return result
    rng = np.random.default_rng(seed)
    pearson: list[float] = []
    spearman: list[float] = []
    for _ in range(rounds):
        sampled = rng.integers(0, x.size, size=x.size)
        value = _correlation(x[sampled], y[sampled])
        if value["pearson"] is not None:
            pearson.append(float(value["pearson"]))
        if value["spearman"] is not None:
            spearman.append(float(value["spearman"]))
    if pearson:
        result["pearson_95"] = {"lower_95": float(np.quantile(pearson, 0.025)), "upper_95": float(np.quantile(pearson, 0.975))}
    if spearman:
        result["spearman_95"] = {"lower_95": float(np.quantile(spearman, 0.025)), "upper_95": float(np.quantile(spearman, 0.975))}
    return result


def _outcome_counts(regrets: np.ndarray) -> dict[str, int]:
    return {
        "spline_wins": int(np.sum(regrets < -_TIE_ATOL)),
        "ties": int(np.sum(np.abs(regrets) <= _TIE_ATOL)),
        "spline_losses": int(np.sum(regrets > _TIE_ATOL)),
    }


def _rank_balanced_quartiles(rows: list[dict[str, Any]], *, predictor: str, outcome: str) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get(predictor) is not None and row.get(outcome) is not None]
    valid = [row for row in valid if np.isfinite(float(row[predictor])) and np.isfinite(float(row[outcome]))]
    ordered = sorted(valid, key=lambda row: (float(row[predictor]), int(row["task_id"])))
    groups = np.array_split(np.asarray(ordered, dtype=object), min(4, len(ordered))) if ordered else []
    output: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        members = list(group)
        predictors = np.asarray([float(row[predictor]) for row in members], dtype=float)
        regrets = np.asarray([float(row[outcome]) for row in members], dtype=float)
        output.append(
            {
                "rank_balanced_quartile": index,
                "n_tasks": len(members),
                "task_ids": [int(row["task_id"]) for row in members],
                "predictor_min": float(np.min(predictors)),
                "predictor_max": float(np.max(predictors)),
                "mean_test_regret": float(np.mean(regrets)),
                "median_test_regret": float(np.median(regrets)),
                "harm_fraction": float(np.mean(regrets > _TIE_ATOL)),
                **_outcome_counts(regrets),
            }
        )
    return output


def _group_summaries(rows: list[dict[str, Any]], *, predictor: str, outcome: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for problem_type in sorted({str(row["problem_type"]) for row in rows}):
        group = [row for row in rows if str(row["problem_type"]) == problem_type]
        x = np.asarray([float(row[predictor]) for row in group], dtype=float)
        y = np.asarray([float(row[outcome]) for row in group], dtype=float)
        result[problem_type] = {
            "n_tasks": len(group),
            "correlation": _correlation(x, y),
            "mean_test_regret": float(np.mean(y)),
            "harm_fraction": float(np.mean(y > _TIE_ATOL)),
            **_outcome_counts(y),
        }
    return result


def _leave_one_out(rows: list[dict[str, Any]], *, predictor: str, outcome: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, excluded in enumerate(rows):
        retained = rows[:index] + rows[index + 1 :]
        x = np.asarray([float(row[predictor]) for row in retained], dtype=float)
        y = np.asarray([float(row[outcome]) for row in retained], dtype=float)
        output.append(
            {
                "excluded_task_id": int(excluded["task_id"]),
                "excluded_dataset_name": str(excluded["dataset_name"]),
                "correlation": _correlation(x, y),
            }
        )
    return output


def _support_association_summary(
    task_records: list[dict[str, Any]], *, bootstrap_rounds: int, bootstrap_seed: int
) -> dict[str, Any]:
    predictor = "test_minus_validation_unsupported_rate"
    outcome = "test_adapted_regret"
    usable = [
        row
        for row in task_records
        if bool(row.get("support_available"))
        and row.get(predictor) is not None
        and row.get(outcome) is not None
        and np.isfinite(float(row[predictor]))
        and np.isfinite(float(row[outcome]))
    ]
    usable = sorted(usable, key=lambda row: int(row["task_id"]))
    x = np.asarray([float(row[predictor]) for row in usable], dtype=float)
    y = np.asarray([float(row[outcome]) for row in usable], dtype=float)
    quartiles = _rank_balanced_quartiles(usable, predictor=predictor, outcome=outcome)
    top_harm = None if not quartiles else quartiles[-1]["harm_fraction"]
    bottom_harm = None if not quartiles else quartiles[0]["harm_fraction"]
    top_to_bottom_ratio = None
    if top_harm is not None and bottom_harm is not None and float(bottom_harm) > 0.0:
        top_to_bottom_ratio = float(float(top_harm) / float(bottom_harm))
    correlation = _correlation(x, y)
    return {
        "analysis_unit": "task; support quantities are averaged across that task's fixed inner bags, numerical features, and actual TabICL normalization views",
        "primary_predictor": predictor,
        "primary_outcome": "test_adapted_regret = adapted deployment error - matched identity deployment error; positive means DirectSpline harmed the frozen test prediction",
        "n_tasks_with_nominal_support": int(sum(bool(row.get("support_available")) for row in task_records)),
        "n_tasks_primary_analysis": len(usable),
        "excluded_task_ids": [int(row["task_id"]) for row in task_records if row not in usable],
        "correlation": correlation,
        "task_bootstrap": _bootstrap_correlation(x, y, rounds=bootstrap_rounds, seed=bootstrap_seed),
        "rank_balanced_quartiles": quartiles,
        "top_vs_bottom_harm": {
            "top_quartile_harm_fraction": top_harm,
            "bottom_quartile_harm_fraction": bottom_harm,
            "ratio_when_bottom_nonzero": top_to_bottom_ratio,
        },
        "by_problem_type": _group_summaries(usable, predictor=predictor, outcome=outcome),
        "leave_one_task_out": _leave_one_out(usable, predictor=predictor, outcome=outcome),
        "predeclared_interpretation_checks": {
            "spearman_at_least_0_30": (
                None if correlation["spearman"] is None else bool(float(correlation["spearman"]) >= 0.30)
            ),
            "absolute_spearman_below_0_15": (
                None if correlation["spearman"] is None else bool(abs(float(correlation["spearman"])) < 0.15)
            ),
            "top_quartile_harm_at_least_twice_bottom_when_defined": (
                None if top_to_bottom_ratio is None else bool(top_to_bottom_ratio >= 2.0)
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_case(
    *,
    case: SourceCase,
    immutable_run: Mapping[str, Any],
    normal_config: Mapping[str, Any],
    sparse_min_count: int,
    sparse_min_fraction: float,
    boundary_inner: float,
    boundary_outer: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    outer_split = immutable_run.get("data_source", {}).get("outer_split")
    if not isinstance(outer_split, Mapping):
        raise ValueError("source manifest has no immutable OpenML outer split")
    task = load_tabarena_openml_task(
        case.task_id,
        outer_repeat=_as_int(outer_split.get("repeat"), name="outer repeat"),
        outer_fold=_as_int(outer_split.get("fold"), name="outer fold"),
        outer_sample=_as_int(outer_split.get("sample"), name="outer sample"),
    )
    for field, source_value, actual_value in (
        ("dataset_id", case.dataset_id, task.dataset_id),
        ("dataset_name", case.dataset_name, task.dataset_name),
        ("problem_type", case.problem_type, task.problem_type),
        ("outer_split_hash", case.outer_split_hash, task.outer_split_hash),
    ):
        if source_value != actual_value:
            raise ValueError(
                f"OpenML {field} changed for task {case.task_id}: source={source_value!r}, current={actual_value!r}"
            )
    config_summary = _load_json(case.config_summary_path, label="source config summary")
    predictions = _load_predictions(case.config_predictions_path)
    protocol_seed = _as_int(immutable_run.get("protocol_seed"), name="source protocol_seed")
    expected_splits = list(
        _bag_splits(task, requested_bags=case.requested_bags, seed=_seed(protocol_seed, task.task_id, 0))
    )
    if len(expected_splits) != case.effective_bags:
        raise RuntimeError(
            f"source effective bags differ from deterministic split replay for task {case.task_id}: "
            f"source={case.effective_bags}, replay={len(expected_splits)}"
        )
    degree = int(case.config.get("degree", 3))
    n_control_points = _as_int(case.config.get("n_control_points"), name="n_control_points")
    standardized_range = float(case.config.get("standardized_range", 4.0))
    if not standardized_range > 0.0:
        raise ValueError("DirectSpline standardized_range must be positive")
    feature_records: list[dict[str, Any]] = []
    bag_records: list[dict[str, Any]] = []
    bag_results: list[Any] = []
    pending_bag_records: list[tuple[int, Any, np.ndarray, list[dict[str, Any]]]] = []
    for bag_index, (fit_indices, validation_indices) in enumerate(expected_splits):
        bag_path = case.config_dir / f"bag_{bag_index}.npz"
        bag_result = _load_bag(bag_path)
        if int(bag_result.metadata.get("bag", -1)) != bag_index:
            raise RuntimeError(f"source bag metadata has wrong bag index: {bag_path}")
        if not np.array_equal(np.asarray(bag_result.validation_indices, dtype=int), np.asarray(validation_indices, dtype=int)):
            raise RuntimeError(f"source bag validation split differs from deterministic replay: {bag_path}")
        if bag_result.metadata.get("run_fingerprint_hash") != config_summary.get("run_fingerprint_hash"):
            raise RuntimeError(f"source bag fingerprint differs from its config summary: {bag_path}")
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
        bag_features = _feature_support_records(
            case=case,
            bag=bag_index,
            views=views,
            feature_names=feature_names,
            degree=degree,
            n_control_points=n_control_points,
            standardized_range=standardized_range,
            boundary_inner=boundary_inner,
            boundary_outer=boundary_outer,
            sparse_min_count=sparse_min_count,
            sparse_min_fraction=sparse_min_fraction,
        )
        feature_records.extend(bag_features)
        pending_bag_records.append((bag_index, bag_result, task.y_train[validation_indices], bag_features))
        bag_results.append(bag_result)
        print(
            f"task {case.task_id} bag {bag_index + 1}/{case.effective_bags}: "
            f"{len(bag_features)} feature-view records",
            flush=True,
        )
    # All nominal support records are now fixed.  Only this scoring block
    # reads outer-test targets, and it cannot affect any saved source artifact
    # or audit support quantity.
    for bag_index, bag_result, validation_labels, bag_features in pending_bag_records:
        bag_records.append(
            _bag_outcome_record(
                case=case,
                bag_index=bag_index,
                bag_result=bag_result,
                validation_labels=validation_labels,
                test_labels=task.y_test,
                feature_records=bag_features,
            )
        )
    _validate_aggregated_predictions(case=case, bag_results=bag_results, predictions=predictions)
    _assert_summary_validation_metrics(
        case=case, config_summary=config_summary, task=task, predictions=predictions
    )
    task_record = _task_record(
        case=case,
        task=task,
        config_summary=config_summary,
        feature_records=feature_records,
        bag_records=bag_records,
        predictions=predictions,
    )
    return feature_records, bag_records, task_record


def _audit_manifest(
    *,
    source_dir: Path,
    source_manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "DirectSpline nominal support-mismatch audit",
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
        "source_repository_revision": source_manifest.get("immutable_run", {}).get("repository_revision"),
        "config_label": args.config_label,
        "task_ids": None if args.task_id is None else sorted(set(args.task_id)),
        "support_definition": {
            "unsupported": "outside empirical bag-fit context range OR inside a knot interval with fewer than the sparse occupancy threshold fit-context values",
            "tail": "absolute final TabICL-preprocessed coordinate exceeds DirectSpline standardized_range",
            "boundary": "absolute final TabICL-preprocessed coordinate lies between boundary_inner and boundary_outer",
            "adapter_state_available": False,
        },
        "parameters": {
            "sparse_min_count": args.sparse_min_count,
            "sparse_min_fraction": args.sparse_min_fraction,
            "boundary_inner": args.boundary_inner,
            "boundary_outer": args.boundary_outer,
            "bootstrap_rounds": args.bootstrap_rounds,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }


def _prepare_output(*, output_dir: Path, audit_manifest: Mapping[str, Any], resume: bool) -> None:
    path = output_dir / "audit_manifest.json"
    if path.exists():
        existing = _load_json(path, label="existing audit manifest")
        if _canonical_json(existing) != _canonical_json(audit_manifest):
            raise ValueError("output directory belongs to a different immutable support audit; choose a new --output-dir")
        if not resume:
            raise ValueError("support-audit output directory already exists; pass --resume to rewrite matching results")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, audit_manifest)


def _run_audit(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    normal_config = _source_normal_config(immutable_run)
    requested_task_ids = None if args.task_id is None else set(args.task_id)
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=requested_task_ids,
    )
    audit_manifest = _audit_manifest(source_dir=source_dir, source_manifest=source_manifest, args=args)
    _prepare_output(output_dir=args.output_dir, audit_manifest=audit_manifest, resume=bool(args.resume))

    feature_records: list[dict[str, Any]] = []
    bag_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        print(
            f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}: replaying CPU preprocessing",
            flush=True,
        )
        task_features, task_bags, task_record = _audit_case(
            case=case,
            immutable_run=immutable_run,
            normal_config=normal_config,
            sparse_min_count=args.sparse_min_count,
            sparse_min_fraction=args.sparse_min_fraction,
            boundary_inner=args.boundary_inner,
            boundary_outer=args.boundary_outer,
        )
        feature_records.extend(task_features)
        bag_records.extend(task_bags)
        task_records.append(task_record)
    task_records.sort(key=lambda row: int(row["task_id"]))
    association = _support_association_summary(
        task_records, bootstrap_rounds=args.bootstrap_rounds, bootstrap_seed=args.bootstrap_seed
    )
    feature_path = args.output_dir / "feature_records.csv"
    bag_path = args.output_dir / "bag_records.csv"
    task_path = args.output_dir / "task_records.csv"
    task_json_path = args.output_dir / "task_records.json"
    _write_csv(feature_path, feature_records)
    _write_csv(bag_path, bag_records)
    _write_csv(task_path, task_records)
    _write_json(task_json_path, task_records)
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "source": {
            "source_dir": str(source_dir),
            "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
            "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
            "source_repository_revision": immutable_run.get("repository_revision"),
            "config_label": args.config_label,
            "normal_tabarena_preprocessing": normal_config,
        },
        "scope": {
            "n_completed_tasks": len(task_records),
            "n_bag_records": len(bag_records),
            "n_feature_branch_records": len(feature_records),
            "test_label_role": "scoring frozen source predictions only after support features are constructed",
            "adapter_state_limitation": (
                "source bag artifacts preserve predictions and metadata but not learned adapter state; "
                "this audit is nominal support only, not an actual learned-coordinate or derivative audit"
            ),
        },
        "support_definition": audit_manifest["support_definition"],
        "parameters": audit_manifest["parameters"],
        "primary_association": association,
        "files": {
            "feature_records_csv": str(feature_path),
            "bag_records_csv": str(bag_path),
            "task_records_csv": str(task_path),
            "task_records_json": str(task_json_path),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def _run_combined_summary(args: argparse.Namespace) -> dict[str, Any]:
    summaries = [_load_json(path.resolve(), label="input support-audit summary") for path in args.input_summary]
    task_records: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    for path, summary in zip(args.input_summary, summaries, strict=True):
        if summary.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported audit schema in {path}")
        records_path = summary.get("files", {}).get("task_records_json")
        if not isinstance(records_path, str):
            raise ValueError(f"support summary has no task_records_json: {path}")
        records = json.loads(Path(records_path).read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError(f"invalid task_records_json in {path}")
        task_records.extend(records)
        cohorts.append(
            {
                "summary_path": str(path.resolve()),
                "n_tasks": len(records),
                "source": summary.get("source"),
                "primary_association": summary.get("primary_association"),
            }
        )
    task_ids = [int(row["task_id"]) for row in task_records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("cannot combine cohorts with duplicate OpenML task IDs")
    association = _support_association_summary(
        task_records, bootstrap_rounds=args.bootstrap_rounds, bootstrap_seed=args.bootstrap_seed
    )
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "combined DirectSpline nominal support-mismatch audit",
        "cohorts": cohorts,
        "n_tasks": len(task_records),
        "primary_association": association,
        "interpretation": (
            "A combined association is descriptive. The predeclared next decision still requires checking whether "
            "support-harm evidence repeats in the separate multiclass and regression cohorts."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "combined_summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    summary = _run_combined_summary(args) if args.input_summary is not None else _run_audit(args)
    association = summary.get("primary_association", {})
    print(
        json.dumps(
            {
                "n_tasks": summary.get("scope", {}).get("n_completed_tasks", summary.get("n_tasks")),
                "spearman": association.get("correlation", {}).get("spearman"),
                "summary": str(args.output_dir / ("combined_summary.json" if args.input_summary is not None else "summary.json")),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
