"""Run a nested, stateful DirectSpline mechanism audit on completed OpenML tasks.

The experiment answers a narrower question than a new benchmark run:

    When a DirectSpline checkpoint and its identity guard are selected on one
    held-out block, does that frozen choice improve a *different* held-out
    block from the same published outer-training split?

Each rotation of the deterministic K-fold partition uses six folds for fitting,
one for checkpoint/guard selection, and one untouched fold for audit scoring.
The official OpenML outer test labels are never read by fitting, selection,
geometry measurement, or instability measurement.  Its *features* may be
used after selection as a common unlabeled probe for the optional prediction-
correction stability diagnostic.

This script is intentionally restricted to the fixed cubic K20 D arm and to
multiclass/regression tasks.  Their deployment metrics decompose into row
losses: log loss and MSE respectively.  Binary AUC needs a separate pairwise
version of this audit.

For every frozen selected spline checkpoint, the audit saves:

* the learned adapter and identity states;
* actual B-spline leverage in the selected adapter coordinate system;
* per-column unmixed deformation and local query-to-context geometry distortion;
* a common-grid learned-function correction, and an unlabeled outer-test
  prediction correction for cross-fit instability analysis.

The learned adapter is never updated using validation, audit, or outer-test
query labels.  Audit and outer-test labels are consumed only after all
state/geometry artefacts are fixed, to score already frozen predictions.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_nested_mechanism_audit.py \
      --source-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828 \
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_nested_mechanism_audit/multiclass_D \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from direct_spline_openml_support_audit import (
    SourceCase,
    _as_int,
    _canonical_json,
    _find_source_cases,
    _load_json,
    _sha256,
    _write_json,
    load_tabarena_openml_task,
)
from tabicl._experiments.direct_spline_openml import (
    OpenMLTaskData,
    _load_bag,
    _resolve_device,
    _save_bag,
    _seed,
    _split_hash,
    effective_inner_bag_count,
    load_frozen_backbone,
)
from tabicl._experiments.direct_spline_openml_standard import (
    _fit_one_bag_standard,
    _normal_prediction,
    _prepare_query,
)
from tabicl._experiments.direct_spline_protocol import deployment_error
from tabicl._hyperspline import DirectSplineTransform


AUDIT_SCHEMA_VERSION = 1
_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class _NestedRotation:
    """One fit/select/audit allocation of the deterministic inner folds."""

    rotation: int
    fit_indices: np.ndarray
    selection_indices: np.ndarray
    audit_indices: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Completed standard-pipeline D result used only for frozen task/config provenance.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D", help="Completed source configuration label (fixed cubic K20 only; default D).")
    parser.add_argument("--task-id", type=int, action="append", help="Audit only these completed task IDs. Repeatable.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--protocol-seed", type=int, default=None, help="Nested-fold seed. Omit to reuse the source run's protocol seed.")
    parser.add_argument("--bags", type=int, default=None, help="Nested partition count. Omit to reuse each source task's completed bag count.")
    parser.add_argument("--leverage-ridge", type=float, default=1e-6, help="Positive ridge added to B^T B for basis leverage (default 1e-6).")
    parser.add_argument("--grid-points", type=int, default=33, help="Common standardized-coordinate grid points for functional instability (default 33).")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse only complete matching nested-rotation artefacts.")
    args = parser.parse_args()
    if args.bags is not None and args.bags < 3:
        raise ValueError("--bags must be at least three: fit, selection, and audit blocks must be disjoint")
    if not np.isfinite(args.leverage_ridge) or args.leverage_ridge <= 0.0:
        raise ValueError("--leverage-ridge must be finite and positive")
    if args.grid_points < 5:
        raise ValueError("--grid-points must be at least five")
    return args


def _load_source_task(*, case: SourceCase, immutable_run: Mapping[str, Any]) -> OpenMLTaskData:
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
    return task


def _validate_case(case: SourceCase) -> None:
    if case.problem_type not in {"multiclass", "regression"}:
        raise ValueError(
            f"nested row-loss audit excludes {case.problem_type!r} task {case.task_id}; binary AUC is pairwise"
        )
    config = case.config
    if str(config.get("adapter_architecture", "fixed_cubic")) != "fixed_cubic":
        raise ValueError("nested mechanism audit currently supports only the fixed cubic D arm")
    if int(config.get("n_control_points", 20)) != 20:
        raise ValueError("nested mechanism audit requires the fixed K20 D arm")


def _nested_rotations(
    *, task: OpenMLTaskData, requested_bags: int, protocol_seed: int
) -> tuple[_NestedRotation, ...]:
    """Produce cyclic six-fit/one-select/one-audit rotations without leakage."""

    effective = effective_inner_bag_count(task, requested_bags=requested_bags)
    if effective < 3:
        raise ValueError(f"task {task.task_id} has only {effective} feasible inner folds; nested audit needs at least three")
    from tabicl._experiments.direct_spline_openml import _bag_splits

    split_seed = _seed(protocol_seed, task.task_id, 0)
    folds = [np.asarray(validation, dtype=int) for _fit, validation in _bag_splits(task, requested_bags=requested_bags, seed=split_seed)]
    if len(folds) != effective:
        raise RuntimeError("deterministic nested partition returned an unexpected fold count")
    all_indices = np.arange(task.y_train.size, dtype=int)
    rotations: list[_NestedRotation] = []
    for rotation, audit_indices in enumerate(folds):
        selection_indices = folds[(rotation + 1) % len(folds)]
        fit_mask = np.ones(all_indices.size, dtype=bool)
        fit_mask[audit_indices] = False
        fit_mask[selection_indices] = False
        fit_indices = all_indices[fit_mask]
        if fit_indices.size == 0:
            raise RuntimeError("nested fold rotation has no fitting rows")
        if np.intersect1d(fit_indices, selection_indices).size or np.intersect1d(fit_indices, audit_indices).size or np.intersect1d(selection_indices, audit_indices).size:
            raise RuntimeError("nested fold rotation is not disjoint")
        rotations.append(
            _NestedRotation(
                rotation=rotation,
                fit_indices=fit_indices,
                selection_indices=selection_indices,
                audit_indices=audit_indices,
            )
        )
    covered = np.concatenate([rotation.audit_indices for rotation in rotations])
    if not np.array_equal(np.sort(covered), all_indices):
        raise RuntimeError("nested audit folds do not cover every outer-training row exactly once")
    return tuple(rotations)


def _uniform_knots(*, n_control_points: int, degree: int) -> np.ndarray:
    internal = np.linspace(-1.0, 1.0, n_control_points - degree + 1, dtype=float)[1:-1]
    return np.concatenate((np.full(degree + 1, -1.0), internal, np.full(degree + 1, 1.0)))


def _bspline_design(
    *, values: np.ndarray, knots: np.ndarray, degree: int, n_control_points: int, standardized_range: float
) -> np.ndarray:
    """Cox--de Boor design matrix matching DirectSpline's clamped evaluation."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or standardized_range <= 0.0:
        raise ValueError("basis inputs must be finite/nonempty and the range positive")
    u = np.clip(values / standardized_range, -1.0, 1.0)
    spans = np.searchsorted(knots, u, side="right") - 1
    spans = np.clip(spans, degree, n_control_points - 1).astype(int, copy=False)
    local = np.ones((u.size, degree + 1), dtype=float)
    left = np.zeros_like(local)
    right = np.zeros_like(local)
    for order in range(1, degree + 1):
        left[:, order] = u - knots[spans + 1 - order]
        right[:, order] = knots[spans + order] - u
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
        raise RuntimeError("B-spline basis must form a partition of unity")
    return design


def _basis_leverage(
    *, context: np.ndarray, query: np.ndarray, knots: np.ndarray, degree: int, n_control_points: int, standardized_range: float, ridge: float
) -> np.ndarray:
    """Return b(q)^T(B^T B + ridge I)^-1 b(q), high for weakly covered directions."""

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
    gram = context_design.T @ context_design
    gram.flat[:: n_control_points + 1] += float(ridge)
    solved = np.linalg.solve(gram, query_design.T)
    values = np.sum(query_design.T * solved, axis=0)
    if not np.isfinite(values).all() or np.any(values < -1e-9):
        raise RuntimeError("basis leverage must be finite and non-negative")
    return np.maximum(values, 0.0)


def _nearest_context_indices(*, context: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Find each query value's nearest context value in one dimension."""

    context = np.asarray(context, dtype=float).reshape(-1)
    query = np.asarray(query, dtype=float).reshape(-1)
    if context.size == 0 or not np.isfinite(context).all() or not np.isfinite(query).all():
        raise ValueError("nearest-context inputs must be finite and context nonempty")
    order = np.argsort(context, kind="stable")
    sorted_context = context[order]
    right = np.searchsorted(sorted_context, query, side="left")
    left = np.clip(right - 1, 0, context.size - 1)
    right = np.clip(right, 0, context.size - 1)
    choose_right = np.abs(sorted_context[right] - query) < np.abs(sorted_context[left] - query)
    return order[np.where(choose_right, right, left)]


def _local_geometry_distortion(
    *, context: np.ndarray, query: np.ndarray, transformed_context: np.ndarray, transformed_query: np.ndarray
) -> np.ndarray:
    """Measure change to each query's difference from its nearest context value."""

    nearest = _nearest_context_indices(context=context, query=query)
    identity_difference = query - context[nearest]
    transformed_difference = transformed_query - transformed_context[nearest]
    return np.abs(transformed_difference - identity_difference)


def _per_row_loss(
    *, problem_type: str, labels: np.ndarray, prediction: np.ndarray, n_classes: int | None
) -> np.ndarray:
    labels = np.asarray(labels)
    prediction = np.asarray(prediction, dtype=float)
    if problem_type == "regression":
        return np.square(prediction.reshape(-1) - labels.astype(float).reshape(-1))
    if problem_type != "multiclass" or n_classes is None:
        raise ValueError("row-local loss is available only for multiclass and regression")
    classes = np.arange(n_classes)
    unique = np.unique(labels)
    encoded = labels.astype(int) if np.array_equal(unique, classes) else np.searchsorted(unique, labels)
    probabilities = prediction / prediction.sum(axis=1, keepdims=True)
    return -np.log(np.clip(probabilities[np.arange(labels.size), encoded], 1e-15, 1.0))


def _row_metrics_from_geometry(path: Path) -> dict[str, np.ndarray]:
    """Average branch-level values while retaining a cross-branch maximum risk."""

    with np.load(path, allow_pickle=False) as payload:
        required = (
            "mean_basis_leverage",
            "max_basis_leverage",
            "mean_unmixed_deformation",
            "max_unmixed_deformation",
            "mean_local_geometry_distortion",
            "max_local_geometry_distortion",
            "mean_leverage_geometry_interaction",
            "max_leverage_geometry_interaction",
            "outside_spline_domain_fraction",
            "mixing_spectral_norm",
        )
        values = {name: np.asarray(payload[name], dtype=float) for name in required}
    result: dict[str, np.ndarray] = {}
    for name, matrix in values.items():
        if matrix.ndim == 1:
            result[name] = matrix
        elif matrix.shape[0] == 0:
            # A categorical-only task has no numerical adapter.  It remains a
            # valid explicit identity tie; keep its audit rows rather than
            # silently dropping it, but mark numerical mechanisms unavailable.
            result[name] = np.full(matrix.shape[1], np.nan, dtype=float)
        elif name in {"max_basis_leverage", "max_unmixed_deformation", "max_local_geometry_distortion", "max_leverage_geometry_interaction", "mixing_spectral_norm"}:
            result[name] = matrix.max(axis=0) if matrix.shape[0] > 1 and matrix.shape[1] > 1 else matrix.reshape(-1)
        else:
            result[name] = matrix.mean(axis=0) if matrix.shape[0] > 1 else matrix.reshape(-1)
    lengths = {value.size for value in result.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"inconsistent geometry-row lengths in {path}")
    return result


def _make_capture_callback(
    *, rotation_dir: Path, probe_x: Any, leverage_ridge: float, grid_points: int
):
    """Create an unlabeled post-selection capture hook for one nested rotation."""

    state_path = rotation_dir / "selected_adapter_state.pt"
    geometry_path = rotation_dir / "unlabeled_geometry.npz"
    probe_path = rotation_dir / "unlabeled_probe_correction.npz"

    def capture(**values: Any) -> Mapping[str, Any]:
        bundle = values["bundle"]
        adapters = values["adapters"]
        device = values["device"]
        deployment_x = values["deployment_x"]
        rotation_dir.mkdir(parents=True, exist_ok=True)
        state_payload = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "identity_adapter_state_dict": values["identity_adapter_state"],
            "selected_adapted_adapter_state_dict": values["selected_adapted_adapter_state"],
            "guard_selected_adapted": bool(values["guard_selected_adapted"]),
        }
        torch.save(state_payload, state_path)
        if adapters is None:
            np.savez_compressed(
                geometry_path,
                methods=np.asarray([], dtype="U1"),
                raw_numerical_feature_positions=np.asarray([], dtype=np.int64),
                mean_basis_leverage=np.empty((0, len(deployment_x))),
                max_basis_leverage=np.empty((0, len(deployment_x))),
                mean_unmixed_deformation=np.empty((0, len(deployment_x))),
                max_unmixed_deformation=np.empty((0, len(deployment_x))),
                mean_local_geometry_distortion=np.empty((0, len(deployment_x))),
                max_local_geometry_distortion=np.empty((0, len(deployment_x))),
                mean_leverage_geometry_interaction=np.empty((0, len(deployment_x))),
                max_leverage_geometry_interaction=np.empty((0, len(deployment_x))),
                outside_spline_domain_fraction=np.empty((0, len(deployment_x))),
                mixing_spectral_norm=np.empty((0, len(deployment_x))),
                grid=np.linspace(-4.0, 4.0, grid_points),
                function_grid_correction=np.empty((0, grid_points, 0)),
            )
        else:
            prepared = _prepare_query(bundle, deployment_x)
            numerical_indices = np.asarray(bundle.numerical_indices, dtype=int)
            keep_mask = np.asarray(
                bundle.estimator.ensemble_generator_.unique_filter_.features_to_keep_, dtype=bool
            )
            kept_original_positions = np.flatnonzero(keep_mask)
            raw_numerical_feature_positions = kept_original_positions[numerical_indices]
            if raw_numerical_feature_positions.shape != numerical_indices.shape:
                raise RuntimeError("cannot map selected adapter features back to raw column positions")
            method_names: list[str] = []
            metric_values: dict[str, list[np.ndarray]] = {
                "mean_basis_leverage": [],
                "max_basis_leverage": [],
                "mean_unmixed_deformation": [],
                "max_unmixed_deformation": [],
                "mean_local_geometry_distortion": [],
                "max_local_geometry_distortion": [],
                "mean_leverage_geometry_interaction": [],
                "max_leverage_geometry_interaction": [],
                "outside_spline_domain_fraction": [],
                "mixing_spectral_norm": [],
            }
            grid = np.linspace(-4.0, 4.0, grid_points, dtype=np.float32)
            grid_corrections: list[np.ndarray] = []
            for method, preprocessor in bundle.estimator.ensemble_generator_.preprocessors_.items():
                adapter = adapters.for_method(method)
                if not isinstance(adapter, DirectSplineTransform):
                    raise ValueError("nested mechanism audit currently supports DirectSplineTransform D branches only")
                context = np.asarray(preprocessor.X_transformed_[bundle.support_indices], dtype=np.float32)[:, numerical_indices]
                query = np.asarray(preprocessor.transform(prepared.filtered), dtype=np.float32)[:, numerical_indices]
                context_tensor = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
                query_tensor = torch.as_tensor(query, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    location, scale, standardized_range = adapter._location_scale_range()  # type: ignore[attr-defined]
                    z_context = ((context_tensor - location.unsqueeze(1)) / scale.unsqueeze(1)).squeeze(0).cpu().numpy()
                    z_query = ((query_tensor - location.unsqueeze(1)) / scale.unsqueeze(1)).squeeze(0).cpu().numpy()
                    transformed_context = adapter.unmixed_transform(context_tensor).squeeze(0).cpu().numpy()
                    transformed_query = adapter.unmixed_transform(query_tensor).squeeze(0).cpu().numpy()
                    grid_tensor = torch.as_tensor(grid, dtype=torch.float32, device=device).view(1, grid_points, 1).expand(1, -1, query.shape[1])
                    transformed_grid = adapter.unmixed_transform(grid_tensor).squeeze(0).cpu().numpy()
                    _mixing_mean, _mixing_max, mixing_spectral = adapter.mixing_diagnostics()
                ranges = standardized_range.squeeze(0).detach().cpu().numpy()
                knots = adapter.knots_for_transform().detach().cpu().numpy()
                if knots.ndim != 1:
                    raise ValueError("nested mechanism audit requires the fixed uniform knot vector")
                degree = int(adapter.degree)
                n_control_points = int(adapter.gap_logits.shape[-1] + 1)
                leverages: list[np.ndarray] = []
                distortions: list[np.ndarray] = []
                outside: list[np.ndarray] = []
                for feature in range(query.shape[1]):
                    leverage = _basis_leverage(
                        context=z_context[:, feature],
                        query=z_query[:, feature],
                        knots=knots,
                        degree=degree,
                        n_control_points=n_control_points,
                        standardized_range=float(ranges[feature]),
                        ridge=leverage_ridge,
                    )
                    leverages.append(leverage)
                    distortions.append(
                        _local_geometry_distortion(
                            context=context[:, feature],
                            query=query[:, feature],
                            transformed_context=transformed_context[:, feature],
                            transformed_query=transformed_query[:, feature],
                        )
                    )
                    outside.append(np.abs(z_query[:, feature]) > float(ranges[feature]))
                leverage_matrix = np.column_stack(leverages)
                distortion_matrix = np.column_stack(distortions)
                deformation_matrix = np.abs(transformed_query - query)
                interaction_matrix = leverage_matrix * distortion_matrix
                metric_values["mean_basis_leverage"].append(leverage_matrix.mean(axis=1))
                metric_values["max_basis_leverage"].append(leverage_matrix.max(axis=1))
                metric_values["mean_unmixed_deformation"].append(deformation_matrix.mean(axis=1))
                metric_values["max_unmixed_deformation"].append(deformation_matrix.max(axis=1))
                metric_values["mean_local_geometry_distortion"].append(distortion_matrix.mean(axis=1))
                metric_values["max_local_geometry_distortion"].append(distortion_matrix.max(axis=1))
                metric_values["mean_leverage_geometry_interaction"].append(interaction_matrix.mean(axis=1))
                metric_values["max_leverage_geometry_interaction"].append(interaction_matrix.max(axis=1))
                metric_values["outside_spline_domain_fraction"].append(np.column_stack(outside).mean(axis=1))
                metric_values["mixing_spectral_norm"].append(
                    np.full(query.shape[0], float(mixing_spectral.detach().cpu()), dtype=float)
                )
                method_names.append(str(method))
                grid_corrections.append(transformed_grid - grid[:, None])
            np.savez_compressed(
                geometry_path,
                methods=np.asarray(method_names, dtype="U32"),
                raw_numerical_feature_positions=raw_numerical_feature_positions.astype(np.int64, copy=False),
                grid=grid,
                function_grid_correction=np.stack(grid_corrections, axis=0),
                **{name: np.stack(entries, axis=0) for name, entries in metric_values.items()},
            )
        # This inference is post-selection and uses only outer-test *features*.
        # It is a common probe for instability, never a training or selection input.
        identity_probe = _normal_prediction(
            bundle=bundle,
            query_x=probe_x,
            context_indices=bundle.support_indices,
            adapters=None,
            device=device,
        )
        adapted_probe = _normal_prediction(
            bundle=bundle,
            query_x=probe_x,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
        np.savez_compressed(probe_path, spline_minus_identity=np.asarray(adapted_probe) - np.asarray(identity_probe))
        return {
            "selected_adapter_state": state_path.name,
            "selected_adapter_state_sha256": _sha256(state_path),
            "unlabeled_geometry": geometry_path.name,
            "unlabeled_geometry_sha256": _sha256(geometry_path),
            "unlabeled_probe_correction": probe_path.name,
            "unlabeled_probe_correction_sha256": _sha256(probe_path),
            "outer_test_feature_role": "common unlabeled post-selection probe for correction-instability measurement only",
        }

    return capture


def _task_proxy_for_audit(*, task: OpenMLTaskData, audit_indices: np.ndarray) -> OpenMLTaskData:
    """Reuse the normal bag fitter with an inner audit fold as its query side."""

    audit_x = task.x_train.iloc[audit_indices].reset_index(drop=True)
    audit_y = np.asarray(task.y_train[audit_indices])
    return OpenMLTaskData(
        task_id=task.task_id,
        dataset_id=task.dataset_id,
        dataset_name=task.dataset_name,
        problem_type=task.problem_type,
        n_classes=task.n_classes,
        x_train=task.x_train,
        y_train=task.y_train,
        x_test=audit_x,
        y_test=audit_y,
        outer_split_hash=task.outer_split_hash,
    )


def _rotation_paths(task_dir: Path, rotation: int) -> dict[str, Path]:
    directory = task_dir / "rotations" / f"rotation_{rotation}"
    return {
        "directory": directory,
        "bag": directory / "bag.npz",
        "state": directory / "selected_adapter_state.pt",
        "geometry": directory / "unlabeled_geometry.npz",
        "probe": directory / "unlabeled_probe_correction.npz",
    }


def _rotation_complete(paths: Mapping[str, Path], fingerprint: str) -> bool:
    required = ("bag", "state", "geometry", "probe")
    if not all(paths[name].is_file() for name in required):
        return False
    result = _load_bag(paths["bag"])
    if result.metadata.get("run_fingerprint_hash") != fingerprint:
        return False
    # A numerical column can be removed by TabICL's ordinary unique-feature
    # filter in one nested fit, while remaining trainable in another.  The
    # saved raw-column map is therefore required to align function curves
    # honestly across rotations.  Treat captures from before that map was
    # added as incomplete: their feature ordinal alone is not a stable
    # identity after a preceding column has been removed.
    try:
        with np.load(paths["geometry"], allow_pickle=False) as payload:
            return "raw_numerical_feature_positions" in payload.files
    except (OSError, ValueError, KeyError):
        return False


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _instability_summary(rotation_paths: Sequence[Mapping[str, Path]]) -> dict[str, Any]:
    """Summarize selected corrections only where their raw column is shared.

    TabICL's standard unique-feature filter is fit independently inside each
    nested rotation.  A raw numeric column can consequently disappear in one
    rotation if it is constant there.  Comparing its replacement by ordinal
    position would compare different columns; omitting the unavailable
    rotation keeps the functional-instability measurement well-defined.
    """

    curves_by_method_feature: dict[tuple[str, int], list[np.ndarray]] = {}
    probes: list[np.ndarray] = []
    n_functional_candidates = 0
    for paths in rotation_paths:
        with np.load(paths["geometry"], allow_pickle=False) as payload:
            required = {"methods", "raw_numerical_feature_positions", "function_grid_correction"}
            missing = required.difference(payload.files)
            if missing:
                return {
                    "available": False,
                    "reason": f"rotation geometry lacks raw-column alignment metadata: {sorted(missing)}",
                    "n_rotations": len(rotation_paths),
                }
            methods = np.asarray(payload["methods"], dtype=str)
            raw_positions = np.asarray(payload["raw_numerical_feature_positions"], dtype=int)
            correction = np.asarray(payload["function_grid_correction"], dtype=float)
        if correction.ndim != 3:
            raise RuntimeError(f"function-grid correction must be 3D, got {correction.shape}")
        expected_shape = (len(methods), correction.shape[1], len(raw_positions))
        if correction.shape != expected_shape:
            raise RuntimeError(
                "invalid function-grid correction shape: "
                f"got {correction.shape}, expected (methods, grid, raw_numerical_features)={expected_shape}"
            )
        for method_index, method in enumerate(methods):
            for feature_index, raw_position in enumerate(raw_positions):
                curves_by_method_feature.setdefault((str(method), int(raw_position)), []).append(
                    correction[method_index, :, feature_index]
                )
                n_functional_candidates += 1
        with np.load(paths["probe"], allow_pickle=False) as payload:
            probes.append(np.asarray(payload["spline_minus_identity"], dtype=float))

    aligned_curve_variances: list[np.ndarray] = []
    n_present_all_rotations = 0
    n_single_rotation_only = 0
    for curves in curves_by_method_feature.values():
        if len(curves) == len(rotation_paths):
            n_present_all_rotations += 1
        if len(curves) < 2:
            n_single_rotation_only += 1
            continue
        aligned_curve_variances.append(np.var(np.stack(curves, axis=0), axis=0))
    functional_summary: dict[str, Any]
    if aligned_curve_variances:
        grid_variance = np.concatenate(aligned_curve_variances)
        functional_summary = {
            "available": True,
            "mean": float(grid_variance.mean()),
            "p95": float(np.quantile(grid_variance, 0.95)),
            "max": float(grid_variance.max()),
            "n_method_feature_pairs_with_at_least_two_rotations": len(aligned_curve_variances),
            "n_method_feature_pairs_present_in_all_rotations": n_present_all_rotations,
            "n_method_feature_pairs_present_in_one_rotation_only": n_single_rotation_only,
        }
    else:
        functional_summary = {
            "available": False,
            "reason": "no raw numerical method-feature pair is available in at least two rotations",
            "n_method_feature_pairs_present_in_one_rotation_only": n_single_rotation_only,
        }

    probe_shapes = {probe.shape for probe in probes}
    if len(probe_shapes) == 1:
        probe_variance = np.var(np.stack(probes, axis=0), axis=0)
        probe_summary: dict[str, Any] = {
            "available": True,
            "mean": float(probe_variance.mean()),
            "p95": float(np.quantile(probe_variance, 0.95)),
            "max": float(probe_variance.max()),
        }
    else:
        probe_summary = {
            "available": False,
            "reason": f"outer-test correction shapes differ across rotations: {sorted(probe_shapes)}",
        }
    return {
        "available": bool(functional_summary["available"] or probe_summary["available"]),
        "n_rotations": len(rotation_paths),
        "n_rotation_method_feature_observations": n_functional_candidates,
        "functional_correction_variance": functional_summary,
        "unlabeled_outer_test_prediction_correction_variance": probe_summary,
    }


def _task_records(
    *, case: SourceCase, task: OpenMLTaskData, rotations: Sequence[_NestedRotation], task_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Score frozen selections only after every geometry/state artefact exists."""

    shape = (task.y_train.size,) if task.problem_type == "regression" else (task.y_train.size, int(task.n_classes))
    selection_identity = np.full(shape, np.nan, dtype=float)
    selection_adapted = np.full(shape, np.nan, dtype=float)
    selection_guarded = np.full(shape, np.nan, dtype=float)
    audit_identity = np.full(shape, np.nan, dtype=float)
    audit_adapted = np.full(shape, np.nan, dtype=float)
    audit_guarded = np.full(shape, np.nan, dtype=float)
    row_records: list[dict[str, Any]] = []
    bag_records: list[dict[str, Any]] = []
    all_paths: list[dict[str, Path]] = []
    for rotation in rotations:
        paths = _rotation_paths(task_dir, rotation.rotation)
        result = _load_bag(paths["bag"])
        metadata = result.metadata
        if not np.array_equal(np.asarray(result.validation_indices, dtype=int), rotation.selection_indices):
            raise RuntimeError(f"rotation {rotation.rotation} selection indices differ from its saved artefact")
        if not np.array_equal(np.asarray(metadata.get("nested_audit_indices"), dtype=int), rotation.audit_indices):
            raise RuntimeError(f"rotation {rotation.rotation} audit indices differ from its saved artefact")
        selection_identity[rotation.selection_indices] = result.identity_validation
        selection_adapted[rotation.selection_indices] = result.adapted_validation
        selection_guarded[rotation.selection_indices] = result.guarded_validation
        audit_identity[rotation.audit_indices] = result.identity_test
        audit_adapted[rotation.audit_indices] = result.adapted_test
        audit_guarded[rotation.audit_indices] = result.guarded_test
        selection_labels = np.asarray(task.y_train[rotation.selection_indices])
        audit_labels = np.asarray(task.y_train[rotation.audit_indices])
        selection_identity_error = deployment_error(task.problem_type, selection_labels, result.identity_validation, n_classes=task.n_classes)
        selection_adapted_error = deployment_error(task.problem_type, selection_labels, result.adapted_validation, n_classes=task.n_classes)
        selection_guarded_error = deployment_error(task.problem_type, selection_labels, result.guarded_validation, n_classes=task.n_classes)
        audit_identity_error = deployment_error(task.problem_type, audit_labels, result.identity_test, n_classes=task.n_classes)
        audit_adapted_error = deployment_error(task.problem_type, audit_labels, result.adapted_test, n_classes=task.n_classes)
        audit_guarded_error = deployment_error(task.problem_type, audit_labels, result.guarded_test, n_classes=task.n_classes)
        bag_records.append(
            {
                "task_id": case.task_id,
                "dataset_id": case.dataset_id,
                "dataset_name": case.dataset_name,
                "problem_type": case.problem_type,
                "rotation": rotation.rotation,
                "fit_rows": int(rotation.fit_indices.size),
                "selection_rows": int(rotation.selection_indices.size),
                "audit_rows": int(rotation.audit_indices.size),
                "guard_selected_adapted": bool(metadata["guard_selected_adapted"]),
                "selection_identity_error": float(selection_identity_error),
                "selection_adapted_error": float(selection_adapted_error),
                "selection_guarded_error": float(selection_guarded_error),
                "audit_identity_error": float(audit_identity_error),
                "audit_adapted_error": float(audit_adapted_error),
                "audit_guarded_error": float(audit_guarded_error),
                "raw_selection_minus_identity": float(selection_adapted_error - selection_identity_error),
                "raw_audit_minus_identity": float(audit_adapted_error - audit_identity_error),
                "guarded_selection_minus_identity": float(selection_guarded_error - selection_identity_error),
                "guarded_audit_minus_identity": float(audit_guarded_error - audit_identity_error),
                "selected_adapter_step": metadata.get("adapter_best_step"),
                "train_seconds": metadata.get("train_seconds"),
            }
        )
        geometry = _row_metrics_from_geometry(paths["geometry"])
        identity_loss = _per_row_loss(problem_type=task.problem_type, labels=audit_labels, prediction=result.identity_test, n_classes=task.n_classes)
        adapted_loss = _per_row_loss(problem_type=task.problem_type, labels=audit_labels, prediction=result.adapted_test, n_classes=task.n_classes)
        guarded_loss = _per_row_loss(problem_type=task.problem_type, labels=audit_labels, prediction=result.guarded_test, n_classes=task.n_classes)
        for row_position, source_index in enumerate(rotation.audit_indices):
            record: dict[str, Any] = {
                "task_id": case.task_id,
                "dataset_id": case.dataset_id,
                "dataset_name": case.dataset_name,
                "problem_type": case.problem_type,
                "rotation": rotation.rotation,
                "audit_source_row_index": int(source_index),
                "guard_selected_adapted": bool(metadata["guard_selected_adapted"]),
                "identity_row_loss": float(identity_loss[row_position]),
                "adapted_row_loss": float(adapted_loss[row_position]),
                "guarded_row_loss": float(guarded_loss[row_position]),
                "raw_adapted_minus_identity_row_loss": float(adapted_loss[row_position] - identity_loss[row_position]),
                "guarded_minus_identity_row_loss": float(guarded_loss[row_position] - identity_loss[row_position]),
            }
            for name, metric in geometry.items():
                record[name] = float(metric[row_position])
            row_records.append(record)
        all_paths.append(paths)
    if not all(np.isfinite(array).all() for array in (selection_identity, selection_adapted, selection_guarded, audit_identity, audit_adapted, audit_guarded)):
        raise RuntimeError(f"nested task {task.task_id} has incomplete selection/audit predictions")
    summary = {
        "task_id": case.task_id,
        "dataset_id": case.dataset_id,
        "dataset_name": case.dataset_name,
        "problem_type": case.problem_type,
        "n_rotations": len(rotations),
        "guard_selected_adapted_fraction": float(np.mean([record["guard_selected_adapted"] for record in bag_records])),
        "selection": {
            "identity_error": float(deployment_error(task.problem_type, task.y_train, selection_identity, n_classes=task.n_classes)),
            "raw_adapted_error": float(deployment_error(task.problem_type, task.y_train, selection_adapted, n_classes=task.n_classes)),
            "guarded_error": float(deployment_error(task.problem_type, task.y_train, selection_guarded, n_classes=task.n_classes)),
        },
        "independent_audit": {
            "identity_error": float(deployment_error(task.problem_type, task.y_train, audit_identity, n_classes=task.n_classes)),
            "raw_adapted_error": float(deployment_error(task.problem_type, task.y_train, audit_adapted, n_classes=task.n_classes)),
            "guarded_error": float(deployment_error(task.problem_type, task.y_train, audit_guarded, n_classes=task.n_classes)),
        },
        "instability": _instability_summary(all_paths),
        "test_label_role": "none: official outer-test labels are not read by this nested mechanism audit",
    }
    summary["selection"]["raw_adapted_minus_identity"] = summary["selection"]["raw_adapted_error"] - summary["selection"]["identity_error"]
    summary["selection"]["guarded_minus_identity"] = summary["selection"]["guarded_error"] - summary["selection"]["identity_error"]
    summary["independent_audit"]["raw_adapted_minus_identity"] = summary["independent_audit"]["raw_adapted_error"] - summary["independent_audit"]["identity_error"]
    summary["independent_audit"]["guarded_minus_identity"] = summary["independent_audit"]["guarded_error"] - summary["independent_audit"]["identity_error"]
    return row_records, bag_records, summary


def _manifest(*, source_dir: Path, source_manifest: Mapping[str, Any], cases: Sequence[SourceCase], args: argparse.Namespace, protocol_seed: int) -> dict[str, Any]:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "nested DirectSpline selection-to-independent-audit mechanism study",
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
        "source_repository_revision": source_manifest.get("immutable_run", {}).get("repository_revision"),
        "config_label": args.config_label,
        "task_ids": [case.task_id for case in cases],
        "protocol_seed": int(protocol_seed),
        "requested_bags": args.bags,
        "fixed_arm_requirement": {"adapter_architecture": "fixed_cubic", "n_control_points": 20},
        "selection_protocol": "cyclic nested K-fold: six-or-more fit folds, one checkpoint/guard selection fold, one distinct untouched audit fold",
        "geometry": {
            "support": "actual selected-state B-spline leverage b(q)^T(B^T B + ridge I)^-1 b(q) in every branch/column",
            "leverage_ridge": float(args.leverage_ridge),
            "relative_geometry": "absolute change to query-minus-nearest-context distance under the selected unmixed spline",
            "deformation": "absolute selected unmixed transform minus input coordinate, diagnostic only",
            "functional_instability": "variance of selected learned function correction T_b(grid)-grid over nested rotations",
            "prediction_instability": "variance of raw spline-minus-identity predictions on a common unlabeled official outer-test feature probe",
            "grid_points": int(args.grid_points),
        },
        "label_policy": {
            "selection_labels": "select checkpoint and per-bag identity guard only",
            "audit_labels": "score frozen selection only after state and geometry are persisted",
            "official_outer_test_labels": "never read",
            "official_outer_test_features": "post-selection common instability probe only; never used by preprocessing fit, adapter training, checkpoint selection, guard, or any metric",
        },
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> str:
    path = output_dir / "experiment_manifest.json"
    fingerprint = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    payload = {**manifest, "run_fingerprint_sha256": fingerprint}
    if path.exists():
        existing = _load_json(path, label="existing nested mechanism manifest")
        if _canonical_json(existing) != _canonical_json(payload):
            raise ValueError("output directory belongs to a different nested mechanism audit; choose a new --output-dir")
        if not resume:
            raise ValueError("nested mechanism audit output directory already exists; pass --resume to reuse matching rotations")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return fingerprint


def _run_case(
    *, case: SourceCase, task: OpenMLTaskData, task_dir: Path, args: argparse.Namespace, protocol_seed: int, fingerprint: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _validate_case(case)
    requested_bags = case.requested_bags if args.bags is None else int(args.bags)
    rotations = _nested_rotations(task=task, requested_bags=requested_bags, protocol_seed=protocol_seed)
    task_dir.mkdir(parents=True, exist_ok=True)
    need_fit = [rotation for rotation in rotations if not _rotation_complete(_rotation_paths(task_dir, rotation.rotation), fingerprint)]
    if need_fit:
        device = _resolve_device(args.device)
        backbone, _checkpoint_path, checkpoint_metadata = load_frozen_backbone(
            problem_type=task.problem_type,
            device=device,
            classifier_checkpoint=args.classifier_checkpoint,
            regressor_checkpoint=args.regressor_checkpoint,
        )
        _write_json(task_dir / "task_provenance.json", {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": task.dataset_name,
            "problem_type": task.problem_type,
            "outer_split_hash": task.outer_split_hash,
            "config": case.config,
            "checkpoint": checkpoint_metadata,
            "rotations": [
                {
                    "rotation": rotation.rotation,
                    "fit_indices_sha256": _split_hash(rotation.fit_indices, np.empty(0, dtype=int)),
                    "selection_indices_sha256": _split_hash(rotation.selection_indices, np.empty(0, dtype=int)),
                    "audit_indices_sha256": _split_hash(rotation.audit_indices, np.empty(0, dtype=int)),
                }
                for rotation in rotations
            ],
        })
        for position, rotation in enumerate(rotations, start=1):
            paths = _rotation_paths(task_dir, rotation.rotation)
            if _rotation_complete(paths, fingerprint):
                print(f"task {task.task_id} rotation {position}/{len(rotations)}: reused", flush=True)
                continue
            print(
                f"task {task.task_id} rotation {position}/{len(rotations)}: "
                f"fit={rotation.fit_indices.size} select={rotation.selection_indices.size} audit={rotation.audit_indices.size}",
                flush=True,
            )
            result = _fit_one_bag_standard(
                task=_task_proxy_for_audit(task=task, audit_indices=rotation.audit_indices),
                fit_indices=rotation.fit_indices,
                validation_indices=rotation.selection_indices,
                bag=rotation.rotation,
                config=dict(case.config),
                protocol_seed=protocol_seed,
                backbone=backbone,
                device=device,
                run_fingerprint_hash=fingerprint,
                progress=None,
                requested_bags=requested_bags,
                effective_bags=len(rotations),
                diagnostic_callback=_make_capture_callback(
                    rotation_dir=paths["directory"],
                    probe_x=task.x_test,
                    leverage_ridge=float(args.leverage_ridge),
                    grid_points=int(args.grid_points),
                ),
            )
            result.metadata["nested_rotation"] = int(rotation.rotation)
            result.metadata["nested_fit_indices"] = rotation.fit_indices.tolist()
            result.metadata["nested_selection_indices"] = rotation.selection_indices.tolist()
            result.metadata["nested_audit_indices"] = rotation.audit_indices.tolist()
            result.metadata["nested_protocol"] = "fit/select/audit disjoint cyclic folds"
            _save_bag(paths["bag"], result)
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _task_records(case=case, task=task, rotations=rotations, task_dir=task_dir)


def main() -> None:
    args = _parse_args()
    source_dir = args.source_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    cases = _find_source_cases(
        source_dir=source_dir,
        manifest=source_manifest,
        config_label=args.config_label,
        requested_task_ids=None if args.task_id is None else set(args.task_id),
    )
    cases = [case for case in cases if case.problem_type in {"multiclass", "regression"}]
    if not cases:
        raise ValueError("source has no completed multiclass/regression tasks for this row-loss mechanism audit")
    source_protocol_seed = _as_int(immutable_run.get("protocol_seed"), name="source protocol_seed")
    protocol_seed = source_protocol_seed if args.protocol_seed is None else int(args.protocol_seed)
    manifest = _manifest(source_dir=source_dir, source_manifest=source_manifest, cases=cases, args=args, protocol_seed=protocol_seed)
    fingerprint = _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))
    all_rows: list[dict[str, Any]] = []
    all_bags: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}: nested mechanism audit", flush=True)
        _validate_case(case)
        task = _load_source_task(case=case, immutable_run=immutable_run)
        task_dir = args.output_dir / "raw" / f"task_{case.task_id}_{case.dataset_name.replace('/', '_')}"
        rows, bags, task_summary = _run_case(
            case=case,
            task=task,
            task_dir=task_dir,
            args=args,
            protocol_seed=protocol_seed,
            fingerprint=fingerprint,
        )
        all_rows.extend(rows)
        all_bags.extend(bags)
        task_summaries.append(task_summary)
        _write_json(task_dir / "task_summary.json", task_summary)
    all_rows.sort(key=lambda row: (int(row["task_id"]), int(row["rotation"]), int(row["audit_source_row_index"])))
    all_bags.sort(key=lambda row: (int(row["task_id"]), int(row["rotation"])))
    task_summaries.sort(key=lambda row: int(row["task_id"]))
    _write_csv(args.output_dir / "row_records.csv", all_rows)
    _write_csv(args.output_dir / "bag_records.csv", all_bags)
    _write_json(args.output_dir / "task_summaries.json", task_summaries)
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "n_tasks": len(task_summaries),
        "n_rows": len(all_rows),
        "n_rotations": len(all_bags),
        "tasks": task_summaries,
        "label_policy": manifest["label_policy"],
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(task_summaries), "n_rows": len(all_rows), "n_rotations": len(all_bags)}), flush=True)


if __name__ == "__main__":
    main()
