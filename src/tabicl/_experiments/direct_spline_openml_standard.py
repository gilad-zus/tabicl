"""Full-pipeline DirectSpline experiment for the public OpenML TabArena suite.

This is the corrected companion to :mod:`direct_spline_openml`.  The older
runner was deliberately a small raw-backbone headroom experiment: it sampled a
single context and bypassed TabICL's usual preprocessing ensemble.  That is a
useful diagnostic, but its identity path is not the normal TabICLv2 estimator.

This module keeps the DirectSpline adapter as the *only* learned addition while
using the normal estimator's per-bag preprocessing views, feature/class
shuffles, temperature, and logit averaging.  Adapted arrays are evaluated by
the public estimator's own batching/inference functions rather than a second
reimplementation.  Before training, every full-query public input view and
label is checked bit-for-bit against both the no-spline and fresh-spline view.

The outer OpenML test labels are never used for preprocessing fitting, adapter
optimisation, identity guarding, or configuration selection.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import csv
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.utils.validation import validate_data

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._experiments.direct_spline_openml import (
    STANDARD_TABICL_CONFIG,
    BagPredictions,
    OpenMLTaskData,
    _bag_splits,
    _config_dir,
    _candidate_metric_bundle,
    _cpu_state_dict,
    _emit,
    _json_dump,
    _json_load,
    _load_bag,
    _metric_bundle,
    _paired_comparison_summary,
    _prediction_shape,
    _relative_error_change,
    _safe_name,
    _save_bag,
    _seed,
    _standard_baseline_dir,
    _set_frozen_autograd_routing,
    effective_inner_bag_count,
    load_frozen_backbone,
)
from tabicl._experiments.direct_spline_protocol import (
    ProblemType,
    choose_identity_guard,
    deployment_error,
    sample_episode_indices,
    sample_prediction_context,
)
from tabicl._hyperspline import AdaptiveDirectSplineTransform, DirectSplineTransform
from tabicl._model.attention import flash_attn3_toggle
from tabicl._model.tabicl import TabICL


# ``None`` deliberately means that the normal estimator receives every fitting
# row of a bag.  A positive cap is available only for hardware-constrained
# diagnostics and is recorded as such in every artifact.
STANDARD_DIRECT_SPLINE_CONFIG: dict[str, Any] = {
    "adapter_steps": 150,
    "adapter_patience": 10,
    "validation_interval": 10,
    "max_context_rows": None,
    # Match deployment as closely as a labelled training episode permits:
    # every non-query fitting row is context.  The launcher still exposes an
    # explicit positive cap for hardware-constrained diagnostic runs.
    "train_context_rows": None,
    # Row interaction attends across features within each row, not across
    # dataset rows.  Keep its differentiable execution in conservative row
    # batches: the public inference manager does the equivalent batching under
    # no_grad, while adapter training must retain the input gradient.
    "row_interaction_chunk_rows": 2_048,
    "query_batch_rows": 256,
    "n_control_points": 20,
    "learning_rate": 0.005,
    "weight_decay": 0.003,
    "grad_clip": 2.0,
    "gate_learning_rate_factor": 3.0,
    "trainable_location_scale": True,
    "cross_column_mixing_rank": 4,
    "cross_column_mixing_bound": 0.10,
    "guard_relative_improvement": 0.005,
    "random_state": 0,
}


@dataclass
class _StandardBag:
    """One normal TabICLv2 estimator fitted on an adapter bag's fit rows."""

    estimator: TabICLClassifier | TabICLRegressor
    fit_labels: np.ndarray
    support_indices: np.ndarray
    numerical_indices: np.ndarray
    problem_type: ProblemType
    n_classes: int | None
    row_interaction_chunk_rows: int = 2_048

    @property
    def backbone(self) -> TabICL:
        return self.estimator.model_


class _AdapterSet(nn.Module):
    """One spline per normalisation branch, all initially exact identity.

    The two normal TabICL branches (``none`` and ``power`` by default) live in
    different coordinates.  They therefore require separate spline parameters,
    but members sharing a normalisation branch share the same adapter before
    their ordinary feature shuffles are applied.
    """

    def __init__(self, adapters: OrderedDict[str, nn.Module]) -> None:
        super().__init__()
        self._keys = {method: f"branch_{index}" for index, method in enumerate(adapters)}
        self.adapters = nn.ModuleDict({self._keys[method]: adapter for method, adapter in adapters.items()})

    def for_method(self, method: str) -> nn.Module:
        return self.adapters[self._keys[method]]


class ValidationSplitInfeasibleError(ValueError):
    """A task cannot retain a valid one-split classification validation arm."""


def standard_direct_spline_config(
    *,
    context_cap: int | None = None,
    train_context_rows: int | None = None,
    adapter_steps: int | None = None,
    adapter_patience: int | None = None,
    validation_interval: int | None = None,
) -> dict[str, Any]:
    """Return the default strong-pipeline DirectSpline configuration.

    ``context_cap=None`` (and ``0`` at the launcher) means all fit rows.  A
    cap is useful to diagnose memory or context effects, but it intentionally
    no longer claims exact parity with a full normal TabICL estimator.
    """

    config = dict(STANDARD_DIRECT_SPLINE_CONFIG)
    if context_cap is not None:
        if context_cap <= 0:
            raise ValueError("context_cap must be positive when provided")
        config["max_context_rows"] = int(context_cap)
        # Unless separately overridden, a hardware diagnostic should train at
        # the same context scale that it deploys.
        if train_context_rows is None:
            config["train_context_rows"] = int(context_cap)
    if train_context_rows is not None:
        if train_context_rows < 0:
            raise ValueError("train_context_rows must be zero (all available rows) or positive")
        config["train_context_rows"] = None if train_context_rows == 0 else int(train_context_rows)
    for name, value in (
        ("adapter_steps", adapter_steps),
        ("adapter_patience", adapter_patience),
        ("validation_interval", validation_interval),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} must be positive when provided")
            config[name] = int(value)
    return config


def shared_standard_direct_spline_configs(
    n_configs: int,
    *,
    seed: int,
    context_cap: int | None = None,
    adapter_steps: int | None = None,
    adapter_patience: int | None = None,
    validation_interval: int | None = None,
) -> list[dict[str, Any]]:
    """The existing ten random DirectSpline draws, but on the strong path."""

    # Import lazily to keep this runner independent of the old runner's
    # default context policy at module import time.
    from tabicl._experiments.direct_spline_protocol import shared_random_direct_spline_configs

    configs = shared_random_direct_spline_configs(n_configs, seed=seed)
    for config in configs:
        config["max_context_rows"] = context_cap
        config["train_context_rows"] = (
            STANDARD_DIRECT_SPLINE_CONFIG["train_context_rows"]
            if context_cap is None
            else int(context_cap)
        )
        for name, value in (
            ("adapter_steps", adapter_steps),
            ("adapter_patience", adapter_patience),
            ("validation_interval", validation_interval),
        ):
            if value is not None:
                if value <= 0:
                    raise ValueError(f"{name} must be positive when provided")
                config[name] = int(value)
    return configs


def _filtered_numerical_indices(estimator: TabICLClassifier | TabICLRegressor) -> np.ndarray:
    """Map raw numerical columns through the normal unique-feature filter."""

    keep_mask = np.asarray(estimator.ensemble_generator_.unique_filter_.features_to_keep_, dtype=bool)
    kept_original_positions = np.flatnonzero(keep_mask)
    raw_numerical_positions = np.asarray(estimator.X_encoder_.numeric_output_positions_, dtype=int)
    return np.flatnonzero(np.isin(kept_original_positions, raw_numerical_positions)).astype(int)


def _fit_standard_bag(
    *,
    task: OpenMLTaskData,
    fit_indices: np.ndarray,
    config: dict[str, Any],
    protocol_seed: int,
    bag: int,
    backbone: TabICL,
    device: torch.device,
) -> _StandardBag:
    """Fit the public sklearn estimator without loading a second checkpoint."""

    fit_x = task.x_train.iloc[fit_indices].reset_index(drop=True)
    raw_labels = np.asarray(task.y_train[fit_indices])
    common = {
        "n_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "norm_methods": list(STANDARD_TABICL_CONFIG["norm_methods"]),
        "feat_shuffle_method": str(STANDARD_TABICL_CONFIG["feat_shuffle_method"]),
        "outlier_threshold": float(STANDARD_TABICL_CONFIG["outlier_threshold"]),
        "batch_size": int(STANDARD_TABICL_CONFIG["batch_size"]),
        "kv_cache": bool(STANDARD_TABICL_CONFIG["kv_cache"]),
        "numerical_preprocessing": "existing",
        "random_state": int(STANDARD_TABICL_CONFIG["random_state"]),
        "device": str(device),
        "verbose": False,
    }
    if task.problem_type == "regression":
        estimator: TabICLClassifier | TabICLRegressor = TabICLRegressor(**common)
    else:
        estimator = TabICLClassifier(
            **common,
            class_shuffle_method=str(STANDARD_TABICL_CONFIG["class_shuffle_method"]),
            softmax_temperature=float(STANDARD_TABICL_CONFIG["softmax_temperature"]),
            average_logits=bool(STANDARD_TABICL_CONFIG["average_logits"]),
            support_many_classes=bool(STANDARD_TABICL_CONFIG["support_many_classes"]),
        )
    # ``fit`` normally loads the checkpoint.  The task/config runner owns one
    # frozen backbone instead, so all bags share those immutable weights.
    estimator.model_ = backbone
    estimator._load_model = lambda: None  # type: ignore[method-assign]
    estimator.fit(fit_x, raw_labels)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    # The public estimators cast/encode labels inside ``fit`` before handing
    # them to the ensemble generator.  Re-transforming the caller's regression
    # targets here can therefore differ by a few float32 ULPs (the caller may
    # still hold float64 values).  With a long in-context sequence those tiny
    # label changes can measurably alter predictions.  Reuse the exact labels
    # consumed by the public path for both regression and classification.
    scaled = np.asarray(estimator.ensemble_generator_.y_, dtype=np.float32)
    context_cap = config.get("max_context_rows")
    if context_cap is None or int(context_cap) >= len(scaled):
        support_indices = np.arange(len(scaled), dtype=int)
    else:
        support_indices = sample_prediction_context(
            scaled,
            problem_type=task.problem_type,
            max_context_rows=int(context_cap),
            rng=np.random.default_rng(_seed(protocol_seed, task.task_id, bag, 101)),
        )
    return _StandardBag(
        estimator=estimator,
        fit_labels=scaled,
        support_indices=support_indices,
        numerical_indices=_filtered_numerical_indices(estimator),
        problem_type=task.problem_type,
        n_classes=task.n_classes,
        row_interaction_chunk_rows=int(config["row_interaction_chunk_rows"]),
    )


def _make_adapters(bundle: _StandardBag, config: dict[str, Any], device: torch.device) -> _AdapterSet | None:
    """Create identity-initialised numerical splines in each normal branch."""

    if bundle.numerical_indices.size == 0:
        return None
    adapters: OrderedDict[str, nn.Module] = OrderedDict()
    n_numerical = int(bundle.numerical_indices.size)
    architecture = str(config.get("adapter_architecture", "fixed_cubic"))
    if architecture not in {"fixed_cubic", "adaptive_columns", "conditional_adaptive_columns"}:
        raise ValueError(f"unknown DirectSpline adapter architecture: {architecture!r}")
    if architecture != "fixed_cubic" and float(config.get("identity_regularization", 0.0)) != 0.0:
        raise ValueError("adaptive DirectSpline phase-1 arms require identity_regularization=0")
    # Standard preprocessing already defines the coordinates consumed by
    # TabICL.  DirectSpline's context statistics are therefore deliberately
    # replaced by location=0/scale=1 below.  Feeding tens of thousands of rows
    # into ``summarize_context`` only to discard its quantiles wasted VRAM and
    # caused avoidable OOMs.  A shape-only dummy constructs the identical
    # uniform-knot adapter.
    coordinate_dummy = torch.zeros((1, 1, n_numerical), dtype=torch.float32, device=device)
    identity_probe = torch.linspace(-5.0, 5.0, 17, device=device).view(1, 17, 1).expand(-1, -1, n_numerical)
    for method in bundle.estimator.ensemble_generator_.preprocessors_:
        if architecture == "fixed_cubic":
            adapter: nn.Module = DirectSplineTransform(
                coordinate_dummy,
                n_control_points=int(config["n_control_points"]),
                trainable_shape=True,
                trainable_location_scale=bool(config["trainable_location_scale"]),
                knot_placement="uniform",
                control_mode="monotone",
                cross_column_mixing_rank=int(config["cross_column_mixing_rank"]),
                cross_column_mixing_bound=float(config["cross_column_mixing_bound"]),
            ).to(device)
        else:
            raw_specs = config.get("adaptive_expert_specs", ((1, 4), (2, 8), (3, 20)))
            expert_specs = tuple((int(item[0]), int(item[1])) for item in raw_specs)
            adapter = AdaptiveDirectSplineTransform(
                coordinate_dummy,
                expert_specs=expert_specs,
                trainable_location_scale=bool(config["trainable_location_scale"]),
                cross_column_mixing_rank=int(config["cross_column_mixing_rank"]),
                cross_column_mixing_bound=float(config["cross_column_mixing_bound"]),
                conditional_rank=(
                    0
                    if architecture == "adaptive_columns"
                    else int(config.get("conditional_interaction_rank", 4))
                ),
                conditional_bound=float(config.get("conditional_interaction_bound", 0.25)),
                routing_temperature=float(config.get("adaptive_routing_temperature", 1.0)),
            ).to(device)
        # The standard preprocessing output is already the coordinate system
        # consumed by the normal estimator.  These values make a fresh spline
        # literally x -> x, not an extra standardisation pass.
        with torch.no_grad():
            if isinstance(adapter, AdaptiveDirectSplineTransform):
                for expert in adapter.experts:
                    expert.location.zero_()
                    expert.scale.fill_(1.0)
            else:
                adapter.location.zero_()
                adapter.scale.fill_(1.0)
            if not torch.equal(adapter.transform(identity_probe), identity_probe):  # type: ignore[attr-defined]
                raise RuntimeError("standard-pipeline DirectSpline did not initialise to bit-exact identity")
        adapters[method] = adapter
    return _AdapterSet(adapters)


def _optimizer(adapters: _AdapterSet, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Use the same weight-decay and fast-gate policy as the lite runner."""

    regular: list[torch.nn.Parameter] = []
    gated: list[torch.nn.Parameter] = []
    for name, parameter in adapters.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in ("gate", "offset", "log_scale")):
            gated.append(parameter)
        else:
            regular.append(parameter)
    groups: list[dict[str, Any]] = []
    if regular:
        groups.append(
            {
                "params": regular,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )
    if gated:
        groups.append(
            {
                "params": gated,
                "lr": float(config["learning_rate"]) * float(config["gate_learning_rate_factor"]),
                "weight_decay": 0.0,
            }
        )
    if not groups:
        raise RuntimeError("DirectSpline has no trainable parameters")
    return torch.optim.AdamW(groups)


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Decay every adapter parameter group along one reproducible cosine prefix.

    A validation-selected duration is only meaningful if its full-data refit
    receives the *same* learning-rate prefix.  ``total_steps`` is therefore
    the declared maximum schedule length, not the selected refit duration.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must lie in (0, 1]")

    def multiplier(step: int) -> float:
        progress = min(max(int(step), 0), total_steps) / float(total_steps)
        return float(
            min_lr_ratio
            + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _adapter_identity_penalty(adapters: _AdapterSet | None) -> torch.Tensor | None:
    """Return differentiable mean function-space deformation from identity.

    This penalises the actual numerical mapping rather than parameter norm.
    It is intentionally evaluated in each normalisation branch's coordinate
    system, mirroring the checkpoint diagnostics used by the preceding audit.
    """

    if adapters is None:
        return None
    penalties: list[torch.Tensor] = []
    for key in adapters._keys.values():
        adapter = adapters.adapters[key]
        parameters = adapter.parameters_for_transform()
        grid_values = torch.linspace(
            -adapter.standardized_range,
            adapter.standardized_range,
            33,
            dtype=parameters.location.dtype,
            device=parameters.location.device,
        )
        # A same-value-in-every-column grid cannot observe a mixing matrix
        # that happens to annihilate the all-ones direction.  Cyclically shift
        # the sweep by column so this remains a compact function-space probe
        # for the independent splines *and* for cross-column residuals.
        grid_rows = torch.arange(grid_values.numel(), device=grid_values.device).view(-1, 1)
        grid_columns = torch.arange(
            parameters.location.shape[1], device=grid_values.device
        ).view(1, -1)
        grid = grid_values[(grid_rows + grid_columns) % grid_values.numel()].unsqueeze(0).expand(
            parameters.location.shape[0], -1, -1
        )
        penalties.append((adapter.transform(grid) - grid).square().mean())
    if not penalties:  # pragma: no cover - _AdapterSet always has at least one branch
        return None
    return torch.stack(penalties).mean()


def _single_validation_split(
    task: OpenMLTaskData,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one deterministic train/validation split inside the outer train set.

    The adapter never sees validation labels while training.  Classification
    uses stratification so the normal TabICL estimator and validation metric
    retain every class on both sides of the split.
    """

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    indices = np.arange(len(task.y_train), dtype=int)
    if task.problem_type == "regression":
        splitter = ShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        fit_indices, validation_indices = next(splitter.split(indices))
        return np.asarray(fit_indices, dtype=int), np.asarray(validation_indices, dtype=int)

    counts = np.bincount(np.asarray(task.y_train, dtype=int))
    if counts.size == 0 or int(counts.min()) < 3:
        raise ValidationSplitInfeasibleError(
            f"task {task.task_id} cannot form a train/validation DirectSpline split: "
            "every classification class needs at least three outer-training rows"
        )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    try:
        fit_indices, validation_indices = next(splitter.split(indices, task.y_train))
    except ValueError as error:
        raise ValidationSplitInfeasibleError(
            f"task {task.task_id} cannot form the requested stratified validation split "
            f"at fraction {validation_fraction}"
        ) from error
    fit_labels = np.asarray(task.y_train[fit_indices], dtype=int)
    validation_labels = np.asarray(task.y_train[validation_indices], dtype=int)
    expected = np.arange(int(task.n_classes or 0), dtype=int)
    if not (np.array_equal(np.unique(fit_labels), expected) and np.array_equal(np.unique(validation_labels), expected)):
        raise ValidationSplitInfeasibleError(
            f"task {task.task_id} validation split omitted a class despite stratification"
        )
    if int(np.bincount(fit_labels, minlength=len(expected)).min()) < 2:
        raise ValidationSplitInfeasibleError(
            f"task {task.task_id} cannot retain two inner-training rows in every class at "
            f"validation_fraction={validation_fraction}"
        )
    return np.asarray(fit_indices, dtype=int), np.asarray(validation_indices, dtype=int)


def _split_sha256(fit_indices: np.ndarray, validation_indices: np.ndarray) -> str:
    """Fingerprint a persisted inner split without embedding every index in JSON."""

    digest = hashlib.sha256()
    for values in (np.asarray(fit_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)):
        digest.update(values.tobytes())
    return digest.hexdigest()


def _apply_adapter(
    canonical: torch.Tensor,
    *,
    numerical_indices: np.ndarray,
    adapter: nn.Module | None,
    filtered_feature_mask: np.ndarray | None = None,
) -> torch.Tensor:
    if canonical.ndim != 2:
        raise ValueError("canonical standard-preprocessed features must have shape (rows, features)")
    if adapter is None or numerical_indices.size == 0:
        return canonical
    indices = torch.as_tensor(numerical_indices, dtype=torch.long, device=canonical.device)
    transformed = canonical.clone()
    values = canonical.index_select(-1, indices).unsqueeze(0)
    effective_mixing = adapter.effective_mixing_matrix()  # type: ignore[attr-defined]
    if filtered_feature_mask is None:
        adapted_values = adapter.transform(values)  # type: ignore[attr-defined]
    else:
        # Public feature masking removes an all-NaN feature from the table.  A
        # learned cross-column residual must not smuggle that feature back into
        # the remaining outputs.  Keep the learned submatrix on unmasked
        # numerical columns and zero only masked input contributions.
        masked_numerical = torch.as_tensor(
            np.asarray(filtered_feature_mask, dtype=bool)[numerical_indices],
            dtype=torch.bool,
            device=canonical.device,
        )
        if hasattr(adapter, "unmixed_transform_masked"):
            unmixed = adapter.unmixed_transform_masked(values, masked_numerical)  # type: ignore[attr-defined]
        else:
            unmixed = adapter.unmixed_transform(values)  # type: ignore[attr-defined]
        if effective_mixing is None:
            adapted_values = unmixed
        else:
            mixing_input = unmixed.masked_fill(masked_numerical.view(1, 1, -1), 0.0)
            adapted_values = unmixed + torch.matmul(mixing_input, effective_mixing)
        adapted_values = adapted_values.to(values.dtype)
    transformed[..., indices] = adapted_values.squeeze(0)
    return transformed


@dataclass(frozen=True)
class _PreparedQuery:
    """Public-estimator-equivalent query representation without input mutation."""

    encoded: np.ndarray
    filtered: np.ndarray
    feature_mask: np.ndarray | None
    filtered_feature_mask: np.ndarray | None


def _prepare_query(bundle: _StandardBag, query_x: Any) -> _PreparedQuery:
    """Validate, copy, encode, and mask a query exactly as public predict does."""

    validated = validate_data(bundle.estimator, query_x, reset=False, dtype=None, skip_check_array=True)
    if hasattr(validated, "copy"):
        try:
            prepared = validated.copy(deep=True)
        except TypeError:
            prepared = validated.copy()
    else:
        prepared = np.array(validated, copy=True)

    if hasattr(prepared, "columns"):
        feature_mask = prepared.isna().all(axis=0).to_numpy(dtype=bool)
    else:
        values = np.asarray(prepared)
        if np.issubdtype(values.dtype, np.number):
            feature_mask = np.isnan(values).all(axis=0)
        else:
            feature_mask = np.asarray(
                [all(value != value for value in values[:, index]) for index in range(values.shape[1])],
                dtype=bool,
            )
    if not np.any(feature_mask):
        feature_mask = None
    elif hasattr(prepared, "iloc"):
        prepared.iloc[:, feature_mask] = 0.0
    else:
        prepared[:, feature_mask] = 0.0

    encoded = bundle.estimator.X_encoder_.transform(prepared)
    generator = bundle.estimator.ensemble_generator_
    filtered = generator.unique_filter_.transform(encoded)
    filtered_feature_mask = None
    if feature_mask is not None:
        filtered_feature_mask = np.asarray(
            feature_mask[np.asarray(generator.unique_filter_.features_to_keep_, dtype=bool)],
            dtype=bool,
        )
        filtered = np.asarray(filtered, dtype=np.float64).copy()
        filtered[:, filtered_feature_mask] = 0.0
    return _PreparedQuery(
        encoded=np.asarray(encoded),
        filtered=np.asarray(filtered),
        feature_mask=feature_mask,
        filtered_feature_mask=filtered_feature_mask,
    )


def _masked_feature_shuffles(
    shuffles: list[list[int]], filtered_feature_mask: np.ndarray | None
) -> list[list[int]]:
    if filtered_feature_mask is None:
        return shuffles
    kept = np.flatnonzero(~filtered_feature_mask)
    remap = {int(old): new for new, old in enumerate(kept)}
    return [[remap[index] for index in shuffle if index in remap] for shuffle in shuffles]


def _build_method_batch(
    *,
    bundle: _StandardBag,
    method: str,
    context_canonical: np.ndarray,
    query_canonical: np.ndarray,
    context_labels: np.ndarray,
    adapters: _AdapterSet | None,
    device: torch.device,
    filtered_feature_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[np.ndarray | None]]:
    """Create normal TabICL ensemble views after an optional spline nudge."""

    generator = bundle.estimator.ensemble_generator_
    configs = generator.ensemble_configs_[method]
    canonical = torch.as_tensor(
        np.concatenate((context_canonical, query_canonical), axis=0), dtype=torch.float32, device=device
    )
    adapter = None if adapters is None else adapters.for_method(method)
    canonical = _apply_adapter(
        canonical,
        numerical_indices=bundle.numerical_indices,
        adapter=adapter,
        filtered_feature_mask=filtered_feature_mask,
    )
    feature_shuffles = _masked_feature_shuffles(
        [feature_shuffle for feature_shuffle, _ in configs], filtered_feature_mask
    )
    if filtered_feature_mask is not None:
        kept = torch.as_tensor(
            np.flatnonzero(~filtered_feature_mask), dtype=torch.long, device=canonical.device
        )
        canonical = canonical.index_select(-1, kept)
    views = torch.stack([canonical[:, feature_shuffle] for feature_shuffle in feature_shuffles], dim=0)
    class_patterns = [pattern for _feature_shuffle, pattern in configs]
    if bundle.problem_type == "regression":
        labels = torch.as_tensor(context_labels, dtype=torch.float32, device=device).unsqueeze(0)
        labels = labels.expand(len(configs), -1)
    else:
        labels = torch.as_tensor(
            np.stack([np.asarray(pattern)[context_labels.astype(int)] for pattern in class_patterns]),
            dtype=torch.float32,
            device=device,
        )
    return views, labels, feature_shuffles, class_patterns


def _build_public_method_arrays(
    *,
    bundle: _StandardBag,
    method: str,
    context_canonical: np.ndarray,
    query_canonical: np.ndarray,
    context_labels: np.ndarray,
    adapters: _AdapterSet | None,
    device: torch.device,
    filtered_feature_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[np.ndarray | None]]:
    """Build public inference arrays without materialising every view on GPU.

    Ordinary TabICL creates shuffled ensemble views in host memory and lets
    ``_batch_forward`` transfer only its current member batch.  The adapter
    needs a device tensor for its canonical table, but stacking eight copies
    there first needlessly multiplied peak VRAM and disadvantaged this path.
    """

    generator = bundle.estimator.ensemble_generator_
    configs = generator.ensemble_configs_[method]
    canonical = np.asarray(
        np.concatenate((context_canonical, query_canonical), axis=0),
        dtype=np.float32,
    )
    adapter = None if adapters is None else adapters.for_method(method)
    if adapter is not None:
        with torch.no_grad():
            canonical_tensor = torch.as_tensor(canonical, dtype=torch.float32, device=device)
            canonical_tensor = _apply_adapter(
                canonical_tensor,
                numerical_indices=bundle.numerical_indices,
                adapter=adapter,
                filtered_feature_mask=filtered_feature_mask,
            )
            if filtered_feature_mask is not None:
                kept = torch.as_tensor(
                    np.flatnonzero(~filtered_feature_mask),
                    dtype=torch.long,
                    device=canonical_tensor.device,
                )
                canonical_tensor = canonical_tensor.index_select(-1, kept)
            canonical = canonical_tensor.cpu().numpy()
            del canonical_tensor
    elif filtered_feature_mask is not None:
        canonical = canonical[:, ~np.asarray(filtered_feature_mask, dtype=bool)]

    feature_shuffles = _masked_feature_shuffles(
        [feature_shuffle for feature_shuffle, _ in configs], filtered_feature_mask
    )
    views = np.stack(
        [canonical[:, feature_shuffle] for feature_shuffle in feature_shuffles],
        axis=0,
    )
    class_patterns = [pattern for _feature_shuffle, pattern in configs]
    if bundle.problem_type == "regression":
        labels = np.broadcast_to(
            np.asarray(context_labels, dtype=np.float32)[None, :],
            (len(configs), len(context_labels)),
        ).copy()
    else:
        labels = np.asarray(
            np.stack(
                [np.asarray(pattern)[context_labels.astype(int)] for pattern in class_patterns]
            ),
            dtype=np.float32,
        )
    return views, labels, feature_shuffles, class_patterns


def _forward_method(
    *,
    bundle: _StandardBag,
    views: torch.Tensor,
    labels: torch.Tensor,
    feature_shuffles: list[list[int]],
    checkpoint_activations: bool,
) -> torch.Tensor:
    """Forward one preprocessed normal-ensemble branch."""

    backbone = bundle.backbone

    def run(features: torch.Tensor) -> torch.Tensor:
        # Checkpoint recomputation re-enters this closure during backward, so
        # establish the deterministic autograd routing inside the closure.
        _enable_frozen_training_path(backbone)
        backbone.clear_cache()
        manager_configs = (
            bundle.estimator.inference_config_.COL_CONFIG,
            bundle.estimator.inference_config_.ROW_CONFIG,
            bundle.estimator.inference_config_.ICL_CONFIG,
        )
        use_amp = bool(manager_configs[0].get("use_amp", False)) and features.device.type == "cuda"
        use_fa3 = any(bool(config.get("use_fa3", False)) for config in manager_configs)
        autocast = torch.autocast(device_type="cuda") if use_amp else nullcontext()
        with flash_attn3_toggle(use_fa3), autocast:
            if not all(
                hasattr(backbone, name)
                for name in ("col_embedder", "row_interactor", "icl_predictor")
            ):
                # Lightweight test doubles do not expose TabICL's stages.
                return backbone(
                    X=features,
                    y_train=labels,
                    feature_shuffles=(
                        feature_shuffles if bundle.problem_type != "regression" else None
                    ),
                    return_logits=True,
                    softmax_temperature=float(
                        getattr(bundle.estimator, "softmax_temperature", 0.9)
                    ),
                    inference_config=bundle.estimator.inference_config_,
                )
            embeddings = _differentiable_public_column_embeddings(
                backbone=backbone,
                views=features,
                labels=labels,
                feature_shuffles=(
                    feature_shuffles if bundle.problem_type != "regression" else None
                ),
            )
            representations = _differentiable_row_interaction(
                backbone=backbone,
                embeddings=embeddings,
                chunk_rows=bundle.row_interaction_chunk_rows,
            )
            return backbone.icl_predictor(representations, y_train=labels)

    if checkpoint_activations:
        return checkpoint(run, views, use_reentrant=False)
    return run(views)


def _differentiable_row_interaction(
    *,
    backbone: TabICL,
    embeddings: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """Run the row-wise feature interaction in gradient-preserving row chunks.

    ``RowInteraction`` applies attention between a row's feature tokens; it
    has no interaction across the table's row dimension.  Sending every row
    at once to PyTorch SDPA creates a flattened batch of
    ``n_ensemble_members * n_rows``.  On large tables that can exceed a CUDA
    kernel launch limit even when the ordinary inference manager succeeds by
    auto-batching the same work.  Chunks therefore preserve the exact
    mathematical result while retaining the adapter gradient and the full
    context for the subsequent in-context-learning stage.

    The clone avoids ``RowInteraction``'s internal class-token assignment
    mutating a view of the complete embedding tensor.  It is differentiable
    and only holds one row chunk at a time.
    """

    if chunk_rows <= 0:
        raise ValueError("row_interaction_chunk_rows must be positive")
    total_rows = int(embeddings.shape[1])
    if total_rows <= chunk_rows:
        return backbone.row_interactor(embeddings)
    chunks: list[torch.Tensor] = []
    for start in range(0, total_rows, chunk_rows):
        stop = min(total_rows, start + chunk_rows)
        chunks.append(backbone.row_interactor(embeddings[:, start:stop].clone()))
    return torch.cat(chunks, dim=1)


def _enable_frozen_training_path(backbone: TabICL) -> None:
    """Enable TabICL's autograd path without making its weights trainable.

    TabICL's evaluation path dispatches through its memory manager, which
    deliberately executes under ``torch.no_grad``.  That is right for normal
    prediction and is exactly what the parity checks exercise, but cannot
    train an upstream spline.  TabICL's train-mode path computes the same
    supplied ensemble views directly and retains the input gradient.  Keep all
    backbone parameters and stochastic child layers in evaluation mode while
    changing only the modules whose ``training`` flag selects mathematical
    routing.
    """

    _set_frozen_autograd_routing(backbone)


def _differentiable_public_column_embeddings(
    *,
    backbone: TabICL,
    views: torch.Tensor,
    labels: torch.Tensor,
    feature_shuffles: list[list[int]] | None,
) -> torch.Tensor:
    """Reproduce public column embedding while retaining the input gradient.

    For a non-feature-grouped classifier, public inference embeds the first
    ensemble table once and obtains later feature-shuffled members by exactly
    remapping those embeddings.  Calling TabICL's ordinary training route
    instead embeds every member independently (and uses each member's class
    shuffle in the target-aware column encoder), which is a different model.
    This is the same public first-view/remapping mathematics without the
    inference manager's intentional ``no_grad`` wrapper.
    """

    embedder = backbone.col_embedder
    train_size = labels.shape[1]
    if embedder.feature_group:
        return embedder._train_forward_with_feature_group(
            views, labels, embed_with_test=False
        )
    if feature_shuffles is None:
        return embedder._train_forward_without_feature_group(
            views, labels, d=None, embed_with_test=False
        )
    if len(feature_shuffles) != views.shape[0]:
        raise RuntimeError("feature-shuffle count does not match ensemble views")

    first_table = views[0]
    if embedder.reserve_cls_tokens > 0:
        first_table = F.pad(first_table, (embedder.reserve_cls_tokens, 0), value=-100.0)
    features = first_table.transpose(0, 1).unsqueeze(-1)
    if embedder.target_aware:
        first_labels = labels[0].unsqueeze(0).expand(features.shape[0], -1)
    else:
        first_labels = None
    first_embeddings = embedder._compute_embeddings(
        features,
        train_size,
        first_labels,
        embed_with_test=False,
    )

    first_pattern = feature_shuffles[0]
    member_embeddings = [first_embeddings]
    for pattern in feature_shuffles[1:]:
        mapping = embedder.map_feature_shuffle(first_pattern, pattern)
        if embedder.reserve_cls_tokens > 0:
            mapping = [index + embedder.reserve_cls_tokens for index in mapping]
            mapping = list(range(embedder.reserve_cls_tokens)) + mapping
        member_embeddings.append(first_embeddings[mapping])
    return torch.stack(member_embeddings, dim=0).transpose(1, 2)


def _aggregate_training_classification_logits(
    outputs: list[tuple[torch.Tensor, list[np.ndarray | None]]],
    *,
    n_classes: int,
) -> torch.Tensor:
    """Undo ordinary class shuffles and average logits exactly as TabICL does."""

    corrected: list[torch.Tensor] = []
    for raw, patterns in outputs:
        # Match TabICLClassifier._batch_forward, which promotes every AMP
        # member to float32 before undoing class shuffles, averaging logits,
        # and applying the ensemble softmax.  Keeping this reduction in
        # float16 can move probabilities by several 1e-3 on larger tasks.
        raw = raw.float()
        for index, pattern in enumerate(patterns):
            if pattern is None:
                raise RuntimeError("classification ensemble member is missing its class shuffle")
            permutation = torch.as_tensor(pattern, dtype=torch.long, device=raw.device)
            corrected.append(raw[index, :, permutation][:, :n_classes])
    if not corrected:
        raise RuntimeError("normal classifier produced no ensemble outputs")
    average = torch.zeros_like(corrected[0])
    for member in corrected:
        average = average + member
    return average / len(corrected)


def _aggregate_public_classification_members(
    bundle: _StandardBag,
    members: list[np.ndarray],
    class_patterns: list[np.ndarray | None],
) -> np.ndarray:
    """Use the classifier's exact float32 NumPy aggregation order."""

    outputs = np.concatenate(members, axis=0)
    if len(class_patterns) != outputs.shape[0]:
        raise RuntimeError("classification member/shuffle count mismatch")
    average = np.zeros_like(outputs[0])
    for output, pattern in zip(outputs, class_patterns, strict=True):
        if pattern is None:
            raise RuntimeError("classification ensemble member is missing its class shuffle")
        average += output[..., np.asarray(pattern, dtype=int)]
    average /= len(class_patterns)
    estimator = bundle.estimator
    if not isinstance(estimator, TabICLClassifier):
        raise TypeError("classification aggregation requires TabICLClassifier")
    if not estimator.average_logits:
        raise RuntimeError("the standard DirectSpline path requires average_logits=True")
    average = estimator.softmax(
        average, axis=-1, temperature=float(estimator.softmax_temperature)
    )
    average = average / average.sum(axis=1, keepdims=True)
    return average.astype(np.float64)


def _normal_prediction(
    *,
    bundle: _StandardBag,
    query_x: Any,
    context_indices: np.ndarray,
    adapters: _AdapterSet | None,
    device: torch.device,
) -> np.ndarray:
    """Predict with normal ensemble views and an optional DirectSpline set."""

    # Adapter-modified arrays are deliberately handed back to the public
    # estimator's own batching/inference implementation.  There is only one
    # no-grad prediction engine; the experiment no longer reconstructs it.
    bundle.backbone.eval()
    generator = bundle.estimator.ensemble_generator_
    prepared = _prepare_query(bundle, query_x)
    classification_members: list[np.ndarray] = []
    class_patterns: list[np.ndarray | None] = []
    regression_members: list[np.ndarray] = []
    for method, preprocessor in generator.preprocessors_.items():
        context = preprocessor.X_transformed_[context_indices]
        query = preprocessor.transform(prepared.filtered)
        public_views, public_labels, feature_shuffles, method_patterns = (
            _build_public_method_arrays(
                bundle=bundle,
                method=method,
                context_canonical=context,
                query_canonical=query,
                context_labels=bundle.fit_labels[context_indices],
                adapters=adapters,
                device=device,
                filtered_feature_mask=prepared.filtered_feature_mask,
            )
        )
        if device.type == "cuda":
            # The canonical adapter tensor has already moved back to host
            # memory. Release its free cache so public auto-batching sees the
            # same available VRAM for identity and adapted calls.
            torch.cuda.empty_cache()
        if bundle.problem_type == "regression":
            estimator = bundle.estimator
            if not isinstance(estimator, TabICLRegressor):
                raise TypeError("regression prediction requires TabICLRegressor")
            regression_members.append(
                np.asarray(
                    estimator._batch_forward(public_views, public_labels, output_type="mean"),
                    dtype=np.float32,
                )
            )
        else:
            estimator = bundle.estimator
            if not isinstance(estimator, TabICLClassifier):
                raise TypeError("classification prediction requires TabICLClassifier")
            classification_members.append(
                np.asarray(
                    estimator._batch_forward(public_views, public_labels, feature_shuffles),
                    dtype=np.float32,
                )
            )
            class_patterns.extend(method_patterns)
        del public_views, public_labels
    if bundle.problem_type == "regression":
        if not regression_members:
            raise RuntimeError("normal regressor produced no ensemble outputs")
        return _aggregate_public_regression_members(bundle, regression_members)
    if bundle.n_classes is None:
        raise RuntimeError("classification bag has no class count")
    return _aggregate_public_classification_members(bundle, classification_members, class_patterns)


def _identity_prediction(bundle: _StandardBag, query_x: Any) -> np.ndarray:
    """Use the public estimator itself as the authoritative identity control."""

    if hasattr(query_x, "copy"):
        try:
            safe_query = query_x.copy(deep=True)
        except TypeError:
            safe_query = query_x.copy()
    else:
        safe_query = np.array(query_x, copy=True)
    if bundle.problem_type == "regression":
        return np.asarray(bundle.estimator.predict(safe_query), dtype=np.float64)
    return np.asarray(bundle.estimator.predict_proba(safe_query), dtype=np.float64)


def _candidate_deployment_error(
    problem_type: ProblemType,
    labels: np.ndarray,
    prediction: np.ndarray,
    *,
    n_classes: int | None,
) -> float:
    """Score an adapted candidate, mapping invalid output to a guarded loss."""

    try:
        value = deployment_error(problem_type, labels, prediction, n_classes=n_classes)
    except (TypeError, ValueError):
        return float("inf")
    return float(value) if np.isfinite(value) else float("inf")


def _difference_summary(left: Any, right: Any) -> dict[str, Any]:
    """Return compact, JSON-safe diagnostics for two numerical arrays."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    summary: dict[str, Any] = {
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
    }
    if left_array.shape != right_array.shape:
        summary["shape_match"] = False
        return summary
    summary["shape_match"] = True
    difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)).ravel()
    finite = difference[np.isfinite(difference)]
    summary["nonfinite_differences"] = int(difference.size - finite.size)
    if finite.size:
        summary.update(
            {
                "max_abs": float(np.max(finite)),
                "mean_abs": float(np.mean(finite)),
                "p95_abs": float(np.quantile(finite, 0.95)),
            }
        )
    return summary


def _aggregate_public_regression_members(bundle: _StandardBag, members: list[np.ndarray]) -> np.ndarray:
    """Apply the public regressor's exact per-member inverse scaling and mean."""

    values = np.concatenate(members, axis=0)
    n_estimators, n_rows = values.shape
    unscaled = bundle.estimator.y_scaler_.inverse_transform(values.reshape(-1, 1))
    return np.mean(unscaled.reshape(n_estimators, n_rows), axis=0).astype(np.float64)


def _identity_view_parity(
    *,
    bundle: _StandardBag,
    adapters: _AdapterSet | None,
    query_x: Any,
    device: torch.device,
    progress: Any,
    task_id: int,
    bag: int,
    split: str,
) -> tuple[float, str, bool]:
    """Require bit-exact public/no-spline/fresh-spline inputs on every row.

    Evaluation itself uses the estimator's own ``_batch_forward`` method, so
    comparing two separate GPU predictions is both redundant and vulnerable
    to memory-plan drift.  The only custom boundary is construction of the
    adapter-modified arrays; that boundary is audited here in float32, exactly
    as the public estimator consumes it.
    """

    prepared = _prepare_query(bundle, query_x)
    generator = bundle.estimator.ensemble_generator_
    public_data = None
    full_public_context = bundle.support_indices.size == bundle.fit_labels.size
    if full_public_context:
        public_data = generator.transform(
            prepared.encoded, mode="both", feature_mask=prepared.feature_mask
        )

    diagnostics: dict[str, Any] = {}
    maximum = 0.0
    for method, preprocessor in generator.preprocessors_.items():
        common = {
            "bundle": bundle,
            "method": method,
            "context_canonical": preprocessor.X_transformed_[bundle.support_indices],
            "query_canonical": preprocessor.transform(prepared.filtered),
            "context_labels": bundle.fit_labels[bundle.support_indices],
            "device": device,
            "filtered_feature_mask": prepared.filtered_feature_mask,
        }
        plain_views_array, plain_labels_array, plain_shuffles, plain_patterns = _build_public_method_arrays(
            **common, adapters=None
        )
        branch: dict[str, Any] = {}
        if public_data is not None:
            public_views, public_labels = public_data[method]
            branch["public_views_vs_no_spline"] = _difference_summary(
                np.asarray(public_views, dtype=np.float32), plain_views_array
            )
            branch["public_labels_vs_no_spline"] = _difference_summary(
                np.asarray(public_labels, dtype=np.float32), plain_labels_array
            )
            public_shuffles = (
                generator.feature_shuffles_[method]
                if prepared.feature_mask is None
                else generator.masked_feature_shuffles_[method]
            )
            branch["public_feature_shuffles_vs_no_spline"] = _difference_summary(
                np.asarray(public_shuffles, dtype=np.int64),
                np.asarray(plain_shuffles, dtype=np.int64),
            )
            if bundle.problem_type != "regression":
                public_patterns = generator.class_shuffles_[method]
                branch["public_class_shuffles_vs_no_spline"] = _difference_summary(
                    np.asarray(public_patterns, dtype=np.int64),
                    np.asarray(plain_patterns, dtype=np.int64),
                )
        if adapters is not None:
            spline_views, spline_labels, _shuffles, _patterns = _build_public_method_arrays(
                **common, adapters=adapters
            )
            branch["fresh_spline_views_vs_no_spline"] = _difference_summary(
                spline_views, plain_views_array
            )
            branch["fresh_spline_labels_vs_no_spline"] = _difference_summary(
                spline_labels, plain_labels_array
            )
            del spline_views, spline_labels
        diagnostics[str(method)] = branch

        for comparison in branch.values():
            if comparison.get("shape_match") is not True or comparison.get("nonfinite_differences") != 0:
                maximum = float("inf")
                break
            maximum = max(maximum, float(comparison.get("max_abs", 0.0)))

    if maximum != 0.0:
        _emit(
            progress,
            event="identity_view_parity_failed",
            task_id=task_id,
            bag=bag,
            split=split,
            problem_type=bundle.problem_type,
            query_rows=len(query_x),
            diagnostics=diagnostics,
        )
        raise RuntimeError(
            "standard-pipeline identity parity failed: public, no-spline, and fresh-spline input views "
            f"must be bit exact on all {len(query_x)} {split} rows. "
            f"parity_diagnostics={json.dumps(diagnostics, sort_keys=True)}"
        )
    reference = (
        "public_exact_input_views_full_query"
        if full_public_context
        else "matched_capped_exact_input_views_full_query"
    )
    return maximum, reference, full_public_context


def _many_class_training_logits(
    *,
    bundle: _StandardBag,
    views: torch.Tensor,
    labels: torch.Tensor,
    feature_shuffles: list[list[int]],
) -> torch.Tensor:
    """Differentiable counterpart of TabICL's public hierarchical classifier."""

    backbone = bundle.backbone
    _enable_frozen_training_path(backbone)
    manager_configs = (
        bundle.estimator.inference_config_.COL_CONFIG,
        bundle.estimator.inference_config_.ROW_CONFIG,
        bundle.estimator.inference_config_.ICL_CONFIG,
    )
    use_amp = bool(manager_configs[0].get("use_amp", False)) and views.device.type == "cuda"
    use_fa3 = any(bool(config.get("use_fa3", False)) for config in manager_configs)
    autocast = torch.autocast(device_type="cuda") if use_amp else nullcontext()
    temperature = float(bundle.estimator.softmax_temperature)

    with flash_attn3_toggle(use_fa3), autocast:
        # Bypass only the no-grad inference managers.  These are the same
        # frozen embedding/row/ICL modules and the same mixed-radix and
        # hierarchical mathematics used by public prediction.
        embeddings = _differentiable_public_column_embeddings(
            backbone=backbone,
            views=views,
            labels=labels,
            feature_shuffles=feature_shuffles,
        )
        representations = _differentiable_row_interaction(
            backbone=backbone,
            embeddings=embeddings,
            chunk_rows=bundle.row_interaction_chunk_rows,
        )
        predictor = backbone.icl_predictor
        train_size = labels.shape[1]
        table_logits: list[torch.Tensor] = []

        for representation, table_labels in zip(representations, labels, strict=True):
            predictor._fit_hierarchical(representation[:train_size], table_labels)
            root = predictor.root
            n_classes = len(root.classes_)
            query_representation = representation[train_size:]

            def process_node(node: Any) -> torch.Tensor:
                node_labels = node.y.to(query_representation.device)
                node_context = node.R.to(query_representation.device)
                node_input = torch.cat((node_context, query_representation), dim=0)
                if node.is_leaf:
                    local_labels = predictor._label_encoding(node_labels)
                    raw = predictor._icl_predictions(
                        node_input.unsqueeze(0).clone(), local_labels.unsqueeze(0)
                    )
                    local_count = len(node.classes_)
                    local_probabilities = torch.softmax(
                        raw[0, node_labels.shape[0] :, :local_count] / temperature,
                        dim=-1,
                    )
                    global_probabilities = local_probabilities.new_zeros(
                        (local_probabilities.shape[0], n_classes)
                    )
                    return global_probabilities.index_copy(
                        1,
                        node.classes_.to(query_representation.device, dtype=torch.long),
                        local_probabilities,
                    )

                group_labels = node.group_indices.to(query_representation.device)
                raw = predictor._icl_predictions(
                    node_input.unsqueeze(0).clone(), group_labels.unsqueeze(0)
                )
                n_groups = len(node.child_nodes)
                group_probabilities = torch.softmax(
                    raw[0, node_labels.shape[0] :, :n_groups] / temperature,
                    dim=-1,
                )
                # Match the public hierarchical predictor's left-to-right
                # accumulation order. A stacked reduction can use a different
                # tree under AMP and move pseudo-logits for 3+ child groups.
                final_probabilities = group_probabilities.new_zeros(
                    (query_representation.shape[0], n_classes)
                )
                for index, child in enumerate(node.child_nodes):
                    weighted_child = (
                        process_node(child)
                        * group_probabilities[:, index : index + 1]
                    )
                    final_probabilities = final_probabilities + weighted_child
                return final_probabilities

            probabilities = process_node(root)
            # Public inference converts hierarchical probabilities back to
            # temperature-scaled pseudo-logits before estimator ensembling.
            table_logits.append(temperature * torch.log(probabilities + 1e-6))

    return torch.stack(table_logits, dim=0)


def _training_logits(
    *,
    bundle: _StandardBag,
    adapters: _AdapterSet,
    context_indices: np.ndarray,
    query_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return normal-ensemble logits/predictions for one sampled training episode."""

    generator = bundle.estimator.ensemble_generator_
    classification_outputs: list[tuple[torch.Tensor, list[np.ndarray | None]]] = []
    regression_outputs: list[torch.Tensor] = []
    for method, preprocessor in generator.preprocessors_.items():
        views, labels, feature_shuffles, patterns = _build_method_batch(
            bundle=bundle,
            method=method,
            context_canonical=preprocessor.X_transformed_[context_indices],
            query_canonical=preprocessor.X_transformed_[query_indices],
            context_labels=bundle.fit_labels[context_indices],
            adapters=adapters,
            device=device,
        )
        if (
            bundle.problem_type != "regression"
            and bundle.n_classes is not None
            and bundle.n_classes > bundle.backbone.max_classes
        ):
            # Bind the per-method tensors now.  Checkpoint recomputes this
            # callback during backward, after the surrounding loop has
            # advanced to the next normalization method.
            def many_class_forward(
                features: torch.Tensor,
                method_labels: torch.Tensor = labels,
                method_feature_shuffles: list[list[int]] = feature_shuffles,
            ) -> torch.Tensor:
                return _many_class_training_logits(
                    bundle=bundle,
                    views=features,
                    labels=method_labels,
                    feature_shuffles=method_feature_shuffles,
                )

            raw = checkpoint(
                many_class_forward,
                views,
                use_reentrant=False,
            )
        else:
            raw = _forward_method(
                bundle=bundle,
                views=views,
                labels=labels,
                feature_shuffles=feature_shuffles,
                checkpoint_activations=True,
            )
        if bundle.problem_type == "regression":
            regression_outputs.append(bundle.backbone.quantile_dist(raw).quantiles.mean(dim=-1).float())
        else:
            classification_outputs.append((raw, patterns))
    if bundle.problem_type == "regression":
        return torch.cat(regression_outputs, dim=0).mean(dim=0)
    if bundle.n_classes is None:
        raise RuntimeError("classification bag has no class count")
    return _aggregate_training_classification_logits(classification_outputs, n_classes=bundle.n_classes)


def _fit_one_bag_standard(
    *,
    task: OpenMLTaskData,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    bag: int,
    config: dict[str, Any],
    protocol_seed: int,
    backbone: TabICL,
    device: torch.device,
    run_fingerprint_hash: str,
    progress: Any,
    requested_bags: int,
    effective_bags: int,
) -> BagPredictions:
    """Fit one full-strength guarded DirectSpline adapter without test labels."""

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=fit_indices,
        config=config,
        protocol_seed=protocol_seed,
        bag=bag,
        backbone=backbone,
        device=device,
    )
    validation_x = task.x_train.iloc[validation_indices].reset_index(drop=True)
    test_x = task.x_test
    raw_validation_y = task.y_train[validation_indices]
    # DirectSpline's optional low-rank mixing starts from random directions.
    # Fix that source per task/bag/config just as the legacy runner does, so a
    # resumed or repeated standard experiment is reproducible.
    adapter_seed = _seed(int(config["random_state"]), task.task_id, bag, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_validation, parity_reference, public_parity_validation = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=validation_x,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=bag,
        split="validation",
    )
    parity_test, parity_reference_test, public_parity_test = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=test_x,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=bag,
        split="test",
    )
    _emit(
        progress,
        event="bag_started",
        task_id=task.task_id,
        bag=bag,
        fit_rows=int(fit_indices.size),
        validation_rows=int(validation_indices.size),
        support_rows=int(bundle.support_indices.size),
        requested_bags=requested_bags,
        effective_bags=effective_bags,
        pipeline="standard_ensemble",
        normal_estimators=int(STANDARD_TABICL_CONFIG["n_estimators"]),
    )
    training_context_sizes: list[int] = []
    if adapters is None:
        # A numerical spline cannot alter a categorical-only (or wholly
        # constant-after-filtering) table.  Keep it as an explicit neutral tie.
        identity_validation = _normal_prediction(
            bundle=bundle,
            query_x=validation_x,
            context_indices=bundle.support_indices,
            adapters=None,
            device=device,
        )
        identity_test = _normal_prediction(
            bundle=bundle,
            query_x=test_x,
            context_indices=bundle.support_indices,
            adapters=None,
            device=device,
        )
        adapted_validation = identity_validation.copy()
        adapted_test = identity_test.copy()
        first_objective = final_objective = float("nan")
        executed_steps = 0
        best_step = 0
        has_valid_adapted_checkpoint = False
    else:
        optimizer = _optimizer(adapters, config)
        cosine_schedule_steps = config.get("cosine_schedule_steps")
        if cosine_schedule_steps is None:
            scheduler = None
        else:
            cosine_schedule_steps = int(cosine_schedule_steps)
            if cosine_schedule_steps < int(config["adapter_steps"]):
                raise ValueError(
                    "cosine_schedule_steps must be at least adapter_steps when a cosine schedule is requested"
                )
            scheduler = _cosine_scheduler(
                optimizer,
                total_steps=cosine_schedule_steps,
                min_lr_ratio=float(config["cosine_min_lr_ratio"]),
            )
        episode_rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 203))
        identity_state = _cpu_state_dict(adapters)
        best_state = identity_state
        best_error = float("inf")
        best_step = 0
        has_valid_adapted_checkpoint = False
        stale = 0
        checkpoint_records: list[dict[str, float | int | list[float]]] = []
        first_objective = final_objective = float("nan")
        executed_steps = 0
        for step in range(1, int(config["adapter_steps"]) + 1):
            configured_context_rows = config.get("train_context_rows")
            context_row_limit = (
                max(1, bundle.fit_labels.size - int(config["query_batch_rows"]))
                if configured_context_rows is None
                else int(configured_context_rows)
            )
            context_rows, query_rows = sample_episode_indices(
                bundle.fit_labels,
                problem_type=task.problem_type,
                context_rows=context_row_limit,
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
            training_context_sizes.append(int(context_rows.size))
            optimizer.zero_grad(set_to_none=True)
            output = _training_logits(
                bundle=bundle,
                adapters=adapters,
                context_indices=context_rows,
                query_indices=query_rows,
                device=device,
            )
            target = torch.as_tensor(bundle.fit_labels[query_rows], device=device)
            if task.problem_type == "regression":
                objective = F.mse_loss(output.flatten(), target.float().flatten())
            else:
                objective = F.cross_entropy(
                    output / float(bundle.estimator.softmax_temperature), target.long().flatten()
                )
            if not torch.isfinite(objective):
                _emit(
                    progress,
                    event="adapter_nonfinite_objective",
                    task_id=task.task_id,
                    bag=bag,
                    step=step,
                )
                del output, target, objective
                break
            if step == 1:
                first_objective = float(objective.detach())
            objective.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            final_objective = float(objective.detach())
            executed_steps = step
            del output, target, objective
            if step % int(config["validation_interval"]) == 0 or step == int(config["adapter_steps"]):
                # Score identity and adapted back-to-back while the live GPU
                # state is the same.  This prevents automatic inference-plan
                # changes from entering checkpoint selection as spline signal.
                paired_identity = _normal_prediction(
                    bundle=bundle,
                    query_x=validation_x,
                    context_indices=bundle.support_indices,
                    adapters=None,
                    device=device,
                )
                candidate = _normal_prediction(
                    bundle=bundle,
                    query_x=validation_x,
                    context_indices=bundle.support_indices,
                    adapters=adapters,
                    device=device,
                )
                paired_identity_error = deployment_error(
                    task.problem_type, raw_validation_y, paired_identity, n_classes=task.n_classes
                )
                candidate_error = _candidate_deployment_error(
                    task.problem_type, raw_validation_y, candidate, n_classes=task.n_classes
                )
                if candidate_error < best_error:
                    best_error = candidate_error
                    best_state = _cpu_state_dict(adapters)
                    best_step = step
                    has_valid_adapted_checkpoint = True
                    stale = 0
                else:
                    stale += 1
                checkpoint_records.append(
                    {
                        "step": int(step),
                        "validation_error": float(candidate_error),
                        "identity_validation_error": float(paired_identity_error),
                        "best_validation_error": float(best_error),
                        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    }
                )
                _emit(
                    progress,
                    event="adapter_validation",
                    task_id=task.task_id,
                    bag=bag,
                    step=step,
                    validation_error=float(candidate_error),
                    identity_validation_error=float(paired_identity_error),
                    best_validation_error=float(best_error),
                    stale_validations=stale,
                    elapsed_seconds=float(time.perf_counter() - started),
                )
                del paired_identity, candidate
                if config.get("adapter_patience") is not None and stale >= int(config["adapter_patience"]):
                    break
        adapters.load_state_dict(best_state, strict=True)
        del scheduler, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Recompute both arms after training cleanup, under the same execution
        # conditions used by the final guard.
        identity_validation = _normal_prediction(
            bundle=bundle,
            query_x=validation_x,
            context_indices=bundle.support_indices,
            adapters=None,
            device=device,
        )
        adapted_validation = _normal_prediction(
            bundle=bundle,
            query_x=validation_x,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
        identity_test = _normal_prediction(
            bundle=bundle,
            query_x=test_x,
            context_indices=bundle.support_indices,
            adapters=None,
            device=device,
        )
        adapted_test = _normal_prediction(
            bundle=bundle,
            query_x=test_x,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
    identity_error = deployment_error(task.problem_type, raw_validation_y, identity_validation, n_classes=task.n_classes)
    adapted_error = deployment_error(task.problem_type, raw_validation_y, adapted_validation, n_classes=task.n_classes)
    decision = choose_identity_guard(
        identity_error=identity_error,
        adapted_error=adapted_error,
        required_relative_improvement=float(config["guard_relative_improvement"]),
    )
    guarded_validation = adapted_validation if decision.use_adapted else identity_validation
    guarded_test = adapted_test if decision.use_adapted else identity_test
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "bag": int(bag),
        "fit_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "support_rows": int(bundle.support_indices.size),
        "requested_bags": requested_bags,
        "effective_bags": effective_bags,
        "n_features": int(task.x_train.shape[1]),
        "n_numerical_features": int(bundle.numerical_indices.size),
        "no_trainable_numerical_features": bool(adapters is None),
        "normal_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "pipeline": "standard_ensemble",
        "context_policy": (
            "all_fit_rows"
            if bundle.support_indices.size == bundle.fit_labels.size
            else "stratified_context_cap"
        ),
        "identity_parity_max_abs_validation": parity_validation,
        "identity_parity_max_abs_test": parity_test,
        "public_path_input_parity_checked_validation": bool(public_parity_validation),
        "public_path_input_parity_checked_test": bool(public_parity_test),
        # ``None`` means this is an intentionally capped-context diagnostic;
        # a capped input-view comparison must never be reported as a failed
        # full public-estimator parity check.
        "public_path_input_parity_passed": (
            True if public_parity_validation and public_parity_test else None
        ),
        "fresh_spline_view_identity_passed": bool(adapters is None or (parity_validation == 0.0 and parity_test == 0.0)),
        "identity_parity_reference": parity_reference,
        "identity_parity_reference_test": parity_reference_test,
        "identity_error": float(identity_error),
        "adapted_error": float(adapted_error),
        "relative_improvement": float(decision.relative_improvement),
        "guard_selected_adapted": bool(decision.use_adapted),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "adapter_steps_executed": executed_steps,
        "adapter_best_step": int(best_step),
        "adapter_has_valid_learned_checkpoint": bool(has_valid_adapted_checkpoint),
        "adapter_checkpoint_records": checkpoint_records if adapters is not None else [],
        "adapter_early_stopping_patience": config.get("adapter_patience"),
        "adapter_schedule": (
            {"kind": "constant"}
            if config.get("cosine_schedule_steps") is None
            else {
                "kind": "cosine",
                "horizon_steps": int(config["cosine_schedule_steps"]),
                "min_lr_ratio": float(config["cosine_min_lr_ratio"]),
            }
        ),
        "adapter_row_interaction_chunk_rows": int(bundle.row_interaction_chunk_rows),
        "adapter_configured_train_context_rows": config.get("train_context_rows"),
        "adapter_train_context_policy": (
            "all_non_query_fit_rows"
            if config.get("train_context_rows") is None
            else "sampled_context_cap"
        ),
        "adapter_observed_train_context_rows_min": (
            None if not training_context_sizes else int(min(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_max": (
            None if not training_context_sizes else int(max(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_mean": (
            None if not training_context_sizes else float(np.mean(training_context_sizes))
        ),
        "adapter_deployment_context_rows": int(bundle.support_indices.size),
        "adapter_train_to_deployment_context_ratio": (
            None
            if not training_context_sizes
            else float(np.mean(training_context_sizes) / bundle.support_indices.size)
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
        "run_fingerprint_hash": run_fingerprint_hash,
    }
    _emit(progress, event="bag_completed", task_id=task.task_id, **metadata)
    del adapters, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return BagPredictions(
        validation_indices=validation_indices,
        identity_validation=identity_validation,
        adapted_validation=adapted_validation,
        guarded_validation=guarded_validation,
        identity_test=identity_test,
        adapted_test=adapted_test,
        guarded_test=guarded_test,
        metadata=metadata,
    )


def run_task_config_standard(
    *,
    task: OpenMLTaskData,
    label: str,
    config: dict[str, Any],
    output_dir: Path,
    bags: int,
    protocol_seed: int,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
    resume: bool,
    run_fingerprint_hash: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Run a DirectSpline configuration through normal TabICLv2 bag models."""

    config_dir = _config_dir(output_dir, task, label)
    summary_path = config_dir / "config_summary.json"
    predictions_path = config_dir / "config_predictions.npz"
    effective_bags = effective_inner_bag_count(task, requested_bags=bags)
    if resume and summary_path.is_file() and predictions_path.is_file():
        summary = _json_load(summary_path)
        if summary.get("run_fingerprint_hash") != run_fingerprint_hash:
            raise RuntimeError(f"refusing to resume {config_dir}: artifacts use another immutable fingerprint")
        _emit(progress, event="config_reused", task_id=task.task_id, config_label=label)
        return summary
    backbone, checkpoint_path, checkpoint_metadata = load_frozen_backbone(
        problem_type=task.problem_type,
        device=device,
        classifier_checkpoint=classifier_checkpoint,
        regressor_checkpoint=regressor_checkpoint,
    )
    _emit(
        progress,
        event="config_started",
        task_id=task.task_id,
        config_label=label,
        requested_bags=bags,
        effective_bags=effective_bags,
        pipeline="standard_ensemble",
    )
    bag_results: list[BagPredictions] = []
    for bag, (fit_indices, validation_indices) in enumerate(
        _bag_splits(task, requested_bags=bags, seed=_seed(protocol_seed, task.task_id, 0))
    ):
        bag_path = config_dir / f"bag_{bag}.npz"
        if resume and bag_path.is_file():
            bag_result = _load_bag(bag_path)
            if bag_result.metadata.get("run_fingerprint_hash") != run_fingerprint_hash:
                raise RuntimeError(f"refusing to resume {bag_path}: artifacts use another immutable fingerprint")
            _emit(progress, event="bag_reused", task_id=task.task_id, config_label=label, bag=bag)
        else:
            bag_result = _fit_one_bag_standard(
                task=task,
                fit_indices=fit_indices,
                validation_indices=validation_indices,
                bag=bag,
                config=config,
                protocol_seed=protocol_seed,
                backbone=backbone,
                device=device,
                run_fingerprint_hash=run_fingerprint_hash,
                progress=progress,
                requested_bags=bags,
                effective_bags=effective_bags,
            )
            _save_bag(bag_path, bag_result)
        bag_results.append(bag_result)
    shape = _prediction_shape(len(task.y_train), task.problem_type, task.n_classes)
    identity_validation = np.full(shape, np.nan, dtype=np.float64)
    adapted_validation = np.full(shape, np.nan, dtype=np.float64)
    guarded_validation = np.full(shape, np.nan, dtype=np.float64)
    for result in bag_results:
        identity_validation[result.validation_indices] = result.identity_validation
        adapted_validation[result.validation_indices] = result.adapted_validation
        guarded_validation[result.validation_indices] = result.guarded_validation
    if not (
        np.isfinite(identity_validation).all()
        and np.isfinite(adapted_validation).all()
        and np.isfinite(guarded_validation).all()
    ):
        raise RuntimeError(f"task {task.task_id} config {label} did not produce complete OOF predictions")
    identity_test = np.mean([result.identity_test for result in bag_results], axis=0)
    adapted_test = np.mean([result.adapted_test for result in bag_results], axis=0)
    # Retouche applies its identity guard independently to every bag-fold and
    # deploys the mean of those eight selected members.  Preserve that exact
    # protocol for the guarded comparison; the raw adapted ensemble above
    # remains available to expose the full magnitude of wins and failures.
    guarded_test = np.mean([result.guarded_test for result in bag_results], axis=0)
    identity_oof_error = deployment_error(
        task.problem_type, task.y_train, identity_validation, n_classes=task.n_classes
    )
    adapted_oof_error = _candidate_deployment_error(
        task.problem_type, task.y_train, adapted_validation, n_classes=task.n_classes
    )
    config_guard = choose_identity_guard(
        identity_error=identity_oof_error,
        adapted_error=adapted_oof_error,
        required_relative_improvement=float(config["guard_relative_improvement"]),
    )
    np.savez_compressed(
        predictions_path,
        identity_validation=identity_validation,
        adapted_validation=adapted_validation,
        guarded_validation=guarded_validation,
        identity_test=identity_test,
        adapted_test=adapted_test,
        guarded_test=guarded_test,
    )
    summary = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "config_label": label,
        "config": config,
        "checkpoint": checkpoint_metadata,
        "run_fingerprint_hash": run_fingerprint_hash,
        "requested_bags": bags,
        "effective_bags": effective_bags,
        "pipeline": "standard_ensemble",
        "identity_definition": (
            "Normal TabICLv2 fitted independently on each inner-bag fit partition, with the same preprocessing "
            "views, class/feature shuffles, temperature, and logit aggregation as the adapted path."
        ),
        "context_policy": (
            "all_fit_rows"
            if config.get("max_context_rows") is None
            else "stratified_context_cap_when_needed"
        ),
        "validation": {
            "identity": _metric_bundle(task.problem_type, task.y_train, identity_validation, task.n_classes),
            "adapted": _metric_bundle(task.problem_type, task.y_train, adapted_validation, task.n_classes),
            "guarded": _metric_bundle(task.problem_type, task.y_train, guarded_validation, task.n_classes),
        },
        "test_metrics_deferred_to_task_summary": True,
        "guard_protocol": "retouche_per_bag_validation_guard_then_test_ensemble",
        "guard_selected_adapted_fraction": float(
            np.mean([bool(result.metadata["guard_selected_adapted"]) for result in bag_results])
        ),
        "adapter_valid_learned_checkpoint_fraction": float(
            np.mean(
                [
                    bool(result.metadata["adapter_has_valid_learned_checkpoint"])
                    for result in bag_results
                ]
            )
        ),
        "global_oof_guard_selected_adapted_diagnostic": bool(config_guard.use_adapted),
        "global_oof_guard_relative_improvement_diagnostic": float(
            config_guard.relative_improvement
        ),
        "mean_train_seconds_per_bag": float(np.mean([float(result.metadata["train_seconds"]) for result in bag_results])),
        "max_peak_allocated_gib": float(np.max([float(result.metadata["peak_allocated_gib"]) for result in bag_results])),
        "max_identity_parity_abs": float(
            np.max(
                [
                    max(
                        float(result.metadata["identity_parity_max_abs_validation"]),
                        float(result.metadata["identity_parity_max_abs_test"]),
                    )
                    for result in bag_results
                ]
            )
        ),
        "bag_metadata": [result.metadata for result in bag_results],
    }
    _json_dump(summary_path, summary)
    _emit(
        progress,
        event="config_completed",
        task_id=task.task_id,
        config_label=label,
        guarded_validation_error=summary["validation"]["guarded"]["deployment_error"],
    )
    del backbone, checkpoint_path
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _full_context_refit_dir(
    output_dir: Path,
    task: OpenMLTaskData,
    label: str,
) -> Path:
    """Return the resumable artifact directory for one all-row refit."""

    return _config_dir(output_dir, task, label) / "full_context_refit"


def _full_context_checkpoint_audit_dir(
    output_dir: Path,
    task: OpenMLTaskData,
    label: str,
) -> Path:
    """Return the artifact directory for a development-only refit learning curve."""

    return _config_dir(output_dir, task, label) / "full_context_checkpoint_audit"


@torch.no_grad()
def _adapter_checkpoint_diagnostics(adapters: _AdapterSet | None) -> dict[str, Any]:
    """Summarize function-space movement at one full-refit checkpoint.

    Parameter norms are not a useful proxy here because each normalisation
    branch has different parameterisations.  These diagnostics instead record
    the generated function's grid deformation from identity, together with the
    effective spline and cross-column gates.
    """

    if adapters is None:
        return {
            "has_trainable_numerical_features": False,
            "mean_grid_deformation": 0.0,
            "mean_gate": 0.0,
            "max_gate": 0.0,
            "mean_abs_location": 0.0,
            "mean_abs_log_scale": 0.0,
            "mean_mixing_spectral_norm": 0.0,
            "branches": {},
        }
    branches: dict[str, dict[str, Any]] = {}
    for method, key in adapters._keys.items():
        adapter = adapters.adapters[key]
        if hasattr(adapter, "checkpoint_diagnostics"):
            branches[method] = adapter.checkpoint_diagnostics()  # type: ignore[attr-defined]
        else:
            parameters = adapter.parameters_for_transform()  # type: ignore[attr-defined]
            _mixing_mean, _mixing_max, mixing_spectral = adapter.mixing_diagnostics()  # type: ignore[attr-defined]
            grid = torch.linspace(
                -adapter.standardized_range,  # type: ignore[attr-defined]
                adapter.standardized_range,  # type: ignore[attr-defined]
                33,
                dtype=parameters.location.dtype,
                device=parameters.location.device,
            ).view(1, -1, 1).expand(parameters.location.shape[0], -1, parameters.location.shape[1])
            grid_deformation = (adapter.transform(grid) - grid).square().mean()  # type: ignore[attr-defined]
            branches[method] = {
                "grid_deformation": float(grid_deformation.detach()),
                "mean_gate": float(parameters.gate.mean().detach()),
                "max_gate": float(parameters.gate.max().detach()),
                "mean_abs_location": float(parameters.location.abs().mean().detach()),
                "mean_abs_log_scale": float(parameters.scale.log().abs().mean().detach()),
                "mixing_spectral_norm": float(mixing_spectral.detach()),
            }
    values = list(branches.values())
    return {
        "has_trainable_numerical_features": True,
        "mean_grid_deformation": float(np.mean([item["grid_deformation"] for item in values])),
        "mean_gate": float(np.mean([item["mean_gate"] for item in values])),
        "max_gate": float(np.max([item["max_gate"] for item in values])),
        "mean_abs_location": float(np.mean([item["mean_abs_location"] for item in values])),
        "mean_abs_log_scale": float(np.mean([item["mean_abs_log_scale"] for item in values])),
        "mean_mixing_spectral_norm": float(
            np.mean([item["mixing_spectral_norm"] for item in values])
        ),
        "branches": branches,
    }


def _fit_full_context_refit_standard(
    *,
    task: OpenMLTaskData,
    config: dict[str, Any],
    refit_steps: int,
    protocol_seed: int,
    backbone: TabICL,
    device: torch.device,
    progress: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a fresh DirectSpline on every outer-training row exactly once.

    The caller supplies the step budget frozen in the experiment manifest.
    This function deliberately receives no test labels: it is safe to use test
    *features* for the final prediction and parity audit, but no test metric is
    computed until the caller has frozen this result.
    """

    if refit_steps < 0:
        raise ValueError("refit_steps must be non-negative")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full_indices = np.arange(len(task.y_train), dtype=int)
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=full_indices,
        config=config,
        protocol_seed=protocol_seed,
        # This is a separate deterministic initialization from any OOF bag.
        # It cannot affect context selection when every fit row is retained.
        bag=-1,
        backbone=backbone,
        device=device,
    )
    adapter_seed = _seed(int(config["random_state"]), task.task_id, -1, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_test, parity_reference_test, public_parity_test = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=task.x_test,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=-1,
        split="full_context_refit_test",
    )
    identity_test = _normal_prediction(
        bundle=bundle,
        query_x=task.x_test,
        context_indices=bundle.support_indices,
        adapters=None,
        device=device,
    )
    training_context_sizes: list[int] = []
    first_objective = final_objective = float("nan")
    executed_steps = 0
    no_trainable_numerical_features = adapters is None
    if adapters is None or refit_steps == 0:
        adapted_test = identity_test.copy()
    else:
        optimizer = _optimizer(adapters, config)
        episode_rng = np.random.default_rng(
            _seed(int(config["random_state"]), task.task_id, -1, 203)
        )
        for step in range(1, refit_steps + 1):
            configured_context_rows = config.get("train_context_rows")
            context_row_limit = (
                max(1, bundle.fit_labels.size - int(config["query_batch_rows"]))
                if configured_context_rows is None
                else int(configured_context_rows)
            )
            context_rows, query_rows = sample_episode_indices(
                bundle.fit_labels,
                problem_type=task.problem_type,
                context_rows=context_row_limit,
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
            training_context_sizes.append(int(context_rows.size))
            optimizer.zero_grad(set_to_none=True)
            output = _training_logits(
                bundle=bundle,
                adapters=adapters,
                context_indices=context_rows,
                query_indices=query_rows,
                device=device,
            )
            target = torch.as_tensor(bundle.fit_labels[query_rows], device=device)
            if task.problem_type == "regression":
                objective = F.mse_loss(output.flatten(), target.float().flatten())
            else:
                objective = F.cross_entropy(
                    output / float(bundle.estimator.softmax_temperature), target.long().flatten()
                )
            if not torch.isfinite(objective):
                _emit(
                    progress,
                    event="full_refit_nonfinite_objective",
                    task_id=task.task_id,
                    step=step,
                )
                del output, target, objective
                break
            if step == 1:
                first_objective = float(objective.detach())
            objective.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
            optimizer.step()
            final_objective = float(objective.detach())
            executed_steps = step
            del output, target, objective
            if step % int(config["validation_interval"]) == 0 or step == refit_steps:
                _emit(
                    progress,
                    event="full_refit_step",
                    task_id=task.task_id,
                    step=step,
                    objective=final_objective,
                    elapsed_seconds=float(time.perf_counter() - started),
                )
        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        adapted_test = _normal_prediction(
            bundle=bundle,
            query_x=task.x_test,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "fit_rows": int(full_indices.size),
        "support_rows": int(bundle.support_indices.size),
        "n_features": int(task.x_train.shape[1]),
        "n_numerical_features": int(bundle.numerical_indices.size),
        "no_trainable_numerical_features": bool(no_trainable_numerical_features),
        "normal_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "context_policy": (
            "all_outer_training_rows"
            if bundle.support_indices.size == bundle.fit_labels.size
            else "stratified_context_cap"
        ),
        "identity_parity_max_abs_test": float(parity_test),
        "identity_parity_reference_test": parity_reference_test,
        "public_path_input_parity_checked_test": bool(public_parity_test),
        "public_path_input_parity_passed": True if public_parity_test else None,
        "fresh_spline_view_identity_passed": bool(adapters is None or parity_test == 0.0),
        "adapter_seed": int(adapter_seed),
        "adapter_refit_steps_requested": int(refit_steps),
        "adapter_steps_executed": int(executed_steps),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "adapter_configured_train_context_rows": config.get("train_context_rows"),
        "adapter_train_context_policy": (
            "all_non_query_outer_training_rows"
            if config.get("train_context_rows") is None
            else "sampled_context_cap"
        ),
        "adapter_observed_train_context_rows_min": (
            None if not training_context_sizes else int(min(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_max": (
            None if not training_context_sizes else int(max(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_mean": (
            None if not training_context_sizes else float(np.mean(training_context_sizes))
        ),
        "adapter_deployment_context_rows": int(bundle.support_indices.size),
        "adapter_train_to_deployment_context_ratio": (
            None
            if not training_context_sizes
            else float(np.mean(training_context_sizes) / bundle.support_indices.size)
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
    }
    del adapters, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return identity_test, adapted_test, metadata


def _fit_full_context_refit_checkpoint_audit_standard(
    *,
    task: OpenMLTaskData,
    config: dict[str, Any],
    checkpoint_steps: tuple[int, ...],
    protocol_seed: int,
    backbone: TabICL,
    device: torch.device,
    checkpoint_state_dir: Path,
    progress: Any,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Freeze a full-refit prediction curve without selecting from it.

    This deliberately differs from :func:`_fit_full_context_refit_standard`:
    every listed checkpoint is retained for a *development-only* learning-curve
    audit.  It receives no outer-test labels, makes no checkpoint decision,
    and does not alter the unconditional full-refit deployment path.
    """

    if not checkpoint_steps:
        raise ValueError("checkpoint_steps must not be empty")
    if checkpoint_steps != tuple(sorted(set(checkpoint_steps))):
        raise ValueError("checkpoint_steps must be sorted and unique")
    if checkpoint_steps[0] != 0 or any(step < 0 for step in checkpoint_steps):
        raise ValueError("checkpoint_steps must start at zero and be non-negative")
    max_steps = int(checkpoint_steps[-1])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full_indices = np.arange(len(task.y_train), dtype=int)
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=full_indices,
        config=config,
        protocol_seed=protocol_seed,
        bag=-1,
        backbone=backbone,
        device=device,
    )
    adapter_seed = _seed(int(config["random_state"]), task.task_id, -1, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_test, parity_reference_test, public_parity_test = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=task.x_test,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=-1,
        split="full_context_checkpoint_audit_test",
    )
    identity_test = _normal_prediction(
        bundle=bundle,
        query_x=task.x_test,
        context_indices=bundle.support_indices,
        adapters=None,
        device=device,
    )
    checkpoint_state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_predictions: dict[int, np.ndarray] = {}
    checkpoint_metadata: list[dict[str, Any]] = []

    def record_checkpoint(step: int, objective: float | None) -> None:
        state_path = checkpoint_state_dir / f"step_{step:06d}.pt"
        state = {} if adapters is None else _cpu_state_dict(adapters)
        torch.save({"step": int(step), "adapter_state_dict": state}, state_path)
        prediction = (
            identity_test.copy()
            if step == 0 or adapters is None
            else _normal_prediction(
                bundle=bundle,
                query_x=task.x_test,
                context_indices=bundle.support_indices,
                adapters=adapters,
                device=device,
            )
        )
        checkpoint_predictions[step] = prediction
        diagnostics = _adapter_checkpoint_diagnostics(adapters)
        record = {
            "step": int(step),
            "objective": objective,
            "elapsed_seconds": float(time.perf_counter() - started),
            "state_path": state_path.name,
            "adapter_diagnostics": diagnostics,
        }
        checkpoint_metadata.append(record)
        _emit(
            progress,
            event="full_refit_checkpoint_frozen",
            task_id=task.task_id,
            step=int(step),
            objective=objective,
            elapsed_seconds=record["elapsed_seconds"],
            mean_grid_deformation=diagnostics["mean_grid_deformation"],
        )

    record_checkpoint(0, None)
    training_context_sizes: list[int] = []
    first_objective = final_objective = float("nan")
    executed_steps = 0
    no_trainable_numerical_features = adapters is None
    if adapters is None:
        for step in checkpoint_steps[1:]:
            record_checkpoint(step, None)
    else:
        optimizer = _optimizer(adapters, config)
        episode_rng = np.random.default_rng(
            _seed(int(config["random_state"]), task.task_id, -1, 203)
        )
        requested_steps = set(checkpoint_steps[1:])
        for step in range(1, max_steps + 1):
            configured_context_rows = config.get("train_context_rows")
            context_row_limit = (
                max(1, bundle.fit_labels.size - int(config["query_batch_rows"]))
                if configured_context_rows is None
                else int(configured_context_rows)
            )
            context_rows, query_rows = sample_episode_indices(
                bundle.fit_labels,
                problem_type=task.problem_type,
                context_rows=context_row_limit,
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
            training_context_sizes.append(int(context_rows.size))
            optimizer.zero_grad(set_to_none=True)
            output = _training_logits(
                bundle=bundle,
                adapters=adapters,
                context_indices=context_rows,
                query_indices=query_rows,
                device=device,
            )
            target = torch.as_tensor(bundle.fit_labels[query_rows], device=device)
            if task.problem_type == "regression":
                objective_tensor = F.mse_loss(output.flatten(), target.float().flatten())
            else:
                objective_tensor = F.cross_entropy(
                    output / float(bundle.estimator.softmax_temperature), target.long().flatten()
                )
            if not torch.isfinite(objective_tensor):
                _emit(
                    progress,
                    event="full_refit_nonfinite_objective",
                    task_id=task.task_id,
                    step=step,
                )
                del output, target, objective_tensor
                break
            if step == 1:
                first_objective = float(objective_tensor.detach())
            objective_tensor.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
            optimizer.step()
            final_objective = float(objective_tensor.detach())
            executed_steps = step
            del output, target, objective_tensor
            if step in requested_steps:
                record_checkpoint(step, final_objective)
        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "fit_rows": int(full_indices.size),
        "support_rows": int(bundle.support_indices.size),
        "n_features": int(task.x_train.shape[1]),
        "n_numerical_features": int(bundle.numerical_indices.size),
        "no_trainable_numerical_features": bool(no_trainable_numerical_features),
        "normal_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "context_policy": (
            "all_outer_training_rows"
            if bundle.support_indices.size == bundle.fit_labels.size
            else "stratified_context_cap"
        ),
        "identity_parity_max_abs_test": float(parity_test),
        "identity_parity_reference_test": parity_reference_test,
        "public_path_input_parity_checked_test": bool(public_parity_test),
        "public_path_input_parity_passed": True if public_parity_test else None,
        "fresh_spline_view_identity_passed": bool(adapters is None or parity_test == 0.0),
        "adapter_seed": int(adapter_seed),
        "adapter_checkpoint_steps_requested": list(checkpoint_steps),
        "adapter_checkpoint_steps_frozen": sorted(checkpoint_predictions),
        "adapter_steps_executed": int(executed_steps),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "adapter_configured_train_context_rows": config.get("train_context_rows"),
        "adapter_train_context_policy": (
            "all_non_query_outer_training_rows"
            if config.get("train_context_rows") is None
            else "sampled_context_cap"
        ),
        "adapter_observed_train_context_rows_min": (
            None if not training_context_sizes else int(min(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_max": (
            None if not training_context_sizes else int(max(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_mean": (
            None if not training_context_sizes else float(np.mean(training_context_sizes))
        ),
        "adapter_deployment_context_rows": int(bundle.support_indices.size),
        "adapter_train_to_deployment_context_ratio": (
            None
            if not training_context_sizes
            else float(np.mean(training_context_sizes) / bundle.support_indices.size)
        ),
        "checkpoint_metadata": checkpoint_metadata,
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
    }
    del adapters, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return identity_test, checkpoint_predictions, metadata


def _load_prediction(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        return np.asarray(artifact[key])


def _validation_selected_refit_dir(
    output_dir: Path,
    task: OpenMLTaskData,
    label: str,
) -> Path:
    """Return the durable artifact directory for one validation/refit arm."""

    return _config_dir(output_dir, task, label) / "validation_selected_refit"


def _artifact_sha256(path: Path) -> str:
    """Hash a durable validation/refit artifact without holding it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _try_load_validation_selected_refit_artifact(
    *,
    task: OpenMLTaskData,
    label: str,
    config: dict[str, Any],
    validation_fraction: float,
    validation_seed: int,
    task_split_seed: int,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    split_hash: str,
    summary_path: Path,
    predictions_path: Path,
    split_path: Path,
    selected_state_path: Path,
    run_fingerprint_hash: str,
) -> dict[str, Any] | None:
    """Return one complete, verified resume artifact or ``None`` to rebuild it.

    The prediction archive is a completion boundary: outer-test metrics are
    scored only after both variants' complete archives are present. A malformed
    or interrupted artifact is deliberately rebuilt, while a different run
    fingerprint remains a hard error rather than an unsafe overwrite.
    """

    try:
        result = _json_load(summary_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if result.get("run_fingerprint_hash") != run_fingerprint_hash:
        raise RuntimeError(
            f"refusing to resume {summary_path.parent}: validation/refit artifacts use another immutable fingerprint"
        )
    if result.get("experiment_type") != "validation_selected_checkpoint_then_full_context_refit":
        return None
    if (
        result.get("task_id") != task.task_id
        or result.get("dataset_id") != task.dataset_id
        or result.get("dataset_name") != task.dataset_name
        or result.get("outer_split_hash") != task.outer_split_hash
        or result.get("config_label") != label
        or result.get("config") != config
    ):
        return None
    split_metadata = result.get("validation_split")
    if not isinstance(split_metadata, dict) or split_metadata != {
        "fraction": float(validation_fraction),
        "root_seed": int(validation_seed),
        "task_seed": int(task_split_seed),
        "inner_train_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "sha256": split_hash,
        "artifact": split_path.name,
    }:
        return None
    selection = result.get("selection")
    refit = result.get("full_context_refit")
    if not isinstance(selection, dict) or not isinstance(refit, dict):
        return None
    try:
        selected_step = int(selection["selected_step"])
        selected_use_adapted = bool(selection["selected_use_adapted"])
    except (KeyError, TypeError, ValueError):
        return None
    if selected_step < 0 or selected_step > int(config["adapter_steps"]):
        return None
    if int(refit.get("selected_steps_requested", -1)) != selected_step:
        return None
    if not isinstance(refit.get("selected_duration_completed"), bool):
        return None
    expected_keys = {
        "inner_identity_validation": _prediction_shape(
            int(validation_indices.size), task.problem_type, task.n_classes
        ),
        "inner_selected_validation": _prediction_shape(
            int(validation_indices.size), task.problem_type, task.n_classes
        ),
        "inner_identity_test": _prediction_shape(len(task.x_test), task.problem_type, task.n_classes),
        "inner_selected_test": _prediction_shape(len(task.x_test), task.problem_type, task.n_classes),
        "full_identity_test": _prediction_shape(len(task.x_test), task.problem_type, task.n_classes),
        "full_selected_test": _prediction_shape(len(task.x_test), task.problem_type, task.n_classes),
    }
    try:
        with np.load(split_path, allow_pickle=False) as persisted_split:
            persisted_fit_indices = np.asarray(persisted_split["inner_train_indices"], dtype=int)
            persisted_validation_indices = np.asarray(persisted_split["validation_indices"], dtype=int)
        if not (
            np.array_equal(persisted_fit_indices, fit_indices)
            and np.array_equal(persisted_validation_indices, validation_indices)
            and _split_sha256(persisted_fit_indices, persisted_validation_indices) == split_hash
        ):
            return None
        if result.get("prediction_artifact_sha256") != _artifact_sha256(predictions_path):
            return None
        if result.get("selected_adapter_state_sha256") != _artifact_sha256(selected_state_path):
            return None
        with np.load(predictions_path, allow_pickle=False) as predictions:
            if set(predictions.files) != set(expected_keys):
                return None
            for key, expected_shape in expected_keys.items():
                array = np.asarray(predictions[key])
                if array.shape != expected_shape or not np.issubdtype(array.dtype, np.number):
                    return None
        try:
            state_payload = torch.load(selected_state_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - old torch compatibility.
            state_payload = torch.load(selected_state_path, map_location="cpu")
        if not isinstance(state_payload, dict):
            return None
        if (
            int(state_payload.get("selected_step", -1)) != selected_step
            or bool(state_payload.get("selected_use_adapted")) != selected_use_adapted
            or not isinstance(state_payload.get("adapter_state_dict"), dict)
        ):
            return None
    except Exception:  # Corrupt/interrupted local artifact: rebuild this arm deterministically.
        return None
    return result


def _adapter_training_objective(
    *,
    bundle: _StandardBag,
    adapters: _AdapterSet,
    context_indices: np.ndarray,
    query_indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return task loss, function-space identity penalty, and total loss."""

    output = _training_logits(
        bundle=bundle,
        adapters=adapters,
        context_indices=context_indices,
        query_indices=query_indices,
        device=device,
    )
    target = torch.as_tensor(bundle.fit_labels[query_indices], device=device)
    if bundle.problem_type == "regression":
        task_loss = F.mse_loss(output.flatten(), target.float().flatten())
    else:
        task_loss = F.cross_entropy(
            output / float(bundle.estimator.softmax_temperature), target.long().flatten()
        )
    regularization_weight = float(config.get("identity_regularization", 0.0))
    if regularization_weight == 0.0:
        # Keep the scheduler-only control free of an unnecessary second
        # transform graph (and of any penalty-only non-finite failure). Grid
        # deformation is still recorded at each selected checkpoint.
        penalty = torch.zeros((), dtype=task_loss.dtype, device=task_loss.device)
        total = task_loss
    else:
        penalty = _adapter_identity_penalty(adapters)
        if penalty is None:  # pragma: no cover - callers only invoke this with adapters
            penalty = torch.zeros((), dtype=task_loss.dtype, device=task_loss.device)
        total = task_loss + regularization_weight * penalty
    del output, target
    return task_loss, penalty, total


def _fit_validation_selected_checkpoint_standard(
    *,
    task: OpenMLTaskData,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict[str, Any],
    protocol_seed: int,
    backbone: TabICL,
    device: torch.device,
    progress: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Choose identity or a spline checkpoint using only inner validation labels.

    No outer-test metric is computed here.  The selected inner-train model's
    test prediction is frozen only after validation selection, and its test
    label is intentionally left to the separate task-summary stage.
    """

    max_steps = int(config["adapter_steps"])
    checkpoint_interval = int(config["selection_checkpoint_interval"])
    scheduler_horizon = int(config["cosine_schedule_steps"])
    if max_steps <= 0 or checkpoint_interval <= 0 or scheduler_horizon < max_steps:
        raise ValueError("invalid validation-selection schedule")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=fit_indices,
        config=config,
        protocol_seed=protocol_seed,
        bag=-2,
        backbone=backbone,
        device=device,
    )
    validation_x = task.x_train.iloc[validation_indices].reset_index(drop=True)
    validation_y = np.asarray(task.y_train[validation_indices])
    adapter_seed = _seed(int(config["random_state"]), task.task_id, -2, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_validation, parity_validation_reference, public_parity_validation = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=validation_x,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=-2,
        split="validation_selection_validation",
    )
    # The outer-test feature view is checked while the adapter is exactly
    # identity.  This uses no outer-test labels and guards the custom input
    # boundary before any selected prediction is frozen.
    parity_test, parity_test_reference, public_parity_test = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=task.x_test,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=-2,
        split="validation_selection_test",
    )
    identity_validation = _normal_prediction(
        bundle=bundle,
        query_x=validation_x,
        context_indices=bundle.support_indices,
        adapters=None,
        device=device,
    )
    identity_validation_error = deployment_error(
        task.problem_type, validation_y, identity_validation, n_classes=task.n_classes
    )
    identity_state = {} if adapters is None else _cpu_state_dict(adapters)
    checkpoint_records: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "identity",
            "validation_error": float(identity_validation_error),
            "validation_relative_improvement_vs_identity": 0.0,
            "task_objective": None,
            "identity_penalty": 0.0,
            "total_objective": None,
            "learning_rates": [],
            "elapsed_seconds": float(time.perf_counter() - started),
            "adapter_diagnostics": _adapter_checkpoint_diagnostics(adapters),
        }
    ]
    training_context_sizes: list[int] = []
    first_task_objective = final_task_objective = float("nan")
    first_identity_penalty = final_identity_penalty = float("nan")
    first_total_objective = final_total_objective = float("nan")
    selected_candidate_state: dict[str, Any] | None = None
    selected_candidate_step: int | None = None
    selected_candidate_error = float("inf")
    executed_steps = 0
    encountered_nonfinite_objective = False
    no_trainable_numerical_features = adapters is None

    if adapters is not None:
        optimizer = _optimizer(adapters, config)
        scheduler = _cosine_scheduler(
            optimizer,
            total_steps=scheduler_horizon,
            min_lr_ratio=float(config["cosine_min_lr_ratio"]),
        )
        episode_rng = np.random.default_rng(
            _seed(int(config["random_state"]), task.task_id, -2, 203)
        )
        for step in range(1, max_steps + 1):
            configured_context_rows = config.get("train_context_rows")
            context_row_limit = (
                max(1, bundle.fit_labels.size - int(config["query_batch_rows"]))
                if configured_context_rows is None
                else int(configured_context_rows)
            )
            context_rows, query_rows = sample_episode_indices(
                bundle.fit_labels,
                problem_type=task.problem_type,
                context_rows=context_row_limit,
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
            training_context_sizes.append(int(context_rows.size))
            optimizer.zero_grad(set_to_none=True)
            task_objective, identity_penalty, total_objective = _adapter_training_objective(
                bundle=bundle,
                adapters=adapters,
                context_indices=context_rows,
                query_indices=query_rows,
                config=config,
                device=device,
            )
            if not torch.isfinite(total_objective):
                encountered_nonfinite_objective = True
                _emit(
                    progress,
                    event="validation_selection_nonfinite_objective",
                    task_id=task.task_id,
                    step=step,
                )
                del task_objective, identity_penalty, total_objective
                break
            if step == 1:
                first_task_objective = float(task_objective.detach())
                first_identity_penalty = float(identity_penalty.detach())
                first_total_objective = float(total_objective.detach())
            total_objective.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
            optimizer.step()
            scheduler.step()
            final_task_objective = float(task_objective.detach())
            final_identity_penalty = float(identity_penalty.detach())
            final_total_objective = float(total_objective.detach())
            executed_steps = step
            if step % checkpoint_interval == 0 or step == max_steps:
                candidate = _normal_prediction(
                    bundle=bundle,
                    query_x=validation_x,
                    context_indices=bundle.support_indices,
                    adapters=adapters,
                    device=device,
                )
                candidate_error = _candidate_deployment_error(
                    task.problem_type, validation_y, candidate, n_classes=task.n_classes
                )
                relative_improvement = (
                    None
                    if not np.isfinite(candidate_error)
                    else float(
                        (identity_validation_error - candidate_error)
                        / max(abs(identity_validation_error), 1e-12)
                    )
                )
                if candidate_error < selected_candidate_error:
                    selected_candidate_error = candidate_error
                    selected_candidate_state = _cpu_state_dict(adapters)
                    selected_candidate_step = int(step)
                diagnostics = _adapter_checkpoint_diagnostics(adapters)
                record = {
                    "step": int(step),
                    "kind": "spline",
                    "validation_error": float(candidate_error),
                    "validation_relative_improvement_vs_identity": relative_improvement,
                    "task_objective": final_task_objective,
                    "identity_penalty": final_identity_penalty,
                    "total_objective": final_total_objective,
                    "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "adapter_diagnostics": diagnostics,
                }
                checkpoint_records.append(record)
                _emit(
                    progress,
                    event="validation_selection_checkpoint",
                    task_id=task.task_id,
                    step=int(step),
                    validation_error=float(candidate_error),
                    identity_validation_error=float(identity_validation_error),
                    validation_relative_improvement=relative_improvement,
                    elapsed_seconds=record["elapsed_seconds"],
                    mean_grid_deformation=diagnostics["mean_grid_deformation"],
                )
                del candidate
            del task_objective, identity_penalty, total_objective
        del scheduler, optimizer

    if selected_candidate_state is None:
        use_adapted = False
        selected_step = 0
        selected_validation_error = float(identity_validation_error)
    else:
        decision = choose_identity_guard(
            identity_error=float(identity_validation_error),
            adapted_error=float(selected_candidate_error),
            required_relative_improvement=float(config["selection_relative_improvement"]),
        )
        use_adapted = bool(decision.use_adapted)
        selected_step = int(selected_candidate_step) if use_adapted and selected_candidate_step is not None else 0
        selected_validation_error = (
            float(selected_candidate_error) if use_adapted else float(identity_validation_error)
        )
    selected_state = selected_candidate_state if use_adapted else identity_state
    if adapters is not None:
        adapters.load_state_dict(selected_state, strict=True)
    selected_validation = (
        identity_validation.copy()
        if not use_adapted or adapters is None
        else _normal_prediction(
            bundle=bundle,
            query_x=validation_x,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
    )
    # Outer-test predictions are deliberately generated only after the
    # validation selection and adapter state are frozen.
    identity_test = _normal_prediction(
        bundle=bundle,
        query_x=task.x_test,
        context_indices=bundle.support_indices,
        adapters=None,
        device=device,
    )
    selected_test = (
        identity_test.copy()
        if not use_adapted or adapters is None
        else _normal_prediction(
            bundle=bundle,
            query_x=task.x_test,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
    )
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "fit_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "support_rows": int(bundle.support_indices.size),
        "n_features": int(task.x_train.shape[1]),
        "n_numerical_features": int(bundle.numerical_indices.size),
        "no_trainable_numerical_features": bool(no_trainable_numerical_features),
        "normal_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "context_policy": (
            "all_inner_training_rows"
            if bundle.support_indices.size == bundle.fit_labels.size
            else "stratified_context_cap"
        ),
        "identity_parity_max_abs_validation": float(parity_validation),
        "identity_parity_max_abs_test": float(parity_test),
        "identity_parity_reference_validation": parity_validation_reference,
        "identity_parity_reference_test": parity_test_reference,
        "public_path_input_parity_checked_validation": bool(public_parity_validation),
        "public_path_input_parity_checked_test": bool(public_parity_test),
        "public_path_input_parity_passed": (
            True if public_parity_validation and public_parity_test else None
        ),
        "fresh_spline_view_identity_passed": bool(
            adapters is None or (parity_validation == 0.0 and parity_test == 0.0)
        ),
        "adapter_seed": int(adapter_seed),
        "adapter_steps_requested": max_steps,
        "scheduler": {
            "kind": "cosine",
            "horizon_steps": scheduler_horizon,
            "min_lr_ratio": float(config["cosine_min_lr_ratio"]),
        },
        "identity_regularization": float(config.get("identity_regularization", 0.0)),
        "selection_relative_improvement": float(config["selection_relative_improvement"]),
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_records": checkpoint_records,
        "best_spline_checkpoint_step": selected_candidate_step,
        "best_spline_validation_error": (
            None if not np.isfinite(selected_candidate_error) else float(selected_candidate_error)
        ),
        "selected_step": int(selected_step),
        "selected_use_adapted": bool(use_adapted),
        "selected_validation_error": float(selected_validation_error),
        "identity_validation_error": float(identity_validation_error),
        "selected_validation_relative_improvement": float(
            (identity_validation_error - selected_validation_error)
            / max(abs(identity_validation_error), 1e-12)
        ),
        "adapter_steps_executed": int(executed_steps),
        "fixed_horizon_completed": bool(adapters is None or executed_steps == max_steps),
        "encountered_nonfinite_objective": bool(encountered_nonfinite_objective),
        "adapter_first_task_objective": first_task_objective,
        "adapter_final_task_objective": final_task_objective,
        "adapter_first_identity_penalty": first_identity_penalty,
        "adapter_final_identity_penalty": final_identity_penalty,
        "adapter_first_total_objective": first_total_objective,
        "adapter_final_total_objective": final_total_objective,
        "adapter_configured_train_context_rows": config.get("train_context_rows"),
        "adapter_train_context_policy": (
            "all_non_query_inner_training_rows"
            if config.get("train_context_rows") is None
            else "sampled_context_cap"
        ),
        "adapter_observed_train_context_rows_min": (
            None if not training_context_sizes else int(min(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_max": (
            None if not training_context_sizes else int(max(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_mean": (
            None if not training_context_sizes else float(np.mean(training_context_sizes))
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
    }
    selected_state_cpu = {} if adapters is None else _cpu_state_dict(adapters)
    del adapters, bundle
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "identity_validation": identity_validation,
        "selected_validation": selected_validation,
        "identity_test": identity_test,
        "selected_test": selected_test,
    }, metadata, selected_state_cpu


def _fit_selected_full_context_refit_standard(
    *,
    task: OpenMLTaskData,
    selected_steps: int,
    config: dict[str, Any],
    protocol_seed: int,
    backbone: TabICL,
    device: torch.device,
    progress: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Refit a validation-selected schedule on all outer-training rows.

    ``selected_steps`` may be zero, which explicitly carries the validation
    choice of identity through to the full-context comparison.  When positive,
    the cosine horizon remains the original maximum schedule length so the
    refit receives the same LR prefix as its validation-selected checkpoint.
    """

    if selected_steps < 0 or selected_steps > int(config["adapter_steps"]):
        raise ValueError("selected_steps must lie within the declared adapter schedule")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full_indices = np.arange(len(task.y_train), dtype=int)
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=full_indices,
        config=config,
        protocol_seed=protocol_seed,
        bag=-2,
        backbone=backbone,
        device=device,
    )
    adapter_seed = _seed(int(config["random_state"]), task.task_id, -2, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_test, parity_reference_test, public_parity_test = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=task.x_test,
        device=device,
        progress=progress,
        task_id=task.task_id,
        bag=-2,
        split="validation_selected_full_refit_test",
    )
    identity_test = _normal_prediction(
        bundle=bundle,
        query_x=task.x_test,
        context_indices=bundle.support_indices,
        adapters=None,
        device=device,
    )
    training_context_sizes: list[int] = []
    first_task_objective = final_task_objective = float("nan")
    first_identity_penalty = final_identity_penalty = float("nan")
    first_total_objective = final_total_objective = float("nan")
    executed_steps = 0
    encountered_nonfinite_objective = False
    no_trainable_numerical_features = adapters is None
    if adapters is None or selected_steps == 0:
        selected_test = identity_test.copy()
    else:
        optimizer = _optimizer(adapters, config)
        scheduler = _cosine_scheduler(
            optimizer,
            total_steps=int(config["cosine_schedule_steps"]),
            min_lr_ratio=float(config["cosine_min_lr_ratio"]),
        )
        episode_rng = np.random.default_rng(
            _seed(int(config["random_state"]), task.task_id, -2, 203)
        )
        for step in range(1, selected_steps + 1):
            configured_context_rows = config.get("train_context_rows")
            context_row_limit = (
                max(1, bundle.fit_labels.size - int(config["query_batch_rows"]))
                if configured_context_rows is None
                else int(configured_context_rows)
            )
            context_rows, query_rows = sample_episode_indices(
                bundle.fit_labels,
                problem_type=task.problem_type,
                context_rows=context_row_limit,
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
            training_context_sizes.append(int(context_rows.size))
            optimizer.zero_grad(set_to_none=True)
            task_objective, identity_penalty, total_objective = _adapter_training_objective(
                bundle=bundle,
                adapters=adapters,
                context_indices=context_rows,
                query_indices=query_rows,
                config=config,
                device=device,
            )
            if not torch.isfinite(total_objective):
                encountered_nonfinite_objective = True
                _emit(
                    progress,
                    event="validation_selected_refit_nonfinite_objective",
                    task_id=task.task_id,
                    step=step,
                )
                del task_objective, identity_penalty, total_objective
                break
            if step == 1:
                first_task_objective = float(task_objective.detach())
                first_identity_penalty = float(identity_penalty.detach())
                first_total_objective = float(total_objective.detach())
            total_objective.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
            optimizer.step()
            scheduler.step()
            final_task_objective = float(task_objective.detach())
            final_identity_penalty = float(identity_penalty.detach())
            final_total_objective = float(total_objective.detach())
            executed_steps = step
            del task_objective, identity_penalty, total_objective
        del scheduler, optimizer
        if executed_steps != selected_steps:
            # A partially fitted model is not the validation-selected duration
            # and must never be silently reported as that endpoint. Retain an
            # explicitly invalid prediction so the paired summary records a
            # failed adaptation rather than treating it as a shorter refit.
            selected_test = np.full_like(identity_test, np.nan)
        else:
            selected_test = _normal_prediction(
                bundle=bundle,
                query_x=task.x_test,
                context_indices=bundle.support_indices,
                adapters=adapters,
                device=device,
            )
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "fit_rows": int(full_indices.size),
        "support_rows": int(bundle.support_indices.size),
        "n_features": int(task.x_train.shape[1]),
        "n_numerical_features": int(bundle.numerical_indices.size),
        "no_trainable_numerical_features": bool(no_trainable_numerical_features),
        "normal_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "context_policy": (
            "all_outer_training_rows"
            if bundle.support_indices.size == bundle.fit_labels.size
            else "stratified_context_cap"
        ),
        "identity_parity_max_abs_test": float(parity_test),
        "identity_parity_reference_test": parity_reference_test,
        "public_path_input_parity_checked_test": bool(public_parity_test),
        "public_path_input_parity_passed": True if public_parity_test else None,
        "fresh_spline_view_identity_passed": bool(adapters is None or parity_test == 0.0),
        "adapter_seed": int(adapter_seed),
        "selected_steps_requested": int(selected_steps),
        "adapter_steps_executed": int(executed_steps),
        "selected_duration_completed": bool(executed_steps == selected_steps),
        "encountered_nonfinite_objective": bool(encountered_nonfinite_objective),
        "selected_prediction_policy": (
            "identity_at_step_zero"
            if selected_steps == 0
            else "fresh_refit_at_selected_duration"
            if executed_steps == selected_steps
            else "invalid_prediction_after_incomplete_refit"
        ),
        "scheduler": {
            "kind": "cosine",
            "horizon_steps": int(config["cosine_schedule_steps"]),
            "min_lr_ratio": float(config["cosine_min_lr_ratio"]),
        },
        "identity_regularization": float(config.get("identity_regularization", 0.0)),
        "adapter_first_task_objective": first_task_objective,
        "adapter_final_task_objective": final_task_objective,
        "adapter_first_identity_penalty": first_identity_penalty,
        "adapter_final_identity_penalty": final_identity_penalty,
        "adapter_first_total_objective": first_total_objective,
        "adapter_final_total_objective": final_total_objective,
        "adapter_configured_train_context_rows": config.get("train_context_rows"),
        "adapter_train_context_policy": (
            "all_non_query_outer_training_rows"
            if config.get("train_context_rows") is None
            else "sampled_context_cap"
        ),
        "adapter_observed_train_context_rows_min": (
            None if not training_context_sizes else int(min(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_max": (
            None if not training_context_sizes else int(max(training_context_sizes))
        ),
        "adapter_observed_train_context_rows_mean": (
            None if not training_context_sizes else float(np.mean(training_context_sizes))
        ),
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
    }
    del adapters, bundle
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return identity_test, selected_test, metadata


def run_task_validation_selected_full_refit_standard(
    *,
    task: OpenMLTaskData,
    config_labels: list[str],
    configs: list[dict[str, Any]],
    validation_fraction: float,
    validation_seed: int,
    output_dir: Path,
    protocol_seed: int,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
    resume: bool,
    run_fingerprint_hash: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Run validation selection and matched full-context refit for each arm.

    Each arm freezes two outer-test prediction pairs: the selected checkpoint
    from the inner-train model, and a fresh all-row refit trained for the
    validation-selected duration.  Test labels are deliberately absent from
    this function; summaries score only the frozen arrays afterward.
    """

    if len(config_labels) != len(configs) or not config_labels:
        raise ValueError("validation-selected refit requires matching non-empty labels and configs")
    if len(set(config_labels)) != len(config_labels):
        raise ValueError("validation-selected refit config labels must be unique")
    task_split_seed = _seed(int(validation_seed), task.task_id, -2, 301)
    fit_indices, validation_indices = _single_validation_split(
        task,
        validation_fraction=validation_fraction,
        seed=task_split_seed,
    )
    split_hash = _split_sha256(fit_indices, validation_indices)
    variants: dict[str, dict[str, Any]] = {}
    for label, config in zip(config_labels, configs, strict=True):
        artifact_dir = _validation_selected_refit_dir(output_dir, task, label)
        summary_path = artifact_dir / "summary.json"
        predictions_path = artifact_dir / "predictions.npz"
        split_path = artifact_dir / "inner_split.npz"
        selected_state_path = artifact_dir / "selected_adapter_state.pt"
        if resume and all(
            path.is_file()
            for path in (summary_path, predictions_path, split_path, selected_state_path)
        ):
            result = _try_load_validation_selected_refit_artifact(
                task=task,
                label=label,
                config=config,
                validation_fraction=validation_fraction,
                validation_seed=validation_seed,
                task_split_seed=task_split_seed,
                fit_indices=fit_indices,
                validation_indices=validation_indices,
                split_hash=split_hash,
                summary_path=summary_path,
                predictions_path=predictions_path,
                split_path=split_path,
                selected_state_path=selected_state_path,
                run_fingerprint_hash=run_fingerprint_hash,
            )
            if result is not None:
                _emit(progress, event="validation_selected_refit_reused", task_id=task.task_id, config_label=label)
                variants[label] = result
                continue
            _emit(
                progress,
                event="validation_selected_refit_rebuilding_incomplete",
                task_id=task.task_id,
                config_label=label,
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            split_path,
            inner_train_indices=fit_indices,
            validation_indices=validation_indices,
        )
        _emit(
            progress,
            event="validation_selected_refit_started",
            task_id=task.task_id,
            config_label=label,
            inner_train_rows=int(fit_indices.size),
            validation_rows=int(validation_indices.size),
        )
        selection_backbone, selection_checkpoint_path, selection_checkpoint_metadata = load_frozen_backbone(
            problem_type=task.problem_type,
            device=device,
            classifier_checkpoint=classifier_checkpoint,
            regressor_checkpoint=regressor_checkpoint,
        )
        selection_predictions, selection_metadata, selected_state = _fit_validation_selected_checkpoint_standard(
            task=task,
            fit_indices=fit_indices,
            validation_indices=validation_indices,
            config=config,
            protocol_seed=protocol_seed,
            backbone=selection_backbone,
            device=device,
            progress=progress,
        )
        torch.save(
            {
                "selected_step": int(selection_metadata["selected_step"]),
                "selected_use_adapted": bool(selection_metadata["selected_use_adapted"]),
                "adapter_state_dict": selected_state,
            },
            selected_state_path,
        )
        del selection_backbone, selection_checkpoint_path
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        refit_backbone, refit_checkpoint_path, refit_checkpoint_metadata = load_frozen_backbone(
            problem_type=task.problem_type,
            device=device,
            classifier_checkpoint=classifier_checkpoint,
            regressor_checkpoint=regressor_checkpoint,
        )
        full_identity_test, full_selected_test, refit_metadata = _fit_selected_full_context_refit_standard(
            task=task,
            selected_steps=int(selection_metadata["selected_step"]),
            config=config,
            protocol_seed=protocol_seed,
            backbone=refit_backbone,
            device=device,
            progress=progress,
        )
        del refit_backbone, refit_checkpoint_path
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        np.savez_compressed(
            predictions_path,
            inner_identity_validation=selection_predictions["identity_validation"],
            inner_selected_validation=selection_predictions["selected_validation"],
            inner_identity_test=selection_predictions["identity_test"],
            inner_selected_test=selection_predictions["selected_test"],
            full_identity_test=full_identity_test,
            full_selected_test=full_selected_test,
        )
        result = {
            "experiment_type": "validation_selected_checkpoint_then_full_context_refit",
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "dataset_name": task.dataset_name,
            "problem_type": task.problem_type,
            "n_classes": task.n_classes,
            "outer_split_hash": task.outer_split_hash,
            "run_fingerprint_hash": run_fingerprint_hash,
            "config_label": label,
            "config": config,
            "validation_split": {
                "fraction": float(validation_fraction),
                "root_seed": int(validation_seed),
                "task_seed": int(task_split_seed),
                "inner_train_rows": int(fit_indices.size),
                "validation_rows": int(validation_indices.size),
                "sha256": split_hash,
                "artifact": split_path.name,
            },
            "selection_checkpoint": selection_checkpoint_metadata,
            "refit_checkpoint": refit_checkpoint_metadata,
            "selection": selection_metadata,
            "full_context_refit": refit_metadata,
            "selected_adapter_state": selected_state_path.name,
            "selected_adapter_state_sha256": _artifact_sha256(selected_state_path),
            "prediction_artifact_sha256": _artifact_sha256(predictions_path),
            "test_metrics_deferred_to_task_summary": True,
            "outer_test_policy": (
                "Validation labels alone select identity or a spline checkpoint before any outer-test prediction "
                "is frozen. A fresh full-context refit then uses that selected duration and the same cosine "
                "learning-rate prefix. Outer-test labels are read only by the separate summary stage."
            ),
        }
        _json_dump(summary_path, result)
        _emit(
            progress,
            event="validation_selected_refit_completed",
            task_id=task.task_id,
            config_label=label,
            selected_step=int(selection_metadata["selected_step"]),
            selected_use_adapted=bool(selection_metadata["selected_use_adapted"]),
            selection_seconds=float(selection_metadata["train_seconds"]),
            refit_seconds=float(refit_metadata["train_seconds"]),
        )
        variants[label] = result
    return {
        "experiment_type": "validation_selected_checkpoint_then_full_context_refit",
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "run_fingerprint_hash": run_fingerprint_hash,
        "validation_split": {
            "fraction": float(validation_fraction),
            "root_seed": int(validation_seed),
            "task_seed": int(task_split_seed),
            "inner_train_rows": int(fit_indices.size),
            "validation_rows": int(validation_indices.size),
            "sha256": split_hash,
        },
        "variants": variants,
    }


def _selected_pair_summary(
    *,
    task: OpenMLTaskData,
    identity_prediction: np.ndarray,
    selected_prediction: np.ndarray,
) -> dict[str, Any]:
    """Score one already-frozen selected prediction against its matched identity."""

    identity_metrics = _metric_bundle(task.problem_type, task.y_test, identity_prediction, task.n_classes)
    selected_metrics = _candidate_metric_bundle(
        task.problem_type, task.y_test, selected_prediction, task.n_classes
    )
    selected_error = float(selected_metrics["deployment_error"])
    identity_error = float(identity_metrics["deployment_error"])
    relative_improvement = (
        None
        if not np.isfinite(selected_error)
        else float((identity_error - selected_error) / max(abs(identity_error), 1e-12))
    )
    return {
        "identity": identity_metrics,
        "selected": selected_metrics,
        "selected_deployment_relative_improvement_vs_identity": relative_improvement,
    }


def summarize_validation_selected_full_refit_task(
    *,
    task: OpenMLTaskData,
    output_dir: Path,
    task_result: dict[str, Any],
) -> dict[str, Any]:
    """Score both frozen test-prediction arms after validation-only selection."""

    if task_result.get("experiment_type") != "validation_selected_checkpoint_then_full_context_refit":
        raise RuntimeError("refusing to score a non-validation/refit task result")
    variants: dict[str, Any] = {}
    for label, result in task_result["variants"].items():
        artifact_dir = _validation_selected_refit_dir(output_dir, task, label)
        prediction_path = artifact_dir / "predictions.npz"
        inner = _selected_pair_summary(
            task=task,
            identity_prediction=_load_prediction(prediction_path, "inner_identity_test"),
            selected_prediction=_load_prediction(prediction_path, "inner_selected_test"),
        )
        full_refit = _selected_pair_summary(
            task=task,
            identity_prediction=_load_prediction(prediction_path, "full_identity_test"),
            selected_prediction=_load_prediction(prediction_path, "full_selected_test"),
        )
        variants[label] = {
            "config": result["config"],
            "validation_selection": result["selection"],
            "inner_train_selected_checkpoint": inner,
            "full_context_refit_selected_duration": full_refit,
            "full_context_refit": result["full_context_refit"],
        }
    summary = {
        "report_schema_version": 1,
        "experiment_type": "validation_selected_checkpoint_then_full_context_refit",
        "outer_test_scored_after_validation_selection_and_refit_predictions_frozen": True,
        "outer_test_used_for_selection": False,
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "validation_split": task_result["validation_split"],
        "variants": variants,
        "comparison_note": (
            "Each selected checkpoint is compared only with the identity prediction from the same fitted context: "
            "inner-train context for the selected-checkpoint arm and all outer-training rows for the refit arm."
        ),
    }
    path = output_dir / "validation_selected_refit_task_summaries" / (
        f"task_{task.task_id}_{_safe_name(task.dataset_name)}.json"
    )
    _json_dump(path, summary)
    return summary


def summarize_validation_selected_full_refit_experiment(
    *,
    task_summaries: list[dict[str, Any]],
    output_dir: Path,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    skipped_tasks: list[dict[str, Any]] | None = None,
    task_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the two paired endpoints for scheduler and regularised arms."""

    if not task_summaries:
        raise ValueError("cannot summarise an empty validation/refit experiment")
    skipped_tasks = [] if skipped_tasks is None else skipped_tasks
    labels = list(task_summaries[0]["variants"])
    if not labels or any(list(item["variants"]) != labels for item in task_summaries):
        raise ValueError("all validation/refit task summaries must contain the same ordered variants")
    rows: list[dict[str, Any]] = []
    paired_results: dict[str, dict[str, Any]] = {}
    selection_behavior: dict[str, dict[str, Any]] = {}
    arms = ("inner_train_selected_checkpoint", "full_context_refit_selected_duration")
    for label_offset, label in enumerate(labels):
        selected_steps = np.asarray(
            [int(item["variants"][label]["validation_selection"]["selected_step"]) for item in task_summaries],
            dtype=int,
        )
        selected_adapted = np.asarray(
            [
                bool(item["variants"][label]["validation_selection"]["selected_use_adapted"])
                for item in task_summaries
            ],
            dtype=bool,
        )
        selection_behavior[label] = {
            "n_identity_selected": int(np.sum(~selected_adapted)),
            "n_spline_selected": int(np.sum(selected_adapted)),
            "spline_selected_fraction": float(np.mean(selected_adapted)),
            "selected_step_min": int(np.min(selected_steps)),
            "selected_step_median": float(np.median(selected_steps)),
            "selected_step_max": int(np.max(selected_steps)),
        }
        paired_results[label] = {}
        for arm_offset, arm in enumerate(arms):
            reference = np.asarray(
                [float(item["variants"][label][arm]["identity"]["benchmark_error"]) for item in task_summaries],
                dtype=float,
            )
            candidate = np.asarray(
                [float(item["variants"][label][arm]["selected"]["benchmark_error"]) for item in task_summaries],
                dtype=float,
            )
            problem_types = np.asarray([str(item["problem_type"]) for item in task_summaries], dtype=object)
            paired_results[label][arm] = _paired_comparison_summary(
                reference=reference,
                candidate=candidate,
                problem_types=problem_types,
                bootstrap_rounds=bootstrap_rounds,
                bootstrap_seed=bootstrap_seed + 10 * label_offset + arm_offset,
                reference_label=(
                    "inner_train_standard_tabicl_identity"
                    if arm == "inner_train_selected_checkpoint"
                    else "full_outer_training_standard_tabicl_identity"
                ),
                candidate_label=f"{label}_{arm}",
            )
        for item in task_summaries:
            variant = item["variants"][label]
            selection = variant["validation_selection"]
            for arm in arms:
                endpoint = variant[arm]
                rows.append(
                    {
                        "task_id": item["task_id"],
                        "dataset_id": item["dataset_id"],
                        "dataset_name": item["dataset_name"],
                        "problem_type": item["problem_type"],
                        "outer_split_hash": item["outer_split_hash"],
                        "variant": label,
                        "arm": arm,
                        "identity_regularization": variant["config"].get("identity_regularization", 0.0),
                        "selected_step": selection["selected_step"],
                        "selected_use_adapted": selection["selected_use_adapted"],
                        "validation_identity_error": selection["identity_validation_error"],
                        "validation_selected_error": selection["selected_validation_error"],
                        "validation_selected_relative_improvement": selection[
                            "selected_validation_relative_improvement"
                        ],
                        "identity_benchmark_error": endpoint["identity"]["benchmark_error"],
                        "identity_deployment_error": endpoint["identity"]["deployment_error"],
                        "selected_benchmark_error": endpoint["selected"]["benchmark_error"],
                        "selected_deployment_error": endpoint["selected"]["deployment_error"],
                        "selected_prediction_valid": endpoint["selected"].get("prediction_valid", True),
                        "selected_deployment_relative_improvement_vs_identity": endpoint[
                            "selected_deployment_relative_improvement_vs_identity"
                        ],
                        "full_refit_selected_duration_completed": variant["full_context_refit"].get(
                            "selected_duration_completed"
                        ),
                        "full_refit_encountered_nonfinite_objective": variant["full_context_refit"].get(
                            "encountered_nonfinite_objective"
                        ),
                    }
                )
    csv_path = output_dir / "validation_selected_refit_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment_type": "validation_selected_checkpoint_then_full_context_refit",
        "n_tasks": len(task_summaries),
        "n_skipped_tasks": len(skipped_tasks),
        "task_eligibility": task_eligibility,
        "skipped_tasks": skipped_tasks,
        "variant_paired_results": paired_results,
        "selection_behavior": selection_behavior,
        "results_csv": str(csv_path),
        "selection_rule": (
            "The inner-train validation metric chooses identity or the best observed spline checkpoint, subject to "
            "the frozen relative-improvement margin. The full refit uses that duration and the same cosine prefix."
        ),
        "metric_note": "Binary: 1-AUC; multiclass: log loss; regression: RMSE for paired comparison.",
    }
    _json_dump(output_dir / "summary.json", summary)
    return summary


def run_task_unconditional_full_context_refit_standard(
    *,
    task: OpenMLTaskData,
    config_labels: list[str],
    configs: list[dict[str, Any]],
    output_dir: Path,
    protocol_seed: int,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
    resume: bool,
    run_fingerprint_hash: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Train the predeclared DirectSpline on every outer-training row.

    This is the raw headroom experiment. There is no OOF configuration
    selection, checkpoint selection, or deployment guard. The sole declared
    configuration and its full ``adapter_steps`` budget are frozen in the run
    manifest before any outer-test label is scored.
    """

    if len(config_labels) != 1 or len(configs) != 1:
        raise ValueError(
            "unconditional full-context refit requires exactly one predeclared configuration"
        )
    selected_label = config_labels[0]
    selected_config = configs[0]
    refit_steps = int(selected_config["adapter_steps"])
    refit_dir = _full_context_refit_dir(output_dir, task, selected_label)
    refit_summary_path = refit_dir / "summary.json"
    refit_predictions_path = refit_dir / "predictions.npz"
    if resume and refit_summary_path.is_file() and refit_predictions_path.is_file():
        result = _json_load(refit_summary_path)
        if result.get("run_fingerprint_hash") != run_fingerprint_hash:
            raise RuntimeError(
                f"refusing to resume {refit_dir}: full-context refit artifacts use another immutable fingerprint"
            )
        if result.get("selection", {}).get("mode") != "predeclared_fixed_schedule_no_guard":
            raise RuntimeError(f"refusing to reuse an OOF-selected refit as an unconditional refit: {refit_dir}")
        _emit(progress, event="full_refit_reused", task_id=task.task_id, config_label=selected_label)
        return result

    refit_dir.mkdir(parents=True, exist_ok=True)
    _emit(
        progress,
        event="full_refit_started",
        task_id=task.task_id,
        config_label=selected_label,
        refit_steps=refit_steps,
        selection_mode="predeclared_fixed_schedule_no_guard",
    )
    backbone, checkpoint_path, checkpoint_metadata = load_frozen_backbone(
        problem_type=task.problem_type,
        device=device,
        classifier_checkpoint=classifier_checkpoint,
        regressor_checkpoint=regressor_checkpoint,
    )
    identity_test, adapted_test, metadata = _fit_full_context_refit_standard(
        task=task,
        config=selected_config,
        refit_steps=refit_steps,
        protocol_seed=protocol_seed,
        backbone=backbone,
        device=device,
        progress=progress,
    )
    # Preserve the historical key only as an explicit no-guard alias. The
    # unconditional spline is deployed on every task regardless of any old
    # bag validation result.
    np.savez_compressed(
        refit_predictions_path,
        identity_test=identity_test,
        adapted_test=adapted_test,
        guarded_test=adapted_test,
    )
    result = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "run_fingerprint_hash": run_fingerprint_hash,
        "selected_config_label": selected_label,
        "selected_config": selected_config,
        "selection": {
            "mode": "predeclared_fixed_schedule_no_guard",
            "config_selection": "sole configuration frozen before the run",
            "refit_steps": {
                "source": "predeclared adapter_steps",
                "value": refit_steps,
            },
            "deployment_guard_applied": False,
            "oof_used_for_training_or_deployment": False,
        },
        "checkpoint": checkpoint_metadata,
        "identity_definition": (
            "The same all-outer-training-row normal TabICLv2 estimator and context used by the adapted path, "
            "with a fresh DirectSpline audited as exact identity before optimisation."
        ),
        "refit": metadata,
        "test_metrics_deferred_to_task_summary": True,
        "outer_test_policy": (
            "Test labels were not used for configuration selection, schedule selection, spline optimisation, "
            "or deployment. The predeclared spline is evaluated on every task."
        ),
    }
    _json_dump(refit_summary_path, result)
    _emit(
        progress,
        event="full_refit_completed",
        task_id=task.task_id,
        config_label=selected_label,
        refit_steps=refit_steps,
        executed_steps=int(metadata["adapter_steps_executed"]),
        guard_applied=False,
        train_seconds=float(metadata["train_seconds"]),
        peak_allocated_gib=float(metadata["peak_allocated_gib"]),
    )
    del backbone, checkpoint_path
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_task_full_context_refit_checkpoint_audit_standard(
    *,
    task: OpenMLTaskData,
    config_labels: list[str],
    configs: list[dict[str, Any]],
    checkpoint_steps: tuple[int, ...],
    output_dir: Path,
    protocol_seed: int,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
    resume: bool,
    run_fingerprint_hash: str,
    progress: Any = None,
) -> dict[str, Any]:
    """Freeze a development-only full-context DirectSpline checkpoint curve.

    The resulting outer-test curve is diagnostic evidence only.  It cannot
    select a deployment checkpoint, a regulariser, or an identity guard.
    """

    if len(config_labels) != 1 or len(configs) != 1:
        raise ValueError("full-context checkpoint audit requires exactly one predeclared configuration")
    if not checkpoint_steps or checkpoint_steps != tuple(sorted(set(checkpoint_steps))):
        raise ValueError("checkpoint_steps must be a non-empty sorted unique sequence")
    if checkpoint_steps[0] != 0 or any(step < 0 for step in checkpoint_steps):
        raise ValueError("checkpoint_steps must start at zero and be non-negative")
    selected_label = config_labels[0]
    selected_config = configs[0]
    audit_dir = _full_context_checkpoint_audit_dir(output_dir, task, selected_label)
    audit_summary_path = audit_dir / "summary.json"
    audit_predictions_path = audit_dir / "predictions.npz"
    if resume and audit_summary_path.is_file() and audit_predictions_path.is_file():
        result = _json_load(audit_summary_path)
        if result.get("run_fingerprint_hash") != run_fingerprint_hash:
            raise RuntimeError(
                f"refusing to resume {audit_dir}: checkpoint-audit artifacts use another immutable fingerprint"
            )
        if result.get("audit_type") != "development_only_full_context_checkpoint_curve":
            raise RuntimeError(f"refusing to reuse a non-audit refit artifact: {audit_dir}")
        if result.get("checkpoint_steps") != list(checkpoint_steps):
            raise RuntimeError(f"refusing to reuse another checkpoint schedule: {audit_dir}")
        _emit(progress, event="full_refit_checkpoint_audit_reused", task_id=task.task_id)
        return result

    audit_dir.mkdir(parents=True, exist_ok=True)
    _emit(
        progress,
        event="full_refit_checkpoint_audit_started",
        task_id=task.task_id,
        config_label=selected_label,
        checkpoint_steps=list(checkpoint_steps),
    )
    backbone, checkpoint_path, checkpoint_metadata = load_frozen_backbone(
        problem_type=task.problem_type,
        device=device,
        classifier_checkpoint=classifier_checkpoint,
        regressor_checkpoint=regressor_checkpoint,
    )
    identity_test, checkpoint_predictions, metadata = _fit_full_context_refit_checkpoint_audit_standard(
        task=task,
        config=selected_config,
        checkpoint_steps=checkpoint_steps,
        protocol_seed=protocol_seed,
        backbone=backbone,
        device=device,
        checkpoint_state_dir=audit_dir / "checkpoint_states",
        progress=progress,
    )
    prediction_payload: dict[str, np.ndarray] = {"identity_test": identity_test}
    for step, prediction in checkpoint_predictions.items():
        prediction_payload[f"adapted_test_step_{step:06d}"] = prediction
    np.savez_compressed(audit_predictions_path, **prediction_payload)
    result = {
        "audit_type": "development_only_full_context_checkpoint_curve",
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "run_fingerprint_hash": run_fingerprint_hash,
        "selected_config_label": selected_label,
        "selected_config": selected_config,
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoint_steps_frozen": sorted(checkpoint_predictions),
        "checkpoint_state_directory": "checkpoint_states",
        "checkpoint": checkpoint_metadata,
        "refit": metadata,
        "selection": {
            "mode": "development_only_checkpoint_curve_no_selection",
            "configuration_selection": "sole configuration frozen before the run",
            "checkpoint_selection": "none; every requested checkpoint is retained",
            "deployment_guard_applied": False,
            "oof_used_for_training_or_deployment": False,
        },
        "outer_test_policy": (
            "Outer-test labels are used only after every requested checkpoint prediction has been frozen, "
            "to diagnose the development learning curve. They must not select a checkpoint, configuration, "
            "regularisation strength, or deployment guard."
        ),
    }
    _json_dump(audit_summary_path, result)
    _emit(
        progress,
        event="full_refit_checkpoint_audit_completed",
        task_id=task.task_id,
        checkpoint_steps_frozen=sorted(checkpoint_predictions),
        executed_steps=int(metadata["adapter_steps_executed"]),
        train_seconds=float(metadata["train_seconds"]),
        peak_allocated_gib=float(metadata["peak_allocated_gib"]),
    )
    del backbone, checkpoint_path
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize_full_context_refit_checkpoint_audit_task(
    *,
    task: OpenMLTaskData,
    output_dir: Path,
    audit_result: dict[str, Any],
) -> dict[str, Any]:
    """Score an already-frozen checkpoint curve for development diagnosis."""

    if audit_result.get("audit_type") != "development_only_full_context_checkpoint_curve":
        raise RuntimeError("refusing to score a non-audit full-context artifact as a checkpoint curve")
    selected_label = str(audit_result["selected_config_label"])
    audit_dir = _full_context_checkpoint_audit_dir(output_dir, task, selected_label)
    prediction_path = audit_dir / "predictions.npz"
    identity_prediction = _load_prediction(prediction_path, "identity_test")
    identity_metrics = _metric_bundle(task.problem_type, task.y_test, identity_prediction, task.n_classes)
    checkpoint_metadata = {
        int(item["step"]): item
        for item in audit_result["refit"].get("checkpoint_metadata", [])
    }
    checkpoint_metrics: list[dict[str, Any]] = []
    frozen_steps = set(int(step) for step in audit_result.get("checkpoint_steps_frozen", []))
    for step in (int(value) for value in audit_result["checkpoint_steps"]):
        record = checkpoint_metadata.get(step)
        if step not in frozen_steps:
            checkpoint_metrics.append(
                {
                    "step": step,
                    "prediction_frozen": False,
                    "candidate": {
                        "deployment_error": float("inf"),
                        "benchmark_error": float("inf"),
                        "prediction_valid": False,
                        "invalid_prediction_error": "checkpoint_not_frozen_after_nonfinite_training_objective",
                    },
                    "deployment_relative_improvement_vs_identity": float("-inf"),
                    "metadata": record,
                }
            )
            continue
        prediction = _load_prediction(prediction_path, f"adapted_test_step_{step:06d}")
        candidate_metrics = _candidate_metric_bundle(
            task.problem_type, task.y_test, prediction, task.n_classes
        )
        deployment_error_identity = float(identity_metrics["deployment_error"])
        deployment_error_candidate = float(candidate_metrics["deployment_error"])
        checkpoint_metrics.append(
            {
                "step": step,
                "prediction_frozen": True,
                "candidate": candidate_metrics,
                "deployment_relative_improvement_vs_identity": float(
                    (deployment_error_identity - deployment_error_candidate)
                    / max(abs(deployment_error_identity), 1e-12)
                ),
                "metadata": record,
            }
        )
    result = {
        "report_schema_version": 1,
        "audit_type": "development_only_full_context_checkpoint_curve",
        "outer_test_scored_after_all_checkpoint_predictions_frozen": True,
        "outer_test_used_for_selection": False,
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "pipeline": "standard_ensemble_full_context_checkpoint_audit",
        "full_context_identity": identity_metrics,
        "checkpoint_metrics": checkpoint_metrics,
        "checkpoint_audit": audit_result,
        "comparison_note": (
            "Every checkpoint uses the same fresh all-outer-training-row TabICL estimator and context as "
            "full_context_identity. This outer-test curve is development-only diagnostic evidence; it is not "
            "a permitted source of checkpoint, configuration, regularisation, or deployment selection."
        ),
    }
    path = output_dir / "full_context_checkpoint_audit_task_summaries" / (
        f"task_{task.task_id}_{_safe_name(task.dataset_name)}.json"
    )
    _json_dump(path, result)
    return result


def summarize_full_context_refit_checkpoint_audit_experiment(
    *,
    task_summaries: list[dict[str, Any]],
    output_dir: Path,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    skipped_tasks: list[dict[str, Any]] | None = None,
    task_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the paired development learning curve across completed tasks."""

    if not task_summaries:
        raise ValueError("cannot summarise an empty full-context checkpoint audit")
    skipped_tasks = [] if skipped_tasks is None else skipped_tasks
    schedules = {tuple(item["checkpoint_audit"]["checkpoint_steps"]) for item in task_summaries}
    if len(schedules) != 1:
        raise ValueError("all checkpoint-audit task summaries must share one checkpoint schedule")
    checkpoint_steps = next(iter(schedules))
    problem_types = np.asarray([str(item["problem_type"]) for item in task_summaries], dtype=object)
    identity_benchmark = np.asarray(
        [float(item["full_context_identity"]["benchmark_error"]) for item in task_summaries], dtype=float
    )
    checkpoint_results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for offset, step in enumerate(checkpoint_steps):
        candidates: list[float] = []
        for item in task_summaries:
            metric = next(record for record in item["checkpoint_metrics"] if int(record["step"]) == step)
            candidates.append(float(metric["candidate"]["benchmark_error"]))
            metadata = metric.get("metadata") or {}
            diagnostics = metadata.get("adapter_diagnostics") or {}
            rows.append(
                {
                    "task_id": item["task_id"],
                    "dataset_id": item["dataset_id"],
                    "dataset_name": item["dataset_name"],
                    "problem_type": item["problem_type"],
                    "outer_split_hash": item["outer_split_hash"],
                    "step": step,
                    "prediction_frozen": metric["prediction_frozen"],
                    "training_objective": metadata.get("objective"),
                    "elapsed_seconds": metadata.get("elapsed_seconds"),
                    "identity_benchmark_error": item["full_context_identity"]["benchmark_error"],
                    "identity_deployment_error": item["full_context_identity"]["deployment_error"],
                    "checkpoint_benchmark_error": metric["candidate"]["benchmark_error"],
                    "checkpoint_deployment_error": metric["candidate"]["deployment_error"],
                    "checkpoint_prediction_valid": metric["candidate"].get("prediction_valid", True),
                    "checkpoint_deployment_relative_improvement_vs_identity": metric[
                        "deployment_relative_improvement_vs_identity"
                    ],
                    "mean_grid_deformation": diagnostics.get("mean_grid_deformation"),
                    "mean_gate": diagnostics.get("mean_gate"),
                    "max_gate": diagnostics.get("max_gate"),
                    "mean_abs_location": diagnostics.get("mean_abs_location"),
                    "mean_abs_log_scale": diagnostics.get("mean_abs_log_scale"),
                    "mean_mixing_spectral_norm": diagnostics.get("mean_mixing_spectral_norm"),
                }
            )
        checkpoint_results[str(step)] = _paired_comparison_summary(
            reference=identity_benchmark,
            candidate=np.asarray(candidates, dtype=float),
            problem_types=problem_types,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed + offset,
            reference_label="full_outer_training_standard_tabicl_identity",
            candidate_label=f"full_context_direct_spline_step_{step}",
        )
    csv_path = output_dir / "checkpoint_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "audit_type": "development_only_full_context_checkpoint_curve",
        "n_tasks": len(task_summaries),
        "n_skipped_tasks": len(skipped_tasks),
        "task_eligibility": task_eligibility,
        "skipped_tasks": skipped_tasks,
        "checkpoint_steps": list(checkpoint_steps),
        "checkpoint_paired_results": checkpoint_results,
        "checkpoint_results_csv": str(csv_path),
        "development_only_rule": (
            "The outer-test curve diagnoses full-refit learning dynamics on this development suite. It must not "
            "be used to select a checkpoint, regularisation strength, or deployment procedure; freeze those on "
            "a subsequent independent evaluation bank."
        ),
        "metric_note": "Binary: 1-AUC; multiclass: log loss; regression: RMSE for paired comparison.",
    }
    _json_dump(output_dir / "summary.json", summary)
    return summary


def summarize_full_context_refit_task(
    *,
    task: OpenMLTaskData,
    output_dir: Path,
    refit_result: dict[str, Any],
    oof_source_dir: Path | None = None,
    oof_diagnostic_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Score the frozen unconditional refit, then optionally inspect old OOF evidence."""

    selected_label = str(refit_result["selected_config_label"])
    if oof_source_dir is not None and oof_diagnostic_unavailable_reason is not None:
        raise ValueError("an OOF diagnostic cannot be both available and unavailable")
    if refit_result.get("selection", {}).get("mode") != "predeclared_fixed_schedule_no_guard":
        raise RuntimeError("refusing to report an OOF-selected refit as the unconditional experiment")
    refit_prediction_path = _full_context_refit_dir(output_dir, task, selected_label) / "predictions.npz"
    full_refit_identity = _load_prediction(refit_prediction_path, "identity_test")
    full_refit_raw = _load_prediction(refit_prediction_path, "adapted_test")
    identity_metrics = _metric_bundle(
        task.problem_type, task.y_test, full_refit_identity, task.n_classes
    )
    raw_metrics = _candidate_metric_bundle(
        task.problem_type, task.y_test, full_refit_raw, task.n_classes
    )
    identity_deployment_error = float(identity_metrics["deployment_error"])
    raw_deployment_error = float(raw_metrics["deployment_error"])
    full_refit_relative_improvement = float(
        (identity_deployment_error - raw_deployment_error)
        / max(abs(identity_deployment_error), 1e-12)
    )
    required_improvement = float(refit_result["selected_config"]["guard_relative_improvement"])

    oof_diagnostic: dict[str, Any] | None = None
    bagged_identity_metrics: dict[str, Any] | None = None
    bagged_adapted_metrics: dict[str, Any] | None = None
    bagged_guarded_metrics: dict[str, Any] | None = None
    if oof_source_dir is not None:
        config_dir = _config_dir(oof_source_dir, task, selected_label)
        config_summary = _json_load(config_dir / "config_summary.json")
        config_prediction_path = config_dir / "config_predictions.npz"
        historical_standard_path = _standard_baseline_dir(oof_source_dir, task) / "predictions.npz"
        historical_standard = _load_prediction(historical_standard_path, "prediction")
        bagged_identity = _load_prediction(config_prediction_path, "identity_test")
        bagged_adapted = _load_prediction(config_prediction_path, "adapted_test")
        bagged_guarded = _load_prediction(config_prediction_path, "guarded_test")
        oof_identity_error = float(config_summary["validation"]["identity"]["deployment_error"])
        oof_adapted_error = float(config_summary["validation"]["adapted"]["deployment_error"])
        historical_guard = choose_identity_guard(
            identity_error=oof_identity_error,
            adapted_error=oof_adapted_error,
            required_relative_improvement=required_improvement,
        )
        historical_identity_difference = _difference_summary(
            full_refit_identity, historical_standard
        )
        historical_identity_exact = bool(
            historical_identity_difference.get("shape_match") is True
            and historical_identity_difference.get("nonfinite_differences") == 0
            and float(historical_identity_difference.get("max_abs", float("inf"))) == 0.0
        )
        oof_diagnostic = {
            "role": "posthoc_correlation_only",
            "loaded_after_full_refit_prediction_frozen": True,
            "used_for_configuration_selection": False,
            "used_for_step_selection": False,
            "used_as_deployment_guard": False,
            "validation_identity_deployment_error": oof_identity_error,
            "validation_adapted_deployment_error": oof_adapted_error,
            "validation_relative_improvement": float(historical_guard.relative_improvement),
            "historical_guard_selected_adapted": bool(historical_guard.use_adapted),
            "guard_required_relative_improvement": required_improvement,
            "full_refit_test_deployment_relative_improvement": full_refit_relative_improvement,
            "full_refit_test_improved": bool(raw_deployment_error < identity_deployment_error),
            "full_refit_test_met_historical_guard_threshold": bool(
                full_refit_relative_improvement >= required_improvement
            ),
            "full_refit_identity_vs_historical_standard": historical_identity_difference,
            "historical_standard_is_exact_current_identity": historical_identity_exact,
        }
        bagged_identity_metrics = _metric_bundle(
            task.problem_type, task.y_test, bagged_identity, task.n_classes
        )
        bagged_adapted_metrics = _candidate_metric_bundle(
            task.problem_type, task.y_test, bagged_adapted, task.n_classes
        )
        bagged_guarded_metrics = _candidate_metric_bundle(
            task.problem_type, task.y_test, bagged_guarded, task.n_classes
        )
    result = {
        "report_schema_version": 2,
        "outer_test_scored_after_predeclared_full_refit": True,
        "oof_loaded_only_after_full_refit_prediction_frozen": oof_source_dir is not None,
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "pipeline": "standard_ensemble_unconditional_full_context_refit",
        "full_context_identity": identity_metrics,
        "full_refit_identity": identity_metrics,
        "full_refit_raw": raw_metrics,
        # Kept only for readers of the earlier report schema. It is explicitly
        # an alias, not a validation-gated arm.
        "full_refit_guarded": raw_metrics,
        "reported_arm_aliases": {"full_refit_guarded": "full_refit_raw"},
        "deployment_guard_applied": False,
        "full_refit_test_deployment_relative_improvement": full_refit_relative_improvement,
        "bagged_identity": bagged_identity_metrics,
        "bagged_adapted": bagged_adapted_metrics,
        "bagged_guarded": bagged_guarded_metrics,
        "oof_validation_diagnostic": oof_diagnostic,
        "oof_validation_diagnostic_unavailable_reason": oof_diagnostic_unavailable_reason,
        "full_refit": refit_result,
        "comparison_note": (
            "full_refit_raw uses the same fresh all-outer-training-row estimator and context as "
            "full_context_identity; the only learned difference is DirectSpline. The spline is used on every "
            "task. Any oof_validation_diagnostic or bagged_* value was loaded only after this prediction was "
            "frozen and cannot affect it."
        ),
    }
    path = output_dir / "full_context_refit_task_summaries" / (
        f"task_{task.task_id}_{_safe_name(task.dataset_name)}.json"
    )
    _json_dump(path, result)
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks without adding a SciPy dependency."""

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


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(left) & np.isfinite(right)
    x = left[finite]
    y = right[finite]
    result: dict[str, Any] = {"n": int(x.size), "pearson": None, "spearman": None}
    if x.size < 2:
        return result
    if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        result["pearson"] = float(np.corrcoef(x, y)[0, 1])
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if float(np.std(x_rank)) > 0.0 and float(np.std(y_rank)) > 0.0:
        result["spearman"] = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return result


def summarize_full_context_refit_experiment(
    *,
    task_summaries: list[dict[str, Any]],
    output_dir: Path,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    skipped_tasks: list[dict[str, Any]] | None = None,
    task_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report direct all-row-refit comparisons against ordinary TabICLv2."""

    if not task_summaries:
        raise ValueError("cannot summarise an empty full-context refit experiment")
    skipped_tasks = [] if skipped_tasks is None else skipped_tasks
    reference = np.asarray(
        [float(item["full_context_identity"]["benchmark_error"]) for item in task_summaries],
        dtype=float,
    )
    problem_types = np.asarray([str(item["problem_type"]) for item in task_summaries], dtype=object)
    arms = ("full_refit_raw",)
    paired_results: dict[str, dict[str, Any]] = {}
    for offset, arm in enumerate(arms):
        candidate = np.asarray(
            [float(item[arm]["benchmark_error"]) for item in task_summaries], dtype=float
        )
        paired_results[arm] = _paired_comparison_summary(
            reference=reference,
            candidate=candidate,
            problem_types=problem_types,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed + offset,
            reference_label="full_outer_training_standard_tabicl",
            candidate_label=arm,
        )
    rows: list[dict[str, Any]] = []
    for item in task_summaries:
        row: dict[str, Any] = {
            "task_id": item["task_id"],
            "dataset_id": item["dataset_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "outer_split_hash": item["outer_split_hash"],
            "selected_config_label": item["full_refit"]["selected_config_label"],
            "full_refit_steps": item["full_refit"]["refit"]["adapter_refit_steps_requested"],
            "full_refit_steps_executed": item["full_refit"]["refit"]["adapter_steps_executed"],
            "deployment_guard_applied": False,
            "full_context_identity_benchmark_error": item["full_context_identity"]["benchmark_error"],
            "full_context_identity_deployment_error": item["full_context_identity"]["deployment_error"],
        }
        for arm in arms:
            row[f"{arm}_benchmark_error"] = item[arm]["benchmark_error"]
            row[f"{arm}_deployment_error"] = item[arm]["deployment_error"]
            row[f"{arm}_minus_full_context_identity_benchmark_error"] = (
                item[arm]["benchmark_error"] - item["full_context_identity"]["benchmark_error"]
            )
            row[f"{arm}_relative_error_change_vs_full_context_identity"] = float(
                _relative_error_change(
                    np.asarray([item["full_context_identity"]["benchmark_error"]], dtype=float),
                    np.asarray([item[arm]["benchmark_error"]], dtype=float),
                )[0]
            )
        diagnostic = item.get("oof_validation_diagnostic")
        if isinstance(diagnostic, dict):
            row.update(
                {
                    "oof_validation_relative_improvement": diagnostic["validation_relative_improvement"],
                    "oof_historical_guard_selected_adapted": diagnostic["historical_guard_selected_adapted"],
                    "full_refit_test_deployment_relative_improvement": diagnostic[
                        "full_refit_test_deployment_relative_improvement"
                    ],
                    "full_refit_test_improved": diagnostic["full_refit_test_improved"],
                    "full_refit_test_met_historical_guard_threshold": diagnostic[
                        "full_refit_test_met_historical_guard_threshold"
                    ],
                    "historical_standard_is_exact_current_identity": diagnostic[
                        "historical_standard_is_exact_current_identity"
                    ],
                }
            )
            for arm in ("bagged_identity", "bagged_adapted", "bagged_guarded"):
                metrics = item.get(arm)
                if isinstance(metrics, dict):
                    row[f"{arm}_benchmark_error"] = metrics["benchmark_error"]
                    row[f"{arm}_deployment_error"] = metrics["deployment_error"]
        rows.append(row)
    csv_path = output_dir / "task_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics = [
        item["oof_validation_diagnostic"]
        for item in task_summaries
        if isinstance(item.get("oof_validation_diagnostic"), dict)
    ]
    posthoc_correlation: dict[str, Any] | None = None
    if diagnostics:
        validation_improvement = np.asarray(
            [float(item["validation_relative_improvement"]) for item in diagnostics], dtype=float
        )
        test_improvement = np.asarray(
            [float(item["full_refit_test_deployment_relative_improvement"]) for item in diagnostics], dtype=float
        )
        old_guard = np.asarray(
            [bool(item["historical_guard_selected_adapted"]) for item in diagnostics], dtype=bool
        )
        actual_threshold = np.asarray(
            [bool(item["full_refit_test_met_historical_guard_threshold"]) for item in diagnostics], dtype=bool
        )
        validation_positive = validation_improvement > 0.0
        test_positive = test_improvement > 0.0
        posthoc_correlation = {
            "role": "diagnostic_only_never_used_to_choose_or_guard_the_full_refit",
            "n_tasks": len(diagnostics),
            "relative_improvement_correlation": _finite_correlation(
                validation_improvement, test_improvement
            ),
            "positive_direction_agreement_fraction": float(
                np.mean(validation_positive == test_positive)
            ),
            "historical_guard_vs_full_refit_same_threshold": {
                "agreement_fraction": float(np.mean(old_guard == actual_threshold)),
                "guard_yes_test_yes": int(np.sum(old_guard & actual_threshold)),
                "guard_yes_test_no": int(np.sum(old_guard & ~actual_threshold)),
                "guard_no_test_yes": int(np.sum(~old_guard & actual_threshold)),
                "guard_no_test_no": int(np.sum(~old_guard & ~actual_threshold)),
            },
        }
    summary = {
        "n_tasks": len(task_summaries),
        "n_skipped_tasks": len(skipped_tasks),
        "task_eligibility": task_eligibility,
        "skipped_tasks": skipped_tasks,
        "paired_elo_note": (
            "Paired Elo is computed per completed task against normal TabICLv2 fitted on all outer-training rows. "
            "The full-refit comparisons therefore share the same context size and ordinary preprocessing/ensemble; "
            "they isolate the DirectSpline addition. These are local paired Elo deltas, not Retouche's absolute Elo pool."
        ),
        "metric_note": (
            "Binary: 1-AUC; multiclass: log loss; regression: RMSE for paired comparison. "
            "The optional historical OOF correlation uses the deployment metric (regression MSE)."
        ),
        "full_refit_protocol": {
            "configuration_selection": "none; sole configuration D is frozen before the run",
            "refit_step_selection": "none; full adapter_steps budget is frozen before the run",
            "deployment_guard": "none; DirectSpline is evaluated on every completed task",
            "identity_reference": "same fresh all-row estimator before DirectSpline optimisation",
            "oof_source_role": "optional post-hoc validation/test correlation only",
            "test_label_use": "scoring only after the unconditional spline prediction is frozen",
        },
        "posthoc_oof_correlation": posthoc_correlation,
        "paired_results": paired_results,
        "distinct_paired_result_keys": list(arms),
        "paired_result_aliases": {"full_refit_guarded": "full_refit_raw"},
        "task_results_csv": str(csv_path),
    }
    _json_dump(output_dir / "summary.json", summary)
    return summary
