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
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from sklearn.utils.validation import validate_data

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._experiments.direct_spline_openml import (
    STANDARD_TABICL_CONFIG,
    BagPredictions,
    OpenMLTaskData,
    _bag_splits,
    _config_dir,
    _cpu_state_dict,
    _emit,
    _json_dump,
    _json_load,
    _load_bag,
    _metric_bundle,
    _prediction_shape,
    _save_bag,
    _seed,
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
from tabicl._hyperspline import DirectSplineTransform
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

    def __init__(self, adapters: OrderedDict[str, DirectSplineTransform]) -> None:
        super().__init__()
        self._keys = {method: f"branch_{index}" for index, method in enumerate(adapters)}
        self.adapters = nn.ModuleDict({self._keys[method]: adapter for method, adapter in adapters.items()})

    def for_method(self, method: str) -> DirectSplineTransform:
        return self.adapters[self._keys[method]]


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
    adapters: OrderedDict[str, DirectSplineTransform] = OrderedDict()
    n_numerical = int(bundle.numerical_indices.size)
    # Standard preprocessing already defines the coordinates consumed by
    # TabICL.  DirectSpline's context statistics are therefore deliberately
    # replaced by location=0/scale=1 below.  Feeding tens of thousands of rows
    # into ``summarize_context`` only to discard its quantiles wasted VRAM and
    # caused avoidable OOMs.  A shape-only dummy constructs the identical
    # uniform-knot adapter.
    coordinate_dummy = torch.zeros((1, 1, n_numerical), dtype=torch.float32, device=device)
    identity_probe = torch.linspace(-5.0, 5.0, 17, device=device).view(1, 17, 1).expand(-1, -1, n_numerical)
    for method in bundle.estimator.ensemble_generator_.preprocessors_:
        adapter = DirectSplineTransform(
            coordinate_dummy,
            n_control_points=int(config["n_control_points"]),
            trainable_shape=True,
            trainable_location_scale=bool(config["trainable_location_scale"]),
            knot_placement="uniform",
            control_mode="monotone",
            cross_column_mixing_rank=int(config["cross_column_mixing_rank"]),
            cross_column_mixing_bound=float(config["cross_column_mixing_bound"]),
        ).to(device)
        # The standard preprocessing output is already the coordinate system
        # consumed by the normal estimator.  These values make a fresh spline
        # literally x -> x, not an extra standardisation pass.
        with torch.no_grad():
            adapter.location.zero_()
            adapter.scale.fill_(1.0)
            if not torch.equal(adapter.transform(identity_probe), identity_probe):
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


def _apply_adapter(
    canonical: torch.Tensor,
    *,
    numerical_indices: np.ndarray,
    adapter: DirectSplineTransform | None,
    filtered_feature_mask: np.ndarray | None = None,
) -> torch.Tensor:
    if canonical.ndim != 2:
        raise ValueError("canonical standard-preprocessed features must have shape (rows, features)")
    if adapter is None or numerical_indices.size == 0:
        return canonical
    indices = torch.as_tensor(numerical_indices, dtype=torch.long, device=canonical.device)
    transformed = canonical.clone()
    values = canonical.index_select(-1, indices).unsqueeze(0)
    effective_mixing = adapter.effective_mixing_matrix()
    if filtered_feature_mask is None or effective_mixing is None:
        adapted_values = adapter.transform(values)
    else:
        # Public feature masking removes an all-NaN feature from the table.  A
        # learned cross-column residual must not smuggle that feature back into
        # the remaining outputs.  Keep the learned submatrix on unmasked
        # numerical columns and zero only masked input contributions.
        unmixed = adapter.unmixed_transform(values)
        masked_numerical = torch.as_tensor(
            np.asarray(filtered_feature_mask, dtype=bool)[numerical_indices],
            dtype=torch.bool,
            device=canonical.device,
        )
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
        episode_rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 203))
        identity_state = _cpu_state_dict(adapters)
        best_state = identity_state
        best_error = float("inf")
        best_step = 0
        has_valid_adapted_checkpoint = False
        stale = 0
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
                if stale >= int(config["adapter_patience"]):
                    break
        adapters.load_state_dict(best_state, strict=True)
        del optimizer
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
