"""Full-pipeline DirectSpline experiment for the public OpenML TabArena suite.

This is the corrected companion to :mod:`direct_spline_openml`.  The older
runner was deliberately a small raw-backbone headroom experiment: it sampled a
single context and bypassed TabICL's usual preprocessing ensemble.  That is a
useful diagnostic, but its identity path is not the normal TabICLv2 estimator.

This module keeps the DirectSpline adapter as the *only* learned addition while
using the normal estimator's per-bag preprocessing views, feature/class
shuffles, temperature, and logit averaging.  Before training an adapter, it
checks that a freshly initialised (therefore exact-identity) spline reproduces
the same predictions as the normal estimator fitted on the same bag.

The outer OpenML test labels are never used for preprocessing fitting, adapter
optimisation, identity guarding, or configuration selection.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

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
from tabicl._model.tabicl import TabICL


# ``None`` deliberately means that the normal estimator receives every fitting
# row of a bag.  A positive cap is available only for hardware-constrained
# diagnostics and is recorded as such in every artifact.
STANDARD_DIRECT_SPLINE_CONFIG: dict[str, Any] = {
    "adapter_steps": 150,
    "adapter_patience": 10,
    "validation_interval": 10,
    "max_context_rows": None,
    "train_context_rows": 1024,
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
    "identity_parity_atol": 5e-5,
    "identity_parity_rtol": 1e-5,
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
    if train_context_rows is not None:
        if train_context_rows <= 0:
            raise ValueError("train_context_rows must be positive when provided")
        config["train_context_rows"] = int(train_context_rows)
    return config


def shared_standard_direct_spline_configs(n_configs: int, *, seed: int, context_cap: int | None = None) -> list[dict[str, Any]]:
    """The existing ten random DirectSpline draws, but on the strong path."""

    # Import lazily to keep this runner independent of the old runner's
    # default context policy at module import time.
    from tabicl._experiments.direct_spline_protocol import shared_random_direct_spline_configs

    configs = shared_random_direct_spline_configs(n_configs, seed=seed)
    for config in configs:
        config["max_context_rows"] = context_cap
        config["train_context_rows"] = STANDARD_DIRECT_SPLINE_CONFIG["train_context_rows"]
        config["identity_parity_atol"] = STANDARD_DIRECT_SPLINE_CONFIG["identity_parity_atol"]
        config["identity_parity_rtol"] = STANDARD_DIRECT_SPLINE_CONFIG["identity_parity_rtol"]
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
        "kv_cache": False,
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

    if task.problem_type == "regression":
        scaled = estimator.y_scaler_.transform(raw_labels.reshape(-1, 1)).ravel().astype(np.float32)
    else:
        # The ensemble generator owns the label-encoded values used by the
        # normal classifier, so use that exact representation for episodes.
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
    )


def _make_adapters(bundle: _StandardBag, config: dict[str, Any], device: torch.device) -> _AdapterSet | None:
    """Create identity-initialised numerical splines in each normal branch."""

    if bundle.numerical_indices.size == 0:
        return None
    adapters: OrderedDict[str, DirectSplineTransform] = OrderedDict()
    for method, preprocessor in bundle.estimator.ensemble_generator_.preprocessors_.items():
        support = torch.as_tensor(
            preprocessor.X_transformed_[bundle.support_indices][:, bundle.numerical_indices],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        adapter = DirectSplineTransform(
            support,
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
            if not torch.allclose(adapter.transform(support), support, atol=2e-5, rtol=2e-5):
                raise RuntimeError("standard-pipeline DirectSpline did not initialise to identity")
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
) -> torch.Tensor:
    if canonical.ndim != 2:
        raise ValueError("canonical standard-preprocessed features must have shape (rows, features)")
    if adapter is None or numerical_indices.size == 0:
        return canonical
    indices = torch.as_tensor(numerical_indices, dtype=torch.long, device=canonical.device)
    transformed = canonical.clone()
    values = canonical.index_select(-1, indices).unsqueeze(0)
    transformed[..., indices] = adapter.transform(values).squeeze(0)
    return transformed


def _encoded_query(bundle: _StandardBag, query_x: Any) -> np.ndarray:
    encoded = bundle.estimator.X_encoder_.transform(query_x)
    return bundle.estimator.ensemble_generator_.unique_filter_.transform(encoded)


def _build_method_batch(
    *,
    bundle: _StandardBag,
    method: str,
    context_canonical: np.ndarray,
    query_canonical: np.ndarray,
    context_labels: np.ndarray,
    adapters: _AdapterSet | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[np.ndarray | None]]:
    """Create normal TabICL ensemble views after an optional spline nudge."""

    generator = bundle.estimator.ensemble_generator_
    configs = generator.ensemble_configs_[method]
    canonical = torch.as_tensor(
        np.concatenate((context_canonical, query_canonical), axis=0), dtype=torch.float32, device=device
    )
    adapter = None if adapters is None else adapters.for_method(method)
    canonical = _apply_adapter(canonical, numerical_indices=bundle.numerical_indices, adapter=adapter)
    views = torch.stack([canonical[:, feature_shuffle] for feature_shuffle, _ in configs], dim=0)
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
    return views, labels, generator.feature_shuffles_[method], class_patterns


def _forward_method(
    *,
    bundle: _StandardBag,
    views: torch.Tensor,
    labels: torch.Tensor,
    feature_shuffles: list[list[int]],
    checkpoint_activations: bool,
) -> torch.Tensor:
    """Forward the frozen normal inference path, retaining gradients to splines."""

    backbone = bundle.backbone

    def run(features: torch.Tensor) -> torch.Tensor:
        backbone.clear_cache()
        return backbone(
            X=features,
            y_train=labels,
            # The public classifier supplies the shuffle metadata so its
            # column embedder can share work across equivalent tables.  The
            # public regressor instead forwards its already-shuffled views
            # without that metadata.  Preserve that small but real pipeline
            # difference here; otherwise an "identity" spline would not be
            # the normal regressor.
            feature_shuffles=feature_shuffles if bundle.problem_type != "regression" else None,
            return_logits=True,
            softmax_temperature=float(STANDARD_TABICL_CONFIG["softmax_temperature"]),
            inference_config=bundle.estimator.inference_config_,
        )

    if checkpoint_activations:
        return checkpoint(run, views, use_reentrant=False)
    return run(views)


def _aggregate_classification_logits(
    outputs: list[tuple[torch.Tensor, list[np.ndarray | None]]],
    *,
    n_classes: int,
) -> torch.Tensor:
    """Undo ordinary class shuffles and average logits exactly as TabICL does."""

    corrected: list[torch.Tensor] = []
    for raw, patterns in outputs:
        for index, pattern in enumerate(patterns):
            if pattern is None:
                raise RuntimeError("classification ensemble member is missing its class shuffle")
            permutation = torch.as_tensor(pattern, dtype=torch.long, device=raw.device)
            corrected.append(raw[index, :, permutation][:, :n_classes])
    if not corrected:
        raise RuntimeError("normal classifier produced no ensemble outputs")
    return torch.stack(corrected, dim=0).mean(dim=0)


def _normal_prediction(
    *,
    bundle: _StandardBag,
    query_x: Any,
    context_indices: np.ndarray,
    adapters: _AdapterSet | None,
    device: torch.device,
) -> np.ndarray:
    """Predict with normal ensemble views and an optional DirectSpline set."""

    generator = bundle.estimator.ensemble_generator_
    query_encoded = _encoded_query(bundle, query_x)
    outputs: list[tuple[torch.Tensor, list[np.ndarray | None]]] = []
    regression_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for method, preprocessor in generator.preprocessors_.items():
            context = preprocessor.X_transformed_[context_indices]
            query = preprocessor.transform(query_encoded)
            views, labels, feature_shuffles, class_patterns = _build_method_batch(
                bundle=bundle,
                method=method,
                context_canonical=context,
                query_canonical=query,
                context_labels=bundle.fit_labels[context_indices],
                adapters=adapters,
                device=device,
            )
            raw = _forward_method(
                bundle=bundle,
                views=views,
                labels=labels,
                feature_shuffles=feature_shuffles,
                checkpoint_activations=False,
            )
            if bundle.problem_type == "regression":
                regression_outputs.append(bundle.backbone.quantile_dist(raw).quantiles.mean(dim=-1))
            else:
                outputs.append((raw, class_patterns))
    if bundle.problem_type == "regression":
        if not regression_outputs:
            raise RuntimeError("normal regressor produced no ensemble outputs")
        scaled = torch.cat(regression_outputs, dim=0).mean(dim=0).cpu().numpy()
        return bundle.estimator.y_scaler_.inverse_transform(scaled.reshape(-1, 1)).ravel().astype(np.float64)
    if bundle.n_classes is None:
        raise RuntimeError("classification bag has no class count")
    logits = _aggregate_classification_logits(outputs, n_classes=bundle.n_classes)
    probabilities = torch.softmax(logits / float(STANDARD_TABICL_CONFIG["softmax_temperature"]), dim=-1)
    return probabilities.cpu().numpy().astype(np.float64)


def _identity_prediction(bundle: _StandardBag, query_x: Any) -> np.ndarray:
    """Use the public estimator itself as the authoritative identity control."""

    if bundle.problem_type == "regression":
        return np.asarray(bundle.estimator.predict(query_x), dtype=np.float64)
    return np.asarray(bundle.estimator.predict_proba(query_x), dtype=np.float64)


def _identity_parity(
    *,
    bundle: _StandardBag,
    adapters: _AdapterSet | None,
    query_x: Any,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[np.ndarray, float, str]:
    """Build the matched identity control and verify the zero spline.

    With all fit rows as context, this additionally checks the custom
    differentiable implementation against the public sklearn estimator.  A
    deliberately capped run cannot make that claim: its matched control uses
    the same capped rows as its spline arm, and is checked only against its
    own no-adapter path.
    """

    matched_identity = _normal_prediction(
        bundle=bundle,
        query_x=query_x,
        context_indices=bundle.support_indices,
        adapters=None,
        device=device,
    )
    parity_errors: list[float] = []
    if bundle.support_indices.size == bundle.fit_labels.size:
        public_identity = _identity_prediction(bundle, query_x)
        max_public_abs = float(np.max(np.abs(matched_identity - public_identity)))
        if not np.allclose(
            matched_identity,
            public_identity,
            rtol=float(config["identity_parity_rtol"]),
            atol=float(config["identity_parity_atol"]),
        ):
            raise RuntimeError(
                "standard-pipeline identity parity failed: the reconstructed normal TabICL path differs from "
                f"the public estimator (max_abs={max_public_abs:.3g}). Refuse to train against a mismatched baseline."
            )
        parity_errors.append(max_public_abs)
        reference = "public_full_context_estimator"
    else:
        reference = "matched_capped_standard_views"
    if adapters is None:
        return matched_identity, max(parity_errors, default=0.0), reference
    spline_identity = _normal_prediction(
        bundle=bundle,
        query_x=query_x,
        context_indices=bundle.support_indices,
        adapters=adapters,
        device=device,
    )
    max_abs = float(np.max(np.abs(spline_identity - matched_identity)))
    if not np.allclose(
        spline_identity,
        matched_identity,
        rtol=float(config["identity_parity_rtol"]),
        atol=float(config["identity_parity_atol"]),
    ):
        raise RuntimeError(
            "standard-pipeline identity parity failed: a fresh DirectSpline changed normal TabICL predictions "
            f"(max_abs={max_abs:.3g}). Refuse to train against a mismatched baseline."
        )
    parity_errors.append(max_abs)
    return matched_identity, max(parity_errors), reference


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
        raw = _forward_method(
            bundle=bundle,
            views=views,
            labels=labels,
            feature_shuffles=feature_shuffles,
            checkpoint_activations=True,
        )
        if bundle.problem_type == "regression":
            regression_outputs.append(bundle.backbone.quantile_dist(raw).quantiles.mean(dim=-1))
        else:
            classification_outputs.append((raw, patterns))
    if bundle.problem_type == "regression":
        return torch.cat(regression_outputs, dim=0).mean(dim=0)
    if bundle.n_classes is None:
        raise RuntimeError("classification bag has no class count")
    return _aggregate_classification_logits(classification_outputs, n_classes=bundle.n_classes)


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
    identity_validation, parity_validation, parity_reference = _identity_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=validation_x,
        device=device,
        config=config,
    )
    identity_test, parity_test, parity_reference_test = _identity_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=test_x,
        device=device,
        config=config,
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
    if adapters is None:
        adapted_validation = identity_validation.copy()
        adapted_test = identity_test.copy()
        first_objective = final_objective = float("nan")
        executed_steps = 0
    else:
        optimizer = _optimizer(adapters, config)
        episode_rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 203))
        best_state = _cpu_state_dict(adapters)
        best_error = float("inf")
        stale = 0
        first_objective = final_objective = float("nan")
        executed_steps = 0
        for step in range(1, int(config["adapter_steps"]) + 1):
            context_rows, query_rows = sample_episode_indices(
                bundle.fit_labels,
                problem_type=task.problem_type,
                context_rows=int(config["train_context_rows"]),
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
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
                    output / float(STANDARD_TABICL_CONFIG["softmax_temperature"]), target.long().flatten()
                )
            if step == 1:
                first_objective = float(objective.detach())
            objective.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
            optimizer.step()
            final_objective = float(objective.detach())
            executed_steps = step
            if step % int(config["validation_interval"]) == 0 or step == int(config["adapter_steps"]):
                candidate = _normal_prediction(
                    bundle=bundle,
                    query_x=validation_x,
                    context_indices=bundle.support_indices,
                    adapters=adapters,
                    device=device,
                )
                candidate_error = deployment_error(
                    task.problem_type, raw_validation_y, candidate, n_classes=task.n_classes
                )
                if candidate_error < best_error:
                    best_error = candidate_error
                    best_state = _cpu_state_dict(adapters)
                    stale = 0
                else:
                    stale += 1
                    if stale >= int(config["adapter_patience"]):
                        break
                _emit(
                    progress,
                    event="adapter_validation",
                    task_id=task.task_id,
                    bag=bag,
                    step=step,
                    validation_error=float(candidate_error),
                    best_validation_error=float(best_error),
                    stale_validations=stale,
                    elapsed_seconds=float(time.perf_counter() - started),
                )
        adapters.load_state_dict(best_state, strict=True)
        adapted_validation = _normal_prediction(
            bundle=bundle,
            query_x=validation_x,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
        adapted_test = _normal_prediction(
            bundle=bundle,
            query_x=test_x,
            context_indices=bundle.support_indices,
            adapters=adapters,
            device=device,
        )
        del optimizer
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
        "normal_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "pipeline": "standard_ensemble",
        "context_policy": (
            "all_fit_rows"
            if bundle.support_indices.size == bundle.fit_labels.size
            else "stratified_context_cap"
        ),
        "identity_parity_max_abs_validation": parity_validation,
        "identity_parity_max_abs_test": parity_test,
        "identity_parity_passed": True,
        "identity_parity_reference": parity_reference,
        "identity_parity_reference_test": parity_reference_test,
        "identity_error": float(identity_error),
        "adapted_error": float(adapted_error),
        "relative_improvement": float(decision.relative_improvement),
        "guard_selected_adapted": bool(decision.use_adapted),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "adapter_steps_executed": executed_steps,
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
    if not np.isfinite(guarded_validation).all():
        raise RuntimeError(f"task {task.task_id} config {label} did not produce complete OOF predictions")
    identity_test = np.mean([result.identity_test for result in bag_results], axis=0)
    adapted_test = np.mean([result.adapted_test for result in bag_results], axis=0)
    guarded_test = np.mean([result.guarded_test for result in bag_results], axis=0)
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
        "test": {
            "identity": _metric_bundle(task.problem_type, task.y_test, identity_test, task.n_classes),
            "adapted": _metric_bundle(task.problem_type, task.y_test, adapted_test, task.n_classes),
            "guarded": _metric_bundle(task.problem_type, task.y_test, guarded_test, task.n_classes),
        },
        "guard_selected_adapted_fraction": float(
            np.mean([bool(result.metadata["guard_selected_adapted"]) for result in bag_results])
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
        guarded_test_error=summary["test"]["guarded"]["benchmark_error"],
    )
    del backbone, checkpoint_path
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
