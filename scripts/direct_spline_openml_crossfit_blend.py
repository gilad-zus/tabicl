"""Evaluate a cross-fitted, table-level DirectSpline blending policy.

This is a follow-up to the preserved-fold DirectSpline experiment.  Its goal
is not to discover another spline architecture.  It asks a sharper deployment
question: can validation decide *how much* of a learned spline prediction to
use, without using the same validation rows both to select a checkpoint and
to judge that selected checkpoint?

For every ordinary eight-fold bag, seven folds are still used for fitting. The
held-out eighth fold is split deterministically into A and B.  One fixed,
full training trajectory produces two checkpoints: A selects the checkpoint
scored on B, and B selects the checkpoint scored on A.  Thus every
cross-fitted OOF prediction is made by a checkpoint whose selection did not
read that row's label.  The two selected checkpoints are averaged at outer
test time.

The pooled cross-fitted OOF predictions choose one *table-level* blend

    p_alpha = (1 - alpha) p_identity + alpha p_spline

from the predeclared grid ``{0, .25, .5, .75, 1}``.  A one-standard-error
rule chooses the smallest alpha within one standard error of the best mean
relative held-out loss.  Alpha zero is an explicit identity decision.  For
classification the convex blend remains a probability distribution and caps
the per-row log-loss increase relative to identity; it does not alter the
input representation row by row.

The outer-test labels are not read until every bag artifact and the
table-level alpha have been fixed.  The script is restricted to multiclass
and regression because the source DirectSpline D arm's binary metric (AUC) is
pairwise and needs a separate cross-fitted selection implementation.

The source D configuration supplies the fixed adapter architecture,
optimiser, learning-rate schedule (cosine where the source used it; constant
otherwise), and inner folds.  Its patience setting is
intentionally overridden to ``None``: a shared trajectory cannot be stopped
on A/B validation evidence without making one half influence the trajectory
that is evaluated on itself.  This preserves seven-fold fitting while avoiding
both label leakage and a wasteful second full training trajectory per bag.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \\
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_crossfit_blend.py \\
      --source-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828 \\
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_crossfit_blend/multiclass_D_260904 \\
      --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

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
    _bag_splits,
    _metric_bundle,
    _paired_comparison_summary,
    _prediction_shape,
    _safe_name,
    _seed,
    _standard_baseline_dir,
    effective_inner_bag_count,
    load_frozen_backbone,
)
from tabicl._experiments.direct_spline_openml_standard import (
    _candidate_deployment_error,
    _classification_training_objective_from_logits,
    _cosine_scheduler,
    _cpu_state_dict,
    _fit_standard_bag,
    _identity_view_parity,
    _make_adapters,
    _normal_prediction,
    _optimizer,
    _training_logits,
)
from tabicl._experiments.direct_spline_protocol import deployment_error, sample_episode_indices


CROSSFIT_BLEND_SCHEMA_VERSION = 1
_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
_HARD_ALPHA_GRID = (0.0, 1.0)
_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class CrossfitBagPredictions:
    """Predictions needed to reconstruct one independent A/B bag result."""

    validation_indices: np.ndarray
    selection_a_indices: np.ndarray
    selection_b_indices: np.ndarray
    identity_selection_a: np.ndarray
    identity_selection_b: np.ndarray
    spline_selected_on_b_selection_a: np.ndarray
    spline_selected_on_a_selection_b: np.ndarray
    identity_test: np.ndarray
    spline_selected_on_a_test: np.ndarray
    spline_selected_on_b_test: np.ndarray
    metadata: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Completed standard-pipeline source run containing D.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D", help="Fixed cubic source configuration label (default: D).")
    parser.add_argument("--task-id", type=int, action="append", help="Run only these completed source task IDs. Repeatable.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--protocol-seed", type=int, default=None, help="Omit to reuse the source inner-fold seed.")
    parser.add_argument("--bags", type=int, default=None, help="Omit to reuse each completed source task's bag count.")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument("--resume", action="store_true", help="Reuse only complete bag artifacts with this exact immutable protocol.")
    args = parser.parse_args()
    if args.bags is not None and args.bags < 2:
        raise ValueError("--bags must be at least two")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
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
            f"cross-fitted blend currently supports multiclass/regression only; task {case.task_id} is {case.problem_type!r}"
        )
    if str(case.config.get("adapter_architecture", "fixed_cubic")) != "fixed_cubic":
        raise ValueError("cross-fitted blend currently requires the fixed cubic D arm")
    if int(case.config.get("n_control_points", 20)) != 20:
        raise ValueError("cross-fitted blend requires fixed-cubic K20 D")
    if case.config.get("cosine_schedule_steps") is not None:
        if int(case.config["cosine_schedule_steps"]) < int(case.config["adapter_steps"]):
            raise ValueError("source cosine schedule must cover the full adapter trajectory")
        if "cosine_min_lr_ratio" not in case.config:
            raise ValueError("source cosine schedule lacks its minimum learning-rate ratio")
    if float(case.config.get("identity_regularization", 0.0)) != 0.0:
        raise ValueError(
            "cross-fitted blend currently requires identity_regularization=0 so its fixed trajectory exactly "
            "matches the source D training objective"
        )


def _crossfit_validation_halves(
    *, task: OpenMLTaskData, validation_indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint, approximately stratified A/B halves.

    The normal inner fold has already been stratified.  Splitting each class
    independently retains both classes whenever the inner fold permits it;
    singleton class rows are alternated so a small valid fold is never made
    empty solely by stratification.
    """

    validation_indices = np.asarray(validation_indices, dtype=int)
    if validation_indices.ndim != 1 or validation_indices.size < 2:
        raise ValueError("cross-fitted checkpoint selection requires at least two held-out rows")
    if np.unique(validation_indices).size != validation_indices.size:
        raise ValueError("validation indices must be unique")
    rng = np.random.default_rng(seed)
    if task.problem_type == "regression":
        shuffled = rng.permutation(validation_indices)
        midpoint = shuffled.size // 2
        a, b = shuffled[:midpoint], shuffled[midpoint:]
    else:
        labels = np.asarray(task.y_train[validation_indices])
        a_parts: list[np.ndarray] = []
        b_parts: list[np.ndarray] = []
        singleton_to_a = True
        for label in np.unique(labels):
            group = validation_indices[labels == label]
            group = rng.permutation(group)
            if group.size == 1:
                (a_parts if singleton_to_a else b_parts).append(group)
                singleton_to_a = not singleton_to_a
                continue
            split = group.size // 2
            a_parts.append(group[:split])
            b_parts.append(group[split:])
        a = np.concatenate(a_parts) if a_parts else np.empty(0, dtype=int)
        b = np.concatenate(b_parts) if b_parts else np.empty(0, dtype=int)
    if a.size == 0 or b.size == 0:
        raise ValueError("cross-fitted validation split produced an empty half")
    a = np.sort(np.asarray(a, dtype=int))
    b = np.sort(np.asarray(b, dtype=int))
    if np.intersect1d(a, b).size or not np.array_equal(np.sort(np.concatenate((a, b))), np.sort(validation_indices)):
        raise RuntimeError("cross-fitted validation halves must partition the original held-out fold")
    return a, b


def _save_bag(path: Path, result: CrossfitBagPredictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        validation_indices=result.validation_indices,
        selection_a_indices=result.selection_a_indices,
        selection_b_indices=result.selection_b_indices,
        identity_selection_a=result.identity_selection_a,
        identity_selection_b=result.identity_selection_b,
        spline_selected_on_b_selection_a=result.spline_selected_on_b_selection_a,
        spline_selected_on_a_selection_b=result.spline_selected_on_a_selection_b,
        identity_test=result.identity_test,
        spline_selected_on_a_test=result.spline_selected_on_a_test,
        spline_selected_on_b_test=result.spline_selected_on_b_test,
        metadata=np.asarray(json.dumps(result.metadata, sort_keys=True)),
    )


def _load_bag(path: Path) -> CrossfitBagPredictions:
    with np.load(path, allow_pickle=False) as payload:
        return CrossfitBagPredictions(
            validation_indices=np.asarray(payload["validation_indices"], dtype=int),
            selection_a_indices=np.asarray(payload["selection_a_indices"], dtype=int),
            selection_b_indices=np.asarray(payload["selection_b_indices"], dtype=int),
            identity_selection_a=np.asarray(payload["identity_selection_a"], dtype=float),
            identity_selection_b=np.asarray(payload["identity_selection_b"], dtype=float),
            spline_selected_on_b_selection_a=np.asarray(payload["spline_selected_on_b_selection_a"], dtype=float),
            spline_selected_on_a_selection_b=np.asarray(payload["spline_selected_on_a_selection_b"], dtype=float),
            identity_test=np.asarray(payload["identity_test"], dtype=float),
            spline_selected_on_a_test=np.asarray(payload["spline_selected_on_a_test"], dtype=float),
            spline_selected_on_b_test=np.asarray(payload["spline_selected_on_b_test"], dtype=float),
            metadata=json.loads(str(payload["metadata"].item())),
        )


def _bag_complete(
    *, path: Path, fingerprint: str, validation_indices: np.ndarray, test_shape: tuple[int, ...]
) -> bool:
    if not path.is_file():
        return False
    try:
        result = _load_bag(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if result.metadata.get("run_fingerprint_hash") != fingerprint:
        return False
    if not np.array_equal(result.validation_indices, validation_indices):
        return False
    if result.identity_test.shape != test_shape:
        return False
    expected_a = (result.selection_a_indices.size, *test_shape[1:])
    expected_b = (result.selection_b_indices.size, *test_shape[1:])
    required = (
        (result.identity_selection_a, expected_a),
        (result.identity_selection_b, expected_b),
        (result.spline_selected_on_b_selection_a, expected_a),
        (result.spline_selected_on_a_selection_b, expected_b),
        (result.identity_test, test_shape),
        (result.spline_selected_on_a_test, test_shape),
        (result.spline_selected_on_b_test, test_shape),
    )
    return bool(
        np.array_equal(np.sort(np.concatenate((result.selection_a_indices, result.selection_b_indices))), np.sort(validation_indices))
        and not np.intersect1d(result.selection_a_indices, result.selection_b_indices).size
        and all(value.shape == expected and np.isfinite(value).all() for value, expected in required)
    )


def _fit_crossfit_bag(
    *,
    task: OpenMLTaskData,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    bag: int,
    config: dict[str, Any],
    protocol_seed: int,
    backbone: Any,
    device: torch.device,
    run_fingerprint_hash: str,
    requested_bags: int,
    effective_bags: int,
) -> CrossfitBagPredictions:
    """Fit one seven-fold bag and retain independently selected A/B states."""

    if config.get("adapter_patience") is not None:
        raise ValueError(
            "cross-fitted selection requires adapter_patience=None: shared early stopping would let one half "
            "change the trajectory scored on itself"
        )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    selection_a_indices, selection_b_indices = _crossfit_validation_halves(
        task=task,
        validation_indices=validation_indices,
        seed=_seed(protocol_seed, task.task_id, bag, 405),
    )
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=fit_indices,
        config=config,
        protocol_seed=protocol_seed,
        bag=bag,
        backbone=backbone,
        device=device,
    )
    selection_a_x = task.x_train.iloc[selection_a_indices].reset_index(drop=True)
    selection_b_x = task.x_train.iloc[selection_b_indices].reset_index(drop=True)
    selection_a_y = np.asarray(task.y_train[selection_a_indices])
    selection_b_y = np.asarray(task.y_train[selection_b_indices])
    test_x = task.x_test
    adapter_seed = _seed(int(config["random_state"]), task.task_id, bag, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_a, parity_a_reference, public_parity_a = _identity_view_parity(
        bundle=bundle, adapters=adapters, query_x=selection_a_x, device=device, progress=None,
        task_id=task.task_id, bag=bag, split="crossfit_selection_a",
    )
    parity_b, parity_b_reference, public_parity_b = _identity_view_parity(
        bundle=bundle, adapters=adapters, query_x=selection_b_x, device=device, progress=None,
        task_id=task.task_id, bag=bag, split="crossfit_selection_b",
    )
    parity_test, parity_test_reference, public_parity_test = _identity_view_parity(
        bundle=bundle, adapters=adapters, query_x=test_x, device=device, progress=None,
        task_id=task.task_id, bag=bag, split="test",
    )
    training_context_sizes: list[int] = []
    first_objective = final_objective = float("nan")
    executed_steps = 0
    checkpoint_records: list[dict[str, Any]] = []
    best_a_step = best_b_step = 0
    best_a_error = best_b_error = float("inf")
    best_a_valid = best_b_valid = False
    if adapters is None:
        identity_a = _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=None, device=device)
        identity_b = _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=None, device=device)
        identity_test = _normal_prediction(bundle=bundle, query_x=test_x, context_indices=bundle.support_indices, adapters=None, device=device)
        spline_b_on_a = identity_a.copy()
        spline_a_on_b = identity_b.copy()
        spline_a_test = identity_test.copy()
        spline_b_test = identity_test.copy()
    else:
        optimizer = _optimizer(adapters, config)
        if config.get("cosine_schedule_steps") is None:
            scheduler = None
        else:
            scheduler = _cosine_scheduler(
                optimizer,
                total_steps=int(config["cosine_schedule_steps"]),
                min_lr_ratio=float(config["cosine_min_lr_ratio"]),
            )
        episode_rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 203))
        identity_state = _cpu_state_dict(adapters)
        best_a_state = identity_state
        best_b_state = identity_state
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
                bundle=bundle, adapters=adapters, context_indices=context_rows, query_indices=query_rows, device=device
            )
            target = torch.as_tensor(bundle.fit_labels[query_rows], device=device)
            if task.problem_type == "regression":
                objective = F.mse_loss(output.flatten(), target.float().flatten())
            else:
                objective = _classification_training_objective_from_logits(
                    logits=output,
                    target=target,
                    problem_type=task.problem_type,
                    n_classes=task.n_classes,
                    softmax_temperature=float(bundle.estimator.softmax_temperature),
                    config=config,
                )
            if not torch.isfinite(objective):
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
                candidate_a = _normal_prediction(
                    bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=adapters, device=device
                )
                candidate_b = _normal_prediction(
                    bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=adapters, device=device
                )
                candidate_a_error = _candidate_deployment_error(
                    task.problem_type, selection_a_y, candidate_a, n_classes=task.n_classes
                )
                candidate_b_error = _candidate_deployment_error(
                    task.problem_type, selection_b_y, candidate_b, n_classes=task.n_classes
                )
                if candidate_a_error < best_a_error:
                    best_a_error = candidate_a_error
                    best_a_state = _cpu_state_dict(adapters)
                    best_a_step = step
                    best_a_valid = True
                if candidate_b_error < best_b_error:
                    best_b_error = candidate_b_error
                    best_b_state = _cpu_state_dict(adapters)
                    best_b_step = step
                    best_b_valid = True
                checkpoint_records.append(
                    {
                        "step": int(step),
                        "selection_a_error": float(candidate_a_error),
                        "selection_b_error": float(candidate_b_error),
                        "best_selection_a_error": float(best_a_error),
                        "best_selection_b_error": float(best_b_error),
                        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                    }
                )
                del candidate_a, candidate_b
        del scheduler, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        identity_a = _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=None, device=device)
        identity_b = _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=None, device=device)
        identity_test = _normal_prediction(bundle=bundle, query_x=test_x, context_indices=bundle.support_indices, adapters=None, device=device)
        adapters.load_state_dict(best_a_state, strict=True)
        spline_a_on_b = _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=adapters, device=device)
        spline_a_test = _normal_prediction(bundle=bundle, query_x=test_x, context_indices=bundle.support_indices, adapters=adapters, device=device)
        adapters.load_state_dict(best_b_state, strict=True)
        spline_b_on_a = _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=adapters, device=device)
        spline_b_test = _normal_prediction(bundle=bundle, query_x=test_x, context_indices=bundle.support_indices, adapters=adapters, device=device)
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "bag": int(bag),
        "fit_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "selection_a_rows": int(selection_a_indices.size),
        "selection_b_rows": int(selection_b_indices.size),
        "support_rows": int(bundle.support_indices.size),
        "requested_bags": int(requested_bags),
        "effective_bags": int(effective_bags),
        "n_features": int(task.x_train.shape[1]),
        "n_numerical_features": int(bundle.numerical_indices.size),
        "no_trainable_numerical_features": bool(adapters is None),
        "pipeline": "standard_ensemble_crossfit_validation_blend",
        "checkpoint_protocol": "fixed full trajectory; A-selected checkpoint scores B and B-selected checkpoint scores A",
        "adapter_patience": None,
        "adapter_schedule": (
            {"kind": "constant"}
            if config.get("cosine_schedule_steps") is None
            else {
                "kind": "cosine",
                "horizon_steps": int(config["cosine_schedule_steps"]),
                "min_lr_ratio": float(config["cosine_min_lr_ratio"]),
            }
        ),
        "adapter_steps_requested": int(config["adapter_steps"]),
        "adapter_steps_executed": int(executed_steps),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "adapter_best_step_selection_a": int(best_a_step),
        "adapter_best_step_selection_b": int(best_b_step),
        "adapter_has_valid_checkpoint_selection_a": bool(best_a_valid),
        "adapter_has_valid_checkpoint_selection_b": bool(best_b_valid),
        "adapter_checkpoint_records": checkpoint_records,
        "identity_parity_max_abs_selection_a": float(parity_a),
        "identity_parity_max_abs_selection_b": float(parity_b),
        "identity_parity_max_abs_test": float(parity_test),
        "identity_parity_reference_selection_a": parity_a_reference,
        "identity_parity_reference_selection_b": parity_b_reference,
        "identity_parity_reference_test": parity_test_reference,
        "public_path_input_parity_checked_selection_a": bool(public_parity_a),
        "public_path_input_parity_checked_selection_b": bool(public_parity_b),
        "public_path_input_parity_checked_test": bool(public_parity_test),
        "adapter_observed_train_context_rows_min": None if not training_context_sizes else int(min(training_context_sizes)),
        "adapter_observed_train_context_rows_max": None if not training_context_sizes else int(max(training_context_sizes)),
        "adapter_observed_train_context_rows_mean": None if not training_context_sizes else float(np.mean(training_context_sizes)),
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
        "run_fingerprint_hash": run_fingerprint_hash,
    }
    del adapters, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return CrossfitBagPredictions(
        validation_indices=np.asarray(validation_indices, dtype=int),
        selection_a_indices=selection_a_indices,
        selection_b_indices=selection_b_indices,
        identity_selection_a=identity_a,
        identity_selection_b=identity_b,
        spline_selected_on_b_selection_a=spline_b_on_a,
        spline_selected_on_a_selection_b=spline_a_on_b,
        identity_test=identity_test,
        spline_selected_on_a_test=spline_a_test,
        spline_selected_on_b_test=spline_b_test,
        metadata=metadata,
    )


def _blend_prediction(identity: np.ndarray, spline: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must lie in [0, 1]")
    identity = np.asarray(identity, dtype=float)
    spline = np.asarray(spline, dtype=float)
    if identity.shape != spline.shape:
        raise ValueError("identity and spline predictions must have matching shapes")
    return (1.0 - alpha) * identity + alpha * spline


def _relative_error(identity_error: float, candidate_error: float) -> float:
    return float((candidate_error - identity_error) / max(abs(identity_error), _EPS))


def _alpha_selection(
    *,
    task: OpenMLTaskData,
    units: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: Sequence[float],
) -> dict[str, Any]:
    """Choose the smallest alpha within one SE of the best held-out mean."""

    if not units:
        raise ValueError("alpha selection needs at least one independent held-out unit")
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        unit_relative: list[float] = []
        unit_identity_error: list[float] = []
        unit_candidate_error: list[float] = []
        for labels, identity, spline in units:
            identity_error = _candidate_deployment_error(task.problem_type, labels, identity, n_classes=task.n_classes)
            candidate = _blend_prediction(identity, spline, float(alpha))
            candidate_error = _candidate_deployment_error(task.problem_type, labels, candidate, n_classes=task.n_classes)
            unit_identity_error.append(float(identity_error))
            unit_candidate_error.append(float(candidate_error))
            unit_relative.append(_relative_error(identity_error, candidate_error))
        values = np.asarray(unit_relative, dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("a cross-fitted alpha has non-finite held-out error")
        standard_error = 0.0 if values.size == 1 else float(values.std(ddof=1) / np.sqrt(values.size))
        records.append(
            {
                "alpha": float(alpha),
                "mean_relative_error_change": float(values.mean()),
                "standard_error_relative_error_change": standard_error,
                "median_relative_error_change": float(np.median(values)),
                "unit_relative_error_changes": values.tolist(),
                "unit_identity_errors": unit_identity_error,
                "unit_candidate_errors": unit_candidate_error,
            }
        )
    best = min(records, key=lambda item: (float(item["mean_relative_error_change"]), float(item["alpha"])))
    threshold = float(best["mean_relative_error_change"]) + float(best["standard_error_relative_error_change"])
    eligible = [item for item in records if float(item["mean_relative_error_change"]) <= threshold + 1e-15]
    selected = min(eligible, key=lambda item: float(item["alpha"]))
    return {
        "selection_rule": "smallest alpha within one standard error of the minimum mean relative independent-heldout loss",
        "n_independent_units": len(units),
        "candidate_alphas": records,
        "best_alpha_before_one_se": float(best["alpha"]),
        "best_mean_relative_error_change": float(best["mean_relative_error_change"]),
        "best_standard_error": float(best["standard_error_relative_error_change"]),
        "one_se_threshold": threshold,
        "eligible_alphas": [float(item["alpha"]) for item in eligible],
        "selected_alpha": float(selected["alpha"]),
    }


def _source_standard_prediction(*, source_dir: Path, task: OpenMLTaskData) -> np.ndarray | None:
    path = _standard_baseline_dir(source_dir, task) / "predictions.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"], dtype=float)
    expected = _prediction_shape(len(task.y_test), task.problem_type, task.n_classes)
    if prediction.shape != expected or not np.isfinite(prediction).all():
        raise ValueError(f"source full TabICLv2 prediction is invalid: {path}")
    return prediction


def _task_artifact_dir(output_dir: Path, task: OpenMLTaskData) -> Path:
    return output_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _assemble_task_predictions(
    *, task: OpenMLTaskData, bag_results: Sequence[CrossfitBagPredictions]
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Assemble independent OOF evidence and the deployable bagged test pair."""

    shape = _prediction_shape(len(task.y_train), task.problem_type, task.n_classes)
    identity_oof = np.full(shape, np.nan, dtype=float)
    spline_oof = np.full(shape, np.nan, dtype=float)
    units: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for result in bag_results:
        identity_oof[result.selection_a_indices] = result.identity_selection_a
        spline_oof[result.selection_a_indices] = result.spline_selected_on_b_selection_a
        identity_oof[result.selection_b_indices] = result.identity_selection_b
        spline_oof[result.selection_b_indices] = result.spline_selected_on_a_selection_b
        units.append((
            np.asarray(task.y_train[result.selection_a_indices]),
            result.identity_selection_a,
            result.spline_selected_on_b_selection_a,
        ))
        units.append((
            np.asarray(task.y_train[result.selection_b_indices]),
            result.identity_selection_b,
            result.spline_selected_on_a_selection_b,
        ))
    if not np.isfinite(identity_oof).all() or not np.isfinite(spline_oof).all():
        raise RuntimeError(f"task {task.task_id} has incomplete cross-fitted OOF predictions")
    identity_test = np.mean([result.identity_test for result in bag_results], axis=0)
    spline_test = np.mean(
        [0.5 * (result.spline_selected_on_a_test + result.spline_selected_on_b_test) for result in bag_results],
        axis=0,
    )
    return identity_oof, spline_oof, units, identity_test, spline_test


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest(
    *, source_dir: Path, source_manifest: Mapping[str, Any], cases: Sequence[SourceCase], args: argparse.Namespace, protocol_seed: int
) -> dict[str, Any]:
    return {
        "crossfit_blend_schema_version": CROSSFIT_BLEND_SCHEMA_VERSION,
        "experiment": "DirectSpline symmetric cross-fitted checkpoint selection and table-level blend",
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
        "source_repository_revision": source_manifest.get("immutable_run", {}).get("repository_revision"),
        "config_label": args.config_label,
        "task_ids": [case.task_id for case in cases],
        "protocol_seed": int(protocol_seed),
        "requested_bags": args.bags,
        "fixed_arm_requirement": {"adapter_architecture": "fixed_cubic", "n_control_points": 20},
        "training": {
            "source_config": "all source D fields retained except adapter_patience",
            "adapter_patience_override": None,
            "reason": "a shared trajectory must not be early-stopped on labels from the half on which it is independently evaluated",
        },
        "crossfit_protocol": {
            "fit_folds_per_bag": "seven of eight",
            "heldout_fold": "deterministically partitioned into disjoint approximately stratified A/B halves",
            "checkpoint_selection": "A-selected state predicts B; B-selected state predicts A",
            "test_spline_ensemble": "mean of the A-selected and B-selected states inside every ordinary bag, then mean across bags",
            "test_identity_ensemble": "ordinary mean across the eight seven-fold identity bags",
        },
        "table_policy": {
            "hard_gate_alphas": list(_HARD_ALPHA_GRID),
            "blend_alphas": list(_ALPHA_GRID),
            "selection_rule": "smallest alpha within one standard error of the minimum mean relative independent-heldout loss",
            "selection_metric": "multiclass log loss or regression MSE",
        },
        "label_policy": {
            "fit_labels": "adapter optimisation on seven inner folds only",
            "A_B_labels": "select their own checkpoint; score only the opposite selected checkpoint and select table alpha",
            "outer_test_labels": "read only after all bag artifacts and both table-level alphas are fixed, for final reporting",
        },
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> str:
    path = output_dir / "experiment_manifest.json"
    fingerprint = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    payload = {**manifest, "run_fingerprint_sha256": fingerprint}
    if path.exists():
        existing = _load_json(path, label="existing cross-fitted blend manifest")
        if _canonical_json(existing) != _canonical_json(payload):
            raise ValueError("output directory belongs to a different cross-fitted blend protocol; choose a new --output-dir")
        if not resume:
            raise ValueError("cross-fitted blend output directory already exists; pass --resume to reuse matching bags")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return fingerprint


def _run_task(
    *,
    case: SourceCase,
    task: OpenMLTaskData,
    output_dir: Path,
    args: argparse.Namespace,
    protocol_seed: int,
    fingerprint: str,
) -> dict[str, Any]:
    _validate_case(case)
    requested_bags = case.requested_bags if args.bags is None else int(args.bags)
    splits = list(_bag_splits(task, requested_bags=requested_bags, seed=_seed(protocol_seed, task.task_id, 0)))
    effective_bags = effective_inner_bag_count(task, requested_bags=requested_bags)
    if len(splits) != effective_bags:
        raise RuntimeError("source inner bag construction did not return the expected number of folds")
    task_dir = _task_artifact_dir(output_dir, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    config = dict(case.config)
    source_patience = config.get("adapter_patience")
    config["adapter_patience"] = None
    test_shape = _prediction_shape(len(task.y_test), task.problem_type, task.n_classes)
    missing = [
        (bag, fit_indices, validation_indices)
        for bag, (fit_indices, validation_indices) in enumerate(splits)
        if not _bag_complete(
            path=task_dir / f"bag_{bag}.npz",
            fingerprint=fingerprint,
            validation_indices=np.asarray(validation_indices, dtype=int),
            test_shape=test_shape,
        )
    ]
    if missing:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested as {args.device!r}, but it is not available")
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
            "source_config": case.config,
            "effective_config": config,
            "source_adapter_patience": source_patience,
            "checkpoint": checkpoint_metadata,
            "bag_splits": [
                {"bag": bag, "fit_rows": int(len(fit)), "heldout_rows": int(len(heldout))}
                for bag, (fit, heldout) in enumerate(splits)
            ],
        })
        for position, (bag, fit_indices, validation_indices) in enumerate(missing, start=1):
            print(
                f"task {task.task_id} bag {bag} ({position}/{len(missing)} missing): "
                f"fit={len(fit_indices)} heldout={len(validation_indices)}; fixed-trajectory A/B checkpoint selection",
                flush=True,
            )
            result = _fit_crossfit_bag(
                task=task,
                fit_indices=np.asarray(fit_indices, dtype=int),
                validation_indices=np.asarray(validation_indices, dtype=int),
                bag=bag,
                config=config,
                protocol_seed=protocol_seed,
                backbone=backbone,
                device=device,
                run_fingerprint_hash=fingerprint,
                requested_bags=requested_bags,
                effective_bags=effective_bags,
            )
            _save_bag(task_dir / f"bag_{bag}.npz", result)
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    bag_results = [_load_bag(task_dir / f"bag_{bag}.npz") for bag in range(effective_bags)]
    identity_oof, spline_oof, units, identity_test, spline_test = _assemble_task_predictions(task=task, bag_results=bag_results)
    hard_selection = _alpha_selection(task=task, units=units, alphas=_HARD_ALPHA_GRID)
    blend_selection = _alpha_selection(task=task, units=units, alphas=_ALPHA_GRID)
    hard_alpha = float(hard_selection["selected_alpha"])
    blend_alpha = float(blend_selection["selected_alpha"])
    hard_oof = _blend_prediction(identity_oof, spline_oof, hard_alpha)
    blend_oof = _blend_prediction(identity_oof, spline_oof, blend_alpha)
    hard_test = _blend_prediction(identity_test, spline_test, hard_alpha)
    blend_test = _blend_prediction(identity_test, spline_test, blend_alpha)
    # The outer-test labels enter only below this point, after both alphas and
    # every deployed prediction have been frozen and written in memory.
    prediction_path = task_dir / "task_predictions.npz"
    np.savez_compressed(
        prediction_path,
        identity_oof=identity_oof,
        spline_oof=spline_oof,
        hard_oof=hard_oof,
        blend_oof=blend_oof,
        identity_test=identity_test,
        spline_test=spline_test,
        hard_test=hard_test,
        blend_test=blend_test,
    )
    full_standard_prediction = _source_standard_prediction(source_dir=case.source_dir, task=task)
    task_summary: dict[str, Any] = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "requested_bags": requested_bags,
        "effective_bags": effective_bags,
        "source_adapter_patience": source_patience,
        "effective_adapter_patience": None,
        "crossfit_oof": {
            "identity": _metric_bundle(task.problem_type, task.y_train, identity_oof, task.n_classes),
            "raw_spline": _metric_bundle(task.problem_type, task.y_train, spline_oof, task.n_classes),
            "hard_gate": _metric_bundle(task.problem_type, task.y_train, hard_oof, task.n_classes),
            "blend": _metric_bundle(task.problem_type, task.y_train, blend_oof, task.n_classes),
        },
        "hard_gate_selection": hard_selection,
        "blend_selection": blend_selection,
        "outer_test_scored_after_crossfit_policy_lock": True,
        "outer_test": {
            "matched_inner_bag_identity": _metric_bundle(task.problem_type, task.y_test, identity_test, task.n_classes),
            "crossfit_raw_spline": _metric_bundle(task.problem_type, task.y_test, spline_test, task.n_classes),
            "crossfit_hard_gate": _metric_bundle(task.problem_type, task.y_test, hard_test, task.n_classes),
            "crossfit_blend": _metric_bundle(task.problem_type, task.y_test, blend_test, task.n_classes),
        },
        "source_full_outer_training_tabiclv2": (
            None if full_standard_prediction is None else _metric_bundle(task.problem_type, task.y_test, full_standard_prediction, task.n_classes)
        ),
        "mean_train_seconds_per_bag": float(np.mean([float(item.metadata["train_seconds"]) for item in bag_results])),
        "max_peak_allocated_gib": float(np.max([float(item.metadata["peak_allocated_gib"]) for item in bag_results])),
        "max_identity_parity_abs": float(max(
            max(
                item.metadata["identity_parity_max_abs_selection_a"],
                item.metadata["identity_parity_max_abs_selection_b"],
                item.metadata["identity_parity_max_abs_test"],
            )
            for item in bag_results
        )),
        "test_prediction_artifact": str(prediction_path),
    }
    _write_json(task_dir / "task_summary.json", task_summary)
    return task_summary


def _comparison(
    *, task_summaries: Sequence[Mapping[str, Any]], candidate: str, bootstrap_rounds: int, bootstrap_seed: int
) -> dict[str, Any]:
    reference = np.asarray(
        [float(item["outer_test"]["matched_inner_bag_identity"]["benchmark_error"]) for item in task_summaries], dtype=float
    )
    values = np.asarray([float(item["outer_test"][candidate]["benchmark_error"]) for item in task_summaries], dtype=float)
    problem_types = np.asarray([str(item["problem_type"]) for item in task_summaries], dtype=object)
    return _paired_comparison_summary(
        reference=reference,
        candidate=values,
        problem_types=problem_types,
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=bootstrap_seed,
        reference_label="matched_seven_fold_tabiclv2_identity",
        candidate_label=candidate,
    )


def _full_standard_comparison(
    *, task_summaries: Sequence[Mapping[str, Any]], candidate: str, bootstrap_rounds: int, bootstrap_seed: int
) -> dict[str, Any] | None:
    usable = [item for item in task_summaries if item["source_full_outer_training_tabiclv2"] is not None]
    if not usable:
        return None
    reference = np.asarray(
        [float(item["source_full_outer_training_tabiclv2"]["benchmark_error"]) for item in usable], dtype=float
    )
    values = np.asarray([float(item["outer_test"][candidate]["benchmark_error"]) for item in usable], dtype=float)
    problem_types = np.asarray([str(item["problem_type"]) for item in usable], dtype=object)
    return _paired_comparison_summary(
        reference=reference,
        candidate=values,
        problem_types=problem_types,
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=bootstrap_seed,
        reference_label="source_full_outer_training_tabiclv2",
        candidate_label=candidate,
    )


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
        raise ValueError("source has no completed multiclass/regression D tasks")
    for case in cases:
        _validate_case(case)
    source_protocol_seed = _as_int(immutable_run.get("protocol_seed"), name="source protocol_seed")
    protocol_seed = source_protocol_seed if args.protocol_seed is None else int(args.protocol_seed)
    manifest = _manifest(source_dir=source_dir, source_manifest=source_manifest, cases=cases, args=args, protocol_seed=protocol_seed)
    fingerprint = _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))
    task_summaries: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}: cross-fitted blend", flush=True)
        task = _load_source_task(case=case, immutable_run=immutable_run)
        task_summaries.append(
            _run_task(
                case=case,
                task=task,
                output_dir=args.output_dir,
                args=args,
                protocol_seed=protocol_seed,
                fingerprint=fingerprint,
            )
        )
    task_summaries.sort(key=lambda item: int(item["task_id"]))
    _write_json(args.output_dir / "task_summaries.json", task_summaries)
    rows = [
        {
            "task_id": item["task_id"],
            "dataset_id": item["dataset_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "hard_gate_alpha": item["hard_gate_selection"]["selected_alpha"],
            "blend_alpha": item["blend_selection"]["selected_alpha"],
            "identity_benchmark_error": item["outer_test"]["matched_inner_bag_identity"]["benchmark_error"],
            "raw_spline_benchmark_error": item["outer_test"]["crossfit_raw_spline"]["benchmark_error"],
            "hard_gate_benchmark_error": item["outer_test"]["crossfit_hard_gate"]["benchmark_error"],
            "blend_benchmark_error": item["outer_test"]["crossfit_blend"]["benchmark_error"],
            "full_tabiclv2_benchmark_error": (
                None if item["source_full_outer_training_tabiclv2"] is None else item["source_full_outer_training_tabiclv2"]["benchmark_error"]
            ),
        }
        for item in task_summaries
    ]
    _write_csv(args.output_dir / "task_results.csv", rows)
    candidates = ("crossfit_raw_spline", "crossfit_hard_gate", "crossfit_blend")
    summary = {
        "crossfit_blend_schema_version": CROSSFIT_BLEND_SCHEMA_VERSION,
        "n_tasks": len(task_summaries),
        "task_summaries": task_summaries,
        "paired_vs_matched_inner_bag_identity": {
            candidate: _comparison(
                task_summaries=task_summaries,
                candidate=candidate,
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.bootstrap_seed + index,
            )
            for index, candidate in enumerate(candidates)
        },
        "end_to_end_vs_source_full_outer_training_tabiclv2": {
            candidate: _full_standard_comparison(
                task_summaries=task_summaries,
                candidate=candidate,
                bootstrap_rounds=args.bootstrap_rounds,
                bootstrap_seed=args.bootstrap_seed + 10_000 + index,
            )
            for index, candidate in enumerate(candidates)
        },
        "alpha_counts": {
            "hard_gate": {str(alpha): int(sum(item["hard_gate_selection"]["selected_alpha"] == alpha for item in task_summaries)) for alpha in _HARD_ALPHA_GRID},
            "blend": {str(alpha): int(sum(item["blend_selection"]["selected_alpha"] == alpha for item in task_summaries)) for alpha in _ALPHA_GRID},
        },
        "metric_note": "Multiclass uses log loss for validation selection and benchmark reporting; regression uses MSE for selection and RMSE for benchmark/Elo comparison.",
        "paired_elo_note": "Paired Elo is a within-this-run DirectSpline-versus-reference delta, not an absolute Retouche-style rating.",
        "outer_test_label_policy": "Outer-test labels were read only after cross-fitted checkpoint selection and table alpha selection were fixed.",
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"n_tasks": len(task_summaries), "alpha_counts": summary["alpha_counts"]}), flush=True)


if __name__ == "__main__":
    main()
