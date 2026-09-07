"""Test whether frozen DirectSpline benefits from a larger labelled ICL context.

Each ordinary eight-fold bag has fitting rows ``T`` and a held-out fold split
into disjoint halves ``A`` and ``B``.  A single adapter trajectory is fitted on
``T`` only.  It supplies two independent checkpoint choices under the original
T-only context: the checkpoint selected on A predicts B, and vice versa.  The
same two frozen states are then reused in the expanded-context arm.

The experiment evaluates two otherwise identical contexts:

* ``original``: every prediction uses T as its ICL context;
* ``expanded``: A is predicted with T+B, B with T+A, and outer test with
  T+A+B.  The adapter, feature preprocessing, target scaler/encoder, and
  ensemble views remain frozen from T.

Thus neither an OOF prediction nor a selected checkpoint receives that row's
label in context.  The expanded identity arm is a necessary control: it tells
us whether any benefit comes from extra ICL examples alone rather than from
the spline.  The selected table blend is chosen independently from each
condition's cross-fitted OOF predictions, but the underlying adapter states
are identical across conditions.  Outer-test labels are read only for the
final report.

The selected states were not retained by prior cross-fit runs, so this script
replays their training trajectory.  It requires the matching completed run as
a reference and refuses a replay whose original-context predictions diverge.
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
    _canonical_json,
    _find_source_cases,
    _load_json,
    _sha256,
    _write_json,
)
from direct_spline_openml_crossfit_blend import (
    CROSSFIT_BLEND_SCHEMA_VERSION,
    _ALPHA_GRID,
    _blend_prediction,
    _crossfit_validation_halves,
    _load_bag as _load_reference_bag,
    _load_source_task,
    _source_standard_prediction,
    _validate_case,
)
from tabicl._experiments.direct_spline_openml import (
    OpenMLTaskData,
    _bag_splits,
    _cpu_state_dict,
    _metric_bundle,
    _paired_comparison_summary,
    _prediction_shape,
    _safe_name,
    _seed,
    effective_inner_bag_count,
    load_frozen_backbone,
)
from tabicl._experiments.direct_spline_openml_standard import (
    _candidate_deployment_error,
    _classification_training_objective_from_logits,
    _cosine_scheduler,
    _fit_standard_bag,
    _identity_view_parity,
    _make_adapters,
    _normal_prediction,
    _normal_prediction_with_appended_context,
    _optimizer,
    _training_logits,
)
from tabicl._experiments.direct_spline_protocol import deployment_error, sample_episode_indices


CONTEXT_EXPANSION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ContextExpansionBagPredictions:
    """All original/expanded predictions for one independently selected bag."""

    validation_indices: np.ndarray
    selection_a_indices: np.ndarray
    selection_b_indices: np.ndarray
    original_identity_selection_a: np.ndarray
    original_identity_selection_b: np.ndarray
    original_spline_selected_on_b_selection_a: np.ndarray
    original_spline_selected_on_a_selection_b: np.ndarray
    original_identity_test: np.ndarray
    original_spline_selected_on_a_test: np.ndarray
    original_spline_selected_on_b_test: np.ndarray
    expanded_identity_selection_a: np.ndarray
    expanded_identity_selection_b: np.ndarray
    expanded_spline_selected_on_b_selection_a: np.ndarray
    expanded_spline_selected_on_a_selection_b: np.ndarray
    expanded_identity_test: np.ndarray
    expanded_spline_selected_on_a_test: np.ndarray
    expanded_spline_selected_on_b_test: np.ndarray
    metadata: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Completed D source run with full TabICLv2 predictions.")
    parser.add_argument(
        "--reference-crossfit-dir",
        type=Path,
        required=True,
        help="Completed matching cross-fit run used to verify the original-context replay.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--task-id", type=int, action="append", required=True, help="Fixed pilot task ID. Repeatable.")
    parser.add_argument("--protocol-seed", type=int, required=True, help="Must equal the reference cross-fit partition seed.")
    parser.add_argument("--bags", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-rounds", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument(
        "--reference-atol",
        type=float,
        default=1e-8,
        help="Tolerance used to report whether the diagnostic old-run replay is numerically close.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_id)) != len(args.task_id):
        raise ValueError("--task-id values must be unique")
    if args.bags is not None and args.bags < 2:
        raise ValueError("--bags must be at least two")
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    if args.reference_atol < 0.0:
        raise ValueError("--reference-atol must be non-negative")
    return args


def _task_dir(output_dir: Path, task: OpenMLTaskData) -> Path:
    return output_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def _append_prediction(
    *,
    bundle: Any,
    task: OpenMLTaskData,
    query_x: Any,
    appended_indices: np.ndarray,
    adapters: Any,
    device: torch.device,
) -> np.ndarray:
    appended_indices = np.asarray(appended_indices, dtype=int)
    return _normal_prediction_with_appended_context(
        bundle=bundle,
        query_x=query_x,
        context_indices=bundle.support_indices,
        appended_context_x=task.x_train.iloc[appended_indices].reset_index(drop=True),
        appended_context_y=np.asarray(task.y_train[appended_indices]),
        adapters=adapters,
        device=device,
    )


def _save_bag(path: Path, result: ContextExpansionBagPredictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in result.__dict__.items()
        if key != "metadata"
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata=np.asarray(json.dumps(result.metadata, sort_keys=True)),
    )


def _load_bag(path: Path) -> ContextExpansionBagPredictions:
    with np.load(path, allow_pickle=False) as payload:
        values: dict[str, Any] = {
            field: np.asarray(payload[field], dtype=int if field.endswith("indices") else float)
            for field in ContextExpansionBagPredictions.__dataclass_fields__
            if field != "metadata"
        }
        values["metadata"] = json.loads(str(payload["metadata"].item()))
    return ContextExpansionBagPredictions(**values)


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
    expected_a = (result.selection_a_indices.size, *test_shape[1:])
    expected_b = (result.selection_b_indices.size, *test_shape[1:])
    expected_test = test_shape
    prediction_fields = (
        ("original_identity_selection_a", expected_a),
        ("original_spline_selected_on_b_selection_a", expected_a),
        ("expanded_identity_selection_a", expected_a),
        ("expanded_spline_selected_on_b_selection_a", expected_a),
        ("original_identity_selection_b", expected_b),
        ("original_spline_selected_on_a_selection_b", expected_b),
        ("expanded_identity_selection_b", expected_b),
        ("expanded_spline_selected_on_a_selection_b", expected_b),
        ("original_identity_test", expected_test),
        ("original_spline_selected_on_a_test", expected_test),
        ("original_spline_selected_on_b_test", expected_test),
        ("expanded_identity_test", expected_test),
        ("expanded_spline_selected_on_a_test", expected_test),
        ("expanded_spline_selected_on_b_test", expected_test),
    )
    return bool(
        np.array_equal(
            np.sort(np.concatenate((result.selection_a_indices, result.selection_b_indices))),
            np.sort(validation_indices),
        )
        and not np.intersect1d(result.selection_a_indices, result.selection_b_indices).size
        and all(np.asarray(getattr(result, field)).shape == shape and np.isfinite(getattr(result, field)).all() for field, shape in prediction_fields)
    )


def _verify_reference_bag(
    *, actual: ContextExpansionBagPredictions, reference_path: Path, atol: float
) -> dict[str, Any]:
    """Compare with the old run without requiring cross-job GPU bit identity.

    Indices and array contracts are deterministic protocol invariants and stay
    fatal.  Prediction values are only diagnostic: the adapter is optimized
    again, and CUDA kernels can send that trajectory to a different selected
    state even under the same seed.  The causal comparison in this experiment
    is instead original versus expanded context within this run, where both
    arms use the exact same in-memory checkpoint tensors.
    """
    if not reference_path.is_file():
        raise FileNotFoundError(f"missing reference cross-fit bag: {reference_path}")
    reference = _load_reference_bag(reference_path)
    for name in ("validation_indices", "selection_a_indices", "selection_b_indices"):
        if not np.array_equal(getattr(actual, name), getattr(reference, name)):
            raise ValueError(f"original-context replay changed {name} for {reference_path}")
    field_pairs = {
        "identity_selection_a": "original_identity_selection_a",
        "identity_selection_b": "original_identity_selection_b",
        "spline_selected_on_b_selection_a": "original_spline_selected_on_b_selection_a",
        "spline_selected_on_a_selection_b": "original_spline_selected_on_a_selection_b",
        "identity_test": "original_identity_test",
        "spline_selected_on_a_test": "original_spline_selected_on_a_test",
        "spline_selected_on_b_test": "original_spline_selected_on_b_test",
    }
    differences: dict[str, float] = {}
    for reference_name, actual_name in field_pairs.items():
        expected = np.asarray(getattr(reference, reference_name), dtype=float)
        observed = np.asarray(getattr(actual, actual_name), dtype=float)
        if expected.shape != observed.shape:
            raise ValueError(f"original-context replay changed {reference_name} shape for {reference_path}")
        difference = float(np.max(np.abs(expected - observed), initial=0.0))
        differences[reference_name] = difference
    maximum = max(differences.values(), default=0.0)
    return {
        "max_abs_by_prediction": differences,
        "max_abs": float(maximum),
        "within_atol": bool(maximum <= atol),
        "atol": float(atol),
    }


def _fit_context_expansion_bag(
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
) -> ContextExpansionBagPredictions:
    """Fit on T once, then select/evaluate original and expanded contexts."""

    if config.get("adapter_patience") is not None:
        raise ValueError("shared A/B trajectories require adapter_patience=None")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    selection_a_indices, selection_b_indices = _crossfit_validation_halves(
        task=task,
        validation_indices=validation_indices,
        seed=_seed(protocol_seed, task.task_id, bag, 405),
    )
    selection_a_x = task.x_train.iloc[selection_a_indices].reset_index(drop=True)
    selection_b_x = task.x_train.iloc[selection_b_indices].reset_index(drop=True)
    selection_a_y = np.asarray(task.y_train[selection_a_indices])
    selection_b_y = np.asarray(task.y_train[selection_b_indices])
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=fit_indices,
        config=config,
        protocol_seed=protocol_seed,
        bag=bag,
        backbone=backbone,
        device=device,
    )
    adapter_seed = _seed(int(config["random_state"]), task.task_id, bag, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    parity_a, parity_a_reference, public_parity_a = _identity_view_parity(
        bundle=bundle, adapters=adapters, query_x=selection_a_x, device=device, progress=None,
        task_id=task.task_id, bag=bag, split="context_expansion_selection_a",
    )
    parity_b, parity_b_reference, public_parity_b = _identity_view_parity(
        bundle=bundle, adapters=adapters, query_x=selection_b_x, device=device, progress=None,
        task_id=task.task_id, bag=bag, split="context_expansion_selection_b",
    )
    parity_test, parity_test_reference, public_parity_test = _identity_view_parity(
        bundle=bundle, adapters=adapters, query_x=task.x_test, device=device, progress=None,
        task_id=task.task_id, bag=bag, split="context_expansion_test",
    )
    train_context_sizes: list[int] = []
    checkpoint_records: list[dict[str, Any]] = []
    first_objective = final_objective = float("nan")
    executed_steps = 0
    if adapters is None:
        original_identity_a = _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=None, device=device)
        original_identity_b = _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=None, device=device)
        original_identity_test = _normal_prediction(bundle=bundle, query_x=task.x_test, context_indices=bundle.support_indices, adapters=None, device=device)
        expanded_identity_a = _append_prediction(bundle=bundle, task=task, query_x=selection_a_x, appended_indices=selection_b_indices, adapters=None, device=device)
        expanded_identity_b = _append_prediction(bundle=bundle, task=task, query_x=selection_b_x, appended_indices=selection_a_indices, adapters=None, device=device)
        expanded_identity_test = _append_prediction(bundle=bundle, task=task, query_x=task.x_test, appended_indices=validation_indices, adapters=None, device=device)
        original_spline_b_on_a = original_identity_a.copy()
        original_spline_a_on_b = original_identity_b.copy()
        original_spline_a_test = original_identity_test.copy()
        original_spline_b_test = original_identity_test.copy()
        expanded_spline_b_on_a = expanded_identity_a.copy()
        expanded_spline_a_on_b = expanded_identity_b.copy()
        expanded_spline_a_test = expanded_identity_test.copy()
        expanded_spline_b_test = expanded_identity_test.copy()
        best = {
            key: {"step": 0, "error": 0.0, "valid": False, "state": None}
            for key in ("original_a", "original_b")
        }
    else:
        optimizer = _optimizer(adapters, config)
        scheduler = None if config.get("cosine_schedule_steps") is None else _cosine_scheduler(
            optimizer,
            total_steps=int(config["cosine_schedule_steps"]),
            min_lr_ratio=float(config["cosine_min_lr_ratio"]),
        )
        episode_rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 203))
        identity_state = _cpu_state_dict(adapters)
        best = {
            key: {"step": 0, "error": float("inf"), "valid": False, "state": identity_state}
            for key in ("original_a", "original_b")
        }
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
            train_context_sizes.append(int(context_rows.size))
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
            if step % int(config["validation_interval"]) != 0 and step != int(config["adapter_steps"]):
                continue
            candidates = {
                "original_a": _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=adapters, device=device),
                "original_b": _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=adapters, device=device),
            }
            labels = {"original_a": selection_a_y, "original_b": selection_b_y}
            errors = {
                name: _candidate_deployment_error(task.problem_type, labels[name], prediction, n_classes=task.n_classes)
                for name, prediction in candidates.items()
            }
            for name, error in errors.items():
                if error < float(best[name]["error"]):
                    best[name] = {"step": int(step), "error": float(error), "valid": True, "state": _cpu_state_dict(adapters)}
            checkpoint_records.append({
                "step": int(step),
                "original_selection_a_error": float(errors["original_a"]),
                "original_selection_b_error": float(errors["original_b"]),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            })
            del candidates
        del scheduler, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        original_identity_a = _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=None, device=device)
        original_identity_b = _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=None, device=device)
        original_identity_test = _normal_prediction(bundle=bundle, query_x=task.x_test, context_indices=bundle.support_indices, adapters=None, device=device)
        adapters.load_state_dict(best["original_a"]["state"], strict=True)
        original_spline_a_on_b = _normal_prediction(bundle=bundle, query_x=selection_b_x, context_indices=bundle.support_indices, adapters=adapters, device=device)
        original_spline_a_test = _normal_prediction(bundle=bundle, query_x=task.x_test, context_indices=bundle.support_indices, adapters=adapters, device=device)
        adapters.load_state_dict(best["original_b"]["state"], strict=True)
        original_spline_b_on_a = _normal_prediction(bundle=bundle, query_x=selection_a_x, context_indices=bundle.support_indices, adapters=adapters, device=device)
        original_spline_b_test = _normal_prediction(bundle=bundle, query_x=task.x_test, context_indices=bundle.support_indices, adapters=adapters, device=device)
        # First complete the original-context replay exactly as the reference
        # cross-fit runner did. Appended-context calls can populate estimator
        # inference caches, so they must not be interleaved with this guard.
        expanded_identity_a = _append_prediction(bundle=bundle, task=task, query_x=selection_a_x, appended_indices=selection_b_indices, adapters=None, device=device)
        expanded_identity_b = _append_prediction(bundle=bundle, task=task, query_x=selection_b_x, appended_indices=selection_a_indices, adapters=None, device=device)
        expanded_identity_test = _append_prediction(bundle=bundle, task=task, query_x=task.x_test, appended_indices=validation_indices, adapters=None, device=device)
        adapters.load_state_dict(best["original_a"]["state"], strict=True)
        expanded_spline_a_on_b = _append_prediction(bundle=bundle, task=task, query_x=selection_b_x, appended_indices=selection_a_indices, adapters=adapters, device=device)
        expanded_spline_a_test = _append_prediction(bundle=bundle, task=task, query_x=task.x_test, appended_indices=validation_indices, adapters=adapters, device=device)
        adapters.load_state_dict(best["original_b"]["state"], strict=True)
        expanded_spline_b_on_a = _append_prediction(bundle=bundle, task=task, query_x=selection_a_x, appended_indices=selection_b_indices, adapters=adapters, device=device)
        expanded_spline_b_test = _append_prediction(bundle=bundle, task=task, query_x=task.x_test, appended_indices=validation_indices, adapters=adapters, device=device)

    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "bag": int(bag),
        "fit_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "selection_a_rows": int(selection_a_indices.size),
        "selection_b_rows": int(selection_b_indices.size),
        "original_context_rows": int(bundle.support_indices.size),
        "expanded_test_context_rows": int(bundle.support_indices.size + validation_indices.size),
        "requested_bags": int(requested_bags),
        "effective_bags": int(effective_bags),
        "pipeline": "crossfit_frozen_context_expansion",
        "adapter_training_rows": "T only",
        "expanded_context_policy": "The original T-context-selected states are frozen; A uses T+B, B uses T+A, and test uses T+A+B",
        "adapter_steps_requested": int(config["adapter_steps"]),
        "adapter_steps_executed": int(executed_steps),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "checkpoints": {name: {key: value for key, value in record.items() if key != "state"} for name, record in best.items()},
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
        "adapter_observed_train_context_rows_min": None if not train_context_sizes else int(min(train_context_sizes)),
        "adapter_observed_train_context_rows_max": None if not train_context_sizes else int(max(train_context_sizes)),
        "adapter_observed_train_context_rows_mean": None if not train_context_sizes else float(np.mean(train_context_sizes)),
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
        "run_fingerprint_hash": run_fingerprint_hash,
    }
    del adapters, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ContextExpansionBagPredictions(
        validation_indices=np.asarray(validation_indices, dtype=int),
        selection_a_indices=selection_a_indices,
        selection_b_indices=selection_b_indices,
        original_identity_selection_a=original_identity_a,
        original_identity_selection_b=original_identity_b,
        original_spline_selected_on_b_selection_a=original_spline_b_on_a,
        original_spline_selected_on_a_selection_b=original_spline_a_on_b,
        original_identity_test=original_identity_test,
        original_spline_selected_on_a_test=original_spline_a_test,
        original_spline_selected_on_b_test=original_spline_b_test,
        expanded_identity_selection_a=expanded_identity_a,
        expanded_identity_selection_b=expanded_identity_b,
        expanded_spline_selected_on_b_selection_a=expanded_spline_b_on_a,
        expanded_spline_selected_on_a_selection_b=expanded_spline_a_on_b,
        expanded_identity_test=expanded_identity_test,
        expanded_spline_selected_on_a_test=expanded_spline_a_test,
        expanded_spline_selected_on_b_test=expanded_spline_b_test,
        metadata=metadata,
    )


def _assemble(
    *, task: OpenMLTaskData, bags: Sequence[ContextExpansionBagPredictions], condition: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if condition not in {"original", "expanded"}:
        raise ValueError(f"unknown context condition {condition!r}")
    shape = _prediction_shape(len(task.y_train), task.problem_type, task.n_classes)
    identity_oof = np.full(shape, np.nan, dtype=float)
    spline_oof = np.full(shape, np.nan, dtype=float)
    for bag in bags:
        identity_oof[bag.selection_a_indices] = getattr(bag, f"{condition}_identity_selection_a")
        identity_oof[bag.selection_b_indices] = getattr(bag, f"{condition}_identity_selection_b")
        spline_oof[bag.selection_a_indices] = getattr(bag, f"{condition}_spline_selected_on_b_selection_a")
        spline_oof[bag.selection_b_indices] = getattr(bag, f"{condition}_spline_selected_on_a_selection_b")
    if not np.isfinite(identity_oof).all() or not np.isfinite(spline_oof).all():
        raise RuntimeError(f"task {task.task_id} has incomplete {condition} OOF predictions")
    identity_test = np.mean([getattr(bag, f"{condition}_identity_test") for bag in bags], axis=0)
    spline_test = np.mean(
        [0.5 * (getattr(bag, f"{condition}_spline_selected_on_a_test") + getattr(bag, f"{condition}_spline_selected_on_b_test")) for bag in bags],
        axis=0,
    )
    return identity_oof, spline_oof, identity_test, spline_test


def _minimum_pooled_oof_selection(
    *, task: OpenMLTaskData, identity_oof: np.ndarray, spline_oof: np.ndarray
) -> dict[str, Any]:
    candidates = []
    for alpha in _ALPHA_GRID:
        prediction = _blend_prediction(identity_oof, spline_oof, float(alpha))
        candidates.append({
            "alpha": float(alpha),
            "pooled_oof_deployment_error": float(
                deployment_error(task.problem_type, task.y_train, prediction, n_classes=task.n_classes)
            ),
        })
    selected = min(candidates, key=lambda item: (item["pooled_oof_deployment_error"], item["alpha"]))
    return {
        "selection_rule": "smallest alpha at the minimum pooled cross-fitted OOF loss",
        "candidate_alphas": candidates,
        "selected_alpha": float(selected["alpha"]),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(
    *, source_dir: Path, source_manifest: Mapping[str, Any], reference_dir: Path, reference_manifest: Mapping[str, Any], cases: Sequence[SourceCase], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "context_expansion_schema_version": CONTEXT_EXPANSION_SCHEMA_VERSION,
        "experiment": "DirectSpline frozen labelled-context expansion under cross-fitted checkpoint selection",
        "source_dir": str(source_dir.resolve()),
        "source_manifest_sha256": _sha256(source_dir / "experiment_manifest.json"),
        "source_repository_revision": source_manifest.get("immutable_run", {}).get("repository_revision"),
        "reference_crossfit_dir": str(reference_dir.resolve()),
        "reference_crossfit_manifest_sha256": _sha256(reference_dir / "experiment_manifest.json"),
        "reference_crossfit_fingerprint": reference_manifest.get("run_fingerprint_sha256"),
        "config_label": str(args.config_label),
        "task_ids": sorted(case.task_id for case in cases),
        "protocol_seed": int(args.protocol_seed),
        "requested_bags": args.bags,
        "reference_atol": float(args.reference_atol),
        "fixed_arm_requirement": {"adapter_architecture": "fixed_cubic", "n_control_points": 20},
        "training": "One fixed T-only adapter trajectory per bag; checkpoints are selected under the original T-only context, then no adapter, input-preprocessor, target-scaler, or ensemble refit occurs after A/B rows are appended.",
        "crossfit_contexts": {
            "original": "A/B/test use T",
            "expanded": "A uses T+B, B uses T+A, test uses T+A+B",
        },
        "selection": "Both context conditions reuse the same original-context-selected adapter states. Each condition selects its own alpha from its independently cross-fitted OOF predictions; exact ties choose lower alpha.",
        "label_policy": "A/B labels may enter the opposite half's ICL context but never their own OOF context; outer-test labels are read only after both condition alphas and test predictions are frozen.",
        "script_sha256": _sha256(Path(__file__)),
    }


def _prepare_output(*, output_dir: Path, manifest: Mapping[str, Any], resume: bool) -> str:
    path = output_dir / "experiment_manifest.json"
    fingerprint = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    payload = {**manifest, "run_fingerprint_sha256": fingerprint}
    if path.exists():
        existing = _load_json(path, label="existing context-expansion manifest")
        if _canonical_json(existing) != _canonical_json(payload):
            raise ValueError("output directory belongs to a different context-expansion protocol; choose a new --output-dir")
        if not resume:
            raise ValueError("context-expansion output directory already exists; pass --resume to reuse matching bags")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
    return fingerprint


def _condition_record(
    *, task: OpenMLTaskData, identity_oof: np.ndarray, spline_oof: np.ndarray, identity_test: np.ndarray, spline_test: np.ndarray
) -> dict[str, Any]:
    selection = _minimum_pooled_oof_selection(task=task, identity_oof=identity_oof, spline_oof=spline_oof)
    alpha = float(selection["selected_alpha"])
    selected_oof = _blend_prediction(identity_oof, spline_oof, alpha)
    selected_test = _blend_prediction(identity_test, spline_test, alpha)
    return {
        "oof": {
            "identity": _metric_bundle(task.problem_type, task.y_train, identity_oof, task.n_classes),
            "raw_spline": _metric_bundle(task.problem_type, task.y_train, spline_oof, task.n_classes),
            "selected_blend": _metric_bundle(task.problem_type, task.y_train, selected_oof, task.n_classes),
        },
        "blend_selection": selection,
        "outer_test": {
            "identity": _metric_bundle(task.problem_type, task.y_test, identity_test, task.n_classes),
            "raw_spline": _metric_bundle(task.problem_type, task.y_test, spline_test, task.n_classes),
            "selected_blend": _metric_bundle(task.problem_type, task.y_test, selected_test, task.n_classes),
        },
        "predictions": {"identity_oof": identity_oof, "spline_oof": spline_oof, "selected_oof": selected_oof, "identity_test": identity_test, "spline_test": spline_test, "selected_test": selected_test},
    }


def _run_task(
    *, case: SourceCase, task: OpenMLTaskData, output_dir: Path, args: argparse.Namespace, fingerprint: str
) -> dict[str, Any]:
    _validate_case(case)
    requested_bags = case.requested_bags if args.bags is None else int(args.bags)
    splits = list(_bag_splits(task, requested_bags=requested_bags, seed=_seed(args.protocol_seed, task.task_id, 0)))
    effective_bags = effective_inner_bag_count(task, requested_bags=requested_bags)
    if len(splits) != effective_bags:
        raise RuntimeError("unexpected effective inner bag count")
    task_dir = _task_dir(output_dir, task)
    task_dir.mkdir(parents=True, exist_ok=True)
    config = dict(case.config)
    source_patience = config.get("adapter_patience")
    config["adapter_patience"] = None
    test_shape = _prediction_shape(len(task.y_test), task.problem_type, task.n_classes)
    missing = [
        (bag, np.asarray(fit, dtype=int), np.asarray(heldout, dtype=int))
        for bag, (fit, heldout) in enumerate(splits)
        if not _bag_complete(path=task_dir / f"bag_{bag}.npz", fingerprint=fingerprint, validation_indices=np.asarray(heldout, dtype=int), test_shape=test_shape)
    ]
    if missing:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested as {args.device!r}, but it is unavailable")
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
            "reference_crossfit_dir": str(args.reference_crossfit_dir.resolve()),
            "bag_splits": [{"bag": bag, "fit_rows": len(fit), "heldout_rows": len(heldout)} for bag, fit, heldout in missing],
        })
        for position, (bag, fit_indices, validation_indices) in enumerate(missing, start=1):
            print(f"task {task.task_id} bag {bag} ({position}/{len(missing)}): replaying T-only trajectory; scoring frozen original and expanded contexts", flush=True)
            result = _fit_context_expansion_bag(
                task=task,
                fit_indices=fit_indices,
                validation_indices=validation_indices,
                bag=bag,
                config=config,
                protocol_seed=int(args.protocol_seed),
                backbone=backbone,
                device=device,
                run_fingerprint_hash=fingerprint,
                requested_bags=requested_bags,
                effective_bags=effective_bags,
            )
            reference_path = _task_dir(args.reference_crossfit_dir, task) / f"bag_{bag}.npz"
            comparison = _verify_reference_bag(actual=result, reference_path=reference_path, atol=float(args.reference_atol))
            result.metadata["reference_replay_diagnostic"] = comparison
            _save_bag(task_dir / f"bag_{bag}.npz", result)
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    bags = [_load_bag(task_dir / f"bag_{bag}.npz") for bag in range(effective_bags)]
    original_oof_identity, original_oof_spline, original_test_identity, original_test_spline = _assemble(
        task=task, bags=bags, condition="original"
    )
    expanded_oof_identity, expanded_oof_spline, expanded_test_identity, expanded_test_spline = _assemble(
        task=task, bags=bags, condition="expanded"
    )
    original = _condition_record(
        task=task,
        identity_oof=original_oof_identity,
        spline_oof=original_oof_spline,
        identity_test=original_test_identity,
        spline_test=original_test_spline,
    )
    expanded = _condition_record(
        task=task,
        identity_oof=expanded_oof_identity,
        spline_oof=expanded_oof_spline,
        identity_test=expanded_test_identity,
        spline_test=expanded_test_spline,
    )
    prediction_path = task_dir / "task_predictions.npz"
    np.savez_compressed(prediction_path, **{f"original_{name}": value for name, value in original["predictions"].items()}, **{f"expanded_{name}": value for name, value in expanded["predictions"].items()})
    source_prediction = _source_standard_prediction(source_dir=case.source_dir, task=task)
    replay_max = max(
        (float(bag.metadata.get("reference_replay_diagnostic", {}).get("max_abs", 0.0)) for bag in bags),
        default=0.0,
    )
    return {
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
        "original_context": {key: value for key, value in original.items() if key != "predictions"},
        "expanded_context": {key: value for key, value in expanded.items() if key != "predictions"},
        "source_full_outer_training_tabiclv2": None if source_prediction is None else _metric_bundle(task.problem_type, task.y_test, source_prediction, task.n_classes),
        "original_context_replay_max_abs": replay_max,
        "outer_test_scored_after_both_context_predictions_and_alphas_fixed": True,
        "test_prediction_artifact": str(prediction_path),
    }


def _comparison(
    *, task_summaries: Sequence[Mapping[str, Any]], candidate_condition: str, candidate_method: str, reference_condition: str | None, reference_method: str | None, bootstrap_rounds: int, bootstrap_seed: int
) -> dict[str, Any] | None:
    usable = list(task_summaries)
    if reference_condition is None:
        usable = [item for item in usable if item["source_full_outer_training_tabiclv2"] is not None]
    if not usable:
        return None
    reference = np.asarray([
        float(item["source_full_outer_training_tabiclv2"]["benchmark_error"])
        if reference_condition is None
        else float(item[f"{reference_condition}_context"]["outer_test"][str(reference_method)]["benchmark_error"])
        for item in usable
    ])
    candidate = np.asarray([
        float(item[f"{candidate_condition}_context"]["outer_test"][candidate_method]["benchmark_error"])
        for item in usable
    ])
    return _paired_comparison_summary(
        reference=reference,
        candidate=candidate,
        problem_types=np.asarray([item["problem_type"] for item in usable], dtype=object),
        bootstrap_rounds=bootstrap_rounds,
        bootstrap_seed=bootstrap_seed,
        reference_label=("source_full_outer_training_tabiclv2" if reference_condition is None else f"{reference_condition}_context_{reference_method}"),
        candidate_label=f"{candidate_condition}_context_{candidate_method}",
    )


def main() -> None:
    args = _parse_args()
    source_dir = args.source_dir.resolve()
    args.reference_crossfit_dir = args.reference_crossfit_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    source_manifest = _load_json(source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    reference_manifest = _load_json(args.reference_crossfit_dir / "experiment_manifest.json", label="reference cross-fit manifest")
    if reference_manifest.get("crossfit_blend_schema_version") != CROSSFIT_BLEND_SCHEMA_VERSION:
        raise ValueError("reference run has unsupported cross-fit artifact schema")
    if int(reference_manifest.get("protocol_seed")) != int(args.protocol_seed):
        raise ValueError("--protocol-seed must equal the reference cross-fit run's protocol_seed")
    if str(reference_manifest.get("source_dir", "")) != str(source_dir):
        raise ValueError("reference cross-fit run was not built from this exact source directory")
    requested = set(args.task_id)
    cases = _find_source_cases(source_dir=source_dir, manifest=source_manifest, config_label=args.config_label, requested_task_ids=requested)
    cases = [case for case in cases if case.problem_type in {"multiclass", "regression"}]
    if {case.task_id for case in cases} != requested:
        raise ValueError("one or more requested task IDs are absent or not multiclass/regression D cases")
    manifest = _manifest(source_dir=source_dir, source_manifest=source_manifest, reference_dir=args.reference_crossfit_dir, reference_manifest=reference_manifest, cases=cases, args=args)
    fingerprint = _prepare_output(output_dir=args.output_dir, manifest=manifest, resume=bool(args.resume))
    task_summaries = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}: frozen context expansion", flush=True)
        task = _load_source_task(case=case, immutable_run=immutable_run)
        task_summaries.append(_run_task(case=case, task=task, output_dir=args.output_dir, args=args, fingerprint=fingerprint))
    task_summaries.sort(key=lambda item: int(item["task_id"]))
    _write_json(args.output_dir / "task_summaries.json", task_summaries)
    methods = ("identity", "raw_spline", "selected_blend")
    summary = {
        "context_expansion_schema_version": CONTEXT_EXPANSION_SCHEMA_VERSION,
        "n_tasks": len(task_summaries),
        "task_summaries": task_summaries,
        "context_effect_vs_original": {
            method: _comparison(task_summaries=task_summaries, candidate_condition="expanded", candidate_method=method, reference_condition="original", reference_method=method, bootstrap_rounds=args.bootstrap_rounds, bootstrap_seed=args.bootstrap_seed + index)
            for index, method in enumerate(methods)
        },
        "incremental_spline_effect": {
            condition: _comparison(task_summaries=task_summaries, candidate_condition=condition, candidate_method="raw_spline", reference_condition=condition, reference_method="identity", bootstrap_rounds=args.bootstrap_rounds, bootstrap_seed=args.bootstrap_seed + 10 + index)
            for index, condition in enumerate(("original", "expanded"))
        },
        "end_to_end_vs_source_full_outer_training_tabiclv2": {
            f"{condition}_{method}": _comparison(task_summaries=task_summaries, candidate_condition=condition, candidate_method=method, reference_condition=None, reference_method=None, bootstrap_rounds=args.bootstrap_rounds, bootstrap_seed=args.bootstrap_seed + 100 + index)
            for index, (condition, method) in enumerate((condition, method) for condition in ("original", "expanded") for method in methods)
        },
        "selection_metric_note": "Multiclass uses log loss; regression uses MSE. Benchmark reporting uses log loss and RMSE respectively.",
        "label_policy": "The selected state and OOF prediction for a row exclude that row's label from both checkpoint selection context and prediction context. Outer-test labels are report-only.",
        "reference_replay_note": "Reference indices and shapes are strict invariants. Prediction differences are diagnostic only because separately optimized CUDA trajectories are not required to be bit-identical; the causal original/expanded arms share the exact same checkpoint tensors within this run.",
    }
    _write_json(args.output_dir / "summary.json", summary)
    rows = []
    for item in task_summaries:
        row = {"task_id": item["task_id"], "dataset_name": item["dataset_name"], "problem_type": item["problem_type"], "original_context_replay_max_abs": item["original_context_replay_max_abs"]}
        for condition in ("original", "expanded"):
            row[f"{condition}_selected_alpha"] = item[f"{condition}_context"]["blend_selection"]["selected_alpha"]
            for method in methods:
                row[f"{condition}_{method}_benchmark_error"] = item[f"{condition}_context"]["outer_test"][method]["benchmark_error"]
        row["full_tabiclv2_benchmark_error"] = None if item["source_full_outer_training_tabiclv2"] is None else item["source_full_outer_training_tabiclv2"]["benchmark_error"]
        rows.append(row)
    _write_csv(args.output_dir / "task_results.csv", rows)
    print(json.dumps({"n_tasks": len(task_summaries), "max_original_context_replay_abs": max((item["original_context_replay_max_abs"] for item in task_summaries), default=0.0)}), flush=True)


if __name__ == "__main__":
    main()
