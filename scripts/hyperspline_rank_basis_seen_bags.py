"""Shared known-dataset HyperSpline with unseen contexts and untouched queries.

This is the missing bridge between one-conditioner-per-dataset feasibility and
dataset-zero-shot transfer.  One shared conditioner trains on bags from every
named dataset, but validation and final evaluation use disjoint episodes made
from that outer seed's untouched holdout.  Thus dataset identities/statistical
families are known while context rows and query labels are unseen.

The rank basis is fitted from *training-bag teacher curves only*.  Teacher
parameters are never training targets; the conditioner learns exclusively
through frozen-TabICL query cross entropy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn

from tabicl._hyperspline import summarize_context

try:
    from scripts.direct_spline_function_basis import TeacherBag, fit_pca
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.direct_spline_dataset_headroom import release_cuda
    from scripts.hyperspline_rank_basis_zero_shot import (
        FactorizedRankBasisSpline,
        metrics,
        parameter_metrics,
        tensors,
        write_csv,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_function_basis import TeacherBag, fit_pca
    from direct_spline_multidataset_headroom import load_backbone
    from direct_spline_dataset_headroom import release_cuda
    from hyperspline_rank_basis_zero_shot import (
        FactorizedRankBasisSpline,
        metrics,
        parameter_metrics,
        tensors,
        write_csv,
    )


@dataclass
class EvaluationEpisode:
    dataset: str
    seed: int
    bag: int
    support_x: torch.Tensor
    support_y: torch.Tensor
    guard_x: np.ndarray
    guard_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    curves: torch.Tensor | None = None
    source: str = "outer_holdout"


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    return values


def bag_key(bag: TeacherBag | EvaluationEpisode) -> tuple[str, int, int]:
    return bag.dataset, int(bag.seed), int(bag.bag)


def split_seen_bags(
    bags: Iterable[TeacherBag],
    *,
    datasets: set[str],
    outer_seed: int,
    train_bag_ids: set[int],
    validation_bag_ids: set[int],
    test_bag_ids: set[int],
) -> tuple[list[TeacherBag], list[TeacherBag], list[TeacherBag]]:
    """Deterministically split exact cache keys within one outer seed."""
    if train_bag_ids & validation_bag_ids or train_bag_ids & test_bag_ids or validation_bag_ids & test_bag_ids:
        raise ValueError("train, validation, and test bag IDs must be disjoint")
    selected = sorted(
        (bag for bag in bags if bag.dataset in datasets and int(bag.seed) == outer_seed),
        key=bag_key,
    )
    train = [bag for bag in selected if int(bag.bag) in train_bag_ids]
    validation = [bag for bag in selected if int(bag.bag) in validation_bag_ids]
    test = [bag for bag in selected if int(bag.bag) in test_bag_ids]
    expected = {
        "train": (train, train_bag_ids),
        "validation": (validation, validation_bag_ids),
        "test": (test, test_bag_ids),
    }
    for split, (items, ids) in expected.items():
        counts = {dataset: sum(item.dataset == dataset for item in items) for dataset in datasets}
        missing = {dataset: len(ids) - count for dataset, count in counts.items() if count != len(ids)}
        if missing:
            raise ValueError(f"incomplete {split} bags for outer seed {outer_seed}: {missing}")
    return train, validation, test


def fit_training_basis(train_bags: list[TeacherBag], rank: int):
    """Fit PCA from exact training keys; validation/test curves cannot leak."""
    curves = torch.cat([bag.curves for bag in train_bags])
    mean, components, explained = fit_pca(curves)
    if rank <= 0 or rank > components.shape[0]:
        raise ValueError(f"rank {rank} unavailable; maximum is {components.shape[0]}")
    components = components[:rank]
    payload = torch.cat((mean.flatten(), components.flatten())).contiguous().numpy().tobytes()
    return mean, components, explained[:rank], hashlib.sha256(payload).hexdigest()


def macro_dataset_mean(rows: list[dict], field: str = "loss") -> float:
    """Equal-weight datasets regardless of their bag/query counts."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(float(row[field]))
    if not grouped:
        raise ValueError("cannot aggregate an empty evaluation")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _stratified_split_indices(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(y.size)
    first, second = train_test_split(indices, test_size=fraction, random_state=seed, stratify=y)
    return np.sort(first), np.sort(second)


def _make_holdout_episode(
    dataset: str,
    outer_seed: int,
    episode_id: int,
    x: np.ndarray,
    y: np.ndarray,
    *,
    context_fraction: float,
    max_context_rows: int,
    seed: int,
    source: str,
) -> EvaluationEpisode:
    context_idx, query_idx = _stratified_split_indices(y, 1.0 - context_fraction, seed)
    if max_context_rows > 0 and context_idx.size > max_context_rows:
        keep, _ = train_test_split(
            context_idx,
            train_size=max_context_rows,
            random_state=seed + 17,
            stratify=y[context_idx],
        )
        context_idx = np.sort(keep)
    return EvaluationEpisode(
        dataset=dataset,
        seed=outer_seed,
        bag=episode_id,
        support_x=torch.as_tensor(x[context_idx], dtype=torch.float32).unsqueeze(0),
        support_y=torch.as_tensor(y[context_idx], dtype=torch.float32).unsqueeze(0),
        guard_x=np.asarray(x[query_idx], dtype=np.float32),
        guard_y=np.asarray(y[query_idx], dtype=np.int64),
        test_x=np.asarray(x[query_idx], dtype=np.float32),
        test_y=np.asarray(y[query_idx], dtype=np.int64),
        source=source,
    )


def make_outer_holdout_episodes(
    cache_bags: list[TeacherBag],
    *,
    datasets: set[str],
    outer_seed: int,
    validation_fraction: float,
    context_fraction: float,
    max_context_rows: int,
    split_seed: int,
) -> tuple[list[EvaluationEpisode], list[EvaluationEpisode], list[dict]]:
    """Create row-disjoint validation/final episodes from untouched outer tests."""
    validation, test, manifest_rows = [], [], []
    for dataset_index, dataset in enumerate(sorted(datasets)):
        items = sorted(
            [bag for bag in cache_bags if bag.dataset == dataset and int(bag.seed) == outer_seed],
            key=bag_key,
        )
        if not items:
            raise ValueError(f"no bags for {dataset}, outer seed {outer_seed}")
        reference_x = np.asarray(items[0].test_x, dtype=np.float32)
        reference_y = np.asarray(items[0].test_y, dtype=np.int64)
        for item in items[1:]:
            if not np.array_equal(reference_y, item.test_y) or not np.array_equal(reference_x, item.test_x):
                raise ValueError(f"outer test differs across bags for {dataset}, seed {outer_seed}")
        row_seed = split_seed + 100_000 * outer_seed + 1_000 * dataset_index
        val_idx, test_idx = _stratified_split_indices(reference_y, 1.0 - validation_fraction, row_seed)
        val_episode = _make_holdout_episode(
            dataset, outer_seed, -1, reference_x[val_idx], reference_y[val_idx],
            context_fraction=context_fraction, max_context_rows=max_context_rows,
            seed=row_seed + 1, source="outer_holdout_validation",
        )
        test_episode = _make_holdout_episode(
            dataset, outer_seed, -2, reference_x[test_idx], reference_y[test_idx],
            context_fraction=context_fraction, max_context_rows=max_context_rows,
            seed=row_seed + 2, source="outer_holdout_final",
        )
        validation.append(val_episode); test.append(test_episode)
        manifest_rows.append({
            "dataset": dataset,
            "outer_seed": outer_seed,
            "outer_rows": int(reference_y.size),
            "validation_pool_rows": int(val_idx.size),
            "validation_context_rows": int(val_episode.support_y.numel()),
            "validation_query_rows": int(val_episode.guard_y.size),
            "test_pool_rows": int(test_idx.size),
            "test_context_rows": int(test_episode.support_y.numel()),
            "test_query_rows": int(test_episode.test_y.size),
            "row_split_seed": row_seed,
        })
    return validation, test, manifest_rows


def reference_predictions(backbone, episode, split: str, device: torch.device):
    """Exact identity spline after ordinary context mean/std standardization."""
    context_x, context_y, query_x, query_y = tensors(episode, split, device)
    statistics = summarize_context(context_x)
    context_z = (context_x.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    query_z = (query_x.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    backbone.clear_cache()
    logits = backbone(torch.cat((context_z, query_z), dim=1), context_y.float().unsqueeze(0))
    return logits, query_y


def model_predictions(backbone, model, episode, split: str, device: torch.device):
    context_x, context_y, query_x, query_y = tensors(episode, split, device)
    backbone.clear_cache()
    transformed, parameters = model(context_x, context_y, query_x)
    logits = backbone(transformed, context_y.float().unsqueeze(0))
    return logits, query_y, parameters, context_x, context_y


@torch.no_grad()
def evaluate_reference(backbone, episodes, split, device, *, stage, step, run_fields):
    rows, probabilities = [], {}
    for episode in episodes:
        logits, target = reference_predictions(backbone, episode, split, device)
        probability = logits.softmax(-1).cpu()
        key = bag_key(episode)
        probabilities[key] = (probability, target.cpu())
        rows.append({**run_fields, "stage": stage, "step": step, "dataset": episode.dataset,
                     "outer_seed": episode.seed, "bag": episode.bag, "source": getattr(episode, "source", "cache"),
                     **metrics(probability, target.cpu())})
    return rows, probabilities


def _curve_diagnostics(model, episode, parameters) -> dict[str, float]:
    if getattr(episode, "curves", None) is None:
        return {}
    generated = (parameters["values"][0] - model.grid.view(1, -1)).detach().cpu()
    teacher = episode.curves.float().cpu()
    if generated.shape != teacher.shape:
        return {"teacher_curve_shape_mismatch": 1.0}
    error = generated - teacher
    per_column = error.square().mean(-1).sqrt()
    teacher_centered = teacher - teacher.mean()
    generated_centered = generated - generated.mean()
    cosine = F.cosine_similarity(generated_centered.flatten(), teacher_centered.flatten(), dim=0)
    oracle = model.mean_curve.cpu() + (teacher - model.mean_curve.cpu()) @ model.components.cpu().T @ model.components.cpu()
    result = {
        "teacher_curve_rmse": float(error.square().mean().sqrt()),
        "teacher_curve_column_median_rmse": float(per_column.median()),
        "teacher_curve_column_max_rmse": float(per_column.max()),
        "teacher_curve_cosine": float(cosine),
        "rank_basis_oracle_curve_rmse": float((oracle - teacher).square().mean().sqrt()),
    }
    teacher_state = getattr(episode, "teacher_state", None)
    if teacher_state and "location_offsets" in teacher_state and "log_scale_offsets" in teacher_state:
        teacher_location = torch.tanh(teacher_state["location_offsets"].float())[0].cpu()
        teacher_log_scale = (math.log(2.0) * torch.tanh(teacher_state["log_scale_offsets"].float()))[0].cpu()
        generated_location = parameters["location_delta"][0].detach().cpu()
        generated_log_scale = parameters["log_scale_delta"][0].detach().cpu()
        for name, generated_value, teacher_value in (
            ("location", generated_location, teacher_location),
            ("log_scale", generated_log_scale, teacher_log_scale),
        ):
            result[f"teacher_{name}_rmse"] = float((generated_value - teacher_value).square().mean().sqrt())
            result[f"teacher_{name}_zero_baseline_rmse"] = float(teacher_value.square().mean().sqrt())
            result[f"teacher_{name}_cosine"] = float(
                F.cosine_similarity(generated_value.flatten(), teacher_value.flatten(), dim=0)
            )
            result[f"teacher_{name}_sign_agreement"] = float(
                (generated_value.sign() == teacher_value.sign()).float().mean()
            )
    return result


@torch.no_grad()
def evaluate_model(backbone, model, episodes, split, device, *, stage, step, run_fields, collect=False):
    rows, probabilities, records = [], {}, []
    for episode in episodes:
        logits, target, parameters, context_x, context_y = model_predictions(backbone, model, episode, split, device)
        probability = logits.softmax(-1).cpu()
        key = bag_key(episode)
        probabilities[key] = (probability, target.cpu())
        diagnostics = parameter_metrics(parameters) | _curve_diagnostics(model, episode, parameters)
        rows.append({**run_fields, "stage": stage, "step": step, "dataset": episode.dataset,
                     "outer_seed": episode.seed, "bag": episode.bag, "source": getattr(episode, "source", "cache"),
                     **metrics(probability, target.cpu()), **diagnostics})
        if collect:
            summary = summarize_context(context_x, y_context=context_y.float().unsqueeze(0)).summary
            descriptor = torch.cat((summary.mean(1), summary.std(1, unbiased=False)), dim=-1)[0].cpu()
            records.append({
                "dataset": episode.dataset,
                "outer_seed": int(episode.seed),
                "bag": int(episode.bag),
                "source": getattr(episode, "source", "cache"),
                "curve": (parameters["values"][0] - model.grid.view(1, -1)).cpu(),
                "coefficients": parameters["coefficients"][0].cpu(),
                "shape_gate": parameters["shape_gate"][0].cpu(),
                "normalization_gate": parameters["normalization_gate"][0].cpu(),
                "location_delta": parameters["location_delta"][0].cpu(),
                "log_scale_delta": parameters["log_scale_delta"][0].cpu(),
                "descriptor": descriptor,
            })
    return rows, probabilities, records


def paired_rows(reference_rows: list[dict], selected_rows: list[dict], *, run_fields: dict) -> list[dict]:
    reference = {(row["dataset"], int(row["outer_seed"]), int(row["bag"])): row for row in reference_rows}
    result = []
    for row in selected_rows:
        key = row["dataset"], int(row["outer_seed"]), int(row["bag"])
        base = reference[key]
        result.append({**run_fields, "dataset": row["dataset"], "outer_seed": row["outer_seed"],
                       "bag": row["bag"], "source": row["source"],
                       **{f"{field}_delta": float(row[field]) - float(base[field])
                          for field in ("loss", "accuracy", "balanced_accuracy", "brier", "ece")}})
    return result


def consistency_rows(train_records: list[dict], test_records: list[dict], *, run_fields: dict) -> list[dict]:
    result = []
    for dataset in sorted({record["dataset"] for record in test_records}):
        train = [record for record in train_records if record["dataset"] == dataset]
        test = [record for record in test_records if record["dataset"] == dataset]
        if not train or not test:
            continue
        row = {**run_fields, "dataset": dataset, "n_train_contexts": len(train), "n_test_contexts": len(test)}
        for field in ("curve", "coefficients", "shape_gate", "normalization_gate", "location_delta", "log_scale_delta", "descriptor"):
            train_stack = torch.stack([record[field] for record in train])
            test_stack = torch.stack([record[field] for record in test])
            train_mean, test_mean = train_stack.mean(0), test_stack.mean(0)
            row[f"{field}_train_within_rms"] = float((train_stack - train_mean).square().mean().sqrt())
            row[f"{field}_test_within_rms"] = float((test_stack - test_mean).square().mean().sqrt())
            row[f"{field}_train_test_mean_rmse"] = float((train_mean - test_mean).square().mean().sqrt())
        result.append(row)
    return result


def parameter_column_rows(records: list[dict], *, run_fields: dict) -> list[dict]:
    """Flatten selected generated outputs for auditable per-column analysis."""
    rows = []
    for record in records:
        for column in range(record["curve"].shape[0]):
            curve = record["curve"][column]
            coefficients = record["coefficients"][column]
            row = {
                **run_fields,
                "dataset": record["dataset"], "outer_seed": record["outer_seed"],
                "bag": record["bag"], "source": record["source"], "column": column,
                "shape_gate": float(record["shape_gate"][column]),
                "normalization_gate": float(record["normalization_gate"][column]),
                "location_delta": float(record["location_delta"][column]),
                "log_scale_delta": float(record["log_scale_delta"][column]),
                "curve_deformation_rms": float(curve.square().mean().sqrt()),
                "curve_deformation_abs_max": float(curve.abs().max()),
                "coefficient_rms": float(coefficients.square().mean().sqrt()),
                "coefficient_abs_max": float(coefficients.abs().max()),
            }
            row.update({f"coefficient_{index}": float(value) for index, value in enumerate(coefficients)})
            rows.append(row)
    return rows


def load_historical_references(root: Path | None, branch: str) -> dict[tuple[str, int], float]:
    if root is None or not root.exists():
        return {}
    result = {}
    for path in root.glob("*/summary.json"):
        payload = json.loads(path.read_text(encoding="utf8"))
        branch_summary = payload.get("branches", {}).get(branch)
        if branch_summary is not None:
            result[(str(payload["dataset"]), int(payload["outer_seed"]))] = float(
                branch_summary["outer_test_minus_identity"]
            )
    return result


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[], help="Repeat; default uses every cached dataset.")
    parser.add_argument("--outer-seeds", type=parse_int_csv, default=parse_int_csv("0,1"))
    parser.add_argument("--model-seeds", type=parse_int_csv, default=parse_int_csv("0,1"))
    parser.add_argument("--train-bags", type=parse_int_csv, default=parse_int_csv("0,1,2,3,4,5"))
    parser.add_argument("--validation-bags", type=parse_int_csv, default=parse_int_csv("6"))
    parser.add_argument("--test-bags", type=parse_int_csv, default=parse_int_csv("7"))
    parser.add_argument("--outer-validation-fraction", type=float, default=0.40)
    parser.add_argument("--evaluation-context-fraction", type=float, default=0.50)
    parser.add_argument("--max-evaluation-context-rows", type=int, default=512)
    parser.add_argument("--episode-split-seed", type=int, default=70_001)
    parser.add_argument("--branch", choices=("shape", "normalization", "joint"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-single-dataset-root", type=Path)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--coefficient-bound", type=float, default=1.5)
    parser.add_argument("--location-bound", type=float, default=1.0)
    parser.add_argument("--log-scale-bound", type=float, default=1.0)
    parser.add_argument("--gate-initial-probability", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patience-validations", type=int, default=20)
    parser.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Skip run directories with complete summary.json files.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    return parser.parse_args()


def run_one(args, backbone, all_bags, datasets, *, outer_seed, model_seed, device, run_dir):
    train_bags, cached_validation, cached_test = split_seen_bags(
        all_bags, datasets=datasets, outer_seed=outer_seed,
        train_bag_ids=set(args.train_bags), validation_bag_ids=set(args.validation_bags),
        test_bag_ids=set(args.test_bags),
    )
    validation, final_test, episode_manifest = make_outer_holdout_episodes(
        all_bags, datasets=datasets, outer_seed=outer_seed,
        validation_fraction=args.outer_validation_fraction,
        context_fraction=args.evaluation_context_fraction,
        max_context_rows=args.max_evaluation_context_rows,
        split_seed=args.episode_split_seed,
    )
    mean, components, explained, basis_hash = fit_training_basis(train_bags, args.rank)
    run_fields = {"branch": args.branch, "outer_seed_run": outer_seed, "model_seed": model_seed}
    manifest = {
        "status": "running", "protocol": "known_datasets + train_bags + disjoint_outer_holdout_validation_test",
        "run": run_fields, "datasets": sorted(datasets),
        "train_keys": [bag_key(bag) for bag in train_bags],
        "cached_validation_keys": [bag_key(bag) for bag in cached_validation],
        "cached_test_keys": [bag_key(bag) for bag in cached_test],
        "episode_splits": episode_manifest,
        "basis_sha256": basis_hash,
        "basis_explained_variance": explained.tolist(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = FactorizedRankBasisSpline(
        mean.to(device), components.to(device), branch=args.branch, hidden_dim=args.hidden_dim,
        coefficient_bound=args.coefficient_bound, location_bound=args.location_bound,
        log_scale_bound=args.log_scale_bound, target_aware=args.target_aware,
        raw_context=args.raw_context, gate_initial_probability=args.gate_initial_probability,
    ).to(device)
    parameters = list(nn.Module.parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    rng = np.random.default_rng(model_seed + 10_000 * outer_seed)
    by_dataset = {dataset: [bag for bag in train_bags if bag.dataset == dataset] for dataset in sorted(datasets)}
    training_rows, evaluation_rows = [], []

    reference_validation, _ = evaluate_reference(
        backbone, validation, "guard", device, stage="reference_validation", step=0, run_fields=run_fields,
    )
    reference_test, reference_test_probabilities = evaluate_reference(
        backbone, final_test, "test", device, stage="reference_test", step=0, run_fields=run_fields,
    )
    initial_validation, _, _ = evaluate_model(
        backbone, model, validation, "guard", device, stage="initial_model_validation", step=0,
        run_fields=run_fields,
    )
    evaluation_rows.extend(reference_validation + reference_test + initial_validation)
    best_loss = macro_dataset_mean(initial_validation)
    best_step, stale, best_state = 0, 0, copy.deepcopy(model.state_dict())
    started = time.time()

    for step in range(1, args.steps + 1):
        dataset = sorted(datasets)[int(rng.integers(len(datasets)))]
        bag = by_dataset[dataset][int(rng.integers(len(by_dataset[dataset])))]
        logits, target, generated, _, _ = model_predictions(backbone, model, bag, "guard", device)
        task_loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        trust = model.trust_region(generated)
        objective = task_loss + args.regularization * trust
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip))
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            task_loss_value = float(task_loss.detach())
            objective_value = float(objective.detach())
            trust_value = float(trust.detach())
            weighted = args.regularization * trust_value
            row = {
                **run_fields, "step": step, "dataset": bag.dataset, "outer_seed": bag.seed, "bag": bag.bag,
                "task_loss": task_loss_value, "objective": objective_value, "trust_region": trust_value,
                "weighted_regularization": weighted,
                "regularization_to_task_loss": weighted / max(task_loss_value, 1e-12),
                "pre_clip_gradient_norm": gradient_norm, "elapsed_seconds": time.time() - started,
                **parameter_metrics(generated),
            }
            if device.type == "cuda":
                row.update({
                    "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                    "cuda_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                })
            training_rows.append(row)
            write_csv(run_dir / "training.csv", training_rows)
            print(
                f"[{args.branch} seed={outer_seed} model={model_seed} train] step={step}/{args.steps} "
                f"dataset={bag.dataset} bag={bag.bag} loss={task_loss_value:.6f} "
                f"grad={gradient_norm:.6g} shape_gate={row['shape_gate_mean']:.4f} "
                f"norm_gate={row['normalization_gate_mean']:.4f} reg/ce={row['regularization_to_task_loss']:.3e}",
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            current, _, _ = evaluate_model(
                backbone, model, validation, "guard", device, stage="validation", step=step,
                run_fields=run_fields,
            )
            evaluation_rows.extend(current)
            score = macro_dataset_mean(current)
            improved = score < best_loss
            if improved:
                best_loss, best_step, stale, best_state = score, step, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            dataset_losses = {dataset: np.mean([row["loss"] for row in current if row["dataset"] == dataset]) for dataset in sorted(datasets)}
            print(
                f"[{args.branch} seed={outer_seed} model={model_seed} validation] step={step} "
                f"macro_loss={score:.6f} best={best_loss:.6f}@{best_step} improved={improved} "
                f"per_dataset={json.dumps(dataset_losses, sort_keys=True)}",
                flush=True,
            )
            write_csv(run_dir / "evaluations.csv", evaluation_rows)
            torch.save({
                "state_dict": model.state_dict(), "best_state_dict": best_state,
                "optimizer": optimizer.state_dict(), "step": step, "best_step": best_step,
                "best_validation_loss": best_loss, "stale_validations": stale, "manifest": manifest,
            }, run_dir / "last.pt")
            if args.patience_validations and stale >= args.patience_validations:
                print(f"[{args.branch} seed={outer_seed} model={model_seed}] early stopping", flush=True)
                break

    model.load_state_dict(best_state)
    selected_validation, _, _ = evaluate_model(
        backbone, model, validation, "guard", device, stage="selected_validation", step=best_step,
        run_fields=run_fields,
    )
    selected_test, selected_test_probabilities, test_records = evaluate_model(
        backbone, model, final_test, "test", device, stage="selected_test", step=best_step,
        run_fields=run_fields, collect=True,
    )
    selected_train, _, train_records = evaluate_model(
        backbone, model, train_bags, "guard", device, stage="selected_train_bags", step=best_step,
        run_fields=run_fields, collect=True,
    )
    cached_diagnostics, _, _ = evaluate_model(
        backbone, model, cached_test, "test", device, stage="cached_unseen_bag_diagnostic", step=best_step,
        run_fields=run_fields,
    )
    evaluation_rows.extend(selected_validation + selected_test + selected_train + cached_diagnostics)
    write_csv(run_dir / "evaluations.csv", evaluation_rows)
    paired = paired_rows(reference_test, selected_test, run_fields=run_fields)
    write_csv(run_dir / "paired_test.csv", paired)
    consistency = consistency_rows(train_records, test_records, run_fields=run_fields)
    write_csv(run_dir / "parameter_consistency.csv", consistency)
    columns = parameter_column_rows(train_records + test_records, run_fields=run_fields)
    write_csv(run_dir / "selected_parameter_columns.csv", columns)
    torch.save({"train": train_records, "test": test_records}, run_dir / "selected_parameters.pt")

    historical = load_historical_references(args.historical_single_dataset_root, args.branch)
    per_dataset = []
    for dataset in sorted(datasets):
        rows = [row for row in paired if row["dataset"] == dataset]
        loss_delta = float(np.mean([row["loss_delta"] for row in rows]))
        reference_delta = historical.get((dataset, outer_seed))
        per_dataset.append({
            **run_fields, "dataset": dataset, "n": len(rows),
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
            "historical_single_dataset_loss_delta": reference_delta,
            "fraction_of_historical_single_dataset_gain": (
                loss_delta / reference_delta if reference_delta is not None and reference_delta < 0 else None
            ),
        })
    write_csv(run_dir / "per_dataset.csv", per_dataset)
    selected_by_validation = macro_dataset_mean(selected_validation) < macro_dataset_mean(reference_validation)
    summary = {
        **run_fields,
        "datasets": sorted(datasets), "best_step": best_step,
        "reference_validation_macro_loss": macro_dataset_mean(reference_validation),
        "initial_model_validation_macro_loss": macro_dataset_mean(initial_validation),
        "selected_validation_macro_loss": macro_dataset_mean(selected_validation),
        "selected_by_reference_guard": selected_by_validation,
        "reference_test_macro_loss": macro_dataset_mean(reference_test),
        "selected_test_macro_loss": macro_dataset_mean(selected_test),
        "test_macro_loss_delta": float(np.mean([row["loss_delta"] for row in paired])),
        "test_macro_accuracy_delta": float(np.mean([row["accuracy_delta"] for row in paired])),
        "dataset_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in per_dataset])),
        "guarded_test_macro_loss_delta": (
            float(np.mean([row["loss_delta"] for row in paired])) if selected_by_validation else 0.0
        ),
        "per_dataset": per_dataset,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    torch.save({
        "state_dict": best_state, "mean": mean, "components": components,
        "summary": summary, "manifest": manifest,
    }, run_dir / "best.pt")
    manifest["status"], manifest["summary"] = "complete", summary
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    del model, optimizer, parameters, best_state
    release_cuda(device)
    return summary, paired, per_dataset, consistency


def main() -> None:
    args = parse_args()
    if not 0 < args.outer_validation_fraction < 1 or not 0 < args.evaluation_context_fraction < 1:
        raise ValueError("validation and context fractions must be strictly between zero and one")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.teacher_cache.resolve(), map_location="cpu", weights_only=False)
    all_bags: list[TeacherBag] = payload["bags"]
    available = {bag.dataset for bag in all_bags}
    datasets = set(args.dataset) if args.dataset else available
    missing = datasets - available
    if missing:
        raise ValueError(f"datasets absent from teacher cache: {sorted(missing)}")
    available_seeds = {int(bag.seed) for bag in all_bags if bag.dataset in datasets}
    if set(args.outer_seeds) - available_seeds:
        raise ValueError(f"outer seeds absent from cache: {sorted(set(args.outer_seeds) - available_seeds)}")
    top_manifest = {
        "status": "running", "args": {**vars(args), "teacher_cache": str(args.teacher_cache),
                                         "output_dir": str(output),
                                         "historical_single_dataset_root": str(args.historical_single_dataset_root) if args.historical_single_dataset_root else None},
        "datasets": sorted(datasets), "available_cache_datasets": sorted(available),
        "cache_config": payload.get("config"),
    }
    (output / "manifest.json").write_text(json.dumps(top_manifest, indent=2, default=str), encoding="utf8")
    device = torch.device(args.device)
    backbone, _ = load_backbone(args, device)
    summaries, paired, per_dataset, consistency = [], [], [], []
    for outer_seed in args.outer_seeds:
        for model_seed in args.model_seeds:
            run_dir = output / f"outer_seed_{outer_seed}" / f"model_seed_{model_seed}"
            summary_path = run_dir / "summary.json"
            if args.resume and summary_path.exists():
                print(f"Reusing completed run: {run_dir}", flush=True)
                summaries.append(json.loads(summary_path.read_text(encoding="utf8")))
                paired.extend(read_csv(run_dir / "paired_test.csv"))
                per_dataset.extend(read_csv(run_dir / "per_dataset.csv"))
                consistency.extend(read_csv(run_dir / "parameter_consistency.csv"))
                continue
            summary, run_paired, run_per_dataset, run_consistency = run_one(
                args, backbone, all_bags, datasets, outer_seed=outer_seed,
                model_seed=model_seed, device=device, run_dir=run_dir,
            )
            summaries.append(summary); paired.extend(run_paired); per_dataset.extend(run_per_dataset); consistency.extend(run_consistency)
            write_csv(output / "runs.csv", summaries)
            write_csv(output / "paired_test.csv", paired)
            write_csv(output / "per_dataset.csv", per_dataset)
            write_csv(output / "parameter_consistency.csv", consistency)

    numeric_paired = [{**row, **{key: float(row[key]) for key in row if key.endswith("_delta")}} for row in paired]
    dataset_seed_groups = defaultdict(list)
    for row in numeric_paired:
        dataset_seed_groups[(row["dataset"], int(row["outer_seed_run"]), int(row["model_seed"]))].append(row)
    units = []
    for (dataset, outer_seed, model_seed), rows in sorted(dataset_seed_groups.items()):
        units.append({
            "dataset": dataset, "outer_seed": outer_seed, "model_seed": model_seed,
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
        })
    write_csv(output / "dataset_seed_units.csv", units)
    aggregate = {
        "branch": args.branch, "datasets": sorted(datasets),
        "outer_seeds": args.outer_seeds, "model_seeds": args.model_seeds,
        "runs": len(summaries), "dataset_seed_units": len(units),
        "macro_loss_delta": float(np.mean([row["loss_delta"] for row in units])),
        "macro_accuracy_delta": float(np.mean([row["accuracy_delta"] for row in units])),
        "macro_balanced_accuracy_delta": float(np.mean([row["balanced_accuracy_delta"] for row in units])),
        "macro_brier_delta": float(np.mean([row["brier_delta"] for row in units])),
        "macro_ece_delta": float(np.mean([row["ece_delta"] for row in units])),
        "dataset_seed_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in units])),
        "guard_selected_fraction": float(np.mean([bool(row["selected_by_reference_guard"]) for row in summaries])),
        "mean_guarded_loss_delta": float(np.mean([float(row["guarded_test_macro_loss_delta"]) for row in summaries])),
        "per_run": summaries,
    }
    historical_rows = []
    for row in per_dataset:
        reference = row.get("historical_single_dataset_loss_delta")
        if reference not in (None, "") and float(reference) < 0:
            historical_rows.append((float(row["loss_delta"]), float(reference)))
    aggregate["fraction_of_historical_single_dataset_gain"] = (
        float(sum(-current for current, _ in historical_rows) / sum(-reference for _, reference in historical_rows))
        if historical_rows else None
    )
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf8")
    top_manifest["status"], top_manifest["summary"] = "complete", aggregate
    (output / "manifest.json").write_text(json.dumps(top_manifest, indent=2, default=str), encoding="utf8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
