"""Teacher-free, per-column context/query HyperSpline experiment.

This is the controlled bridge from DirectSpline split-specific headroom to a
simple learned HyperSpline.  Four known PMLB datasets are divided once into
permanent, row-disjoint train/validation/test pools.  A single shared MLP is
trained across fresh training-pool episodes and evaluated only on episodes
drawn from the held-out pools.

The three paired arms differ *only* in the per-column conditioner input:

``context``
    Context marginal and label-aware context statistics.
``query_marginal``
    Query marginal and label-aware context statistics.
``context_query_shift``
    Context statistics, query marginal statistics, and signed/absolute
    context-to-query marginal shifts, plus label-aware context statistics.

Every arm has the same capacity-matched small shared MLP, direct bounded monotone spline
output, identity initialization, frozen TabICL backbone, row pools, episodes,
seeds, optimizer, and the sole training objective: TabICL query NLL.  Query
labels never enter parameter generation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import HyperSplineTransform, summarize_context

try:  # Support both ``python scripts/...`` and package-style imports.
    from scripts.direct_spline_dataset_headroom import release_cuda
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_rank_basis_zero_shot import metrics
    from scripts.hyperspline_real_task_bank import load_pmlb_frame
except ModuleNotFoundError:  # pragma: no cover - direct script invocation.
    from direct_spline_dataset_headroom import release_cuda
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_rank_basis_zero_shot import metrics
    from hyperspline_real_task_bank import load_pmlb_frame


DEFAULT_DATASETS = ("magic", "phoneme", "spambase", "pendigits")
FORMAT_VERSION = 2


@dataclass(frozen=True)
class RowPool:
    dataset: str
    split: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Episode:
    dataset: str
    stage: str
    episode_id: int
    source_seed: int
    x_context: torch.Tensor  # CPU, (N_context, D)
    y_context: torch.Tensor  # CPU, (N_context,)
    x_query: torch.Tensor  # CPU, (N_query, D)
    y_query: torch.Tensor  # CPU, (N_query,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("context", "query_marginal", "context_query_shift"), required=True)
    parser.add_argument("--pmlb-dataset", action="append", default=None,
                        help="Repeat; defaults to magic, phoneme, spambase, pendigits.")
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--pool-bank", type=Path, default=Path("results/hyperspline_query_conditioning/banks/row_pools.pt"))
    parser.add_argument("--validation-bank", type=Path, default=Path("results/hyperspline_query_conditioning/banks/validation.pt"))
    parser.add_argument("--test-bank", type=Path, default=Path("results/hyperspline_query_conditioning/banks/test.pt"))
    parser.add_argument("--prepare-banks-only", action="store_true")
    parser.add_argument("--pool-seed", type=int, default=91_001)
    parser.add_argument("--validation-seed", type=int, default=101_001)
    parser.add_argument("--test-seed", type=int, default=201_001)
    parser.add_argument("--train-row-fraction", type=float, default=0.60)
    parser.add_argument("--validation-row-fraction", type=float, default=0.20)
    parser.add_argument("--test-row-fraction", type=float, default=0.20)
    parser.add_argument("--validation-episodes-per-dataset", type=int, default=16)
    parser.add_argument("--test-episodes-per-dataset", type=int, default=32)
    parser.add_argument("--context-rows", type=int, default=384)
    parser.add_argument("--query-rows", type=int, default=128)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--train-seed", type=int, default=1_001)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--tasks-per-step", type=int, default=2,
                        help="Compatible same-dataset episodes stacked on the TabICL batch axis.")
    parser.add_argument("--max-backbone-batch-size", type=int, default=2,
                        help="Automatic CUDA-OOM fallback halves this at runtime if necessary.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--gate-initial-probability", type=float, default=0.01)
    parser.add_argument("--generate-location", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generate-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patience-validations", type=int, default=20)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf8")


def _pool_payload(pools: Sequence[RowPool], *, datasets: Sequence[str], args: argparse.Namespace) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "datasets": list(datasets),
        "pool_seed": args.pool_seed,
        "fractions": {
            "train": args.train_row_fraction,
            "validation": args.validation_row_fraction,
            "test": args.test_row_fraction,
        },
        "pools": [
            {"dataset": pool.dataset, "split": pool.split, "x": torch.as_tensor(pool.x), "y": torch.as_tensor(pool.y)}
            for pool in pools
        ],
    }


def save_pools(path: Path, pools: Sequence[RowPool], *, datasets: Sequence[str], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_pool_payload(pools, datasets=datasets, args=args), path)
    print(f"Saved permanent row pools for {len(datasets)} datasets to {path}", flush=True)


def _fractions(args: argparse.Namespace) -> dict[str, float]:
    return {
        "train": args.train_row_fraction,
        "validation": args.validation_row_fraction,
        "test": args.test_row_fraction,
    }


def pool_signature(pools: Sequence[RowPool]) -> str:
    """Stable identity for the exact rows backing a fixed episode bank."""
    digest = hashlib.sha256()
    for pool in sorted(pools, key=lambda item: (item.dataset, item.split)):
        digest.update(pool.dataset.encode("utf8"))
        digest.update(pool.split.encode("utf8"))
        for array in (pool.x, pool.y):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_pools(path: Path, *, datasets: Sequence[str], args: argparse.Namespace) -> list[RowPool]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported row-pool bank format: {path}")
    if tuple(payload.get("datasets", ())) != tuple(datasets):
        raise ValueError(f"row-pool bank datasets differ from requested datasets: {path}")
    if payload.get("pool_seed") != args.pool_seed or payload.get("fractions") != _fractions(args):
        raise ValueError(
            f"row-pool bank settings differ from the requested pool seed/fractions: {path}. "
            "Use matching arguments or regenerate all fixed banks."
        )
    pools = [
        RowPool(
            dataset=str(item["dataset"]), split=str(item["split"]),
            x=np.asarray(item["x"], dtype=np.float32), y=np.asarray(item["y"], dtype=np.int64),
        )
        for item in payload["pools"]
    ]
    expected = {(dataset, split) for dataset in datasets for split in ("train", "validation", "test")}
    actual = {(pool.dataset, pool.split) for pool in pools}
    if actual != expected:
        raise ValueError(f"row-pool bank has unexpected dataset/split keys: {actual ^ expected}")
    return pools


def make_pools(args: argparse.Namespace, datasets: Sequence[str]) -> list[RowPool]:
    pools: list[RowPool] = []
    for dataset_index, dataset in enumerate(datasets):
        x, raw_y, _ = load_pmlb_frame(dataset, cache_dir=args.pmlb_cache_dir)
        _, y = np.unique(raw_y, return_inverse=True)
        labels, counts = np.unique(y, return_counts=True)
        if labels.size < 2 or labels.size > 10 or counts.min() < 6:
            raise ValueError(f"{dataset}: need 2..10 classes with at least six examples each")
        indices = np.arange(y.size)
        random_state = args.pool_seed + 10_000 * dataset_index
        train_indices, heldout_indices = train_test_split(
            indices,
            test_size=args.validation_row_fraction + args.test_row_fraction,
            random_state=random_state,
            stratify=y,
        )
        test_share_of_heldout = args.test_row_fraction / (args.validation_row_fraction + args.test_row_fraction)
        validation_indices, test_indices = train_test_split(
            heldout_indices,
            test_size=test_share_of_heldout,
            random_state=random_state + 1,
            stratify=y[heldout_indices],
        )
        for split, split_indices in (("train", train_indices), ("validation", validation_indices), ("test", test_indices)):
            pools.append(RowPool(dataset, split, np.asarray(x[split_indices], dtype=np.float32), np.asarray(y[split_indices], dtype=np.int64)))
        print(
            f"[{dataset}] permanent pools: train={train_indices.size}, validation={validation_indices.size}, test={test_indices.size}",
            flush=True,
        )
    return pools


def _stratified_rows(y: np.ndarray, n_rows: int, rng_seed: int) -> np.ndarray:
    indices = np.arange(y.size)
    if n_rows >= y.size:
        return indices
    selected, _ = train_test_split(indices, train_size=n_rows, random_state=rng_seed, stratify=y)
    return np.asarray(selected)


def _repair_nonfinite(context_x: np.ndarray, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Context-only median repair, matching the numerical boundary in prior real runs."""
    context_x, query_x = np.asarray(context_x, dtype=np.float32).copy(), np.asarray(query_x, dtype=np.float32).copy()
    context_finite, query_finite = np.isfinite(context_x), np.isfinite(query_x)
    replacements = int((~context_finite).sum() + (~query_finite).sum())
    if replacements == 0:
        return context_x, query_x, 0
    medians = np.nanmedian(np.where(context_finite, context_x, np.nan), axis=0)
    if not np.isfinite(medians).all():
        bad = np.flatnonzero(~np.isfinite(medians)).tolist()
        raise ValueError(f"context has no finite values in columns {bad}")
    return (
        np.where(context_finite, context_x, medians).astype(np.float32, copy=False),
        np.where(query_finite, query_x, medians).astype(np.float32, copy=False),
        replacements,
    )


def make_episode(pool: RowPool, *, stage: str, episode_id: int, source_seed: int, args: argparse.Namespace) -> Episode:
    labels, counts = np.unique(pool.y, return_counts=True)
    desired_total = min(pool.y.size, args.context_rows + args.query_rows)
    if desired_total < 2 * labels.size or counts.min() < 2:
        raise ValueError(f"{pool.dataset}/{pool.split} is too small for a class-preserving episode")
    selected = _stratified_rows(pool.y, desired_total, source_seed)
    # Preserve the requested context/query ratio when a held-out pool is
    # smaller than the nominal episode budget (notably Pendigits' test pool),
    # while retaining at least one row per class on each side.
    requested_fraction = args.context_rows / (args.context_rows + args.query_rows)
    context_rows = int(round(desired_total * requested_fraction))
    context_rows = min(max(context_rows, labels.size), desired_total - labels.size)
    context_indices, query_indices = train_test_split(
        selected, train_size=context_rows, random_state=source_seed + 1, stratify=pool.y[selected]
    )
    context_x, query_x, repaired = _repair_nonfinite(pool.x[context_indices], pool.x[query_indices])
    if repaired and episode_id == 0:
        print(f"[{pool.dataset} {stage}] context-median-imputed {repaired} non-finite numerical cells", flush=True)
    return Episode(
        dataset=pool.dataset, stage=stage, episode_id=episode_id, source_seed=source_seed,
        x_context=torch.as_tensor(context_x), y_context=torch.as_tensor(pool.y[context_indices], dtype=torch.float32),
        x_query=torch.as_tensor(query_x), y_query=torch.as_tensor(pool.y[query_indices], dtype=torch.long),
    )


def save_episode_bank(
    path: Path, episodes: Sequence[Episode], *, stage: str, datasets: Sequence[str], seed: int,
    episodes_per_dataset: int, pools: Sequence[RowPool], args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION, "stage": stage, "datasets": list(datasets), "seed": seed,
            "episodes_per_dataset": episodes_per_dataset,
            "context_rows": args.context_rows,
            "query_rows": args.query_rows,
            "pool_signature": pool_signature(pools),
            "episodes": [
                {
                    "dataset": episode.dataset, "stage": episode.stage, "episode_id": episode.episode_id,
                    "source_seed": episode.source_seed, "x_context": episode.x_context, "y_context": episode.y_context,
                    "x_query": episode.x_query, "y_query": episode.y_query,
                }
                for episode in episodes
            ],
        }, path,
    )
    print(f"Saved {stage} episode bank with {len(episodes)} episodes to {path}", flush=True)


def load_episode_bank(
    path: Path, *, stage: str, datasets: Sequence[str], seed: int, episodes_per_dataset: int,
    pools: Sequence[RowPool], args: argparse.Namespace,
) -> list[Episode]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("stage") != stage:
        raise ValueError(f"unsupported or wrong-stage episode bank: {path}")
    if tuple(payload.get("datasets", ())) != tuple(datasets):
        raise ValueError(f"episode bank datasets differ from requested datasets: {path}")
    expected = {
        "seed": seed,
        "episodes_per_dataset": episodes_per_dataset,
        "context_rows": args.context_rows,
        "query_rows": args.query_rows,
        "pool_signature": pool_signature(pools),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"episode bank settings or source rows differ from this run: {path}. "
            "Use matching arguments or regenerate all fixed banks."
        )
    return [Episode(**item) for item in payload["episodes"]]


def make_fixed_episodes(
    pools: Sequence[RowPool], *, split: str, stage: str, episodes_per_dataset: int, seed: int, args: argparse.Namespace
) -> list[Episode]:
    pool_by_dataset = {pool.dataset: pool for pool in pools if pool.split == split}
    result = []
    for dataset_index, dataset in enumerate(sorted(pool_by_dataset)):
        for episode_id in range(episodes_per_dataset):
            result.append(make_episode(
                pool_by_dataset[dataset], stage=stage, episode_id=episode_id,
                source_seed=seed + 100_000 * dataset_index + episode_id, args=args,
            ))
    return result


def get_banks(args: argparse.Namespace, datasets: Sequence[str]) -> tuple[list[RowPool], list[Episode], list[Episode]]:
    pools = load_pools(args.pool_bank, datasets=datasets, args=args) if args.pool_bank.is_file() else make_pools(args, datasets)
    if not args.pool_bank.is_file():
        save_pools(args.pool_bank, pools, datasets=datasets, args=args)
    if args.validation_bank.is_file():
        validation = load_episode_bank(
            args.validation_bank, stage="validation", datasets=datasets, seed=args.validation_seed,
            episodes_per_dataset=args.validation_episodes_per_dataset, pools=pools, args=args,
        )
    else:
        validation = make_fixed_episodes(
            pools, split="validation", stage="validation", episodes_per_dataset=args.validation_episodes_per_dataset,
            seed=args.validation_seed, args=args,
        )
        save_episode_bank(
            args.validation_bank, validation, stage="validation", datasets=datasets, seed=args.validation_seed,
            episodes_per_dataset=args.validation_episodes_per_dataset, pools=pools, args=args,
        )
    if args.test_bank.is_file():
        test = load_episode_bank(
            args.test_bank, stage="test", datasets=datasets, seed=args.test_seed,
            episodes_per_dataset=args.test_episodes_per_dataset, pools=pools, args=args,
        )
    else:
        test = make_fixed_episodes(
            pools, split="test", stage="test", episodes_per_dataset=args.test_episodes_per_dataset,
            seed=args.test_seed, args=args,
        )
        save_episode_bank(
            args.test_bank, test, stage="test", datasets=datasets, seed=args.test_seed,
            episodes_per_dataset=args.test_episodes_per_dataset, pools=pools, args=args,
        )
    return pools, validation, test


def stack_episodes(episodes: Sequence[Episode], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not episodes:
        raise ValueError("cannot stack no episodes")
    shape = (episodes[0].x_context.shape, episodes[0].x_query.shape)
    if any((episode.x_context.shape, episode.x_query.shape) != shape for episode in episodes):
        raise ValueError("only compatible episodes can share a TabICL batch")
    return (
        torch.stack([episode.x_context for episode in episodes]).to(device),
        torch.stack([episode.y_context for episode in episodes]).to(device),
        torch.stack([episode.x_query for episode in episodes]).to(device),
        torch.stack([episode.y_query for episode in episodes]).to(device),
    )


def identity_logits(backbone, x_context: torch.Tensor, y_context: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    statistics = summarize_context(x_context)
    context = (x_context.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    query = (x_query.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    backbone.clear_cache()
    return backbone(torch.cat((context, query), dim=1), y_context)


def model_logits(backbone, model: HyperSplineTransform, x_context, y_context, x_query):
    context, query, parameters = model(
        x_context, x_query, y_context=y_context if model.target_aware else None, return_parameters=True
    )
    backbone.clear_cache()
    return backbone(torch.cat((context, query), dim=1), y_context), parameters


def parameter_diagnostics(parameters) -> dict[str, float]:
    identity = torch.linspace(-1.0, 1.0, parameters.control_points.shape[-1], device=parameters.control_points.device)
    return {
        "mean_gate": float(parameters.gate.detach().mean()),
        "max_gate": float(parameters.gate.detach().max()),
        "mean_abs_control_displacement": float((parameters.control_points.detach() - identity).abs().mean()),
        "location_rms": float(parameters.location.detach().square().mean().sqrt()),
        "scale_mean": float(parameters.scale.detach().mean()),
    }


def macro_dataset_mean(rows: Sequence[dict], field: str = "loss") -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


@torch.no_grad()
def evaluate(
    backbone, model: HyperSplineTransform, episodes: Sequence[Episode], *, stage: str, step: int,
    device: torch.device, batch_size: int, identity: bool, run_fields: dict,
) -> list[dict]:
    rows = []
    grouped: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.dataset].append(episode)
    for dataset in sorted(grouped):
        start, safe_batch = 0, batch_size
        while start < len(grouped[dataset]):
            batch_episodes = grouped[dataset][start : start + safe_batch]
            try:
                x_context, y_context, x_query, y_query = stack_episodes(batch_episodes, device)
                if identity:
                    logits = identity_logits(backbone, x_context, y_context, x_query)
                    parameters = None
                else:
                    logits, parameters = model_logits(backbone, model, x_context, y_context, x_query)
            except RuntimeError as error:
                if not (device.type == "cuda" and len(batch_episodes) > 1 and _is_cuda_oom(error)):
                    raise
                next_batch = max(1, len(batch_episodes) // 2)
                print(f"[{stage}] CUDA OOM for {dataset}; retrying evaluation batch {len(batch_episodes)}->{next_batch}", flush=True)
                safe_batch = next_batch
                gc.collect(); torch.cuda.empty_cache()
                continue
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logits at {stage} for {dataset}")
            probability = logits.softmax(-1).cpu()
            for index, episode in enumerate(batch_episodes):
                row = {
                    **run_fields, "stage": stage, "step": step, "dataset": episode.dataset,
                    "episode_id": episode.episode_id, "source_seed": episode.source_seed,
                    "n_context": int(episode.x_context.shape[0]), "n_query": int(episode.x_query.shape[0]),
                    "n_features": int(episode.x_context.shape[1]), **metrics(probability[index : index + 1], y_query[index : index + 1].cpu()),
                }
                if parameters is not None:
                    row.update(parameter_diagnostics(type(parameters)(
                        parameters.control_points[index : index + 1], parameters.gate[index : index + 1],
                        parameters.location[index : index + 1], parameters.scale[index : index + 1],
                        None if parameters.supervised_residual_gate is None else parameters.supervised_residual_gate[index : index + 1],
                    )))
                rows.append(row)
            start += len(batch_episodes)
    return rows


def train_batch(
    backbone, model: HyperSplineTransform, episodes: Sequence[Episode], *, device: torch.device
) -> tuple[torch.Tensor, object]:
    x_context, y_context, x_query, y_query = stack_episodes(episodes, device)
    logits, parameters = model_logits(backbone, model, x_context, y_context, x_query)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite training logits")
    return F.cross_entropy(logits.flatten(0, 1), y_query.flatten()), parameters


def accumulated_train_step(
    backbone, model: HyperSplineTransform, optimizer: torch.optim.Optimizer, episodes: Sequence[Episode], *,
    device: torch.device, max_microbatch_size: int, trainable: Sequence[torch.Tensor], gradient_clip: float,
) -> tuple[float, dict[str, float], float]:
    """Take one optimizer step over all episodes, using bounded microbatches.

    The weighting is by query rows, so this is equivalent to one cross-entropy
    over the complete logical meta-batch even when CUDA requires batch size one.
    """
    if not episodes:
        raise ValueError("cannot train on an empty logical meta-batch")
    total_query_rows = sum(int(episode.y_query.numel()) for episode in episodes)
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    weighted_diagnostics: dict[str, float] = defaultdict(float)
    for start in range(0, len(episodes), max_microbatch_size):
        microbatch = episodes[start : start + max_microbatch_size]
        loss, parameters = train_batch(backbone, model, microbatch, device=device)
        rows = sum(int(episode.y_query.numel()) for episode in microbatch)
        weight = rows / total_query_rows
        (weight * loss).backward()
        weighted_loss += weight * float(loss.detach())
        for key, value in parameter_diagnostics(parameters).items():
            weighted_diagnostics[key] += weight * value
        del loss, parameters
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, gradient_clip))
    optimizer.step()
    return weighted_loss, dict(weighted_diagnostics), gradient_norm


def _is_cuda_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def run_model_seed(
    args: argparse.Namespace, *, backbone, pools: Sequence[RowPool], validation: Sequence[Episode], test: Sequence[Episode],
    model_seed: int, device: torch.device,
) -> dict:
    run_dir = args.output_dir / f"model_seed_{model_seed}"
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf8"))
        if existing.get("status") == "complete":
            print(f"[{args.arm} seed={model_seed}] already complete; skipping due to --resume", flush=True)
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = HyperSplineTransform(
        n_control_points=args.n_control_points, hidden_dim=args.hidden_dim,
        generate_location=args.generate_location, generate_scale=args.generate_scale,
        gate_initial_probability=args.gate_initial_probability, target_aware=args.target_aware,
        conditioning_mode=args.arm, capacity_matched_conditioning=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    trainable = list(model.parameters())
    train_pools = {pool.dataset: pool for pool in pools if pool.split == "train"}
    datasets = tuple(sorted(train_pools))
    run_fields = {"arm": args.arm, "model_seed": model_seed}
    manifest = {
        "status": "running", "run": run_fields, "datasets": list(datasets),
        "teacher_used": False, "teacher_parameters_used_as_targets": False,
        "query_labels_enter_parameter_generation": False,
        "train_objective": "TabICL query cross entropy only",
        "conditioning": args.arm, "output": "direct bounded monotone cubic spline controls per numerical column",
        "conditioning_capacity_matched": True,
        "shared_weights_across_columns": True, "same_transform_applied_to_context_and_query": True,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(run_dir / "manifest.json", manifest)
    identity_validation = evaluate(
        backbone, model, validation, stage="identity_validation", step=0, device=device,
        batch_size=args.evaluation_batch_size, identity=True, run_fields=run_fields,
    )
    identity_test = evaluate(
        backbone, model, test, stage="identity_test", step=0, device=device,
        batch_size=args.evaluation_batch_size, identity=True, run_fields=run_fields,
    )
    evaluations = identity_validation + identity_test
    identity_validation_loss = macro_dataset_mean(identity_validation)
    best_loss, best_step, best_state, stale = identity_validation_loss, 0, copy.deepcopy(model.state_dict()), 0
    training_rows: list[dict] = []
    safe_microbatch = min(args.tasks_per_step, args.max_backbone_batch_size)
    started = time.time()
    print(
        f"[{args.arm} seed={model_seed}] shared direct HyperSpline: datasets={list(datasets)}, "
        f"conditioning_dim={model.conditioning_dim}, tasks/optimizer_step={args.tasks_per_step}, "
        f"initial_max_microbatch={safe_microbatch}; "
        f"reference validation macro NLL={identity_validation_loss:.6f}", flush=True,
    )
    for step in range(1, args.steps + 1):
        dataset = datasets[(step - 1) % len(datasets)]
        episodes = [make_episode(
            train_pools[dataset], stage="train", episode_id=(step - 1) * args.tasks_per_step + offset,
            source_seed=args.train_seed + 10_000_000 * model_seed + 100_000 * datasets.index(dataset) + (step - 1) * args.tasks_per_step + offset,
            args=args,
        ) for offset in range(args.tasks_per_step)]
        attempt_microbatch = safe_microbatch
        while True:
            try:
                task_loss, diagnostics, gradient_norm = accumulated_train_step(
                    backbone, model, optimizer, episodes, device=device, max_microbatch_size=attempt_microbatch,
                    trainable=trainable, gradient_clip=args.gradient_clip,
                )
                break
            except RuntimeError as error:
                optimizer.zero_grad(set_to_none=True)
                if not (device.type == "cuda" and attempt_microbatch > 1 and _is_cuda_oom(error)):
                    raise
                next_microbatch = max(1, attempt_microbatch // 2)
                print(
                    f"[{args.arm} seed={model_seed}] CUDA OOM at step={step}; retrying microbatch "
                    f"{attempt_microbatch}->{next_microbatch} while retaining all {args.tasks_per_step} tasks",
                    flush=True,
                )
                attempt_microbatch = safe_microbatch = next_microbatch
                gc.collect(); torch.cuda.empty_cache()
        if step == 1 or step % args.log_every == 0:
            row = {
                **run_fields, "step": step, "dataset": dataset,
                "tasks_per_optimizer_step": args.tasks_per_step, "backbone_microbatch_size": attempt_microbatch,
                "task_loss": task_loss, "pre_clip_gradient_norm": gradient_norm,
                "elapsed_seconds": time.time() - started, **diagnostics,
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
                f"[{args.arm} seed={model_seed} train] step={step}/{args.steps} dataset={dataset} "
                f"loss={row['task_loss']:.6f} grad={gradient_norm:.5g} tasks={args.tasks_per_step} "
                f"microbatch={attempt_microbatch} gate={row['mean_gate']:.4f}",
                flush=True,
            )
        del episodes
        if step % args.validate_every == 0 or step == args.steps:
            current = evaluate(
                backbone, model, validation, stage="validation", step=step, device=device,
                batch_size=args.evaluation_batch_size, identity=False, run_fields=run_fields,
            )
            evaluations.extend(current)
            current_loss = macro_dataset_mean(current)
            improved = current_loss < best_loss
            if improved:
                best_loss, best_step, best_state, stale = current_loss, step, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            write_csv(run_dir / "evaluations.csv", evaluations)
            torch.save({"state_dict": model.state_dict(), "best_state_dict": best_state, "step": step, "best_step": best_step}, run_dir / "last.pt")
            print(
                f"[{args.arm} seed={model_seed} validation] step={step} macro_nll={current_loss:.6f} "
                f"best={best_loss:.6f}@{best_step} improved={improved}", flush=True,
            )
            if args.patience_validations and stale >= args.patience_validations:
                print(f"[{args.arm} seed={model_seed}] early stopping after {stale} validations without improvement", flush=True)
                break
    model.load_state_dict(best_state)
    selected_validation = evaluate(
        backbone, model, validation, stage="selected_validation", step=best_step, device=device,
        batch_size=args.evaluation_batch_size, identity=False, run_fields=run_fields,
    )
    selected_test = evaluate(
        backbone, model, test, stage="selected_test", step=best_step, device=device,
        batch_size=args.evaluation_batch_size, identity=False, run_fields=run_fields,
    )
    evaluations.extend(selected_validation + selected_test)
    write_csv(run_dir / "evaluations.csv", evaluations)
    identity_by_key = {(row["dataset"], row["episode_id"]): row for row in identity_test}
    paired = []
    for row in selected_test:
        base = identity_by_key[row["dataset"], row["episode_id"]]
        paired.append({
            **run_fields, "dataset": row["dataset"], "episode_id": row["episode_id"], "source_seed": row["source_seed"],
            **{f"{name}_delta": float(row[name]) - float(base[name]) for name in ("loss", "accuracy", "balanced_accuracy", "brier", "ece")},
        })
    per_dataset = []
    for dataset in datasets:
        group = [row for row in paired if row["dataset"] == dataset]
        per_dataset.append({"dataset": dataset, "n": len(group), **{
            key: float(np.mean([float(row[key]) for row in group]))
            for key in ("loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta")
        }})
    write_csv(run_dir / "paired_test.csv", paired)
    write_csv(run_dir / "per_dataset.csv", per_dataset)
    summary = {
        "status": "complete", **run_fields, "datasets": list(datasets), "best_step": best_step,
        "identity_validation_macro_nll": identity_validation_loss, "selected_validation_macro_nll": best_loss,
        "validation_nll_delta": best_loss - identity_validation_loss,
        "test_episodes": len(paired),
        "macro_test_nll_delta": float(np.mean([row["loss_delta"] for row in per_dataset])),
        "macro_test_accuracy_delta": float(np.mean([row["accuracy_delta"] for row in per_dataset])),
        "test_dataset_win_fraction_nll": float(np.mean([row["loss_delta"] < 0.0 for row in per_dataset])),
        "test_episode_win_fraction_nll": float(np.mean([row["loss_delta"] < 0.0 for row in paired])),
        "max_backbone_microbatch_size_final": safe_microbatch,
        "protocol": "permanent row-disjoint pools + teacher-free shared per-column HyperSpline + TabICL query NLL only",
    }
    torch.save({"state_dict": best_state, "best_step": best_step, "summary": summary, "manifest": manifest}, run_dir / "best.pt")
    write_json(run_dir / "summary.json", summary)
    release_cuda(device)
    return summary


def main() -> None:
    args = parse_args()
    datasets = tuple(args.pmlb_dataset or DEFAULT_DATASETS)
    if len(datasets) < 2 or len(set(datasets)) != len(datasets):
        raise ValueError("provide at least two unique datasets")
    if abs(args.train_row_fraction + args.validation_row_fraction + args.test_row_fraction - 1.0) > 1e-8:
        raise ValueError("row-pool fractions must sum to one")
    if min(args.train_row_fraction, args.validation_row_fraction, args.test_row_fraction) <= 0:
        raise ValueError("row-pool fractions must be positive")
    if min(args.context_rows, args.query_rows, args.steps, args.tasks_per_step, args.max_backbone_batch_size,
           args.validation_episodes_per_dataset, args.test_episodes_per_dataset, args.validate_every, args.log_every) <= 0:
        raise ValueError("row budgets, episode counts, steps, batch sizes, and log intervals must be positive")
    if args.lr <= 0 or args.n_control_points <= 3 or args.hidden_dim <= 0 or args.gradient_clip <= 0:
        raise ValueError("invalid model or optimizer settings")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        raise ValueError("model seeds must be unique")
    pools, validation, test = get_banks(args, datasets)
    if args.prepare_banks_only:
        print("Prepared row-disjoint pools and fixed validation/test episode banks; no model was trained.", flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone, _ = load_backbone(args, device)
    # TabICL's ``train`` path is the frozen differentiable forward used by
    # every DirectSpline/HyperSpline optimisation script.  Its ``eval`` path
    # dispatches to a separate inference manager, which is not suitable for
    # backpropagation and can allocate a much larger internal cache even for
    # a single episode.  The checkpoint loader disables dropout; only the
    # HyperSpline parameters remain trainable below.
    backbone.train()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    summaries = [run_model_seed(
        args, backbone=backbone, pools=pools, validation=validation, test=test, model_seed=model_seed, device=device,
    ) for model_seed in args.model_seeds]
    macro = {
        key: float(np.mean([float(summary[key]) for summary in summaries]))
        for key in ("validation_nll_delta", "macro_test_nll_delta", "macro_test_accuracy_delta",
                    "test_dataset_win_fraction_nll", "test_episode_win_fraction_nll")
    }
    write_json(args.output_dir / "summary.json", {
        "arm": args.arm, "datasets": list(datasets), "runs": summaries,
        "macro_mean_over_model_seeds": macro,
        "protocol": "all arms must reuse the same pool/validation/test banks; test rows never enter training",
    })
    print(json.dumps({"arm": args.arm, **macro}, indent=2), flush=True)


if __name__ == "__main__":
    main()
