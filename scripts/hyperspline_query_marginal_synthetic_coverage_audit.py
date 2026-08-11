"""Audit native versus coverage-expanded synthetic tasks in HyperSpline's exact input space.

This is deliberately a *read-only diagnostic*, not a training script and not
a DirectSpline teacher experiment.  The primary analysis compares numerical
columns using the 33 values consumed by the winning ``query_marginal``
HyperSpline conditioner:

* 23 unlabeled marginal statistics of the query column;
* 8 class-permutation-invariant context-label statistics; and
* 2 context/query alignment values (relative location and log scale ratio).

It creates matched synthetic banks: the native control and
``coverage_expanded`` have the same prior seeds, task sizes, active feature
counts, labels and context fractions.  They differ only in the deterministic,
label-free observed-column model.  Real validation datasets are never used to
fit the descriptor scaler or choose the synthetic bank.

Optional DirectSpline measurements are oracle headroom diagnostics only.  They
optimise a separate spline using each synthetic query's labels, and are never
fed into coverage, model selection, or HyperSpline training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

try:  # Supports tests and ``python scripts/...`` invocation.
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import load_bank
    from scripts.hyperspline_real_zero_shot_eval import RealEpisode
    from scripts.hyperspline_synthetic_train import (
        SyntheticEpisode,
        generate_scheduled_episodes,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation.
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import load_bank
    from hyperspline_real_zero_shot_eval import RealEpisode
    from hyperspline_synthetic_train import SyntheticEpisode, generate_scheduled_episodes

from tabicl._hyperspline import DirectSplineTransform, summarize_context
from tabicl._hyperspline.statistics import SUPERVISED_SUMMARY_DIM, UNSUPERVISED_SUMMARY_DIM


QUERY_MARGINAL_DIM = UNSUPERVISED_SUMMARY_DIM + SUPERVISED_SUMMARY_DIM + 2


@dataclass(frozen=True)
class ColumnPoint:
    source: str
    identity: str
    episode_id: int
    column: int
    n_context: int
    n_query: int
    n_features: int
    n_numerical_features: int
    n_classes: int
    descriptor: np.ndarray


@dataclass(frozen=True)
class RobustScaler:
    """Coordinate-wise robust scaling fitted only on real meta-training data."""

    center: np.ndarray
    scale: np.ndarray
    clip: float

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(matrix, dtype=np.float64) - self.center) / self.scale, -self.clip, self.clip)


class Progress:
    def __init__(self, label: str, total: int, every: int) -> None:
        self.label, self.total, self.every = label, max(total, 1), max(every, 1)
        self.done, self.started = 0, time.monotonic()

    def update(self, count: int = 1) -> None:
        previous = self.done
        self.done += count
        if self.done != self.total and self.done // self.every == previous // self.every:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate else float("inf")
        print(
            f"[{self.label}] {self.done}/{self.total} ({100 * self.done / self.total:.1f}%) "
            f"rate={rate:.2f} episodes/s eta={remaining:.0f}s",
            flush=True,
        )


def parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)) or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected unique positive comma-separated integers")
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)) or not all(0.0 < value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("expected unique context fractions in (0, 1)")
    return values


def _json_default(value: object):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def episode_identity(episode: RealEpisode | SyntheticEpisode) -> tuple[str, int, int, int, int, int]:
    """Return ID and dimensional metadata without reading query labels."""
    if isinstance(episode, RealEpisode):
        return (
            episode.dataset,
            int(episode.split_seed),
            int(episode.n_context),
            int(episode.n_query),
            int(episode.n_numerical_features),
            int(episode.n_classes),
        )
    width = int(episode.x_context.shape[-1])
    return (
        f"synthetic_{episode.task_id}",
        int(episode.task_id),
        int(episode.x_context.shape[1]),
        int(episode.x_query.shape[1]),
        width,
        int(episode.n_classes),
    )


def numerical_mask(episode: RealEpisode | SyntheticEpisode) -> torch.Tensor:
    if isinstance(episode, RealEpisode):
        return episode.numerical_mask.detach().cpu().bool()
    return torch.ones(episode.x_context.shape[-1], dtype=torch.bool)


@torch.no_grad()
def query_marginal_descriptors(
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    """Return the exact 33-value query-marginal input per numerical column.

    Query labels are purposefully not an argument.  This is the same algebra
    as ``HyperSplineTransform.conditioning_summary(..., query_marginal)``.
    """
    context = summarize_context(x_context.float(), y_context=y_context.float())
    query = summarize_context(x_query.float())
    alignment = torch.stack(
        (
            (query.location - context.location) / context.scale,
            (query.scale / context.scale).clamp_min(1e-6).log(),
        ),
        dim=-1,
    )
    descriptor = torch.cat(
        (
            query.summary[..., :UNSUPERVISED_SUMMARY_DIM],
            context.summary[..., -SUPERVISED_SUMMARY_DIM:],
            alignment,
        ),
        dim=-1,
    )
    if descriptor.shape[-1] != QUERY_MARGINAL_DIM:
        raise AssertionError(f"unexpected descriptor dimension {descriptor.shape[-1]}")
    return descriptor.nan_to_num(0.0, posinf=10.0, neginf=-10.0)


@torch.no_grad()
def extract_points_batch(
    episodes: Sequence[RealEpisode | SyntheticEpisode], *, source: str, device: torch.device
) -> list[ColumnPoint]:
    if not episodes:
        return []
    shape = (tuple(episodes[0].x_context.shape[1:]), tuple(episodes[0].x_query.shape[1:]))
    if any((tuple(item.x_context.shape[1:]), tuple(item.x_query.shape[1:])) != shape for item in episodes):
        raise ValueError("descriptor batches need equal context/query row and feature shapes")
    # Banks deliberately remain on CPU.  Only this shape-compatible slice is
    # resident on CUDA, preventing hundreds of stored episodes from consuming
    # VRAM before the actual descriptor reductions begin.
    context = torch.cat([item.x_context.float() for item in episodes], dim=0).to(device)
    labels = torch.cat([item.y_context.float() for item in episodes], dim=0).to(device)
    query = torch.cat([item.x_query.float() for item in episodes], dim=0).to(device)
    values = query_marginal_descriptors(context, labels, query).cpu().numpy().astype(np.float64)
    output: list[ColumnPoint] = []
    for row, episode in enumerate(episodes):
        identity, episode_id, n_context, n_query, n_numerical, n_classes = episode_identity(episode)
        for column in np.flatnonzero(numerical_mask(episode).numpy()):
            output.append(
                ColumnPoint(
                    source=source,
                    identity=identity,
                    episode_id=episode_id,
                    column=int(column),
                    n_context=n_context,
                    n_query=n_query,
                    n_features=int(episode.x_context.shape[-1]),
                    n_numerical_features=n_numerical,
                    n_classes=n_classes,
                    descriptor=values[row, column],
                )
            )
    return output


def extract_points(
    episodes: Sequence[RealEpisode | SyntheticEpisode], *, source: str, batch_size: int, progress_every: int,
    device: torch.device,
) -> list[ColumnPoint]:
    """Batch descriptor extraction by exact shape; CUDA is optional acceleration."""
    grouped: dict[tuple[tuple[int, int], tuple[int, int]], list[RealEpisode | SyntheticEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[(tuple(episode.x_context.shape[1:]), tuple(episode.x_query.shape[1:]))].append(episode)
    reporter = Progress(f"descriptors:{source}", len(episodes), progress_every)
    result: list[ColumnPoint] = []
    for key in sorted(grouped):
        values = grouped[key]
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            result.extend(extract_points_batch(batch, source=source, device=device))
            reporter.update(len(batch))
    return result


def descriptor_matrix(points: Sequence[ColumnPoint]) -> np.ndarray:
    if not points:
        raise ValueError("expected at least one column point")
    return np.stack([point.descriptor for point in points]).astype(np.float64, copy=False)


def fit_robust_scaler(points: Sequence[ColumnPoint], *, clip: float) -> RobustScaler:
    values = descriptor_matrix(points)
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return RobustScaler(center, scale, clip)


def nearest_distances(reference: np.ndarray, targets: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(reference)
    return model.kneighbors(targets, return_distance=True)[0][:, 0]


def source_auc(real_points: Sequence[ColumnPoint], synthetic_points: Sequence[ColumnPoint], *, seed: int) -> dict:
    """Source separability with dataset/task-group-disjoint folds.

    AUC near 0.5 means a linear classifier cannot distinguish a real numerical
    column from the synthetic source after robust scaling.  Splits are grouped
    by dataset/task identity so sibling columns and repeat splits never leak
    between a classifier's train and test folds.
    """
    combined = list(real_points) + list(synthetic_points)
    labels = np.asarray([0] * len(real_points) + [1] * len(synthetic_points))
    groups = np.asarray([f"real::{point.identity}" for point in real_points] + [f"synthetic::{point.identity}" for point in synthetic_points])
    values = descriptor_matrix(combined)
    n_real_groups = len({point.identity for point in real_points})
    n_synthetic_groups = len({point.identity for point in synthetic_points})
    n_splits = min(5, n_real_groups, n_synthetic_groups)
    if n_splits < 2:
        return {"auc": float("nan"), "n_splits": n_splits, "n_points": len(combined), "reason": "too_few_groups"}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    held_labels, held_scores = [], []
    for train, test in splitter.split(values, labels, groups):
        # Fit every numeric preprocessing quantity inside the fold.  This is
        # stricter than necessary for an unsupervised scaler, but guarantees
        # that no sibling dataset/task descriptor influences a held-out source
        # prediction through any path.
        real_training_points = [combined[index] for index in train if labels[index] == 0]
        scaler = fit_robust_scaler(real_training_points, clip=10.0)
        model = LogisticRegression(max_iter=2_000, class_weight="balanced", C=1.0)
        model.fit(scaler.transform(values[train]), labels[train])
        held_labels.append(labels[test])
        held_scores.append(model.predict_proba(scaler.transform(values[test]))[:, 1])
    return {
        "auc": float(roc_auc_score(np.concatenate(held_labels), np.concatenate(held_scores))),
        "n_splits": n_splits,
        "n_points": len(combined),
        "n_real_groups": n_real_groups,
        "n_synthetic_groups": n_synthetic_groups,
    }


def descriptor_effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    weights = singular.square() / max(float(singular.square().sum()), 1e-12)
    entropy = -(weights[weights > 0] * np.log(weights[weights > 0])).sum()
    return float(math.exp(entropy))


def source_profile(points: Sequence[ColumnPoint], *, scaler: RobustScaler) -> dict:
    values = scaler.transform(descriptor_matrix(points))
    return {
        "n_points": len(points),
        "n_identities": len({point.identity for point in points}),
        "n_context_min": min(point.n_context for point in points),
        "n_context_max": max(point.n_context for point in points),
        "n_query_min": min(point.n_query for point in points),
        "n_query_max": max(point.n_query for point in points),
        "n_features_min": min(point.n_features for point in points),
        "n_features_max": max(point.n_features for point in points),
        "n_classes_min": min(point.n_classes for point in points),
        "n_classes_max": max(point.n_classes for point in points),
        "descriptor_effective_rank": descriptor_effective_rank(values),
    }


def validate_matched_synthetic_banks(native: Sequence[SyntheticEpisode], expanded: Sequence[SyntheticEpisode]) -> dict:
    if len(native) != len(expanded):
        raise ValueError("matched synthetic banks differ in number of tasks")
    changed_columns, total_columns = 0, 0
    for first, second in zip(native, expanded, strict=True):
        if (
            first.task_id != second.task_id
            or first.n_classes != second.n_classes
            or first.x_context.shape != second.x_context.shape
            or first.x_query.shape != second.x_query.shape
            or not torch.equal(first.y_context.cpu(), second.y_context.cpu())
            or not torch.equal(first.y_query.cpu(), second.y_query.cpu())
        ):
            raise ValueError("native and expanded synthetic banks are not metadata/label matched")
        raw_native = torch.cat((first.x_context.cpu(), first.x_query.cpu()), dim=1)
        raw_expanded = torch.cat((second.x_context.cpu(), second.x_query.cpu()), dim=1)
        changed_columns += int((raw_native != raw_expanded).any(dim=1).sum())
        total_columns += raw_native.shape[-1]
    return {
        "matched_tasks": len(native),
        "label_and_shape_matched": True,
        "changed_column_fraction": changed_columns / max(total_columns, 1),
    }


def column_rows(points: Sequence[ColumnPoint]) -> list[dict]:
    rows = []
    for point in points:
        row = {
            "source": point.source,
            "identity": point.identity,
            "episode_id": point.episode_id,
            "column": point.column,
            "n_context": point.n_context,
            "n_query": point.n_query,
            "n_features": point.n_features,
            "n_numerical_features": point.n_numerical_features,
            "n_classes": point.n_classes,
        }
        row.update({f"descriptor_{index:02d}": float(value) for index, value in enumerate(point.descriptor)})
        rows.append(row)
    return rows


def evaluate_direct_headroom(
    args: argparse.Namespace,
    native: Sequence[SyntheticEpisode],
    expanded: Sequence[SyntheticEpisode],
) -> list[dict]:
    """Optional oracle diagnostic; it has no path back to the coverage audit."""
    if args.headroom_tasks <= 0:
        return []
    device = torch.device(args.headroom_device or args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--headroom-device requested CUDA but CUDA is unavailable")
    backbone, _ = load_backbone(args, device)
    results: list[dict] = []
    selected_native = list(native[: args.headroom_tasks])
    selected_expanded = list(expanded[: args.headroom_tasks])
    reporter = Progress("direct_headroom", 2 * len(selected_native), 1)
    for source, episodes in (("native", selected_native), ("coverage_expanded", selected_expanded)):
        for episode in episodes:
            x_context = episode.x_context.to(device)
            x_query = episode.x_query.to(device)
            y_context = episode.y_context.to(device)
            y_query = episode.y_query.to(device)
            spline = DirectSplineTransform(
                x_context, args.n_control_points, trainable_shape=True, trainable_location_scale=True
            ).to(device)
            optimizer = torch.optim.Adam(spline.parameters(), lr=args.headroom_lr)

            def loss_value() -> torch.Tensor:
                backbone.clear_cache()
                transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
                logits = backbone(transformed, y_context)
                return F.cross_entropy(logits.flatten(0, 1), y_query.flatten())

            with torch.no_grad():
                initial = float(loss_value())
            for _ in range(args.headroom_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_value()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final = float(loss_value())
            results.append(
                {
                    "source": source,
                    "task_id": episode.task_id,
                    "n_context": int(x_context.shape[1]),
                    "n_query": int(x_query.shape[1]),
                    "n_features": int(x_context.shape[-1]),
                    "initial_nll": initial,
                    "final_nll": final,
                    "nll_improvement": initial - final,
                    "steps": args.headroom_steps,
                }
            )
            reporter.update()
            del spline, optimizer
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-train-bank", type=Path, required=True)
    parser.add_argument("--real-validation-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu is sufficient; cuda batches descriptor reductions if available.")
    parser.add_argument("--summary-batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--synthetic-tasks", type=int, default=192)
    parser.add_argument("--synthetic-sequence-lengths", type=parse_int_csv, default=parse_int_csv("128,256,512,1024"))
    parser.add_argument("--synthetic-context-fractions", type=parse_float_csv, default=parse_float_csv("0.50,0.70,0.85"))
    parser.add_argument("--synthetic-seed", type=int, default=71_001)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--scaler-clip", type=float, default=10.0)
    parser.add_argument("--auc-seed", type=int, default=17)
    # Optional DirectSpline oracle diagnostic.  It does not run by default.
    parser.add_argument("--headroom-tasks", type=int, default=0)
    parser.add_argument("--headroom-steps", type=int, default=300)
    parser.add_argument("--headroom-lr", type=float, default=0.03)
    parser.add_argument(
        "--headroom-device",
        default=None,
        help="Optional DirectSpline GPU; defaults to --device rather than silently using cuda:0.",
    )
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    args = parser.parse_args()
    if args.summary_batch_size <= 0 or args.progress_every <= 0 or args.synthetic_tasks <= 0:
        raise ValueError("summary batch size, progress interval, and synthetic tasks must be positive")
    if not 0 < args.min_features <= args.max_features or args.max_classes < 2:
        raise ValueError("invalid synthetic feature/class limits")
    if args.headroom_tasks < 0 or args.headroom_steps <= 0 or args.headroom_lr <= 0:
        raise ValueError("invalid optional DirectSpline headroom arguments")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requested CUDA but CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Do not materialize the entire audit corpus on CUDA.  ``extract_points``
    # transfers one homogeneous mini-batch at a time; optional DirectSpline
    # headroom separately moves its one task at a time.
    storage_device = torch.device("cpu")
    _, real_train = load_bank(args.real_train_bank, device=storage_device)
    _, real_validation = load_bank(args.real_validation_bank, device=storage_device)
    if not real_train or not real_validation:
        raise ValueError("real train and validation banks must both contain episodes")
    print(
        f"Loaded real banks: train={len(real_train)} episodes/{len({item.dataset for item in real_train})} datasets, "
        f"validation={len(real_validation)} episodes/{len({item.dataset for item in real_validation})} datasets; "
        f"descriptor=query_marginal[{QUERY_MARGINAL_DIM}]",
        flush=True,
    )
    schedule = dict(
        sequence_lengths=args.synthetic_sequence_lengths,
        context_fractions=args.synthetic_context_fractions,
    )
    print(
        f"Generating matched synthetic banks: tasks={args.synthetic_tasks}, rows={args.synthetic_sequence_lengths}, "
        f"context_fractions={args.synthetic_context_fractions}, prior={args.prior_type}",
        flush=True,
    )
    native = generate_scheduled_episodes(
        args, args.synthetic_tasks, source_seed=args.synthetic_seed, task_offset=0, device=storage_device,
        observation_mode="native", **schedule,
    )
    expanded = generate_scheduled_episodes(
        args, args.synthetic_tasks, source_seed=args.synthetic_seed, task_offset=0, device=storage_device,
        observation_mode="coverage_expanded", **schedule,
    )
    matching = validate_matched_synthetic_banks(native, expanded)
    print(
        f"Matched banks verified: {matching['matched_tasks']} tasks, "
        f"changed synthetic columns={matching['changed_column_fraction']:.1%}",
        flush=True,
    )
    sources: dict[str, Sequence[RealEpisode | SyntheticEpisode]] = {
        "real_train": real_train,
        "real_validation": real_validation,
        "synthetic_native": native,
        "synthetic_coverage_expanded": expanded,
    }
    points = {
        name: extract_points(
            items, source=name, batch_size=args.summary_batch_size,
            progress_every=args.progress_every, device=device,
        )
        for name, items in sources.items()
    }
    for name, values in points.items():
        print(f"[{name}] extracted {len(values)} numerical-column descriptors", flush=True)
        write_csv(args.output_dir / f"{name}_columns.csv", column_rows(values))
    scaler = fit_robust_scaler(points["real_train"], clip=args.scaler_clip)
    train_values = scaler.transform(descriptor_matrix(points["real_train"]))
    validation_values = scaler.transform(descriptor_matrix(points["real_validation"]))
    native_values = scaler.transform(descriptor_matrix(points["synthetic_native"]))
    expanded_values = scaler.transform(descriptor_matrix(points["synthetic_coverage_expanded"]))
    distances = {
        "nearest_real_train": nearest_distances(train_values, validation_values),
        "nearest_synthetic_native": nearest_distances(native_values, validation_values),
        "nearest_synthetic_coverage_expanded": nearest_distances(expanded_values, validation_values),
    }
    coverage_rows = []
    for index, point in enumerate(points["real_validation"]):
        native_distance = float(distances["nearest_synthetic_native"][index])
        expanded_distance = float(distances["nearest_synthetic_coverage_expanded"][index])
        coverage_rows.append(
            {
                "identity": point.identity,
                "episode_id": point.episode_id,
                "column": point.column,
                "n_context": point.n_context,
                "n_query": point.n_query,
                "n_features": point.n_features,
                "n_classes": point.n_classes,
                "nearest_real_train_distance": float(distances["nearest_real_train"][index]),
                "nearest_synthetic_native_distance": native_distance,
                "nearest_synthetic_coverage_expanded_distance": expanded_distance,
                "expanded_minus_native_distance": expanded_distance - native_distance,
                "expanded_is_closer_than_native": expanded_distance < native_distance,
            }
        )
    write_csv(args.output_dir / "real_validation_coverage.csv", coverage_rows)
    auc_rows = []
    for source in ("synthetic_native", "synthetic_coverage_expanded"):
        auc_rows.append({"synthetic_source": source, **source_auc(points["real_train"], points[source], seed=args.auc_seed)})
    write_csv(args.output_dir / "source_separability.csv", auc_rows)
    headroom_rows = evaluate_direct_headroom(args, native, expanded)
    if headroom_rows:
        write_csv(args.output_dir / "synthetic_direct_headroom.csv", headroom_rows)
    summary = {
        "protocol": {
            "descriptor": "23 query unlabeled marginals + 8 context class-invariant label statistics + 2 context/query alignment values",
            "descriptor_dimension": QUERY_MARGINAL_DIM,
            "real_scaler_fit": "real_train only",
            "coverage_target": "real_validation only; never used to construct native or coverage-expanded tasks",
            "nearest_neighbor": "Euclidean after real-train robust coordinate scaling",
            "source_auc": "group-disjoint logistic regression, groups are dataset/task identities",
            "missingness": "not injected: current numeric synthetic episode path has no missing-value mask",
            "direct_headroom": "optional query-label oracle diagnostic; excluded from all coverage calculations",
        },
        "arguments": vars(args),
        "matched_synthetic_banks": matching,
        "source_profiles": {name: source_profile(values, scaler=scaler) for name, values in points.items()},
        "real_validation_coverage": {
            "n_columns": len(coverage_rows),
            "mean_nearest_real_train": float(np.mean(distances["nearest_real_train"])),
            "mean_nearest_native": float(np.mean(distances["nearest_synthetic_native"])),
            "mean_nearest_expanded": float(np.mean(distances["nearest_synthetic_coverage_expanded"])),
            "median_nearest_native": float(np.median(distances["nearest_synthetic_native"])),
            "median_nearest_expanded": float(np.median(distances["nearest_synthetic_coverage_expanded"])),
            "expanded_closer_fraction": float(np.mean([row["expanded_is_closer_than_native"] for row in coverage_rows])),
            "mean_expanded_minus_native_distance": float(np.mean([row["expanded_minus_native_distance"] for row in coverage_rows])),
        },
        "source_separability": auc_rows,
        "optional_direct_headroom": {
            "enabled": bool(headroom_rows),
            "n_records": len(headroom_rows),
            "mean_nll_improvement_by_source": {
                source: float(np.mean([row["nll_improvement"] for row in headroom_rows if row["source"] == source]))
                for source in ("native", "coverage_expanded")
            } if headroom_rows else {},
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        "Coverage audit finished: "
        f"validation nearest-native={summary['real_validation_coverage']['mean_nearest_native']:.4f}, "
        f"nearest-expanded={summary['real_validation_coverage']['mean_nearest_expanded']:.4f}, "
        f"expanded-closer={summary['real_validation_coverage']['expanded_closer_fraction']:.1%}. "
        f"Outputs: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
