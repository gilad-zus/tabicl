"""Audit statistical coverage for the teacher-free learned-rank HyperSpline.

This program does *not* load TabICL, fit a spline, optimise a model, or use a
DirectSpline teacher.  It only asks whether the labelled-context statistics
available to the HyperSpline cover the clouds of contexts seen at validation
and final evaluation.

The unit entering the HyperSpline is a numerical column in one labelled
context split.  A dataset is therefore represented as a cloud of such points,
and as an aggregate dataset profile.  The audit reports:

* per-column nearest-neighbour coverage by real meta-training contexts;
* per-dataset coverage by real training datasets, and by real plus synthetic;
* within-dataset split variability versus distance to the nearest other
  dataset; and
* source separability (real train/validation/final/synthetic) using only the
  fixed context-statistics representation.

It is a diagnostic for the dataset-shift hypothesis, not an evaluation and
not a source of HyperSpline training targets.  In particular, any optional
final datasets are read only after the existing experiments are complete.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

try:  # Supports imports from tests and direct ``python scripts/...`` execution.
    from scripts.hyperspline_real_task_bank import load_bank
    from scripts.hyperspline_real_zero_shot_eval import MIXED_SPECS, NUMERICAL_SPECS, RealEpisode, build_episode
    from scripts.hyperspline_synthetic_train import SyntheticEpisode, generate_episodes
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from hyperspline_real_task_bank import load_bank
    from hyperspline_real_zero_shot_eval import MIXED_SPECS, NUMERICAL_SPECS, RealEpisode, build_episode
    from hyperspline_synthetic_train import SyntheticEpisode, generate_episodes

from tabicl._hyperspline import summarize_context
from tabicl._hyperspline.statistics import SUMMARY_DIM


@dataclass(frozen=True)
class ColumnPoint:
    """One numerical-column, labelled-context descriptor."""

    source: str
    dataset: str
    episode_id: int
    column: int
    n_context: int
    n_features: int
    n_numerical_features: int
    n_categorical_features: int
    n_classes: int
    descriptor: np.ndarray


@dataclass(frozen=True)
class EpisodeProfile:
    """A set summary for one context episode's numerical-column cloud."""

    source: str
    dataset: str
    episode_id: int
    n_context: int
    n_features: int
    n_numerical_features: int
    n_categorical_features: int
    n_classes: int
    descriptor: np.ndarray


@dataclass(frozen=True)
class DatasetProfile:
    """An aggregate point for one dataset identity."""

    source: str
    dataset: str
    n_episodes: int
    descriptor: np.ndarray


@dataclass(frozen=True)
class RobustScaler:
    """Robust coordinate-wise scaling fit only on real meta-training data."""

    center: np.ndarray
    scale: np.ndarray
    clip: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(values, dtype=np.float64) - self.center) / self.scale
        return np.clip(transformed, -self.clip, self.clip)


def parse_int_csv(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected non-empty comma-separated unique integers")
    return result


def parse_name_csv(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected non-empty comma-separated unique names")
    return result


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf8")


def _json_default(value: object):
    """Serialize experiment arguments and NumPy diagnostics safely."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def write_summary_from_existing_output(args: argparse.Namespace) -> None:
    """Recover ``summary.json`` after an interrupted final serialization.

    CSV files are deliberately written before the summary.  This mode avoids
    repeating context extraction when an old version failed only at the final
    JSON serialization step.
    """
    required = {
        "column_context_points.csv",
        "episode_profiles.csv",
        "dataset_profiles.csv",
        "dataset_coverage.csv",
        "source_separability.csv",
    }
    missing = sorted(name for name in required if not (args.output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"--summarize-existing requires completed CSV outputs in {args.output_dir}; missing {missing}"
        )
    points = read_csv(args.output_dir / "column_context_points.csv")
    episodes = read_csv(args.output_dir / "episode_profiles.csv")
    datasets = read_csv(args.output_dir / "dataset_profiles.csv")
    coverage = read_csv(args.output_dir / "dataset_coverage.csv")
    separability = read_csv(args.output_dir / "source_separability.csv")
    correlation_path = args.output_dir / "result_coverage_correlations.csv"
    correlations = read_csv(correlation_path) if correlation_path.is_file() else []
    sources = ("real_train", "real_validation", "real_final", "synthetic")
    counts = {
        source: {
            "column_context_points": sum(row.get("source") == source for row in points),
            "context_episodes": sum(row.get("source") == source for row in episodes),
            "dataset_identities": sum(row.get("source") == source for row in datasets),
        }
        for source in sources
    }
    final_coverage = [row for row in coverage if row.get("source") == "real_final"]
    real_distances = [float(row["nearest_real_train_distance"]) for row in final_coverage]
    union_distances = [
        float(row["nearest_real_or_synthetic_distance"])
        for row in final_coverage
        if row.get("nearest_real_or_synthetic_distance", "")
    ]
    profile_dimension = sum(name.startswith("dataset_profile_") for name in datasets[0]) if datasets else 0
    summary = {
        "protocol": "context_statistics_coverage_and_split_stability_audit",
        "recovered_from_existing_csvs": True,
        "teacher_used": False,
        "tabicl_loaded": False,
        "query_features_or_labels_used": False,
        "summary_dimension": SUMMARY_DIM,
        "dataset_profile_dimension": profile_dimension,
        "counts": counts,
        "mean_final_nearest_real_train_distance": float(np.mean(real_distances)) if real_distances else None,
        "mean_final_nearest_real_or_synthetic_distance": float(np.mean(union_distances)) if union_distances else None,
        "source_separability": separability,
        "result_coverage_correlations": correlations,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    print("[recover] writing missing summary.json from existing CSV outputs", flush=True)
    write_json(args.output_dir / "summary.json", summary)


class ProgressReporter:
    """TTY-independent progress and ETA logging for long CPU/GPU summaries."""

    def __init__(self, label: str, total: int, every: int) -> None:
        self.label = label
        self.total = total
        self.every = max(1, every)
        self.completed = 0
        self.started = time.monotonic()
        print(f"[{label}] 0/{total} (0.0%)", flush=True)

    def update(self, count: int) -> None:
        previous = self.completed
        self.completed += count
        should_log = self.completed == self.total or self.completed // self.every > previous // self.every
        if not should_log:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.completed / elapsed
        remaining = (self.total - self.completed) / rate if rate else float("inf")
        print(
            f"[{self.label}] {self.completed}/{self.total} ({100 * self.completed / self.total:.1f}%) "
            f"rate={rate:.2f} episodes/s eta={remaining:.0f}s",
            flush=True,
        )


def _episode_metadata(episode: RealEpisode | SyntheticEpisode) -> tuple[str, int, int, int, int, int, int]:
    """Return immutable task metadata without inspecting query features/labels."""
    if isinstance(episode, RealEpisode):
        return (
            episode.dataset,
            int(episode.split_seed),
            int(episode.n_context),
            int(episode.n_features),
            int(episode.n_numerical_features),
            int(episode.n_categorical_features),
            int(episode.n_classes),
        )
    n_features = int(episode.x_context.shape[-1])
    return (
        f"synthetic_task_{episode.task_id}",
        int(episode.task_id),
        int(episode.x_context.shape[1]),
        n_features,
        n_features,
        0,
        int(episode.n_classes),
    )


def numerical_mask(episode: RealEpisode | SyntheticEpisode) -> torch.Tensor:
    if isinstance(episode, RealEpisode):
        return episode.numerical_mask.detach().cpu().bool()
    return torch.ones(episode.x_context.shape[-1], dtype=torch.bool)


@torch.no_grad()
def extract_column_points_batch(
    episodes: Sequence[RealEpisode | SyntheticEpisode], *, source: str
) -> list[ColumnPoint]:
    """Extract fixed HyperSpline summaries for same-shaped context episodes.

    Query rows and query labels are deliberately absent.  Categorical columns
    are excluded after calculating summaries, exactly as spline parameters are
    generated only for numerical columns.
    """
    if not episodes:
        return []
    shape = tuple(episodes[0].x_context.shape[1:])
    if any(tuple(episode.x_context.shape[1:]) != shape for episode in episodes):
        raise ValueError("summary batches require equal context row and feature dimensions")
    # Keep data on the selected device.  CPU remains the default, while CUDA
    # batches the costly quantile/reduction kernels for compatible episodes.
    context = torch.cat([episode.x_context.detach().float() for episode in episodes], dim=0)
    labels = torch.cat([episode.y_context.detach().float() for episode in episodes], dim=0)
    summary = summarize_context(context, y_context=labels).summary.detach().cpu().numpy().astype(np.float64)
    result = []
    for batch_index, episode in enumerate(episodes):
        dataset, episode_id, n_context, n_features, n_numerical, n_categorical, n_classes = _episode_metadata(episode)
        current = summary[batch_index]
        if current.shape != (n_features, SUMMARY_DIM):
            raise AssertionError(f"unexpected summary shape {current.shape}; expected {(n_features, SUMMARY_DIM)}")
        for column in np.flatnonzero(numerical_mask(episode).numpy()):
            result.append(
                ColumnPoint(
                    source=source,
                    dataset=dataset,
                    episode_id=episode_id,
                    column=int(column),
                    n_context=n_context,
                    n_features=n_features,
                    n_numerical_features=n_numerical,
                    n_categorical_features=n_categorical,
                    n_classes=n_classes,
                    descriptor=current[column],
                )
            )
    return result


def extract_column_points(episode: RealEpisode | SyntheticEpisode, *, source: str) -> list[ColumnPoint]:
    """Single-episode public helper retained for tests and programmatic callers."""
    return extract_column_points_batch([episode], source=source)


def extract_source_points(
    episodes: Sequence[RealEpisode | SyntheticEpisode], *, source: str, summary_batch_size: int, progress_every: int
) -> list[ColumnPoint]:
    """Group compatible contexts so ``--device cuda`` does useful batched work."""
    if not episodes:
        return []
    grouped: dict[tuple[int, int], list[RealEpisode | SyntheticEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[(int(episode.x_context.shape[1]), int(episode.x_context.shape[2]))].append(episode)
    reporter = ProgressReporter(f"summaries:{source}", len(episodes), progress_every)
    points = []
    for key in sorted(grouped):
        items = grouped[key]
        for start in range(0, len(items), summary_batch_size):
            batch = items[start : start + summary_batch_size]
            points.extend(extract_column_points_batch(batch, source=source))
            reporter.update(len(batch))
    return points


def fit_robust_scaler(values: np.ndarray, *, clip: float = 10.0) -> RobustScaler:
    """Fit a finite robust scaler with useful fallbacks for constant entries."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0]:
        raise ValueError("expected a non-empty two-dimensional reference matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("reference descriptors must be finite")
    center = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349  # IQR of a unit Gaussian is approximately 1.349.
    standard_deviation = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, standard_deviation)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return RobustScaler(center=center, scale=scale, clip=clip)


def descriptor_matrix(items: Sequence[ColumnPoint | EpisodeProfile | DatasetProfile]) -> np.ndarray:
    if not items:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack([item.descriptor for item in items]).astype(np.float64, copy=False)


def episode_profiles(points: Sequence[ColumnPoint]) -> list[EpisodeProfile]:
    """Aggregate a column cloud into one profile for each context episode."""
    grouped: dict[tuple[str, str, int], list[ColumnPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.source, point.dataset, point.episode_id)].append(point)
    profiles = []
    for (_, _, _), group in sorted(grouped.items()):
        first = group[0]
        matrix = descriptor_matrix(group)
        # Mean/standard deviation/five-number style description preserves
        # heterogeneity between numerical columns without pretending they are
        # independent datasets.
        descriptor = np.concatenate(
            (
                matrix.mean(axis=0),
                matrix.std(axis=0),
                np.quantile(matrix, 0.25, axis=0),
                np.quantile(matrix, 0.50, axis=0),
                np.quantile(matrix, 0.75, axis=0),
                np.asarray(
                    (
                        math.log1p(first.n_context),
                        math.log1p(first.n_features),
                        first.n_numerical_features / max(first.n_features, 1),
                        first.n_classes,
                    ),
                    dtype=np.float64,
                ),
            )
        )
        profiles.append(
            EpisodeProfile(
                source=first.source,
                dataset=first.dataset,
                episode_id=first.episode_id,
                n_context=first.n_context,
                n_features=first.n_features,
                n_numerical_features=first.n_numerical_features,
                n_categorical_features=first.n_categorical_features,
                n_classes=first.n_classes,
                descriptor=descriptor,
            )
        )
    return profiles


def dataset_profiles(profiles: Sequence[EpisodeProfile]) -> list[DatasetProfile]:
    grouped: dict[tuple[str, str], list[EpisodeProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile.source, profile.dataset)].append(profile)
    return [
        DatasetProfile(
            source=source,
            dataset=dataset,
            n_episodes=len(group),
            descriptor=descriptor_matrix(group).mean(axis=0),
        )
        for (source, dataset), group in sorted(grouped.items())
    ]


def nearest_neighbours(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean nearest-neighbour distances and row indices."""
    if not len(query) or not len(reference):
        return np.empty(0), np.empty(0, dtype=int)
    model = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(reference)
    distances, indices = model.kneighbors(query, return_distance=True)
    return distances[:, 0], indices[:, 0]


def mean_pairwise_distance(values: np.ndarray) -> float:
    """Mean upper-triangular Euclidean distance; NaN if fewer than two rows."""
    if len(values) < 2:
        return float("nan")
    differences = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.square(differences).sum(axis=-1))
    return float(distances[np.triu_indices(len(values), k=1)].mean())


def profile_rows(profiles: Sequence[EpisodeProfile | DatasetProfile], *, prefix: str) -> list[dict]:
    rows = []
    for profile in profiles:
        row = {
            "source": profile.source,
            "dataset": profile.dataset,
            **{f"{prefix}_{index}": float(value) for index, value in enumerate(profile.descriptor)},
        }
        if isinstance(profile, EpisodeProfile):
            row.update(
                {
                    "episode_id": profile.episode_id,
                    "n_context": profile.n_context,
                    "n_features": profile.n_features,
                    "n_numerical_features": profile.n_numerical_features,
                    "n_categorical_features": profile.n_categorical_features,
                    "n_classes": profile.n_classes,
                }
            )
        else:
            row["n_episodes"] = profile.n_episodes
        rows.append(row)
    return rows


def column_rows(points: Sequence[ColumnPoint]) -> list[dict]:
    return [
        {
            "source": point.source,
            "dataset": point.dataset,
            "episode_id": point.episode_id,
            "column": point.column,
            "n_context": point.n_context,
            "n_features": point.n_features,
            "n_numerical_features": point.n_numerical_features,
            "n_categorical_features": point.n_categorical_features,
            "n_classes": point.n_classes,
            **{f"summary_{index}": float(value) for index, value in enumerate(point.descriptor)},
        }
        for point in points
    ]


def _other_train_reference(
    item: DatasetProfile, train: Sequence[DatasetProfile], scaled_train: np.ndarray
) -> tuple[np.ndarray, list[DatasetProfile]]:
    if item.source != "real_train":
        return scaled_train, list(train)
    keep = [index for index, candidate in enumerate(train) if candidate.dataset != item.dataset]
    return scaled_train[keep], [train[index] for index in keep]


def dataset_coverage_rows(
    profiles: Sequence[DatasetProfile], *, profile_scaler: RobustScaler
) -> list[dict]:
    """Nearest real and synthetic dataset clouds plus within-cloud stability."""
    train = [item for item in profiles if item.source == "real_train"]
    synthetic = [item for item in profiles if item.source == "synthetic"]
    if not train:
        raise ValueError("at least one real_train dataset profile is required")
    scaled_all = profile_scaler.transform(descriptor_matrix(profiles))
    scaled_by_key = {
        (item.source, item.dataset): scaled_all[index] for index, item in enumerate(profiles)
    }
    scaled_train = profile_scaler.transform(descriptor_matrix(train))
    scaled_synthetic = profile_scaler.transform(descriptor_matrix(synthetic)) if synthetic else np.empty((0, scaled_train.shape[1]))

    # A dataset's radius is the variation across independently generated
    # labelled contexts.  It is not treated as extra dataset-level evidence.
    episode_by_dataset: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    # Stored temporarily by caller through an attribute-like key is ugly; the
    # rows below intentionally leave it empty and ``attach_within_radius`` adds
    # it from episode profiles.
    _ = episode_by_dataset

    rows = []
    for item in profiles:
        query = scaled_by_key[(item.source, item.dataset)].reshape(1, -1)
        real_reference, real_items = _other_train_reference(item, train, scaled_train)
        real_distance, real_index = nearest_neighbours(query, real_reference)
        nearest_real = real_items[int(real_index[0])] if len(real_index) else None
        row = {
            "source": item.source,
            "dataset": item.dataset,
            "n_episodes": item.n_episodes,
            "nearest_real_train_dataset": nearest_real.dataset if nearest_real else "",
            "nearest_real_train_distance": float(real_distance[0]) if len(real_distance) else float("nan"),
        }
        if item.source in {"real_validation", "real_final"} and synthetic:
            combined = np.concatenate((scaled_train, scaled_synthetic), axis=0)
            combined_items = [*train, *synthetic]
            distance, index = nearest_neighbours(query, combined)
            nearest = combined_items[int(index[0])]
            row.update(
                {
                    "nearest_real_or_synthetic_source": nearest.source,
                    "nearest_real_or_synthetic_dataset": nearest.dataset,
                    "nearest_real_or_synthetic_distance": float(distance[0]),
                    "synthetic_reduces_profile_distance": bool(float(distance[0]) < row["nearest_real_train_distance"]),
                }
            )
        rows.append(row)
    return rows


def attach_within_dataset_radius(rows: list[dict], profiles: Sequence[EpisodeProfile], scaler: RobustScaler) -> None:
    grouped: dict[tuple[str, str], list[EpisodeProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[(profile.source, profile.dataset)].append(profile)
    for row in rows:
        group = grouped[(row["source"], row["dataset"])]
        radius = mean_pairwise_distance(scaler.transform(descriptor_matrix(group)))
        row["within_dataset_context_distance"] = radius
        nearest = row["nearest_real_train_distance"]
        row["within_to_nearest_real_ratio"] = (
            float(radius / nearest) if np.isfinite(radius) and np.isfinite(nearest) and nearest > 0 else float("nan")
        )


def column_coverage_rows(points: Sequence[ColumnPoint], *, column_scaler: RobustScaler) -> list[dict]:
    """Per-column nearest real-training statistic coverage.

    Validation/final points also receive a second distance to the union of
    real-training and synthetic points.  Synthetic points never use that union
    because they would otherwise select themselves with distance zero.
    """
    train = [item for item in points if item.source == "real_train"]
    synthetic = [item for item in points if item.source == "synthetic"]
    if not train:
        raise ValueError("at least one real_train column point is required")
    scaled_train = column_scaler.transform(descriptor_matrix(train))
    scaled_synthetic = column_scaler.transform(descriptor_matrix(synthetic)) if synthetic else np.empty((0, scaled_train.shape[1]))
    result = []
    for source in ("real_validation", "real_final", "synthetic"):
        query_points = [item for item in points if item.source == source]
        if not query_points:
            continue
        query = column_scaler.transform(descriptor_matrix(query_points))
        distance, index = nearest_neighbours(query, scaled_train)
        union_distance: np.ndarray | None = None
        union_index: np.ndarray | None = None
        combined_points: list[ColumnPoint] | None = None
        if source in {"real_validation", "real_final"} and synthetic:
            combined = np.concatenate((scaled_train, scaled_synthetic), axis=0)
            combined_points = [*train, *synthetic]
            union_distance, union_index = nearest_neighbours(query, combined)
        for row_index, (point, current_distance, current_index) in enumerate(zip(query_points, distance, index, strict=True)):
            nearest = train[int(current_index)]
            row = {
                "source": point.source,
                "dataset": point.dataset,
                "episode_id": point.episode_id,
                "column": point.column,
                "nearest_real_train_dataset": nearest.dataset,
                "nearest_real_train_episode_id": nearest.episode_id,
                "nearest_real_train_column": nearest.column,
                "nearest_real_train_distance": float(current_distance),
            }
            if union_distance is not None and union_index is not None and combined_points is not None:
                nearest_union = combined_points[int(union_index[row_index])]
                row.update(
                    {
                        "nearest_real_or_synthetic_source": nearest_union.source,
                        "nearest_real_or_synthetic_dataset": nearest_union.dataset,
                        "nearest_real_or_synthetic_distance": float(union_distance[row_index]),
                        "synthetic_reduces_column_distance": bool(float(union_distance[row_index]) < float(current_distance)),
                    }
                )
            result.append(row)
    return result


def mean_rows_by_dataset(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["dataset"]))].append(row)
    summary = []
    for (source, dataset), group in sorted(grouped.items()):
        real_distances = np.asarray([float(row["nearest_real_train_distance"]) for row in group])
        row = {
            "source": source,
            "dataset": dataset,
            "n_column_context_points": len(group),
            "mean_nearest_real_train_column_distance": float(real_distances.mean()),
            "median_nearest_real_train_column_distance": float(np.median(real_distances)),
            "p90_nearest_real_train_column_distance": float(np.quantile(real_distances, 0.90)),
        }
        if "nearest_real_or_synthetic_distance" in group[0]:
            combined = np.asarray([float(item["nearest_real_or_synthetic_distance"]) for item in group])
            row.update(
                {
                    "mean_nearest_real_or_synthetic_column_distance": float(combined.mean()),
                    "synthetic_reduces_column_distance_fraction": float(
                        np.mean([bool(item["synthetic_reduces_column_distance"]) for item in group])
                    ),
                }
            )
        summary.append(row)
    return summary


def binary_source_auc(
    profiles: Sequence[DatasetProfile], *, source_a: str, source_b: str, seed: int
) -> dict:
    """Dataset-identity-level cross-validated source separability.

    Columns and bags are intentionally collapsed first: the score asks whether
    *dataset identities* are distinguishable, not whether repeated rows from
    the same dataset make a classifier look accurate.
    """
    selected = [profile for profile in profiles if profile.source in {source_a, source_b}]
    labels = np.asarray([int(profile.source == source_b) for profile in selected], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if len(selected) < 4 or counts.min() < 2:
        return {"source_a": source_a, "source_b": source_b, "n": len(selected), "auc_mean": float("nan"), "auc_std": float("nan")}
    folds = min(5, int(counts.min()))
    matrix = descriptor_matrix(selected)
    aucs = []
    for train_index, test_index in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(matrix, labels):
        scaler = fit_robust_scaler(matrix[train_index])
        classifier = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
        classifier.fit(scaler.transform(matrix[train_index]), labels[train_index])
        probabilities = classifier.predict_proba(scaler.transform(matrix[test_index]))[:, 1]
        aucs.append(float(roc_auc_score(labels[test_index], probabilities)))
    return {
        "source_a": source_a,
        "source_b": source_b,
        "n": len(selected),
        "n_source_a": int(counts[0]),
        "n_source_b": int(counts[1]),
        "folds": folds,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def pca_rows(profiles: Sequence[DatasetProfile], scaler: RobustScaler) -> list[dict]:
    if not profiles:
        return []
    matrix = scaler.transform(descriptor_matrix(profiles))
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    coordinates = PCA(n_components=n_components, random_state=0).fit_transform(matrix)
    if n_components == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    return [
        {"source": profile.source, "dataset": profile.dataset, "n_episodes": profile.n_episodes, "pc1": float(coordinate[0]), "pc2": float(coordinate[1])}
        for profile, coordinate in zip(profiles, coordinates, strict=True)
    ]


def attach_result_deltas(
    coverage_rows: Sequence[dict], result_paths: Sequence[Path]
) -> tuple[list[dict], list[dict]]:
    """Join optional final-dataset results for descriptive distance correlations."""
    final_rows = {(row["dataset"], row["source"]): dict(row) for row in coverage_rows if row["source"] == "real_final"}
    joined, correlations = [], []
    for path in result_paths:
        with path.open(newline="", encoding="utf8") as handle:
            result = list(csv.DictReader(handle))
        if not result or "dataset" not in result[0] or "loss_delta" not in result[0]:
            raise ValueError(f"{path} must contain dataset and loss_delta columns")
        current = []
        for row in result:
            key = (str(row["dataset"]), "real_final")
            if key not in final_rows:
                continue
            merged = {"result_file": path.name, **final_rows[key], "loss_delta": float(row["loss_delta"])}
            current.append(merged)
        joined.extend(current)
        x = np.asarray([float(row["nearest_real_train_distance"]) for row in current])
        y = np.asarray([float(row["loss_delta"]) for row in current])
        correlations.append(
            {
                "result_file": path.name,
                "n_final_datasets_joined": len(current),
                "pearson_distance_vs_loss_delta": (
                    float(np.corrcoef(x, y)[0, 1]) if len(current) >= 3 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
                ),
            }
        )
    return joined, correlations


def generate_synthetic_points(args: argparse.Namespace, *, device: torch.device) -> list[SyntheticEpisode]:
    if args.synthetic_tasks <= 0:
        return []
    synthetic_args = argparse.Namespace(
        prior_type=args.prior_type,
        sequence_length=args.synthetic_sequence_length,
        context_fraction=args.context_fraction,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        prior_n_jobs=args.prior_n_jobs,
        train_seed=args.synthetic_seed,
    )
    print(
        f"Generating {args.synthetic_tasks} synthetic contexts: prior={args.prior_type}, "
        f"rows={args.synthetic_sequence_length}, features={args.min_features}-{args.max_features}",
        flush=True,
    )
    return generate_episodes(
        synthetic_args,
        args.synthetic_tasks,
        source_seed=args.synthetic_seed,
        task_offset=0,
        device=device,
    )


def build_final_episodes(args: argparse.Namespace, *, device: torch.device) -> list[RealEpisode]:
    specs_by_name = {spec.name: spec for spec in (*NUMERICAL_SPECS, *MIXED_SPECS)}
    unknown = sorted(set(args.final_datasets).difference(specs_by_name))
    if unknown:
        raise ValueError(f"unknown --final-datasets {unknown}; choices are {sorted(specs_by_name)}")
    builder_args = argparse.Namespace(max_rows=args.final_max_rows, test_size=args.final_test_size)
    episodes = []
    for name in args.final_datasets:
        for seed in args.final_seeds:
            episodes.append(build_episode(specs_by_name[name], seed, builder_args, device))
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-train-bank", type=Path, default=None)
    parser.add_argument("--real-validation-bank", type=Path, default=None)
    parser.add_argument("--final-bank", type=Path, default=None, help="Optional serialized real bank used as real_final.")
    parser.add_argument(
        "--include-final", action="store_true",
        help="Build the seven current final datasets locally (may use cached OpenML data or download it if absent).",
    )
    parser.add_argument(
        "--final-datasets", type=parse_name_csv,
        default=[spec.name for spec in (*NUMERICAL_SPECS, *MIXED_SPECS)],
        help="Comma-separated names when --include-final is set.",
    )
    parser.add_argument("--final-seeds", type=parse_int_csv, default=list(range(10)))
    parser.add_argument("--final-max-rows", type=int, default=1024)
    parser.add_argument("--final-test-size", type=float, default=0.30)
    parser.add_argument("--synthetic-tasks", type=int, default=256, help="Fresh native-prior task identities; zero skips synthetic coverage.")
    parser.add_argument("--synthetic-seed", type=int, default=91_001)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--synthetic-sequence-length", type=int, default=512)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument(
        "--device", default="cpu",
        help="Where to compute context summaries. CUDA batches compatible episodes; no TabICL model is loaded.",
    )
    parser.add_argument("--summary-batch-size", type=int, default=32, help="Compatible context episodes per summary batch.")
    parser.add_argument("--progress-every", type=int, default=16, help="Print summary-extraction progress after this many episodes.")
    parser.add_argument("--descriptor-clip", type=float, default=10.0)
    parser.add_argument("--source-classifier-seed", type=int, default=0)
    parser.add_argument(
        "--results-per-dataset-csv", type=Path, action="append", default=[],
        help="Optional per_final_dataset.csv files; joins their loss_delta to real_final coverage descriptively.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/hyperspline_rank_basis_coverage_audit"))
    parser.add_argument(
        "--summarize-existing", action="store_true",
        help="Only write summary.json from already-complete CSV outputs; use after an interrupted old audit run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_existing:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_summary_from_existing_output(args)
        return
    if args.real_train_bank is None or args.real_validation_bank is None:
        raise ValueError("--real-train-bank and --real-validation-bank are required unless --summarize-existing is set")
    if args.final_bank is not None and args.include_final:
        raise ValueError("use only one of --final-bank and --include-final")
    if args.synthetic_tasks < 0 or args.synthetic_sequence_length < 4:
        raise ValueError("--synthetic-tasks must be non-negative and --synthetic-sequence-length must be at least four")
    if not 0 < args.context_fraction < 1 or not 0 < args.final_test_size < 1:
        raise ValueError("context and final test fractions must be between zero and one")
    if not 0 < args.min_features <= args.max_features or args.max_classes < 2:
        raise ValueError("invalid synthetic feature or class limits")
    if args.descriptor_clip <= 0 or args.summary_batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("--descriptor-clip, --summary-batch-size, and --progress-every must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device={args.device} was requested but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] device={device}; loading real context banks", flush=True)
    _, real_train = load_bank(args.real_train_bank, device=device)
    _, real_validation = load_bank(args.real_validation_bank, device=device)
    print(f"[load] real_train={len(real_train)} episodes; real_validation={len(real_validation)} episodes", flush=True)
    final: list[RealEpisode] = []
    if args.final_bank is not None:
        print(f"[load] loading final bank {args.final_bank}", flush=True)
        _, final = load_bank(args.final_bank, device=device)
    elif args.include_final:
        print(f"[load] building {len(args.final_datasets) * len(args.final_seeds)} final contexts", flush=True)
        final = build_final_episodes(args, device=device)
    if final:
        print(f"[load] real_final={len(final)} episodes", flush=True)
    synthetic = generate_synthetic_points(args, device=device)

    all_points = []
    for source, episodes in (
        ("real_train", real_train),
        ("real_validation", real_validation),
        ("real_final", final),
        ("synthetic", synthetic),
    ):
        all_points.extend(
            extract_source_points(
                episodes,
                source=source,
                summary_batch_size=args.summary_batch_size,
                progress_every=args.progress_every,
            )
        )
    print(f"[analysis] extracted {len(all_points)} numerical column/context points", flush=True)
    train_points = [point for point in all_points if point.source == "real_train"]
    if not train_points:
        raise ValueError("--real-train-bank contains no numerical column contexts")
    column_scaler = fit_robust_scaler(descriptor_matrix(train_points), clip=args.descriptor_clip)
    profiles = episode_profiles(all_points)
    train_episode_profiles = [profile for profile in profiles if profile.source == "real_train"]
    profile_scaler = fit_robust_scaler(descriptor_matrix(train_episode_profiles), clip=args.descriptor_clip)
    datasets = dataset_profiles(profiles)

    print("[analysis] computing profile and nearest-neighbour coverage", flush=True)
    coverage = dataset_coverage_rows(datasets, profile_scaler=profile_scaler)
    attach_within_dataset_radius(coverage, profiles, profile_scaler)
    point_coverage = column_coverage_rows(all_points, column_scaler=column_scaler)
    point_summary = mean_rows_by_dataset(point_coverage)
    pca = pca_rows(datasets, profile_scaler)
    source_pairs = (("real_train", "real_validation"), ("real_train", "real_final"), ("synthetic", "real_train"), ("synthetic", "real_final"))
    separability = [
        binary_source_auc(datasets, source_a=source_a, source_b=source_b, seed=args.source_classifier_seed)
        for source_a, source_b in source_pairs
    ]
    result_join, result_correlations = attach_result_deltas(coverage, args.results_per_dataset_csv)

    outputs = (
        ("column context points", args.output_dir / "column_context_points.csv", column_rows(all_points)),
        ("episode profiles", args.output_dir / "episode_profiles.csv", profile_rows(profiles, prefix="profile")),
        ("dataset profiles", args.output_dir / "dataset_profiles.csv", profile_rows(datasets, prefix="dataset_profile")),
        ("dataset coverage", args.output_dir / "dataset_coverage.csv", coverage),
        ("column coverage", args.output_dir / "column_coverage.csv", point_coverage),
        ("column coverage by dataset", args.output_dir / "column_coverage_by_dataset.csv", point_summary),
        ("dataset PCA", args.output_dir / "dataset_profile_pca.csv", pca),
        ("source separability", args.output_dir / "source_separability.csv", separability),
        ("result coverage join", args.output_dir / "result_coverage_join.csv", result_join),
        ("result coverage correlations", args.output_dir / "result_coverage_correlations.csv", result_correlations),
    )
    for label, path, rows in outputs:
        print(f"[write] {label}: {path.name} ({len(rows)} rows)", flush=True)
        write_csv(path, rows)

    counts = {
        source: {
            "column_context_points": sum(point.source == source for point in all_points),
            "context_episodes": sum(profile.source == source for profile in profiles),
            "dataset_identities": sum(profile.source == source for profile in datasets),
        }
        for source in ("real_train", "real_validation", "real_final", "synthetic")
    }
    final_coverage = [row for row in coverage if row["source"] == "real_final"]
    summary = {
        "protocol": "context_statistics_coverage_and_split_stability_audit",
        "teacher_used": False,
        "tabicl_loaded": False,
        "query_features_or_labels_used": False,
        "summary_dimension": SUMMARY_DIM,
        "dataset_profile_dimension": int(descriptor_matrix(datasets).shape[1]),
        "counts": counts,
        "mean_final_nearest_real_train_distance": (
            float(np.mean([row["nearest_real_train_distance"] for row in final_coverage])) if final_coverage else None
        ),
        "mean_final_nearest_real_or_synthetic_distance": (
            float(np.mean([row["nearest_real_or_synthetic_distance"] for row in final_coverage if "nearest_real_or_synthetic_distance" in row]))
            if any("nearest_real_or_synthetic_distance" in row for row in final_coverage) else None
        ),
        "source_separability": separability,
        "result_coverage_correlations": result_correlations,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    print("[write] summary: summary.json", flush=True)
    write_json(args.output_dir / "summary.json", summary)
    print(
        f"Wrote audit to {args.output_dir}: "
        f"{counts['real_train']['dataset_identities']} real-train dataset clouds, "
        f"{counts['real_validation']['dataset_identities']} validation, "
        f"{counts['real_final']['dataset_identities']} final, "
        f"{counts['synthetic']['dataset_identities']} synthetic task identities.",
        flush=True,
    )


if __name__ == "__main__":
    main()
