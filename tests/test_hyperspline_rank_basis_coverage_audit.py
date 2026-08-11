"""CPU-only invariants for the context-statistics coverage audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.hyperspline_rank_basis_coverage_audit import (
    ColumnPoint,
    _json_default,
    attach_within_dataset_radius,
    column_coverage_rows,
    dataset_coverage_rows,
    dataset_profiles,
    descriptor_matrix,
    episode_profiles,
    extract_column_points,
    fit_robust_scaler,
)
from scripts.hyperspline_real_zero_shot_eval import RealEpisode
from tabicl._hyperspline.statistics import SUMMARY_DIM


def _episode(labels: torch.Tensor) -> RealEpisode:
    context = torch.tensor(
        [[[0.0, 10.0, 1.0], [1.0, 11.0, 2.0], [2.0, 12.0, 5.0], [3.0, 13.0, 8.0],
          [4.0, 14.0, 1.0], [5.0, 15.0, 2.0], [6.0, 16.0, 5.0], [7.0, 17.0, 8.0]]]
    )
    return RealEpisode(
        dataset="pmlb_toy",
        dataset_group="real_meta",
        split_seed=7,
        n_context=8,
        n_query=2,
        n_features=3,
        n_numerical_features=2,
        n_categorical_features=1,
        n_classes=2,
        x_context=context,
        x_query=context[:, :2],
        y_context=labels.float().unsqueeze(0),
        y_query=torch.tensor([0, 1]),
        numerical_mask=torch.tensor([True, False, True]),
    )


def _point(source: str, dataset: str, episode_id: int, column: int, values: tuple[float, float]) -> ColumnPoint:
    return ColumnPoint(
        source=source,
        dataset=dataset,
        episode_id=episode_id,
        column=column,
        n_context=16,
        n_features=2,
        n_numerical_features=2,
        n_categorical_features=0,
        n_classes=2,
        descriptor=np.asarray(values, dtype=np.float64),
    )


def test_audit_uses_only_numerical_columns_and_is_class_id_invariant():
    original = extract_column_points(_episode(torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])), source="real_train")
    permuted = extract_column_points(_episode(torch.tensor([4, 9, 4, 9, 4, 9, 4, 9])), source="real_train")
    assert [point.column for point in original] == [0, 2]
    assert all(point.descriptor.shape == (SUMMARY_DIM,) for point in original)
    assert np.allclose(descriptor_matrix(original), descriptor_matrix(permuted), atol=1e-6)


def test_dataset_cloud_reports_within_context_radius_and_synthetic_coverage():
    points = [
        _point("real_train", "train_a", 0, 0, (0.0, 0.0)),
        _point("real_train", "train_a", 0, 1, (0.1, 0.1)),
        _point("real_train", "train_a", 1, 0, (0.02, 0.0)),
        _point("real_train", "train_a", 1, 1, (0.12, 0.1)),
        _point("real_train", "train_b", 0, 0, (10.0, 10.0)),
        _point("real_train", "train_b", 0, 1, (10.1, 10.1)),
        _point("real_final", "final_c", 0, 0, (5.0, 5.0)),
        _point("real_final", "final_c", 0, 1, (5.1, 5.1)),
        _point("real_final", "final_c", 1, 0, (5.02, 5.0)),
        _point("real_final", "final_c", 1, 1, (5.12, 5.1)),
        _point("synthetic", "synthetic_task_0", 0, 0, (5.0, 5.0)),
        _point("synthetic", "synthetic_task_0", 0, 1, (5.1, 5.1)),
    ]
    episodes = episode_profiles(points)
    profiles = dataset_profiles(episodes)
    scaler = fit_robust_scaler(descriptor_matrix([item for item in episodes if item.source == "real_train"]))
    coverage = dataset_coverage_rows(profiles, profile_scaler=scaler)
    attach_within_dataset_radius(coverage, episodes, scaler)
    final = next(row for row in coverage if row["dataset"] == "final_c")
    assert final["nearest_real_or_synthetic_source"] == "synthetic"
    assert final["synthetic_reduces_profile_distance"]
    assert np.isfinite(final["within_dataset_context_distance"])
    assert final["within_to_nearest_real_ratio"] < 1.0

    column_scaler = fit_robust_scaler(descriptor_matrix([item for item in points if item.source == "real_train"]))
    per_column = column_coverage_rows(points, column_scaler=column_scaler)
    final_columns = [row for row in per_column if row["source"] == "real_final"]
    assert final_columns and all(row["nearest_real_or_synthetic_source"] == "synthetic" for row in final_columns)
    assert all(row["synthetic_reduces_column_distance"] for row in final_columns)


def test_audit_summary_serializes_paths_and_numpy_values():
    payload = json.loads(
        json.dumps(
            {"path": Path("results/bank.pt"), "value": np.float64(0.125), "array": np.asarray([1, 2])},
            default=_json_default,
        )
    )
    assert payload["path"].endswith("bank.pt")
    assert payload["value"] == 0.125
    assert payload["array"] == [1, 2]
