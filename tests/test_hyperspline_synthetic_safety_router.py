"""Unit tests for the fixed-candidate synthetic safety-router protocol."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import hyperspline_synthetic_safety_router as router
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline import HyperSplineTransform


class _ToyBackbone:
    def clear_cache(self) -> None:
        pass

    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        query = features[:, labels.shape[1] :, 0]
        return torch.stack((query, -query), dim=-1)


def _episode() -> SyntheticEpisode:
    return SyntheticEpisode(
        task_id=4,
        source_seed=7,
        x_context=torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]]),
        x_query=torch.tensor([[[0.5, 0.5], [2.5, 0.5]]]),
        y_context=torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        y_query=torch.tensor([0, 1]),
        n_classes=2,
        observation_mode="coverage_expanded",
    )


def _routing_data() -> dict[str, np.ndarray]:
    return {
        "features": np.zeros((4, 3), dtype=np.float32),
        "task_id": np.arange(4),
        "n_classes": np.full(4, 2),
        "identity_nll": np.full(4, 0.70),
        "identity_accuracy": np.full(4, 0.50),
        "identity_auc": np.full(4, 0.50),
        "identity_deployment_error": np.full(4, 0.50),
        # The high-probability tasks are genuine wins; the low-probability
        # ones are material AUC harms.
        "candidate_nll": np.asarray([0.65, 0.65, 0.75, 0.75]),
        "candidate_accuracy": np.asarray([0.60, 0.60, 0.40, 0.40]),
        "candidate_auc": np.asarray([0.60, 0.60, 0.40, 0.40]),
        "candidate_deployment_error": np.asarray([0.40, 0.40, 0.60, 0.60]),
        "candidate_win": np.asarray([1, 1, 0, 0]),
        "candidate_material_harm": np.asarray([0, 0, 1, 1]),
    }


def test_descriptor_is_invariant_to_hidden_query_labels():
    torch.manual_seed(1)
    candidate = HyperSplineTransform(
        n_control_points=6,
        hidden_dim=8,
        target_aware=True,
        conditioning_mode="query_marginal",
        capacity_matched_conditioning=True,
    ).eval()
    episode = _episode()
    changed_labels = replace(episode, y_query=torch.tensor([1, 0]))
    first, names, _ = router.router_descriptor(_ToyBackbone(), candidate, episode)
    second, other_names, _ = router.router_descriptor(_ToyBackbone(), candidate, changed_labels)
    assert names == other_names
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_feature_sets_are_nested_and_router_runs_are_isolated():
    groups = ["current", "shift_global", "probe", "transform"]
    assert router.feature_indices(groups, "current").tolist() == [0]
    assert router.feature_indices(groups, "shift_global").tolist() == [0, 1]
    assert router.feature_indices(groups, "shift_global_probe").tolist() == [0, 1, 2]
    assert router.feature_indices(groups, "all").tolist() == [0, 1, 2, 3]
    output_dir = Path("routing-output")
    assert router.router_path(output_dir, "current_mlp") != router.router_path(output_dir, "all_mlp")


def test_validation_threshold_abstains_on_predicted_material_harms():
    threshold, summary, table = router.choose_threshold(
        _routing_data(), np.asarray([0.90, 0.80, 0.20, 0.10]), [0.0, 0.50], max_material_harm_ratio=0.75
    )
    assert threshold == pytest.approx(0.50)
    assert summary["material_harms"] == 0
    assert summary["applied_tasks"] == 2
    # Applying every task would include the two designed material harms.
    assert next(row for row in table if row["threshold"] == 0.0)["material_harms"] == 2
