"""Backbone-free protocol tests for the synthetic zero-shot benchmark."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from scripts import hyperspline_synthetic_zero_shot as zero_shot
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline import HyperSplineTransform


class _ToyBackbone:
    """Small differentiable stand-in that returns logits only for query rows."""

    def clear_cache(self) -> None:
        pass

    def __call__(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        query = features[:, labels.shape[1] :, 0]
        return torch.stack((query, -query), dim=-1)


def _episode() -> SyntheticEpisode:
    return SyntheticEpisode(
        task_id=1,
        source_seed=1,
        x_context=torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]]),
        x_query=torch.tensor([[[0.5, 0.5], [2.5, 0.5]]]),
        y_context=torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        y_query=torch.tensor([0, 1]),
        n_classes=2,
        observation_mode="coverage_expanded",
    )


def test_paired_elo_delta_counts_wins_losses_and_ties():
    # Candidate wins twice, loses once, and ties once: 62.5% game score.
    result = zero_shot.paired_elo_delta([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.0])
    assert (result["wins"], result["losses"], result["ties"]) == (2, 1, 1)
    assert result["score"] == pytest.approx(0.625)
    assert result["elo_delta"] > 0


def test_binary_deployment_surrogate_rewards_correct_ranking():
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[[2.0, -2.0], [1.0, -1.0], [-1.0, 1.0], [-2.0, 2.0]]])
    bad = good.flip(-1)
    assert zero_shot.deployment_surrogate(good, labels, 2, 0.1) < zero_shot.deployment_surrogate(bad, labels, 2, 0.1)


def test_cdf_shape_arm_uses_rich_input_without_location_scale_outputs():
    args = argparse.Namespace(
        arm="cdf_elo_shape", n_control_points=20, hidden_dim=64,
        gate_initial_probability=0.1, target_aware=True,
        conditioning_mode="query_marginal", capacity_matched_conditioning=True,
        cdf_quantiles=33, cdf_num_heads=4,
    )
    config = zero_shot.hyperspline_config(args)
    assert config["conditioning_mode"] == "cdf"
    assert config["generate_location"] is False
    assert config["generate_scale"] is False
    assert config["gate_location_scale"] is False
    assert config["capacity_matched_conditioning"] is False


def test_cdf_conditioner_is_identity_initialized_and_row_invariant():
    torch.manual_seed(3)
    model = HyperSplineTransform(
        n_control_points=8, hidden_dim=16, conditioning_mode="cdf",
        target_aware=True, generate_location=True, generate_scale=True,
        gate_location_scale=True, cdf_quantiles=9, cdf_num_heads=4,
    )
    episode = _episode()
    context, query = model(episode.x_context, episode.x_query, y_context=episode.y_context)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_context, permuted_query = model(
        episode.x_context[:, permutation], episode.x_query,
        y_context=episode.y_context[:, permutation],
    )
    assert torch.allclose(query, permuted_query, atol=1e-6)
    assert torch.allclose(context[:, permutation], permuted_context, atol=1e-6)


def test_training_stream_seed_wraps_at_numpy_uint32_boundary_without_repeating():
    # The prior failure occurred here: old arithmetic exceeded NumPy's valid
    # [0, 2**32 - 1] seed range around step 4,295.
    values = [zero_shot.training_source_seed(61_001, step) for step in range(1, 5_001)]
    assert all(0 <= value < 2**32 for value in values)
    assert len(values) == len(set(values))
    # Keep the already-completed pre-wrap run reproducible.
    assert values[0] == 61_001 + 1_000_003


def test_intermediate_checkpoint_budget_can_be_added_without_changing_primary_schedule():
    args = argparse.Namespace(scale_tasks=[40_000, 160_000, 640_000], extra_checkpoint_tasks=[80_000])
    assert zero_shot.checkpoint_budgets(args) == [40_000, 80_000, 160_000, 640_000]
    args.extra_checkpoint_tasks = [40_000]
    with pytest.raises(ValueError, match="must not duplicate"):
        zero_shot.checkpoint_budgets(args)


def test_probability_normalisation_removes_float32_softmax_row_sum_drift():
    raw = np.asarray([[0.10000001, 0.2, 0.7], [0.3, 0.3, 0.40000004]], dtype=np.float32)
    probabilities = zero_shot.normalise_probability_rows(raw)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=0.0, atol=1e-15)
    assert probabilities.dtype == np.float64


def test_query_marginal_transform_receives_query_features_but_not_query_labels():
    torch.manual_seed(3)
    model = HyperSplineTransform(
        n_control_points=6,
        hidden_dim=8,
        target_aware=True,
        conditioning_mode="query_marginal",
        capacity_matched_conditioning=True,
    ).eval()
    # Make the initially zero output head visibly depend on the conditioner.
    with torch.no_grad():
        model.mlp[-1].weight.zero_()
        model.mlp[-1].bias.zero_()
        model.mlp[-1].weight[5].fill_(1.0)  # The scalar spline gate.
    episode = _episode()
    _, _, first = model(episode.x_context, episode.x_query, y_context=episode.y_context, return_parameters=True)
    # An unlabeled query-location shift changes the two alignment features.
    _, _, shifted = model(episode.x_context, episode.x_query + 5.0, y_context=episode.y_context, return_parameters=True)
    assert not torch.allclose(first.gate, shifted.gate)


def test_identity_baseline_matches_an_untrained_query_marginal_hyperspline():
    torch.manual_seed(4)
    model = HyperSplineTransform(
        n_control_points=6,
        hidden_dim=8,
        gate_initial_probability=0.1,
        target_aware=True,
        conditioning_mode="query_marginal",
        capacity_matched_conditioning=True,
    ).eval()
    episode, backbone = _episode(), _ToyBackbone()
    identity_loss, identity_logits = zero_shot.forward_identity(backbone, episode)
    model_loss, model_logits, _ = zero_shot.forward_hyperspline(backbone, model, episode)
    torch.testing.assert_close(model_logits, identity_logits)
    torch.testing.assert_close(model_loss, identity_loss)


def test_seed_averaging_keeps_one_elo_game_per_table():
    rows = [
        {
            "task_id": 3,
            "identity_nll": 1.0,
            "identity_accuracy": 0.5,
            "identity_auc": 0.5,
            "identity_deployment_error": 0.5,
            "candidate_nll": candidate_nll,
            "candidate_accuracy": candidate_accuracy,
            "candidate_auc": candidate_auc,
            "candidate_deployment_error": candidate_error,
        }
        for candidate_nll, candidate_accuracy, candidate_auc, candidate_error in ((0.8, 0.6, 0.6, 0.4), (1.0, 0.5, 0.5, 0.5))
    ]
    averaged = zero_shot.average_model_seed_rows(rows)
    assert len(averaged) == 1
    assert averaged[0]["model_seed_runs"] == 2
    assert averaged[0]["candidate_nll"] == pytest.approx(0.9)
    assert averaged[0]["candidate_deployment_error"] == pytest.approx(0.45)


def test_training_distribution_must_match_prepared_banks():
    args = argparse.Namespace(
        prior_type="mix_scm",
        min_features=5,
        max_features=100,
        max_classes=10,
        prior_n_jobs=1,
        synthetic_observation_mode="coverage_expanded",
        sequence_lengths=[128, 256],
        context_fractions=[0.5, 0.7],
    )
    manifest = {"bank_generation": zero_shot._bank_config(args)}
    zero_shot.validate_training_distribution(args, manifest)
    args.max_features = 101
    with pytest.raises(ValueError, match="fresh training distribution"):
        zero_shot.validate_training_distribution(args, manifest)


def test_report_aggregate_tracks_metric_magnitude_not_only_elo():
    rows = [
        {
            "identity_nll": 1.0,
            "candidate_nll": 0.97,
            "identity_accuracy": 0.50,
            "candidate_accuracy": 0.53,
            "identity_auc": 0.60,
            "candidate_auc": 0.61,
            "identity_deployment_error": 0.40,
            "candidate_deployment_error": 0.39,
        },
        {
            "identity_nll": 1.0,
            "candidate_nll": 1.02,
            "identity_accuracy": 0.50,
            "candidate_accuracy": 0.48,
            "identity_auc": np.nan,
            "candidate_auc": np.nan,
            "identity_deployment_error": 1.0,
            "candidate_deployment_error": 1.02,
        },
    ]
    result = zero_shot.aggregate_report(rows, bootstrap_seed=7, bootstrap_samples=30)
    assert result["mean_nll_delta"] == pytest.approx(-0.005)
    assert result["binary_tables"] == 1
    assert result["mean_auc_delta"] == pytest.approx(0.01)
    assert result["elo"]["n_tasks"] == 2
