"""Backbone-free protocol tests for the synthetic zero-shot benchmark."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from scripts import hyperspline_synthetic_zero_shot as zero_shot


def test_paired_elo_delta_counts_wins_losses_and_ties():
    # Candidate wins twice, loses once, and ties once: 62.5% game score.
    result = zero_shot.paired_elo_delta([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.0])
    assert (result["wins"], result["losses"], result["ties"]) == (2, 1, 1)
    assert result["score"] == pytest.approx(0.625)
    assert result["elo_delta"] > 0


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
