"""Checks for the staged checkpoint-size comparison."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direct_spline_staged_validation_size import _matched_config, _select_best  # noqa: E402


def test_checkpoint_choice_can_change_when_the_same_trajectory_gets_more_selection_rows():
    records = [
        {"step": 0, "small_error": 0.9, "large_error": 0.9},
        {"step": 25, "small_error": 0.7, "large_error": 0.8},
        {"step": 50, "small_error": 0.75, "large_error": 0.6},
        {"step": 75, "small_error": 0.7, "large_error": 0.6},
    ]
    assert _select_best(records, size="small") == 25
    assert _select_best(records, size="large") == 50


def test_staged_config_preserves_optimization_and_sampler_from_source():
    source = {
        "adapter_architecture": "fixed_cubic", "n_control_points": 20,
        "learning_rate": 0.005, "weight_decay": 0.003,
        "random_state": 20260828, "validation_interval": 25,
        "query_batch_rows": 256, "cosine_schedule_steps": 500,
        "cosine_min_lr_ratio": 0.01,
    }
    config = _matched_config(source, steps=250)
    assert config["adapter_steps"] == config["cosine_schedule_steps"] == 250
    assert config["random_state"] == source["random_state"]
    assert config["learning_rate"] == source["learning_rate"]
    assert config["query_fraction_min"] == pytest.approx(0.05)
    assert config["query_fraction_max"] == pytest.approx(0.20)
    assert config["coordinate_mapping"] == "arctan"
    assert config["trainable_shape"] is True
    assert config["direct_spline_output"] is True
