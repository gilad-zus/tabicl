import numpy as np
import pandas as pd
import pytest
import torch

from tabicl._experiments.direct_spline_openml import (
    OpenMLTaskData,
    _fit_one_bag,
    greedy_validation_ensemble,
    summarize_experiment,
)
from tabicl._experiments.direct_spline_protocol import (
    DEFAULT_DIRECT_SPLINE_CONFIG,
    FoldPreprocessor,
    benchmark_error,
    bootstrap_paired_elo,
    choose_identity_guard,
    deployment_error,
    paired_elo_delta,
    sample_episode_indices,
    sample_prediction_context,
    shared_random_direct_spline_configs,
)


def test_fold_preprocessor_uses_train_statistics_and_maps_unknown_categories():
    train = pd.DataFrame(
        {"numeric": [0.0, 2.0, np.nan], "constant": [5, 5, 5], "category": ["a", "b", None]}
    )
    validation = pd.DataFrame(
        {"numeric": [1_000.0, np.nan], "constant": [5, 5], "category": ["c", None]}
    )
    preprocessor = FoldPreprocessor.fit(train)
    transformed_train = preprocessor.transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert preprocessor.numerical_columns == ("numeric", "constant")
    assert preprocessor.categorical_columns == ("category",)
    assert preprocessor.numerical_medians["numeric"] == 1.0
    assert preprocessor.category_maps["category"] == {
        "<MISSING>": 0,
        "<VALUE>a": 1,
        "<VALUE>b": 2,
    }
    assert np.allclose(transformed_train.mean(axis=0), 0.0)
    assert np.allclose(transformed_train.std(axis=0)[:1], 1.0)
    assert np.allclose(transformed_train[:, 1], 0.0)
    assert transformed_validation[0, 0] > 100.0
    assert transformed_validation[0, 2] < transformed_train[:, 2].min()


def test_preprocessor_rejects_reordered_columns():
    preprocessor = FoldPreprocessor.fit(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    with pytest.raises(ValueError, match="columns"):
        preprocessor.transform(pd.DataFrame({"b": ["x"], "a": [1]}))


def test_classification_episode_is_disjoint_and_preserves_classes():
    labels = np.repeat(np.array([0, 1, 2]), 6)
    context, query = sample_episode_indices(
        labels,
        problem_type="multiclass",
        context_rows=8,
        query_rows=7,
        rng=np.random.default_rng(5),
    )
    assert not np.intersect1d(context, query).size
    assert set(labels[context]) == {0, 1, 2}
    assert set(labels[query]) == {0, 1, 2}


def test_prediction_context_is_train_only_stratified_subset():
    labels = np.array([0] * 8 + [1] * 4)
    rows = sample_prediction_context(
        labels,
        problem_type="binary",
        max_context_rows=6,
        rng=np.random.default_rng(9),
    )
    assert rows.size == 6
    assert set(labels[rows]) == {0, 1}


def test_identity_guard_requires_full_half_percent_relative_improvement():
    rejected = choose_identity_guard(identity_error=0.20, adapted_error=0.1991)
    selected = choose_identity_guard(identity_error=0.20, adapted_error=0.1990)
    assert not rejected.use_adapted
    assert selected.use_adapted
    assert selected.relative_improvement == pytest.approx(0.005)


def test_deployment_and_benchmark_errors_keep_regression_metrics_distinct():
    binary_probability = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])
    assert deployment_error("binary", np.array([0, 1, 0, 1]), binary_probability) == pytest.approx(0.0)
    multiclass_probability = np.eye(3)[np.array([0, 1, 2])]
    assert deployment_error("multiclass", np.array([0, 1, 2]), multiclass_probability, n_classes=3) == pytest.approx(0.0)
    regression_labels = np.array([1.0, 3.0])
    regression_prediction = np.array([2.0, 1.0])
    assert deployment_error("regression", regression_labels, regression_prediction) == pytest.approx(2.5)
    assert benchmark_error("regression", regression_labels, regression_prediction) == pytest.approx(np.sqrt(2.5))


def test_shared_tuning_configs_are_fixed_and_keep_protocol_controls():
    first = shared_random_direct_spline_configs(10, seed=20260813)
    second = shared_random_direct_spline_configs(10, seed=20260813)
    assert first == second
    assert len(first) == 10
    for config in first:
        assert config["max_context_rows"] == DEFAULT_DIRECT_SPLINE_CONFIG["max_context_rows"]
        assert 100 <= config["adapter_steps"] <= 200
        assert config["random_state"] != DEFAULT_DIRECT_SPLINE_CONFIG["random_state"]


def test_paired_elo_is_symmetric_and_bootstrapped():
    identity = np.array([0.4, 0.3, 0.2, 0.2])
    candidate = np.array([0.3, 0.35, 0.1, 0.2])
    result = paired_elo_delta(identity, candidate)
    inverse = paired_elo_delta(candidate, identity)
    interval = bootstrap_paired_elo(identity, candidate, rounds=20, seed=1)
    assert result["candidate_wins"] == 2
    assert result["identity_wins"] == 1
    assert result["ties"] == 1
    assert result["paired_elo_delta"] == pytest.approx(-inverse["paired_elo_delta"])
    assert interval["lower_95"] <= interval["upper_95"]


def test_greedy_ensemble_uses_validation_to_select_the_better_candidate():
    labels = np.array([0, 1, 0, 1])
    poor = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    strong = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    selected, prediction = greedy_validation_ensemble(
        predictions=[poor, strong], labels=labels, problem_type="binary", n_classes=2, rounds=3
    )
    assert selected[0] == 1
    assert deployment_error("binary", labels, prediction) == pytest.approx(0.0)


def test_one_bag_is_train_only_and_returns_complete_prediction_artifacts():
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Match the frozen TabICL checkpoint: it exposes a wider head than
            # this binary task, so the runner must select its first two logits.
            self.head = torch.nn.Linear(2, 10)

        def clear_cache(self):
            pass

        def forward(self, features, context_labels):
            return self.head(features[:, context_labels.shape[1] :])

    rows = 32
    features = pd.DataFrame(
        {"x0": np.linspace(-1.0, 1.0, rows), "x1": np.tile([0.0, 1.0], rows // 2)}
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="test",
    )
    config = {
        **DEFAULT_DIRECT_SPLINE_CONFIG,
        "adapter_steps": 2,
        "adapter_patience": 2,
        "validation_interval": 1,
        "max_context_rows": 8,
        "train_context_rows": 4,
        "query_batch_rows": 4,
        "evaluation_query_chunk_rows": 8,
        "cross_column_mixing_rank": 0,
    }
    result = _fit_one_bag(
        task=task,
        fit_indices=np.arange(16),
        validation_indices=np.arange(16, 24),
        bag=0,
        config=config,
        protocol_seed=0,
        backbone=TinyBackbone(),
        device=torch.device("cpu"),
    )
    assert np.array_equal(result.validation_indices, np.arange(16, 24))
    assert result.identity_validation.shape == (8, 2)
    assert result.adapted_validation.shape == (8, 2)
    assert result.guarded_validation.shape == (8, 2)
    assert result.guarded_test.shape == (8, 2)
    assert np.allclose(result.guarded_validation.sum(axis=1), 1.0)
    assert result.metadata["fit_rows"] == 16
    assert result.metadata["validation_rows"] == 8


def test_summary_keeps_standard_tabarena_baseline_out_of_internal_paired_elo(tmp_path):
    task_summary = {
        "task_id": 1,
        "dataset_id": 2,
        "dataset_name": "tiny",
        "problem_type": "binary",
        "outer_split_hash": "split",
        "identity": {"benchmark_error": 0.30, "deployment_error": 0.30},
        "default": {"benchmark_error": 0.25, "deployment_error": 0.25},
        "tuned": {"benchmark_error": 0.24, "deployment_error": 0.24},
        "tuned_ensemble": {"benchmark_error": 0.23, "deployment_error": 0.23},
        "tuned_config_label": "D",
        "tuned_validation_deployment_error": 0.22,
        "default_guard_selected_adapted_fraction": 1.0,
        "tuned_ensemble_selected_config_labels": ["D"],
        "standard_tabarena": {"benchmark_error": 0.20, "deployment_error": 0.20},
    }
    summary = summarize_experiment(
        task_summaries=[task_summary], output_dir=tmp_path, bootstrap_rounds=5, bootstrap_seed=0
    )
    row = (tmp_path / "task_results.csv").read_text(encoding="utf-8").splitlines()[1]
    assert summary["standard_tabarena"]["mean_benchmark_error"] == pytest.approx(0.20)
    assert "standard_tabarena_benchmark_error" in (tmp_path / "task_results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "0.2" in row
    assert set(summary["paired_results"]) == {"default", "tuned", "tuned_ensemble"}
