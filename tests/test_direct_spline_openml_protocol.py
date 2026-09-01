import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl._experiments.direct_spline_openml import (
    OpenMLTaskData,
    _fit_one_bag,
    _bag_splits,
    effective_inner_bag_count,
    greedy_validation_ensemble,
    run_standard_tabarena_baseline,
    summarize_experiment,
    summarize_task_tuning,
    tabarena_v0pt1_task_ids,
)
import tabicl._experiments.direct_spline_openml as direct_spline_openml
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
from tabicl._experiments.direct_spline_random_forest_anchor import (
    DEFAULT_BT_CONFIG,
    METHOD_RANDOM_FOREST,
    bradley_terry_fit_config,
    fit_bradley_terry_elo,
    fit_random_forest_task,
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


def test_multiclass_deployment_error_normalizes_softmax_rounding_without_warning():
    labels = np.array([0, 1, 2])
    # These are representative float32 softmax rows after conversion to
    # float64: valid probabilities, but not exact unit sums.
    prediction = np.array(
        [[0.70000005, 0.20000002, 0.09999999], [0.1, 0.80000007, 0.09999998], [0.2, 0.3, 0.49999994]]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        error = deployment_error("multiclass", labels, prediction, n_classes=3)
    assert np.isfinite(error)


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


def test_tabarena_suite_lookup_retries_a_transient_openml_failure(monkeypatch):
    calls = 0

    def get_suite(_suite_id):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("temporary gateway failure")
        return types.SimpleNamespace(tasks=list(range(51)))

    fake_openml = types.SimpleNamespace(study=types.SimpleNamespace(get_suite=get_suite))
    monkeypatch.setitem(sys.modules, "openml", fake_openml)
    delays: list[float] = []
    monkeypatch.setattr(direct_spline_openml.time, "sleep", delays.append)

    assert tabarena_v0pt1_task_ids(attempts=3, initial_retry_seconds=0.25) == list(range(51))
    assert calls == 3
    assert delays == [0.25, 0.5]


def test_standard_baseline_uses_existing_preprocessing_and_does_not_mutate_test_data(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    observed: dict[str, object] = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            observed["kwargs"] = kwargs
            self.model_path_ = checkpoint

        def fit(self, features, labels):
            observed["fit_features"] = features
            observed["fit_labels"] = labels
            return self

        def predict_proba(self, features):
            observed["prediction_input"] = features
            features.iloc[:, 0] = 0.0
            return np.tile(np.array([[0.75, 0.25]]), (len(features), 1))

    monkeypatch.setattr(sys.modules["tabicl"], "TabICLClassifier", FakeClassifier)
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"all_nan_at_test": [0.0, 1.0], "x": [1.0, 2.0]}),
        y_train=np.array([0, 1]),
        x_test=pd.DataFrame({"all_nan_at_test": [np.nan, np.nan], "x": [3.0, 4.0]}),
        y_test=np.array([0, 1]),
        outer_split_hash="split",
    )

    metadata = run_standard_tabarena_baseline(
        task=task,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        classifier_checkpoint=None,
        regressor_checkpoint=None,
        resume=False,
        run_fingerprint_hash="fingerprint",
    )

    assert observed["kwargs"]["numerical_preprocessing"] == "existing"
    assert observed["prediction_input"] is not task.x_test
    assert task.x_test["all_nan_at_test"].isna().all()
    assert "test" not in metadata
    assert metadata["test_metrics_deferred_to_task_summary"] is True


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
    assert result["rating_kind"] == "paired_head_to_head_elo_equivalent"
    assert result["paired_head_to_head_elo_equivalent"] == result["paired_elo_delta"]
    assert result["paired_elo_delta"] == pytest.approx(-inverse["paired_elo_delta"])
    assert interval["lower_95"] <= interval["upper_95"]


def test_local_bradley_terry_anchor_is_fixed_and_orders_clear_methods():
    board = fit_bradley_terry_elo(
        errors_by_method={
            METHOD_RANDOM_FOREST: [0.40, 0.40, 0.40, 0.40],
            "TabICLv2_D": [0.30, 0.30, 0.30, 0.30],
            "DirectSpline_D": [0.20, 0.20, 0.20, 0.20],
        }
    )
    ratings = board["ratings"]
    assert ratings[METHOD_RANDOM_FOREST]["elo"] == pytest.approx(1000.0)
    assert ratings["DirectSpline_D"]["elo"] > ratings["TabICLv2_D"]["elo"] > 1000.0
    assert board["rating_kind"] == "local_anchored_bradley_terry_elo"


def test_bradley_terry_fit_config_keeps_manifest_metadata_out_of_fit_kwargs():
    fit_config = bradley_terry_fit_config(DEFAULT_BT_CONFIG)

    assert "model" not in fit_config
    assert fit_config["anchor_method"] == METHOD_RANDOM_FOREST


def test_random_forest_anchor_uses_only_outer_training_preprocessing():
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny_rf",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"numeric": [0.0, 0.1, 0.9, 1.0], "category": ["a", "a", "b", "b"]}),
        y_train=np.array([0, 0, 1, 1]),
        x_test=pd.DataFrame({"numeric": [0.05, 0.95], "category": ["missing-at-train", "b"]}),
        y_test=np.array([0, 1]),
        outer_split_hash="split",
    )
    result = fit_random_forest_task(task=task, config={"n_estimators": 10}, n_jobs=1)
    assert result.prediction.shape == (2, 2)
    assert np.allclose(result.prediction.sum(axis=1), 1.0)
    assert np.isfinite(result.metadata["benchmark_error"])
    assert result.metadata["preprocessor"] == "FoldPreprocessor.fit(outer_training_rows_only)"


def test_invalid_candidate_prediction_is_retained_as_a_paired_loss():
    result = direct_spline_openml._paired_comparison_summary(
        reference=np.asarray([0.2, 0.3]),
        candidate=np.asarray([np.inf, 0.1]),
        problem_types=np.asarray(["binary", "binary"], dtype=object),
        bootstrap_rounds=20,
        bootstrap_seed=0,
        reference_label="identity",
        candidate_label="adapted",
    )

    assert result["n_tasks"] == 2
    assert result["n_invalid_candidate_predictions"] == 1
    assert result["candidate_wins"] == 1
    assert result["identity_wins"] == 1


def test_greedy_ensemble_uses_validation_to_select_the_better_candidate():
    labels = np.array([0, 1, 0, 1])
    poor = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    strong = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    selected, prediction = greedy_validation_ensemble(
        predictions=[poor, strong], labels=labels, problem_type="binary", n_classes=2, rounds=3
    )
    assert selected[0] == 1
    assert deployment_error("binary", labels, prediction) == pytest.approx(0.0)


def test_rare_class_reduces_stratified_bag_count_without_dropping_the_task():
    labels = np.concatenate((np.repeat(0, 100), np.repeat(1, 100), np.repeat(2, 5)))
    task = OpenMLTaskData(
        task_id=363614,
        dataset_id=2,
        dataset_name="anneal_like",
        problem_type="multiclass",
        n_classes=3,
        x_train=pd.DataFrame({"x": np.arange(labels.size)}),
        y_train=labels,
        x_test=pd.DataFrame({"x": [0, 1]}),
        y_test=np.array([0, 1]),
        outer_split_hash="split",
    )
    assert effective_inner_bag_count(task, requested_bags=8) == 5
    splits = list(_bag_splits(task, requested_bags=8, seed=0))
    assert len(splits) == 5
    for fit_indices, validation_indices in splits:
        assert set(labels[validation_indices]) == {0, 1, 2}
        assert np.bincount(labels[fit_indices], minlength=3).min() >= 2


def test_direct_spline_rejects_class_with_too_few_rows_for_train_episodes():
    labels = np.array([0, 0, 1, 1])
    task = OpenMLTaskData(
        task_id=99,
        dataset_id=2,
        dataset_name="too_rare",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": np.arange(labels.size)}),
        y_train=labels,
        x_test=pd.DataFrame({"x": [0, 1]}),
        y_test=np.array([0, 1]),
        outer_split_hash="split",
    )
    with pytest.raises(ValueError, match="at least three rows"):
        effective_inner_bag_count(task, requested_bags=8)


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
    assert result.metadata["backbone_activation_checkpointing"] is True


def test_single_default_task_summary_marks_tuning_and_ensemble_as_aliases(tmp_path):
    labels = np.array([0, 1, 0, 1])
    identity = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7]])
    adapted = np.array([[0.8, 0.2], [0.3, 0.7], [0.7, 0.3], [0.2, 0.8]])
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": np.arange(4)}),
        y_train=labels,
        x_test=pd.DataFrame({"x": np.arange(4, 8)}),
        y_test=labels,
        outer_split_hash="split",
    )
    identity_metrics = {
        "benchmark_error": benchmark_error("binary", labels, identity),
        "deployment_error": deployment_error("binary", labels, identity),
    }
    adapted_metrics = {
        "benchmark_error": benchmark_error("binary", labels, adapted),
        "deployment_error": deployment_error("binary", labels, adapted),
    }
    config_dir = tmp_path / "raw" / "task_1_tiny" / "config_D"
    config_dir.mkdir(parents=True)
    direct_spline_openml._json_dump(
        config_dir / "config_summary.json",
        {
            "pipeline": "standard",
            "identity_definition": "matched",
            "validation": {
                "identity": identity_metrics,
                "adapted": adapted_metrics,
                "guarded": adapted_metrics,
            },
            "test_metrics_deferred_to_task_summary": True,
            "guard_selected_adapted_fraction": 1.0,
            "requested_bags": 8,
            "effective_bags": 8,
        },
    )
    np.savez_compressed(
        config_dir / "config_predictions.npz",
        identity_validation=identity,
        adapted_validation=adapted,
        guarded_validation=adapted,
        identity_test=identity,
        adapted_test=adapted,
        guarded_test=adapted,
    )

    result = summarize_task_tuning(
        task=task,
        config_labels=["D"],
        output_dir=tmp_path,
        ensemble_rounds=10,
    )

    assert result["default"] == result["guarded_default"]
    assert result["tuned"] == result["guarded_default"]
    assert result["tuned_ensemble"] == result["guarded_default"]
    assert result["raw_adapted_tuned"] == result["raw_adapted_default"]
    assert result["tuned_ensemble_selected_config_labels"] == ["D"]
    assert result["distinct_reported_arms"] == ["raw_adapted_default", "guarded_default"]
    assert result["reported_arm_aliases"]["guarded_tuned_ensemble"] == "guarded_default"


def test_predeclared_loss_arms_require_and_record_matched_identity_predictions(tmp_path):
    labels = np.array([0, 1, 0, 1])
    identity = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7]])
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": np.arange(4)}),
        y_train=labels,
        x_test=pd.DataFrame({"x": np.arange(4, 8)}),
        y_test=labels,
        outer_split_hash="split",
    )
    identity_metrics = {
        "benchmark_error": benchmark_error("binary", labels, identity),
        "deployment_error": deployment_error("binary", labels, identity),
    }
    for label, objective in (
        ("D_CE", "cross_entropy"),
        ("D_pairwise_auc", "pairwise_auc"),
        ("D_CE_plus_pairwise_auc", "cross_entropy_plus_pairwise_auc"),
    ):
        config_dir = tmp_path / "raw" / "task_1_tiny" / f"config_{label}"
        config_dir.mkdir(parents=True)
        direct_spline_openml._json_dump(
            config_dir / "config_summary.json",
            {
                "pipeline": "standard",
                "identity_definition": "matched",
                "config": {"classification_objective": objective},
                "validation": {
                    "identity": identity_metrics,
                    "adapted": identity_metrics,
                    "guarded": identity_metrics,
                },
                "test_metrics_deferred_to_task_summary": True,
                "guard_selected_adapted_fraction": 0.0,
                "requested_bags": 8,
                "effective_bags": 8,
            },
        )
        np.savez_compressed(
            config_dir / "config_predictions.npz",
            identity_validation=identity,
            adapted_validation=identity,
            guarded_validation=identity,
            identity_test=identity,
            adapted_test=identity,
            guarded_test=identity,
        )

    result = summarize_task_tuning(
        task=task,
        config_labels=["D_CE", "D_pairwise_auc", "D_CE_plus_pairwise_auc"],
        output_dir=tmp_path,
        ensemble_rounds=3,
        report_predeclared_config_arms=True,
    )

    matched_identity = result["predeclared_config_arms"]["D_pairwise_auc"]["matched_identity"]
    assert matched_identity["benchmark_error"] == result["identity"]["benchmark_error"]
    assert matched_identity["deployment_error"] == result["identity"]["deployment_error"]
    assert matched_identity["prediction_valid"] is True
    assert result["predeclared_config_identity_consistency"]["max_abs_test_difference_by_config"] == {
        "D_CE": 0.0,
        "D_pairwise_auc": 0.0,
        "D_CE_plus_pairwise_auc": 0.0,
    }

    mismatch_dir = tmp_path / "raw" / "task_1_tiny" / "config_D_pairwise_auc"
    np.savez_compressed(
        mismatch_dir / "config_predictions.npz",
        identity_validation=identity,
        adapted_validation=identity,
        guarded_validation=identity,
        identity_test=np.flip(identity, axis=1),
        adapted_test=identity,
        guarded_test=identity,
    )
    with pytest.raises(RuntimeError, match="identity differed"):
        summarize_task_tuning(
            task=task,
            config_labels=["D_CE", "D_pairwise_auc", "D_CE_plus_pairwise_auc"],
            output_dir=tmp_path,
            ensemble_rounds=3,
            report_predeclared_config_arms=True,
        )


def test_summary_separates_raw_guarded_and_end_to_end_standard_comparisons(tmp_path):
    task_summary = {
        "report_schema_version": 2,
        "task_id": 1,
        "dataset_id": 2,
        "dataset_name": "tiny",
        "problem_type": "binary",
        "outer_split_hash": "split",
        "identity": {"benchmark_error": 0.30, "deployment_error": 0.30},
        "raw_adapted_default": {"benchmark_error": 0.32, "deployment_error": 0.32},
        "raw_adapted_tuned": {"benchmark_error": 0.21, "deployment_error": 0.21},
        "raw_adapted_tuned_config_label": "T1",
        "raw_adapted_tuned_validation_deployment_error": 0.20,
        "guarded_default": {"benchmark_error": 0.25, "deployment_error": 0.25},
        "guarded_tuned": {"benchmark_error": 0.24, "deployment_error": 0.24},
        "guarded_tuned_config_label": "T2",
        "guarded_tuned_validation_deployment_error": 0.22,
        "guarded_tuned_ensemble": {"benchmark_error": 0.23, "deployment_error": 0.23},
        "guarded_tuned_ensemble_validation": {"benchmark_error": 0.22, "deployment_error": 0.22},
        "guarded_tuned_ensemble_selected_config_labels": ["D", "T2"],
        "reported_arm_aliases": {
            "default": "guarded_default",
            "tuned": "guarded_tuned",
            "tuned_ensemble": "guarded_tuned_ensemble",
        },
        "distinct_reported_arms": [
            "raw_adapted_default",
            "raw_adapted_tuned",
            "guarded_default",
            "guarded_tuned",
            "guarded_tuned_ensemble",
        ],
        "default": {"benchmark_error": 0.25, "deployment_error": 0.25},
        "tuned": {"benchmark_error": 0.24, "deployment_error": 0.24},
        "tuned_ensemble": {"benchmark_error": 0.23, "deployment_error": 0.23},
        "tuned_config_label": "D",
        "tuned_validation_deployment_error": 0.22,
        "default_guard_selected_adapted_fraction": 1.0,
        "direct_spline_requested_bags": 8,
        "direct_spline_effective_bags": 8,
        "tuned_ensemble_selected_config_labels": ["D"],
        "standard_tabarena": {"benchmark_error": 0.20, "deployment_error": 0.20},
        "predeclared_config_arms": {
            "D_CE": {
                "training_objective": "cross_entropy",
                "matched_identity": {"benchmark_error": 0.30, "deployment_error": 0.30},
                "raw_adapted": {"benchmark_error": 0.32, "deployment_error": 0.32},
                "guarded": {"benchmark_error": 0.25, "deployment_error": 0.25},
            },
            "D_pairwise_auc": {
                "training_objective": "pairwise_auc",
                "matched_identity": {"benchmark_error": 0.20, "deployment_error": 0.20},
                "raw_adapted": {"benchmark_error": 0.25, "deployment_error": 0.25},
                "guarded": {"benchmark_error": 0.24, "deployment_error": 0.24},
            },
            "D_CE_plus_pairwise_auc": {
                "training_objective": "cross_entropy_plus_pairwise_auc",
                "matched_identity": {"benchmark_error": 0.23, "deployment_error": 0.23},
                "raw_adapted": {"benchmark_error": 0.22, "deployment_error": 0.22},
                "guarded": {"benchmark_error": 0.22, "deployment_error": 0.22},
            },
        },
    }
    summary = summarize_experiment(
        task_summaries=[task_summary],
        output_dir=tmp_path,
        bootstrap_rounds=5,
        bootstrap_seed=0,
        skipped_tasks=[
            {
                "task_id": 99,
                "dataset_name": "wide",
                "n_features": 1776,
                "max_features": 100,
                "reason": "n_features_exceeds_max_features",
            }
        ],
        task_eligibility={"max_features": 100},
    )
    row = (tmp_path / "task_results.csv").read_text(encoding="utf-8").splitlines()[1]
    assert summary["standard_tabarena"]["n_tasks_available"] == 1
    assert "mean_benchmark_error" not in summary["standard_tabarena"]
    assert "standard_tabarena_benchmark_error" in (tmp_path / "task_results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "0.2" in row
    assert summary["distinct_paired_result_keys"] == task_summary["distinct_reported_arms"]
    assert summary["paired_results"]["default"]["alias_of"] == "guarded_default"
    assert summary["paired_results"]["raw_adapted_default"]["candidate_wins"] == 0
    assert summary["paired_results"]["raw_adapted_tuned"]["candidate_wins"] == 1
    assert "mean_benchmark_error" not in summary["paired_results"]["guarded_default"]
    assert summary["paired_results"]["guarded_default"]["median_relative_error_change"] == pytest.approx(
        (0.25 - 0.30) / 0.30
    )
    assert summary["paired_results"]["guarded_default"]["by_problem_type"]["binary"]["n_tasks"] == 1
    pairwise_raw = summary["predeclared_config_objective_results"]["D_pairwise_auc"][
        "raw_adapted_vs_matched_inner_bag_identity"
    ]
    assert pairwise_raw["candidate_wins"] == 0
    assert pairwise_raw["identity_wins"] == 1
    end_to_end = summary["end_to_end_vs_standard_tabarena"]
    assert end_to_end["results"]["guarded_default"]["identity_wins"] == 1
    assert "not an isolated spline effect" in end_to_end["note"]
    assert summary["n_skipped_tasks"] == 1
    assert summary["skipped_tasks"][0]["dataset_name"] == "wide"
    assert summary["task_eligibility"] == {"max_features": 100}
