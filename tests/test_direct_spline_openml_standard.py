from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import tabicl._experiments.direct_spline_openml_standard as standard_openml
from tabicl._experiments.direct_spline_openml import BagPredictions, OpenMLTaskData
from tabicl._experiments.direct_spline_openml_standard import (
    _apply_adapter,
    _aggregate_public_classification_members,
    _enable_frozen_training_path,
    _fit_standard_bag,
    _forward_method,
    _fit_one_bag_standard,
    _identity_prediction,
    _identity_view_parity,
    _make_adapters,
    _many_class_training_logits,
    _normal_prediction,
    _StandardBag,
    run_task_config_standard,
    standard_direct_spline_config,
)
from tabicl._hyperspline import DirectSplineTransform
from tabicl._model.inference_config import InferenceConfig
from tabicl._model.tabicl import TabICL


class _TinyStandardBackbone(torch.nn.Module):
    """Minimal differentiable stand-in for the public classifier checkpoint."""

    def __init__(self):
        super().__init__()
        self.max_classes = 10
        self.head = torch.nn.Linear(2, 10)
        self.forward_training_flags = []

    def clear_cache(self):
        pass

    def forward(
        self,
        X,
        y_train,
        feature_shuffles=None,
        return_logits=True,
        softmax_temperature=0.9,
        inference_config=None,
    ):
        del feature_shuffles, return_logits, softmax_temperature, inference_config
        self.forward_training_flags.append(self.training)
        # The real inference head exposes exactly the task's label width,
        # rather than the checkpoint's maximum class count.
        return self.head(X[:, y_train.shape[1] :])[..., :2]


class _HalfPrecisionEvalBackbone(_TinyStandardBackbone):
    """Mimic the reduced-precision output produced by AMP inference."""

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return output if self.training else output.half()


class _OffsetPublicRegressionBackbone(torch.nn.Module):
    """Regression stand-in whose public and reconstructed forwards disagree."""

    def __init__(self):
        super().__init__()
        self.max_classes = 0

    def clear_cache(self):
        pass

    def forward(
        self,
        X,
        y_train,
        feature_shuffles=None,
        return_logits=True,
        softmax_temperature=0.9,
        inference_config=None,
    ):
        del feature_shuffles, return_logits, softmax_temperature, inference_config
        prediction = X[:, y_train.shape[1] :].mean(dim=-1, keepdim=True)
        return prediction.expand(-1, -1, 3)

    def quantile_dist(self, raw):
        class _Distribution:
            quantiles = raw

        return _Distribution()

    def predict_stats(self, X, y_train, output_type="mean", alphas=None, inference_config=None):
        del alphas, inference_config
        assert output_type == ["mean"]
        # Deliberately model a public/private-path bug for the failure report.
        return {"mean": self.forward(X, y_train).mean(dim=-1) + 0.25}


class _TransientPublicRegressionBackbone(_OffsetPublicRegressionBackbone):
    """Model public repeat drift that disappears in an exact branch replay."""

    def __init__(self):
        super().__init__()
        self.public_calls = 0

    def predict_stats(self, X, y_train, output_type="mean", alphas=None, inference_config=None):
        del alphas, inference_config
        assert output_type == ["mean"]
        self.public_calls += 1
        drift = 0.25 if self.public_calls <= 2 else 0.0
        return {"mean": self.forward(X, y_train).mean(dim=-1) + drift}


def test_standard_config_uses_all_fit_rows_unless_explicitly_capped():
    assert standard_direct_spline_config()["max_context_rows"] is None
    assert standard_direct_spline_config()["train_context_rows"] is None
    assert standard_direct_spline_config(train_context_rows=0)["train_context_rows"] is None
    assert standard_direct_spline_config(train_context_rows=128)["train_context_rows"] == 128
    assert standard_direct_spline_config(context_cap=128)["max_context_rows"] == 128
    assert standard_direct_spline_config(context_cap=128)["train_context_rows"] == 128


def test_standard_regression_reuses_public_estimator_scaled_labels_exactly():
    rows = 16
    features = pd.DataFrame(
        {
            "x0": np.linspace(-2.0, 2.0, rows),
            "x1": np.linspace(10.0, 20.0, rows),
        }
    )
    # Keep the caller-side representation float64, as it is for OpenML
    # regression tasks, while the public regressor casts to float32 in fit().
    targets = np.linspace(326.123456789, 18_823.987654321, rows, dtype=np.float64)
    task = OpenMLTaskData(
        task_id=5,
        dataset_id=6,
        dataset_name="regression_label_parity",
        problem_type="regression",
        n_classes=None,
        x_train=features,
        y_train=targets,
        x_test=features.iloc[:4].reset_index(drop=True),
        y_test=targets[:4],
        outer_split_hash="test",
    )

    bundle = _fit_standard_bag(
        task=task,
        fit_indices=np.arange(rows),
        config=standard_direct_spline_config(),
        protocol_seed=0,
        bag=0,
        backbone=_OffsetPublicRegressionBackbone(),
        device=torch.device("cpu"),
    )

    public_labels = np.asarray(bundle.estimator.ensemble_generator_.y_, dtype=np.float32)
    caller_recomputed = (
        bundle.estimator.y_scaler_.transform(targets.reshape(-1, 1)).ravel().astype(np.float32)
    )
    assert np.array_equal(bundle.fit_labels, public_labels)
    assert not np.array_equal(caller_recomputed, public_labels)


def test_standard_adapter_identity_changes_only_numeric_positions():
    canonical = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    support = canonical[:, [0, 2]].unsqueeze(0)
    adapter = DirectSplineTransform(support, n_control_points=8, trainable_location_scale=True)
    with torch.no_grad():
        adapter.location.zero_()
        adapter.scale.fill_(1.0)
    transformed = _apply_adapter(
        canonical,
        numerical_indices=np.array([0, 2]),
        adapter=adapter,
    )
    assert torch.allclose(transformed, canonical, atol=2e-5, rtol=2e-5)
    assert torch.equal(transformed[..., 1], canonical[..., 1])


def test_all_nan_feature_mask_removes_cross_mixing_input_contributions():
    first = torch.tensor([[1.0, -100.0], [2.0, 200.0]])
    second = torch.tensor([[1.0, 500.0], [2.0, -700.0]])
    adapter = DirectSplineTransform(
        torch.zeros((1, 1, 2)),
        n_control_points=8,
        trainable_location_scale=True,
        cross_column_mixing_rank=2,
        cross_column_mixing_bound=0.5,
    )
    with torch.no_grad():
        adapter.location.zero_()
        adapter.scale.fill_(1.0)
        adapter.mixing_gate.fill_(1.0)

    unmasked_first = _apply_adapter(
        first, numerical_indices=np.array([0, 1]), adapter=adapter
    )
    unmasked_second = _apply_adapter(
        second, numerical_indices=np.array([0, 1]), adapter=adapter
    )
    masked_first = _apply_adapter(
        first,
        numerical_indices=np.array([0, 1]),
        adapter=adapter,
        filtered_feature_mask=np.array([False, True]),
    )
    masked_second = _apply_adapter(
        second,
        numerical_indices=np.array([0, 1]),
        adapter=adapter,
        filtered_feature_mask=np.array([False, True]),
    )

    assert not torch.allclose(unmasked_first[:, 0], unmasked_second[:, 0])
    assert torch.equal(masked_first[:, 0], masked_second[:, 0])


def test_standard_direct_spline_is_bit_exact_identity_before_first_update():
    canonical = torch.tensor(
        [
            [-3.1251, -0.5001, 0.3334],
            [-1.0002, 0.4999, 1.7501],
            [0.1251, 2.0002, 4.1251],
        ],
        dtype=torch.float32,
    )
    support = canonical.unsqueeze(0)
    adapter = DirectSplineTransform(
        support,
        n_control_points=20,
        trainable_location_scale=True,
        cross_column_mixing_rank=3,
        cross_column_mixing_bound=0.1,
    )
    with torch.no_grad():
        adapter.location.zero_()
        adapter.scale.fill_(1.0)

    transformed = adapter.transform(support)

    # Exact equality matters here: an allclose-sized perturbation before an
    # AMP backbone can cross a float16 rounding boundary.
    assert torch.equal(transformed, support)
    transformed.sum().backward()
    assert adapter.gap_logits.grad is not None


def test_standard_runner_checks_public_identity_and_preserves_prediction_shapes():
    rows = 32
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny_standard",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="test",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "adapter_steps": 2,
        "adapter_patience": 2,
        "validation_interval": 1,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
    }
    backbone = _HalfPrecisionEvalBackbone()
    result = _fit_one_bag_standard(
        task=task,
        fit_indices=np.arange(16),
        validation_indices=np.arange(16, 24),
        bag=0,
        config=config,
        protocol_seed=0,
        backbone=backbone,
        device=torch.device("cpu"),
        run_fingerprint_hash="test",
        progress=None,
        requested_bags=8,
        effective_bags=8,
    )
    assert result.identity_validation.shape == (8, 2)
    assert result.adapted_validation.shape == (8, 2)
    assert result.guarded_test.shape == (8, 2)
    assert np.allclose(result.identity_validation.sum(axis=1), 1.0)
    assert result.metadata["pipeline"] == "standard_ensemble"
    assert result.metadata["support_rows"] == 16
    assert result.metadata["identity_parity_reference"] == "public_exact_input_views_full_query"
    assert result.metadata["identity_parity_max_abs_validation"] == 0.0
    assert result.metadata["public_path_input_parity_passed"]
    assert result.metadata["fresh_spline_view_identity_passed"]
    assert result.metadata["adapter_has_valid_learned_checkpoint"]
    assert result.metadata["adapter_best_step"] > 0
    # Evaluation/parity runs use the public inference mode; adapter updates
    # switch only the frozen backbone execution path back to autograd mode.
    assert any(backbone.forward_training_flags)
    assert any(not flag for flag in backbone.forward_training_flags)


def test_categorical_only_task_is_recorded_as_a_public_identity_tie():
    rows = 32
    features = pd.DataFrame(
        {
            "role": np.tile(["analyst", "manager"], rows // 2),
            "resource": np.tile(["a", "b", "c", "d"], rows // 4),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=2,
        dataset_id=3,
        dataset_name="categorical_only",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="test",
    )
    result = _fit_one_bag_standard(
        task=task,
        fit_indices=np.arange(16),
        validation_indices=np.arange(16, 24),
        bag=0,
        config=standard_direct_spline_config(train_context_rows=4),
        protocol_seed=0,
        backbone=_TinyStandardBackbone(),
        device=torch.device("cpu"),
        run_fingerprint_hash="test",
        progress=None,
        requested_bags=8,
        effective_bags=8,
    )
    assert result.metadata["no_trainable_numerical_features"]
    assert result.metadata["identity_parity_reference"] == "public_exact_input_views_full_query"
    assert result.metadata["adapter_steps_executed"] == 0
    assert np.array_equal(result.identity_validation, result.adapted_validation)
    assert np.array_equal(result.identity_test, result.guarded_test)


def test_capped_context_records_unchecked_public_parity_not_failure():
    task = OpenMLTaskData(
        task_id=89,
        dataset_id=90,
        dataset_name="capped_context",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame(
            {
                "x0": np.linspace(-1.0, 1.0, 16),
                "x1": np.linspace(1.0, 3.0, 16),
            }
        ),
        y_train=np.tile([0, 1], 8),
        x_test=pd.DataFrame(
            {
                "x0": np.linspace(1.1, 2.0, 8),
                "x1": np.linspace(3.1, 4.0, 8),
            }
        ),
        y_test=np.tile([0, 1], 4),
        outer_split_hash="test",
    )
    result = _fit_one_bag_standard(
        task=task,
        fit_indices=np.arange(12),
        validation_indices=np.arange(12, 16),
        bag=0,
        config={
            **standard_direct_spline_config(context_cap=6, train_context_rows=6),
            "adapter_steps": 1,
            "adapter_patience": 1,
            "validation_interval": 1,
            "query_batch_rows": 2,
            "cross_column_mixing_rank": 0,
        },
        protocol_seed=0,
        backbone=_TinyStandardBackbone(),
        device=torch.device("cpu"),
        run_fingerprint_hash="test",
        progress=None,
        requested_bags=8,
        effective_bags=8,
    )

    assert not result.metadata["public_path_input_parity_checked_validation"]
    assert not result.metadata["public_path_input_parity_checked_test"]
    assert result.metadata["public_path_input_parity_passed"] is None
    assert result.metadata["identity_parity_reference"] == "matched_capped_exact_input_views_full_query"


def test_standard_runner_uses_retouches_per_bag_guard_for_test_ensemble(tmp_path, monkeypatch):
    task = OpenMLTaskData(
        task_id=91,
        dataset_id=92,
        dataset_name="per_bag_guard",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        y_train=np.asarray([0, 1, 0, 1]),
        x_test=pd.DataFrame({"x": [4.0, 5.0]}),
        y_test=np.asarray([0, 1]),
        outer_split_hash="test",
    )

    identity_test = np.asarray([[0.9, 0.1], [0.8, 0.2]])
    adapted_test = np.asarray([[0.4, 0.6], [0.3, 0.7]])

    def fake_splits(*_args, **_kwargs):
        yield np.asarray([2, 3]), np.asarray([0, 1])
        yield np.asarray([0, 1]), np.asarray([2, 3])

    def fake_load_backbone(**_kwargs):
        return object(), None, {"test": "checkpoint"}

    def fake_fit_one_bag(*, bag, validation_indices, **_kwargs):
        guarded_test = identity_test if bag == 0 else adapted_test
        identity_validation = np.asarray([[0.9, 0.1], [0.2, 0.8]])
        adapted_validation = np.asarray([[0.4, 0.6], [0.3, 0.7]])
        guarded_validation = identity_validation if bag == 0 else adapted_validation
        return BagPredictions(
            validation_indices=validation_indices,
            identity_validation=identity_validation,
            adapted_validation=adapted_validation,
            guarded_validation=guarded_validation,
            identity_test=identity_test,
            adapted_test=adapted_test,
            guarded_test=guarded_test,
            metadata={
                "run_fingerprint_hash": "test",
                "train_seconds": 0.0,
                "peak_allocated_gib": 0.0,
                "identity_parity_max_abs_validation": 0.0,
                "identity_parity_max_abs_test": 0.0,
                "guard_selected_adapted": bool(bag == 1),
                "adapter_has_valid_learned_checkpoint": True,
            },
        )

    monkeypatch.setattr(standard_openml, "_bag_splits", fake_splits)
    monkeypatch.setattr(
        standard_openml, "effective_inner_bag_count", lambda *_args, **_kwargs: 2
    )
    monkeypatch.setattr(standard_openml, "load_frozen_backbone", fake_load_backbone)
    monkeypatch.setattr(standard_openml, "_fit_one_bag_standard", fake_fit_one_bag)

    summary = run_task_config_standard(
        task=task,
        label="D",
        config=standard_direct_spline_config(),
        output_dir=tmp_path,
        bags=2,
        protocol_seed=0,
        device=torch.device("cpu"),
        classifier_checkpoint=None,
        regressor_checkpoint=None,
        resume=False,
        run_fingerprint_hash="test",
    )
    predictions = np.load(
        tmp_path / "raw" / "task_91_per_bag_guard" / "config_D" / "config_predictions.npz"
    )
    expected_guarded_test = (identity_test + adapted_test) / 2.0

    assert summary["guard_protocol"] == "retouche_per_bag_validation_guard_then_test_ensemble"
    assert summary["guard_selected_adapted_fraction"] == pytest.approx(0.5)
    assert np.array_equal(predictions["guarded_test"], expected_guarded_test)
    assert not np.array_equal(predictions["guarded_test"], predictions["adapted_test"])


def test_full_query_parity_does_not_ignore_rows_after_old_probe_boundary(monkeypatch):
    rows = 8_230
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=3,
        dataset_id=4,
        dataset_name="bounded_public_probe",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="test",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "cross_column_mixing_rank": 0,
    }
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=np.arange(16),
        config=config,
        protocol_seed=0,
        bag=0,
        backbone=_HalfPrecisionEvalBackbone(),
        device=torch.device("cpu"),
    )
    adapters = _make_adapters(bundle, config, torch.device("cpu"))
    original = standard_openml._build_public_method_arrays

    def corrupt_last_query_row(**kwargs):
        views, labels_out, shuffles, patterns = original(**kwargs)
        if kwargs["adapters"] is None and views.shape[1] > 8_192:
            views = views.copy()
            views[:, -1, 0] += 1.0
        return views, labels_out, shuffles, patterns

    monkeypatch.setattr(standard_openml, "_build_public_method_arrays", corrupt_last_query_row)
    with pytest.raises(RuntimeError, match="all 8206 test rows"):
        _identity_view_parity(
            bundle=bundle,
            adapters=adapters,
            query_x=task.x_test,
            device=torch.device("cpu"),
            progress=None,
            task_id=task.task_id,
            bag=0,
            split="test",
        )


def test_regression_evaluation_uses_the_public_batch_forward():
    rows = 24
    features = pd.DataFrame(
        {
            "x0": np.linspace(-2.0, 2.0, rows),
            "x1": np.linspace(1.0, 3.0, rows),
        }
    )
    targets = np.linspace(10.0, 30.0, rows)
    task = OpenMLTaskData(
        task_id=4,
        dataset_id=5,
        dataset_name="regression_parity_diagnostic",
        problem_type="regression",
        n_classes=None,
        x_train=features.iloc[:16].reset_index(drop=True),
        y_train=targets[:16],
        x_test=features.iloc[16:].reset_index(drop=True),
        y_test=targets[16:],
        outer_split_hash="test",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "adapter_steps": 1,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
    }

    bundle = _fit_standard_bag(
        task=task,
        fit_indices=np.arange(12),
        config=config,
        protocol_seed=0,
        bag=0,
        backbone=_OffsetPublicRegressionBackbone(),
        device=torch.device("cpu"),
    )
    calls = 0
    original = bundle.estimator._batch_forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    bundle.estimator._batch_forward = counted
    reconstructed = _normal_prediction(
        bundle=bundle,
        query_x=task.x_test,
        context_indices=bundle.support_indices,
        adapters=None,
        device=torch.device("cpu"),
    )
    public = _identity_prediction(bundle, task.x_test)
    assert calls >= len(bundle.estimator.ensemble_generator_.preprocessors_)
    assert np.array_equal(reconstructed, public)


def test_all_nan_query_mask_is_a_full_exact_input_check_without_mutation():
    rows = 24
    features = pd.DataFrame(
        {
            "x0": np.linspace(-2.0, 2.0, rows),
            "x1": np.linspace(1.0, 3.0, rows),
        }
    )
    targets = np.linspace(10.0, 30.0, rows)
    task = OpenMLTaskData(
        task_id=6,
        dataset_id=7,
        dataset_name="regression_all_nan_mask",
        problem_type="regression",
        n_classes=None,
        x_train=features.iloc[:16].reset_index(drop=True),
        y_train=targets[:16],
        x_test=features.iloc[16:].reset_index(drop=True),
        y_test=targets[16:],
        outer_split_hash="test",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "adapter_steps": 1,
        "adapter_patience": 1,
        "validation_interval": 1,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
    }
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=np.arange(12),
        config=config,
        protocol_seed=0,
        bag=0,
        backbone=_TransientPublicRegressionBackbone(),
        device=torch.device("cpu"),
    )
    adapters = _make_adapters(bundle, config, torch.device("cpu"))
    masked_query = task.x_test.copy(deep=True)
    masked_query.loc[:, "x1"] = np.nan
    before = masked_query.copy(deep=True)
    maximum, reference, public_checked = _identity_view_parity(
        bundle=bundle,
        adapters=adapters,
        query_x=masked_query,
        device=torch.device("cpu"),
        progress=None,
        task_id=task.task_id,
        bag=0,
        split="test",
    )
    assert maximum == 0.0
    assert reference == "public_exact_input_views_full_query"
    assert public_checked
    pd.testing.assert_frame_equal(masked_query, before)


def test_many_class_training_route_has_all_public_classes_and_input_gradients():
    backbone = TabICL(
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=2,
        col_feature_group=False,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        col_ssmax=False,
        icl_ssmax=False,
        dropout=0.2,
        zero_init=False,
    )
    estimator = SimpleNamespace(
        model_=backbone,
        inference_config_=InferenceConfig(),
        softmax_temperature=0.9,
    )
    bundle = _StandardBag(
        estimator=estimator,
        fit_labels=np.resize(np.arange(7), 14).astype(np.float32),
        support_indices=np.arange(14),
        numerical_indices=np.asarray([0, 1, 2]),
        problem_type="multiclass",
        n_classes=7,
    )
    canonical = torch.randn(15, 3)
    feature_shuffles = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    views = torch.stack([canonical[:, pattern] for pattern in feature_shuffles]).requires_grad_()
    base_labels = torch.as_tensor(np.resize(np.arange(7), 14), dtype=torch.long)
    class_shuffles = [
        torch.arange(7),
        torch.roll(torch.arange(7), shifts=-1),
        torch.roll(torch.arange(7), shifts=-2),
    ]
    labels = torch.stack([pattern[base_labels] for pattern in class_shuffles]).float()

    backbone.eval()
    public_logits = backbone(
        X=views.detach().clone(),
        y_train=labels,
        feature_shuffles=feature_shuffles,
        return_logits=True,
        softmax_temperature=estimator.softmax_temperature,
        inference_config=estimator.inference_config_,
    )
    logits = _many_class_training_logits(
        bundle=bundle,
        views=views,
        labels=labels,
        feature_shuffles=feature_shuffles,
    )

    assert logits.shape == (3, 1, 7)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.detach(), public_logits, atol=1e-6, rtol=1e-6)
    logits.sum().backward()
    assert views.grad is not None
    assert torch.isfinite(views.grad).all()
    # Only direct-mathematics routing flags are true; every stochastic child
    # remains in eval mode.
    routing_modules = {
        id(backbone),
        id(backbone.col_embedder),
        id(backbone.row_interactor),
        id(backbone.icl_predictor),
    }
    assert all(
        module.training for module in backbone.modules() if id(module) in routing_modules
    )
    assert not any(
        module.training for module in backbone.modules() if id(module) not in routing_modules
    )


def test_many_class_checkpoint_recompute_keeps_each_normalization_branch(monkeypatch):
    class ShiftAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = torch.nn.Parameter(torch.zeros(()))

        def effective_mixing_matrix(self):
            return None

        def transform(self, values):
            return values + self.shift

    first = np.arange(15, dtype=np.float32).reshape(5, 3)
    second = first + 100.0
    identity_pattern = np.asarray([0, 1, 2])
    shifted_pattern = np.asarray([2, 0, 1])
    generator = SimpleNamespace(
        preprocessors_=OrderedDict(
            (
                ("first", SimpleNamespace(X_transformed_=first)),
                ("second", SimpleNamespace(X_transformed_=second)),
            )
        ),
        ensemble_configs_={
            "first": [([0, 1, 2], identity_pattern)],
            "second": [([0, 1, 2], shifted_pattern)],
        },
    )
    bundle = _StandardBag(
        estimator=SimpleNamespace(ensemble_generator_=generator),
        fit_labels=np.asarray([0, 1, 2, 0, 1], dtype=np.float32),
        support_indices=np.arange(5),
        numerical_indices=np.asarray([0, 1, 2]),
        problem_type="multiclass",
        n_classes=3,
    )
    bundle.estimator.model_ = SimpleNamespace(max_classes=2)
    adapters = standard_openml._AdapterSet(
        OrderedDict((method, ShiftAdapter()) for method in generator.preprocessors_)
    )
    recompute_markers: list[int] = []

    def fake_many_class_training_logits(*, bundle, views, labels, feature_shuffles):
        del bundle, feature_shuffles
        recompute_markers.append(int(labels[0, 0].item()))
        query = views[:, labels.shape[1] :, :3]
        return torch.sin(query)

    monkeypatch.setattr(
        standard_openml,
        "_many_class_training_logits",
        fake_many_class_training_logits,
    )
    logits = standard_openml._training_logits(
        bundle=bundle,
        adapters=adapters,
        context_indices=np.asarray([0, 1, 2]),
        query_indices=np.asarray([3, 4]),
        device=torch.device("cpu"),
    )
    logits.sum().backward()

    # Each branch is called once in the initial forward and once in backward
    # recomputation. A late-bound loop closure would use marker 2 twice during
    # backward and silently give the first adapter the second branch's labels.
    assert recompute_markers.count(0) == 2
    assert recompute_markers.count(2) == 2
    assert all(adapter.shift.grad is not None for adapter in adapters.adapters.values())


def test_differentiable_standard_forward_matches_public_eval_without_dropout():
    backbone = TabICL(
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=2,
        col_feature_group=False,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        col_ssmax=False,
        icl_ssmax=False,
        dropout=0.3,
        zero_init=False,
    )
    estimator = SimpleNamespace(
        model_=backbone,
        inference_config_=InferenceConfig(),
        softmax_temperature=0.9,
    )
    bundle = _StandardBag(
        estimator=estimator,
        fit_labels=np.asarray([0, 1, 0, 1], dtype=np.float32),
        support_indices=np.arange(4),
        numerical_indices=np.asarray([0, 1, 2]),
        problem_type="binary",
        n_classes=2,
    )
    canonical = torch.randn(5, 3)
    feature_shuffles = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    views = torch.stack([canonical[:, pattern] for pattern in feature_shuffles]).requires_grad_()
    base_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    labels = torch.stack(
        [base_labels, 1 - base_labels, base_labels], dim=0
    ).float()

    backbone.eval()
    public_logits = backbone(
        X=views.detach().clone(),
        y_train=labels,
        feature_shuffles=feature_shuffles,
        return_logits=True,
        softmax_temperature=estimator.softmax_temperature,
        inference_config=estimator.inference_config_,
    )
    _enable_frozen_training_path(backbone)
    differentiable_logits = _forward_method(
        bundle=bundle,
        views=views,
        labels=labels,
        feature_shuffles=feature_shuffles,
        checkpoint_activations=False,
    )

    assert torch.allclose(
        differentiable_logits.detach()[..., : bundle.n_classes],
        public_logits,
        atol=1e-6,
        rtol=1e-6,
    )
    differentiable_logits.sum().backward()
    assert views.grad is not None
    assert torch.isfinite(views.grad).all()
    # Only the four mathematical-routing flags are true; every child layer,
    # including TabICL's custom attention modules, remains in eval mode.
    routing_modules = {
        id(backbone),
        id(backbone.col_embedder),
        id(backbone.row_interactor),
        id(backbone.icl_predictor),
    }
    assert all(
        module.training for module in backbone.modules() if id(module) in routing_modules
    )
    assert not any(
        module.training for module in backbone.modules() if id(module) not in routing_modules
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP parity requires a GPU")
def test_cuda_amp_three_shuffle_forward_matches_public_inference_and_backpropagates():
    device = torch.device("cuda")
    backbone = TabICL(
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=2,
        col_feature_group=False,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        col_ssmax=False,
        icl_ssmax=False,
        dropout=0.3,
        zero_init=False,
    ).to(device)
    inference_config = InferenceConfig()
    for name in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"):
        getattr(inference_config, name).update(
            {"use_amp": True, "use_fa3": False, "offload": False}
        )
    inference_config = inference_config.with_default_device(device)
    estimator = SimpleNamespace(
        model_=backbone,
        inference_config_=inference_config,
        softmax_temperature=0.9,
    )
    bundle = _StandardBag(
        estimator=estimator,
        fit_labels=np.asarray([0, 1, 0, 1], dtype=np.float32),
        support_indices=np.arange(4),
        numerical_indices=np.asarray([0, 1, 2]),
        problem_type="binary",
        n_classes=2,
    )
    canonical = torch.randn(5, 3, device=device)
    feature_shuffles = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    views = torch.stack(
        [canonical[:, pattern] for pattern in feature_shuffles]
    ).requires_grad_()
    base_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    labels = torch.stack([base_labels, 1 - base_labels, base_labels], dim=0).float()

    backbone.eval()
    public_logits = backbone(
        X=views.detach().clone(),
        y_train=labels,
        feature_shuffles=feature_shuffles,
        return_logits=True,
        softmax_temperature=estimator.softmax_temperature,
        inference_config=inference_config,
    )
    differentiable_logits = _forward_method(
        bundle=bundle,
        views=views,
        labels=labels,
        feature_shuffles=feature_shuffles,
        checkpoint_activations=True,
    )

    assert torch.allclose(
        differentiable_logits.float(), public_logits.float(), atol=2e-3, rtol=2e-3
    )
    differentiable_logits.float().sum().backward()
    assert views.grad is not None
    assert torch.isfinite(views.grad).all()


def test_classification_eval_uses_exact_public_numpy_aggregation_order():
    rows = 16
    task = OpenMLTaskData(
        task_id=7,
        dataset_id=8,
        dataset_name="classification_aggregation",
        problem_type="multiclass",
        n_classes=3,
        x_train=pd.DataFrame(
            {
                "x0": np.linspace(-1.0, 1.0, rows),
                "x1": np.arange(rows) % 4,
            }
        ),
        y_train=np.resize(np.arange(3), rows),
        x_test=pd.DataFrame({"x0": [0.1], "x1": [1]}),
        y_test=np.asarray([1]),
        outer_split_hash="test",
    )
    bundle = _fit_standard_bag(
        task=task,
        fit_indices=np.arange(rows),
        config=standard_direct_spline_config(),
        protocol_seed=0,
        bag=0,
        backbone=_TinyStandardBackbone(),
        device=torch.device("cpu"),
    )
    patterns = [
        pattern
        for method_patterns in bundle.estimator.ensemble_generator_.class_shuffles_.values()
        for pattern in method_patterns
    ]
    values = np.asarray(
        [
            [[1e4, -1e4, 0.25]],
            [[-1e4, 1e4, -0.25]],
            [[0.125, 0.12500001, 0.12499999]],
            [[25.0, 25.000002, 24.999998]],
            [[-25.0, -24.999998, -25.000002]],
            [[3.0, -7.0, 11.0]],
            [[-9.0, 2.0, 4.0]],
            [[0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )[: len(patterns)]
    method_sizes = [
        len(method_patterns)
        for method_patterns in bundle.estimator.ensemble_generator_.class_shuffles_.values()
    ]
    members = []
    offset = 0
    for size in method_sizes:
        members.append(values[offset : offset + size])
        offset += size

    actual = _aggregate_public_classification_members(bundle, members, patterns)
    expected = np.zeros_like(values[0])
    for output, pattern in zip(values, patterns, strict=True):
        expected += output[..., pattern]
    expected /= len(patterns)
    expected = bundle.estimator.softmax(
        expected,
        axis=-1,
        temperature=bundle.estimator.softmax_temperature,
    )
    expected = expected / expected.sum(axis=1, keepdims=True)

    assert np.array_equal(actual, expected.astype(np.float64))
