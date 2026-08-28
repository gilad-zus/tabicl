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
    _adapter_identity_penalty,
    _aggregate_public_classification_members,
    _cosine_scheduler,
    _differentiable_row_interaction,
    _enable_frozen_training_path,
    _fit_standard_bag,
    _forward_method,
    _fit_one_bag_standard,
    _identity_prediction,
    _identity_view_parity,
    _make_adapters,
    _many_class_training_logits,
    _normal_prediction,
    _single_validation_split,
    _StandardBag,
    run_task_full_context_refit_checkpoint_audit_standard,
    run_task_unconditional_full_context_refit_standard,
    run_task_validation_selected_full_refit_standard,
    run_task_config_standard,
    standard_direct_spline_config,
    summarize_full_context_refit_checkpoint_audit_experiment,
    summarize_full_context_refit_checkpoint_audit_task,
    summarize_full_context_refit_experiment,
    summarize_full_context_refit_task,
    summarize_validation_selected_full_refit_experiment,
    summarize_validation_selected_full_refit_task,
)
from tabicl._hyperspline import AdaptiveDirectSplineTransform, DirectSplineTransform
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
    assert standard_direct_spline_config()["row_interaction_chunk_rows"] == 2_048
    assert standard_direct_spline_config(train_context_rows=0)["train_context_rows"] is None
    assert standard_direct_spline_config(train_context_rows=128)["train_context_rows"] == 128
    assert standard_direct_spline_config(context_cap=128)["max_context_rows"] == 128
    assert standard_direct_spline_config(context_cap=128)["train_context_rows"] == 128


def test_standard_config_accepts_explicit_adapter_schedule():
    config = standard_direct_spline_config(
        adapter_steps=500,
        adapter_patience=10,
        validation_interval=10,
    )

    assert config["adapter_steps"] == 500
    assert config["adapter_patience"] == 10
    assert config["validation_interval"] == 10


def test_validation_selected_split_is_deterministic_disjoint_and_stratified():
    labels = np.tile([0, 1], 15)
    task = OpenMLTaskData(
        task_id=78,
        dataset_id=79,
        dataset_name="deterministic_validation_split",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": np.arange(labels.size)}),
        y_train=labels,
        x_test=pd.DataFrame({"x": [31.0, 32.0]}),
        y_test=np.asarray([0, 1]),
        outer_split_hash="split",
    )

    first_fit, first_validation = _single_validation_split(
        task, validation_fraction=0.2, seed=12_345
    )
    second_fit, second_validation = _single_validation_split(
        task, validation_fraction=0.2, seed=12_345
    )

    assert np.array_equal(first_fit, second_fit)
    assert np.array_equal(first_validation, second_validation)
    assert not np.intersect1d(first_fit, first_validation).size
    assert np.array_equal(np.sort(np.concatenate((first_fit, first_validation))), np.arange(labels.size))
    assert np.array_equal(np.unique(labels[first_fit]), np.asarray([0, 1]))
    assert np.array_equal(np.unique(labels[first_validation]), np.asarray([0, 1]))
    assert np.bincount(labels[first_fit]).min() >= 2


def test_identity_regularizer_is_zero_at_identity_and_differentiable_after_movement():
    adapter = DirectSplineTransform(
        torch.zeros((1, 3, 2)),
        n_control_points=8,
        trainable_location_scale=True,
        cross_column_mixing_rank=2,
    )
    with torch.no_grad():
        adapter.location.zero_()
        adapter.scale.fill_(1.0)
    adapters = standard_openml._AdapterSet(OrderedDict({"none": adapter}))

    initial = _adapter_identity_penalty(adapters)
    assert initial is not None
    assert float(initial.detach()) == 0.0
    with torch.no_grad():
        adapter.gap_logits[..., 0].add_(0.5)

    penalty = _adapter_identity_penalty(adapters)
    assert penalty is not None
    assert float(penalty.detach()) > 0.0
    penalty.backward()
    assert adapter.gap_logits.grad is not None
    assert torch.isfinite(adapter.gap_logits.grad).all()

    adapter.zero_grad(set_to_none=True)
    with torch.no_grad():
        adapter.gap_logits.zero_()
        assert adapter.mixing_left is not None
        assert adapter.mixing_right is not None
        assert adapter.mixing_weight_logits is not None
        assert adapter.mixing_gate is not None
        adapter.mixing_left.copy_(torch.eye(2).unsqueeze(0))
        adapter.mixing_right.copy_(torch.eye(2).unsqueeze(0))
        adapter.mixing_weight_logits.fill_(0.5)
        adapter.mixing_gate.fill_(0.5)
    mixing_penalty = _adapter_identity_penalty(adapters)
    assert mixing_penalty is not None
    assert float(mixing_penalty.detach()) > 0.0
    mixing_penalty.backward()
    assert adapter.mixing_gate.grad is not None


def test_cosine_schedule_preserves_the_long_horizon_prefix_for_a_short_refit():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = _cosine_scheduler(optimizer, total_steps=500, min_lr_ratio=0.01)
    for _ in range(25):
        optimizer.step()
        scheduler.step()
    long_horizon_rate_at_25 = optimizer.param_groups[0]["lr"]

    short_parameter = torch.nn.Parameter(torch.zeros(()))
    short_optimizer = torch.optim.SGD([short_parameter], lr=1.0)
    short_scheduler = _cosine_scheduler(short_optimizer, total_steps=25, min_lr_ratio=0.01)
    for _ in range(25):
        short_optimizer.step()
        short_scheduler.step()

    assert long_horizon_rate_at_25 > 0.9
    assert short_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


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


def test_standard_adaptive_direct_spline_is_bit_exact_identity_before_first_update():
    canonical = torch.tensor(
        [[-3.1251, -0.5001, 0.3334], [-1.0002, 0.4999, 1.7501], [0.1251, 2.0002, 4.1251]],
        dtype=torch.float32,
    )
    adapter = AdaptiveDirectSplineTransform(
        canonical.unsqueeze(0),
        expert_specs=((1, 4), (2, 8), (3, 20)),
        trainable_location_scale=True,
        cross_column_mixing_rank=3,
        cross_column_mixing_bound=0.1,
        conditional_rank=2,
    )
    with torch.no_grad():
        for expert in adapter.experts:
            expert.location.zero_()
            expert.scale.fill_(1.0)

    assert torch.equal(adapter.transform(canonical.unsqueeze(0)), canonical.unsqueeze(0))
    adapted = _apply_adapter(
        canonical,
        numerical_indices=np.array([0, 1, 2]),
        adapter=adapter,
    )
    assert torch.equal(adapted, canonical)


def test_standard_adapter_factory_accepts_the_frozen_adaptive_phase1_config():
    bundle = SimpleNamespace(
        numerical_indices=np.array([0, 1, 2]),
        estimator=SimpleNamespace(
            ensemble_generator_=SimpleNamespace(preprocessors_=["none", "power"])
        ),
    )
    config = {
        **standard_direct_spline_config(),
        "adapter_architecture": "conditional_adaptive_columns",
        "adaptive_expert_specs": ((1, 4), (2, 8), (3, 20)),
        "adaptive_routing_temperature": 1.0,
        "conditional_interaction_rank": 4,
        "conditional_interaction_bound": 0.25,
        "identity_regularization": 0.0,
    }
    adapters = _make_adapters(bundle, config, torch.device("cpu"))
    assert adapters is not None
    for method in ("none", "power"):
        adapter = adapters.for_method(method)
        assert isinstance(adapter, AdaptiveDirectSplineTransform)
        probe = torch.tensor([[[1.0, -2.0, 3.0]]])
        assert torch.equal(adapter.transform(probe), probe)


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


def test_standard_runner_can_preserve_full_cosine_checkpoint_trajectory_without_patience():
    rows = 32
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=111,
        dataset_id=112,
        dataset_name="cosine_preserved_checkpoint",
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
        "adapter_patience": None,
        "validation_interval": 1,
        "cosine_schedule_steps": 2,
        "cosine_min_lr_ratio": 0.01,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
    }
    result = _fit_one_bag_standard(
        task=task,
        fit_indices=np.arange(16),
        validation_indices=np.arange(16, 24),
        bag=0,
        config=config,
        protocol_seed=0,
        backbone=_HalfPrecisionEvalBackbone(),
        device=torch.device("cpu"),
        run_fingerprint_hash="test",
        progress=None,
        requested_bags=8,
        effective_bags=8,
    )

    assert result.metadata["adapter_steps_executed"] == 2
    assert result.metadata["adapter_early_stopping_patience"] is None
    assert result.metadata["adapter_schedule"] == {
        "kind": "cosine",
        "horizon_steps": 2,
        "min_lr_ratio": 0.01,
    }
    assert [record["step"] for record in result.metadata["adapter_checkpoint_records"]] == [1, 2]
    learning_rates = [record["learning_rates"][0] for record in result.metadata["adapter_checkpoint_records"]]
    assert learning_rates[1] < learning_rates[0]


def test_full_context_checkpoint_audit_runs_and_persists_real_adapter_states(tmp_path):
    rows = 32
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=96,
        dataset_id=97,
        dataset_name="real_checkpoint_audit",
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
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
    }

    identity, curve, metadata = standard_openml._fit_full_context_refit_checkpoint_audit_standard(
        task=task,
        config=config,
        checkpoint_steps=(0, 1, 2),
        protocol_seed=0,
        backbone=_HalfPrecisionEvalBackbone(),
        device=torch.device("cpu"),
        checkpoint_state_dir=tmp_path / "states",
        progress=None,
    )

    assert identity.shape == (8, 2)
    assert sorted(curve) == [0, 1, 2]
    assert np.array_equal(curve[0], identity)
    assert metadata["adapter_steps_executed"] == 2
    assert [item["step"] for item in metadata["checkpoint_metadata"]] == [0, 1, 2]
    assert metadata["checkpoint_metadata"][0]["adapter_diagnostics"]["mean_grid_deformation"] == 0.0
    assert (tmp_path / "states" / "step_000000.pt").is_file()
    assert (tmp_path / "states" / "step_000001.pt").is_file()
    assert (tmp_path / "states" / "step_000002.pt").is_file()


@pytest.mark.parametrize("selected_step", [0, 2])
def test_validation_selected_refit_freezes_both_arms_before_summary(
    tmp_path, monkeypatch, selected_step
):
    """The full refit receives the validation-selected duration, including zero."""

    labels = np.tile([0, 1], 6)
    task = OpenMLTaskData(
        task_id=98 + selected_step,
        dataset_id=99 + selected_step,
        dataset_name=f"validation_refit_{selected_step}",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": np.arange(labels.size, dtype=float)}),
        y_train=labels,
        x_test=pd.DataFrame({"x": [12.0, 13.0, 14.0, 15.0]}),
        y_test=np.asarray([0, 1, 0, 1]),
        outer_split_hash="split",
    )
    identity = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    adapted = np.asarray([[0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]])
    observed_steps: list[int] = []
    selection_calls: list[float] = []

    def fake_load_backbone(**_kwargs):
        return object(), None, {"test": "checkpoint"}

    def fake_selection_fit(**_kwargs):
        selection_calls.append(float(_kwargs["config"]["identity_regularization"]))
        validation_rows = len(_kwargs["validation_indices"])
        validation_identity = identity[:validation_rows].copy()
        validation_adapted = adapted[:validation_rows].copy()
        return {
            "identity_validation": validation_identity,
            "selected_validation": validation_adapted if selected_step else validation_identity.copy(),
            "identity_test": identity.copy(),
            "selected_test": adapted.copy() if selected_step else identity.copy(),
        }, {
            "selected_step": selected_step,
            "selected_use_adapted": bool(selected_step),
            "identity_validation_error": 0.5,
            "selected_validation_error": 0.4 if selected_step else 0.5,
            "selected_validation_relative_improvement": 0.2 if selected_step else 0.0,
            "train_seconds": 0.0,
        }, {}

    def fake_refit_fit(*, selected_steps, config, **_kwargs):
        observed_steps.append(selected_steps)
        assert config["cosine_schedule_steps"] == 5
        return identity.copy(), adapted.copy() if selected_steps else identity.copy(), {
            "selected_steps_requested": selected_steps,
            "adapter_steps_executed": selected_steps,
            "selected_duration_completed": True,
            "encountered_nonfinite_objective": False,
            "scheduler": {"horizon_steps": 5},
            "train_seconds": 0.0,
            "peak_allocated_gib": 0.0,
        }

    monkeypatch.setattr(standard_openml, "load_frozen_backbone", fake_load_backbone)
    monkeypatch.setattr(
        standard_openml, "_fit_validation_selected_checkpoint_standard", fake_selection_fit
    )
    monkeypatch.setattr(
        standard_openml, "_fit_selected_full_context_refit_standard", fake_refit_fit
    )
    config = {
        **standard_direct_spline_config(adapter_steps=5),
        "random_state": 20_260_826,
        "selection_checkpoint_interval": 1,
        "cosine_schedule_steps": 5,
        "cosine_min_lr_ratio": 0.01,
        "selection_relative_improvement": 0.005,
        "identity_regularization": 0.0,
    }
    result = run_task_validation_selected_full_refit_standard(
        task=task,
        config_labels=["cosine", "cosine_identity_regularized"],
        configs=[config, {**config, "identity_regularization": 0.01}],
        validation_fraction=0.25,
        validation_seed=20_260_826,
        output_dir=tmp_path,
        protocol_seed=0,
        device=torch.device("cpu"),
        classifier_checkpoint=None,
        regressor_checkpoint=None,
        resume=False,
        run_fingerprint_hash="test",
    )

    assert observed_steps == [selected_step, selected_step]
    for label in result["variants"]:
        artifact_dir = (
            tmp_path
            / "raw"
            / f"task_{task.task_id}_{task.dataset_name}"
            / f"config_{label}"
            / "validation_selected_refit"
        )
        assert (artifact_dir / "inner_split.npz").is_file()
        assert (artifact_dir / "selected_adapter_state.pt").is_file()
        predictions = np.load(artifact_dir / "predictions.npz")
        assert set(predictions.files) == {
            "inner_identity_validation",
            "inner_selected_validation",
            "inner_identity_test",
            "inner_selected_test",
            "full_identity_test",
            "full_selected_test",
        }

    task_summary = summarize_validation_selected_full_refit_task(
        task=task,
        output_dir=tmp_path,
        task_result=result,
    )
    assert task_summary["outer_test_scored_after_validation_selection_and_refit_predictions_frozen"]
    assert set(task_summary["variants"]) == {"cosine", "cosine_identity_regularized"}
    experiment_summary = summarize_validation_selected_full_refit_experiment(
        task_summaries=[task_summary],
        output_dir=tmp_path,
        bootstrap_rounds=10,
        bootstrap_seed=0,
    )
    assert set(experiment_summary["variant_paired_results"]) == {
        "cosine",
        "cosine_identity_regularized",
    }
    assert experiment_summary["selection_behavior"]["cosine"]["n_spline_selected"] == int(
        bool(selected_step)
    )

    corrupted_summary = (
        tmp_path
        / "raw"
        / f"task_{task.task_id}_{task.dataset_name}"
        / "config_cosine"
        / "validation_selected_refit"
        / "summary.json"
    )
    corrupted_summary.write_text("{", encoding="utf-8")
    observed_steps.clear()
    selection_calls.clear()
    resumed = run_task_validation_selected_full_refit_standard(
        task=task,
        config_labels=["cosine", "cosine_identity_regularized"],
        configs=[config, {**config, "identity_regularization": 0.01}],
        validation_fraction=0.25,
        validation_seed=20_260_826,
        output_dir=tmp_path,
        protocol_seed=0,
        device=torch.device("cpu"),
        classifier_checkpoint=None,
        regressor_checkpoint=None,
        resume=True,
        run_fingerprint_hash="test",
    )
    assert set(resumed["variants"]) == {"cosine", "cosine_identity_regularized"}
    assert observed_steps == [selected_step]
    assert selection_calls == [0.0]


@pytest.mark.parametrize("identity_regularization", [0.0, 0.01])
def test_validation_selected_checkpoint_and_fresh_refit_run_on_a_tiny_real_adapter(
    identity_regularization,
):
    rows = 32
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=103,
        dataset_id=104,
        dataset_name="tiny_validation_refit_real",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="split",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "adapter_steps": 2,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
        "random_state": 20_260_826,
        "selection_checkpoint_interval": 1,
        "cosine_schedule_steps": 2,
        "cosine_min_lr_ratio": 0.01,
        "selection_relative_improvement": 0.0,
        "identity_regularization": identity_regularization,
    }
    predictions, selection, selected_state = standard_openml._fit_validation_selected_checkpoint_standard(
        task=task,
        fit_indices=np.arange(18),
        validation_indices=np.arange(18, 24),
        config=config,
        protocol_seed=0,
        backbone=_HalfPrecisionEvalBackbone(),
        device=torch.device("cpu"),
        progress=None,
    )

    assert predictions["identity_validation"].shape == (6, 2)
    assert predictions["selected_test"].shape == (8, 2)
    assert [item["step"] for item in selection["checkpoint_records"]] == [0, 1, 2]
    assert selection["scheduler"]["horizon_steps"] == 2
    assert selection["fixed_horizon_completed"] is True
    assert selection["encountered_nonfinite_objective"] is False
    assert isinstance(selected_state, dict)
    identity, refit, refit_metadata = standard_openml._fit_selected_full_context_refit_standard(
        task=task,
        selected_steps=2,
        config=config,
        protocol_seed=0,
        backbone=_HalfPrecisionEvalBackbone(),
        device=torch.device("cpu"),
        progress=None,
    )

    assert identity.shape == (8, 2)
    assert refit.shape == (8, 2)
    assert refit_metadata["adapter_steps_executed"] == 2
    assert refit_metadata["scheduler"]["horizon_steps"] == 2


def test_nonfinite_selected_refit_is_reported_as_an_invalid_endpoint(monkeypatch):
    rows = 32
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=105,
        dataset_id=106,
        dataset_name="nonfinite_refit",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="split",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "adapter_steps": 2,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
        "random_state": 20_260_826,
        "cosine_schedule_steps": 2,
        "cosine_min_lr_ratio": 0.01,
        "identity_regularization": 0.0,
    }

    def nonfinite_objective(**_kwargs):
        value = torch.full((), float("nan"))
        return value, value, value

    monkeypatch.setattr(standard_openml, "_adapter_training_objective", nonfinite_objective)
    identity, selected, metadata = standard_openml._fit_selected_full_context_refit_standard(
        task=task,
        selected_steps=2,
        config=config,
        protocol_seed=0,
        backbone=_HalfPrecisionEvalBackbone(),
        device=torch.device("cpu"),
        progress=None,
    )

    assert identity.shape == (8, 2)
    assert np.isnan(selected).all()
    assert metadata["adapter_steps_executed"] == 0
    assert metadata["selected_duration_completed"] is False
    assert metadata["encountered_nonfinite_objective"] is True
    assert metadata["selected_prediction_policy"] == "invalid_prediction_after_incomplete_refit"


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


def test_full_context_refit_is_unconditional_and_oof_is_posthoc_only(tmp_path, monkeypatch):
    """Old bag evidence must not select the full-row schedule or prediction."""

    task = OpenMLTaskData(
        task_id=93,
        dataset_id=94,
        dataset_name="full_refit",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        y_train=np.asarray([0, 1, 0, 1]),
        x_test=pd.DataFrame({"x": [4.0, 5.0]}),
        y_test=np.asarray([0, 1]),
        outer_split_hash="split",
    )
    config = standard_direct_spline_config()
    config["adapter_steps"] = 500
    identity = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    adapted = np.asarray([[0.7, 0.3], [0.3, 0.7]])
    source_dir = tmp_path / "completed_oof_source"
    destination_dir = tmp_path / "new_full_refit"
    config_dir = source_dir / "raw" / "task_93_full_refit" / "config_D"
    config_dir.mkdir(parents=True)
    standard_openml._json_dump(
        config_dir / "config_summary.json",
        {
            "run_fingerprint_hash": "test",
            "validation": {
                # This is below the historical 0.5% gate. The new experiment
                # must nevertheless report the unconditional full-row spline.
                "identity": {"deployment_error": 0.3000},
                "adapted": {"deployment_error": 0.2990},
            },
            "bag_metadata": [
                {"adapter_has_valid_learned_checkpoint": True, "adapter_best_step": 20},
                {"adapter_has_valid_learned_checkpoint": True, "adapter_best_step": 40},
                {"adapter_has_valid_learned_checkpoint": False, "adapter_best_step": 500},
            ],
        },
    )
    np.savez_compressed(
        config_dir / "config_predictions.npz",
        identity_test=identity,
        adapted_test=adapted,
        guarded_test=identity,
    )
    baseline_dir = source_dir / "standard_tabarena_baseline" / "task_93_full_refit"
    baseline_dir.mkdir(parents=True)
    np.savez_compressed(baseline_dir / "predictions.npz", prediction=identity)

    observed: dict[str, int] = {}

    def fake_load_backbone(**_kwargs):
        return object(), None, {"test": "checkpoint"}

    def fake_full_fit(*, refit_steps, **_kwargs):
        observed["refit_steps"] = refit_steps
        return identity.copy(), adapted.copy(), {
            "adapter_refit_steps_requested": refit_steps,
            "adapter_steps_executed": refit_steps,
            "train_seconds": 0.0,
            "peak_allocated_gib": 0.0,
        }

    monkeypatch.setattr(standard_openml, "load_frozen_backbone", fake_load_backbone)
    monkeypatch.setattr(standard_openml, "_fit_full_context_refit_standard", fake_full_fit)

    refit = run_task_unconditional_full_context_refit_standard(
        task=task,
        config_labels=["D"],
        configs=[config],
        output_dir=destination_dir,
        protocol_seed=0,
        device=torch.device("cpu"),
        classifier_checkpoint=None,
        regressor_checkpoint=None,
        resume=False,
        run_fingerprint_hash="test",
    )
    predictions = np.load(
        destination_dir / "raw" / "task_93_full_refit" / "config_D" / "full_context_refit" / "predictions.npz"
    )
    assert observed["refit_steps"] == 500
    assert refit["selection"]["mode"] == "predeclared_fixed_schedule_no_guard"
    assert refit["selection"]["oof_used_for_training_or_deployment"] is False
    assert np.array_equal(predictions["guarded_test"], adapted)

    task_summary = summarize_full_context_refit_task(
        task=task,
        output_dir=destination_dir,
        refit_result=refit,
        oof_source_dir=source_dir,
    )
    assert task_summary["deployment_guard_applied"] is False
    assert task_summary["reported_arm_aliases"] == {"full_refit_guarded": "full_refit_raw"}
    assert task_summary["full_refit_guarded"]["deployment_error"] == pytest.approx(
        task_summary["full_refit_raw"]["deployment_error"]
    )
    diagnostic = task_summary["oof_validation_diagnostic"]
    assert diagnostic["historical_guard_selected_adapted"] is False
    assert diagnostic["used_for_step_selection"] is False
    assert diagnostic["used_as_deployment_guard"] is False
    assert diagnostic["loaded_after_full_refit_prediction_frozen"] is True
    experiment_summary = summarize_full_context_refit_experiment(
        task_summaries=[task_summary],
        output_dir=destination_dir,
        bootstrap_rounds=20,
        bootstrap_seed=0,
    )
    assert experiment_summary["distinct_paired_result_keys"] == ["full_refit_raw"]
    assert experiment_summary["paired_result_aliases"] == {
        "full_refit_guarded": "full_refit_raw"
    }
    assert experiment_summary["posthoc_oof_correlation"]["n_tasks"] == 1


def test_full_context_checkpoint_audit_freezes_curve_before_scoring(tmp_path, monkeypatch):
    """The audit stores all requested predictions and never selects a checkpoint."""

    task = OpenMLTaskData(
        task_id=94,
        dataset_id=95,
        dataset_name="checkpoint_audit",
        problem_type="binary",
        n_classes=2,
        x_train=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
        y_train=np.asarray([0, 1, 0, 1]),
        x_test=pd.DataFrame({"x": [4.0, 5.0]}),
        y_test=np.asarray([0, 1]),
        outer_split_hash="split",
    )
    config = standard_direct_spline_config(adapter_steps=5)
    identity = np.asarray([[0.1, 0.9], [0.8, 0.2]])
    step_two = np.asarray([[0.7, 0.3], [0.6, 0.4]])
    step_five = np.asarray([[0.2, 0.8], [0.1, 0.9]])
    observed: dict[str, object] = {}

    def fake_load_backbone(**_kwargs):
        return object(), None, {"test": "checkpoint"}

    def fake_audit_fit(*, checkpoint_steps, checkpoint_state_dir, **_kwargs):
        observed["checkpoint_steps"] = checkpoint_steps
        observed["checkpoint_state_dir"] = checkpoint_state_dir
        return identity.copy(), {0: identity.copy(), 2: step_two.copy(), 5: step_five.copy()}, {
            "adapter_steps_executed": 5,
            "train_seconds": 0.0,
            "peak_allocated_gib": 0.0,
            "checkpoint_metadata": [
                {"step": 0, "objective": None, "elapsed_seconds": 0.0, "adapter_diagnostics": {}},
                {"step": 2, "objective": 0.6, "elapsed_seconds": 1.0, "adapter_diagnostics": {}},
                {"step": 5, "objective": 0.2, "elapsed_seconds": 2.0, "adapter_diagnostics": {}},
            ],
        }

    monkeypatch.setattr(standard_openml, "load_frozen_backbone", fake_load_backbone)
    monkeypatch.setattr(
        standard_openml, "_fit_full_context_refit_checkpoint_audit_standard", fake_audit_fit
    )

    audit = run_task_full_context_refit_checkpoint_audit_standard(
        task=task,
        config_labels=["D"],
        configs=[config],
        checkpoint_steps=(0, 2, 5),
        output_dir=tmp_path,
        protocol_seed=0,
        device=torch.device("cpu"),
        classifier_checkpoint=None,
        regressor_checkpoint=None,
        resume=False,
        run_fingerprint_hash="test",
    )

    assert observed["checkpoint_steps"] == (0, 2, 5)
    assert audit["selection"]["checkpoint_selection"] == "none; every requested checkpoint is retained"
    assert audit["checkpoint_steps_frozen"] == [0, 2, 5]
    prediction_path = (
        tmp_path
        / "raw"
        / "task_94_checkpoint_audit"
        / "config_D"
        / "full_context_checkpoint_audit"
        / "predictions.npz"
    )
    predictions = np.load(prediction_path)
    assert set(predictions.files) == {
        "identity_test",
        "adapted_test_step_000000",
        "adapted_test_step_000002",
        "adapted_test_step_000005",
    }

    task_summary = summarize_full_context_refit_checkpoint_audit_task(
        task=task,
        output_dir=tmp_path,
        audit_result=audit,
    )

    assert task_summary["outer_test_scored_after_all_checkpoint_predictions_frozen"] is True
    assert task_summary["outer_test_used_for_selection"] is False
    assert [item["step"] for item in task_summary["checkpoint_metrics"]] == [0, 2, 5]
    assert task_summary["checkpoint_metrics"][-1]["candidate"]["deployment_error"] < (
        task_summary["full_context_identity"]["deployment_error"]
    )
    experiment_summary = summarize_full_context_refit_checkpoint_audit_experiment(
        task_summaries=[task_summary],
        output_dir=tmp_path,
        bootstrap_rounds=20,
        bootstrap_seed=0,
    )
    assert experiment_summary["checkpoint_steps"] == [0, 2, 5]
    assert set(experiment_summary["checkpoint_paired_results"]) == {"0", "2", "5"}


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


def test_differentiable_row_chunking_matches_unbatched_forward_and_input_gradient():
    """Large-table row batching is mathematically exact and retains gradients."""

    torch.manual_seed(17)
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
        dropout=0.0,
        zero_init=False,
    )
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    _enable_frozen_training_path(backbone)

    # The column encoder produces a non-leaf embedding tensor. RowInteraction
    # writes its learned CLS token into that tensor, so model the real path
    # rather than passing it a leaf test fixture.
    embedding_source = torch.randn(3, 23, 4, 8, requires_grad=True)
    embeddings = embedding_source * 1
    calls: list[int] = []
    hook = backbone.row_interactor.register_forward_hook(
        lambda _module, inputs, _output: calls.append(int(inputs[0].shape[1]))
    )
    try:
        unbatched = _differentiable_row_interaction(
            backbone=backbone,
            embeddings=embeddings,
            chunk_rows=64,
        )
        unbatched.sum().backward()
        unbatched_gradient = embedding_source.grad.detach().clone()
        assert calls == [23]

        embedding_source_chunked = embedding_source.detach().clone().requires_grad_()
        embeddings_chunked = embedding_source_chunked * 1
        calls.clear()
        chunked = _differentiable_row_interaction(
            backbone=backbone,
            embeddings=embeddings_chunked,
            chunk_rows=5,
        )
        chunked.sum().backward()
        assert calls == [5, 5, 5, 5, 3]
    finally:
        hook.remove()

    assert torch.allclose(chunked, unbatched, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        embedding_source_chunked.grad, unbatched_gradient, atol=1e-6, rtol=1e-6
    )
    assert torch.isfinite(embedding_source_chunked.grad).all()


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
