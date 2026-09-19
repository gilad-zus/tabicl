from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl._hyperspline.feature_expansion import SplineFeatureExpansion
from tabicl._hyperspline.bspline import greville_abscissae
from tabicl._experiments import direct_spline_openml_standard as standard
from tabicl._experiments.direct_spline_openml import OpenMLTaskData
from tabicl._model.tabicl import TabICL


def _script(name):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_script("direct_spline_openml_support_audit")
_script("direct_spline_openml_crossfit_blend")
runner = _script("direct_spline_openml_feature_expansion")
replacement = _script("direct_spline_openml_crossfit_context_expansion")


@pytest.mark.parametrize("capacity", [8, 12, 20])
def test_initial_features_are_bit_exact_line_and_trainable(capacity):
    values = torch.linspace(-1e4, 1e4, 201).reshape(1, -1, 1).expand(1, -1, 3)
    line = SplineFeatureExpansion(3, n_control_points=None)
    spline = SplineFeatureExpansion(3, n_control_points=capacity)
    assert torch.equal(line.transform(values), spline.transform(values))
    assert spline.smoothness() == 0
    spline.transform(values).square().mean().backward()
    assert torch.isfinite(spline.coefficients.grad).all()
    assert spline.coefficients.grad.abs().sum() > 0


@pytest.mark.parametrize("capacity", [8, 20])
def test_smoothness_measures_function_not_number_of_controls(capacity):
    spline = SplineFeatureExpansion(1, n_control_points=capacity).double()
    knots = spline.knots
    with torch.no_grad():
        # Exact coefficients for u^2 in a cubic B-spline basis.
        quadratic = torch.stack([
            (knots[i+1]*knots[i+2] + knots[i+1]*knots[i+3] + knots[i+2]*knots[i+3]) / 3
            for i in range(capacity)
        ])
        spline.coefficients.copy_(quadratic.reshape(1, 1, -1))
    assert float(spline.smoothness().detach()) == pytest.approx(0.25, abs=1e-10)
    with torch.no_grad():
        spline.coefficients.copy_(3 * greville_abscissae(knots, 3, capacity) + 2)
    assert float(spline.smoothness().detach()) < 1e-20


def test_augmented_public_and_training_views_match_with_masking():
    arrays = np.arange(21, dtype=np.float32).reshape(7, 3)
    generator = SimpleNamespace(ensemble_configs_={"none": [([2, 0, 1], np.array([0, 1])), ([1, 2, 0], np.array([1, 0]))]})
    bundle = SimpleNamespace(estimator=SimpleNamespace(ensemble_generator_=generator), numerical_indices=np.array([0, 2]), problem_type="binary")
    adapters = standard._AdapterSet(OrderedDict(none=SplineFeatureExpansion(2, n_control_points=8)))
    common = dict(bundle=bundle, method="none", context_canonical=arrays[:4], query_canonical=arrays[4:],
                  context_labels=np.array([0, 1, 0, 1]), adapters=adapters, device=torch.device("cpu"))
    for mask in (None, np.array([True, False, False])):
        actual = standard._build_method_batch(**common, filtered_feature_mask=mask)
        public = standard._build_public_method_arrays(**common, filtered_feature_mask=mask)
        assert np.array_equal(actual[0].detach().numpy(), public[0])
        assert np.array_equal(actual[1].numpy(), public[1])
        assert actual[2] == public[2]
        assert actual[0].shape[-1] == (5 if mask is None else 3)
        for member, permutation in zip(actual[0], actual[2]):
            canonical = member[:, np.argsort(permutation)].detach().numpy()
            assert np.array_equal(canonical[:, :3 if mask is None else 2], arrays if mask is None else arrays[:, 1:])
    actual[0].sum().backward()
    assert adapters.for_method("none").coefficients.grad is not None


def test_cross_capacity_start_preserves_trained_line_and_mixer():
    from tabicl._hyperspline import DirectSplineTransform
    def adapter(capacity, shape):
        module = DirectSplineTransform(torch.zeros(1, 4, 3), n_control_points=capacity,
                                       trainable_shape=shape, trainable_location_scale=False,
                                       coordinate_mapping="arctan", direct_spline_output=True,
                                       cross_column_mixing_rank=2)
        with torch.no_grad():
            module.location.zero_()
            module.scale.fill_(1.0)
        return standard._AdapterSet(OrderedDict(none=module))
    source, target = adapter(20, False), adapter(12, True)
    with torch.no_grad():
        source.for_method("none").direct_center.uniform_(-0.2, 0.2)
        source.for_method("none").direct_log_span.uniform_(-0.2, 0.2)
        source.for_method("none").mixing_gate.uniform_(-0.2, 0.2)
    replacement._load_staged_line_state(target, source.state_dict())
    grid = torch.randn(1, 41, 3) * 4
    assert torch.equal(source.for_method("none").transform(grid), target.for_method("none").transform(grid))
    with torch.no_grad():
        target.for_method("none").gap_logits.uniform_(-1.0, 1.0)
    replacement._load_staged_line_state(target, source.state_dict())
    assert torch.equal(source.for_method("none").transform(grid), target.for_method("none").transform(grid))
    with torch.no_grad():
        source.for_method("none").gap_logits.fill_(0.1)
    with pytest.raises(ValueError, match="shape-frozen"):
        replacement._load_staged_line_state(target, source.state_dict())


def _tiny_bundle(problem_type="multiclass"):
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(size=(36, 2)), columns=["x", "z"])
    labels = np.tile(np.arange(3), 12) if problem_type == "multiclass" else np.asarray(x.x + 0.2 * x.z)
    task = OpenMLTaskData(task_id=1, dataset_id=1, dataset_name="tiny", problem_type=problem_type,
                          n_classes=3 if problem_type == "multiclass" else None,
                          x_train=x, y_train=labels, x_test=x.iloc[:4].copy(),
                          y_test=labels[:4], outer_split_hash="test")
    backbone = TabICL(max_classes=3 if problem_type == "multiclass" else 0, num_quantiles=31,
                     embed_dim=8, col_num_blocks=1, col_nhead=1, col_num_inds=2,
                     col_feature_group="same", row_num_blocks=1, row_nhead=1, row_num_cls=1,
                     icl_num_blocks=1, icl_nhead=1, col_ssmax=False, icl_ssmax=False, dropout=0.0, zero_init=False)
    fit = np.arange(24)
    bundle = standard._fit_standard_bag(task=task, fit_indices=fit,
                 config={"max_context_rows": None, "row_interaction_chunk_rows": 16},
                 protocol_seed=1, bag=0, backbone=backbone, device=torch.device("cpu"))
    return task, bundle, fit


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"])
def test_end_to_end_feature_training_and_completed_resume(tmp_path, problem_type):
    task, bundle, fit = _tiny_bundle(problem_type)
    args = SimpleNamespace(steps=2, validation_interval=1, learning_rate=1e-3,
                           smoothness_weight=1e-4, training_seed=10, audit_episodes=1)
    a, b = np.arange(24, 30), np.arange(30, 36)
    baseline = runner._baselines(task, bundle, 0, fit, a, b, torch.device("cpu"), "test", tmp_path)
    result = runner._fit_arm(args, task, bundle, 0, fit, a, b, "spline8", torch.device("cpu"), "test", tmp_path)
    assert result["test_a"].shape == ((4, 3) if problem_type == "multiclass" else (4,))
    assert np.isfinite(result["test_a"]).all()
    assert len(result["metadata"]["checkpoints"]) == 3
    assert set(result["metadata"]["state_diagnostics"]) == {"final", "selected_a", "selected_b"}
    assert all(parameter.grad is None for parameter in bundle.backbone.parameters())
    again = runner._fit_arm(args, task, bundle, 0, fit, a, b, "spline8", torch.device("cpu"), "test", tmp_path)
    assert np.array_equal(again["test_a"], result["test_a"])
    assert baseline["identity_test"].shape == result["test_a"].shape


def test_interrupted_optimizer_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    task, bundle, fit = _tiny_bundle()
    args = SimpleNamespace(steps=2, validation_interval=1, learning_rate=1e-3,
                           smoothness_weight=1e-4, training_seed=10, audit_episodes=1)
    a, b = np.arange(24, 30), np.arange(30, 36)
    reference_dir, resume_dir = tmp_path / "reference", tmp_path / "resume"
    reference_dir.mkdir()
    resume_dir.mkdir()
    reference = runner._fit_arm(args, task, bundle, 0, fit, a, b, "spline8", torch.device("cpu"), "test", reference_dir)
    real_save = runner._atomic_torch_save
    def interrupt_after_save(value, path):
        real_save(value, path)
        if str(path).endswith("progress.pt") and value["step"] == 1:
            raise RuntimeError("simulated time limit")
    monkeypatch.setattr(runner, "_atomic_torch_save", interrupt_after_save)
    with pytest.raises(RuntimeError, match="simulated"):
        runner._fit_arm(args, task, bundle, 0, fit, a, b, "spline8", torch.device("cpu"), "test", resume_dir)
    monkeypatch.setattr(runner, "_atomic_torch_save", real_save)
    resumed = runner._fit_arm(args, task, bundle, 0, fit, a, b, "spline8", torch.device("cpu"), "test", resume_dir)
    assert np.array_equal(reference["final_test"], resumed["final_test"])
    assert reference["metadata"]["checkpoints"] == resumed["metadata"]["checkpoints"]


def test_oof_uses_opposite_selected_states():
    task = SimpleNamespace(y_train=np.zeros(2))
    oof, test = runner._assemble(task, [{"a": np.array([0]), "b": np.array([1]),
                         "oof_a": np.array([4.0]), "oof_b": np.array([5.0]),
                         "test_a": np.array([6.0]), "test_b": np.array([8.0])}])
    assert np.array_equal(oof, [4.0, 5.0])
    assert np.array_equal(test, [7.0])


def test_linearization_preserves_a_straight_spline_using_only_training_coordinates():
    _, bundle, _ = _tiny_bundle()
    adapters = runner._adapters(bundle, 8, torch.device("cpu"))
    with torch.no_grad():
        for adapter in adapters.adapters.values():
            adapter.coefficients.copy_(1.3 * greville_abscissae(adapter.knots, 3, 8) + 0.7)
    lines = runner._linearized_features(bundle, adapters, torch.device("cpu"))
    values = torch.linspace(-8, 8, 51).view(1, -1, 1).expand(1, -1, len(bundle.numerical_indices))
    for method in bundle.estimator.ensemble_generator_.preprocessors_:
        assert torch.allclose(lines.for_method(method).transform(values), adapters.for_method(method).transform(values), atol=3e-6, rtol=1e-6)
