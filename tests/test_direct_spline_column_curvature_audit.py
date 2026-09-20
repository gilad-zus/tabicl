from __future__ import annotations

from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("column_curvature_audit", SCRIPTS / "direct_spline_openml_column_curvature_audit.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

from tabicl._hyperspline.module import DirectSplineTransform
from tabicl._experiments.direct_spline_openml_standard import _AdapterSet, _apply_adapter


def adapter(columns=3, mixing=0):
    result = DirectSplineTransform(torch.zeros(1, 4, columns), n_control_points=8,
                                   coordinate_mapping="arctan", direct_spline_output=True,
                                   cross_column_mixing_rank=mixing)
    with torch.no_grad():
        result.location.zero_()
        result.scale.fill_(1)
        result.gap_logits.copy_(torch.linspace(-1.1, 0.8, result.gap_logits.numel()).reshape_as(result.gap_logits))
        result.direct_center.fill_(0.13)
        result.direct_log_span.fill_(0.17)
        if mixing:
            result.mixing_gate.fill_(0.5)
    result.eval().requires_grad_(False)
    return result


def test_empirical_projection_recovers_lines_and_handles_constant_column():
    u = torch.tensor([[[-1., 2.], [0., 2.], [1., 2.]]])
    y = 3 * u + 7
    slope, intercept, residual = audit._fit_lines(u, y)
    torch.testing.assert_close(slope, torch.tensor([[3., 0.]]))
    torch.testing.assert_close(intercept, torch.tensor([[7., 13.]]))
    assert residual.max() == 0
    with pytest.raises(ValueError, match="nonfinite"):
        audit._fit_lines(u * float("nan"), y)


def test_projection_preserves_training_mean_and_is_orthogonal_to_coordinate():
    base = adapter()
    x = torch.linspace(-8, 8, 81).reshape(1, -1, 1).expand(-1, -1, 3)
    u, y = audit._coordinate(base, x), base.unmixed_transform(x)
    slope, intercept, residual = audit._fit_lines(u, y)
    error = y - (slope.unsqueeze(1) * u + intercept.unsqueeze(1))
    assert residual.max() > 1e-4
    assert error.mean(1).abs().max() < 2e-6
    assert (error * u).mean(1).abs().max() < 2e-6


def test_only_chosen_unmixed_column_changes_and_original_mixer_is_preserved():
    base = adapter(mixing=2)
    x = torch.linspace(-4, 4, 45).reshape(1, 15, 3)
    state = {key: value.clone() for key, value in base.state_dict().items()}
    slope, intercept, _ = audit._fit_lines(audit._coordinate(base, x), base.unmixed_transform(x))
    changed = audit._ProjectedColumn(base, slope, intercept, [1])
    before, after = base.unmixed_transform(x), changed.unmixed_transform(x)
    assert torch.equal(before[..., [0, 2]], after[..., [0, 2]])
    assert not torch.equal(before[..., 1], after[..., 1])
    torch.testing.assert_close(changed.transform(x), after + after @ base.effective_mixing_matrix())
    for key, value in state.items():
        assert torch.equal(base.state_dict()[key], value)
    unchanged = audit._ProjectedColumn(base, slope, intercept, [])
    assert torch.equal(unchanged.transform(x), base.transform(x))


def test_intervention_works_through_public_adapter_and_missing_feature_mask():
    base = adapter(mixing=2)
    x = torch.linspace(-3, 3, 36).reshape(12, 3)
    slope, intercept, _ = audit._fit_lines(audit._coordinate(base, x[None]), base.unmixed_transform(x[None]))
    changed = audit._ProjectedColumn(base, slope, intercept, [0])
    actual = _apply_adapter(x, numerical_indices=np.array([0, 1, 2]), adapter=changed)
    torch.testing.assert_close(actual, changed.transform(x[None])[0])
    mask = np.array([False, True, False])
    actual = _apply_adapter(x, numerical_indices=np.array([0, 1, 2]), adapter=changed, filtered_feature_mask=mask)
    unmixed = changed.unmixed_transform(x[None])
    inputs = unmixed.clone()
    inputs[..., 1] = 0
    torch.testing.assert_close(actual, (unmixed + inputs @ base.effective_mixing_matrix())[0])


def test_removal_from_straight_column_is_bit_exact():
    base = adapter(mixing=2)
    with torch.no_grad():
        base.gap_logits[:, 0].zero_()
    x = torch.randn(1, 50, 3)
    slope, intercept, _ = audit._fit_lines(audit._coordinate(base, x), base.unmixed_transform(x))
    changed = audit._ProjectedColumn(base, slope, intercept, [0])
    assert not changed.columns
    assert torch.equal(changed.transform(x), base.transform(x))


def test_column_mapping_survives_reorder_and_bag_constant_filter():
    estimator = SimpleNamespace(
        X_encoder_=SimpleNamespace(numeric_output_positions_=np.array([2, 3, 4]), numeric_input_positions_=np.array([0, 2, 4])),
        ensemble_generator_=SimpleNamespace(unique_filter_=SimpleNamespace(features_to_keep_=np.array([1, 1, 0, 1, 1], dtype=bool))),
    )
    bundle = SimpleNamespace(estimator=estimator, numerical_indices=np.array([2, 3]))
    assert audit._column_map(bundle) == {2: 0, 4: 1}


def test_selection_uses_only_oof_and_excludes_all_columns_line():
    assert audit._select({"unchanged": 1., "column_0": .9, "column_1": .8, "all_columns_line": 0.}) == "column_1"
    assert audit._select({"unchanged": 1., "column_0": 1.}) == "unchanged"
    with pytest.raises(ValueError):
        audit._select({"unchanged": float("nan")})
    assert audit._gain(0, .1) == {"error_reduction": -.1, "relative_error_reduction": None}


def test_resume_predictions_are_atomic_and_bound_to_fingerprint(tmp_path):
    path = tmp_path / "a.npz"
    values = {"oof": np.array([1., 2.]), "test": np.array([3.])}
    audit._save_predictions(path, "abc", values)
    np.testing.assert_array_equal(audit._cached(path, "abc", (2,), (1,))["test"], [3.])
    assert not path.with_name("a.npz.tmp").exists()
    with pytest.raises(ValueError, match="provenance"):
        audit._cached(path, "def", (2,), (1,))
    with pytest.raises(ValueError, match="invalid audit prediction"):
        audit._cached(path, "abc", (3,), (1,))


def test_replay_rejects_mismatch_nonfinite_and_shape_change():
    assert audit._replay(np.array([1.]), np.array([1.]), atol=0, rtol=0) == 0
    for actual in (np.array([2.]), np.array([np.nan]), np.ones((1, 1))):
        with pytest.raises(ValueError):
            audit._replay(actual, np.array([1.]), atol=1e-5, rtol=1e-5)


def test_splits_exclude_query_labels_from_context():
    audit._validate_splits([0, 1], [2, 3], [2], [3], 4)
    for a, b in (([2], [2]), ([2, 2], [3]), ([2], [4])):
        with pytest.raises(ValueError):
            audit._validate_splits([0, 1], [2, 3], a, b, 4)


def test_resume_allows_only_allocation_changes_on_equivalent_hardware():
    previous = {"semantic": {"code": "abc"}, "runtime": {"host": "one", "device": "cuda:0", "stable": {"gpu": "same"}}}
    moved = {"host": "two", "device": "cuda:1", "stable": {"gpu": "same"}}
    audit._validate_resume(previous, {"code": "abc"}, moved, resume=True, equivalent=True)
    with pytest.raises(ValueError, match="runtime"):
        audit._validate_resume(previous, {"code": "abc"}, moved, resume=True, equivalent=False)
    with pytest.raises(ValueError, match="semantics"):
        audit._validate_resume(previous, {"code": "changed"}, moved, resume=True, equivalent=True)
    moved["stable"]["gpu"] = "different"
    with pytest.raises(ValueError, match="runtime"):
        audit._validate_resume(previous, {"code": "abc"}, moved, resume=True, equivalent=True)


def test_projection_fits_each_branch_only_on_T(monkeypatch):
    branches = OrderedDict(none=adapter(2), power=adapter(2))
    values = np.array([[0., 1., 4.], [0., 2., 7.], [0., 5., 9.]], dtype=np.float32)
    generator = SimpleNamespace(preprocessors_=OrderedDict(none=SimpleNamespace(X_transformed_=values),
                                                           power=SimpleNamespace(X_transformed_=values * .25)))
    bundle = SimpleNamespace(numerical_indices=np.array([1, 2]), estimator=SimpleNamespace(ensemble_generator_=generator))
    coefficients, diagnostics = audit._project(bundle, _AdapterSet(branches), torch.device("cpu"))
    assert set(coefficients) == {"none", "power"}
    assert all(row["fit_rows"] == 3 for row in diagnostics.values())
    assert not torch.equal(coefficients["none"][0], coefficients["power"][0])


def test_tiny_end_to_end_audit_and_resume_without_new_predictions(tmp_path, monkeypatch):
    """Exercise A/B routing, saved baselines, projection, reports and resume."""
    from direct_spline_openml_crossfit_context_expansion import ContextExpansionBagPredictions, _save_bag
    train = pd.DataFrame({"x": np.linspace(-2, 2, 8), "category": pd.Categorical(["a", "b"] * 4), "z": np.arange(8.)})
    task = SimpleNamespace(task_id=7, dataset_name="tiny", problem_type="regression", n_classes=None,
                           x_train=train, y_train=np.arange(8.), x_test=train.iloc[:2].copy(), y_test=np.array([1., 2.]), outer_split_hash="split")
    staged, output = tmp_path / "staged", tmp_path / "audit"
    source = audit._task_path(staged / "continued_spline", task)
    audit._write(staged / "experiment_manifest.json", {"protocol_seed": 12, "bags": 2})
    audit._write(source / "task_provenance.json", {"checkpoint": {"sha256": "weights"}})
    audit._write(staged / "continued_line/task_summaries.json", [{"task_id": 7, "expanded_context": {"outer_test": {"raw_spline": {"benchmark_error": 1.}}}}])
    splits = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
    monkeypatch.setattr(audit, "_bag_splits", lambda *a, **k: iter(splits))
    monkeypatch.setattr(audit, "load_frozen_backbone", lambda **k: (torch.nn.Identity(), None, {"sha256": "weights"}))
    monkeypatch.setattr(audit, "_source_standard_prediction", lambda **k: np.array([1., 2.]))
    def make_bundle(**kwargs):
        fit = kwargs["fit_indices"]
        values = np.column_stack([np.zeros(len(fit)), train.x.to_numpy()[fit], train.z.to_numpy()[fit]])
        estimator = SimpleNamespace(
            X_encoder_=SimpleNamespace(numeric_output_positions_=np.array([1, 2]), numeric_input_positions_=np.array([0, 2])),
            ensemble_generator_=SimpleNamespace(preprocessors_=OrderedDict(none=SimpleNamespace(X_transformed_=values)),
                                                 unique_filter_=SimpleNamespace(features_to_keep_=np.ones(3, dtype=bool))),
        )
        return SimpleNamespace(estimator=estimator, numerical_indices=np.array([1, 2]), fit_labels=np.zeros(4), support_indices=np.arange(4))
    monkeypatch.setattr(audit, "_fit_standard_bag", make_bundle)
    monkeypatch.setattr(audit, "_make_adapters", lambda *a: _AdapterSet(OrderedDict(none=adapter(2))))
    calls = []
    def predict(**kwargs):
        query = kwargs["query_x"]
        calls.append((query.x.to_numpy(), kwargs["appended_indices"].copy()))
        x = torch.as_tensor(query[["x", "z"]].to_numpy().copy(), dtype=torch.float32)[None]
        with torch.no_grad():
            return kwargs["adapters"].for_method("none").transform(x).sum(-1)[0].numpy()
    monkeypatch.setattr(audit, "_append_prediction", predict)
    for bag, (fit, validation) in enumerate(splits):
        a, b = validation[:2], validation[2:]
        adapters = _AdapterSet(OrderedDict(none=adapter(2)))
        state = adapters.state_dict()
        checkpoint = {"config": {}, "provenance": {"task_id": 7, "outer_split_hash": "split", "bag": bag, "protocol_seed": 12,
                      "numerical_indices": [1, 2], "normalization_methods": ["none"]},
                      "checkpoints": {f"original_{side}": {"valid": True, "step": 2, "state_dict": state} for side in "ab"}}
        torch.save(checkpoint, source / f"bag_{bag}.adapters.pt")
        pa = predict(query_x=train.iloc[a], appended_indices=b, adapters=adapters)
        pb = predict(query_x=train.iloc[b], appended_indices=a, adapters=adapters)
        pt = predict(query_x=task.x_test, appended_indices=validation, adapters=adapters)
        fields = {}
        for field in ContextExpansionBagPredictions.__dataclass_fields__:
            if field.endswith("selection_a"):
                fields[field] = pa
            elif field.endswith("selection_b"):
                fields[field] = pb
            elif field.endswith("test"):
                fields[field] = pt
        _save_bag(source / f"bag_{bag}.npz", ContextExpansionBagPredictions(validation_indices=validation, selection_a_indices=a,
                  selection_b_indices=b, metadata={}, **fields))
    calls.clear()
    args = SimpleNamespace(replay_atol=1e-6, replay_rtol=1e-6)
    result = audit._run_task(args, SimpleNamespace(source_dir=tmp_path), task, staged, output, torch.device("cpu"), "fingerprint")
    assert result["n_columns"] == 2
    assert len(result["candidates"]) == 4
    assert result["ordinary_tabiclv2_test_error"] == 0
    assert all(len(context) in (2, 4) for _, context in calls)
    count = len(calls)
    resumed = audit._run_task(args, SimpleNamespace(source_dir=tmp_path), task, staged, output, torch.device("cpu"), "fingerprint")
    assert resumed == result
    assert len(calls) == count
    report = audit._task_path(output, task) / "column_curvature_summary.json"
    report.unlink()  # Exercise granular resume, not just completed-task resume.
    resumed = audit._run_task(args, SimpleNamespace(source_dir=tmp_path), task, staged, output, torch.device("cpu"), "fingerprint")
    assert resumed == result
    assert len(calls) == count
