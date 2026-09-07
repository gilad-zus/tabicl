from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module(name: str, filename: str):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "direct_spline_openml_support_audit" not in sys.modules:
    _load_module("direct_spline_openml_support_audit", "direct_spline_openml_support_audit.py")
if "direct_spline_openml_crossfit_blend" not in sys.modules:
    _load_module("direct_spline_openml_crossfit_blend", "direct_spline_openml_crossfit_blend.py")
experiment = _load_module(
    "direct_spline_openml_crossfit_context_expansion",
    "direct_spline_openml_crossfit_context_expansion.py",
)


def _bag() -> object:
    return experiment.ContextExpansionBagPredictions(
        validation_indices=np.asarray([0, 1]),
        selection_a_indices=np.asarray([0]),
        selection_b_indices=np.asarray([1]),
        original_identity_selection_a=np.asarray([10.0]),
        original_identity_selection_b=np.asarray([11.0]),
        # The A row must receive the B-selected state (20), and vice versa.
        original_spline_selected_on_b_selection_a=np.asarray([20.0]),
        original_spline_selected_on_a_selection_b=np.asarray([21.0]),
        original_identity_test=np.asarray([1.0]),
        original_spline_selected_on_a_test=np.asarray([3.0]),
        original_spline_selected_on_b_test=np.asarray([5.0]),
        expanded_identity_selection_a=np.asarray([30.0]),
        expanded_identity_selection_b=np.asarray([31.0]),
        expanded_spline_selected_on_b_selection_a=np.asarray([40.0]),
        expanded_spline_selected_on_a_selection_b=np.asarray([41.0]),
        expanded_identity_test=np.asarray([7.0]),
        expanded_spline_selected_on_a_test=np.asarray([9.0]),
        expanded_spline_selected_on_b_test=np.asarray([11.0]),
        metadata={},
    )


def test_assembly_keeps_opposite_half_checkpoint_and_separates_context_arms():
    task = SimpleNamespace(y_train=np.asarray([0.0, 1.0]), problem_type="regression", n_classes=None)

    original = experiment._assemble(task=task, bags=[_bag()], condition="original")
    expanded = experiment._assemble(task=task, bags=[_bag()], condition="expanded")

    assert np.array_equal(original[0], np.asarray([10.0, 11.0]))
    assert np.array_equal(original[1], np.asarray([20.0, 21.0]))
    assert np.array_equal(original[2], np.asarray([1.0]))
    assert np.array_equal(original[3], np.asarray([4.0]))
    assert np.array_equal(expanded[0], np.asarray([30.0, 31.0]))
    assert np.array_equal(expanded[1], np.asarray([40.0, 41.0]))
    assert np.array_equal(expanded[2], np.asarray([7.0]))
    assert np.array_equal(expanded[3], np.asarray([10.0]))


def test_minimum_pooled_oof_blend_uses_only_crossfitted_oof_rows_and_tie_breaks_lower():
    task = SimpleNamespace(problem_type="regression", n_classes=None, y_train=np.asarray([0.0, 1.0]))
    selection = experiment._minimum_pooled_oof_selection(
        task=task,
        identity_oof=np.asarray([0.0, 0.0]),
        spline_oof=np.asarray([0.0, 2.0]),
    )

    # alpha=.5 predicts the targets exactly. It is selected without any test
    # prediction or outer-test label being supplied to the function.
    assert selection["selected_alpha"] == 0.5


def test_reference_prediction_drift_is_diagnostic_but_indices_are_strict(tmp_path):
    reference = SimpleNamespace(
        validation_indices=np.asarray([0, 1]),
        selection_a_indices=np.asarray([0]),
        selection_b_indices=np.asarray([1]),
        identity_selection_a=np.asarray([0.0]),
        identity_selection_b=np.asarray([0.0]),
        spline_selected_on_b_selection_a=np.asarray([0.0]),
        spline_selected_on_a_selection_b=np.asarray([0.0]),
        identity_test=np.asarray([0.0]),
        spline_selected_on_a_test=np.asarray([0.0]),
        spline_selected_on_b_test=np.asarray([0.0]),
    )
    path = tmp_path / "bag.npz"
    path.touch()
    original_loader = experiment._load_reference_bag
    experiment._load_reference_bag = lambda _path: reference
    try:
        actual = _bag()
        diagnostic = experiment._verify_reference_bag(actual=actual, reference_path=path, atol=1e-8)
        assert diagnostic["within_atol"] is False
        assert diagnostic["max_abs"] == 21.0

        changed = experiment.ContextExpansionBagPredictions(
            **{**actual.__dict__, "selection_a_indices": np.asarray([1])}
        )
        with pytest.raises(ValueError, match="changed selection_a_indices"):
            experiment._verify_reference_bag(actual=changed, reference_path=path, atol=1e-8)
    finally:
        experiment._load_reference_bag = original_loader
