from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


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
blend = _load_module("direct_spline_openml_crossfit_blend", "direct_spline_openml_crossfit_blend.py")


def test_crossfit_validation_halves_are_deterministic_disjoint_and_keep_available_classes():
    task = SimpleNamespace(
        task_id=42,
        problem_type="multiclass",
        y_train=np.asarray([0, 0, 0, 1, 1, 1, 2, 2]),
    )
    heldout = np.arange(8)

    first_a, first_b = blend._crossfit_validation_halves(task=task, validation_indices=heldout, seed=17)
    second_a, second_b = blend._crossfit_validation_halves(task=task, validation_indices=heldout, seed=17)

    assert np.array_equal(first_a, second_a)
    assert np.array_equal(first_b, second_b)
    assert not np.intersect1d(first_a, first_b).size
    assert np.array_equal(np.sort(np.concatenate((first_a, first_b))), heldout)
    # Classes with at least two held-out examples are represented on both
    # sides; the singleton class is deliberately allowed on one side only.
    for label in (0, 1, 2):
        positions = heldout[task.y_train == label]
        assert np.intersect1d(first_a, positions).size
        assert np.intersect1d(first_b, positions).size


def test_one_standard_error_rule_prefers_identity_when_small_gain_is_not_resolved():
    task = SimpleNamespace(problem_type="regression", n_classes=None)
    units = [
        (np.asarray([1.0]), np.asarray([0.0]), np.asarray([1.0 - np.sqrt(0.7)])),
        (np.asarray([1.0]), np.asarray([0.0]), np.asarray([1.0 - np.sqrt(1.2)])),
    ]

    result = blend._alpha_selection(task=task, units=units, alphas=(0.0, 1.0))

    assert result["best_alpha_before_one_se"] == 1.0
    assert result["selected_alpha"] == 0.0
    assert result["eligible_alphas"] == [0.0, 1.0]


def test_one_standard_error_rule_keeps_a_large_stable_spline_gain():
    task = SimpleNamespace(problem_type="regression", n_classes=None)
    units = [
        (np.asarray([1.0]), np.asarray([0.0]), np.asarray([0.5])),
        (np.asarray([2.0]), np.asarray([0.0]), np.asarray([1.0])),
    ]

    result = blend._alpha_selection(task=task, units=units, alphas=(0.0, 1.0))

    assert result["best_alpha_before_one_se"] == 1.0
    assert result["selected_alpha"] == 1.0
    assert result["eligible_alphas"] == [1.0]


def test_oof_assembly_uses_only_the_opposite_half_selected_state():
    task = SimpleNamespace(
        y_train=np.asarray([0.0, 1.0, 2.0, 3.0]),
        problem_type="regression",
        n_classes=None,
    )
    first = blend.CrossfitBagPredictions(
        validation_indices=np.asarray([0, 1]),
        selection_a_indices=np.asarray([0]),
        selection_b_indices=np.asarray([1]),
        identity_selection_a=np.asarray([10.0]),
        identity_selection_b=np.asarray([11.0]),
        spline_selected_on_b_selection_a=np.asarray([20.0]),
        spline_selected_on_a_selection_b=np.asarray([21.0]),
        identity_test=np.asarray([1.0]),
        spline_selected_on_a_test=np.asarray([3.0]),
        spline_selected_on_b_test=np.asarray([5.0]),
        metadata={},
    )
    second = blend.CrossfitBagPredictions(
        validation_indices=np.asarray([2, 3]),
        selection_a_indices=np.asarray([2]),
        selection_b_indices=np.asarray([3]),
        identity_selection_a=np.asarray([12.0]),
        identity_selection_b=np.asarray([13.0]),
        spline_selected_on_b_selection_a=np.asarray([22.0]),
        spline_selected_on_a_selection_b=np.asarray([23.0]),
        identity_test=np.asarray([3.0]),
        spline_selected_on_a_test=np.asarray([7.0]),
        spline_selected_on_b_test=np.asarray([9.0]),
        metadata={},
    )

    identity_oof, spline_oof, units, identity_test, spline_test = blend._assemble_task_predictions(
        task=task, bag_results=[first, second]
    )

    assert np.array_equal(identity_oof, np.asarray([10.0, 11.0, 12.0, 13.0]))
    assert np.array_equal(spline_oof, np.asarray([20.0, 21.0, 22.0, 23.0]))
    assert len(units) == 4
    assert np.array_equal(identity_test, np.asarray([2.0]))
    assert np.array_equal(spline_test, np.asarray([6.0]))
