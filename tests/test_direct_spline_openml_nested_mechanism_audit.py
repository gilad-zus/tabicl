from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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
audit = _load_module("direct_spline_openml_nested_mechanism_audit", "direct_spline_openml_nested_mechanism_audit.py")


def test_basis_leverage_is_larger_away_from_the_dense_context_region():
    knots = audit._uniform_knots(n_control_points=20, degree=3)
    context = np.concatenate((np.linspace(-0.25, 0.25, 101), np.array([-3.9, 3.9])))
    leverage = audit._basis_leverage(
        context=context,
        query=np.array([0.0, 3.0]),
        knots=knots,
        degree=3,
        n_control_points=20,
        standardized_range=4.0,
        ridge=1e-6,
    )
    assert leverage[1] > leverage[0]
    assert np.all(leverage >= 0.0)


def test_geometry_distortion_is_zero_for_identity_and_translation():
    context = np.array([-1.0, 0.0, 1.0])
    query = np.array([-0.8, 0.4])
    identity = audit._local_geometry_distortion(
        context=context,
        query=query,
        transformed_context=context,
        transformed_query=query,
    )
    translated = audit._local_geometry_distortion(
        context=context,
        query=query,
        transformed_context=context + 2.0,
        transformed_query=query + 2.0,
    )
    mismatched = audit._local_geometry_distortion(
        context=context,
        query=query,
        transformed_context=context + np.array([0.0, 1.0, 0.0]),
        transformed_query=query,
    )
    assert np.allclose(identity, 0.0)
    assert np.allclose(translated, 0.0)
    assert np.any(mismatched > 0.0)


def test_nested_rotations_are_disjoint_and_audit_every_row_once(monkeypatch):
    from types import SimpleNamespace

    task = SimpleNamespace(task_id=17, problem_type="regression", y_train=np.arange(12), x_train=None)
    folds = [
        (np.array([3, 4, 5, 6, 7, 8, 9, 10, 11]), np.array([0, 1, 2])),
        (np.array([0, 1, 2, 6, 7, 8, 9, 10, 11]), np.array([3, 4, 5])),
        (np.array([0, 1, 2, 3, 4, 5, 9, 10, 11]), np.array([6, 7, 8])),
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), np.array([9, 10, 11])),
    ]

    monkeypatch.setattr(audit, "effective_inner_bag_count", lambda task, requested_bags: 4)
    import tabicl._experiments.direct_spline_openml as base

    monkeypatch.setattr(base, "_bag_splits", lambda task, requested_bags, seed: iter(folds))
    rotations = audit._nested_rotations(task=task, requested_bags=4, protocol_seed=3)

    assert len(rotations) == 4
    assert np.array_equal(np.sort(np.concatenate([rotation.audit_indices for rotation in rotations])), np.arange(12))
    for rotation in rotations:
        assert not np.intersect1d(rotation.fit_indices, rotation.selection_indices).size
        assert not np.intersect1d(rotation.fit_indices, rotation.audit_indices).size
        assert not np.intersect1d(rotation.selection_indices, rotation.audit_indices).size
