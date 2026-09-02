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


# The basis audit deliberately reuses the source/split/preprocessing helpers
# from the support audit.  Load that sibling module exactly as the test runner
# loads standalone scripts, before loading the dependent script.
_load_module("direct_spline_openml_support_audit", "direct_spline_openml_support_audit.py")
audit = _load_module("direct_spline_openml_basis_leverage_audit", "direct_spline_openml_basis_leverage_audit.py")


def test_uniform_bspline_design_has_partition_and_identity_reproduction():
    controls = 8
    degree = 3
    knots = audit._uniform_knots(n_control_points=controls, degree=degree)
    values = np.array([-5.0, -4.0, -2.5, 0.0, 3.75, 4.0, 5.0])
    design = audit._bspline_design(
        values=values,
        knots=knots,
        degree=degree,
        n_control_points=controls,
        standardized_range=4.0,
    )
    assert design.shape == (values.size, controls)
    assert np.allclose(design.sum(axis=1), 1.0)
    greville = np.array([np.mean(knots[index + 1 : index + degree + 1]) for index in range(controls)])
    assert np.allclose(design @ greville, np.clip(values / 4.0, -1.0, 1.0))


def test_quantile_knots_are_strictly_ordered_even_for_repeated_context_values():
    knots = audit._quantile_knots(
        context=np.array([-4.0] * 8 + [0.0] * 80 + [4.0] * 8),
        n_control_points=20,
        degree=3,
        standardized_range=4.0,
    )
    internal_boundaries = knots[4:-4]
    widths = np.diff(np.concatenate((np.array([-1.0]), internal_boundaries, np.array([1.0]))))
    assert np.all(widths >= audit._MIN_KNOT_INTERVAL - 1e-12)
    assert np.all(np.diff(knots) >= 0.0)


def test_leverage_is_finite_and_normalised_to_context_mean():
    candidate = audit.BasisCandidate("uniform_k8", "uniform", 3, 8)
    context = np.linspace(-4.0, 4.0, 101)
    result = audit._split_leverage(
        context=context,
        validation=np.array([-0.1, 0.0, 0.1]),
        test=np.array([-4.0, 4.0]),
        candidate=candidate,
        standardized_range=4.0,
        ridge_relative=1e-6,
    )
    assert np.isfinite(list(result.values())).all()
    assert np.isclose(result["context_mean_relative_leverage"], 1.0)
    # Boundary coordinates are less represented than the average interior
    # context point for this uniform basis.
    assert result["test_mean_relative_leverage"] > 1.0


def test_candidate_summary_uses_dimensionless_relative_regret():
    records = [
        {
            "candidate": "uniform_k20",
            "test_minus_validation_mean_relative_leverage": shift,
            "test_adapted_relative_regret": regret,
            "test_adapted_outcome": outcome,
        }
        for shift, regret, outcome in [(-0.2, -0.1, "win"), (0.0, 0.0, "tie"), (0.3, 0.2, "loss")]
    ]
    result = audit._candidate_summary(records, candidate="uniform_k20", bootstrap_rounds=25, bootstrap_seed=7)
    assert result["n_tasks"] == 3
    assert np.isclose(result["correlation"]["spearman"], 1.0)
    assert result["by_frozen_directspline_outcome"]["loss"]["n_tasks"] == 1
