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
if "direct_spline_openml_crossfit_blend" not in sys.modules:
    _load_module("direct_spline_openml_crossfit_blend", "direct_spline_openml_crossfit_blend.py")
audit = _load_module(
    "direct_spline_openml_crossfit_blend_ensemble_audit",
    "direct_spline_openml_crossfit_blend_ensemble_audit.py",
)


def test_ensemble_can_improve_a_pair_of_individually_worse_regression_predictions():
    labels = np.asarray([0.0])
    identity_metrics = [
        {"deployment_error": 1.0, "benchmark_error": 1.0},
        {"deployment_error": 1.0, "benchmark_error": 1.0},
    ]
    spline_metrics = [
        {"deployment_error": 4.0, "benchmark_error": 2.0},
        {"deployment_error": 0.01, "benchmark_error": 0.1},
    ]

    identity_member = audit._mean_metrics(identity_metrics)
    spline_member = audit._mean_metrics(spline_metrics)
    assert audit._outcome(identity_member["deployment_error"], spline_member["deployment_error"]) == "loss"

    identity_ensemble = np.asarray([1.0])
    spline_ensemble = np.asarray([0.95])
    assert np.square(spline_ensemble - labels).mean() < np.square(identity_ensemble - labels).mean()
