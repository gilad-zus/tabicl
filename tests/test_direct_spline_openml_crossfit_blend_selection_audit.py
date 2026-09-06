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
if "direct_spline_openml_crossfit_blend" not in sys.modules:
    _load_module("direct_spline_openml_crossfit_blend", "direct_spline_openml_crossfit_blend.py")
audit = _load_module(
    "direct_spline_openml_crossfit_blend_selection_audit",
    "direct_spline_openml_crossfit_blend_selection_audit.py",
)


def test_pooled_oof_loss_does_not_overweight_a_tiny_relative_loss_unit():
    task = SimpleNamespace(
        problem_type="regression",
        n_classes=None,
        y_train=np.zeros(101),
    )
    identity = np.concatenate((np.asarray([0.1]), np.ones(100)))
    spline = np.concatenate((np.asarray([np.sqrt(0.02)]), np.full(100, np.sqrt(0.8))))
    units = [
        (task.y_train[:1], identity[:1], spline[:1]),
        (task.y_train[1:], identity[1:], spline[1:]),
    ]

    relative = audit._argmin_record(
        audit._alpha_selection(task=task, units=units, alphas=(0.0, 1.0))["candidate_alphas"],
        key="mean_relative_error_change",
    )
    pooled = audit._pooled_oof_selection(task=task, identity_oof=identity, spline_oof=spline, alphas=(0.0, 1.0))

    assert relative["alpha"] == 0.0
    assert pooled["selected_alpha"] == 1.0
