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
    "direct_spline_openml_crossfit_repeat_selection_audit",
    "direct_spline_openml_crossfit_repeat_selection_audit.py",
)


def test_prediction_averaging_can_choose_a_spline_when_mean_member_loss_rejects_it():
    task = SimpleNamespace(problem_type="regression", n_classes=None, y_train=np.asarray([0.0]))
    selection = audit._two_repeat_selections(
        task=task,
        first_identity_oof=np.asarray([1.0]),
        first_spline_oof=np.asarray([4.0]),
        second_identity_oof=np.asarray([1.0]),
        second_spline_oof=np.asarray([-4.0]),
    )

    assert selection["mean_member_pooled_oof_loss"]["selected_alpha"] == 0.0
    assert selection["mean_prediction_pooled_oof_loss"]["selected_alpha"] == 1.0
