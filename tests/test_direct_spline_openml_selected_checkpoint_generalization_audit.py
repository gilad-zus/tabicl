from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, filename: str):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if "direct_spline_openml_support_audit" not in sys.modules:
    _load_module("direct_spline_openml_support_audit", "direct_spline_openml_support_audit.py")
if "direct_spline_openml_crossfit_blend" not in sys.modules:
    _load_module("direct_spline_openml_crossfit_blend", "direct_spline_openml_crossfit_blend.py")
experiment = _load_module(
    "selected_checkpoint_generalization_audit",
    "direct_spline_openml_selected_checkpoint_generalization_audit.py",
)


def test_aggregate_identifies_train_win_test_loss():
    rows = [
        {
            "dataset_name": "overfit",
            "training_relative_curvature_gain": 0.2,
            "oof_raw_relative_curvature_gain": -0.1,
            "outer_test_raw_relative_curvature_gain": -0.05,
            "outer_test_selected_relative_curvature_gain": 0.0,
        },
        {
            "dataset_name": "generalizes",
            "training_relative_curvature_gain": 0.1,
            "oof_raw_relative_curvature_gain": 0.02,
            "outer_test_raw_relative_curvature_gain": 0.03,
            "outer_test_selected_relative_curvature_gain": 0.01,
        },
    ]
    result = experiment._aggregate(rows)
    assert result["train_win_test_loss_datasets"] == ["overfit"]
    assert result["comparisons"]["training_relative_curvature_gain"]["wins"] == 2
    assert result["comparisons"]["outer_test_raw_relative_curvature_gain"]["losses"] == 1
