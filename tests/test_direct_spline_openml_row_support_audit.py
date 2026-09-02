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


_load_module("direct_spline_openml_support_audit", "direct_spline_openml_support_audit.py")
audit = _load_module("direct_spline_openml_row_support_audit", "direct_spline_openml_row_support_audit.py")


def test_effective_basis_support_is_larger_in_a_dense_context_region():
    knots = audit._uniform_knots(n_control_points=20, degree=3)
    context = np.concatenate((np.linspace(-0.3, 0.3, 101), np.array([-3.9, 3.9])))
    support = audit._basis_mass(
        context=context,
        query=np.array([0.0, 3.0]),
        knots=knots,
        degree=3,
        n_control_points=20,
        standardized_range=4.0,
    )
    assert support[0] > support[1]
    assert np.all(support >= 0.0)


def test_per_row_loss_matches_multiclass_log_loss_and_regression_mse():
    classification = audit._per_row_loss(
        problem_type="multiclass",
        labels=np.array([0, 1]),
        prediction=np.array([[0.8, 0.2], [0.25, 0.75]]),
        n_classes=2,
    )
    assert np.allclose(classification, -np.log(np.array([0.8, 0.75])))
    regression = audit._per_row_loss(
        problem_type="regression",
        labels=np.array([2.0, -1.0]),
        prediction=np.array([1.0, 2.0]),
        n_classes=None,
    )
    assert np.allclose(regression, np.array([1.0, 9.0]))


def test_rank_strata_are_balanced_and_ordered_by_risk():
    strata = audit._rank_strata(np.array([4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0]), n_strata=4)
    assert np.bincount(strata).tolist() == [2, 2, 2, 2]
    assert strata[1] == 0
    assert strata[-1] == 3


def test_cell_flip_summary_requires_adequate_rows_and_detects_validation_help_test_harm():
    base = {
        "task_id": 1,
        "dataset_name": "synthetic",
        "problem_type": "regression",
        "normalization_method": "none",
        "numerical_feature_index": 0,
        "numerical_feature_name": "x",
        "knot_interval": 3,
        "context_interval_count": 2,
        "n_query_rows": 5,
    }
    records = [
        {**base, "split": "validation", "mean_adapted_minus_identity_row_loss": -0.4},
        {**base, "split": "test", "mean_adapted_minus_identity_row_loss": 0.3},
    ]
    summary = audit._cell_flip_summary(cell_records=records, minimum_query_rows=5)
    assert summary[0]["eligible_validation_test_cells"] == 1
    assert summary[0]["validation_help_test_harm_cells"] == 1


def test_source_task_replay_uses_the_frozen_outer_split(monkeypatch):
    observed = {}

    def fake_loader(task_id, *, outer_repeat, outer_fold, outer_sample):
        observed.update(
            task_id=task_id,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            outer_sample=outer_sample,
        )
        return "task"

    monkeypatch.setattr(audit, "load_tabarena_openml_task", fake_loader)
    result = audit._load_source_task(
        case=SimpleNamespace(task_id=123),
        immutable_run={"data_source": {"outer_split": {"repeat": 2, "fold": 3, "sample": 4}}},
    )

    assert result == "task"
    assert observed == {"task_id": 123, "outer_repeat": 2, "outer_fold": 3, "outer_sample": 4}
