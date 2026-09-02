from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _load_audit_module():
    path = Path(__file__).parents[1] / "scripts" / "direct_spline_openml_support_audit.py"
    spec = importlib.util.spec_from_file_location("direct_spline_openml_support_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


def test_support_metrics_separate_tail_sparse_and_empirical_outside():
    edges = np.linspace(-4.0, 4.0, 18)
    context = np.array([-3.9, -3.8, -0.1, 0.0, 0.1, 3.8, 3.9])
    query = np.array([-4.5, -3.85, 0.0, 2.5, 4.5])

    result = audit._support_split_metrics(
        context=context,
        query=query,
        interval_edges=edges,
        standardized_range=4.0,
        boundary_inner=3.5,
        boundary_outer=4.5,
        sparse_min_count=2,
        sparse_min_fraction=0.0 + 0.001,
    )

    assert result["tail_rate"] == 2 / 5
    assert result["boundary_rate"] == 3 / 5
    assert result["outside_context_range_rate"] == 2 / 5
    # The 2.5 coordinate lands in an empty interior polynomial interval.
    assert result["sparse_interval_rate"] >= 1 / 5
    assert result["unsupported_rate"] >= result["outside_context_range_rate"]
    assert result["sparse_interval_count_threshold"] == 2


def test_preprocessing_replay_uses_numeric_columns_and_both_actual_views():
    fit = pd.DataFrame(
        {
            "numeric": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "constant": [7.0] * 6,
            "category": ["a", "a", "b", "b", "a", "b"],
        }
    )
    validation = pd.DataFrame({"numeric": [-3.0, 0.5], "constant": [7.0, 7.0], "category": ["a", "b"]})
    test = pd.DataFrame({"numeric": [4.0, 10.0], "constant": [7.0, 7.0], "category": ["b", "missing"]})
    config = {
        "n_estimators": 8,
        "norm_methods": ["none", "power"],
        "feat_shuffle_method": "latin",
        "class_shuffle_method": "shift",
        "outlier_threshold": 4.0,
        "random_state": 0,
    }

    views, names = audit._fit_preprocessing_views(
        fit_x=fit,
        validation_x=validation,
        test_x=test,
        fit_labels=np.array([0, 1, 0, 1, 0, 1]),
        problem_type="binary",
        normal_config=config,
    )

    assert set(views) == {"none", "power"}
    assert names == ["numeric"]
    for context, held_out, outer_test in views.values():
        assert context.shape == (6, 1)
        assert held_out.shape == (2, 1)
        assert outer_test.shape == (2, 1)
        assert np.isfinite(context).all()


def test_task_level_association_has_bootstrap_quartiles_and_leave_one_out():
    rows = []
    for index, (support_shift, regret) in enumerate(
        [(-0.2, -0.04), (-0.1, -0.02), (0.0, 0.0), (0.2, 0.03), (0.5, 0.08)], start=1
    ):
        rows.append(
            {
                "task_id": index,
                "dataset_name": f"task_{index}",
                "problem_type": "multiclass" if index % 2 else "regression",
                "support_available": True,
                "test_minus_validation_unsupported_rate": support_shift,
                "test_adapted_regret": regret,
            }
        )

    summary = audit._support_association_summary(rows, bootstrap_rounds=50, bootstrap_seed=12)

    assert summary["n_tasks_primary_analysis"] == 5
    assert np.isclose(summary["correlation"]["spearman"], 1.0)
    assert summary["task_bootstrap"]["spearman_95"] is not None
    assert len(summary["rank_balanced_quartiles"]) == 4
    assert len(summary["leave_one_task_out"]) == 5


def test_source_config_allows_only_documented_patience_resume_migration():
    expected = {"adapter_steps": 500, "adapter_patience": None, "n_control_points": 20}
    resumed = {"adapter_steps": 500, "adapter_patience": 12, "n_control_points": 20}
    manifest = {"resume_protocol_migrations": [{"reason": "Retouche efficiency resume"}]}

    allowed, differences = audit._source_config_matches_manifest(
        source_config=resumed, expected_config=expected, manifest=manifest
    )
    assert allowed
    assert differences == ["adapter_patience"]

    changed_spline = {**resumed, "n_control_points": 12}
    allowed, differences = audit._source_config_matches_manifest(
        source_config=changed_spline, expected_config=expected, manifest=manifest
    )
    assert not allowed
    assert differences == ["adapter_patience", "n_control_points"]
