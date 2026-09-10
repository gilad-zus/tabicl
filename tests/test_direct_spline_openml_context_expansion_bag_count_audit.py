from __future__ import annotations

import importlib.util
import json
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
if "direct_spline_openml_crossfit_context_expansion" not in sys.modules:
    _load_module("direct_spline_openml_crossfit_context_expansion", "direct_spline_openml_crossfit_context_expansion.py")
audit = _load_module(
    "direct_spline_openml_context_expansion_bag_count_audit",
    "direct_spline_openml_context_expansion_bag_count_audit.py",
)
context = sys.modules["direct_spline_openml_crossfit_context_expansion"]


def test_load_archived_task_summaries_uses_root_level_run_file(tmp_path):
    (tmp_path / "task_summaries.json").write_text(
        json.dumps(
            [
                {"task_id": 7, "effective_bags": 8, "expanded_context": {}},
                {"task_id": 11, "effective_bags": 8, "expanded_context": {}},
            ]
        ),
        encoding="utf-8",
    )

    summaries = audit._load_archived_task_summaries(tmp_path)

    assert set(summaries) == {7, 11}
    assert summaries[11]["effective_bags"] == 8


def _bag(*, index: int, identity_oof: float, spline_oof: float, identity_test: float, spline_a: float, spline_b: float):
    return context.ContextExpansionBagPredictions(
        validation_indices=np.asarray([2 * index, 2 * index + 1]),
        selection_a_indices=np.asarray([2 * index]),
        selection_b_indices=np.asarray([2 * index + 1]),
        original_identity_selection_a=np.asarray([0.0]),
        original_identity_selection_b=np.asarray([0.0]),
        original_spline_selected_on_b_selection_a=np.asarray([0.0]),
        original_spline_selected_on_a_selection_b=np.asarray([0.0]),
        original_identity_test=np.asarray([0.0]),
        original_spline_selected_on_a_test=np.asarray([0.0]),
        original_spline_selected_on_b_test=np.asarray([0.0]),
        expanded_identity_selection_a=np.asarray([identity_oof]),
        expanded_identity_selection_b=np.asarray([identity_oof]),
        expanded_spline_selected_on_b_selection_a=np.asarray([spline_oof]),
        expanded_spline_selected_on_a_selection_b=np.asarray([spline_oof]),
        expanded_identity_test=np.asarray([identity_test]),
        expanded_spline_selected_on_a_test=np.asarray([spline_a]),
        expanded_spline_selected_on_b_test=np.asarray([spline_b]),
        metadata={},
    )


def test_subset_uses_only_its_held_out_rows_and_averages_its_two_checkpoint_states():
    task = SimpleNamespace(
        problem_type="regression",
        n_classes=None,
        y_train=np.asarray([0.0, 0.0, 1.0, 1.0]),
        y_test=np.asarray([1.0]),
    )
    first = _bag(index=0, identity_oof=1.0, spline_oof=0.0, identity_test=3.0, spline_a=1.0, spline_b=3.0)
    second = _bag(index=1, identity_oof=0.0, spline_oof=1.0, identity_test=5.0, spline_a=5.0, spline_b=7.0)

    labels, identity_oof, spline_oof, identity_test, spline_test = audit._prediction_and_oof_for_subset(
        task=task, bags=[second]
    )

    assert np.array_equal(labels, np.asarray([1.0, 1.0]))
    assert np.array_equal(identity_oof, np.asarray([0.0, 0.0]))
    assert np.array_equal(spline_oof, np.asarray([1.0, 1.0]))
    assert np.array_equal(identity_test, np.asarray([5.0]))
    assert np.array_equal(spline_test, np.asarray([6.0]))

    selection = audit._select_alpha(
        task=task,
        labels=labels,
        identity_prediction=identity_oof,
        spline_prediction=spline_oof,
    )
    assert selection["selected_alpha"] == 1.0

    record = audit._score_subset(task=task, bags=[first, second], fixed_alpha=0.5, subset=(0, 1))
    assert record["policies"]["fixed_eight_bag_alpha"]["alpha"] == 0.5
    assert record["policies"]["subset_oof_alpha"]["alpha"] == 1.0
    assert record["policies"]["fixed_eight_bag_alpha"]["metrics"]["benchmark_error"] == 3.0


def test_subset_summary_keeps_policy_and_identity_comparisons_separate():
    subsets = [
        {
            "identity": {"benchmark_error": 2.0},
            "policies": {
                "fixed_eight_bag_alpha": {"alpha": 1.0, "metrics": {"benchmark_error": 1.0}},
                "subset_oof_alpha": {"alpha": 0.0, "metrics": {"benchmark_error": 2.0}},
            },
        },
        {
            "identity": {"benchmark_error": 2.0},
            "policies": {
                "fixed_eight_bag_alpha": {"alpha": 1.0, "metrics": {"benchmark_error": 3.0}},
                "subset_oof_alpha": {"alpha": 0.5, "metrics": {"benchmark_error": 1.5}},
            },
        },
    ]

    result = audit._summarize_subsets(
        subset_records=subsets,
        policy="subset_oof_alpha",
        full_reference={"benchmark_error": 1.5},
    )

    assert result["n_subsets"] == 2
    assert result["alpha_counts"] == {"0.0": 1, "0.25": 0, "0.5": 1, "0.75": 0, "1.0": 0}
    assert result["subset_outcomes_vs_identity"] == {"win": 1, "tie": 1, "loss": 0}
    assert result["mean_subset_outcomes_vs_full_tabiclv2"] == {"win": 0, "tie": 1, "loss": 1}
