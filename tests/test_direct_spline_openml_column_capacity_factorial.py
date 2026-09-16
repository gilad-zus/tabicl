from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_experiment():
    path = Path(__file__).parents[1] / "scripts" / "direct_spline_openml_column_capacity_factorial.py"
    spec = importlib.util.spec_from_file_location("direct_spline_openml_column_capacity_factorial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def _method(error: float) -> dict[str, float]:
    return {"benchmark_error": error, "deployment_error": error**2}


def test_aggregate_selects_capacity_from_oof_not_outer_test(tmp_path):
    for index, assignment in enumerate(experiment.ASSIGNMENTS):
        label = experiment._label(assignment)
        condition_dir = tmp_path / label
        condition_dir.mkdir()
        is_target = assignment == (4, 20, 4)
        oof = 0.1 if is_target else 0.2 + index / 100
        # Deliberately make the OOF-selected mixed assignment worse on test;
        # the aggregator must not silently switch to the outer-test oracle.
        outer = 0.4 if is_target else 0.3 + index / 100
        record = {
            "task_id": 4999,
            "dataset_name": "visualizing_soil",
            "expanded_context": {
                "blend_selection": {"selected_alpha": 0.5},
                "oof": {
                    "raw_spline": _method(oof),
                    "selected_blend": _method(oof),
                },
                "outer_test": {
                    "identity": _method(0.5),
                    "raw_spline": _method(outer),
                    "selected_blend": _method(outer),
                },
            },
            "source_full_outer_training_tabiclv2": _method(0.6),
        }
        (condition_dir / "task_summaries.json").write_text(
            json.dumps([record]), encoding="utf-8"
        )

    result = experiment._aggregate(SimpleNamespace(output_dir=tmp_path, task_id=4999))

    assert result["oof_selected_raw_capacity"]["label"] == "k4_20_4"
    assert result["oof_selected_end_to_end_capacity"]["label"] == "k4_20_4"
    assert result["predeclared_heterogeneity_result"] == {
        "selected_assignment_is_mixed": True,
        "selected_outer_error": 0.4,
        "beats_both_uniforms_on_outer_test": False,
    }
