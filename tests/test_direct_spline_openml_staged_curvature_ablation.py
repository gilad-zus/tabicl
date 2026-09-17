from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "direct_spline_openml_staged_curvature_ablation.py"
    spec = importlib.util.spec_from_file_location("staged_curvature_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_module()


def _task(task_id: int, raw: float, selected: float) -> dict:
    metrics = {
        "oof": {
            "identity": {"benchmark_error": 1.0},
            "raw_spline": {"benchmark_error": raw},
            "selected_blend": {"benchmark_error": selected},
        },
        "outer_test": {
            "identity": {"benchmark_error": 1.0},
            "raw_spline": {"benchmark_error": raw},
            "selected_blend": {"benchmark_error": selected},
        },
    }
    return {
        "task_id": task_id,
        "dataset_name": f"task-{task_id}",
        "problem_type": "multiclass",
        "expanded_context": metrics,
    }


def test_aggregate_reports_only_paired_continuation_curvature_gain(tmp_path):
    for arm, raw, selected in (
        ("continued_line", 0.8, 0.9),
        ("continued_spline", 0.6, 0.9),
    ):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / "task_summaries.json").write_text(
            json.dumps([_task(7, raw, selected)]), encoding="utf-8"
        )
    result = experiment._aggregate(SimpleNamespace(output_dir=tmp_path, task_id=[7]))
    raw = result["outer_test_spline_vs_line_continuation"]["raw_spline"]
    assert (raw["spline_wins"], raw["ties"], raw["spline_losses"]) == (1, 0, 0)
    assert raw["mean_relative_curvature_gain"] == pytest.approx(0.25)
    assert result["outer_test_spline_vs_line_continuation"]["selected_blend"]["ties"] == 1


def test_manifest_binds_exact_initial_line_artifact(tmp_path):
    source = tmp_path / "source"
    initial = tmp_path / "line"
    output = tmp_path / "output"
    source.mkdir()
    initial.mkdir()
    (initial / "experiment_manifest.json").write_text('{"line": true}\n', encoding="utf-8")
    args = SimpleNamespace(
        source_dir=source,
        initial_line_dir=initial,
        output_dir=output,
        task_id=[2, 1],
        config_label="D",
        protocol_seed=11,
        bags=4,
        continuation_steps=250,
        query_fraction_min=0.05,
        query_fraction_max=0.2,
        resume=False,
    )
    experiment._prepare(args)
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_ids"] == [1, 2]
    assert manifest["initial_line_dir"] == str(initial.resolve())
    assert manifest["continuation_steps"] == 250
