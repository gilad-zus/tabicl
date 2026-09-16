from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_experiment():
    path = Path(__file__).parents[1] / "scripts" / "direct_spline_openml_arctan_ablation.py"
    spec = importlib.util.spec_from_file_location("direct_spline_openml_arctan_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def _method(error: float) -> dict[str, float]:
    return {"benchmark_error": error, "deployment_error": error**2}


def _record(task_id: int, *, identity: float, raw: float, selected: float) -> dict:
    return {
        "task_id": task_id,
        "dataset_name": f"task-{task_id}",
        "problem_type": "regression",
        "expanded_context": {
            "oof": {
                "identity": _method(identity + 0.1),
                "raw_spline": _method(raw + 0.1),
                "selected_blend": _method(selected + 0.1),
            },
            "outer_test": {
                "identity": _method(identity),
                "raw_spline": _method(raw),
                "selected_blend": _method(selected),
            },
        },
    }


def test_aggregate_reports_paired_incremental_spline_effect(tmp_path):
    control_dir = tmp_path / "arctan_no_spline"
    spline_dir = tmp_path / "arctan_spline"
    control_dir.mkdir()
    spline_dir.mkdir()
    (control_dir / "task_summaries.json").write_text(
        json.dumps([
            _record(1, identity=0.5, raw=0.4, selected=0.4),
            _record(2, identity=0.7, raw=0.5, selected=0.45),
        ]),
        encoding="utf-8",
    )
    (spline_dir / "task_summaries.json").write_text(
        json.dumps([
            _record(1, identity=0.5, raw=0.3, selected=0.35),
            _record(2, identity=0.7, raw=0.6, selected=0.45),
        ]),
        encoding="utf-8",
    )

    result = experiment._aggregate(SimpleNamespace(output_dir=tmp_path, task_id=[1, 2]))

    assert result["outer_test_spline_vs_no_spline"]["raw_spline"] == pytest.approx(
        {
            "spline_wins": 1,
            "ties": 0,
            "spline_losses": 1,
            "mean_relative_spline_gain": 0.025,
        }
    )
    assert result["outer_test_spline_vs_no_spline"]["selected_blend"]["spline_wins"] == 1
    assert result["outer_test_spline_vs_no_spline"]["selected_blend"]["ties"] == 1


def test_aggregate_rejects_unmatched_identity_baselines(tmp_path):
    for label, identity in (("arctan_no_spline", 0.5), ("arctan_spline", 0.6)):
        arm_dir = tmp_path / label
        arm_dir.mkdir()
        (arm_dir / "task_summaries.json").write_text(
            json.dumps([_record(1, identity=identity, raw=0.4, selected=0.4)]),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="identity baseline differs"):
        experiment._aggregate(SimpleNamespace(output_dir=tmp_path, task_id=[1]))
