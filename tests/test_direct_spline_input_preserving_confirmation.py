from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_experiment():
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "direct_spline_openml_input_preserving_confirmation.py"
    spec = importlib.util.spec_from_file_location("direct_spline_openml_input_preserving_confirmation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def _record(task_id: int, raw: float, selected: float, alpha: float) -> dict:
    def metrics(value: float) -> dict:
        return {"benchmark_error": value}

    return {
        "task_id": task_id,
        "dataset_id": task_id + 100,
        "dataset_name": f"task-{task_id}",
        "problem_type": "multiclass",
        "outer_split_hash": f"split-{task_id}",
        "source_full_outer_training_tabiclv2": metrics(0.45),
        "expanded_context": {
            "oof": {"identity": metrics(0.50), "raw_spline": metrics(raw + 0.1), "selected_blend": metrics(selected + 0.1)},
            "outer_test": {"identity": metrics(0.40), "raw_spline": metrics(raw), "selected_blend": metrics(selected)},
            "blend_selection": {"selected_alpha": alpha},
        },
    }


def _write_arm(root: Path, name: str, records: list[dict]) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "task_summaries.json").write_text(json.dumps(records), encoding="utf-8")


def test_confirmation_reports_curvature_and_full_baseline_separately(tmp_path):
    _write_arm(tmp_path, "preserved_line", [_record(1, 0.35, 0.35, 1.0), _record(2, 0.30, 0.31, 0.75)])
    _write_arm(tmp_path, "preserved_spline", [_record(1, 0.25, 0.40, 0.0), _record(2, 0.33, 0.32, 0.50)])

    summary = experiment._aggregate(SimpleNamespace(output_dir=tmp_path, task_id=[1, 2]))

    assert summary["comparisons"]["raw_curvature_vs_line"]["wins"] == 1
    assert summary["comparisons"]["raw_curvature_vs_line"]["losses"] == 1
    assert summary["comparisons"]["raw_spline_vs_full_tabiclv2"]["wins"] == 2
    assert summary["task_results"][0]["selected_alpha_spline"] == 0.0
    assert (tmp_path / "confirmation_results.csv").is_file()


def test_confirmation_rejects_changed_identity_baseline(tmp_path):
    line = _record(1, 0.35, 0.35, 1.0)
    spline = _record(1, 0.25, 0.25, 1.0)
    spline["expanded_context"]["outer_test"]["identity"]["benchmark_error"] = 0.41
    _write_arm(tmp_path, "preserved_line", [line])
    _write_arm(tmp_path, "preserved_spline", [spline])

    with pytest.raises(ValueError, match="identity error differs"):
        experiment._aggregate(SimpleNamespace(output_dir=tmp_path, task_id=[1]))


def test_confirmation_compares_cosine_and_saved_constant_lr_on_same_tasks(tmp_path):
    _write_arm(tmp_path, "preserved_line", [_record(1, 0.30, 0.30, 1.0)])
    _write_arm(tmp_path, "preserved_spline", [_record(1, 0.25, 0.25, 1.0)])
    reference = tmp_path / "constant"
    reference.mkdir()
    (reference / "confirmation_summary.json").write_text(json.dumps({
        "task_results": [{
            "task_id": 1,
            "dataset_name": "task-1",
            "outer_test_full_tabiclv2": 0.45,
            "outer_test_raw_spline_line": 0.35,
            "outer_test_raw_spline_spline": 0.40,
            "outer_test_selected_blend_line": 0.35,
            "outer_test_selected_blend_spline": 0.40,
        }],
    }), encoding="utf-8")

    summary = experiment._aggregate(SimpleNamespace(
        output_dir=tmp_path, reference_dir=reference, task_id=[1]
    ))

    assert summary["comparisons"]["cosine_vs_constant_line"]["wins"] == 1
    assert summary["comparisons"]["cosine_vs_constant_spline"]["wins"] == 1
    assert summary["task_results"][0]["outer_test_raw_spline_spline_cosine_gain"] == pytest.approx(0.375)
