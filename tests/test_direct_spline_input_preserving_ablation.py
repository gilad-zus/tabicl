from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tabicl._experiments.direct_spline_openml_standard import _make_adapters
from tabicl._hyperspline import DirectSplineTransform


def _load_experiment():
    path = Path(__file__).parents[1] / "scripts" / "direct_spline_openml_input_preserving_ablation.py"
    spec = importlib.util.spec_from_file_location("direct_spline_openml_input_preserving_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def _method(error: float) -> dict[str, float]:
    return {"benchmark_error": error, "deployment_error": error**2}


def test_native_input_residual_starts_exactly_at_tabicl_coordinates():
    context = torch.zeros(1, 1, 2)
    line = DirectSplineTransform(
        context,
        n_control_points=8,
        trainable_shape=False,
        trainable_location_scale=False,
        coordinate_mapping="arctan",
        direct_spline_output=True,
        preserve_input_base=True,
    )
    spline = DirectSplineTransform(
        context,
        n_control_points=8,
        trainable_shape=True,
        trainable_location_scale=False,
        coordinate_mapping="arctan",
        direct_spline_output=True,
        preserve_input_base=True,
    )
    probe = torch.tensor([[[-100.0, -4.0], [0.0, 0.0], [4.0, 100.0]]])
    with torch.no_grad():
        for adapter in (line, spline):
            adapter.location.zero_()
            adapter.scale.fill_(1.0)

    assert torch.equal(line.transform(probe), probe)
    assert torch.equal(spline.transform(probe), probe)
    assert not line.gap_logits.requires_grad
    assert spline.gap_logits.requires_grad

    with torch.no_grad():
        spline.gap_logits[..., 2].fill_(0.5)
    changed = spline.transform(probe)
    assert not torch.allclose(changed, probe)
    changed.square().mean().backward()
    assert spline.gap_logits.grad is not None
    assert spline.direct_center.grad is not None


def test_native_input_residual_is_exact_for_standard_probe_and_random_values():
    n_columns = 7
    probe = torch.linspace(-5.0, 5.0, 17).view(1, 17, 1).expand(-1, -1, n_columns)
    values = torch.cat((probe, torch.randn(1, 4096, n_columns) * 7.0), dim=1)
    for trainable_shape in (False, True):
        adapter = DirectSplineTransform(
            torch.zeros(1, 1, n_columns),
            n_control_points=20,
            trainable_shape=trainable_shape,
            trainable_location_scale=False,
            coordinate_mapping="arctan",
            direct_spline_output=True,
            preserve_input_base=True,
            cross_column_mixing_rank=4,
        )
        with torch.no_grad():
            adapter.location.zero_()
            adapter.scale.fill_(1.0)
            assert torch.equal(adapter.transform(values), values)


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_standard_adapter_accepts_input_preserving_initialization(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bundle = SimpleNamespace(
        numerical_indices=np.arange(7),
        estimator=SimpleNamespace(
            ensemble_generator_=SimpleNamespace(preprocessors_=["normal"])
        ),
    )
    config = {
        "n_control_points": 20,
        "trainable_shape": True,
        "trainable_location_scale": False,
        "coordinate_mapping": "arctan",
        "direct_spline_output": True,
        "preserve_input_base": True,
        "cross_column_mixing_rank": 4,
        "cross_column_mixing_bound": 0.1,
    }
    assert _make_adapters(bundle, config, torch.device(device_name)) is not None


def test_native_input_residual_rejects_non_arctan_direct_output():
    with pytest.raises(ValueError, match="direct spline output requires arctan"):
        DirectSplineTransform(
            torch.zeros(1, 1, 1),
            coordinate_mapping="linear",
            direct_spline_output=True,
            preserve_input_base=True,
        )


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


def _write_arm(root: Path, label: str, records: list[dict]) -> None:
    directory = root / label
    directory.mkdir(parents=True)
    (directory / "task_summaries.json").write_text(json.dumps(records), encoding="utf-8")


def test_aggregate_separates_base_initialization_from_curvature(tmp_path):
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    _write_arm(
        reference,
        "direct_arctan_line",
        [
            _record(1, identity=0.50, raw=0.40, selected=0.40),
            _record(2, identity=0.70, raw=0.60, selected=0.60),
        ],
    )
    _write_arm(
        reference,
        "direct_arctan_spline",
        [
            _record(1, identity=0.50, raw=0.35, selected=0.35),
            _record(2, identity=0.70, raw=0.65, selected=0.60),
        ],
    )
    _write_arm(
        output,
        "preserved_line",
        [
            _record(1, identity=0.50, raw=0.38, selected=0.38),
            _record(2, identity=0.70, raw=0.55, selected=0.55),
        ],
    )
    _write_arm(
        output,
        "preserved_spline",
        [
            _record(1, identity=0.50, raw=0.30, selected=0.32),
            _record(2, identity=0.70, raw=0.58, selected=0.55),
        ],
    )

    result = experiment._aggregate(
        SimpleNamespace(reference_dir=reference, output_dir=output, task_id=[1, 2])
    )

    contrasts = result["outer_test_paired_contrasts"]
    curvature = contrasts["input_preserving_curvature"]["raw_spline"]
    assert curvature["reference_arm"] == "preserved_line"
    assert curvature["candidate_arm"] == "preserved_spline"
    assert curvature["candidate_wins"] == 1
    assert curvature["ties"] == 0
    assert curvature["candidate_losses"] == 1
    assert curvature["mean_relative_candidate_gain"] == pytest.approx(0.07799043062200956)
    assert curvature["median_relative_candidate_gain"] == pytest.approx(0.07799043062200956)
    assert curvature["n_relative_gain_tasks"] == 2
    assert contrasts["input_base_effect_without_curvature"]["raw_spline"]["candidate_wins"] == 2
    assert contrasts["compressed_curvature"]["selected_blend"]["candidate_wins"] == 1


def test_aggregate_rejects_identity_baseline_drift(tmp_path):
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    for root, label, identity in (
        (reference, "direct_arctan_line", 0.5),
        (reference, "direct_arctan_spline", 0.5),
        (output, "preserved_line", 0.5),
        (output, "preserved_spline", 0.6),
    ):
        _write_arm(root, label, [_record(1, identity=identity, raw=0.4, selected=0.4)])

    with pytest.raises(ValueError, match="identity baseline differs"):
        experiment._aggregate(
            SimpleNamespace(reference_dir=reference, output_dir=output, task_id=[1])
        )
