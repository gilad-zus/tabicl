from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


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
experiment = _load_module(
    "direct_spline_openml_crossfit_context_expansion",
    "direct_spline_openml_crossfit_context_expansion.py",
)


def test_cosine_override_changes_only_learning_rate_schedule_in_training_config():
    source = {
        "learning_rate": 0.005,
        "validation_interval": 10,
        "random_state": 0,
        "adapter_steps": 500,
        "trainable_location_scale": True,
    }
    args = SimpleNamespace(
        adapter_arm="direct_spline",
        coordinate_mapping="arctan",
        preserve_input_base=True,
        n_control_points=20,
        cosine_min_lr_ratio=0.01,
        adapter_steps=500,
        query_fraction_min=0.05,
        query_fraction_max=0.20,
        column_control_points=None,
    )

    configured = experiment._updated_adapter_config(source, args)

    assert configured["cosine_schedule_steps"] == 500
    assert configured["cosine_min_lr_ratio"] == 0.01
    assert configured["learning_rate"] == 0.005
    assert configured["validation_interval"] == 10
    assert configured["random_state"] == 0
    assert configured["trainable_shape"] is True
    assert configured["direct_spline_output"] is True
    assert configured["trainable_location_scale"] is False
    assert "cosine_schedule_steps" not in source


def _bag() -> object:
    return experiment.ContextExpansionBagPredictions(
        validation_indices=np.asarray([0, 1]),
        selection_a_indices=np.asarray([0]),
        selection_b_indices=np.asarray([1]),
        original_identity_selection_a=np.asarray([10.0]),
        original_identity_selection_b=np.asarray([11.0]),
        # The A row must receive the B-selected state (20), and vice versa.
        original_spline_selected_on_b_selection_a=np.asarray([20.0]),
        original_spline_selected_on_a_selection_b=np.asarray([21.0]),
        original_identity_test=np.asarray([1.0]),
        original_spline_selected_on_a_test=np.asarray([3.0]),
        original_spline_selected_on_b_test=np.asarray([5.0]),
        expanded_identity_selection_a=np.asarray([30.0]),
        expanded_identity_selection_b=np.asarray([31.0]),
        expanded_spline_selected_on_b_selection_a=np.asarray([40.0]),
        expanded_spline_selected_on_a_selection_b=np.asarray([41.0]),
        expanded_identity_test=np.asarray([7.0]),
        expanded_spline_selected_on_a_test=np.asarray([9.0]),
        expanded_spline_selected_on_b_test=np.asarray([11.0]),
        metadata={},
    )


def test_assembly_keeps_opposite_half_checkpoint_and_separates_context_arms():
    task = SimpleNamespace(y_train=np.asarray([0.0, 1.0]), problem_type="regression", n_classes=None)

    original = experiment._assemble(task=task, bags=[_bag()], condition="original")
    expanded = experiment._assemble(task=task, bags=[_bag()], condition="expanded")

    assert np.array_equal(original[0], np.asarray([10.0, 11.0]))
    assert np.array_equal(original[1], np.asarray([20.0, 21.0]))
    assert np.array_equal(original[2], np.asarray([1.0]))
    assert np.array_equal(original[3], np.asarray([4.0]))
    assert np.array_equal(expanded[0], np.asarray([30.0, 31.0]))
    assert np.array_equal(expanded[1], np.asarray([40.0, 41.0]))
    assert np.array_equal(expanded[2], np.asarray([7.0]))
    assert np.array_equal(expanded[3], np.asarray([10.0]))


def test_minimum_pooled_oof_blend_uses_only_crossfitted_oof_rows_and_tie_breaks_lower():
    task = SimpleNamespace(problem_type="regression", n_classes=None, y_train=np.asarray([0.0, 1.0]))
    selection = experiment._minimum_pooled_oof_selection(
        task=task,
        identity_oof=np.asarray([0.0, 0.0]),
        spline_oof=np.asarray([0.0, 2.0]),
    )

    # alpha=.5 predicts the targets exactly. It is selected without any test
    # prediction or outer-test label being supplied to the function.
    assert selection["selected_alpha"] == 0.5


def test_selected_adapter_weights_roundtrip_without_backbone_or_optimizer():
    destination = io.BytesIO()
    best = {
        "original_a": {"step": 10, "error": 0.2, "valid": True, "state": {"shape": torch.tensor([1.0])}},
        "original_b": {"step": 30, "error": 0.1, "valid": True, "state": {"shape": torch.tensor([2.0])}},
    }
    experiment._save_selected_adapters(
        destination,
        best=best,
        config={"adapter_architecture": "fixed_cubic", "random_state": 123},
        provenance={"bag": 0, "split_artifact": "bag_0.npz", "outer_split_hash": "test-split"},
    )
    destination.seek(0)
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(saved) == {"adapter_checkpoint_schema_version", "config", "provenance", "checkpoints"}
    for name, record in best.items():
        selected = saved["checkpoints"][name]
        assert selected["step"] == record["step"]
        assert selected["valid"]
        assert torch.equal(selected["state_dict"]["shape"], record["state"]["shape"])
    assert saved["provenance"]["outer_split_hash"] == "test-split"


def test_ablation_manifest_records_arm_sampler_and_implementation_hashes(monkeypatch):
    source_dir = Path.cwd()
    monkeypatch.setattr(experiment, "_sha256", lambda _path: "hash")
    args = SimpleNamespace(
        config_label="D",
        protocol_seed=9,
        bags=4,
        reference_atol=1e-8,
        adapter_arm="affine_mixing",
        coordinate_mapping="arctan",
        query_fraction_min=0.05,
        query_fraction_max=0.2,
        column_control_points=(4, 20, 4),
    )
    case = SimpleNamespace(task_id=1)
    manifest = experiment._manifest(
        source_dir=source_dir,
        source_manifest={"immutable_run": {"repository_revision": "abc"}},
        reference_dir=None,
        reference_manifest=None,
        cases=[case],
        args=args,
    )
    assert manifest["adapter_arm"] == "affine_mixing"
    assert manifest["coordinate_mapping"] == "arctan"
    assert manifest["query_fraction_range"] == [0.05, 0.2]
    assert manifest["column_control_points"] == [4, 20, 4]
    assert set(manifest["implementation_sha256"]) == {
        "script",
        "standard_adapter",
        "episode_protocol",
        "adapter_module",
        "numerical_preparation",
    }


def test_reference_prediction_drift_is_diagnostic_but_indices_are_strict(tmp_path):
    reference = SimpleNamespace(
        validation_indices=np.asarray([0, 1]),
        selection_a_indices=np.asarray([0]),
        selection_b_indices=np.asarray([1]),
        identity_selection_a=np.asarray([0.0]),
        identity_selection_b=np.asarray([0.0]),
        spline_selected_on_b_selection_a=np.asarray([0.0]),
        spline_selected_on_a_selection_b=np.asarray([0.0]),
        identity_test=np.asarray([0.0]),
        spline_selected_on_a_test=np.asarray([0.0]),
        spline_selected_on_b_test=np.asarray([0.0]),
    )
    path = tmp_path / "bag.npz"
    path.touch()
    original_loader = experiment._load_reference_bag
    experiment._load_reference_bag = lambda _path: reference
    try:
        actual = _bag()
        diagnostic = experiment._verify_reference_bag(actual=actual, reference_path=path, atol=1e-8)
        assert diagnostic["within_atol"] is False
        assert diagnostic["max_abs"] == 21.0

        changed = experiment.ContextExpansionBagPredictions(
            **{**actual.__dict__, "selection_a_indices": np.asarray([1])}
        )
        with pytest.raises(ValueError, match="changed selection_a_indices"):
            experiment._verify_reference_bag(actual=changed, reference_path=path, atol=1e-8)
    finally:
        experiment._load_reference_bag = original_loader
