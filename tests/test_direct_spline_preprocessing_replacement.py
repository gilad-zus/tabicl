from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl._experiments.direct_spline_preprocessing import (
    MinimalNumericalPreprocessor, install_minimal_numerical_preprocessing,
)
from tabicl._sklearn.preprocessing import PreprocessingPipeline


def _module(filename):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    name = filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, scripts / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


experiment = _module("direct_spline_openml_preprocessing_replacement.py")
crossfit = _module("direct_spline_openml_crossfit_context_expansion.py")


def test_minimal_preparation_keeps_category_transform_but_does_not_clip_numeric_tail():
    train = np.array([[0, -2], [1, -1], [0, 0], [2, 1], [1, 8]], dtype=float)
    original = PreprocessingPipeline(normalization_method="power").fit(train)
    minimal = MinimalNumericalPreprocessor(original, train, np.array([1]))
    query = np.array([[0, 1e8], [1, -1e8]], dtype=float)
    output = minimal.transform(query)
    category_probe = query.copy()
    category_probe[:, 1] = train[0, 1]
    assert np.array_equal(output[:, 0], original.transform(category_probe)[:, 0])
    assert np.allclose(output[:, 1], (query[:, 1]-train[:, 1].mean())/train[:, 1].std())
    assert abs(output[0, 1]) > 1e6
    assert np.array_equal(minimal.mean_, train[:, 1:2].mean(axis=0))
    assert np.array_equal(minimal.X_transformed_, minimal.transform(train))


def test_minimal_slots_have_shared_numeric_coordinates_and_unchanged_permutations():
    values = np.array([[0, -2], [1, -1], [0, 0], [2, 1], [1, 8]], dtype=float)
    configs = {"none": [([0, 1], [0, 1])], "power": [([1, 0], [1, 0])]}
    generator = SimpleNamespace(X_=values, ensemble_configs_=configs, preprocessors_={
        name: PreprocessingPipeline(normalization_method=name).fit(values)
        for name in ("none", "power")
    })
    bundle = SimpleNamespace(estimator=SimpleNamespace(ensemble_generator_=generator), numerical_indices=np.array([1]))
    install_minimal_numerical_preprocessing(bundle)
    assert generator.ensemble_configs_ is configs
    assert tuple(generator.preprocessors_) == ("none", "power")
    assert np.array_equal(generator.preprocessors_["none"].X_transformed_[:, 1],
                          generator.preprocessors_["power"].X_transformed_[:, 1])


def test_constant_override_removes_inherited_schedule_and_retains_matched_optimizer():
    args = SimpleNamespace(
        adapter_arm="direct_line", coordinate_mapping="arctan", preserve_input_base=True,
        n_control_points=20, cosine_min_lr_ratio=None, adapter_steps=500,
        query_fraction_min=.05, query_fraction_max=.2, column_control_points=None,
        constant_lr=True, training_random_state=20260828, validation_interval=25,
        numerical_preparation="minimal", branch_diagnostics=True,
    )
    source = {"cosine_schedule_steps": 500, "cosine_min_lr_ratio": .01,
              "trainable_location_scale": True, "learning_rate": .005, "weight_decay": .003}
    configured = crossfit._updated_adapter_config(source, args)
    assert "cosine_schedule_steps" not in configured
    assert configured["learning_rate"] == .005
    assert configured["weight_decay"] == .003
    assert not configured["trainable_shape"]
    assert configured["preserve_input_base"]
    assert configured["numerical_preparation"] == "minimal"
    assert "cosine_schedule_steps" in source


def _record(task, oof, test):
    return {
        "task_id": task, "dataset_id": task+100, "dataset_name": f"task-{task}",
        "problem_type": "multiclass", "effective_bags": 4, "outer_split_hash": f"split-{task}",
        "source_full_outer_training_tabiclv2": {"benchmark_error": 1.},
        "expanded_context": {
            split: {"raw_spline": {"benchmark_error": value}, "identity": {"benchmark_error": 1.}}
            for split, value in (("oof", oof), ("outer_test", test))
        } | {"blend_selection": {"selected_alpha": 1.}},
    }


def test_schedule_selection_is_independent_per_family_and_ignores_test_oracle(tmp_path):
    for arm in experiment.ARMS:
        for schedule in experiment.SCHEDULES:
            directory = tmp_path / "multiclass" / arm / schedule
            directory.mkdir(parents=True)
            # Lines prefer constant on OOF but cosine on test. Splines vice versa.
            selected_oof = ("constant" if arm.endswith("_line") else "cosine")
            selected_test = ("cosine" if arm.endswith("_line") else "constant")
            items = [_record(task, .5 if schedule == selected_oof else .7,
                             .1 if schedule == selected_test else .9)
                     for task in experiment.TASKS["multiclass"]]
            (directory / "task_summaries.json").write_text(json.dumps(items), encoding="utf-8")
    result = experiment._aggregate(SimpleNamespace(output_dir=tmp_path), "multiclass")
    for row in result["task_results"]:
        assert row["standard_line_selected_schedule"] == "constant"
        assert row["standard_spline_selected_schedule"] == "cosine"
        assert row["minimal_line_selected_schedule"] == "constant"
        assert row["minimal_spline_selected_schedule"] == "cosine"
        assert row["minimal_spline_selected_outer_test"] == .9
    assert result["comparisons"]["selected"]["minimal_curvature"]["outer_test"]["ties"] == 4


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"])
def test_real_cpu_bag_trains_minimal_adapter_and_saves_both_branch_predictions(tmp_path, problem_type):
    from tabicl._experiments.direct_spline_openml import OpenMLTaskData
    from tabicl._model.tabicl import TabICL
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"x": rng.normal(size=24), "y": rng.normal(size=24),
                              "category": pd.Categorical(np.resize(["a", "b"], 24))})
        targets = np.resize(np.arange(3), 24) if problem_type == "multiclass" else rng.normal(size=24)
        task = OpenMLTaskData(task_id=1, dataset_id=2, dataset_name="tiny", problem_type=problem_type,
                             n_classes=3 if problem_type == "multiclass" else None, x_train=frame, y_train=targets,
                             x_test=frame.iloc[:3].copy(), y_test=targets[:3], outer_split_hash="split")
        backbone = TabICL(max_classes=3 if problem_type == "multiclass" else 0, num_quantiles=9,
                         embed_dim=8, col_num_blocks=1, col_nhead=1,
                         col_num_inds=2, col_feature_group=False, row_num_blocks=1, row_nhead=1,
                         row_num_cls=1, icl_num_blocks=1, icl_nhead=1,
                         col_ssmax=False, icl_ssmax=False, dropout=0., zero_init=False)
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        config = {"adapter_architecture": "fixed_cubic", "n_control_points": 20,
                  "trainable_shape": True, "trainable_location_scale": False,
                  "coordinate_mapping": "arctan", "direct_spline_output": True,
                  "preserve_input_base": True, "numerical_preparation": "minimal",
                  "cross_column_mixing_rank": 4, "cross_column_mixing_bound": .1,
                  "random_state": 42, "adapter_steps": 1, "adapter_patience": None,
                  "max_context_rows": None, "train_context_rows": None, "query_batch_rows": 256,
                  "query_fraction_min": .1, "query_fraction_max": .2, "validation_interval": 1,
                  "learning_rate": .005, "weight_decay": .003,
                  "row_interaction_chunk_rows": 16,
                  "gate_learning_rate_factor": 3., "grad_clip": 2.,
                  "training_audit_episodes": 1, "branch_diagnostics": True}
        result = crossfit._fit_context_expansion_bag(
            task=task, fit_indices=np.arange(18), validation_indices=np.arange(18, 24),
            bag=0, config=config, protocol_seed=20260915, backbone=backbone, device=torch.device("cpu"),
            run_fingerprint_hash="test", requested_bags=4, effective_bags=4,
            adapter_checkpoint_path=tmp_path / "bag_0.adapters.pt",
        )
        assert result.metadata["adapter_steps_executed"] == 1
        assert result.metadata["identity_parity_max_abs_test"] == 0
        assert set(result.metadata["branch_predictions"]) == {"none", "power"}
        assert "fixed_training_loss" in result.metadata["adapter_checkpoint_records"][0]
        shape = (3, 3) if problem_type == "multiclass" else (3,)
        assert result.expanded_spline_selected_on_a_test.shape == shape
        assert np.isfinite(result.expanded_spline_selected_on_a_test).all()
        for branch in result.metadata["branch_predictions"].values():
            assert np.asarray(branch["expanded_spline_selected_on_a_test"]).shape == shape
        assert all(parameter.grad is None for parameter in backbone.parameters())
    finally:
        torch.set_num_threads(previous_threads)
