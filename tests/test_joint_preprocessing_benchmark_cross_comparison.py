"""Verify genuine cross interactions, gradient flow, paired budgets and test locks."""
import copy

import pytest
import torch
import torch.nn.functional as functional

from scripts import joint_preprocessing_benchmark_sampling as runner
from tests.test_joint_preprocessing_benchmark_sampling import args_for, entry_for, one_thread
from tests.test_joint_preprocessing_benchmark_restricted import completed_source, restricted_args
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal, read_rows
from tests.test_joint_preprocessing_residual_conditioning import tiny


def comparison_args(root, **changes):
    settings = dict(transformation="comparison", source_dir=root.parent / "source",
                    restricted_reference_dir=root.parent / "reference", lr=.0003)
    settings.update(changes)
    return args_for(root, **settings)


def make_model(root, arm="cross_low_lr"):
    entry, _ = entry_for(root)
    initial = runner.diagnostic.new_model("statistics")
    return entry, runner.direct_model(comparison_args(root), initial, entry, torch.device("cpu"), arm=arm)


def parameters_for(model, numerical_column_count, column_positions=None):
    if column_positions is None:
        column_positions = torch.arange(numerical_column_count)
    return model.generate(torch.zeros(1, 4, len(column_positions)), torch.zeros(1, 4),
                          numerical_column_positions=column_positions)


def test_cross_identity_parameter_budget_and_first_step_learning(tmp_path):
    entry, model = make_model(tmp_path)
    numerical_column_count = int(entry["metadata"]["numerical_mask"].sum())
    assert sum(parameter.numel() for parameter in model.parameters()) == 2 * numerical_column_count * (4 + 2 * model.rank)
    assert not any("spline" in parameter_name or "neural" in parameter_name for parameter_name in model.raw_outputs)
    backbone = tiny()
    baseline = runner.score_partition(backbone, None, entry, entry["validation"], 16)
    assert runner.score_partition(backbone, model, entry, entry["validation"], 16) == baseline
    runner.residual.ensemble_backward(backbone, model, runner.training_episode(entry, "cross_low_lr", 1), scale=1.)
    output_gradient = model.raw_outputs["cross_output_factors"].grad
    assert torch.isfinite(output_gradient).all() and output_gradient.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    optimizer.step()
    assert model.raw_outputs["cross_output_factors"].abs().sum() > 0
    optimizer.zero_grad(set_to_none=True)
    runner.residual.ensemble_backward(backbone, model, runner.training_episode(entry, "cross_low_lr", 2), scale=1.)
    for parameter_name in ("cross_input_factors", "cross_output_factors", "cross_gate"):
        gradient = model.raw_outputs[parameter_name].grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_cross_is_multiplicative_preserves_missing_and_original_column_alignment(tmp_path):
    entry, model = make_model(tmp_path)
    numerical_column_count = int(entry["metadata"]["numerical_mask"].sum())
    with torch.no_grad():
        model.raw_outputs["cross_input_factors"].zero_()
        model.raw_outputs["cross_output_factors"].zero_()
        # Column zero receives the product of its own value and column two.
        model.raw_outputs["cross_input_factors"][:, :, 2, 0] = 1.
        model.raw_outputs["cross_output_factors"][:, :, 0, 0] = 1.
    column_positions = torch.tensor([2, 0, 4])
    parameters = parameters_for(model, numerical_column_count, column_positions)
    native_values = torch.tensor([[[3., 2., 9.], [5., 2., 9.]]])
    missing = torch.zeros_like(native_values, dtype=torch.bool)
    actual = model.apply_native(native_values, parameters, 0, missing)
    expected = native_values.clone()
    expected[..., 1] += .02 * native_values[..., 1] * native_values[..., 0]
    torch.testing.assert_close(actual, expected)
    missing[:, 1, 1] = True
    masked = model.apply_native(native_values, parameters, 0, missing)
    assert masked[0, 1, 1] == native_values[0, 1, 1]
    # Reordering columns reorders the transform rather than changing its meaning.
    permutation = torch.tensor([1, 2, 0])
    reordered = model.apply_native(native_values[..., permutation],
        parameters_for(model, numerical_column_count, column_positions[permutation]), 0, missing[..., permutation])
    torch.testing.assert_close(reordered, masked[..., permutation])


def test_cross_two_pass_gradients_match_full_autograd(tmp_path):
    entry, model = make_model(tmp_path)
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(8)
        model.raw_outputs["cross_output_factors"].normal_(0, .01)
        model.raw_outputs["affine_head"].normal_(0, .02)
    staged_model = copy.deepcopy(model)
    backbone = tiny()
    episode = runner.training_episode(entry, "cross_low_lr", 2)
    context, views = runner.residual.prepared_episode(episode, torch.device("cpu"))
    numerical_values, _ = runner.residual.numeric_views(model, backbone, context, views)
    logits = runner.residual.mean_logits(backbone, views, episode["n_classes"], numerical_values)
    expected_loss = functional.cross_entropy((logits / .9).flatten(0, 1), episode["y_query"].flatten())
    expected_loss.backward()
    actual_loss = runner.residual.ensemble_backward(backbone, staged_model, episode, scale=1.)
    torch.testing.assert_close(actual_loss, expected_loss.detach(), rtol=1e-6, atol=1e-6)
    for parameter_name, parameter in model.named_parameters():
        staged_parameter = dict(staged_model.named_parameters())[parameter_name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(staged_parameter.grad, parameter.grad, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("arm", runner.COMPARISON_ARMS)
def test_comparison_resume_preserves_optimizer_maps_and_paired_rows(tmp_path, arm):
    entry, _ = entry_for(tmp_path)
    initial = runner.diagnostic.new_model("statistics")
    manifest = dict(fingerprint="fixture")
    uninterrupted_root, resumed_root = tmp_path / "uninterrupted", tmp_path / "resumed"
    runner.fit_arm(comparison_args(uninterrupted_root), tiny(), initial, manifest, entry, arm)
    runner.fit_arm(comparison_args(resumed_root, max_steps=1), tiny(), initial, manifest, entry, arm)
    runner.fit_arm(comparison_args(resumed_root, resume=True), tiny(), initial, manifest, entry, arm)
    for filename in ("state.pt", "selected.pt", "final.pt"):
        expected = torch.load(uninterrupted_root / "tasks/363613" / arm / filename, weights_only=True)
        actual = torch.load(resumed_root / "tasks/363613" / arm / filename, weights_only=True)
        assert_nested_equal(actual, expected)
    for step in (1, 2, 3):
        assert_nested_equal(runner.training_episode(entry, arm, step), runner.training_episode(entry, "resampled", step))
    training_rows = read_rows(resumed_root / "tasks/363613" / arm / "training.csv")
    assert all(float(row["learning_rate"]) == .0003 for row in training_rows)
    assert all(float(row["non_affine_gradient_norm"]) > 0 for row in training_rows)
    validation_rows = read_rows(resumed_root / "tasks/363613" / arm / "validation.csv")
    assert float(validation_rows[0]["slot0_correction_rms"]) == 0
    assert float(validation_rows[-1]["slot0_correction_rms"]) > 0
    if arm == "cross_low_lr":
        assert float(validation_rows[0]["cross_output_factor_norm"]) == 0
        assert float(validation_rows[-1]["cross_output_factor_norm"]) > 0


def make_references(tmp_path, monkeypatch):
    source_root, reference_root = tmp_path / "source", tmp_path / "reference"
    source_manifest = completed_source(source_root, monkeypatch)
    reference_args = restricted_args(reference_root)
    monkeypatch.setattr(runner.pilot, "load_frozen", lambda *arguments: (tiny(), None, runner.CHECKPOINT_HASH))
    runner.run(reference_args)
    reference_manifest = runner.previous.read(reference_root / "manifest.json")
    monkeypatch.setattr(runner, "RESTRICTED_FINGERPRINT", reference_manifest["fingerprint"])
    monkeypatch.setattr(runner, "RESTRICTED_REPORT_HASH", runner.pilot.hash_file(reference_root / "complete.json"))
    return source_manifest


def test_serial_conditions_lock_all_maps_before_test_and_reuse_references(tmp_path, monkeypatch):
    source_manifest = make_references(tmp_path, monkeypatch)
    target_root = tmp_path / "comparison"
    original_hashes = {path: runner.pilot.hash_file(path) for root in (tmp_path / "source", tmp_path / "reference")
                       for path in root.rglob("*") if path.is_file()}
    original_load = torch.load

    def forbid_test_banks(path, *arguments, **keywords):
        assert path.name != "test.pt", "test labels must wait for every new condition's map locks"
        return original_load(path, *arguments, **keywords)

    with monkeypatch.context() as fitting_patch:
        fitting_patch.setattr(torch, "load", forbid_test_banks)
        runner.run(comparison_args(target_root, arm="spline_low_lr"))
        assert not (target_root / "complete.json").exists()
        runner.run(comparison_args(target_root, arm="cross_low_lr", max_steps=1))
        with pytest.raises(FileNotFoundError):
            runner.report(comparison_args(target_root), tiny(), runner.diagnostic.new_model("statistics"),
                          runner.checked_manifest(comparison_args(target_root)))
    runner.run(comparison_args(target_root, arm="cross_low_lr", resume=True))
    report = runner.previous.read(target_root / "complete.json")
    assert report["datasets"] == 1
    assert "spline_lower_vs_original_lr_selected" in report["comparisons"]
    assert "cross_vs_spline_same_lr_selected" in report["comparisons"]
    assert report["results"][0]["ordinary16"]["rows"] == source_manifest["accepted"][0]["official_test_rows"]
    assert all(runner.pilot.hash_file(path) == expected_hash for path, expected_hash in original_hashes.items())
    assert not list(target_root.rglob("test.pt")) and not list(target_root.rglob("fit_validation.pt"))
    with pytest.raises(ValueError, match="protocol differs"):
        runner.prepare_restricted(comparison_args(tmp_path / "wrong-budget", steps=4))
    with pytest.raises(ValueError, match="learning rate"):
        runner.settings(comparison_args(target_root, lr=.001))
    (tmp_path / "reference" / "complete.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference report changed"):
        runner.checked_manifest(comparison_args(target_root))
