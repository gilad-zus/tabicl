"""Verify restricted map gradients, restart continuity, and reuse of frozen banks."""
import copy

import pytest
import torch
import torch.nn.functional as functional

from scripts import joint_preprocessing_benchmark_sampling as runner
from tests.test_joint_preprocessing_benchmark_sampling import args_for, entry_for, one_thread
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal
from tests.test_joint_preprocessing_residual_conditioning import tiny


def restricted_args(root, **changes):
    return args_for(root, transformation="restricted", source_dir=root.parent / "source", **changes)


def test_restricted_map_has_only_affine_spline_parameters_and_matches_retained_heads(tmp_path):
    entry, _ = entry_for(tmp_path)
    initial = runner.diagnostic.new_model("statistics")
    model = runner.direct_model(restricted_args(tmp_path), initial, entry, torch.device("cpu"))
    full_model = runner.direct_model(args_for(tmp_path), initial, entry, torch.device("cpu"))
    numerical_columns = int(entry["metadata"]["numerical_mask"].sum())
    assert sum(parameter.numel() for parameter in model.parameters()) == 44 * numerical_columns
    assert sum(parameter.numel() for parameter in full_model.parameters()) == 114 * numerical_columns
    assert set(model.raw_outputs) == {"affine_head", "spline_head", "spline_gate_head"}
    for head_name, values in model.raw_outputs.items():
        torch.testing.assert_close(values, full_model.raw_outputs[head_name], rtol=0, atol=0)
    baseline = runner.score_partition(tiny(), None, entry, entry["validation"], 16)
    assert runner.score_partition(tiny(), model, entry, entry["validation"], 16) == baseline


def test_restricted_transform_is_monotone_preserves_missing_and_matches_disabled_full_map(tmp_path):
    entry, _ = entry_for(tmp_path)
    initial = runner.diagnostic.new_model("statistics")
    model = runner.direct_model(restricted_args(tmp_path), initial, entry, torch.device("cpu"))
    full_model = runner.direct_model(args_for(tmp_path), initial, entry, torch.device("cpu"))
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(5)
        model.raw_outputs["affine_head"].normal_(0, .1)
        model.raw_outputs["spline_head"].normal_(0, .2)
        for head_name, values in model.raw_outputs.items():
            full_model.raw_outputs[head_name].copy_(values)
    numerical_context = entry["fitting"]["values"][..., entry["metadata"]["numerical_mask"]]
    column_positions = torch.arange(numerical_context.shape[-1])
    generate_arguments = dict(numerical_column_positions=column_positions)
    restricted_parameters = model.generate(numerical_context, entry["fitting"]["labels"].float()[None],
                                           entry["fitting"]["missing"][None], **generate_arguments)
    full_parameters = full_model.generate(numerical_context, entry["fitting"]["labels"].float()[None],
                                         entry["fitting"]["missing"][None], **generate_arguments)
    assert restricted_parameters.neural_first_weight is None and restricted_parameters.mixing is None
    native_values = torch.linspace(-10, 10, 101)[None, :, None].expand(1, 101, len(column_positions))
    missing = torch.zeros_like(native_values, dtype=torch.bool)
    for slot in (0, 1):
        transformed = model.apply_native(native_values, restricted_parameters, slot, missing)
        assert (transformed[:, 1:] >= transformed[:, :-1]).all()
        torch.testing.assert_close(transformed, full_model.apply_native(native_values, full_parameters, slot, missing), rtol=0, atol=0)
        missing[:, ::7] = True
        masked = model.apply_native(native_values, restricted_parameters, slot, missing)
        torch.testing.assert_close(masked[missing], native_values[missing], rtol=0, atol=0)
        missing.zero_()


def test_restricted_two_pass_gradients_match_full_autograd(tmp_path):
    entry, _ = entry_for(tmp_path)
    backbone = tiny()
    model = runner.direct_model(restricted_args(tmp_path), runner.diagnostic.new_model("statistics"), entry, torch.device("cpu"))
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(8)
        model.raw_outputs["affine_head"].normal_(0, .03)
        model.raw_outputs["spline_head"].normal_(0, .02)
    staged_model = copy.deepcopy(model)
    episode = runner.training_episode(entry, "resampled", 2)
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
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_restricted_resume_matches_uninterrupted_fitting(tmp_path):
    entry, _ = entry_for(tmp_path)
    initial = runner.diagnostic.new_model("statistics")
    manifest = dict(fingerprint="fixture")
    uninterrupted_root, resumed_root = tmp_path / "uninterrupted", tmp_path / "resumed"
    runner.fit_arm(restricted_args(uninterrupted_root), tiny(), initial, manifest, entry, "resampled")
    runner.fit_arm(restricted_args(resumed_root, max_steps=1), tiny(), initial, manifest, entry, "resampled")
    runner.fit_arm(restricted_args(resumed_root, resume=True), tiny(), initial, manifest, entry, "resampled")
    for filename in ("state.pt", "selected.pt", "final.pt"):
        expected = torch.load(uninterrupted_root / "tasks/363613/resampled" / filename, weights_only=True)
        actual = torch.load(resumed_root / "tasks/363613/resampled" / filename, weights_only=True)
        assert_nested_equal(actual, expected)


def completed_source(root, monkeypatch):
    source_args = args_for(root)
    entry, test = entry_for(root)
    task_folder = runner.task_folder(source_args, 363613)
    runner.pilot.atomic_save(task_folder / "fit_validation.pt", entry)
    runner.pilot.atomic_save(task_folder / "test.pt", test)
    record = dict(task_id=363613, family="fixture", official_test_rows=len(test["labels"]),
        fitting_sha256=runner.pilot.hash_file(task_folder / "fit_validation.pt"),
        test_sha256=runner.pilot.hash_file(task_folder / "test.pt"))
    manifest = dict(settings=runner.settings(source_args), accepted=[record], excluded=[])
    manifest["fingerprint"] = runner.previous.digest(manifest)
    runner.pilot.json_write(root / "manifest.json", manifest)
    initial = runner.diagnostic.new_model("statistics")
    for arm in runner.ARMS:
        runner.fit_arm(source_args, tiny(), initial, manifest, entry, arm)
    runner.report(source_args, tiny(), initial, manifest)
    monkeypatch.setattr(runner, "SOURCE_FINGERPRINT", manifest["fingerprint"])
    monkeypatch.setattr(runner, "SOURCE_REPORT_HASH", runner.pilot.hash_file(root / "complete.json"))
    return manifest


def test_restricted_run_reuses_banks_keeps_source_immutable_and_locks_before_test(tmp_path, monkeypatch):
    source_root, target_root = tmp_path / "source", tmp_path / "restricted"
    source_manifest = completed_source(source_root, monkeypatch)
    target_args = restricted_args(target_root, max_steps=1)
    monkeypatch.setattr(runner.pilot, "load_frozen", lambda *arguments: (tiny(), None, runner.CHECKPOINT_HASH))
    source_hashes = {path: runner.pilot.hash_file(path) for path in source_root.rglob("*") if path.is_file()}
    original_load = torch.load

    def prevent_test_loading(path, *arguments, **keywords):
        assert path.name != "test.pt", "training must not deserialize the official test bank"
        return original_load(path, *arguments, **keywords)

    with monkeypatch.context() as training_patch:
        training_patch.setattr(torch, "load", prevent_test_loading)
        runner.run(target_args)
        with pytest.raises(FileNotFoundError):
            runner.report(target_args, tiny(), runner.diagnostic.new_model("statistics"), runner.checked_manifest(target_args))
    assert not list(target_root.rglob("test.pt")) and not list(target_root.rglob("fit_validation.pt"))
    runner.run(restricted_args(target_root, resume=True))
    result = runner.previous.read(target_root / "complete.json")
    assert result["datasets"] == 1
    assert "restricted_vs_full_resampled_selected" in result["comparisons"]
    assert result["results"][0]["ordinary16"]["rows"] == source_manifest["accepted"][0]["official_test_rows"]
    assert all(runner.pilot.hash_file(path) == expected_hash for path, expected_hash in source_hashes.items())
    with pytest.raises(ValueError, match="protocol differs"):
        runner.prepare_restricted(restricted_args(tmp_path / "different-budget", steps=4))
    (source_root / "complete.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="report changed"):
        runner.checked_manifest(restricted_args(target_root))
