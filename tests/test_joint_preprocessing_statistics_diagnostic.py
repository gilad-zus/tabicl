import argparse
import copy

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_statistics_diagnostic as runner
from tabicl._hyperspline.statistics import summarize_context
from tests.test_joint_preprocessing_residual_conditioning import tiny
from tests.test_joint_preprocessing_real_meta_continuation import (
    assert_nested_equal, real_episode, source_family, read_rows,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous_count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_count)


def args_for(root, **changes):
    values = dict(output_dir=root, candidate_manifest=runner.residual.DEFAULT_CANDIDATES,
        cache_dir=root / "cache", arm="statistics", continuation_seed=0, device="cpu",
        checkpoint=None, steps=2, evaluate_every=1, save_every=1, lr=.0003,
        direct_datasets=1, direct_steps=2, direct_rows=64, direct_evaluate_every=1,
        direct_lr=.001, resume=False, max_steps=None, expected_revision=None,
        reuse_bank_dir=None, preflight_family="source", preflight_split_seed=33)
    values.update(changes)
    return argparse.Namespace(**values)


def diagnostic_entry():
    outer = runner.bank.bank.sample_real_episode(source_family(), 64, .75, 71)
    context_positions, fitting_positions, selection_positions = runner.partition_context(outer, 13)
    return dict(outer=outer,
        fitting=runner.inner_episode(outer, context_positions, fitting_positions, "fitting"),
        selection=runner.inner_episode(outer, context_positions, selection_positions, "selection"))


def test_statistics_model_has_no_attention_and_preserves_context_symmetries():
    model, episode = runner.new_model("statistics"), real_episode()
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
    context_values = episode["x_context"][..., episode["numerical_mask"]]
    labels, missing = episode["y_context"], episode["context_missing"][None]
    statistics = summarize_context(context_values, missing, labels)
    expected_tokens = model.encoder(context_values, labels, statistics, missing)
    row_order = torch.arange(context_values.shape[1] - 1, -1, -1)
    column_order = torch.arange(context_values.shape[2] - 1, -1, -1)
    reordered_values = context_values[:, row_order][:, :, column_order]
    reordered_labels = 1 - labels[:, row_order]
    reordered_missing = missing[:, row_order][:, :, column_order]
    reordered_statistics = summarize_context(reordered_values, reordered_missing, reordered_labels)
    actual_tokens = model.encoder(reordered_values, reordered_labels, reordered_statistics, reordered_missing)
    torch.testing.assert_close(actual_tokens, expected_tokens[:, column_order], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("kind", ["statistics", "direct"])
def test_native16_exact_identity_and_staged_gradients(kind):
    backbone, episode, initial = tiny(), real_episode(), runner.new_model("statistics")
    model = initial if kind == "statistics" else runner.direct_model(initial, episode, torch.device("cpu"))
    torch.testing.assert_close(runner.residual.episode_logits(backbone, model, episode),
        runner.residual.episode_logits(backbone, None, episode), rtol=0, atol=0)
    # A nonzero residual exercises all branches of the two-pass input-gradient path.
    with torch.no_grad():
        if kind == "statistics":
            model.affine_head.weight.fill_(.003)
            model.neural_last_head.weight.fill_(.002)
            model.mix_gate_head.bias.fill_(.01)
        else:
            model.raw_outputs["affine_head"].fill_(.003)
            model.raw_outputs["neural_last_head"].fill_(.002)
            model.raw_outputs["mix_gate_head"].fill_(.01)
    staged_model = copy.deepcopy(model)
    context, views = runner.residual.prepared_episode(episode, torch.device("cpu"))
    numerical_views, _ = runner.residual.numeric_views(model, backbone, context, views)
    logits = runner.residual.mean_logits(backbone, views, episode["n_classes"], numerical_views)
    expected_loss = torch.nn.functional.cross_entropy((logits / .9).flatten(0, 1), episode["y_query"])
    expected_loss.backward()
    staged_loss = runner.residual.ensemble_backward(backbone, staged_model, episode, scale=1.)
    torch.testing.assert_close(staged_loss, expected_loss.detach())
    for (name, expected), (_, actual) in zip(model.named_parameters(), staged_model.named_parameters(), strict=True):
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-4, atol=5e-6, msg=name)
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_direct_and_generated_parameter_bounds_match_exactly():
    initial, episode = runner.new_model("statistics"), real_episode()
    with torch.no_grad():
        for head_name, head in initial.named_children():
            if head_name.endswith("_head"):
                head.weight.normal_(0, .01)
                head.bias.add_(.01)
    numerical_context = episode["x_context"][..., episode["numerical_mask"]]
    labels, missing = episode["y_context"], episode["context_missing"][None]
    direct = runner.direct_model(initial, episode, torch.device("cpu"))
    expected = initial.generate(numerical_context, labels, missing)
    actual = direct.generate(numerical_context, labels, missing,
        numerical_column_positions=torch.arange(numerical_context.shape[-1]))
    for field_name in expected.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, field_name), getattr(expected, field_name), rtol=0, atol=0)
    # Parameter ordering follows original numerical columns after native filtering.
    selected_columns = torch.tensor([2, 0])
    subset = direct.generate(numerical_context[..., selected_columns], labels, missing[..., selected_columns],
        numerical_column_positions=selected_columns)
    torch.testing.assert_close(subset.shift, actual.shift[..., selected_columns], rtol=0, atol=0)
    assert subset.mixing.norm(dim=(-2, -1)).max() <= .1


def test_inner_partitions_exclude_all_outer_queries_and_are_label_complete():
    entry = diagnostic_entry()
    outer = entry["outer"]
    outer_query_rows = set(outer["query_indices"].tolist())
    fitting_context_rows = set(entry["fitting"]["context_indices"].tolist())
    fitting_query_rows = set(entry["fitting"]["query_indices"].tolist())
    selection_query_rows = set(entry["selection"]["query_indices"].tolist())
    assert not outer_query_rows & (fitting_context_rows | fitting_query_rows | selection_query_rows)
    assert not fitting_context_rows & fitting_query_rows
    assert not fitting_query_rows & selection_query_rows
    assert fitting_context_rows | fitting_query_rows | selection_query_rows == set(outer["context_indices"].tolist())
    for episode in (entry["fitting"], entry["selection"]):
        assert torch.unique(episode["y_context"]).numel() == outer["n_classes"]
        assert torch.unique(episode["y_query"]).numel() == outer["n_classes"]


def test_reserved_rows_never_enter_shared_batches_and_caps_are_independent(tmp_path, monkeypatch):
    entry = diagnostic_entry()
    sources = [source_family(), source_family("second", rows=300)]
    monkeypatch.setattr(runner.bank, "load_bank", lambda *arguments: sources)
    monkeypatch.setattr(runner, "load_direct_bank", lambda *arguments: [entry])
    filtered = runner.load_source_banks(args_for(tmp_path), {})
    reserved_rows = set(entry["outer"]["query_indices"].tolist())
    assert not reserved_rows & set(filtered["large"][0]["source_indices"])
    monkeypatch.setattr(runner.core, "shape_for_step", lambda *arguments: (256, .7))
    monkeypatch.setattr(runner.residual.diversity, "scheduled_sources", lambda *arguments: filtered["large"] * 2)
    episodes = runner.training_batch("statistics", 1, filtered)
    assert episodes[0]["actual_rows"] == len(filtered["large"][0]["labels"])
    assert episodes[1]["actual_rows"] == 256
    for episode in episodes:
        if episode["family"] == "source":
            assert not reserved_rows & set(episode["context_indices"].tolist() + episode["query_indices"].tolist())


def test_direct_bank_lock_and_changed_outer_labels_cannot_change_fitted_weights(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    manifest = dict(fingerprint="fixture", settings=dict(direct_reference=runner.settings(args)["direct_reference"]))
    monkeypatch.setattr(runner.bank, "load_bank", lambda *arguments: [source_family()])
    runner.prepare_direct_bank(args, manifest)
    original = runner.load_direct_bank(args, manifest)[0]
    changed = copy.deepcopy(original)
    changed["outer"]["y_query"] = 1 - changed["outer"]["y_query"]
    monkeypatch.setattr(runner, "setup", lambda arguments: (tiny(), runner.new_model("statistics"), manifest, torch.device("cpu")))
    for root, entry in ((tmp_path / "original", original), (tmp_path / "changed", changed)):
        monkeypatch.setattr(runner, "load_direct_bank", lambda *arguments, entry=entry: [entry])
        runner.direct(args_for(root))
    original_state = torch.load(tmp_path / "original/direct/source_00/selected.pt", weights_only=True)
    changed_state = torch.load(tmp_path / "changed/direct/source_00/selected.pt", weights_only=True)
    assert_nested_equal(original_state, changed_state)
    completion = runner.previous.read(tmp_path / "original/direct/source_00/complete.json")
    assert completion["steps"] == 2 and not completion["outer_query_used_for_selection"]


def test_shared_train_resume_and_final_report(tmp_path, monkeypatch):
    entry = diagnostic_entry()
    source_banks = dict(large=[source_family(f"source_{position}") for position in range(4)])
    episode = real_episode()
    values = {panel: [episode] for panel in runner.residual.PANELS}
    manifest = dict(fingerprint="fixture", initial_sha256=dict(statistics="initial"),
        settings={}, banks={panel: dict(count=1) for panel in values})
    monkeypatch.setattr(runner, "setup", lambda arguments: (tiny(), runner.new_model("statistics"), manifest, torch.device("cpu")))
    monkeypatch.setattr(runner, "load_source_banks", lambda *arguments: source_banks)
    monkeypatch.setattr(runner, "load_direct_bank", lambda *arguments: [entry])
    monkeypatch.setattr(runner.bank, "load_bank", lambda root, manifest, panel: values[panel])
    monkeypatch.setattr(runner.core, "shape_for_step", lambda *arguments: (20, .5))
    for root, max_steps in ((tmp_path / "full", None), (tmp_path / "resumed", 1)):
        runner.pilot.json_write(root / "backbone_lock.json", dict(sha256="fixture"))
        runner.train(args_for(root, max_steps=max_steps))
    runner.train(args_for(tmp_path / "resumed", resume=True))
    full_state = torch.load(runner.core.run_dir(tmp_path / "full", "statistics", 0) / "state.pt", weights_only=True)
    resumed_state = torch.load(runner.core.run_dir(tmp_path / "resumed", "statistics", 0) / "state.pt", weights_only=True)
    for field_name in ("model", "optimizer", "best_model", "best_step", "best_score", "rng", "other_rng"):
        assert_nested_equal(full_state[field_name], resumed_state[field_name])
    runner.direct(args_for(tmp_path / "full"))
    runner.report(args_for(tmp_path / "full"))
    result = runner.previous.read(tmp_path / "full/complete.json")
    assert set(result["source_comparisons"]) == {"selected", "final"}
    assert len(result["direct_references"]) == 1
    assert len(read_rows(runner.core.run_dir(tmp_path / "full", "statistics", 0) / "presentations.csv")) == 8
    runner.report(args_for(tmp_path / "full"))


def test_prepare_locks_both_data_banks_and_is_idempotent(tmp_path, monkeypatch):
    source_root, output_root = tmp_path / "source", tmp_path / "prepared"
    source_root.mkdir()
    source_specification = dict(counts=dict(small=40, large=160, validation=25),
        candidate_sha256=runner.pilot.hash_file(runner.residual.DEFAULT_CANDIDATES), banks={})
    for panel in ("small_train", "large_train", *runner.residual.PANELS):
        values = [source_family()] if panel == "large_train" else []
        bank_path = source_root / "banks" / f"{panel}.pt"
        runner.pilot.atomic_save(bank_path, dict(values=values))
        source_specification["banks"][panel] = dict(path=bank_path.relative_to(source_root).as_posix(),
            count=len(values), sha256=runner.pilot.hash_file(bank_path))
    runner.pilot.json_write(source_root / "banks_manifest.json", source_specification)
    source_manifest = dict(banks_manifest_sha256=runner.pilot.hash_file(source_root / "banks_manifest.json"))
    source_manifest["fingerprint"] = runner.previous.digest(source_manifest)
    runner.pilot.json_write(source_root / "manifest.json", source_manifest)
    checkpoint_path = tmp_path / "frozen.pt"
    checkpoint_path.write_bytes(b"fixture weights")
    monkeypatch.setattr(runner.pilot, "load_frozen", lambda arguments, device:
        (tiny(), checkpoint_path, runner.pilot.hash_file(checkpoint_path)))
    args = args_for(output_root, reuse_bank_dir=source_root, source_fingerprint=source_manifest["fingerprint"])
    runner.prepare(args)
    manifest = runner.checked_manifest(args)
    runner.prepare(args)
    assert runner.checked_manifest(args) == manifest
    assert manifest["parameter_counts"] == dict(statistics=8187)
    assert len(runner.load_direct_bank(args, manifest)) == 1
    with (output_root / "direct_bank_manifest.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="direct-reference manifest changed"):
        runner.checked_manifest(args)
