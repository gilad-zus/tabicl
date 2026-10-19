import argparse
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import joint_preprocessing_lr_response as runner
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal, source_family
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def single_thread():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def parent_state():
    model = runner.base.prior.new_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .001)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(model=runner.base.pilot.state_cpu(model), optimizer=optimizer.state_dict(), coverage={})


def test_both_arms_preserve_parent_weights_and_moments_but_override_rate():
    parent = parent_state()
    untouched = copy.deepcopy(parent)
    models, optimizers = runner.load_arms(parent, "cpu")
    for arm, learning_rate in runner.ARMS.items():
        assert_nested_equal(models[arm].state_dict(), parent["model"])
        assert_nested_equal(optimizers[arm].state_dict()["state"], parent["optimizer"]["state"])
        assert optimizers[arm].param_groups[0]["lr"] == learning_rate
    first_parameter = next(models["control"].parameters())
    with torch.no_grad():
        first_parameter.add_(1)
    assert not torch.equal(first_parameter, next(models["larger_lr"].parameters()))
    assert_nested_equal(parent, untouched)


def fixture_run(tmp_path):
    source = source_family("train", rows=96, mixed=False)
    cache_path = tmp_path / "source.pt"
    runner.base.pilot.atomic_save(cache_path, source)
    entry = dict(family="train", panel="train", source_group="train", full_rows=96,
        cache_path=str(cache_path), cache_sha256=runner.base.pilot.hash_file(cache_path),
        fitting_positions=torch.arange(88), probe_query_positions=torch.arange(88, 96))
    parent = parent_state()
    parent["coverage"] = {"train": {role: torch.zeros(96, dtype=torch.bool) for role in ("context", "query")}}
    return parent, [entry]


def test_paired_training_and_update_boundary_resume_are_identical(tmp_path):
    parent, entries = fixture_run(tmp_path)
    manifest = dict(fingerprint="fixture")
    for folder in (tmp_path / "whole", tmp_path / "resumed"):
        models, optimizers = runner.load_arms(parent, "cpu")
        options = argparse.Namespace(output_dir=folder, device="cpu", additional_updates=2, resume=False)
        first_target = 2 if folder.name == "whole" else 1
        options.additional_updates = first_target
        assert runner.train(options, tiny(), models, optimizers, entries, manifest, float("inf"), parent) == first_target
        if first_target == 1:
            # Simulate records flushed after the durable pair, then interrupt and restart.
            runner.base.append_jsonl(folder / "episodes.jsonl", dict(additional_update=2, arm="incomplete"))
            models, optimizers = runner.load_arms(parent, "cpu")
            options.additional_updates, options.resume = 2, True
            assert runner.train(options, tiny(), models, optimizers, entries, manifest, float("inf"), parent) == 2
    whole = torch.load(tmp_path / "whole/state.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/state.pt", weights_only=True)
    assert_nested_equal(whole, resumed)
    episodes = [json.loads(line) for line in (tmp_path / "resumed/episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 16
    for control, larger in zip(episodes[::2], episodes[1::2], strict=True):
        for field in ("context_hash", "query_hash", "family", "step", "episode_position", "native_nll"):
            assert control[field] == larger[field]
        assert control["context_rows"] <= 4096 and control["query_rows"] <= 512
    assert not resumed["coverage"]["train"]["context"][88:].any()
    assert not resumed["coverage"]["train"]["query"][88:].any()
    updates = [json.loads(line) for line in (tmp_path / "resumed/updates.jsonl").read_text().splitlines()]
    assert all(record["parameter_update_norm"] > 0 for record in updates)
    assert {record["learning_rate"] for record in updates} == {.0003, .003}


def test_budget_stop_saves_equal_zero_update_state(tmp_path):
    parent, entries = fixture_run(tmp_path)
    models, optimizers = runner.load_arms(parent, "cpu")
    options = argparse.Namespace(output_dir=tmp_path, device="cpu", additional_updates=2, resume=False)
    assert runner.train(options, tiny(), models, optimizers, entries, dict(fingerprint="fixture"), 0, parent) == 0
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    assert state["completed_updates"] == 0
    assert_nested_equal(state["models"]["control"], state["models"]["larger_lr"])


def test_fitting_panel_uses_first_record_per_score_blind_id(tmp_path):
    visits = [dict(family=family, step=step) for step in (143, 144) for family in ("z", "b", "a")]
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(record) for record in visits))
    chosen = runner.fitting_panel(SimpleNamespace(parent_dir=tmp_path))
    assert [record["family"] for record in chosen] == ["a", "b", "z"]
    assert all(record["step"] == 143 for record in chosen)


def test_validation_keeps_full_queries_and_20k_context(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(runner.base.prior, "SourceCache", lambda *args, **kwargs: object())
    def episode_for(entry, cache, **kwargs):
        captured.append(kwargs)
        return dict(family=entry["family"], split=kwargs["split"])
    monkeypatch.setattr(runner.base, "episode_for", episode_for)
    monkeypatch.setattr(runner.base.pilot, "hash_file", lambda *args: "fixture")
    monkeypatch.setattr(runner.base, "prediction_record", lambda options, backbone, models, episode, *args:
        dict(episode, scores=dict(control=dict(nll=.4), native16=dict(nll=.5))))
    assert runner.evaluate_panel(SimpleNamespace(output_dir=tmp_path), None, dict(control=None),
        [dict(family="validation", source_group="validation", panel="validation")],
        dict(fingerprint="fixture"), "validation", float("inf"))
    assert captured == [dict(split=split, context_row_cap=20000) for split in range(3)]


def test_nonfinite_gradients_fail_before_optimizer_update():
    model = runner.base.prior.new_model()
    optimizer = torch.optim.AdamW(model.parameters())
    next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("nan"))
    with pytest.raises(FloatingPointError, match="gradient"):
        runner.optimizer_step(model, optimizer)


def test_evaluation_resume_keeps_locked_endpoint_and_checkpoint_bytes(tmp_path, monkeypatch):
    parent, entries = fixture_run(tmp_path)
    runner.base.pilot.atomic_save(tmp_path / "sources.pt", dict(entries=entries))
    manifest = dict(fingerprint="fixture", backbone=dict(path="fixture.pt", sha256="hash"))
    options = SimpleNamespace(output_dir=tmp_path, device="cpu", additional_updates=1,
                              resume=True, wall_minutes=10, evaluation_reserve_minutes=1)
    monkeypatch.setattr(runner, "prepare", lambda *args: None)
    monkeypatch.setattr(runner, "checked_manifest", lambda *args: manifest)
    monkeypatch.setattr(runner, "read_parent", lambda *args: ({}, parent))
    monkeypatch.setattr(runner.base.pilot, "load_frozen", lambda *args: (tiny(), "fixture.pt", "hash"))
    monkeypatch.setattr(runner, "evaluate_panel", lambda *args: False)
    runner.pipeline(options)
    checkpoint_hashes = {arm: runner.base.pilot.hash_file(tmp_path / f"{arm}.pt")
                         for arm in (*runner.ARMS, "starting")}
    def unexpected_training(*args):
        raise AssertionError("evaluation resume must not train again")
    monkeypatch.setattr(runner, "train", unexpected_training)
    monkeypatch.setattr(runner, "evaluate_panel", lambda *args: True)
    runner.pipeline(options)
    assert checkpoint_hashes == {arm: runner.base.pilot.hash_file(tmp_path / f"{arm}.pt")
                                for arm in checkpoint_hashes}
    assert json.loads((tmp_path / "complete.json").read_text())["completed_updates_per_arm"] == 1


def test_parent_rejects_pending_gradients_or_wrong_hash(tmp_path, monkeypatch):
    parent = parent_state()
    parent.update(fingerprint=runner.PARENT_FINGERPRINT, step=320, episode_position=0,
                  query_offset=0, pending_validation=False, gradients=dict(weight=torch.ones(1)))
    runner.base.pilot.atomic_save(tmp_path / "state.pt", parent)
    monkeypatch.setattr(runner.base, "verified_json", lambda *args: dict(fingerprint=runner.PARENT_FINGERPRINT))
    options = SimpleNamespace(parent_dir=tmp_path, parent_state_sha256="wrong")
    with pytest.raises(ValueError, match="audited bytes"):
        runner.read_parent(options)
    options.parent_state_sha256 = runner.base.pilot.hash_file(tmp_path / "state.pt")
    with pytest.raises(ValueError, match="pending gradients"):
        runner.read_parent(options)


def test_continuation_replay_uses_actual_logged_rows_and_training_sizes(tmp_path, monkeypatch):
    parent, entries = fixture_run(tmp_path)
    models, optimizers = runner.load_arms(parent, "cpu")
    options = SimpleNamespace(output_dir=tmp_path, device="cpu", additional_updates=1, resume=False)
    manifest = dict(fingerprint="fixture")
    runner.train(options, tiny(), models, optimizers, entries, manifest, float("inf"), parent)
    actual_episode_for = runner.base.episode_for
    captured = []
    def episode_for(*args, **kwargs):
        captured.append(kwargs)
        return actual_episode_for(*args, **kwargs)
    monkeypatch.setattr(runner.base, "episode_for", episode_for)
    actual_hash_file = runner.base.pilot.hash_file
    monkeypatch.setattr(runner.base.pilot, "hash_file", lambda path:
        "fixture" if path.name == "control.pt" else actual_hash_file(path))
    monkeypatch.setattr(runner.base, "prediction_record", lambda options, backbone, models, episode, *args:
        dict(family=episode["family"], scores=dict(control=dict(nll=.4), native16=dict(nll=.5))))
    assert runner.evaluate_panel(options, None, dict(control=None), entries, manifest, "training_replay", float("inf"))
    assert len(captured) == 4
    assert all(visit["context_row_cap"] == 4096 and visit["training_query_row_cap"] == 512 for visit in captured)
    summary = json.loads((tmp_path / "training_replay/complete.json").read_text())
    assert summary["qualification"] == "optimized continuation rows"
