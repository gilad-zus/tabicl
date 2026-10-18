import argparse
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_fraction_context as runner
from tests.test_joint_preprocessing_real_meta_continuation import assert_nested_equal, real_episode, source_family
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def args(root, **changes):
    values = dict(output_dir=root, source_dir=root / "source", catalog_dir=root / "catalog",
        candidate_manifest=runner.residual.DEFAULT_CANDIDATES, expected_revision=None, checkpoint=None,
        device="cpu", steps=2, evaluate_every=1, save_every=2, query_chunk_rows=7,
        continuation_seed=0, lr=.0003, wall_minutes=10, resume=False, max_records=None)
    values.update(changes)
    return argparse.Namespace(**values)


def trained_model():
    model = runner.prior.new_model()
    with torch.no_grad():
        model.affine_head.weight.fill_(.003)
        model.spline_head.weight.fill_(.002)
    return model


def test_fraction_split_exhausts_full_pool_and_preserves_class_support():
    labels = np.arange(10003) % 3
    available = np.arange(10003)[256:]
    split = runner.split_positions(labels, available, 13)
    assert len(split["context"]) == int(.8 * len(available))
    assert len(split["query"]) > 256
    assert not set(split["context"]) & set(split["query"])
    assert set(np.concatenate(list(split.values()))) == set(available)
    assert set(labels[split["context"]]) == set(labels)
    changed = runner.split_positions(labels, available, 14)
    assert not np.array_equal(changed["context"], split["context"])
    for role, positions in split.items():
        np.testing.assert_array_equal(positions, runner.split_positions(labels, available, 13)[role])
    with pytest.raises(ValueError, match="unique"):
        runner.split_positions(labels, [1, 1, 2], 13)


def test_chunked_gradients_match_all_queries_and_activation_recomputation():
    episode, model, backbone = real_episode(), trained_model(), tiny()
    expected = copy.deepcopy(model)
    expected_loss = runner.residual.ensemble_backward(backbone, expected, episode, scale=.25)
    runner.configure_backbone(backbone)
    prepared = runner.PreparedEpisode(episode)
    generated, actual_loss = None, 0.
    for start in range(0, episode["n_query"], 5):
        stop = min(start + 5, episode["n_query"])
        context, views = prepared.chunk(start, stop, "cpu")
        generated = generated or runner.GeneratedMap(backbone, model, context, episode["n_classes"])
        weight = (stop - start) / episode["n_query"]
        actual_loss += generated.backward_chunk(backbone, views, context,
            episode["y_query"][start:stop], weight=weight / 4.) * weight
    assert actual_loss == pytest.approx(float(expected_loss), rel=2e-6)
    for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), expected.named_parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=8e-4, atol=5e-6, msg=name)
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_prediction_chunks_match_whole_query_and_generate_once(tmp_path, monkeypatch):
    episode, model, backbone = dict(real_episode(), split=0), trained_model(), tiny()
    runner.configure_backbone(backbone)
    context, views = runner.residual.prepared_episode(episode, torch.device("cpu"))
    with torch.no_grad():
        numerical_values, _ = runner.residual.numeric_views(model, backbone, context, views)
        whole_logits = runner.residual.mean_logits(backbone, views, episode["n_classes"], numerical_values)
    generated_calls = []
    original_generate = model.generate

    def counted_generate(*arguments, **keywords):
        generated_calls.append(1)
        return original_generate(*arguments, **keywords)

    monkeypatch.setattr(model, "generate", counted_generate)
    result = runner.prediction_record(args(tmp_path, query_chunk_rows=5), backbone, dict(candidate=model),
        episode, tmp_path / "record", "fixture", float("inf"))
    assert len(generated_calls) == 1
    assert result["scores"]["candidate"]["nll"] == pytest.approx(runner.prior.score(whole_logits, episode["y_query"])["nll"], abs=1e-7)
    saved = torch.load(tmp_path / "record/progress.pt", weights_only=True)
    torch.testing.assert_close(torch.cat(saved["predictions"]["candidate"], 1), whole_logits, rtol=1e-5, atol=1e-6)
    assert saved["query_offset"] == episode["n_query"]


def test_prediction_resume_preserves_all_queries_and_context(tmp_path, monkeypatch):
    episode, model, backbone = dict(real_episode(), split=0), trained_model(), tiny()
    ticks = iter([0., 100.])
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    options = args(tmp_path, query_chunk_rows=5)
    assert runner.prediction_record(options, backbone, dict(candidate=model), episode, tmp_path / "resumed", "fixture", 50.) is None
    progress = torch.load(tmp_path / "resumed/progress.pt", weights_only=True)
    assert progress["query_offset"] == 5
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 0.))
    result = runner.prediction_record(options, backbone, dict(candidate=model), episode, tmp_path / "resumed", "fixture", 50.)
    reference = runner.prediction_record(options, backbone, dict(candidate=model), episode, tmp_path / "full", "fixture", 50.)
    assert result == reference


def test_equal_dataset_aggregation_after_partition_averaging():
    records = [dict(family="small", scores=dict(candidate=dict(nll=.4), native16=dict(nll=.5))),
               dict(family="small", scores=dict(candidate=dict(nll=.6), native16=dict(nll=.5))),
               dict(family="large", scores=dict(candidate=dict(nll=.8), native16=dict(nll=1.)))]
    summary = runner.aggregate_records(records, "candidate")
    assert summary["mean_nll"] == pytest.approx(.65)
    assert summary["native_mean_nll"] == pytest.approx(.75)
    assert summary["datasets"] == 2


def test_query_labels_and_values_cannot_change_generated_map():
    episode, model, backbone = real_episode(), trained_model(), tiny()
    changed = copy.deepcopy(episode)
    changed["y_query"] = 1 - changed["y_query"]
    changed["x_query"] += 1000.
    maps = []
    for current_episode in (episode, changed):
        prepared = runner.PreparedEpisode(current_episode)
        context, _ = prepared.chunk(0, 5, "cpu")
        maps.append(runner.GeneratedMap(backbone, model, context, episode["n_classes"]).original)
    for field in runner.dataclasses.fields(maps[0]):
        original_value, changed_value = getattr(maps[0], field.name), getattr(maps[1], field.name)
        if isinstance(original_value, torch.Tensor):
            torch.testing.assert_close(original_value, changed_value, rtol=0, atol=0)


def test_mid_episode_training_resume_matches_uninterrupted(tmp_path, monkeypatch):
    source = source_family("train", rows=96, mixed=False)
    cache_path = tmp_path / "source.pt"
    runner.pilot.atomic_save(cache_path, source)
    entry = dict(family="train", panel="train", source_group="train", full_rows=96,
        cache_path=str(cache_path), cache_sha256=runner.pilot.hash_file(cache_path),
        fitting_positions=torch.arange(88), probe_query_positions=torch.arange(88, 96))
    manifest = dict(fingerprint="fixture")
    monkeypatch.setattr(runner, "setup", lambda options: (tiny(), runner.prior.new_model(), [entry], manifest))
    monkeypatch.setattr(runner, "validation", lambda *arguments: dict(mean_nll=1., native_mean_nll=1.))
    for root in (tmp_path / "full", tmp_path / "resumed"):
        runner.pilot.json_write(root / "profile/complete.json", dict(fingerprint="fixture", passed=True))
    assert runner.train(args(tmp_path / "full", query_chunk_rows=11))
    assert not runner.train(args(tmp_path / "resumed", query_chunk_rows=11, max_records=3))
    state = torch.load(tmp_path / "resumed/state.pt", weights_only=True)
    assert state["step"] == 0 and state["episode_position"] == 1 and state["query_offset"] == 11
    assert runner.train(args(tmp_path / "resumed", query_chunk_rows=11, resume=True))
    whole = torch.load(tmp_path / "full/state.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/state.pt", weights_only=True)
    assert_nested_equal(whole, resumed)
    records = [json.loads(line) for line in (tmp_path / "resumed/chunks.jsonl").read_text().splitlines()]
    assert [record["record"] for record in records] == list(range(1, 17))
    assert len({(record["step"], record["episode_position"], record["query_start"]) for record in records}) == 16


def test_specification_pins_full_fraction_and_chunk_execution(tmp_path):
    settings = runner.specification(args(tmp_path))
    assert settings["context_fraction"] == .8 and settings["query_fraction"] == .2
    assert settings["context_row_cap"] is None and settings["query_row_cap"] is None
    assert settings["query_chunk_rows"] == 7
    assert "scripts/joint_preprocessing_fraction_context.py" in settings["code_hashes"]


def test_prepare_replays_catalog_raw_cache_and_frozen_reservations(tmp_path, monkeypatch):
    options = args(tmp_path / "experiment")
    source_cache_dir = options.source_dir / "raw"
    entries = []
    for family, panel in (("training", "train"), ("validation", "real_validation")):
        source = source_family(family, rows=96, mixed=False)
        cache_path = source_cache_dir / f"{family}.pt"
        runner.pilot.atomic_save(cache_path, source)
        entries.append(dict(family=family, source_group=family, panel=panel, full_rows=96,
            cache_path=str(cache_path), cache_sha256=runner.pilot.hash_file(cache_path),
            fitting_positions=torch.arange(88), probe_query_positions=torch.arange(88, 96)))
    runner.pilot.atomic_save(options.source_dir / "sources.pt", dict(entries=entries))
    source_manifest = dict(settings=dict(source_cache_dirs=[str(source_cache_dir)]),
        files={"sources.pt": runner.pilot.hash_file(options.source_dir / "sources.pt")})
    source_manifest["fingerprint"] = runner.previous.digest(source_manifest)
    runner.pilot.json_write(options.source_dir / "manifest.json", source_manifest)
    runner.pilot.json_write(options.source_dir / "backbone_lock.json", dict(path="fixture", sha256="fixture"))
    evaluation_source = source_family("evaluation", rows=101, mixed=False)
    cache_path = source_cache_dir / "evaluation.pt"
    runner.pilot.atomic_save(cache_path, evaluation_source)
    fingerprint = runner.bank.content_fingerprints(runner.bank.unpack_frame(evaluation_source["columns"]),
                                                 np.asarray(evaluation_source["labels"]))
    catalog = dict(accepted=[dict(family="evaluation", source_group="evaluation", rows=101,
        episode_file="episodes/evaluation.pt", inputs_sha256=fingerprint["inputs_sha256"],
        exposure="previously_declared_source")], source_lock=dict(checkpoints={}))
    catalog["fingerprint"] = runner.previous.digest(catalog)
    runner.pilot.json_write(options.catalog_dir / "manifest.json", catalog)
    monkeypatch.setattr(runner, "SOURCE_FINGERPRINT", source_manifest["fingerprint"])
    monkeypatch.setattr(runner, "CATALOG_FINGERPRINT", catalog["fingerprint"])
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", dict(train=1, validation=1, evaluation=1))
    runner.prepare(options)
    manifest = runner.checked_manifest(options)
    assert manifest["counts"] == dict(train=1, validation=1, evaluation=1)
    saved = torch.load(options.output_dir / "sources.pt", weights_only=True)["entries"]
    assert [(entry["context_rows"], entry["query_rows"]) for entry in saved] == [(70, 18), (76, 20), (80, 21)]
    torch.testing.assert_close(saved[0]["probe_query_positions"], entries[0]["probe_query_positions"])
    cache = runner.prior.SourceCache(saved)
    training_episode = runner.episode_for(saved[0], cache, step=1)
    assert not set(training_episode["query_indices"].tolist()) & set(range(88, 96))
    assert not set(training_episode["context_indices"].tolist()) & set(range(88, 96))
    runner.prepare(options)
    assert runner.checked_manifest(options) == manifest
    cache_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        runner.prior.SourceCache(saved).get("evaluation")


def test_resource_profile_uses_full_context_and_preserves_initial_model(tmp_path, monkeypatch):
    entries = []
    for family, panel, rows in (("train", "train", 96), ("validation", "validation", 128)):
        source = source_family(family, rows=rows, mixed=False)
        cache_path = tmp_path / f"{family}.pt"
        runner.pilot.atomic_save(cache_path, source)
        available_rows = rows - 8 if panel == "train" else rows
        context_rows = int(.8 * available_rows)
        entries.append(dict(family=family, source_group=family, panel=panel,
            cache_path=str(cache_path), cache_sha256=runner.pilot.hash_file(cache_path),
            full_rows=rows, fitting_positions=torch.arange(rows - 8), probe_query_positions=torch.arange(rows - 8, rows),
            context_rows=context_rows, query_rows=available_rows - context_rows, input_columns=6))
    model = runner.prior.new_model()
    initial = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(runner, "setup", lambda options: (tiny(), model, entries, dict(fingerprint="fixture")))
    runner.profile(args(tmp_path / "experiment"))
    result = runner.previous.read(tmp_path / "experiment/profile/complete.json")
    assert result["passed"] and len(result["reports"]) == 2
    assert {report["context_rows"] for report in result["reports"]} == {70, 102}
    assert all(report["gradient_norm"] > 0 and report["no_optimizer_update"] for report in result["reports"])
    assert_nested_equal(initial, model.state_dict())


def test_context_cap_preserves_every_original_query_and_class_support():
    labels = np.arange(60003) % 7
    available_positions = np.arange(256, len(labels))
    uncapped = runner.split_positions(labels, available_positions, 71)
    capped = runner.split_positions(labels, available_positions, 71, context_row_cap=20000)
    assert len(capped["context"]) == 20000
    np.testing.assert_array_equal(capped["query"], uncapped["query"])
    assert set(capped["context"]) <= set(uncapped["context"])
    assert not set(capped["context"]) & set(capped["query"])
    assert set(labels[capped["context"]]) == set(range(7))
    small = runner.split_positions(labels, np.arange(1000), 71, context_row_cap=20000)
    assert len(small["context"]) == 800 and len(small["query"]) == 200


def test_reuse_prepared_bank_changes_only_context_limit(tmp_path, monkeypatch):
    options = args(tmp_path / "capped")
    options.context_row_cap = 20000
    options.reuse_prepared_dir = tmp_path / "uncapped"
    entry = dict(family="large", panel="train", full_rows=60000, context_rows=47795,
        query_rows=11949, fitting_positions=torch.arange(59744), probe_query_positions=torch.arange(59744, 60000))
    runner.pilot.atomic_save(options.reuse_prepared_dir / "sources.pt", dict(entries=[entry]))
    source_manifest = dict(fingerprint="ea1185039ddb00a6f2aad0b5b79147802da34a03a8cdc82e13b86f0e91fe130e",
        source_fingerprint=runner.SOURCE_FINGERPRINT, catalog_fingerprint=runner.CATALOG_FINGERPRINT,
        counts=dict(train=1), backbone=dict(path="fixture"), previous_checkpoints={})
    monkeypatch.setattr(runner, "verified_json", lambda *arguments: source_manifest)
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", dict(train=1))
    runner.prepare_from_locked(options, dict(context_row_cap=20000))
    saved = torch.load(options.output_dir / "sources.pt", weights_only=True)["entries"][0]
    assert saved["context_rows"] == 20000 and saved["query_rows"] == entry["query_rows"]
    assert saved["nominal_context_rows"] == entry["context_rows"]
    torch.testing.assert_close(saved["fitting_positions"], entry["fitting_positions"])
    torch.testing.assert_close(saved["probe_query_positions"], entry["probe_query_positions"])


def test_training_query_sampling_preserves_context_and_complete_evaluation(tmp_path):
    source = source_family("training", rows=12000, mixed=False)
    cache_path = tmp_path / "source.pt"
    runner.pilot.atomic_save(cache_path, source)
    entry = dict(family="training", panel="train", source_group="training", full_rows=12000,
        cache_path=str(cache_path), cache_sha256=runner.pilot.hash_file(cache_path),
        fitting_positions=torch.arange(11744), probe_query_positions=torch.arange(11744, 12000))
    cache = runner.prior.SourceCache([entry])
    complete = runner.episode_for(entry, cache, step=47, context_row_cap=4000)
    sampled = runner.episode_for(entry, cache, step=47, context_row_cap=4000, training_query_row_cap=2048)
    repeated = runner.episode_for(entry, cache, step=47, context_row_cap=4000, training_query_row_cap=2048)
    revisited = runner.episode_for(entry, cache, step=48, context_row_cap=4000, training_query_row_cap=2048)
    assert sampled["n_query"] == 2048 and sampled["query_pool_rows"] == complete["n_query"] == 2349
    torch.testing.assert_close(sampled["context_indices"], complete["context_indices"])
    torch.testing.assert_close(sampled["query_indices"], repeated["query_indices"])
    assert len(set(sampled["query_indices"].tolist())) == 2048
    assert set(sampled["query_indices"].tolist()) <= set(complete["query_indices"].tolist())
    assert not set(sampled["query_indices"].tolist()) & set(sampled["context_indices"].tolist())
    assert not set(sampled["query_indices"].tolist()) & set(range(11744, 12000))
    assert not torch.equal(sampled["query_indices"], revisited["query_indices"])
    # No step means evaluation, even on a training source; held-out probes stay reserved.
    evaluated = runner.episode_for(entry, cache, context_row_cap=4000, training_query_row_cap=2048)
    assert evaluated["n_query"] == evaluated["query_pool_rows"] == 2349
    probe = runner.episode_for(entry, cache, probe=True, context_row_cap=4000, training_query_row_cap=100)
    torch.testing.assert_close(probe["query_indices"], entry["probe_query_positions"])
    validation_entry = dict(entry, panel="validation")
    validation = runner.episode_for(validation_entry, cache, context_row_cap=4000, training_query_row_cap=2048)
    assert validation["n_query"] == 2400


def test_query_budget_requires_one_chunk_and_preserves_small_query_pools(tmp_path):
    settings = runner.specification(args(tmp_path, training_query_row_cap=7))
    assert settings["query_row_cap"] == 7
    with pytest.raises(ValueError, match="one positive query chunk"):
        runner.specification(args(tmp_path, training_query_row_cap=8))
    with pytest.raises(ValueError, match="one positive query chunk"):
        runner.specification(args(tmp_path, training_query_row_cap=0))
    source = source_family("small", rows=96, mixed=False)
    cache_path = tmp_path / "small.pt"
    runner.pilot.atomic_save(cache_path, source)
    entry = dict(family="small", panel="train", source_group="small", cache_path=str(cache_path),
        cache_sha256=runner.pilot.hash_file(cache_path), fitting_positions=torch.arange(88))
    cache = runner.prior.SourceCache([entry])
    complete = runner.episode_for(entry, cache, step=1)
    sampled = runner.episode_for(entry, cache, step=1, training_query_row_cap=2048)
    torch.testing.assert_close(sampled["query_indices"], complete["query_indices"])


@pytest.mark.parametrize("parent_step,validate_imported", [(46, True), (142, False)])
def test_continuation_validation_schedule_and_sampled_query_resume(tmp_path, monkeypatch, parent_step, validate_imported):
    source = source_family("training", rows=96, mixed=False)
    cache_path = tmp_path / "source.pt"
    runner.pilot.atomic_save(cache_path, source)
    entry = dict(family="training", panel="train", source_group="training", full_rows=96,
        cache_path=str(cache_path), cache_sha256=runner.pilot.hash_file(cache_path),
        fitting_positions=torch.arange(88), probe_query_positions=torch.arange(88, 96))
    parent_model = trained_model()
    optimizer = torch.optim.AdamW(parent_model.parameters(), lr=.0003, weight_decay=1e-4)
    sum(parameter.square().sum() for parameter in parent_model.parameters()).backward()
    optimizer.step()
    parent_state = dict(fingerprint="parent", step=parent_step, episode_position=int(validate_imported), query_offset=0,
        episode_loss_sum=0., update_loss_sum=.25, record=681, pending_validation=False,
        best_score=float("inf"), best_step=0, best_model=runner.pilot.state_cpu(runner.prior.new_model()),
        model=runner.pilot.state_cpu(parent_model), optimizer=optimizer.state_dict(),
        gradients={name: torch.ones_like(parameter) for name, parameter in parent_model.named_parameters()},
        coverage={"training": {role: torch.zeros(96, dtype=torch.bool) for role in ("context", "query")}})
    parent_dir = tmp_path / "parent"
    runner.pilot.atomic_save(parent_dir / "state.pt", parent_state)
    locked = dict(parent_dir=str(parent_dir), state_sha256=runner.pilot.hash_file(parent_dir / "state.pt"),
        parent_fingerprint="parent", completed_updates=parent_step, validate_imported_checkpoint=validate_imported)
    manifest = dict(fingerprint="continuation", continuation=locked)
    options = args(tmp_path / "continued", steps=parent_step + 2, evaluate_every=320,
        training_query_row_cap=7, resume=True, max_records=1)
    imported = runner.imported_state(options, manifest)
    assert_nested_equal(imported["model"], parent_state["model"])
    assert_nested_equal(imported["optimizer"], parent_state["optimizer"])
    assert imported["step"] == parent_step and imported["episode_position"] == 0 and imported["record"] == 0
    assert all(gradient is None for gradient in imported["gradients"].values())
    validation_steps = []

    def validation_before_updates(options, backbone, model, entries, manifest, step, deadline):
        validation_steps.append(step)
        if step == parent_step:
            assert_nested_equal(runner.pilot.state_cpu(model), parent_state["model"])
        return dict(mean_nll=1., native_mean_nll=1.)

    monkeypatch.setattr(runner, "setup", lambda options: (tiny(), runner.prior.new_model(), [entry], manifest))
    monkeypatch.setattr(runner, "validation", validation_before_updates)
    runner.pilot.json_write(options.output_dir / "profile/complete.json", dict(fingerprint="continuation", passed=True))
    assert not runner.train(options)
    options.max_records = None
    assert runner.train(options)
    assert validation_steps == ([parent_step, parent_step + 2] if validate_imported else [parent_step + 2])
    records = [json.loads(line) for line in (options.output_dir / "chunks.jsonl").read_text().splitlines()]
    assert len(records) == 8 and {record["step"] for record in records} == {parent_step + 1, parent_step + 2}
    assert all(record["query_start"] == 0 and record["query_stop"] == 7 and record["query_pool_rows"] == 18 for record in records)
    # The old state remains byte-for-byte unchanged after continuation and resume.
    assert runner.pilot.hash_file(parent_dir / "state.pt") == locked["state_sha256"]
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.imported_state(options, dict(manifest, continuation=dict(locked, state_sha256="wrong")))


@pytest.mark.parametrize("bounded", [False, True])
def test_continuation_lock_rejects_optimizer_or_model_changes(tmp_path, monkeypatch, bounded):
    parent_dir = tmp_path / "parent"
    options = args(tmp_path / "new", continue_from_dir=parent_dir, bounded_continuation=bounded)
    settings = dict(lr=.0003, context_row_cap=20000, query_chunk_rows=2048, query_row_cap=2048,
        evaluate_every=32, code_hashes={"model.py": "unchanged", "scripts/joint_preprocessing_fraction_context.py": "new"})
    parent_settings = dict(settings, query_row_cap=2048 if bounded else None, evaluate_every=32 if bounded else 256,
        code_hashes=dict(settings["code_hashes"], **{"scripts/joint_preprocessing_fraction_context.py": "old"}))
    parent_manifest = dict(fingerprint="parent", settings=parent_settings)
    monkeypatch.setattr(runner, "verified_json", lambda *arguments: parent_manifest)
    monkeypatch.setattr(runner, "BOUNDED_PARENT_FINGERPRINT" if bounded else "CONTINUATION_FINGERPRINT", "parent")
    parent_step = 142 if bounded else 46
    runner.pilot.atomic_save(parent_dir / "state.pt", dict(fingerprint="parent", step=parent_step,
        episode_position=0 if bounded else 1, query_offset=0, pending_validation=False, gradients={"head": None}))
    runner.pilot.json_write(parent_dir / "profile/complete.json", dict(fingerprint="parent", passed=True))
    monkeypatch.setattr(runner, "BOUNDED_PARENT_STATE_SHA256" if bounded else "CONTINUATION_STATE_SHA256",
        runner.pilot.hash_file(parent_dir / "state.pt"))
    locked = runner.lock_continuation(options, settings)
    assert locked["completed_updates"] == parent_step
    assert locked["validate_imported_checkpoint"] is not bounded
    with pytest.raises(ValueError, match="setting change: lr"):
        runner.lock_continuation(options, dict(settings, lr=.001))
    with pytest.raises(ValueError, match="model/dependency changed"):
        runner.lock_continuation(options, dict(settings, code_hashes=dict(settings["code_hashes"], **{"model.py": "changed"})))
    if bounded:
        with pytest.raises(ValueError, match="setting change: query_row_cap"):
            runner.lock_continuation(options, dict(settings, query_row_cap=256))


def test_continuation_reuses_passed_gpu_shape_check_without_another_backward(tmp_path, monkeypatch):
    parent_profile_path = tmp_path / "parent/profile/complete.json"
    runner.pilot.json_write(parent_profile_path, dict(fingerprint="parent", passed=True, reports=[dict(context_rows=20000)]))
    locked = dict(parent_dir=str(tmp_path / "parent"), parent_fingerprint="parent",
        parent_profile_sha256=runner.pilot.hash_file(parent_profile_path))
    manifest = dict(fingerprint="continuation", continuation=locked)
    monkeypatch.setattr(runner, "setup", lambda options: (None, None, [], manifest))
    options = args(tmp_path / "continued")
    runner.profile(options)
    inherited = runner.previous.read(options.output_dir / "profile/complete.json")
    assert inherited["passed"] and inherited["fingerprint"] == "continuation"
    assert inherited["reused_parent_fingerprint"] == "parent"
    assert inherited["reports"] == [dict(context_rows=20000)]


def test_bounded_specification_requires_exact_stop_and_validation_schedule(tmp_path):
    options = args(tmp_path, continue_from_dir=tmp_path / "parent", reuse_prepared_dir=tmp_path / "bank",
        bounded_continuation=True, training_query_row_cap=7, steps=320, evaluate_every=320)
    settings = runner.specification(options)
    assert settings["bounded_continuation"] and "step142" in settings["initialization"]
    assert settings["steps"] == settings["evaluate_every"] == 320
    options.evaluate_every = 32
    with pytest.raises(ValueError, match="validate at320 only"):
        runner.specification(options)


def test_bounded_pipeline_stops_after_validation_without_catalog_inference(tmp_path, monkeypatch):
    options = args(tmp_path, command="pipeline", bounded_continuation=True)
    monkeypatch.setattr(runner, "parse_args", lambda: options)
    monkeypatch.setattr(runner, "profile", lambda options: None)
    monkeypatch.setattr(runner, "train", lambda options: True)
    monkeypatch.setattr(runner, "evaluate", lambda options: pytest.fail("bounded allocation must stop after validation"))
    runner.main()
