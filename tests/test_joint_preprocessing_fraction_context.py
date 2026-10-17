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
    monkeypatch.setattr(runner, "validation", lambda *arguments: dict(mean_nll=1.))
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
