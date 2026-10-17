import argparse
import copy

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_catalog_inference as runner
from tests.test_joint_preprocessing_real_meta_continuation import source_family, real_episode
from tests.test_joint_preprocessing_residual_conditioning import tiny


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_shared_native_predictions_match_original_for_trained_models():
    episode, backbone = real_episode(), tiny()
    model = runner.context_run.new_model().eval().requires_grad_(False)
    with torch.no_grad():
        model.affine_head.weight.fill_(.003)
        model.spline_head.weight.fill_(.002)
    before = copy.deepcopy(model.state_dict())
    result = runner.predict_methods(backbone, {"learned": model, "identity": None}, episode)
    torch.testing.assert_close(result["native16"], runner.residual.episode_logits(backbone, None, episode), rtol=0, atol=0)
    torch.testing.assert_close(result["learned"], runner.residual.episode_logits(backbone, model, episode), rtol=0, atol=0)
    assert torch.equal(result["identity"], result["native16"])
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("rows,expected_contexts,queries", [(800, {"512"}, 288), (2500, {"512", "2048"}, 452), (5000, {"512", "2048"}, 1024)])
def test_actual_context_sizes_common_queries_and_deterministic_splits(rows, expected_contexts, queries):
    source = source_family(rows=rows)
    positions = runner.row_partitions(source, None, 0)
    repeat = runner.row_partitions(source, None, 0)
    assert set(positions)-{"query"} == expected_contexts
    assert len(positions["query"]) == queries
    for size in expected_contexts:
        assert len(positions[size]) == int(size)
        assert not set(positions[size]) & set(positions["query"])
        np.testing.assert_array_equal(positions[size], repeat[size])
    if "2048" in positions:
        assert set(positions["512"]) < set(positions["2048"])
    other = runner.row_partitions(source, None, 1)
    assert not np.array_equal(positions["query"], other["query"])


def test_training_reservations_never_enter_contexts_on_either_split():
    source = source_family(rows=5000)
    fitting, queries = runner.context_run.partition_source(np.asarray(source["labels"]), 91)
    reservation = dict(fitting_positions=torch.as_tensor(fitting), probe_query_positions=torch.as_tensor(queries))
    for split in (0, 1):
        positions = runner.row_partitions(source, reservation, split)
        np.testing.assert_array_equal(positions["query"], queries)
        assert set(positions["2048"]) <= set(fitting)
        assert not set(positions["2048"]) & set(queries)


def test_exposure_priority_and_conservative_history():
    locked = dict(training_groups=["group-a"], validation_groups=["group-b"])
    registry = dict(historical_groups=["groupc"], historical_names=["alias"])
    assert runner.exposure_for(dict(name="source", source_group="group-a"), locked, registry) == "training_source"
    assert runner.exposure_for(dict(name="source", source_group="group-b"), locked, registry) == "validation_source"
    assert runner.exposure_for(dict(name="source", source_group="group-c"), locked, registry) == "previously_declared_source"
    assert runner.exposure_for(dict(name="source", source_group="new", aliases=["alias"]), locked, registry) == "previously_declared_source"
    assert runner.exposure_for(dict(name="unlisted", source_group="new"), locked, registry) == "catalog_only_source"


def test_dataset_summary_averages_splits_and_separates_seen_sources():
    records = []
    for family, exposure, candidate_losses in (("seen", "training_source", [2., 2.]),
                                                ("unseen", "catalog_only_source", [.4, .6])):
        for split, loss in enumerate(candidate_losses):
            scores = {name: dict(nll=1. if name == "native16" else loss,
                accuracy=.5, macro_f1=.5, binary_auc="") for name in runner.METHODS}
            records.append(dict(family=family, exposure=exposure, split=split, context_rows=512, scores=scores))
    result = runner.aggregate(records)
    unseen = result["all_unseen_training_sources/512/small_final"]
    assert unseen["datasets"] == 1 and unseen["mean_nll_reduction"] == .5
    assert unseen["wins"] == 1
    assert result["all_sources/512/small_final"]["datasets"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        runner.aggregate(records[:-1])


def test_model_lock_rejects_checkpoint_tampering(tmp_path):
    args = argparse.Namespace(source_dir=tmp_path)
    files = {}
    for filename in ("sources.pt", "eligibility.json"):
        path = tmp_path / filename
        if filename.endswith("json"):
            runner.pilot.json_write(path, dict(accepted=[]))
        else:
            path.write_bytes(b"source rows")
        files[filename] = runner.pilot.hash_file(path)
    runner.pilot.json_write(tmp_path / "manifest.json", dict(fingerprint="source", files=files, original_allocation_sha256="original"))
    runner.pilot.json_write(tmp_path / "backbone_lock.json", dict(path="checkpoint", sha256="hash"))
    runs = {}
    for arm in ("small", "large"):
        folder = tmp_path / "runs" / f"{arm}_seed0"
        folder.mkdir(parents=True)
        for name in ("final", "selected"):
            (folder / f"{name}.pt").write_bytes(name.encode())
        runs[arm] = dict(steps=2048, selected_step=0,
            **{f"{name}_sha256": runner.pilot.hash_file(folder / f"{name}.pt") for name in ("final", "selected")})
        runner.pilot.json_write(folder / "complete.json", runs[arm])
    runner.pilot.json_write(tmp_path / "complete.json", dict(fingerprint="source", runs=runs))
    assert len(runner.source_lock(args)["checkpoints"]) == 4
    (tmp_path / "runs/small_seed0/final.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.source_lock(args)


def test_preparation_reuses_full_cache_and_locks_all_eligible_contexts(tmp_path, monkeypatch):
    source = source_family(rows=3000, mixed=False)
    entry = dict(source="openml", data_id=123, name="source", source_group="source")
    cache = tmp_path / "cache"
    source_path = cache / f"{runner.hashlib.sha256(b'openml:123').hexdigest()[:20]}.pt"
    runner.pilot.atomic_save(source_path, dict(entry=entry, columns=source["columns"], labels=source["labels"], description="real observations"))
    runner.pilot.json_write(tmp_path / "catalog.json", dict(candidates=[entry]))
    runner.pilot.json_write(tmp_path / "history.json", dict(historical_groups=[], historical_names=[], input_hash_groups={}))
    runner.pilot.json_write(tmp_path / "original/banks_manifest.json", dict(selection=dict(large=[], validation=[])))
    runner.pilot.atomic_save(tmp_path / "source/sources.pt", dict(entries=[]))
    args = argparse.Namespace(output_dir=tmp_path / "output", source_dir=tmp_path / "source",
        original_bank_dir=tmp_path / "original", registry=tmp_path / "history.json",
        candidate_manifest=tmp_path / "catalog.json", source_cache_dirs=[cache], cache_dir=tmp_path / "download")
    monkeypatch.setattr(runner, "specification", lambda args: dict(expected_revision=None, code_hashes={}))
    monkeypatch.setattr(runner, "source_lock", lambda args: dict(training_groups=[], validation_groups=[],
        original_allocation_sha256=runner.pilot.hash_file(args.original_bank_dir / "banks_manifest.json")))
    monkeypatch.setattr(runner.residual.bank, "load_candidate", lambda *args: pytest.fail("cached source downloaded"))
    runner.prepare(args)
    manifest = runner.checked_manifest(args)
    assert len(manifest["accepted"]) == 1
    accepted = manifest["accepted"][0]
    assert accepted["contexts"] == [512, 2048] and accepted["queries"] == 952
    assert accepted["exposure"] == "catalog_only_source"
    runner.prepare(args)
    assert runner.checked_manifest(args) == manifest


def test_inference_resume_skips_predictions_and_keeps_all_checkpoint_labels(tmp_path, monkeypatch):
    episode = real_episode()
    episodes = [dict(episode, split=split, context_rows=512) for split in (0, 1)]
    episode_path = tmp_path / "output/episodes/source.pt"
    runner.pilot.atomic_save(episode_path, dict(episodes=episodes))
    source = tmp_path / "source"
    locks = {}
    for method in runner.METHODS[1:]:
        arm, kind = method.split("_")
        step = 0 if method == "large_selected" else 1
        locks[method] = dict(step=step)
        runner.pilot.atomic_save(source / "runs" / f"{arm}_seed0" / f"{kind}.pt",
            dict(step=step, model=runner.context_run.new_model().state_dict()))
    manifest = dict(fingerprint="frozen", counts=dict(catalog_only_source=1),
        source_lock=dict(backbone=dict(path="checkpoint", sha256="hash"), checkpoints=locks),
        accepted=[dict(episode_file="episodes/source.pt", episode_sha256=runner.pilot.hash_file(episode_path),
            family="source", source_group="source", exposure="catalog_only_source")])
    monkeypatch.setattr(runner, "checked_manifest", lambda args: manifest)
    monkeypatch.setattr(runner.pilot, "load_frozen", lambda *args: (tiny(), "checkpoint", "hash"))
    calls = []
    def predictions(backbone, models, episode):
        calls.append(episode["split"])
        return {method: torch.zeros(1, episode["n_query"], 2) for method in runner.METHODS}
    monkeypatch.setattr(runner, "predict_methods", predictions)
    args = argparse.Namespace(output_dir=tmp_path / "output", source_dir=source, device="cpu", wall_minutes=1.)
    runner.infer(args)
    runner.infer(args)
    assert calls == [0, 1]
    report = runner.previous.read(args.output_dir / "complete.json")
    assert report["episodes"] == 2
    assert len(report["checkpoints"]) == 4
    assert report["comparisons"]["catalog_only_source/512/large_selected"]["ties"] == 1
