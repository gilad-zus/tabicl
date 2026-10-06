import argparse
import copy
import random

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_related_transfer as runner
from tests.test_joint_preprocessing_real_meta_continuation import (
    Backbone, assert_nested_equal, initial_model, read_rows, source_family,
)


@pytest.fixture(autouse=True)
def one_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def args_for(root, **changes):
    values = dict(output_dir=root, bank_dir=root / "old", source_dir=root / "source", group="clinical",
                  group_manifest=runner.DEFAULT_GROUPS, continuation_seed=0, device="cpu", checkpoint=None,
                  steps=2, evaluate_every=1, save_every=1, lr=.0003, resume=False, max_steps=None)
    values.update(changes)
    return argparse.Namespace(**values)


def declared_metadata(groups):
    return dict(families={p: [dict(name=n, source_group=n, inputs_sha256=n)
                for g in runner.GROUPS for n in groups["groups"][g][p]] for p in ("train", "validation", "test")})


def test_actual_declared_groups_are_independent_and_available():
    groups = runner.previous.read(runner.DEFAULT_GROUPS)
    candidates = runner.previous.read(runner.DEFAULT_GROUPS.with_name("joint_preprocessing_real_meta_candidates_20261004.json"))
    manifest = dict(families={p: [dict(e, inputs_sha256=e["name"]) for e in entries]
                    for p, entries in candidates["candidates"].items()})
    available = runner.validate_groups(groups, manifest)
    selected = {n for g in runner.GROUPS for p in ("train", "validation", "test") for n in groups["groups"][g][p]}
    assert len(selected) == 14
    assert all(n in available for n in selected)


@pytest.mark.parametrize("duplicate", ["name", "source_group", "inputs_sha256"])
def test_source_overlap_is_rejected_even_under_different_names(duplicate):
    groups = runner.previous.read(runner.DEFAULT_GROUPS)
    manifest = declared_metadata(groups)
    if duplicate == "name":
        groups["groups"]["clinical"]["test"][0] = groups["groups"]["clinical"]["train"][0]
    else:
        manifest["families"]["test"][0][duplicate] = manifest["families"]["train"][0][duplicate]
    with pytest.raises(ValueError, match="duplicate/related"):
        runner.validate_groups(groups, manifest)


def test_paired_training_schedule_equalizes_rows_and_uses_all_four_sources(monkeypatch):
    a = [source_family(f"a{i}", rows=64 + i) for i in range(4)]
    b = [source_family(f"b{i}", rows=96 + i) for i in range(4)]
    monkeypatch.setattr(runner.core, "shape_for_step", lambda seed, step: (256, .7))
    for step in (1, 7, 1024):
        aa, bb = runner.training_batch(a, 64, step), runner.training_batch(b, 64, step)
        assert {e["family"] for e in aa} == {f"a{i}" for i in range(4)}
        assert {e["family"] for e in bb} == {f"b{i}" for i in range(4)}
        for x, y in zip(aa, bb, strict=True):
            assert (x["source_seed"], x["task_id"], x["n_context"], x["n_query"]) == (y["source_seed"], y["task_id"], y["n_context"], y["n_query"])
            assert x["actual_rows"] == y["actual_rows"] == 64
            assert not set(x["context_indices"].tolist()) & set(x["query_indices"].tolist())


@pytest.fixture
def fixture(monkeypatch):
    groups = runner.previous.read(runner.DEFAULT_GROUPS)
    manifest = dict(fingerprint="fixed-transfer", settings={}, training_row_cap=32, groups=groups, banks={})
    values = {}
    for group in runner.GROUPS:
        families = [source_family(n, rows=40) for n in groups["groups"][group]["train"]]
        values[f"{group}_train"] = families
        values[f"{group}_probe"] = [runner.bank.sample_real_episode(f, 32, .7, 9) for f in families]
    for role in ("validation", "test"):
        values[f"real_{role}"] = [runner.bank.sample_real_episode(source_family(n, rows=40), 32, .7, seed)
            for g in runner.GROUPS for n in groups["groups"][g][role] for seed in (0, 1)]
    manifest["banks"] = {p: dict(count=len(v)) for p, v in values.items()}
    loaded = []
    initial = initial_model()
    monkeypatch.setattr(runner, "checked_manifest", lambda _: manifest)
    monkeypatch.setattr(runner, "setup", lambda _: (Backbone(), initial, manifest, torch.device("cpu")))
    def load(root, m, panel):
        loaded.append(panel)
        return values[panel]
    monkeypatch.setattr(runner, "load_bank", load)
    monkeypatch.setattr(runner.core, "shape_for_step", lambda seed, step: (32, .7))
    return manifest, initial, values, loaded


def test_resume_preserves_model_optimizer_rng_learning_logs_and_excludes_test(tmp_path, fixture):
    full, interrupted = tmp_path / "full", tmp_path / "resume"
    original = copy.deepcopy(fixture[1].state_dict())
    runner.train(args_for(full))
    runner.train(args_for(interrupted, max_steps=1))
    random.random()
    np.random.rand(3)
    torch.rand(3)
    runner.train(args_for(interrupted, resume=True))
    aa, bb = [torch.load(runner.core.run_dir(p, "clinical", 0) / "state.pt", weights_only=True) for p in (full, interrupted)]
    for key in ("model", "optimizer", "best_model", "best_step", "best_score", "rng", "other_rng"):
        assert_nested_equal(aa[key], bb[key])
    folder = runner.core.run_dir(interrupted, "clinical", 0)
    assert len(read_rows(folder / "presentations.csv")) == 8
    assert len(read_rows(folder / "learning.csv")) == 6
    assert "real_test" not in fixture[3]
    assert_nested_equal(fixture[1].state_dict(), original)


def test_test_bank_not_loaded_until_both_runs_lock_and_report_is_idempotent(tmp_path, fixture):
    root = tmp_path / "run"
    args = args_for(root)
    runner.train(args)
    with pytest.raises(FileNotFoundError):
        runner.test(args)
    assert "real_test" not in fixture[3]
    runner.train(args_for(root, group="financial"))
    runner.test(args)
    report = runner.previous.read(root / "complete.json")
    assert len(report["summaries"]["final/clinical"]["names"]) == 2
    assert len(report["summaries"]["final/financial"]["names"]) == 2
    assert report["summaries"]["final/pooled_related_versus_cross"]["tasks"] == 4
    runner.test(args)
    with pytest.raises(ValueError, match="locked"):
        runner.train(args_for(root, resume=True))
    # A modified checkpoint is rejected before any further inference.
    (runner.core.run_dir(root, "financial", 0) / "final.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.test(args)


def test_missing_or_duplicate_evaluation_partitions_rejected():
    rows = [dict(family="dataset", split_seed=0, blend_nll=.2), dict(family="dataset", split_seed=1, blend_nll=.4)]
    np.testing.assert_allclose(runner.grouped_scores(rows, "blend_nll", ["dataset"]), [.3])
    with pytest.raises(ValueError, match="duplicated"):
        runner.grouped_scores(rows + rows[:1], "blend_nll", ["dataset"])


def test_preparation_freezes_separate_banks_and_rejects_changed_settings(tmp_path, monkeypatch):
    root, old_root = tmp_path / "run", tmp_path / "old"
    groups = runner.previous.read(runner.DEFAULT_GROUPS)
    metadata = declared_metadata(groups)
    runner.pilot.json_write(old_root / "real_manifest.json", metadata)
    source = dict(arms=dict(repeated=dict(selected_step=5120)))
    old = dict(source_lock=source, real_manifest_sha256=runner.pilot.hash_file(old_root / "real_manifest.json"))
    old["fingerprint"] = runner.previous.digest(old)
    runner.pilot.json_write(old_root / "manifest.json", old)
    raw = [source_family(n, rows=40) for g in runner.GROUPS for p in ("train", "validation", "test") for n in groups["groups"][g][p]]
    monkeypatch.setattr(runner.core, "load_panel", lambda r, m, p: raw if p == "real_train" else [])
    monkeypatch.setattr(runner.core.transfer, "model_lock", lambda _: source)
    args = args_for(root, bank_dir=old_root)
    runner.prepare(args)
    runner.prepare(args)
    manifest = runner.checked_manifest(args)
    assert manifest["training_row_cap"] == 40
    assert manifest["banks"]["real_validation"]["count"] == 4
    assert manifest["banks"]["real_test"]["count"] == 8
    for g in runner.GROUPS:
        assert len(runner.load_bank(root, manifest, f"{g}_train")) == 4
    with pytest.raises(ValueError, match="changed"):
        runner.prepare(args_for(root, bank_dir=old_root, steps=3))
