import argparse
import copy
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts import joint_preprocessing_ensemble_objective as runner
from tests.test_joint_preprocessing_real_meta_continuation import (
    Backbone, assert_nested_equal, initial_model, read_rows, real_episode, source_family,
)
from tabicl._model.tabicl import TabICL


@pytest.fixture(autouse=True)
def one_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def naive_loss(backbone, model, e):
    context, views = runner.prepared_episode(e, torch.device("cpu"))
    slots = runner.numeric_slots(model, context)
    with torch.no_grad():
        ordinary = runner.ensemble_mean(backbone, views, e["n_classes"])
    learned = runner.ensemble_mean(backbone, views, e["n_classes"], slots)
    return F.cross_entropy((.5 * (ordinary + learned) / .9).flatten(0, 1), e["y_query"].flatten())


@pytest.mark.parametrize("mixed", [False, True])
def test_memory_bounded_backward_matches_full_autograd_for_every_parameter(mixed):
    e = real_episode(mixed=mixed)
    naive, staged = initial_model(), initial_model()
    backbone = Backbone()
    expected = naive_loss(backbone, naive, e)
    (expected * .25).backward()
    actual = runner.ensemble_backward(backbone, staged, e, .25)
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    for (name, a), (_, b) in zip(naive.named_parameters(), staged.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6, msg=name)
    assert all(p.grad is None for p in backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in staged.encoder.parameters())


@pytest.mark.parametrize("grouping", [False, True])
def test_actual_tabicl_deployment_value_and_backpropagation_parity(tmp_path, grouping):
    backbone = TabICL(max_classes=2, embed_dim=8, col_num_blocks=1, col_nhead=1,
        col_num_inds=2, col_feature_group=grouping, row_num_blocks=1, row_nhead=1,
        row_num_cls=1, icl_num_blocks=1, icl_nhead=1, col_ssmax=True,
        icl_ssmax=True, dropout=0., zero_init=False).train().requires_grad_(False)
    model, e = initial_model(), real_episode()
    runner.execution_audit(backbone, model, e, tmp_path, torch.device("cpu"))
    expected = naive_loss(backbone, model, e)
    expected.backward()
    staged = initial_model()
    runner.ensemble_backward(backbone, staged, e, 1.)
    for (_, a), (_, b) in zip(model.named_parameters(), staged.named_parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
    assert all(p.grad is None for p in backbone.parameters())


def test_all_views_preserve_categories_and_do_not_read_query_labels():
    e = real_episode()
    del e["y_query"]
    context, views = runner.prepared_episode(e, torch.device("cpu"))
    model = initial_model()
    slots = runner.numeric_slots(model, context)
    backbone = Backbone()
    learned = runner.ensemble_mean(backbone, views, 2, slots)
    deployed, count = runner.core.panel_logits(backbone, model, e, 8)
    torch.testing.assert_close(learned, deployed, rtol=1e-6, atol=1e-6)
    assert count == len(views) == 8
    for view, actual in zip(views, backbone.inputs[:8], strict=True):
        categories = sorted(set(range(view["x"].shape[-1])) - set(view["columns"].tolist()))
        assert categories
        torch.testing.assert_close(actual[..., categories], view["x"][..., categories], rtol=0, atol=0)


def args_for(root, **changes):
    args = dict(output_dir=root, bank_dir=root / "old_bank", source_dir=root / "source",
                arm="ensemble", continuation_seed=0, device="cpu", checkpoint=None, steps=2,
                evaluate_every=1, save_every=1, lr=.0003, resume=False, max_steps=None)
    args.update(changes)
    return argparse.Namespace(**args)


@pytest.fixture
def training_fixture(monkeypatch):
    initial = initial_model()
    families = [source_family(f"family{i}") for i in range(4)]
    manifest = dict(settings={"fixture": True}, episode_settings={},
        banks={p: dict(count=1) for p in runner.PANELS},
        source_lock=dict(arms=dict(repeated=dict(checkpoint_sha256="same-initial-source"))))
    loaded = []

    def setup(args):
        return Backbone(), initial, manifest, "fixed-pilot", torch.device("cpu")

    def load(root, manifest, panel):
        loaded.append(panel)
        assert "test" not in panel and "synthetic" not in panel
        return families if panel == "real_train" else [real_episode()]

    monkeypatch.setattr(runner, "setup", setup)
    monkeypatch.setattr(runner.core, "load_panel", load)
    monkeypatch.setattr(runner.core, "shape_for_step", lambda seed, step: (20, .5))
    return initial, manifest, loaded


@pytest.mark.parametrize("arm", runner.ARMS)
def test_resuming_pilot_preserves_optimizer_model_rng_and_reports(tmp_path, monkeypatch, training_fixture, arm):
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    runner.train(args_for(full, arm=arm))
    runner.train(args_for(interrupted, arm=arm, max_steps=1))
    folder = runner.core.run_dir(interrupted, arm, 0)
    with (folder / "training.csv").open("a", encoding="utf8") as handle:
        handle.write("99,0,0\n")
    random.random()
    np.random.random(4)
    torch.rand(4)
    runner.train(args_for(interrupted, arm=arm, resume=True))
    a = torch.load(runner.core.run_dir(full, arm, 0) / "state.pt", weights_only=True)
    b = torch.load(folder / "state.pt", weights_only=True)
    for key in ("model", "optimizer", "best_model", "best_step", "best_score", "clipped", "rng", "other_rng"):
        assert_nested_equal(a[key], b[key])
    assert {int(s["step"]) for s in b["optimizer"]["state"].values()} == {2}
    assert len(read_rows(folder / "presentations.csv")) == 8
    assert [int(row["step"]) for row in read_rows(folder / "training.csv")] == [1, 2]
    assert [(int(r["step"]), r["panel"]) for r in read_rows(folder / "learning.csv")] == [
        (step, panel) for step in (0, 1, 2) for panel in runner.PANELS]
    assert set(training_fixture[2]) == {"real_train", *runner.PANELS}


def test_shared_references_same_episode_schedule_and_completed_report(tmp_path, training_fixture):
    root = tmp_path / "pilot"
    initial, manifest, _ = training_fixture
    original = copy.deepcopy(initial.state_dict())
    runner.pilot.json_write(root / "manifest.json", dict(manifest, fingerprint="fixed-pilot"))
    for arm in runner.ARMS:
        runner.train(args_for(root, arm=arm))
    assert_nested_equal(initial.state_dict(), original)
    assert len(list((root / "references").glob("*.pt"))) == len(runner.PANELS)
    a, b = [read_rows(runner.core.run_dir(root, arm, 0) / "presentations.csv") for arm in runner.ARMS]
    assert a == b
    runner.report(args_for(root))
    report = runner.previous.read(root / "complete.json")
    assert report["continuation_seeds"] == [0]
    assert set(report["runs"]) == set(runner.ARMS)
    assert set(report["ensemble_versus_single"]) == {f"{c}/{p}" for c in ("selected", "final") for p in runner.PANELS}
    runner.report(args_for(root))
    with pytest.raises(ValueError, match="locked"):
        runner.train(args_for(root, resume=True))


def test_only_one_seed_and_canonical_line_endings(tmp_path):
    with pytest.raises(ValueError, match="seed zero"):
        runner.settings(args_for(tmp_path, continuation_seed=1))
    path = tmp_path / "example.py"
    path.write_bytes(b"a\r\nb\r\n")
    first = runner.canonical_hash(path)
    path.write_bytes(b"a\nb\n")
    assert runner.canonical_hash(path) == first


def test_preparation_freezes_existing_inputs_and_never_opens_tests(tmp_path, monkeypatch):
    args = args_for(tmp_path / "pilot", bank_dir=tmp_path / "bank")
    source = dict(arms=dict(repeated=dict(selected_step=5120)))
    bank = dict(source_lock=source, settings={}, banks={})
    for panel in ("real_train", *runner.PANELS):
        path = args.bank_dir / f"{panel}.pt"
        runner.pilot.atomic_save(path, {"fixture": True})
        bank["banks"][panel] = dict(path=path.name, count=1, sha256=runner.pilot.hash_file(path))
    # Deliberately absent test file: no filesystem access/deserialization permitted.
    bank["banks"]["real_test"] = dict(path="DO_NOT_OPEN.pt", count=60, sha256="not-needed")
    bank["fingerprint"] = runner.previous.digest(bank)
    runner.pilot.json_write(args.bank_dir / "manifest.json", bank)
    monkeypatch.setattr(runner.core.transfer, "model_lock", lambda _: source)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("prepare must not deserialize any bank"))
    runner.prepare(args)
    runner.prepare(args)
    manifest = runner.previous.read(args.output_dir / "manifest.json")
    assert set(manifest["banks"]) == {"real_train", *runner.PANELS}
    assert manifest["settings"]["continuation_seeds"] == [0]
    with pytest.raises(ValueError, match="changed"):
        runner.prepare(args_for(args.output_dir, bank_dir=args.bank_dir, steps=3))
    (args.bank_dir / "real_train.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bank hash changed"):
        runner.prepare(args)
