import argparse
import copy
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts import joint_preprocessing_residual_conditioning as runner
from tabicl._model.tabicl import TabICL
from tabicl._hyperspline.statistics import summarize_context
from tests.test_joint_preprocessing_real_meta_continuation import (
    Backbone, assert_nested_equal, real_episode, source_family, read_rows,
)


@pytest.fixture(autouse=True)
def one_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def tiny(grouping="same"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        return TabICL(max_classes=2, embed_dim=8, col_num_blocks=1, col_nhead=1,
            col_num_inds=2, col_feature_group=grouping, row_num_blocks=1, row_nhead=1,
            row_num_cls=2, icl_num_blocks=1, icl_nhead=1, col_ssmax=True,
            icl_ssmax=True, dropout=0., zero_init=False).train().requires_grad_(False)


def model_for(arm, trained=False):
    model = runner.new_model(arm, 8)
    if trained:
        with torch.no_grad():
            model.affine_head.weight.fill_(.001)
            model.neural_last_head.weight.normal_(0, .01)
            model.spline_head.weight.normal_(0, .005)
            model.mix_gate_head.bias.fill_(.01)
    return model


@pytest.mark.parametrize("arm", runner.ARMS)
@pytest.mark.parametrize("grouping", [False, "same", "valid"])
def test_zero_residual_is_exact_native16_and_actual_tabicl_parity(tmp_path, arm, grouping):
    backbone, model, episode = tiny(grouping), model_for(arm), real_episode()
    episode.pop("task_id")
    episode.pop("source_seed")
    # A finite query outlier and existing missing/category cells must survive.
    episode["x_query"][0, 0, 0] = 1e20
    context, views = runner.prepared_episode(episode, torch.device("cpu"))
    values, _ = runner.numeric_views(model, backbone, context, views)
    assert len(views) == 16 and sum(v["adapted"] for v in views) == 8
    assert {v["slot"]: sum(w["adapted"] for w in views if w["slot"] == v["slot"]) for v in views} == {0: 4, 1: 4}
    for view, value in zip(views, values, strict=True):
        if value is not None:
            torch.testing.assert_close(value, view["x"].index_select(-1, view["columns"]), rtol=0, atol=0)
            actual = view["x"].index_copy(-1, view["columns"], value)
            torch.testing.assert_close(actual, view["x"], rtol=0, atol=0)
    learned = runner.episode_logits(backbone, model, episode)
    ordinary, count = runner.objective.real.episode_logits(backbone, "ordinary", episode, 16)
    assert count == 16
    torch.testing.assert_close(learned, ordinary, rtol=0, atol=0)
    audit = runner.execution_audit(backbone, model, episode, tmp_path, torch.device("cpu"))
    assert audit["task_id"] is None and audit["exact_identity_default_amp_logits"]
    assert audit["post_two_updates_amp_finite"]
    assert all(p.grad is None for p in backbone.parameters())


@pytest.mark.parametrize("arm", runner.ARMS)
def test_staged_gradient_matches_full_autograd_for_every_parameter(arm):
    backbone, episode = tiny(), real_episode()
    model = model_for(arm, True)
    other = copy.deepcopy(model)
    context, views = runner.prepared_episode(episode, torch.device("cpu"))
    values, _ = runner.numeric_views(model, backbone, context, views)
    logits = runner.mean_logits(backbone, views, episode["n_classes"], values)
    expected = F.cross_entropy((logits / .9).flatten(0, 1), episode["y_query"].flatten())
    (expected * .25).backward()
    actual = runner.ensemble_backward(backbone, other, episode)
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-6)
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6, msg=name)
    assert all(p.grad is None for p in backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in other.encoder.parameters())


@pytest.mark.parametrize("arm", runner.ARMS)
def test_conditioning_never_reads_queries_and_keeps_categories_and_missing_values(arm):
    episode, backbone, model = real_episode(), tiny(), model_for(arm, True)
    first, views = runner.prepared_episode(episode, torch.device("cpu"))
    before = runner.generate(model, backbone, first)
    changed = copy.deepcopy(episode)
    changed["x_query"] = changed["x_query"] * 123 + 11
    changed["query_missing"] = ~changed["query_missing"]
    changed.pop("y_query")
    second, _ = runner.prepared_episode(changed, torch.device("cpu"))
    after = runner.generate(model, backbone, second)
    for key in before.__dataclass_fields__:
        a, b = getattr(before, key), getattr(after, key)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    values, _ = runner.numeric_views(model, backbone, first, views)
    for view, value in zip(views, values, strict=True):
        if value is None:
            continue
        original = view["x"].index_select(-1, view["columns"])
        torch.testing.assert_close(value[first["all_missing"]], original[first["all_missing"]], rtol=0, atol=0)
        actual = view["x"].index_copy(-1, view["columns"], value)
        cats = sorted(set(range(actual.shape[-1])) - set(view["columns"].tolist()))
        assert cats
        torch.testing.assert_close(actual[..., cats], view["x"][..., cats], rtol=0, atol=0)


def test_backbone_conditioner_sees_categories_and_shared_heads_are_paired():
    a, b = model_for("raw"), model_for("backbone")
    for key, value in a.state_dict().items():
        if not key.startswith("encoder.cell_encoder."):
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    backbone, e = tiny(), real_episode()
    context, _ = runner.prepared_episode(e, torch.device("cpu"))
    features = runner.frozen_context_features(backbone, context["full"], context["y"], context["positions"])
    other = context["full"].clone()
    cats = sorted(set(range(other.shape[-1])) - set(context["positions"].tolist()))
    other[..., cats] += 5
    changed = runner.frozen_context_features(backbone, other, context["y"], context["positions"])
    assert not torch.equal(features[1], changed[1])
    assert all(not v.requires_grad for v in features)
    stats = summarize_context(context["x"], context["missing"], context["y"])
    x = b.encoder(context["x"], context["y"], stats, context["missing"], features)
    y = b.encoder(context["x"], context["y"], stats, context["missing"], changed)
    assert not torch.equal(x, y)


def args_for(root, **changes):
    values = dict(output_dir=root, candidate_manifest=runner.DEFAULT_CANDIDATES, cache_dir=root / "cache",
        arm="raw", continuation_seed=0, device="cpu", checkpoint=None, steps=2,
        evaluate_every=1, save_every=1, lr=.0003, resume=False, max_steps=None,
        expected_revision=None, reuse_bank_dir=None)
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture
def fixture(monkeypatch):
    source_banks = dict(small=[source_family(f"f{i}") for i in range(4)], large=[source_family(f"f{i}") for i in range(8)])
    values = {f"{k}_train": v for k, v in source_banks.items()}
    episode = real_episode()
    episode.pop("task_id")
    episode.pop("source_seed")
    values.update({p: [episode] for p in runner.PANELS})
    manifest = dict(fingerprint="fixture", initial_sha256={a: "initial" for a in runner.ARMS},
        settings={}, banks={p: dict(count=len(v)) for p, v in values.items()})
    loaded = []
    monkeypatch.setattr(runner, "setup", lambda args: (tiny(), model_for(args.arm), manifest, torch.device("cpu")))
    monkeypatch.setattr(runner, "checked_manifest", lambda args: manifest)
    def load(root, manifest, panel):
        assert "test" not in panel and "synthetic" not in panel
        loaded.append(panel)
        return values[panel]
    monkeypatch.setattr(runner.bank, "load_bank", load)
    monkeypatch.setattr(runner.core, "shape_for_step", lambda seed, step: (20, .5))
    return manifest, loaded, episode


def train_fixture(args):
    runner.pilot.json_write(args.output_dir / "backbone_lock.json", dict(sha256="fixture"))
    runner.train(args)


@pytest.mark.parametrize("arm", runner.ARMS)
def test_resume_bit_exact_and_locks_completed_runs(tmp_path, fixture, arm):
    full, part = tmp_path / "full", tmp_path / "partial"
    train_fixture(args_for(full, arm=arm))
    train_fixture(args_for(part, arm=arm, max_steps=1))
    folder = runner.core.run_dir(part, arm, 0)
    with (folder / "training.csv").open("a", encoding="utf-8") as handle:
        handle.write("99,0,0\n")
    random.random()
    np.random.rand(3)
    torch.rand(4)
    runner.train(args_for(part, arm=arm, resume=True))
    a = torch.load(runner.core.run_dir(full, arm, 0) / "state.pt", weights_only=True)
    b = torch.load(folder / "state.pt", weights_only=True)
    for key in ("model", "optimizer", "best_model", "best_step", "best_score", "clipped", "rng", "other_rng"):
        assert_nested_equal(a[key], b[key])
    assert [int(r["step"]) for r in read_rows(folder / "training.csv")] == [1, 2]
    assert len(read_rows(folder / "presentations.csv")) == 8
    assert set(fixture[1]) == {"small_train", "large_train", *runner.PANELS}


def test_both_arms_share_sources_references_and_report(tmp_path, fixture):
    for arm in runner.ARMS:
        train_fixture(args_for(tmp_path, arm=arm))
    a, b = [read_rows(runner.core.run_dir(tmp_path, arm, 0) / "presentations.csv") for arm in runner.ARMS]
    for x, y in zip(a, b, strict=True):
        assert {k: v for k, v in x.items() if k != "domain"} == {k: v for k, v in y.items() if k != "domain"}
    assert len(list((tmp_path / "references").glob("*.pt"))) == len(runner.PANELS)
    runner.report(args_for(tmp_path))
    report = runner.previous.read(tmp_path / "complete.json")
    assert set(report["runs"]) == set(runner.ARMS)
    assert len(report["backbone_versus_raw"]) == 2 * len(runner.PANELS)
    runner.report(args_for(tmp_path))
    with pytest.raises(ValueError, match="locked"):
        runner.train(args_for(tmp_path, resume=True))


def test_preflight_covers_both_arms_and_keeps_saved_initials(tmp_path, fixture):
    runner.preflight(args_for(tmp_path, preflight_family=fixture[2]["family"], preflight_split_seed=fixture[2]["split_seed"]))
    result = runner.previous.read(tmp_path / "preflight" / "complete.json")
    assert result["both_passed"] and result["arms"] == list(runner.ARMS)
    assert not (tmp_path / "runs").exists()

def test_prepare_and_manifest_lock_data_initials_and_backbone(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    spec = dict(counts=dict(small=40, large=160, validation=25),
        candidate_sha256=runner.pilot.hash_file(runner.DEFAULT_CANDIDATES), banks={})
    for panel in ("small_train", "large_train", *runner.PANELS):
        path = source / "banks" / f"{panel}.pt"
        runner.pilot.atomic_save(path, dict(values=[]))
        spec["banks"][panel] = dict(path=path.relative_to(source).as_posix(), count=0, sha256=runner.pilot.hash_file(path))
    runner.pilot.json_write(source / "banks_manifest.json", spec)
    manifest = dict(banks_manifest_sha256=runner.pilot.hash_file(source / "banks_manifest.json"))
    manifest["fingerprint"] = runner.previous.digest(manifest)
    runner.pilot.json_write(source / "manifest.json", manifest)
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"fixture weights")
    sha = runner.pilot.hash_file(checkpoint)
    monkeypatch.setattr(runner.pilot, "load_frozen", lambda args, device: (tiny(), checkpoint, sha))
    args = args_for(root, reuse_bank_dir=source, source_fingerprint=manifest["fingerprint"])
    runner.prepare(args)
    locked = runner.checked_manifest(args)
    runner.prepare(args)
    assert runner.checked_manifest(args) == locked
    assert locked["source_fingerprint"] == manifest["fingerprint"]
    assert locked["backbone"]["embedding_dim"] == 8 and locked["no_test_bank"]
    assert locked["parameter_counts"]["backbone"] > locked["parameter_counts"]["raw"]
    assert not (root / "references").exists() and not (root / "runs").exists()
    with pytest.raises(ValueError, match="settings"):
        runner.checked_manifest(args_for(root, steps=3))
    (root / "initial_raw.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="initial weights"):
        runner.checked_manifest(args)
