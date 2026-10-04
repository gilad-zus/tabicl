import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import joint_preprocessing_real_transfer as real
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts import joint_preprocessing_zero_shot_comparison as synthetic
from scripts import joint_preprocessing_zero_shot_real_transfer as transfer
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def clear_cache(self):
        pass

    def forward(self, x, y_train, **kwargs):
        assert not torch.is_grad_enabled()
        values = x[:, y_train.shape[1]:, 0] * self.weight
        return torch.stack((-values, values), dim=-1)


def make_episode(mixed=False):
    rng = np.random.default_rng(104)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=300) for i in range(5)})
    frame["constant"] = 1.0
    if mixed:
        frame["category"] = pd.Categorical(np.resize(["a", "b", "c"], 300))
        frame.loc[::17, "x1"] = np.nan
    return real._episode(frame, np.arange(300) % 2, family="example", seed=0,
                         max_rows=256, test_fraction=.3)


@pytest.mark.parametrize("views", [8, 16])
def test_ordinary_logits_match_completed_synthetic_runner(views):
    e = make_episode()
    synthetic_episode = SyntheticEpisode(1, 1, e["x_context"], e["x_query"],
        e["y_context"], e["y_query"], e["n_classes"])
    expected, expected_views = synthetic.ordinary_logits(Backbone(), synthetic_episode, views)
    actual, actual_views = real.episode_logits(Backbone(), "ordinary", e, views)
    assert actual_views == expected_views == views
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_learned_logits_match_synthetic_path_and_do_not_read_query_labels():
    e = make_episode()
    assert not set(e["context_indices"].tolist()) & set(e["query_indices"].tolist())
    model = JointPreprocessor("joint").eval().requires_grad_(False)
    with torch.no_grad():
        model.affine_head.bias.fill_(.08)
    weights = copy.deepcopy(model.state_dict())
    synthetic_episode = SyntheticEpisode(1, 1, e["x_context"], e["x_query"],
        e["y_context"], e["y_query"], e["n_classes"])
    with torch.no_grad(), pilot.frozen_inference(Backbone()):
        expected = pilot.forward_views(Backbone(), model, synthetic_episode)
    e.pop("y_query")
    actual, views = real.episode_logits(Backbone(), model, e)
    assert views == 8
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_mixed_missing_predictions_and_blend_use_aligned_logits():
    e = make_episode(mixed=True)
    model = JointPreprocessor("joint").eval().requires_grad_(False)
    ordinary, _ = real.episode_logits(Backbone(), "ordinary", e)
    learned, _ = real.episode_logits(Backbone(), model, e)
    for logits in (ordinary, learned, .5*ordinary+.5*learned):
        scores = transfer.metrics(logits, e["y_query"], 2)
        assert np.isfinite(scores["nll"])
        assert 0 <= scores["accuracy"] <= 1
        assert 0 <= scores["auc"] <= 1


def build_source(root):
    pilot.json_write(root / "manifest.json", dict(experiment=dict(fixture=True)))
    manifest_hash = pilot.hash_file(root / "manifest.json")
    fp = synthetic.digest(dict(settings=dict(fixture=True), manifest_sha256=manifest_hash, backbone_hash="backbone"))
    pilot.json_write(root / "config.json", dict(fingerprint=fp, revision="training", backbone_hash="backbone"))
    choices = {}
    for arm in transfer.ARMS:
        arm_fp = synthetic.digest(dict(arm=arm))
        pilot.atomic_save(root / arm / "selected.pt", dict(fingerprint=arm_fp, step=1,
            model=JointPreprocessor("joint").state_dict()))
        checkpoint_hash = pilot.hash_file(root / arm / "selected.pt")
        pilot.json_write(root / arm / "complete.json", dict(experiment_fingerprint=fp,
            fingerprint=arm_fp, selected_step=1, selected_sha256=checkpoint_hash))
        choices[arm] = dict(alpha=.5, selected_step=1, checkpoint_sha256=checkpoint_hash,
            complete_sha256=pilot.hash_file(root / arm / "complete.json"))
    pilot.json_write(root / "lock.json", dict(fingerprint=fp, manifest_sha256=manifest_hash, arms=choices))
    pilot.json_write(root / "test_report/complete.json", dict(fingerprint=fp, lock_hash=pilot.hash_file(root / "lock.json")))


def test_lock_rejects_changed_checkpoint_and_changed_weight(tmp_path):
    source = tmp_path / "synthetic"
    build_source(source)
    assert set(transfer.model_lock(source)["arms"]) == set(transfer.ARMS)
    path = source / "fresh/selected.pt"
    original = path.read_bytes()
    path.write_bytes(original+b"modified")
    with pytest.raises(ValueError, match="model changed"):
        transfer.model_lock(source)
    path.write_bytes(original)
    locked = synthetic.read(source / "lock.json")
    locked["arms"]["fresh"]["alpha"] = .25
    pilot.json_write(source / "lock.json", locked)
    pilot.json_write(source / "test_report/complete.json", dict(fingerprint=locked["fingerprint"], lock_hash=pilot.hash_file(source / "lock.json")))
    with pytest.raises(ValueError, match="alpha=0.5"):
        transfer.model_lock(source)


def test_full_report_resumes_and_aggregates_families_before_bootstrap(tmp_path, monkeypatch):
    source, output = tmp_path / "synthetic", tmp_path / "real"
    build_source(source)
    families = [f"family{i}" for i in range(20)]
    source_manifest = tmp_path / "families.json"
    pilot.json_write(source_manifest, dict(split_seeds=[0, 1], families=[dict(name=f) for f in families]))
    args = argparse.Namespace(source_dir=source, output_dir=output, family_manifest=source_manifest,
        bootstrap_samples=20, cache_dir=tmp_path, device="cpu", checkpoint=None)
    def prepare_bank(args):
        e = make_episode(mixed=True)
        episodes = [dict(e, family=f, split_seed=s) for f in families for s in (0, 1)]
        root = args.output_dir / "real_transfer"
        pilot.atomic_save(root / "bank.pt", dict(format_version=1, episodes=episodes))
        pilot.json_write(root / "manifest.json", dict(source_manifest_sha256=pilot.hash_file(args.family_manifest),
            bank_sha256=pilot.hash_file(root / "bank.pt"), n_episodes=40))
    monkeypatch.setattr(real, "prepare", prepare_bank)
    monkeypatch.setattr(pilot, "load_frozen", lambda *args: (Backbone(), Path("unused"), "backbone"))
    transfer.prepare(args)
    transfer.prepare(args)
    original = transfer._timed_prediction
    calls = []
    def interrupt_after_one_episode(*args):
        calls.append(1)
        if len(calls) == 6:
            raise RuntimeError("simulate interruption")
        return original(*args)
    monkeypatch.setattr(transfer, "_timed_prediction", interrupt_after_one_episode)
    with pytest.raises(RuntimeError, match="interruption"):
        transfer.report(args)
    assert len(list((output / "real_transfer/report/predictions").glob("*.pt"))) == 1
    monkeypatch.setattr(transfer, "_timed_prediction", original)
    transfer.report(args)
    finished = synthetic.read(output / "real_transfer/report/complete.json")
    assert len(finished["comparisons"]) == 19
    assert all(c["families"] == 20 for c in finished["comparisons"].values())
    baseline = finished["comparisons"]["fresh_blend_vs_ordinary16"]
    assert baseline["wins"]+baseline["losses"]+baseline["ties"] == 20
    caches = list((output / "real_transfer/report/predictions").glob("*.pt"))
    assert len(caches) == 40
    for path in caches:
        saved = torch.load(path, weights_only=True)
        for arm in transfer.ARMS:
            torch.testing.assert_close(saved["predictions"][f"{arm}_blend"]["logits"],
                .5*saved["predictions"]["ordinary8"]["logits"]+.5*saved["predictions"][f"{arm}_learned"]["logits"])
    monkeypatch.setattr(transfer, "_timed_prediction", lambda *args: pytest.fail("completed report reran inference"))
    transfer.report(args)


def test_family_summary_cannot_treat_splits_as_independent_datasets():
    families = ["a", "b"]
    rows = []
    methods = ("ordinary8", "ordinary16", *[f"{arm}_{kind}" for arm in transfer.ARMS for kind in ("learned", "blend")])
    for family in families:
        for split in (0, 1):
            for method in methods:
                nll = (1 if split == 0 else 3) if method.startswith("ordinary") else 1.9
                rows.append(dict(family=family, split_seed=split, method=method, nll=nll,
                    accuracy=.6, seconds=1., auc=None, views=8))
    _, comparisons = transfer.aggregate(rows, families, 100)
    result = comparisons["fresh_learned_vs_ordinary8"]
    assert result["families"] == 2 and result["wins"] == 2
    assert result["geometric_gain"] == pytest.approx(1-(1.9+1e-4)/(2+1e-4))
    with pytest.raises(ValueError, match="split coverage"):
        transfer.aggregate(rows[:-1], families, 100)
