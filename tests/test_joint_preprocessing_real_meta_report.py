import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_real_meta_report as report
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts import joint_preprocessing_zero_shot_comparison as synthetic


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def episode(family, split, task_id):
    x_context = torch.tensor([[[-1.], [1.], [-.5], [.5]]])
    return dict(family=family, split_seed=split, task_id=task_id,
        x_context=x_context, x_query=torch.tensor([[[-.8], [.7]]]),
        y_context=torch.tensor([[0., 1., 0., 1.]]), y_query=torch.tensor([0, 1]),
        numerical_mask=torch.tensor([True]), n_classes=2)


def build_fixture(tmp_path, monkeypatch, include_final=True):
    args = argparse.Namespace(output_dir=tmp_path, bootstrap_samples=32,
        steps=2, include_final=include_final)
    initial = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        initial.weight.fill_(.8)
    fp = "shared-experiment"
    manifest = dict(fixture=True)
    pilot.json_write(tmp_path / "manifest.json", manifest)
    banks = dict(real_test=[episode(f, s, 10+i*2+s) for i, f in enumerate(("a", "b")) for s in (0, 1)],
        synthetic_test=[episode("synthetic", 0, 100+i) for i in range(3)])
    loaded, predictions = [], []
    def run_dir(root, arm, seed):
        return root / "runs" / f"{arm}_seed{seed}"
    def setup(_):
        return torch.nn.Linear(1, 1), copy.deepcopy(initial), manifest, fp, torch.device("cpu")
    def load_panel(_, __, panel):
        assert (tmp_path / "lock.json").exists()
        assert (tmp_path / "test_report/started.json").exists()
        loaded.append(panel)
        return banks[panel]
    def logits(_, model, e, estimators=8):
        predictions.append((e["task_id"], estimators, model if isinstance(model, str) else "learned"))
        assert not torch.is_grad_enabled()
        strength = 1.0 if isinstance(model, str) else float(model.weight[0, 0])
        values = e["x_query"][..., 0]*strength
        return torch.stack((-values, values), -1), estimators
    runner = SimpleNamespace(ARMS=("synthetic", "real", "mixed"), SEEDS=(0, 1),
        setup=setup, run_dir=run_dir, load_panel=load_panel, panel_logits=logits,
        run_fingerprint=lambda _, arm, seed: synthetic.digest(dict(arm=arm, seed=seed)),
        __file__=__file__)
    monkeypatch.setattr(report, "_runner", lambda: runner)
    for index, arm in enumerate(runner.ARMS):
        for seed in runner.SEEDS:
            directory = run_dir(tmp_path, arm, seed)
            runfp = synthetic.digest(dict(arm=arm, seed=seed))
            hashes = {}
            for state, step in (("selected", 1), ("final", 2)):
                model = copy.deepcopy(initial)
                with torch.no_grad():
                    model.weight.fill_(.9+.1*index+.01*seed+(.02 if state == "final" else 0))
                pilot.atomic_save(directory / f"{state}.pt", dict(
                    fingerprint=runfp, model=model.state_dict(), step=step))
                hashes[f"{state}_sha256"] = pilot.hash_file(directory / f"{state}.pt")
            pilot.json_write(directory / "complete.json", dict(experiment_fingerprint=fp,
                fingerprint=runfp, steps=2, selected_step=1, selected_score=-.01, **hashes))
    return args, runner, banks, loaded, predictions


def test_lock_requires_every_completion_before_any_test_loading(tmp_path, monkeypatch):
    args, runner, _, loaded, _ = build_fixture(tmp_path, monkeypatch)
    (runner.run_dir(tmp_path, "mixed", 1) / "complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        report.report(args)
    assert loaded == []
    assert not (tmp_path / "lock.json").exists()
    assert not (tmp_path / "test_report/started.json").exists()


def test_lock_rejects_changed_final_and_changed_completion_after_lock(tmp_path, monkeypatch):
    args, runner, _, _, _ = build_fixture(tmp_path, monkeypatch)
    locked = report.lock(args)
    assert locked["alpha"] == .5 and len(locked["choices"]) == 6
    directory = runner.run_dir(tmp_path, "real", 0)
    path = directory / "final.pt"
    original = path.read_bytes()
    path.write_bytes(original+b"changed")
    with pytest.raises(ValueError, match="final checkpoint hash"):
        report.lock(args)
    path.write_bytes(original)
    complete = synthetic.read(directory / "complete.json")
    complete["selected_score"] = -.02
    pilot.json_write(directory / "complete.json", complete)
    with pytest.raises(ValueError, match="choices changed"):
        report.lock(args)
    assert synthetic.read(tmp_path / "lock.json") == locked


def test_lock_checks_checkpoint_metadata_not_just_file_hash(tmp_path, monkeypatch):
    args, runner, _, _, _ = build_fixture(tmp_path, monkeypatch)
    directory = runner.run_dir(tmp_path, "synthetic", 0)
    path = directory / "selected.pt"
    saved = torch.load(path, weights_only=True)
    saved["step"] = 0
    pilot.atomic_save(path, saved)
    complete = synthetic.read(directory / "complete.json")
    complete["selected_sha256"] = pilot.hash_file(path)
    pilot.json_write(directory / "complete.json", complete)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        report.lock(args)


def test_lock_rejects_completion_from_another_continuation_seed(tmp_path, monkeypatch):
    args, runner, _, _, _ = build_fixture(tmp_path, monkeypatch)
    directory = runner.run_dir(tmp_path, "synthetic", 0)
    complete = synthetic.read(directory / "complete.json")
    complete["fingerprint"] = runner.run_fingerprint("shared-experiment", "synthetic", 1)
    pilot.json_write(directory / "complete.json", complete)
    with pytest.raises(ValueError, match="arm/seed fingerprint"):
        report.lock(args)


def metric_rows(runner, methods, expected):
    rows = []
    for family, splits in expected.items():
        for split in splits:
            for method in methods:
                nll = 1. if split == 0 else 3.
                if method not in ("ordinary8", "ordinary16"):
                    nll = 1.9
                rows.append(dict(panel="real_test", family=family, split_seed=split,
                    method=method, nll=nll, accuracy=.6, auc=.7, seconds=1., views=16))
    return rows


def test_aggregate_averages_splits_and_keeps_seeds_within_families():
    runner = SimpleNamespace(ARMS=("synthetic", "real", "mixed"), SEEDS=(0, 1))
    expected = dict(a={0, 1}, b={0, 1})
    methods = report._methods(runner, False)
    rows = metric_rows(runner, methods, expected)
    _, comparisons = report.aggregate(rows, expected, methods, runner, 32, include_final=False)
    selected = comparisons["real_paired_seed_mean_selected_blend_vs_ordinary16"]
    assert selected["families"] == 2 and selected["wins"] == 2
    assert selected["geometric_gain"] == pytest.approx(1-(1.9+1e-4)/(2+1e-4))
    assert selected["seed_consistency"]["both_seed_wins"] == 2
    with pytest.raises(ValueError, match="coverage"):
        report.aggregate(rows[:-1], expected, methods, runner, 32, include_final=False)
    with pytest.raises(ValueError, match="duplicate"):
        report.aggregate(rows+[rows[0]], expected, methods, runner, 32, include_final=False)


def test_seed_consistency_exposes_opposite_seed_directions():
    runner = SimpleNamespace(ARMS=("synthetic", "real", "mixed"), SEEDS=(0, 1))
    expected = dict(a={0, 1})
    methods = report._methods(runner, False)
    rows = metric_rows(runner, methods, expected)
    for row in rows:
        if row["method"] == "real_seed0_selected_blend":
            row["nll"] = 1.5
        if row["method"] == "real_seed1_selected_blend":
            row["nll"] = 2.3
    _, comparisons = report.aggregate(rows, expected, methods, runner, 32, include_final=False)
    paired = comparisons["real_paired_seed_mean_selected_blend_vs_ordinary16"]
    assert paired["families"] == 1 and paired["wins"] == 1
    assert paired["seed_consistency"]["both_seed_wins"] == 0
    assert paired["seed_consistency"]["other_seed_direction"] == 1
    assert paired["seed_consistency"]["per_seed_geometric_gain"][0] > 0
    assert paired["seed_consistency"]["per_seed_geometric_gain"][1] < 0


def test_complete_report_resumes_caches_and_preserves_aligned_blends(tmp_path, monkeypatch):
    args, _, _, loaded, predictions = build_fixture(tmp_path, monkeypatch)
    original = report._timed_prediction
    calls = []
    def interrupt_after_one_episode(*values):
        calls.append(1)
        if len(calls) == 16:
            raise RuntimeError("simulated interruption")
        return original(*values)
    monkeypatch.setattr(report, "_timed_prediction", interrupt_after_one_episode)
    with pytest.raises(RuntimeError, match="interruption"):
        report.report(args)
    caches = list((tmp_path / "test_report/predictions/real_test").glob("*.pt"))
    assert len(caches) == 1
    first_task_calls = sum(task == 10 for task, _, _ in predictions)
    assert first_task_calls == 15
    monkeypatch.setattr(report, "_timed_prediction", original)
    report.report(args)
    assert sum(task == 10 for task, _, _ in predictions) == first_task_calls
    complete = synthetic.read(tmp_path / "test_report/complete.json")
    assert complete["panels"]["real_test"]["families"] == 2
    assert complete["panels"]["synthetic_test"]["families"] == 3
    for panel, count in (("real_test", 4), ("synthetic_test", 3)):
        caches = list((tmp_path / "test_report/predictions" / panel).glob("*.pt"))
        assert len(caches) == count
        for path in caches:
            saved = torch.load(path, weights_only=True)
            assert len(saved["predictions"]) == 28
            ordinary = saved["predictions"]["ordinary8"]["logits"]
            for name, value in saved["predictions"].items():
                if name.endswith("blend"):
                    learned = saved["predictions"][name.removesuffix("blend")+"learned"]
                    torch.testing.assert_close(value["logits"], .5*ordinary+.5*learned["logits"])
                    assert value["views"] == 16
    synthetic_units = complete["panels"]["synthetic_test"]["comparisons"]["mixed_paired_seed_mean_selected_blend_vs_ordinary16"]
    assert synthetic_units["families"] == 3
    loads_before = len(loaded)
    monkeypatch.setattr(report, "_timed_prediction", lambda *args: pytest.fail("completed report reran inference"))
    report.report(args)
    assert len(loaded) == loads_before


def test_resume_rejects_query_label_changes_and_incomplete_prediction_cache(tmp_path, monkeypatch):
    args, _, banks, _, _ = build_fixture(tmp_path, monkeypatch, include_final=False)
    original = report._timed_prediction
    count = 0
    def interrupt(*values):
        nonlocal count
        count += 1
        if count == 10:
            raise RuntimeError("interrupted")
        return original(*values)
    monkeypatch.setattr(report, "_timed_prediction", interrupt)
    with pytest.raises(RuntimeError):
        report.report(args)
    monkeypatch.setattr(report, "_timed_prediction", original)
    labels = banks["real_test"][0]["y_query"].clone()
    banks["real_test"][0]["y_query"] = labels.flip(0)
    with pytest.raises(ValueError, match="cache identity/methods/labels"):
        report.report(args)
    banks["real_test"][0]["y_query"] = labels
    path = next((tmp_path / "test_report/predictions/real_test").glob("*.pt"))
    saved = torch.load(path, weights_only=True)
    saved["predictions"].pop("real_seed0_selected_blend")
    pilot.atomic_save(path, saved)
    with pytest.raises(ValueError, match="cache identity/methods/labels"):
        report.report(args)


def test_identical_checkpoint_states_share_inference_and_keep_deployment_timing(tmp_path, monkeypatch):
    args, runner, _, _, calls = build_fixture(tmp_path, monkeypatch)
    initial = runner.setup(args)[1]
    for arm in runner.ARMS:
        for seed in runner.SEEDS:
            directory = runner.run_dir(tmp_path, arm, seed)
            complete = synthetic.read(directory / "complete.json")
            for state in ("selected", "final"):
                path = directory / f"{state}.pt"
                saved = torch.load(path, weights_only=True)
                saved["model"] = initial.state_dict()
                pilot.atomic_save(path, saved)
                complete[f"{state}_sha256"] = pilot.hash_file(path)
            pilot.json_write(directory / "complete.json", complete)
    report.report(args)
    assert len(calls) == 7*3  # Two ordinary predictions plus one unique learned state.
    cache = torch.load(next((tmp_path / "test_report/predictions/real_test").glob("*.pt")), weights_only=True)
    baseline = cache["predictions"]["initial_learned"]
    repeated = cache["predictions"]["real_seed0_selected_learned"]
    assert repeated["inference_reused"]
    assert repeated["prediction_source_method"] == "initial_learned"
    assert repeated["seconds"] == baseline["seconds"] > 0
    torch.testing.assert_close(repeated["logits"], baseline["logits"])


def test_prediction_requires_all_requested_views(tmp_path, monkeypatch):
    args, runner, banks, _, _ = build_fixture(tmp_path, monkeypatch)
    original = runner.panel_logits
    runner.panel_logits = lambda *a, **kw: (original(*a, **kw)[0], 4)
    with pytest.raises(ValueError, match="view count"):
        report._timed_prediction(runner, runner.setup(args)[0], "ordinary", banks["real_test"][0], 8, torch.device("cpu"))


def test_binary_single_class_query_skips_auc_without_breaking_nll():
    scores = report.metrics(torch.tensor([[[-1., 1.], [-.5, .5]]]), torch.tensor([1, 1]), 2)
    assert np.isfinite(scores["nll"]) and scores["auc"] is None
