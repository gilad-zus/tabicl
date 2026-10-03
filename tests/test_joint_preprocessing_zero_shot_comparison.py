import argparse
import copy
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_zero_shot_comparison as experiment
from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import SyntheticEpisode


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Backbone(torch.nn.Module):
    max_classes = 10

    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def clear_cache(self):
        pass

    def forward(self, x, labels, **kwargs):
        if torch.is_grad_enabled() and x.requires_grad:
            assert labels.shape[1] == 6, "held-out task used for training"
        values = x[:, labels.shape[1]:, 0]
        return torch.stack((-values, values), dim=-1)


def episode(task_id, source_seed):
    x = torch.tensor([[[.1, -1., 2.], [2., 1., 0.], [1., -.2, 3.],
                       [-1., 2., 1.], [.7, -1.5, 2.5], [2.3, .4, -1.]]])
    x = x + (task_id % 1000) / 1000
    q = torch.tensor([[[.3, .5, 1.], [-.4, 1.1, .5]]]) + (task_id % 1000) / 2000
    return SyntheticEpisode(task_id, source_seed, x, q,
        torch.tensor([[0., 1., 0., 1., 0., 1.]]), torch.tensor([0, 1]), 2,
        observation_mode="coverage_expanded")


def settings(root, **changes):
    values = dict(output_dir=root, steps=4, repeated_tasks=8, validation_tasks=2,
        test_tasks=2, probe_tasks=4, teacher_steps=2, evaluate_every=2, save_every=1,
        fit_steps=12, fit_evaluate_every=4, lr=.0003, bootstrap_samples=20,
        device="cpu", checkpoint=None, resume=False, max_steps=None, max_teachers=None, arm="fresh")
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def generate(args, count, *, source_seed, task_offset, **kwargs):
        return [episode(task_offset + i, source_seed) for i in range(count)]
    def scheduled(args, count, *, source_seed, task_offset, **kwargs):
        values = generate(args, count, source_seed=source_seed, task_offset=task_offset)
        length = 4 if source_seed == experiment.VAL_SEED else 3
        return [replace(e, x_context=e.x_context[:, :length], y_context=e.y_context[:, :length]) for e in values]
    monkeypatch.setattr(experiment.synthetic, "generate_episodes", generate)
    monkeypatch.setattr(experiment.synthetic, "generate_scheduled_episodes", scheduled)
    monkeypatch.setattr(pilot, "load_frozen", lambda args, device: (Backbone(), tmp_path / "checkpoint", "hash"))
    args = settings(tmp_path / "experiment")
    experiment.prepare(args)
    return args


def test_budget_seed_audit_and_repeated_order():
    args = settings(None, steps=10240, repeated_tasks=512, validation_tasks=512,
        test_tasks=1024, probe_tasks=48, teacher_steps=2048, evaluate_every=1024)
    spec = experiment.settings(args)
    audit = experiment.seed_audit(spec)
    assert len(audit["train_seeds"]) == 10240
    assert not set(audit["train_seeds"]) & set(audit["validation_seeds"])
    visits = [i for step in range(1, 10241) for i in experiment.batch_indices(step, 512)]
    assert np.bincount(visits).tolist() == [80] * 512
    for step in range(1, 130):
        assert len(set(experiment.batch_indices(step, 512))) == 4
    assert len({experiment.shape_for_step(s) for s in range(1, 13)}) == 12
    args.teacher_steps = 2049
    with pytest.raises(ValueError, match="budgets"):
        experiment.settings(args)


def test_preparation_locks_content_and_repeated_is_exact_fresh_subset(prepared):
    manifest = experiment.read(prepared.output_dir / "manifest.json")
    bank = pilot.load_bank(prepared.output_dir, manifest, "repeated")
    generated = [e for s in range(1, 3) for e in experiment.fresh_batch(s, manifest["experiment"])]
    assert [experiment.episode_hash(e) for e in bank] == [experiment.episode_hash(e) for e in generated]
    prepared.resume = True
    experiment.prepare(prepared)
    prepared.lr *= 2
    with pytest.raises(ValueError, match="fingerprint"):
        experiment.prepare(prepared)


def test_teacher_chunks_advance_and_weak_fits_have_identity_targets(prepared):
    prepared.max_teachers = 2
    for expected in (2, 4, 6, 8):
        experiment.teachers(prepared)
        assert len(list((prepared.output_dir / "teachers/fits").glob("*/complete.json"))) == expected
    _, _, _, fp, _ = experiment.setup(prepared)
    targets, _ = experiment.load_targets(prepared.output_dir, fp, list(range(experiment.TRAIN_OFFSET, experiment.TRAIN_OFFSET + 8)))
    done = experiment.read(prepared.output_dir / "teachers/complete.json")
    assert done["useful"] + done["neutral"] == 8
    assert done["independent_updates"] == 96
    for task, target in targets.items():
        fit = experiment.read(prepared.output_dir / "teachers/fits" / str(task) / "complete.json")
        assert target["useful"] == (fit["best_gain"] >= .01)
        if not target["useful"]:
            assert target["baseline_mse"] < 1e-10
    cache_path = prepared.output_dir / "teachers/targets.pt"
    with cache_path.open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="cache differs"):
        experiment.load_targets(prepared.output_dir, fp, list(targets))


def test_all_arms_global_validation_selection_and_locked_test(prepared, monkeypatch):
    with pytest.raises(ValueError, match="requires lock"):
        experiment.test(prepared)
    experiment.teachers(prepared)
    original_load = pilot.load_bank
    opened = []
    def monitored(root, manifest, panel):
        opened.append(panel)
        return original_load(root, manifest, panel)
    monkeypatch.setattr(pilot, "load_bank", monitored)
    for arm in experiment.ARMS:
        prepared.arm = arm
        experiment.train(prepared)
        folder = prepared.output_dir / arm
        done = experiment.read(folder / "complete.json")
        assert done["steps"] == 4 and done["optimizer_resets"] == 1
        history = [r for r in capacity.read_rows(folder / "evaluation.csv") if r["panel"] == "validation"]
        assert done["selected_step"] == int(min(history, key=lambda r: float(r["validation_score"]))["step"])
        assert [int(r["step"]) for r in history] == [0, 2, 4]
        trained = capacity.read_rows(folder / "training.csv")
        assert done["recorded_training_seconds"] > 0 and done["recorded_evaluation_seconds"] > 0
        assert all((r["normalized_function_loss"] != "") == (arm == "teacher") for r in trained)
        assert [r["stage"] for r in trained] == (["function", "function", "prediction", "prediction"] if arm == "teacher" else ["prediction"] * 4)
        saved = torch.load(folder / "state.pt", weights_only=True)
        assert all(float(v["step"]) == 2 for v in saved["optimizer"]["state"].values())
    assert "test" not in opened
    fresh = capacity.read_rows(prepared.output_dir / "fresh/fresh_tasks.csv")
    assert len(fresh) == 16 and len({r["task_id"] for r in fresh}) == 16
    experiment.lock(prepared)
    assert "test" not in opened
    locked = experiment.read(prepared.output_dir / "lock.json")
    assert set(locked["arms"]) == set(experiment.ARMS)
    with pytest.raises(ValueError, match="locked"):
        experiment.train(prepared)
    experiment.test(prepared)
    assert "test" in opened
    report = experiment.read(prepared.output_dir / "test_report/complete.json")
    assert report["ordinary16_vs_ordinary8"]["tasks"] == 2
    assert set(report["pairs"]) == {"repeated_vs_fresh", "teacher_vs_repeated", "teacher_vs_fresh"}
    rows = capacity.read_rows(prepared.output_dir / "test_report/tasks.csv")
    assert len(rows) == 6
    for row in rows:
        assert int(row["equal_blend_views"]) == int(row["ordinary8_views"]) + int(row["learned_views"])
        expected = int(row["ordinary8_views"]) + (int(row["learned_views"]) if float(row["selected_alpha"]) else 0)
        assert int(row["selected_blend_deployment_views"]) == expected
    experiment.test(prepared)
    assert len(capacity.read_rows(prepared.output_dir / "test_report/tasks.csv")) == 6
    lock_path = prepared.output_dir / "lock.json"
    original_lock = lock_path.read_bytes()
    lock_path.unlink()
    with pytest.raises(ValueError, match="cannot be reselected"):
        experiment.lock(prepared)
    lock_path.write_bytes(original_lock + b"\n")
    with pytest.raises(ValueError, match="different lock"):
        experiment.test(prepared)
    lock_path.write_bytes(original_lock)
    # An interrupted test also verifies the frozen checkpoints before loading test tasks.
    (prepared.output_dir / "test_report/complete.json").unlink()
    with (prepared.output_dir / "fresh/selected.pt").open("ab") as handle:
        handle.write(b"changed")
    opened.clear()
    with pytest.raises(ValueError, match="locked model changed"):
        experiment.test(prepared)
    assert "test" not in opened


def test_ordinary_logits_match_existing_baseline_and_zero_blend():
    e = episode(experiment.TRAIN_OFFSET, experiment.TRAIN_SEED)
    backbone = Backbone()
    logits, views = experiment.ordinary_logits(backbone, e)
    expected = pilot.ordinary_episode_metrics(backbone, e)
    assert views == len(pilot.view_specs(3, 2))
    assert experiment.score(logits, e.y_query)["nll"] == pytest.approx(expected["nll"], abs=1e-7)
    reference = {e.task_id: dict(ordinary=logits, ordinary_nll=expected["nll"])}
    model = experiment.JointPreprocessor("joint")
    _, scores = experiment.prediction_panel(backbone, model, [e], reference, torch.device("cpu"))
    assert scores[e.task_id]["alpha0"] == pytest.approx(expected["nll"], abs=1e-7)


def test_single_view_diagnostics_use_context_filtered_schedule(tmp_path, monkeypatch):
    raw = episode(experiment.TRAIN_OFFSET, experiment.TRAIN_SEED)
    raw.x_context[..., 0] = 1.
    backbone = Backbone()
    original = experiment.diagnostic.surrogate_nll
    calls = []
    def check(backbone, model, e):
        assert e.x_context.shape[-1] == 2
        calls.append(e.task_id)
        return original(backbone, model, e)
    monkeypatch.setattr(experiment.diagnostic, "surrogate_nll", check)
    refs = experiment.references(backbone, [raw], tmp_path, "probe", "hash", torch.device("cpu"))
    model = experiment.JointPreprocessor("joint")
    result = experiment.evaluate(backbone, model, {"probe": [raw]}, {"probe": refs},
        tmp_path / "run", 0, torch.device("cpu"))
    assert len(calls) == 2
    assert result["probe"]["mean_single_nll"] == pytest.approx(refs[raw.task_id]["identity_single_nll"], abs=1e-6)


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_exact_resume_across_common_optimizer_reset(prepared, tmp_path, arm):
    if arm == "teacher":
        experiment.teachers(prepared)
    partial_root = tmp_path / "partial"
    shutil.copytree(prepared.output_dir, partial_root)
    prepared.arm = arm
    experiment.train(prepared)
    full = torch.load(prepared.output_dir / arm / "state.pt", weights_only=True)
    partial = copy.copy(prepared)
    partial.output_dir, partial.max_steps = partial_root, 2
    experiment.train(partial)
    with (partial_root / arm / "training.csv").open("a") as handle:
        handle.write("99,stale\n")
    partial.max_steps, partial.resume = None, True
    experiment.train(partial)
    resumed = torch.load(partial_root / arm / "state.pt", weights_only=True)
    for field in ("model", "best_model"):
        for name in full[field]:
            torch.testing.assert_close(full[field][name], resumed[field][name], atol=0, rtol=0)
    assert full["best_step"] == resumed["best_step"] and full["resets"] == resumed["resets"] == 1
    assert [int(r["step"]) for r in capacity.read_rows(partial_root / arm / "training.csv")] == [1,2,3,4]
    if arm == "fresh":
        assert len(capacity.read_rows(partial_root / arm / "fresh_tasks.csv")) == 16


def test_alpha_ties_and_paired_bootstrap():
    refs = {1: dict(ordinary_nll=1.), 2: dict(ordinary_nll=2.)}
    nlls = {t: {f"alpha{a:g}": ref["ordinary_nll"] for a in experiment.ALPHAS} for t, ref in refs.items()}
    assert experiment.select_alpha(nlls, refs) == 0.
    nlls[1]["alpha0.25"] = .5
    assert experiment.select_alpha(nlls, refs) == .25
    report = experiment.bootstrap_summary([1.,2.], [1.,2.], 100)
    assert report["gain_ci95"] == [0.,0.] and report["ties"] == 2
    assert experiment.bootstrap_summary([.5,1.], [1.,2.], 100)["wins"] == 2
