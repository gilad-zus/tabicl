import argparse
import copy
import json

import numpy as np
import pytest
import torch

from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_fromscratch_diagnostic as experiment
from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def episode(task_id, source_seed=181001):
    x = torch.tensor([[[.1, -1., 2.], [2., 1., 0.], [1., -.2, 3.],
                       [-1., 2., 1.], [.7, -1.5, 2.5], [2.3, .4, -1.]]])
    query = torch.tensor([[[.3, .5, 1.], [-.4, 1.1, .5]]])
    return SyntheticEpisode(task_id, source_seed, x, query,
        torch.tensor([[0., 1., 0., 1., 0., 1.]]), torch.tensor([0, 1]), 2,
        observation_mode="coverage_expanded")


class Backbone(torch.nn.Module):
    max_classes = 10

    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def clear_cache(self):
        pass

    def forward(self, x, labels, **kwargs):
        values = x[:, labels.shape[1]:, 0]
        return torch.stack((-values, values), dim=-1)


def settings(**changes):
    values = dict(tasks=4, visits=2, evaluate_every=1, save_every=1,
        lrs=[.001, .0003], checkpoint=None, device="cpu", resume=False)
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    def generate(generation, count, *, source_seed, task_offset, **kwargs):
        return [episode(task_offset + i, source_seed) for i in range(count)]
    monkeypatch.setattr(diagnostic, "generate_scheduled_episodes", generate)
    monkeypatch.setattr(pilot, "load_frozen", lambda args, device: (Backbone(), tmp_path / "checkpoint", "hash"))
    bank = tmp_path / "bank"
    diagnostic.run(argparse.Namespace(output_dir=bank, train_tasks=4, validation_tasks=2,
        steps=2, evaluate_every=1, save_every=1, lr=.001, device="cpu",
        checkpoint=None, max_steps=None, resume=False))
    teacher = tmp_path / "teacher"
    capacity.run(argparse.Namespace(source_dir=bank, output_dir=teacher, fit_steps=2,
        fit_evaluate_every=1, fit_lrs=[.001, .003], distill_steps=1,
        distill_evaluate_every=1, distill_lr=.001, teacher_gain=.99,
        min_useful_tasks=4, checkpoint=None, device="cpu", resume=False))
    return settings(bank_dir=bank, teacher_dir=teacher, output_dir=tmp_path / "experiment")


def data(args):
    torch.manual_seed(0)
    initial = JointPreprocessor("joint")
    backbone = Backbone()
    episodes, refs, _, _, _ = experiment.source_data(args, backbone, "hash", initial, torch.device("cpu"))
    return backbone, initial, episodes, refs


def test_task_selection_and_schedule_ignore_scores_and_match_exposure():
    bank = [episode(i) for i in [9, 2, 5, 1]]
    assert [e.task_id for e in experiment.select_tasks(bank, 2)] == [1, 2]
    visits = [experiment.task_index(step, 8) for step in range(1, 2001)]
    assert np.bincount(visits).tolist() == [250] * 8
    for start in range(0, 2000, 8):
        assert sorted(visits[start:start + 8]) == list(range(8))
    with pytest.raises(ValueError, match="selection"):
        experiment.select_tasks(bank + [bank[0]], 2)


def test_reference_recovery_requires_meaningful_gain_and_is_not_clipped():
    assert experiment.recovery(.8, 1., .999) is None
    assert experiment.recovery(.8, 1., 1.1) is None
    assert experiment.recovery(.6, 1., .2) == pytest.approx(.5)
    assert experiment.recovery(1.2, 1., .2) < 0
    assert experiment.recovery(.1, 1., .2) > 1


def test_end_to_end_replays_teachers_and_trains_every_parameter_from_scratch(sources):
    experiment.run(sources)
    complete = experiment.read_json(sources.output_dir / "complete.json")
    assert complete["total_new_ce_updates"] == 32
    assert len(complete["summary"]) == 8
    comparison = capacity.read_rows(sources.output_dir / "comparison.csv")
    assert len(comparison) == 16
    for row in comparison:
        summary = complete["summary"][f"{row['method']}:lr{float(row['lr']):g}:final"]
        matched = [r for r in comparison if (r["method"], r["lr"]) == (row["method"], row["lr"])]
        assert summary["mean_nll"] == pytest.approx(np.mean([float(r["final_nll"]) for r in matched]))
    teacher_rows = capacity.read_rows(sources.output_dir / "teacher_reference.csv")
    assert len(teacher_rows) == 8
    assert sum(int(r["primary_rate"]) for r in teacher_rows) == 4
    for rate_index, lr in enumerate(sources.lrs):
        result = complete["rates"][str(lr)]
        shared = result["shared"]
        assert shared["steps"] == 8
        assert shared["initial"]["geometric_gain"] == pytest.approx(0, abs=1e-7)
        root = sources.output_dir / f"lr{rate_index}"
        rows = capacity.read_rows(root / "shared" / "training.csv")
        assert {int(r["task_id"]) for r in rows} == set(shared["tasks"])
        assert all(sum(int(r["task_id"]) == task for r in rows) == 2 for task in shared["tasks"])
        assert len(capacity.read_rows(root / "shared" / "evaluation_tasks.csv")) == 12
        saved = torch.load(root / "shared" / "state.pt", weights_only=True)
        parameter_count = len(list(JointPreprocessor("joint").parameters()))
        assert len(saved["optimizer"]["param_groups"][0]["params"]) == parameter_count
        assert len(saved["optimizer"]["state"]) == parameter_count
        for task, separate in result["separate"].items():
            assert separate["steps"] == 2
            assert separate["initial"]["geometric_gain"] == pytest.approx(0, abs=1e-7)
            assert len(capacity.read_rows(root / "separate" / task / "training.csv")) == 2
    sources.resume = True
    experiment.run(sources)
    assert len(capacity.read_rows(sources.output_dir / "teacher_reference.csv")) == 8
    sources.lrs = [.002]
    with pytest.raises(ValueError, match="fingerprint"):
        experiment.run(sources)


def test_teacher_checkpoint_and_bank_integrity_are_checked(sources):
    data(sources)
    path = sources.teacher_dir / "fits" / "4000000000" / "lr0" / "selected.pt"
    original = torch.load(path, weights_only=True)
    altered = copy.deepcopy(original)
    altered["fingerprint"] = "altered"
    torch.save(altered, path)
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        data(sources)
    torch.save(original, path)
    manifest = sources.bank_dir / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        data(sources)


def test_teacher_scores_cannot_influence_training(sources, tmp_path):
    backbone, initial, episodes, refs = data(sources)
    changed = copy.deepcopy(refs)
    for ref in changed.values():
        ref["teacher_nll"] = ref["sweep_teacher_nll"] = 0.
    for name, reference in (("original", refs), ("changed", changed)):
        experiment.train_network(backbone, initial, episodes, reference, tmp_path / name,
            sources, .001, "same", name)
    original = torch.load(tmp_path / "original" / "state.pt", weights_only=True)
    changed_state = torch.load(tmp_path / "changed" / "state.pt", weights_only=True)
    for key in original["model"]:
        torch.testing.assert_close(original["model"][key], changed_state["model"][key], atol=0, rtol=0)
    assert backbone.frozen.grad is None


def test_interrupted_resume_is_exact_and_initial_weights_are_preserved(sources, tmp_path):
    backbone, initial, episodes, refs = data(sources)
    original = pilot.state_cpu(initial)
    full, partial = tmp_path / "full", tmp_path / "partial"
    a = experiment.train_network(backbone, initial, episodes, refs, full, sources, .001, "same", "run")
    assert experiment.train_network(backbone, initial, episodes, refs, partial, sources,
        .001, "same", "run", stop_at=3) is None
    with (partial / "training.csv").open("a") as handle:
        handle.write("99,stale\n")
    b = experiment.train_network(backbone, initial, episodes, refs, partial, sources, .001, "same", "run")
    assert a == b
    for field in ("model", "best_model"):
        x = torch.load(full / "state.pt", weights_only=True)[field]
        y = torch.load(partial / "state.pt", weights_only=True)[field]
        for key in x:
            torch.testing.assert_close(x[key], y[key], atol=0, rtol=0)
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
    assert [int(r["step"]) for r in capacity.read_rows(partial / "training.csv")] == list(range(1, 9))


def test_real_tiny_tabicl_updates_heads_then_encoder_from_fresh_initialization():
    from tabicl._model.tabicl import TabICL
    torch.manual_seed(0)
    model = JointPreprocessor("joint")
    backbone = TabICL(max_classes=10, embed_dim=8, col_num_blocks=1,
        col_nhead=1, col_num_inds=2, col_feature_group=False, row_num_blocks=1,
        row_nhead=1, row_num_cls=1, icl_num_blocks=1, icl_nhead=1,
        col_ssmax=False, icl_ssmax=False, dropout=0., zero_init=False)
    backbone.requires_grad_(False)
    e = episode(4000000000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    before = model.affine_head.weight.detach().clone()
    diagnostic.surrogate_nll(backbone, model, e).backward()
    optimizer.step()
    assert not torch.equal(before, model.affine_head.weight)
    optimizer.zero_grad(set_to_none=True)
    diagnostic.surrogate_nll(backbone, model, e).backward()
    assert model.encoder.cell_encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
