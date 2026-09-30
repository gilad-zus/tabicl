import argparse
import csv

import pytest
import torch

from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


def episode(task_id, source_seed=181001):
    x = torch.tensor([[[.1, -1., 2.], [2., 1., 0.], [1., -.2, 3.],
                       [-1., 2., 1.], [.7, -1.5, 2.5], [2.3, .4, -1.]]])
    query = torch.tensor([[[.3, .5, 1.], [-.4, 1.1, .5]]])
    return SyntheticEpisode(task_id, source_seed, x, query,
                            torch.tensor([[0., 1., 0., 1., 0., 1.]]),
                            torch.tensor([0, 1]), 2, observation_mode="coverage_expanded")


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


def args(root, **changes):
    values = dict(output_dir=root, train_tasks=4, validation_tasks=2,
                  steps=2, evaluate_every=1, save_every=1, lr=1e-3,
                  device="cpu", checkpoint=None, max_steps=None, resume=False)
    values.update(changes)
    return argparse.Namespace(**values)


def fake_banks(monkeypatch):
    def generate(generation, count, *, source_seed, task_offset, **kwargs):
        return [episode(task_offset + i, source_seed) for i in range(count)]
    monkeypatch.setattr(diagnostic, "generate_scheduled_episodes", generate)


def rows(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def test_epoch_schedule_and_task_separation(tmp_path, monkeypatch):
    assert sorted(i for step in range(1, 9) for i in diagnostic.batch_indices(step, 32)) == list(range(32))
    assert diagnostic.batch_indices(10, 32) == diagnostic.batch_indices(10, 32)
    fake_banks(monkeypatch)
    diagnostic.prepare(args(tmp_path))
    manifest = pilot.read_manifest(tmp_path)
    train = pilot.load_bank(tmp_path, manifest, "train")
    val = pilot.load_bank(tmp_path, manifest, "validation")
    assert len(train) == 4 and len(val) == 2
    assert not set(e.task_id for e in train) & set(e.task_id for e in val)
    assert manifest["banks"]["train"]["source_seed"] != manifest["banks"]["validation"]["source_seed"]
    assert "test" not in manifest["banks"]
    with pytest.raises(ValueError):
        diagnostic.batch_indices(1, 5)


def test_effective_branch_diagnostics_at_identity():
    model = JointPreprocessor("joint")
    with torch.no_grad():
        values = diagnostic.transform_diagnostics(model, episode(4_000_000_000))
    for slot in (0, 1):
        for branch in ("affine", "spline", "neural", "mixing", "total"):
            assert values[f"slot{slot}_{branch}_rms"] < 3e-5
        assert values[f"slot{slot}_neural_gate_mean"] == pytest.approx(.1)
        assert values[f"slot{slot}_mixing_frobenius_norm"] == 0


def test_run_resume_and_validation_never_supply_gradients(tmp_path, monkeypatch):
    fake_banks(monkeypatch)
    backbones = []

    def load(args, device):
        backbone = Backbone()
        backbones.append(backbone)
        return backbone, tmp_path / "checkpoint", "frozen-hash"

    monkeypatch.setattr(pilot, "load_frozen", load)
    original = diagnostic.surrogate_nll
    fitting_tasks = []

    def tracked(backbone, model, ep):
        if model is not None and torch.is_grad_enabled():
            fitting_tasks.append(ep.task_id)
        return original(backbone, model, ep)

    monkeypatch.setattr(diagnostic, "surrogate_nll", tracked)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    diagnostic.run(args(full))
    diagnostic.run(args(resumed, max_steps=1))
    # Simulate rows written after the most recent checkpoint before interruption.
    with (resumed / "training.csv").open("a", encoding="utf8") as handle:
        handle.write("99,0,0,0\n")
    diagnostic.run(args(resumed, resume=True))
    a = torch.load(full / "state.pt", weights_only=True)
    b = torch.load(resumed / "state.pt", weights_only=True)
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    assert all(4_000_000_000 <= task < 4_000_000_004 for task in fitting_tasks)
    assert all(backbone.frozen.grad is None for backbone in backbones)
    assert [int(row["step"]) for row in rows(resumed / "training.csv")] == [1, 2]
    assert len(rows(resumed / "evaluation_tasks.csv")) == 18
    curve = rows(resumed / "evaluation.csv")
    assert {(int(row["step"]), row["panel"]) for row in curve} == {
        (step, panel) for step in (0, 1, 2) for panel in ("train", "validation")}
    assert "grad_encoder" in rows(resumed / "training.csv")[0]
    assert float(rows(resumed / "training.csv")[0]["grad_neural_last_head"]) > 0
    assert (resumed / "selected.pt").exists() and (resumed / "complete.json").exists()
    with pytest.raises(ValueError, match="fingerprint"):
        diagnostic.run(args(resumed, steps=3, resume=True))
