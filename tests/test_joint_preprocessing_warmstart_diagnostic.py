import argparse
import copy
import json

import pytest
import torch

from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts import joint_preprocessing_warmstart_diagnostic as warmstart
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def episode(task_id, source_seed):
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
        # Validation receives this marker, detecting any differentiable use.
        if torch.is_grad_enabled() and x.requires_grad:
            assert labels.shape[1] == 6, "validation used for training"
        values = x[:, labels.shape[1]:, 0]
        return torch.stack((-values, values), dim=-1)


def settings(**changes):
    values = dict(steps=3, evaluate_every=1, save_every=1, lr=.0003,
                  checkpoint=None, device="cpu", resume=False)
    values.update(changes)
    return argparse.Namespace(**values)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    def generate(generation, count, *, source_seed, task_offset, **kwargs):
        bank = [episode(task_offset + i, source_seed) for i in range(count)]
        if source_seed == 182001:
            from dataclasses import replace
            bank = [replace(e, x_context=e.x_context[:, :4], y_context=e.y_context[:, :4]) for e in bank]
        return bank
    monkeypatch.setattr(diagnostic, "generate_scheduled_episodes", generate)
    monkeypatch.setattr(pilot, "load_frozen", lambda args, device: (Backbone(), tmp_path / "checkpoint", "hash"))
    bank = tmp_path / "bank"
    diagnostic.run(argparse.Namespace(output_dir=bank, train_tasks=4, validation_tasks=2,
                    steps=2, evaluate_every=1, save_every=1, lr=.001, device="cpu",
                    checkpoint=None, max_steps=None, resume=False))
    source = tmp_path / "capacity"
    capacity.run(argparse.Namespace(source_dir=bank, output_dir=source, fit_steps=2,
                 fit_evaluate_every=1, fit_lrs=[.01], distill_steps=2,
                 distill_evaluate_every=1, distill_lr=.001, teacher_gain=1e-7,
                 min_useful_tasks=1, checkpoint=None, device="cpu", resume=False))
    return settings(source_dir=source, bank_dir=bank, output_dir=tmp_path / "warmstart")


def load_training_data(args):
    banks, refs, targets, weights, _, _, taught = warmstart.source_data(args, "hash")
    banks = {name: [pilot.filtered_episode(e) for e in bank] for name, bank in banks.items()}
    initial = JointPreprocessor("joint")
    initial.load_state_dict(weights)
    return banks, refs, targets, initial, taught["selected_step"]


def test_end_to_end_arms_share_taught_initialization_and_reset_optimizer(sources):
    args = sources
    warmstart.run(args)
    complete = json.loads((args.output_dir / "complete.json").read_text())
    for arm in warmstart.ARMS:
        result = complete["arms"][arm]
        assert result["steps_completed"] == 3
        assert result["initial"]["train"]["tasks"] == 4
        assert result["initial"]["validation"]["tasks"] == 2
        assert len(capacity.read_rows(args.output_dir / arm / "evaluation_tasks.csv")) == 24
        rows = capacity.read_rows(args.output_dir / arm / "training.csv")
        assert all(float(r["lr"]) == .0003 for r in rows)
        assert float(rows[-1]["grad_encoder"]) > 0
        saved = torch.load(args.output_dir / arm / "state.pt", weights_only=True)
        assert all(float(v["step"]) == 3 for v in saved["optimizer"]["state"].values())
    initial_a = complete["arms"]["imitation"]["initial"]
    initial_b = complete["arms"]["prediction"]["initial"]
    for panel in ("train", "validation"):
        assert initial_a[panel]["single_mean_nll"] == initial_b[panel]["single_mean_nll"]
        assert initial_a[panel]["ordinary_mean_nll"] == initial_b[panel]["ordinary_mean_nll"]
    args.resume = True
    warmstart.run(args)
    assert len(capacity.read_rows(args.output_dir / "prediction" / "training.csv")) == 3
    args.lr *= 2
    with pytest.raises(ValueError, match="fingerprint"):
        warmstart.run(args)


@pytest.mark.parametrize("arm", warmstart.ARMS)
def test_interrupted_arm_replays_exactly_and_preserves_source_weights(sources, tmp_path, arm):
    banks, refs, targets, initial, source_step = load_training_data(sources)
    unchanged = pilot.state_cpu(initial)
    full, interrupted = tmp_path / "full", tmp_path / "partial"
    backbone = Backbone()
    a = warmstart.train_arm(backbone, initial, banks, targets, refs, full,
                            sources, "locked", arm, source_step)
    assert warmstart.train_arm(backbone, initial, banks, targets, refs, interrupted,
                               sources, "locked", arm, source_step, stop_at=1) is None
    with (interrupted / "training.csv").open("a") as handle:
        handle.write("99,99,stale\n")
    b = warmstart.train_arm(backbone, initial, banks, targets, refs, interrupted,
                            sources, "locked", arm, source_step)
    assert a == b
    for name in ("model", "best_model"):
        state_a = torch.load(full / "state.pt", weights_only=True)[name]
        state_b = torch.load(interrupted / "state.pt", weights_only=True)[name]
        for key in state_a:
            torch.testing.assert_close(state_a[key], state_b[key], atol=0, rtol=0)
    for key, value in initial.state_dict().items():
        torch.testing.assert_close(value, unchanged[key], atol=0, rtol=0)
    assert backbone.frozen.grad is None
    assert [int(r["step"]) for r in capacity.read_rows(interrupted / "training.csv")] == [1, 2, 3]


def test_teacher_bank_and_checkpoint_integrity_are_required(sources):
    warmstart.source_data(sources, "hash")
    with pytest.raises(ValueError, match="backbone"):
        warmstart.source_data(sources, "different")
    selected_path = sources.source_dir / "distillation" / "selected.pt"
    checkpoint = torch.load(selected_path, weights_only=True)
    altered = copy.deepcopy(checkpoint)
    altered["fingerprint"] = "changed"
    torch.save(altered, selected_path)
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        warmstart.source_data(sources, "hash")
    torch.save(checkpoint, selected_path)
    manifest = sources.bank_dir / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="source hash"):
        warmstart.source_data(sources, "hash")


def test_validation_selection_uses_prediction_quality():
    summary = dict(validation=dict(ordinary_geometric_gain=.1, normalized_function_mse=100.))
    assert warmstart.selection_score(summary) == pytest.approx(torch.log(torch.tensor(.9)).item())
    summary["validation"]["ordinary_geometric_gain"] = .2
    assert warmstart.selection_score(summary) < -0.2


def test_initial_checkpoint_must_replay_recorded_teaching_score(sources, tmp_path):
    banks, refs, targets, initial, source_step = load_training_data(sources)
    with pytest.raises(ValueError, match="does not replay"):
        warmstart.train_arm(Backbone(), initial, banks, targets, refs, tmp_path / "wrong",
                             sources, "locked", "imitation", source_step,
                             expected_initial=dict(single_mean_nll=123., normalized_function_mse=123.))
    assert not (tmp_path / "wrong" / "state.pt").exists()


def test_actual_tiny_tabicl_backpropagates_through_warmstarted_hypernetwork():
    from tabicl._model.tabicl import TabICL
    torch.manual_seed(0)
    e = episode(4_000_000_000, 181001)
    model = JointPreprocessor("joint")
    with torch.no_grad():
        model.affine_head.weight.normal_(std=.01)
        model.affine_head.bias.add_(.03)
    backbone = TabICL(max_classes=10, embed_dim=8, col_num_blocks=1,
                     col_nhead=1, col_num_inds=2, col_feature_group=False, row_num_blocks=1,
                     row_nhead=1, row_num_cls=1, icl_num_blocks=1, icl_nhead=1,
                     col_ssmax=False, icl_ssmax=False, dropout=0., zero_init=False)
    backbone.requires_grad_(False)
    diagnostic.surrogate_nll(backbone, model, e).backward()
    assert model.encoder.cell_encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
