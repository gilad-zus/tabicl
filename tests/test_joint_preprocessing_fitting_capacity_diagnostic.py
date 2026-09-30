import argparse
import json

import pytest
import torch

from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
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


def episode(task_id=4_000_000_000, source_seed=181001):
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


def settings(**changes):
    values = dict(fit_steps=2, fit_evaluate_every=1, fit_lrs=[.001, .01],
                  distill_steps=2, distill_evaluate_every=1, distill_lr=.001,
                  teacher_gain=1e-7, min_useful_tasks=1, checkpoint=None, device="cpu", resume=False)
    values.update(changes)
    return argparse.Namespace(**values)


def initial_model():
    torch.manual_seed(0)
    return JointPreprocessor("joint")


def test_independent_initialization_bounds_and_parameter_isolation():
    initial, e = initial_model(), episode()
    original = pilot.state_cpu(initial)
    model = capacity.independent_map(initial, e)
    other = capacity.independent_map(initial, e)
    initial_parameters = initial.generate(e.x_context, e.y_context)
    independent_parameters = model.generate(e.x_context, e.y_context)
    for name in initial_parameters.__dataclass_fields__:
        torch.testing.assert_close(getattr(initial_parameters, name), getattr(independent_parameters, name),
                                   atol=0, rtol=0)
    assert isinstance(model.encoder, capacity.IgnoredContextEncoder)
    assert all(name.endswith(".raw") for name, p in model.named_parameters() if p.requires_grad)
    assert not model.slot_embeddings.requires_grad
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 4)
    p = model.generate(e.x_context, e.y_context)
    assert p.shift.abs().max() <= 1 and p.log_scale.abs().max() <= 1
    assert (p.spline_controls.diff(dim=-1) > 0).all()
    assert p.spline_controls[..., 0].eq(-1).all()
    torch.testing.assert_close(p.spline_controls[..., -1], torch.ones_like(p.shift), atol=1e-6, rtol=0)
    assert p.mixing.norm(dim=(-2, -1)).max() <= .100001
    assert torch.linalg.matrix_rank(p.mixing).max() <= min(initial.rank, 3)
    for name, value in original.items():
        torch.testing.assert_close(initial.state_dict()[name], value, atol=0, rtol=0)
    torch.testing.assert_close(other.affine_head.raw, torch.zeros_like(other.affine_head.raw))
    # Per-task parameters do not change with query labels or context labels.
    changed = model.generate(e.x_context, 1 - e.y_context)
    torch.testing.assert_close(changed.shift, p.shift, atol=0, rtol=0)
    with pytest.raises(ValueError, match="feature count"):
        model.generate(e.x_context[..., :2], e.y_context)


def test_direct_fit_resume_replays_and_backbone_stays_frozen(tmp_path):
    initial, backbone, e = initial_model(), Backbone(), episode()
    reference = float(diagnostic.surrogate_nll(backbone, None, e))
    args = settings()
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    a = capacity.fit_task(backbone, initial, e, reference, full, args, .01, "locked")
    assert capacity.fit_task(backbone, initial, e, reference, interrupted, args, .01, "locked", stop_at=1) is None
    with (interrupted / "training.csv").open("a") as handle:
        handle.write("99,0,0,0,0\n")
    b = capacity.fit_task(backbone, initial, e, reference, interrupted, args, .01, "locked")
    assert a == b
    state_a = torch.load(full / "state.pt", weights_only=True)
    state_b = torch.load(interrupted / "state.pt", weights_only=True)
    for name in state_a["model"]:
        torch.testing.assert_close(state_a["model"][name], state_b["model"][name], atol=0, rtol=0)
    assert [int(r["step"]) for r in capacity.read_rows(interrupted / "training.csv")] == [1, 2]
    assert backbone.frozen.grad is None
    assert a["best_nll"] <= reference + 1e-6
    assert float(capacity.read_rows(full / "training.csv")[0]["grad_affine_head"]) > 0
    with pytest.raises(ValueError, match="fingerprint"):
        capacity.fit_task(backbone, initial, e, reference, full, args, .01, "changed")


def targets_and_references(initial, episodes):
    backbone, targets, refs = Backbone(), [], {}
    for e in episodes:
        teacher = capacity.independent_map(initial, e)
        slot = pilot.view_specs(3, 2)[diagnostic.training_view(e)][0]
        fitting_loss = diagnostic.surrogate_nll(backbone, teacher, e)
        fitting_loss.backward()
        with torch.no_grad():
            # A small gradient step supplies a genuinely improving target for
            # each class/feature shuffle, rather than assuming scaling helps.
            teacher.affine_head.raw.sub_(.1 * teacher.affine_head.raw.grad)
            p = teacher.generate(e.x_context, e.y_context)
            x = torch.cat((e.x_context, e.x_query), dim=1)
            transformed = teacher.apply(x, p, slot)
            baseline = (x - p.location[:, None]) / p.scale[:, None]
            logits = pilot.forward_views(backbone, teacher, e, view_index=diagnostic.training_view(e))
            nll = float(diagnostic.surrogate_nll(backbone, teacher, e))
            identity = float(diagnostic.surrogate_nll(backbone, None, e))
        assert identity > nll
        targets.append(dict(task_id=e.task_id, slot=slot, transformed=transformed,
                            baseline_mse=float((transformed - baseline).square().mean()),
                            logits=logits, nll=nll, useful=True))
        refs[e.task_id] = dict(identity_surrogate_nll=identity,
                              ordinary_ensemble_nll=pilot.ordinary_episode_metrics(backbone, e)["nll"])
    return targets, refs


def test_function_teaching_resume_has_encoder_gradients_and_no_backbone_gradients(tmp_path):
    initial = initial_model()
    episodes = [episode(4_000_000_000 + i) for i in range(4)]
    targets, refs = targets_and_references(initial, episodes)
    for t in targets:
        assert not t["transformed"].requires_grad and not t["logits"].requires_grad
    args = settings(distill_steps=3)
    backbone = Backbone()
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    a = capacity.distill(backbone, initial, episodes, targets, refs, full, args, "locked")
    assert capacity.distill(backbone, initial, episodes, targets, refs, interrupted, args, "locked", stop_at=1) is None
    b = capacity.distill(backbone, initial, episodes, targets, refs, interrupted, args, "locked")
    assert a == b
    rows = capacity.read_rows(full / "training.csv")
    assert float(rows[-1]["grad_encoder"]) > 0
    assert a["selected"]["normalized_function_mse"] < 1
    assert backbone.frozen.grad is None
    assert len(capacity.read_rows(full / "evaluation_tasks.csv")) == 16
    assert len(capacity.read_rows(full / "selected_report" / "evaluation_tasks.csv")) == 4
    for name, value in torch.load(full / "state.pt", weights_only=True)["model"].items():
        torch.testing.assert_close(value, torch.load(interrupted / "state.pt", weights_only=True)["model"][name],
                                   atol=0, rtol=0)


def make_source(tmp_path, monkeypatch):
    def generate(generation, count, *, source_seed, task_offset, **kwargs):
        return [episode(task_offset + i, source_seed) for i in range(count)]
    monkeypatch.setattr(diagnostic, "generate_scheduled_episodes", generate)
    monkeypatch.setattr(pilot, "load_frozen", lambda args, device: (Backbone(), tmp_path / "checkpoint", "hash"))
    source = tmp_path / "source"
    diagnostic.run(argparse.Namespace(output_dir=source, train_tasks=4, validation_tasks=2,
                    steps=2, evaluate_every=1, save_every=1, lr=.001, device="cpu",
                    checkpoint=None, max_steps=None, resume=False))
    return source


def test_end_to_end_source_reuse_conditional_teaching_and_integrity(tmp_path, monkeypatch):
    source = make_source(tmp_path, monkeypatch)
    root = tmp_path / "capacity"
    args = settings(source_dir=source, output_dir=root)
    capacity.run(args)
    complete = json.loads((root / "complete.json").read_text())
    assert complete["fitting"]["tasks"] == 4
    assert complete["fitting"]["useful_teachers"] >= 1
    assert "selected" in complete["teaching"]
    assert len(capacity.read_rows(root / "fitting_comparison.csv")) == 8
    # Resume regenerates derived tables but cannot duplicate optimizer visits.
    args.resume = True
    capacity.run(args)
    assert len(capacity.read_rows(root / "fitting_comparison.csv")) == 8
    assert len(capacity.read_rows(root / "distillation" / "training.csv")) == 2
    args.fit_steps = 3
    with pytest.raises(ValueError, match="fingerprint"):
        capacity.run(args)
    args.fit_steps = 2
    manifest = source / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        capacity.run(args)


def test_skip_teaching_when_no_sufficient_teachers(tmp_path, monkeypatch):
    source = make_source(tmp_path, monkeypatch)
    root = tmp_path / "skip"
    capacity.run(settings(source_dir=source, output_dir=root, teacher_gain=.9, min_useful_tasks=4))
    complete = json.loads((root / "complete.json").read_text())
    assert complete["teaching"]["skipped"] and complete["fitting"]["useful_teachers"] == 0
    assert not (root / "distillation").exists()
    cache = torch.load(root / "teachers.pt", weights_only=True)
    assert all(not target["useful"] for target in cache["targets"])


def test_actual_tiny_tabicl_can_backpropagate_to_independent_parameters():
    from tabicl._model.tabicl import TabICL
    initial, e = initial_model(), episode()
    backbone = TabICL(max_classes=10, embed_dim=8, col_num_blocks=1,
                     col_nhead=1, col_num_inds=2, col_feature_group=False, row_num_blocks=1,
                     row_nhead=1, row_num_cls=1, icl_num_blocks=1, icl_nhead=1,
                     col_ssmax=False, icl_ssmax=False, dropout=0., zero_init=False)
    backbone.requires_grad_(False)
    model = capacity.independent_map(initial, e)
    loss = diagnostic.surrogate_nll(backbone, model, e)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.affine_head.raw.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
