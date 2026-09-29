import argparse
import json
import numpy as np
import pytest
import torch

from scripts.hyperspline_synthetic_train import SyntheticEpisode
from scripts.joint_preprocessing_synthetic_pilot import episode_metrics, forward_views, ordinary_episode_metrics, view_specs
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor
from tabicl._model.tabicl import TabICL
from tabicl._sklearn.preprocessing import EnsembleGenerator


def example():
    x = torch.tensor([[[0.1, -1.0, 2.0], [2.0, 1.0, 0.0],
                       [1.0, -0.2, 3.0], [-1.0, 2.0, 1.0],
                       [0.7, -1.5, 2.5], [2.3, 0.4, -1.0]]])
    y = torch.tensor([[0., 1., 0., 1., 0., 1.]])
    query = torch.tensor([[[0.3, 0.5, 1.0], [-0.4, 1.1, 0.5]]])
    return x, y, query


@pytest.mark.parametrize("arm", ("restricted", "joint", "no_spline"))
def test_joint_preprocessor_identity_and_context_invariances(arm):
    x, y, query = example()
    torch.manual_seed(7)
    model = JointPreprocessor(arm, hidden_dim=16)
    model.eval()
    context, queries = model(x, query, y)
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    torch.testing.assert_close(context, (x - mean) / std, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(queries, (query - mean) / std, atol=2e-5, rtol=2e-5)

    row_order = torch.tensor([4, 1, 3, 0, 5, 2])
    relabelled = 10 - y
    columns = torch.tensor([2, 0, 1])
    a = model.generate(x, y)
    b = model.generate(x[:, row_order], y[:, row_order])
    c = model.generate(x, relabelled)
    torch.testing.assert_close(model.apply(query, a, 0), model.apply(query, b, 0), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(model.apply(query, a, 0), model.apply(query, c, 0), atol=2e-5, rtol=2e-5)
    shuffled = model.generate(x[..., columns], y)
    torch.testing.assert_close(model.apply(query[..., columns], shuffled, 0),
                               model.apply(query, a, 0)[..., columns], atol=2e-5, rtol=2e-5)
    if a.mixing is not None:
        assert float(torch.linalg.matrix_norm(a.mixing.detach(), ord=2).max()) <= 0.100001


@pytest.mark.parametrize("arm", ("restricted", "joint", "no_spline"))
def test_joint_preprocessor_enabled_heads_receive_gradients(arm):
    x, y, query = example()
    torch.manual_seed(7)
    model = JointPreprocessor(arm, hidden_dim=16)
    params = model.generate(x, y)
    altered = model.apply(query, params, 0)
    (altered * torch.tensor([[[1., 2., -1.], [3., -1., 2.]]])).sum().backward()
    assert model.affine_head.weight.grad is not None
    assert model.affine_head.weight.grad.abs().sum() > 0
    if arm != "no_spline":
        assert model.spline_head.weight.grad is not None
        assert model.spline_head.weight.grad.abs().sum() > 0
    if arm != "restricted":
        assert model.neural_last_head.weight.grad is not None
        assert model.neural_last_head.weight.grad.abs().sum() > 0
        assert model.mix_gate_head.weight.grad is not None
        assert model.mix_gate_head.weight.grad.abs().sum() > 0


def test_view_schedule_matches_standard_tabicl_generator():
    x = np.arange(60, dtype=np.float64).reshape(20, 3)
    y = np.arange(20) % 2
    standard = EnsembleGenerator(classification=True, n_estimators=8,
                                 norm_methods=["none", "power"], feat_shuffle_method="latin",
                                 class_shuffle_method="shift", random_state=0).fit(x, y)
    expected = tuple((0 if method == "none" else 1, tuple(map(int, features)), tuple(map(int, labels)))
                     for method, configs in standard.ensemble_configs_.items() for features, labels in configs)
    assert view_specs(3, 2) == expected


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def clear_cache(self):
        pass

    def forward(self, x, labels, **kwargs):
        values = x[:, labels.shape[1]:, 0]
        return torch.stack((-values, values), dim=-1)


def test_training_view_is_differentiable_and_all_views_are_finite():
    x, y, query = example()
    episode = SyntheticEpisode(12, 42, x, query, y, torch.tensor([0, 1]), 2)
    model = JointPreprocessor("joint", hidden_dim=16)
    single = forward_views(FakeBackbone(), model, episode, view_index=2)
    ensemble = forward_views(FakeBackbone(), model, episode)
    assert single.shape == ensemble.shape == (1, 2, 2)
    assert torch.isfinite(ensemble).all()
    single.sum().backward()
    assert model.affine_head.weight.grad is not None


def test_standard_preprocessing_reference_is_finite():
    x, y, query = example()
    episode = SyntheticEpisode(12, 42, x, query, y, torch.tensor([0, 1]), 2)
    metrics = ordinary_episode_metrics(FakeBackbone(), episode)
    assert np.isfinite(metrics["nll"])
    assert 0 <= metrics["accuracy"] <= 1


def test_one_step_runner_writes_selected_checkpoint(tmp_path, monkeypatch, capsys):
    from scripts import joint_preprocessing_synthetic_pilot as pilot

    x, y, query = example()
    episode = SyntheticEpisode(2_000_000_000, 171001, x, query, y, torch.tensor([0, 1]), 2)
    backbone = FakeBackbone()
    backbone.max_classes = 10
    backbone.dummy.requires_grad_(False)
    monkeypatch.setattr(pilot, "load_frozen", lambda args, device: (backbone, tmp_path / "backbone", "hash"))
    monkeypatch.setattr(pilot, "load_bank", lambda root, manifest, name: [episode])
    monkeypatch.setattr(pilot, "train_episodes", lambda args, step, device: [
        SyntheticEpisode(1_000_000_000 + i, 161001, x, query, y, torch.tensor([0, 1]), 2)
        for i in range(4)
    ])
    pilot.json_write(pilot.manifest_path(tmp_path), {"format_version": 1})
    args = argparse.Namespace(output_dir=tmp_path, arm="joint", model_seed=0, device="cpu",
                              checkpoint=None, lr=1e-3, steps=1, validate_every=1,
                              save_every=1, max_steps=None, resume=False)
    pilot.train(args)
    progress = capsys.readouterr().out
    assert "step=0 val_nll=" in progress
    assert "step=1 train_nll=" in progress
    assert "step=1/1 train_nll_recent=" in progress
    checkpoint = torch.load(tmp_path / "runs" / "joint_seed0" / "selected.pt", weights_only=True)
    assert checkpoint["selected_step"] in (0, 1)
    assert (tmp_path / "runs" / "joint_seed0" / "complete.json").exists()


def test_report_accepts_only_completed_seed_zero_runs(tmp_path, monkeypatch):
    from scripts import joint_preprocessing_synthetic_pilot as pilot

    x, y, query = example()
    episode = SyntheticEpisode(3_000_000_000, 172001, x, query, y,
                               torch.tensor([0, 1]), 2)
    backbone = FakeBackbone()
    backbone.dummy.requires_grad_(False)
    monkeypatch.setattr(pilot, "load_frozen", lambda args, device: (backbone, tmp_path / "backbone", "hash"))
    monkeypatch.setattr(pilot, "load_bank", lambda root, manifest, name: [episode])
    pilot.json_write(pilot.manifest_path(tmp_path),
                     {"format_version": 1, "banks": {"test": {"sha256": "bank-hash"}}})
    manifest_hash = pilot.hash_file(pilot.manifest_path(tmp_path))
    for arm in pilot.ARMS:
        run = tmp_path / "runs" / f"{arm}_seed0"
        run.mkdir(parents=True)
        pilot.atomic_save(run / "selected.pt", dict(model=pilot.state_cpu(JointPreprocessor(arm)),
                                                     backbone_hash="hash", manifest_sha256=manifest_hash))
        if arm != "no_spline":
            pilot.json_write(run / "complete.json", {"steps_completed": 1})
    args = argparse.Namespace(output_dir=tmp_path, model_seeds=[0], device="cpu", checkpoint=None)
    with pytest.raises(FileNotFoundError):
        pilot.report(args)
    pilot.json_write(tmp_path / "runs" / "no_spline_seed0" / "complete.json", {"steps_completed": 1})
    pilot.report(args)
    result = json.loads((tmp_path / "report_seed0" / "synthetic_summary.json").read_text())
    assert result["model_seed_ids"] == [0]
    assert all(result["arms"][arm]["model_seeds"] == 1 for arm in pilot.ARMS)
    assert not (tmp_path / "report" / "synthetic_summary.json").exists()


def test_actual_tiny_tabicl_training_and_inference_paths():
    x, y, query = example()
    episode = SyntheticEpisode(12, 42, x, query, y, torch.tensor([0, 1]), 2)
    backbone = TabICL(max_classes=2, embed_dim=8, col_num_blocks=1, col_nhead=1,
                      col_num_inds=2, col_feature_group=False, row_num_blocks=1,
                      row_nhead=1, row_num_cls=1, icl_num_blocks=1, icl_nhead=1,
                      col_ssmax=False, icl_ssmax=False, dropout=0., zero_init=False)
    backbone.train()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model = JointPreprocessor("joint", hidden_dim=16)
    logits = forward_views(backbone, model, episode, view_index=0)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), episode.y_query).backward()
    assert model.affine_head.weight.grad is not None
    assert np.isfinite(episode_metrics(backbone, model, episode)["nll"])
    assert np.isfinite(ordinary_episode_metrics(backbone, episode)["nll"])
