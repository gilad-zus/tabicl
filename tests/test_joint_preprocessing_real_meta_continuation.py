import argparse
import copy
import csv
import random

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_real_meta_bank as bank
from scripts import joint_preprocessing_real_meta_continuation as runner
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import SyntheticEpisode
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor
from tabicl._model.tabicl import TabICL


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Backbone(torch.nn.Module):
    max_classes = 2

    def __init__(self, stochastic=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3), requires_grad=False)
        self.stochastic = stochastic
        self.inputs = []

    def clear_cache(self):
        pass

    def forward(self, x, labels, **kwargs):
        self.inputs.append(x.detach().clone())
        weights = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
        value = (x[:, labels.shape[1]:] * weights).mean(-1) * self.weight
        if self.stochastic and self.training and torch.is_grad_enabled():
            value = value * (.5 + torch.rand(()) + np.random.random() + random.random())
        return torch.stack((-value, value), -1)


def source_family(name="source", rows=96, mixed=True):
    rng = np.random.default_rng(91)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(5)})
    frame["constant"] = 1.
    if mixed:
        frame["category"] = pd.Series(np.resize(["a", "b", "c"], rows), dtype=object)
        frame.loc[::7, "x1"] = np.nan
    return dict(family=name, source_group=name, columns=bank.pack_frame(frame),
                labels=(np.arange(rows) % 2).tolist(), source_indices=list(range(rows)))


def real_episode(mixed=True, task_id=17):
    e = bank.sample_real_episode(source_family(mixed=mixed), 32, .625, 33)
    e.update(domain="real", task_id=task_id, source_seed=33)
    return e


def initial_model():
    with torch.random.fork_rng():
        torch.manual_seed(72)
        model = JointPreprocessor("joint", hidden_dim=16)
        # A continuation checkpoint has nonzero heads, allowing encoder gradients.
        with torch.no_grad():
            model.affine_head.weight.normal_(0, .025)
            model.neural_last_head.weight.normal_(0, .01)
    return model


def generated_episodes(args, count, *, source_seed, task_offset, **kwargs):
    rng = np.random.default_rng(source_seed)
    rows = args.sequence_length
    context = int(rows * args.context_fraction)
    result = []
    for i in range(count):
        x = torch.tensor(rng.normal(size=(1, rows, 5)), dtype=torch.float32)
        labels = torch.arange(rows) % 2
        result.append(SyntheticEpisode(task_offset + i, source_seed,
            x[:, :context], x[:, context:], labels[:context].float()[None],
            labels[context:], 2, observation_mode="coverage_expanded"))
    return result


def read_rows(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_real_single_view_keeps_categories_and_missing_values_and_backpropagates():
    e = real_episode()
    assert (~e["numerical_mask"]).any() and e["context_missing"].any()
    model, backbone = initial_model(), Backbone()
    before = copy.deepcopy(backbone.state_dict())
    loss = runner.surrogate_nll(backbone, model, e)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("affine_head", "neural_last_head", "encoder"):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    assert_nested_equal(before, backbone.state_dict())
    assert torch.isfinite(backbone.inputs[-1]).all()

    # The learned path may alter numerical columns, but categorical columns must
    # retain the ordinary view's encoded values, including its feature ordering.
    generator, members, positions, _, _ = runner.real._prepared_views(e, 8)
    views = [(xs, index, features) for method, (xs, ys) in members.items()
             for index, (features, classes) in enumerate(generator.ensemble_configs_[method])]
    xs, index, features = views[random.Random(e["task_id"]).randrange(len(views))]
    categorical_positions = [j for j, original in enumerate(features) if original not in positions]
    assert categorical_positions
    torch.testing.assert_close(backbone.inputs[-1][..., categorical_positions],
        torch.tensor(xs[index:index + 1], dtype=torch.float32)[..., categorical_positions], rtol=0, atol=0)


@pytest.mark.parametrize("model_present", [False, True])
def test_numeric_real_and_synthetic_single_view_match_existing_surrogate(model_present):
    e = real_episode(mixed=False)
    model = initial_model() if model_present else None
    synthetic = runner.to_synthetic(e)
    expected = diagnostic.surrogate_nll(Backbone(), model, synthetic)
    actual_real = runner.surrogate_nll(Backbone(), model, e)
    actual_synthetic = runner.surrogate_nll(Backbone(), model, dict(e, domain="synthetic"))
    torch.testing.assert_close(actual_real, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_synthetic, expected, rtol=0, atol=0)


def test_query_labels_do_not_condition_real_predictions_or_generated_parameters(monkeypatch):
    e, model, backbone = real_episode(), initial_model(), Backbone()
    contexts = []
    generate = model.generate

    def record(x, y, missing=None):
        contexts.append((x.detach().clone(), y.detach().clone()))
        return generate(x, y, missing)

    monkeypatch.setattr(model, "generate", record)
    original = runner.surrogate_logits(backbone, model, e)
    altered = dict(e, y_query=1 - e["y_query"])
    torch.testing.assert_close(runner.surrogate_logits(backbone, model, altered), original, rtol=0, atol=0)
    without_labels = dict(e)
    without_labels.pop("y_query")
    torch.testing.assert_close(runner.surrogate_logits(backbone, model, without_labels), original, rtol=0, atol=0)
    assert len(contexts) == 3
    for x, y in contexts:
        assert x.shape[1] == e["x_context"].shape[1]
        torch.testing.assert_close(y, e["y_context"], rtol=0, atol=0)


def test_paired_domains_shapes_seeds_and_balanced_real_family_sampling(monkeypatch):
    monkeypatch.setattr(runner.synthetic, "generate_episodes", generated_episodes)
    families = [source_family(f"family{i}", rows=96 + 16 * i) for i in range(8)]
    spec = dict(generator=runner.GENERATOR)
    all_real = []
    for seed in (0, 1):
        for step in (1, 2):
            batches = {arm: runner.training_batch(arm, seed, step, families, spec) for arm in runner.ARMS}
            assert all(len(batch) == 4 for batch in batches.values())
            assert [e["domain"] for e in batches["synthetic"]] == ["synthetic"] * 4
            assert [e["domain"] for e in batches["real"]] == ["real"] * 4
            assert sum(e["domain"] == "real" for e in batches["mixed"]) == 2
            repeated = runner.training_batch("mixed", seed, step, families, spec)
            for i, mixed in enumerate(batches["mixed"]):
                counterpart = batches[mixed["domain"]][i]
                for key in ("x_context", "x_query", "y_context", "y_query"):
                    torch.testing.assert_close(mixed[key], counterpart[key], rtol=0, atol=0)
                    torch.testing.assert_close(mixed[key], repeated[i][key], rtol=0, atol=0)
                assert mixed["source_seed"] == counterpart["source_seed"] == repeated[i]["source_seed"]
                assert mixed["task_id"] == counterpart["task_id"]
            sizes = {(e["x_context"].shape[1], e["x_query"].shape[1])
                     for batch in batches.values() for e in batch}
            assert len(sizes) == 1
            if seed == 0:
                all_real.extend(e["family"] for e in batches["real"])
    assert sorted(all_real) == sorted(f["family"] for f in families)
    schedules = [{runner.shape_for_step(seed, step) for step in range(1, 13)} for seed in (0, 1)]
    assert schedules[0] == schedules[1] == {(n, f) for n in pilot.LENGTHS for f in pilot.FRACTIONS}
    audit = runner.seed_audit(dict(steps=4096))
    assert audit["training_seed_counts"] == [4096, 4096]
    assert runner.fresh_seed(0, 1) != runner.fresh_seed(1, 1)


@pytest.fixture
def training_fixture(monkeypatch):
    initial = initial_model().eval().requires_grad_(False)
    initial_state = copy.deepcopy(initial.state_dict())
    families = [source_family(f"family{i}") for i in range(4)]
    real_value = real_episode()
    numerical = real_episode(mixed=False)
    synthetic_value = runner.from_synthetic(runner.to_synthetic(numerical))
    loaded, backbones = [], []
    spec = dict(generator=runner.GENERATOR, fixture=True)
    manifest = dict(settings=spec, source_lock=dict(arms=dict(repeated=dict(checkpoint_sha256="same-initial-source"))))

    def setup(args):
        backbone = Backbone(stochastic=True)
        backbones.append(backbone)
        return backbone, initial, manifest, "fixed-experiment", torch.device("cpu")

    def load(root, manifest, panel):
        loaded.append(panel)
        assert "test" not in panel, "training attempted to load a final test bank"
        if panel == "real_train":
            return families
        return [copy.deepcopy(real_value if panel.startswith("real") else synthetic_value)]

    monkeypatch.setattr(runner, "setup", setup)
    monkeypatch.setattr(runner, "load_panel", load)
    monkeypatch.setattr(runner, "shape_for_step", lambda seed, step: (16 + 4 * (step % 2), .5))
    monkeypatch.setattr(runner.synthetic, "generate_episodes", generated_episodes)
    return dict(initial=initial, initial_state=initial_state, loaded=loaded, backbones=backbones)


def train_args(root, **changes):
    values = dict(output_dir=root, source_dir=root / "source", arm="mixed", continuation_seed=0,
                  device="cpu", checkpoint=None, steps=2, evaluate_every=1, save_every=1,
                  lr=.0003, resume=False, max_steps=None)
    values.update(changes)
    return argparse.Namespace(**values)


def test_real_training_resume_preserves_updates_optimizer_and_rng(tmp_path, monkeypatch, training_fixture):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    runner.train(train_args(full))
    runner.train(train_args(resumed, max_steps=1))
    folder = runner.run_dir(resumed, "mixed", 0)
    with (folder / "training.csv").open("a", encoding="utf8") as handle:
        handle.write("99,0,0\n")
    random.random()
    np.random.random(4)
    torch.rand(4)
    runner.train(train_args(resumed, resume=True))
    a = torch.load(runner.run_dir(full, "mixed", 0) / "state.pt", weights_only=True)
    b = torch.load(folder / "state.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "best_model", "best_step", "best_score", "clipped"):
        assert_nested_equal(a[key], b[key])
    optimizer_steps = [int(value["step"]) for value in b["optimizer"]["state"].values()]
    assert optimizer_steps and set(optimizer_steps) == {2}
    assert [int(row["step"]) for row in read_rows(folder / "training.csv")] == [1, 2]
    assert len(read_rows(folder / "presentations.csv")) == 8
    assert {(int(row["step"]), row["panel"]) for row in read_rows(folder / "evaluation.csv")} == {
        (step, panel) for step in (0, 1, 2)
        for panel in ("real_probe", "real_validation", "synthetic_probe", "synthetic_validation")}
    assert all(p.grad is None for backbone in training_fixture["backbones"] for p in backbone.parameters())
    assert_nested_equal(training_fixture["initial_state"], training_fixture["initial"].state_dict())
    assert any(not torch.equal(a["model"][key], training_fixture["initial_state"][key]) for key in a["model"])
    assert set(training_fixture["loaded"]) == {"real_train", "real_probe", "real_validation", "synthetic_probe", "synthetic_validation"}


def test_step_zero_remains_eligible_and_every_arm_starts_from_same_weights(tmp_path, monkeypatch, training_fixture):
    original_evaluate = runner.evaluate

    def score_worsens(*args, **kwargs):
        results = original_evaluate(*args, **kwargs)
        step = args[5]
        results["real_validation"]["validation_score"] = float(step)
        return results

    monkeypatch.setattr(runner, "evaluate", score_worsens)
    for arm in runner.ARMS:
        runner.train(train_args(tmp_path / arm, arm=arm))
        folder = runner.run_dir(tmp_path / arm, arm, 0)
        selected = torch.load(folder / "selected.pt", weights_only=True)
        assert selected["step"] == 0
        assert_nested_equal(selected["model"], training_fixture["initial_state"])
        config = runner.previous.read(folder / "config.json")
        assert config["source_checkpoint_sha256"] == "same-initial-source"
        assert config["source_step"] == 5120


@pytest.mark.parametrize("marker", ["lock.json", "test_report/started.json"])
def test_training_refuses_updates_after_test_choices_are_locked(tmp_path, marker):
    path = tmp_path / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="locked"):
        runner.train(train_args(tmp_path))


def test_tiny_actual_tabicl_real_training_path_has_input_gradients():
    backbone = TabICL(max_classes=2, embed_dim=8, col_num_blocks=1, col_nhead=1,
        col_num_inds=2, col_feature_group=False, row_num_blocks=1, row_nhead=1,
        row_num_cls=1, icl_num_blocks=1, icl_nhead=1, col_ssmax=False,
        icl_ssmax=False, dropout=0., zero_init=False).train().requires_grad_(False)
    model = initial_model()
    loss = runner.surrogate_nll(backbone, model, real_episode())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.affine_head.weight.grad is not None
    assert model.affine_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
