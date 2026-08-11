"""CPU-only protocol invariants for the teacher-free scaling experiment."""

from __future__ import annotations

import numpy as np
import torch

from scripts.hyperspline_rank_basis_meta_scaling import (
    BatchedLearnedSharedRankBasisSpline,
    choose_alpha,
    macro_dataset_mean,
    resolve_real_train_datasets,
    sanitize_real_episode,
    scale_generated_parameters,
    select_real_episodes,
)
from scripts.hyperspline_real_zero_shot_eval import RealEpisode


def _episode(dataset: str, seed: int) -> RealEpisode:
    context = torch.randn(1, 8, 3)
    query = torch.randn(1, 4, 3)
    return RealEpisode(
        dataset=dataset,
        dataset_group="real_meta",
        split_seed=seed,
        n_context=8,
        n_query=4,
        n_features=3,
        n_numerical_features=3,
        n_categorical_features=0,
        n_classes=2,
        x_context=context,
        x_query=query,
        y_context=torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
        y_query=torch.tensor([0, 1, 0, 1]),
        numerical_mask=torch.tensor([True, True, True]),
    )


def test_real_arm_resolution_never_silently_shrinks_real4():
    available = {"pmlb_magic", "pmlb_pendigits", "pmlb_phoneme", "pmlb_spambase", "pmlb_ecoli"}
    assert resolve_real_train_datasets(
        "real_4", available, ["magic", "pendigits", "phoneme", "spambase"]
    ) == ["pmlb_magic", "pmlb_pendigits", "pmlb_phoneme", "pmlb_spambase"]
    assert resolve_real_train_datasets("real_all", available, ["magic"]) == sorted(available)
    try:
        resolve_real_train_datasets("real_4", available, ["magic", "missing"])
    except ValueError as error:
        assert "pmlb_missing" in str(error)
    else:  # pragma: no cover - makes the failure reason explicit.
        raise AssertionError("missing real_4 identity must fail rather than change the arm")


def test_global_alpha_has_exact_identity_endpoint_and_preserves_full_endpoint():
    torch.manual_seed(0)
    model = BatchedLearnedSharedRankBasisSpline(
        rank=4,
        hidden_dim=16,
        coefficient_bound=1.5,
        basis_bound=0.75,
        basis_init_rms=0.12,
        target_aware=True,
        raw_context=False,
    )
    # Make a non-identity table independently of the freshly zeroed encoder.
    values = model.grid.view(1, 1, -1).expand(1, 2, -1).clone()
    values[:, :, 1:-1] += 0.1
    generated = {
        "values": values,
        "location": torch.zeros(1, 2),
        "scale": torch.ones(1, 2),
        "shape_gate": torch.ones(1, 2),
        "normalization_gate": torch.zeros(1, 2),
        "coefficients": torch.zeros(1, 2, 4),
        "location_delta": torch.zeros(1, 2),
        "log_scale_delta": torch.zeros(1, 2),
    }
    identity = model.grid.view(1, 1, -1).expand_as(values)
    assert torch.equal(scale_generated_parameters(model, generated, 0.0)["values"], identity)
    assert torch.equal(scale_generated_parameters(model, generated, 1.0)["values"], values)
    halfway = scale_generated_parameters(model, generated, 0.5)["values"]
    assert torch.allclose(halfway, (identity + values) / 2)


def test_batched_conditioner_accepts_multiple_labelled_contexts_without_query_labels():
    model = BatchedLearnedSharedRankBasisSpline(
        rank=4,
        hidden_dim=16,
        coefficient_bound=1.5,
        basis_bound=0.75,
        basis_init_rms=0.12,
        target_aware=True,
        raw_context=True,
    )
    context = torch.randn(3, 12, 2)
    labels = torch.tensor([[0, 1] * 6, [1, 0] * 6, [0, 0, 1, 1] * 3])
    parameters = model.generated_parameters(context, labels)
    assert parameters["values"].shape == (3, 2, model.grid.numel())
    assert torch.allclose(
        parameters["values"], model.grid.view(1, 1, -1).expand_as(parameters["values"]), atol=1e-6
    )


def test_balanced_sampler_visits_each_selected_identity_equally_when_possible():
    by_dataset = {
        name: [_episode(name, seed) for seed in range(3)]
        for name in ("pmlb_a", "pmlb_b", "pmlb_c", "pmlb_d")
    }
    selected = select_real_episodes(
        by_dataset, task_count=12, datasets_per_update=4, rng=np.random.default_rng(7)
    )
    counts = {name: sum(item.dataset == name for item in selected) for name in by_dataset}
    assert counts == {name: 3 for name in by_dataset}


def test_alpha_selection_is_validation_only_and_tie_prefers_smaller_residual():
    alpha, score = choose_alpha({1.0: 0.1, 0.25: 0.1, 0.0: 0.11})
    assert alpha == 0.25
    assert score == 0.1


def test_macro_dataset_mean_is_not_bag_weighted():
    rows = [{"dataset": "small", "loss": 0.0}] + [{"dataset": "large", "loss": 0.6}] * 9
    assert macro_dataset_mean(rows) == 0.3


def test_nonfinite_bank_values_are_repaired_from_context_only():
    episode = _episode("pmlb_missing", 7)
    context = episode.x_context.clone()
    query = episode.x_query.clone()
    context[0, :, 0] = torch.tensor([1.0, float("nan"), 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    query[0, :, 0] = torch.tensor([float("nan"), 500.0, 600.0, 700.0])
    repaired, count = sanitize_real_episode(
        RealEpisode(**{**episode.__dict__, "x_context": context, "x_query": query})
    )
    # The median of finite context values is 9; query 500/600/700 cannot alter it.
    assert count == 2
    assert repaired.x_context[0, 1, 0] == 9.0
    assert repaired.x_query[0, 0, 0] == 9.0
    assert torch.isfinite(repaired.x_context).all() and torch.isfinite(repaired.x_query).all()
