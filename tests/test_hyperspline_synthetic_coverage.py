"""Fast, backbone-free tests for synthetic coverage expansion and its audit."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts import hyperspline_synthetic_train as synthetic
from scripts.hyperspline_query_marginal_synthetic_coverage_audit import (
    QUERY_MARGINAL_DIM,
    ColumnPoint,
    descriptor_effective_rank,
    query_marginal_descriptors,
    source_auc,
)
from scripts.hyperspline_synthetic_generator_calibration import (
    calibrate_profiles,
    split_episodes_by_dataset,
)


def test_coverage_expanded_observation_is_deterministic_finite_non_decreasing_and_nontrivial():
    base = torch.linspace(-3.0, 3.0, 257).unsqueeze(-1).repeat(1, 6)
    transformed = synthetic.apply_synthetic_observation(base, observation_mode="coverage_expanded", seed=17)
    repeated = synthetic.apply_synthetic_observation(base, observation_mode="coverage_expanded", seed=17)
    assert torch.equal(transformed, repeated)
    assert torch.isfinite(transformed).all()
    # Each generated observation model is non-decreasing.  This verifies that
    # synthetic coverage does not create an artificial label signal by
    # reordering raw feature values.
    assert (transformed[1:] - transformed[:-1] >= -1e-6).all()
    assert not torch.equal(transformed, base)
    assert torch.equal(synthetic.apply_synthetic_observation(base, observation_mode="native", seed=17), base)


def test_scheduled_generation_balances_shapes_and_keeps_task_order(monkeypatch):
    def fake_generate(args, count, *, source_seed, task_offset, device):
        return [
            synthetic.SyntheticEpisode(
                task_id=task_offset + index,
                source_seed=source_seed,
                x_context=torch.zeros(1, int(args.sequence_length * args.context_fraction), 2),
                x_query=torch.zeros(1, int(args.sequence_length - int(args.sequence_length * args.context_fraction)), 2),
                y_context=torch.zeros(1, int(args.sequence_length * args.context_fraction)),
                y_query=torch.zeros(int(args.sequence_length - int(args.sequence_length * args.context_fraction)), dtype=torch.long),
                n_classes=2,
                observation_mode=args.synthetic_observation_mode,
            )
            for index in range(count)
        ]

    monkeypatch.setattr(synthetic, "generate_episodes", fake_generate)
    args = argparse.Namespace(prior_type="dummy")
    episodes = synthetic.generate_scheduled_episodes(
        args,
        13,
        source_seed=9,
        task_offset=100,
        device=torch.device("cpu"),
        sequence_lengths=(64, 128),
        context_fractions=(0.5, 0.75),
        observation_mode="coverage_expanded",
    )
    assert [item.task_id for item in episodes] == list(range(100, 113))
    assert {item.observation_mode for item in episodes} == {"coverage_expanded"}
    # Four schedule cells over 13 tasks: all receive either 3 or 4 tasks.
    cells = [(item.x_context.shape[1] + item.x_query.shape[1], item.x_context.shape[1]) for item in episodes]
    counts = sorted(cells.count(cell) for cell in set(cells))
    assert counts[-1] - counts[0] <= 1


def test_query_marginal_descriptor_is_33_dimensional_and_context_class_permutation_invariant():
    x_context = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]])
    x_query = torch.tensor([[[0.5, 1.5], [2.0, 3.0], [4.0, 6.0]]])
    labels = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    original = query_marginal_descriptors(x_context, labels, x_query)
    permuted = query_marginal_descriptors(x_context, 1.0 - labels, x_query)
    assert original.shape == (1, 2, QUERY_MARGINAL_DIM)
    assert torch.allclose(original, permuted, atol=1e-6)


def test_group_disjoint_source_auc_is_available_for_the_audit():
    def point(source: str, identity: str, value: float) -> ColumnPoint:
        return ColumnPoint(source, identity, 0, 0, 10, 10, 1, 1, 2, np.asarray([value, value * 2]))

    real = [point("real", f"real_{index}", float(index)) for index in range(5)]
    synthetic_points = [point("synthetic", f"synthetic_{index}", float(index) + 0.1) for index in range(5)]
    result = source_auc(real, synthetic_points, seed=0)
    assert result["n_splits"] >= 2
    assert 0.0 <= result["auc"] <= 1.0


def test_descriptor_effective_rank_accepts_numpy_singular_values():
    rank = descriptor_effective_rank(np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    assert np.isfinite(rank)
    assert 1.0 <= rank <= 2.0


def test_structural_synthetic_observation_is_deterministic_and_changes_task_structure():
    x = torch.linspace(-2.0, 2.0, 60).reshape(30, 2)
    y = torch.tensor([0, 1, 2] * 10)
    first_x, first_y = synthetic.apply_synthetic_task_structure(
        x, y, observation_mode="coverage_structural_broad", seed=91
    )
    repeated_x, repeated_y = synthetic.apply_synthetic_task_structure(
        x, y, observation_mode="coverage_structural_broad", seed=91
    )
    assert torch.equal(first_x, repeated_x)
    assert torch.equal(first_y, repeated_y)
    assert torch.isfinite(first_x).all()
    assert set(first_y.tolist()).issubset({0, 1, 2})
    native_x, native_y = synthetic.apply_synthetic_task_structure(x, y, observation_mode="native", seed=91)
    assert torch.equal(native_x, x)
    assert torch.equal(native_y, y)


def test_generator_calibration_splits_entire_dataset_identities_and_ranks_fit_only():
    class Episode:
        def __init__(self, dataset: str) -> None:
            self.dataset = dataset

    episodes = [Episode(f"dataset_{dataset}") for dataset in range(8) for _ in range(3)]
    fit, selection, fit_ids, selection_ids = split_episodes_by_dataset(episodes, fit_fraction=0.75, seed=5)
    assert set(fit_ids).isdisjoint(selection_ids)
    assert {item.dataset for item in fit} == set(fit_ids)
    assert {item.dataset for item in selection} == set(selection_ids)
    ranked = calibrate_profiles([
        {"profile": "native", "macro_real_to_synthetic_nn": 4.0, "macro_synthetic_to_real_nn": 4.0,
         "descriptor_quantile_l1_gap": 4.0, "source_auc": 0.95},
        {"profile": "better", "macro_real_to_synthetic_nn": 2.0, "macro_synthetic_to_real_nn": 2.0,
         "descriptor_quantile_l1_gap": 2.0, "source_auc": 0.70},
    ])
    assert ranked[0]["profile"] == "better"
    assert ranked[0]["fit_rank"] == 1
