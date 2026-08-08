"""CPU-only invariants for the dataset-disjoint teacher-free LODO experiment."""

from __future__ import annotations

import torch

from scripts.hyperspline_rank_basis_lodo import LearnedSharedRankBasisSpline, lodo_folds


def test_lodo_folds_are_deterministic_and_dataset_disjoint():
    datasets = {"magic", "pendigits", "phoneme", "spambase"}
    folds = lodo_folds(datasets)
    assert [heldout for heldout, _ in folds] == sorted(datasets)
    assert len(folds) == len(datasets)
    for heldout, train in folds:
        assert heldout not in train
        assert train | {heldout} == datasets
        assert len(train) == 3


def test_learned_shared_rank_starts_at_exact_identity_and_uses_no_teacher_basis():
    torch.manual_seed(0)
    model = LearnedSharedRankBasisSpline(
        rank=4,
        hidden_dim=16,
        coefficient_bound=1.5,
        basis_bound=0.75,
        basis_init_rms=0.12,
        target_aware=True,
        raw_context=False,
    )
    context_x = torch.randn(1, 20, 3)
    context_y = torch.tensor([0, 1] * 10)
    parameters = model.generated_parameters(context_x, context_y)
    identity = model.grid.view(1, 1, -1)
    assert torch.allclose(parameters["values"], identity.expand_as(parameters["values"]), atol=1e-6)
    assert torch.count_nonzero(parameters["coefficients"]) == 0
    assert model.shared_basis_raw.requires_grad
    assert "components" not in dict(model.named_buffers())


def test_learned_shared_rank_receives_gradients_in_encoder_then_shared_dictionary():
    torch.manual_seed(1)
    model = LearnedSharedRankBasisSpline(
        rank=4,
        hidden_dim=16,
        coefficient_bound=1.5,
        basis_bound=0.75,
        basis_init_rms=0.12,
        target_aware=True,
        raw_context=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    context_x = torch.randn(1, 24, 3)
    context_y = torch.tensor([0, 1] * 12)
    for _ in range(2):
        parameters = model.generated_parameters(context_x, context_y)
        loss = parameters["values"].square().mean() + 0.01 * model.shared_basis_regularization()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert model.shared_basis_raw.grad is not None
    assert torch.isfinite(model.shared_basis_raw.grad).all()
    assert float(model.shared_basis_regularization().detach()) > 0.0
