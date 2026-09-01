from __future__ import annotations

import pytest
import torch

from tabicl._experiments.direct_spline_openml_standard import (
    _classification_training_objective_from_logits,
)


def _objective(logits: torch.Tensor, name: str, **config: float) -> torch.Tensor:
    return _classification_training_objective_from_logits(
        logits=logits,
        target=torch.tensor([0, 1, 0, 1]),
        problem_type="binary",
        n_classes=2,
        softmax_temperature=1.0,
        config={"classification_objective": name, **config},
    )


def test_pairwise_auc_is_finite_and_invariant_to_per_row_logit_offsets():
    logits = torch.tensor(
        [[1.2, -0.3], [0.2, 0.8], [0.9, -0.1], [-0.2, 0.6]], requires_grad=True
    )
    loss = _objective(logits, "pairwise_auc")
    shifted = _objective(
        logits.detach() + torch.tensor([[9.0], [-4.0], [1.5], [0.25]]), "pairwise_auc"
    )

    assert torch.isfinite(loss)
    assert torch.allclose(loss.detach(), shifted)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_hybrid_uses_declared_normalized_weights():
    logits = torch.tensor([[1.2, -0.3], [0.2, 0.8], [0.9, -0.1], [-0.2, 0.6]])
    cross_entropy = _objective(logits, "cross_entropy")
    pairwise_auc = _objective(logits, "pairwise_auc")
    hybrid = _objective(
        logits,
        "cross_entropy_plus_pairwise_auc",
        cross_entropy_weight=3.0,
        pairwise_auc_weight=1.0,
    )

    assert torch.allclose(hybrid, 0.75 * cross_entropy + 0.25 * pairwise_auc)


def test_pairwise_objective_rejects_multiclass_tasks():
    with pytest.raises(ValueError, match="binary-only"):
        _classification_training_objective_from_logits(
            logits=torch.zeros((3, 3)),
            target=torch.tensor([0, 1, 2]),
            problem_type="multiclass",
            n_classes=3,
            softmax_temperature=1.0,
            config={"classification_objective": "pairwise_auc"},
        )
