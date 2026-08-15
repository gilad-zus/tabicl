import numpy as np
import pandas as pd
import torch

from tabicl._experiments.direct_spline_openml import OpenMLTaskData
from tabicl._experiments.direct_spline_openml_standard import (
    _apply_adapter,
    _fit_one_bag_standard,
    standard_direct_spline_config,
)
from tabicl._hyperspline import DirectSplineTransform


class _TinyStandardBackbone(torch.nn.Module):
    """Minimal differentiable stand-in for the public classifier checkpoint."""

    def __init__(self):
        super().__init__()
        self.max_classes = 10
        self.head = torch.nn.Linear(2, 10)

    def clear_cache(self):
        pass

    def forward(
        self,
        X,
        y_train,
        feature_shuffles=None,
        return_logits=True,
        softmax_temperature=0.9,
        inference_config=None,
    ):
        del feature_shuffles, return_logits, softmax_temperature, inference_config
        # The real inference head exposes exactly the task's label width,
        # rather than the checkpoint's maximum class count.
        return self.head(X[:, y_train.shape[1] :])[..., :2]


def test_standard_config_uses_all_fit_rows_unless_explicitly_capped():
    assert standard_direct_spline_config()["max_context_rows"] is None
    assert standard_direct_spline_config(context_cap=128)["max_context_rows"] == 128


def test_standard_adapter_identity_changes_only_numeric_positions():
    canonical = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    support = canonical[:, [0, 2]].unsqueeze(0)
    adapter = DirectSplineTransform(support, n_control_points=8, trainable_location_scale=True)
    with torch.no_grad():
        adapter.location.zero_()
        adapter.scale.fill_(1.0)
    transformed = _apply_adapter(
        canonical,
        numerical_indices=np.array([0, 2]),
        adapter=adapter,
    )
    assert torch.allclose(transformed, canonical, atol=2e-5, rtol=2e-5)
    assert torch.equal(transformed[..., 1], canonical[..., 1])


def test_standard_runner_checks_public_identity_and_preserves_prediction_shapes():
    rows = 32
    features = pd.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, rows),
            "x1": np.tile([0.0, 1.0], rows // 2),
        }
    )
    labels = np.tile([0, 1], rows // 2)
    task = OpenMLTaskData(
        task_id=1,
        dataset_id=2,
        dataset_name="tiny_standard",
        problem_type="binary",
        n_classes=2,
        x_train=features.iloc[:24].reset_index(drop=True),
        y_train=labels[:24],
        x_test=features.iloc[24:].reset_index(drop=True),
        y_test=labels[24:],
        outer_split_hash="test",
    )
    config = {
        **standard_direct_spline_config(train_context_rows=4),
        "adapter_steps": 2,
        "adapter_patience": 2,
        "validation_interval": 1,
        "query_batch_rows": 4,
        "cross_column_mixing_rank": 0,
    }
    result = _fit_one_bag_standard(
        task=task,
        fit_indices=np.arange(16),
        validation_indices=np.arange(16, 24),
        bag=0,
        config=config,
        protocol_seed=0,
        backbone=_TinyStandardBackbone(),
        device=torch.device("cpu"),
        run_fingerprint_hash="test",
        progress=None,
        requested_bags=8,
        effective_bags=8,
    )
    assert result.identity_validation.shape == (8, 2)
    assert result.adapted_validation.shape == (8, 2)
    assert result.guarded_test.shape == (8, 2)
    assert np.allclose(result.identity_validation.sum(axis=1), 1.0)
    assert result.metadata["pipeline"] == "standard_ensemble"
    assert result.metadata["support_rows"] == 16
    assert result.metadata["identity_parity_reference"] == "public_full_context_estimator"
    assert result.metadata["identity_parity_max_abs_validation"] <= config["identity_parity_atol"]
