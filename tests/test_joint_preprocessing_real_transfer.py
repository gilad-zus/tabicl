import numpy as np
import pandas as pd
import pytest
import torch

from scripts.joint_preprocessing_real_transfer import _episode, evaluate_episode, episode_logits
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def clear_cache(self):
        pass

    def forward(self, x, y_train, **kwargs):
        values = x[:, y_train.shape[1]:, 0] * self.weight
        return torch.stack((-values, values), dim=-1)


def test_context_fitted_mixed_real_episode_and_all_references():
    rng = np.random.default_rng(37)
    rows = 300
    frame = pd.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(5)})
    frame["category"] = pd.Categorical(np.resize(["a", "b", "c"], rows))
    frame.loc[::17, "x0"] = np.nan
    labels = np.resize([0, 1], rows)
    episode = _episode(frame, labels, family="tiny_mixed", seed=0, max_rows=256, test_fraction=0.3)
    assert episode["context_missing"].any()
    assert int(episode["numerical_mask"].sum()) == 5
    backbone = TinyBackbone().eval()
    ordinary = evaluate_episode(backbone, "ordinary", episode)
    identity = evaluate_episode(backbone, "identity", episode)
    joint = evaluate_episode(backbone, JointPreprocessor("joint", hidden_dim=16).eval(), episode)
    for result in (ordinary, identity, joint):
        assert np.isfinite(result["nll"])
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["auc"] <= 1


def test_real_episode_rejects_ineligible_row_count():
    frame = pd.DataFrame({f"x{i}": np.arange(100) for i in range(5)})
    labels = np.arange(100) % 2
    try:
        _episode(frame, labels, family="small", seed=0, max_rows=1024, test_fraction=.3)
    except ValueError as error:
        assert "256" in str(error)
    else:
        raise AssertionError("short real dataset was accepted")


def test_nonfinite_prediction_identifies_source_split_and_view():
    class NonfiniteBackbone(TinyBackbone):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs) * float('nan')
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({f'x{i}': rng.normal(size=300) for i in range(5)})
    episode = _episode(frame, np.arange(300) % 2, family='unstable', seed=0,
                       max_rows=256, test_fraction=.3)
    with pytest.raises(FloatingPointError, match='unstable split 0 ordinary none view 0: nonfinite backbone logits'):
        episode_logits(NonfiniteBackbone(), 'ordinary', episode)
