"""Regression tests for multi-GPU inference-manager device binding."""

import torch

from tabicl._model.inference_config import InferenceConfig


def test_inference_config_binds_unspecified_managers_without_mutating_the_caller_config():
    original = InferenceConfig(COL_CONFIG={"device": "cpu"})
    bound = original.with_default_device(torch.device("cuda:3"))

    # An explicit caller choice wins, while all manager defaults inherit the
    # model's actual GPU.  The original is reusable and remains untouched.
    assert original.COL_CONFIG.device == "cpu"
    assert original.ROW_CONFIG.device is None
    assert original.ICL_CONFIG.device is None
    assert bound.COL_CONFIG.device == "cpu"
    assert bound.ROW_CONFIG.device == torch.device("cuda:3")
    assert bound.ICL_CONFIG.device == torch.device("cuda:3")
