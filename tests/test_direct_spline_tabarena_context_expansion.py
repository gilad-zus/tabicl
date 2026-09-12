from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from direct_spline_tabarena_context_expansion import (  # noqa: E402
    TABARENA_LITE_MULTICLASS_TASK_IDS,
    TABARENA_LITE_REGRESSION_TASK_IDS,
    TABARENA_LITE_SUPPORTED_TASK_IDS,
    frozen_config,
)


def test_tabarena_lite_supported_task_bank_is_complete_and_unique() -> None:
    assert len(TABARENA_LITE_MULTICLASS_TASK_IDS) == 8
    assert len(TABARENA_LITE_REGRESSION_TASK_IDS) == 13
    assert len(TABARENA_LITE_SUPPORTED_TASK_IDS) == 21
    assert len(set(TABARENA_LITE_SUPPORTED_TASK_IDS)) == 21
    assert set(TABARENA_LITE_MULTICLASS_TASK_IDS).isdisjoint(TABARENA_LITE_REGRESSION_TASK_IDS)


def test_frozen_config_matches_heldout_context_expansion_source() -> None:
    config = frozen_config()
    assert config["adapter_architecture"] == "fixed_cubic"
    assert config["n_control_points"] == 20
    assert config["adapter_steps"] == 500
    assert config["validation_interval"] == 10
    assert config["adapter_patience"] == 10
    assert config["learning_rate"] == 0.005
    assert config["weight_decay"] == 0.003
    assert config["random_state"] == 0
    assert config["identity_regularization"] == 0.0
    assert "cosine_schedule_steps" not in config
