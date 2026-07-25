"""Standalone HyperSpline checkpoint helpers.

The frozen TabICL state dict is intentionally never included in these files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .module import HyperSplineTransform


FORMAT_VERSION = 1
STATISTICS_SCHEMA_VERSION = 1


def backbone_state_dict_hash(backbone: torch.nn.Module) -> str:
    """Return a stable digest of the exact frozen backbone weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_hyperspline_checkpoint(
    path: str | Path,
    module: HyperSplineTransform,
    config: Mapping[str, Any],
    *,
    backbone_reference: str | None = None,
    backbone_hash: str | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    step: int | None = None,
) -> None:
    """Save HyperSpline state and metadata without duplicating TabICL."""
    payload = {
        "format_version": FORMAT_VERSION,
        "hyperspline_config": dict(config),
        "state_dict": module.state_dict(),
        "backbone_reference": backbone_reference,
        "backbone_hash": backbone_hash,
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
        "optimizer_state": optimizer_state,
        "step": step,
    }
    torch.save(payload, Path(path))


def load_hyperspline_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_backbone_reference: str | None = None,
    expected_backbone_hash: str | None = None,
) -> tuple[HyperSplineTransform, dict[str, Any]]:
    """Load a strictly validated standalone HyperSpline checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported HyperSpline checkpoint format")
    if payload.get("statistics_schema_version") != STATISTICS_SCHEMA_VERSION:
        raise ValueError("unsupported HyperSpline statistics schema")
    if expected_backbone_reference is not None and payload.get("backbone_reference") != expected_backbone_reference:
        raise ValueError("HyperSpline checkpoint was trained for a different backbone reference")
    if expected_backbone_hash is not None and payload.get("backbone_hash") != expected_backbone_hash:
        raise ValueError("HyperSpline checkpoint was trained for a different backbone hash")
    config = dict(payload["hyperspline_config"])
    module = HyperSplineTransform(**config)
    module.load_state_dict(payload["state_dict"], strict=True)
    if module.has_supervised_residual:
        # requires_grad flags are not encoded in a state dict; restore the
        # architectural guarantee when loading a residual checkpoint.
        module.freeze_marginal_policy()
    module.to(device)
    return module, payload
