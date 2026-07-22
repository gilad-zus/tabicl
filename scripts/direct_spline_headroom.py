"""Minimal frozen-TabICL direct-spline headroom experiment.

This intentionally uses a small numerical-only sklearn dataset.  It is a
diagnostic, not the zero-shot HyperSpline evaluation path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import DirectSplineTransform
from tabicl._model.tabicl import TabICL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Existing TabICL classifier checkpoint. Omit to download/cache the released v2 checkpoint.",
    )
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-control-points", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running on device: {device}", flush=True)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)

    if args.checkpoint is None:
        print(
            f"No checkpoint provided; resolving {args.checkpoint_version!r} from jingang/TabICL.",
            flush=True,
        )
        checkpoint_path = Path(
            hf_hub_download(repo_id="jingang/TabICL", filename=args.checkpoint_version)
        )
    else:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"TabICL checkpoint does not exist: {checkpoint_path}")
    print(f"Loading TabICL checkpoint: {checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain 'config' and 'state_dict'")
    backbone = TabICL(**checkpoint["config"])
    backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    backbone.to(device).train()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    print(
        "Loaded frozen TabICL backbone in differentiable training mode "
        f"({sum(p.numel() for p in backbone.parameters()):,} parameters).",
        flush=True,
    )

    data = load_iris()
    print(f"Loaded Iris dataset: X={data.data.shape}, y={data.target.shape}", flush=True)
    x_context, x_query, y_context, y_query = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
    )
    print(
        f"Split dataset: context={x_context.shape}, query={x_query.shape}; "
        f"context labels={y_context.shape}, query labels={y_query.shape}",
        flush=True,
    )
    x_context = torch.tensor(x_context, dtype=torch.float32, device=device).unsqueeze(0)
    x_query = torch.tensor(x_query, dtype=torch.float32, device=device).unsqueeze(0)
    y_context = torch.tensor(y_context, dtype=torch.float32, device=device).unsqueeze(0)
    y_query = torch.tensor(y_query, dtype=torch.long, device=device)
    spline = DirectSplineTransform(x_context, args.n_control_points).to(device)
    optimizer = torch.optim.AdamW(spline.parameters(), lr=args.lr)
    print(
        f"Initialized DirectSpline: control_points={args.n_control_points}, "
        f"trainable_parameters={sum(p.numel() for p in spline.parameters())}, "
        f"steps={args.steps}, lr={args.lr}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
        logits = backbone(transformed, y_context)
        loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
        if step == 0:
            print(
                f"First forward: transformed={tuple(transformed.shape)}, logits={tuple(logits.shape)}, "
                f"transformed.requires_grad={transformed.requires_grad}, "
                f"logits.requires_grad={logits.requires_grad}, loss.requires_grad={loss.requires_grad}",
                flush=True,
            )
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == args.steps:
            gate_grad = spline.gate_logits.grad
            gap_grad = spline.gap_logits.grad
            print(
                f"step={step + 1} loss={loss.item():.6f} "
                f"gate_grad_norm={gate_grad.norm().item():.6g} "
                f"gap_grad_norm={gap_grad.norm().item():.6g}",
                flush=True,
            )

    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        print(f"Peak CUDA memory: allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
