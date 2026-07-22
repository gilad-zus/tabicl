"""Minimal frozen-TabICL direct-spline headroom experiment.

This intentionally uses a small numerical-only sklearn dataset.  It is a
diagnostic, not the zero-shot HyperSpline evaluation path.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import DirectSplineTransform
from tabicl._model.tabicl import TabICL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-control-points", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone = TabICL(**checkpoint["config"])
    backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    data = load_iris()
    x_context, x_query, y_context, y_query = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
    )
    x_context = torch.tensor(x_context, dtype=torch.float32, device=device).unsqueeze(0)
    x_query = torch.tensor(x_query, dtype=torch.float32, device=device).unsqueeze(0)
    y_context = torch.tensor(y_context, dtype=torch.float32, device=device).unsqueeze(0)
    y_query = torch.tensor(y_query, dtype=torch.long, device=device)
    spline = DirectSplineTransform(x_context, args.n_control_points).to(device)
    optimizer = torch.optim.AdamW(spline.parameters(), lr=args.lr)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
        logits = backbone(transformed, y_context)
        loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
