"""Single-dataset feasibility ladder for the bounded rank-basis HyperSpline.

Stages: direct parameters fit one task; HyperSpline overfits that same task;
then one HyperSpline trains on a fixed bank and selects checkpoints on unseen
episodes from the same dataset.  The outer test is read only after selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split

from tabicl._hyperspline import HyperSplineTransform, summarize_context

try:
    # The cache may have been saved while function-basis ran as a script.
    # Keeping this name at module scope lets torch resolve __main__.TeacherBag
    # during unpickling without rewriting the existing cache.
    from scripts.direct_spline_function_basis import TeacherBag, fit_pca
    from scripts.direct_spline_dataset_headroom import release_cuda, stratified_subset
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import load_pmlb_frame
except ModuleNotFoundError:  # pragma: no cover
    from direct_spline_function_basis import TeacherBag, fit_pca
    from direct_spline_dataset_headroom import release_cuda, stratified_subset
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import load_pmlb_frame


class RankBasisSpline(nn.Module):
    """Exact-identity-at-zero monotone linear-spline output family."""
    def __init__(self, basis: torch.Tensor, *, hidden_dim: int, coefficient_bound: float, branch: str) -> None:
        super().__init__()
        self.register_buffer("grid", torch.linspace(-4.0, 4.0, basis.shape[-1]))
        self.register_buffer("basis", basis)
        self.n_coefficients = basis.shape[0]
        self.coefficient_bound, self.branch = coefficient_bound, branch
        self.encoder = HyperSplineTransform(
            n_control_points=self.n_coefficients + 1, hidden_dim=hidden_dim,
            target_aware=True, raw_context_residual=True, raw_context_num_heads=4,
            generate_location=True, generate_scale=True,
            gate_initial_probability=0.01, raw_context_gate_initial_probability=0.01,
        )

    def raw_from_context(self, x_context: torch.Tensor, y_context: torch.Tensor) -> tuple[torch.Tensor, object]:
        statistics = summarize_context(x_context, y_context=y_context.float().unsqueeze(0))
        raw, _ = self.encoder.generate_raw(
            statistics, x_context=x_context, y_context=y_context.float().unsqueeze(0)
        )
        return raw, statistics

    def parameters_from_raw(self, raw: torch.Tensor, statistics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficient_raw = raw[..., :self.n_coefficients]
        loc_raw, scale_raw = raw[..., self.n_coefficients + 1], raw[..., self.n_coefficients + 2]
        if self.branch == "normalization":
            coefficient_raw = coefficient_raw * 0.0
        if self.branch == "shape":
            loc_raw, scale_raw = loc_raw * 0.0, scale_raw * 0.0
        coefficients = self.coefficient_bound * torch.tanh(coefficient_raw)
        candidate = self.grid.view(1, 1, -1) + torch.matmul(coefficients, self.basis)
        # Identity increments are already positive and pass unchanged.  The
        # smooth correction only acts if a bounded curve would reverse slope.
        increments = torch.diff(candidate, dim=-1)
        increments = increments + F.softplus(-increments / 1e-4) * 1e-4
        values = torch.cat((candidate[..., :1], candidate[..., :1] + increments.cumsum(-1)), dim=-1)
        location = statistics.location + statistics.scale * torch.tanh(loc_raw)
        scale = statistics.scale * torch.exp(torch.tanh(scale_raw))
        return values, location, scale.clamp_min(1e-6)

    def transform(self, x: torch.Tensor, values: torch.Tensor, location: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        z = (x.float() - location.unsqueeze(1)) / scale.unsqueeze(1)
        clipped = z.clamp(float(self.grid[0]), float(self.grid[-1]))
        position = (clipped - self.grid[0]) / (self.grid[-1] - self.grid[0]) * (self.grid.numel() - 1)
        left = position.floor().long().clamp(0, self.grid.numel() - 2)
        fraction = (position - left).to(x.dtype)
        table = values.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        low = table.gather(3, left.unsqueeze(-1)).squeeze(-1)
        high = table.gather(3, (left + 1).unsqueeze(-1)).squeeze(-1)
        return low + fraction * (high - low) + (z - clipped)

    def forward_from_raw(self, raw: torch.Tensor, statistics, x_context: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        values, location, scale = self.parameters_from_raw(raw, statistics)
        return torch.cat((self.transform(x_context, values, location, scale), self.transform(x_query, values, location, scale)), dim=1)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        raw, statistics = self.raw_from_context(x_context, y_context)
        return self.forward_from_raw(raw, statistics, x_context, x_query)

    def deformation_penalty(self, x_context: torch.Tensor, y_context: torch.Tensor) -> torch.Tensor:
        """Graph-connected trust region around the identity transform.

        Keep this separate from :func:`diagnostics`: diagnostics deliberately
        detaches values for logging, while a training penalty must remain a
        tensor connected to the conditioner parameters.
        """
        raw, statistics = self.raw_from_context(x_context, y_context)
        values, location, scale = self.parameters_from_raw(raw, statistics)
        shape = (values - self.grid.view(1, 1, -1)).square().mean()
        location_residual = ((location - statistics.location) / statistics.scale.clamp_min(1e-6)).square().mean()
        log_scale_residual = (scale.log() - statistics.scale.clamp_min(1e-6).log()).square().mean()
        if self.branch == "shape":
            return shape
        if self.branch == "normalization":
            return location_residual + log_scale_residual
        return shape + location_residual + log_scale_residual


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pmlb-dataset", default="spambase")
    p.add_argument("--outer-seed", type=int, default=0)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default=None); p.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    p.add_argument("--device", default="cuda"); p.add_argument("--rank", type=int, default=8)
    p.add_argument("--branches", default="normalization,shape,joint")
    p.add_argument("--single-steps", type=int, default=2000); p.add_argument("--bank-steps", type=int, default=10000)
    p.add_argument("--bank-tasks", type=int, default=32); p.add_argument("--validation-tasks", type=int, default=32)
    p.add_argument("--context-sizes", default="128,256,384,512"); p.add_argument("--query-rows", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--coefficient-bound", type=float, default=1.5); p.add_argument("--regularization", type=float, default=1e-4)
    p.add_argument("--validate-every", type=int, default=250); p.add_argument("--model-seed", type=int, default=0)
    return p.parse_args()


def episode(x: np.ndarray, y: np.ndarray, context_rows: int, query_rows: int, rng: np.random.Generator):
    context_indices = stratified_subset(y, min(context_rows, y.size - np.unique(y).size), rng)
    remaining = np.setdiff1d(np.arange(y.size), context_indices, assume_unique=False)
    query_indices = remaining[stratified_subset(y[remaining], min(query_rows, remaining.size), rng)]
    return x[context_indices], y[context_indices], x[query_indices], y[query_indices]


def tensors(item, device: torch.device):
    cx, cy, qx, qy = item
    return (torch.as_tensor(cx, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(cy, dtype=torch.long, device=device),
            torch.as_tensor(qx, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(qy, dtype=torch.long, device=device))


def forward_loss(backbone, model, item, device, raw=None):
    cx, cy, qx, qy = tensors(item, device)
    backbone.clear_cache()
    if raw is None:
        transformed = model(cx, cy, qx)
    else:
        statistics = summarize_context(cx, y_context=cy.float().unsqueeze(0))
        transformed = model.forward_from_raw(raw, statistics, cx, qx)
    logits = backbone(transformed, cy.float().unsqueeze(0))
    return F.cross_entropy(logits.flatten(0, 1), qy.flatten()), (cx, cy, qx, qy)


@torch.no_grad()
def average_loss(backbone, model, tasks, device):
    return float(np.mean([float(forward_loss(backbone, model, item, device)[0]) for item in tasks]))


@torch.no_grad()
def diagnostics(model, item, device):
    cx, cy, _, _ = tensors(item, device)
    raw, stats = model.raw_from_context(cx, cy)
    values, loc, scale = model.parameters_from_raw(raw, stats)
    return {"coefficient_rms": float(raw[..., :model.n_coefficients].square().mean().sqrt()),
            "location_residual_rms": float((loc - stats.location).square().mean().sqrt()),
            "log_scale_residual_rms": float((scale.log() - stats.scale.log()).square().mean().sqrt()),
            "deformation_rms": float((values - model.grid.view(1, 1, -1)).square().mean().sqrt())}


def main() -> None:
    args = parse_args()
    branches = [name.strip() for name in args.branches.split(",") if name.strip()]
    if set(branches) - {"normalization", "shape", "joint"}: raise ValueError("invalid branches")
    args.context_sizes = [int(value) for value in args.context_sizes.split(",")]
    device = torch.device(args.device)
    x, y, _ = load_pmlb_frame(args.pmlb_dataset, cache_dir=args.pmlb_cache_dir); _, y = np.unique(y, return_inverse=True)
    x_dev, x_test, y_dev, y_test = train_test_split(x, y, test_size=.20, random_state=args.outer_seed, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=.25, random_state=args.outer_seed + 101, stratify=y_dev)
    payload = torch.load(args.teacher_cache.resolve(), map_location="cpu", weights_only=False)
    curves = torch.cat([bag.curves for bag in payload["bags"] if bag.dataset == args.pmlb_dataset and bag.seed == args.outer_seed])
    mean, components, _ = fit_pca(curves)
    basis = torch.cat((mean.unsqueeze(0), components[:args.rank]), dim=0).to(device)
    backbone, _ = load_backbone(args, device)
    rng = np.random.default_rng(args.model_seed)
    fixed = episode(x_train, y_train, max(args.context_sizes), args.query_rows, rng)
    bank = [episode(x_train, y_train, int(rng.choice(args.context_sizes)), args.query_rows, rng) for _ in range(args.bank_tasks)]
    validation = [episode(x_val, y_val, int(rng.choice(args.context_sizes)), args.query_rows, rng) for _ in range(args.validation_tasks)]
    test_context = episode(x_dev, y_dev, max(args.context_sizes), args.query_rows, np.random.default_rng(args.outer_seed + 909))[0:2]
    test_item = (test_context[0], test_context[1], x_test, y_test)
    rows, summary = [], {}
    for branch in branches:
        branch_rng = np.random.default_rng(args.model_seed)
        torch.manual_seed(args.model_seed)
        model = RankBasisSpline(basis, hidden_dim=args.hidden_dim, coefficient_bound=args.coefficient_bound, branch=branch).to(device)
        # Stage A: direct per-column rank-basis parameters on the fixed task.
        cx, cy, _, _ = tensors(fixed, device)
        raw_direct = nn.Parameter(torch.zeros(1, cx.shape[-1], model.n_coefficients + 3, device=device))
        direct_opt = torch.optim.Adam([raw_direct], lr=args.lr)
        initial_direct = float(forward_loss(backbone, model, fixed, device, raw_direct)[0])
        for _ in range(args.single_steps):
            loss, _ = forward_loss(backbone, model, fixed, device, raw_direct)
            direct_opt.zero_grad(set_to_none=True); loss.backward(); direct_opt.step()
        final_direct = float(forward_loss(backbone, model, fixed, device, raw_direct)[0])
        # Stage B: same fixed task through the HyperSpline conditioner.
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        initial_hyper = float(forward_loss(backbone, model, fixed, device)[0])
        for _ in range(args.single_steps):
            loss, _ = forward_loss(backbone, model, fixed, device)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        final_hyper = float(forward_loss(backbone, model, fixed, device)[0])
        rows.append({"branch": branch, "stage": "single_overfit", "step": args.single_steps, "direct_initial_loss": initial_direct, "direct_final_loss": final_direct, "hyper_initial_loss": initial_hyper, "hyper_final_loss": final_hyper, **diagnostics(model, fixed, device)})
        print(f"[{branch} single] direct={final_direct:.6f} hyper={final_hyper:.6f}", flush=True)
        # Stage C/D: fresh model, fixed task bank, identity-selected validation checkpoint.
        torch.manual_seed(args.model_seed)
        model = RankBasisSpline(basis, hidden_dim=args.hidden_dim, coefficient_bound=args.coefficient_bound, branch=branch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        identity_val = average_loss(backbone, model, validation, device)
        best_state, best_val, best_step = copy.deepcopy(model.state_dict()), identity_val, 0
        for step in range(1, args.bank_steps + 1):
            task = bank[int(branch_rng.integers(len(bank)))]
            loss, _ = forward_loss(backbone, model, task, device)
            cx, cy, _, _ = tensors(task, device)
            penalty = model.deformation_penalty(cx, cy)
            objective = loss + args.regularization * penalty
            opt.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step == 1 or step % args.validate_every == 0 or step == args.bank_steps:
                train_loss, val_loss = average_loss(backbone, model, bank, device), average_loss(backbone, model, validation, device)
                if val_loss < best_val:
                    best_state, best_val, best_step = copy.deepcopy(model.state_dict()), val_loss, step
                rows.append({"branch": branch, "stage": "bank", "step": step, "train_bank_loss": train_loss, "validation_loss": val_loss, "identity_validation_loss": identity_val, **diagnostics(model, fixed, device)})
                print(f"[{branch} bank] step={step}/{args.bank_steps} train={train_loss:.6f} validation={val_loss:.6f} best={best_val:.6f}", flush=True)
        model.load_state_dict(best_state)
        test_loss = float(forward_loss(backbone, model, test_item, device)[0])
        identity_model = RankBasisSpline(basis, hidden_dim=args.hidden_dim, coefficient_bound=args.coefficient_bound, branch=branch).to(device)
        identity_test = float(forward_loss(backbone, identity_model, test_item, device)[0])
        summary[branch] = {"direct_fixed_task_loss_delta": final_direct - initial_direct, "hyper_fixed_task_loss_delta": final_hyper - initial_hyper, "identity_validation_loss": identity_val, "best_validation_loss": best_val, "best_step": best_step, "best_validation_minus_identity": best_val - identity_val, "outer_test_loss": test_loss, "outer_test_identity_loss": identity_test, "outer_test_minus_identity": test_loss - identity_test}
        del model, identity_model, raw_direct, opt; release_cuda(device)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({"dataset": args.pmlb_dataset, "outer_seed": args.outer_seed, "rank": args.rank, "branches": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
