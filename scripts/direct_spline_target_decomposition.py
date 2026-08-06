"""Final DirectSpline target diagnostics before amortizing HyperSpline.

The normalization mode asks which oracle normalization blocks are required
when a rank-R function-space curve is already known.  The consensus mode asks
whether a leave-one-bag-out robust consensus of other teacher curves is a
useful stable target.  Both only read the completed teacher cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.direct_spline_function_basis import TeacherBag, fit_pca, reconstruct_curves, restore_spline
    from scripts.direct_spline_dataset_headroom import ensemble_metrics, evaluate_chunked, release_cuda
    from scripts.direct_spline_multidataset_headroom import load_backbone
except ModuleNotFoundError:  # pragma: no cover
    from direct_spline_function_basis import TeacherBag, fit_pca, reconstruct_curves, restore_spline
    from direct_spline_dataset_headroom import ensemble_metrics, evaluate_chunked, release_cuda
    from direct_spline_multidataset_headroom import load_backbone


NORMALIZATION_CONDITIONS = ("oracle_all", "default", "location", "scale", "range")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("normalization", "consensus"), required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--consensus", choices=("mean", "median"), default="median")
    parser.add_argument("--projection-steps", type=int, default=250)
    parser.add_argument("--projection-lr", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--evaluation-query-chunk-rows", type=int, default=256)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_compat_args(args: argparse.Namespace, config: dict[str, object]) -> None:
    args.n_control_points = int(config["n_control_points"])
    args.grid_points = int(config["grid_points"])
    args.grid_range = float(config["grid_range"])
    args.freeze_spline_shape = False
    args.trainable_range = False
    args.trainable_location_scale = True
    args.control_mode = "monotone"
    args.free_control_bound = 1.0
    args.knot_placement = "uniform"
    args.cross_column_mixing_rank = 0
    args.cross_column_mixing_bound = 0.1


def copy_normalization(candidate, teacher, condition: str) -> None:
    with torch.no_grad():
        if condition in {"oracle_all", "location"}:
            candidate.location_offsets.copy_(teacher.location_offsets)
        if condition in {"oracle_all", "scale"}:
            candidate.log_scale_offsets.copy_(teacher.log_scale_offsets)
        if condition in {"oracle_all", "range"}:
            candidate.range_logits.copy_(teacher.range_logits)


def project_curve(identity, teacher, curve: torch.Tensor, args: argparse.Namespace, condition: str):
    candidate = copy.deepcopy(identity).train()
    copy_normalization(candidate, teacher, condition)
    with torch.no_grad():
        candidate.gap_logits.zero_()
    location, scale, _ = candidate._location_scale_range()
    z = torch.linspace(-args.grid_range, args.grid_range, args.grid_points, device=location.device)
    raw = (location.unsqueeze(1) + scale.unsqueeze(1) * z.view(1, -1, 1)).detach()
    target = (z.view(1, -1, 1) + curve.transpose(0, 1).unsqueeze(0).to(raw)).detach()
    optimizer = torch.optim.Adam((candidate.gap_logits, candidate.gate_logits), lr=args.projection_lr)
    for _ in range(args.projection_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (candidate.transform(raw) - target).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        rmse = float((candidate.transform(raw) - target).square().mean().sqrt())
    return candidate.eval(), rmse


def candidate_curve(args, bags: list[TeacherBag], bag: TeacherBag) -> torch.Tensor:
    if args.mode == "consensus":
        source = [item.curves for item in bags if item.dataset == bag.dataset and item.seed == bag.seed and item.bag != bag.bag]
        curves = torch.stack(source)
        return curves.median(0).values if args.consensus == "median" else curves.mean(0)
    train = torch.cat([item.curves for item in bags if item.dataset != bag.dataset])
    mean, components, _ = fit_pca(train)
    if args.rank > components.shape[0]:
        raise ValueError(f"rank {args.rank} unavailable")
    return reconstruct_curves(bag.curves, mean, components, args.rank)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.teacher_cache.resolve(), map_location="cpu", weights_only=False)
    config, bags = payload["config"], payload["bags"]
    expected = len(config["datasets"]) * len(config["seeds"]) * int(config["bags"])
    if len(bags) != expected:
        raise ValueError(f"incomplete teacher cache: {len(bags)}/{expected}")
    configure_compat_args(args, config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    backbone, _ = load_backbone(args, device)
    conditions = NORMALIZATION_CONDITIONS if args.mode == "normalization" else ("consensus_default",)
    rows: list[dict[str, object]] = []
    candidate_probs: dict[tuple[str, str, int], list[torch.Tensor]] = defaultdict(list)
    identity_probs: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    teacher_probs: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    labels: dict[tuple[str, int], np.ndarray] = {}
    for index, bag in enumerate(bags, start=1):
        support_x, support_y = bag.support_x.to(device), bag.support_y.to(device)
        identity = restore_spline(support_x, args, bag.identity_state)
        teacher = restore_spline(support_x, args, bag.teacher_state)
        identity_loss, identity_acc, identity_prob = evaluate_chunked(backbone, identity, support_x, support_y, bag.test_x, bag.test_y, chunk_rows=args.evaluation_query_chunk_rows, device=device)
        teacher_loss, teacher_acc, teacher_prob = evaluate_chunked(backbone, teacher, support_x, support_y, bag.test_x, bag.test_y, chunk_rows=args.evaluation_query_chunk_rows, device=device)
        key = (bag.dataset, bag.seed)
        identity_probs[key].append(identity_prob)
        teacher_probs[key].append(teacher_prob)
        labels[key] = bag.test_y
        curve = candidate_curve(args, bags, bag)
        for condition in conditions:
            choice = "default" if condition == "consensus_default" else condition
            candidate, projection_rmse = project_curve(identity, teacher, curve, args, choice)
            loss, acc, prob = evaluate_chunked(backbone, candidate, support_x, support_y, bag.test_x, bag.test_y, chunk_rows=args.evaluation_query_chunk_rows, device=device)
            candidate_probs[(condition, *key)].append(prob)
            available = identity_loss - teacher_loss
            rows.append({"mode": args.mode, "condition": condition, "dataset": bag.dataset, "outer_seed": bag.seed, "bag": bag.bag,
                         "identity_outer_test_loss": identity_loss, "teacher_outer_test_loss": teacher_loss, "candidate_outer_test_loss": loss,
                         "identity_outer_test_accuracy": identity_acc, "teacher_outer_test_accuracy": teacher_acc, "candidate_outer_test_accuracy": acc,
                         "headroom_recovery_loss": None if available <= 1e-8 else (identity_loss - loss) / available,
                         "projection_rmse": projection_rmse})
            del candidate
        del identity, teacher, support_x, support_y
        release_cuda(device)
        if index % 8 == 0 or index == len(bags):
            print(f"Evaluated {index}/{len(bags)} bags", flush=True)
    summaries = []
    for condition in conditions:
        runs = []
        for dataset, seed in sorted(labels):
            key = (dataset, seed)
            identity_loss, identity_acc = ensemble_metrics(identity_probs[key], labels[key])
            teacher_loss, teacher_acc = ensemble_metrics(teacher_probs[key], labels[key])
            loss, acc = ensemble_metrics(candidate_probs[(condition, *key)], labels[key])
            available = identity_loss - teacher_loss
            group = [row for row in rows if row["condition"] == condition and row["dataset"] == dataset and row["outer_seed"] == seed]
            runs.append({"dataset": dataset, "outer_seed": seed, "candidate_minus_identity_loss": loss - identity_loss,
                         "teacher_minus_identity_loss": teacher_loss - identity_loss, "candidate_minus_identity_accuracy": acc - identity_acc,
                         "teacher_minus_identity_accuracy": teacher_acc - identity_acc,
                         "headroom_recovery_loss": None if available <= 1e-8 else (identity_loss - loss) / available,
                         "mean_projection_rmse": float(np.mean([row["projection_rmse"] for row in group]))})
        fields = ("candidate_minus_identity_loss", "teacher_minus_identity_loss", "candidate_minus_identity_accuracy", "teacher_minus_identity_accuracy", "headroom_recovery_loss", "mean_projection_rmse")
        summaries.append({"condition": condition, "runs": runs, "macro_mean": {field: float(np.mean([run[field] for run in runs if run[field] is not None])) for field in fields}})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "folds.csv", rows)
    (output / "summary.json").write_text(json.dumps({"mode": args.mode, "rank": args.rank, "consensus": args.consensus, "summaries": summaries}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
