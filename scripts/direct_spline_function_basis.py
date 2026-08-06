"""Oracle function-space PCA of DirectSpline teachers.

This experiment asks whether independently fitted per-column spline functions
can be represented by a small shared vocabulary of curve shapes.  It is
invariant to raw spline parameters and knot locations: each teacher is first
represented by its actual residual curve on a canonical standardized grid.

For every held-out dataset, PCA is fitted only on teacher curves from the
other supplied datasets.  Target-teacher coefficients are then used as an
oracle to reconstruct each target curve at ranks 0, 1, 2, 4, 8, and 16.  The
reconstruction is projected into a valid monotone DirectSpline before frozen
TabICL evaluation.  Thus this is an output-capacity experiment, not a
HyperSpline-conditioning experiment yet.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from scripts.direct_spline_dataset_headroom import (
        ensemble_metrics, evaluate_chunked, make_direct_spline,
        make_direct_spline_optimizer, optimize_direct_spline, release_cuda,
        stratified_subset, to_device,
    )
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import load_pmlb_frame
except ModuleNotFoundError:  # pragma: no cover
    from direct_spline_dataset_headroom import (
        ensemble_metrics, evaluate_chunked, make_direct_spline,
        make_direct_spline_optimizer, optimize_direct_spline, release_cuda,
        stratified_subset, to_device,
    )
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import load_pmlb_frame


@dataclass
class TeacherBag:
    dataset: str
    seed: int
    bag: int
    support_x: torch.Tensor
    support_y: torch.Tensor
    identity_state: dict[str, torch.Tensor]
    teacher_state: dict[str, torch.Tensor]
    curves: torch.Tensor  # (D, G), full teacher residual curves in standardized output space
    guard_x: np.ndarray
    guard_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    teacher_initial_train_loss: float
    teacher_final_train_loss: float


def parse_csv(value: str, converter):
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True,
                        help="Repeat at least twice; PCA for each dataset excludes its own teachers.")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer-test-size", type=float, default=0.20)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--max-context-rows", type=int, default=512)
    parser.add_argument("--train-context-rows", type=int, default=384)
    parser.add_argument("--query-batch-rows", type=int, default=256)
    parser.add_argument("--evaluation-query-chunk-rows", type=int, default=256)
    parser.add_argument("--teacher-steps", type=int, default=1_750)
    parser.add_argument("--teacher-lr", type=float, default=0.03)
    parser.add_argument("--teacher-log-every", type=int, default=250)
    parser.add_argument("--transform-regularization", type=float, default=0.0)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--grid-points", type=int, default=129)
    parser.add_argument("--grid-range", type=float, default=4.0)
    parser.add_argument("--ranks", default="0,1,2,4,8,16")
    parser.add_argument("--projection-steps", type=int, default=250,
                        help="Label-free steps to project reconstructed curves into valid monotone splines.")
    parser.add_argument("--projection-lr", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed teacher bags from output-dir/teacher_bags.pt.")
    return parser.parse_args()


def state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_spline(support_x: torch.Tensor, args: argparse.Namespace, state: dict[str, torch.Tensor]):
    spline = make_direct_spline(support_x, args, knot_placement="uniform", control_mode="monotone").to(support_x.device)
    spline.load_state_dict(state)
    return spline.eval()


@torch.no_grad()
def teacher_residual_curves(teacher, grid_points: int, grid_range: float) -> torch.Tensor:
    """Return T(z)-z for every column on a common standardized-coordinate grid."""
    location, scale, _ = teacher._location_scale_range()
    z = torch.linspace(-grid_range, grid_range, grid_points, device=location.device, dtype=location.dtype)
    raw = location.unsqueeze(1) + scale.unsqueeze(1) * z.view(1, -1, 1)
    residual = teacher.transform(raw) - z.view(1, -1, 1)
    return residual.squeeze(0).transpose(0, 1).cpu()  # (D, G)


def fit_pca(curves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean curve, right-singular vectors, and explained-variance ratios."""
    if curves.ndim != 2 or curves.shape[0] < 2:
        raise ValueError("PCA needs at least two curve samples")
    mean = curves.mean(dim=0)
    _, singular, right = torch.linalg.svd(curves - mean, full_matrices=False)
    variance = singular.square()
    ratio = variance / variance.sum().clamp_min(1e-12)
    return mean, right, ratio


def reconstruct_curves(curves: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return mean.expand_as(curves)
    basis = components[:rank]
    return mean + (curves - mean) @ basis.transpose(0, 1) @ basis


def project_reconstructed_curves(identity, teacher, reconstructed: torch.Tensor, args: argparse.Namespace):
    """Fit valid monotone controls to PCA-reconstructed full transformation curves.

    Location/scale/range remain those of the target teacher.  Only controls and
    the per-column spline gate are projected, with no labels or TabICL calls.
    """
    candidate = copy.deepcopy(identity).train()
    with torch.no_grad():
        candidate.location_offsets.copy_(teacher.location_offsets)
        candidate.log_scale_offsets.copy_(teacher.log_scale_offsets)
        candidate.range_logits.copy_(teacher.range_logits)
        candidate.gate_logits.copy_(teacher.gate_logits)
        candidate.gap_logits.zero_()
    location, scale, _ = teacher._location_scale_range()
    z = torch.linspace(-args.grid_range, args.grid_range, args.grid_points, device=location.device, dtype=location.dtype)
    raw = (location.unsqueeze(1) + scale.unsqueeze(1) * z.view(1, -1, 1)).detach()
    target = (z.view(1, -1, 1) + reconstructed.transpose(0, 1).unsqueeze(0).to(raw)).detach()
    optimizer = torch.optim.Adam((candidate.gap_logits, candidate.gate_logits), lr=args.projection_lr)
    for _ in range(args.projection_steps):
        optimizer.zero_grad(set_to_none=True)
        error = (candidate.transform(raw) - target).square().mean()
        error.backward()
        optimizer.step()
    with torch.no_grad():
        curve_rmse = float((candidate.transform(raw) - target).square().mean().sqrt())
        teacher_rmse = float((candidate.transform(raw) - teacher.transform(raw)).square().mean().sqrt())
    return candidate.eval(), curve_rmse, teacher_rmse


def teacher_cache_config(args: argparse.Namespace) -> dict[str, object]:
    """Fields that must agree before cached teachers are safe to reuse."""
    return {
        "datasets": list(args.pmlb_dataset), "seeds": list(args.seeds), "outer_test_size": args.outer_test_size,
        "bags": args.bags, "max_context_rows": args.max_context_rows,
        "teacher_steps": args.teacher_steps, "teacher_lr": args.teacher_lr,
        "transform_regularization": args.transform_regularization,
        "n_control_points": args.n_control_points, "grid_points": args.grid_points,
        "grid_range": args.grid_range,
    }


def train_teacher_bags(args: argparse.Namespace, backbone, device: torch.device) -> list[TeacherBag]:
    cache_path = args.output_dir / "teacher_bags.pt"
    config = teacher_cache_config(args)
    bags: list[TeacherBag] = []
    if args.resume and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("config") != config:
            raise ValueError(
                f"teacher cache {cache_path} was made with different training settings; "
                "use a new output directory or remove that cache deliberately"
            )
        bags = payload["bags"]
        print(f"Resuming with {len(bags)} completed teacher bags from {cache_path}", flush=True)
    completed = {(bag.dataset, bag.seed, bag.bag) for bag in bags}
    for dataset in args.pmlb_dataset:
        x, y, _ = load_pmlb_frame(dataset, cache_dir=args.pmlb_cache_dir)
        _, y = np.unique(y, return_inverse=True)
        labels, counts = np.unique(y, return_counts=True)
        if labels.size < 2 or labels.size > backbone.max_classes or counts.min() < args.bags:
            raise ValueError(f"{dataset}: unsupported class count or insufficient rows")
        for seed in args.seeds:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=args.outer_test_size, random_state=seed, stratify=y
            )
            bagger = StratifiedKFold(n_splits=args.bags, shuffle=True, random_state=seed + 1)
            for bag, (fit_indices, guard_indices) in enumerate(bagger.split(x_train, y_train)):
                if (dataset, seed, bag) in completed:
                    continue
                fit_x, fit_y = x_train[fit_indices], y_train[fit_indices]
                guard_x, guard_y = x_train[guard_indices], y_train[guard_indices]
                n_classes = np.unique(fit_y).size
                support_cap = min(args.max_context_rows, max(n_classes, fit_y.size - n_classes))
                support_size = min(support_cap, max(n_classes, int(round(fit_y.size * 0.50))))
                support_rows = stratified_subset(fit_y, support_size, np.random.default_rng(seed + 10_000 + bag))
                query_mask = np.ones(fit_y.size, dtype=bool); query_mask[support_rows] = False
                adaptation_x, adaptation_y = fit_x[query_mask], fit_y[query_mask]
                support_x, support_y = to_device(fit_x, fit_y, support_rows, device)
                torch.manual_seed(seed + 10_000_000 + bag)
                identity = make_direct_spline(support_x, args, knot_placement="uniform", control_mode="monotone").to(device).eval()
                teacher = copy.deepcopy(identity).train()
                print(f"[{dataset} seed={seed} bag={bag}] fitting teacher", flush=True)
                initial, final = optimize_direct_spline(
                    backbone, teacher, make_direct_spline_optimizer(teacher, args), support_x, support_y,
                    adaptation_x, adaptation_y, steps=args.teacher_steps,
                    sample_rng=np.random.default_rng(seed + 100_000 + bag), args=args,
                    log_every=args.teacher_log_every, log_prefix=f"[{dataset} seed={seed} bag={bag} teacher]",
                )
                teacher.eval()
                bags.append(TeacherBag(
                    dataset, seed, bag, support_x.cpu(), support_y.cpu(), state_dict_cpu(identity), state_dict_cpu(teacher),
                    teacher_residual_curves(teacher, args.grid_points, args.grid_range), guard_x, guard_y, x_test, y_test,
                    initial, final,
                ))
                torch.save({"config": config, "bags": bags}, cache_path)
                del identity, teacher, support_x, support_y
                release_cuda(device)
    return bags


def evaluate_bag_ranks(
    args: argparse.Namespace, backbone, device: torch.device, bags: list[TeacherBag]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[int, str, int], list[torch.Tensor]],
    dict[tuple[str, int], list[torch.Tensor]],
    dict[tuple[str, int], list[torch.Tensor]],
    dict[tuple[str, int], np.ndarray],
]:
    rows: list[dict[str, object]] = []
    basis_rows: list[dict[str, object]] = []
    candidate_probabilities: dict[tuple[int, str, int], list[torch.Tensor]] = {}
    identity_probabilities: dict[tuple[str, int], list[torch.Tensor]] = {}
    teacher_probabilities: dict[tuple[str, int], list[torch.Tensor]] = {}
    test_labels: dict[tuple[str, int], np.ndarray] = {}
    for dataset in args.pmlb_dataset:
        train_curves = torch.cat([bag.curves for bag in bags if bag.dataset != dataset], dim=0)
        mean, components, ratio = fit_pca(train_curves)
        available = components.shape[0]
        valid_ranks = [rank for rank in args.ranks if rank <= available]
        basis_rows.append({
            "held_out_dataset": dataset, "train_curve_count": int(train_curves.shape[0]), "grid_points": args.grid_points,
            "available_components": int(available), "explained_variance_ratio": json.dumps(ratio[:min(32, available)].tolist()),
            "cumulative_explained_variance": json.dumps(ratio.cumsum(0)[:min(32, available)].tolist()),
        })
        for bag in (item for item in bags if item.dataset == dataset):
            support_x, support_y = bag.support_x.to(device), bag.support_y.to(device)
            identity = restore_spline(support_x, args, bag.identity_state)
            teacher = restore_spline(support_x, args, bag.teacher_state)
            identity_guard_loss, identity_guard_accuracy, _ = evaluate_chunked(
                backbone, identity, support_x, support_y, bag.guard_x, bag.guard_y,
                chunk_rows=args.evaluation_query_chunk_rows, device=device)
            teacher_guard_loss, teacher_guard_accuracy, _ = evaluate_chunked(
                backbone, teacher, support_x, support_y, bag.guard_x, bag.guard_y,
                chunk_rows=args.evaluation_query_chunk_rows, device=device)
            identity_test_loss, identity_test_accuracy, identity_probability = evaluate_chunked(
                backbone, identity, support_x, support_y, bag.test_x, bag.test_y,
                chunk_rows=args.evaluation_query_chunk_rows, device=device)
            teacher_test_loss, teacher_test_accuracy, teacher_probability = evaluate_chunked(
                backbone, teacher, support_x, support_y, bag.test_x, bag.test_y,
                chunk_rows=args.evaluation_query_chunk_rows, device=device)
            run_key = (dataset, bag.seed)
            identity_probabilities.setdefault(run_key, []).append(identity_probability)
            teacher_probabilities.setdefault(run_key, []).append(teacher_probability)
            if run_key in test_labels:
                if not np.array_equal(test_labels[run_key], bag.test_y):
                    raise RuntimeError("bags from one outer split must share test labels")
            else:
                test_labels[run_key] = bag.test_y
            for rank in valid_ranks:
                reconstructed = reconstruct_curves(bag.curves, mean, components, rank)
                candidate, projection_rmse, teacher_rmse = project_reconstructed_curves(identity, teacher, reconstructed, args)
                guard_loss, guard_accuracy, _ = evaluate_chunked(
                    backbone, candidate, support_x, support_y, bag.guard_x, bag.guard_y,
                    chunk_rows=args.evaluation_query_chunk_rows, device=device)
                test_loss, test_accuracy, candidate_probability = evaluate_chunked(
                    backbone, candidate, support_x, support_y, bag.test_x, bag.test_y,
                    chunk_rows=args.evaluation_query_chunk_rows, device=device)
                teacher_headroom = identity_test_loss - teacher_test_loss
                recovery = float("nan") if teacher_headroom <= 1e-8 else (identity_test_loss - test_loss) / teacher_headroom
                curve_rmse = float((reconstructed - bag.curves).square().mean().sqrt())
                candidate_probabilities.setdefault((rank, *run_key), []).append(candidate_probability)
                rows.append({
                    "dataset": dataset, "outer_seed": bag.seed, "bag": bag.bag, "rank": rank,
                    "train_curve_count": int(train_curves.shape[0]), "teacher_initial_train_loss": bag.teacher_initial_train_loss,
                    "teacher_final_train_loss": bag.teacher_final_train_loss,
                    "identity_guard_loss": identity_guard_loss, "teacher_guard_loss": teacher_guard_loss,
                    "candidate_guard_loss": guard_loss, "identity_guard_accuracy": identity_guard_accuracy,
                    "teacher_guard_accuracy": teacher_guard_accuracy, "candidate_guard_accuracy": guard_accuracy,
                    "identity_outer_test_loss": identity_test_loss, "teacher_outer_test_loss": teacher_test_loss,
                    "candidate_outer_test_loss": test_loss, "identity_outer_test_accuracy": identity_test_accuracy,
                    "teacher_outer_test_accuracy": teacher_test_accuracy, "candidate_outer_test_accuracy": test_accuracy,
                    "curve_reconstruction_rmse": curve_rmse, "projection_to_monotone_rmse": projection_rmse,
                    "candidate_vs_teacher_curve_rmse": teacher_rmse, "teacher_headroom_recovery_loss": recovery,
                })
                del candidate
            del identity, teacher, support_x, support_y
            release_cuda(device)
    return rows, basis_rows, candidate_probabilities, identity_probabilities, teacher_probabilities, test_labels


def summarize(
    rows: list[dict[str, object]],
    args: argparse.Namespace,
    candidate_probabilities: dict[tuple[int, str, int], list[torch.Tensor]],
    identity_probabilities: dict[tuple[str, int], list[torch.Tensor]],
    teacher_probabilities: dict[tuple[str, int], list[torch.Tensor]],
    test_labels: dict[tuple[str, int], np.ndarray],
) -> dict[str, object]:
    output: list[dict[str, object]] = []
    for rank in sorted({int(row["rank"]) for row in rows}):
        rank_rows = [row for row in rows if int(row["rank"]) == rank]
        by_run: dict[tuple[str, int], list[dict[str, object]]] = {}
        for row in rank_rows:
            by_run.setdefault((str(row["dataset"]), int(row["outer_seed"])), []).append(row)
        runs = []
        for (dataset, seed), group in by_run.items():
            key = (dataset, seed)
            labels = test_labels[key]
            identity_loss, identity_accuracy = ensemble_metrics(identity_probabilities[key], labels)
            teacher_loss, teacher_accuracy = ensemble_metrics(teacher_probabilities[key], labels)
            candidate_loss, candidate_accuracy = ensemble_metrics(candidate_probabilities[(rank, dataset, seed)], labels)
            teacher_headroom = identity_loss - teacher_loss
            recovery = float("nan") if teacher_headroom <= 1e-8 else (identity_loss - candidate_loss) / teacher_headroom
            runs.append({"dataset": dataset, "outer_seed": seed,
                         "candidate_minus_identity_loss": candidate_loss - identity_loss,
                         "teacher_minus_identity_loss": teacher_loss - identity_loss,
                         "candidate_minus_identity_accuracy": candidate_accuracy - identity_accuracy,
                         "teacher_minus_identity_accuracy": teacher_accuracy - identity_accuracy,
                         "headroom_recovery_loss": recovery,
                         "mean_curve_reconstruction_rmse": float(np.mean([row["curve_reconstruction_rmse"] for row in group])),
                         "mean_projection_to_monotone_rmse": float(np.mean([row["projection_to_monotone_rmse"] for row in group]))})
        output.append({"rank": rank, "runs": runs,
                       "macro_mean": {key: float(np.mean([run[key] for run in runs])) for key in runs[0] if key not in {"dataset", "outer_seed"}}})
    return {"protocol": "leave_one_dataset_out_function_space_pca + oracle_target_coefficients + monotone_projection + outer_holdout",
            "teacher_steps": args.teacher_steps, "grid_points": args.grid_points, "grid_range": args.grid_range,
            "projection_steps": args.projection_steps, "rank_summaries": output}


def main() -> None:
    args = parse_args()
    args.seeds, args.ranks = parse_csv(args.seeds, int), parse_csv(args.ranks, int)
    if len(args.pmlb_dataset) < 2 or len(set(args.pmlb_dataset)) != len(args.pmlb_dataset):
        raise ValueError("supply at least two unique datasets for leave-one-dataset-out PCA")
    if sorted(set(args.ranks)) != args.ranks or any(rank < 0 for rank in args.ranks):
        raise ValueError("ranks must be sorted, unique, and non-negative")
    if min(args.bags, args.max_context_rows, args.train_context_rows, args.query_batch_rows,
           args.evaluation_query_chunk_rows, args.teacher_steps, args.grid_points, args.projection_steps) <= 0:
        raise ValueError("row budgets, steps, and grid points must be positive")
    if args.teacher_lr <= 0 or args.projection_lr <= 0 or args.grid_range <= 0 or args.teacher_log_every < 0:
        raise ValueError("invalid learning rate, range, or log interval")
    args.output_dir, args.pmlb_cache_dir = args.output_dir.resolve(), args.pmlb_cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Compatibility attributes for the shared, unchanged DirectSpline helpers.
    args.lr = args.teacher_lr; args.freeze_spline_shape = False; args.trainable_range = False
    args.trainable_location_scale = True; args.control_mode = "monotone"; args.free_control_bound = 1.0
    args.free_control_curvature_regularization = 0.0; args.free_control_reference_regularization = 0.0
    args.knot_placement = "uniform"; args.cross_column_mixing_rank = 0; args.cross_column_mixing_bound = 0.1
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    backbone, _ = load_backbone(args, device)
    bags = train_teacher_bags(args, backbone, device)
    rows, basis_rows, candidate_probabilities, identity_probabilities, teacher_probabilities, test_labels = evaluate_bag_ranks(
        args, backbone, device, bags
    )
    write_csv(args.output_dir / "folds.csv", rows)
    write_csv(args.output_dir / "basis.csv", basis_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            summarize(
                rows, args, candidate_probabilities, identity_probabilities,
                teacher_probabilities, test_labels,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rank-by-bag records and {len(basis_rows)} held-out PCA bases to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
