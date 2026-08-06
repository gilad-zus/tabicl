"""Test whether a sparse subset of DirectSpline columns retains teacher headroom.

Each bag first fits the unchanged monotone, uniform-knot DirectSpline teacher.
The teacher and identity transform are then frozen.  A small vector of
per-column mask logits alone is fitted on the same train-only adaptation pool:

    identity(x) + sigmoid(mask_j) * (teacher(x) - identity(x)).

The guard selects identity, the full teacher, or one sparse-mask strength;
the outer test is never used for that choice.  This is deliberately a new
script: it does not alter the quantile or monotonicity experiments.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


def parse_csv(value: str, converter):
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True)
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
    parser.add_argument("--transform-regularization", type=float, default=0.0)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--mask-steps", type=int, default=300)
    parser.add_argument("--mask-lr", type=float, default=0.10)
    parser.add_argument("--teacher-log-every", type=int, default=250)
    parser.add_argument("--mask-log-every", type=int, default=100)
    parser.add_argument("--sparsity-penalties", default="0,0.003,0.01,0.03,0.10",
                        help="L1 penalties on mean sigmoid mask; each is a guard candidate.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def masked_transform(identity, teacher, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Function-space interpolation; independent of the teacher's knot representation."""
    with torch.no_grad():
        baseline = identity.transform(x)
        delta = teacher.transform(x) - baseline
    return baseline + mask.view(1, 1, -1) * delta


def masked_logits(backbone, identity, teacher, mask, context_x, context_y, query_x):
    backbone.clear_cache()
    transformed = torch.cat((
        masked_transform(identity, teacher, context_x, mask),
        masked_transform(identity, teacher, query_x, mask),
    ), dim=1)
    return backbone(transformed, context_y)


@torch.no_grad()
def evaluate_masked_chunked(backbone, identity, teacher, mask, context_x, context_y, query_x, query_y, args, device):
    transformed_context = masked_transform(identity, teacher, context_x, mask)
    probabilities = []
    for start in range(0, query_y.size, args.evaluation_query_chunk_rows):
        end = min(start + args.evaluation_query_chunk_rows, query_y.size)
        query = torch.as_tensor(query_x[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        backbone.clear_cache()
        logits = backbone(torch.cat((transformed_context, masked_transform(identity, teacher, query, mask)), dim=1), context_y)
        probabilities.append(logits.softmax(dim=-1).cpu())
        del query, logits
    probability = torch.cat(probabilities, dim=1).clamp_min(1e-12)
    labels = torch.as_tensor(query_y, dtype=torch.long)
    return (
        float(F.nll_loss(probability.log().flatten(0, 1), labels.flatten())),
        float((probability.argmax(dim=-1).flatten() == labels).float().mean()),
        probability,
    )


def fit_mask(backbone, identity, teacher, support_x, support_y, adaptation_x, adaptation_y, penalty, args, rng, log_prefix):
    logits = torch.nn.Parameter(torch.full((support_x.shape[-1],), torch.logit(torch.tensor(0.90)), device=support_x.device))
    optimizer = torch.optim.Adam((logits,), lr=args.mask_lr)
    for step in range(1, args.mask_steps + 1):
        context_rows = stratified_subset(support_y.squeeze(0).cpu().numpy(), min(args.train_context_rows, support_y.shape[1]), rng)
        query_rows = stratified_subset(adaptation_y, min(args.query_batch_rows, adaptation_y.size), rng)
        query_x, query_y = to_device(adaptation_x, adaptation_y, query_rows, support_x.device)
        mask = logits.sigmoid()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(masked_logits(
            backbone, identity, teacher, mask, support_x[:, context_rows], support_y[:, context_rows], query_x
        ).flatten(0, 1), query_y.long().flatten()) + penalty * mask.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((logits,), 1.0)
        optimizer.step()
        if args.mask_log_every and (step == 1 or step % args.mask_log_every == 0 or step == args.mask_steps):
            print(f"{log_prefix} step={step}/{args.mask_steps} objective={float(loss.detach()):.6f} mask_mean={float(mask.mean().detach()):.3f}", flush=True)
        del query_x, query_y, loss
    return logits.detach().sigmoid()


def run_one(args, backbone, dataset: str, seed: int, device: torch.device):
    x, y, _ = load_pmlb_frame(dataset, cache_dir=args.pmlb_cache_dir)
    _, y = np.unique(y, return_inverse=True)
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2 or labels.size > backbone.max_classes or counts.min() < args.bags:
        raise ValueError("dataset must have 2..max_classes labels and enough rows per class")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.outer_test_size, random_state=seed, stratify=y)
    rows, selected_probabilities, identity_probabilities, teacher_probabilities = [], [], [], []
    bagger = StratifiedKFold(n_splits=args.bags, shuffle=True, random_state=seed + 1)
    for bag, (fit_indices, guard_indices) in enumerate(bagger.split(x_train, y_train)):
        print(f"[{dataset} seed={seed} bag={bag}] fitting DirectSpline teacher", flush=True)
        fit_x, fit_y, guard_x, guard_y = x_train[fit_indices], y_train[fit_indices], x_train[guard_indices], y_train[guard_indices]
        n_classes = np.unique(fit_y).size
        support_size = min(args.max_context_rows, max(n_classes, int(round(fit_y.size * 0.50))))
        support_rows = stratified_subset(fit_y, support_size, np.random.default_rng(seed + 10_000 + bag))
        query_mask = np.ones(fit_y.size, dtype=bool); query_mask[support_rows] = False
        adaptation_x, adaptation_y = fit_x[query_mask], fit_y[query_mask]
        support_x, support_y = to_device(fit_x, fit_y, support_rows, device)
        torch.manual_seed(seed + 10_000_000 + bag)
        identity = make_direct_spline(support_x, args, knot_placement="uniform", control_mode="monotone").to(device).eval()
        teacher = copy.deepcopy(identity).train()
        teacher_optimizer = make_direct_spline_optimizer(teacher, args)
        teacher_initial, teacher_final = optimize_direct_spline(
            backbone, teacher, teacher_optimizer, support_x, support_y, adaptation_x, adaptation_y,
            steps=args.teacher_steps, sample_rng=np.random.default_rng(seed + 100_000 + bag), args=args,
            log_every=args.teacher_log_every, log_prefix=f"[{dataset} seed={seed} bag={bag} teacher]",
        )
        teacher.eval()
        identity_guard_loss, identity_guard_accuracy, identity_test_probability = evaluate_chunked(
            backbone, identity, support_x, support_y, guard_x, guard_y, chunk_rows=args.evaluation_query_chunk_rows, device=device)
        teacher_guard_loss, teacher_guard_accuracy, teacher_test_probability = evaluate_chunked(
            backbone, teacher, support_x, support_y, guard_x, guard_y, chunk_rows=args.evaluation_query_chunk_rows, device=device)
        identity_test_loss, identity_test_accuracy, identity_test_probability = evaluate_chunked(
            backbone, identity, support_x, support_y, x_test, y_test, chunk_rows=args.evaluation_query_chunk_rows, device=device)
        teacher_test_loss, teacher_test_accuracy, teacher_test_probability = evaluate_chunked(
            backbone, teacher, support_x, support_y, x_test, y_test, chunk_rows=args.evaluation_query_chunk_rows, device=device)
        candidates = [("identity", None, identity_guard_loss, identity_guard_accuracy, identity_test_loss, identity_test_accuracy, identity_test_probability),
                      ("teacher", None, teacher_guard_loss, teacher_guard_accuracy, teacher_test_loss, teacher_test_accuracy, teacher_test_probability)]
        for penalty in args.sparsity_penalty_values:
            print(f"[{dataset} seed={seed} bag={bag}] fitting sparse mask penalty={penalty:g}", flush=True)
            mask = fit_mask(backbone, identity, teacher, support_x, support_y, adaptation_x, adaptation_y, penalty, args,
                            np.random.default_rng(seed + 200_000 + bag),
                            f"[{dataset} seed={seed} bag={bag} mask penalty={penalty:g}]")
            guard_loss, guard_accuracy, _ = evaluate_masked_chunked(backbone, identity, teacher, mask, support_x, support_y, guard_x, guard_y, args, device)
            test_loss, test_accuracy, test_probability = evaluate_masked_chunked(backbone, identity, teacher, mask, support_x, support_y, x_test, y_test, args, device)
            candidates.append(("sparse_mask", penalty, guard_loss, guard_accuracy, test_loss, test_accuracy, test_probability, mask.detach().cpu()))
        selected = min(candidates, key=lambda item: item[2])
        selected_probabilities.append(selected[6]); identity_probabilities.append(identity_test_probability); teacher_probabilities.append(teacher_test_probability)
        for candidate in candidates:
            kind, penalty, guard_loss, guard_accuracy, test_loss, test_accuracy, _ = candidate[:7]
            mask = candidate[7] if len(candidate) > 7 else None
            rows.append({
                "dataset": dataset, "outer_seed": seed, "bag": bag, "candidate": kind, "sparsity_penalty": penalty,
                "selected_by_guard": candidate is selected, "teacher_initial_train_loss": teacher_initial,
                "teacher_final_train_loss": teacher_final, "identity_guard_loss": identity_guard_loss,
                "candidate_guard_loss": guard_loss, "identity_guard_accuracy": identity_guard_accuracy,
                "candidate_guard_accuracy": guard_accuracy, "identity_outer_test_loss": identity_test_loss,
                "candidate_outer_test_loss": test_loss, "identity_outer_test_accuracy": identity_test_accuracy,
                "candidate_outer_test_accuracy": test_accuracy,
                "mask_mean": None if mask is None else float(mask.mean()),
                "mask_active_0_5": None if mask is None else int((mask >= 0.5).sum()),
                "mask_values": None if mask is None else json.dumps(mask.tolist()),
                "posthoc_outer_test_diagnostic": True,
            })
        del teacher_optimizer, teacher, identity, support_x, support_y
        release_cuda(device)
    identity_loss, identity_accuracy = ensemble_metrics(identity_probabilities, y_test)
    teacher_loss, teacher_accuracy = ensemble_metrics(teacher_probabilities, y_test)
    selected_loss, selected_accuracy = ensemble_metrics(selected_probabilities, y_test)
    return rows, {"dataset": dataset, "outer_seed": seed, "bags": args.bags, "n_features": int(x.shape[1]),
                  "identity_outer_test_loss": identity_loss, "identity_outer_test_accuracy": identity_accuracy,
                  "teacher_outer_test_loss": teacher_loss, "teacher_outer_test_accuracy": teacher_accuracy,
                  "guarded_sparse_outer_test_loss": selected_loss, "guarded_sparse_outer_test_accuracy": selected_accuracy,
                  "teacher_minus_identity_loss": teacher_loss - identity_loss,
                  "teacher_minus_identity_accuracy": teacher_accuracy - identity_accuracy,
                  "sparse_minus_identity_loss": selected_loss - identity_loss,
                  "sparse_minus_identity_accuracy": selected_accuracy - identity_accuracy,
                  "protocol": "outer_holdout + nested_bagging + train_only_teacher_and_masks + guard_selection"}


def main() -> None:
    args = parse_args()
    args.seeds, args.sparsity_penalty_values = parse_csv(args.seeds, int), parse_csv(args.sparsity_penalties, float)
    if args.teacher_steps <= 0 or args.mask_steps <= 0 or args.teacher_lr <= 0 or args.mask_lr <= 0 or args.bags < 2:
        raise ValueError("invalid optimisation configuration")
    if args.teacher_log_every < 0 or args.mask_log_every < 0:
        raise ValueError("log intervals must be non-negative")
    if any(value < 0 for value in args.sparsity_penalty_values):
        raise ValueError("sparsity penalties must be non-negative")
    args.pmlb_cache_dir, args.output_dir = args.pmlb_cache_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Compatibility attributes consumed by the unchanged DirectSpline helpers.
    args.n_control_points = args.n_control_points; args.lr = args.teacher_lr
    args.freeze_spline_shape = False; args.trainable_range = False; args.trainable_location_scale = True
    args.control_mode = "monotone"; args.free_control_bound = 1.0
    args.free_control_curvature_regularization = 0.0; args.free_control_reference_regularization = 0.0
    args.knot_placement = "uniform"; args.cross_column_mixing_rank = 0; args.cross_column_mixing_bound = 0.1
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    backbone, _ = load_backbone(args, device)
    folds_path, summary_path = args.output_dir / "folds.csv", args.output_dir / "summary.json"
    all_rows, summaries = [], []
    completed = set()
    if args.resume and summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8")); summaries = payload.get("runs", [])
        completed = {(item["dataset"], int(item["outer_seed"])) for item in summaries}
        if folds_path.is_file():
            with folds_path.open(newline="", encoding="utf-8") as handle: all_rows = list(csv.DictReader(handle))
    for dataset in args.pmlb_dataset:
        for seed in args.seeds:
            if (dataset, seed) in completed:
                continue
            rows, summary = run_one(args, backbone, dataset, seed, device)
            all_rows.extend(rows); summaries.append(summary)
            write_csv(folds_path, all_rows)
            macro = {key: float(np.mean([run[key] for run in summaries])) for key in ("teacher_minus_identity_loss", "teacher_minus_identity_accuracy", "sparse_minus_identity_loss", "sparse_minus_identity_accuracy")}
            summary_path.write_text(json.dumps({"runs": summaries, "macro_mean_over_dataset_seed_runs": macro}, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
