"""Measure DirectSpline oracle headroom on native TabICL synthetic tasks.

Each task is drawn directly from :class:`tabicl.prior.PriorDataset` without
additional numerical warping.  A separate DirectSpline is optimized on each
task's query labels, so this is an upper-bound diagnostic rather than a
zero-shot result.  It answers whether native synthetic tasks provide useful
numerical-preprocessing headroom before using them to train HyperSpline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from direct_spline_multidataset_headroom import load_backbone
from tabicl._hyperspline import DirectSplineTransform


def evaluate(
    backbone,
    spline: DirectSplineTransform,
    x_context: torch.Tensor,
    x_query: torch.Tensor,
    y_context: torch.Tensor,
    y_query: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
        logits = backbone(transformed, y_context)
        loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
        accuracy = (logits.argmax(dim=-1).flatten() == y_query.flatten()).float().mean()
    return loss.item(), accuracy.item()


def make_prior(args: argparse.Namespace):
    try:
        from tabicl.prior import PriorDataset
    except ModuleNotFoundError as error:
        if error.name == "xgboost":
            raise RuntimeError(
                "Native TabICL prior generation requires xgboost. Install the project's pretraining dependencies "
                "before running this experiment (for example: pip install -e '.[pretrain]')."
            ) from error
        raise
    return PriorDataset(
        regression=False,
        batch_size=args.tasks,
        batch_size_per_gp=1,
        batch_size_per_subgp=1,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        # PriorDataset samples with randint(min_seq_len, max_seq_len), whose
        # upper bound is exclusive.  The one-element interval below therefore
        # gives exactly --sequence-length while remaining valid to the prior.
        min_seq_len=args.sequence_length,
        max_seq_len=args.sequence_length + 1,
        min_train_size=args.context_fraction,
        max_train_size=args.context_fraction,
        prior_type=args.prior_type,
        n_jobs=args.prior_n_jobs,
        num_threads_per_generate=1,
        device="cpu",
    )


def prepare_task(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    active_features: torch.Tensor,
    sequence_length: torch.Tensor,
    context_size: torch.Tensor,
    index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows = int(sequence_length[index])
    n_context = int(context_size[index])
    n_features = int(active_features[index])
    if not 0 < n_context < n_rows:
        raise ValueError(f"synthetic task {index} has invalid context/query split {n_context}/{n_rows}")
    x = x_batch[index, :n_rows, :n_features].to(device=device, dtype=torch.float32)
    # Map the generator's classes to contiguous indices for cross entropy.
    _, y = torch.unique(y_batch[index, :n_rows].to(torch.long), sorted=True, return_inverse=True)
    x_context, x_query = x[:n_context].unsqueeze(0), x[n_context:].unsqueeze(0)
    y_context = y[:n_context].to(device=device, dtype=torch.float32).unsqueeze(0)
    y_query = y[n_context:].to(device=device, dtype=torch.long)
    return x_context, x_query, y_context, y_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=int, default=8, help="Independent synthetic tasks in this diagnostic.")
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1, help="Use 1 for portable, deterministic task generation.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=Path("results/direct_spline_synthetic_headroom.csv"))
    args = parser.parse_args()

    if args.tasks <= 0 or args.sequence_length < 4 or args.steps <= 0 or args.log_every <= 0:
        raise ValueError("--tasks, --steps, and --log-every must be positive; --sequence-length must be at least 4")
    if not 0 < args.context_fraction < 1:
        raise ValueError("--context-fraction must be between 0 and 1")
    if not 0 < args.min_features <= args.max_features:
        raise ValueError("invalid feature range")
    if args.n_control_points <= 3:
        raise ValueError("--n-control-points must exceed 3")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    backbone, _ = load_backbone(args, device)
    if args.max_classes > backbone.max_classes:
        raise ValueError(
            f"--max-classes={args.max_classes} exceeds the frozen backbone maximum {backbone.max_classes}"
        )
    print(
        f"Generating {args.tasks} native {args.prior_type} tasks: sequence_length={args.sequence_length}, "
        f"features={args.min_features}-{args.max_features}, context_fraction={args.context_fraction}",
        flush=True,
    )
    prior = make_prior(args)
    x_batch, y_batch, active_features, sequence_lengths, context_sizes = prior.get_batch()
    print(
        f"Generated batch: X={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}, "
        f"active_features={active_features.tolist()}, context_sizes={context_sizes.tolist()}",
        flush=True,
    )
    results: list[dict[str, float | int | str]] = []
    for index in range(args.tasks):
        x_context, x_query, y_context, y_query = prepare_task(
            x_batch, y_batch, active_features, sequence_lengths, context_sizes, index, device
        )
        n_classes = int(torch.unique(y_context).numel())
        if n_classes > backbone.max_classes:
            raise ValueError(f"synthetic task {index} has {n_classes} context classes; backbone supports {backbone.max_classes}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        backbone.clear_cache()
        spline = DirectSplineTransform(x_context, args.n_control_points).to(device)
        optimizer = torch.optim.Adam(spline.parameters(), lr=args.lr)
        baseline_loss, baseline_accuracy = evaluate(backbone, spline, x_context, x_query, y_context, y_query)
        print(
            f"[task={index}] context={tuple(x_context.shape)}, query={tuple(x_query.shape)}, "
            f"classes={n_classes}, baseline_loss={baseline_loss:.6f}, baseline_accuracy={baseline_accuracy:.4f}",
            flush=True,
        )
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            backbone.clear_cache()
            transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
            logits = backbone(transformed, y_context)
            loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
            loss.backward()
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"[task={index}] step={step} loss={loss.item():.6f}", flush=True)
        backbone.clear_cache()
        final_loss, final_accuracy = evaluate(backbone, spline, x_context, x_query, y_context, y_query)
        result = {
            "task": index,
            "seed": args.seed,
            "prior_type": args.prior_type,
            "n_context": int(x_context.shape[1]),
            "n_query": int(x_query.shape[1]),
            "n_features": int(x_context.shape[2]),
            "n_classes": n_classes,
            "baseline_loss": baseline_loss,
            "final_loss": final_loss,
            "loss_delta": baseline_loss - final_loss,
            "relative_loss_improvement": (baseline_loss - final_loss) / max(baseline_loss, 1e-12),
            "baseline_accuracy": baseline_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_delta": final_accuracy - baseline_accuracy,
            "final_gate_mean": spline.parameters_for_transform().gate.mean().item(),
            "steps": args.steps,
            "learning_rate": args.lr,
            "n_control_points": args.n_control_points,
        }
        results.append(result)
        print(
            f"[task={index}] final_loss={final_loss:.6f} (delta={result['loss_delta']:+.6f}), "
            f"final_accuracy={final_accuracy:.4f} (delta={result['accuracy_delta']:+.4f}), "
            f"gate_mean={result['final_gate_mean']:.4f}",
            flush=True,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} synthetic headroom records to {args.output_csv}", flush=True)
    print(
        f"Summary: mean_relative_loss_improvement="
        f"{np.mean([row['relative_loss_improvement'] for row in results]):.2%}, "
        f"mean_accuracy_delta={np.mean([row['accuracy_delta'] for row in results]):+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
