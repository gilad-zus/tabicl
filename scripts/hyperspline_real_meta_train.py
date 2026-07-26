"""Meta-train a raw-context HyperSpline on held-out real tabular tasks.

The OpenML datasets in the training and validation banks must be disjoint from
the seven datasets used by ``hyperspline_real_paired_eval.py``.  Selection is
made exclusively on the real validation bank; the final suite is never read
by this program.
"""

from __future__ import annotations

import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import RealEpisode, load_bank
    from scripts.hyperspline_synthetic_train import (
        SyntheticEpisode,
        generate_episodes,
        seed_generator,
        validate_episode_classes,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct execution
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import RealEpisode, load_bank
    from hyperspline_synthetic_train import SyntheticEpisode, generate_episodes, seed_generator, validate_episode_classes

from tabicl._hyperspline import (
    HyperSplineTransform,
    backbone_state_dict_hash,
    save_hyperspline_checkpoint,
    summarize_context,
)
from tabicl._hyperspline.checkpoint import load_hyperspline_checkpoint


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_context_subset(y_context: torch.Tensor, fraction: float) -> torch.Tensor:
    """Choose a non-empty, class-balanced subset of a single episode's context rows."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("context subset fraction must be in (0, 1)")
    if y_context.shape[0] != 1:
        raise ValueError("real-meta episodes currently require batch size one")
    labels = y_context[0]
    indices = []
    for label in torch.unique(labels, sorted=True):
        class_indices = torch.where(labels == label)[0]
        count = min(class_indices.numel(), max(1, int(np.ceil(class_indices.numel() * fraction))))
        indices.append(class_indices[torch.randperm(class_indices.numel(), device=labels.device)[:count]])
    return torch.cat(indices).sort().values


def context_subset_consistency_penalty(
    hyperspline: HyperSplineTransform,
    full_parameters,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    fraction: float | None,
) -> torch.Tensor:
    """Keep the generated transform stable under a stratified context subsample."""
    if fraction is None:
        return full_parameters.gate.new_zeros(())
    indices = stratified_context_subset(y_context, fraction)
    subset_x, subset_y = x_context[:, indices], y_context[:, indices]
    with torch.no_grad():
        subset_statistics = summarize_context(subset_x, y_context=subset_y, eps=hyperspline.eps)
    subset_parameters = hyperspline.generate_parameters(
        subset_statistics, x_context=subset_x, y_context=subset_y
    )
    return hyperspline.grid_deformation_penalty(full_parameters, reference_parameters=subset_parameters)


def real_forward(
    backbone, hyperspline: HyperSplineTransform, episode: RealEpisode, marginal: HyperSplineTransform,
    *, subset_fraction: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply HyperSpline only to numerical columns; categorical columns pass through unchanged."""
    numeric_context = episode.x_context[..., episode.numerical_mask]
    numeric_query = episode.x_query[..., episode.numerical_mask]
    with torch.no_grad():
        statistics = summarize_context(numeric_context, y_context=episode.y_context, eps=hyperspline.eps)
        marginal_statistics = summarize_context(numeric_context, eps=marginal.eps)
    parameters = hyperspline.generate_parameters(
        statistics, x_context=numeric_context, y_context=episode.y_context
    )
    # This is the *original* marginal policy, not the potentially unfrozen copy.
    reference_parameters = marginal.generate_parameters(marginal_statistics)
    current_marginal = hyperspline.generate_marginal_parameters(statistics)
    transformed_context = hyperspline.apply_transform(numeric_context, parameters)
    transformed_query = hyperspline.apply_transform(numeric_query, parameters)
    x_context = episode.x_context.clone()
    x_query = episode.x_query.clone()
    x_context[..., episode.numerical_mask] = transformed_context
    x_query[..., episode.numerical_mask] = transformed_query
    logits = backbone(torch.cat((x_context, x_query), dim=1), episode.y_context)
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    diagnostics = {
        "residual_grid_penalty": hyperspline.grid_deformation_penalty(
            parameters, reference_parameters=current_marginal
        ),
        "marginal_trust_penalty": hyperspline.grid_deformation_penalty(
            current_marginal, reference_parameters=reference_parameters
        ),
        "context_subset_consistency_penalty": context_subset_consistency_penalty(
            hyperspline, parameters, numeric_context, episode.y_context, subset_fraction
        ),
        "mean_residual_gate": parameters.supervised_residual_gate.mean(),
    }
    return loss, logits, diagnostics


def synthetic_forward(
    backbone, hyperspline: HyperSplineTransform, episode: SyntheticEpisode, marginal: HyperSplineTransform,
    *, subset_fraction: float | None = None,
):
    numeric_context, numeric_query = episode.x_context, episode.x_query
    with torch.no_grad():
        statistics = summarize_context(numeric_context, y_context=episode.y_context, eps=hyperspline.eps)
        marginal_statistics = summarize_context(numeric_context, eps=marginal.eps)
    parameters = hyperspline.generate_parameters(
        statistics, x_context=numeric_context, y_context=episode.y_context
    )
    reference_parameters = marginal.generate_parameters(marginal_statistics)
    current_marginal = hyperspline.generate_marginal_parameters(statistics)
    transformed = torch.cat(
        (hyperspline.apply_transform(numeric_context, parameters), hyperspline.apply_transform(numeric_query, parameters)),
        dim=1,
    )
    logits = backbone(transformed, episode.y_context)
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    return loss, logits, {
        "residual_grid_penalty": hyperspline.grid_deformation_penalty(parameters, reference_parameters=current_marginal),
        "marginal_trust_penalty": hyperspline.grid_deformation_penalty(
            current_marginal, reference_parameters=reference_parameters
        ),
        "context_subset_consistency_penalty": context_subset_consistency_penalty(
            hyperspline, parameters, numeric_context, episode.y_context, subset_fraction
        ),
        "mean_residual_gate": parameters.supervised_residual_gate.mean(),
    }


@torch.no_grad()
def evaluate_real_bank(backbone, hyperspline, marginal, episodes: list[RealEpisode]) -> tuple[float, list[dict]]:
    """Average datasets equally, so a dataset with more cached splits cannot dominate selection."""
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        backbone.clear_cache()
        loss, logits, diagnostics = real_forward(backbone, hyperspline, episode, marginal)
        by_dataset[episode.dataset].append(
            {
                "dataset": episode.dataset,
                "split_seed": episode.split_seed,
                "loss": float(loss),
                "accuracy": float((logits.argmax(dim=-1).flatten() == episode.y_query).float().mean()),
                **{name: float(value) for name, value in diagnostics.items()},
            }
        )
    rows = [row for values in by_dataset.values() for row in values]
    dataset_loss = [np.mean([row["loss"] for row in values]) for values in by_dataset.values()]
    return float(np.mean(dataset_loss)), rows


def add_synthetic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--synthetic-long-sequence-length", type=int, default=1024)
    parser.add_argument("--synthetic-long-fraction", type=float, default=0.75)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--marginal-checkpoint", type=Path, required=True)
    parser.add_argument("--real-train-bank", type=Path, required=True)
    parser.add_argument("--real-validation-bank", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--real-episode-fraction", type=float, default=0.75)
    parser.add_argument("--marginal-warmup-steps", type=int, default=2_000)
    parser.add_argument("--raw-lr", type=float, default=1e-3)
    parser.add_argument("--marginal-lr", type=float, default=1e-4)
    parser.add_argument("--residual-transform-regularization", type=float, default=1e-3)
    parser.add_argument("--marginal-trust-region-regularization", type=float, default=1e-3)
    parser.add_argument("--context-subset-consistency-regularization", type=float, default=1e-2)
    parser.add_argument("--context-subset-fraction", type=float, default=0.70)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--raw-context-num-heads", type=int, default=4)
    parser.add_argument("--raw-context-residual-bound", type=float, default=0.5)
    parser.add_argument("--raw-context-gate-initial-probability", type=float, default=0.5)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-train-csv", type=Path, required=True)
    parser.add_argument("--output-real-validation-csv", type=Path, required=True)
    add_synthetic_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0 or args.tasks_per_step <= 0 or args.validate_every <= 0:
        raise ValueError("train steps, tasks per step, and validation interval must be positive")
    if not 0.0 <= args.real_episode_fraction <= 1.0:
        raise ValueError("--real-episode-fraction must be in [0, 1]")
    if args.marginal_warmup_steps < 0 or args.marginal_warmup_steps > args.train_steps:
        raise ValueError("--marginal-warmup-steps must be between zero and train steps")
    if not 0.0 <= args.synthetic_long_fraction <= 1.0:
        raise ValueError("--synthetic-long-fraction must be in [0, 1]")
    if args.sequence_length < 4 or args.synthetic_long_sequence_length < 4:
        raise ValueError("synthetic sequence lengths must be at least four")
    if not 0.0 < args.context_subset_fraction < 1.0:
        raise ValueError("--context-subset-fraction must be in (0, 1)")
    if min(
        args.residual_transform_regularization,
        args.marginal_trust_region_regularization,
        args.context_subset_consistency_regularization,
    ) < 0:
        raise ValueError("regularization weights must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        print(f"Running on CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    backbone, _ = load_backbone(args, device)
    base, payload = load_hyperspline_checkpoint(
        args.marginal_checkpoint,
        device=device,
        expected_backbone_reference=args.checkpoint_version,
        expected_backbone_hash=backbone_state_dict_hash(backbone),
    )
    if base.target_aware or base.has_supervised_residual:
        raise ValueError("--marginal-checkpoint must be a marginal-only HyperSpline")
    config = dict(payload["hyperspline_config"])
    config.update(
        target_aware=True,
        raw_context_residual=True,
        raw_context_num_heads=args.raw_context_num_heads,
        raw_context_residual_bound=args.raw_context_residual_bound,
        raw_context_gate_initial_probability=args.raw_context_gate_initial_probability,
    )
    # These two are mutually exclusive architectural variants.
    config["supervised_residual"] = False
    config["cross_column_residual"] = False
    torch.manual_seed(args.model_seed)
    hyperspline = HyperSplineTransform(**config).to(device).train()
    hyperspline.initialize_supervised_residual_from(base)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    _, train_episodes = load_bank(args.real_train_bank, device=device)
    _, validation_episodes = load_bank(args.real_validation_bank, device=device)
    if not train_episodes or not validation_episodes:
        raise ValueError("real meta banks must each contain at least one episode")
    if {episode.dataset for episode in train_episodes}.intersection(episode.dataset for episode in validation_episodes):
        raise ValueError("real meta train and validation banks must be dataset-disjoint")
    if args.max_classes > backbone.max_classes:
        raise ValueError(f"--max-classes exceeds backbone maximum {backbone.max_classes}")
    marginal_parameters = list(hyperspline.mlp.parameters())
    raw_parameters = [parameter for name, parameter in hyperspline.named_parameters() if not name.startswith("mlp.")]
    optimizer = torch.optim.Adam(
        [{"params": raw_parameters, "lr": args.raw_lr}, {"params": marginal_parameters, "lr": args.marginal_lr}]
    )
    rng = np.random.default_rng(args.train_seed)
    seed_generator(args.train_seed)
    print(
        f"Initialized real-meta raw HyperSpline: real episodes={len(train_episodes)}, "
        f"validation episodes={len(validation_episodes)}, real_fraction={args.real_episode_fraction:.0%}, "
        f"marginal frozen through step {args.marginal_warmup_steps}",
        flush=True,
    )
    validation_rows: list[dict] = []
    train_rows: list[dict] = []
    hyperspline.eval()
    baseline_validation, baseline_rows = evaluate_real_bank(backbone, hyperspline, base, validation_episodes)
    for row in baseline_rows:
        row.update(step=0, phase="initial", loss_delta=0.0)
    validation_rows.extend(baseline_rows)
    best_loss, best_step = baseline_validation, 0
    best_state = copy.deepcopy({name: value.detach().cpu() for name, value in hyperspline.state_dict().items()})
    print(f"[real_validation initial] equal-dataset mean_loss={baseline_validation:.6f}", flush=True)

    for step in range(1, args.train_steps + 1):
        if step == args.marginal_warmup_steps + 1 and args.marginal_warmup_steps < args.train_steps:
            hyperspline.unfreeze_marginal_policy()
            print(f"[stage] unfroze marginal policy at step={step}", flush=True)
        hyperspline.train()
        optimizer.zero_grad(set_to_none=True)
        losses, grids, trusts, consistencies, gates, real_count, long_synthetic_count = [], [], [], [], [], 0, 0
        selected: list[tuple[str, object]] = []
        for _ in range(args.tasks_per_step):
            if rng.random() < args.real_episode_fraction:
                selected.append(("real", train_episodes[int(rng.integers(len(train_episodes)))]))
                real_count += 1
            else:
                sequence_length = (
                    args.synthetic_long_sequence_length if rng.random() < args.synthetic_long_fraction else args.sequence_length
                )
                long_synthetic_count += int(sequence_length == args.synthetic_long_sequence_length)
                synthetic_args = argparse.Namespace(**vars(args), sequence_length=sequence_length)
                selected.append(("synthetic", generate_episodes(
                    synthetic_args, 1, source_seed=None, task_offset=step * args.tasks_per_step + len(selected), device=device
                )[0]))
        for kind, episode in selected:
            backbone.clear_cache()
            if kind == "real":
                loss, _, diagnostics = real_forward(
                    backbone, hyperspline, episode, base, subset_fraction=args.context_subset_fraction
                )
            else:
                loss, _, diagnostics = synthetic_forward(
                    backbone, hyperspline, episode, base, subset_fraction=args.context_subset_fraction
                )
            objective = loss + args.residual_transform_regularization * diagnostics["residual_grid_penalty"]
            objective = objective + (
                args.context_subset_consistency_regularization * diagnostics["context_subset_consistency_penalty"]
            )
            if step > args.marginal_warmup_steps:
                objective = objective + args.marginal_trust_region_regularization * diagnostics["marginal_trust_penalty"]
            (objective / len(selected)).backward()
            losses.append(float(loss.detach()))
            grids.append(float(diagnostics["residual_grid_penalty"].detach()))
            trusts.append(float(diagnostics["marginal_trust_penalty"].detach()))
            consistencies.append(float(diagnostics["context_subset_consistency_penalty"].detach()))
            gates.append(float(diagnostics["mean_residual_gate"].detach()))
        norm = torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), 1.0)
        optimizer.step()
        train_rows.append({
            "step": step, "mean_query_loss": float(np.mean(losses)), "real_episodes": real_count,
            "synthetic_episodes": len(selected) - real_count, "long_synthetic_episodes": long_synthetic_count,
            "mean_residual_grid_penalty": float(np.mean(grids)),
            "mean_marginal_trust_penalty": float(np.mean(trusts)), "mean_residual_gate": float(np.mean(gates)),
            "mean_context_subset_consistency_penalty": float(np.mean(consistencies)),
            "marginal_policy_trainable": int(step > args.marginal_warmup_steps),
            "pre_clip_gradient_norm": float(norm),
        })
        if step == 1 or step % args.validate_every == 0 or step == args.train_steps:
            hyperspline.eval()
            validation_loss, rows = evaluate_real_bank(backbone, hyperspline, base, validation_episodes)
            for row in rows:
                row.update(step=step, phase="real_validation", loss_delta=baseline_validation - row["loss"])
            validation_rows.extend(rows)
            print(
                f"[step={step}] train_loss={np.mean(losses):.6f}, real_validation={validation_loss:.6f}, "
                f"grad_norm={float(norm):.5g}", flush=True
            )
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                best_state = copy.deepcopy({name: value.detach().cpu() for name, value in hyperspline.state_dict().items()})
                print(f"[selection] real validation improved; selected_step={best_step}", flush=True)

    hyperspline.load_state_dict(best_state)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_hyperspline_checkpoint(
        args.output_checkpoint, hyperspline, config,
        backbone_reference=args.checkpoint_version, backbone_hash=backbone_state_dict_hash(backbone), step=best_step,
    )
    write_csv(args.output_train_csv, train_rows)
    write_csv(args.output_real_validation_csv, validation_rows)
    print(
        f"Finished real-meta training: selected_step={best_step}, best_real_validation_loss={best_loss:.6f}, "
        f"checkpoint={args.output_checkpoint}", flush=True,
    )


if __name__ == "__main__":
    main()
