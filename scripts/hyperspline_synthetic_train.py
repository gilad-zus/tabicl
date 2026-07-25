"""Train the current shared HyperSpline on a stream of native TabICL tasks.

Only HyperSpline is optimized.  Every update receives newly generated native
``PriorDataset`` tasks; validation and test use fixed, disjoint task banks.
The epoch-0 identity state is an eligible checkpoint, so the experiment never
selects a learned transform unless it improves synthetic validation loss.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from direct_spline_multidataset_headroom import load_backbone
from direct_spline_synthetic_headroom import make_prior
from tabicl._hyperspline import (
    HyperSplineTransform,
    backbone_state_dict_hash,
    save_hyperspline_checkpoint,
    summarize_context,
)
from tabicl._hyperspline.checkpoint import load_hyperspline_checkpoint


@dataclass(frozen=True)
class SyntheticEpisode:
    """One fully numeric synthetic classification task with a fixed ICL split."""

    task_id: int
    source_seed: int
    x_context: torch.Tensor  # (1, N_C, D)
    x_query: torch.Tensor  # (1, N_Q, D)
    y_context: torch.Tensor  # (1, N_C), float for TabICL
    y_query: torch.Tensor  # (N_Q,), long for CE
    n_classes: int


def seed_generator(seed: int) -> None:
    """Seed every RNG used by the native prior deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_episode(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    active_features: torch.Tensor,
    sequence_lengths: torch.Tensor,
    context_sizes: torch.Tensor,
    index: int,
    *,
    task_id: int,
    source_seed: int,
    device: torch.device,
) -> SyntheticEpisode:
    n_rows = int(sequence_lengths[index])
    n_context = int(context_sizes[index])
    n_features = int(active_features[index])
    if not 0 < n_context < n_rows:
        raise ValueError(f"synthetic task {task_id} has invalid split {n_context}/{n_rows}")
    x = x_batch[index, :n_rows, :n_features].to(device=device, dtype=torch.float32)
    # The generator labels are made contiguous once over the complete task, not
    # separately for context/query, so query classes retain their true index.
    classes, y = torch.unique(y_batch[index, :n_rows].to(torch.long), sorted=True, return_inverse=True)
    return SyntheticEpisode(
        task_id=task_id,
        source_seed=source_seed,
        x_context=x[:n_context].unsqueeze(0),
        x_query=x[n_context:].unsqueeze(0),
        y_context=y[:n_context].to(device=device, dtype=torch.float32).unsqueeze(0),
        y_query=y[n_context:].to(device=device, dtype=torch.long),
        n_classes=int(classes.numel()),
    )


def generate_episodes(
    args: argparse.Namespace,
    count: int,
    *,
    source_seed: int | None,
    task_offset: int,
    device: torch.device,
) -> list[SyntheticEpisode]:
    """Generate ``count`` tasks; a seed is used only for fixed evaluation banks."""
    if source_seed is not None:
        seed_generator(source_seed)
    prior_args = argparse.Namespace(**vars(args), tasks=count)
    prior = make_prior(prior_args)
    x_batch, y_batch, active_features, sequence_lengths, context_sizes = prior.get_batch()
    episodes = [
        prepare_episode(
            x_batch,
            y_batch,
            active_features,
            sequence_lengths,
            context_sizes,
            index,
            task_id=task_offset + index,
            source_seed=source_seed if source_seed is not None else args.train_seed,
            device=device,
        )
        for index in range(count)
    ]
    return episodes


def save_episode_bank(path: Path, episodes: list[SyntheticEpisode], *, source_seed: int) -> None:
    """Persist a fixed synthetic bank so every ablation consumes identical tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "source_seed": source_seed,
        "episodes": [
            {
                "task_id": episode.task_id,
                "source_seed": episode.source_seed,
                "x_context": episode.x_context.cpu(),
                "x_query": episode.x_query.cpu(),
                "y_context": episode.y_context.cpu(),
                "y_query": episode.y_query.cpu(),
                "n_classes": episode.n_classes,
            }
            for episode in episodes
        ],
    }
    torch.save(payload, path)
    print(f"Saved fixed synthetic bank with {len(episodes)} episodes to {path}", flush=True)


def load_episode_bank(path: Path, *, expected_seed: int, expected_count: int, device: torch.device) -> list[SyntheticEpisode]:
    """Load and validate a fixed synthetic bank created by :func:`save_episode_bank`."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported synthetic bank format: {path}")
    if payload.get("source_seed") != expected_seed:
        raise ValueError(f"synthetic bank {path} has source seed {payload.get('source_seed')}, expected {expected_seed}")
    stored = payload.get("episodes", [])
    if len(stored) != expected_count:
        raise ValueError(f"synthetic bank {path} has {len(stored)} episodes, expected {expected_count}")
    episodes = [
        SyntheticEpisode(
            task_id=int(item["task_id"]),
            source_seed=int(item["source_seed"]),
            x_context=item["x_context"].to(device=device),
            x_query=item["x_query"].to(device=device),
            y_context=item["y_context"].to(device=device),
            y_query=item["y_query"].to(device=device),
            n_classes=int(item["n_classes"]),
        )
        for item in stored
    ]
    print(f"Loaded fixed synthetic bank with {len(episodes)} episodes from {path}", flush=True)
    return episodes


def get_fixed_episode_bank(
    args: argparse.Namespace,
    path: Path | None,
    count: int,
    *,
    source_seed: int,
    task_offset: int,
    device: torch.device,
) -> list[SyntheticEpisode]:
    if path is not None and path.is_file():
        return load_episode_bank(path, expected_seed=source_seed, expected_count=count, device=device)
    episodes = generate_episodes(args, count, source_seed=source_seed, task_offset=task_offset, device=device)
    if path is not None:
        save_episode_bank(path, episodes, source_seed=source_seed)
    return episodes


def validate_episode_classes(episodes: Iterable[SyntheticEpisode], max_classes: int) -> None:
    too_many = [episode.task_id for episode in episodes if episode.n_classes > max_classes]
    if too_many:
        raise ValueError(f"synthetic tasks exceed backbone max_classes={max_classes}: {too_many[:5]}")


def forward_episode(
    backbone,
    hyperspline: HyperSplineTransform,
    episode: SyntheticEpisode,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    # Parameter generation is context-only. Statistics have no trainable
    # weights, so their graph is intentionally omitted while the generated
    # parameter path remains differentiable with respect to HyperSpline.
    with torch.no_grad():
        statistics = summarize_context(
            episode.x_context,
            y_context=episode.y_context if hyperspline.target_aware else None,
            eps=hyperspline.eps,
        )
    parameters = hyperspline.generate_parameters(statistics)
    transformed_context = hyperspline.apply_transform(episode.x_context, parameters)
    transformed_query = hyperspline.apply_transform(episode.x_query, parameters)
    transformed = torch.cat((transformed_context, transformed_query), dim=1)
    logits = backbone(transformed, episode.y_context)
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    # Measure the spline's actual residual relative to its standardized base
    # path, not relative to raw values (which would include normalisation).
    raw = torch.cat((episode.x_context, episode.x_query), dim=1).float()
    z = (raw - parameters.location.unsqueeze(1)) / parameters.scale.unsqueeze(1)
    deformation = (transformed.float() - z).abs()
    clip_fraction = ((z / hyperspline.standardized_range).abs() >= 1).float().mean()
    diagnostics = {
        "mean_gate": parameters.gate.mean(),
        "min_gate": parameters.gate.min(),
        "max_gate": parameters.gate.max(),
        "mean_abs_deformation": deformation.mean(),
        "max_abs_deformation": deformation.max(),
        "clip_fraction": clip_fraction,
        "mean_supervised_residual_gate": (
            parameters.supervised_residual_gate.mean()
            if parameters.supervised_residual_gate is not None
            else parameters.gate.new_zeros(())
        ),
    }
    return loss, logits, diagnostics


def evaluate_episode(backbone, hyperspline: HyperSplineTransform, episode: SyntheticEpisode) -> dict[str, float | int]:
    with torch.no_grad():
        loss, logits, diagnostics = forward_episode(backbone, hyperspline, episode)
        accuracy = (logits.argmax(dim=-1).flatten() == episode.y_query).float().mean()
    return {
        "task": episode.task_id,
        "source_seed": episode.source_seed,
        "n_context": int(episode.x_context.shape[1]),
        "n_query": int(episode.x_query.shape[1]),
        "n_features": int(episode.x_context.shape[2]),
        "n_classes": episode.n_classes,
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        **{name: value.item() for name, value in diagnostics.items()},
    }


def append_bank_evaluation(
    records: list[dict[str, float | int | str]],
    backbone,
    hyperspline: HyperSplineTransform,
    episodes: list[SyntheticEpisode],
    identity: dict[int, dict[str, float | int]],
    *,
    phase: str,
    step: int,
) -> float:
    current_rows = []
    for episode in episodes:
        backbone.clear_cache()
        metrics = evaluate_episode(backbone, hyperspline, episode)
        baseline = identity[episode.task_id]
        metrics.update(
            {
                "phase": phase,
                "step": step,
                "identity_loss": baseline["loss"],
                "identity_accuracy": baseline["accuracy"],
                "loss_delta": float(baseline["loss"]) - float(metrics["loss"]),
                "relative_loss_improvement": (float(baseline["loss"]) - float(metrics["loss"]))
                / max(float(baseline["loss"]), 1e-12),
                "accuracy_delta": float(metrics["accuracy"]) - float(baseline["accuracy"]),
            }
        )
        records.append(metrics)
        current_rows.append(metrics)
    mean_loss = float(np.mean([float(row["loss"]) for row in current_rows]))
    improved = float(np.mean([float(row["loss_delta"]) > 0.0 for row in current_rows]))
    print(
        f"[{phase} step={step}] mean_loss={mean_loss:.6f}, "
        f"mean_loss_delta={np.mean([float(row['loss_delta']) for row in current_rows]):+.6f}, "
        f"loss_improved_fraction={improved:.2%}",
        flush=True,
    )
    return mean_loss


def collect_identity_metrics(backbone, hyperspline: HyperSplineTransform, episodes: list[SyntheticEpisode]) -> dict[int, dict[str, float | int]]:
    """Evaluate the fixed identity state with the same cache discipline as later evaluations."""
    metrics = {}
    for episode in episodes:
        backbone.clear_cache()
        metrics[episode.task_id] = evaluate_episode(backbone, hyperspline, episode)
    return metrics


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=64)
    parser.add_argument("--test-tasks", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--gate-initial-probability", type=float, default=0.10)
    parser.add_argument(
        "--target-aware",
        action="store_true",
        help="Append class-permutation-invariant context-label statistics to each column summary.",
    )
    parser.add_argument(
        "--supervised-residual",
        action="store_true",
        help="Add a separately gated label-only residual branch on a frozen marginal HyperSpline.",
    )
    parser.add_argument(
        "--cross-column-residual",
        action="store_true",
        help="Use permutation-equivariant cross-column conditioning with bounded per-column residual gates.",
    )
    parser.add_argument(
        "--marginal-checkpoint",
        type=Path,
        default=None,
        help="Required trained marginal checkpoint used as the frozen base for --supervised-residual.",
    )
    parser.add_argument("--supervised-residual-gate-initial-probability", type=float, default=0.01)
    parser.add_argument("--cross-column-num-heads", type=int, default=4)
    parser.add_argument("--cross-column-residual-bound", type=float, default=0.1)
    parser.add_argument("--cross-column-gate-initial-probability", type=float, default=0.01)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=0,
        help="Seed used only for HyperSpline initialization; set this identically for ablations.",
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=10_001)
    parser.add_argument("--test-seed", type=int, default=20_001)
    parser.add_argument("--validation-bank", type=Path, default=None, help="Reusable serialized fixed validation bank.")
    parser.add_argument("--test-bank", type=Path, default=None, help="Reusable serialized fixed test bank.")
    parser.add_argument("--output-csv", type=Path, default=Path("results/hyperspline_synthetic_evaluation.csv"))
    parser.add_argument("--output-train-csv", type=Path, default=Path("results/hyperspline_synthetic_training.csv"))
    parser.add_argument("--output-checkpoint", type=Path, default=Path("results/hyperspline_synthetic_best.pt"))
    args = parser.parse_args()

    if args.train_steps <= 0 or args.tasks_per_step <= 0 or args.validation_tasks <= 0 or args.test_tasks <= 0:
        raise ValueError("task counts and --train-steps must be positive")
    if args.sequence_length < 4 or not 0 < args.context_fraction < 1:
        raise ValueError("invalid sequence length or context fraction")
    if not 0 < args.min_features <= args.max_features or args.max_classes < 2:
        raise ValueError("invalid feature/class range")
    if args.validate_every <= 0 or args.n_control_points <= 3 or not 0 < args.gate_initial_probability < 1:
        raise ValueError("invalid validation or HyperSpline configuration")
    if len({args.train_seed, args.validation_seed, args.test_seed}) != 3:
        raise ValueError("train, validation, and test generator seeds must be distinct")
    if args.supervised_residual and args.cross_column_residual:
        raise ValueError("--supervised-residual and --cross-column-residual are mutually exclusive")
    if (args.supervised_residual or args.cross_column_residual) and (
        not args.target_aware or args.marginal_checkpoint is None
    ):
        raise ValueError("supervised residuals require --target-aware and --marginal-checkpoint")
    if args.marginal_checkpoint is not None and not (args.supervised_residual or args.cross_column_residual):
        raise ValueError("--marginal-checkpoint is only valid with a supervised residual")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.train_seed)
        print(f"Running on CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    backbone, _ = load_backbone(args, device)
    if args.max_classes > backbone.max_classes:
        raise ValueError(f"--max-classes exceeds frozen backbone maximum {backbone.max_classes}")

    # Keep initialization controlled separately from the synthetic-task stream:
    # paired marginal/label-aware ablations should differ only in label access.
    torch.manual_seed(args.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.model_seed)
    config = {
        "n_control_points": args.n_control_points,
        "hidden_dim": args.hidden_dim,
        "gate_initial_probability": args.gate_initial_probability,
        "target_aware": args.target_aware,
        "supervised_residual": args.supervised_residual,
        "supervised_residual_gate_initial_probability": args.supervised_residual_gate_initial_probability,
        "cross_column_residual": args.cross_column_residual,
        "cross_column_num_heads": args.cross_column_num_heads,
        "cross_column_residual_bound": args.cross_column_residual_bound,
        "cross_column_gate_initial_probability": args.cross_column_gate_initial_probability,
    }
    hyperspline = HyperSplineTransform(**config).to(device).train()
    if args.supervised_residual or args.cross_column_residual:
        marginal, _ = load_hyperspline_checkpoint(
            args.marginal_checkpoint,
            device=device,
            expected_backbone_reference=args.checkpoint_version,
            expected_backbone_hash=backbone_state_dict_hash(backbone),
        )
        hyperspline.initialize_supervised_residual_from(marginal)
    optimizer = torch.optim.Adam(hyperspline.parameters(), lr=args.lr)
    print(
        f"Initialized current HyperSpline baseline: "
        f"trainable_parameters={sum(parameter.numel() for parameter in hyperspline.parameters() if parameter.requires_grad):,}, "
        f"target_aware={args.target_aware}, train_steps={args.train_steps}, tasks_per_step={args.tasks_per_step}",
        flush=True,
    )

    print("Loading or generating fixed validation and test banks from disjoint prior seeds...", flush=True)
    validation_episodes = get_fixed_episode_bank(
        args, args.validation_bank, args.validation_tasks, source_seed=args.validation_seed, task_offset=0, device=device
    )
    test_episodes = get_fixed_episode_bank(
        args,
        args.test_bank,
        args.test_tasks,
        source_seed=args.test_seed,
        task_offset=args.validation_tasks,
        device=device,
    )
    validate_episode_classes(validation_episodes + test_episodes, backbone.max_classes)
    seed_generator(args.train_seed)

    evaluation_records: list[dict[str, float | int | str]] = []
    train_records: list[dict[str, float | int]] = []
    hyperspline.eval()
    identity_validation = collect_identity_metrics(backbone, hyperspline, validation_episodes)
    identity_test = collect_identity_metrics(backbone, hyperspline, test_episodes)
    # Identity is the initial best checkpoint; later states must beat it on the
    # fixed validation task bank to be selected.
    append_bank_evaluation(
        evaluation_records, backbone, hyperspline, validation_episodes, identity_validation, phase="identity_validation", step=0
    )
    append_bank_evaluation(
        evaluation_records, backbone, hyperspline, test_episodes, identity_test, phase="identity_test", step=0
    )
    best_validation_loss = float(np.mean([float(row["loss"]) for row in identity_validation.values()]))
    best_step = 0
    best_state = {name: value.detach().cpu().clone() for name, value in hyperspline.state_dict().items()}

    for step in range(1, args.train_steps + 1):
        hyperspline.train()
        episodes = generate_episodes(args, args.tasks_per_step, source_seed=None, task_offset=step * args.tasks_per_step, device=device)
        validate_episode_classes(episodes, backbone.max_classes)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for episode in episodes:
            backbone.clear_cache()
            loss, _, _ = forward_episode(backbone, hyperspline, episode)
            (loss / len(episodes)).backward()
            losses.append(loss.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), max_norm=1.0)
        optimizer.step()
        train_records.append(
            {
                "step": step,
                "mean_query_loss": float(np.mean(losses)),
                "pre_clip_gradient_norm": gradient_norm.item(),
                "tasks_per_step": len(episodes),
            }
        )
        if step == 1 or step % args.validate_every == 0 or step == args.train_steps:
            print(
                f"[train step={step}] mean_fresh_task_loss={np.mean(losses):.6f}, "
                f"pre_clip_gradient_norm={gradient_norm.item():.6g}",
                flush=True,
            )
            hyperspline.eval()
            validation_loss = append_bank_evaluation(
                evaluation_records, backbone, hyperspline, validation_episodes, identity_validation, phase="validation", step=step
            )
            append_bank_evaluation(
                evaluation_records, backbone, hyperspline, test_episodes, identity_test, phase="test", step=step
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in hyperspline.state_dict().items()}
                print(f"[selection] validation improved; selected_step={best_step}", flush=True)

    hyperspline.load_state_dict(best_state)
    hyperspline.eval()
    append_bank_evaluation(
        evaluation_records, backbone, hyperspline, validation_episodes, identity_validation, phase="selected_validation", step=best_step
    )
    append_bank_evaluation(
        evaluation_records, backbone, hyperspline, test_episodes, identity_test, phase="selected_test", step=best_step
    )
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_hyperspline_checkpoint(
        args.output_checkpoint,
        hyperspline,
        config,
        backbone_reference=args.checkpoint_version,
        backbone_hash=backbone_state_dict_hash(backbone),
        step=best_step,
    )
    write_csv(args.output_csv, evaluation_records)
    write_csv(args.output_train_csv, train_records)
    print(
        f"Finished synthetic HyperSpline training: selected_step={best_step}, "
        f"best_validation_loss={best_validation_loss:.6f}, checkpoint={args.output_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
