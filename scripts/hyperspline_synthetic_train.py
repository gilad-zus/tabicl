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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:  # Supports imports from tests as well as ``python scripts/file.py``.
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.direct_spline_synthetic_headroom import make_prior
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
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
    # ``native`` is exactly the historical TabICL prior output.  The expanded
    # mode is an opt-in, deterministic *observation model* applied to feature
    # values before the ICL split; it never accesses labels.
    observation_mode: str = "native"


SYNTHETIC_OBSERVATION_MODES = ("native", "coverage_expanded")


def apply_synthetic_observation(
    x: torch.Tensor,
    *,
    observation_mode: str,
    seed: int,
) -> torch.Tensor:
    """Apply a deterministic, label-free observation model to synthetic columns.

    The native TabICL prior standardises and clips numerical columns before it
    returns them.  ``coverage_expanded`` deliberately keeps the *underlying
    task and labels* fixed but restores a wider family of realistic observed
    column shapes: skewed and heavy/light-tailed monotone responses, bounded
    sensors, censoring, rounded measurements, and near-zero resolution floors.
    Every operation is non-decreasing per column, so it does not fabricate a
    feature--target relationship by reordering values.  It is applied before
    the context/query split, avoiding a synthetic covariate shift by design.

    Missingness is intentionally not injected here.  TabICL's current numeric
    episode path has no missing-value mask; adding NaNs would only test an
    accidental imputation path.  Missingness needs a separate mask-aware
    episode protocol rather than arbitrary corruption.
    """
    if observation_mode not in SYNTHETIC_OBSERVATION_MODES:
        raise ValueError(f"unknown synthetic observation mode {observation_mode!r}")
    if observation_mode == "native":
        return x
    if x.ndim != 2:
        raise ValueError("synthetic observation model expects one task shaped (N, D)")

    # The prior itself generates on CPU.  Keeping this small, deterministic
    # augmentation on CPU makes fixed banks bitwise reproducible regardless of
    # the GPU used for later HyperSpline training.
    original_device, original_dtype = x.device, x.dtype
    values = x.detach().to(device="cpu", dtype=torch.float32).clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    n_features = values.shape[1]
    primary = torch.randint(0, 7, (n_features,), generator=generator)

    def uniform(low: float, high: float) -> float:
        return float(torch.empty((), dtype=values.dtype).uniform_(low, high, generator=generator))

    for column in range(n_features):
        z = values[:, column].clamp(-12.0, 12.0)
        family = int(primary[column])
        if family == 0:
            # Concave/convex signed powers vary concentration and tail weight.
            exponent = uniform(0.35, 0.80) if bool(torch.randint(0, 2, (), generator=generator)) else uniform(1.35, 2.40)
            z = z.sign() * z.abs().pow(exponent)
        elif family == 1:
            # Smooth asymmetric exponential sensor response (and its mirror).
            strength, direction = uniform(0.18, 0.70), -1.0 if bool(torch.randint(0, 2, (), generator=generator)) else 1.0
            z = direction * torch.expm1(direction * strength * z) / strength
        elif family == 2:
            # Bounded/saturating measurement scale.
            bound, strength = uniform(0.8, 2.5), uniform(0.35, 1.20)
            z = bound * torch.tanh(strength * z)
        elif family == 3:
            # Symmetric log compression creates lighter tails without a bound.
            strength = uniform(0.25, 1.25)
            z = z.sign() * torch.log1p(strength * z.abs()) / strength
        elif family == 4:
            # Asymmetric tail response: a simple monotone piecewise sensor.
            positive, negative = uniform(1.0, 3.5), uniform(0.35, 1.0)
            z = torch.where(z >= 0, positive * z, negative * z)
        elif family == 5:
            # Censored instrument with a random lower/upper reporting range.
            low, high = uniform(-2.5, -0.25), uniform(0.25, 2.5)
            z = z.clamp(low, high)
        else:
            # Outlier-prone response: only far tails are stretched.
            threshold, stretch = uniform(0.6, 1.8), uniform(1.5, 4.0)
            z = z.sign() * torch.where(z.abs() <= threshold, z.abs(), threshold + stretch * (z.abs() - threshold))

        # Measurement resolution and a near-zero dead zone are each optional.
        # Both remain non-decreasing, so any lost information is realistic
        # coarsening rather than a label-dependent perturbation.
        if bool(torch.randint(0, 2, (), generator=generator)):
            step = uniform(0.04, 0.35) * z.detach().std().clamp_min(0.25).item()
            z = torch.round(z / step) * step
        if bool(torch.randint(0, 2, (), generator=generator)):
            dead_zone = uniform(0.02, 0.35) * z.detach().std().clamp_min(0.25).item()
            z = torch.where(z.abs() < dead_zone, torch.zeros_like(z), z)
        values[:, column] = torch.nan_to_num(z, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    return values.to(device=original_device, dtype=original_dtype)


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
    observation_mode: str = "native",
) -> SyntheticEpisode:
    n_rows = int(sequence_lengths[index])
    n_context = int(context_sizes[index])
    n_features = int(active_features[index])
    if not 0 < n_context < n_rows:
        raise ValueError(f"synthetic task {task_id} has invalid split {n_context}/{n_rows}")
    native_x = x_batch[index, :n_rows, :n_features]
    x = apply_synthetic_observation(
        native_x,
        observation_mode=observation_mode,
        # Distinct, reproducible column draws for every prior task, including
        # streaming training tasks whose task_id is monotonically increasing.
        seed=int(source_seed) + 1_000_003 * int(task_id),
    ).to(device=device, dtype=torch.float32)
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
        observation_mode=observation_mode,
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
    observation_mode = getattr(args, "synthetic_observation_mode", "native")
    if observation_mode not in SYNTHETIC_OBSERVATION_MODES:
        raise ValueError(f"unknown synthetic observation mode {observation_mode!r}")
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
            observation_mode=observation_mode,
        )
        for index in range(count)
    ]
    return episodes


def generate_scheduled_episodes(
    args: argparse.Namespace,
    count: int,
    *,
    source_seed: int,
    task_offset: int,
    device: torch.device,
    sequence_lengths: Sequence[int],
    context_fractions: Sequence[float],
    observation_mode: str,
) -> list[SyntheticEpisode]:
    """Generate a balanced fixed bank across row counts and context fractions.

    This helper is intentionally for diagnostics/bank construction, where
    variable task shapes are welcome.  Existing streaming training remains
    unchanged and therefore retains uniform shapes and batching behavior.
    Calling this with ``native`` and ``coverage_expanded`` uses identical
    schedules, prior seeds, labels, sizes, and feature counts; only observed
    numerical values differ.
    """
    if count <= 0:
        return []
    if observation_mode not in SYNTHETIC_OBSERVATION_MODES:
        raise ValueError(f"unknown synthetic observation mode {observation_mode!r}")
    lengths = tuple(sorted({int(value) for value in sequence_lengths}))
    fractions = tuple(sorted({float(value) for value in context_fractions}))
    if not lengths or min(lengths) < 4 or not fractions or not all(0.0 < value < 1.0 for value in fractions):
        raise ValueError("scheduled synthetic lengths must be >=4 and context fractions must lie in (0, 1)")
    choices = [(length, fraction) for length in lengths for fraction in fractions]
    # Cycle a shuffled copy of the full Cartesian schedule.  This ensures every
    # requested size/split receives almost exactly equal coverage.
    selector = np.random.default_rng(int(source_seed) + 73_991)
    order = np.asarray([choices[index % len(choices)] for index in range(count)], dtype=object)
    selector.shuffle(order)
    grouped: dict[tuple[int, float], list[int]] = {}
    for index, (length, fraction) in enumerate(order.tolist()):
        grouped.setdefault((int(length), float(fraction)), []).append(index)
    result: list[SyntheticEpisode | None] = [None] * count
    for group_index, ((length, fraction), indices) in enumerate(sorted(grouped.items())):
        group_args = argparse.Namespace(**vars(args))
        group_args.sequence_length = length
        group_args.context_fraction = fraction
        group_args.synthetic_observation_mode = observation_mode
        generated = generate_episodes(
            group_args,
            len(indices),
            source_seed=int(source_seed) + 10_000_019 * (group_index + 1),
            task_offset=0,
            device=device,
        )
        for local_index, output_index in enumerate(indices):
            result[output_index] = replace(generated[local_index], task_id=task_offset + output_index)
    if any(episode is None for episode in result):  # Defensive: a schedule bug should never silently drop a task.
        raise AssertionError("scheduled synthetic generation did not fill every task")
    return [episode for episode in result if episode is not None]


def save_episode_bank(path: Path, episodes: list[SyntheticEpisode], *, source_seed: int) -> None:
    """Persist a fixed synthetic bank so every ablation consumes identical tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "source_seed": source_seed,
        "observation_mode": episodes[0].observation_mode if episodes else "native",
        "episodes": [
            {
                "task_id": episode.task_id,
                "source_seed": episode.source_seed,
                "x_context": episode.x_context.cpu(),
                "x_query": episode.x_query.cpu(),
                "y_context": episode.y_context.cpu(),
                "y_query": episode.y_query.cpu(),
                "n_classes": episode.n_classes,
                "observation_mode": episode.observation_mode,
            }
            for episode in episodes
        ],
    }
    torch.save(payload, path)
    print(f"Saved fixed synthetic bank with {len(episodes)} episodes to {path}", flush=True)


def load_episode_bank(
    path: Path,
    *,
    expected_seed: int,
    expected_count: int,
    device: torch.device,
    expected_observation_mode: str = "native",
) -> list[SyntheticEpisode]:
    """Load and validate a fixed synthetic bank created by :func:`save_episode_bank`."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError(f"unsupported synthetic bank format: {path}")
    if payload.get("source_seed") != expected_seed:
        raise ValueError(f"synthetic bank {path} has source seed {payload.get('source_seed')}, expected {expected_seed}")
    stored = payload.get("episodes", [])
    stored_mode = payload.get("observation_mode", "native")
    if stored_mode != expected_observation_mode:
        raise ValueError(
            f"synthetic bank {path} has observation mode {stored_mode!r}, expected {expected_observation_mode!r}"
        )
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
            observation_mode=str(item.get("observation_mode", stored_mode)),
        )
        for item in stored
    ]
    print(
        f"Loaded fixed synthetic bank with {len(episodes)} episodes ({stored_mode}) from {path}", flush=True
    )
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
        return load_episode_bank(
            path,
            expected_seed=source_seed,
            expected_count=count,
            device=device,
            expected_observation_mode=getattr(args, "synthetic_observation_mode", "native"),
        )
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
    parameters = hyperspline.generate_parameters(
        statistics,
        x_context=episode.x_context,
        y_context=episode.y_context if hyperspline.target_aware else None,
    )
    marginal_parameters = (
        hyperspline.generate_marginal_parameters(statistics)
        if hyperspline.has_supervised_residual
        else None
    )
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
        "grid_deformation_penalty": hyperspline.grid_deformation_penalty(
            parameters, reference_parameters=marginal_parameters
        ),
        "supervised_gate_penalty": (
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
    parser.add_argument(
        "--synthetic-observation-mode",
        choices=SYNTHETIC_OBSERVATION_MODES,
        default="native",
        help=(
            "Observed numerical-column family after native prior generation. "
            "native preserves historical experiments exactly; coverage_expanded is opt-in."
        ),
    )
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
        "--raw-context-residual",
        action="store_true",
        help="Use class-invariant raw labelled-context encoding with row-level and cross-column attention.",
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
    parser.add_argument("--raw-context-num-heads", type=int, default=4)
    parser.add_argument("--raw-context-residual-bound", type=float, default=0.5)
    parser.add_argument("--raw-context-gate-initial-probability", type=float, default=0.5)
    parser.add_argument(
        "--transform-regularization",
        type=float,
        default=0.0,
        help="Weight on mean squared spline deformation over a fixed standardized grid.",
    )
    parser.add_argument(
        "--supervised-gate-regularization",
        type=float,
        default=0.0,
        help="Weight on mean supervised residual gate activation; leave zero unless needed for stability.",
    )
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
    residual_flags = (args.supervised_residual, args.cross_column_residual, args.raw_context_residual)
    if sum(residual_flags) > 1:
        raise ValueError("supervised residual variants are mutually exclusive")
    if any(residual_flags) and (
        not args.target_aware or args.marginal_checkpoint is None
    ):
        raise ValueError("supervised residuals require --target-aware and --marginal-checkpoint")
    if args.marginal_checkpoint is not None and not any(residual_flags):
        raise ValueError("--marginal-checkpoint is only valid with a supervised residual")
    if args.transform_regularization < 0 or args.supervised_gate_regularization < 0:
        raise ValueError("regularization weights must be non-negative")

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
        "raw_context_residual": args.raw_context_residual,
        "raw_context_num_heads": args.raw_context_num_heads,
        "raw_context_residual_bound": args.raw_context_residual_bound,
        "raw_context_gate_initial_probability": args.raw_context_gate_initial_probability,
    }
    hyperspline = HyperSplineTransform(**config).to(device).train()
    if any(residual_flags):
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
        losses, grid_penalties, gate_penalties = [], [], []
        for episode in episodes:
            backbone.clear_cache()
            loss, _, diagnostics = forward_episode(backbone, hyperspline, episode)
            objective = (
                loss
                + args.transform_regularization * diagnostics["grid_deformation_penalty"]
                + args.supervised_gate_regularization * diagnostics["supervised_gate_penalty"]
            )
            (objective / len(episodes)).backward()
            losses.append(loss.detach().item())
            grid_penalties.append(diagnostics["grid_deformation_penalty"].detach().item())
            gate_penalties.append(diagnostics["supervised_gate_penalty"].detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), max_norm=1.0)
        optimizer.step()
        train_records.append(
            {
                "step": step,
                "mean_query_loss": float(np.mean(losses)),
                "mean_grid_deformation_penalty": float(np.mean(grid_penalties)),
                "mean_supervised_gate_penalty": float(np.mean(gate_penalties)),
                "transform_regularization": args.transform_regularization,
                "supervised_gate_regularization": args.supervised_gate_regularization,
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
