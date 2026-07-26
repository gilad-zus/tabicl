"""Paired DirectSpline headroom experiment on HyperSpline's fixed banks.

For every synthetic or real episode this script evaluates three conditions on
the *same* tensors:

``identity``
    No numerical spline.
``context_only``
    DirectSpline is fitted using an inner stratified split of the labelled
    context.  Its best state is selected by the held-out context rows, then
    evaluated once on the untouched outer query rows.  No query label enters
    fitting or selection.
``oracle_query_labels``
    DirectSpline is fitted directly to outer query labels.  This is deliberately
    invalid for zero-shot use, and is only an upper bound on the current scalar
    spline family.

The synthetic input banks are the serialized banks created by
``hyperspline_label_aware_ablation.py``.  Real episodes are built exactly once
per dataset/seed and then shared by all three conditions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:  # Support both ``python scripts/file.py`` and package-style imports.
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_zero_shot_eval import (
        RealEpisode,
        build_episode,
        parse_csv,
        resolve_specs,
    )
    from scripts.hyperspline_synthetic_train import SyntheticEpisode, load_episode_bank
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_zero_shot_eval import RealEpisode, build_episode, parse_csv, resolve_specs
    from hyperspline_synthetic_train import SyntheticEpisode, load_episode_bank
from tabicl._hyperspline import DirectSplineTransform


@dataclass(frozen=True)
class Episode:
    name: str
    phase: str
    seed: int
    x_context: torch.Tensor
    x_query: torch.Tensor
    y_context: torch.Tensor
    y_query: torch.Tensor
    numerical_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic-validation-bank", type=Path, required=True)
    parser.add_argument("--synthetic-test-bank", type=Path, required=True)
    parser.add_argument("--synthetic-validation-tasks", type=int, default=64)
    parser.add_argument("--synthetic-test-tasks", type=int, default=64)
    parser.add_argument("--synthetic-validation-seed", type=int, default=10_001)
    parser.add_argument("--synthetic-test-seed", type=int, default=20_001)
    parser.add_argument("--real-dataset-suite", choices=("all", "numerical_only", "mixed"), default="all")
    parser.add_argument("--real-datasets", default=None)
    parser.add_argument("--real-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--real-test-size", type=float, default=0.30)
    parser.add_argument("--real-max-rows", type=int, default=1024)
    parser.add_argument("--context-validation-fraction", type=float, default=0.25)
    parser.add_argument("--context-steps", type=int, default=250)
    parser.add_argument("--oracle-steps", type=int, default=500)
    parser.add_argument("--select-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Controls inner context split and optimizer initialization.")
    parser.add_argument(
        "--conditions",
        default="identity,context_only,oracle_query_labels",
        help="Comma-separated subset of identity,context_only,oracle_query_labels. Identity is always written as the paired baseline.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    return parser.parse_args()


def transform_all(spline: DirectSplineTransform, episode: Episode) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform numerical columns only; categorical columns stay byte-for-byte unchanged."""
    x_context, x_query = episode.x_context.clone(), episode.x_query.clone()
    x_context[..., episode.numerical_mask] = spline.transform(x_context[..., episode.numerical_mask])
    x_query[..., episode.numerical_mask] = spline.transform(x_query[..., episode.numerical_mask])
    return x_context, x_query


def loss_on_split(
    backbone,
    spline: DirectSplineTransform,
    episode: Episode,
    context_rows: torch.Tensor,
    query_rows: torch.Tensor,
    *,
    query_from_context: bool,
) -> torch.Tensor:
    x_context, x_query = transform_all(spline, episode)
    if query_from_context:
        target_x = x_context[:, query_rows]
        target_y = episode.y_context[:, query_rows].long()
    else:
        target_x = x_query[:, query_rows]
        target_y = episode.y_query[query_rows]
    backbone.clear_cache()
    logits = backbone(
        torch.cat((x_context[:, context_rows], target_x), dim=1),
        episode.y_context[:, context_rows],
    )
    return F.cross_entropy(logits.flatten(0, 1), target_y.flatten())


@torch.no_grad()
def evaluate_outer(backbone, spline: DirectSplineTransform | None, episode: Episode) -> tuple[float, float, float]:
    if spline is None:
        x_context, x_query = episode.x_context, episode.x_query
        gate_mean = 0.0
    else:
        x_context, x_query = transform_all(spline, episode)
        gate_mean = float(spline.parameters_for_transform().gate.mean())
    backbone.clear_cache()
    logits = backbone(torch.cat((x_context, x_query), dim=1), episode.y_context)
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    accuracy = (logits.argmax(dim=-1).flatten() == episode.y_query.flatten()).float().mean()
    return float(loss), float(accuracy), gate_mean


def stratified_inner_split(y: torch.Tensor, fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at least one labelled support row for every class where possible."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    y_cpu = y.detach().reshape(-1).long().cpu()
    fit, validation = [], []
    for label in torch.unique(y_cpu, sorted=True):
        rows = torch.where(y_cpu == label)[0]
        rows = rows[torch.randperm(rows.numel(), generator=generator)]
        if rows.numel() == 1:
            fit.append(rows)
            continue
        n_validation = min(max(1, round(rows.numel() * fraction)), rows.numel() - 1)
        validation.append(rows[:n_validation])
        fit.append(rows[n_validation:])
    fit_rows = torch.cat(fit)
    validation_rows = torch.cat(validation) if validation else fit_rows[:0]
    if validation_rows.numel() == 0:
        raise ValueError("context-only adaptation needs at least one class with two context rows")
    return fit_rows.to(y.device), validation_rows.to(y.device)


def fit_spline(
    backbone,
    episode: Episode,
    *,
    training_context_rows: torch.Tensor,
    training_query_rows: torch.Tensor,
    selection_context_rows: torch.Tensor | None,
    selection_query_rows: torch.Tensor | None,
    training_query_from_context: bool,
    selection_query_from_context: bool,
    steps: int,
    lr: float,
    select_every: int,
    n_control_points: int,
) -> tuple[DirectSplineTransform, int]:
    spline = DirectSplineTransform(episode.x_context[..., episode.numerical_mask], n_control_points).to(episode.x_context.device)
    optimizer = torch.optim.Adam(spline.parameters(), lr=lr)
    best_step, best_loss = 0, float("inf")
    best_state = {name: value.detach().cpu().clone() for name, value in spline.state_dict().items()}

    def select_loss() -> float:
        if selection_context_rows is None:
            return float(loss_on_split(
                backbone, spline, episode, training_context_rows, training_query_rows,
                query_from_context=training_query_from_context,
            ).detach())
        return float(loss_on_split(
            backbone, spline, episode, selection_context_rows, selection_query_rows,
            query_from_context=selection_query_from_context,
        ).detach())

    # Identity is eligible for both the context-selected and oracle conditions.
    best_loss = select_loss()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on_split(
            backbone, spline, episode, training_context_rows, training_query_rows,
            query_from_context=training_query_from_context,
        )
        loss.backward()
        optimizer.step()
        if step % select_every == 0 or step == steps:
            candidate_loss = select_loss()
            if candidate_loss < best_loss:
                best_step, best_loss = step, candidate_loss
                best_state = {name: value.detach().cpu().clone() for name, value in spline.state_dict().items()}
    spline.load_state_dict(best_state)
    return spline, best_step


def make_synthetic_episode(source: SyntheticEpisode, phase: str) -> Episode:
    return Episode(
        name=f"synthetic_task_{source.task_id}", phase=phase, seed=source.source_seed,
        x_context=source.x_context, x_query=source.x_query, y_context=source.y_context,
        y_query=source.y_query, numerical_mask=torch.ones(source.x_context.shape[-1], dtype=torch.bool, device=source.x_context.device),
    )


def make_real_episode(source: RealEpisode) -> Episode:
    return Episode(
        name=source.dataset, phase="real", seed=source.split_seed,
        x_context=source.x_context, x_query=source.x_query, y_context=source.y_context,
        y_query=source.y_query, numerical_mask=source.numerical_mask,
    )


def run_episode(
    backbone, episode: Episode, args: argparse.Namespace, ordinal: int, requested_conditions: set[str]
) -> list[dict[str, object]]:
    baseline_loss, baseline_accuracy, _ = evaluate_outer(backbone, None, episode)
    all_context = torch.arange(episode.x_context.shape[1], device=episode.x_context.device)
    all_query = torch.arange(episode.x_query.shape[1], device=episode.x_context.device)
    fit_rows, validation_rows = stratified_inner_split(
        episode.y_context, args.context_validation_fraction, args.seed + 10_000 * ordinal
    )
    conditions: list[tuple[str, DirectSplineTransform | None, int]] = [("identity", None, 0)]
    if "context_only" in requested_conditions:
        context_spline, context_step = fit_spline(
            backbone, episode, training_context_rows=fit_rows, training_query_rows=validation_rows,
            selection_context_rows=fit_rows, selection_query_rows=validation_rows,
            training_query_from_context=True, selection_query_from_context=True,
            steps=args.context_steps, lr=args.lr, select_every=args.select_every,
            n_control_points=args.n_control_points,
        )
        conditions.append(("context_only", context_spline, context_step))
    if "oracle_query_labels" in requested_conditions:
        # This is intentionally an invalid upper bound: the *actual outer
        # query features and labels* are both used for fitting and selection.
        oracle_spline, oracle_step = fit_spline(
            backbone, episode, training_context_rows=all_context, training_query_rows=all_query,
            selection_context_rows=None, selection_query_rows=None,
            training_query_from_context=False, selection_query_from_context=False,
            steps=args.oracle_steps, lr=args.lr, select_every=args.select_every,
            n_control_points=args.n_control_points,
        )
        conditions.append(("oracle_query_labels", oracle_spline, oracle_step))
    rows = []
    for condition, spline, selected_step in conditions:
        loss, accuracy, gate = evaluate_outer(backbone, spline, episode)
        rows.append({
            "phase": episode.phase, "dataset": episode.name, "seed": episode.seed,
            "condition": condition, "n_context": episode.x_context.shape[1], "n_query": episode.x_query.shape[1],
            "n_features": episode.x_context.shape[-1], "n_numerical_features": int(episode.numerical_mask.sum()),
            "baseline_loss": baseline_loss, "loss": loss, "loss_delta": baseline_loss - loss,
            "baseline_accuracy": baseline_accuracy, "accuracy": accuracy, "accuracy_delta": accuracy - baseline_accuracy,
            "selected_step": selected_step, "mean_spline_gate": gate,
            "inner_fit_rows": int(fit_rows.numel()), "inner_validation_rows": int(validation_rows.numel()),
        })
    deltas = "; ".join(f"{row['condition']} loss_delta={row['loss_delta']:+.6f}" for row in rows[1:])
    print(f"[{episode.phase} {episode.name} seed={episode.seed}] {deltas or 'identity only'}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for phase in sorted({str(row["phase"]) for row in rows}):
        for condition in ("identity", "context_only", "oracle_query_labels"):
            subset = [row for row in rows if row["phase"] == phase and row["condition"] == condition]
            if not subset:
                continue
            loss_deltas = np.asarray([float(row["loss_delta"]) for row in subset])
            accuracy_deltas = np.asarray([float(row["accuracy_delta"]) for row in subset])
            def interval(values: np.ndarray) -> tuple[float, float]:
                if len(values) <= 1:
                    return float(values[0]), float(values[0])
                radius = 1.96 * float(values.std(ddof=1) / np.sqrt(len(values)))
                return float(values.mean() - radius), float(values.mean() + radius)
            loss_low, loss_high = interval(loss_deltas)
            accuracy_low, accuracy_high = interval(accuracy_deltas)
            output.append({
                "phase": phase, "condition": condition, "n": len(subset),
                "mean_loss": np.mean([float(row["loss"]) for row in subset]),
                "mean_loss_delta": float(loss_deltas.mean()),
                "loss_delta_ci_low": loss_low,
                "loss_delta_ci_high": loss_high,
                "mean_accuracy": np.mean([float(row["accuracy"]) for row in subset]),
                "mean_accuracy_delta": float(accuracy_deltas.mean()),
                "accuracy_delta_ci_low": accuracy_low,
                "accuracy_delta_ci_high": accuracy_high,
                "mean_spline_gate": np.mean([float(row["mean_spline_gate"]) for row in subset]),
            })
    return output


def main() -> None:
    args = parse_args()
    if not 0 < args.context_validation_fraction < 1:
        raise ValueError("--context-validation-fraction must be in (0, 1)")
    if min(args.context_steps, args.oracle_steps, args.select_every, args.n_control_points) <= 0:
        raise ValueError("step counts, --select-every, and --n-control-points must be positive")
    allowed_conditions = {"identity", "context_only", "oracle_query_labels"}
    requested_conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
    unknown_conditions = requested_conditions.difference(allowed_conditions)
    if not requested_conditions or unknown_conditions:
        raise ValueError(f"--conditions must be drawn from {sorted(allowed_conditions)}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    backbone, _ = load_backbone(args, device)
    validation = load_episode_bank(args.synthetic_validation_bank, expected_seed=args.synthetic_validation_seed,
                                   expected_count=args.synthetic_validation_tasks, device=device)
    test = load_episode_bank(args.synthetic_test_bank, expected_seed=args.synthetic_test_seed,
                             expected_count=args.synthetic_test_tasks, device=device)
    episodes = [make_synthetic_episode(item, "synthetic_validation") for item in validation]
    episodes += [make_synthetic_episode(item, "synthetic_test") for item in test]
    real_args = argparse.Namespace(dataset_suite=args.real_dataset_suite, datasets=args.real_datasets,
                                   test_size=args.real_test_size, max_rows=args.real_max_rows)
    for spec in resolve_specs(real_args):
        for seed in parse_csv(args.real_seeds, int):
            episodes.append(make_real_episode(build_episode(spec, seed, real_args, device)))
    rows = [
        row for ordinal, episode in enumerate(episodes)
        for row in run_episode(backbone, episode, args, ordinal, requested_conditions)
    ]
    write_csv(args.output_csv, rows)
    summaries = summarize(rows)
    write_csv(args.output_summary_csv, summaries)
    print(f"Wrote {len(rows)} paired records to {args.output_csv}", flush=True)
    for row in summaries:
        print(f"[{row['phase']} {row['condition']}] loss_delta={row['mean_loss_delta']:+.6f}, "
              f"accuracy_delta={row['mean_accuracy_delta']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
