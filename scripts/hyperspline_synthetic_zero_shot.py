"""Synthetic-to-unseen-synthetic HyperSpline zero-shot benchmark.

This is deliberately separate from ``hyperspline_synthetic_train.py``.  It
freezes disjoint synthetic calibration, train-audit, validation, and final-test
banks; trains on a fresh, deterministic stream of *different* synthetic tasks;
selects checkpoints using validation NLL only; and unlocks the final test bank
only in ``report``.  Thus a final test table is never used for generation,
training, early stopping, or checkpoint selection.

The primary output is paired, per-table performance against the unmodified
TabICL backbone.  It reports NLL/AUC/accuracy deltas as well as a paired Elo
*delta*.  That delta is a compact win/loss summary within this synthetic suite;
it is not numerically comparable to TabArena / TFM-Retouche Elo.

Typical use::

  python scripts/hyperspline_synthetic_zero_shot.py prepare --output-dir results/hs_synth_zs
  python scripts/hyperspline_synthetic_zero_shot.py audit --output-dir results/hs_synth_zs
  python scripts/hyperspline_synthetic_zero_shot.py train --output-dir results/hs_synth_zs \
      --run-name expanded_seed0 --scale-tasks 40000,160000 --device cuda:1
  python scripts/hyperspline_synthetic_zero_shot.py report --output-dir results/hs_synth_zs \
      --run-dir results/hs_synth_zs/runs/expanded_seed0 --scale-tasks 40000,160000 --device cuda:1

Run ``prepare`` once, then repeat ``train`` with different ``--model-seed`` and
run names.  A training run can be resumed safely with ``--resume``: every
streaming episode is a pure function of the global step, so resuming does not
repeat or change future synthetic tasks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

try:  # Supports tests and ``python scripts/...`` invocation.
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_query_marginal_synthetic_coverage_audit import (
        column_rows,
        extract_points,
        fit_robust_scaler,
        nearest_distances,
        source_auc,
        source_profile,
    )
    from scripts.hyperspline_synthetic_train import (
        SYNTHETIC_OBSERVATION_MODES,
        SyntheticEpisode,
        generate_episodes,
        generate_scheduled_episodes,
        load_episode_bank,
        save_episode_bank,
        validate_episode_classes,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation.
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_query_marginal_synthetic_coverage_audit import (
        column_rows,
        extract_points,
        fit_robust_scaler,
        nearest_distances,
        source_auc,
        source_profile,
    )
    from hyperspline_synthetic_train import (
        SYNTHETIC_OBSERVATION_MODES,
        SyntheticEpisode,
        generate_episodes,
        generate_scheduled_episodes,
        load_episode_bank,
        save_episode_bank,
        validate_episode_classes,
    )

from tabicl._hyperspline import (
    HyperSplineTransform,
    backbone_state_dict_hash,
    load_hyperspline_checkpoint,
    save_hyperspline_checkpoint,
    summarize_context,
)


MANIFEST_VERSION = 1
STATE_VERSION = 1
BANK_NAMES = ("calibration", "train_audit", "validation", "test")


def parse_int_csv(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)) or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected unique positive comma-separated integers")
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(values) != len(set(values)) or not all(0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("expected unique fractions in (0, 1)")
    return values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append stable streaming logs without losing a long run on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def bank_path(output_dir: Path, name: str) -> Path:
    return output_dir / "banks" / f"{name}.pt"


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "experiment_manifest.json"


def _bank_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prior_type": args.prior_type,
        "min_features": args.min_features,
        "max_features": args.max_features,
        "max_classes": args.max_classes,
        "prior_n_jobs": args.prior_n_jobs,
        "synthetic_observation_mode": args.synthetic_observation_mode,
        "sequence_lengths": list(args.sequence_lengths),
        "context_fractions": list(args.context_fractions),
    }


def _bank_specs(args: argparse.Namespace) -> dict[str, tuple[int, int, int]]:
    """Name -> (task count, generation seed, non-overlapping task-id offset)."""
    counts = (args.calibration_tasks, args.train_audit_tasks, args.validation_tasks, args.test_tasks)
    seeds = (args.calibration_seed, args.train_audit_seed, args.validation_seed, args.test_seed)
    if any(value <= 0 for value in counts) or len(set(seeds)) != len(seeds):
        raise ValueError("bank task counts must be positive and all bank seeds must be distinct")
    offset, result = 0, {}
    for name, count, seed in zip(BANK_NAMES, counts, seeds, strict=True):
        result[name] = (count, seed, offset)
        offset += count
    return result


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    manifest = manifest_path(output_dir)
    paths = [manifest, *(bank_path(output_dir, name) for name in BANK_NAMES)]
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to replace frozen experiment artifacts; use a new --output-dir or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        # This only replaces explicitly named benchmark artifacts, never the output directory or runs beneath it.
        for path in existing:
            path.unlink()

    device = torch.device("cpu")  # Frozen banks must be independent of a GPU choice.
    specs = _bank_specs(args)
    bank_metadata: dict[str, dict[str, Any]] = {}
    print("Generating four disjoint, frozen synthetic banks on CPU...", flush=True)
    for name, (count, seed, offset) in specs.items():
        episodes = generate_scheduled_episodes(
            args,
            count,
            source_seed=seed,
            task_offset=offset,
            device=device,
            sequence_lengths=args.sequence_lengths,
            context_fractions=args.context_fractions,
            observation_mode=args.synthetic_observation_mode,
        )
        path = bank_path(output_dir, name)
        save_episode_bank(path, episodes, source_seed=seed)
        bank_metadata[name] = {
            "path": str(path.relative_to(output_dir)),
            "count": count,
            "seed": seed,
            "task_id_offset": offset,
            "sha256": sha256_file(path),
        }
    payload = {
        "format_version": MANIFEST_VERSION,
        "created_unix": time.time(),
        "purpose": "synthetic-to-unseen-synthetic HyperSpline zero-shot benchmark",
        "bank_generation": _bank_config(args),
        "banks": bank_metadata,
        "protocol": {
            "train_stream": "fresh deterministic synthetic tasks disjoint from all frozen banks",
            "selection": "validation mean NLL only",
            "test_access": "report subcommand only after checkpoint selection",
            "elo": "paired delta relative to identity; do not compare its absolute value to TabArena Elo",
        },
    }
    write_json(manifest, payload)
    print(f"Prepared frozen benchmark manifest: {manifest}", flush=True)


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark manifest; run prepare first: {path}")
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("format_version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version in {path}")
    return payload


def load_bank_from_manifest(output_dir: Path, manifest: dict[str, Any], name: str, device: torch.device) -> list[SyntheticEpisode]:
    metadata = manifest["banks"].get(name)
    if metadata is None:
        raise ValueError(f"manifest has no {name!r} bank")
    path = output_dir / metadata["path"]
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {name} bank: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != metadata["sha256"]:
        raise ValueError(f"frozen {name} bank hash changed: {path}")
    return load_episode_bank(
        path,
        expected_seed=int(metadata["seed"]),
        expected_count=int(metadata["count"]),
        device=device,
        expected_observation_mode=manifest["bank_generation"]["synthetic_observation_mode"],
    )


def bank_shape_summary(episodes: Iterable[SyntheticEpisode]) -> list[dict[str, int]]:
    counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    for episode in episodes:
        counts[(episode.x_context.shape[1], episode.x_query.shape[1], episode.x_context.shape[2], episode.n_classes)] += 1
    return [
        {"n_context": context, "n_query": query, "n_features": features, "n_classes": classes, "n_tasks": count}
        for (context, query, features, classes), count in sorted(counts.items())
    ]


def audit(args: argparse.Namespace) -> None:
    output_dir, manifest = args.output_dir, load_manifest(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for the descriptor audit but is unavailable")
    banks = {name: load_bank_from_manifest(output_dir, manifest, name, torch.device("cpu")) for name in BANK_NAMES}
    audit_dir = output_dir / "audit"
    points = {}
    for name in BANK_NAMES:
        print(f"Extracting query-marginal descriptors for frozen {name} bank...", flush=True)
        points[name] = extract_points(
            banks[name], source=name, batch_size=args.summary_batch_size, progress_every=args.progress_every, device=device
        )
        write_csv(audit_dir / f"{name}_columns.csv", column_rows(points[name]))
    scaler = fit_robust_scaler(points["train_audit"], clip=args.scaler_clip)
    reference = scaler.transform(np.stack([point.descriptor for point in points["train_audit"]]))
    summary: dict[str, Any] = {
        "protocol": "fit all descriptor normalization on train_audit only; validation and test are hold-outs",
        "profiles": {name: source_profile(value, scaler=scaler) for name, value in points.items()},
        "bank_shapes": {name: bank_shape_summary(value) for name, value in banks.items()},
        "comparisons": {},
    }
    for name in ("validation", "test"):
        target = scaler.transform(np.stack([point.descriptor for point in points[name]]))
        distances = nearest_distances(reference, target)
        separability = source_auc(points["train_audit"], points[name], seed=args.auc_seed)
        summary["comparisons"][f"train_audit_to_{name}"] = {
            "nearest_distance_mean": float(distances.mean()),
            "nearest_distance_median": float(np.median(distances)),
            "nearest_distance_p95": float(np.quantile(distances, 0.95)),
            "source_auc": separability,
        }
    write_json(audit_dir / "alignment_summary.json", summary)
    print(f"Wrote synthetic train-to-holdout alignment audit: {audit_dir / 'alignment_summary.json'}", flush=True)


def hyperspline_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_control_points": args.n_control_points,
        "hidden_dim": args.hidden_dim,
        "gate_initial_probability": args.gate_initial_probability,
        "target_aware": args.target_aware,
        "conditioning_mode": args.conditioning_mode,
        "capacity_matched_conditioning": args.capacity_matched_conditioning,
    }


def validate_training_distribution(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Keep the fresh stream in the same declared family as the frozen tables."""
    expected = manifest["bank_generation"]
    actual = _bank_config(args)
    mismatches = {key: (expected[key], actual[key]) for key in expected if expected[key] != actual[key]}
    if mismatches:
        formatted = ", ".join(f"{key}: prepared={before!r}, train={after!r}" for key, (before, after) in mismatches.items())
        raise ValueError("fresh training distribution differs from frozen benchmark: " + formatted)


def model_state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def forward_identity(backbone, episode: SyntheticEpisode) -> tuple[torch.Tensor, torch.Tensor]:
    """The fixed context-standardisation baseline used by an identity HyperSpline.

    A zero-deformation HyperSpline standardises both partitions by context
    location/scale before reaching TabICL.  Comparing a learned spline to raw
    features would therefore accidentally credit basic standardisation rather
    than the learned monotone deformation.
    """
    statistics = summarize_context(episode.x_context.float())
    context = (episode.x_context.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    query = (episode.x_query.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    logits = backbone(torch.cat((context, query), dim=1), episode.y_context)
    logits = logits[..., : episode.n_classes]
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    return loss, logits


def forward_hyperspline(
    backbone, hyperspline: HyperSplineTransform, episode: SyntheticEpisode
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the audited conditioner without ever exposing query labels.

    ``HyperSplineTransform.forward`` is essential for ``query_marginal``: it
    builds a label-free summary of ``x_query`` alongside target-aware context
    statistics.  The older streaming helper only supported context-only
    conditioning, so using it here would silently run a different experiment.
    """
    transformed_context, transformed_query, parameters = hyperspline(
        episode.x_context,
        episode.x_query,
        y_context=episode.y_context if hyperspline.target_aware else None,
        return_parameters=True,
    )
    logits = backbone(torch.cat((transformed_context, transformed_query), dim=1), episode.y_context)
    logits = logits[..., : episode.n_classes]
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    diagnostics = {"grid_deformation_penalty": hyperspline.grid_deformation_penalty(parameters)}
    return loss, logits, diagnostics


def validation_mean_nll(backbone, hyperspline: HyperSplineTransform, episodes: Sequence[SyntheticEpisode]) -> float:
    losses = []
    hyperspline.eval()
    with torch.no_grad():
        for episode in episodes:
            backbone.clear_cache()
            loss, _, _ = forward_hyperspline(backbone, hyperspline, episode)
            losses.append(float(loss))
    return float(np.mean(losses))


def identity_validation_mean_nll(backbone, episodes: Sequence[SyntheticEpisode]) -> float:
    """Selection reference matching the deployed identity preprocessing exactly."""
    losses = []
    with torch.no_grad():
        for episode in episodes:
            backbone.clear_cache()
            loss, _ = forward_identity(backbone, episode)
            losses.append(float(loss))
    return float(np.mean(losses))


def training_source_seed(train_seed: int, step: int) -> int:
    """Return a unique-in-practice NumPy-compatible seed for one stream step.

    NumPy's legacy global RNG accepts only unsigned 32-bit seeds.  The prior
    stream deliberately spaces step seeds by 1,000,003; reducing modulo
    ``2**32`` preserves every already-valid historical seed and gives this
    odd stride a full 2**32-step period, far beyond this benchmark's budget.
    """
    if step <= 0:
        raise ValueError("training step must be positive")
    return (int(train_seed) + int(step) * 1_000_003) % (2**32)


def scheduled_training_episodes(args: argparse.Namespace, step: int, device: torch.device) -> list[SyntheticEpisode]:
    """Fresh task batch whose shape cycle matches the frozen hold-out schedule."""
    choices = [(length, fraction) for length in args.sequence_lengths for fraction in args.context_fractions]
    cycle = np.random.default_rng(args.train_seed).permutation(len(choices))
    length, fraction = choices[int(cycle[(step - 1) % len(cycle)])]
    step_args = argparse.Namespace(**vars(args))
    step_args.sequence_length = int(length)
    step_args.context_fraction = float(fraction)
    # A step-specific uint32 seed makes generation independent of resume history.
    return generate_episodes(
        step_args,
        args.tasks_per_step,
        source_seed=training_source_seed(args.train_seed, step),
        task_offset=1_000_000_000 + step * args.tasks_per_step,
        device=device,
    )


def run_fingerprint(args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    return stable_digest(
        {
            "manifest_hash": sha256_file(manifest_path(args.output_dir)),
            "model": hyperspline_config(args),
            "optimization": {
                "model_seed": args.model_seed,
                "train_seed": args.train_seed,
                "lr": args.lr,
                "tasks_per_step": args.tasks_per_step,
                "scale_tasks": args.scale_tasks,
                "validate_every": args.validate_every,
                "transform_regularization": args.transform_regularization,
            },
            "backbone_reference": args.checkpoint_version,
        }
    )


def save_training_state(
    path: Path,
    *,
    fingerprint: str,
    step: int,
    module: HyperSplineTransform,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    best_step: int,
    best_state: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": STATE_VERSION,
            "fingerprint": fingerprint,
            "step": step,
            "module_state": model_state_cpu(module),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_step": best_step,
            "best_state": best_state,
        },
        temporary,
    )
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.output_dir)
    validate_training_distribution(args, manifest)
    if any(tasks % args.tasks_per_step for tasks in args.scale_tasks):
        raise ValueError("every --scale-tasks value must be divisible by --tasks-per-step")
    if args.scale_tasks != sorted(args.scale_tasks):
        raise ValueError("--scale-tasks must be strictly increasing")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive")
    target_step = max(args.scale_tasks) // args.tasks_per_step
    if args.max_train_steps is not None:
        target_step = min(target_step, args.max_train_steps)
    if args.validate_every <= 0 or args.tasks_per_step <= 0 or args.lr <= 0:
        raise ValueError("invalid optimization configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    backbone, _ = load_backbone(args, device)
    if manifest["bank_generation"]["max_classes"] > backbone.max_classes:
        raise ValueError("frozen banks include more classes than this TabICL backbone supports")
    validation = load_bank_from_manifest(args.output_dir, manifest, "validation", device)
    validate_episode_classes(validation, backbone.max_classes)
    torch.manual_seed(args.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.model_seed)
    model_config = hyperspline_config(args)
    hyperspline = HyperSplineTransform(**model_config).to(device)
    optimizer = torch.optim.Adam(hyperspline.parameters(), lr=args.lr)

    run_dir = args.output_dir / "runs" / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    state_path = run_dir / "training_state.pt"
    fingerprint = run_fingerprint(args, manifest)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory exists; pass --resume or choose a new --run-name: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", {"fingerprint": fingerprint, "args": vars(args), "manifest": str(manifest_path(args.output_dir))})
    current_step, best_step = 0, 0
    hyperspline.eval()
    best_loss = identity_validation_mean_nll(backbone, validation)
    best_state = model_state_cpu(hyperspline)
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requires saved state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("format_version") != STATE_VERSION or state.get("fingerprint") != fingerprint:
            raise ValueError("resume state belongs to a different immutable benchmark or training configuration")
        hyperspline.load_state_dict(state["module_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(device)
        current_step = int(state["step"])
        best_loss, best_step, best_state = float(state["best_loss"]), int(state["best_step"]), state["best_state"]
        print(f"Resumed {args.run_name} after step={current_step}; selected validation NLL={best_loss:.6f} at step={best_step}", flush=True)
    else:
        append_csv(
            run_dir / "validation.csv",
            {"step": 0, "tasks_seen": 0, "mean_validation_nll": best_loss, "selected": True, "selection_reason": "identity"},
        )
        print(f"[identity] mean validation NLL={best_loss:.6f}; final test bank remains unopened", flush=True)
        save_training_state(
            state_path,
            fingerprint=fingerprint,
            step=0,
            module=hyperspline,
            optimizer=optimizer,
            best_loss=best_loss,
            best_step=best_step,
            best_state=best_state,
        )

    scale_steps = {tasks: tasks // args.tasks_per_step for tasks in args.scale_tasks}
    for step in range(current_step + 1, target_step + 1):
        hyperspline.train()
        episodes = scheduled_training_episodes(args, step, device)
        validate_episode_classes(episodes, backbone.max_classes)
        optimizer.zero_grad(set_to_none=True)
        task_losses, grid_penalties = [], []
        for episode in episodes:
            backbone.clear_cache()
            loss, _, diagnostics = forward_hyperspline(backbone, hyperspline, episode)
            objective = loss + args.transform_regularization * diagnostics["grid_deformation_penalty"]
            (objective / len(episodes)).backward()
            task_losses.append(float(loss.detach()))
            grid_penalties.append(float(diagnostics["grid_deformation_penalty"].detach()))
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), max_norm=1.0))
        optimizer.step()
        append_csv(
            run_dir / "training.csv",
            {
                "step": step,
                "tasks_seen": step * args.tasks_per_step,
                "mean_fresh_task_nll": float(np.mean(task_losses)),
                "mean_grid_deformation_penalty": float(np.mean(grid_penalties)),
                "pre_clip_gradient_norm": gradient_norm,
            },
        )
        is_scale = step in scale_steps.values()
        if step == 1 or step % args.validate_every == 0 or is_scale or step == target_step:
            validation_loss = validation_mean_nll(backbone, hyperspline, validation)
            selected = validation_loss < best_loss
            if selected:
                best_loss, best_step, best_state = validation_loss, step, model_state_cpu(hyperspline)
            append_csv(
                run_dir / "validation.csv",
                {
                    "step": step,
                    "tasks_seen": step * args.tasks_per_step,
                    "mean_validation_nll": validation_loss,
                    "selected": selected,
                    "selection_reason": "lower_nll" if selected else "not_selected",
                },
            )
            print(
                f"[step={step} tasks={step * args.tasks_per_step}] fresh_nll={np.mean(task_losses):.6f} "
                f"val_nll={validation_loss:.6f} best={best_loss:.6f}@{best_step}",
                flush=True,
            )
            if is_scale:
                budget_tasks = next(tasks for tasks, budget_step in scale_steps.items() if budget_step == step)
                current_state = model_state_cpu(hyperspline)
                hyperspline.load_state_dict(best_state, strict=True)
                checkpoint_path = checkpoint_dir / f"scale_{budget_tasks}.pt"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_hyperspline_checkpoint(
                    checkpoint_path,
                    hyperspline,
                    model_config,
                    backbone_reference=args.checkpoint_version,
                    backbone_hash=backbone_state_dict_hash(backbone),
                    step=best_step,
                )
                write_json(
                    checkpoint_path.with_suffix(".json"),
                    {
                        "budget_tasks": budget_tasks,
                        "budget_step": step,
                        "selected_step": best_step,
                        "selected_validation_nll": best_loss,
                        "model_config": model_config,
                        "manifest_hash": sha256_file(manifest_path(args.output_dir)),
                        "raw_final_test_used": False,
                    },
                )
                hyperspline.load_state_dict(current_state, strict=True)
                print(f"Saved validation-selected checkpoint for {budget_tasks:,} tasks: {checkpoint_path}", flush=True)
            save_training_state(
                state_path,
                fingerprint=fingerprint,
                step=step,
                module=hyperspline,
                optimizer=optimizer,
                best_loss=best_loss,
                best_step=best_step,
                best_state=best_state,
            )
    status = "complete" if target_step == max(args.scale_tasks) // args.tasks_per_step else "stopped early"
    print(f"Training {status} through {target_step * args.tasks_per_step:,} tasks. Run report only after a requested scale checkpoint exists.", flush=True)


def metric_row(backbone, hyperspline: HyperSplineTransform | None, episode: SyntheticEpisode) -> dict[str, float]:
    with torch.no_grad():
        backbone.clear_cache()
        if hyperspline is None:
            _, logits = forward_identity(backbone, episode)
        else:
            _, logits, _ = forward_hyperspline(backbone, hyperspline, episode)
        probabilities = torch.softmax(logits.flatten(0, 1), dim=-1).cpu().numpy().astype(np.float64)
    labels = episode.y_query.detach().cpu().numpy().astype(int)
    nll = float(log_loss(labels, probabilities, labels=list(range(episode.n_classes))))
    accuracy = float((probabilities.argmax(axis=1) == labels).mean())
    result = {"nll": nll, "accuracy": accuracy}
    if episode.n_classes == 2 and np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, probabilities[:, 1]))
        result.update({"auc": auc, "deployment_error": 1.0 - auc})
    else:
        result.update({"auc": float("nan"), "deployment_error": nll})
    return result


def paired_elo_delta(identity_errors: Sequence[float], candidate_errors: Sequence[float]) -> dict[str, float | int]:
    """Paired Elo delta from task outcomes, with ties worth half a point.

    Unlike standard global rating fitting, this has an exact interpretation:
    it is the rating difference whose expected win probability equals the
    observed candidate win/tie score against the identity baseline.
    """
    baseline = np.asarray(identity_errors, dtype=np.float64)
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("paired Elo requires equally sized non-empty one-dimensional vectors")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired Elo requires finite deployment errors")
    wins = int(np.count_nonzero(candidate < baseline))
    losses = int(np.count_nonzero(candidate > baseline))
    ties = int(baseline.size - wins - losses)
    score = (wins + 0.5 * ties) / baseline.size
    # Infinite ratings for an all-win/all-loss finite suite are unhelpful in a
    # report, so use the conventional half-game continuity correction.
    adjusted = (wins + 0.5 * ties + 0.5) / (baseline.size + 1.0)
    return {
        "n_tasks": int(baseline.size),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "score": float(score),
        "elo_delta": float(400.0 * math.log10(adjusted / (1.0 - adjusted))),
    }


def bootstrap_elo(identity_errors: Sequence[float], candidate_errors: Sequence[float], *, seed: int, samples: int) -> dict[str, float]:
    baseline, candidate = np.asarray(identity_errors), np.asarray(candidate_errors)
    if samples <= 0:
        return {"elo_delta_ci_low": float("nan"), "elo_delta_ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [paired_elo_delta(baseline[index], candidate[index])["elo_delta"] for index in rng.integers(0, baseline.size, (samples, baseline.size))],
        dtype=np.float64,
    )
    return {"elo_delta_ci_low": float(np.quantile(draws, 0.025)), "elo_delta_ci_high": float(np.quantile(draws, 0.975))}


def aggregate_report(rows: Sequence[dict[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("no report rows")
    base = np.asarray([float(row["identity_nll"]) for row in rows])
    candidate = np.asarray([float(row["candidate_nll"]) for row in rows])
    deployment_base = np.asarray([float(row["identity_deployment_error"]) for row in rows])
    deployment_candidate = np.asarray([float(row["candidate_deployment_error"]) for row in rows])
    nll_delta = candidate - base
    accuracy_delta = np.asarray([float(row["candidate_accuracy"]) - float(row["identity_accuracy"]) for row in rows])
    finite_auc = [row for row in rows if math.isfinite(float(row["identity_auc"])) and math.isfinite(float(row["candidate_auc"]))]
    summary: dict[str, Any] = {
        "n_tables": len(rows),
        "mean_nll_delta": float(nll_delta.mean()),
        "median_nll_delta": float(np.median(nll_delta)),
        "mean_relative_nll_delta": float(np.mean(nll_delta / np.maximum(base, 1e-12))),
        "mean_accuracy_delta": float(accuracy_delta.mean()),
        "nll_wins": int(np.count_nonzero(nll_delta < 0)),
        "nll_losses": int(np.count_nonzero(nll_delta > 0)),
        "material_nll_wins_1pct": int(np.count_nonzero(nll_delta / np.maximum(base, 1e-12) <= -0.01)),
        "material_nll_harms_1pct": int(np.count_nonzero(nll_delta / np.maximum(base, 1e-12) >= 0.01)),
        "elo": paired_elo_delta(deployment_base, deployment_candidate),
        "metric_note": "deployment error is 1-AUC for binary tables and NLL for multiclass tables",
    }
    summary["elo"].update(bootstrap_elo(deployment_base, deployment_candidate, seed=bootstrap_seed, samples=bootstrap_samples))
    if finite_auc:
        auc_delta = np.asarray([float(row["candidate_auc"]) - float(row["identity_auc"]) for row in finite_auc])
        summary.update(
            {
                "binary_tables": len(finite_auc),
                "mean_auc_delta": float(auc_delta.mean()),
                "median_auc_delta": float(np.median(auc_delta)),
                "material_auc_wins_0p005": int(np.count_nonzero(auc_delta >= 0.005)),
                "material_auc_harms_0p005": int(np.count_nonzero(auc_delta <= -0.005)),
            }
        )
    return summary


def average_model_seed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average independent model-seed metrics before treating tables as Elo games.

    This avoids accidentally giving a 1,024-table suite three times its
    statistical weight merely because it was trained from three random model
    initializations.  It is a mean-of-model-seeds analysis, not an ensemble.
    """
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    combined = []
    metric_fields = (
        "identity_nll", "identity_accuracy", "identity_auc", "identity_deployment_error",
        "candidate_nll", "candidate_accuracy", "candidate_auc", "candidate_deployment_error",
    )
    for task_id, task_rows in sorted(by_task.items()):
        first = dict(task_rows[0])
        first["model_seed_runs"] = len(task_rows)
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in task_rows], dtype=np.float64)
            # Identity metrics should match exactly.  nanmean also retains the
            # correct multiclass AUC marker rather than turning it into zero.
            first[field] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        combined.append(first)
    return combined


def report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for report but unavailable")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    backbone, _ = load_backbone(args, device)
    test = load_bank_from_manifest(args.output_dir, manifest, "test", device)
    validate_episode_classes(test, backbone.max_classes)
    baseline = {episode.task_id: metric_row(backbone, None, episode) for episode in test}
    output_dir = args.output_dir / "reports"
    all_rows: list[dict[str, Any]] = []
    all_summary: dict[str, Any] = {"protocol": manifest["protocol"], "runs": {}}
    rows_by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run_dir in args.run_dir:
        run_dir = Path(run_dir)
        for budget_tasks in args.scale_tasks:
            checkpoint = run_dir / "checkpoints" / f"scale_{budget_tasks}.pt"
            sidecar = checkpoint.with_suffix(".json")
            if not checkpoint.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"missing selected scale checkpoint/sidecar: {checkpoint}")
            checkpoint_info = json.loads(sidecar.read_text(encoding="utf8"))
            if checkpoint_info.get("manifest_hash") != sha256_file(manifest_path(args.output_dir)):
                raise ValueError(f"checkpoint was selected on a different frozen benchmark: {checkpoint}")
            hyperspline, _ = load_hyperspline_checkpoint(
                checkpoint,
                device=device,
                expected_backbone_reference=args.checkpoint_version,
                expected_backbone_hash=backbone_state_dict_hash(backbone),
            )
            hyperspline.eval()
            rows = []
            for episode in test:
                candidate = metric_row(backbone, hyperspline, episode)
                row: dict[str, Any] = {
                    "run": run_dir.name,
                    "checkpoint": str(checkpoint),
                    "budget_tasks": budget_tasks,
                    "selected_step": checkpoint_info["selected_step"],
                    "task_id": episode.task_id,
                    "n_context": episode.x_context.shape[1],
                    "n_query": episode.x_query.shape[1],
                    "n_features": episode.x_context.shape[2],
                    "n_classes": episode.n_classes,
                }
                row.update({f"identity_{name}": value for name, value in baseline[episode.task_id].items()})
                row.update({f"candidate_{name}": value for name, value in candidate.items()})
                rows.append(row)
            key = f"{run_dir.name}/scale_{budget_tasks}"
            all_rows.extend(rows)
            rows_by_budget[budget_tasks].extend(rows)
            all_summary["runs"][key] = aggregate_report(rows, bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples)
            print(
                f"[{key}] mean NLL delta={all_summary['runs'][key]['mean_nll_delta']:.6g}; "
                f"paired Elo delta={all_summary['runs'][key]['elo']['elo_delta']:.1f}",
                flush=True,
            )
    if len(args.run_dir) > 1:
        all_summary["mean_over_model_seeds"] = {
            f"scale_{budget_tasks}": aggregate_report(
                average_model_seed_rows(rows), bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples
            )
            for budget_tasks, rows in sorted(rows_by_budget.items())
        }
    write_csv(output_dir / "per_table_metrics.csv", all_rows)
    write_json(output_dir / "summary.json", all_summary)
    print(f"Wrote held-out zero-shot metrics: {output_dir}", flush=True)


def add_shared_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--synthetic-observation-mode", choices=SYNTHETIC_OBSERVATION_MODES, default="coverage_expanded")
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--sequence-lengths", type=parse_int_csv, default=parse_int_csv("128,256,512,1024"))
    parser.add_argument("--context-fractions", type=parse_float_csv, default=parse_float_csv("0.50,0.70,0.85"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate the immutable synthetic table banks.")
    add_shared_benchmark_arguments(prepare_parser)
    prepare_parser.add_argument("--calibration-tasks", type=int, default=256)
    prepare_parser.add_argument("--train-audit-tasks", type=int, default=1024)
    prepare_parser.add_argument("--validation-tasks", type=int, default=512)
    prepare_parser.add_argument("--test-tasks", type=int, default=1024)
    prepare_parser.add_argument("--calibration-seed", type=int, default=71_001)
    prepare_parser.add_argument("--train-audit-seed", type=int, default=72_001)
    prepare_parser.add_argument("--validation-seed", type=int, default=73_001)
    prepare_parser.add_argument("--test-seed", type=int, default=74_001)
    prepare_parser.add_argument("--overwrite", action="store_true", help="Replace only this benchmark's manifest and four named bank files.")

    audit_parser = subparsers.add_parser("audit", help="Measure train-to-holdout descriptor alignment before training.")
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.add_argument("--device", default="cpu")
    audit_parser.add_argument("--summary-batch-size", type=int, default=16)
    audit_parser.add_argument("--progress-every", type=int, default=32)
    audit_parser.add_argument("--scaler-clip", type=float, default=10.0)
    audit_parser.add_argument("--auc-seed", type=int, default=17)

    train_parser = subparsers.add_parser("train", help="Train on fresh tasks; select only on frozen validation tables.")
    add_shared_benchmark_arguments(train_parser)
    train_parser.add_argument("--checkpoint", default=None)
    train_parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--run-name", required=True)
    train_parser.add_argument("--scale-tasks", type=parse_int_csv, required=True, help="Cumulative fresh-task budgets, e.g. 40000,160000,640000.")
    train_parser.add_argument("--max-train-steps", type=int, default=None, help="Smoke-test cap; never marks an unfinished scale as complete.")
    train_parser.add_argument("--tasks-per-step", type=int, default=4)
    train_parser.add_argument("--validate-every", type=int, default=1000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--n-control-points", type=int, default=20)
    train_parser.add_argument("--gate-initial-probability", type=float, default=0.10)
    train_parser.add_argument(
        "--conditioning-mode",
        choices=("context", "query_marginal", "context_query_shift"),
        default="query_marginal",
        help="Primary arm is query_marginal, exactly the 33D conditioning input audited above.",
    )
    train_parser.add_argument(
        "--target-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the context-label statistics in query_marginal conditioning; query labels are never accepted.",
    )
    train_parser.add_argument(
        "--capacity-matched-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad control arms to the largest conditioner width so arm comparisons isolate available information.",
    )
    train_parser.add_argument("--transform-regularization", type=float, default=0.0)
    train_parser.add_argument("--model-seed", type=int, default=0)
    train_parser.add_argument("--train-seed", type=int, default=61_001)
    train_parser.add_argument("--resume", action="store_true")

    report_parser = subparsers.add_parser("report", help="Open frozen test bank after all selection is complete.")
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.add_argument("--checkpoint", default=None)
    report_parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    report_parser.add_argument("--device", default="cuda")
    report_parser.add_argument("--run-dir", type=Path, action="append", required=True, help="May be repeated for independently seeded runs.")
    report_parser.add_argument("--scale-tasks", type=parse_int_csv, required=True)
    report_parser.add_argument("--bootstrap-seed", type=int, default=9_001)
    report_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "train":
        train(args)
    elif args.command == "report":
        report(args)
    else:  # pragma: no cover - argparse protects this branch.
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
