"""Teacher-free real-meta scaling experiment for the learned-rank HyperSpline.

This is the decisive data-coverage bridge after the small LODO experiment.
The transform is exactly the teacher-free ``LearnedSharedRankBasisSpline``:
the context-only HyperSpline predicts direct amplitudes for a globally learned,
smooth rank basis.  A DirectSpline teacher is not loaded, fitted, queried, or
used as a target anywhere in this program.

The three arms keep the output architecture and total task budget fixed:

``real_4``
    Four fixed real PMLB dataset identities, matching the earlier LODO scale.
``real_all``
    Every accepted real PMLB identity in the supplied training bank.
``real_all_plus_synthetic``
    Half real / half native TabICL synthetic episodes in each update.

Checkpoint and one *global* residual interpolation scale are selected solely
on a dataset-disjoint real validation bank.  The final seven-dataset suite is
constructed only after selection.  Thus it is a genuine test of whether a
context-to-spline policy becomes more transferable as its meta-dataset grows,
not a teacher-distillation experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:  # Allows tests/imports and direct ``python scripts/...py`` execution.
    from scripts.direct_spline_dataset_headroom import release_cuda
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_rank_basis_lodo import LearnedSharedRankBasisSpline
    from scripts.hyperspline_rank_basis_zero_shot import metrics, parameter_metrics
    from scripts.hyperspline_real_task_bank import RealEpisode, load_bank
    from scripts.hyperspline_real_zero_shot_eval import (
        MIXED_SPECS,
        NUMERICAL_SPECS,
        build_episode as build_final_episode,
    )
    from scripts.hyperspline_synthetic_train import (
        SyntheticEpisode,
        generate_episodes,
        seed_generator,
        validate_episode_classes,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from direct_spline_dataset_headroom import release_cuda
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_rank_basis_lodo import LearnedSharedRankBasisSpline
    from hyperspline_rank_basis_zero_shot import metrics, parameter_metrics
    from hyperspline_real_task_bank import RealEpisode, load_bank
    from hyperspline_real_zero_shot_eval import MIXED_SPECS, NUMERICAL_SPECS, build_episode as build_final_episode
    from hyperspline_synthetic_train import SyntheticEpisode, generate_episodes, seed_generator, validate_episode_classes

from tabicl._hyperspline import summarize_context


DEFAULT_REAL4_DATASETS = ("magic", "pendigits", "phoneme", "spambase")
DEFAULT_RESIDUAL_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class EpisodeBatch:
    """Several episodes with identical table dimensions, stacked on batch axis."""

    datasets: tuple[str, ...]
    split_seeds: tuple[int, ...]
    episode_ids: tuple[int, ...]
    source: str
    x_context: torch.Tensor  # (B, N_context, D)
    x_query: torch.Tensor  # (B, N_query, D)
    y_context: torch.Tensor  # (B, N_context), float for TabICL
    y_query: torch.Tensor  # (B, N_query), long
    numerical_mask: torch.Tensor  # (D,)

    @property
    def batch_size(self) -> int:
        return int(self.x_context.shape[0])


class BatchedLearnedSharedRankBasisSpline(LearnedSharedRankBasisSpline):
    """The teacher-free model with a BxN labelled-context interface.

    The earlier scripts use a single task at a time and therefore pass labels
    as ``(N,)``.  Supporting both forms lets this experiment stack compatible
    real tasks into true frozen-backbone mini-batches without changing the
    older experiments.
    """

    def _raw(self, x: torch.Tensor, y: torch.Tensor):
        labels = y.float()
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if labels.ndim != 2 or labels.shape[:2] != x.shape[:2]:
            raise ValueError("context labels must have shape (N,) or (B, N)")
        labelled = labels if self.target_aware else None
        statistics = summarize_context(x, y_context=labelled)
        raw, _ = self.encoder.generate_raw(statistics, x_context=x, y_context=labelled)
        return raw, statistics


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected non-empty comma-separated unique integers")
    return values


def parse_float_csv(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected non-empty comma-separated unique floats")
    return values


def parse_name_csv(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("expected non-empty comma-separated unique dataset names")
    return names


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")


def canonical_pmlb_name(name: str) -> str:
    return name if name.startswith("pmlb_") else f"pmlb_{name}"


def resolve_real_train_datasets(
    arm: str, available: Iterable[str], real4_datasets: Sequence[str]
) -> list[str]:
    """Choose real identities without silently falling back to a smaller arm."""
    known = sorted(set(available))
    if arm != "real_4":
        return known
    requested = [canonical_pmlb_name(name) for name in real4_datasets]
    missing = sorted(set(requested).difference(known))
    if missing:
        raise ValueError(
            "--real4-datasets are absent from --real-train-bank; rebuild the bank or choose available names: "
            f"{missing}"
        )
    return requested


def sanitize_real_episode(episode: RealEpisode) -> tuple[RealEpisode, int]:
    """Median-impute non-finite numerical values using that episode's context.

    Older PMLB banks predate this guard and may contain NaNs from source TSVs.
    The statistic is intentionally fit on labelled-context *features* only,
    never on query rows or labels, so repairing an existing bank does not leak
    query information into the transform or into TabICL.
    """
    context, query = episode.x_context.clone(), episode.x_query.clone()
    numeric_context = context[..., episode.numerical_mask]
    numeric_query = query[..., episode.numerical_mask]
    context_finite = torch.isfinite(numeric_context)
    query_finite = torch.isfinite(numeric_query)
    replacements = int((~context_finite).sum() + (~query_finite).sum())
    if replacements == 0:
        return episode, 0
    context_with_nan = torch.where(context_finite, numeric_context, torch.full_like(numeric_context, float("nan")))
    medians = torch.nanmedian(context_with_nan, dim=1).values
    if not torch.isfinite(medians).all():
        bad = torch.where(~torch.isfinite(medians).all(dim=0))[0].tolist()
        raise ValueError(f"{episode.dataset} has numerical columns with no finite context value: {bad}")
    context[..., episode.numerical_mask] = torch.where(
        context_finite, numeric_context, medians.unsqueeze(1)
    )
    query[..., episode.numerical_mask] = torch.where(
        query_finite, numeric_query, medians.unsqueeze(1)
    )
    return replace(episode, x_context=context, x_query=query), replacements


def sanitize_real_bank(episodes: Sequence[RealEpisode]) -> tuple[list[RealEpisode], int]:
    repaired, replacements = [], 0
    for episode in episodes:
        fixed, count = sanitize_real_episode(episode)
        repaired.append(fixed)
        replacements += count
    return repaired, replacements


def batch_key(episode: RealEpisode | SyntheticEpisode) -> tuple[int, int, int, tuple[bool, ...]]:
    if isinstance(episode, SyntheticEpisode):
        mask = (True,) * int(episode.x_context.shape[-1])
    else:
        mask = tuple(bool(value) for value in episode.numerical_mask.detach().cpu().tolist())
    return (
        int(episode.x_context.shape[1]),
        int(episode.x_query.shape[1]),
        int(episode.x_context.shape[2]),
        mask,
    )


def episode_dataset(episode: RealEpisode | SyntheticEpisode) -> str:
    return episode.dataset if isinstance(episode, RealEpisode) else "synthetic_native_prior"


def episode_seed(episode: RealEpisode | SyntheticEpisode) -> int:
    return int(episode.split_seed if isinstance(episode, RealEpisode) else episode.source_seed)


def episode_id(episode: RealEpisode | SyntheticEpisode) -> int:
    return int(episode.split_seed if isinstance(episode, RealEpisode) else episode.task_id)


def stack_episode_batch(episodes: Sequence[RealEpisode | SyntheticEpisode], *, source: str) -> EpisodeBatch:
    if not episodes:
        raise ValueError("cannot stack an empty episode batch")
    key = batch_key(episodes[0])
    if any(batch_key(episode) != key for episode in episodes):
        raise ValueError("only shape-compatible episodes may be stacked")
    if isinstance(episodes[0], RealEpisode):
        mask = episodes[0].numerical_mask
    else:
        mask = torch.ones(episodes[0].x_context.shape[-1], dtype=torch.bool, device=episodes[0].x_context.device)
    return EpisodeBatch(
        datasets=tuple(episode_dataset(episode) for episode in episodes),
        split_seeds=tuple(episode_seed(episode) for episode in episodes),
        episode_ids=tuple(episode_id(episode) for episode in episodes),
        source=source,
        x_context=torch.cat([episode.x_context for episode in episodes], dim=0),
        x_query=torch.cat([episode.x_query for episode in episodes], dim=0),
        y_context=torch.cat([episode.y_context for episode in episodes], dim=0),
        y_query=torch.stack([episode.y_query for episode in episodes], dim=0),
        numerical_mask=mask,
    )


def make_compatible_batches(
    episodes: Sequence[RealEpisode | SyntheticEpisode], *, source: str, max_batch_size: int
) -> list[EpisodeBatch]:
    """Group same-shape episodes so a frozen TabICL call has B > 1 when possible."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    grouped: dict[tuple[int, int, int, tuple[bool, ...]], list[RealEpisode | SyntheticEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[batch_key(episode)].append(episode)
    batches = []
    for key in sorted(grouped):
        items = grouped[key]
        for start in range(0, len(items), max_batch_size):
            batches.append(stack_episode_batch(items[start : start + max_batch_size], source=source))
    return batches


def scale_generated_parameters(
    model: BatchedLearnedSharedRankBasisSpline,
    parameters: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Convexly interpolate identity and the predicted table, preserving monotonicity."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("residual scale alpha must be in [0, 1]")
    identity = model.grid.view(1, 1, -1)
    return {**parameters, "values": identity + alpha * (parameters["values"] - identity)}


def _replace_numerical(
    model: BatchedLearnedSharedRankBasisSpline,
    batch: EpisodeBatch,
    parameters: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep categoricals exactly untouched; transform only selected numerical columns."""
    numeric_context = batch.x_context[..., batch.numerical_mask]
    numeric_query = batch.x_query[..., batch.numerical_mask]
    if parameters is None:
        statistics = summarize_context(numeric_context)
        transformed_context = (numeric_context.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
        transformed_query = (numeric_query.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
    else:
        transformed_context = model.transform(numeric_context, parameters)
        transformed_query = model.transform(numeric_query, parameters)
    context, query = batch.x_context.clone(), batch.x_query.clone()
    context[..., batch.numerical_mask] = transformed_context
    query[..., batch.numerical_mask] = transformed_query
    return context, query


def reference_predictions(backbone, model: BatchedLearnedSharedRankBasisSpline, batch: EpisodeBatch):
    context, query = _replace_numerical(model, batch, None)
    backbone.clear_cache()
    return backbone(torch.cat((context, query), dim=1), batch.y_context), batch.y_query


def model_predictions(
    backbone,
    model: BatchedLearnedSharedRankBasisSpline,
    batch: EpisodeBatch,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    numeric_context = batch.x_context[..., batch.numerical_mask]
    numeric_query = batch.x_query[..., batch.numerical_mask]
    # Crucially, only y_context reaches parameter generation.  y_query is used
    # later, solely by cross entropy / reported metrics.
    parameters = model.generated_parameters(numeric_context, batch.y_context.long())
    parameters = scale_generated_parameters(model, parameters, alpha)
    transformed_context = model.transform(numeric_context, parameters)
    transformed_query = model.transform(numeric_query, parameters)
    context, query = batch.x_context.clone(), batch.x_query.clone()
    context[..., batch.numerical_mask] = transformed_context
    query[..., batch.numerical_mask] = transformed_query
    backbone.clear_cache()
    logits = backbone(torch.cat((context, query), dim=1), batch.y_context)
    return logits, batch.y_query, parameters


def macro_dataset_mean(rows: Sequence[dict], field: str = "loss") -> float:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row["dataset"])].append(float(row[field]))
    if not values:
        raise ValueError("cannot aggregate an empty evaluation")
    return float(np.mean([np.mean(dataset_values) for dataset_values in values.values()]))


def choose_alpha(scores: dict[float, float]) -> tuple[float, float]:
    """Deterministically favour the smaller residual if scores are tied."""
    if not scores:
        raise ValueError("at least one validation alpha score is required")
    return min(scores.items(), key=lambda item: (item[1], item[0]))


def _slice_parameters(parameters: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in parameters.items():
        result[name] = value[index : index + 1] if torch.is_tensor(value) and value.ndim else value
    return result


@torch.no_grad()
def evaluation_rows(
    backbone,
    model: BatchedLearnedSharedRankBasisSpline,
    episodes: Sequence[RealEpisode],
    *,
    stage: str,
    step: int,
    alpha: float | None,
    batch_size: int,
    run_fields: dict,
    collect_parameters: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Evaluate each stored real episode but use compatible mini-batches internally."""
    rows, records = [], []
    for batch in make_compatible_batches(episodes, source="real", max_batch_size=batch_size):
        if alpha is None:
            logits, targets = reference_predictions(backbone, model, batch)
            parameters = None
        else:
            logits, targets, parameters = model_predictions(backbone, model, batch, alpha=alpha)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"non-finite TabICL logits during {stage} for datasets={list(batch.datasets)}; "
                "inspect the context/query features in the bank"
            )
        probabilities = logits.softmax(-1).detach().cpu()
        for index in range(batch.batch_size):
            row = {
                **run_fields,
                "stage": stage,
                "step": step,
                "alpha": "reference" if alpha is None else alpha,
                "dataset": batch.datasets[index],
                "split_seed": batch.split_seeds[index],
                "episode_id": batch.episode_ids[index],
                "source": batch.source,
                "n_context": int(batch.x_context.shape[1]),
                "n_query": int(batch.x_query.shape[1]),
                "n_features": int(batch.x_context.shape[2]),
                "n_numerical_features": int(batch.numerical_mask.sum()),
                **metrics(probabilities[index : index + 1], targets[index : index + 1].detach().cpu()),
            }
            if parameters is not None:
                parameter = _slice_parameters(parameters, index)
                row.update(parameter_metrics(parameter))
                curve = parameter["values"][0] - model.grid.view(1, -1)
                row["curve_deformation_rms"] = float(curve.square().mean().sqrt())
                row["curve_deformation_abs_max"] = float(curve.abs().max())
                if collect_parameters:
                    records.append(
                        {
                            **run_fields,
                            "stage": stage,
                            "step": step,
                            "alpha": alpha,
                            "dataset": batch.datasets[index],
                            "split_seed": batch.split_seeds[index],
                            "episode_id": batch.episode_ids[index],
                            "parameters": {name: value.detach().cpu() for name, value in parameter.items()},
                        }
                    )
            rows.append(row)
    return rows, records


def parameter_column_rows(records: Sequence[dict]) -> list[dict]:
    rows = []
    for record in records:
        parameters = record.pop("parameters")
        values = parameters["values"][0]
        coefficients = parameters["coefficients"][0]
        identity = torch.linspace(-4.0, 4.0, values.shape[-1], dtype=values.dtype)
        for column in range(values.shape[0]):
            curve = values[column] - identity
            coefficient = coefficients[column]
            rows.append(
                {
                    **record,
                    "column": column,
                    "curve_deformation_rms": float(curve.square().mean().sqrt()),
                    "curve_deformation_abs_max": float(curve.abs().max()),
                    "coefficient_rms": float(coefficient.square().mean().sqrt()),
                    "coefficient_abs_max": float(coefficient.abs().max()),
                    **{f"coefficient_{index}": float(value) for index, value in enumerate(coefficient)},
                }
            )
    return rows


def select_real_episodes(
    by_dataset: dict[str, list[RealEpisode]],
    *,
    task_count: int,
    datasets_per_update: int,
    rng: np.random.Generator,
) -> list[RealEpisode]:
    """Sample identities uniformly, then episodes uniformly within identity."""
    if task_count <= 0:
        return []
    names = sorted(by_dataset)
    if not names:
        raise ValueError("no real meta-training episodes")
    selected_names = list(rng.choice(names, size=min(len(names), datasets_per_update), replace=False))
    repeats = math.ceil(task_count / len(selected_names))
    result = []
    for dataset in selected_names:
        choices = rng.choice(len(by_dataset[dataset]), size=repeats, replace=repeats > len(by_dataset[dataset]))
        result.extend(by_dataset[dataset][int(choice)] for choice in np.atleast_1d(choices))
    return result[:task_count]


def extra_regularization(model: BatchedLearnedSharedRankBasisSpline) -> torch.Tensor:
    return model.shared_basis_regularization()


def effective_real_task_count(args: argparse.Namespace) -> int:
    if args.arm != "real_all_plus_synthetic":
        return args.meta_batch_episodes
    return int(round(args.meta_batch_episodes * (1.0 - args.synthetic_fraction)))


def run_model_seed(
    args: argparse.Namespace,
    *,
    backbone,
    train_episodes: list[RealEpisode],
    validation_episodes: list[RealEpisode],
    model_seed: int,
    output: Path,
    device: torch.device,
) -> dict:
    run_dir = output / f"model_seed_{model_seed}"
    summary_path = run_dir / "summary.json"
    if args.resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf8"))
        if existing.get("status") == "complete":
            print(f"[resume] completed model_seed={model_seed}; skipping {run_dir}", flush=True)
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)

    real_datasets = resolve_real_train_datasets(args.arm, {episode.dataset for episode in train_episodes}, args.real4_datasets)
    train_episodes = [episode for episode in train_episodes if episode.dataset in real_datasets]
    by_dataset = {dataset: [episode for episode in train_episodes if episode.dataset == dataset] for dataset in real_datasets}
    if any(not episodes for episodes in by_dataset.values()):  # Defensive: keys should be exact.
        raise ValueError("a requested real training dataset has no usable episodes")
    validation_datasets = sorted({episode.dataset for episode in validation_episodes})
    if set(real_datasets).intersection(validation_datasets):
        raise ValueError("real training and validation dataset identities must be disjoint")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = BatchedLearnedSharedRankBasisSpline(
        rank=args.learned_rank,
        hidden_dim=args.hidden_dim,
        coefficient_bound=args.coefficient_bound,
        basis_bound=args.shared_basis_bound,
        basis_init_rms=args.shared_basis_init_rms,
        target_aware=args.target_aware,
        raw_context=args.raw_context,
    ).to(device)
    parameters = list(nn.Module.parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.train_seed + 100_003 * model_seed)
    run_fields = {"arm": args.arm, "model_seed": model_seed}
    manifest = {
        "status": "running",
        "protocol": (
            "teacher_free_learned_rank_basis + balanced_real_meta_batches + "
            "dataset_disjoint_real_validation + validation_selected_global_residual_scale"
        ),
        "teacher_used": False,
        "teacher_parameters_used_as_targets": False,
        "query_labels_enter_parameter_generation": False,
        "run": run_fields,
        "real_train_datasets": real_datasets,
        "real_validation_datasets": validation_datasets,
        "real_train_episode_count": len(train_episodes),
        "real_validation_episode_count": len(validation_episodes),
        "total_meta_batch_episodes": args.meta_batch_episodes,
        "real_meta_batch_episodes": effective_real_task_count(args),
        "synthetic_meta_batch_episodes": args.meta_batch_episodes - effective_real_task_count(args),
        "residual_alphas": args.residual_alphas,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(run_dir / "manifest.json", manifest)

    reference_validation, _ = evaluation_rows(
        backbone, model, validation_episodes, stage="reference_validation", step=0, alpha=None,
        batch_size=args.evaluation_batch_size, run_fields=run_fields,
    )
    reference_validation_score = macro_dataset_mean(reference_validation)
    evaluation_log = list(reference_validation)
    selection_log = [{
        **run_fields, "step": 0, "selected_alpha": 0.0, "candidate_validation_loss": reference_validation_score,
        "reference_validation_loss": reference_validation_score, "improved": True, "stale_validations": 0,
    }]
    training_log: list[dict] = []
    best_score, best_step, best_alpha, stale = reference_validation_score, 0, 0.0, 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.time()
    real_task_count = effective_real_task_count(args)
    working_backbone_batch_size = args.max_backbone_batch_size

    print(
        f"[{args.arm} seed={model_seed}] teacher-free learned rank={args.learned_rank}; "
        f"real identities={len(real_datasets)}, validation identities={len(validation_datasets)}, "
        f"effective meta batch={args.meta_batch_episodes} (real={real_task_count}, synthetic={args.meta_batch_episodes-real_task_count}), "
        f"max backbone batch={args.max_backbone_batch_size}",
        flush=True,
    )
    print(f"[{args.arm} seed={model_seed}] reference validation macro NLL={reference_validation_score:.6f}", flush=True)

    for step in range(1, args.steps + 1):
        model.train()
        selected_real = select_real_episodes(
            by_dataset, task_count=real_task_count, datasets_per_update=args.datasets_per_update, rng=rng
        )
        selected_synthetic: list[SyntheticEpisode] = []
        synthetic_count = args.meta_batch_episodes - len(selected_real)
        if synthetic_count:
            synthetic_args = argparse.Namespace(**vars(args), sequence_length=args.synthetic_sequence_length)
            selected_synthetic = generate_episodes(
                synthetic_args, synthetic_count, source_seed=None,
                task_offset=step * args.meta_batch_episodes, device=device,
            )
            validate_episode_classes(selected_synthetic, backbone.max_classes)
        oom_backoffs = 0
        while True:
            batches = make_compatible_batches(
                selected_real, source="real_meta", max_batch_size=working_backbone_batch_size
            ) + make_compatible_batches(
                selected_synthetic, source="synthetic_native_prior", max_batch_size=working_backbone_batch_size
            )
            if sum(batch.batch_size for batch in batches) != args.meta_batch_episodes:
                raise AssertionError("meta-batch assembly lost an episode")
            optimizer.zero_grad(set_to_none=True)
            task_losses, trusts, generated = [], [], []
            try:
                for batch in batches:
                    logits, targets, current = model_predictions(backbone, model, batch, alpha=1.0)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            f"non-finite training logits at step={step} for datasets={list(batch.datasets)}"
                        )
                    per_example_loss = F.cross_entropy(
                        logits.flatten(0, 1), targets.flatten(), reduction="none"
                    ).reshape(batch.batch_size, -1).mean(dim=1)
                    task_loss = per_example_loss.mean()
                    trust = model.trust_region(current)
                    weight = batch.batch_size / args.meta_batch_episodes
                    # Backward per compatible mini-batch frees the frozen backbone graph
                    # promptly, while weighting makes this exactly the 24-task mean.
                    (weight * (task_loss + args.trust_regularization * trust)).backward()
                    task_losses.extend(per_example_loss.detach().cpu().tolist())
                    trusts.extend([float(trust.detach())] * batch.batch_size)
                    # Retain diagnostics only, never a finished frozen-backbone graph.
                    generated.append({name: value.detach() for name, value in current.items()})
                    # The backbone is frozen but its large activations are kept
                    # until these references disappear, because gradients still
                    # flow through its input to the HyperSpline.
                    del logits, targets, current, per_example_loss, task_loss, trust
                basis_penalty = extra_regularization(model)
                (args.shared_basis_regularization * basis_penalty).backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip))
                optimizer.step()
                break
            except torch.OutOfMemoryError as error:
                # Re-run the *same selected tasks* with a smaller genuine
                # backbone batch.  No optimizer step has occurred, so this is
                # semantically identical to choosing the smaller batch up front.
                optimizer.zero_grad(set_to_none=True)
                backbone.clear_cache()
                # ``batches`` are cheap input tensors; discard every
                # diagnostic/activation reference before emptying the cache.
                del batches, task_losses, trusts, generated
                release_cuda(device)
                if working_backbone_batch_size == 1:
                    raise RuntimeError(
                        "CUDA OOM even at one full table per TabICL microbatch. "
                        "Lower --max-rows or move to a larger GPU slice."
                    ) from error
                previous = working_backbone_batch_size
                working_backbone_batch_size = max(1, working_backbone_batch_size // 2)
                oom_backoffs += 1
                print(
                    f"[{args.arm} seed={model_seed}] CUDA OOM at step={step}; "
                    f"retrying the same meta-batch with max backbone batch "
                    f"{previous}->{working_backbone_batch_size}",
                    flush=True,
                )

        if step == 1 or step % args.log_every == 0:
            task_loss_value = float(np.mean(task_losses))
            trust_value = float(np.mean(trusts))
            weighted_trust = args.trust_regularization * trust_value
            basis_penalty_value = float(basis_penalty.detach())
            row = {
                **run_fields,
                "step": step,
                "meta_batch_episodes": args.meta_batch_episodes,
                "real_episodes": len(selected_real),
                "synthetic_episodes": len(selected_synthetic),
                "real_dataset_identities_in_update": len({episode.dataset for episode in selected_real}),
                "backbone_microbatches": len(batches),
                "max_observed_backbone_batch": max(batch.batch_size for batch in batches),
                "configured_max_backbone_batch": args.max_backbone_batch_size,
                "working_max_backbone_batch": working_backbone_batch_size,
                "oom_backoffs_this_update": oom_backoffs,
                "task_loss": task_loss_value,
                "task_loss_std": float(np.std(task_losses)),
                "trust_region": trust_value,
                "weighted_trust_regularization": weighted_trust,
                "shared_basis_penalty": basis_penalty_value,
                "weighted_shared_basis_regularization": args.shared_basis_regularization * basis_penalty_value,
                "objective": task_loss_value + weighted_trust + args.shared_basis_regularization * basis_penalty_value,
                "trust_regularization_to_task_loss": weighted_trust / max(task_loss_value, 1e-12),
                "pre_clip_gradient_norm": gradient_norm,
                "elapsed_seconds": time.time() - started,
                **parameter_metrics(generated[0]),
                **model.basis_diagnostics(),
            }
            if device.type == "cuda":
                row.update({
                    "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                    "cuda_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                })
            training_log.append(row)
            write_csv(run_dir / "training.csv", training_log)
            print(
                f"[{args.arm} seed={model_seed} train] step={step}/{args.steps} loss={task_loss_value:.6f} "
                f"trust={trust_value:.6g} reg/ce={row['trust_regularization_to_task_loss']:.3e} "
                f"grad={gradient_norm:.5g} microbatch<={row['max_observed_backbone_batch']}",
                flush=True,
            )

        if step % args.validate_every == 0 or step == args.steps:
            model.eval()
            score_by_alpha = {0.0: reference_validation_score}
            current_by_alpha: dict[float, list[dict]] = {}
            for alpha in args.residual_alphas:
                if alpha == 0.0:
                    continue
                rows, _ = evaluation_rows(
                    backbone, model, validation_episodes, stage="validation", step=step, alpha=alpha,
                    batch_size=args.evaluation_batch_size, run_fields=run_fields,
                )
                current_by_alpha[alpha] = rows
                score_by_alpha[alpha] = macro_dataset_mean(rows)
                evaluation_log.extend(rows)
            candidate_alpha, candidate_score = choose_alpha(score_by_alpha)
            improved = candidate_score < best_score
            if improved:
                best_score, best_step, best_alpha, stale = candidate_score, step, candidate_alpha, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            selection_log.append({
                **run_fields,
                "step": step,
                "selected_alpha": candidate_alpha,
                "candidate_validation_loss": candidate_score,
                "reference_validation_loss": reference_validation_score,
                "best_validation_loss": best_score,
                "best_step": best_step,
                "best_alpha": best_alpha,
                "improved": improved,
                "stale_validations": stale,
                **{f"validation_loss_alpha_{alpha:g}": score for alpha, score in sorted(score_by_alpha.items())},
            })
            write_csv(run_dir / "validation_selection.csv", selection_log)
            write_csv(run_dir / "evaluations.csv", evaluation_log)
            torch.save({
                "state_dict": model.state_dict(), "best_state_dict": best_state,
                "optimizer": optimizer.state_dict(), "step": step, "best_step": best_step,
                "best_alpha": best_alpha, "best_validation_loss": best_score, "manifest": manifest,
            }, run_dir / "last.pt")
            print(
                f"[{args.arm} seed={model_seed} validation] step={step} best_alpha={candidate_alpha:g} "
                f"macro_nll={candidate_score:.6f}; selected={best_score:.6f}@{best_step} alpha={best_alpha:g}",
                flush=True,
            )
            if args.patience_validations and stale >= args.patience_validations:
                print(f"[{args.arm} seed={model_seed}] early stopping after {stale} validations", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    if best_alpha == 0.0:
        selected_validation = reference_validation
    else:
        selected_validation, _ = evaluation_rows(
            backbone, model, validation_episodes, stage="selected_validation", step=best_step, alpha=best_alpha,
            batch_size=args.evaluation_batch_size, run_fields=run_fields,
        )
        evaluation_log.extend(selected_validation)
    # This is deliberately the first point at which the final benchmark data
    # is constructed/read.  It cannot affect training, early stopping, or alpha selection.
    final_episodes = [
        build_final_episode(spec, seed, args, device)
        for spec in (NUMERICAL_SPECS + MIXED_SPECS)
        for seed in args.final_seeds
    ]
    reference_final, _ = evaluation_rows(
        backbone, model, final_episodes, stage="reference_final", step=best_step, alpha=None,
        batch_size=args.evaluation_batch_size, run_fields=run_fields,
    )
    final_by_alpha: dict[float, list[dict]] = {0.0: reference_final}
    selected_records: list[dict] = []
    for alpha in args.residual_alphas:
        if alpha == 0.0:
            continue
        rows, records = evaluation_rows(
            backbone, model, final_episodes, stage="final", step=best_step, alpha=alpha,
            batch_size=args.evaluation_batch_size, run_fields=run_fields, collect_parameters=(alpha == best_alpha),
        )
        final_by_alpha[alpha] = rows
        if alpha == best_alpha:
            selected_records = records
    selected_final = final_by_alpha[best_alpha]
    if best_alpha == 0.0:
        selected_records = []
    evaluation_log.extend(reference_final)
    for alpha, rows in final_by_alpha.items():
        if alpha != 0.0:
            evaluation_log.extend(rows)
    write_csv(run_dir / "evaluations.csv", evaluation_log)
    write_csv(run_dir / "selected_parameter_columns.csv", parameter_column_rows(selected_records))

    reference_by_key = {(row["dataset"], int(row["split_seed"])): row for row in reference_final}
    paired = []
    for row in selected_final:
        reference = reference_by_key[(row["dataset"], int(row["split_seed"]))]
        paired.append({
            **run_fields,
            "dataset": row["dataset"], "split_seed": row["split_seed"], "selected_alpha": best_alpha,
            **{f"{field}_delta": float(row[field]) - float(reference[field])
               for field in ("loss", "accuracy", "balanced_accuracy", "brier", "ece")},
        })
    write_csv(run_dir / "paired_final.csv", paired)
    final_scores = {str(alpha): macro_dataset_mean(rows) for alpha, rows in final_by_alpha.items()}
    summary = {
        "status": "complete",
        **run_fields,
        "real_train_dataset_count": len(real_datasets),
        "real_validation_dataset_count": len(validation_datasets),
        "best_step": best_step,
        "selected_alpha": best_alpha,
        "reference_validation_loss": reference_validation_score,
        "selected_validation_loss": macro_dataset_mean(selected_validation),
        "reference_final_loss": macro_dataset_mean(reference_final),
        "selected_final_loss": macro_dataset_mean(selected_final),
        **{f"selected_final_{field}_delta": float(np.mean([row[f"{field}_delta"] for row in paired]))
           for field in ("loss", "accuracy", "balanced_accuracy", "brier", "ece")},
        "selected_final_dataset_seed_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in paired])),
        "final_macro_loss_by_alpha": final_scores,
        "basis_diagnostics": model.basis_diagnostics(),
    }
    torch.save({
        "state_dict": best_state,
        "summary": summary,
        "config": {
            "learned_rank": args.learned_rank, "hidden_dim": args.hidden_dim,
            "coefficient_bound": args.coefficient_bound, "shared_basis_bound": args.shared_basis_bound,
            "shared_basis_init_rms": args.shared_basis_init_rms, "target_aware": args.target_aware,
            "raw_context": args.raw_context,
        },
        "shared_basis": model.basis_components().detach().cpu(),
        "manifest": manifest,
    }, run_dir / "best.pt")
    write_json(summary_path, summary)
    manifest["status"] = "complete"
    manifest["summary"] = summary
    write_json(run_dir / "manifest.json", manifest)
    print(
        f"[{args.arm} seed={model_seed}] complete: selected alpha={best_alpha:g}, "
        f"final macro DeltaNLL={summary['selected_final_loss_delta']:+.6f}", flush=True,
    )
    del model, optimizer, parameters, best_state
    release_cuda(device)
    return summary


def aggregate(output: Path, args: argparse.Namespace) -> dict:
    summaries, paired = [], []
    for path in sorted(output.glob("model_seed_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf8"))
        if summary.get("status") != "complete":
            continue
        summaries.append(summary)
        paired_path = path.with_name("paired_final.csv")
        if paired_path.is_file():
            with paired_path.open(newline="", encoding="utf8") as handle:
                paired.extend(list(csv.DictReader(handle)))
    if not summaries:
        return {"status": "incomplete", "completed_model_seeds": []}
    numeric = []
    for row in paired:
        numeric.append({
            **row,
            "model_seed": int(row["model_seed"]),
            "split_seed": int(row["split_seed"]),
            **{field: float(row[field]) for field in row if field.endswith("_delta")},
        })
    grouped_units: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in numeric:
        grouped_units[(str(row["dataset"]), int(row["split_seed"]), int(row["model_seed"]))].append(row)
    units = []
    for (dataset, seed, model_seed), rows in sorted(grouped_units.items()):
        # A unit normally has one final episode.  Keep the mean so future
        # multi-context final evaluation does not inflate the evidence count.
        units.append({
            "dataset": dataset, "split_seed": seed, "model_seed": model_seed,
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
        })
    write_csv(output / "final_dataset_seed_model_units.csv", units)
    per_dataset = []
    for dataset in sorted({row["dataset"] for row in units}):
        rows = [row for row in units if row["dataset"] == dataset]
        per_dataset.append({
            "dataset": dataset, "runs": len(rows),
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
            "loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in rows])),
        })
    write_csv(output / "per_final_dataset.csv", per_dataset)
    result = {
        "status": "complete" if len(summaries) == len(args.model_seeds) else "partial",
        "arm": args.arm,
        "completed_model_seeds": sorted(summary["model_seed"] for summary in summaries),
        "expected_model_seeds": args.model_seeds,
        "final_units": len(units),
        **{f"macro_{field}": float(np.mean([row[field] for row in units])) for field in (
            "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
        )},
        "dataset_seed_model_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in units])),
        "dataset_mean_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in per_dataset])),
        "per_dataset": per_dataset,
        "per_model_seed": summaries,
    }
    write_json(output / "summary.json", result)
    return result


def add_synthetic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm", "graph_scm", "dummy"), default="mix_scm")
    parser.add_argument("--synthetic-sequence-length", type=int, default=512)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("real_4", "real_all", "real_all_plus_synthetic"), required=True)
    parser.add_argument("--real-train-bank", type=Path, required=True)
    parser.add_argument("--real-validation-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real4-datasets", type=parse_name_csv, default=list(DEFAULT_REAL4_DATASETS))
    parser.add_argument("--model-seeds", type=parse_int_csv, default=parse_int_csv("0,1"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--meta-batch-episodes", type=int, default=24)
    parser.add_argument(
        "--datasets-per-update", type=int, default=3,
        help="Uniformly sampled real identities per update; 3 with a 24-task update yields true batches of up to 8.",
    )
    parser.add_argument(
        "--max-backbone-batch-size", type=int, default=2,
        help="Initial true TabICL microbatch cap; training automatically backs this off after an OOM.",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--synthetic-fraction", type=float, default=0.5)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patience-validations", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--trust-regularization", type=float, default=0.1)
    parser.add_argument("--shared-basis-regularization", type=float, default=1e-3)
    parser.add_argument("--learned-rank", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--coefficient-bound", type=float, default=1.5)
    parser.add_argument("--shared-basis-bound", type=float, default=0.75)
    parser.add_argument("--shared-basis-init-rms", type=float, default=0.12)
    parser.add_argument("--residual-alphas", type=parse_float_csv, default=list(DEFAULT_RESIDUAL_ALPHAS))
    parser.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-seed", type=int, default=73_001)
    parser.add_argument("--final-seeds", type=parse_int_csv, default=parse_int_csv("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    add_synthetic_args(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.meta_batch_episodes <= 0 or args.datasets_per_update <= 0:
        raise ValueError("steps, meta-batch episodes, and datasets per update must be positive")
    if args.max_backbone_batch_size <= 0 or args.evaluation_batch_size <= 0:
        raise ValueError("backbone batch sizes must be positive")
    if args.validate_every <= 0 or args.log_every <= 0 or args.patience_validations < 0:
        raise ValueError("invalid logging/validation configuration")
    if not 0.0 <= args.synthetic_fraction <= 1.0:
        raise ValueError("--synthetic-fraction must be in [0, 1]")
    if args.arm == "real_all_plus_synthetic" and not 0.0 < args.synthetic_fraction < 1.0:
        raise ValueError("mixed arm requires a nonzero, nonunit synthetic fraction")
    if args.arm != "real_all_plus_synthetic" and args.synthetic_fraction != 0.5:
        print("[note] --synthetic-fraction is ignored outside real_all_plus_synthetic", flush=True)
    if not args.residual_alphas or any(not 0.0 <= alpha <= 1.0 for alpha in args.residual_alphas):
        raise ValueError("all --residual-alphas must be in [0, 1]")
    if 0.0 not in args.residual_alphas or 1.0 not in args.residual_alphas:
        raise ValueError("--residual-alphas must include exact identity 0 and full residual 1")
    if min(args.trust_regularization, args.shared_basis_regularization, args.weight_decay) < 0:
        raise ValueError("regularization values must be non-negative")
    if args.learned_rank <= 3 or args.hidden_dim <= 0 or args.coefficient_bound <= 0:
        raise ValueError("--learned-rank must exceed 3; hidden dimension and coefficient bound must be positive")
    if not 0 < args.shared_basis_init_rms < args.shared_basis_bound:
        raise ValueError("shared-basis init RMS must be positive and smaller than its bound")
    if not 0.0 < args.context_fraction < 1.0 or args.synthetic_sequence_length < 4:
        raise ValueError("invalid synthetic context fraction or sequence length")
    if not 0 < args.min_features <= args.max_features or args.max_classes < 2:
        raise ValueError("invalid synthetic feature/class configuration")
    if not 0.0 < args.test_size < 1.0 or args.max_rows < 0:
        raise ValueError("invalid final-evaluation split configuration")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(json.dumps(aggregate(output, args), indent=2), flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        print(f"Running on CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    backbone, _ = load_backbone(args, device)
    if args.max_classes > backbone.max_classes:
        raise ValueError(f"--max-classes={args.max_classes} exceeds frozen backbone maximum {backbone.max_classes}")
    _, train_episodes = load_bank(args.real_train_bank, device=device)
    _, validation_episodes = load_bank(args.real_validation_bank, device=device)
    train_episodes, train_replacements = sanitize_real_bank(train_episodes)
    validation_episodes, validation_replacements = sanitize_real_bank(validation_episodes)
    if not train_episodes or not validation_episodes:
        raise ValueError("both real banks must contain at least one episode")
    if train_replacements or validation_replacements:
        print(
            "[data] repaired non-finite numerical cells using context-only medians: "
            f"train={train_replacements}, validation={validation_replacements}. "
            "Regenerate the banks after this run to persist the repair.",
            flush=True,
        )
    # Seed the CPU native prior once before its stochastic per-update draws.
    seed_generator(args.train_seed)
    for model_seed in args.model_seeds:
        run_model_seed(
            args, backbone=backbone, train_episodes=train_episodes, validation_episodes=validation_episodes,
            model_seed=model_seed, output=output, device=device,
        )
    print(json.dumps(aggregate(output, args), indent=2), flush=True)


if __name__ == "__main__":
    main()
