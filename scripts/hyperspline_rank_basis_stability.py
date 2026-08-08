"""Stability ablation for shared, context-conditioned numerical transforms.

The experiment holds the seen-dataset protocol fixed while changing two things
that can make the conditioner brittle: single-episode SGD becomes balanced
meta-batches, and the shape decoder can be compared with two zero-identity,
ungated alternatives.  The three arms are:

``gated_rank``
    The existing teacher-PCA rank basis with a separate shape gate.
``direct_rank``
    The teacher-PCA mean plus components with directly effective amplitudes.
``uniform_spline``
    A teacher-free, fixed-knot cubic B-spline whose monotone control-point
    gaps are generated directly.

Teachers are used only to construct the fixed PCA dictionary for the rank
arms.  They are never parameter targets and are never consulted at inference.
``uniform_spline`` is the corresponding teacher-free control.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn

from tabicl._hyperspline import HyperSplineTransform, summarize_context
from tabicl._hyperspline.bspline import evaluate_bspline, greville_abscissae, uniform_augmented_knots

try:
    from scripts.direct_spline_function_basis import TeacherBag
    from scripts.direct_spline_dataset_headroom import release_cuda
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_rank_basis_zero_shot import FactorizedRankBasisSpline, metrics, parameter_metrics
    from scripts import hyperspline_rank_basis_seen_bags as seen
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_function_basis import TeacherBag
    from direct_spline_dataset_headroom import release_cuda
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_rank_basis_zero_shot import FactorizedRankBasisSpline, metrics, parameter_metrics
    import hyperspline_rank_basis_seen_bags as seen


GRID_SIZE = 129
STANDARDIZED_RANGE = 4.0


def strictly_increasing(values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Project sampled values to a strictly increasing curve, differentiably."""
    increments = torch.diff(values, dim=-1)
    increments = increments + F.softplus(-increments / eps) * eps
    return torch.cat((values[..., :1], values[..., :1] + increments.cumsum(-1)), dim=-1)


def interpolate_table(
    x: torch.Tensor,
    *,
    grid: torch.Tensor,
    values: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply a per-column table, retaining an identity-slope extrapolation."""
    z = (x.float() - location.unsqueeze(1)) / scale.unsqueeze(1).clamp_min(1e-6)
    clipped = z.clamp(float(grid[0]), float(grid[-1]))
    position = (clipped - grid[0]) / (grid[-1] - grid[0]) * (grid.numel() - 1)
    left = position.floor().long().clamp(0, grid.numel() - 2)
    fraction = (position - left).to(x.dtype)
    table = values.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
    low = table.gather(3, left.unsqueeze(-1)).squeeze(-1)
    high = table.gather(3, (left + 1).unsqueeze(-1)).squeeze(-1)
    return low + fraction * (high - low) + z - clipped


class _ContextConditionedTable(nn.Module):
    """Shared context encoder plus common table application machinery."""

    def __init__(
        self,
        *,
        n_control_points: int,
        hidden_dim: int,
        target_aware: bool,
        raw_context: bool,
        grid_size: int = GRID_SIZE,
        standardized_range: float = STANDARDIZED_RANGE,
    ) -> None:
        super().__init__()
        self.standardized_range = standardized_range
        self.register_buffer("grid", torch.linspace(-standardized_range, standardized_range, grid_size))
        self.encoder = HyperSplineTransform(
            n_control_points=n_control_points,
            hidden_dim=hidden_dim,
            target_aware=target_aware,
            raw_context_residual=raw_context,
            raw_context_num_heads=4,
            generate_location=True,
            generate_scale=True,
            # The native gate is not part of these ungated output arms.
            gate_initial_probability=0.5,
            raw_context_gate_initial_probability=0.01,
        )
        # HyperSplineTransform initializes its native gate logit at an output
        # index that is an amplitude for direct_rank.  Resetting this head makes
        # raw=0 and therefore the transformed table exactly identity.
        nn.init.zeros_(self.encoder.mlp[-1].weight)
        nn.init.zeros_(self.encoder.mlp[-1].bias)

    @property
    def target_aware(self) -> bool:
        return self.encoder.target_aware

    def _raw(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, object]:
        labels = y.float().unsqueeze(0) if self.target_aware else None
        statistics = summarize_context(x, y_context=labels)
        raw, _ = self.encoder.generate_raw(statistics, x_context=x, y_context=labels)
        return raw, statistics

    def transform(self, x: torch.Tensor, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        return interpolate_table(
            x,
            grid=self.grid,
            values=parameters["values"],
            location=parameters["location"],
            scale=parameters["scale"],
        )

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor):
        parameters = self.generated_parameters(context_x, context_y)
        return torch.cat((self.transform(context_x, parameters), self.transform(query_x, parameters)), dim=1), parameters

    def trust_region(self, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        return (parameters["values"] - self.grid.view(1, 1, -1)).square().mean()


class DirectEffectiveRankBasisSpline(_ContextConditionedTable):
    """Teacher-PCA basis with one directly effective amplitude per basis curve.

    The mean curve is basis element zero.  Thus the all-zero initialized output
    is identity without relying on a multiplicative gate.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        *,
        hidden_dim: int,
        coefficient_bound: float,
        mean_bound: float,
        target_aware: bool,
        raw_context: bool,
    ) -> None:
        self.rank = int(components.shape[0])
        if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != mean.numel():
            raise ValueError("mean and components must have shapes (G,) and (R, G)")
        super().__init__(
            n_control_points=self.rank + 1,
            hidden_dim=hidden_dim,
            target_aware=target_aware,
            raw_context=raw_context,
            grid_size=mean.numel(),
        )
        self.coefficient_bound = coefficient_bound
        self.mean_bound = mean_bound
        self.register_buffer("mean_curve", mean)
        self.register_buffer("components", components)

    def generated_parameters(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        raw, statistics = self._raw(x, y)
        mean_amplitude = self.mean_bound * torch.tanh(raw[..., :1])
        component_amplitudes = self.coefficient_bound * torch.tanh(raw[..., 1 : self.rank + 1])
        identity = self.grid.view(1, 1, -1)
        values = identity + mean_amplitude * self.mean_curve.view(1, 1, -1)
        values = values + torch.matmul(component_amplitudes, self.components)
        values = strictly_increasing(values)
        batch, _, columns = x.shape
        zeros = torch.zeros(batch, columns, device=x.device, dtype=x.dtype)
        return {
            "values": values,
            "location": statistics.location,
            "scale": statistics.scale.clamp_min(1e-6),
            "shape_gate": torch.ones_like(zeros),
            "normalization_gate": zeros,
            "coefficients": torch.cat((mean_amplitude, component_amplitudes), dim=-1),
            "location_delta": zeros,
            "log_scale_delta": zeros,
        }


class UniformControlSpline(_ContextConditionedTable):
    """Teacher-free cubic spline with generated monotone control-point gaps."""

    def __init__(
        self,
        *,
        n_control_points: int,
        hidden_dim: int,
        gap_adjustment_bound: float,
        target_aware: bool,
        raw_context: bool,
    ) -> None:
        if n_control_points <= 3:
            raise ValueError("uniform_spline needs more control points than cubic degree")
        super().__init__(
            n_control_points=n_control_points,
            hidden_dim=hidden_dim,
            target_aware=target_aware,
            raw_context=raw_context,
        )
        self.n_control_points = n_control_points
        self.gap_adjustment_bound = gap_adjustment_bound
        knots = uniform_augmented_knots(n_control_points, degree=3)
        identity_controls = greville_abscissae(knots, degree=3, n_control_points=n_control_points)
        self.register_buffer("knots", knots)
        self.register_buffer("identity_gaps", identity_controls[1:] - identity_controls[:-1])
        # These two buffers make generic curve diagnostics well defined while
        # explicitly representing a rank-zero teacher basis.
        self.register_buffer("mean_curve", torch.zeros_like(self.grid))
        self.register_buffer("components", torch.empty(0, self.grid.numel()))

    def generated_parameters(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        raw, statistics = self._raw(x, y)
        gap_raw = raw[..., : self.n_control_points - 1]
        gaps = self.identity_gaps * torch.exp(self.gap_adjustment_bound * torch.tanh(gap_raw))
        gaps = 2.0 * gaps / gaps.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        controls = torch.cat((torch.full_like(gaps[..., :1], -1.0), -1.0 + gaps.cumsum(dim=-1)), dim=-1)
        batch, _, columns = x.shape
        u = (self.grid / self.standardized_range).view(1, -1, 1).expand(batch, -1, columns)
        spline = evaluate_bspline(u, controls.float(), self.knots, degree=3).transpose(1, 2)
        values = strictly_increasing(self.standardized_range * spline)
        zeros = torch.zeros(batch, columns, device=x.device, dtype=x.dtype)
        return {
            "values": values,
            "location": statistics.location,
            "scale": statistics.scale.clamp_min(1e-6),
            "shape_gate": torch.ones_like(zeros),
            "normalization_gate": zeros,
            "coefficients": torch.tanh(gap_raw),
            "location_delta": zeros,
            "log_scale_delta": zeros,
            "control_points": controls,
        }


def make_outer_holdout_episode_banks(
    cache_bags: list[TeacherBag],
    *,
    datasets: set[str],
    outer_seed: int,
    validation_fraction: float,
    context_fraction: float,
    max_context_rows: int,
    split_seed: int,
    episodes_per_dataset: int,
) -> tuple[list[seen.EvaluationEpisode], list[seen.EvaluationEpisode], list[dict]]:
    """Make several episodes from disjoint validation and final row pools."""
    validation, final, manifest = [], [], []
    for dataset_index, dataset in enumerate(sorted(datasets)):
        items = sorted(
            [bag for bag in cache_bags if bag.dataset == dataset and int(bag.seed) == outer_seed],
            key=seen.bag_key,
        )
        if not items:
            raise ValueError(f"no bags for {dataset}, outer seed {outer_seed}")
        x = np.asarray(items[0].test_x, dtype=np.float32)
        y = np.asarray(items[0].test_y, dtype=np.int64)
        if any(not np.array_equal(x, item.test_x) or not np.array_equal(y, item.test_y) for item in items[1:]):
            raise ValueError(f"outer test differs across bags for {dataset}, seed {outer_seed}")
        row_seed = split_seed + 100_000 * outer_seed + 1_000 * dataset_index
        validation_idx, final_idx = seen._stratified_split_indices(y, 1.0 - validation_fraction, row_seed)
        for episode_index in range(episodes_per_dataset):
            validation.append(seen._make_holdout_episode(
                dataset, outer_seed, -(episode_index + 1), x[validation_idx], y[validation_idx],
                context_fraction=context_fraction, max_context_rows=max_context_rows,
                seed=row_seed + 100 + episode_index, source="outer_holdout_validation",
            ))
            final.append(seen._make_holdout_episode(
                dataset, outer_seed, -(1_000 + episode_index), x[final_idx], y[final_idx],
                context_fraction=context_fraction, max_context_rows=max_context_rows,
                seed=row_seed + 10_000 + episode_index, source="outer_holdout_final",
            ))
        manifest.append({
            "dataset": dataset,
            "outer_seed": outer_seed,
            "outer_rows": int(y.size),
            "validation_pool_rows": int(validation_idx.size),
            "final_pool_rows": int(final_idx.size),
            "episodes_per_dataset": episodes_per_dataset,
            "validation_context_rows": int(validation[ -episodes_per_dataset].support_y.numel()),
            "validation_query_rows": int(validation[ -episodes_per_dataset].guard_y.size),
            "final_context_rows": int(final[ -episodes_per_dataset].support_y.numel()),
            "final_query_rows": int(final[ -episodes_per_dataset].test_y.size),
            "row_split_seed": row_seed,
        })
    return validation, final, manifest


def build_model(args: argparse.Namespace, mean: torch.Tensor, components: torch.Tensor) -> nn.Module:
    if args.arm == "gated_rank":
        return FactorizedRankBasisSpline(
            mean, components, branch="shape", hidden_dim=args.hidden_dim,
            coefficient_bound=args.coefficient_bound, location_bound=0.0,
            log_scale_bound=0.0, target_aware=args.target_aware,
            raw_context=args.raw_context, gate_initial_probability=args.gate_initial_probability,
        )
    if args.arm == "direct_rank":
        return DirectEffectiveRankBasisSpline(
            mean, components, hidden_dim=args.hidden_dim,
            coefficient_bound=args.coefficient_bound, mean_bound=args.mean_coefficient_bound,
            target_aware=args.target_aware, raw_context=args.raw_context,
        )
    return UniformControlSpline(
        n_control_points=args.uniform_control_points, hidden_dim=args.hidden_dim,
        gap_adjustment_bound=args.gap_adjustment_bound,
        target_aware=args.target_aware, raw_context=args.raw_context,
    )


def mean_parameter_metrics(parameters: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    rows = [parameter_metrics(item) for item in parameters]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def gradient_alignment(
    backbone,
    model: nn.Module,
    bags_by_dataset: dict[str, list[TeacherBag]],
    parameters: list[nn.Parameter],
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    """Diagnostic-only pairwise cosine of one fresh episode gradient per dataset."""
    gradients: dict[str, torch.Tensor] = {}
    for dataset, bags in bags_by_dataset.items():
        bag = bags[int(rng.integers(len(bags)))]
        logits, target, _, _, _ = seen.model_predictions(backbone, model, bag, "guard", device)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        pieces = torch.autograd.grad(loss, parameters, allow_unused=True)
        flat = torch.cat([piece.detach().flatten() for piece in pieces if piece is not None])
        gradients[dataset] = flat
    result = {f"gradient_norm_{dataset}": float(vector.norm()) for dataset, vector in gradients.items()}
    for first, second in combinations(sorted(gradients), 2):
        result[f"gradient_cosine_{first}__{second}"] = float(
            F.cosine_similarity(gradients[first], gradients[second], dim=0)
        )
    return result


def rank_oracle_rows(
    backbone,
    episodes: list[seen.EvaluationEpisode],
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    coefficient_bound: float,
    mean_bound: float,
    steps: int,
    lr: float,
    device: torch.device,
    outer_seed: int,
) -> list[dict]:
    """Diagnostic query-label oracle for rank-basis headroom on final episodes.

    This is never used to select a HyperSpline checkpoint.  It deliberately
    optimizes against final query labels and is reported only as an attainable
    headroom reference for the fixed output dictionary.
    """
    if steps <= 0:
        return []
    rows = []
    for episode in episodes:
        context_x, context_y, query_x, target = seen.tensors(episode, "test", device)
        statistics = summarize_context(context_x)
        raw = nn.Parameter(torch.zeros(1, context_x.shape[-1], components.shape[0] + 1, device=device))
        optimizer = torch.optim.AdamW([raw], lr=lr)
        identity = torch.linspace(-STANDARDIZED_RANGE, STANDARDIZED_RANGE, mean.numel(), device=device).view(1, 1, -1)
        for _ in range(steps):
            amplitudes = torch.cat((mean_bound * torch.tanh(raw[..., :1]), coefficient_bound * torch.tanh(raw[..., 1:])), dim=-1)
            values = identity + amplitudes[..., :1] * mean.view(1, 1, -1)
            values = strictly_increasing(values + torch.matmul(amplitudes[..., 1:], components))
            transformed = torch.cat((
                interpolate_table(context_x, grid=identity.flatten(), values=values, location=statistics.location, scale=statistics.scale),
                interpolate_table(query_x, grid=identity.flatten(), values=values, location=statistics.location, scale=statistics.scale),
            ), dim=1)
            backbone.clear_cache()
            logits = backbone(transformed, context_y.float().unsqueeze(0))
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            amplitudes = torch.cat((mean_bound * torch.tanh(raw[..., :1]), coefficient_bound * torch.tanh(raw[..., 1:])), dim=-1)
            values = identity + amplitudes[..., :1] * mean.view(1, 1, -1)
            values = strictly_increasing(values + torch.matmul(amplitudes[..., 1:], components))
            transformed = torch.cat((
                interpolate_table(context_x, grid=identity.flatten(), values=values, location=statistics.location, scale=statistics.scale),
                interpolate_table(query_x, grid=identity.flatten(), values=values, location=statistics.location, scale=statistics.scale),
            ), dim=1)
            backbone.clear_cache()
            probability = backbone(transformed, context_y.float().unsqueeze(0)).softmax(-1).cpu()
            reference_logits, reference_target = seen.reference_predictions(backbone, episode, "test", device)
            reference_probability = reference_logits.softmax(-1).cpu()
            current = metrics(probability, target.cpu())
            reference = metrics(reference_probability, reference_target.cpu())
        rows.append({
            "dataset": episode.dataset, "outer_seed": outer_seed, "bag": episode.bag,
            "oracle_steps": steps, **current,
            **{f"{field}_delta": current[field] - reference[field]
               for field in ("loss", "accuracy", "balanced_accuracy", "brier", "ece")},
        })
    return rows


def parse_int_csv(value: str) -> list[int]:
    return seen.parse_int_csv(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[], help="Repeat; default uses every cache dataset.")
    parser.add_argument("--outer-seeds", type=parse_int_csv, default=parse_int_csv("0,1"))
    parser.add_argument("--model-seeds", type=parse_int_csv, default=parse_int_csv("0,1,2"))
    parser.add_argument("--train-bags", type=parse_int_csv, default=parse_int_csv("0,1,2,3,4,5"))
    parser.add_argument("--validation-bags", type=parse_int_csv, default=parse_int_csv("6"))
    parser.add_argument("--test-bags", type=parse_int_csv, default=parse_int_csv("7"))
    parser.add_argument("--outer-validation-fraction", type=float, default=0.40)
    parser.add_argument("--evaluation-context-fraction", type=float, default=0.50)
    parser.add_argument("--max-evaluation-context-rows", type=int, default=512)
    parser.add_argument("--evaluation-episodes-per-dataset", type=int, default=8)
    parser.add_argument("--episode-split-seed", type=int, default=70_001)
    parser.add_argument("--arm", choices=("gated_rank", "direct_rank", "uniform_spline"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--uniform-control-points", type=int, default=9)
    parser.add_argument("--steps", type=int, default=1_250, help="Optimizer updates; each sees a balanced meta-batch.")
    parser.add_argument("--episodes-per-dataset-per-step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--coefficient-bound", type=float, default=1.5)
    parser.add_argument("--mean-coefficient-bound", type=float, default=1.0)
    parser.add_argument("--gap-adjustment-bound", type=float, default=2.0)
    parser.add_argument("--gate-initial-probability", type=float, default=0.01)
    parser.add_argument(
        "--regularization", type=float, default=1e-2,
        help="Effective-curve trust weight; monitor reg/ce and target roughly 0.1%%-1%%.",
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostics-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patience-validations", type=int, default=20)
    parser.add_argument("--rank-oracle-steps", type=int, default=250)
    parser.add_argument("--rank-oracle-lr", type=float, default=0.03)
    parser.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    return parser.parse_args()


def run_one(
    args: argparse.Namespace,
    backbone,
    all_bags: list[TeacherBag],
    datasets: set[str],
    *,
    outer_seed: int,
    model_seed: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict, list[dict]]:
    train_bags, cached_validation, cached_test = seen.split_seen_bags(
        all_bags, datasets=datasets, outer_seed=outer_seed,
        train_bag_ids=set(args.train_bags), validation_bag_ids=set(args.validation_bags),
        test_bag_ids=set(args.test_bags),
    )
    validation, final_test, episode_manifest = make_outer_holdout_episode_banks(
        all_bags, datasets=datasets, outer_seed=outer_seed,
        validation_fraction=args.outer_validation_fraction,
        context_fraction=args.evaluation_context_fraction,
        max_context_rows=args.max_evaluation_context_rows,
        split_seed=args.episode_split_seed,
        episodes_per_dataset=args.evaluation_episodes_per_dataset,
    )
    mean, components, explained, basis_hash = seen.fit_training_basis(train_bags, args.rank)
    run_fields = {"arm": args.arm, "outer_seed_run": outer_seed, "model_seed": model_seed}
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "protocol": "known_datasets + balanced_meta_batches + disjoint_outer_holdout_validation_test",
        "run": run_fields, "datasets": sorted(datasets),
        "train_keys": [seen.bag_key(bag) for bag in train_bags],
        "cached_validation_keys": [seen.bag_key(bag) for bag in cached_validation],
        "cached_test_keys": [seen.bag_key(bag) for bag in cached_test],
        "episode_splits": episode_manifest,
        "basis_sha256": basis_hash, "basis_explained_variance": explained.tolist(),
        "teacher_basis_used": args.arm != "uniform_spline",
        "teacher_parameters_used_as_targets": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args, mean.to(device), components.to(device)).to(device)
    parameters = list(nn.Module.parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    rng = np.random.default_rng(model_seed + 10_000 * outer_seed)
    by_dataset = {dataset: [bag for bag in train_bags if bag.dataset == dataset] for dataset in sorted(datasets)}
    reference_validation, _ = seen.evaluate_reference(
        backbone, validation, "guard", device, stage="reference_validation", step=0, run_fields=run_fields,
    )
    reference_test, _ = seen.evaluate_reference(
        backbone, final_test, "test", device, stage="reference_test", step=0, run_fields=run_fields,
    )
    initial_validation, _, _ = seen.evaluate_model(
        backbone, model, validation, "guard", device, stage="initial_model_validation", step=0, run_fields=run_fields,
    )
    training_rows, evaluation_rows = [], reference_validation + reference_test + initial_validation
    best_loss = seen.macro_dataset_mean(initial_validation)
    best_step, stale, best_state = 0, 0, copy.deepcopy(model.state_dict())
    started = time.time()
    for step in range(1, args.steps + 1):
        selected = []
        for dataset, bags in by_dataset.items():
            count = args.episodes_per_dataset_per_step
            indices = rng.choice(len(bags), size=count, replace=count > len(bags))
            selected.extend((dataset, bags[int(index)]) for index in np.atleast_1d(indices))
        optimizer.zero_grad(set_to_none=True)
        task_losses, trusts, generated_list = [], [], []
        for _, bag in selected:
            logits, target, generated, _, _ = seen.model_predictions(backbone, model, bag, "guard", device)
            task_loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
            trust = model.trust_region(generated)
            ((task_loss + args.regularization * trust) / len(selected)).backward()
            task_losses.append(task_loss.detach())
            trusts.append(trust.detach())
            generated_list.append(generated)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip))
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            task_loss_value = float(torch.stack(task_losses).mean())
            trust_value = float(torch.stack(trusts).mean())
            weighted = args.regularization * trust_value
            row = {
                **run_fields, "step": step, "meta_batch_episodes": len(selected),
                "datasets_per_update": len(by_dataset), "task_loss": task_loss_value,
                "task_loss_std": float(torch.stack(task_losses).std(unbiased=False)),
                "objective": task_loss_value + weighted, "trust_region": trust_value,
                "weighted_regularization": weighted,
                "regularization_to_task_loss": weighted / max(task_loss_value, 1e-12),
                "pre_clip_gradient_norm": gradient_norm, "elapsed_seconds": time.time() - started,
                **mean_parameter_metrics(generated_list),
            }
            if args.gradient_diagnostics_every and step % args.gradient_diagnostics_every == 0:
                row.update(gradient_alignment(backbone, model, by_dataset, parameters, rng, device))
            if device.type == "cuda":
                row.update({
                    "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                    "cuda_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                })
            training_rows.append(row)
            seen.write_csv(run_dir / "training.csv", training_rows)
            print(
                f"[{args.arm} outer={outer_seed} model={model_seed} train] "
                f"step={step}/{args.steps} loss={task_loss_value:.6f} std={row['task_loss_std']:.6f} "
                f"grad={gradient_norm:.6g} shape_gate={row['shape_gate_mean']:.4f} "
                f"reg/ce={row['regularization_to_task_loss']:.3e}", flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            current, _, _ = seen.evaluate_model(
                backbone, model, validation, "guard", device, stage="validation", step=step, run_fields=run_fields,
            )
            evaluation_rows.extend(current)
            score = seen.macro_dataset_mean(current)
            if score < best_loss:
                best_loss, best_step, stale, best_state = score, step, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            per_dataset = {dataset: float(np.mean([row["loss"] for row in current if row["dataset"] == dataset])) for dataset in sorted(datasets)}
            print(
                f"[{args.arm} outer={outer_seed} model={model_seed} validation] step={step} "
                f"macro_loss={score:.6f} best={best_loss:.6f}@{best_step} "
                f"per_dataset={json.dumps(per_dataset, sort_keys=True)}", flush=True,
            )
            seen.write_csv(run_dir / "evaluations.csv", evaluation_rows)
            torch.save({"state_dict": model.state_dict(), "best_state_dict": best_state,
                        "optimizer": optimizer.state_dict(), "step": step, "best_step": best_step,
                        "best_validation_loss": best_loss, "manifest": manifest}, run_dir / "last.pt")
            if args.patience_validations and stale >= args.patience_validations:
                print(f"[{args.arm} outer={outer_seed} model={model_seed}] early stopping", flush=True)
                break

    model.load_state_dict(best_state)
    selected_validation, _, _ = seen.evaluate_model(
        backbone, model, validation, "guard", device, stage="selected_validation", step=best_step, run_fields=run_fields,
    )
    selected_test, _, test_records = seen.evaluate_model(
        backbone, model, final_test, "test", device, stage="selected_test", step=best_step,
        run_fields=run_fields, collect=True,
    )
    selected_train, _, train_records = seen.evaluate_model(
        backbone, model, train_bags, "guard", device, stage="selected_train_bags", step=best_step,
        run_fields=run_fields, collect=True,
    )
    cached, _, _ = seen.evaluate_model(
        backbone, model, cached_test, "test", device, stage="cached_unseen_bag_diagnostic", step=best_step,
        run_fields=run_fields,
    )
    evaluation_rows.extend(selected_validation + selected_test + selected_train + cached)
    seen.write_csv(run_dir / "evaluations.csv", evaluation_rows)
    paired = seen.paired_rows(reference_test, selected_test, run_fields=run_fields)
    seen.write_csv(run_dir / "paired_test.csv", paired)
    consistency = seen.consistency_rows(train_records, test_records, run_fields=run_fields)
    seen.write_csv(run_dir / "parameter_consistency.csv", consistency)
    seen.write_csv(run_dir / "selected_parameter_columns.csv", seen.parameter_column_rows(train_records + test_records, run_fields=run_fields))
    torch.save({"train": train_records, "test": test_records}, run_dir / "selected_parameters.pt")
    per_dataset = []
    for dataset in sorted(datasets):
        rows = [row for row in paired if row["dataset"] == dataset]
        per_dataset.append({
            **run_fields, "dataset": dataset, "n": len(rows),
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
        })
    seen.write_csv(run_dir / "per_dataset.csv", per_dataset)
    selected_by_reference = seen.macro_dataset_mean(selected_validation) < seen.macro_dataset_mean(reference_validation)
    summary = {
        **run_fields, "datasets": sorted(datasets), "best_step": best_step,
        "reference_validation_macro_loss": seen.macro_dataset_mean(reference_validation),
        "initial_model_validation_macro_loss": seen.macro_dataset_mean(initial_validation),
        "selected_validation_macro_loss": seen.macro_dataset_mean(selected_validation),
        "selected_by_reference_guard": selected_by_reference,
        "reference_test_macro_loss": seen.macro_dataset_mean(reference_test),
        "selected_test_macro_loss": seen.macro_dataset_mean(selected_test),
        "test_macro_loss_delta": float(np.mean([row["loss_delta"] for row in paired])),
        "test_macro_accuracy_delta": float(np.mean([row["accuracy_delta"] for row in paired])),
        "dataset_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in per_dataset])),
        "guarded_test_macro_loss_delta": float(np.mean([row["loss_delta"] for row in paired])) if selected_by_reference else 0.0,
        "per_dataset": per_dataset,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    torch.save({"state_dict": best_state, "mean": mean, "components": components,
                "summary": summary, "manifest": manifest}, run_dir / "best.pt")
    manifest["status"], manifest["summary"] = "complete", summary
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    del model, optimizer, parameters, best_state
    release_cuda(device)
    return summary, paired


def main() -> None:
    args = parse_args()
    if not 0 < args.outer_validation_fraction < 1 or not 0 < args.evaluation_context_fraction < 1:
        raise ValueError("validation and context fractions must be strictly between zero and one")
    if args.episodes_per_dataset_per_step <= 0 or args.evaluation_episodes_per_dataset <= 0:
        raise ValueError("episode counts must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.teacher_cache.resolve(), map_location="cpu", weights_only=False)
    all_bags: list[TeacherBag] = payload["bags"]
    available = {bag.dataset for bag in all_bags}
    datasets = set(args.dataset) if args.dataset else available
    if missing := datasets - available:
        raise ValueError(f"datasets absent from teacher cache: {sorted(missing)}")
    available_seeds = {int(bag.seed) for bag in all_bags if bag.dataset in datasets}
    if missing := set(args.outer_seeds) - available_seeds:
        raise ValueError(f"outer seeds absent from cache: {sorted(missing)}")
    top_manifest = {
        "status": "running", "args": {**vars(args), "teacher_cache": str(args.teacher_cache), "output_dir": str(output)},
        "datasets": sorted(datasets), "cache_config": payload.get("config"),
        "teacher_basis_used": args.arm != "uniform_spline",
        "teacher_parameters_used_as_targets": False,
    }
    (output / "manifest.json").write_text(json.dumps(top_manifest, indent=2, default=str), encoding="utf8")
    device = torch.device(args.device)
    backbone, _ = load_backbone(args, device)
    summaries, paired = [], []
    oracle_rows = []
    for outer_seed in args.outer_seeds:
        train_bags, _, _ = seen.split_seen_bags(
            all_bags, datasets=datasets, outer_seed=outer_seed,
            train_bag_ids=set(args.train_bags), validation_bag_ids=set(args.validation_bags), test_bag_ids=set(args.test_bags),
        )
        mean, components, _, _ = seen.fit_training_basis(train_bags, args.rank)
        _, final_episodes, _ = make_outer_holdout_episode_banks(
            all_bags, datasets=datasets, outer_seed=outer_seed,
            validation_fraction=args.outer_validation_fraction, context_fraction=args.evaluation_context_fraction,
            max_context_rows=args.max_evaluation_context_rows, split_seed=args.episode_split_seed,
            episodes_per_dataset=args.evaluation_episodes_per_dataset,
        )
        oracle_path = output / f"rank_oracle_outer_seed_{outer_seed}.csv"
        if args.rank_oracle_steps > 0 and not (args.resume and oracle_path.exists()):
            oracle = rank_oracle_rows(
                backbone, [episode for episode in final_episodes if episode.bag == -1000],
                mean=mean.to(device), components=components.to(device), coefficient_bound=args.coefficient_bound,
                mean_bound=args.mean_coefficient_bound, steps=args.rank_oracle_steps, lr=args.rank_oracle_lr,
                device=device, outer_seed=outer_seed,
            )
            seen.write_csv(oracle_path, oracle)
        oracle_rows.extend(seen.read_csv(oracle_path))
        for model_seed in args.model_seeds:
            run_dir = output / f"outer_seed_{outer_seed}" / f"model_seed_{model_seed}"
            summary_path = run_dir / "summary.json"
            if args.resume and summary_path.exists():
                print(f"Reusing completed run: {run_dir}", flush=True)
                summaries.append(json.loads(summary_path.read_text(encoding="utf8")))
                paired.extend(seen.read_csv(run_dir / "paired_test.csv"))
                continue
            summary, current_paired = run_one(
                args, backbone, all_bags, datasets, outer_seed=outer_seed, model_seed=model_seed,
                device=device, run_dir=run_dir,
            )
            summaries.append(summary); paired.extend(current_paired)
            seen.write_csv(output / "runs.csv", summaries)
            seen.write_csv(output / "paired_test.csv", paired)

    numeric = [{**row, **{key: float(row[key]) for key in row if key.endswith("_delta")}} for row in paired]
    units = []
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in numeric:
        grouped[(row["dataset"], int(row["outer_seed_run"]), int(row["model_seed"]))].append(row)
    for (dataset, outer_seed, model_seed), rows in sorted(grouped.items()):
        units.append({"dataset": dataset, "outer_seed": outer_seed, "model_seed": model_seed,
                      **{field: float(np.mean([row[field] for row in rows])) for field in (
                          "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
                      )}})
    seen.write_csv(output / "dataset_seed_units.csv", units)
    aggregate = {
        "arm": args.arm, "datasets": sorted(datasets), "outer_seeds": args.outer_seeds,
        "model_seeds": args.model_seeds, "runs": len(summaries), "dataset_seed_units": len(units),
        **{f"macro_{field}": float(np.mean([row[field] for row in units])) for field in (
            "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
        )},
        "dataset_seed_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in units])),
        "guard_selected_fraction": float(np.mean([bool(row["selected_by_reference_guard"]) for row in summaries])),
        "mean_guarded_loss_delta": float(np.mean([float(row["guarded_test_macro_loss_delta"]) for row in summaries])),
        "rank_oracle_macro_loss_delta": (
            float(np.mean([float(row["loss_delta"]) for row in oracle_rows])) if oracle_rows else None
        ),
        "per_run": summaries,
    }
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf8")
    top_manifest["status"], top_manifest["summary"] = "complete", aggregate
    (output / "manifest.json").write_text(json.dumps(top_manifest, indent=2, default=str), encoding="utf8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
