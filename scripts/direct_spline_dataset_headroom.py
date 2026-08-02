"""Bounded-memory, train-only DirectSpline headroom on PMLB datasets.

This is the diagnostic immediately before returning to HyperSpline or
zero-shot transfer.  DirectSpline learns one set of per-numerical-column
spline parameters for a bag, rather than asking a hypernetwork to generate
them.  Thus a positive outer-test result demonstrates genuine, within-dataset
preprocessing headroom; a null result says that a more complex HyperSpline is
unlikely to solve the problem by itself.

For each dataset and outer seed:

* hold out an untouched outer test split;
* form nested stratified bags in the outer training partition;
* use one fold as a never-trained-on guard;
* take a bounded, fixed support context from each fit fold;
* repeatedly resample smaller support contexts and labelled fit queries while
  fitting DirectSpline; and
* select identity versus DirectSpline using guard log loss, then evaluate the
  already-fixed choices on the outer test set; and
* measure how much DirectSpline parameter error can be tolerated through
  interpolation, controlled perturbations, cross-bag transfer, and consensus.

All TabICL calls are bounded by ``max_context_rows + query_chunk_rows``.
The CSV has bag-level post-hoc diagnostics, including whether the guard made
the right decision on the untouched test set.  Those diagnostics never affect
selection.  Progress is saved after every completed dataset/seed run.
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

try:  # Support both package-style imports and ``python scripts/file.py``.
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import load_pmlb_frame
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import load_pmlb_frame

from tabicl._hyperspline import DirectSplineTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True,
                        help="PMLB classification dataset; repeat for several datasets.")
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--outer-test-size", type=float, default=0.20)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--max-context-rows", type=int, default=512,
                        help="Maximum fixed TabICL support rows per bag.")
    parser.add_argument("--train-context-rows", type=int, default=384,
                        help="Support rows sampled per DirectSpline update.")
    parser.add_argument("--query-batch-rows", type=int, default=256,
                        help="Labelled fit-query rows sampled per DirectSpline update.")
    parser.add_argument("--evaluation-query-chunk-rows", type=int, default=256,
                        help="Maximum guard/test rows per TabICL evaluation call.")
    parser.add_argument("--steps", type=int, default=1_250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--transform-regularization", type=float, default=0.0)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--freeze-spline-shape", action="store_true",
                        help="Keep spline controls and nonlinear gate at identity; adapt only enabled basis freedoms.")
    parser.add_argument("--trainable-range", action="store_true",
                        help="Learn bounded per-column spline ranges instead of fixed R=4.")
    parser.add_argument("--trainable-location-scale", action="store_true",
                        help="Learn bounded residual location/scale adjustments.")
    parser.add_argument("--basis-variant", default="uniform_fixed",
                        help="Label written to outputs; use one output directory per variant.")
    parser.add_argument("--output-fold-csv", type=Path, required=True)
    parser.add_argument("--output-margin-csv", type=Path, default=None,
                        help="Per-bag interpolation/perturbation/transfer diagnostics. Defaults beside fold CSV.")
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--interpolation-alphas", default="0,0.25,0.5,0.75,1",
                        help="Comma-separated deformation strengths in [0, 1].")
    parser.add_argument("--perturbation-scales", default="0.05,0.10,0.25",
                        help="Comma-separated RMS fractions of the learned parameter displacement.")
    parser.add_argument("--perturbation-repeats", type=int, default=2)
    parser.add_argument("--cross-bag-max-sources", type=int, default=3,
                        help="Maximum other teachers evaluated on each target bag; 0 evaluates all.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed dataset/seed runs found in the summary JSON.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float_list(value: str, *, name: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must be non-empty")
    return values


def release_cuda(device: torch.device) -> None:
    """Release objects between bags/runs; row budgets remain the main guard."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def stratified_subset(y: np.ndarray, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Return a reproducible class-stratified subset without replacement."""
    if n_rows <= 0:
        raise ValueError("subset size must be positive")
    if n_rows >= y.size:
        return rng.permutation(y.size)
    labels, counts = np.unique(y, return_counts=True)
    if n_rows < labels.size:
        raise ValueError("subset size must contain at least one row per class")
    ideal = counts.astype(np.float64) * n_rows / y.size
    take = np.floor(ideal).astype(int)
    take = np.maximum(take, 1)
    take = np.minimum(take, counts)
    while take.sum() > n_rows:
        candidates = np.where(take > 1)[0]
        if not candidates.size:
            raise ValueError("unable to make a class-preserving subset")
        index = candidates[np.argmax(take[candidates] - ideal[candidates])]
        take[index] -= 1
    while take.sum() < n_rows:
        candidates = np.where(take < counts)[0]
        index = candidates[np.argmax(ideal[candidates] - take[candidates])]
        take[index] += 1
    parts = []
    for label, count in zip(labels, take, strict=True):
        rows = np.flatnonzero(y == label)
        parts.append(rng.permutation(rows)[:count])
    return rng.permutation(np.concatenate(parts))


def to_device(x: np.ndarray, y: np.ndarray, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(x[rows], dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(y[rows], dtype=torch.float32, device=device).unsqueeze(0),
    )


def make_direct_spline(x_context: torch.Tensor, args: argparse.Namespace) -> DirectSplineTransform:
    return DirectSplineTransform(
        x_context,
        args.n_control_points,
        trainable_shape=not args.freeze_spline_shape,
        trainable_range=args.trainable_range,
        trainable_location_scale=args.trainable_location_scale,
    )


def transform_logits(
    backbone,
    spline: DirectSplineTransform,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    backbone.clear_cache()
    transformed = torch.cat((spline.transform(x_context), spline.transform(x_query)), dim=1)
    return backbone(transformed, y_context)


def train_loss(
    backbone,
    spline: DirectSplineTransform,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    logits = transform_logits(backbone, spline, x_context, y_context, x_query)
    loss = F.cross_entropy(logits.flatten(0, 1), y_query.long().flatten())
    if regularization:
        # DirectSpline uses tanh-bounded log gap adjustments.  Penalizing their
        # magnitude is an identity-relative trust region without changing the
        # output parameterization.
        loss = loss + regularization * (spline.gap_logits.tanh().square().mean() + spline.gate_logits.sigmoid().square().mean())
    return loss


@torch.no_grad()
def evaluate_chunked(
    backbone,
    spline: DirectSplineTransform,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query_np: np.ndarray,
    y_query_np: np.ndarray,
    *,
    chunk_rows: int,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    """Evaluate arbitrary query length with bounded GPU memory; return CPU probabilities."""
    transformed_context = spline.transform(x_context)
    probabilities: list[torch.Tensor] = []
    for start in range(0, y_query_np.size, chunk_rows):
        end = min(start + chunk_rows, y_query_np.size)
        query = torch.as_tensor(x_query_np[start:end], dtype=torch.float32, device=device).unsqueeze(0)
        backbone.clear_cache()
        logits = backbone(torch.cat((transformed_context, spline.transform(query)), dim=1), y_context)
        probabilities.append(logits.softmax(dim=-1).cpu())
        del query, logits
    probability = torch.cat(probabilities, dim=1).clamp_min(1e-12)
    labels = torch.as_tensor(y_query_np, dtype=torch.long)
    loss = F.nll_loss(probability.log().flatten(0, 1), labels.flatten())
    accuracy = (probability.argmax(dim=-1).flatten() == labels).float().mean()
    return float(loss), float(accuracy), probability


def spline_diagnostics(spline: DirectSplineTransform) -> tuple[float, float, float]:
    with torch.no_grad():
        parameters = spline.parameters_for_transform()
        identity = torch.linspace(-1.0, 1.0, parameters.control_points.shape[-1], device=parameters.control_points.device)
        control_displacement = (parameters.control_points - identity).abs().mean()
        return float(parameters.gate.mean()), float(parameters.gate.max()), float(control_displacement)


def clone_with_shape(
    identity: DirectSplineTransform,
    *,
    gap_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    location_offsets: torch.Tensor | None = None,
    log_scale_offsets: torch.Tensor | None = None,
    range_logits: torch.Tensor | None = None,
) -> DirectSplineTransform:
    """Use a target bag's normalization with a complete learned spline state.

    Offsets are dimensionless, so transferring them deliberately applies the
    same bounded residual in the target bag's native location/scale units.
    ``None`` preserves the target identity value for a non-adaptive basis.
    """
    candidate = copy.deepcopy(identity).eval()
    with torch.no_grad():
        candidate.gap_logits.copy_(gap_logits.to(candidate.gap_logits))
        candidate.gate_logits.copy_(gate_logits.to(candidate.gate_logits))
        if location_offsets is not None:
            candidate.location_offsets.copy_(location_offsets.to(candidate.location_offsets))
        if log_scale_offsets is not None:
            candidate.log_scale_offsets.copy_(log_scale_offsets.to(candidate.log_scale_offsets))
        if range_logits is not None:
            candidate.range_logits.copy_(range_logits.to(candidate.range_logits))
    return candidate


def interpolated_shape(identity: DirectSplineTransform, teacher: DirectSplineTransform, alpha: float) -> DirectSplineTransform:
    """Interpolate every learned basis residual; endpoints reproduce both models."""
    with torch.no_grad():
        if alpha == 0.0:
            return copy.deepcopy(identity).eval()
        if alpha == 1.0:
            return clone_with_shape(
                identity, gap_logits=teacher.gap_logits, gate_logits=teacher.gate_logits,
                location_offsets=teacher.location_offsets,
                log_scale_offsets=teacher.log_scale_offsets,
                range_logits=teacher.range_logits,
            )
        teacher_parameters = teacher.parameters_for_transform()
        gate = (alpha * teacher_parameters.gate).clamp(1e-6, 1.0 - 1e-6)
        return clone_with_shape(
            identity,
            gap_logits=identity.gap_logits + alpha * (teacher.gap_logits - identity.gap_logits),
            gate_logits=torch.logit(gate),
            location_offsets=identity.location_offsets + alpha * (teacher.location_offsets - identity.location_offsets),
            log_scale_offsets=identity.log_scale_offsets + alpha * (teacher.log_scale_offsets - identity.log_scale_offsets),
            range_logits=identity.range_logits + alpha * (teacher.range_logits - identity.range_logits),
        )


def perturb_shape(
    identity: DirectSplineTransform,
    teacher: DirectSplineTransform,
    scale: float,
    generator: torch.Generator,
) -> DirectSplineTransform:
    """Add normalized independent parameter noise around the learned teacher."""
    with torch.no_grad():
        gap_delta = teacher.gap_logits - identity.gap_logits
        gate_delta = teacher.gate_logits - identity.gate_logits
        location_delta = teacher.location_offsets - identity.location_offsets
        log_scale_delta = teacher.log_scale_offsets - identity.log_scale_offsets
        range_delta = teacher.range_logits - identity.range_logits

        def noise_like(delta: torch.Tensor) -> torch.Tensor:
            rms = delta.square().mean().sqrt().clamp_min(1e-6)
            return torch.randn(delta.shape, generator=generator, device="cpu", dtype=delta.dtype).to(delta) * rms * scale

        return clone_with_shape(
            identity,
            gap_logits=teacher.gap_logits + noise_like(gap_delta),
            gate_logits=teacher.gate_logits + noise_like(gate_delta),
            location_offsets=teacher.location_offsets + noise_like(location_delta),
            log_scale_offsets=teacher.log_scale_offsets + noise_like(log_scale_delta),
            range_logits=teacher.range_logits + noise_like(range_delta),
        )


@torch.no_grad()
def functional_relative_error(
    candidate: DirectSplineTransform,
    teacher: DirectSplineTransform,
    identity: DirectSplineTransform,
    support_x: torch.Tensor,
) -> float:
    """RMS functional error, normalized by the teacher's non-identity deformation."""
    identity_output = identity.transform(support_x)
    teacher_deformation = (teacher.transform(support_x) - identity_output).square().mean().sqrt()
    candidate_error = (candidate.transform(support_x) - teacher.transform(support_x)).square().mean().sqrt()
    return float(candidate_error / teacher_deformation.clamp_min(1e-6))


def ensemble_metrics(probabilities: list[torch.Tensor], labels: np.ndarray) -> tuple[float, float]:
    probability = torch.stack(probabilities).mean(dim=0).clamp_min(1e-12)
    target = torch.as_tensor(labels, dtype=torch.long)
    loss = F.nll_loss(probability.log().flatten(0, 1), target.flatten())
    accuracy = (probability.argmax(dim=-1).flatten() == target).float().mean()
    return float(loss), float(accuracy)


def evaluate_margin_candidate(
    backbone,
    candidate: DirectSplineTransform,
    teacher: DirectSplineTransform,
    identity: DirectSplineTransform,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    identity_loss: float,
    identity_accuracy: float,
    teacher_loss: float,
    teacher_accuracy: float,
    chunk_rows: int,
    device: torch.device,
    dataset: str,
    seed: int,
    target_bag: int,
    condition: str,
    alpha: float | None = None,
    perturbation_scale: float | None = None,
    perturbation_repeat: int | None = None,
    source_bag: int | None = None,
) -> dict[str, object]:
    loss, accuracy, _ = evaluate_chunked(
        backbone, candidate, support_x, support_y, x_test, y_test,
        chunk_rows=chunk_rows, device=device,
    )
    available_headroom = identity_loss - teacher_loss
    recovery = float("nan") if available_headroom <= 1e-8 else (identity_loss - loss) / available_headroom
    gate_mean, gate_max, displacement = spline_diagnostics(candidate)
    return {
        "dataset": dataset, "outer_seed": seed, "target_bag": target_bag,
        "source_bag": source_bag, "condition": condition, "alpha": alpha,
        "perturbation_scale": perturbation_scale, "perturbation_repeat": perturbation_repeat,
        "identity_outer_test_loss": identity_loss, "teacher_outer_test_loss": teacher_loss,
        "candidate_outer_test_loss": loss, "candidate_minus_identity_loss": loss - identity_loss,
        "headroom_recovery_loss": recovery,
        "identity_outer_test_accuracy": identity_accuracy, "teacher_outer_test_accuracy": teacher_accuracy,
        "candidate_outer_test_accuracy": accuracy, "candidate_minus_identity_accuracy": accuracy - identity_accuracy,
        "functional_relative_error": functional_relative_error(candidate, teacher, identity, support_x),
        "candidate_mean_gate": gate_mean, "candidate_max_gate": gate_max,
        "candidate_mean_abs_control_displacement": displacement,
    }


def margin_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate only finite recovery measurements, preserving diagnostic strata."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["condition"], row["alpha"], row["perturbation_scale"])
        groups.setdefault(key, []).append(row)
    output = []
    for (condition, alpha, scale), group in sorted(groups.items(), key=lambda item: str(item[0])):
        recovery = np.asarray([float(row["headroom_recovery_loss"]) for row in group], dtype=float)
        recovery = recovery[np.isfinite(recovery)]
        output.append({
            "condition": condition, "alpha": alpha, "perturbation_scale": scale,
            "n": len(group), "mean_headroom_recovery_loss": float(recovery.mean()) if recovery.size else None,
            "fraction_better_than_identity": float(np.mean([float(row["candidate_minus_identity_loss"]) < 0.0 for row in group])),
            "mean_candidate_minus_identity_loss": float(np.mean([float(row["candidate_minus_identity_loss"]) for row in group])),
            "mean_functional_relative_error": float(np.mean([float(row["functional_relative_error"]) for row in group])),
            "mean_candidate_minus_identity_accuracy": float(np.mean([float(row["candidate_minus_identity_accuracy"]) for row in group])),
        })
    return output


def run_one_protocol(
    args: argparse.Namespace,
    backbone,
    *,
    dataset: str,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    x, y, _ = load_pmlb_frame(dataset, cache_dir=args.pmlb_cache_dir)
    _, y = np.unique(y, return_inverse=True)
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2 or labels.size > backbone.max_classes or counts.min() < args.bags:
        raise ValueError("dataset must have 2..max_classes labels and enough rows per class for all bags")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.outer_test_size, random_state=seed, stratify=y
    )
    bagger = StratifiedKFold(n_splits=args.bags, shuffle=True, random_state=seed + 1)
    rows: list[dict[str, object]] = []
    identity_probabilities: list[torch.Tensor] = []
    adapted_probabilities: list[torch.Tensor] = []
    guarded_probabilities: list[torch.Tensor] = []
    margin_rows: list[dict[str, object]] = []
    bag_artifacts: list[dict[str, object]] = []

    for bag, (fit_indices, guard_indices) in enumerate(bagger.split(x_train, y_train)):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fit_x, fit_y = x_train[fit_indices], y_train[fit_indices]
        guard_x, guard_y = x_train[guard_indices], y_train[guard_indices]
        n_classes = np.unique(fit_y).size
        # Keep an independent labelled query pool.  On small datasets this
        # backs off from the cap so every class can remain in both pools.
        support_cap = min(args.max_context_rows, max(n_classes, fit_y.size - n_classes))
        support_size = min(support_cap, max(n_classes, int(round(fit_y.size * 0.50))))
        split_rng = np.random.default_rng(seed + 10_000 + bag)
        support_rows = stratified_subset(fit_y, support_size, split_rng)
        query_mask = np.ones(fit_y.size, dtype=bool)
        query_mask[support_rows] = False
        adaptation_x, adaptation_y = fit_x[query_mask], fit_y[query_mask]
        if adaptation_y.size < n_classes or np.unique(adaptation_y).size != n_classes:
            raise ValueError("fit fold could not leave a labelled adaptation query pool")
        support_x, support_y = to_device(fit_x, fit_y, support_rows, device)
        identity = make_direct_spline(support_x, args).to(device).eval()
        spline = copy.deepcopy(identity).train()
        optimizer = torch.optim.Adam(spline.parameters(), lr=args.lr)
        sample_rng = np.random.default_rng(seed + 100_000 + bag)
        initial_loss = final_loss = float("nan")
        for step in range(1, args.steps + 1):
            context_rows = stratified_subset(
                support_y.squeeze(0).detach().cpu().numpy(), min(args.train_context_rows, support_y.shape[1]), sample_rng
            )
            query_rows = stratified_subset(adaptation_y, min(args.query_batch_rows, adaptation_y.size), sample_rng)
            context_x = support_x[:, context_rows]
            context_y = support_y[:, context_rows]
            query_x, query_y = to_device(adaptation_x, adaptation_y, query_rows, device)
            optimizer.zero_grad(set_to_none=True)
            objective = train_loss(
                backbone, spline, context_x, context_y, query_x, query_y, args.transform_regularization
            )
            if step == 1:
                initial_loss = float(objective.detach())
            objective.backward()
            torch.nn.utils.clip_grad_norm_(spline.parameters(), 1.0)
            optimizer.step()
            final_loss = float(objective.detach())
            del context_x, context_y, query_x, query_y, objective
        spline.eval()
        identity_guard_loss, identity_guard_accuracy, _ = evaluate_chunked(
            backbone, identity, support_x, support_y, guard_x, guard_y,
            chunk_rows=args.evaluation_query_chunk_rows, device=device,
        )
        adapted_guard_loss, adapted_guard_accuracy, _ = evaluate_chunked(
            backbone, spline, support_x, support_y, guard_x, guard_y,
            chunk_rows=args.evaluation_query_chunk_rows, device=device,
        )
        use_adapted = adapted_guard_loss < identity_guard_loss
        identity_test_loss, identity_test_accuracy, identity_test_probability = evaluate_chunked(
            backbone, identity, support_x, support_y, x_test, y_test,
            chunk_rows=args.evaluation_query_chunk_rows, device=device,
        )
        adapted_test_loss, adapted_test_accuracy, adapted_test_probability = evaluate_chunked(
            backbone, spline, support_x, support_y, x_test, y_test,
            chunk_rows=args.evaluation_query_chunk_rows, device=device,
        )
        gate_mean, gate_max, control_displacement = spline_diagnostics(spline)
        test_loss_delta = adapted_test_loss - identity_test_loss
        test_accuracy_delta = adapted_test_accuracy - identity_test_accuracy
        test_prefers_adapted_loss = test_loss_delta < 0.0
        identity_probabilities.append(identity_test_probability)
        adapted_probabilities.append(adapted_test_probability)
        guarded_probabilities.append(adapted_test_probability if use_adapted else identity_test_probability)
        # Interpolation asks whether a student merely needs the correct
        # direction/magnitude, rather than an exact reproduction of controls.
        for alpha in args.interpolation_values:
            candidate = copy.deepcopy(identity) if alpha == 0.0 else interpolated_shape(identity, spline, alpha)
            margin_rows.append(evaluate_margin_candidate(
                backbone, candidate, spline, identity, support_x, support_y, x_test, y_test,
                identity_loss=identity_test_loss, identity_accuracy=identity_test_accuracy,
                teacher_loss=adapted_test_loss, teacher_accuracy=adapted_test_accuracy,
                chunk_rows=args.evaluation_query_chunk_rows, device=device,
                dataset=dataset, seed=seed, target_bag=bag, condition="interpolation", alpha=alpha,
            ))
            del candidate
        # Parameter noise is normalized independently for every learned raw
        # parameter block by its teacher displacement from identity.  The
        # recorded functional error is the meaningful common scale across
        # controls, gates, and adaptive basis parameters.
        for scale in args.perturbation_values:
            for repeat in range(args.perturbation_repeats):
                generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000 + 10_000 * bag + 100 * repeat)
                candidate = perturb_shape(identity, spline, scale, generator)
                margin_rows.append(evaluate_margin_candidate(
                    backbone, candidate, spline, identity, support_x, support_y, x_test, y_test,
                    identity_loss=identity_test_loss, identity_accuracy=identity_test_accuracy,
                    teacher_loss=adapted_test_loss, teacher_accuracy=adapted_test_accuracy,
                    chunk_rows=args.evaluation_query_chunk_rows, device=device,
                    dataset=dataset, seed=seed, target_bag=bag, condition="parameter_perturbation",
                    perturbation_scale=scale, perturbation_repeat=repeat,
                ))
                del candidate, generator
        # Preserve only CPU state.  This enables cross-bag tests after every
        # teacher has been trained without retaining GPU allocations.
        bag_artifacts.append({
            "bag": bag,
            "support_x": support_x.detach().cpu(), "support_y": support_y.detach().cpu(),
            "teacher_gap_logits": spline.gap_logits.detach().cpu().clone(),
            "teacher_gate_logits": spline.gate_logits.detach().cpu().clone(),
            "teacher_location_offsets": spline.location_offsets.detach().cpu().clone(),
            "teacher_log_scale_offsets": spline.log_scale_offsets.detach().cpu().clone(),
            "teacher_range_logits": spline.range_logits.detach().cpu().clone(),
            "identity_test_loss": identity_test_loss, "identity_test_accuracy": identity_test_accuracy,
            "teacher_test_loss": adapted_test_loss, "teacher_test_accuracy": adapted_test_accuracy,
        })
        peak_allocated_gib = peak_reserved_gib = 0.0
        if device.type == "cuda":
            peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
        rows.append({
            "basis_variant": args.basis_variant, "n_control_points": args.n_control_points,
            "trainable_shape": not args.freeze_spline_shape, "trainable_range": args.trainable_range,
            "trainable_location_scale": args.trainable_location_scale,
            "dataset": dataset, "outer_seed": seed, "bag": bag,
            "fit_rows": int(fit_y.size), "support_context_rows": int(support_y.shape[1]),
            "adaptation_query_pool_rows": int(adaptation_y.size), "guard_rows": int(guard_y.size),
            "outer_test_rows": int(y_test.size), "train_context_rows_per_step": min(args.train_context_rows, support_y.shape[1]),
            "query_batch_rows_per_step": min(args.query_batch_rows, adaptation_y.size),
            "initial_train_objective": initial_loss, "final_train_objective": final_loss,
            "identity_guard_loss": identity_guard_loss, "adapted_guard_loss": adapted_guard_loss,
            "identity_guard_accuracy": identity_guard_accuracy, "adapted_guard_accuracy": adapted_guard_accuracy,
            "guard_loss_delta": adapted_guard_loss - identity_guard_loss,
            "guard_accuracy_delta": adapted_guard_accuracy - identity_guard_accuracy,
            "identity_guard_selected_adapted": use_adapted,
            "identity_outer_test_loss": identity_test_loss, "adapted_outer_test_loss": adapted_test_loss,
            "identity_outer_test_accuracy": identity_test_accuracy, "adapted_outer_test_accuracy": adapted_test_accuracy,
            "outer_test_loss_delta": test_loss_delta, "outer_test_accuracy_delta": test_accuracy_delta,
            "outer_test_prefers_adapted_loss": test_prefers_adapted_loss,
            "outer_test_prefers_adapted_accuracy": test_accuracy_delta > 0.0,
            "guard_correct_for_outer_test_loss": use_adapted == test_prefers_adapted_loss,
            "guard_false_positive_loss": use_adapted and not test_prefers_adapted_loss,
            "guard_false_negative_loss": not use_adapted and test_prefers_adapted_loss,
            "adapted_mean_gate": gate_mean, "adapted_max_gate": gate_max,
            "adapted_mean_abs_control_displacement": control_displacement,
            "peak_allocated_gib": peak_allocated_gib, "peak_reserved_gib": peak_reserved_gib,
        })
        print(
            f"[{dataset} seed={seed} bag={bag}] train={initial_loss:.4f}->{final_loss:.4f}; "
            f"guard identity={identity_guard_loss:.4f}, adapted={adapted_guard_loss:.4f}; "
            f"test_delta={test_loss_delta:+.5f}; selected={'adapted' if use_adapted else 'identity'}; "
            f"peak={peak_allocated_gib:.2f} GiB",
            flush=True,
        )
        del optimizer, spline, identity, support_x, support_y, identity_test_probability, adapted_test_probability
        release_cuda(device)

    # A source teacher's complete spline state is transplanted into the target
    # support normalization.  This tests whether its learned bounded residual
    # policy is stable across independent samples from the same dataset.
    for target in bag_artifacts:
        target_x, target_y = target["support_x"].to(device), target["support_y"].to(device)
        target_identity = make_direct_spline(target_x, args).to(device).eval()
        target_teacher = clone_with_shape(
            target_identity,
            gap_logits=target["teacher_gap_logits"], gate_logits=target["teacher_gate_logits"],
            location_offsets=target["teacher_location_offsets"],
            log_scale_offsets=target["teacher_log_scale_offsets"],
            range_logits=target["teacher_range_logits"],
        )
        all_sources = [source for source in bag_artifacts if source["bag"] != target["bag"]]
        transfer_sources = all_sources
        if args.cross_bag_max_sources > 0 and len(transfer_sources) > args.cross_bag_max_sources:
            selected = np.linspace(0, len(transfer_sources) - 1, args.cross_bag_max_sources, dtype=int)
            transfer_sources = [transfer_sources[index] for index in selected]
        for source in transfer_sources:
            candidate = clone_with_shape(
                target_identity,
                gap_logits=source["teacher_gap_logits"], gate_logits=source["teacher_gate_logits"],
                location_offsets=source["teacher_location_offsets"],
                log_scale_offsets=source["teacher_log_scale_offsets"],
                range_logits=source["teacher_range_logits"],
            )
            margin_rows.append(evaluate_margin_candidate(
                backbone, candidate, target_teacher, target_identity, target_x, target_y, x_test, y_test,
                identity_loss=float(target["identity_test_loss"]), identity_accuracy=float(target["identity_test_accuracy"]),
                teacher_loss=float(target["teacher_test_loss"]), teacher_accuracy=float(target["teacher_test_accuracy"]),
                chunk_rows=args.evaluation_query_chunk_rows, device=device,
                dataset=dataset, seed=seed, target_bag=int(target["bag"]), condition="cross_bag_transfer",
                source_bag=int(source["bag"]),
            ))
            del candidate
        # Leave-one-bag-out consensus has no access to the target teacher.  It
        # is a simple, non-neural proxy for an amortized dataset-level policy.
        consensus_gap = torch.stack([source["teacher_gap_logits"] for source in all_sources]).mean(dim=0)
        consensus_gate = torch.stack([source["teacher_gate_logits"] for source in all_sources]).mean(dim=0)
        consensus_location = torch.stack([source["teacher_location_offsets"] for source in all_sources]).mean(dim=0)
        consensus_log_scale = torch.stack([source["teacher_log_scale_offsets"] for source in all_sources]).mean(dim=0)
        consensus_range = torch.stack([source["teacher_range_logits"] for source in all_sources]).mean(dim=0)
        candidate = clone_with_shape(
            target_identity, gap_logits=consensus_gap, gate_logits=consensus_gate,
            location_offsets=consensus_location, log_scale_offsets=consensus_log_scale,
            range_logits=consensus_range,
        )
        margin_rows.append(evaluate_margin_candidate(
            backbone, candidate, target_teacher, target_identity, target_x, target_y, x_test, y_test,
            identity_loss=float(target["identity_test_loss"]), identity_accuracy=float(target["identity_test_accuracy"]),
            teacher_loss=float(target["teacher_test_loss"]), teacher_accuracy=float(target["teacher_test_accuracy"]),
            chunk_rows=args.evaluation_query_chunk_rows, device=device,
            dataset=dataset, seed=seed, target_bag=int(target["bag"]), condition="leave_one_bag_out_consensus",
        ))
        del candidate, target_teacher, target_identity, target_x, target_y
        release_cuda(device)

    identity_loss, identity_accuracy = ensemble_metrics(identity_probabilities, y_test)
    adapted_loss, adapted_accuracy = ensemble_metrics(adapted_probabilities, y_test)
    guarded_loss, guarded_accuracy = ensemble_metrics(guarded_probabilities, y_test)
    oracle_probability = [
        adapted_probabilities[index] if bool(row["outer_test_prefers_adapted_loss"]) else identity_probabilities[index]
        for index, row in enumerate(rows)
    ]
    oracle_loss, oracle_accuracy = ensemble_metrics(oracle_probability, y_test)
    summary = {
        "basis_variant": args.basis_variant, "n_control_points": args.n_control_points,
        "trainable_shape": not args.freeze_spline_shape, "trainable_range": args.trainable_range,
        "trainable_location_scale": args.trainable_location_scale,
        "dataset": dataset, "outer_seed": seed, "outer_train_rows": int(y_train.size),
        "outer_test_rows": int(y_test.size), "n_features": int(x.shape[1]), "n_classes": int(labels.size),
        "bags": args.bags, "steps_per_bag": args.steps,
        "identity_outer_test_loss": identity_loss, "identity_outer_test_accuracy": identity_accuracy,
        "adapted_outer_test_loss": adapted_loss, "adapted_outer_test_accuracy": adapted_accuracy,
        "guarded_outer_test_loss": guarded_loss, "guarded_outer_test_accuracy": guarded_accuracy,
        "oracle_per_bag_outer_test_loss": oracle_loss, "oracle_per_bag_outer_test_accuracy": oracle_accuracy,
        "adapted_minus_identity_loss": adapted_loss - identity_loss,
        "adapted_minus_identity_accuracy": adapted_accuracy - identity_accuracy,
        "guarded_minus_identity_loss": guarded_loss - identity_loss,
        "guarded_minus_identity_accuracy": guarded_accuracy - identity_accuracy,
        "oracle_per_bag_minus_identity_loss": oracle_loss - identity_loss,
        "oracle_per_bag_minus_identity_accuracy": oracle_accuracy - identity_accuracy,
        "guard_selected_adapted_fraction": float(np.mean([bool(row["identity_guard_selected_adapted"]) for row in rows])),
        "guard_correct_fraction_for_outer_test_loss": float(np.mean([bool(row["guard_correct_for_outer_test_loss"]) for row in rows])),
        "guard_false_positive_fraction_loss": float(np.mean([bool(row["guard_false_positive_loss"]) for row in rows])),
        "guard_false_negative_fraction_loss": float(np.mean([bool(row["guard_false_negative_loss"]) for row in rows])),
        "mean_adapted_gate": float(np.mean([float(row["adapted_mean_gate"]) for row in rows])),
        "mean_peak_allocated_gib": float(np.mean([float(row["peak_allocated_gib"]) for row in rows])),
        "protocol": "outer_holdout + nested_bagging + fixed_support_context + resampled_train_only_episodes + external_identity_guard",
    }
    del identity_probabilities, adapted_probabilities, guarded_probabilities, oracle_probability
    release_cuda(device)
    return rows, summary, margin_rows


def save_progress(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    margin_rows: list[dict[str, object]],
) -> None:
    write_csv(args.output_fold_csv, rows)
    write_csv(args.output_margin_csv, margin_rows)
    macro_names = (
        "adapted_minus_identity_loss", "adapted_minus_identity_accuracy",
        "guarded_minus_identity_loss", "guarded_minus_identity_accuracy",
        "oracle_per_bag_minus_identity_loss", "guard_correct_fraction_for_outer_test_loss",
        "guard_false_positive_fraction_loss", "guard_false_negative_fraction_loss",
    )
    macro = {name: float(np.mean([float(item[name]) for item in summaries])) for name in macro_names} if summaries else {}
    payload = {
        "runs": summaries, "macro_mean_over_dataset_seed_runs": macro,
        "n_runs": len(summaries), "n_bag_decisions": len(rows),
        "margin_summary": margin_summary(margin_rows), "n_margin_evaluations": len(margin_rows),
        "protocol": "oracle_per_bag fields are intentionally post-hoc diagnostics and never used for selection",
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_margin_csv = args.output_margin_csv or args.output_fold_csv.with_name(
        f"{args.output_fold_csv.stem}_margin.csv"
    )
    args.interpolation_values = parse_float_list(args.interpolation_alphas, name="--interpolation-alphas")
    args.perturbation_values = parse_float_list(args.perturbation_scales, name="--perturbation-scales")
    if not 0 < args.outer_test_size < 1 or args.bags < 2:
        raise ValueError("outer-test-size must lie in (0, 1) and bags must be at least two")
    if min(args.max_context_rows, args.train_context_rows, args.query_batch_rows,
           args.evaluation_query_chunk_rows, args.steps) <= 0:
        raise ValueError("all row budgets and steps must be positive")
    if args.n_control_points <= 3 or args.lr <= 0 or args.transform_regularization < 0:
        raise ValueError("invalid spline or optimization configuration")
    if args.freeze_spline_shape and not args.trainable_location_scale:
        raise ValueError("--freeze-spline-shape requires --trainable-location-scale; range alone has no effect on identity shape")
    if any(value < 0.0 or value > 1.0 for value in args.interpolation_values):
        raise ValueError("--interpolation-alphas must lie in [0, 1]")
    if any(value < 0.0 for value in args.perturbation_values) or args.perturbation_repeats <= 0:
        raise ValueError("perturbation scales must be non-negative and repeats must be positive")
    if len(set(args.pmlb_dataset)) != len(args.pmlb_dataset) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("datasets and seeds must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    backbone, _ = load_backbone(args, device)
    all_rows = read_csv(args.output_fold_csv) if args.resume else []
    all_margin_rows = read_csv(args.output_margin_csv) if args.resume else []
    existing_summaries: list[dict[str, object]] = []
    if args.resume and args.output_summary_json.is_file():
        existing_summaries = list(json.loads(args.output_summary_json.read_text(encoding="utf-8")).get("runs", []))
    completed = {(str(item["dataset"]), int(item["outer_seed"])) for item in existing_summaries}
    for dataset in args.pmlb_dataset:
        for seed in args.seeds:
            if (dataset, seed) in completed:
                print(f"[{dataset} seed={seed}] already complete; skipping due to --resume", flush=True)
                continue
            rows, summary, margin_rows = run_one_protocol(args, backbone, dataset=dataset, seed=seed, device=device)
            all_rows.extend(rows)
            all_margin_rows.extend(margin_rows)
            existing_summaries.append(summary)
            save_progress(args, all_rows, existing_summaries, all_margin_rows)
            print(json.dumps(summary, indent=2), flush=True)
    save_progress(args, all_rows, existing_summaries, all_margin_rows)
    print(
        f"Wrote {len(all_rows)} bag records, {len(all_margin_rows)} margin records, and "
        f"{len(existing_summaries)} run summaries.",
        flush=True,
    )


if __name__ == "__main__":
    main()
