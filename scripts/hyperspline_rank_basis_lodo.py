"""Dataset-disjoint HyperSpline ablation: teacher dictionary vs learned dictionary.

For every leave-one-dataset-out (LODO) fold, the held-out dataset is absent
from basis construction, HyperSpline training, and checkpoint selection.  It
only supplies the labelled context at final evaluation, exactly as it would at
zero-shot inference.

Two output spaces are compared under the same direct-amplitude conditioner:

``teacher_rank``
    A rank basis obtained by PCA over DirectSpline curves from meta-train
    datasets only.  Teacher curves are a *training-time dictionary*, never
    targets and never used on the held-out dataset.
``learned_shared_rank``
    A teacher-free shared dictionary.  Smooth zero-endpoint curve directions
    start from a deterministic Fourier-like basis and are learned jointly with
    the context-to-amplitude HyperSpline from frozen-TabICL cross entropy.

The experiment therefore distinguishes transfer of the conditioner from the
need for an offline DirectSpline teacher dictionary.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from scripts.direct_spline_dataset_headroom import release_cuda
    from scripts.direct_spline_function_basis import TeacherBag
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts import hyperspline_rank_basis_seen_bags as seen
    from scripts import hyperspline_rank_basis_stability as stability
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_dataset_headroom import release_cuda
    from direct_spline_function_basis import TeacherBag
    from direct_spline_multidataset_headroom import load_backbone
    import hyperspline_rank_basis_seen_bags as seen
    import hyperspline_rank_basis_stability as stability


class LearnedSharedRankBasisSpline(stability._ContextConditionedTable):
    """Teacher-free direct-amplitude spline with a learned shared dictionary.

    The components are global parameters, not produced per episode.  The
    shared encoder predicts only their per-column amplitudes.  Coefficients
    start at zero, so the model starts at the exact standardization identity;
    nonzero deterministic initial curves ensure the coefficient head receives
    a meaningful gradient at that point.
    """

    def __init__(
        self,
        *,
        rank: int,
        hidden_dim: int,
        coefficient_bound: float,
        basis_bound: float,
        basis_init_rms: float,
        target_aware: bool,
        raw_context: bool,
    ) -> None:
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0 < basis_init_rms < basis_bound:
            raise ValueError("basis_init_rms must lie strictly between zero and basis_bound")
        self.rank = int(rank)
        self.coefficient_bound = float(coefficient_bound)
        self.basis_bound = float(basis_bound)
        super().__init__(
            n_control_points=rank,
            hidden_dim=hidden_dim,
            target_aware=target_aware,
            raw_context=raw_context,
        )
        t = torch.linspace(0.0, 1.0, self.grid.numel())
        # Each mode vanishes at both endpoints, preserving the standardized
        # range anchors.  The modes are only an initialization, not a fixed
        # analytic basis: ``shared_basis_raw`` is fully trainable.
        mode_numbers = torch.arange(1, rank + 1, dtype=t.dtype).unsqueeze(1)
        templates = torch.sin(math.pi * mode_numbers * t.unsqueeze(0))
        templates = templates / templates.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        templates = basis_init_rms * templates
        initial_raw = torch.atanh((templates / basis_bound).clamp(-0.999, 0.999))
        self.shared_basis_raw = nn.Parameter(initial_raw)
        self.register_buffer("basis_window", torch.sin(math.pi * t))
        self.register_buffer("initial_components", self.basis_components().detach().clone())
        # Keep the diagnostic interface of the teacher-basis models.  The mean
        # is exactly zero; components are exposed through ``basis_components``.
        self.register_buffer("mean_curve", torch.zeros_like(self.grid))

    def basis_components(self) -> torch.Tensor:
        return self.basis_bound * torch.tanh(self.shared_basis_raw) * self.basis_window.unsqueeze(0)

    def shared_basis_regularization(self) -> torch.Tensor:
        """Stops an arbitrary dictionary rescaling/drift from hiding in coefficients."""
        return (self.basis_components() - self.initial_components).square().mean()

    def basis_diagnostics(self) -> dict[str, float]:
        current = self.basis_components().detach()
        delta = current - self.initial_components
        smoothness = torch.diff(current, n=2, dim=-1)
        gram = F.normalize(current, dim=-1) @ F.normalize(current, dim=-1).T
        off_diagonal = gram - torch.eye(self.rank, device=gram.device)
        return {
            "shared_basis_rms": float(current.square().mean().sqrt()),
            "shared_basis_abs_max": float(current.abs().max()),
            "shared_basis_delta_rms": float(delta.square().mean().sqrt()),
            "shared_basis_second_difference_rms": float(smoothness.square().mean().sqrt()),
            "shared_basis_off_diagonal_cosine_abs_mean": float(off_diagonal.abs().sum() / max(self.rank * (self.rank - 1), 1)),
        }

    def generated_parameters(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        raw, statistics = self._raw(x, y)
        coefficients = self.coefficient_bound * torch.tanh(raw[..., : self.rank])
        identity = self.grid.view(1, 1, -1)
        values = stability.strictly_increasing(identity + torch.matmul(coefficients, self.basis_components()))
        batch, _, columns = x.shape
        zeros = torch.zeros(batch, columns, device=x.device, dtype=x.dtype)
        return {
            "values": values,
            "location": statistics.location,
            "scale": statistics.scale.clamp_min(1e-6),
            "shape_gate": torch.ones_like(zeros),
            "normalization_gate": zeros,
            "coefficients": coefficients,
            "location_delta": zeros,
            "log_scale_delta": zeros,
        }


def lodo_folds(datasets: set[str]) -> list[tuple[str, set[str]]]:
    """Deterministic held-out dataset / meta-train dataset pairs."""
    if len(datasets) < 2:
        raise ValueError("LODO needs at least two datasets")
    return [(heldout, set(datasets) - {heldout}) for heldout in sorted(datasets)]


def build_model(
    args: argparse.Namespace,
    *,
    mean: torch.Tensor | None,
    components: torch.Tensor | None,
) -> nn.Module:
    if args.arm == "teacher_rank":
        if mean is None or components is None:
            raise ValueError("teacher_rank requires a training-only PCA basis")
        return stability.DirectEffectiveRankBasisSpline(
            mean,
            components,
            hidden_dim=args.hidden_dim,
            coefficient_bound=args.coefficient_bound,
            mean_bound=args.mean_coefficient_bound,
            target_aware=args.target_aware,
            raw_context=args.raw_context,
        )
    return LearnedSharedRankBasisSpline(
        rank=args.learned_rank,
        hidden_dim=args.hidden_dim,
        coefficient_bound=args.coefficient_bound,
        basis_bound=args.shared_basis_bound,
        basis_init_rms=args.shared_basis_init_rms,
        target_aware=args.target_aware,
        raw_context=args.raw_context,
    )


def model_basis_for_oracle(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return a fixed selected dictionary for a deliberately query-label oracle."""
    if isinstance(model, stability.DirectEffectiveRankBasisSpline):
        return model.mean_curve.detach(), model.components.detach(), model.mean_bound
    if isinstance(model, LearnedSharedRankBasisSpline):
        return torch.zeros_like(model.grid), model.basis_components().detach(), 0.0
    raise TypeError(f"unsupported model type: {type(model).__name__}")


def extra_regularization(model: nn.Module) -> torch.Tensor:
    if isinstance(model, LearnedSharedRankBasisSpline):
        return model.shared_basis_regularization()
    return next(nn.Module.parameters(model)).new_zeros(())


def write_run_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf8")


def run_one(
    args: argparse.Namespace,
    backbone,
    all_bags: list[TeacherBag],
    *,
    heldout_dataset: str,
    train_datasets: set[str],
    outer_seed: int,
    model_seed: int,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict, list[dict]]:
    train_bags, cached_validation, cached_test = seen.split_seen_bags(
        all_bags,
        datasets=train_datasets,
        outer_seed=outer_seed,
        train_bag_ids=set(args.train_bags),
        validation_bag_ids=set(args.validation_bags),
        test_bag_ids=set(args.test_bags),
    )
    validation, _, validation_manifest = stability.make_outer_holdout_episode_banks(
        all_bags,
        datasets=train_datasets,
        outer_seed=outer_seed,
        validation_fraction=args.outer_validation_fraction,
        context_fraction=args.evaluation_context_fraction,
        max_context_rows=args.max_evaluation_context_rows,
        split_seed=args.episode_split_seed,
        episodes_per_dataset=args.evaluation_episodes_per_dataset,
    )
    _, final_test, test_manifest = stability.make_outer_holdout_episode_banks(
        all_bags,
        datasets={heldout_dataset},
        outer_seed=outer_seed,
        validation_fraction=args.outer_validation_fraction,
        context_fraction=args.evaluation_context_fraction,
        max_context_rows=args.max_evaluation_context_rows,
        split_seed=args.episode_split_seed,
        episodes_per_dataset=args.evaluation_episodes_per_dataset,
    )
    mean: torch.Tensor | None = None
    components: torch.Tensor | None = None
    explained: list[float] = []
    basis_hash: str | None = None
    if args.arm == "teacher_rank":
        mean, components, explained_tensor, basis_hash = seen.fit_training_basis(train_bags, args.rank)
        explained = explained_tensor.tolist()

    run_fields = {
        "arm": args.arm,
        "heldout_dataset": heldout_dataset,
        "outer_seed_run": outer_seed,
        "model_seed": model_seed,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "protocol": "LODO + balanced_meta_batches + disjoint_outer_holdout_validation_test",
        "run": run_fields,
        "meta_train_datasets": sorted(train_datasets),
        "heldout_dataset": heldout_dataset,
        "train_keys": [seen.bag_key(bag) for bag in train_bags],
        "cached_validation_keys": [seen.bag_key(bag) for bag in cached_validation],
        "cached_test_keys": [seen.bag_key(bag) for bag in cached_test],
        "validation_episode_splits": validation_manifest,
        "heldout_final_episode_splits": test_manifest,
        "teacher_basis_used": args.arm == "teacher_rank",
        "teacher_parameters_used_as_targets": False,
        "basis_sha256": basis_hash,
        "basis_explained_variance": explained,
        "heldout_teacher_curves_used": False,
    }
    write_run_manifest(run_dir / "manifest.json", manifest)

    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        args,
        mean=None if mean is None else mean.to(device),
        components=None if components is None else components.to(device),
    ).to(device)
    parameters = list(nn.Module.parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    fold_offset = sum((index + 1) * ord(character) for index, character in enumerate(heldout_dataset))
    rng = np.random.default_rng(model_seed + 10_000 * outer_seed + 1_000_000 * fold_offset)
    by_dataset = {dataset: [bag for bag in train_bags if bag.dataset == dataset] for dataset in sorted(train_datasets)}

    reference_validation, _ = seen.evaluate_reference(
        backbone, validation, "guard", device, stage="reference_meta_validation", step=0, run_fields=run_fields,
    )
    reference_test, _ = seen.evaluate_reference(
        backbone, final_test, "test", device, stage="reference_heldout_test", step=0, run_fields=run_fields,
    )
    initial_validation, _, _ = seen.evaluate_model(
        backbone, model, validation, "guard", device, stage="initial_model_meta_validation", step=0, run_fields=run_fields,
    )
    training_rows, evaluation_rows = [], reference_validation + reference_test + initial_validation
    best_loss = seen.macro_dataset_mean(initial_validation)
    best_step, stale, best_state = 0, 0, copy.deepcopy(model.state_dict())
    started = time.time()

    for step in range(1, args.steps + 1):
        selected: list[TeacherBag] = []
        for dataset, bags in by_dataset.items():
            indices = rng.choice(
                len(bags),
                size=args.episodes_per_dataset_per_step,
                replace=args.episodes_per_dataset_per_step > len(bags),
            )
            selected.extend(bags[int(index)] for index in np.atleast_1d(indices))
        optimizer.zero_grad(set_to_none=True)
        task_losses, trusts, generated_list = [], [], []
        for bag in selected:
            logits, target, generated, _, _ = seen.model_predictions(backbone, model, bag, "guard", device)
            task_loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
            trust = model.trust_region(generated)
            objective = task_loss + args.regularization * trust
            if args.shared_basis_regularization:
                objective = objective + args.shared_basis_regularization * extra_regularization(model)
            (objective / len(selected)).backward()
            task_losses.append(task_loss.detach())
            trusts.append(trust.detach())
            generated_list.append(generated)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip))
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            task_loss_value = float(torch.stack(task_losses).mean())
            trust_value = float(torch.stack(trusts).mean())
            weighted = args.regularization * trust_value
            basis_penalty = float(extra_regularization(model).detach())
            row = {
                **run_fields,
                "step": step,
                "meta_batch_episodes": len(selected),
                "datasets_per_update": len(by_dataset),
                "task_loss": task_loss_value,
                "task_loss_std": float(torch.stack(task_losses).std(unbiased=False)),
                "trust_region": trust_value,
                "weighted_regularization": weighted,
                "shared_basis_penalty": basis_penalty,
                "weighted_shared_basis_regularization": args.shared_basis_regularization * basis_penalty,
                "objective": task_loss_value + weighted + args.shared_basis_regularization * basis_penalty,
                "regularization_to_task_loss": weighted / max(task_loss_value, 1e-12),
                "pre_clip_gradient_norm": gradient_norm,
                "elapsed_seconds": time.time() - started,
                **stability.mean_parameter_metrics(generated_list),
            }
            if isinstance(model, LearnedSharedRankBasisSpline):
                row.update(model.basis_diagnostics())
            if args.gradient_diagnostics_every and step % args.gradient_diagnostics_every == 0:
                row.update(stability.gradient_alignment(backbone, model, by_dataset, parameters, rng, device))
            if device.type == "cuda":
                row.update({
                    "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                    "cuda_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                })
            training_rows.append(row)
            seen.write_csv(run_dir / "training.csv", training_rows)
            print(
                f"[{args.arm} heldout={heldout_dataset} outer={outer_seed} model={model_seed} train] "
                f"step={step}/{args.steps} loss={task_loss_value:.6f} grad={gradient_norm:.6g} "
                f"reg/ce={row['regularization_to_task_loss']:.3e}",
                flush=True,
            )

        if step % args.validate_every == 0 or step == args.steps:
            current, _, _ = seen.evaluate_model(
                backbone, model, validation, "guard", device,
                stage="meta_validation", step=step, run_fields=run_fields,
            )
            evaluation_rows.extend(current)
            score = seen.macro_dataset_mean(current)
            if score < best_loss:
                best_loss, best_step, stale, best_state = score, step, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            seen.write_csv(run_dir / "evaluations.csv", evaluation_rows)
            torch.save(
                {
                    "state_dict": model.state_dict(), "best_state_dict": best_state,
                    "optimizer": optimizer.state_dict(), "step": step, "best_step": best_step,
                    "best_validation_loss": best_loss, "manifest": manifest,
                },
                run_dir / "last.pt",
            )
            print(
                f"[{args.arm} heldout={heldout_dataset} outer={outer_seed} model={model_seed} validation] "
                f"step={step} macro_loss={score:.6f} best={best_loss:.6f}@{best_step}",
                flush=True,
            )
            if args.patience_validations and stale >= args.patience_validations:
                print(f"[{args.arm} heldout={heldout_dataset} outer={outer_seed} model={model_seed}] early stopping", flush=True)
                break

    model.load_state_dict(best_state)
    selected_validation, _, _ = seen.evaluate_model(
        backbone, model, validation, "guard", device,
        stage="selected_meta_validation", step=best_step, run_fields=run_fields,
    )
    selected_test, _, records = seen.evaluate_model(
        backbone, model, final_test, "test", device,
        stage="selected_heldout_test", step=best_step, run_fields=run_fields, collect=True,
    )
    evaluation_rows.extend(selected_validation + selected_test)
    seen.write_csv(run_dir / "evaluations.csv", evaluation_rows)
    paired = seen.paired_rows(reference_test, selected_test, run_fields=run_fields)
    seen.write_csv(run_dir / "paired_heldout_test.csv", paired)
    seen.write_csv(run_dir / "selected_parameter_columns.csv", seen.parameter_column_rows(records, run_fields=run_fields))

    oracle_rows: list[dict] = []
    if args.oracle_steps > 0:
        oracle_mean, oracle_components, oracle_mean_bound = model_basis_for_oracle(model)
        diagnostic_episodes = [episode for episode in final_test if episode.bag == -1000]
        oracle_rows = stability.rank_oracle_rows(
            backbone,
            diagnostic_episodes,
            mean=oracle_mean,
            components=oracle_components,
            coefficient_bound=args.coefficient_bound,
            mean_bound=oracle_mean_bound,
            steps=args.oracle_steps,
            lr=args.oracle_lr,
            device=device,
            outer_seed=outer_seed,
        )
        for row in oracle_rows:
            row.update(run_fields)
        seen.write_csv(run_dir / "query_label_oracle.csv", oracle_rows)

    selected_by_reference = seen.macro_dataset_mean(selected_validation) < seen.macro_dataset_mean(reference_validation)
    summary = {
        **run_fields,
        "meta_train_datasets": sorted(train_datasets),
        "best_step": best_step,
        "reference_meta_validation_loss": seen.macro_dataset_mean(reference_validation),
        "initial_model_meta_validation_loss": seen.macro_dataset_mean(initial_validation),
        "selected_meta_validation_loss": seen.macro_dataset_mean(selected_validation),
        "selected_by_reference_guard": selected_by_reference,
        "reference_heldout_test_loss": seen.macro_dataset_mean(reference_test),
        "selected_heldout_test_loss": seen.macro_dataset_mean(selected_test),
        **{f"heldout_test_{field}": float(np.mean([row[f"{field}_delta"] for row in paired]))
           for field in ("loss", "accuracy", "balanced_accuracy", "brier", "ece")},
        "heldout_episode_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in paired])),
        "guarded_heldout_test_loss_delta": float(np.mean([row["loss_delta"] for row in paired])) if selected_by_reference else 0.0,
        "oracle_heldout_loss_delta": float(np.mean([row["loss_delta"] for row in oracle_rows])) if oracle_rows else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    if isinstance(model, LearnedSharedRankBasisSpline):
        saved_basis = model.basis_components().detach().cpu()
        saved_mean = torch.zeros_like(saved_basis[0])
    else:
        saved_basis = model.components.detach().cpu()
        saved_mean = model.mean_curve.detach().cpu()
    torch.save(
        {"state_dict": best_state, "mean": saved_mean, "components": saved_basis,
         "summary": summary, "manifest": manifest},
        run_dir / "best.pt",
    )
    manifest["status"], manifest["summary"] = "complete", summary
    write_run_manifest(run_dir / "manifest.json", manifest)
    del model, optimizer, parameters, best_state
    release_cuda(device)
    return summary, paired


def load_completed_runs(output: Path) -> tuple[list[dict], list[dict]]:
    summaries, paired = [], []
    for path in sorted(output.glob("heldout_*/*/model_seed_*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf8")))
        paired.extend(seen.read_csv(path.with_name("paired_heldout_test.csv")))
    return summaries, paired


def aggregate(output: Path, args: argparse.Namespace, datasets: set[str]) -> dict:
    summaries, paired = load_completed_runs(output)
    expected = len(datasets) * len(args.outer_seeds) * len(args.model_seeds)
    if len(summaries) != expected:
        raise ValueError(f"cannot summarize: found {len(summaries)} completed runs, expected {expected}")
    numeric = [{**row, **{key: float(row[key]) for key in row if key.endswith("_delta")}} for row in paired]
    units: list[dict] = []
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in numeric:
        grouped[(str(row["heldout_dataset"]), int(row["outer_seed_run"]), int(row["model_seed"]))].append(row)
    for (dataset, outer_seed, model_seed), rows in sorted(grouped.items()):
        units.append({
            "heldout_dataset": dataset,
            "outer_seed": outer_seed,
            "model_seed": model_seed,
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
        })
    seen.write_csv(output / "heldout_dataset_seed_units.csv", units)
    per_dataset = []
    for dataset in sorted(datasets):
        rows = [row for row in units if row["heldout_dataset"] == dataset]
        per_dataset.append({
            "heldout_dataset": dataset,
            "runs": len(rows),
            **{field: float(np.mean([row[field] for row in rows])) for field in (
                "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
            )},
            "loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in rows])),
        })
    seen.write_csv(output / "per_heldout_dataset.csv", per_dataset)
    aggregate_result = {
        "arm": args.arm,
        "datasets": sorted(datasets),
        "outer_seeds": args.outer_seeds,
        "model_seeds": args.model_seeds,
        "runs": len(summaries),
        "heldout_dataset_seed_units": len(units),
        **{f"macro_{field}": float(np.mean([row[field] for row in units])) for field in (
            "loss_delta", "accuracy_delta", "balanced_accuracy_delta", "brier_delta", "ece_delta"
        )},
        "dataset_seed_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in units])),
        "heldout_dataset_mean_loss_win_fraction": float(np.mean([row["loss_delta"] < 0 for row in per_dataset])),
        "guard_selected_fraction": float(np.mean([bool(row["selected_by_reference_guard"]) for row in summaries])),
        "mean_guarded_loss_delta": float(np.mean([float(row["guarded_heldout_test_loss_delta"]) for row in summaries])),
        "mean_query_label_oracle_loss_delta": float(np.mean([
            float(row["oracle_heldout_loss_delta"])
            for row in summaries if row["oracle_heldout_loss_delta"] is not None
        ])) if any(row["oracle_heldout_loss_delta"] is not None for row in summaries) else None,
        "per_heldout_dataset": per_dataset,
        "per_run": summaries,
    }
    (output / "summary.json").write_text(json.dumps(aggregate_result, indent=2), encoding="utf8")
    return aggregate_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True, help="Episode cache; the teacher-free arm does not read curves.")
    parser.add_argument("--dataset", action="append", default=[], help="Repeat; default uses every cached dataset.")
    parser.add_argument("--arm", choices=("teacher_rank", "learned_shared_rank"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-seeds", type=seen.parse_int_csv, default=seen.parse_int_csv("0,1"))
    parser.add_argument("--model-seeds", type=seen.parse_int_csv, default=seen.parse_int_csv("0,1,2"))
    parser.add_argument(
        "--worker-model-seeds", type=seen.parse_int_csv, default=None,
        help="Optional subset for one parallel worker.  --model-seeds remains the complete expected plan.",
    )
    parser.add_argument("--train-bags", type=seen.parse_int_csv, default=seen.parse_int_csv("0,1,2,3,4,5"))
    parser.add_argument("--validation-bags", type=seen.parse_int_csv, default=seen.parse_int_csv("6"))
    parser.add_argument("--test-bags", type=seen.parse_int_csv, default=seen.parse_int_csv("7"))
    parser.add_argument("--outer-validation-fraction", type=float, default=0.40)
    parser.add_argument("--evaluation-context-fraction", type=float, default=0.50)
    parser.add_argument("--max-evaluation-context-rows", type=int, default=512)
    parser.add_argument("--evaluation-episodes-per-dataset", type=int, default=8)
    parser.add_argument("--episode-split-seed", type=int, default=70_001)
    parser.add_argument("--rank", type=int, default=8, help="Teacher-PCA component count.")
    parser.add_argument("--learned-rank", type=int, default=9, help="Teacher-free shared component count.")
    parser.add_argument("--shared-basis-bound", type=float, default=0.75)
    parser.add_argument("--shared-basis-init-rms", type=float, default=0.12)
    parser.add_argument("--shared-basis-regularization", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1_250)
    parser.add_argument("--episodes-per-dataset-per-step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--coefficient-bound", type=float, default=1.5)
    parser.add_argument("--mean-coefficient-bound", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostics-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patience-validations", type=int, default=20)
    parser.add_argument("--oracle-steps", type=int, default=250, help="Query-label diagnostic only; set zero to skip.")
    parser.add_argument("--oracle-lr", type=float, default=0.03)
    parser.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Aggregate completed parallel workers without loading TabICL.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.outer_validation_fraction < 1 or not 0 < args.evaluation_context_fraction < 1:
        raise ValueError("validation and context fractions must be strictly between zero and one")
    if args.episodes_per_dataset_per_step <= 0 or args.evaluation_episodes_per_dataset <= 0:
        raise ValueError("episode counts must be positive")
    if args.oracle_steps < 0 or args.shared_basis_regularization < 0:
        raise ValueError("regularization weights and oracle steps must be nonnegative")
    if args.worker_model_seeds is not None and not set(args.worker_model_seeds) <= set(args.model_seeds):
        raise ValueError("worker-model-seeds must be a subset of model-seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.teacher_cache.resolve(), map_location="cpu", weights_only=False)
    all_bags: list[TeacherBag] = payload["bags"]
    available = {bag.dataset for bag in all_bags}
    datasets = set(args.dataset) if args.dataset else available
    if missing := datasets - available:
        raise ValueError(f"datasets absent from cache: {sorted(missing)}")
    if missing := set(args.outer_seeds) - {int(bag.seed) for bag in all_bags if bag.dataset in datasets}:
        raise ValueError(f"outer seeds absent from cache: {sorted(missing)}")

    top_manifest = {
        "status": "running",
        "args": {**vars(args), "teacher_cache": str(args.teacher_cache), "output_dir": str(output)},
        "datasets": sorted(datasets),
        "cache_config": payload.get("config"),
        "protocol": "dataset-disjoint leave-one-dataset-out",
        "teacher_basis_used": args.arm == "teacher_rank",
        "teacher_parameters_used_as_targets": False,
        "heldout_teacher_curves_used": False,
    }
    write_run_manifest(output / "manifest.json", top_manifest)
    if args.summarize_only:
        result = aggregate(output, args, datasets)
        top_manifest["status"], top_manifest["summary"] = "complete", result
        write_run_manifest(output / "manifest.json", top_manifest)
        print(json.dumps(result, indent=2), flush=True)
        return

    device = torch.device(args.device)
    backbone, _ = load_backbone(args, device)
    worker_model_seeds = args.worker_model_seeds or args.model_seeds
    for heldout_dataset, train_datasets in lodo_folds(datasets):
        for outer_seed in args.outer_seeds:
            for model_seed in worker_model_seeds:
                run_dir = output / f"heldout_{heldout_dataset}" / f"outer_seed_{outer_seed}" / f"model_seed_{model_seed}"
                if args.resume and (run_dir / "summary.json").exists():
                    print(f"Reusing completed run: {run_dir}", flush=True)
                    continue
                run_one(
                    args,
                    backbone,
                    all_bags,
                    heldout_dataset=heldout_dataset,
                    train_datasets=train_datasets,
                    outer_seed=outer_seed,
                    model_seed=model_seed,
                    device=device,
                    run_dir=run_dir,
                )
    # A one-process run can aggregate immediately.  Parallel workers should
    # invoke --summarize-only once all workers finish.
    summaries, _ = load_completed_runs(output)
    expected = len(datasets) * len(args.outer_seeds) * len(args.model_seeds)
    if len(summaries) == expected:
        result = aggregate(output, args, datasets)
        top_manifest["status"], top_manifest["summary"] = "complete", result
        write_run_manifest(output / "manifest.json", top_manifest)
        print(json.dumps(result, indent=2), flush=True)
    else:
        print(f"Worker completed {len(summaries)}/{expected} runs; run --summarize-only after the other workers finish.", flush=True)


if __name__ == "__main__":
    main()
