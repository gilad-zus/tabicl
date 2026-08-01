"""Dataset-specific HyperSpline headroom with nested 8-fold bagging.

This is deliberately *not* a zero-shot evaluation.  It asks whether a
HyperSpline can be tuned on one dataset's training rows and still improve an
untouched outer test set.  Each bag has three disjoint roles inside the outer
training partition:

* adaptation context: labels condition TabICL and HyperSpline;
* adaptation query: labels provide gradients to tune HyperSpline;
* bag guard: labels choose adapted versus identity, never provide gradients.

The outer test labels are used exactly once, after all eight bag decisions are
fixed.  Bag probabilities are averaged at test time.  The runner can repeat
the complete protocol over multiple PMLB datasets and independent outer-split
seeds; its CSV is deliberately rich enough to audit whether the guard made the
right decision after the fact (without ever using test labels for selection).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from scripts.direct_spline_multidataset_headroom import load_backbone
    from scripts.hyperspline_real_task_bank import load_pmlb_frame
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_multidataset_headroom import load_backbone
    from hyperspline_real_task_bank import load_pmlb_frame

from tabicl._hyperspline import HyperSplineTransform, backbone_state_dict_hash, summarize_context
from tabicl._hyperspline.checkpoint import load_hyperspline_checkpoint


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_episode(x: np.ndarray, y: np.ndarray, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )


def make_identity(config: dict, device: torch.device) -> HyperSplineTransform:
    """The standardization-only HyperSpline used by the identity guard."""
    identity_config = dict(config)
    identity_config.update(target_aware=False, supervised_residual=False, cross_column_residual=False, raw_context_residual=False)
    return HyperSplineTransform(**identity_config).to(device).eval()


def forward_logits(
    backbone,
    hyperspline: HyperSplineTransform,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    """Differentiable HyperSpline → frozen TabICL forward for numerical PMLB tables."""
    with torch.no_grad():
        statistics = summarize_context(
            x_context,
            y_context=y_context.unsqueeze(0).float() if hyperspline.target_aware else None,
            eps=hyperspline.eps,
        )
    y_context_float = y_context.unsqueeze(0).float()
    parameters = hyperspline.generate_parameters(
        statistics,
        x_context=x_context,
        y_context=y_context_float if hyperspline.target_aware else None,
    )
    transformed = torch.cat(
        (hyperspline.apply_transform(x_context, parameters), hyperspline.apply_transform(x_query, parameters)), dim=1
    )
    return backbone(transformed, y_context_float)


@torch.no_grad()
def evaluate(
    backbone,
    hyperspline: HyperSplineTransform,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_query: torch.Tensor,
    y_query: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    backbone.clear_cache()
    logits = forward_logits(backbone, hyperspline, x_context, y_context, x_query)
    loss = F.cross_entropy(logits.flatten(0, 1), y_query.flatten())
    accuracy = (logits.argmax(dim=-1).flatten() == y_query).float().mean()
    return float(loss), float(accuracy), logits.softmax(dim=-1)


def build_tuned_hyperspline(args: argparse.Namespace, backbone, device: torch.device) -> tuple[HyperSplineTransform, dict]:
    if args.initial_hyperspline_checkpoint is not None:
        base, payload = load_hyperspline_checkpoint(
            args.initial_hyperspline_checkpoint,
            device=device,
            expected_backbone_reference=args.checkpoint_version,
            expected_backbone_hash=backbone_state_dict_hash(backbone),
        )
        config = dict(payload["hyperspline_config"])
        if args.raw_context_residual:
            if base.target_aware or base.has_supervised_residual:
                raise ValueError("raw-context headroom initialization requires a marginal-only checkpoint")
            config.update(
                target_aware=True,
                supervised_residual=False,
                cross_column_residual=False,
                raw_context_residual=True,
                raw_context_num_heads=args.raw_context_num_heads,
                raw_context_residual_bound=args.raw_context_residual_bound,
                raw_context_gate_initial_probability=args.raw_context_gate_initial_probability,
            )
            hyperspline = HyperSplineTransform(**config).to(device)
            hyperspline.initialize_supervised_residual_from(base)
        elif args.target_aware and not base.target_aware:
            # Preserve the trained marginal mapping, but allow the MLP to use
            # the eight supervised summary entries during dataset tuning.
            config.update(target_aware=True)
            hyperspline = HyperSplineTransform(**config).to(device)
            hyperspline.mlp.load_state_dict(base.mlp.state_dict())
        else:
            hyperspline = base
        # This is headroom tuning: all HyperSpline parameters, including the
        # copied marginal policy of a residual model, may receive gradients.
        hyperspline.unfreeze_marginal_policy()
        return hyperspline.train(), config
    config = {
        "n_control_points": args.n_control_points,
        "hidden_dim": args.hidden_dim,
        "gate_initial_probability": args.gate_initial_probability,
        "target_aware": args.target_aware,
        "raw_context_residual": args.raw_context_residual,
        "raw_context_num_heads": args.raw_context_num_heads,
        "raw_context_residual_bound": args.raw_context_residual_bound,
        "raw_context_gate_initial_probability": args.raw_context_gate_initial_probability,
    }
    return HyperSplineTransform(**config).to(device).train(), config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pmlb-dataset",
        action="append",
        required=True,
        help="PMLB classification dataset; repeat this option for each dataset.",
    )
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-hyperspline-checkpoint", type=Path, default=None)
    parser.add_argument("--outer-test-size", type=float, default=0.20)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--adaptation-context-fraction", type=float, default=0.70)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transform-regularization", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Independent outer-split seeds (default: 0).")
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gate-initial-probability", type=float, default=0.10)
    parser.add_argument("--target-aware", action="store_true")
    parser.add_argument("--raw-context-residual", action="store_true")
    parser.add_argument("--raw-context-num-heads", type=int, default=4)
    parser.add_argument("--raw-context-residual-bound", type=float, default=0.5)
    parser.add_argument("--raw-context-gate-initial-probability", type=float, default=0.5)
    parser.add_argument("--output-fold-csv", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    return parser.parse_args()


def run_one_protocol(
    args: argparse.Namespace,
    backbone,
    config: dict,
    *,
    dataset: str,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run one fully isolated outer split and return audit rows plus its summary."""
    x, y, _ = load_pmlb_frame(dataset, cache_dir=args.pmlb_cache_dir)
    _, y = np.unique(y, return_inverse=True)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2 or classes.size > 10 or counts.min() < max(args.bags, 3):
        raise ValueError("dataset must have 2..10 classes and enough examples per class for outer/bag splits")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.outer_test_size, random_state=seed, stratify=y
    )
    if classes.size > backbone.max_classes:
        raise ValueError(f"dataset has {classes.size} classes but backbone supports {backbone.max_classes}")
    identity = make_identity(config, device)
    bagger = StratifiedKFold(n_splits=args.bags, shuffle=True, random_state=seed + 1)
    test_x, test_y = tensor_episode(x_test, y_test, device=device)
    identity_probabilities, adapted_probabilities, guarded_probabilities = [], [], []
    rows: list[dict[str, object]] = []

    for bag, (fit_indices, guard_indices) in enumerate(bagger.split(x_train, y_train)):
        fit_x, fit_y = x_train[fit_indices], y_train[fit_indices]
        guard_x, guard_y = x_train[guard_indices], y_train[guard_indices]
        context_x_np, tune_x_np, context_y_np, tune_y_np = train_test_split(
            fit_x, fit_y, train_size=args.adaptation_context_fraction,
            random_state=seed + 10_000 + bag, stratify=fit_y,
        )
        context_x, context_y = tensor_episode(context_x_np, context_y_np, device=device)
        tune_x, tune_y = tensor_episode(tune_x_np, tune_y_np, device=device)
        fit_context_x, fit_context_y = tensor_episode(fit_x, fit_y, device=device)
        guard_x_tensor, guard_y_tensor = tensor_episode(guard_x, guard_y, device=device)
        torch.manual_seed(seed + 100_000 + bag)
        hyperspline, _ = build_tuned_hyperspline(args, backbone, device)
        optimizer = torch.optim.Adam(hyperspline.parameters(), lr=args.lr)
        first_tune_loss = None
        for step in range(1, args.train_steps + 1):
            hyperspline.train()
            backbone.clear_cache()
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(backbone, hyperspline, context_x, context_y, tune_x)
            query_loss = F.cross_entropy(logits.flatten(0, 1), tune_y.flatten())
            # Optional identity-relative deformation regularizer.  It is zero
            # at initialization and deliberately defaults to zero for headroom.
            with torch.no_grad():
                statistics = summarize_context(context_x, y_context=context_y.unsqueeze(0).float() if hyperspline.target_aware else None)
            parameters = hyperspline.generate_parameters(
                statistics, x_context=context_x, y_context=context_y.unsqueeze(0).float() if hyperspline.target_aware else None)
            objective = query_loss + args.transform_regularization * hyperspline.grid_deformation_penalty(parameters)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(hyperspline.parameters(), 1.0)
            optimizer.step()
            if first_tune_loss is None:
                first_tune_loss = float(query_loss.detach())
        hyperspline.eval()
        identity_guard_loss, identity_guard_accuracy, _ = evaluate(
            backbone, identity, fit_context_x, fit_context_y, guard_x_tensor, guard_y_tensor
        )
        adapted_guard_loss, adapted_guard_accuracy, _ = evaluate(
            backbone, hyperspline, fit_context_x, fit_context_y, guard_x_tensor, guard_y_tensor
        )
        use_adapted = adapted_guard_loss < identity_guard_loss
        identity_test_loss, identity_test_accuracy, identity_test_probs = evaluate(
            backbone, identity, fit_context_x, fit_context_y, test_x, test_y
        )
        adapted_test_loss, adapted_test_accuracy, adapted_test_probs = evaluate(
            backbone, hyperspline, fit_context_x, fit_context_y, test_x, test_y
        )
        guard_loss_delta = adapted_guard_loss - identity_guard_loss
        test_loss_delta = adapted_test_loss - identity_test_loss
        guard_accuracy_delta = adapted_guard_accuracy - identity_guard_accuracy
        test_accuracy_delta = adapted_test_accuracy - identity_test_accuracy
        test_prefers_adapted_loss = test_loss_delta < 0.0
        test_prefers_adapted_accuracy = test_accuracy_delta > 0.0
        # These fields are analysis only.  They explicitly use outer-test
        # labels *after* the decision has been made, to show whether the guard
        # is a useful proxy rather than to affect the deployed ensemble.
        guard_correct_for_test_loss = use_adapted == test_prefers_adapted_loss
        guard_correct_for_test_accuracy = (adapted_guard_accuracy > identity_guard_accuracy) == test_prefers_adapted_accuracy
        identity_probabilities.append(identity_test_probs)
        adapted_probabilities.append(adapted_test_probs)
        guarded_probabilities.append(adapted_test_probs if use_adapted else identity_test_probs)
        rows.append({
            "dataset": dataset, "outer_seed": seed, "bag": bag,
            "fit_rows": len(fit_indices), "adaptation_context_rows": len(context_y_np),
            "adaptation_query_rows": len(tune_y_np), "guard_rows": len(guard_indices),
            "initial_tune_loss": first_tune_loss, "final_tune_loss": float(query_loss.detach()),
            "identity_guard_loss": identity_guard_loss, "adapted_guard_loss": adapted_guard_loss,
            "identity_guard_accuracy": identity_guard_accuracy, "adapted_guard_accuracy": adapted_guard_accuracy,
            "guard_loss_delta": guard_loss_delta, "guard_accuracy_delta": guard_accuracy_delta,
            "identity_guard_selected_adapted": use_adapted,
            "identity_outer_test_loss": identity_test_loss, "adapted_outer_test_loss": adapted_test_loss,
            "identity_outer_test_accuracy": identity_test_accuracy, "adapted_outer_test_accuracy": adapted_test_accuracy,
            "outer_test_loss_delta": test_loss_delta, "outer_test_accuracy_delta": test_accuracy_delta,
            "outer_test_prefers_adapted_loss": test_prefers_adapted_loss,
            "outer_test_prefers_adapted_accuracy": test_prefers_adapted_accuracy,
            "guard_correct_for_outer_test_loss": guard_correct_for_test_loss,
            "guard_correct_for_outer_test_accuracy": guard_correct_for_test_accuracy,
            "guard_false_positive_loss": use_adapted and not test_prefers_adapted_loss,
            "guard_false_negative_loss": not use_adapted and test_prefers_adapted_loss,
        })
        print(
            f"[{dataset} seed={seed} bag={bag}] tune_loss={first_tune_loss:.4f}->{float(query_loss.detach()):.4f}, "
            f"guard identity={identity_guard_loss:.4f}, adapted={adapted_guard_loss:.4f}, "
            f"selected={'adapted' if use_adapted else 'identity'}", flush=True,
        )

    def ensemble_metrics(probabilities: list[torch.Tensor]) -> tuple[float, float]:
        probability = torch.stack(probabilities).mean(dim=0).clamp_min(1e-12)
        loss = F.nll_loss(probability.log().flatten(0, 1), test_y.flatten())
        accuracy = (probability.argmax(dim=-1).flatten() == test_y).float().mean()
        return float(loss), float(accuracy)

    identity_loss, identity_accuracy = ensemble_metrics(identity_probabilities)
    adapted_loss, adapted_accuracy = ensemble_metrics(adapted_probabilities)
    guarded_loss, guarded_accuracy = ensemble_metrics(guarded_probabilities)
    # Oracle bag selection is intentionally post-hoc and leaky.  Its distance
    # above the guarded result measures the guard's remaining headroom.
    oracle_probabilities = [
        adapted_probabilities[index] if bool(row["outer_test_prefers_adapted_loss"]) else identity_probabilities[index]
        for index, row in enumerate(rows)
    ]
    oracle_loss, oracle_accuracy = ensemble_metrics(oracle_probabilities)
    summary = {
        "dataset": dataset, "outer_seed": seed,
        "outer_train_rows": int(y_train.size), "outer_test_rows": int(y_test.size), "n_features": int(x.shape[1]),
        "n_classes": int(classes.size), "bags": args.bags, "train_steps_per_bag": args.train_steps,
        "identity_outer_test_loss": identity_loss, "identity_outer_test_accuracy": identity_accuracy,
        "adapted_outer_test_loss": adapted_loss, "adapted_outer_test_accuracy": adapted_accuracy,
        "guarded_outer_test_loss": guarded_loss, "guarded_outer_test_accuracy": guarded_accuracy,
        "oracle_per_bag_outer_test_loss": oracle_loss, "oracle_per_bag_outer_test_accuracy": oracle_accuracy,
        "adapted_minus_identity_loss": adapted_loss - identity_loss,
        "guarded_minus_identity_loss": guarded_loss - identity_loss,
        "adapted_minus_identity_accuracy": adapted_accuracy - identity_accuracy,
        "guarded_minus_identity_accuracy": guarded_accuracy - identity_accuracy,
        "guard_selected_adapted_fraction": float(np.mean([bool(row["identity_guard_selected_adapted"]) for row in rows])),
        "outer_test_prefers_adapted_fraction_loss": float(np.mean([bool(row["outer_test_prefers_adapted_loss"]) for row in rows])),
        "outer_test_prefers_adapted_fraction_accuracy": float(np.mean([bool(row["outer_test_prefers_adapted_accuracy"]) for row in rows])),
        "guard_correct_fraction_for_outer_test_loss": float(np.mean([bool(row["guard_correct_for_outer_test_loss"]) for row in rows])),
        "guard_correct_fraction_for_outer_test_accuracy": float(np.mean([bool(row["guard_correct_for_outer_test_accuracy"]) for row in rows])),
        "guard_false_positive_fraction_loss": float(np.mean([bool(row["guard_false_positive_loss"]) for row in rows])),
        "guard_false_negative_fraction_loss": float(np.mean([bool(row["guard_false_negative_loss"]) for row in rows])),
        "guard_false_positive_count_loss": int(sum(bool(row["guard_false_positive_loss"]) for row in rows)),
        "guard_false_negative_count_loss": int(sum(bool(row["guard_false_negative_loss"]) for row in rows)),
        "oracle_per_bag_minus_identity_loss": oracle_loss - identity_loss,
        "oracle_per_bag_minus_identity_accuracy": oracle_accuracy - identity_accuracy,
        "protocol": "outer_holdout + nested_8_fold_bagging + train_only_adaptation + external_identity_guard",
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    if not 0 < args.outer_test_size < 1 or not 0 < args.adaptation_context_fraction < 1:
        raise ValueError("outer test size and adaptation context fraction must lie in (0, 1)")
    if args.bags < 2 or args.train_steps <= 0 or args.lr <= 0 or args.transform_regularization < 0:
        raise ValueError("invalid bagging or optimization configuration")
    if args.raw_context_residual and not args.target_aware:
        raise ValueError("--raw-context-residual requires --target-aware")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    backbone, _ = load_backbone(args, device)
    # All runs use identical HyperSpline construction; every bag still starts
    # independently from the supplied initial checkpoint.
    prototype, config = build_tuned_hyperspline(args, backbone, device)
    del prototype
    all_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for dataset in args.pmlb_dataset:
        for seed in args.seeds:
            rows, summary = run_one_protocol(args, backbone, config, dataset=dataset, seed=seed, device=device)
            all_rows.extend(rows)
            run_summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
    if not all_rows:
        raise RuntimeError("no dataset/seed runs were requested")
    write_csv(args.output_fold_csv, all_rows)
    macro_metric_names = (
        "adapted_minus_identity_loss", "guarded_minus_identity_loss",
        "adapted_minus_identity_accuracy", "guarded_minus_identity_accuracy",
        "guard_correct_fraction_for_outer_test_loss", "guard_false_positive_fraction_loss",
        "guard_false_negative_fraction_loss", "oracle_per_bag_minus_identity_loss",
    )
    macro = {name: float(np.mean([float(run[name]) for run in run_summaries])) for name in macro_metric_names}
    summary = {
        "runs": run_summaries,
        "macro_mean_over_dataset_seed_runs": macro,
        "n_datasets": len(args.pmlb_dataset), "n_seeds": len(args.seeds), "n_runs": len(run_summaries),
        "n_bag_decisions": len(all_rows),
        "datasets": args.pmlb_dataset, "seeds": args.seeds,
        "protocol": "outer_holdout + nested_bagging + train_only_adaptation + external_identity_guard; oracle fields are post-hoc diagnostics",
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
