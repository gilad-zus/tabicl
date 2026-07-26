"""Evaluate marginal and supervised-residual HyperSpline checkpoints on one real bank.

All real episodes are built once, then identity, marginal, and supervised
checkpoints are evaluated sequentially against those exact tensors.  This is
the paired counterpart to ``hyperspline_real_zero_shot_eval.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from hyperspline_real_zero_shot_eval import (
    build_episode,
    evaluate,
    feature_bin,
    load_backbone,
    parse_csv,
    resolve_specs,
    summarize_rows,
    write_csv,
)
from tabicl._hyperspline import HyperSplineTransform, backbone_state_dict_hash, load_hyperspline_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginal-checkpoint", type=Path, required=True)
    parser.add_argument("--supervised-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-suite", choices=("all", "numerical_only", "mixed"), default="all")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--marginal-output-csv", type=Path, required=True)
    parser.add_argument("--marginal-output-summary-csv", type=Path, required=True)
    parser.add_argument("--supervised-output-csv", type=Path, required=True)
    parser.add_argument("--supervised-output-summary-csv", type=Path, required=True)
    parser.add_argument("--comparison-output-csv", type=Path, required=True)
    return parser.parse_args()


def output_row(episode, transformed: dict[str, float], baseline: dict[str, float]) -> dict[str, object]:
    loss_delta = baseline["loss"] - transformed["loss"]
    return {
        "dataset": episode.dataset,
        "dataset_group": episode.dataset_group,
        "split_seed": episode.split_seed,
        "n_context": episode.n_context,
        "n_query": episode.n_query,
        "n_features": episode.n_features,
        "feature_bin": feature_bin(episode.n_features),
        "n_numerical_features": episode.n_numerical_features,
        "n_categorical_features": episode.n_categorical_features,
        "n_classes": episode.n_classes,
        **transformed,
        "identity_loss": baseline["loss"],
        "identity_accuracy": baseline["accuracy"],
        "loss_delta": loss_delta,
        "relative_loss_improvement": loss_delta / max(baseline["loss"], 1e-12),
        "accuracy_delta": transformed["accuracy"] - baseline["accuracy"],
        "loss_is_finite": bool(np.isfinite(baseline["loss"]) and np.isfinite(transformed["loss"])),
    }


def write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1 or args.max_rows < 0:
        raise ValueError("--test-size must be in (0, 1) and --max-rows must be non-negative")
    if not args.marginal_checkpoint.is_file() or not args.supervised_checkpoint.is_file():
        raise FileNotFoundError("both HyperSpline checkpoints must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    seeds, specs = parse_csv(args.seeds, int), resolve_specs(args)
    backbone, _ = load_backbone(args, device)
    expected_hash = backbone_state_dict_hash(backbone)
    marginal, marginal_metadata = load_hyperspline_checkpoint(
        args.marginal_checkpoint, device=device, expected_backbone_reference=args.checkpoint_version, expected_backbone_hash=expected_hash
    )
    supervised, supervised_metadata = load_hyperspline_checkpoint(
        args.supervised_checkpoint, device=device, expected_backbone_reference=args.checkpoint_version, expected_backbone_hash=expected_hash
    )
    marginal_config, supervised_config = dict(marginal_metadata["hyperspline_config"]), dict(supervised_metadata["hyperspline_config"])
    variant_only_keys = {
        "target_aware",
        "supervised_residual",
        "supervised_residual_gate_initial_probability",
        "cross_column_residual",
        "cross_column_num_heads",
        "cross_column_residual_bound",
        "cross_column_gate_initial_probability",
        "raw_context_residual",
        "raw_context_num_heads",
        "raw_context_residual_bound",
        "raw_context_gate_initial_probability",
    }
    if {key: value for key, value in marginal_config.items() if key not in variant_only_keys} != {
        key: value for key, value in supervised_config.items() if key not in variant_only_keys
    }:
        raise ValueError("paired evaluation requires matching marginal HyperSpline configurations")
    if (
        marginal_config.get("target_aware")
        or marginal_config.get("supervised_residual")
        or marginal_config.get("cross_column_residual")
        or marginal_config.get("raw_context_residual")
    ):
        raise ValueError("--marginal-checkpoint must be marginal-only")
    if not supervised_config.get("target_aware") or not (
        supervised_config.get("cross_column_residual") or supervised_config.get("raw_context_residual")
    ):
        raise ValueError("--supervised-checkpoint must use a supported supervised residual")
    identity_config = {
        **marginal_config,
        "target_aware": False,
        "supervised_residual": False,
        "cross_column_residual": False,
        "raw_context_residual": False,
    }
    identity = HyperSplineTransform(**identity_config).to(device).eval()
    marginal.eval()
    supervised.eval()

    marginal_rows, supervised_rows, comparison_rows = [], [], []
    print(f"Building one fixed real episode bank: datasets={[spec.name for spec in specs]}, seeds={seeds}", flush=True)
    for spec in specs:
        for seed in seeds:
            episode = build_episode(spec, seed, args, device)
            if episode.n_classes > backbone.max_classes:
                raise ValueError(f"{spec.name} has {episode.n_classes} classes; backbone supports {backbone.max_classes}")
            baseline = evaluate(backbone, identity, episode)
            marginal_metrics = evaluate(backbone, marginal, episode)
            supervised_metrics = evaluate(backbone, supervised, episode)
            marginal_row = output_row(episode, marginal_metrics, baseline)
            supervised_row = output_row(episode, supervised_metrics, baseline)
            marginal_rows.append(marginal_row)
            supervised_rows.append(supervised_row)
            comparison_rows.append(
                {
                    "dataset": episode.dataset,
                    "split_seed": episode.split_seed,
                    "identity_loss": baseline["loss"],
                    "identity_accuracy": baseline["accuracy"],
                    "marginal_loss": marginal_metrics["loss"],
                    "supervised_loss": supervised_metrics["loss"],
                    "supervised_minus_marginal_loss": supervised_metrics["loss"] - marginal_metrics["loss"],
                    "marginal_accuracy": marginal_metrics["accuracy"],
                    "supervised_accuracy": supervised_metrics["accuracy"],
                    "supervised_minus_marginal_accuracy": supervised_metrics["accuracy"] - marginal_metrics["accuracy"],
                    "loss_is_finite": marginal_row["loss_is_finite"] and supervised_row["loss_is_finite"],
                }
            )

    write_csv(args.marginal_output_csv, marginal_rows)
    write_csv(args.marginal_output_summary_csv, summarize_rows(marginal_rows))
    write_csv(args.supervised_output_csv, supervised_rows)
    write_csv(args.supervised_output_summary_csv, summarize_rows(supervised_rows))
    write_comparison(args.comparison_output_csv, comparison_rows)
    print(f"Wrote paired comparison rows to {args.comparison_output_csv}", flush=True)


if __name__ == "__main__":
    main()
