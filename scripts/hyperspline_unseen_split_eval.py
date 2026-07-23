"""Evaluate a trained shared HyperSpline on unseen context/query splits.

This script never optimizes HyperSpline parameters.  Train a checkpoint on
one set of split seeds, then run this script on disjoint seeds for the same
datasets.  This separates split-level generalization from training-split fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from hyperspline_multidataset_overfit import (
    DATASETS,
    build_episode,
    evaluate_episode,
    load_backbone,
    parse_csv,
)
from tabicl._hyperspline import HyperSplineTransform, backbone_state_dict_hash, load_hyperspline_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyperspline-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", default="iris,wine,breast_cancer,digits")
    parser.add_argument("--seeds", default="3,4,5", help="Must not overlap the training split seeds.")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--output-csv", type=Path, default=Path("results/hyperspline_unseen_splits.csv"))
    args = parser.parse_args()

    if not args.hyperspline_checkpoint.is_file():
        raise FileNotFoundError(f"HyperSpline checkpoint does not exist: {args.hyperspline_checkpoint}")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    datasets = parse_csv(args.datasets, str)
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choices are {sorted(DATASETS)}")
    seeds = parse_csv(args.seeds, int)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    print(f"Running frozen unseen-split evaluation on device: {device}", flush=True)
    backbone, _ = load_backbone(args, device)
    hyperspline, metadata = load_hyperspline_checkpoint(
        args.hyperspline_checkpoint,
        device=device,
        expected_backbone_reference=args.checkpoint_version,
        expected_backbone_hash=backbone_state_dict_hash(backbone),
    )
    hyperspline.eval()
    # A newly constructed module has identity spline control points.  Measuring
    # it on the same episodes gives the frozen TabICL baseline needed to judge
    # whether the checkpoint helps unseen splits.
    identity = HyperSplineTransform(**metadata["hyperspline_config"]).to(device).eval()
    print(
        f"Loaded HyperSpline step={metadata.get('step')} and will not optimize it; "
        f"datasets={datasets}, unseen_seeds={seeds}",
        flush=True,
    )

    records = []
    for dataset in datasets:
        for seed in seeds:
            episode = build_episode(dataset, seed, args, device)
            if torch.unique(episode.y_context).numel() > backbone.max_classes:
                raise ValueError(f"{dataset} exceeds the backbone's {backbone.max_classes}-class limit")
            backbone.clear_cache()
            baseline = evaluate_episode(backbone, identity, episode)
            backbone.clear_cache()
            metrics = evaluate_episode(backbone, hyperspline, episode)
            record = {
                "phase": "unseen_split",
                "checkpoint_step": metadata.get("step"),
                **metrics,
                "identity_loss": baseline["loss"],
                "identity_accuracy": baseline["accuracy"],
                "loss_delta": float(baseline["loss"]) - float(metrics["loss"]),
                "accuracy_delta": float(metrics["accuracy"]) - float(baseline["accuracy"]),
            }
            records.append(record)
            print(
                f"[{dataset} seed={seed}] loss={metrics['loss']:.6f}, "
                f"accuracy={metrics['accuracy']:.4f}, loss_delta={record['loss_delta']:+.6f}, "
                f"mean_gate={metrics['mean_gate']:.4f}",
                flush=True,
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} frozen unseen-split records to {args.output_csv}", flush=True)
    for dataset in datasets:
        rows = [row for row in records if row["dataset"] == dataset]
        print(
            f"  {dataset}: mean_loss={np.mean([row['loss'] for row in rows]):.6f}, "
            f"mean_accuracy={np.mean([row['accuracy'] for row in rows]):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
