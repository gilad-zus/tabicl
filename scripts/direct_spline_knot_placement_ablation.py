"""Measure whether spline resolution or individual knot placement adds headroom.

The experiment remains strictly a numerical per-column spline experiment.  It
compares fixed uniform knots at several control-point counts with two K=20
alternatives: deterministic context-quantile knots and gradient-optimized,
strictly ordered knot locations.  Every condition reuses the established
train-only DirectSpline protocol, identity guard, and parameter-margin tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_csv(value: str, converter):
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer-test-size", type=float, default=0.20)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--max-context-rows", type=int, default=512)
    parser.add_argument("--train-context-rows", type=int, default=384)
    parser.add_argument("--query-batch-rows", type=int, default=256)
    parser.add_argument("--evaluation-query-chunk-rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1_250)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--transform-regularization", type=float, default=0.0)
    parser.add_argument("--uniform-control-points", default="10,20,40",
                        help="Comma-separated K values for fixed uniform-knot capacity checks.")
    parser.add_argument("--adaptive-control-points", type=int, default=20,
                        help="K used for the paired quantile-knot and learned-knot conditions.")
    parser.add_argument("--interpolation-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--perturbation-scales", default="0.05,0.10,0.25")
    parser.add_argument("--perturbation-repeats", type=int, default=2)
    parser.add_argument("--cross-bag-max-sources", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_csv(args.seeds, int)
    uniform_controls = parse_csv(args.uniform_control_points, int)
    if any(k <= 3 for k in uniform_controls) or args.adaptive_control_points <= 3:
        raise ValueError("every control-point count must exceed cubic degree 3")
    if len(set(uniform_controls)) != len(uniform_controls):
        raise ValueError("--uniform-control-points must not contain duplicates")
    if args.bags < 2 or args.steps <= 0 or args.lr <= 0:
        raise ValueError("invalid DirectSpline optimization configuration")
    if len(set(args.pmlb_dataset)) != len(args.pmlb_dataset) or len(set(seeds)) != len(seeds):
        raise ValueError("datasets and seeds must be unique")

    args.output_dir = args.output_dir.resolve()
    args.pmlb_cache_dir = args.pmlb_cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("direct_spline_dataset_headroom.py")
    conditions = [("uniform", k) for k in uniform_controls]
    conditions.extend((("quantile", args.adaptive_control_points), ("learned", args.adaptive_control_points)))
    completed = []

    for placement, controls in conditions:
        name = f"{placement}_k{controls}"
        condition_dir = args.output_dir / name
        command = [
            sys.executable, str(runner), "--basis-variant", name,
            "--knot-placement", placement, "--n-control-points", str(controls),
            "--trainable-location-scale",
            "--pmlb-cache-dir", str(args.pmlb_cache_dir),
            "--checkpoint-version", args.checkpoint_version, "--device", args.device,
            "--seeds", *[str(seed) for seed in seeds], "--outer-test-size", str(args.outer_test_size),
            "--bags", str(args.bags), "--max-context-rows", str(args.max_context_rows),
            "--train-context-rows", str(args.train_context_rows),
            "--query-batch-rows", str(args.query_batch_rows),
            "--evaluation-query-chunk-rows", str(args.evaluation_query_chunk_rows),
            "--steps", str(args.steps), "--lr", str(args.lr),
            "--transform-regularization", str(args.transform_regularization),
            "--interpolation-alphas", args.interpolation_alphas,
            "--perturbation-scales", args.perturbation_scales,
            "--perturbation-repeats", str(args.perturbation_repeats),
            "--cross-bag-max-sources", str(args.cross_bag_max_sources),
            "--output-fold-csv", str(condition_dir / "folds.csv"),
            "--output-margin-csv", str(condition_dir / "margin.csv"),
            "--output-summary-json", str(condition_dir / "summary.json"),
        ]
        for dataset in args.pmlb_dataset:
            command.extend(("--pmlb-dataset", dataset))
        if args.checkpoint is not None:
            command.extend(("--checkpoint", args.checkpoint))
        if args.resume:
            command.append("--resume")
        print(f"Running {name}", flush=True)
        subprocess.run(command, check=True)
        completed.append({
            "condition": name,
            "knot_placement": placement,
            "n_control_points": controls,
            "summary": str(condition_dir / "summary.json"),
            "margin": str(condition_dir / "margin.csv"),
        })
        (args.output_dir / "manifest.json").write_text(
            json.dumps({"conditions": completed}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
