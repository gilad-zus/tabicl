"""Map DirectSpline low-rank mixing capacity around the current rank-4 model.

The full sweep runs one univariate reference and five paired low-rank mixing
conditions.  ``--new-arms-only`` omits the two already-completed references
(``joint_univariate`` and ``rank4_bound010``), leaving only the four new
conditions needed to extend ``direct_spline_cross_column``.

All conditions use the same train-only DirectSpline protocol, outer splits,
identity guard, and margin diagnostics.  The resulting fold CSV additionally
records the effective mixing spectrum, stable rank, and output-energy ratio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FULL_CONDITIONS = (
    ("joint_univariate", 0, 0.0),
    ("rank2_bound010", 2, 0.10),
    ("rank4_bound005", 4, 0.05),
    ("rank4_bound010", 4, 0.10),
    ("rank4_bound020", 4, 0.20),
    ("rank8_bound010", 8, 0.10),
)
NEW_CONDITIONS = tuple(
    condition for condition in FULL_CONDITIONS
    if condition[0] not in {"joint_univariate", "rank4_bound010"}
)


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
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--interpolation-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--perturbation-scales", default="0.05,0.10,0.25")
    parser.add_argument("--perturbation-repeats", type=int, default=2)
    parser.add_argument("--cross-bag-max-sources", type=int, default=3)
    parser.add_argument("--new-arms-only", action="store_true",
                        help="Run only rank2/bound0.1, rank4/bound0.05, rank4/bound0.2, and rank8/bound0.1.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_csv(args.seeds, int)
    if args.n_control_points <= 3 or args.bags < 2 or args.steps <= 0 or args.lr <= 0:
        raise ValueError("invalid DirectSpline sweep configuration")
    if len(set(args.pmlb_dataset)) != len(args.pmlb_dataset) or len(set(seeds)) != len(seeds):
        raise ValueError("datasets and seeds must be unique")
    conditions = NEW_CONDITIONS if args.new_arms_only else FULL_CONDITIONS
    args.output_dir = args.output_dir.resolve()
    args.pmlb_cache_dir = args.pmlb_cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("direct_spline_dataset_headroom.py")
    completed = []

    for name, rank, bound in conditions:
        condition_dir = args.output_dir / f"{name}_k{args.n_control_points}"
        command = [
            sys.executable, str(runner), "--basis-variant", name,
            "--n-control-points", str(args.n_control_points),
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
            "--trainable-location-scale",
        ]
        for dataset in args.pmlb_dataset:
            command.extend(("--pmlb-dataset", dataset))
        if rank:
            command.extend((
                "--cross-column-mixing-rank", str(rank),
                "--cross-column-mixing-bound", str(bound),
            ))
        if args.checkpoint is not None:
            command.extend(("--checkpoint", args.checkpoint))
        if args.resume:
            command.append("--resume")
        print(f"Running {name}: rank={rank}, bound={bound}", flush=True)
        subprocess.run(command, check=True)
        completed.append({
            "condition": name, "mixing_rank": rank, "mixing_bound": bound,
            "summary": str(condition_dir / "summary.json"),
            "margin": str(condition_dir / "margin.csv"),
        })
        (args.output_dir / "manifest.json").write_text(
            json.dumps({"conditions": completed}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
