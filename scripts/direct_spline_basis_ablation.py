"""Run a paired DirectSpline basis ablation before expanding HyperSpline.

Each condition uses the bounded train-only DirectSpline protocol.  The only
differences are freedoms that a future HyperSpline could predict: number of
control points, a bounded per-column spline range, and bounded location/scale
residuals.  Results are written per condition plus one manifest, keeping the
existing teacher/margin diagnostics intact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASIS_VARIANTS = {
    "uniform_fixed": (),
    "uniform_learned_range": ("--trainable-range",),
    "uniform_learned_location_scale": ("--trainable-location-scale",),
    "uniform_learned_all": ("--trainable-range", "--trainable-location-scale"),
}

# This uses the basis result to ask a narrower question: which trainable
# block is responsible for the held-out benefit?  Range is deliberately not
# included because its earlier standalone ablation added little headroom.
DECOMPOSITION_VARIANTS = {
    "shape_only": (),
    "location_scale_only": ("--freeze-spline-shape", "--trainable-location-scale"),
    "joint_shape_location_scale": ("--trainable-location-scale",),
}

# This asks whether a qualitatively new, bounded multivariate freedom raises
# the DirectSpline headroom beyond the best independent-column transform.
CROSS_COLUMN_VARIANTS = {
    "joint_univariate": ("--trainable-location-scale",),
    "joint_low_rank_mixing": ("--trainable-location-scale",),
}


def parse_csv(value: str, converter):
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True)
    parser.add_argument("--experiment", choices=("basis", "decomposition", "cross_column"), default="basis",
                        help="Run basis freedoms, shape-versus-affine decomposition, or cross-column mixing.")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--variants", default=None,
                        help="Comma-separated condition names; defaults depend on --experiment.")
    parser.add_argument("--control-points", default="20", help="Comma-separated K values, e.g. 8,12,20,32.")
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
    parser.add_argument("--mixing-rank", type=int, default=4,
                        help="Low rank used by the joint_low_rank_mixing condition.")
    parser.add_argument("--mixing-bound", type=float, default=0.1,
                        help="Maximum spectral scale of the low-rank mixing residual.")
    parser.add_argument("--interpolation-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--perturbation-scales", default="0.05,0.10,0.25")
    parser.add_argument("--perturbation-repeats", type=int, default=2)
    parser.add_argument("--cross-bag-max-sources", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    variant_definitions = {
        "basis": BASIS_VARIANTS,
        "decomposition": DECOMPOSITION_VARIANTS,
        "cross_column": CROSS_COLUMN_VARIANTS,
    }[args.experiment]
    if args.variants is None:
        args.variants = ",".join(variant_definitions)
    variants = parse_csv(args.variants, str)
    unknown = sorted(set(variants).difference(variant_definitions))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choices are {sorted(variant_definitions)}")
    control_points = parse_csv(args.control_points, int)
    if any(value <= 3 for value in control_points):
        raise ValueError("all control-point counts must exceed cubic degree 3")
    if args.mixing_rank <= 0 or args.mixing_bound < 0:
        raise ValueError("--mixing-rank must be positive and --mixing-bound non-negative")
    seeds = parse_csv(args.seeds, int)
    args.output_dir = args.output_dir.resolve()
    args.pmlb_cache_dir = args.pmlb_cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("direct_spline_dataset_headroom.py")
    completed = []
    for variant in variants:
        for controls in control_points:
            condition_dir = args.output_dir / f"{variant}_k{controls}"
            command = [
                sys.executable, str(runner), "--basis-variant", variant,
                "--n-control-points", str(controls), "--pmlb-cache-dir", str(args.pmlb_cache_dir),
                "--checkpoint-version", args.checkpoint_version, "--device", args.device,
                "--seeds", *[str(seed) for seed in seeds], "--outer-test-size", str(args.outer_test_size),
                "--bags", str(args.bags), "--max-context-rows", str(args.max_context_rows),
                "--train-context-rows", str(args.train_context_rows), "--query-batch-rows", str(args.query_batch_rows),
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
            command.extend(variant_definitions[variant])
            if args.experiment == "cross_column" and variant == "joint_low_rank_mixing":
                command.extend((
                    "--cross-column-mixing-rank", str(args.mixing_rank),
                    "--cross-column-mixing-bound", str(args.mixing_bound),
                ))
            print(f"Running {variant}, K={controls}", flush=True)
            subprocess.run(command, check=True)
            completed.append({
                "experiment": args.experiment, "variant": variant, "n_control_points": controls,
                "mixing_rank": args.mixing_rank if args.experiment == "cross_column" else None,
                "mixing_bound": args.mixing_bound if args.experiment == "cross_column" else None,
                "summary": str(condition_dir / "summary.json"),
                "margin": str(condition_dir / "margin.csv"),
            })
            (args.output_dir / "manifest.json").write_text(json.dumps({"conditions": completed}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
