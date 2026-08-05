"""Clean paired study of deterministic context-quantile spline knots.

The first two arms share exactly the same uniform DirectSpline teacher after
``--base-steps``.  ``quantile_continuation`` function-matches that teacher in
the context-quantile basis without labels before its matched refinement
updates.  It records the residual projection error, so any interpretation can
verify that the two branches started sufficiently close.

``quantile_from_identity`` is a complementary full-budget arm.  It answers
whether quantile knots help mainly because they change the optimisation path
from the initial identity transform.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = ("uniform_continuation", "quantile_continuation", "quantile_from_identity")


def parse_csv(value: str, converter):
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--variants", default=",".join(CONDITIONS),
                        help=f"Comma-separated subset of: {', '.join(CONDITIONS)}.")
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
    parser.add_argument("--base-steps", type=int, default=1_250)
    parser.add_argument("--refinement-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--transform-regularization", type=float, default=0.0)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--knot-projection-steps", type=int, default=250)
    parser.add_argument("--knot-projection-lr", type=float, default=0.10)
    parser.add_argument("--knot-projection-grid-points", type=int, default=257)
    parser.add_argument("--interpolation-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--perturbation-scales", default="0.05,0.10,0.25")
    parser.add_argument("--perturbation-repeats", type=int, default=2)
    parser.add_argument("--cross-bag-max-sources", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def common_command(args: argparse.Namespace, runner: Path, variant: str, seeds: list[int]) -> list[str]:
    command = [
        sys.executable, str(runner), "--basis-variant", variant,
        "--n-control-points", str(args.n_control_points), "--trainable-location-scale",
        "--pmlb-cache-dir", str(args.pmlb_cache_dir),
        "--checkpoint-version", args.checkpoint_version, "--device", args.device,
        "--seeds", *[str(seed) for seed in seeds], "--outer-test-size", str(args.outer_test_size),
        "--bags", str(args.bags), "--max-context-rows", str(args.max_context_rows),
        "--train-context-rows", str(args.train_context_rows),
        "--query-batch-rows", str(args.query_batch_rows),
        "--evaluation-query-chunk-rows", str(args.evaluation_query_chunk_rows),
        "--lr", str(args.lr), "--transform-regularization", str(args.transform_regularization),
        "--knot-projection-steps", str(args.knot_projection_steps),
        "--knot-projection-lr", str(args.knot_projection_lr),
        "--knot-projection-grid-points", str(args.knot_projection_grid_points),
        "--interpolation-alphas", args.interpolation_alphas,
        "--perturbation-scales", args.perturbation_scales,
        "--perturbation-repeats", str(args.perturbation_repeats),
        "--cross-bag-max-sources", str(args.cross_bag_max_sources),
    ]
    for dataset in args.pmlb_dataset:
        command.extend(("--pmlb-dataset", dataset))
    if args.checkpoint is not None:
        command.extend(("--checkpoint", args.checkpoint))
    if args.resume:
        command.append("--resume")
    return command


def main() -> None:
    args = parse_args()
    seeds = parse_csv(args.seeds, int)
    variants = parse_csv(args.variants, str)
    unknown = sorted(set(variants).difference(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(CONDITIONS)}")
    if args.n_control_points <= 3 or args.base_steps <= 0 or args.refinement_steps <= 0:
        raise ValueError("control points must exceed 3 and both phase lengths must be positive")
    if args.lr <= 0 or args.knot_projection_steps <= 0 or args.knot_projection_lr <= 0:
        raise ValueError("learning rates and projection steps must be positive")
    if len(set(args.pmlb_dataset)) != len(args.pmlb_dataset) or len(set(seeds)) != len(seeds):
        raise ValueError("datasets and seeds must be unique")

    args.output_dir = args.output_dir.resolve()
    args.pmlb_cache_dir = args.pmlb_cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("direct_spline_dataset_headroom.py")
    base_state_dir = args.output_dir / "base_states"
    if "quantile_continuation" in variants and "uniform_continuation" not in variants and not base_state_dir.is_dir():
        raise ValueError("quantile_continuation requires saved uniform base states; run it with uniform_continuation first.")

    completed = []
    for variant in variants:
        condition_dir = args.output_dir / variant
        command = common_command(args, runner, variant, seeds)
        command.extend((
            "--output-fold-csv", str(condition_dir / "folds.csv"),
            "--output-margin-csv", str(condition_dir / "margin.csv"),
            "--output-summary-json", str(condition_dir / "summary.json"),
        ))
        if variant == "uniform_continuation":
            command.extend((
                "--knot-placement", "uniform", "--steps", str(args.base_steps),
                "--knot-refinement-steps", str(args.refinement_steps),
                "--knot-refinement-placement", "uniform",
                "--save-base-state-dir", str(base_state_dir),
            ))
        elif variant == "quantile_continuation":
            command.extend((
                "--knot-placement", "uniform", "--steps", str(args.base_steps),
                "--knot-refinement-steps", str(args.refinement_steps),
                "--knot-refinement-placement", "quantile",
                "--base-state-dir", str(base_state_dir),
            ))
        else:
            command.extend((
                "--knot-placement", "quantile", "--steps", str(args.base_steps + args.refinement_steps),
            ))
        print(f"Running {variant}", flush=True)
        subprocess.run(command, check=True)
        completed.append({
            "condition": variant,
            "n_control_points": args.n_control_points,
            "base_steps": args.base_steps,
            "refinement_steps": args.refinement_steps,
            "total_steps": args.base_steps + args.refinement_steps,
            "base_state_dir": str(base_state_dir) if variant != "quantile_from_identity" else None,
            "function_matched_projection": variant == "quantile_continuation",
            "projection_grid_points": args.knot_projection_grid_points if variant == "quantile_continuation" else None,
            "summary": str(condition_dir / "summary.json"),
            "margin": str(condition_dir / "margin.csv"),
        })
        (args.output_dir / "manifest.json").write_text(
            json.dumps({"conditions": completed}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
