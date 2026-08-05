"""Paired DirectSpline test of monotone versus bounded free-control curves.

``free_refinement`` starts from the exact trained monotone spline function and
unlocks only bounded, endpoint-pinned interior controls.  It therefore tests
whether non-monotonic curve shape adds held-out headroom rather than merely
following a different initial optimisation path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = ("monotone_continuation", "free_refinement", "free_from_identity")


def parse_csv(value: str, converter):
    values = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmlb-dataset", action="append", required=True)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--variants", default=",".join(CONDITIONS))
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
    parser.add_argument("--free-control-bound", type=float, default=1.0)
    parser.add_argument("--free-control-curvature-regularization", type=float, default=1e-3)
    parser.add_argument("--free-control-reference-regularization", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def command_prefix(args: argparse.Namespace, runner: Path, variant: str, seeds: list[int]) -> list[str]:
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
        "--free-control-bound", str(args.free_control_bound),
        "--free-control-curvature-regularization", str(args.free_control_curvature_regularization),
        "--free-control-reference-regularization", str(args.free_control_reference_regularization),
        "--skip-margin-diagnostics",
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
    seeds, variants = parse_csv(args.seeds, int), parse_csv(args.variants, str)
    unknown = sorted(set(variants).difference(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(CONDITIONS)}")
    if args.n_control_points <= 3 or args.base_steps <= 0 or args.refinement_steps <= 0:
        raise ValueError("control points must exceed 3 and both phase lengths must be positive")
    if (args.lr <= 0 or args.free_control_bound <= 0 or args.free_control_curvature_regularization < 0
            or args.free_control_reference_regularization < 0):
        raise ValueError("invalid free-control optimisation arguments")
    args.output_dir, args.pmlb_cache_dir = args.output_dir.resolve(), args.pmlb_cache_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner, base_state_dir = Path(__file__).with_name("direct_spline_dataset_headroom.py"), args.output_dir / "base_states"
    if "free_refinement" in variants and "monotone_continuation" not in variants and not base_state_dir.is_dir():
        raise ValueError("free_refinement requires saved monotone base states; run it with monotone_continuation first.")

    completed = []
    for variant in variants:
        condition_dir = args.output_dir / variant
        command = command_prefix(args, runner, variant, seeds)
        command.extend((
            "--output-fold-csv", str(condition_dir / "folds.csv"),
            "--output-margin-csv", str(condition_dir / "margin.csv"),
            "--output-summary-json", str(condition_dir / "summary.json"),
        ))
        if variant == "monotone_continuation":
            command.extend((
                "--control-mode", "monotone", "--knot-placement", "uniform", "--steps", str(args.base_steps),
                "--knot-refinement-steps", str(args.refinement_steps),
                "--knot-refinement-placement", "uniform", "--save-base-state-dir", str(base_state_dir),
            ))
        elif variant == "free_refinement":
            command.extend((
                "--control-mode", "free", "--knot-placement", "uniform", "--steps", str(args.base_steps),
                "--knot-refinement-steps", str(args.refinement_steps),
                "--knot-refinement-placement", "uniform", "--base-state-dir", str(base_state_dir),
            ))
        else:
            command.extend((
                "--control-mode", "free", "--knot-placement", "uniform",
                "--steps", str(args.base_steps + args.refinement_steps),
            ))
        print(f"Running {variant}", flush=True)
        subprocess.run(command, check=True)
        completed.append({
            "condition": variant, "n_control_points": args.n_control_points,
            "total_steps": args.base_steps + args.refinement_steps,
            "free_control_bound": args.free_control_bound,
            "free_control_curvature_regularization": args.free_control_curvature_regularization,
            "free_control_reference_regularization": args.free_control_reference_regularization,
            "exact_monotone_base_state_dir": str(base_state_dir) if variant == "free_refinement" else None,
            "summary": str(condition_dir / "summary.json"),
        })
        (args.output_dir / "manifest.json").write_text(json.dumps({"conditions": completed}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
