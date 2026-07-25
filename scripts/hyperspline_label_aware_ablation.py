"""Run paired marginal and class-invariant-supervised HyperSpline experiments.

Use ``--variants class_invariant_supervised`` to resume an interrupted run
after the marginal condition has completed.  Existing manifest entries are
preserved, and the default seeds remain identical to a full paired run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
VARIANTS = {"marginal": False, "class_invariant_supervised": True}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hyperspline_label_aware_ablation"))
    parser.add_argument(
        "--variants",
        default="marginal,class_invariant_supervised",
        help="Comma-separated: marginal,class_invariant_supervised. Select only the unfinished condition to resume.",
    )
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--validation-tasks", type=int, default=64)
    parser.add_argument("--test-tasks", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--gate-initial-probability", type=float, default=0.10)
    parser.add_argument("--prior-type", default="mix_scm")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=10_001)
    parser.add_argument("--test-seed", type=int, default=20_001)
    parser.add_argument("--real-dataset-suite", choices=("all", "numerical_only", "mixed"), default="all")
    parser.add_argument("--real-datasets", default=None)
    parser.add_argument("--real-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--real-test-size", type=float, default=0.30)
    parser.add_argument("--real-max-rows", type=int, default=1024)
    return parser.parse_args()


def train_options(args: argparse.Namespace) -> list[str]:
    options = {
        "--checkpoint-version": args.checkpoint_version,
        "--device": args.device,
        "--prior-type": args.prior_type,
        "--sequence-length": args.sequence_length,
        "--context-fraction": args.context_fraction,
        "--min-features": args.min_features,
        "--max-features": args.max_features,
        "--max-classes": args.max_classes,
        "--prior-n-jobs": args.prior_n_jobs,
        "--train-steps": args.train_steps,
        "--tasks-per-step": args.tasks_per_step,
        "--validation-tasks": args.validation_tasks,
        "--test-tasks": args.test_tasks,
        "--validate-every": args.validate_every,
        "--lr": args.lr,
        "--hidden-dim": args.hidden_dim,
        "--n-control-points": args.n_control_points,
        "--gate-initial-probability": args.gate_initial_probability,
        "--model-seed": args.model_seed,
        "--train-seed": args.train_seed,
        "--validation-seed": args.validation_seed,
        "--test-seed": args.test_seed,
    }
    result = [part for key, value in options.items() for part in (key, str(value))]
    if args.checkpoint is not None:
        result.extend(("--checkpoint", args.checkpoint))
    return result


def main() -> None:
    args = parse_args()
    requested = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(requested).difference(VARIANTS))
    if not requested or unknown:
        raise ValueError(f"--variants must be drawn from {sorted(VARIANTS)}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"variants": {}}
    manifest.setdefault("variants", {})
    validation_bank = output_dir / "synthetic_validation_bank.pt"
    test_bank = output_dir / "synthetic_test_bank.pt"

    for name in requested:
        checkpoint = output_dir / f"{name}.pt"
        train_command = [sys.executable, str(SCRIPTS / "hyperspline_synthetic_train.py"), *train_options(args)]
        if VARIANTS[name]:
            train_command.append("--target-aware")
        train_command.extend((
            "--validation-bank", str(validation_bank),
            "--test-bank", str(test_bank),
            "--output-csv", str(output_dir / f"{name}_synthetic.csv"),
            "--output-train-csv", str(output_dir / f"{name}_training.csv"),
            "--output-checkpoint", str(checkpoint),
        ))
        run(train_command)
        manifest["variants"][name] = {"target_aware": VARIANTS[name], "checkpoint": str(checkpoint)}

    marginal_checkpoint = output_dir / "marginal.pt"
    supervised_checkpoint = output_dir / "class_invariant_supervised.pt"
    if marginal_checkpoint.is_file() and supervised_checkpoint.is_file():
        paired_command = [
            sys.executable, str(SCRIPTS / "hyperspline_real_paired_eval.py"),
            "--marginal-checkpoint", str(marginal_checkpoint),
            "--supervised-checkpoint", str(supervised_checkpoint),
            "--checkpoint-version", args.checkpoint_version,
            "--device", args.device,
            "--dataset-suite", args.real_dataset_suite,
            "--seeds", args.real_seeds,
            "--test-size", str(args.real_test_size),
            "--max-rows", str(args.real_max_rows),
            "--marginal-output-csv", str(output_dir / "marginal_real.csv"),
            "--marginal-output-summary-csv", str(output_dir / "marginal_real_summary.csv"),
            "--supervised-output-csv", str(output_dir / "class_invariant_supervised_real.csv"),
            "--supervised-output-summary-csv", str(output_dir / "class_invariant_supervised_real_summary.csv"),
            "--comparison-output-csv", str(output_dir / "real_paired_comparison.csv"),
        ]
        if args.checkpoint is not None:
            paired_command.extend(("--checkpoint", args.checkpoint))
        if args.real_datasets is not None:
            paired_command.extend(("--datasets", args.real_datasets))
        run(paired_command)
    else:
        print("Skipping real evaluation until both marginal and supervised checkpoints exist.", flush=True)

    manifest["shared_fixed_banks"] = {
        "model_seed": args.model_seed,
        "synthetic_validation_seed": args.validation_seed,
        "synthetic_test_seed": args.test_seed,
        "real_split_seeds": args.real_seeds,
        "real_dataset_suite": args.real_dataset_suite,
        "synthetic_validation_bank": str(validation_bank),
        "synthetic_validation_bank_sha256": sha256(validation_bank) if validation_bank.is_file() else None,
        "synthetic_test_bank": str(test_bank),
        "synthetic_test_bank_sha256": sha256(test_bank) if test_bank.is_file() else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Completed requested conditions; manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
