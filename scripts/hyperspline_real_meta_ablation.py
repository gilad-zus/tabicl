"""Run the dataset-disjoint real-meta HyperSpline experiment end to end.

It creates the real train/validation banks once, trains the mixed real/synthetic
model, then evaluates the selected checkpoint on the unchanged final seven-dataset
paired suite.  The final suite is intentionally not used before the last call.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marginal-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-openml-ids", default="11,23,29,37,44,50,54,59")
    parser.add_argument("--validation-openml-ids", default="14,16,18,22")
    parser.add_argument("--episodes-per-dataset", type=int, default=4)
    parser.add_argument("--min-train-datasets", type=int, default=20)
    parser.add_argument("--min-validation-datasets", type=int, default=8)
    parser.add_argument("--rebuild-real-banks", action="store_true")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--real-episode-fraction", type=float, default=0.75)
    parser.add_argument("--marginal-warmup-steps", type=int, default=2_000)
    parser.add_argument("--raw-lr", type=float, default=1e-3)
    parser.add_argument("--marginal-lr", type=float, default=1e-4)
    parser.add_argument("--residual-transform-regularization", type=float, default=1e-3)
    parser.add_argument("--marginal-trust-region-regularization", type=float, default=1e-3)
    parser.add_argument("--context-subset-consistency-regularization", type=float, default=1e-2)
    parser.add_argument("--context-subset-fraction", type=float, default=0.70)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--prior-type", default="mix_scm")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--synthetic-long-sequence-length", type=int, default=1024)
    parser.add_argument("--synthetic-long-fraction", type=float, default=0.75)
    parser.add_argument("--context-fraction", type=float, default=0.70)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--prior-n-jobs", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--final-seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_bank, validation_bank = output / "real_meta_train_bank.pt", output / "real_meta_validation_bank.pt"
    common = ["--checkpoint-version", args.checkpoint_version, "--device", args.device]
    if args.checkpoint:
        common += ["--checkpoint", args.checkpoint]
    if args.rebuild_real_banks or not train_bank.exists() or not validation_bank.exists():
        run([
            sys.executable, "scripts/hyperspline_real_task_bank.py",
            "--train-openml-ids", args.train_openml_ids,
            "--validation-openml-ids", args.validation_openml_ids,
            "--episodes-per-dataset", str(args.episodes_per_dataset),
            "--min-train-datasets", str(args.min_train_datasets),
            "--min-validation-datasets", str(args.min_validation_datasets),
            "--train-output", str(train_bank), "--validation-output", str(validation_bank),
            "--availability-manifest", str(output / "real_meta_dataset_availability.json"),
        ])
    else:
        print(f"Reusing real meta banks: {train_bank} and {validation_bank}", flush=True)
    checkpoint = output / "real_meta_raw_context.pt"
    run([
        sys.executable, "scripts/hyperspline_real_meta_train.py", *common,
        "--marginal-checkpoint", str(args.marginal_checkpoint),
        "--real-train-bank", str(train_bank), "--real-validation-bank", str(validation_bank),
        "--train-steps", str(args.train_steps), "--tasks-per-step", str(args.tasks_per_step),
        "--real-episode-fraction", str(args.real_episode_fraction),
        "--marginal-warmup-steps", str(args.marginal_warmup_steps),
        "--raw-lr", str(args.raw_lr), "--marginal-lr", str(args.marginal_lr),
        "--residual-transform-regularization", str(args.residual_transform_regularization),
        "--marginal-trust-region-regularization", str(args.marginal_trust_region_regularization),
        "--context-subset-consistency-regularization", str(args.context_subset_consistency_regularization),
        "--context-subset-fraction", str(args.context_subset_fraction),
        "--validate-every", str(args.validate_every), "--prior-type", args.prior_type,
        "--sequence-length", str(args.sequence_length), "--context-fraction", str(args.context_fraction),
        "--synthetic-long-sequence-length", str(args.synthetic_long_sequence_length),
        "--synthetic-long-fraction", str(args.synthetic_long_fraction),
        "--min-features", str(args.min_features), "--max-features", str(args.max_features),
        "--max-classes", str(args.max_classes), "--prior-n-jobs", str(args.prior_n_jobs),
        "--model-seed", str(args.model_seed), "--train-seed", str(args.train_seed),
        "--output-checkpoint", str(checkpoint),
        "--output-train-csv", str(output / "real_meta_training.csv"),
        "--output-real-validation-csv", str(output / "real_meta_validation.csv"),
    ])
    run([
        sys.executable, "scripts/hyperspline_real_paired_eval.py", *common,
        "--marginal-checkpoint", str(args.marginal_checkpoint),
        "--supervised-checkpoint", str(checkpoint), "--seeds", args.final_seeds,
        "--marginal-output-csv", str(output / "real_meta_final_marginal.csv"),
        "--marginal-output-summary-csv", str(output / "real_meta_final_marginal_summary.csv"),
        "--supervised-output-csv", str(output / "real_meta_final_supervised.csv"),
        "--supervised-output-summary-csv", str(output / "real_meta_final_supervised_summary.csv"),
        "--comparison-output-csv", str(output / "real_meta_final_paired.csv"),
    ])


if __name__ == "__main__":
    main()
