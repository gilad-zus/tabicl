"""Run the TFM-Retouche-style DirectSpline experiment without TabArena.

Only the small OpenML client is required.  It downloads the public TabArena
v0.1 suite (OpenML suite 457), uses each task's published split
``repeat=0, fold=0, sample=0`` as the untouched outer test set, and performs
eight guarded training bags locally.

The output directory is self-contained: it records the task IDs, split hashes,
frozen configuration draws, per-bag validation/test predictions, model choices,
and a paired-Elo report.  It never imports TabArena, AutoGluon, or Ray.

Examples
--------
Install the only additional dependency in the existing server environment::

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -m pip install openml

First run a one-task smoke test (this executes only the predeclared default)::

    python scripts/direct_spline_openml_lite.py \
      --output-dir results/openml_direct_spline/smoke --task-id 363621 --device cuda

Then run the full default arm, followed by the shared 10-config tuning arm::

    python scripts/direct_spline_openml_lite.py \
      --output-dir results/openml_direct_spline/default --device cuda

    python scripts/direct_spline_openml_lite.py \
      --output-dir results/openml_direct_spline/tuned --device cuda \
      --n-random-configs 10 --tuning-seed 20260813

Do not alter the default configuration after inspecting a smoke-test outer-test
score.  Its purpose is only to verify data access, memory, and artifact layout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tabicl._experiments.direct_spline_openml import (
    TABARENA_V0PT1_OPENML_SUITE_ID,
    _json_dump,
    _resolve_device,
    load_tabarena_openml_task,
    run_task_config,
    summarize_experiment,
    summarize_task_tuning,
    tabarena_v0pt1_task_ids,
)
from tabicl._experiments.direct_spline_protocol import (
    DEFAULT_DIRECT_SPLINE_CONFIG,
    shared_random_direct_spline_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-id", action="append", type=int,
        help="OpenML task ID. Repeat for a pilot; omit to run the public 51-task suite.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Take the first N suite tasks; for smoke tests only.")
    parser.add_argument("--outer-repeat", type=int, default=0)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--outer-sample", type=int, default=0)
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--n-random-configs", type=int, default=0)
    parser.add_argument("--tuning-seed", type=int, default=20_260_813)
    parser.add_argument("--ensemble-rounds", type=int, default=None, help="Defaults to twice the number of configs.")
    parser.add_argument("--protocol-seed", type=int, default=0, help="Controls shared bags and support contexts, not HPO draws.")
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse completed bag/config artifacts from this exact output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write the frozen manifest but do not download task data or run fits.")
    return parser.parse_args()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _configs(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    default = dict(DEFAULT_DIRECT_SPLINE_CONFIG)
    random = shared_random_direct_spline_configs(args.n_random_configs, seed=args.tuning_seed)
    return ["D", *(f"R{index + 1}" for index in range(len(random)))], [default, *random]


def _validate(args: argparse.Namespace) -> None:
    if args.bags < 2:
        raise ValueError("--bags must be at least two")
    if args.n_random_configs < 0 or args.bootstrap_rounds <= 0:
        raise ValueError("random-config and bootstrap counts must be non-negative/positive")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive")
    if min(args.outer_repeat, args.outer_fold, args.outer_sample) < 0:
        raise ValueError("outer repeat/fold/sample values must be non-negative")


def main() -> None:
    args = parse_args()
    _validate(args)
    labels, configs = _configs(args)
    task_ids = list(args.task_id) if args.task_id else tabarena_v0pt1_task_ids()
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task IDs must be unique")
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "experiment_manifest.json"
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_data = previous.get("data_source", {})
        if (
            prior_data.get("task_ids") != task_ids
            or prior_data.get("outer_split") != {
                "repeat": args.outer_repeat,
                "fold": args.outer_fold,
                "sample": args.outer_sample,
            }
            or previous.get("inner_bags") != args.bags
            or previous.get("config_labels") != labels
            or previous.get("configs") != configs
        ):
            raise ValueError(
                "--resume output directory was created for a different task/split/bag/configuration manifest; "
                "choose the original arguments or a new --output-dir."
            )
    manifest = {
        "experiment": "DirectSpline OpenML TabArena-v0.1 Lite reproduction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _git_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "data_source": {
            "provider": "OpenML",
            "suite_id": TABARENA_V0PT1_OPENML_SUITE_ID,
            "task_ids": task_ids,
            "outer_split": {
                "repeat": args.outer_repeat,
                "fold": args.outer_fold,
                "sample": args.outer_sample,
            },
        },
        "inner_bags": args.bags,
        "protocol_seed": args.protocol_seed,
        "guard": {
            "binary": "1 - ROC-AUC",
            "multiclass": "log loss",
            "regression": "MSE",
            "required_relative_improvement": DEFAULT_DIRECT_SPLINE_CONFIG["guard_relative_improvement"],
        },
        "leaderboard_metric": {"binary": "1 - ROC-AUC", "multiclass": "log loss", "regression": "RMSE"},
        "config_labels": labels,
        "configs": configs,
        "outer_test_policy": "never read by preprocessing fitting, adapter optimisation, guard, HPO, or ensembling",
        "absolute_elo_note": (
            "This run computes paired Elo deltas versus its own frozen TabICL identity baseline. "
            "Those deltas cannot be compared numerically with absolute ELO on TabArena's large published method pool."
        ),
    }
    _json_dump(manifest_path, manifest)
    print(f"Wrote frozen manifest for {len(task_ids)} task(s): {manifest_path}", flush=True)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, default=str), flush=True)
        return
    device = _resolve_device(args.device)
    task_summaries: list[dict[str, Any]] = []
    ensemble_rounds = args.ensemble_rounds or max(1, 2 * len(configs))
    for task_id in task_ids:
        task = load_tabarena_openml_task(
            task_id,
            outer_repeat=args.outer_repeat,
            outer_fold=args.outer_fold,
            outer_sample=args.outer_sample,
        )
        print(
            f"[task={task.task_id} dataset={task.dataset_name}] "
            f"{task.problem_type}; outer train={len(task.y_train)}, test={len(task.y_test)}, "
            f"features={task.x_train.shape[1]}",
            flush=True,
        )
        for label, config in zip(labels, configs, strict=True):
            print(f"[task={task.task_id} config={label}] starting/recovering {args.bags} bags", flush=True)
            result = run_task_config(
                task=task,
                label=label,
                config=config,
                output_dir=args.output_dir,
                bags=args.bags,
                protocol_seed=args.protocol_seed,
                device=device,
                classifier_checkpoint=args.classifier_checkpoint,
                regressor_checkpoint=args.regressor_checkpoint,
                resume=args.resume,
            )
            print(
                f"[task={task.task_id} config={label}] "
                f"guarded validation={result['validation']['guarded']['deployment_error']:.6g}; "
                f"guarded test={result['test']['guarded']['benchmark_error']:.6g}",
                flush=True,
            )
        task_summary = summarize_task_tuning(
            task=task,
            config_labels=labels,
            output_dir=args.output_dir,
            ensemble_rounds=ensemble_rounds,
        )
        task_summaries.append(task_summary)
        _json_dump(args.output_dir / "run_progress.json", {"completed_task_ids": [item["task_id"] for item in task_summaries]})
    summary = summarize_experiment(
        task_summaries=task_summaries,
        output_dir=args.output_dir,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_seed=args.protocol_seed,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
