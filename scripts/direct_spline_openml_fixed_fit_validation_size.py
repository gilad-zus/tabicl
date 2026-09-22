"""Test whether checkpoint-selection evidence, rather than fitted rows, is limiting DirectSpline.

For each task and split seed this experiment partitions the published outer-training
rows into a fixed 60% adapter-fit set and a 40% held-out selection pool.  The same
adapter trajectory is trained once on the 60%.  Two nested rules then choose among
its identity state and checkpoint states:

* ``small`` uses a stratified 20% subset of the outer-training rows;
* ``large`` uses the full 40% selection pool, including the small subset.

The selected states are evaluated unchanged with the same 60%-fit TabICL context on
the untouched published outer test set.  Thus small and large decisions differ only
in the amount of labelled checkpoint-selection evidence.  There is deliberately no
full-context refit: a refit would add a second training trajectory and defeat that
control.  This is a mechanism diagnostic, not a replacement for the normal 80/20
training/validation protocol or a fresh benchmark result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from tabicl._experiments.direct_spline_openml import (
    _json_dump,
    _resolve_device,
    load_tabarena_openml_task,
)
from tabicl._experiments.direct_spline_openml_standard import (
    summarize_fixed_fit_validation_size_experiment,
    summarize_fixed_fit_validation_size_task,
    run_task_fixed_fit_validation_size_standard,
    standard_direct_spline_config,
)


DEFAULT_MULTICLASS_TASK_IDS = (4602, 75158, 167186, 361539)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument(
        "--split-seed", type=int, action="append", default=None,
        help="Repeat to run another independently drawn nested selection split (default: two seeds).",
    )
    parser.add_argument("--protocol-seed", type=int, default=20_260_922)
    parser.add_argument("--adapter-seed", type=int, default=20_260_922)
    parser.add_argument("--adapter-steps", type=int, default=500)
    parser.add_argument("--selection-checkpoint-interval", type=int, default=25)
    parser.add_argument("--small-validation-fraction", type=float, default=0.20)
    parser.add_argument("--large-validation-fraction", type=float, default=0.40)
    parser.add_argument("--cosine-min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--selection-relative-improvement", type=float, default=0.005)
    parser.add_argument("--identity-regularization", type=float, default=0.0)
    parser.add_argument("--bootstrap-rounds", type=int, default=2_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--regressor-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.task_id = list(DEFAULT_MULTICLASS_TASK_IDS if args.task_id is None else args.task_id)
    args.split_seed = [20_260_922, 20_260_923] if args.split_seed is None else args.split_seed
    if len(set(args.task_id)) != len(args.task_id) or len(set(args.split_seed)) != len(args.split_seed):
        raise ValueError("task IDs and split seeds must each be unique")
    if args.adapter_steps <= 0 or args.selection_checkpoint_interval <= 0:
        raise ValueError("adapter steps and checkpoint interval must be positive")
    if not 0.0 < args.small_validation_fraction < args.large_validation_fraction < 0.5:
        raise ValueError("fractions must satisfy 0 < small < large < 0.5")
    if not 0.0 < args.cosine_min_lr_ratio <= 1.0:
        raise ValueError("--cosine-min-lr-ratio must lie in (0, 1]")
    if not 0.0 <= args.selection_relative_improvement < 1.0:
        raise ValueError("--selection-relative-improvement must lie in [0, 1)")
    if args.identity_regularization < 0.0 or args.bootstrap_rounds < 1:
        raise ValueError("regularization must be non-negative and bootstrap rounds positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(args: argparse.Namespace) -> dict[str, Any]:
    config = standard_direct_spline_config(
        context_cap=None, train_context_rows=None, adapter_steps=args.adapter_steps,
        validation_interval=args.selection_checkpoint_interval,
    )
    config.update(
        {
            "random_state": int(args.adapter_seed),
            "adapter_patience": None,
            "selection_checkpoint_interval": int(args.selection_checkpoint_interval),
            "cosine_schedule_steps": int(args.adapter_steps),
            "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
            "selection_relative_improvement": float(args.selection_relative_improvement),
            "guard_relative_improvement": float(args.selection_relative_improvement),
            "identity_regularization": float(args.identity_regularization),
        }
    )
    return config


def _manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    core = root / "src" / "tabicl" / "_experiments" / "direct_spline_openml_standard.py"
    return {
        "experiment": "fixed-fit nested validation-size checkpoint-selection diagnostic",
        "task_ids": args.task_id,
        "split_seeds": args.split_seed,
        "outer_split": {"repeat": 0, "fold": 0, "sample": 0},
        "fit_fraction": 1.0 - float(args.large_validation_fraction),
        "small_selection_fraction": float(args.small_validation_fraction),
        "large_selection_fraction": float(args.large_validation_fraction),
        "adapter_protocol_seed": int(args.protocol_seed),
        "config": config,
        "test_label_policy": "outer test labels are read only after both nested selectors freeze predictions",
        "no_full_context_refit": True,
        "implementation_sha256": {
            "launcher": _sha256(Path(__file__).resolve()),
            "standard_runner": _sha256(core),
        },
    }


def main() -> None:
    args = _parse_args()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    device = _resolve_device(args.device)
    config = _config(args)
    manifest = _manifest(args, config)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf8")).hexdigest()
    manifest_path = args.output_dir / "experiment_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf8"))
        if existing != manifest:
            raise ValueError("output directory belongs to a different immutable experiment; choose another --output-dir")
        if not args.resume:
            raise ValueError("output directory already exists; pass --resume for this exact experiment")
    elif args.resume:
        raise FileNotFoundError("cannot safely resume without experiment_manifest.json")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(manifest_path, manifest)

    combined: dict[str, Any] = {"manifest": manifest, "runs": []}
    for split_seed in args.split_seed:
        seed_dir = args.output_dir / f"split_seed_{split_seed}"
        summaries: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for ordinal, task_id in enumerate(args.task_id, start=1):
            task = load_tabarena_openml_task(task_id, outer_repeat=0, outer_fold=0, outer_sample=0)
            if task.problem_type != "multiclass":
                raise ValueError(f"task {task_id} is {task.problem_type}; this multiclass diagnostic refuses it")
            print(f"[{split_seed} {ordinal}/{len(args.task_id)}] task {task.task_id} {task.dataset_name}", flush=True)
            try:
                task_result = run_task_fixed_fit_validation_size_standard(
                    task=task, config_labels=["direct_spline_k20"], configs=[config],
                    small_validation_fraction=args.small_validation_fraction,
                    large_validation_fraction=args.large_validation_fraction,
                    validation_seed=split_seed, output_dir=seed_dir,
                    protocol_seed=args.protocol_seed, device=device,
                    classifier_checkpoint=args.classifier_checkpoint,
                    regressor_checkpoint=args.regressor_checkpoint,
                    resume=args.resume, run_fingerprint_hash=fingerprint,
                    progress=lambda event: print(json.dumps(event, sort_keys=True), flush=True),
                )
                summaries.append(
                    summarize_fixed_fit_validation_size_task(
                        task=task, output_dir=seed_dir, task_result=task_result
                    )
                )
            except torch.OutOfMemoryError as error:
                skipped.append({"task_id": task.task_id, "dataset_name": task.dataset_name, "reason": "cuda_out_of_memory", "error": str(error)})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if not summaries:
            raise RuntimeError(f"all tasks failed for split seed {split_seed}")
        summary = summarize_fixed_fit_validation_size_experiment(
            task_summaries=summaries, output_dir=seed_dir,
            bootstrap_rounds=args.bootstrap_rounds, bootstrap_seed=split_seed,
            skipped_tasks=skipped,
            task_eligibility={"requested_task_ids": args.task_id, "problem_type": "multiclass"},
        )
        combined["runs"].append({"split_seed": split_seed, "summary": summary})
        _json_dump(args.output_dir / "combined_summary.json", combined)
        print(json.dumps(summary["paired_results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
