"""Anchor saved DirectSpline results with deterministic Random Forest fits.

This is a CPU-only scoring experiment.  It loads the completed DirectSpline
task summaries, replays their exact OpenML outer split, fits one predeclared
Random Forest on all outer-training rows, and reports a three-method local
Bradley--Terry Elo board anchored at ``RandomForest_D = 1000``.

It does *not* reproduce Retouche's published Elo: this run has a smaller,
user-selected task pool and only three methods.  It is useful as a transparent
local reference scale, while a Retouche-comparable result still needs the
official 51-task suite and published method pool.

Example (do not run until the code and command are reviewed)::

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_random_forest_anchor.py \
      --source-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828 \
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche_rf_anchor/multiclass_seed20260828 \
      --n-jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tabicl._experiments.direct_spline_openml import _json_dump, load_tabarena_openml_task
from tabicl._experiments.direct_spline_random_forest_anchor import (
    DEFAULT_BT_CONFIG,
    METHOD_DIRECT_SPLINE,
    METHOD_RANDOM_FOREST,
    METHOD_TABICL,
    bradley_terry_fit_config,
    bootstrap_bradley_terry_elo,
    fit_bradley_terry_elo,
    fit_random_forest_task,
    load_random_forest_task_result,
    make_anchor_manifest,
    resolve_random_forest_config,
    save_random_forest_task_result,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direct-spline-arm", default="guarded_default")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--bootstrap-rounds", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path} ({type(error).__name__}: {error})") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _source_task_records(source_dir: Path, *, direct_spline_arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_mapping(source_dir / "experiment_manifest.json", label="source manifest")
    immutable = manifest.get("immutable_run")
    if not isinstance(immutable, dict):
        raise ValueError("source manifest has no immutable_run")
    data_source = immutable.get("data_source")
    if not isinstance(data_source, dict) or not isinstance(data_source.get("outer_split"), dict):
        raise ValueError("source manifest has no frozen outer split")
    outer_split = data_source["outer_split"]
    task_paths = sorted((source_dir / "task_summaries").glob("task_*.json"))
    if not task_paths:
        raise ValueError("source directory has no completed task summaries")
    records: list[dict[str, Any]] = []
    for path in task_paths:
        summary = _load_mapping(path, label="source task summary")
        standard = summary.get("standard_tabarena")
        candidate = summary.get(direct_spline_arm)
        if not isinstance(standard, dict) or not isinstance(candidate, dict):
            raise ValueError(
                f"{path} lacks standard_tabarena or requested DirectSpline arm {direct_spline_arm!r}"
            )
        records.append(
            {
                "task_id": int(summary["task_id"]),
                "dataset_id": int(summary["dataset_id"]),
                "dataset_name": str(summary["dataset_name"]),
                "problem_type": str(summary["problem_type"]),
                "n_classes": summary.get("n_classes"),
                "outer_split_hash": str(summary["outer_split_hash"]),
                "task_summary_path": str(path),
                "tabicl_benchmark_error": float(standard["benchmark_error"]),
                "direct_spline_benchmark_error": float(candidate["benchmark_error"]),
            }
        )
    task_ids = [record["task_id"] for record in records]
    frozen_ids = [int(value) for value in data_source.get("task_ids", [])]
    if sorted(task_ids) != sorted(frozen_ids):
        raise ValueError("completed source task summaries do not match the source manifest's frozen task ids")
    for name in ("repeat", "fold", "sample"):
        if name not in outer_split:
            raise ValueError(f"source manifest outer split has no {name}")
    return manifest, records


def _rf_config(args: argparse.Namespace) -> dict[str, Any]:
    return {"n_estimators": int(args.n_estimators)}


def _write_summary(
    *,
    output_dir: Path,
    task_rows: list[dict[str, Any]],
    bt_config: dict[str, Any],
    bootstrap_rounds: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    errors = {
        METHOD_RANDOM_FOREST: np.asarray([row["random_forest_benchmark_error"] for row in task_rows], dtype=float),
        METHOD_TABICL: np.asarray([row["tabicl_benchmark_error"] for row in task_rows], dtype=float),
        METHOD_DIRECT_SPLINE: np.asarray([row["direct_spline_benchmark_error"] for row in task_rows], dtype=float),
    }
    fit_config = bradley_terry_fit_config(bt_config)
    board = fit_bradley_terry_elo(errors_by_method=errors, **fit_config)
    bootstrap = bootstrap_bradley_terry_elo(
        errors_by_method=errors,
        rounds=bootstrap_rounds,
        seed=bootstrap_seed,
        **fit_config,
    )
    for method, interval in bootstrap["ratings"].items():
        board["ratings"][method]["bootstrap_95"] = interval
    csv_path = output_dir / "task_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(dict.fromkeys(key for row in task_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(task_rows)
    summary = {
        "scope": (
            "Local three-method Elo anchored to the sklearn Random Forest. This is not an absolute "
            "Retouche/TabArena leaderboard result."
        ),
        "n_tasks": len(task_rows),
        "metric_note": "Binary: 1-AUC; multiclass: log loss; regression: RMSE.",
        "methods": [METHOD_RANDOM_FOREST, METHOD_TABICL, METHOD_DIRECT_SPLINE],
        "bradley_terry": board,
        "task_results_csv": str(csv_path),
    }
    _json_dump(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    if args.bootstrap_rounds < 1:
        raise ValueError("--bootstrap-rounds must be positive")
    source_manifest, source_records = _source_task_records(args.source_dir, direct_spline_arm=args.direct_spline_arm)
    immutable = source_manifest["immutable_run"]
    outer_split = immutable["data_source"]["outer_split"]
    random_forest_config = resolve_random_forest_config(_rf_config(args), n_jobs=args.n_jobs)
    bt_config = dict(DEFAULT_BT_CONFIG)
    expected_manifest = make_anchor_manifest(
        source_dir=args.source_dir,
        source_manifest=source_manifest,
        direct_spline_arm=args.direct_spline_arm,
        random_forest_config=random_forest_config,
        bt_config=bt_config,
        task_records=source_records,
    )
    manifest_path = args.output_dir / "experiment_manifest.json"
    if manifest_path.exists():
        existing = _load_mapping(manifest_path, label="anchor manifest")
        if existing != expected_manifest:
            raise ValueError("anchor output directory belongs to a different immutable experiment; choose a new --output-dir")
        if not args.resume:
            raise ValueError("anchor output directory already exists; pass --resume to reuse completed Random Forest tasks")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(manifest_path, expected_manifest)

    rows: list[dict[str, Any]] = []
    for position, record in enumerate(source_records, start=1):
        task = load_tabarena_openml_task(
            int(record["task_id"]),
            outer_repeat=int(outer_split["repeat"]),
            outer_fold=int(outer_split["fold"]),
            outer_sample=int(outer_split["sample"]),
        )
        if task.outer_split_hash != record["outer_split_hash"]:
            raise ValueError(f"OpenML outer split changed for task {task.task_id}; refusing to mix results")
        if args.resume:
            try:
                rf_result = load_random_forest_task_result(
                    output_dir=args.output_dir,
                    task=task,
                    expected_config=random_forest_config,
                )
                status = "reused"
            except FileNotFoundError:
                rf_result = fit_random_forest_task(task=task, config=random_forest_config, n_jobs=args.n_jobs)
                save_random_forest_task_result(output_dir=args.output_dir, result=rf_result)
                status = "completed"
        else:
            rf_result = fit_random_forest_task(task=task, config=random_forest_config, n_jobs=args.n_jobs)
            save_random_forest_task_result(output_dir=args.output_dir, result=rf_result)
            status = "completed"
        print(
            f"[{position}/{len(source_records)}] {status}: task {task.task_id} {task.dataset_name} "
            f"RF benchmark error={rf_result.metadata['benchmark_error']:.8g}",
            flush=True,
        )
        rows.append(
            {
                **record,
                "random_forest_benchmark_error": float(rf_result.metadata["benchmark_error"]),
                "random_forest_deployment_error": float(rf_result.metadata["deployment_error"]),
            }
        )
    summary = _write_summary(
        output_dir=args.output_dir,
        task_rows=rows,
        bt_config=bt_config,
        bootstrap_rounds=int(args.bootstrap_rounds),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps(summary["bradley_terry"], indent=2), flush=True)


if __name__ == "__main__":
    main()
