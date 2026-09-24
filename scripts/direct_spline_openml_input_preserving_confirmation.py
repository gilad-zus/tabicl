"""Confirm input-preserving spline curvature on a fixed multiclass cohort.

Train matched line and cubic adapters from the same source run. Both begin at
TabICL's original numerical input, use the same protocol, and differ only in
whether spline curvature is trainable. Report raw curvature separately from
the independently validation-selected identity blends.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from direct_spline_openml_input_preserving_ablation import relative_error_reduction


ARMS = {"preserved_line": "direct_line", "preserved_spline": "direct_spline"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Completed constant-LR run on the same tasks and source.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--adapter-steps", type=int, default=500)
    parser.add_argument("--cosine-min-lr-ratio", type=float)
    parser.add_argument("--n-control-points", type=int, default=20)
    parser.add_argument("--query-fraction-min", type=float, default=0.05)
    parser.add_argument("--query-fraction-max", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_id)) != len(args.task_id):
        parser.error("--task-id values must be unique")
    if args.bags < 2 or args.adapter_steps < 1 or args.n_control_points < 4:
        parser.error("bags >= 2, adapter-steps >= 1, and n-control-points >= 4 are required")
    if not 0 < args.query_fraction_min <= args.query_fraction_max < 1:
        parser.error("query fraction range must lie inside (0, 1)")
    if args.cosine_min_lr_ratio is not None and not 0 < args.cosine_min_lr_ratio <= 1:
        parser.error("cosine-min-lr-ratio must lie inside (0, 1]")
    return args


def _validate_source_and_reference(args: argparse.Namespace) -> None:
    source = _load_json(args.source_dir / "experiment_manifest.json")
    immutable = source["immutable_run"]
    if set(map(int, immutable["data_source"]["task_ids"])) != set(args.task_id):
        raise ValueError("source task IDs differ from the requested cohort")
    configs = immutable["configs"]
    if len(configs) != 1:
        raise ValueError("confirmation source must have exactly one D configuration")
    config = configs[0]
    expected = {
        "learning_rate": 0.005,
        "validation_interval": 10,
        "random_state": 0,
        "adapter_steps": args.adapter_steps,
        "n_control_points": args.n_control_points,
    }
    mismatched = [key for key, value in expected.items() if config.get(key) != value]
    if config.get("cosine_schedule_steps") is not None:
        mismatched.append("cosine_schedule_steps")
    if mismatched:
        raise ValueError(f"source training configuration changed: {', '.join(mismatched)}")
    if args.reference_dir is not None:
        reference = _load_json(args.reference_dir / "experiment_manifest.json")
        expected_reference = {
            "source_manifest_sha256": _sha256(args.source_dir / "experiment_manifest.json"),
            "task_ids": sorted(args.task_id),
            "config_label": args.config_label,
            "protocol_seed": args.protocol_seed,
            "bags": args.bags,
            "adapter_steps": args.adapter_steps,
            "n_control_points": args.n_control_points,
            "query_fraction_range": [args.query_fraction_min, args.query_fraction_max],
            "coordinate_mapping": "arctan",
            "preserve_input_base": True,
            "arms": ARMS,
        }
        different = [key for key, value in expected_reference.items() if reference.get(key) != value]
        if reference.get("cosine_min_lr_ratio") is not None:
            different.append("cosine_min_lr_ratio")
        if different:
            raise ValueError(f"constant-LR reference differs: {', '.join(different)}")
        if not (args.reference_dir / "confirmation_summary.json").is_file():
            raise FileNotFoundError(args.reference_dir / "confirmation_summary.json")


def _prepare_manifest(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve()
    manifest = {
        "experiment": "input-preserving multiclass curvature confirmation",
        "source_dir": str(args.source_dir.resolve()),
        "source_manifest_sha256": _sha256(args.source_dir / "experiment_manifest.json"),
        "reference_dir": None if args.reference_dir is None else str(args.reference_dir),
        "reference_summary_sha256": (
            None if args.reference_dir is None
            else _sha256(args.reference_dir / "confirmation_summary.json")
        ),
        "task_ids": sorted(args.task_id),
        "config_label": args.config_label,
        "protocol_seed": args.protocol_seed,
        "bags": args.bags,
        "adapter_steps": args.adapter_steps,
        "cosine_min_lr_ratio": args.cosine_min_lr_ratio,
        "inherited_training_random_state": 0,
        "inherited_validation_interval": 10,
        "inherited_learning_rate": 0.005,
        "n_control_points": args.n_control_points,
        "query_fraction_range": [args.query_fraction_min, args.query_fraction_max],
        "arms": ARMS,
        "coordinate_mapping": "arctan",
        "preserve_input_base": True,
        "confirmation_script_sha256": _sha256(here),
        "runner_sha256": _sha256(here.with_name("direct_spline_openml_crossfit_context_expansion.py")),
        "adapter_module_sha256": _sha256(here.parents[1] / "src/tabicl/_hyperspline/module.py"),
        "standard_adapter_sha256": _sha256(here.parents[1] / "src/tabicl/_experiments/direct_spline_openml_standard.py"),
    }
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        if _load_json(path) != manifest:
            raise ValueError("output directory has a different experiment manifest")
        if not args.resume:
            raise ValueError("output directory exists; pass --resume")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def _run_arms(args: argparse.Namespace) -> None:
    runner = Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")
    for arm_name, adapter_arm in ARMS.items():
        command = [
            sys.executable, str(runner),
            "--source-dir", str(args.source_dir),
            "--output-dir", str(args.output_dir / arm_name),
            "--config-label", args.config_label,
            "--protocol-seed", str(args.protocol_seed),
            "--bags", str(args.bags),
            "--adapter-arm", adapter_arm,
            "--coordinate-mapping", "arctan",
            "--preserve-input-base",
            "--n-control-points", str(args.n_control_points),
            "--adapter-steps", str(args.adapter_steps),
            "--query-fraction-min", str(args.query_fraction_min),
            "--query-fraction-max", str(args.query_fraction_max),
            "--device", args.device,
            "--resume",
        ]
        for task_id in args.task_id:
            command.extend(("--task-id", str(task_id)))
        if args.classifier_checkpoint is not None:
            command.extend(("--classifier-checkpoint", str(args.classifier_checkpoint)))
        if args.openml_cache_dir is not None:
            command.extend(("--openml-cache-dir", str(args.openml_cache_dir)))
        if args.cosine_min_lr_ratio is not None:
            command.extend(("--cosine-min-lr-ratio", str(args.cosine_min_lr_ratio)))
        print(f"Running {arm_name} on {len(args.task_id)} tasks", flush=True)
        subprocess.run(command, check=True)


def _task_records(directory: Path, expected: set[int]) -> dict[int, dict[str, Any]]:
    records = _load_json(directory / "task_summaries.json")
    by_id = {int(item["task_id"]): item for item in records}
    if len(records) != len(expected) or set(by_id) != expected:
        raise ValueError(f"incomplete or duplicated task records in {directory}")
    return by_id


def _error(record: dict[str, Any], split: str, method: str) -> float:
    return float(record["expanded_context"][split][method]["benchmark_error"])


def _summarize_pairs(rows: list[dict[str, Any]], reference: str, candidate: str) -> dict[str, Any]:
    pairs = [(float(row[reference]), float(row[candidate])) for row in rows]
    gains = [relative_error_reduction(ref, cand) for ref, cand in pairs]
    usable = [gain for gain in gains if gain is not None]
    wins = sum(cand < ref for ref, cand in pairs)
    losses = sum(cand > ref for ref, cand in pairs)
    return {
        "wins": wins, "losses": losses, "ties": len(pairs) - wins - losses,
        "mean_relative_gain": statistics.mean(usable) if usable else None,
        "median_relative_gain": statistics.median(usable) if usable else None,
    }


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    expected = set(args.task_id)
    line = _task_records(args.output_dir / "preserved_line", expected)
    spline = _task_records(args.output_dir / "preserved_spline", expected)
    rows = []
    for task_id in sorted(expected):
        left, right = line[task_id], spline[task_id]
        if left["problem_type"] != "multiclass" or right["problem_type"] != "multiclass":
            raise ValueError(f"task {task_id} is not multiclass")
        for key in ("dataset_id", "dataset_name", "outer_split_hash"):
            if left[key] != right[key]:
                raise ValueError(f"task {task_id} has different {key} across arms")
        row = {"task_id": task_id, "dataset_name": left["dataset_name"]}
        for split in ("oof", "outer_test"):
            if _error(left, split, "identity") != _error(right, split, "identity"):
                raise ValueError(f"task {task_id} identity error differs across arms on {split}")
            for method in ("identity", "raw_spline", "selected_blend"):
                row[f"{split}_{method}_line"] = _error(left, split, method)
                row[f"{split}_{method}_spline"] = _error(right, split, method)
        for arm, record in (("line", left), ("spline", right)):
            source = record["source_full_outer_training_tabiclv2"]
            if source is None:
                raise ValueError(f"task {task_id} has no ordinary full TabICLv2 baseline")
            row[f"outer_test_full_tabiclv2_{arm}"] = float(source["benchmark_error"])
            row[f"selected_alpha_{arm}"] = float(record["expanded_context"]["blend_selection"]["selected_alpha"])
        if row["outer_test_full_tabiclv2_line"] != row["outer_test_full_tabiclv2_spline"]:
            raise ValueError(f"task {task_id} full TabICLv2 error differs across arms")
        row["outer_test_full_tabiclv2"] = row.pop("outer_test_full_tabiclv2_line")
        row.pop("outer_test_full_tabiclv2_spline")
        for split in ("oof", "outer_test"):
            for method in ("raw_spline", "selected_blend"):
                row[f"{split}_{method}_curvature_gain"] = relative_error_reduction(
                    row[f"{split}_{method}_line"], row[f"{split}_{method}_spline"]
                )
        rows.append(row)
    if getattr(args, "reference_dir", None) is not None:
        old = _load_json(args.reference_dir / "confirmation_summary.json")
        old_rows = {int(item["task_id"]): item for item in old["task_results"]}
        if len(old["task_results"]) != len(expected) or set(old_rows) != expected:
            raise ValueError("constant-LR reference has a different task set")
        for row in rows:
            previous = old_rows[row["task_id"]]
            if row["dataset_name"] != previous["dataset_name"]:
                raise ValueError(f"task {row['task_id']} dataset name changed")
            if row["outer_test_full_tabiclv2"] != previous["outer_test_full_tabiclv2"]:
                raise ValueError(f"task {row['task_id']} full TabICLv2 baseline changed")
            for arm in ("line", "spline"):
                for method in ("raw_spline", "selected_blend"):
                    key = f"outer_test_{method}_{arm}"
                    row[f"{key}_constant_lr"] = float(previous[key])
                    row[f"{key}_cosine_gain"] = relative_error_reduction(
                        row[f"{key}_constant_lr"], row[key]
                    )
    summary = {
        "n_tasks": len(rows),
        "primary_contrast": "outer_test_raw_spline_spline versus outer_test_raw_spline_line",
        "task_results": rows,
        "comparisons": {
            "raw_curvature_vs_line": _summarize_pairs(rows, "outer_test_raw_spline_line", "outer_test_raw_spline_spline"),
            "selected_spline_vs_selected_line": _summarize_pairs(rows, "outer_test_selected_blend_line", "outer_test_selected_blend_spline"),
            "raw_spline_vs_full_tabiclv2": _summarize_pairs(rows, "outer_test_full_tabiclv2", "outer_test_raw_spline_spline"),
            "selected_spline_vs_full_tabiclv2": _summarize_pairs(rows, "outer_test_full_tabiclv2", "outer_test_selected_blend_spline"),
        },
        "interpretation": "Raw spline versus matched line attributes incremental curvature. A selected blend with alpha zero is an identity deployment and does not count as a spline contribution.",
    }
    if getattr(args, "reference_dir", None) is not None:
        summary["comparisons"]["cosine_vs_constant_line"] = _summarize_pairs(
            rows, "outer_test_raw_spline_line_constant_lr", "outer_test_raw_spline_line"
        )
        summary["comparisons"]["cosine_vs_constant_spline"] = _summarize_pairs(
            rows, "outer_test_raw_spline_spline_constant_lr", "outer_test_raw_spline_spline"
        )
    _write_json(args.output_dir / "confirmation_summary.json", summary)
    with (args.output_dir / "confirmation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    args = _parse_args()
    args.source_dir = args.source_dir.resolve()
    if args.reference_dir is not None:
        args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    _validate_source_and_reference(args)
    if args.preflight_only:
        print("Source and constant-LR reference match the fixed ten-task protocol.", flush=True)
        return
    _prepare_manifest(args)
    _run_arms(args)
    summary = _aggregate(args)
    print(json.dumps(summary["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
