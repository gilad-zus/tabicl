"""Test whether spline curvature helps after a direct line has already been learned.

Both continuation arms start from the exact A/B-selected checkpoints of a
completed direct-line run.  They receive identical continuation episodes and
optimization budgets.  One keeps the shape frozen as a line; the other enables
zero-initialized monotone spline curvature.  The inherited checkpoint at
continuation step zero remains selectable in both arms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ARMS = {
    "continued_line": "direct_line",
    "continued_spline": "direct_spline",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--initial-line-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--continuation-steps", type=int, default=250)
    parser.add_argument("--query-fraction-min", type=float, default=0.05)
    parser.add_argument("--query-fraction-max", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--regressor-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_id)) != len(args.task_id):
        raise ValueError("--task-id values must be unique")
    if args.bags < 2 or args.continuation_steps < 1:
        raise ValueError("--bags must be at least two and --continuation-steps positive")
    if not 0.0 < args.query_fraction_min <= args.query_fraction_max < 1.0:
        raise ValueError("query fractions must satisfy 0 < min <= max < 1")
    return args


def _prepare(args: argparse.Namespace) -> None:
    script = Path(__file__)
    initial_manifest = args.initial_line_dir / "experiment_manifest.json"
    manifest = {
        "experiment": "staged direct-arctan curvature ablation",
        "source_dir": str(args.source_dir.resolve()),
        "initial_line_dir": str(args.initial_line_dir.resolve()),
        "initial_line_manifest_sha256": hashlib.sha256(initial_manifest.read_bytes()).hexdigest(),
        "task_ids": sorted(args.task_id),
        "config_label": args.config_label,
        "protocol_seed": args.protocol_seed,
        "bags": args.bags,
        "continuation_steps": args.continuation_steps,
        "query_fraction_range": [args.query_fraction_min, args.query_fraction_max],
        "arms": ARMS,
        "causal_contrast": (
            "Same selected direct-line starting states and continuation episodes; only the "
            "continued-spline arm can learn curvature. Step zero is selectable."
        ),
        "implementation_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        if _canonical(_load(path)) != _canonical(manifest):
            raise ValueError("output directory belongs to a different staged experiment")
        if not args.resume:
            raise ValueError("output directory exists; pass --resume")
    else:
        _write(path, manifest)


def _run(args: argparse.Namespace) -> None:
    runner = Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")
    for position, (label, adapter_arm) in enumerate(ARMS.items(), 1):
        command = [
            sys.executable, str(runner),
            "--source-dir", str(args.source_dir),
            "--initial-adapter-dir", str(args.initial_line_dir),
            "--output-dir", str(args.output_dir / label),
            "--config-label", args.config_label,
            "--protocol-seed", str(args.protocol_seed),
            "--bags", str(args.bags),
            "--adapter-arm", adapter_arm,
            "--coordinate-mapping", "arctan",
            "--adapter-steps", str(args.continuation_steps),
            "--query-fraction-min", str(args.query_fraction_min),
            "--query-fraction-max", str(args.query_fraction_max),
            "--device", args.device,
            "--resume",
        ]
        for task_id in args.task_id:
            command.extend(("--task-id", str(task_id)))
        for flag, value in (
            ("--openml-cache-dir", args.openml_cache_dir),
            ("--classifier-checkpoint", args.classifier_checkpoint),
            ("--regressor-checkpoint", args.regressor_checkpoint),
        ):
            if value is not None:
                command.extend((flag, str(value)))
        print(f"[{position}/{len(ARMS)}] {label}", flush=True)
        subprocess.run(command, check=True)


def _metric(record: dict[str, Any], split: str, method: str) -> float:
    return float(record["expanded_context"][split][method]["benchmark_error"])


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    records = {
        arm: {int(row["task_id"]): row for row in _load(args.output_dir / arm / "task_summaries.json")}
        for arm in ARMS
    }
    expected = set(args.task_id)
    if any(set(rows) != expected for rows in records.values()):
        raise ValueError("a continuation arm has an incomplete task set")
    rows: list[dict[str, Any]] = []
    for task_id in sorted(expected):
        line = records["continued_line"][task_id]
        spline = records["continued_spline"][task_id]
        row: dict[str, Any] = {
            "task_id": task_id,
            "dataset_name": line["dataset_name"],
            "problem_type": line["problem_type"],
        }
        for split in ("oof", "outer_test"):
            for method in ("raw_spline", "selected_blend"):
                line_error = _metric(line, split, method)
                spline_error = _metric(spline, split, method)
                prefix = f"{split}_{method}"
                row[f"{prefix}_continued_line"] = line_error
                row[f"{prefix}_continued_spline"] = spline_error
                row[f"{prefix}_relative_curvature_gain"] = (
                    (line_error - spline_error) / line_error if line_error else 0.0
                )
        rows.append(row)
    comparisons = {}
    for method in ("raw_spline", "selected_blend"):
        pairs = [
            (row[f"outer_test_{method}_continued_line"], row[f"outer_test_{method}_continued_spline"])
            for row in rows
        ]
        wins = sum(spline < line for line, spline in pairs)
        losses = sum(spline > line for line, spline in pairs)
        comparisons[method] = {
            "spline_wins": wins,
            "ties": len(pairs) - wins - losses,
            "spline_losses": losses,
            "mean_relative_curvature_gain": sum(
                (line - spline) / line if line else 0.0 for line, spline in pairs
            ) / len(pairs),
        }
    result = {
        "n_tasks": len(rows),
        "task_results": rows,
        "outer_test_spline_vs_line_continuation": comparisons,
        "interpretation": (
            "Both arms inherit the same validation-selected direct-line checkpoints and use "
            "matched continuation episodes. Their difference isolates enabling curvature after "
            "the useful linear transformation has already been learned."
        ),
    }
    _write(args.output_dir / "staged_curvature_ablation_summary.json", result)
    with (args.output_dir / "staged_curvature_ablation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    args = _parse_args()
    _prepare(args)
    _run(args)
    result = _aggregate(args)
    print(json.dumps(result["outer_test_spline_vs_line_continuation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
