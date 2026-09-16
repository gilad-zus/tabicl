"""Compare an arctan coordinate map with and without learned spline shape.

Both arms inherit the same trainable location/scale, low-rank mixing, training
schedule, context expansion, and OOF blend selection.  The only difference is
whether the cubic spline residual is trainable.  This isolates the incremental
value of spline shape after replacing the hard-clamped residual coordinate by
a smooth bounded arctan coordinate.
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
    "arctan_no_spline": "affine_mixing",
    "arctan_spline": "full_spline",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--query-fraction-min", type=float, default=0.05)
    parser.add_argument("--query-fraction-max", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_id)) != len(args.task_id):
        raise ValueError("--task-id values must be unique")
    if args.bags < 2:
        raise ValueError("--bags must be at least two")
    if not 0.0 < args.query_fraction_min <= args.query_fraction_max < 1.0:
        raise ValueError("query fractions must satisfy 0 < min <= max < 1")
    return args


def _prepare_manifest(args: argparse.Namespace) -> None:
    script = Path(__file__)
    manifest = {
        "experiment": "arctan coordinate mapping spline-shape ablation",
        "source_dir": str(args.source_dir.resolve()),
        "task_ids": sorted(int(task_id) for task_id in args.task_id),
        "config_label": str(args.config_label),
        "protocol_seed": int(args.protocol_seed),
        "bags": int(args.bags),
        "query_fraction_range": [float(args.query_fraction_min), float(args.query_fraction_max)],
        "coordinate_mapping": "arctan",
        "arms": ARMS,
        "causal_contrast": (
            "Both arms train location/scale and cross-column mixing. Only arctan_spline "
            "trains the cubic spline shape; arctan_no_spline freezes it at identity."
        ),
        "implementation_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        if _canonical_json(_load_json(path)) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different arctan ablation")
        if not args.resume:
            raise ValueError("output directory exists; pass --resume")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def _run_arms(args: argparse.Namespace) -> None:
    runner = Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")
    for position, (label, adapter_arm) in enumerate(ARMS.items(), start=1):
        command = [
            sys.executable,
            str(runner),
            "--source-dir",
            str(args.source_dir),
            "--output-dir",
            str(args.output_dir / label),
            "--config-label",
            str(args.config_label),
            "--protocol-seed",
            str(args.protocol_seed),
            "--bags",
            str(args.bags),
            "--adapter-arm",
            adapter_arm,
            "--coordinate-mapping",
            "arctan",
            "--query-fraction-min",
            str(args.query_fraction_min),
            "--query-fraction-max",
            str(args.query_fraction_max),
            "--device",
            str(args.device),
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
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for label in ARMS:
        records = _load_json(args.output_dir / label / "task_summaries.json")
        by_arm[label] = {int(record["task_id"]): record for record in records}
    expected = set(int(task_id) for task_id in args.task_id)
    if any(set(records) != expected for records in by_arm.values()):
        raise ValueError("one or more arctan arms has an incomplete or unexpected task set")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(expected):
        control = by_arm["arctan_no_spline"][task_id]
        spline = by_arm["arctan_spline"][task_id]
        if control["problem_type"] != spline["problem_type"]:
            raise ValueError(f"task {task_id} changed problem type between arms")
        for split in ("oof", "outer_test"):
            if _metric(control, split, "identity") != _metric(spline, split, "identity"):
                raise ValueError(f"task {task_id} identity baseline differs between matched arms on {split}")
        row: dict[str, Any] = {
            "task_id": task_id,
            "dataset_name": control["dataset_name"],
            "problem_type": control["problem_type"],
        }
        for split in ("oof", "outer_test"):
            for method in ("raw_spline", "selected_blend"):
                control_error = _metric(control, split, method)
                spline_error = _metric(spline, split, method)
                prefix = f"{split}_{method}"
                row[f"{prefix}_arctan_no_spline"] = control_error
                row[f"{prefix}_arctan_spline"] = spline_error
                row[f"{prefix}_relative_spline_gain"] = (
                    (control_error - spline_error) / control_error if control_error != 0.0 else 0.0
                )
        row["outer_test_identity"] = _metric(control, "outer_test", "identity")
        rows.append(row)

    def comparison(method: str) -> dict[str, Any]:
        control = [float(row[f"outer_test_{method}_arctan_no_spline"]) for row in rows]
        spline = [float(row[f"outer_test_{method}_arctan_spline"]) for row in rows]
        wins = sum(candidate < reference for reference, candidate in zip(control, spline))
        losses = sum(candidate > reference for reference, candidate in zip(control, spline))
        ties = len(rows) - wins - losses
        return {
            "spline_wins": wins,
            "ties": ties,
            "spline_losses": losses,
            "mean_relative_spline_gain": sum(
                (reference - candidate) / reference if reference != 0.0 else 0.0
                for reference, candidate in zip(control, spline)
            )
            / len(rows),
        }

    result = {
        "n_tasks": len(rows),
        "task_results": rows,
        "outer_test_spline_vs_no_spline": {
            method: comparison(method) for method in ("raw_spline", "selected_blend")
        },
        "interpretation": (
            "The paired difference isolates learned spline shape inside the arctan pipeline. "
            "Outer-test labels are report-only; each arm's blend strength was selected from its OOF rows."
        ),
    }
    _write_json(args.output_dir / "arctan_ablation_summary.json", result)
    with (args.output_dir / "arctan_ablation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    args = _parse_args()
    _prepare_manifest(args)
    _run_arms(args)
    result = _aggregate(args)
    print(json.dumps(result["outer_test_spline_vs_no_spline"], indent=2), flush=True)


if __name__ == "__main__":
    main()
