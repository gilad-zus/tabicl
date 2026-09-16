"""Run the three-column 4/20-control-point DirectSpline factorial.

The eight cubic architectures differ only in the number of control points
assigned to each numerical column.  Every architecture is trained by the
same cross-fitted A/B context-expansion protocol.  Architecture and blend
selection use pooled OOF error; outer-test errors are reported afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ASSIGNMENTS = tuple(itertools.product((4, 20), repeat=3))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _label(assignment: tuple[int, ...]) -> str:
    return "k" + "_".join(str(value) for value in assignment)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=4999)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--query-fraction-min", type=float, default=0.05)
    parser.add_argument("--query-fraction-max", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openml-cache-dir", type=Path, default=None)
    parser.add_argument("--regressor-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.bags < 2:
        raise ValueError("--bags must be at least two")
    if not 0 < args.query_fraction_min <= args.query_fraction_max < 1:
        raise ValueError("query fractions must satisfy 0 < min <= max < 1")
    return args


def _prepare_manifest(args: argparse.Namespace) -> None:
    script = Path(__file__)
    manifest = {
        "experiment": "three-column cubic DirectSpline 4/20-control-point full factorial",
        "task_id": int(args.task_id),
        "source_dir": str(args.source_dir.resolve()),
        "config_label": str(args.config_label),
        "protocol_seed": int(args.protocol_seed),
        "bags": int(args.bags),
        "query_fraction_range": [float(args.query_fraction_min), float(args.query_fraction_max)],
        "assignments": [list(item) for item in ASSIGNMENTS],
        "selection_policy": (
            "Select capacity assignment by minimum pooled cross-fitted OOF error; "
            "ties prefer fewer total control points then lexicographic order. Outer-test labels are report-only."
        ),
        "implementation_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        if _canonical_json(_load_json(path)) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different factorial run")
        if not args.resume:
            raise ValueError("output directory exists; pass --resume")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def _run_conditions(args: argparse.Namespace) -> None:
    runner = Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")
    for position, assignment in enumerate(ASSIGNMENTS, start=1):
        label = _label(assignment)
        command = [
            sys.executable,
            str(runner),
            "--source-dir",
            str(args.source_dir),
            "--output-dir",
            str(args.output_dir / label),
            "--config-label",
            str(args.config_label),
            "--task-id",
            str(args.task_id),
            "--protocol-seed",
            str(args.protocol_seed),
            "--bags",
            str(args.bags),
            "--adapter-arm",
            "full_spline",
            "--query-fraction-min",
            str(args.query_fraction_min),
            "--query-fraction-max",
            str(args.query_fraction_max),
            "--column-control-points",
            ",".join(str(value) for value in assignment),
            "--device",
            str(args.device),
            "--resume",
        ]
        if args.openml_cache_dir is not None:
            command.extend(("--openml-cache-dir", str(args.openml_cache_dir)))
        if args.regressor_checkpoint is not None:
            command.extend(("--regressor-checkpoint", str(args.regressor_checkpoint)))
        print(f"[{position}/{len(ASSIGNMENTS)}] {label}", flush=True)
        subprocess.run(command, check=True)


def _metric(record: dict[str, Any], split: str, method: str) -> float:
    return float(record["expanded_context"][split][method]["benchmark_error"])


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for assignment in ASSIGNMENTS:
        label = _label(assignment)
        records = _load_json(args.output_dir / label / "task_summaries.json")
        if len(records) != 1 or int(records[0]["task_id"]) != int(args.task_id):
            raise ValueError(f"{label} does not contain exactly task {args.task_id}")
        record = records[0]
        rows.append(
            {
                "label": label,
                "column_control_points": list(assignment),
                "total_control_points": sum(assignment),
                "is_mixed": len(set(assignment)) > 1,
                "oof_raw_spline_error": _metric(record, "oof", "raw_spline"),
                "oof_selected_blend_error": _metric(record, "oof", "selected_blend"),
                "selected_alpha": float(record["expanded_context"]["blend_selection"]["selected_alpha"]),
                "outer_raw_spline_error": _metric(record, "outer_test", "raw_spline"),
                "outer_selected_blend_error": _metric(record, "outer_test", "selected_blend"),
                "outer_identity_error": _metric(record, "outer_test", "identity"),
                "outer_full_tabiclv2_error": float(record["source_full_outer_training_tabiclv2"]["benchmark_error"]),
            }
        )

    def selected(method: str) -> dict[str, Any]:
        return min(
            rows,
            key=lambda row: (
                float(row[f"oof_{method}_error"]),
                int(row["total_control_points"]),
                tuple(row["column_control_points"]),
            ),
        )

    raw_selection = selected("raw_spline")
    end_to_end_selection = selected("selected_blend")
    uniform = {row["label"]: row for row in rows if not row["is_mixed"]}
    result = {
        "task_id": int(args.task_id),
        "dataset_name": _load_json(args.output_dir / rows[0]["label"] / "task_summaries.json")[0]["dataset_name"],
        "conditions": rows,
        "oof_selected_raw_capacity": raw_selection,
        "oof_selected_end_to_end_capacity": end_to_end_selection,
        "uniform_references": uniform,
        "predeclared_heterogeneity_result": {
            "selected_assignment_is_mixed": bool(end_to_end_selection["is_mixed"]),
            "selected_outer_error": float(end_to_end_selection["outer_selected_blend_error"]),
            "beats_both_uniforms_on_outer_test": bool(
                end_to_end_selection["outer_selected_blend_error"]
                < min(row["outer_selected_blend_error"] for row in uniform.values())
            ),
        },
        "interpretation_guard": (
            "All outer-test condition scores are diagnostic. Only the OOF-selected condition is a deployable estimate; "
            "choosing a different condition after viewing outer-test results is forbidden."
        ),
    }
    _write_json(args.output_dir / "factorial_summary.json", result)
    with (args.output_dir / "factorial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "label", "column_control_points", "total_control_points", "is_mixed",
            "oof_raw_spline_error", "oof_selected_blend_error", "selected_alpha",
            "outer_raw_spline_error", "outer_selected_blend_error", "outer_identity_error",
            "outer_full_tabiclv2_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "column_control_points": ",".join(map(str, row["column_control_points"]))})
    return result


def main() -> None:
    args = _parse_args()
    _prepare_manifest(args)
    _run_conditions(args)
    result = _aggregate(args)
    print(json.dumps({
        "oof_selected_raw": result["oof_selected_raw_capacity"]["label"],
        "oof_selected_end_to_end": result["oof_selected_end_to_end_capacity"]["label"],
        **result["predeclared_heterogeneity_result"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
