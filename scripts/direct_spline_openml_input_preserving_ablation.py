"""Test whether arctan-basis splines train better when they start at TabICL input.

The existing direct-arctan experiment starts both its straight-line and cubic
arms at ``R * u(z)``, where ``z`` is TabICL's already-standardized numerical
feature and ``u(z) = 2/pi * atan(pi*z/(2R))``.  This experiment keeps that
coordinate only as the spline's bounded evaluation coordinate and instead uses

    g(z) = z + c + s S(u(z)) - R u(z).

At initialization ``c=0``, ``s=R``, and ``S(u)=u``, so both new arms are
exactly the ordinary TabICL numerical input.  The line arm can learn only the
arctan-coordinate shift/span residual; the cubic arm can additionally learn
curvature.  Their difference is therefore the incremental effect of spline
curvature conditional on starting from the frozen backbone's native input.

The prior direct-arctan line/spline runs are supplied as a read-only matched
reference.  They use the identical tasks, source run, folds, context protocol,
and schedule, so this report can also separate an input-base effect from a
curvature effect without rerunning their already-completed trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REFERENCE_ARMS = {
    "compressed_line": "direct_arctan_line",
    "compressed_spline": "direct_arctan_spline",
}
PRESERVING_ARMS = {
    "preserved_line": "direct_line",
    "preserved_spline": "direct_spline",
}


def relative_error_reduction(reference: float, candidate: float) -> float | None:
    if not all(math.isfinite(value) and value >= 0 for value in (reference, candidate)):
        raise ValueError("paired errors must be finite non-negative values")
    return (reference - candidate) / reference if reference > 0 else None


def summarize_error_pairs(pairs: list[tuple[float, float]]) -> dict[str, int | float | None]:
    gains = [relative_error_reduction(reference, candidate) for reference, candidate in pairs]
    relative = [gain for gain in gains if gain is not None]
    wins = sum(reference > candidate for reference, candidate in pairs)
    losses = sum(candidate > reference for reference, candidate in pairs)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(pairs) - wins - losses,
        "n_relative_gain_tasks": len(relative),
        "mean_relative_gain": statistics.mean(relative) if relative else None,
        "median_relative_gain": statistics.median(relative) if relative else None,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Completed direct-arctan ablation directory containing direct_arctan_line and direct_arctan_spline.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--query-fraction-min", type=float, default=0.05)
    parser.add_argument("--query-fraction-max", type=float, default=0.20)
    parser.add_argument("--adapter-steps", type=int, default=None)
    parser.add_argument("--n-control-points", type=int, default=20)
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
    if args.adapter_steps is not None and args.adapter_steps < 1:
        raise ValueError("--adapter-steps must be positive")
    if args.n_control_points < 4:
        raise ValueError("cubic splines require at least four control points")
    return args


def _validate_reference(args: argparse.Namespace) -> None:
    manifest_path = args.reference_dir / "experiment_manifest.json"
    manifest = _load_json(manifest_path)
    expected = set(int(task_id) for task_id in args.task_id)
    if Path(str(manifest.get("source_dir", ""))).resolve() != args.source_dir.resolve():
        raise ValueError("reference direct-arctan run was built from a different source directory")
    if set(int(task_id) for task_id in manifest.get("task_ids", ())) != expected:
        raise ValueError("reference direct-arctan run has a different task set")
    for key, expected_value in {
        "config_label": str(args.config_label),
        "protocol_seed": int(args.protocol_seed),
        "bags": int(args.bags),
    }.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"reference direct-arctan run has different {key}")
    if manifest.get("query_fraction_range") != [
        float(args.query_fraction_min), float(args.query_fraction_max)
    ]:
        raise ValueError("reference direct-arctan run has a different query-fraction range")
    if manifest.get("arms") != {
        "direct_arctan_line": "direct_line",
        "direct_arctan_spline": "direct_spline",
    }:
        raise ValueError("reference directory is not the expected direct-output arctan ablation")
    for arm in REFERENCE_ARMS.values():
        if not (args.reference_dir / arm / "task_summaries.json").is_file():
            raise FileNotFoundError(args.reference_dir / arm / "task_summaries.json")


def _prepare_manifest(args: argparse.Namespace) -> None:
    script = Path(__file__)
    manifest = {
        "experiment": "input-preserving direct-arctan spline curvature ablation",
        "source_dir": str(args.source_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "reference_manifest_sha256": _sha256(args.reference_dir / "experiment_manifest.json"),
        "task_ids": sorted(int(task_id) for task_id in args.task_id),
        "config_label": str(args.config_label),
        "protocol_seed": int(args.protocol_seed),
        "bags": int(args.bags),
        "query_fraction_range": [float(args.query_fraction_min), float(args.query_fraction_max)],
        "adapter_steps_override": args.adapter_steps,
        "n_control_points": int(args.n_control_points),
        "coordinate_mapping": "arctan",
        "preserve_input_base": True,
        "reference_arms": REFERENCE_ARMS,
        "new_arms": PRESERVING_ARMS,
        "formula": "g(z) = z + c + s*S(u(z)) - R*u(z), u(z)=2/pi*atan(pi*z/(2R))",
        "initialization": (
            "c=0, s=R, S(u)=u; the fresh transformed feature equals z exactly. "
            "The direct line freezes S(u)=u and the direct spline learns its monotone cubic shape."
        ),
        "primary_causal_contrast": (
            "preserved_spline versus preserved_line: matched learned arctan-coordinate "
            "shift/span and mixer, differing only in trainable cubic curvature."
        ),
        "secondary_causal_contrast": (
            "preserved versus compressed corresponding arm: whether retaining TabICL's "
            "native standardized numerical input at initialization changes trainability."
        ),
        "implementation_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "crossfit_runner_sha256": _sha256(
            script.with_name("direct_spline_openml_crossfit_context_expansion.py")
        ),
        "adapter_module_sha256": _sha256(
            script.resolve().parents[1] / "src/tabicl/_hyperspline/module.py"
        ),
        "standard_adapter_sha256": _sha256(
            script.resolve().parents[1] / "src/tabicl/_experiments/direct_spline_openml_standard.py"
        ),
    }
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        if _canonical_json(_load_json(path)) != _canonical_json(manifest):
            raise ValueError("output directory belongs to a different input-preserving ablation")
        if not args.resume:
            raise ValueError("output directory exists; pass --resume")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(path, manifest)


def _run_arms(args: argparse.Namespace) -> None:
    runner = Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")
    for position, (label, adapter_arm) in enumerate(PRESERVING_ARMS.items(), start=1):
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
            "--preserve-input-base",
            "--n-control-points",
            str(args.n_control_points),
            "--query-fraction-min",
            str(args.query_fraction_min),
            "--query-fraction-max",
            str(args.query_fraction_max),
            "--device",
            str(args.device),
            "--resume",
        ]
        if args.adapter_steps is not None:
            command.extend(("--adapter-steps", str(args.adapter_steps)))
        for task_id in args.task_id:
            command.extend(("--task-id", str(task_id)))
        for flag, value in (
            ("--openml-cache-dir", args.openml_cache_dir),
            ("--classifier-checkpoint", args.classifier_checkpoint),
            ("--regressor-checkpoint", args.regressor_checkpoint),
        ):
            if value is not None:
                command.extend((flag, str(value)))
        print(f"[{position}/{len(PRESERVING_ARMS)}] {label}", flush=True)
        subprocess.run(command, check=True)


def _metric(record: Mapping[str, Any], split: str, method: str) -> float:
    return float(record["expanded_context"][split][method]["benchmark_error"])


def _load_records(path: Path) -> dict[int, dict[str, Any]]:
    records = _load_json(path / "task_summaries.json")
    return {int(record["task_id"]): record for record in records}


def _comparison(
    rows: list[dict[str, Any]], *, reference: str, candidate: str, method: str
) -> dict[str, Any]:
    pairs = [
        (
            float(row[f"outer_test_{method}_{reference}"]),
            float(row[f"outer_test_{method}_{candidate}"]),
        )
        for row in rows
    ]
    paired = summarize_error_pairs(pairs)
    return {
        "reference_arm": reference,
        "candidate_arm": candidate,
        "candidate_wins": paired["wins"],
        "ties": paired["ties"],
        "candidate_losses": paired["losses"],
        "mean_relative_candidate_gain": paired["mean_relative_gain"],
        "median_relative_candidate_gain": paired["median_relative_gain"],
        "n_relative_gain_tasks": paired["n_relative_gain_tasks"],
    }


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    by_arm = {
        label: _load_records(args.reference_dir / source_label)
        for label, source_label in REFERENCE_ARMS.items()
    }
    by_arm.update(
        {label: _load_records(args.output_dir / label) for label in PRESERVING_ARMS}
    )
    expected = set(int(task_id) for task_id in args.task_id)
    if any(set(records) != expected for records in by_arm.values()):
        raise ValueError("one or more input-preserving or reference arms has an incomplete task set")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(expected):
        records = {label: values[task_id] for label, values in by_arm.items()}
        problem_types = {str(record["problem_type"]) for record in records.values()}
        if len(problem_types) != 1:
            raise ValueError(f"task {task_id} changed problem type between matched arms")
        for split in ("oof", "outer_test"):
            identities = {_metric(record, split, "identity") for record in records.values()}
            if len(identities) != 1:
                raise ValueError(
                    f"task {task_id} identity baseline differs between matched arms on {split}"
                )
        reference = records["compressed_line"]
        row: dict[str, Any] = {
            "task_id": task_id,
            "dataset_name": reference["dataset_name"],
            "problem_type": reference["problem_type"],
            "outer_test_identity": _metric(reference, "outer_test", "identity"),
        }
        for split in ("oof", "outer_test"):
            for method in ("raw_spline", "selected_blend"):
                for label, record in records.items():
                    row[f"{split}_{method}_{label}"] = _metric(record, split, method)
                for reference_arm, candidate_arm in (
                    ("compressed_line", "compressed_spline"),
                    ("preserved_line", "preserved_spline"),
                    ("compressed_line", "preserved_line"),
                    ("compressed_spline", "preserved_spline"),
                ):
                    row[
                        f"{split}_{method}_relative_gain_{candidate_arm}_vs_{reference_arm}"
                    ] = relative_error_reduction(
                        float(row[f"{split}_{method}_{reference_arm}"]),
                        float(row[f"{split}_{method}_{candidate_arm}"]),
                    )
        rows.append(row)

    contrasts = {
        "compressed_curvature": ("compressed_line", "compressed_spline"),
        "input_preserving_curvature": ("preserved_line", "preserved_spline"),
        "input_base_effect_without_curvature": ("compressed_line", "preserved_line"),
        "input_base_effect_with_curvature": ("compressed_spline", "preserved_spline"),
    }
    result = {
        "n_tasks": len(rows),
        "task_results": rows,
        "outer_test_paired_contrasts": {
            label: {
                method: _comparison(rows, reference=reference, candidate=candidate, method=method)
                for method in ("raw_spline", "selected_blend")
            }
            for label, (reference, candidate) in contrasts.items()
        },
        "primary_result": "input_preserving_curvature.raw_spline",
        "interpretation": (
            "Raw spline is the primary curvature attribution: it compares two matched trained "
            "arms before an identity blend can hide their difference. The selected blend is "
            "reported separately because its alpha is chosen independently from each arm's OOF rows. "
            "Outer-test labels are read only for this final report."
        ),
    }
    _write_json(args.output_dir / "input_preserving_ablation_summary.json", result)
    with (args.output_dir / "input_preserving_ablation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    args = _parse_args()
    args.source_dir = args.source_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    _validate_reference(args)
    _prepare_manifest(args)
    _run_arms(args)
    result = _aggregate(args)
    print(json.dumps(result["outer_test_paired_contrasts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
