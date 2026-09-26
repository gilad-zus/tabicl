"""Compare learned spline preprocessing with post-TabICL spline adaptation.

Four model arms, two schedules each, on the shared eight development tasks.
Every model family chooses constant/cosine using pooled cross-fitted OOF loss;
the corresponding test prediction is evaluated only after that choice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import importlib.metadata
import math
import platform
from pathlib import Path
import subprocess
import sys

from direct_spline_openml_input_preserving_ablation import (
    _load_json, _sha256, _write_json, relative_error_reduction, summarize_error_pairs,
)
from direct_spline_openml_lite import (
    _normalise_equivalent_hardware_environment, _resolve_execution_environment,
)


TASKS = {
    "multiclass": (4602, 75158, 167186, 361539),
    "regression": (4999, 5042, 362096, 362343),
}
ARMS = {
    "standard_line": ("standard", "direct_line"),
    "standard_spline": ("standard", "direct_spline"),
    "minimal_line": ("minimal", "direct_line"),
    "minimal_spline": ("minimal", "direct_spline"),
}
SCHEDULES = ("constant", "cosine")
CONTRASTS = {
    "standard_curvature": ("standard_line", "standard_spline"),
    "minimal_curvature": ("minimal_line", "minimal_spline"),
    "preprocessing_effect_line": ("standard_line", "minimal_line"),
    "preprocessing_effect_spline": ("standard_spline", "minimal_spline"),
    "minimal_spline_vs_full_tabiclv2": ("full_tabiclv2", "minimal_spline"),
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-type", choices=("all", *TASKS), default="all")
    parser.add_argument("--multiclass-source", type=Path)
    parser.add_argument("--regression-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--regressor-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-equivalent-hardware-resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--metadata-only-preflight", action="store_true", help="Local metadata check when prediction arrays were not downloaded.")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.allow_equivalent_hardware_resume and not args.resume:
        parser.error("equivalent-hardware resume requires --resume")
    if args.metadata_only_preflight and not args.preflight_only:
        parser.error("metadata-only check requires --preflight-only")
    return args


def _sources(args):
    defaults = {
        "multiclass": "openml_direct_spline_adaptive_retouche/multiclass_seed20260828",
        "regression": "openml_direct_spline_regression_confirmation/full_D_500_v1",
    }
    return {
        kind: getattr(args, f"{kind}_source") or args.results_root / defaults[kind]
        for kind in (TASKS if args.problem_type == "all" else (args.problem_type,))
    }


def _validate_sources(sources, *, metadata_only=False):
    from direct_spline_openml_support_audit import _find_source_cases
    from direct_spline_openml_crossfit_blend import _validate_case

    for kind, directory in sources.items():
        source = _load_json(directory / "experiment_manifest.json")
        if metadata_only:
            immutable = source["immutable_run"]
            if immutable["pipeline"] != "standard":
                raise ValueError("source must use the standard pipeline")
            if not set(TASKS[kind]).issubset(set(immutable["data_source"]["task_ids"])):
                raise ValueError("source metadata does not include all development tasks")
            configs = dict(zip(immutable["config_labels"], immutable["configs"]))
            if configs["D"]["n_control_points"] != 20:
                raise ValueError("source D capacity is not cubic K20")
            continue
        cases = _find_source_cases(
            source_dir=directory, manifest=source, config_label="D",
            requested_task_ids=set(TASKS[kind]),
        )
        if {item.task_id for item in cases} != set(TASKS[kind]):
            raise ValueError(f"{kind}: missing development source cases")
        for item in cases:
            _validate_case(item)
            if item.problem_type != kind:
                raise ValueError(f"task {item.task_id} has unexpected problem type")
            for key, value in {
                "learning_rate": .005, "weight_decay": .003,
                "gate_learning_rate_factor": 3., "grad_clip": 2.,
                "cross_column_mixing_rank": 4, "cross_column_mixing_bound": .1,
            }.items():
                if item.config.get(key) != value:
                    raise ValueError(f"source task {item.task_id} changed {key}")


def _prepare(args, sources):
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__), root / "scripts/direct_spline_openml_crossfit_context_expansion.py",
        root / "src/tabicl/_experiments/direct_spline_openml_standard.py",
        root / "src/tabicl/_experiments/direct_spline_preprocessing.py",
        root / "src/tabicl/_experiments/tabarena_direct_spline_protocol.py",
        root / "src/tabicl/_hyperspline/module.py",
        root / "src/tabicl/_hyperspline/bspline.py",
        root / "src/tabicl/_sklearn/preprocessing.py",
    ]
    semantic = {
        "experiment": "numerical preprocessing replacement factorial v1",
        "sources": {kind: {"path": str(path.resolve()), "manifest_sha256": _sha256(path / "experiment_manifest.json")}
                    for kind, path in sources.items()},
        "tasks": {kind: list(TASKS[kind]) for kind in sources},
        "arms": ARMS, "schedules": list(SCHEDULES),
        "protocol_seed": 20260915, "training_seed": 20260828,
        "bags": 4, "steps": 500, "validation_interval": 25,
        "query_fraction_range": [.05, .20], "n_control_points": 20,
        "cosine_min_lr_ratio": .01, "fixed_training_episodes": 4,
        "formula": "z+c+(s-4)*u+s*(S(u)-u), u=2/pi*atan(pi*z/8)",
        "minimal_preparation": "retain ordinary T-fitted encoder imputation/categories; numerical mean/std only; no power or outlier transform",
        "ensemble": "original two branch slots and eight views; original feature/class permutations and aggregation; separate adapters in each slot",
        "selection": "minimum pooled raw OOF error per model family; constant schedule wins exact ties; no identity blend in primary results",
        "implementation_sha256": {str(path.relative_to(root)): _sha256(path) for path in paths},
        "checkpoint_overrides": {
            key: None if getattr(args, key) is None else _sha256(getattr(args, key))
            for key in ("classifier_checkpoint", "regressor_checkpoint")
        },
    }
    _, runtime = _resolve_execution_environment(args.device, dry_run=False)
    runtime["python_version"] = platform.python_version()
    runtime["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "scipy", "scikit-learn", "openml", "tabicl")
    }
    if runtime.get("resolution_status") != "resolved":
        raise RuntimeError("could not resolve experiment execution environment")
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        old = _load_json(path)
        # JSON turns tuples into lists; compare canonical serialized values.
        if not args.resume or json.dumps(old["semantic"], sort_keys=True) != json.dumps(semantic, sort_keys=True):
            raise ValueError("resume requires matching experiment semantics and source hashes")
        if old["runtime"] != runtime:
            if not args.allow_equivalent_hardware_resume or (
                _normalise_equivalent_hardware_environment(old["runtime"])
                != _normalise_equivalent_hardware_environment(runtime)
            ):
                raise ValueError("hardware/software changed; equivalent hardware requires the resume flag and stable environment equality")
    else:
        _write_json(path, {"semantic": semantic, "runtime": runtime})


def _command(args, kind, source, arm, schedule):
    preparation, adapter = ARMS[arm]
    command = [
        sys.executable, str(Path(__file__).with_name("direct_spline_openml_crossfit_context_expansion.py")),
        "--source-dir", str(source), "--output-dir", str(args.output_dir / kind / arm / schedule),
        "--config-label", "D", "--protocol-seed", "20260915", "--bags", "4",
        "--adapter-arm", adapter, "--coordinate-mapping", "arctan", "--preserve-input-base",
        "--numerical-preparation", preparation, "--n-control-points", "20", "--adapter-steps", "500",
        "--training-random-state", "20260828", "--validation-interval", "25",
        "--query-fraction-min", ".05", "--query-fraction-max", ".20",
        "--training-audit-episodes", "4", "--branch-diagnostics", "--device", args.device, "--resume",
    ]
    command += ["--constant-lr"] if schedule == "constant" else ["--cosine-min-lr-ratio", ".01"]
    for task_id in TASKS[kind]:
        command += ["--task-id", str(task_id)]
    for key in ("openml_cache_dir", "classifier_checkpoint", "regressor_checkpoint"):
        if getattr(args, key) is not None:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return command


def _error(record, split, method="raw_spline"):
    value = float(record["expanded_context"][split][method]["benchmark_error"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("cannot select or compare invalid errors")
    return value


def _aggregate(args, kind):
    records = {}
    for arm in ARMS:
        records[arm] = {}
        for schedule in SCHEDULES:
            path = args.output_dir / kind / arm / schedule / "task_summaries.json"
            items = _load_json(path)
            by_id = {int(item["task_id"]): item for item in items}
            if len(items) != len(TASKS[kind]) or set(by_id) != set(TASKS[kind]):
                raise ValueError(f"incomplete/duplicated task records: {path}")
            records[arm][schedule] = by_id
    rows = []
    for task_id in TASKS[kind]:
        all_records = [records[arm][schedule][task_id] for arm in ARMS for schedule in SCHEDULES]
        reference = all_records[0]
        for record in all_records:
            for key in ("dataset_id", "dataset_name", "outer_split_hash", "problem_type", "effective_bags"):
                if record[key] != reference[key]:
                    raise ValueError(f"task {task_id}: mismatched {key}")
        baselines = [float(item["source_full_outer_training_tabiclv2"]["benchmark_error"]) for item in all_records]
        if len(set(baselines)) != 1:
            raise ValueError(f"task {task_id}: ordinary baseline changed")
        row = {"task_id": task_id, "dataset_name": reference["dataset_name"], "full_tabiclv2": baselines[0]}
        for arm in ARMS:
            choices = {schedule: records[arm][schedule][task_id] for schedule in SCHEDULES}
            for schedule, record in choices.items():
                for split in ("oof", "outer_test"):
                    row[f"{arm}_{schedule}_{split}"] = _error(record, split)
                    row[f"{arm}_{schedule}_{split}_identity"] = _error(record, split, "identity")
            # Select only on OOF. Test scores are not passed to this key.
            chosen = min(SCHEDULES, key=lambda name: (_error(choices[name], "oof"), SCHEDULES.index(name)))
            row[f"{arm}_selected_schedule"] = chosen
            for split in ("oof", "outer_test"):
                row[f"{arm}_selected_{split}"] = _error(choices[chosen], split)
            row[f"{arm}_selected_alpha_secondary"] = choices[chosen]["expanded_context"]["blend_selection"]["selected_alpha"]
        for preparation in ("standard", "minimal"):
            for schedule in SCHEDULES:
                for split in ("oof", "outer_test"):
                    left = row[f"{preparation}_line_{schedule}_{split}_identity"]
                    right = row[f"{preparation}_spline_{schedule}_{split}_identity"]
                    if left != right:
                        raise ValueError(f"task {task_id}: matched identity differs for {preparation}/{schedule}/{split}")
        standard_gain = relative_error_reduction(row["standard_line_selected_outer_test"], row["standard_spline_selected_outer_test"])
        minimal_gain = relative_error_reduction(row["minimal_line_selected_outer_test"], row["minimal_spline_selected_outer_test"])
        row["standard_curvature_gain"] = standard_gain
        row["minimal_curvature_gain"] = minimal_gain
        row["curvature_gain_difference"] = None if standard_gain is None or minimal_gain is None else minimal_gain-standard_gain
        rows.append(row)
    comparisons = {}
    for condition in (*SCHEDULES, "selected"):
        comparisons[condition] = {}
        for name, (control, candidate) in CONTRASTS.items():
            comparisons[condition][name] = {}
            for split in ("oof", "outer_test"):
                if control == "full_tabiclv2" and split == "oof":
                    continue
                comparisons[condition][name][split] = summarize_error_pairs([
                    (row["full_tabiclv2"] if control == "full_tabiclv2" else row[f"{control}_{condition}_{split}"],
                     row[f"{candidate}_{condition}_{split}"])
                    for row in rows
                ])
    result = {
        "problem_type": kind, "n_tasks": len(rows), "comparisons": comparisons, "task_results": rows,
        "selection_warning": "OOF is schedule selection data; outer test is subsequent evaluation. Development tasks were previously inspected.",
        "attribution_warning": "Removal of power/outlier transformations changes representation diversity despite identical view counts; larger curvature gain alone is insufficient if minimal line deteriorates.",
    }
    _write_json(args.output_dir / kind / "preprocessing_replacement_summary.json", result)
    with (args.output_dir / kind / "preprocessing_replacement_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    args = _parse_args()
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    sources = {kind: path.resolve() for kind, path in _sources(args).items()}
    _validate_sources(sources, metadata_only=args.metadata_only_preflight)
    if args.preflight_only:
        for kind, source in sources.items():
            for arm in ARMS:
                for schedule in SCHEDULES:
                    print(json.dumps(_command(args, kind, source, arm, schedule)), flush=True)
        return
    if not args.summarize_only:
        _prepare(args, sources)
        for kind, source in sources.items():
            for arm in ARMS:
                for schedule in SCHEDULES:
                    print(f"{kind}: {arm}, {schedule}", flush=True)
                    subprocess.run(_command(args, kind, source, arm, schedule), check=True)
            result = _aggregate(args, kind)
            print(json.dumps(result["comparisons"]["selected"], indent=2), flush=True)
    else:
        for kind in sources:
            print(json.dumps(_aggregate(args, kind)["comparisons"]["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
