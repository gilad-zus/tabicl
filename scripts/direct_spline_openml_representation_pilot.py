"""Run the separate replacement-K12 and appended-feature development tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


TASKS = {"multiclass": [4602, 75158, 167186, 361539], "regression": [4999, 5042, 362096, 362343]}
SOURCES = {
    "multiclass": "openml_direct_spline_adaptive_retouche/multiclass_seed20260828",
    "regression": "openml_direct_spline_regression_confirmation/full_D_500_v1",
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _replacement_comparison(output, old_staged, kind):
    candidates = _load(output / "task_summaries.json")
    references = {
        arm: {int(row["task_id"]): row for row in _load(old_staged / arm / "task_summaries.json")}
        for arm in ("continued_line", "continued_spline")
    }
    rows = []
    for row in candidates:
        record = {"task_id": row["task_id"], "dataset_name": row["dataset_name"], "problem_type": kind}
        for split in ("oof", "outer_test"):
            for method in ("raw_spline", "selected_blend"):
                error = float(row["expanded_context"][split][method]["benchmark_error"])
                record[f"{split}_{method}_k12"] = error
                for arm, label in (("continued_line", "line"), ("continued_spline", "k20")):
                    baseline = float(references[arm][row["task_id"]]["expanded_context"][split][method]["benchmark_error"])
                    record[f"{split}_{method}_{label}"] = baseline
                    record[f"{split}_{method}_gain_vs_{label}"] = (baseline - error) / baseline if baseline else 0.0
        rows.append(record)
    summary = {"n_tasks": len(rows), "task_results": rows,
               "primary": "raw_spline: validation-selected checkpoints before identity blending",
               "reference": str(old_staged),
               "interpretation": "Cubic K12 versus existing K20 and line continuations from the same saved line states; new feature-expansion results are a separate experiment."}
    (output / "replacement_capacity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-type", choices=("both", "multiclass", "regression"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-equivalent-hardware-resume", action="store_true")
    args = parser.parse_args()
    if args.allow_equivalent_hardware_resume and not args.resume:
        parser.error("equivalent-hardware resume requires --resume")
    args.results_root, args.output_dir = args.results_root.resolve(), args.output_dir.resolve()
    scripts = Path(__file__).resolve().parent
    kinds = tuple(TASKS) if args.problem_type == "both" else (args.problem_type,)
    # Validate all dependencies before starting any costly fitting.
    for kind in kinds:
        source = args.results_root / SOURCES[kind]
        old_staged = args.results_root / "openml_direct_spline_staged_curvature_ablation/dev8_seed20260915" / kind
        initial = args.results_root / "openml_direct_spline_direct_arctan_ablation/dev8_seed20260915_v2" / kind / "direct_arctan_line"
        manifest = _load(old_staged / "experiment_manifest.json")
        if manifest["bags"] != 4 or manifest["protocol_seed"] != 20260915 or manifest["continuation_steps"] != 250:
            raise ValueError("existing K20 comparison does not match the predeclared protocol")
        if Path(manifest["source_dir"]).resolve() != source or Path(manifest["initial_line_dir"]).resolve() != initial:
            raise ValueError("reference source/initialization does not match")
        if sorted(manifest["task_ids"]) != sorted(TASKS[kind]):
            raise ValueError("reference task set differs")
        for path in (source / "experiment_manifest.json", initial / "experiment_manifest.json",
                     old_staged / "continued_line/task_summaries.json", old_staged / "continued_spline/task_summaries.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
    for kind in kinds:
        source = args.results_root / SOURCES[kind]
        task_flags = [item for task in TASKS[kind] for item in ("--task-id", str(task))]
        feature_command = [sys.executable, str(scripts / "direct_spline_openml_feature_expansion.py"),
                           "--source-dir", str(source), "--output-dir", str(args.output_dir / "feature_expansion" / kind),
                           "--device", args.device, *task_flags]
        if args.resume:
            feature_command.append("--resume")
        if args.allow_equivalent_hardware_resume:
            feature_command.append("--allow-equivalent-hardware-resume")
        print(f"{kind}: appended-feature line / cubic8 / cubic20", flush=True)
        subprocess.run(feature_command, check=True)
        initial = args.results_root / "openml_direct_spline_direct_arctan_ablation/dev8_seed20260915_v2" / kind / "direct_arctan_line"
        replacement_output = args.output_dir / "replacement12" / kind
        replacement_command = [sys.executable, str(scripts / "direct_spline_openml_crossfit_context_expansion.py"),
                               "--source-dir", str(source), "--initial-adapter-dir", str(initial),
                               "--output-dir", str(replacement_output), "--config-label", "D",
                               "--protocol-seed", "20260915", "--bags", "4", "--adapter-arm", "direct_spline",
                               "--coordinate-mapping", "arctan", "--n-control-points", "12",
                               "--adapter-steps", "250", "--query-fraction-min", "0.05", "--query-fraction-max", "0.20",
                               "--training-audit-episodes", "8", "--device", args.device, *task_flags]
        if args.resume:
            replacement_command.append("--resume")
        print(f"{kind}: existing replacement pipeline, cubic12 continuation", flush=True)
        subprocess.run(replacement_command, check=True)
        old_staged = args.results_root / "openml_direct_spline_staged_curvature_ablation/dev8_seed20260915" / kind
        _replacement_comparison(replacement_output, old_staged, kind)


if __name__ == "__main__":
    main()
