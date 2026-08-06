"""Measure whether DirectSpline teachers are stable across equivalent splits.

This is a label-free analysis of already trained monotone DirectSpline
teachers.  For every outer dataset/seed, it compares the functional residual
curve of each column across the nested bag splits.  It therefore answers a
necessary question before fitting a HyperSpline teacher target: does an
approximately unique target function exist, or do different valid splits
produce unrelated equally-good transformations?

The runner intentionally consumes ``teacher_bags.pt`` written by
``direct_spline_function_basis.py``.  It neither modifies that cache nor loads
TabICL, so it is inexpensive once the teachers have been fitted.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    # Importing the class is required for torch.load to unpickle TeacherBag.
    from scripts.direct_spline_function_basis import TeacherBag
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_function_basis import TeacherBag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True,
                        help="teacher_bags.pt from direct_spline_function_basis.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial-cache", action="store_true",
                        help="Analyse an intentionally incomplete cache; unsafe while it is being written.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_metrics(first: TeacherBag, second: TeacherBag) -> tuple[dict[str, object], list[dict[str, object]]]:
    if first.curves.shape != second.curves.shape:
        raise ValueError("teachers in one dataset/seed must have aligned column and grid shapes")
    first_curve, second_curve = first.curves.float(), second.curves.float()
    delta = first_curve - second_curve
    rmse = delta.square().mean(dim=-1).sqrt()
    first_energy = first_curve.square().mean(dim=-1).sqrt()
    second_energy = second_curve.square().mean(dim=-1).sqrt()
    mean_energy = 0.5 * (first_energy + second_energy)
    relative_rmse = rmse / mean_energy.clamp_min(1e-6)
    cosine = (first_curve * second_curve).sum(-1) / (
        first_curve.square().sum(-1).sqrt() * second_curve.square().sum(-1).sqrt()
    ).clamp_min(1e-6)
    first_gate = torch.sigmoid(first.teacher_state["gate_logits"]).reshape(-1)
    second_gate = torch.sigmoid(second.teacher_state["gate_logits"]).reshape(-1)
    rows = []
    for column in range(first_curve.shape[0]):
        rows.append({
            "dataset": first.dataset, "outer_seed": first.seed,
            "first_bag": first.bag, "second_bag": second.bag, "column": column,
            "curve_rmse": float(rmse[column]),
            "relative_curve_rmse": float(relative_rmse[column]),
            "curve_cosine": float(cosine[column]),
            "mean_deformation_rms": float(mean_energy[column]),
            "first_gate": float(first_gate[column]), "second_gate": float(second_gate[column]),
            "absolute_gate_difference": float((first_gate[column] - second_gate[column]).abs()),
        })
    pair = {
        "dataset": first.dataset, "outer_seed": first.seed,
        "first_bag": first.bag, "second_bag": second.bag,
        "n_columns": first_curve.shape[0],
        "mean_curve_rmse": float(rmse.mean()),
        "median_curve_rmse": float(rmse.median()),
        "mean_relative_curve_rmse": float(relative_rmse.mean()),
        "median_relative_curve_rmse": float(relative_rmse.median()),
        "mean_curve_cosine": float(cosine.mean()),
        "median_curve_cosine": float(cosine.median()),
        "mean_deformation_rms": float(mean_energy.mean()),
        "mean_absolute_gate_difference": float((first_gate - second_gate).abs().mean()),
        "mean_teacher_train_objective_difference": abs(
            first.teacher_final_train_loss - second.teacher_final_train_loss
        ),
    }
    return pair, rows


def summarize(pair_rows: list[dict[str, object]], column_rows: list[dict[str, object]], cache: Path) -> dict[str, object]:
    by_run: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_run[(str(row["dataset"]), int(row["outer_seed"]))].append(row)
    run_summaries = []
    fields = (
        "mean_curve_rmse", "mean_relative_curve_rmse", "mean_curve_cosine",
        "mean_deformation_rms", "mean_absolute_gate_difference",
    )
    for (dataset, seed), rows in sorted(by_run.items()):
        run_summaries.append({
            "dataset": dataset, "outer_seed": seed, "teacher_pairs": len(rows),
            **{field: float(np.mean([float(row[field]) for row in rows])) for field in fields},
        })
    macro = {
        field: float(np.mean([run[field] for run in run_summaries]))
        for field in fields
    }
    return {
        "protocol": "same_outer_dataset_seed__different_nested_fit_support_query_splits__canonical_function_comparison",
        "teacher_cache": str(cache),
        "pair_count": len(pair_rows), "column_pair_count": len(column_rows),
        "per_dataset_seed": run_summaries, "macro_mean_over_dataset_seed": macro,
        "interpretation": {
            "low_relative_curve_rmse_and_high_cosine": "A stable teacher target exists across split noise.",
            "high_relative_curve_rmse_with_small_deformation_rms": "Transforms are near identity; relative disagreement can be unimportant.",
            "high_relative_curve_rmse_with_large_deformation_rms": "The current teacher target is split-dependent; regress functions or consensus targets rather than raw parameters.",
        },
    }


def main() -> None:
    args = parse_args()
    cache_path, output_dir = args.teacher_cache.resolve(), args.output_dir.resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    bags: list[TeacherBag] = payload.get("bags", [])
    config = payload.get("config", {})
    expected = len(config.get("datasets", [])) * len(config.get("seeds", [])) * int(config.get("bags", 0))
    if not args.allow_partial_cache and expected and len(bags) != expected:
        raise ValueError(
            f"cache has {len(bags)}/{expected} teacher bags and may still be being written; "
            "wait for function-basis teacher fitting or explicitly pass --allow-partial-cache"
        )
    groups: dict[tuple[str, int], list[TeacherBag]] = defaultdict(list)
    for bag in bags:
        groups[(bag.dataset, bag.seed)].append(bag)
    pair_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        for first, second in itertools.combinations(sorted(group, key=lambda item: item.bag), 2):
            pair, columns = pair_metrics(first, second)
            pair_rows.append(pair)
            column_rows.extend(columns)
    if not pair_rows:
        raise ValueError("need at least two completed teachers for one dataset/seed")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pairs.csv", pair_rows)
    write_csv(output_dir / "columns.csv", column_rows)
    summary = summarize(pair_rows, column_rows, cache_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(pair_rows)} teacher-pair and {len(column_rows)} column-pair records to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
