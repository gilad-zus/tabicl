"""Test whether context-derived descriptors can predict DirectSpline functions.

This is a teacher-regression diagnostic, not an end-to-end HyperSpline run.
It uses the cached DirectSpline teachers from ``direct_spline_function_basis``
and predicts every teacher's full canonical residual curve T(z)-z directly.
The target representation is therefore independent of raw control points,
PCA, knot locations, or any future HyperSpline output basis.

Each predictor is evaluated leave-one-dataset-out.  The conditions are:

* ``marginal``: the existing 23 unsupervised per-column statistics;
* ``supervised``: marginal plus the eight class-permutation-invariant stats;
* ``pooled_cross_column``: supervised stats plus leave-one-column-out mean and
  standard deviation of the other numerical columns' supervised descriptors.

The DirectSpline teacher used labelled adaptation-query rows, which a
zero-shot HyperSpline will not receive.  Consequently a poor held-out result
is strong evidence that the available context cannot identify this teacher;
a good result is only evidence that it *may* be identifiable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tabicl._hyperspline.statistics import UNSUPERVISED_SUMMARY_DIM, summarize_context

try:
    from scripts.direct_spline_function_basis import TeacherBag
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from direct_spline_function_basis import TeacherBag


CONDITIONS = ("marginal", "supervised", "pooled_cross_column")


class CurveRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True,
                        help="teacher_bags.pt from direct_spline_function_basis.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu",
                        help="CPU is usually sufficient and avoids competing with DirectSpline runs.")
    parser.add_argument("--allow-partial-cache", action="store_true",
                        help="Analyse an intentionally incomplete cache; unsafe while it is being written.")
    return parser.parse_args()


def parse_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names or any(name not in CONDITIONS for name in names) or len(set(names)) != len(names):
        raise ValueError(f"conditions must be a unique comma-separated subset of {CONDITIONS}")
    return names


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def descriptors_for_bag(bag: TeacherBag) -> dict[str, torch.Tensor]:
    """One permutation-equivariant descriptor vector per original column."""
    x_context = bag.support_x.float()
    y_context = bag.support_y.long()
    summary = summarize_context(x_context, y_context=y_context).summary.squeeze(0).float()
    marginal = summary[:, :UNSUPERVISED_SUMMARY_DIM]
    supervised = summary
    if summary.shape[0] <= 1:
        other_mean, other_std = torch.zeros_like(summary), torch.zeros_like(summary)
    else:
        total = summary.sum(dim=0, keepdim=True)
        other_mean = (total - summary) / (summary.shape[0] - 1)
        # This is leave-one-column-out, so a column cannot leak its own
        # descriptor twice through the pooled context token.
        centered = summary.unsqueeze(0) - other_mean.unsqueeze(1)
        # The diagonal itself is excluded; the explicit loop is tiny (D <= 100)
        # and avoids relying on an approximate population correction.
        other_std = torch.stack([
            summary[torch.arange(summary.shape[0]) != column].std(dim=0, unbiased=False)
            for column in range(summary.shape[0])
        ])
        del centered
    return {
        "marginal": marginal,
        "supervised": supervised,
        "pooled_cross_column": torch.cat((supervised, other_mean, other_std), dim=-1),
    }


def cache_is_complete(payload: dict[str, object], bags: list[TeacherBag]) -> bool:
    config = payload.get("config", {})
    if not isinstance(config, dict):
        return False
    expected = len(config.get("datasets", [])) * len(config.get("seeds", [])) * int(config.get("bags", 0))
    return bool(expected and len(bags) == expected)


def collect_records(bags: list[TeacherBag], conditions: list[str]) -> dict[str, dict[str, torch.Tensor | list[str] | list[int]]]:
    records: dict[str, dict[str, list]] = {
        condition: {"x": [], "y": [], "dataset": [], "seed": [], "bag": [], "column": []}
        for condition in conditions
    }
    for index, bag in enumerate(bags, start=1):
        features = descriptors_for_bag(bag)
        if features["marginal"].shape[0] != bag.curves.shape[0]:
            raise ValueError("descriptor and teacher curve column count disagree")
        for condition in conditions:
            records[condition]["x"].append(features[condition])
            records[condition]["y"].append(bag.curves.float())
            records[condition]["dataset"].extend([bag.dataset] * bag.curves.shape[0])
            records[condition]["seed"].extend([bag.seed] * bag.curves.shape[0])
            records[condition]["bag"].extend([bag.bag] * bag.curves.shape[0])
            records[condition]["column"].extend(range(bag.curves.shape[0]))
        if index % 16 == 0 or index == len(bags):
            print(f"Extracted descriptors for {index}/{len(bags)} teacher bags", flush=True)
    output = {}
    for condition, data in records.items():
        output[condition] = {
            "x": torch.cat(data["x"], dim=0), "y": torch.cat(data["y"], dim=0),
            "dataset": data["dataset"], "seed": data["seed"], "bag": data["bag"], "column": data["column"],
        }
    return output


def balanced_batch_indices(dataset_names: list[str], allowed: np.ndarray, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    names = sorted({dataset_names[index] for index in allowed.tolist()})
    pools = [torch.as_tensor([index for index in allowed.tolist() if dataset_names[index] == name]) for name in names]
    counts = [batch_size // len(pools) + (position < batch_size % len(pools)) for position in range(len(pools))]
    parts = [pool[torch.randint(pool.numel(), (count,), generator=generator)] for pool, count in zip(pools, counts, strict=True)]
    return torch.cat(parts)[torch.randperm(batch_size, generator=generator)]


def fit_and_score(
    *,
    condition: str,
    held_out_dataset: str,
    data: dict[str, torch.Tensor | list[str] | list[int]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset_names = data["dataset"]
    assert isinstance(dataset_names, list)
    x, y = data["x"], data["y"]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    train_indices = np.asarray([index for index, name in enumerate(dataset_names) if name != held_out_dataset])
    test_indices = np.asarray([index for index, name in enumerate(dataset_names) if name == held_out_dataset])
    if not train_indices.size or not test_indices.size:
        raise ValueError("leave-one-dataset-out split is empty")
    mean, std = x[train_indices].mean(0), x[train_indices].std(0, unbiased=False).clamp_min(1e-6)
    train_x = ((x[train_indices] - mean) / std).to(device)
    train_y = y[train_indices].to(device)
    test_x = ((x[test_indices] - mean) / std).to(device)
    test_y = y[test_indices].to(device)
    torch.manual_seed(args.model_seed + sum(ord(char) for char in condition + held_out_dataset))
    model = CurveRegressor(train_x.shape[-1], args.hidden_dim, train_y.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(args.model_seed + 10_000 + sum(ord(char) for char in held_out_dataset))
    for step in range(1, args.steps + 1):
        local = balanced_batch_indices(dataset_names, train_indices, args.batch_size, generator)
        # ``local`` indexes the original arrays; map into the train slice once
        # rather than moving the complete descriptors to the GPU each step.
        position = torch.searchsorted(torch.as_tensor(train_indices), local)
        prediction = model(train_x[position.to(device)])
        loss = (prediction - train_y[position.to(device)]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            print(f"[{condition} held_out={held_out_dataset}] step={step}/{args.steps} curve_mse={float(loss):.6f}", flush=True)
    with torch.no_grad():
        prediction = model(test_x).cpu()
    mean_curve = y[train_indices].mean(0, keepdim=True)
    zero_curve = torch.zeros_like(prediction)
    z = torch.linspace(-4.0, 4.0, prediction.shape[-1])
    output: list[dict[str, object]] = []
    for local_index, original_index in enumerate(test_indices.tolist()):
        target = y[original_index]
        predicted = prediction[local_index]
        transform = z + predicted
        output.append({
            "condition": condition, "held_out_dataset": held_out_dataset,
            "dataset": dataset_names[original_index], "outer_seed": data["seed"][original_index],
            "bag": data["bag"][original_index], "column": data["column"][original_index],
            "prediction_curve_rmse": float((predicted - target).square().mean().sqrt()),
            "mean_curve_baseline_rmse": float((mean_curve[0] - target).square().mean().sqrt()),
            "identity_curve_baseline_rmse": float((zero_curve[local_index] - target).square().mean().sqrt()),
            "prediction_transform_negative_slope_fraction": float((torch.diff(transform) < 0).float().mean()),
            "teacher_deformation_rms": float(target.square().mean().sqrt()),
        })
    del model, optimizer, train_x, train_y, test_x, test_y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def summarize(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["held_out_dataset"]))].append(row)
    per_dataset = []
    for (condition, dataset), group in sorted(grouped.items()):
        prediction = np.mean([float(row["prediction_curve_rmse"]) for row in group])
        baseline = np.mean([float(row["mean_curve_baseline_rmse"]) for row in group])
        per_dataset.append({
            "condition": condition, "held_out_dataset": dataset, "n_column_teachers": len(group),
            "mean_prediction_curve_rmse": float(prediction), "mean_curve_baseline_rmse": float(baseline),
            "relative_rmse_to_mean_curve": float(prediction / max(baseline, 1e-8)),
            "mean_negative_slope_fraction": float(np.mean([float(row["prediction_transform_negative_slope_fraction"]) for row in group])),
        })
    macro = []
    for condition in args.conditions:
        group = [row for row in per_dataset if row["condition"] == condition]
        macro.append({
            "condition": condition,
            **{key: float(np.mean([float(row[key]) for row in group])) for key in (
                "mean_prediction_curve_rmse", "mean_curve_baseline_rmse",
                "relative_rmse_to_mean_curve", "mean_negative_slope_fraction",
            )},
        })
    return {
        "protocol": "leave_one_dataset_out__balanced_teacher_curve_regression__function_space_target",
        "conditions": args.conditions, "model": {"hidden_dim": args.hidden_dim, "steps": args.steps, "lr": args.lr},
        "per_held_out_dataset": per_dataset, "macro_mean_over_held_out_datasets": macro,
        "interpretation": {
            "relative_rmse_below_one": "The descriptors predict teacher functions better than the training-set mean curve.",
            "marginal_vs_supervised": "Tests whether label-aware, class-ID-invariant information is useful for the target.",
            "supervised_vs_pooled_cross_column": "Tests whether a simple permutation-equivariant global-column token adds information before implementing cross-column HyperSpline attention.",
            "negative_slope_fraction": "A high value means the regressor predicts curves that require monotone projection before use as a spline transform.",
        },
    }


def main() -> None:
    args = parse_args()
    args.conditions = parse_names(args.conditions)
    if min(args.hidden_dim, args.steps, args.batch_size) <= 0 or args.lr <= 0:
        raise ValueError("hidden dimension, steps, batch size, and learning rate must be positive")
    cache_path, output_dir = args.teacher_cache.resolve(), args.output_dir.resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    bags: list[TeacherBag] = payload.get("bags", [])
    if not args.allow_partial_cache and not cache_is_complete(payload, bags):
        raise ValueError("teacher cache is incomplete or unrecognized; wait for it to finish or pass --allow-partial-cache")
    datasets = sorted({bag.dataset for bag in bags})
    if len(datasets) < 2:
        raise ValueError("at least two datasets are required for leave-one-dataset-out regression")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    records = collect_records(bags, args.conditions)
    rows: list[dict[str, object]] = []
    for condition in args.conditions:
        for held_out_dataset in datasets:
            rows.extend(fit_and_score(
                condition=condition, held_out_dataset=held_out_dataset,
                data=records[condition], args=args, device=device,
            ))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summarize(rows, args), indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} held-out column-teacher predictions to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
