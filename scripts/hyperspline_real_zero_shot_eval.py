"""Zero-shot real-data evaluation for a trained synthetic HyperSpline.

This script never trains or selects a HyperSpline checkpoint on real data.
For each stratified context/query split it compares the fixed checkpoint to a
fresh identity HyperSpline with the *same* configuration.  Numerical columns
are median-imputed and passed through HyperSpline; categorical columns are
ordinal-encoded once and copied through unchanged in both conditions.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

try:  # Supports both ``python scripts/file.py`` and module-style imports in tests.
    from scripts.direct_spline_multidataset_headroom import load_backbone
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script invocation
    from direct_spline_multidataset_headroom import load_backbone
from tabicl._hyperspline import (
    FrozenTabICLHyperSpline,
    HyperSplineTransform,
    backbone_state_dict_hash,
    load_hyperspline_checkpoint,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    group: str  # numerical_only | mixed


@dataclass(frozen=True)
class RealEpisode:
    dataset: str
    dataset_group: str
    split_seed: int
    n_context: int
    n_query: int
    n_features: int
    n_numerical_features: int
    n_categorical_features: int
    n_classes: int
    x_context: torch.Tensor  # (1, N_context, D)
    x_query: torch.Tensor  # (1, N_query, D)
    y_context: torch.Tensor  # (1, N_context), float for TabICL
    y_query: torch.Tensor  # (N_query,), long for cross entropy
    numerical_mask: torch.Tensor  # (D,), bool


NUMERICAL_SPECS = [
    DatasetSpec("iris", "numerical_only"),
    DatasetSpec("wine", "numerical_only"),
    DatasetSpec("breast_cancer", "numerical_only"),
    DatasetSpec("digits", "numerical_only"),
]

MIXED_SPECS = [
    DatasetSpec("adult", "mixed"),
    DatasetSpec("credit-g", "mixed"),
    DatasetSpec("bank-marketing", "mixed"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyperspline-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional local frozen TabICLv2 checkpoint.")
    parser.add_argument("--checkpoint-version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-suite", choices=("all", "numerical_only", "mixed"), default="all")
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset names.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024, help="0 disables a stratified row cap.")
    parser.add_argument("--output-csv", type=Path, default=Path("results/hyperspline_real_zero_shot.csv"))
    parser.add_argument("--output-summary-csv", type=Path, default=Path("results/hyperspline_real_zero_shot_summary.csv"))
    return parser.parse_args()


def parse_csv(value: str, converter: Callable[[str], object]) -> list:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return [converter(item) for item in values]


def resolve_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    all_specs = NUMERICAL_SPECS + MIXED_SPECS
    by_name = {spec.name: spec for spec in all_specs}
    if args.datasets is not None:
        names = parse_csv(args.datasets, str)
        unknown = sorted(set(names).difference(by_name))
        if unknown:
            raise ValueError(f"unknown datasets {unknown}; choices are {sorted(by_name)}")
        return [by_name[name] for name in names]
    if args.dataset_suite == "numerical_only":
        return NUMERICAL_SPECS
    if args.dataset_suite == "mixed":
        return MIXED_SPECS
    return all_specs


def load_dataset_frame(spec: DatasetSpec):
    print(f"[{spec.name}] loading dataset source...", flush=True)
    if spec.name == "iris":
        dataset = load_iris(as_frame=True)
    elif spec.name == "wine":
        dataset = load_wine(as_frame=True)
    elif spec.name == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
    elif spec.name == "digits":
        dataset = load_digits(as_frame=True)
    elif spec.name == "adult":
        dataset = fetch_openml("adult", version=2, as_frame=True)
    elif spec.name == "credit-g":
        dataset = fetch_openml("credit-g", version=1, as_frame=True)
    elif spec.name == "bank-marketing":
        dataset = fetch_openml("bank-marketing", version=1, as_frame=True)
    else:  # pragma: no cover - guarded by resolve_specs
        raise ValueError(f"unsupported dataset: {spec.name}")
    return dataset.data.copy(), dataset.target.copy()


def split_columns(frame) -> tuple[list[str], list[str]]:
    numerical, categorical = [], []
    for column in frame.columns:
        dtype = frame[column].dtype
        if (
            str(dtype).startswith("category")
            or dtype == object
            or str(dtype).startswith("string")
            or str(dtype) == "bool"
        ):
            categorical.append(column)
        else:
            numerical.append(column)
    return numerical, categorical


def make_preprocessor(numerical_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numerical_columns:
        transformers.append(("numerical", Pipeline([("impute", SimpleImputer(strategy="median"))]), numerical_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("dataset contains no usable feature columns")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def feature_bin(n_features: int) -> str:
    if n_features <= 10:
        return "01-10"
    if n_features <= 50:
        return "11-50"
    return "51+"


def build_episode(spec: DatasetSpec, seed: int, args: argparse.Namespace, device: torch.device) -> RealEpisode:
    frame, target = load_dataset_frame(spec)
    valid = np.asarray([value is not None and str(value) != "nan" for value in target], dtype=bool)
    frame = frame.loc[valid].reset_index(drop=True)
    target = LabelEncoder().fit_transform(np.asarray(target)[valid])
    if args.max_rows > 0 and frame.shape[0] > args.max_rows:
        frame, _, target, _ = train_test_split(
            frame, target, train_size=args.max_rows, random_state=seed, stratify=target
        )
        frame = frame.reset_index(drop=True)

    numerical_columns, categorical_columns = split_columns(frame)
    context_frame, query_frame, y_context, y_query = train_test_split(
        frame, target, test_size=args.test_size, random_state=seed, stratify=target
    )
    preprocessor = make_preprocessor(numerical_columns, categorical_columns)
    x_context = np.asarray(preprocessor.fit_transform(context_frame), dtype=np.float32)
    x_query = np.asarray(preprocessor.transform(query_frame), dtype=np.float32)
    numerical_mask = torch.tensor(
        [True] * len(numerical_columns) + [False] * len(categorical_columns), device=device, dtype=torch.bool
    )
    n_classes = int(np.unique(y_context).size)
    print(
        f"[{spec.name} seed={seed}] loaded X={frame.shape}, classes={np.unique(target).size}; "
        f"split context={x_context.shape}, query={x_query.shape}, "
        f"numerical={len(numerical_columns)}, categorical={len(categorical_columns)}",
        flush=True,
    )
    return RealEpisode(
        dataset=spec.name,
        dataset_group=spec.group,
        split_seed=seed,
        n_context=x_context.shape[0],
        n_query=x_query.shape[0],
        n_features=x_context.shape[1],
        n_numerical_features=len(numerical_columns),
        n_categorical_features=len(categorical_columns),
        n_classes=n_classes,
        x_context=torch.as_tensor(x_context, device=device).unsqueeze(0),
        x_query=torch.as_tensor(x_query, device=device).unsqueeze(0),
        y_context=torch.as_tensor(y_context, dtype=torch.float32, device=device).unsqueeze(0),
        y_query=torch.as_tensor(y_query, dtype=torch.long, device=device),
        numerical_mask=numerical_mask,
    )


@torch.no_grad()
def evaluate(backbone, hyperspline: HyperSplineTransform, episode: RealEpisode) -> dict[str, float]:
    adapter = FrozenTabICLHyperSpline(backbone, hyperspline).to(episode.x_context.device).eval()
    backbone.clear_cache()
    logits, parameters = adapter(
        episode.x_context, episode.x_query, episode.y_context, episode.numerical_mask, return_parameters=True
    )
    loss = F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())
    accuracy = (logits.argmax(dim=-1).flatten() == episode.y_query).float().mean()
    if parameters is None:
        diagnostics = {"mean_gate": 0.0, "min_gate": 0.0, "max_gate": 0.0, "mean_abs_deformation": 0.0, "max_abs_deformation": 0.0, "clip_fraction": 0.0}
    else:
        raw = torch.cat((episode.x_context[..., episode.numerical_mask], episode.x_query[..., episode.numerical_mask]), dim=1)
        transformed_context = hyperspline.apply_transform(
            episode.x_context[..., episode.numerical_mask], parameters
        )
        transformed_query = hyperspline.apply_transform(
            episode.x_query[..., episode.numerical_mask], parameters
        )
        transformed = torch.cat((transformed_context, transformed_query), dim=1).float()
        z = (raw.float() - parameters.location.unsqueeze(1)) / parameters.scale.unsqueeze(1)
        deformation = (transformed - z).abs()
        clip_fraction = ((z / hyperspline.standardized_range).abs() >= 1).float().mean()
        diagnostics = {
            "mean_gate": float(parameters.gate.mean()),
            "min_gate": float(parameters.gate.min()),
            "max_gate": float(parameters.gate.max()),
            "mean_abs_deformation": float(deformation.mean()),
            "max_abs_deformation": float(deformation.max()),
            "clip_fraction": float(clip_fraction),
        }
    return {"loss": float(loss), "accuracy": float(accuracy), **diagnostics}


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, mean, 0.0
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean - 1.96 * se, mean + 1.96 * se, se


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []

    def emit(group_name: str, group_value: str, group_rows: list[dict[str, object]]) -> None:
        loss_delta = np.asarray([row["loss_delta"] for row in group_rows], dtype=float)
        accuracy_delta = np.asarray([row["accuracy_delta"] for row in group_rows], dtype=float)
        relative = np.asarray([row["relative_loss_improvement"] for row in group_rows], dtype=float)
        loss_low, loss_high, loss_se = confidence_interval(loss_delta)
        acc_low, acc_high, acc_se = confidence_interval(accuracy_delta)
        summaries.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "n": len(group_rows),
                "mean_identity_loss": float(np.mean([row["identity_loss"] for row in group_rows])),
                "mean_loss": float(np.mean([row["loss"] for row in group_rows])),
                "mean_loss_delta": float(loss_delta.mean()),
                "median_loss_delta": float(np.median(loss_delta)),
                "loss_win_count": int((loss_delta > 0).sum()),
                "loss_win_rate": float((loss_delta > 0).mean()),
                "mean_relative_loss_improvement": float(relative.mean()),
                "loss_delta_se": loss_se,
                "loss_delta_ci95_low": loss_low,
                "loss_delta_ci95_high": loss_high,
                "mean_identity_accuracy": float(np.mean([row["identity_accuracy"] for row in group_rows])),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in group_rows])),
                "mean_accuracy_delta": float(accuracy_delta.mean()),
                "median_accuracy_delta": float(np.median(accuracy_delta)),
                "accuracy_win_count": int((accuracy_delta > 0).sum()),
                "accuracy_win_rate": float((accuracy_delta > 0).mean()),
                "accuracy_delta_se": acc_se,
                "accuracy_delta_ci95_low": acc_low,
                "accuracy_delta_ci95_high": acc_high,
                "mean_gate": float(np.mean([row["mean_gate"] for row in group_rows])),
                "mean_abs_deformation": float(np.mean([row["mean_abs_deformation"] for row in group_rows])),
                "mean_clip_fraction": float(np.mean([row["clip_fraction"] for row in group_rows])),
            }
        )

    emit("overall", "all", rows)
    groups: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups["dataset"][str(row["dataset"])].append(row)
        groups["dataset_group"][str(row["dataset_group"])].append(row)
        groups["n_classes"][str(row["n_classes"])].append(row)
        groups["n_features"][str(row["n_features"])].append(row)
        groups["feature_bin"][str(row["feature_bin"])].append(row)
    for group_name, values in groups.items():
        for group_value, group_rows in sorted(values.items()):
            emit(group_name, group_value, group_rows)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.hyperspline_checkpoint.is_file():
        raise FileNotFoundError(f"HyperSpline checkpoint does not exist: {args.hyperspline_checkpoint}")
    if not 0 < args.test_size < 1 or args.max_rows < 0:
        raise ValueError("--test-size must be in (0, 1) and --max-rows must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    seeds = parse_csv(args.seeds, int)
    specs = resolve_specs(args)
    print(f"Running zero-shot real-data evaluation on device: {device}; datasets={[spec.name for spec in specs]}; seeds={seeds}", flush=True)

    backbone, _ = load_backbone(args, device)
    hyperspline, metadata = load_hyperspline_checkpoint(
        args.hyperspline_checkpoint,
        device=device,
        expected_backbone_reference=args.checkpoint_version,
        expected_backbone_hash=backbone_state_dict_hash(backbone),
    )
    hyperspline.eval()
    identity = HyperSplineTransform(**metadata["hyperspline_config"]).to(device).eval()
    print(
        f"Loaded zero-shot HyperSpline checkpoint: selected_step={metadata.get('step')}; "
        f"config={metadata['hyperspline_config']}. No real-data optimization will run.",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for spec in specs:
        for seed in seeds:
            episode = build_episode(spec, seed, args, device)
            if episode.n_classes > backbone.max_classes:
                raise ValueError(f"{spec.name} has {episode.n_classes} classes; backbone supports {backbone.max_classes}")
            baseline = evaluate(backbone, identity, episode)
            transformed = evaluate(backbone, hyperspline, episode)
            loss_delta = baseline["loss"] - transformed["loss"]
            accuracy_delta = transformed["accuracy"] - baseline["accuracy"]
            row = {
                "dataset": episode.dataset,
                "dataset_group": episode.dataset_group,
                "split_seed": episode.split_seed,
                "n_context": episode.n_context,
                "n_query": episode.n_query,
                "n_features": episode.n_features,
                "feature_bin": feature_bin(episode.n_features),
                "n_numerical_features": episode.n_numerical_features,
                "n_categorical_features": episode.n_categorical_features,
                "n_classes": episode.n_classes,
                **transformed,
                "identity_loss": baseline["loss"],
                "identity_accuracy": baseline["accuracy"],
                "loss_delta": loss_delta,
                "relative_loss_improvement": loss_delta / max(baseline["loss"], 1e-12),
                "accuracy_delta": accuracy_delta,
            }
            rows.append(row)
            print(
                f"[{spec.name} seed={seed}] loss={row['loss']:.6f} (delta={loss_delta:+.6f}), "
                f"accuracy={row['accuracy']:.4f} (delta={accuracy_delta:+.4f}), "
                f"gate={row['mean_gate']:.4f}, deformation={row['mean_abs_deformation']:.4f}",
                flush=True,
            )

    summaries = summarize_rows(rows)
    write_csv(args.output_csv, rows)
    write_csv(args.output_summary_csv, summaries)
    overall = summaries[0]
    print(
        f"Overall: mean_loss_delta={overall['mean_loss_delta']:+.6f}, "
        f"loss_wins={overall['loss_win_count']}/{overall['n']}, "
        f"mean_accuracy_delta={overall['mean_accuracy_delta']:+.4f}",
        flush=True,
    )
    print(f"Wrote detailed rows to {args.output_csv}", flush=True)
    print(f"Wrote grouped summary to {args.output_summary_csv}", flush=True)


if __name__ == "__main__":
    main()
