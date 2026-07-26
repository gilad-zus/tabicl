"""Build deterministic, dataset-disjoint real tabular episode banks.

The bank is deliberately separate from the seven final real evaluation
datasets.  Each stored episode fits imputation and categorical encoding on its
context rows only, exactly as the final real evaluator does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

try:
    from scripts.hyperspline_real_zero_shot_eval import (
        DatasetSpec,
        RealEpisode,
        make_preprocessor,
        split_columns,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from hyperspline_real_zero_shot_eval import DatasetSpec, RealEpisode, make_preprocessor, split_columns
from sklearn.model_selection import train_test_split


# These OpenML IDs are intentionally disjoint from the final evaluation suite.
# They are exposed as CLI defaults rather than hidden assumptions, so a study
# can record and revise the exact real meta-distribution it uses.
# Candidate pools rather than a tiny fixed benchmark.  Some OpenML entries
# change metadata or are unsuitable (regression, too many classes, no numeric
# columns); ``--min-*-datasets`` turns those skips into an explicit validation
# result rather than silently reducing the intended dataset diversity.
DEFAULT_TRAIN_OPENML_IDS = "3,11,12,17,19,21,23,24,28,29,37,43,44,46,50,54,59,104,105,106,111,146,151,159,179,180,181,182,188,300,307,333"
DEFAULT_VALIDATION_OPENML_IDS = "4,7,8,9,13,14,16,18,22,32,36,38,40,41,49,53,57,58"
# OpenML counterparts of iris, wine, breast-cancer, digits, adult, credit-g,
# and bank-marketing: the permanent final paired-evaluation suite.
FINAL_EVALUATION_OPENML_IDS = frozenset({15, 31, 61, 187, 554, 1461, 1590})
FORMAT_VERSION = 1


def parse_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids or any(item <= 0 for item in ids):
        raise ValueError("expected a non-empty comma-separated list of positive OpenML IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("OpenML IDs must be unique within a split")
    return ids


def load_openml_frame(data_id: int):
    dataset = fetch_openml(data_id=data_id, as_frame=True)
    return dataset.data.copy(), dataset.target.copy(), str(dataset.details.get("name", f"openml_{data_id}"))


def build_episode_from_openml(
    data_id: int,
    seed: int,
    *,
    test_size: float,
    max_rows: int,
    max_classes: int,
    device: torch.device,
) -> RealEpisode:
    frame, target, source_name = load_openml_frame(data_id)
    valid_target = np.asarray([value is not None and str(value) != "nan" for value in target], dtype=bool)
    frame = frame.loc[valid_target].reset_index(drop=True)
    target = LabelEncoder().fit_transform(np.asarray(target)[valid_target])
    if np.unique(target).size < 2 or np.unique(target).size > max_classes:
        raise ValueError(
            f"OpenML {data_id} ({source_name}) has {np.unique(target).size} classes; expected 2..{max_classes}"
        )
    if max_rows > 0 and frame.shape[0] > max_rows:
        frame, _, target, _ = train_test_split(
            frame, target, train_size=max_rows, random_state=seed, stratify=target
        )
        frame = frame.reset_index(drop=True)
    numerical_columns, categorical_columns = split_columns(frame)
    if not numerical_columns:
        raise ValueError(f"OpenML {data_id} ({source_name}) has no numerical columns for HyperSpline")
    context_frame, query_frame, y_context, y_query = train_test_split(
        frame, target, test_size=test_size, random_state=seed, stratify=target
    )
    preprocessor = make_preprocessor(numerical_columns, categorical_columns)
    x_context = np.asarray(preprocessor.fit_transform(context_frame), dtype=np.float32)
    x_query = np.asarray(preprocessor.transform(query_frame), dtype=np.float32)
    return RealEpisode(
        dataset=f"openml_{data_id}",
        dataset_group="real_meta",
        split_seed=seed,
        n_context=x_context.shape[0],
        n_query=x_query.shape[0],
        n_features=x_context.shape[1],
        n_numerical_features=len(numerical_columns),
        n_categorical_features=len(categorical_columns),
        n_classes=int(np.unique(y_context).size),
        x_context=torch.as_tensor(x_context, dtype=torch.float32, device=device).unsqueeze(0),
        x_query=torch.as_tensor(x_query, dtype=torch.float32, device=device).unsqueeze(0),
        y_context=torch.as_tensor(y_context, dtype=torch.float32, device=device).unsqueeze(0),
        y_query=torch.as_tensor(y_query, dtype=torch.long, device=device),
        numerical_mask=torch.tensor(
            [True] * len(numerical_columns) + [False] * len(categorical_columns), dtype=torch.bool, device=device
        ),
    )


def save_bank(path: Path, episodes: list[RealEpisode], *, openml_ids: list[int], split: str, args: argparse.Namespace) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "openml_ids": openml_ids,
        "test_size": args.test_size,
        "max_rows": args.max_rows,
        "max_classes": args.max_classes,
        "episodes": [
            {
                "dataset": episode.dataset,
                "dataset_group": episode.dataset_group,
                "split_seed": episode.split_seed,
                "n_context": episode.n_context,
                "n_query": episode.n_query,
                "n_features": episode.n_features,
                "n_numerical_features": episode.n_numerical_features,
                "n_categorical_features": episode.n_categorical_features,
                "n_classes": episode.n_classes,
                "x_context": episode.x_context.cpu(),
                "x_query": episode.x_query.cpu(),
                "y_context": episode.y_context.cpu(),
                "y_query": episode.y_query.cpu(),
                "numerical_mask": episode.numerical_mask.cpu(),
            }
            for episode in episodes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved {split} real meta bank with {len(episodes)} episodes to {path}", flush=True)


def load_bank(path: Path, *, device: torch.device) -> tuple[dict, list[RealEpisode]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported real meta bank format: {path}")
    episodes = [
        RealEpisode(
            **{
                **item,
                "x_context": item["x_context"].to(device),
                "x_query": item["x_query"].to(device),
                "y_context": item["y_context"].to(device),
                "y_query": item["y_query"].to(device),
                "numerical_mask": item["numerical_mask"].to(device),
            }
        )
        for item in payload["episodes"]
    ]
    return payload, episodes


def make_bank(
    ids: list[int], *, split: str, episodes_per_dataset: int, seed: int, args: argparse.Namespace, device: torch.device
) -> tuple[list[RealEpisode], list[int], dict[int, str]]:
    episodes = []
    accepted_ids, skipped = [], {}
    for data_id in ids:
        try:
            for offset in range(episodes_per_dataset):
                episode_seed = seed + 10_000 * data_id + offset
                episode = build_episode_from_openml(
                    data_id, episode_seed, test_size=args.test_size, max_rows=args.max_rows,
                    max_classes=args.max_classes, device=device,
                )
                episodes.append(episode)
                print(
                    f"[{split} OpenML={data_id} seed={episode_seed}] context={episode.n_context}, "
                    f"query={episode.n_query}, numerical={episode.n_numerical_features}, "
                    f"categorical={episode.n_categorical_features}", flush=True,
                )
            accepted_ids.append(data_id)
        except Exception as error:  # Availability and schema are tested before a costly run.
            skipped[data_id] = f"{type(error).__name__}: {error}"
            print(f"[{split} OpenML={data_id}] skipped: {skipped[data_id]}", flush=True)
    return episodes, accepted_ids, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-openml-ids", default=DEFAULT_TRAIN_OPENML_IDS)
    parser.add_argument("--validation-openml-ids", default=DEFAULT_VALIDATION_OPENML_IDS)
    parser.add_argument("--episodes-per-dataset", type=int, default=4)
    parser.add_argument("--min-train-datasets", type=int, default=20)
    parser.add_argument("--min-validation-datasets", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=31_001)
    parser.add_argument("--validation-seed", type=int, default=41_001)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--availability-manifest", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true", help="Fetch and validate candidates without saving banks.")
    args = parser.parse_args()
    train_ids, validation_ids = parse_ids(args.train_openml_ids), parse_ids(args.validation_openml_ids)
    if set(train_ids).intersection(validation_ids):
        raise ValueError("real meta train and validation OpenML ID sets must be disjoint")
    leaked = (set(train_ids) | set(validation_ids)).intersection(FINAL_EVALUATION_OPENML_IDS)
    if leaked:
        raise ValueError(f"real meta banks must exclude final-evaluation OpenML IDs: {sorted(leaked)}")
    if args.episodes_per_dataset <= 0 or args.min_train_datasets <= 0 or args.min_validation_datasets <= 0 or not 0 < args.test_size < 1 or args.max_rows < 0:
        raise ValueError("invalid episode count, test size, or max rows")
    if not args.verify_only and (args.train_output is None or args.validation_output is None):
        raise ValueError("--train-output and --validation-output are required unless --verify-only is used")
    device = torch.device("cpu")
    # Availability validation needs one real preprocessing pass per candidate;
    # avoid needlessly repeating seed splits before the actual experiment.
    episodes_per_dataset = 1 if args.verify_only else args.episodes_per_dataset
    train, accepted_train, skipped_train = make_bank(
        train_ids, split="train", episodes_per_dataset=episodes_per_dataset,
        seed=args.train_seed, args=args, device=device,
    )
    validation, accepted_validation, skipped_validation = make_bank(
        validation_ids, split="validation", episodes_per_dataset=episodes_per_dataset,
        seed=args.validation_seed, args=args, device=device,
    )
    if len(accepted_train) < args.min_train_datasets or len(accepted_validation) < args.min_validation_datasets:
        raise RuntimeError(
            f"insufficient eligible datasets: train={len(accepted_train)}/{args.min_train_datasets}, "
            f"validation={len(accepted_validation)}/{args.min_validation_datasets}"
        )
    manifest = {
        "candidate_train_ids": train_ids, "candidate_validation_ids": validation_ids,
        "accepted_train_ids": accepted_train, "accepted_validation_ids": accepted_validation,
        "skipped_train": skipped_train, "skipped_validation": skipped_validation,
        "episodes_per_dataset": episodes_per_dataset, "max_rows": args.max_rows,
        "max_classes": args.max_classes,
    }
    manifest_path = args.availability_manifest or (
        args.train_output.parent / "real_meta_dataset_availability.json" if args.train_output else Path("real_meta_dataset_availability.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote dataset availability manifest to {manifest_path}", flush=True)
    if not args.verify_only:
        save_bank(args.train_output, train, openml_ids=accepted_train, split="train", args=args)
        save_bank(args.validation_output, validation, openml_ids=accepted_validation, split="validation", args=args)


if __name__ == "__main__":
    main()
