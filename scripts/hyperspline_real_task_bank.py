"""Build deterministic, dataset-disjoint real tabular episode banks.

The bank is deliberately separate from the seven final real evaluation
datasets.  Each stored episode fits imputation and categorical encoding on its
context rows only, exactly as the final real evaluator does.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

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


# PMLB is a curated, versioned benchmark suite published by Olson et al.
# (2017). It avoids depending on the OpenML API endpoint that returned 504
# for every previous candidate. These pools exclude the final seven datasets.
DEFAULT_TRAIN_PMLB_DATASETS = (
    "allbp,allrep,backache,balance_scale,biomed,bupa,chess,coil2000,connect_4,dermatology,"
    "ecoli,hepatitis,ionosphere,kr_vs_kp,lymphography,magic,mfeat_factors,mfeat_karhunen,"
    "mfeat_morphological,mfeat_zernike,mushroom,new_thyroid,nursery,page_blocks,pendigits,"
    "phoneme,saheart,satimage,sonar,spambase,spect,spectf,splice,tic_tac_toe,vehicle,"
    "waveform_21,waveform_40,wine_quality_red,yeast"
)
DEFAULT_VALIDATION_PMLB_DATASETS = (
    "analcatdata_authorship,appendicitis,clean1,clean2,confidence,haberman,hayes_roth,"
    "labor,postoperative_patient_data,prnn_crabs,tae,titanic"
)
# Existing callers refer to these names. Keep aliases while changing the
# actual source from numeric OpenML IDs to stable PMLB dataset names.
DEFAULT_TRAIN_OPENML_IDS = DEFAULT_TRAIN_PMLB_DATASETS
DEFAULT_VALIDATION_OPENML_IDS = DEFAULT_VALIDATION_PMLB_DATASETS
# OpenML counterparts of iris, wine, breast-cancer, digits, adult, credit-g,
# and bank-marketing: the permanent final paired-evaluation suite.
FINAL_EVALUATION_OPENML_IDS = frozenset({15, 31, 61, 187, 554, 1461, 1590})
FINAL_EVALUATION_PMLB_DATASETS = frozenset(
    {"adult", "bank_marketing", "breast_cancer", "breast_cancer_wisconsin", "credit_g", "digits", "iris", "optdigits", "wine"}
)
FORMAT_VERSION = 1
PMLB_RAW_URLS = (
    "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz",
    "https://github.com/EpistasisLab/pmlb/raw/master/datasets/{name}/{name}.tsv.gz",
)


def parse_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids or any(item <= 0 for item in ids):
        raise ValueError("expected a non-empty comma-separated list of positive OpenML IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("OpenML IDs must be unique within a split")
    return ids


def parse_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("expected a non-empty comma-separated list of unique dataset names")
    return names


def load_pmlb_frame(name: str, *, cache_dir: Path):
    """Fetch one compressed PMLB table from GitHub once, then reuse it locally."""
    cache_path = cache_dir / f"{name}.tsv.gz"
    if not cache_path.is_file():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for template in PMLB_RAW_URLS:
            try:
                with urlopen(template.format(name=name), timeout=120) as response:
                    payload = response.read()
                if payload[:2] != b"\x1f\x8b":
                    raise ValueError("response was not a gzip PMLB table")
                cache_path.write_bytes(payload)
                break
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")
        else:
            raise RuntimeError("; ".join(errors))
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        table = np.genfromtxt(handle, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    if table.dtype.names is None or "target" not in table.dtype.names:
        raise ValueError(f"PMLB dataset {name} has no target column")
    feature_names = [column for column in table.dtype.names if column != "target"]
    try:
        x = np.column_stack([np.asarray(table[column], dtype=np.float32) for column in feature_names])
    except (TypeError, ValueError) as error:
        raise ValueError(f"PMLB dataset {name} contains non-numeric features after PMLB encoding") from error
    return x, np.asarray(table["target"]), name


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


def build_episode_from_pmlb(
    name: str,
    seed: int,
    *,
    test_size: float,
    max_rows: int,
    max_classes: int,
    cache_dir: Path,
    device: torch.device,
) -> RealEpisode:
    x, target, source_name = load_pmlb_frame(name, cache_dir=cache_dir)
    target = LabelEncoder().fit_transform(np.asarray(target))
    if np.unique(target).size < 2 or np.unique(target).size > max_classes:
        raise ValueError(
            f"PMLB {name} has {np.unique(target).size} classes; expected 2..{max_classes}"
        )
    if max_rows > 0 and x.shape[0] > max_rows:
        x, _, target, _ = train_test_split(
            x, target, train_size=max_rows, random_state=seed, stratify=target
        )
    if x.shape[1] == 0:
        raise ValueError(f"PMLB {source_name} has no numerical columns for HyperSpline")
    x_context, x_query, y_context, y_query = train_test_split(
        x, target, test_size=test_size, random_state=seed, stratify=target
    )
    return RealEpisode(
        dataset=f"pmlb_{name}",
        dataset_group="real_meta",
        split_seed=seed,
        n_context=x_context.shape[0],
        n_query=x_query.shape[0],
        n_features=x_context.shape[1],
        n_numerical_features=x_context.shape[1],
        n_categorical_features=0,
        n_classes=int(np.unique(y_context).size),
        x_context=torch.as_tensor(x_context, dtype=torch.float32, device=device).unsqueeze(0),
        x_query=torch.as_tensor(x_query, dtype=torch.float32, device=device).unsqueeze(0),
        y_context=torch.as_tensor(y_context, dtype=torch.float32, device=device).unsqueeze(0),
        y_query=torch.as_tensor(y_query, dtype=torch.long, device=device),
        numerical_mask=torch.tensor(
            [True] * x_context.shape[1], dtype=torch.bool, device=device
        ),
    )


def save_bank(path: Path, episodes: list[RealEpisode], *, dataset_names: list[str], split: str, args: argparse.Namespace) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "source": "pmlb",
        "dataset_names": dataset_names,
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
    names: list[str], *, split: str, episodes_per_dataset: int, seed: int, args: argparse.Namespace, device: torch.device
) -> tuple[list[RealEpisode], list[str], dict[str, str]]:
    episodes = []
    accepted_names, skipped = [], {}
    for dataset_index, name in enumerate(names):
        try:
            for offset in range(episodes_per_dataset):
                episode_seed = seed + 10_000 * dataset_index + offset
                episode = build_episode_from_pmlb(
                    name, episode_seed, test_size=args.test_size, max_rows=args.max_rows,
                    max_classes=args.max_classes, cache_dir=args.pmlb_cache_dir, device=device,
                )
                episodes.append(episode)
                print(
                    f"[{split} PMLB={name} seed={episode_seed}] context={episode.n_context}, "
                    f"query={episode.n_query}, numerical={episode.n_numerical_features}, "
                    f"categorical={episode.n_categorical_features}", flush=True,
                )
            accepted_names.append(name)
        except Exception as error:  # Availability and schema are tested before a costly run.
            skipped[name] = f"{type(error).__name__}: {error}"
            print(f"[{split} PMLB={name}] skipped: {skipped[name]}", flush=True)
    return episodes, accepted_names, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pmlb-datasets", "--train-openml-ids", dest="train_pmlb_datasets", default=DEFAULT_TRAIN_PMLB_DATASETS)
    parser.add_argument("--validation-pmlb-datasets", "--validation-openml-ids", dest="validation_pmlb_datasets", default=DEFAULT_VALIDATION_PMLB_DATASETS)
    parser.add_argument("--pmlb-cache-dir", type=Path, default=Path("results/pmlb_cache"))
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
    train_names, validation_names = parse_names(args.train_pmlb_datasets), parse_names(args.validation_pmlb_datasets)
    if set(train_names).intersection(validation_names):
        raise ValueError("real meta train and validation PMLB datasets must be disjoint")
    leaked = (set(train_names) | set(validation_names)).intersection(FINAL_EVALUATION_PMLB_DATASETS)
    if leaked:
        raise ValueError(f"real meta banks must exclude final-evaluation PMLB datasets: {sorted(leaked)}")
    if args.episodes_per_dataset <= 0 or args.min_train_datasets <= 0 or args.min_validation_datasets <= 0 or not 0 < args.test_size < 1 or args.max_rows < 0:
        raise ValueError("invalid episode count, test size, or max rows")
    if not args.verify_only and (args.train_output is None or args.validation_output is None):
        raise ValueError("--train-output and --validation-output are required unless --verify-only is used")
    device = torch.device("cpu")
    # Availability validation needs one real preprocessing pass per candidate;
    # avoid needlessly repeating seed splits before the actual experiment.
    episodes_per_dataset = 1 if args.verify_only else args.episodes_per_dataset
    train, accepted_train, skipped_train = make_bank(
        train_names, split="train", episodes_per_dataset=episodes_per_dataset,
        seed=args.train_seed, args=args, device=device,
    )
    validation, accepted_validation, skipped_validation = make_bank(
        validation_names, split="validation", episodes_per_dataset=episodes_per_dataset,
        seed=args.validation_seed, args=args, device=device,
    )
    if len(accepted_train) < args.min_train_datasets or len(accepted_validation) < args.min_validation_datasets:
        raise RuntimeError(
            f"insufficient eligible datasets: train={len(accepted_train)}/{args.min_train_datasets}, "
            f"validation={len(accepted_validation)}/{args.min_validation_datasets}"
        )
    manifest = {
        "source": "pmlb", "candidate_train_datasets": train_names,
        "candidate_validation_datasets": validation_names,
        "accepted_train_datasets": accepted_train, "accepted_validation_datasets": accepted_validation,
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
        save_bank(args.train_output, train, dataset_names=accepted_train, split="train", args=args)
        save_bank(args.validation_output, validation, dataset_names=accepted_validation, split="validation", args=args)


if __name__ == "__main__":
    main()
