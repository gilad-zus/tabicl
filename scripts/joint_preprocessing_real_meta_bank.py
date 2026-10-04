"""Freeze family-disjoint real banks before preprocessing continuation training.

Candidate order is declared in a checked-in manifest. Replacements depend only
on data availability/eligibility, never on model scores. The four bank files
keep final-test features and labels out of the training/validation loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from scripts import joint_preprocessing_real_transfer as real
from scripts.joint_preprocessing_synthetic_pilot import atomic_save, hash_file, json_write
from tabicl._sklearn.preprocessing import TransformToNumerical


DEFAULT_CANDIDATES = Path(__file__).resolve().parents[1] / "docs/experiments/joint_preprocessing_real_meta_candidates_20261004.json"
PANELS = ("train", "validation", "test")


def normalized_name(value: str) -> str:
    value = re.sub(r"_seed_\d+_nrows_.*", "", value.lower())
    return re.sub(r"[^a-z0-9]", "", value)


def _python_value(value: Any) -> str | int | float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def pack_frame(frame) -> list[dict[str, Any]]:
    """Keep dtype information while using only weights-only-safe containers."""
    if not hasattr(frame, "columns"):
        frame = pd.DataFrame(np.asarray(frame), columns=[f"x{i}" for i in range(np.asarray(frame).shape[1])])
    columns = []
    for index in range(frame.shape[1]):
        series = frame.iloc[:, index]
        categorical = (str(series.dtype).startswith(("category", "string"))
                       or series.dtype == object or str(series.dtype) == "bool")
        values = [_python_value(value) for value in series]
        if categorical:
            values = [None if value is None else str(value) for value in values]
        columns.append(dict(name=str(frame.columns[index]), kind="categorical" if categorical else "numerical", values=values))
    return columns


def unpack_frame(columns: list[dict[str, Any]], indices: np.ndarray | None = None) -> pd.DataFrame:
    values = {}
    for index, column in enumerate(columns):
        # Unique internal names prevent duplicate source headers collapsing.
        key = f"column_{index}"
        selected = column["values"] if indices is None else [column["values"][int(i)] for i in indices]
        if column["kind"] == "categorical":
            values[key] = pd.Series([np.nan if v is None else str(v) for v in selected], dtype=object)
        else:
            values[key] = pd.Series(selected, dtype=np.float64)
    return pd.DataFrame(values)


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    try:
        number = float(value)
        if math.isfinite(number):
            return format(number, ".12g")
    except (TypeError, ValueError):
        pass
    return "s:" + str(value).strip()


def content_fingerprints(frame, labels: np.ndarray) -> dict[str, Any]:
    """Detect copies despite row/column order and harmless numeric formatting.

    The feature-only hash also catches identical tables with different targets.
    Known source-group aliases handle feature subsets/alternate target tables;
    hashes alone cannot prove that all independently collected tables are unrelated.
    """
    if not hasattr(frame, "columns"):
        frame = pd.DataFrame(np.asarray(frame))
    columns = [[_canonical_value(value) for value in frame.iloc[:, i]] for i in range(frame.shape[1])]
    ordered = sorted(columns, key=lambda column: hashlib.sha256("\x1e".join(sorted(column)).encode()).hexdigest())
    rows = [json.dumps(row, ensure_ascii=True, separators=(",", ":")) for row in zip(*ordered)]
    row_hashes = [hashlib.sha256(row.encode()).hexdigest() for row in rows]
    inputs_hash = hashlib.sha256("\n".join(sorted(row_hashes)).encode()).hexdigest()
    labeled_rows = [f"{row}:{int(label)}" for row, label in zip(row_hashes, labels)]
    return dict(inputs_sha256=inputs_hash,
                table_sha256=hashlib.sha256("\n".join(sorted(labeled_rows)).encode()).hexdigest(),
                row_hashes=set(row_hashes), n_features=frame.shape[1])


def _allocation(counts: np.ndarray, total: int, *, minimum: int, upper: np.ndarray | None = None) -> np.ndarray:
    """Preserve proportions subject to explicit class-coverage constraints."""
    upper = counts if upper is None else upper
    if total < minimum * len(counts) or total > int(upper.sum()) or np.any(upper < minimum):
        raise ValueError("requested sample cannot preserve class coverage")
    target = total * counts / counts.sum()
    result = np.minimum(upper, np.maximum(minimum, np.floor(target).astype(int)))
    while int(result.sum()) < total:
        residual = np.where(result < upper, target - result, -np.inf)
        result[int(np.argmax(residual))] += 1
    while int(result.sum()) > total:
        residual = np.where(result > minimum, result - target, -np.inf)
        result[int(np.argmax(residual))] -= 1
    return result


def _sample_indices(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    quota = _allocation(counts, size, minimum=2)
    indices = np.concatenate([rng.choice(np.flatnonzero(labels == label), int(count), replace=False)
                              for label, count in zip(classes, quota)])
    rng.shuffle(indices)
    return indices


def sample_real_episode(family_dict: dict[str, Any], n_rows: int, context_fraction: float,
                        seed: int) -> dict[str, Any]:
    """Draw new context/query rows and fit all encoders on context only.

    ``n_rows`` may exceed the retained family pool; actual/requested sizes are
    both recorded. Both parts contain every family class, including rare classes.
    """
    if not 0 < context_fraction < 1:
        raise ValueError("context_fraction must be strictly between zero and one")
    if n_rows < 4:
        raise ValueError("n_rows must be at least four")
    labels = np.asarray(family_dict["labels"], dtype=np.int64)
    indices = _sample_indices(labels, min(n_rows, len(labels)), np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1_000_003)
    selected_labels = labels[indices]
    classes, counts = np.unique(selected_labels, return_counts=True)
    query_size = int(math.ceil(len(indices) * (1 - context_fraction)))
    query_quota = _allocation(counts, query_size, minimum=1, upper=counts - 1)
    context, query = [], []
    for label, count in zip(classes, query_quota):
        local = np.flatnonzero(selected_labels == label)
        rng.shuffle(local)
        query.extend(local[:count])
        context.extend(local[count:])
    context, query = np.asarray(context, dtype=int), np.asarray(query, dtype=int)
    rng.shuffle(context)
    rng.shuffle(query)
    frame = unpack_frame(family_dict["columns"], indices)
    x_context_raw = frame.iloc[context].reset_index(drop=True)
    x_query_raw = frame.iloc[query].reset_index(drop=True)
    encoder = TransformToNumerical().fit(x_context_raw)
    cp, qp = encoder.transform_parts(x_context_raw), encoder.transform_parts(x_query_raw)
    n_categorical, n_numerical = cp.categorical.shape[1], cp.numerical.shape[1]
    if not 5 <= n_categorical + n_numerical <= 100 or n_numerical < 1:
        raise ValueError(f"{family_dict['family']}: expected 5-100 encoded features and at least one numerical feature")
    xc = np.concatenate((cp.categorical, cp.numerical), axis=1).astype(np.float32)
    xq = np.concatenate((qp.categorical, qp.numerical), axis=1).astype(np.float32)
    if not np.isfinite(xc).all() or not np.isfinite(xq).all():
        raise ValueError(f"{family_dict['family']}: nonfinite context/query features")
    source_indices = np.asarray(family_dict["source_indices"], dtype=np.int64)
    return dict(family=family_dict["family"], source_group=family_dict["source_group"], split_seed=int(seed),
                n_classes=int(len(classes)), categorical_features=list(range(n_categorical)),
                context_indices=torch.from_numpy(source_indices[indices[context]].copy()),
                query_indices=torch.from_numpy(source_indices[indices[query]].copy()),
                x_context=torch.from_numpy(xc).unsqueeze(0), x_query=torch.from_numpy(xq).unsqueeze(0),
                y_context=torch.from_numpy(selected_labels[context].astype(np.float32)).unsqueeze(0),
                y_query=torch.from_numpy(selected_labels[query].copy()),
                numerical_mask=torch.tensor([False] * n_categorical + [True] * n_numerical),
                context_missing=torch.from_numpy(cp.numerical_missing.copy()),
                query_missing=torch.from_numpy(qp.numerical_missing.copy()),
                requested_rows=int(n_rows), actual_rows=int(len(indices)),
                n_context=int(len(context)), n_query=int(len(query)),
                requested_context_fraction=float(context_fraction), actual_context_fraction=float(len(context) / len(indices)))


def _load_candidate(entry: dict[str, Any], cache_dir: Path):
    if entry["source"] == "openml" and "data_id" in entry:
        dataset = fetch_openml(data_id=int(entry["data_id"]), as_frame=True, data_home=str(cache_dir / "openml"))
        actual_name = str(dataset.details.get("name", ""))
        if normalized_name(actual_name) not in {normalized_name(x) for x in [entry["name"], *entry.get("aliases", [])]}:
            raise ValueError(f"OpenML ID {entry['data_id']} unexpectedly names {actual_name!r}")
        frame, target, source_hash = dataset.data.copy(), np.asarray(dataset.target), None
    else:
        frame, target, source_hash = real._load_source(entry, cache_dir)
    target = np.asarray(target)
    valid = np.asarray([not pd.isna(value) for value in target], dtype=bool)
    frame = real._take_rows(frame, np.flatnonzero(valid))
    labels = LabelEncoder().fit_transform(target[valid])
    return frame, labels, source_hash


def _entry_identity(entry: dict[str, Any]) -> str:
    return f"{entry['source']}:{entry.get('data_id', entry['name'])}"


def _validate_candidates(source: dict[str, Any]) -> None:
    if source.get("format_version") != 1:
        raise ValueError("unsupported candidate manifest format")
    used_groups, used_names, used_sources = {}, {}, {}
    historical = {normalized_name(name) for name in source.get("historically_inspected_names", [])}
    historical_groups = {normalized_name(name) for name in source.get("historically_inspected_source_groups", [])}
    for panel in PANELS:
        if not source["candidates"].get(panel):
            raise ValueError(f"empty {panel} candidate list")
        for entry in source["candidates"][panel]:
            group = normalized_name(entry["source_group"])
            names = {normalized_name(name) for name in [entry["name"], *entry.get("aliases", [])]}
            identity = _entry_identity(entry)
            for value, owners, label in [(group, used_groups, "source group"), (identity, used_sources, "source ID")]:
                if value in owners and owners[value] != panel:
                    raise ValueError(f"{label} {value!r} crosses {owners[value]}/{panel}")
                owners[value] = panel
            for name in names:
                if name in used_names and used_names[name] != panel:
                    raise ValueError(f"source alias {name!r} crosses {used_names[name]}/{panel}")
                used_names[name] = panel
            if panel == "test" and (names & historical or group in historical_groups):
                raise ValueError(f"fresh test candidate {entry['name']} overlaps historical development")


def _reject_duplicate(fingerprints: dict[str, Any], family: str, accepted: list[dict[str, Any]]) -> None:
    for previous in accepted:
        old = previous["fingerprints"]
        same = fingerprints["inputs_sha256"] == old["inputs_sha256"]
        # Large shared row subsets flag copies/subsamples even under relabeling.
        shared = 0
        smaller = min(len(fingerprints["row_hashes"]), len(old["row_hashes"]))
        if fingerprints["n_features"] == old["n_features"] and smaller >= 128:
            shared = len(fingerprints["row_hashes"] & old["row_hashes"])
        if same or (shared >= 128 and shared / smaller >= .8):
            raise ValueError(f"unsafe duplicate source content: {family} and {previous['family']}")


def prepare_real_bank(output_dir: Path, candidate_manifest_path: Path, cache_dir: Path) -> dict[str, Any]:
    output_dir, candidate_manifest_path, cache_dir = map(Path, (output_dir, candidate_manifest_path, cache_dir))
    source = json.loads(candidate_manifest_path.read_text(encoding="utf8"))
    _validate_candidates(source)
    source_hash = hash_file(candidate_manifest_path)
    lock_path = output_dir / "real_manifest.json"
    if lock_path.exists():
        manifest = json.loads(lock_path.read_text(encoding="utf8"))
        if manifest["candidate_manifest_sha256"] != source_hash:
            raise ValueError("candidate manifest changed after bank was frozen")
        for entry in manifest["banks"].values():
            if hash_file(output_dir / entry["path"]) != entry["sha256"]:
                raise ValueError("frozen real bank hash mismatch")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    availability_path = output_dir / "availability.json"
    status = dict(format_version=1, candidate_manifest_sha256=source_hash, policy="availability and eligibility only; no model scores",
                  accepted={}, failures=[], skipped=[])
    panels: dict[str, list[dict[str, Any]]] = {panel: [] for panel in PANELS}
    accepted, retained_groups = [], set()
    cached_dir = output_dir / "source_cache"
    settings = source["eligibility"]
    for panel in PANELS:
        wanted = int(source["target_counts"][panel])
        for rank, entry in enumerate(source["candidates"][panel]):
            if len(panels[panel]) == wanted:
                break
            group = normalized_name(entry["source_group"])
            if group in retained_groups:
                status["skipped"].append(dict(panel=panel, rank=rank, entry=entry, reason="related source group already selected"))
                continue
            try:
                identity_hash = hashlib.sha256((_entry_identity(entry) + source_hash).encode()).hexdigest()[:20]
                source_path = cached_dir / f"{identity_hash}.pt"
                if source_path.exists():
                    cached = torch.load(source_path, map_location="cpu", weights_only=True)
                    frame = unpack_frame(cached["columns"])
                    labels = np.asarray(cached["labels"], dtype=np.int64)
                    raw_hash = cached["raw_source_sha256"]
                else:
                    frame, labels, raw_hash = _load_candidate(entry, cache_dir)
                    atomic_save(source_path, dict(columns=pack_frame(frame), labels=labels.tolist(), raw_source_sha256=raw_hash))
                classes, counts = np.unique(labels, return_counts=True)
                if len(labels) < int(settings["min_rows"]) or not int(settings["min_classes"]) <= len(classes) <= int(settings["max_classes"]):
                    raise ValueError(f"ineligible rows/classes: {len(labels)}/{len(classes)}")
                if counts.min() < 2:
                    raise ValueError("at least one class has fewer than two usable rows")
                fingerprints = content_fingerprints(frame, labels)
                # Content duplicates across declared groups are a manifest error:
                # do not replace silently and proceed to potentially unsafe GPU work.
                _reject_duplicate(fingerprints, entry["name"], accepted)
                retained = _sample_indices(labels, min(len(labels), int(source["max_source_rows"])),
                                           np.random.default_rng(int(source["source_sample_seed"])))
                raw = dict(family=entry["name"], source_group=entry["source_group"], source=entry["source"],
                           source_entry=entry, columns=pack_frame(real._take_rows(frame, retained)),
                           labels=labels[retained].tolist(), source_indices=retained.tolist(),
                           n_original_rows=int(len(labels)), n_retained_rows=int(len(retained)),
                           inputs_sha256=fingerprints["inputs_sha256"], table_sha256=fingerprints["table_sha256"],
                           raw_source_sha256=raw_hash)
                episodes = [sample_real_episode(raw, int(source["max_episode_rows"]), float(source["context_fraction"]), int(seed))
                            for seed in source["split_seeds"]]
                for episode in episodes:
                    real._prepared_views(episode, 8)
                # Cover every possible training size/fraction before GPU work.
                if panel == "train":
                    for rows in (128, 256, 512, 1024):
                        for fraction in (.5, .7, .85):
                            training_episode = sample_real_episode(raw, rows, fraction, 20_261_004)
                            real._prepared_views(training_episode, 8)
                retained_groups.add(group)
                accepted.append(dict(family=entry["name"], panel=panel, fingerprints=fingerprints))
                panels[panel].append(raw if panel == "train" else episodes)
                metadata = dict(entry, candidate_rank=rank, n_original_rows=int(len(labels)), n_retained_rows=int(len(retained)),
                                encoded_features=int(episodes[0]["x_context"].shape[-1]),
                                numerical_features=int(episodes[0]["numerical_mask"].sum()),
                                inputs_sha256=fingerprints["inputs_sha256"], table_sha256=fingerprints["table_sha256"],
                                retained_source_indices_sha256=hashlib.sha256(np.asarray(retained, dtype=np.int64).tobytes()).hexdigest())
                status["accepted"].setdefault(panel, []).append(metadata)
                print(f"Accepted {panel} {len(panels[panel])}/{wanted}: {entry['name']} ({len(labels)} rows)", flush=True)
            except ValueError as error:
                if "unsafe duplicate" in str(error):
                    status["failures"].append(dict(panel=panel, rank=rank, entry=entry, error=str(error), fatal=True))
                    json_write(availability_path, status)
                    raise
                status["failures"].append(dict(panel=panel, rank=rank, entry=entry, error=f"{type(error).__name__}: {error}"))
                print(f"Rejected {panel} {entry['name']}: {error}", flush=True)
            except Exception as error:
                status["failures"].append(dict(panel=panel, rank=rank, entry=entry, error=f"{type(error).__name__}: {error}"))
                print(f"Unavailable {panel} {entry['name']}: {type(error).__name__}: {error}", flush=True)
            json_write(availability_path, status)
        if len(panels[panel]) != wanted:
            raise RuntimeError(f"real {panel} pool has {len(panels[panel])}/{wanted} eligible independent families; no GPU run may begin")
    banks = {}
    episodes_by_panel = {panel: [episode for pair in panels[panel] for episode in pair] for panel in ("validation", "test")}
    episodes_by_panel["probe"] = [sample_real_episode(raw, int(source["max_episode_rows"]), float(source["context_fraction"]),
                                                      int(source["probe_seed"])) for raw in panels["train"]]
    for panel in ("train", "probe", "validation", "test"):
        path = output_dir / "banks" / f"real_{panel}.pt"
        payload = dict(format_version=1, **({"families": panels[panel]} if panel == "train" else {"episodes": episodes_by_panel[panel]}))
        atomic_save(path, payload)
        banks[f"real_{panel}"] = dict(path=path.relative_to(output_dir).as_posix(), sha256=hash_file(path),
                            count=len(panels[panel]) if panel == "train" else len(episodes_by_panel[panel]),
                            n_families=len(panels["train"]) if panel == "probe" else len(panels[panel]),
                            n_episodes=0 if panel == "train" else len(episodes_by_panel[panel]))
    manifest = dict(format_version=1, candidate_manifest_sha256=source_hash,
                    preparer_sha256=hash_file(Path(__file__)), target_counts=source["target_counts"],
                    banks=banks, families=status["accepted"],
                    split_seeds=source["split_seeds"], context_fraction=source["context_fraction"],
                    maximum_episode_rows=source["max_episode_rows"], maximum_retained_source_rows=source["max_source_rows"],
                    deduplication="declared source groups/aliases plus row/column-order-invariant input hashes and >=80% shared-row copies",
                    final_test_policy="separate file; do not deserialize until all six selected models are locked")
    json_write(lock_path, manifest)
    print(f"Frozen real bank manifest: {lock_path}", flush=True)
    return manifest


def load_real_bank(output_dir: Path, panel: str) -> list[dict[str, Any]]:
    if panel not in {"train", "probe", "validation", "test"}:
        raise ValueError(f"unknown real bank panel {panel!r}")
    manifest = json.loads((Path(output_dir) / "real_manifest.json").read_text(encoding="utf8"))
    metadata = manifest["banks"][f"real_{panel}"]
    path = Path(output_dir) / metadata["path"]
    if hash_file(path) != metadata["sha256"]:
        raise ValueError(f"{panel} bank hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("unsupported real bank format")
    values = payload["families" if panel == "train" else "episodes"]
    expected = metadata["n_families"] if panel == "train" else metadata["n_episodes"]
    if len(values) != expected:
        raise ValueError(f"incomplete {panel} bank")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/pmlb_cache"))
    args = parser.parse_args()
    prepare_real_bank(args.output_dir, args.candidate_manifest, args.cache_dir)


if __name__ == "__main__":
    main()
