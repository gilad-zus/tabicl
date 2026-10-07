"""Prepare nested real-data banks without model scores or test-bank access."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

from scripts import joint_preprocessing_real_meta_bank as bank
from scripts import joint_preprocessing_synthetic_pilot as pilot


GENERATED_DESCRIPTION = re.compile(
    r"synthetic (?:data(?:set)?|examples)|artificial (?:data(?:set)?|examples)|"
    r"synthetic (?:version|content)|(?:all )?content is synthetic|"
    r"randomly generated (?:data|examples)|simulated (?:data(?:set)?|examples)|"
    r"monte.?carlo (?:simulation|generated)", re.I)
KNOWN_GENERATED = {"higgs", "magictelescope", "magic", "estimationofobesitylevels",
                   "fitnessclub", "mobileprice", "ibmemployeeattrition", "ibmemployeeperformance",
                   "loanapprovalstatus", "dynamicallygeneratedhatespeechdataset", "studentsscores"}


def validate_provenance(entry, description):
    if bank.normalized_name(entry["name"]) in KNOWN_GENERATED:
        raise ValueError("excluded generated or uncertain source")
    if GENERATED_DESCRIPTION.search(description or ""):
        raise ValueError("source description identifies generated/simulated data")


def stable_order(values, seed, key):
    return sorted(values, key=lambda v: hashlib.sha256(f"{seed}:{key(v)}".encode()).hexdigest())


def descriptors(frame, labels, e):
    """Unsupervised shape descriptors plus permitted source-label class balance."""
    columns = bank.pack_frame(frame)
    if not hasattr(frame, "iloc"):
        frame = bank.unpack_frame(columns)
    numeric = [i for i, c in enumerate(columns) if c["kind"] == "numerical"]
    raw = frame.iloc[np.linspace(0, len(frame) - 1, min(1024, len(frame)), dtype=int)]
    missing = float(raw.isna().to_numpy().mean())
    skew = correlation = 0.
    if numeric:
        x = raw.iloc[:, numeric].to_numpy(dtype=np.float64, copy=True)
        skew_values = raw.iloc[:, numeric].skew().to_numpy(dtype=np.float64)
        finite = np.isfinite(skew_values)
        skew = float(np.median(np.abs(skew_values[finite]))) if finite.any() else 0.
        if x.shape[1] > 1:
            # Bound cost, and make the descriptor finite for all-missing columns.
            x = x[:, :64]
            for i in range(x.shape[1]):
                v = x[:, i]
                good = np.isfinite(v)
                v[~good] = np.median(v[good]) if good.any() else 0.
            x = x[:, x.std(0) > 1e-10]
            if x.shape[1] > 1:
                c = np.corrcoef(x, rowvar=False)
                correlation = float(np.median(np.abs(c[np.triu_indices_from(c, 1)])))
    _, counts = np.unique(labels, return_counts=True)
    return dict(rows=len(labels), classes=len(counts), features=e["x_context"].shape[-1],
                categorical_fraction=1 - float(e["numerical_mask"].float().mean()),
                missing_fraction=missing, median_absolute_skew=skew,
                median_absolute_correlation=correlation, minority_fraction=float(counts.min() / counts.sum()))


def stratum(d):
    return f"{'binary' if d['classes'] == 2 else 'multi'}:{np.searchsorted([10, 30], d['features'])}:{int(d['categorical_fraction'] >= .25)}"


def balanced_select(records, count, seed):
    """Round-robin cells, proportional deficit selection within represented cells."""
    if not 0 <= count <= len(records):
        raise ValueError("balanced selection count exceeds available sources")
    cells = {}
    for r in records:
        cells.setdefault(stratum(r["descriptors"]), []).append(r)
    cells = {k: stable_order(v, seed, lambda r: r["family"]) for k, v in cells.items()}
    selected, used = [], dict.fromkeys(cells, 0)
    for _ in range(count):
        remaining = [k for k in cells if used[k] < len(cells[k])]
        # Equal fraction selected across cells; deterministic tie breaking.
        k = min(remaining, key=lambda k: (used[k] / len(cells[k]), k))
        selected.append(cells[k][used[k]])
        used[k] += 1
    return selected


def allocate(records, counts, seed):
    if len(records) != counts["large"] + counts["validation"]:
        raise ValueError("exact accepted-source count required")
    validation = balanced_select(records, counts["validation"], seed)
    val_names = {r["family"] for r in validation}
    large = [r for r in records if r["family"] not in val_names]
    small = balanced_select(large, counts["small"], seed + 1)
    return {"large": large, "small": small, "validation": validation}


def load_candidate(entry, cache_dir):
    validate_provenance(entry, None)
    description = None
    if entry["source"] == "openml":
        path = cache_dir / "metadata" / f"openml_{entry['data_id']}.json"
        if not path.exists():
            with urlopen(f"https://www.openml.org/api/v1/json/data/{entry['data_id']}", timeout=60) as response:
                pilot.json_write(path, json.load(response))
        metadata = json.loads(path.read_text())["data_set_description"]
        description = metadata.get("description", "")
        validate_provenance(entry, description)
    frame, labels, raw_hash = bank._load_candidate(entry, cache_dir)
    return frame, labels, raw_hash, description


def cached_entry_identity(entry):
    """Allow curation annotations to change, preserving every data-load field."""
    return {k: v for k, v in entry.items() if k not in {"source_group", "provenance"}}


def prepare(output_dir, candidate_path, cache_dir, reuse_source_cache=None):
    source = json.loads(candidate_path.read_text())
    source_hash = pilot.hash_file(candidate_path)
    lock = output_dir / "banks_manifest.json"
    if lock.exists():
        result = json.loads(lock.read_text())
        if result["candidate_sha256"] != source_hash:
            raise ValueError("frozen candidates changed")
        for e in result["banks"].values():
            if pilot.hash_file(output_dir / e["path"]) != e["sha256"]:
                raise ValueError("frozen bank changed")
        return result
    counts = source["target_counts"]
    wanted = counts["large"] + counts["validation"]
    status = dict(candidate_sha256=source_hash, accepted=[], failures=[], skipped=[], reused_cached_sources=0)
    accepted, fingerprints, used_groups = [], [], set()
    for entry in source["candidates"]:
        if len(accepted) == wanted:
            break
        group = bank.normalized_name(entry["source_group"])
        if group in used_groups:
            continue
        path = output_dir / "source_cache" / f"{hashlib.sha256(bank._entry_identity(entry).encode()).hexdigest()[:20]}.pt"
        read_path = path
        if not path.exists() and reuse_source_cache is not None:
            fallback = reuse_source_cache / path.name
            if fallback.exists():
                read_path = fallback
        try:
            if read_path.exists():
                saved = torch.load(read_path, weights_only=True)
                if cached_entry_identity(saved["entry"]) != cached_entry_identity(entry):
                    raise ValueError("cached source identity changed")
                frame, labels = bank.unpack_frame(saved["columns"]), np.asarray(saved["labels"])
                raw_hash, description = saved["raw_hash"], saved["description"]
                status["reused_cached_sources"] += 1
            else:
                frame, labels, raw_hash, description = load_candidate(entry, cache_dir)
                pilot.atomic_save(path, dict(entry=entry, columns=bank.pack_frame(frame),
                    labels=labels.tolist(), raw_hash=raw_hash, description=description))
            # A reused raw cache must never bypass a stricter provenance audit.
            validate_provenance(entry, description)
            _, class_counts = np.unique(labels, return_counts=True)
            if len(labels) < source["minimum_rows"] or not 2 <= len(class_counts) <= 10 or class_counts.min() < 2:
                raise ValueError("ineligible source rows/classes/class coverage")
            fp = bank.content_fingerprints(frame, labels)
            try:
                bank._reject_duplicate(fp, entry["name"], fingerprints)
            except ValueError as error:
                # Allocation happens after this audit, so removing a copied source
                # cannot create train/validation overlap or use model outcomes.
                status["skipped"].append(dict(entry=entry, reason=str(error)))
                print(f"Duplicate skipped: {entry['name']}: {error}", flush=True)
                continue
            indices = bank._sample_indices(labels, min(len(labels), source["max_source_rows"]), np.random.default_rng(source["seed"]))
            raw = dict(family=entry["name"], source_group=entry["source_group"], source=entry["source"],
                columns=bank.pack_frame(bank.real._take_rows(frame, indices)), labels=labels[indices].tolist(),
                source_indices=indices.tolist(), inputs_sha256=fp["inputs_sha256"], source_entry=entry)
            e = bank.sample_real_episode(raw, 1024, .7, source["seed"])
            bank.real._prepared_views(e, 8)
            for rows in (128, 256, 512, 1024):
                for fraction in (.5, .7, .85):
                    bank.real._prepared_views(bank.sample_real_episode(raw, rows, fraction, source["seed"]), 8)
            raw["descriptors"] = descriptors(frame, labels, e)
            accepted.append(raw)
            fingerprints.append(dict(family=entry["name"], fingerprints=fp))
            used_groups.add(group)
            status["accepted"].append(dict(entry=entry, descriptors=raw["descriptors"], inputs_sha256=fp["inputs_sha256"],
                description_sha256=hashlib.sha256((description or "").encode()).hexdigest(), n_retained_rows=len(indices)))
            print(f"Accepted {len(accepted)}/{wanted}: {entry['name']} [{entry['source_group']}]", flush=True)
        except Exception as error:
            status["failures"].append(dict(entry=entry, reason=f"{type(error).__name__}: {error}"))
            print(f"Rejected {entry['name']}: {type(error).__name__}: {error}", flush=True)
        pilot.json_write(output_dir / "availability.json", status)
    if len(accepted) != wanted:
        raise RuntimeError(f"Only {len(accepted)}/{wanted} independent eligible sources; no GPU run may begin")
    panels = allocate(accepted, counts, source["seed"])
    extra = [r for r in panels["large"] if r["family"] not in {r["family"] for r in panels["small"]}]
    extra_probe = balanced_select(extra, counts["small"], source["seed"] + 2)
    values = dict(small_train=panels["small"], large_train=panels["large"],
        real_probe=[bank.sample_real_episode(f, 1024, .7, source["seed"]) for f in panels["small"]],
        large_only_probe=[bank.sample_real_episode(f, 1024, .7, source["seed"]) for f in extra_probe],
        real_validation=[bank.sample_real_episode(f, 1024, .7, s) for f in panels["validation"] for s in source["split_seeds"]])
    banks = {}
    for name, v in values.items():
        path = output_dir / "banks" / f"{name}.pt"
        pilot.atomic_save(path, dict(values=v))
        banks[name] = dict(path=path.relative_to(output_dir).as_posix(), count=len(v), sha256=pilot.hash_file(path))
    selection = {k: [dict(name=r["family"], source_group=r["source_group"], descriptors=r["descriptors"],
                          inputs_sha256=r["inputs_sha256"], source_entry=r["source_entry"]) for r in v] for k, v in panels.items()}
    # Coverage is descriptive, rather than a claimed distance between useful maps.
    coverage = {}
    for name in ("small", "large", "validation"):
        rows = panels[name]
        coverage[name] = {k: {str(q): float(np.percentile([r["descriptors"][k] for r in rows], q))
                             for q in (0, 25, 50, 75, 100)} for k in rows[0]["descriptors"]}
        coverage[name]["strata"] = {s: sum(stratum(r["descriptors"]) == s for r in rows)
                                     for s in sorted({stratum(r["descriptors"]) for r in accepted})}
    for name in ("small", "large"):
        coverage[name]["validation_outside_training_range"] = {
            k: sum(not min(r["descriptors"][k] for r in panels[name]) <= v["descriptors"][k]
                       <= max(r["descriptors"][k] for r in panels[name]) for v in panels["validation"])
            for k in panels["validation"][0]["descriptors"]}
    result = dict(format_version=1, candidate_sha256=source_hash, counts=counts,
        selection=selection, banks=banks, coverage=coverage, scope=source["scope"],
        note="25 independent validation source groups, reused for development only; no confirmation test bank opened")
    pilot.json_write(lock, result)
    print(f"Frozen nested banks: {lock}", flush=True)
    return result


def load_bank(root, manifest, panel):
    if panel not in {"small_train", "large_train", "real_probe", "large_only_probe", "real_validation"}:
        raise ValueError("unknown panel; this runner has no test-bank loader")
    entry = manifest["banks"][panel]
    path = root / entry["path"]
    if pilot.hash_file(path) != entry["sha256"]:
        raise ValueError("bank hash changed")
    values = torch.load(path, weights_only=True, map_location="cpu")["values"]
    if len(values) != entry["count"]:
        raise ValueError("incomplete bank")
    return values
