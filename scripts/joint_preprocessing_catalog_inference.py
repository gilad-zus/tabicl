"""Evaluate every eligible catalog source with frozen context-size checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_context_size as context_run

residual, pilot, previous = context_run.residual, context_run.pilot, context_run.previous
bank = context_run.bank
SEED = 20261017
METHODS = ("native16", "small_final", "small_selected", "large_final", "large_selected")
EXPOSURES = ("training_source", "validation_source", "previously_declared_source", "catalog_only_source")
ROOT = Path(__file__).resolve().parents[1]


def specification(args):
    settings = residual.settings(args)
    for path in (Path(__file__), Path(context_run.__file__), args.registry,
                 Path(__file__).with_name("run_joint_preprocessing_catalog_inference.sh"),
                 ROOT / "src/tabicl/_hyperspline/statistics_preprocessing.py"):
        settings["code_hashes"][path.relative_to(ROOT).as_posix()] = residual.objective.canonical_hash(path)
    settings.update(protocol="all eligible catalog groups; inference only", seed=SEED,
        methods=list(METHODS), contexts=[512, 2048], split_seeds=[0, 1],
        queries="256 permanently held-out rows on training sources; up to1024 elsewhere, minimum256",
        primary="small_final at512 on sources excluded from current training and validation",
        secondary="all other checkpoint/context/exposure comparisons; no reselection",
        deduplication="one accepted table per declared source group; whole-table content hashes",
        caveat="catalog-only means absent from conservative history registry, not independently proved pristine provenance",
        candidate_hash=pilot.hash_file(args.candidate_manifest), registry_hash=pilot.hash_file(args.registry))
    return settings


def source_lock(args):
    root = args.source_dir
    source_manifest = previous.read(root / "manifest.json")
    eligibility = previous.read(root / "eligibility.json")
    for filename in ("sources.pt", "eligibility.json"):
        if pilot.hash_file(root / filename) != source_manifest["files"][filename]:
            raise ValueError(f"source experiment input changed: {filename}")
    completed = previous.read(root / "complete.json")
    if completed["fingerprint"] != source_manifest["fingerprint"]:
        raise ValueError("source completion fingerprint differs")
    checkpoints = {}
    for arm in ("small", "large"):
        completion = previous.read(root / "runs" / f"{arm}_seed0" / "complete.json")
        if completion != completed["runs"][arm]:
            raise ValueError("source run completion changed")
        for kind in ("final", "selected"):
            path = root / "runs" / f"{arm}_seed0" / f"{kind}.pt"
            if pilot.hash_file(path) != completion[f"{kind}_sha256"]:
                raise ValueError("source checkpoint changed")
            checkpoints[f"{arm}_{kind}"] = dict(sha256=pilot.hash_file(path),
                step=completion["steps"] if kind == "final" else completion["selected_step"])
    return dict(fingerprint=source_manifest["fingerprint"], checkpoints=checkpoints,
        source_manifest_sha256=pilot.hash_file(root / "manifest.json"),
        original_allocation_sha256=source_manifest["original_allocation_sha256"],
        source_rows_sha256=pilot.hash_file(root / "sources.pt"),
        eligibility_sha256=pilot.hash_file(root / "eligibility.json"),
        training_groups=sorted(entry["source_group"] for entry in eligibility["accepted"] if entry["panel"] == "train"),
        validation_groups=sorted(entry["source_group"] for entry in eligibility["accepted"] if entry["panel"] == "real_validation"),
        backbone=previous.read(root / "backbone_lock.json"))


def exposure_for(entry, locked, registry):
    group = bank.normalized_name(entry["source_group"])
    for label, key in (("training_source", "training_groups"), ("validation_source", "validation_groups")):
        if group in {bank.normalized_name(value) for value in locked[key]}:
            return label
    names = {bank.normalized_name(entry["name"]), *(bank.normalized_name(value) for value in entry.get("aliases", []))}
    if group in registry["historical_groups"] or names & set(registry["historical_names"]):
        return "previously_declared_source"
    return "catalog_only_source"


def row_partitions(source, reservation, split_seed):
    """Use common queries and nested contexts; training-source queries stay held out."""
    labels = np.asarray(source["labels"], dtype=np.int64)
    random_generator = np.random.default_rng(SEED + split_seed)
    if reservation is not None:
        query_positions = reservation["probe_query_positions"].numpy()
        context_pool = reservation["fitting_positions"].numpy()
    else:
        if len(labels) < 768:
            raise ValueError("need at least512 context plus256 query rows")
        largest_context = 2048 if len(labels) >= 2304 else 512
        query_count = min(1024, len(labels) - largest_context)
        query_positions = context_run.stratified_positions(labels, query_count, random_generator, reserve=1)
        context_pool = np.setdiff1d(np.arange(len(labels)), query_positions)
    largest_context = 2048 if len(context_pool) >= 2048 else 512
    large_local = context_run.stratified_positions(labels[context_pool], largest_context, random_generator)
    largest_positions = context_pool[large_local]
    result = {str(largest_context): largest_positions, "query": query_positions}
    if largest_context == 2048:
        small_local = context_run.stratified_positions(labels[largest_positions], 512, random_generator)
        result["512"] = largest_positions[small_local]
    if np.intersect1d(context_pool, query_positions).size:
        raise ValueError("query/context row overlap")
    return result


def prepare(args):
    settings, locked = specification(args), source_lock(args)
    residual.diversity.verify_revision(settings)
    intent = dict(settings=settings, source_lock=locked)
    intent_path = args.output_dir / "intent.json"
    if intent_path.exists() and previous.read(intent_path) != intent:
        raise ValueError("preparation settings or source checkpoint changed")
    pilot.json_write(intent_path, intent)
    if (args.output_dir / "manifest.json").exists():
        checked_manifest(args)
        return
    catalog = previous.read(args.candidate_manifest)
    registry = previous.read(args.registry)
    reservations = torch.load(args.source_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    reservations = {entry["source_group"]: entry for entry in reservations if entry["panel"] == "train"}
    # Preserve the original representative for every previously allocated group.
    original = previous.read(args.original_bank_dir / "banks_manifest.json")
    if pilot.hash_file(args.original_bank_dir / "banks_manifest.json") != locked["original_allocation_sha256"]:
        raise ValueError("original source allocation changed")
    ordered = [entry["source_entry"] for panel in ("large", "validation") for entry in original["selection"][panel]]
    ordered.extend(catalog["candidates"])
    accepted, rejected, identities, used_groups, used_hashes = [], [], set(), set(), set()
    for entry in ordered:
        identity = bank._entry_identity(entry)
        group = bank.normalized_name(entry["source_group"])
        if identity in identities or group in used_groups:
            continue
        identities.add(identity)
        stem = hashlib.sha256(identity.encode()).hexdigest()[:20]
        episode_path = args.output_dir / "episodes" / f"{stem}.pt"
        cache_filename = f"{stem}.pt"
        try:
            candidates = [args.output_dir / "source_cache" / cache_filename,
                          *(directory / cache_filename for directory in args.source_cache_dirs)]
            raw_path = next((path for path in candidates if path.exists()), candidates[0])
            if raw_path.exists():
                raw = torch.load(raw_path, weights_only=True, map_location="cpu")
                if residual.bank.cached_entry_identity(raw["entry"]) != residual.bank.cached_entry_identity(entry):
                    raise ValueError("cached loading identity changed")
                frame, labels = bank.unpack_frame(raw["columns"]), np.asarray(raw["labels"], dtype=np.int64)
                description = raw["description"]
            else:
                frame, labels, raw_hash, description = residual.bank.load_candidate(entry, args.cache_dir)
                raw = dict(entry=entry, columns=bank.pack_frame(frame), labels=labels.tolist(),
                           raw_hash=raw_hash, description=description)
                pilot.atomic_save(raw_path, raw)
            residual.bank.validate_provenance(entry, description)
            _, counts = np.unique(labels, return_counts=True)
            if not 2 <= len(counts) <= 10 or counts.min() < 2 or len(labels) < 768:
                raise ValueError("need>=768 rows,2-10 classes and>=2 rows per class")
            fingerprints = bank.content_fingerprints(frame, labels)
            input_hash = fingerprints["inputs_sha256"]
            if input_hash in used_hashes:
                raise ValueError("duplicate accepted feature table")
            historical_group = registry["input_hash_groups"].get(input_hash)
            if historical_group is not None and bank.normalized_name(historical_group) != group:
                raise ValueError(f"duplicate of historical source group {historical_group}")
            raw.update(family=entry["name"], source_group=entry["source_group"])
            reservation = reservations.get(entry["source_group"])
            if reservation is not None and pilot.hash_file(raw_path) != reservation["cache_sha256"]:
                raise ValueError("training-source reservation requires identical full-table cache")
            episodes = []
            for split_seed in (0, 1):
                positions = row_partitions(raw, reservation, split_seed)
                for size in ("512", "2048"):
                    if size not in positions:
                        continue
                    episode = context_run.episode_from_rows(raw, positions[size], positions["query"], SEED + split_seed)
                    # Validate the actual native preprocessing interface before GPU allocation.
                    residual.objective.real._prepared_views(episode, 16)
                    episode.update(split=split_seed, context_rows=int(size))
                    episodes.append(episode)
            pilot.atomic_save(episode_path, dict(episodes=episodes))
            accepted.append(dict(family=entry["name"], source_group=entry["source_group"],
                exposure=exposure_for(entry, locked, registry), identity=identity, rows=len(labels),
                queries=episodes[0]["n_query"], contexts=sorted({episode["n_context"] for episode in episodes}),
                training_queries_permanently_held_out=reservation is not None,
                episode_file=episode_path.relative_to(args.output_dir).as_posix(),
                episode_sha256=pilot.hash_file(episode_path), inputs_sha256=input_hash,
                description_sha256=hashlib.sha256((description or "").encode()).hexdigest()))
            used_groups.add(group)
            used_hashes.add(input_hash)
            print(f"Accepted {len(accepted)} {entry['name']} ({accepted[-1]['exposure']})", flush=True)
        except Exception as error:
            rejected.append(dict(identity=identity, source_group=entry["source_group"], reason=f"{type(error).__name__}: {error}"))
            print(f"Excluded {entry['name']}: {error}", flush=True)
        pilot.json_write(args.output_dir / "availability.json", dict(accepted=accepted, rejected=rejected))
    if not accepted:
        raise ValueError("no eligible catalog datasets")
    manifest = dict(**intent, accepted=accepted, rejected=rejected,
        counts={label: sum(entry["exposure"] == label for entry in accepted) for label in EXPOSURES})
    manifest["fingerprint"] = previous.digest(manifest)
    pilot.json_write(args.output_dir / "manifest.json", manifest)
    print(f"Preparation complete: {manifest['counts']}", flush=True)


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != specification(args) or manifest["source_lock"] != source_lock(args):
        raise ValueError("locked experiment settings or models changed")
    if previous.digest({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("manifest fingerprint changed")
    residual.diversity.verify_revision(manifest["settings"])
    return manifest


def predict_methods(backbone, models, episode):
    """Reuse identical native views across checkpoints; no parameter updates."""
    context, views = residual.prepared_episode(episode, next(backbone.parameters()).device)
    with torch.no_grad(), pilot.frozen_inference(backbone):
        native_outputs = [residual.objective.view_logits(backbone, view, episode["n_classes"], None,
                          residual.InferenceConfig()) for view in views]
        result = {"native16": torch.stack(native_outputs).mean(0)}
        for name, model in models.items():
            if model is None:
                result[name] = result["native16"]
                continue
            numerical_values, _ = residual.numeric_views(model, backbone, context, views)
            outputs = [native if values is None else residual.objective.view_logits(
                backbone, view, episode["n_classes"], values, residual.InferenceConfig())
                for native, view, values in zip(native_outputs, views, numerical_values, strict=True)]
            result[name] = torch.stack(outputs).mean(0)
    if any(not torch.isfinite(logits).all() for logits in result.values()):
        raise FloatingPointError("nonfinite catalog prediction")
    return result


def aggregate(records):
    """Average two row splits within each source before dataset-level summaries."""
    comparisons = {}
    for size in (512, 2048):
        for exposure in (*EXPOSURES, "all_unseen_training_sources", "unseen_training_and_validation", "all_sources"):
            selected = [row for row in records if row["context_rows"] == size and (
                exposure == "all_sources" or row["exposure"] == exposure or (
                    exposure == "all_unseen_training_sources" and row["exposure"] != "training_source") or (
                    exposure == "unseen_training_and_validation" and row["exposure"] not in ("training_source", "validation_source")))]
            families = sorted({row["family"] for row in selected})
            if not families:
                continue
            for method in METHODS[1:]:
                rows = []
                for family in families:
                    pair = [row for row in selected if row["family"] == family]
                    if len(pair) != 2 or {row["split"] for row in pair} != {0, 1}:
                        raise ValueError("incomplete two-split coverage")
                    candidate = np.mean([row["scores"][method]["nll"] for row in pair])
                    native = np.mean([row["scores"]["native16"]["nll"] for row in pair])
                    rows.append(dict(candidate_nll=float(candidate), native_nll=float(native)))
                summary = context_run.metrics_summary(rows)
                candidate = np.asarray([row["candidate_nll"] for row in rows])
                native = np.asarray([row["native_nll"] for row in rows])
                bootstrap = np.random.default_rng(SEED).integers(len(rows), size=(20000, len(rows)))
                summary["mean_nll_reduction_ci95"] = np.percentile(
                    1 - candidate[bootstrap].mean(1) / native[bootstrap].mean(1), [2.5, 97.5]).tolist()
                summary["secondary_metric_changes"] = {
                    metric: float(np.mean([row["scores"][method][metric] - row["scores"]["native16"][metric]
                        for row in selected if row["scores"][method][metric] != ""]))
                    for metric in ("accuracy", "macro_f1", "binary_auc")
                    if any(row["scores"][method][metric] != "" for row in selected)}
                comparisons[f"{exposure}/{size}/{method}"] = summary
    return comparisons


def infer(args):
    manifest = checked_manifest(args)
    backbone, checkpoint_path, checkpoint_hash = pilot.load_frozen(args, torch.device(args.device))
    if dict(path=Path(checkpoint_path).name, sha256=checkpoint_hash) != manifest["source_lock"]["backbone"]:
        raise ValueError("frozen TabICL checkpoint changed")
    backbone.eval().requires_grad_(False)
    models = {}
    for name, lock in manifest["source_lock"]["checkpoints"].items():
        arm, kind = name.split("_")
        saved = torch.load(args.source_dir / "runs" / f"{arm}_seed0" / f"{kind}.pt", weights_only=True, map_location="cpu")
        if saved["step"] != lock["step"]:
            raise ValueError("checkpoint update differs")
        model = context_run.new_model().to(args.device)
        model.load_state_dict(saved["model"])
        # Step-zero large selection is exact native identity; verify before reusing logits.
        if lock["step"] == 0:
            initial = context_run.new_model().state_dict()
            if any(not torch.equal(value.cpu(), initial[key]) for key, value in saved["model"].items()):
                raise ValueError("step-zero checkpoint is not native initialization")
            models[name] = None
        else:
            models[name] = model.eval().requires_grad_(False)
    deadline = time.monotonic() + args.wall_minutes * 60
    records = []
    for entry in manifest["accepted"]:
        episode_path = args.output_dir / entry["episode_file"]
        if pilot.hash_file(episode_path) != entry["episode_sha256"]:
            raise ValueError("locked episode file changed")
        episodes = torch.load(episode_path, weights_only=True, map_location="cpu")["episodes"]
        for episode in episodes:
            result_path = args.output_dir / "predictions" / f"{episode_path.stem}_{episode['split']}_{episode['context_rows']}.json"
            if result_path.exists():
                result = previous.read(result_path)
                if result["fingerprint"] != manifest["fingerprint"]:
                    raise ValueError("cached result fingerprint changed")
            else:
                if time.monotonic() >= deadline:
                    pilot.json_write(args.output_dir / "progress.json", dict(fingerprint=manifest["fingerprint"], episodes=len(records), complete=False))
                    print("Clean wall-budget stop; resume skips completed predictions", flush=True)
                    return
                logits = predict_methods(backbone, models, episode)
                result = dict(fingerprint=manifest["fingerprint"], family=entry["family"], source_group=entry["source_group"],
                    exposure=entry["exposure"], split=episode["split"], context_rows=episode["context_rows"],
                    query_rows=episode["n_query"], query_sha256=context_run.row_digest(episode["query_indices"]),
                    context_sha256=context_run.row_digest(episode["context_indices"]),
                    scores={name: context_run.score(values, episode["y_query"].to(values.device)) for name, values in logits.items()})
                if any(not math.isfinite(value) for scores in result["scores"].values() for value in scores.values() if isinstance(value, (float, int))):
                    raise FloatingPointError("nonfinite catalog metric")
                pilot.json_write(result_path, result)
                print(f"Evaluated {entry['family']} split={episode['split']} context={episode['context_rows']} ({entry['exposure']})", flush=True)
            records.append(result)
    pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=manifest["fingerprint"],
        datasets=len(manifest["accepted"]), episodes=len(records), exposure_counts=manifest["counts"],
        comparisons=aggregate(records), checkpoints=manifest["source_lock"]["checkpoints"],
        note="source overlap labels mandatory; catalog-only freshness is conservative metadata evidence, not proof"))
    print(f"Complete: {len(manifest['accepted'])} datasets, {len(records)} episodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "infer"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--original-bank-dir", type=Path, required=True)
    parser.add_argument("--source-cache-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "docs/experiments/joint_preprocessing_catalog_history_20261017.json")
    parser.add_argument("--candidate-manifest", type=Path, default=residual.DEFAULT_CANDIDATES)
    parser.add_argument("--expected-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wall-minutes", type=float, default=450.)
    # Inherited dependency fingerprint builder requires the original optimizer fields;
    # these are metadata only: this runner has no optimizer or backward operation.
    parser.set_defaults(steps=2048, evaluate_every=256, save_every=25, lr=.0003, continuation_seed=0)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
