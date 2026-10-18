"""Shared zero-shot spline learning with large contexts and sampled training queries."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts import joint_preprocessing_context_size as prior
from tabicl._hyperspline.joint_preprocessing import JointParameters

residual, pilot, previous, bank = prior.residual, prior.pilot, prior.previous, prior.bank
SEED = 20261018
CONTEXT_FRACTION = .8
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = dict(train=83, validation=10, evaluation=58)
SOURCE_FINGERPRINT = "f28b919a88dac747f4a4c998edf9ce996cd374bf0c8b307ff01af6fd4ea2c427"
CATALOG_FINGERPRINT = "4aae70bd8544a962a8c83448b5b00ea6779632b575a2bb9d027c7d0c781a8405"
CONTINUATION_FINGERPRINT = "8e406f6e46e0550b01c089b846753a3570f40dbabb467d28a48d34a82760cf10"
CONTINUATION_STATE_SHA256 = "930fa1981520f689e665242ac7fc278bdf223536ad7d985bbf2c92af16c21b1e"
BOUNDED_PARENT_FINGERPRINT = "facd59803d5ae333b3c712d498db20cc41f45d2a175c3061d5115a7dae9cff0b"
BOUNDED_PARENT_STATE_SHA256 = "807e0643a7534b88e722ad191dff842f2e486cc09f982f1297405dbb2ce11d17"


def split_positions(labels, available_positions, seed, context_row_cap=None):
    """Keep all20% queries; optionally subsample only the nominal80% context."""
    available_positions = np.asarray(available_positions, dtype=np.int64)
    if len(np.unique(available_positions)) != len(available_positions):
        raise ValueError("available rows must be unique")
    query_count = len(available_positions) - math.floor(CONTEXT_FRACTION * len(available_positions))
    random_generator = np.random.default_rng(seed)
    query_local_positions = prior.stratified_positions(
        np.asarray(labels)[available_positions], query_count, random_generator, reserve=1)
    query_positions = available_positions[query_local_positions]
    context_positions = random_generator.permutation(np.setdiff1d(available_positions, query_positions))
    if context_row_cap is not None and len(context_positions) > context_row_cap:
        context_local_positions = prior.stratified_positions(
            np.asarray(labels)[context_positions], context_row_cap, random_generator)
        context_positions = context_positions[context_local_positions]
    return dict(context=context_positions, query=query_positions)


def episode_for(entry, source_cache, *, step=None, episode_position=0, split=0, probe=False,
                context_row_cap=None, training_query_row_cap=None):
    source = source_cache.get(entry["family"])
    if probe:
        query_positions = entry["probe_query_positions"].numpy()
        random_generator = np.random.default_rng(SEED + split)
        fitting_positions = entry["fitting_positions"].numpy()
        context_count = math.floor(CONTEXT_FRACTION * len(fitting_positions))
        if context_row_cap is not None:
            context_count = min(context_count, context_row_cap)
        context_local_positions = prior.stratified_positions(
            np.asarray(source["labels"])[fitting_positions], context_count, random_generator)
        positions = dict(context=fitting_positions[context_local_positions], query=query_positions)
        seed = SEED + split
    else:
        seed = int(np.random.SeedSequence([SEED, step or 0, episode_position, split]).generate_state(1)[0])
        available_positions = entry["fitting_positions"].numpy() if entry["panel"] == "train" else np.arange(len(source["labels"]))
        positions = split_positions(source["labels"], available_positions, seed, context_row_cap)
    query_pool_rows = len(positions["query"])
    # Only training visits sample supervision; evaluation retains the complete query pool.
    if step is not None and entry["panel"] == "train" and not probe and training_query_row_cap is not None:
        if training_query_row_cap < 1:
            raise ValueError("training query cap must be positive")
        if query_pool_rows > training_query_row_cap:
            query_generator = np.random.default_rng(np.random.SeedSequence([seed, 731]))
            sampled_query_positions = query_generator.choice(query_pool_rows, training_query_row_cap, replace=False)
            positions["query"] = positions["query"][sampled_query_positions]
    episode = prior.episode_from_rows(source, positions["context"], positions["query"], seed)
    episode.update(source_seed=seed, split=split, query_pool_rows=query_pool_rows)
    return episode


class PreparedEpisode:
    """Fit native preprocessing once; keep all queries on CPU until their chunk."""

    def __init__(self, episode):
        self.episode = episode
        self.context, self.views = residual.prepared_episode(episode, torch.device("cpu"))

    def chunk(self, query_start, query_stop, device):
        context_count = self.episode["n_context"]
        context = {name: value.to(device) if isinstance(value, torch.Tensor) else value
                   for name, value in self.context.items()}
        context["all_missing"] = torch.cat((self.context["all_missing"][:, :context_count],
            self.context["all_missing"][:, context_count + query_start:context_count + query_stop]), 1).to(device)
        views = []
        for view in self.views:
            chunk_view = {name: value.to(device) if isinstance(value, torch.Tensor) and name != "x" else value
                          for name, value in view.items()}
            chunk_view["x"] = torch.cat((view["x"][:, :context_count],
                view["x"][:, context_count + query_start:context_count + query_stop]), 1).to(device)
            views.append(chunk_view)
        return context, views


class GeneratedMap:
    """Generate one map, then accumulate exact chunk gradients into its generator."""

    def __init__(self, backbone, model, context, n_classes):
        self.model = model
        self.n_classes = n_classes
        self.original = residual.generate(model, backbone, context)
        detached = {}
        self.targets = []
        self.leaves = []
        for field in dataclasses.fields(self.original):
            value = getattr(self.original, field.name)
            detached[field.name] = value.detach() if isinstance(value, torch.Tensor) else value
            if isinstance(value, torch.Tensor) and value.requires_grad:
                detached[field.name] = value.detach().requires_grad_(True)
                self.targets.append(value)
                self.leaves.append(detached[field.name])
        self.parameters = JointParameters(**detached)

    def values(self, view, context):
        if not view["adapted"]:
            return None
        return self.model.apply_native(view["x"].index_select(-1, view["columns"]),
            self.parameters, view["slot"], context["all_missing"])

    def backward_chunk(self, backbone, views, context, query_labels, *, weight, measure_native=False):
        # Replay one adapted view at a time to bound memory at the complete context size.
        with torch.no_grad():
            outputs = [residual.objective.view_logits(backbone, view,
                self.n_classes, self.values(view, context)) for view in views]
            logits = torch.stack(outputs).mean(0)
            self.last_native_nll = None
            if measure_native:
                native_outputs = [output if not view["adapted"] else residual.objective.view_logits(
                    backbone, view, self.n_classes) for output, view in zip(outputs, views, strict=True)]
                native_logits = torch.stack(native_outputs).mean(0)
                self.last_native_nll = float(F.cross_entropy((native_logits / .9).flatten(0, 1), query_labels.to(logits.device)))
        logit_leaf = logits.detach().requires_grad_(True)
        loss = F.cross_entropy((logit_leaf / .9).flatten(0, 1), query_labels.to(logit_leaf.device))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite all-query objective")
        upstream, = torch.autograd.grad(loss, logit_leaf)
        for view in views:
            if view["adapted"]:
                output = residual.objective.view_logits(backbone, view, self.n_classes, self.values(view, context))
                torch.autograd.backward(output, upstream * (weight / 16.))
        # Propagating each chunk frees its backbone graph; the small generator graph is reused.
        gradients = [leaf.grad for leaf in self.leaves]
        if any(value is None or not torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("missing/nonfinite generated-map gradient")
        torch.autograd.backward(self.targets, gradients, retain_graph=True)
        for leaf in self.leaves:
            leaf.grad = None
        return float(loss.detach())


def configure_backbone(backbone):
    backbone.train().requires_grad_(False)
    for module in backbone.modules():
        if hasattr(module, "recompute"):
            module.recompute = True
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.


def specification(args):
    if min(args.steps, args.evaluate_every, args.save_every, args.query_chunk_rows) < 1:
        raise ValueError("budgets and chunk sizes must be positive")
    settings = residual.settings(args)
    for path in (Path(__file__), Path(prior.__file__), ROOT / "scripts/run_joint_preprocessing_fraction_context.sh",
                 ROOT / "src/tabicl/_hyperspline/statistics_preprocessing.py"):
        settings["code_hashes"][path.relative_to(ROOT).as_posix()] = residual.objective.canonical_hash(path)
    context_row_cap = getattr(args, "context_row_cap", None)
    training_query_row_cap = getattr(args, "training_query_row_cap", None)
    if context_row_cap is not None and context_row_cap < 10:
        raise ValueError("context cap must retain support for all supported classes")
    if training_query_row_cap is not None and not 1 <= training_query_row_cap <= args.query_chunk_rows:
        raise ValueError("sampled training queries must fit one positive query chunk")
    settings.update(arms=["fraction"], alpha=None, seed=SEED, context_fraction=.8, query_fraction=.2,
        context_row_cap=context_row_cap, query_row_cap=training_query_row_cap, query_chunk_rows=args.query_chunk_rows,
        reuse_prepared_dir=str(getattr(args, "reuse_prepared_dir", None)),
        initialization="fresh seed0 restricted statistics hypernetwork; exact native16 identity",
        model_parameters=5912, training_precision="FP32; per-view replay; encoder activation recomputation",
        evaluation_precision="same FP32 forward as training; complete context retained in every query chunk",
        validation_splits=3, evaluation_splits=2, panels=["source_probe", "validation", "evaluation"],
        validation_selection="one shared checkpoint, minimum equal-dataset NLL averaged over3 fixed partitions",
        objective="all query rows, row-weighted chunks, four equal-weight source episodes per update",
        confirmation="58 historically inspected sources outside current training/validation; not pristine test",
        source_dir=str(args.source_dir), catalog_dir=str(args.catalog_dir),
        evaluation_timing="final and selected only after training locks; no target fitting",
        hypothesis="substantial context and query fractions improve zero-shot preprocessing",
        runtime_gate="one maximum-cell-count capped gradient check inside training job" if context_row_cap else
            "resource profile must pass at actual uncapped representative sizes before training")
    if training_query_row_cap is not None:
        settings["objective"] = "uniform sampled query mean, one chunk, four equal-weight source episodes per update"
    if getattr(args, "continue_from_dir", None) is not None:
        if training_query_row_cap is None or args.reuse_prepared_dir is None:
            raise ValueError("continuation requires sampled queries and the locked preparation")
        settings.update(continue_from_dir=str(args.continue_from_dir),
            initialization="completed step46 model and AdamW state; discard unfinished step47 gradients",
            continuation_validation="validate imported checkpoint before any update; then every32 absolute updates")
        if getattr(args, "bounded_continuation", False):
            if args.steps != 320 or args.evaluate_every != 320:
                raise ValueError("bounded continuation must stop and validate at320 only")
            settings.update(bounded_continuation=True,
                initialization="completed step142 model, AdamW moments and historical selection; exact update-boundary resume",
                continuation_validation="no initial sweep; full validation at320 only",
                evaluation_timing="stop after step320 validation; catalog inference deferred pending decision")
    elif getattr(args, "bounded_continuation", False):
        raise ValueError("bounded continuation requires a parent experiment")
    return settings


def verified_json(root, filename):
    manifest = previous.read(root / "manifest.json")
    if previous.digest({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("source manifest fingerprint changed")
    if pilot.hash_file(root / filename) != manifest["files"][filename]:
        raise ValueError(f"source input changed: {filename}")
    return manifest


def prepare(args):
    settings = specification(args)
    residual.diversity.verify_revision(settings)
    if (args.output_dir / "manifest.json").exists():
        checked_manifest(args)
        return
    if getattr(args, "reuse_prepared_dir", None) is not None:
        return prepare_from_locked(args, settings)
    source_manifest = verified_json(args.source_dir, "sources.pt")
    if source_manifest["fingerprint"] != SOURCE_FINGERPRINT:
        raise ValueError("unexpected original context bank")
    entries = torch.load(args.source_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    for entry in entries:
        entry["panel"] = "train" if entry["panel"] == "train" else "validation"
    catalog = previous.read(args.catalog_dir / "manifest.json")
    if previous.digest({key: value for key, value in catalog.items() if key != "fingerprint"}) != catalog["fingerprint"]:
        raise ValueError("catalog fingerprint changed")
    if catalog["fingerprint"] != CATALOG_FINGERPRINT:
        raise ValueError("unexpected catalog source allocation")
    trained_groups = {entry["source_group"] for entry in entries}
    catalog_entries = [entry for entry in catalog["accepted"] if entry["source_group"] not in trained_groups]
    for entry in catalog_entries:
        # Replay the catalog's cache search; its manifest identifies raw caches by episode stem.
        cache_filename = Path(entry["episode_file"]).stem + ".pt"
        cache_directories = [args.catalog_dir / "source_cache", *map(Path, source_manifest["settings"]["source_cache_dirs"])]
        raw_path = next((directory / cache_filename for directory in cache_directories
                         if (directory / cache_filename).exists()), None)
        if raw_path is None:
            raise FileNotFoundError(f"full catalog cache missing: {cache_filename}")
        raw = torch.load(raw_path, weights_only=True, map_location="cpu")
        fingerprints = bank.content_fingerprints(bank.unpack_frame(raw["columns"]), np.asarray(raw["labels"]))
        if fingerprints["inputs_sha256"] != entry["inputs_sha256"]:
            raise ValueError("catalog full input content changed")
        entries.append(dict(family=entry["family"], source_group=entry["source_group"], panel="evaluation",
            cache_path=str(raw_path), cache_sha256=pilot.hash_file(raw_path), full_rows=entry["rows"],
            exposure=entry["exposure"]))
        if entry["family"] == "KDDCup09_upselling":
            entries[-1]["historical_relation"] = "Earlier research inspected KDD2009 appetency; not a pristine catalog-only source."
    counts = {panel: sum(entry["panel"] == panel for entry in entries) for panel in ("train", "validation", "evaluation")}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected frozen allocation: {counts}")
    if len({entry["source_group"] for entry in entries}) != len(entries):
        raise ValueError("source groups overlap")
    if len({entry["family"] for entry in entries}) != len(entries):
        raise ValueError("ambiguous dataset names in source cache")
    # Lock the full raw cache bytes once before any GPU resource is requested.
    cache = prior.SourceCache(entries, capacity=1)
    for entry in entries:
        raw = cache.get(entry["family"])
        labels = np.asarray(raw["labels"])
        available_positions = entry["fitting_positions"].numpy() if entry["panel"] == "train" else np.arange(len(labels))
        positions = split_positions(labels, available_positions, SEED)
        entry["context_rows"] = len(positions["context"])
        entry["query_rows"] = len(positions["query"])
        entry["input_columns"] = len(raw["columns"])
        print(f"Locked {entry['panel']} {entry['family']}: context={entry['context_rows']} query={entry['query_rows']}", flush=True)
    pilot.atomic_save(args.output_dir / "sources.pt", dict(entries=entries))
    initial = prior.new_model()
    pilot.atomic_save(args.output_dir / "initial.pt", dict(model=pilot.state_cpu(initial)))
    pilot.json_write(args.output_dir / "allocation.json", dict(counts=counts, entries=[
        {name: value for name, value in entry.items() if not isinstance(value, torch.Tensor)} for entry in entries]))
    manifest = dict(settings=settings, counts=counts, source_fingerprint=source_manifest["fingerprint"],
        catalog_fingerprint=catalog["fingerprint"],
        backbone=previous.read(args.source_dir / "backbone_lock.json"),
        previous_checkpoints=catalog["source_lock"]["checkpoints"],
        files={filename: pilot.hash_file(args.output_dir / filename)
               for filename in ("sources.pt", "initial.pt", "allocation.json")})
    manifest["fingerprint"] = previous.digest(manifest)
    pilot.json_write(args.output_dir / "manifest.json", manifest)


def prepare_from_locked(args, settings):
    """Reuse audited full-table membership/reservations; change only episode context limits."""
    source_manifest = verified_json(args.reuse_prepared_dir, "sources.pt")
    if source_manifest["fingerprint"] != "ea1185039ddb00a6f2aad0b5b79147802da34a03a8cdc82e13b86f0e91fe130e":
        raise ValueError("wrong uncapped preparation to reuse")
    if source_manifest["source_fingerprint"] != SOURCE_FINGERPRINT or source_manifest["catalog_fingerprint"] != CATALOG_FINGERPRINT:
        raise ValueError("source membership changed")
    entries = torch.load(args.reuse_prepared_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    if source_manifest["counts"] != EXPECTED_COUNTS:
        raise ValueError("source counts changed")
    for entry in entries:
        entry["nominal_context_rows"] = entry["context_rows"]
        entry["context_rows"] = min(entry["context_rows"], args.context_row_cap)
        entry["unused_context_rows"] = entry["nominal_context_rows"] - entry["context_rows"]
        # Allocation distinguishes the full query pool from supervision consumed per visit.
        if getattr(args, "training_query_row_cap", None) is not None:
            entry["training_query_rows"] = min(entry["query_rows"], args.training_query_row_cap) if entry["panel"] == "train" else None
    pilot.atomic_save(args.output_dir / "sources.pt", dict(entries=entries))
    pilot.atomic_save(args.output_dir / "initial.pt", dict(model=pilot.state_cpu(prior.new_model())))
    pilot.json_write(args.output_dir / "allocation.json", dict(counts=source_manifest["counts"], entries=[
        {name: value for name, value in entry.items() if not isinstance(value, torch.Tensor)} for entry in entries]))
    manifest = dict(settings=settings, counts=source_manifest["counts"], source_fingerprint=SOURCE_FINGERPRINT,
        catalog_fingerprint=CATALOG_FINGERPRINT, reused_preparation_fingerprint=source_manifest["fingerprint"],
        backbone=source_manifest["backbone"], previous_checkpoints=source_manifest["previous_checkpoints"],
        files={filename: pilot.hash_file(args.output_dir / filename) for filename in ("sources.pt", "initial.pt", "allocation.json")})
    if getattr(args, "continue_from_dir", None) is not None:
        manifest["continuation"] = lock_continuation(args, settings)
    manifest["fingerprint"] = previous.digest(manifest)
    pilot.json_write(args.output_dir / "manifest.json", manifest)
    print(f"Prepared capped experiment: counts={manifest['counts']}, context_cap={args.context_row_cap}, "
          f"training_query_cap={getattr(args, 'training_query_row_cap', None)}, "
          f"evaluation retains all20% queries, fingerprint={manifest['fingerprint']}", flush=True)


def lock_continuation(args, settings):
    """Lock the audited parent and permit only the requested continuation changes."""
    bounded_continuation = getattr(args, "bounded_continuation", False)
    parent_fingerprint = BOUNDED_PARENT_FINGERPRINT if bounded_continuation else CONTINUATION_FINGERPRINT
    expected_state_sha256 = BOUNDED_PARENT_STATE_SHA256 if bounded_continuation else CONTINUATION_STATE_SHA256
    completed_updates = 142 if bounded_continuation else 46
    parent_manifest = verified_json(args.continue_from_dir, "sources.pt")
    if parent_manifest["fingerprint"] != parent_fingerprint:
        raise ValueError("unexpected continuation experiment")
    state_path = args.continue_from_dir / "state.pt"
    if pilot.hash_file(state_path) != expected_state_sha256:
        raise ValueError("continuation checkpoint differs from downloaded audited state")
    # Optimizer, backbone, source membership and transformation implementation must match.
    allowed_settings = {"code_hashes", "expected_revision", "query_row_cap", "evaluate_every", "objective",
                        "initialization", "continue_from_dir", "continuation_validation"}
    if bounded_continuation:
        # Training policy is identical; only the stopping/evaluation schedule and provenance change.
        allowed_settings -= {"query_row_cap", "objective"}
        allowed_settings |= {"steps", "bounded_continuation", "evaluation_timing"}
    parent_settings = parent_manifest["settings"]
    for name in set(parent_settings) | set(settings):
        if name not in allowed_settings and parent_settings.get(name) != settings.get(name):
            raise ValueError(f"unauthorized continuation setting change: {name}")
    allowed_code = {"scripts/joint_preprocessing_fraction_context.py", "scripts/run_joint_preprocessing_fraction_context.sh"}
    for filename in set(parent_settings["code_hashes"]) | set(settings["code_hashes"]):
        if filename not in allowed_code and parent_settings["code_hashes"].get(filename) != settings["code_hashes"].get(filename):
            raise ValueError(f"continuation model/dependency changed: {filename}")
    parent_state = torch.load(state_path, weights_only=True, map_location="cpu")
    if parent_state["fingerprint"] != parent_fingerprint or parent_state["step"] != completed_updates:
        raise ValueError(f"expected completed step{completed_updates} checkpoint")
    expected_cursor = (0, 0) if bounded_continuation else (1, 0)
    if (parent_state["episode_position"], parent_state["query_offset"]) != expected_cursor:
        raise ValueError("unexpected continuation episode/query cursor")
    if bounded_continuation and (parent_state["pending_validation"] or
            any(gradient is not None for gradient in parent_state["gradients"].values())):
        raise ValueError("bounded continuation requires a clean update boundary")
    profile_path = args.continue_from_dir / "profile/complete.json"
    profile_result = previous.read(profile_path)
    if profile_result["fingerprint"] != parent_fingerprint or not profile_result["passed"]:
        raise ValueError("parent profile did not pass")
    return dict(parent_dir=str(args.continue_from_dir), parent_fingerprint=parent_fingerprint,
        state_sha256=expected_state_sha256, completed_updates=completed_updates,
        parent_profile_sha256=pilot.hash_file(profile_path),
        validate_imported_checkpoint=not bounded_continuation,
        discarded_partial_update=None if bounded_continuation else dict(step=47, completed_episodes=1),
        coverage="historical exposure retained, including discarded partial update; new chunks logged separately")


def imported_state(args, manifest):
    """Copy completed weights/moments without applying the unfinished old-policy gradients."""
    locked = manifest["continuation"]
    state_path = Path(locked["parent_dir"]) / "state.pt"
    if pilot.hash_file(state_path) != locked["state_sha256"]:
        raise ValueError("locked continuation checkpoint changed")
    state = torch.load(state_path, weights_only=True, map_location="cpu")
    if state["fingerprint"] != locked["parent_fingerprint"] or state["step"] != locked["completed_updates"]:
        raise ValueError("locked continuation cursor changed")
    state.update(fingerprint=manifest["fingerprint"], episode_position=0, query_offset=0,
        episode_loss_sum=0., update_loss_sum=0., record=0,
        pending_validation=locked.get("validate_imported_checkpoint", True),
        gradients={name: None for name in state["gradients"]}, continuation=locked)
    return state


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != specification(args) or previous.digest({key: value for key, value in manifest.items()
            if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("experiment/code/settings changed")
    for filename, expected in manifest["files"].items():
        if pilot.hash_file(args.output_dir / filename) != expected:
            raise ValueError(f"locked experiment input changed: {filename}")
    residual.diversity.verify_revision(manifest["settings"])
    return manifest


def setup(args):
    manifest = checked_manifest(args)
    backbone, checkpoint_path, checkpoint_hash = pilot.load_frozen(args, torch.device(args.device))
    if dict(path=Path(checkpoint_path).name, sha256=checkpoint_hash) != manifest["backbone"]:
        raise ValueError("backbone checkpoint changed")
    configure_backbone(backbone)
    model = prior.new_model().to(args.device)
    model.load_state_dict(torch.load(args.output_dir / "initial.pt", weights_only=True, map_location="cpu")["model"])
    entries = torch.load(args.output_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    return backbone, model, entries, manifest


def append_jsonl(path, record):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def profile(args):
    backbone, model, entries, manifest = setup(args)
    if (args.output_dir / "profile/complete.json").exists():
        if previous.read(args.output_dir / "profile/complete.json")["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("profile fingerprint changed")
        return
    if "continuation" in manifest:
        # The context and execution chunk sizes are unchanged; reuse the passed GPU shape check.
        locked = manifest["continuation"]
        parent_profile_path = Path(locked["parent_dir"]) / "profile/complete.json"
        if pilot.hash_file(parent_profile_path) != locked["parent_profile_sha256"]:
            raise ValueError("locked parent GPU profile changed")
        inherited_profile = previous.read(parent_profile_path)
        pilot.json_write(args.output_dir / "profile/complete.json", dict(inherited_profile,
            fingerprint=manifest["fingerprint"], reused_parent_fingerprint=locked["parent_fingerprint"]))
        next_action = "training next update" if not locked.get("validate_imported_checkpoint", True) else "validating imported checkpoint"
        print(f"Reusing passed 20000-context/2048-query GPU check; {next_action}", flush=True)
        return
    candidates = [entry for entry in entries if entry["panel"] != "evaluation"]
    # Include the longest context, median training table and largest cell count.
    cache = prior.SourceCache(candidates, capacity=1)
    maximum_cells = max(candidates, key=lambda entry: entry["context_rows"] * entry["input_columns"])
    training_candidates = sorted([entry for entry in candidates if entry["panel"] == "train"], key=lambda entry: entry["context_rows"])
    median_training = training_candidates[len(training_candidates) // 2]
    selected = {entry["family"]: entry for entry in (
        median_training, max(candidates, key=lambda entry: entry["context_rows"]), maximum_cells)}
    if getattr(args, "context_row_cap", None) is not None:
        selected = {maximum_cells["family"]: maximum_cells}
    reports = []
    for entry in selected.values():
        report_path = args.output_dir / "profile" / f"{entry['family'].replace('/', '_')}.json"
        if report_path.exists():
            saved = previous.read(report_path)
            if saved["fingerprint"] != manifest["fingerprint"] or not saved["passed"]:
                raise ValueError("profile incomplete/failed")
            reports.append(saved)
            continue
        started = time.monotonic()
        episode = episode_for(entry, cache, step=1, context_row_cap=getattr(args, "context_row_cap", None))
        print(f"STARTUP CHECK {entry['family']}: context={episode['n_context']} queries={episode['n_query']}", flush=True)
        prepared = PreparedEpisode(episode)
        preparation_seconds = time.monotonic() - started
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        chunk_start = time.monotonic()
        context, views = prepared.chunk(0, min(args.query_chunk_rows, episode["n_query"]), args.device)
        probe_model = copy.deepcopy(model)
        # Perturb heads so the profile also exercises nonidentity generator gradients.
        with torch.no_grad():
            probe_model.affine_head.weight.fill_(.0001)
            probe_model.spline_head.weight.fill_(.0001)
        generated = GeneratedMap(backbone, probe_model, context, episode["n_classes"])
        loss = generated.backward_chunk(backbone, views, context, episode["y_query"][:args.query_chunk_rows], weight=1.)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(probe_model.parameters(), float("inf")))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise FloatingPointError("profile did not propagate useful gradients")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        chunk_seconds = time.monotonic() - chunk_start
        report = dict(fingerprint=manifest["fingerprint"], family=entry["family"], passed=True,
            context_rows=episode["n_context"], query_rows=episode["n_query"],
            query_chunk_rows=min(args.query_chunk_rows, episode["n_query"]),
            preparation_seconds=preparation_seconds, chunk_seconds=chunk_seconds,
            projected_episode_seconds=preparation_seconds + chunk_seconds * math.ceil(episode["n_query"] / args.query_chunk_rows),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0,
            gpu_name=torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU fixture",
            gpu_total_memory=torch.cuda.get_device_properties(0).total_memory if args.device.startswith("cuda") else 0,
            loss=loss, gradient_norm=gradient_norm,
            gradient_groups=residual.core.diagnostic.gradient_norms(probe_model), no_optimizer_update=True)
        pilot.json_write(report_path, report)
        reports.append(report)
        print(f"PROFILE {report}", flush=True)
        del generated, probe_model, views, context, prepared, episode
        gc.collect()
    pilot.json_write(args.output_dir / "profile/complete.json", dict(fingerprint=manifest["fingerprint"], passed=True, reports=reports))


def prediction_record(args, backbone, models, episode, folder, fingerprint, deadline):
    """Persist query progress so even validation can span an allocation boundary."""
    completed_path, progress_path = folder / "complete.json", folder / "progress.pt"
    if completed_path.exists():
        result = previous.read(completed_path)
        if result["fingerprint"] != fingerprint:
            raise ValueError("prediction record fingerprint changed")
        return result
    predictions = {name: [] for name in ("native16", *models)}
    query_offset = 0
    if progress_path.exists():
        progress = torch.load(progress_path, weights_only=True, map_location="cpu")
        if progress["fingerprint"] != fingerprint or progress["context_hash"] != prior.row_digest(episode["context_indices"]):
            raise ValueError("prediction progress changed")
        predictions, query_offset = progress["predictions"], progress["query_offset"]
    prepared = PreparedEpisode(episode)
    maps = None
    while query_offset < episode["n_query"]:
        if time.monotonic() >= deadline:
            return None
        query_stop = min(query_offset + args.query_chunk_rows, episode["n_query"])
        with torch.no_grad():
            context, views = prepared.chunk(query_offset, query_stop, args.device)
            if maps is None:
                maps = {name: GeneratedMap(backbone, model, context, episode["n_classes"])
                        for name, model in models.items() if model is not None}
            native_outputs = [residual.objective.view_logits(backbone, view, episode["n_classes"]) for view in views]
            native_logits = torch.stack(native_outputs).mean(0)
            predictions["native16"].append(native_logits.cpu())
            for name, model in models.items():
                outputs = native_outputs if model is None else [native if not view["adapted"] else
                    residual.objective.view_logits(backbone, view, episode["n_classes"], maps[name].values(view, context))
                    for native, view in zip(native_outputs, views, strict=True)]
                logits = native_logits if model is None else torch.stack(outputs).mean(0)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite prediction")
                predictions[name].append(logits.cpu())
        query_offset = query_stop
        pilot.atomic_save(progress_path, dict(fingerprint=fingerprint, context_hash=prior.row_digest(episode["context_indices"]),
            query_offset=query_offset, predictions=predictions))
        print(f"EVAL {episode['family']} split={episode['split']} queries={query_offset}/{episode['n_query']}", flush=True)
        del context, views, native_outputs
    scores = {name: prior.score(torch.cat(chunks, dim=1), episode["y_query"])
              for name, chunks in predictions.items()}
    if any(not math.isfinite(value) for metrics in scores.values() for value in metrics.values()
           if isinstance(value, (float, int))):
        raise FloatingPointError("nonfinite full-query score")
    result = dict(fingerprint=fingerprint, family=episode["family"], split=episode["split"],
        context_rows=episode["n_context"], query_rows=episode["n_query"],
        context_hash=prior.row_digest(episode["context_indices"]), query_hash=prior.row_digest(episode["query_indices"]), scores=scores)
    pilot.json_write(completed_path, result)
    return result


def aggregate_records(records, method):
    per_dataset = []
    for family in sorted({record["family"] for record in records}):
        partitions = [record for record in records if record["family"] == family]
        per_dataset.append(dict(family=family,
            candidate_nll=float(np.mean([record["scores"][method]["nll"] for record in partitions])),
            native_nll=float(np.mean([record["scores"]["native16"]["nll"] for record in partitions]))))
    summary = prior.metrics_summary(per_dataset)
    candidate = np.asarray([row["candidate_nll"] for row in per_dataset])
    native = np.asarray([row["native_nll"] for row in per_dataset])
    bootstrap = np.random.default_rng(SEED).integers(len(per_dataset), size=(20000, len(per_dataset)))
    summary["mean_nll_reduction_ci95"] = np.percentile(
        1 - candidate[bootstrap].mean(1) / native[bootstrap].mean(1), [2.5, 97.5]).tolist()
    secondary = {}
    for metric in ("accuracy", "macro_f1", "binary_auc"):
        changes = []
        for family in sorted({record["family"] for record in records}):
            differences = [record["scores"][method][metric] - record["scores"]["native16"][metric]
                for record in records if record["family"] == family and
                record["scores"][method].get(metric, "") != ""]
            if differences:
                changes.append(float(np.mean(differences)))
        if changes:
            secondary[metric] = dict(mean_change_percentage_points=100 * float(np.mean(changes)), datasets=len(changes))
    summary["secondary_metrics"] = secondary
    summary["per_dataset"] = per_dataset
    return summary


def validation(args, backbone, model, entries, manifest, step, deadline):
    records = []
    cache = prior.SourceCache(entries, capacity=1)
    for source_position, entry in enumerate(entry for entry in entries if entry["panel"] == "validation"):
        for split in range(3):
            episode = episode_for(entry, cache, split=split, context_row_cap=getattr(args, "context_row_cap", None))
            fingerprint = previous.digest(dict(experiment=manifest["fingerprint"], step=step,
                source=entry["source_group"], split=split, panel="validation"))
            record = prediction_record(args, backbone, dict(candidate=model), episode,
                args.output_dir / "validation" / f"step{step}" / f"source{source_position}_split{split}", fingerprint, deadline)
            if record is None:
                return None
            records.append(record)
    summary = aggregate_records(records, "candidate")
    pilot.json_write(args.output_dir / "validation" / f"step{step}" / "complete.json", summary)
    print(f"VALIDATION step={step} gain={100*summary['mean_nll_reduction']:.4f}% "
          f"W/L/T={summary['wins']}/{summary['losses']}/{summary['ties']}", flush=True)
    return summary


def parameter_gradients(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()}


def train(args):
    deadline = time.monotonic() + args.wall_minutes * 60
    backbone, model, entries, manifest = setup(args)
    profile_result = previous.read(args.output_dir / "profile/complete.json")
    if profile_result["fingerprint"] != manifest["fingerprint"] or not profile_result["passed"]:
        raise ValueError("actual-size GPU profile must pass before training")
    if (args.output_dir / "training_complete.json").exists():
        return True
    train_entries = [entry for entry in entries if entry["panel"] == "train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    state_path = args.output_dir / "state.pt"
    if not state_path.exists() and "continuation" in manifest:
        pilot.atomic_save(state_path, imported_state(args, manifest))
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("training state exists; pass --resume")
        state = torch.load(state_path, weights_only=True, map_location="cpu")
        if state["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("training resume fingerprint changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        residual.core.capacity.optimizer_to(optimizer, torch.device(args.device))
        for name, parameter in model.named_parameters():
            saved_gradient = state["gradients"][name]
            parameter.grad = None if saved_gradient is None else saved_gradient.to(args.device)
        # Discard only log records newer than the durable model/gradient cursor.
        log_path = args.output_dir / "chunks.jsonl"
        if log_path.exists():
            import json
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines()
                     if json.loads(line)["record"] <= state["record"]]
            log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        state = dict(fingerprint=manifest["fingerprint"], step=0, episode_position=0, query_offset=0,
            episode_loss_sum=0., update_loss_sum=0., record=0, pending_validation=False,
            best_score=float("inf"), best_step=0, best_model=pilot.state_cpu(model),
            coverage={entry["family"]: {role: torch.zeros(entry["full_rows"], dtype=torch.bool)
                      for role in ("context", "query")} for entry in train_entries})

    def save():
        state.update(model=pilot.state_cpu(model), optimizer=optimizer.state_dict(), gradients=parameter_gradients(model))
        pilot.atomic_save(state_path, state)

    cache = prior.SourceCache(train_entries, capacity=1)
    save()
    while True:
        if state["pending_validation"]:
            result = validation(args, backbone, model, entries, manifest, state["step"], deadline)
            if result is None:
                save()
                return False
            # Exact identity is an eligible checkpoint without delaying training for its full evaluation.
            if math.isinf(state["best_score"]):
                state["best_score"] = result["native_mean_nll"]
            if result["mean_nll"] < state["best_score"]:
                state.update(best_score=result["mean_nll"], best_step=state["step"], best_model=pilot.state_cpu(model))
            state["pending_validation"] = False
            save()
        if state["step"] >= args.steps:
            break
        if time.monotonic() >= deadline:
            save()
            return False
        scheduled_entries = residual.diversity.scheduled_sources(train_entries, state["step"] + 1)
        entry = scheduled_entries[state["episode_position"]]
        episode = episode_for(entry, cache, step=state["step"] + 1, episode_position=state["episode_position"],
            context_row_cap=getattr(args, "context_row_cap", None),
            training_query_row_cap=getattr(args, "training_query_row_cap", None))
        coverage = state["coverage"][entry["family"]]
        coverage["context"][episode["context_indices"]] = True
        prepared = PreparedEpisode(episode)
        generated = None
        while state["query_offset"] < episode["n_query"]:
            if time.monotonic() >= deadline:
                save()
                return False
            started = time.monotonic()
            query_start = state["query_offset"]
            query_stop = min(query_start + args.query_chunk_rows, episode["n_query"])
            context, views = prepared.chunk(query_start, query_stop, args.device)
            if generated is None:
                generated = GeneratedMap(backbone, model, context, episode["n_classes"])
            query_weight = (query_stop - query_start) / episode["n_query"]
            loss = generated.backward_chunk(backbone, views, context, episode["y_query"][query_start:query_stop],
                weight=query_weight / 4., measure_native=query_start == 0)
            state["episode_loss_sum"] += loss * query_weight
            coverage["query"][episode["query_indices"][query_start:query_stop]] = True
            state["query_offset"] = query_stop
            state["record"] += 1
            append_jsonl(args.output_dir / "chunks.jsonl", dict(record=state["record"], step=state["step"] + 1,
                episode_position=state["episode_position"], family=entry["family"], n_context=episode["n_context"],
                n_query=episode["n_query"], query_pool_rows=episode["query_pool_rows"],
                query_start=query_start, query_stop=query_stop, chunk_loss=loss,
                context_hash=prior.row_digest(episode["context_indices"]), query_hash=prior.row_digest(episode["query_indices"][query_start:query_stop]),
                seconds=time.monotonic() - started,
                native_chunk_nll=generated.last_native_nll,
                mean_spline_gate=float(generated.parameters.spline_gate.detach().mean()),
                mean_absolute_shift=float(generated.parameters.shift.detach().abs().mean()),
                mean_absolute_log_scale=float(generated.parameters.log_scale.detach().abs().mean()),
                unique_context=int(coverage["context"].sum()), unique_queries=int(coverage["query"].sum())))
            if state["record"] % args.save_every == 0:
                save()
            print(f"TRAIN step={state['step']+1} episode={state['episode_position']} {entry['family']} "
                  f"context={episode['n_context']} queries={query_stop}/{episode['n_query']} "
                  f"query_pool={episode['query_pool_rows']} CE={loss:.6f}", flush=True)
            del context, views
            if args.max_records is not None and state["record"] >= args.max_records:
                save()
                return False
        state["update_loss_sum"] += state["episode_loss_sum"] / 4.
        state.update(episode_position=state["episode_position"] + 1, query_offset=0, episode_loss_sum=0.)
        del generated, prepared, episode
        if state["episode_position"] == 4:
            gradient_norms = residual.core.diagnostic.gradient_norms(model)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError("nonfinite/zero optimizer-update gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            pilot.json_write(args.output_dir / "updates" / f"step{state['step']}.json", dict(
                step=state["step"], mean_episode_loss=state["update_loss_sum"],
                preclip_gradient_norm=gradient_norm, **gradient_norms))
            state.update(episode_position=0, update_loss_sum=0.,
                pending_validation=state["step"] % args.evaluate_every == 0 or state["step"] == args.steps)
        save()
    pilot.atomic_save(args.output_dir / "final.pt", dict(model=pilot.state_cpu(model), step=state["step"]))
    pilot.atomic_save(args.output_dir / "selected.pt", dict(model=state["best_model"], step=state["best_step"]))
    pilot.json_write(args.output_dir / "training_complete.json", dict(fingerprint=manifest["fingerprint"],
        final_step=state["step"], selected_step=state["best_step"],
        checkpoints={kind: pilot.hash_file(args.output_dir / f"{kind}.pt") for kind in ("final", "selected")}))
    return True


def evaluate(args):
    deadline = time.monotonic() + args.wall_minutes * 60
    backbone, initial, entries, manifest = setup(args)
    completed = previous.read(args.output_dir / "training_complete.json")
    if completed["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("training completion changed")
    models = {}
    for kind in ("final", "selected"):
        path = args.output_dir / f"{kind}.pt"
        if pilot.hash_file(path) != completed["checkpoints"][kind]:
            raise ValueError("evaluation checkpoint changed")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        models[kind] = copy.deepcopy(initial).eval().requires_grad_(False)
        models[kind].load_state_dict(saved["model"])
    for name, locked in manifest["previous_checkpoints"].items():
        arm, kind = name.split("_")
        path = args.source_dir / "runs" / f"{arm}_seed0" / f"{kind}.pt"
        if pilot.hash_file(path) != locked["sha256"]:
            raise ValueError("previous checkpoint changed")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if locked["step"] == 0:
            if any(not torch.equal(value, initial.state_dict()[name].cpu()) for name, value in saved["model"].items()):
                raise ValueError("identity checkpoint changed")
            models[name] = None
        else:
            models[name] = copy.deepcopy(initial).eval().requires_grad_(False)
            models[name].load_state_dict(saved["model"])
    records = []
    cache = prior.SourceCache(entries, capacity=1)
    for source_position, entry in enumerate(entries):
        for split in range(2):
            episode = episode_for(entry, cache, split=split, probe=entry["panel"] == "train",
                context_row_cap=getattr(args, "context_row_cap", None))
            fingerprint = previous.digest(dict(experiment=manifest["fingerprint"], checkpoints=completed["checkpoints"],
                source=entry["source_group"], split=split, stage="final_evaluation"))
            record = prediction_record(args, backbone, models, episode,
                args.output_dir / "evaluation" / f"source{source_position}_split{split}", fingerprint, deadline)
            if record is None:
                return False
            records.append(dict(record, panel=entry["panel"], exposure=entry.get("exposure", entry["panel"]),
                historical_relation=entry.get("historical_relation", "")))
    summaries = {f"{panel}/{method}": aggregate_records([record for record in records if record["panel"] == panel], method)
                 for panel in ("train", "validation", "evaluation") for method in models}
    pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=manifest["fingerprint"],
        summaries=summaries, records=records, nominal_evaluation_context_fraction=.8,
        context_row_cap=getattr(args, "context_row_cap", None),
        additional_context_fractions_status="50% and100% sweeps not run; actual-size resource measurement first"))
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "profile", "train", "evaluate", "pipeline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--reuse-prepared-dir", type=Path)
    parser.add_argument("--continue-from-dir", type=Path, help="Import the audited step46 checkpoint and AdamW moments")
    parser.add_argument("--bounded-continuation", action="store_true", help="Resume audited step142; train/validate through320, then stop")
    parser.add_argument("--context-row-cap", type=int, default=20000)
    parser.add_argument("--training-query-row-cap", type=int, help="Uniform query subset per training visit; evaluation remains complete")
    parser.add_argument("--candidate-manifest", type=Path, default=residual.DEFAULT_CANDIDATES)
    parser.add_argument("--expected-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--evaluate-every", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--query-chunk-rows", type=int, default=2048)
    parser.add_argument("--continuation-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--wall-minutes", type=float, default=450.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-records", type=int, help="Bounded local interruption/resume check")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    try:
        if args.command == "pipeline":
            profile(args)
            args.wall_minutes -= (time.monotonic() - started) / 60
            if args.wall_minutes <= 0:
                return
            training_started = time.monotonic()
            if train(args) and not args.bounded_continuation:
                args.wall_minutes -= (time.monotonic() - training_started) / 60
                if args.wall_minutes > 0:
                    evaluate(args)
        else:
            globals()[args.command](args)
    except Exception as error:
        pilot.json_write(args.output_dir / "failure.json", dict(command=args.command,
            error_type=type(error).__name__, error=str(error), no_row_cap_or_source_exclusion_applied=True))
        raise


if __name__ == "__main__":
    main()
