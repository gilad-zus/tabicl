"""Paired small/large-context zero-shot spline hypernetworks, seed zero."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from scripts import joint_preprocessing_residual_conditioning as residual
from tabicl._hyperspline.statistics_preprocessing import StatisticsSplineResidualPreprocessor

bank, pilot, previous = residual.bank.bank, residual.pilot, residual.previous
ARMS = ("small", "large")
CONTEXT_ROWS = dict(small=512, large=2048)
QUERY_ROWS = 256
SEED = 20261016
SOURCE_FINGERPRINT = "7149269ed57b83ce024a07934f293bae6b69b80b0bd41bba1686c997a9f149f1"


def new_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return StatisticsSplineResidualPreprocessor()


def settings(args):
    if args.probe_sources < 1:
        raise ValueError("at least one training-source probe is required")
    specification = residual.settings(args)
    repository = Path(__file__).resolve().parents[1]
    for source_path in (Path(__file__), Path(inspect.getfile(StatisticsSplineResidualPreprocessor)),
                        Path(__file__).with_name("run_joint_preprocessing_context_size.sh")):
        specification["code_hashes"][source_path.relative_to(repository).as_posix()] = residual.objective.canonical_hash(source_path)
    specification.update(
        arms=list(ARMS), context_rows=CONTEXT_ROWS, query_rows=QUERY_ROWS,
        lengths=None, fractions=None, seed=SEED, tasks_per_update=4,
        initialization="identical seed-zero statistics-only affine/K20 spline heads; exact native16 identity",
        transformation_family="affine and K20 monotone cubic spline; two native slots; no neural or mixing heads",
        hypothesis="larger labelled contexts improve shared learning and unseen-dataset transfer",
        paired_rows="same four sources and 256 queries; small context is a subset of large; no size capping",
        training_sources="size/class-eligible subset of original 160 source groups; full cached tables",
        panels=["source_probe", "real_validation"],
        row_policy="256 fixed source-probe queries excluded from training; full remaining pools sampled without a row cap",
        source_fingerprint=args.source_fingerprint, probe_sources=args.probe_sources,
        source_cache_dirs=[str(path) for path in (getattr(args, "source_cache_dirs", None) or [])],
        evaluation="both trained models at both context sizes; native16 uses the identical context/query rows",
        validation_selection="mean dataset NLL at 2048 context, step zero eligible; shared checkpoint selection only",
        primary="final equal-update unseen-dataset large-context comparison and improvement over native16",
        confirmation="reused development validation allocation, size filtered; fresh confirmation required after success",
        success_rule=dict(minimum_mean_nll_reduction=.005, minimum_win_fraction=.7,
                          positive_median_gain=True, require_adjacent_passing_checkpoints=True),
    )
    return specification


def stratified_positions(labels, count, random_generator, *, reserve=0):
    classes, class_counts = np.unique(labels, return_counts=True)
    quotas = bank._allocation(class_counts, count, minimum=1, upper=class_counts - reserve)
    selected_positions = np.concatenate([
        random_generator.choice(np.flatnonzero(labels == label), int(quota), replace=False)
        for label, quota in zip(classes, quotas, strict=True)
    ])
    return random_generator.permutation(selected_positions)


def partition_source(labels, seed):
    """Keep probe queries permanently outside every shared-training episode."""
    random_generator = np.random.default_rng(seed)
    probe_query_positions = stratified_positions(labels, QUERY_ROWS, random_generator, reserve=2)
    fitting_positions = np.setdiff1d(np.arange(len(labels)), probe_query_positions)
    if len(fitting_positions) < CONTEXT_ROWS["large"] + QUERY_ROWS:
        raise ValueError("fewer than 2304 fitting rows after reserving probe queries")
    return fitting_positions, probe_query_positions


def paired_positions(labels, fitting_positions, seed, query_positions=None):
    """Sample paired queries first, then nested label-complete contexts."""
    random_generator = np.random.default_rng(seed)
    if query_positions is None:
        query_local_positions = stratified_positions(labels[fitting_positions], QUERY_ROWS,
                                                     random_generator, reserve=1)
        query_positions = fitting_positions[query_local_positions]
        context_pool_positions = np.setdiff1d(fitting_positions, query_positions)
    else:
        query_positions = np.asarray(query_positions, dtype=np.int64)
        context_pool_positions = fitting_positions
        if np.intersect1d(context_pool_positions, query_positions).size:
            raise ValueError("probe queries overlap the fitting pool")
    large_local_positions = stratified_positions(labels[context_pool_positions], CONTEXT_ROWS["large"], random_generator)
    large_context_positions = context_pool_positions[large_local_positions]
    small_local_positions = stratified_positions(labels[large_context_positions], CONTEXT_ROWS["small"], random_generator)
    return dict(small=large_context_positions[small_local_positions], large=large_context_positions,
                query=query_positions)


def episode_from_rows(source, context_positions, query_positions, seed):
    """Fit category encoding and imputation using only the selected context."""
    context_frame = bank.unpack_frame(source["columns"], context_positions)
    query_frame = bank.unpack_frame(source["columns"], query_positions)
    encoder = bank.TransformToNumerical().fit(context_frame)
    context_parts = encoder.transform_parts(context_frame)
    query_parts = encoder.transform_parts(query_frame)
    categorical_count = context_parts.categorical.shape[1]
    numerical_count = context_parts.numerical.shape[1]
    if not 5 <= categorical_count + numerical_count <= 100 or numerical_count < 1:
        raise ValueError("expected 5-100 encoded columns and at least one numerical column")
    context_values = np.concatenate((context_parts.categorical, context_parts.numerical), axis=1).astype(np.float32)
    query_values = np.concatenate((query_parts.categorical, query_parts.numerical), axis=1).astype(np.float32)
    labels = np.asarray(source["labels"], dtype=np.int64)
    if not np.isfinite(context_values).all() or not np.isfinite(query_values).all():
        raise FloatingPointError("nonfinite encoded episode")
    if np.intersect1d(context_positions, query_positions).size:
        raise ValueError("query rows entered their own context")
    return dict(family=source["family"], source_group=source["source_group"], split_seed=int(seed),
        n_classes=int(np.unique(labels).size), categorical_features=list(range(categorical_count)),
        numerical_mask=torch.tensor([False] * categorical_count + [True] * numerical_count),
        x_context=torch.from_numpy(context_values)[None], x_query=torch.from_numpy(query_values)[None],
        y_context=torch.from_numpy(labels[context_positions].astype(np.float32))[None],
        y_query=torch.from_numpy(labels[query_positions].copy()),
        context_missing=torch.from_numpy(context_parts.numerical_missing.copy()),
        query_missing=torch.from_numpy(query_parts.numerical_missing.copy()),
        context_indices=torch.as_tensor(context_positions.copy()), query_indices=torch.as_tensor(query_positions.copy()),
        n_context=len(context_positions), n_query=len(query_positions))


class SourceCache:
    """Bound memory while keeping full original tables available on disk."""

    def __init__(self, entries, capacity=4):
        self.entries = {entry["family"]: entry for entry in entries}
        self.capacity, self.loaded = capacity, OrderedDict()

    def get(self, family):
        if family in self.loaded:
            self.loaded.move_to_end(family)
            return self.loaded[family]
        entry = self.entries[family]
        cache_path = Path(entry["cache_path"])
        if pilot.hash_file(cache_path) != entry["cache_sha256"]:
            raise ValueError(f"full-table source cache changed: {family}")
        source = torch.load(cache_path, weights_only=True, map_location="cpu")
        source.update(family=family, source_group=entry["source_group"])
        self.loaded[family] = source
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
        return source


def prepare(args):
    specification = settings(args)
    residual.diversity.verify_revision(specification)
    if (args.output_dir / "manifest.json").exists():
        checked_manifest(args)
        return
    intent_path = args.output_dir / "preparation.json"
    intent = dict(settings=specification)
    if intent_path.exists() and previous.read(intent_path) != intent:
        raise ValueError("preparation settings changed")
    pilot.json_write(intent_path, intent)
    source_manifest = previous.read(args.reuse_bank_dir / "manifest.json")
    if previous.digest({name: value for name, value in source_manifest.items() if name != "fingerprint"}) != args.source_fingerprint:
        raise ValueError("wrong original source bank fingerprint")
    original_banks_path = args.reuse_bank_dir / "banks_manifest.json"
    if pilot.hash_file(original_banks_path) != source_manifest["banks_manifest_sha256"]:
        raise ValueError("original source allocation changed")
    original_banks = previous.read(original_banks_path)
    entries, rejected, episodes = [], [], []
    for panel, original_panel in (("train", "large"), ("real_validation", "validation")):
        for source_position, selected_source in enumerate(original_banks["selection"][original_panel]):
            family = selected_source["name"]
            minimum_rows = CONTEXT_ROWS["large"] + QUERY_ROWS * (2 if panel == "train" else 1)
            if selected_source["descriptors"]["rows"] < minimum_rows:
                rejected.append(dict(family=family, panel=panel, reason=f"fewer than {minimum_rows} original rows"))
                continue
            identity = bank._entry_identity(selected_source["source_entry"])
            cache_filename = f"{hashlib.sha256(identity.encode()).hexdigest()[:20]}.pt"
            cache_directories = [args.reuse_bank_dir / "source_cache", *(getattr(args, "source_cache_dirs", None) or [])]
            cache_path = next((directory / cache_filename for directory in cache_directories
                               if (directory / cache_filename).exists()), None)
            if cache_path is None:
                raise FileNotFoundError(f"full source cache missing; refusing silently capped fallback: {cache_filename}")
            source = torch.load(cache_path, weights_only=True, map_location="cpu")
            if residual.bank.cached_entry_identity(source["entry"]) != residual.bank.cached_entry_identity(selected_source["source_entry"]):
                raise ValueError("full source identity differs from frozen allocation")
            source.update(family=family, source_group=selected_source["source_group"])
            labels = np.asarray(source["labels"], dtype=np.int64)
            if len(labels) != selected_source["descriptors"]["rows"]:
                raise ValueError("full source row count changed")
            # The earlier allocation fingerprinted complete tables before its 16K cap.
            # Replaying that hash verifies that restoring full pools adds no new source content.
            full_fingerprints = bank.content_fingerprints(bank.unpack_frame(source["columns"]), labels)
            if full_fingerprints["inputs_sha256"] != selected_source["inputs_sha256"]:
                raise ValueError("full source content differs from the original allocation")
            source_seed = SEED + source_position + (0 if panel == "train" else 10000)
            try:
                if panel == "train":
                    fitting_positions, probe_query_positions = partition_source(labels, source_seed)
                else:
                    probe_query_positions = stratified_positions(labels, QUERY_ROWS, np.random.default_rng(source_seed), reserve=1)
                    fitting_positions = np.setdiff1d(np.arange(len(labels)), probe_query_positions)
                positions = paired_positions(labels, fitting_positions, source_seed + 100000, probe_query_positions)
                paired_episodes = {arm: episode_from_rows(source, positions[arm], positions["query"], source_seed)
                                   for arm in ARMS}
                # Check the actual native preprocessing interface before spending GPU time.
                for episode in paired_episodes.values():
                    residual.objective.real._prepared_views(episode, 16)
            except ValueError as error:
                rejected.append(dict(family=family, panel=panel, reason=str(error)))
                continue
            entry = dict(family=family, source_group=source["source_group"], panel=panel,
                cache_path=str(cache_path.resolve()), cache_sha256=pilot.hash_file(cache_path),
                original_inputs_sha256=selected_source["inputs_sha256"],
                full_rows=len(labels), fitting_positions=torch.as_tensor(fitting_positions),
                probe_query_positions=torch.as_tensor(probe_query_positions), classes=int(np.unique(labels).size))
            entries.append(entry)
            episodes.append(dict(family=family, panel="source_probe" if panel == "train" else panel, episodes=paired_episodes))
            print(f"Eligible {panel}: {family}, full={len(labels)} fitting={len(fitting_positions)}", flush=True)
    train_entries = [entry for entry in entries if entry["panel"] == "train"]
    validation_entries = [entry for entry in entries if entry["panel"] == "real_validation"]
    if len(train_entries) < 16 or len(validation_entries) < 5:
        raise ValueError("insufficient eligible independent sources: need at least 16 train / 5 validation")
    # Probe membership uses only a seeded ordering, never previously observed outcomes.
    ordered_train = sorted(train_entries, key=lambda entry: entry["family"])
    probe_order = np.random.default_rng(SEED).permutation(len(ordered_train))[:args.probe_sources]
    probe_families = {ordered_train[int(position)]["family"] for position in probe_order}
    episodes = [entry for entry in episodes if entry["panel"] == "real_validation" or entry["family"] in probe_families]
    source_groups = [{entry["source_group"] for entry in panel_entries}
                     for panel_entries in (train_entries, validation_entries)]
    if source_groups[0] & source_groups[1]:
        raise ValueError("source groups overlap training and validation")
    pilot.atomic_save(args.output_dir / "sources.pt", dict(entries=entries))
    pilot.atomic_save(args.output_dir / "evaluation.pt", dict(entries=episodes))
    initial = new_model()
    pilot.atomic_save(args.output_dir / "initial.pt", dict(model=pilot.state_cpu(initial)))
    audit_entries = [{name: value for name, value in entry.items() if not isinstance(value, torch.Tensor)}
                     for entry in entries]
    pilot.json_write(args.output_dir / "eligibility.json", dict(accepted=audit_entries, rejected=rejected,
        train_sources=len(train_entries), validation_sources=len(validation_entries),
        source_probe_families=sorted(probe_families), allocation_policy="original train/validation assignment, size/class/interface only"))
    manifest = dict(settings=specification, counts=dict(train=len(train_entries), validation=len(validation_entries)),
        files={name: pilot.hash_file(args.output_dir / name) for name in ("sources.pt", "evaluation.pt", "initial.pt", "eligibility.json")},
        original_allocation_sha256=pilot.hash_file(original_banks_path),
        parameter_count=sum(parameter.numel() for parameter in initial.parameters()), no_test_bank=True)
    manifest["fingerprint"] = previous.digest(manifest)
    pilot.json_write(args.output_dir / "manifest.json", manifest)
    print(f"Prepared: {manifest['counts']}, parameters={manifest['parameter_count']}, fingerprint={manifest['fingerprint']}", flush=True)


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != settings(args) or previous.digest(
            {name: value for name, value in manifest.items() if name != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("experiment settings/code changed")
    for filename, expected_hash in manifest["files"].items():
        if pilot.hash_file(args.output_dir / filename) != expected_hash:
            raise ValueError(f"locked input changed: {filename}")
    residual.diversity.verify_revision(manifest["settings"])
    return manifest


def evaluation_bank(args):
    return torch.load(args.output_dir / "evaluation.pt", weights_only=True, map_location="cpu")["entries"]


def training_batch(arm, step, source_entries, source_cache):
    scheduled = residual.diversity.scheduled_sources(source_entries, step)
    episodes = []
    for batch_position, entry in enumerate(scheduled):
        source_seed = int(np.random.SeedSequence([SEED, step, batch_position]).generate_state(1)[0])
        source = source_cache.get(entry["family"])
        positions = paired_positions(np.asarray(source["labels"]), entry["fitting_positions"].numpy(), source_seed)
        episode = episode_from_rows(source, positions[arm], positions["query"], source_seed)
        episode.update(source_seed=source_seed, full_fitting_rows=len(entry["fitting_positions"]))
        episodes.append(episode)
    return episodes


def setup(args):
    manifest = checked_manifest(args)
    backbone, checkpoint_path, checkpoint_hash = pilot.load_frozen(args, torch.device(args.device))
    backbone.train().requires_grad_(False)
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    lock = dict(path=Path(checkpoint_path).name, sha256=checkpoint_hash)
    lock_path = args.output_dir / "backbone_lock.json"
    if lock_path.exists() and previous.read(lock_path) != lock:
        raise ValueError("backbone changed")
    pilot.json_write(lock_path, lock)
    model = new_model().to(args.device)
    model.load_state_dict(torch.load(args.output_dir / "initial.pt", weights_only=True, map_location="cpu")["model"])
    return backbone, model, manifest


def preflight(args):
    backbone, initial, manifest = setup(args)
    largest_probe = max(evaluation_bank(args), key=lambda entry: entry["episodes"]["large"]["x_context"].shape[-1])
    for arm in ARMS:
        residual.execution_audit(backbone, initial, largest_probe["episodes"][arm],
                                 args.output_dir / "preflight" / arm, torch.device(args.device))
    pilot.json_write(args.output_dir / "preflight" / "complete.json", dict(fingerprint=manifest["fingerprint"], passed=True))


def metrics_summary(rows):
    candidate = np.asarray([row["candidate_nll"] for row in rows])
    baseline = np.asarray([row["native_nll"] for row in rows])
    relative_gains = 1 - (candidate + 1e-4) / (baseline + 1e-4)
    differences = baseline - candidate
    return dict(datasets=len(rows), mean_nll=float(candidate.mean()), native_mean_nll=float(baseline.mean()),
        mean_nll_reduction=float(1 - candidate.mean() / baseline.mean()),
        wins=int((differences > 1e-6).sum()), losses=int((differences < -1e-6).sum()),
        ties=int((np.abs(differences) <= 1e-6).sum()), median_relative_gain=float(np.median(relative_gains)),
        harms_over_1pct=int((relative_gains < -.01).sum()), harms_over_5pct=int((relative_gains < -.05).sum()),
        gain_percentiles={str(percentile): float(np.percentile(relative_gains, percentile)) for percentile in (0, 10, 25, 50, 75, 90, 100)})


def score(logits, labels):
    scores = previous.score(logits, labels)
    probabilities = (logits.float() / .9).softmax(-1).flatten(0, 1).detach().cpu().numpy()
    actual_labels = labels.flatten().cpu().numpy()
    scores["macro_f1"] = float(f1_score(actual_labels, probabilities.argmax(-1), average="macro", zero_division=0))
    # Keep the CSV schema fixed while marking AUC as inapplicable to multiclass tasks.
    scores["binary_auc"] = float(roc_auc_score(actual_labels, probabilities[:, 1])) if probabilities.shape[-1] == 2 else ""
    return scores


def correction_summary(backbone, model, episode):
    context, views = residual.prepared_episode(episode, next(model.parameters()).device)
    with torch.no_grad():
        numerical_values, parameters = residual.numeric_views(model, backbone, context, views)
        correction_sizes = []
        for view, values in zip(views, numerical_values, strict=True):
            if values is not None:
                native_values = view["x"].index_select(-1, view["columns"])
                valid_values = ~context["all_missing"]
                correction_sizes.append(float((values - native_values)[valid_values].square().mean().sqrt())
                                        if valid_values.any() else 0.)
    return dict(correction_rms=float(np.mean(correction_sizes)), spline_gate=float(parameters.spline_gate.mean()))


def evaluate(backbone, model, entries, references, folder, step):
    summaries = {}
    model.eval()
    for panel in ("source_probe", "real_validation"):
        for evaluation_arm in ARMS:
            rows = []
            for entry in entries:
                if entry["panel"] != panel:
                    continue
                episode = entry["episodes"][evaluation_arm]
                reference_key = f"{panel}/{entry['family']}/{evaluation_arm}"
                if reference_key not in references:
                    references[reference_key] = residual.episode_logits(backbone, None, episode).cpu()
                logits = residual.episode_logits(backbone, model, episode)
                labels = episode["y_query"].to(logits.device)
                candidate_scores = score(logits, labels)
                native_scores = score(references[reference_key].to(logits.device), labels)
                row = dict(step=step, panel=panel, evaluation_context=evaluation_arm, family=entry["family"],
                    n_context=episode["n_context"], n_query=episode["n_query"],
                    **{f"candidate_{name}": value for name, value in candidate_scores.items()},
                    **{f"native_{name}": value for name, value in native_scores.items()},
                    **correction_summary(backbone, model, episode))
                if not all(math.isfinite(value) for value in row.values() if isinstance(value, (float, int))):
                    raise FloatingPointError("nonfinite evaluation metrics")
                pilot.csv_append(folder / "evaluation_episodes.csv", row)
                rows.append(row)
            summary = metrics_summary(rows)
            summary_key = f"{panel}/{evaluation_arm}"
            summaries[summary_key] = summary
            pilot.csv_append(folder / "evaluation.csv", dict(step=step, panel=panel, evaluation_context=evaluation_arm,
                **{name: value for name, value in summary.items() if name != "gain_percentiles"}))
            print(f"step={step} {summary_key}: gain={100*summary['mean_nll_reduction']:.3f}% "
                  f"W/L/T={summary['wins']}/{summary['losses']}/{summary['ties']} "
                  f"median={100*summary['median_relative_gain']:.3f}% harms>5%={summary['harms_over_5pct']}", flush=True)
    return summaries


def row_digest(row_ids):
    return hashlib.sha256(np.asarray(row_ids, dtype="<i8").tobytes()).hexdigest()


def train(args, deadline=None):
    folder = args.output_dir / "runs" / f"{args.arm}_seed0"
    backbone, initial, manifest = setup(args)
    fingerprint = previous.digest(dict(experiment=manifest["fingerprint"], arm=args.arm, seed=0))
    if (folder / "complete.json").exists():
        if previous.read(folder / "complete.json")["fingerprint"] != fingerprint:
            raise ValueError("completed run changed")
        return True
    if args.device.startswith("cuda"):
        audit = previous.read(args.output_dir / "preflight" / "complete.json")
        if audit != dict(fingerprint=manifest["fingerprint"], passed=True):
            raise ValueError("paired GPU preflight must pass before training")
    entries = torch.load(args.output_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    train_entries = [entry for entry in entries if entry["panel"] == "train"]
    source_cache = SourceCache(train_entries)
    panels = evaluation_bank(args)
    model = copy.deepcopy(initial).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    state_path = folder / "state.pt"
    references_path = folder / "references.pt"
    references = torch.load(references_path, weights_only=True, map_location="cpu") if references_path.exists() else {}
    # Coverage is reconstructed from durable state, so resuming cannot inflate counts.
    coverage = {entry["family"]: dict(context=set(), query=set()) for entry in train_entries}
    step, best_step, clipped = 0, 0, 0
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("existing training state; pass --resume")
        state = torch.load(state_path, weights_only=True, map_location="cpu")
        if state["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        residual.core.capacity.optimizer_to(optimizer, torch.device(args.device))
        step, best_step, clipped = state["step"], state["best_step"], state["clipped"]
        best_score, best_model, last = state["best_score"], state["best_model"], state["last"]
        torch.set_rng_state(state["rng"])
        residual.core.restore_rng(state["other_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for family, roles in state["coverage"].items():
            coverage[family] = {role: set(row_ids.tolist()) for role, row_ids in roles.items()}
        for filename in ("training.csv", "presentations.csv", "evaluation.csv", "evaluation_episodes.csv", "coverage.csv"):
            residual.core.capacity.trim_csv(folder / filename, step)
    else:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError("partial run without durable state; use a fresh run directory")
        torch.manual_seed(SEED)
        last = evaluate(backbone, model, panels, references, folder, 0)
        best_score = last["real_validation/large"]["mean_nll"]
        best_model = pilot.state_cpu(model)
    pilot.atomic_save(references_path, references)

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=fingerprint, step=step, best_step=best_step, clipped=clipped,
            best_score=best_score, best_model=best_model, model=pilot.state_cpu(model), optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(), other_rng=residual.core.rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
            last=last, coverage={family: {role: torch.tensor(sorted(row_ids), dtype=torch.long)
                                        for role, row_ids in roles.items()} for family, roles in coverage.items()}))

    save()
    stop_step = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    if stop_step < step:
        raise ValueError("requested stop precedes durable checkpoint")
    for next_step in range(step + 1, stop_step + 1):
        if deadline is not None and time.monotonic() >= deadline:
            save()
            print(f"Clean wall-budget stop: {args.arm} at {step}/{args.steps}", flush=True)
            return False
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        model.train()
        losses = []
        for batch_position, episode in enumerate(training_batch(args.arm, next_step, train_entries, source_cache)):
            losses.append(float(residual.ensemble_backward(backbone, model, episode, scale=.25)))
            family_coverage = coverage[episode["family"]]
            family_coverage["context"].update(episode["context_indices"].tolist())
            family_coverage["query"].update(episode["query_indices"].tolist())
            pilot.csv_append(folder / "presentations.csv", dict(step=next_step, batch_position=batch_position,
                family=episode["family"], source_seed=episode["source_seed"], n_context=episode["n_context"],
                n_query=episode["n_query"], fitting_pool=episode["full_fitting_rows"],
                context_sha256=row_digest(episode["context_indices"]), query_sha256=row_digest(episode["query_indices"])))
        gradient_norms = residual.core.diagnostic.gradient_norms(model)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0 or not all(math.isfinite(value) for value in gradient_norms.values()):
            raise FloatingPointError("missing finite nonzero shared gradient")
        clipped += int(gradient_norm > 1.)
        optimizer.step()
        step = next_step
        pilot.csv_append(folder / "training.csv", dict(step=step, seconds=time.monotonic() - started,
            objective=float(np.mean(losses)), lr=args.lr, preclip_gradient_norm=gradient_norm,
            clipped_fraction=clipped / step, **gradient_norms))
        evaluated = step % args.evaluate_every == 0 or step == args.steps
        if evaluated:
            last = evaluate(backbone, model, panels, references, folder, step)
            current_score = last["real_validation/large"]["mean_nll"]
            if current_score < best_score:
                best_score, best_step, best_model = current_score, step, pilot.state_cpu(model)
            for entry in train_entries:
                roles = coverage[entry["family"]]
                pilot.csv_append(folder / "coverage.csv", dict(step=step, family=entry["family"],
                    fitting_pool=len(entry["fitting_positions"]), unique_context=len(roles["context"]),
                    unique_queries=len(roles["query"]), unique_either=len(roles["context"] | roles["query"])))
        if evaluated or step % args.save_every == 0 or step == stop_step:
            save()
            print(f"arm={args.arm} step={step}/{args.steps} CE={np.mean(losses):.6f} "
                  f"gradient={gradient_norm:.5g} best_val={best_score:.6f}@{best_step}", flush=True)
    if step != args.steps:
        return False
    for name, weights, saved_step in (("selected", best_model, best_step), ("final", pilot.state_cpu(model), step)):
        pilot.atomic_save(folder / f"{name}.pt", dict(fingerprint=fingerprint, model=weights, step=saved_step))
    pilot.json_write(folder / "complete.json", dict(fingerprint=fingerprint, arm=args.arm, steps=step,
        selected_step=best_step, selected_score=best_score, final=last,
        selected_sha256=pilot.hash_file(folder / "selected.pt"), final_sha256=pilot.hash_file(folder / "final.pt")))
    return True


def report(args):
    manifest = checked_manifest(args)
    completion = {}
    for arm in ARMS:
        folder = args.output_dir / "runs" / f"{arm}_seed0"
        completion[arm] = previous.read(folder / "complete.json")
        expected = previous.digest(dict(experiment=manifest["fingerprint"], arm=arm, seed=0))
        if completion[arm]["fingerprint"] != expected or completion[arm]["steps"] != args.steps:
            raise ValueError("incomplete or changed paired run")
        for checkpoint_name in ("selected", "final"):
            if pilot.hash_file(folder / f"{checkpoint_name}.pt") != completion[arm][f"{checkpoint_name}_sha256"]:
                raise ValueError("completed checkpoint changed")
    presentations = {arm: residual.core.capacity.read_rows(args.output_dir / "runs" / f"{arm}_seed0" / "presentations.csv")
                     for arm in ARMS}
    if any(len(rows) != args.steps * 4 for rows in presentations.values()):
        raise ValueError("incomplete source presentations")
    for small_row, large_row in zip(presentations["small"], presentations["large"], strict=True):
        for field_name in ("step", "batch_position", "family", "source_seed", "n_query", "fitting_pool", "query_sha256"):
            if small_row[field_name] != large_row[field_name]:
                raise ValueError(f"unpaired training data: {field_name}")
        if int(small_row["n_context"]) != CONTEXT_ROWS["small"] or int(large_row["n_context"]) != CONTEXT_ROWS["large"]:
            raise ValueError("training context was silently capped")
        if int(small_row["n_query"]) != QUERY_ROWS:
            raise ValueError("training query size changed")
    comparisons = {}
    for checkpoint_name in ("selected", "final"):
        selected_rows = {}
        for arm in ARMS:
            checkpoint_step = completion[arm]["selected_step"] if checkpoint_name == "selected" else args.steps
            rows = residual.core.capacity.read_rows(args.output_dir / "runs" / f"{arm}_seed0" / "evaluation_episodes.csv")
            selected_rows[arm] = {(row["panel"], row["evaluation_context"], row["family"]): row
                                  for row in rows if int(row["step"]) == checkpoint_step}
        if selected_rows["small"].keys() != selected_rows["large"].keys():
            raise ValueError("paired evaluation membership differs")
        comparisons[checkpoint_name] = {}
        for panel in ("source_probe", "real_validation"):
            for evaluation_arm in ARMS:
                keys = sorted(key for key in selected_rows["small"] if key[:2] == (panel, evaluation_arm))
                large_values = np.asarray([float(selected_rows["large"][key]["candidate_nll"]) for key in keys])
                small_values = np.asarray([float(selected_rows["small"][key]["candidate_nll"]) for key in keys])
                large_native = np.asarray([float(selected_rows["large"][key]["native_nll"]) for key in keys])
                small_native = np.asarray([float(selected_rows["small"][key]["native_nll"]) for key in keys])
                np.testing.assert_allclose(large_native, small_native, rtol=1e-6, atol=1e-7)
                paired_difference = small_values - large_values
                draws = np.random.default_rng(0).integers(len(keys), size=(20000, len(keys)))
                comparisons[checkpoint_name][f"{panel}/{evaluation_arm}"] = dict(
                    large_vs_small=metrics_summary([dict(candidate_nll=large, native_nll=small)
                                                   for large, small in zip(large_values, small_values, strict=True)]),
                    mean_nll_difference_ci95=np.percentile(paired_difference[draws].mean(1), [2.5, 97.5]).tolist(),
                    large_vs_native=metrics_summary([dict(candidate_nll=large, native_nll=native)
                                                    for large, native in zip(large_values, large_native, strict=True)]),
                    small_vs_native=metrics_summary([dict(candidate_nll=small, native_nll=native)
                                                    for small, native in zip(small_values, small_native, strict=True)]))
    pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=manifest["fingerprint"], runs=completion,
        comparisons=comparisons, paired_presentations_verified=args.steps * 4,
        scope="size-filtered reused development sources; no final confirmation claim"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "report", "pipeline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-bank-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=residual.DEFAULT_CANDIDATES)
    parser.add_argument("--source-fingerprint", default=SOURCE_FINGERPRINT)
    parser.add_argument("--source-cache-dirs", type=Path, nargs="+", help="Read full raw caches from earlier preparations; hash replay is required")
    parser.add_argument("--expected-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", choices=ARMS, default="small")
    parser.add_argument("--continuation-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--evaluate-every", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--probe-sources", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-minutes", type=float, default=450.)
    args = parser.parse_args()
    if args.command != "pipeline":
        globals()[args.command](args)
        return
    checked_manifest(args)
    deadline = time.monotonic() + args.wall_minutes * 60
    if not (args.output_dir / "preflight" / "complete.json").exists():
        preflight(args)
    for arm in ARMS:
        args.arm = arm
        if not train(args, deadline):
            return
    report(args)


if __name__ == "__main__":
    main()
