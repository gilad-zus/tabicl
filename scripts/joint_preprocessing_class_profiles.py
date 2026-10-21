"""Compare marginal and conditional class-profile inputs from frozen step576."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_frozen_transfer as frozen
from scripts import joint_preprocessing_lr_response as paired
from tabicl._hyperspline.class_profile_preprocessing import ClassProfileSplinePreprocessor

base = paired.base
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("marginal_profiles", "conditional_profiles")
PARENT_STEP = 576
PARENT_STATE_SHA256 = "9aeba9a41782b8c54cef145350e40f68fa182eabbb67c0916e73fea308faa615"


def specification(args):
    settings = frozen.specification(args)
    for source_path in (Path(__file__), ROOT / "scripts/run_joint_preprocessing_class_profiles.sh",
                        ROOT / "src/tabicl/_hyperspline/class_profile_preprocessing.py"):
        settings["code_hashes"][source_path.relative_to(ROOT).as_posix()] = base.residual.objective.canonical_hash(source_path)
    settings.update(arms=list(ARMS), panels=["training_replay", "reserved_training", "validation_context4096", "validation", "evaluation"],
        training_updates=args.additional_updates, additional_updates=args.additional_updates, parent_step=PARENT_STEP,
        training_context_cap=4096, training_query_cap=512, tasks_per_update=4, lr=.0003,
        initialization="common control576; identical zero-output profile branches; preserve old AdamW states",
        conditioning="existing31 column summaries plus16 interior/2 tail/1 missing histogram categories and class frequency",
        profile_architecture="20->32 GELU->32 GELU, class-frequency pool, zero-initialized32->64 projection",
        intervention="actual class-conditional probabilities versus marginal probabilities repeated per class",
        objective="unchanged equal-source native8 plus adapted8 mean-logit CE, temperature0.9",
        hypothesis="preserving each class distribution before pooling improves useful learning and dataset transfer",
        primary="fixed equal-update final candidate versus capacity-matched control, frozen start and native16",
        validation_selection="none; final endpoint only; no target fitting or validation guard",
        selection_qualification="all outside groups now historically inspected development evidence",
        parent_state_sha256=PARENT_STATE_SHA256,
        evaluation="16 actual fitting episodes;83 reserved-training panels at4096;10 validation split0 at4096;10 validation x3 and58 outside x2 at20k; all queries",
        wall_minutes=args.wall_minutes, evaluation_reserve_minutes=args.evaluation_reserve_minutes)
    return settings


def prepare(args):
    settings = specification(args)
    base.residual.diversity.verify_revision(settings)
    _, checkpoint, source_lock = frozen.parent_inputs(args, settings)
    if source_lock["files"]["state.pt"] != PARENT_STATE_SHA256:
        raise ValueError("parent AdamW state differs from the audited completed bytes")
    entries = torch.load(args.source_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    parent_state = torch.load(args.source_dir / "state.pt", weights_only=True, map_location="cpu")
    manifest = dict(settings=settings, source_lock=source_lock)
    manifest["fingerprint"] = base.previous.digest(manifest)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if base.previous.read(manifest_path) != manifest:
            raise ValueError("profile experiment inputs, code or settings changed")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("new experiment requires an empty output directory")
        base.pilot.json_write(manifest_path, manifest)
    print(f"LOCKED class-profile comparison {manifest['fingerprint']};83/10/58 sources; seed0", flush=True)
    return manifest, entries, checkpoint, parent_state


def load_arms(parent_state, device):
    models, optimizers = {}, {}
    reference = base.prior.new_model()
    parameter_names = [name for name, _ in reference.named_parameters()]
    saved_optimizer = parent_state["optimizers"]["control"]
    if len(saved_optimizer["param_groups"]) != 1:
        raise ValueError("expected the parent's single AdamW parameter group")
    saved_parameter_ids = saved_optimizer["param_groups"][0]["params"]
    if len(saved_parameter_ids) != len(parameter_names):
        raise ValueError("parent optimizer parameter ordering changed")
    for arm in ARMS:
        # Local RNG scopes ensure both new branches have identical initial weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = ClassProfileSplinePreprocessor(arm == "conditional_profiles")
        incompatible = model.load_state_dict(parent_state["models"]["control"], strict=False)
        if incompatible.unexpected_keys or any(not name.startswith(("encoder.class_profile_mlp.",
                "encoder.profile_projection.", "encoder.histogram_boundaries")) for name in incompatible.missing_keys):
            raise ValueError("parent model differs beyond the added profile branch")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
        named_parameters = dict(model.named_parameters())
        # Copy old moments by parameter name; new branch parameters start without moments.
        for name, parameter_id in zip(parameter_names, saved_parameter_ids, strict=True):
            if parameter_id in saved_optimizer["state"]:
                optimizer.state[named_parameters[name]] = copy.deepcopy(saved_optimizer["state"][parameter_id])
        base.residual.core.capacity.optimizer_to(optimizer, torch.device(device))
        models[arm], optimizers[arm] = model, optimizer
    return models, optimizers


def load_saved_pair(args, manifest, models, optimizers):
    saved = torch.load(args.output_dir / "state.pt", weights_only=True, map_location="cpu")
    if saved["fingerprint"] != manifest["fingerprint"] or set(saved["models"]) != set(ARMS):
        raise ValueError("paired resume changed")
    for arm in ARMS:
        models[arm].load_state_dict(saved["models"][arm])
        optimizers[arm].load_state_dict(saved["optimizers"][arm])
        base.residual.core.capacity.optimizer_to(optimizers[arm], torch.device(args.device))
    return saved


def train(args, backbone, models, optimizers, entries, manifest, deadline, parent_state):
    training_entries = [entry for entry in entries if entry["panel"] == "train"]
    completed_updates = 0
    coverage = copy.deepcopy(parent_state["coverage"])
    state_path = args.output_dir / "state.pt"
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("paired state exists; pass --resume")
        saved = load_saved_pair(args, manifest, models, optimizers)
        completed_updates, coverage = saved["completed_updates"], saved["coverage"]
    paired.trim_logs(args.output_dir, completed_updates)
    paired.save_pair(state_path, manifest, models, optimizers, completed_updates, coverage)
    source_cache = base.prior.SourceCache(training_entries, capacity=1)
    while completed_updates < args.additional_updates and time.monotonic() < deadline:
        update_started = time.monotonic()
        absolute_step = PARENT_STEP + completed_updates + 1
        episode_losses, native_losses = {arm: [] for arm in ARMS}, []
        for episode_position, entry in enumerate(base.residual.diversity.scheduled_sources(training_entries, absolute_step)):
            episode = base.episode_for(entry, source_cache, step=absolute_step, episode_position=episode_position,
                context_row_cap=4096, training_query_row_cap=512)
            prepared = base.PreparedEpisode(episode)
            context, views = prepared.chunk(0, episode["n_query"], args.device)
            for arm, model in models.items():
                generated = base.GeneratedMap(backbone, model, context, episode["n_classes"])
                loss = generated.backward_chunk(backbone, views, context, episode["y_query"], weight=.25, measure_native=True)
                episode_losses[arm].append(loss)
                if arm == ARMS[0]:
                    native_losses.append(generated.last_native_nll)
                elif not math.isclose(native_losses[-1], generated.last_native_nll, rel_tol=0., abs_tol=1e-7):
                    raise ValueError("paired native losses differ")
                base.append_jsonl(args.output_dir / "episodes.jsonl", dict(
                    additional_update=completed_updates + 1, step=absolute_step, arm=arm,
                    episode_position=episode_position, family=entry["family"],
                    context_rows=episode["n_context"], query_rows=episode["n_query"],
                    context_hash=base.prior.row_digest(episode["context_indices"]),
                    query_hash=base.prior.row_digest(episode["query_indices"]), candidate_nll=loss,
                    native_nll=generated.last_native_nll, profile_branch_rms=model.encoder.last_profile_rms,
                    mean_spline_gate=float(generated.parameters.spline_gate.detach().mean()),
                    mean_absolute_shift=float(generated.parameters.shift.detach().abs().mean()),
                    mean_absolute_log_scale=float(generated.parameters.log_scale.detach().abs().mean())))
                del generated
            coverage[entry["family"]]["context"][episode["context_indices"]] = True
            coverage[entry["family"]]["query"][episode["query_indices"]] = True
            del context, views, prepared, episode
        for arm in ARMS:
            profile_gradient_norm = math.sqrt(sum(float(parameter.grad.detach().square().sum())
                for name, parameter in models[arm].named_parameters()
                if name.startswith(("encoder.class_profile_mlp.", "encoder.profile_projection.")) and parameter.grad is not None))
            diagnostics = paired.optimizer_step(models[arm], optimizers[arm])
            base.append_jsonl(args.output_dir / "updates.jsonl", dict(additional_update=completed_updates + 1,
                step=absolute_step, arm=arm, learning_rate=.0003, profile_gradient_norm=profile_gradient_norm,
                mean_episode_nll=float(np.mean(episode_losses[arm])), mean_native_nll=float(np.mean(native_losses)), **diagnostics))
        completed_updates += 1
        paired.save_pair(state_path, manifest, models, optimizers, completed_updates, coverage)
        print(f"PAIRED UPDATE {completed_updates}/{args.additional_updates} absolute={absolute_step} "
              f"seconds={time.monotonic()-update_started:.2f} "
              f"CE={dict((arm, float(np.mean(values))) for arm, values in episode_losses.items())}", flush=True)
    return completed_updates


def panel_visits(args, entries, stage):
    if stage == "training_replay":
        log_path = args.output_dir / "episodes.jsonl"
        visits = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
        return [visit for visit in visits if visit["arm"] == ARMS[0]][:16]
    panel_name = "train" if stage == "reserved_training" else "validation" if stage == "validation_context4096" else stage
    split_count = 3 if stage == "validation" else 2 if stage == "evaluation" else 1
    return [dict(family=entry["family"], split=split) for entry in entries if entry["panel"] == panel_name
            for split in range(split_count)]


def evaluate_panel(args, backbone, models, entries, manifest, stage, deadline):
    completed_path = args.output_dir / stage / "complete.json"
    checkpoint_hashes = {name: base.pilot.hash_file(args.output_dir / f"{name}.pt") for name in models}
    panel_fingerprint = base.previous.digest(dict(experiment=manifest["fingerprint"], stage=stage, checkpoints=checkpoint_hashes))
    if completed_path.exists():
        if base.previous.read(completed_path)["fingerprint"] != panel_fingerprint:
            raise ValueError("completed panel changed")
        return True
    source_cache = base.prior.SourceCache(entries, capacity=1)
    entry_by_family = {entry["family"]: entry for entry in entries}
    records = []
    for panel_position, visit in enumerate(panel_visits(args, entries, stage)):
        fingerprint = base.previous.digest(dict(panel=panel_fingerprint, visit=visit))
        folder = args.output_dir / stage / f"episode{panel_position}"
        if (folder / "complete.json").exists():
            record = base.previous.read(folder / "complete.json")
            if record["fingerprint"] != fingerprint:
                raise ValueError("cached episode changed")
        else:
            if time.monotonic() >= deadline:
                return False
            entry = entry_by_family[visit["family"]]
            if stage == "training_replay":
                episode = base.episode_for(entry, source_cache, step=visit["step"],
                    episode_position=visit["episode_position"], context_row_cap=4096, training_query_row_cap=512)
                if any(base.prior.row_digest(episode[row_name]) != visit[hash_name]
                       for row_name, hash_name in (("context_indices", "context_hash"), ("query_indices", "query_hash"))):
                    raise ValueError("fitting replay differs from optimized rows")
            else:
                episode = base.episode_for(entry, source_cache, split=visit["split"],
                    probe=stage == "reserved_training",
                    context_row_cap=4096 if stage in ("reserved_training", "validation_context4096") else 20000)
            record = base.prediction_record(args, backbone, models, episode, folder, fingerprint, deadline)
            if record is None:
                return False
        entry = entry_by_family[visit["family"]]
        exposure = "previously_declared_source" if entry.get("historical_relation") else entry.get("exposure", stage)
        records.append(dict(record, source_group=entry["source_group"], exposure=exposure,
                            historical_relation=entry.get("historical_relation", "")))
    summaries = {name: base.aggregate_records(records, name) for name in models} if records else {}
    # Candidate/control comparisons use paired per-dataset losses, not separate win counts.
    comparisons = {}
    if records:
        for reference in (ARMS[0], "starting"):
            comparison_records = [dict(record, scores=dict(candidate=record["scores"][ARMS[1]],
                native16=record["scores"][reference])) for record in records]
            comparisons[f"conditional_vs_{reference}"] = base.aggregate_records(comparison_records, "candidate")
    outside_details = {arm: frozen.summaries([dict(record, scores=dict(candidate=record["scores"][arm],
        native16=record["scores"]["native16"])) for record in records]) for arm in ARMS} if stage == "evaluation" else {}
    base.pilot.json_write(completed_path, dict(fingerprint=panel_fingerprint, summaries=summaries,
        comparisons=comparisons, outside_details=outside_details, records=records,
        qualification="optimized fitting labels" if stage == "training_replay" else
        "permanently held-out rows of training datasets" if stage == "reserved_training" else
        "historically inspected dataset-disjoint development sources"))
    print(f"{stage.upper()} COMPLETE: {len(records)} episodes; "
          f"gains={dict((name, summary['mean_nll_reduction']) for name, summary in summaries.items())}", flush=True)
    return True


def pipeline(args):
    started = time.monotonic()
    deadline = started + args.wall_minutes * 60
    manifest, entries, checkpoint, parent_state = prepare(args)
    if (args.output_dir / "complete.json").exists():
        if base.previous.read(args.output_dir / "complete.json")["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("completed experiment changed")
        return
    models, optimizers = load_arms(parent_state, args.device)
    backbone, checkpoint_path, checkpoint_hash = base.pilot.load_frozen(args, torch.device(args.device))
    if dict(path=Path(checkpoint_path).name, sha256=checkpoint_hash) != manifest["source_lock"]["backbone"]:
        raise ValueError("TabICL backbone changed")
    base.configure_backbone(backbone)
    completion_path = args.output_dir / "training_complete.json"
    if completion_path.exists():
        completion = base.previous.read(completion_path)
        saved = load_saved_pair(args, manifest, models, optimizers)
        completed_updates = saved["completed_updates"]
        if completion["fingerprint"] != manifest["fingerprint"] or completion["completed_updates_per_arm"] != completed_updates:
            raise ValueError("locked training endpoint changed")
    else:
        completed_updates = train(args, backbone, models, optimizers, entries, manifest,
            deadline - args.evaluation_reserve_minutes * 60, parent_state)
        base.pilot.json_write(completion_path, dict(fingerprint=manifest["fingerprint"],
            completed_updates_per_arm=completed_updates, absolute_step=PARENT_STEP + completed_updates,
            reached_target=completed_updates == args.additional_updates))
    models["starting"] = base.prior.new_model().to(args.device)
    models["starting"].load_state_dict(checkpoint["model"])
    for name, model in models.items():
        model.eval().requires_grad_(False)
        model_state = base.pilot.state_cpu(model)
        export_path = args.output_dir / f"{name}.pt"
        absolute_step = PARENT_STEP if name == "starting" else PARENT_STEP + completed_updates
        if export_path.exists():
            saved = torch.load(export_path, weights_only=True, map_location="cpu")
            if saved["step"] != absolute_step or any(not torch.equal(value, saved["model"][key]) for key, value in model_state.items()):
                raise ValueError("locked evaluation checkpoint changed")
        else:
            base.pilot.atomic_save(export_path, dict(model=model_state, step=absolute_step))
    for stage in ("training_replay", "reserved_training", "validation_context4096", "validation", "evaluation"):
        if not evaluate_panel(args, backbone, models, entries, manifest, stage, deadline):
            print(f"BUDGET STOP: equal updates={completed_updates}; {stage} resumable; no automatic job", flush=True)
            return
    base.pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=manifest["fingerprint"],
        completed_updates_per_arm=completed_updates, reached_target=completed_updates == args.additional_updates,
        elapsed_seconds=time.monotonic() - started, qualification="development comparison; no pristine test claim"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pipeline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=base.residual.DEFAULT_CANDIDATES)
    parser.add_argument("--expected-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additional-updates", type=int, default=384)
    parser.add_argument("--wall-minutes", type=float, default=450.)
    parser.add_argument("--evaluation-reserve-minutes", type=float, default=150.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.query_chunk_rows, args.lr = 2048, .0003
    args.steps, args.evaluate_every, args.save_every, args.continuation_seed = args.additional_updates, args.additional_updates, 1, 0
    if args.additional_updates < 1 or not math.isfinite(args.wall_minutes) or not 0 < args.evaluation_reserve_minutes < args.wall_minutes:
        parser.error("positive updates and finite wall budget with a shorter positive evaluation reserve required")
    return args


if __name__ == "__main__":
    options = parse_args()
    try:
        globals()[options.command](options)
    except Exception as error:
        base.pilot.json_write(options.output_dir / "failure.json", dict(error_type=type(error).__name__, error=str(error)))
        raise
