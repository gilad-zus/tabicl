"""Matched seed-zero learning-rate continuations of the completed step320 model."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_fraction_context as base

ARMS = {"control": .0003, "larger_lr": .003}
PARENT_FINGERPRINT = "33149f2872af1e3b887edd6b699addff4091dfca33670073ae6d8a994afc4708"
PARENT_STEP = 320
ROOT = Path(__file__).resolve().parents[1]


def specification(args):
    settings = base.residual.settings(args)
    # Lock the reused implementation as well as the new orchestration and launch command.
    for source_path in (Path(__file__), Path(base.__file__), Path(base.prior.__file__),
                        ROOT / "src/tabicl/_hyperspline/statistics_preprocessing.py",
                        ROOT / "scripts/run_joint_preprocessing_fraction_context.sh",
                        ROOT / "scripts/run_joint_preprocessing_lr_response.sh"):
        settings["code_hashes"][source_path.relative_to(ROOT).as_posix()] = base.residual.objective.canonical_hash(source_path)
    settings.update(arms=ARMS, model_seed=0, parent_dir=str(args.parent_dir),
        parent_fingerprint=PARENT_FINGERPRINT, parent_state_sha256=args.parent_state_sha256,
        parent_step=PARENT_STEP, additional_updates=args.additional_updates,
        training_context_cap=4096, training_query_cap=512, evaluation_context_cap=20000,
        evaluation_query_chunk=2048, tasks_per_update=4,
        optimizer_resets="none; preserve parent moments and step counters", lengths=None, fractions=None,
        conditioning="31 per-column context summaries, table-average summaries and three size descriptors",
        transformation_family="5912-parameter statistics-only affine/K20 monotone spline native residuals",
        training_precision="FP32; eight adapted-view gradient replays; frozen backbone/dropout0",
        evaluation_precision="FP32; complete context in every query chunk",
        initialization="identical step320 weights and AdamW moments; override only each arm's learning rate",
        objective="CE of aligned native8 plus adapted8 logits; temperature0.9; equal source weights",
        row_policy="same source/context/query/view draws in both arms; reserved probes remain excluded",
        validation="all queries, ten existing validation sources, three fixed partitions; no checkpoint selection",
        fitting_probe="eight score-blind source IDs; replay actual parent training episodes at original sizes",
        training_replay="first eight actual continuation episodes, identical4096/512 training sizes; fitting evidence",
        hypothesis="larger optimizer updates improve useful learning in the current compact hypernetwork",
        primary="paired final equal-update arms and common starting checkpoint; development evidence",
        selection_qualification="reused validation; no test or catalog inference",
        success_rule="learning-response diagnostic; report magnitude, breadth and intervals; no automatic scaling decision",
        wall_minutes=args.wall_minutes, evaluation_reserve_minutes=args.evaluation_reserve_minutes)
    return settings


def read_parent(args):
    manifest = base.verified_json(args.parent_dir, "sources.pt")
    if manifest["fingerprint"] != PARENT_FINGERPRINT:
        raise ValueError("unexpected parent experiment")
    state_path = args.parent_dir / "state.pt"
    if base.pilot.hash_file(state_path) != args.parent_state_sha256:
        raise ValueError("parent state differs from locally audited bytes")
    state = torch.load(state_path, weights_only=True, map_location="cpu")
    if state["fingerprint"] != PARENT_FINGERPRINT or state["step"] != PARENT_STEP:
        raise ValueError("parent must be completed step320")
    if state["episode_position"] or state["query_offset"] or state["pending_validation"]:
        raise ValueError("parent is not at a clean completed update boundary")
    if any(gradient is not None for gradient in state["gradients"].values()):
        raise ValueError("parent has pending gradients")
    return manifest, state


def fitting_panel(args):
    # Choose source IDs without scores, then find each ID's first actual parent presentation.
    records = [json.loads(line) for line in (args.parent_dir / "chunks.jsonl").read_text().splitlines()]
    selected_families = sorted({record["family"] for record in records})[:8]
    return [next(record for record in records if record["family"] == family) for family in selected_families]


def prepare(args):
    settings = specification(args)
    base.residual.diversity.verify_revision(settings)
    parent_manifest, _ = read_parent(args)
    # Every model/backbone dependency must match the parent; only orchestration is new.
    for filename, expected_hash in parent_manifest["settings"]["code_hashes"].items():
        if filename not in settings["code_hashes"] or settings["code_hashes"][filename] != expected_hash:
            raise ValueError(f"parent dependency changed: {filename}")
    if (args.output_dir / "manifest.json").exists():
        return checked_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent_sources = torch.load(args.parent_dir / "sources.pt", weights_only=True, map_location="cpu")
    base.pilot.atomic_save(args.output_dir / "sources.pt", parent_sources)
    base.pilot.json_write(args.output_dir / "fitting_panel.json", fitting_panel(args))
    manifest = dict(settings=settings, counts=parent_manifest["counts"], backbone=parent_manifest["backbone"],
        parent_sources_sha256=parent_manifest["files"]["sources.pt"],
        files={filename: base.pilot.hash_file(args.output_dir / filename)
               for filename in ("sources.pt", "fitting_panel.json")})
    manifest["fingerprint"] = base.previous.digest(manifest)
    base.pilot.json_write(args.output_dir / "manifest.json", manifest)
    print(f"PREPARED {manifest['fingerprint']} counts={manifest['counts']}", flush=True)
    return manifest


def checked_manifest(args):
    manifest = base.previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != specification(args):
        raise ValueError("experiment settings changed")
    if base.previous.digest({name: value for name, value in manifest.items() if name != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("manifest fingerprint changed")
    for filename, expected_hash in manifest["files"].items():
        if base.pilot.hash_file(args.output_dir / filename) != expected_hash:
            raise ValueError(f"experiment input changed: {filename}")
    base.residual.diversity.verify_revision(manifest["settings"])
    return manifest


def load_arms(parent_state, device):
    models, optimizers = {}, {}
    for arm, learning_rate in ARMS.items():
        model = base.prior.new_model().to(device)
        model.load_state_dict(parent_state["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4,
                                     betas=(.9, .999), eps=1e-8)
        optimizer.load_state_dict(copy.deepcopy(parent_state["optimizer"]))
        # Loading AdamW also restores its old rate; apply the experimental intervention afterwards.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        base.residual.core.capacity.optimizer_to(optimizer, torch.device(device))
        optimizer.zero_grad(set_to_none=True)
        models[arm], optimizers[arm] = model, optimizer
    return models, optimizers


def optimizer_step(model, optimizer):
    before_update = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    gradient_groups = base.residual.core.diagnostic.gradient_norms(model)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise FloatingPointError("nonfinite or zero gradient norm")
    optimizer.step()
    update_squared = sum(float((parameter.detach() - before_update[name]).square().sum())
                         for name, parameter in model.named_parameters())
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("nonfinite updated parameters")
    optimizer.zero_grad(set_to_none=True)
    return dict(preclip_gradient_norm=gradient_norm, clipped=gradient_norm > 1.,
                parameter_update_norm=math.sqrt(update_squared), **gradient_groups)


def save_pair(path, manifest, models, optimizers, completed_updates, coverage):
    # One atomic file commits both arms together; an interrupted pair is replayed in full.
    base.pilot.atomic_save(path, dict(fingerprint=manifest["fingerprint"], completed_updates=completed_updates,
        models={arm: base.pilot.state_cpu(model) for arm, model in models.items()},
        optimizers={arm: optimizer.state_dict() for arm, optimizer in optimizers.items()}, coverage=coverage))


def trim_logs(output_dir, completed_updates):
    for filename in ("episodes.jsonl", "updates.jsonl"):
        log_path = output_dir / filename
        if log_path.exists():
            retained = [line for line in log_path.read_text().splitlines()
                        if json.loads(line)["additional_update"] <= completed_updates]
            log_path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")


def load_saved_pair(args, models, optimizers, manifest):
    saved = torch.load(args.output_dir / "state.pt", weights_only=True, map_location="cpu")
    if saved["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("paired resume fingerprint changed")
    for arm in ARMS:
        models[arm].load_state_dict(saved["models"][arm])
        optimizers[arm].load_state_dict(saved["optimizers"][arm])
        base.residual.core.capacity.optimizer_to(optimizers[arm], torch.device(args.device))
    return saved


def train(args, backbone, models, optimizers, entries, manifest, deadline, parent_state):
    train_entries = [entry for entry in entries if entry["panel"] == "train"]
    state_path = args.output_dir / "state.pt"
    completed_updates, coverage = 0, copy.deepcopy(parent_state["coverage"])
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("paired state exists; pass --resume")
        saved = load_saved_pair(args, models, optimizers, manifest)
        completed_updates, coverage = saved["completed_updates"], saved["coverage"]
    trim_logs(args.output_dir, completed_updates)
    save_pair(state_path, manifest, models, optimizers, completed_updates, coverage)
    cache = base.prior.SourceCache(train_entries, capacity=1)
    training_started = time.monotonic()
    initial_updates = completed_updates
    while completed_updates < args.additional_updates and time.monotonic() < deadline:
        update_started = time.monotonic()
        absolute_step = PARENT_STEP + completed_updates + 1
        episode_losses = {arm: [] for arm in ARMS}
        for episode_position, entry in enumerate(base.residual.diversity.scheduled_sources(train_entries, absolute_step)):
            episode = base.episode_for(entry, cache, step=absolute_step, episode_position=episode_position,
                                       context_row_cap=4096, training_query_row_cap=512)
            prepared = base.PreparedEpisode(episode)
            context, views = prepared.chunk(0, episode["n_query"], args.device)
            for arm, model in models.items():
                started = time.monotonic()
                generated = base.GeneratedMap(backbone, model, context, episode["n_classes"])
                loss = generated.backward_chunk(backbone, views, context, episode["y_query"],
                                                weight=.25, measure_native=True)
                episode_losses[arm].append(loss)
                base.append_jsonl(args.output_dir / "episodes.jsonl", dict(
                    additional_update=completed_updates + 1, step=absolute_step, arm=arm,
                    episode_position=episode_position, family=entry["family"],
                    context_rows=episode["n_context"], query_rows=episode["n_query"],
                    context_hash=base.prior.row_digest(episode["context_indices"]),
                    query_hash=base.prior.row_digest(episode["query_indices"]),
                    candidate_nll=loss, native_nll=generated.last_native_nll,
                    mean_spline_gate=float(generated.parameters.spline_gate.detach().mean()),
                    mean_absolute_shift=float(generated.parameters.shift.detach().abs().mean()),
                    mean_absolute_log_scale=float(generated.parameters.log_scale.detach().abs().mean()),
                    seconds=time.monotonic() - started))
                del generated
            coverage[entry["family"]]["context"][episode["context_indices"]] = True
            coverage[entry["family"]]["query"][episode["query_indices"]] = True
            del context, views, prepared, episode
        for arm in ARMS:
            diagnostics = optimizer_step(models[arm], optimizers[arm])
            base.append_jsonl(args.output_dir / "updates.jsonl", dict(
                additional_update=completed_updates + 1, step=absolute_step, arm=arm,
                learning_rate=optimizers[arm].param_groups[0]["lr"],
                mean_episode_nll=float(np.mean(episode_losses[arm])), **diagnostics))
        completed_updates += 1
        save_pair(state_path, manifest, models, optimizers, completed_updates, coverage)
        elapsed = time.monotonic() - training_started
        mean_seconds = elapsed / (completed_updates - initial_updates)
        print(f"PAIRED UPDATE {completed_updates}/{args.additional_updates} absolute={absolute_step} "
              f"seconds={time.monotonic()-update_started:.2f} mean_seconds={mean_seconds:.2f} "
              f"remaining_training_estimate_minutes={(args.additional_updates-completed_updates)*mean_seconds/60:.1f} "
              f"CE={dict((arm, round(float(np.mean(values)), 6)) for arm, values in episode_losses.items())}", flush=True)
    return completed_updates


def evaluate_panel(args, backbone, models, entries, manifest, stage, deadline):
    records = []
    cache = base.prior.SourceCache(entries, capacity=1)
    fitting = stage in ("fitting", "training_replay")
    if stage == "training_replay":
        # These exact episodes were optimized by both arms, so row-size changes do not confound fitting assessment.
        visits = [json.loads(line) for line in (args.output_dir / "episodes.jsonl").read_text().splitlines()]
        panel = [visit for visit in visits if visit["arm"] == "control"][:8]
    else:
        panel = base.previous.read(args.output_dir / "fitting_panel.json") if fitting else [
            dict(family=entry["family"], split=split)
            for entry in entries if entry["panel"] == "validation" for split in range(3)]
    entry_by_family = {entry["family"]: entry for entry in entries}
    for panel_position, visit in enumerate(panel):
        entry = entry_by_family[visit["family"]]
        if fitting:
            # Both replay panels use exact logged fitting rows; neither measures generalization.
            episode = base.episode_for(entry, cache, step=visit["step"],
                episode_position=visit["episode_position"],
                context_row_cap=4096 if stage == "training_replay" else 20000,
                training_query_row_cap=512 if stage == "training_replay" else 2048)
            if (base.prior.row_digest(episode["context_indices"]) != visit["context_hash"] or
                    base.prior.row_digest(episode["query_indices"]) != visit["query_hash"]):
                raise ValueError("fitting replay rows differ from actual parent presentation")
        else:
            episode = base.episode_for(entry, cache, split=visit["split"], context_row_cap=20000)
        fingerprint = base.previous.digest(dict(experiment=manifest["fingerprint"], stage=stage,
            checkpoints={name: base.pilot.hash_file(args.output_dir / f"{name}.pt") for name in models},
            source=entry["source_group"], panel_position=panel_position))
        record = base.prediction_record(args, backbone, models, episode,
            args.output_dir / stage / f"episode{panel_position}", fingerprint, deadline)
        if record is None:
            return False
        records.append(record)
    summaries = {name: base.aggregate_records(records, name) for name in models} if records else {}
    base.pilot.json_write(args.output_dir / stage / "complete.json", dict(
        fingerprint=manifest["fingerprint"], summaries=summaries, records=records,
        qualification="optimized continuation rows" if stage == "training_replay" else
            "previously optimized parent fitting rows" if fitting else "reused unseen-source development validation"))
    print(f"{stage.upper()} SUMMARY {json.dumps(summaries)}", flush=True)
    return True


def pipeline(args):
    started = time.monotonic()
    deadline = started + args.wall_minutes * 60
    prepare(args)
    manifest = checked_manifest(args)
    _, parent_state = read_parent(args)
    backbone, checkpoint_path, checkpoint_hash = base.pilot.load_frozen(args, torch.device(args.device))
    if dict(path=Path(checkpoint_path).name, sha256=checkpoint_hash) != manifest["backbone"]:
        raise ValueError("backbone changed")
    base.configure_backbone(backbone)
    models, optimizers = load_arms(parent_state, args.device)
    entries = torch.load(args.output_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    training_completion_path = args.output_dir / "training_complete.json"
    if training_completion_path.exists():
        # Resume evaluation at the locked equal-update endpoint, including a budget-limited endpoint.
        completed = base.previous.read(training_completion_path)
        saved = load_saved_pair(args, models, optimizers, manifest)
        completed_updates = saved["completed_updates"]
        if completed["fingerprint"] != manifest["fingerprint"] or completed["completed_updates_per_arm"] != completed_updates:
            raise ValueError("locked training endpoint changed")
    else:
        completed_updates = train(args, backbone, models, optimizers, entries, manifest,
            deadline - args.evaluation_reserve_minutes * 60, parent_state)
        base.pilot.json_write(training_completion_path, dict(
            fingerprint=manifest["fingerprint"], completed_updates_per_arm=completed_updates,
            requested_updates_per_arm=args.additional_updates, reached_target=completed_updates == args.additional_updates,
            absolute_step=PARENT_STEP + completed_updates))
    models["starting"] = base.prior.new_model().to(args.device)
    models["starting"].load_state_dict(parent_state["model"])
    for name, model in models.items():
        model.eval().requires_grad_(False)
        checkpoint_path = args.output_dir / f"{name}.pt"
        checkpoint_step = PARENT_STEP if name == "starting" else PARENT_STEP + completed_updates
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            if saved["step"] != checkpoint_step or any(not torch.equal(value.cpu(), saved["model"][key])
                    for key, value in model.state_dict().items()):
                raise ValueError("locked evaluation checkpoint changed")
        else:
            base.pilot.atomic_save(checkpoint_path, dict(model=base.pilot.state_cpu(model), step=checkpoint_step))
    for stage in ("training_replay", "fitting", "validation"):
        if not evaluate_panel(args, backbone, models, entries, manifest, stage, deadline):
            print(f"BUDGET STOP: equal updates={completed_updates}; {stage} resumable; no job chained", flush=True)
            return
    base.pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=manifest["fingerprint"],
        completed_updates_per_arm=completed_updates, reached_target=completed_updates == args.additional_updates,
        elapsed_seconds=time.monotonic() - started, no_test_evaluation=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pipeline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--parent-state-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=base.residual.DEFAULT_CANDIDATES)
    parser.add_argument("--expected-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additional-updates", type=int, default=256)
    parser.add_argument("--wall-minutes", type=float, default=450.)
    parser.add_argument("--evaluation-reserve-minutes", type=float, default=120.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Shared prediction code chunks evaluation only; training always uses its complete <=512 queries.
    args.query_chunk_rows = 2048
    args.lr = .0003
    args.steps = args.additional_updates
    args.evaluate_every = args.additional_updates
    args.save_every = 1
    args.continuation_seed = 0
    if args.additional_updates < 1 or not 0 < args.evaluation_reserve_minutes < args.wall_minutes:
        parser.error("positive update budget and evaluation reserve shorter than total wall time required")
    return args


def main():
    args = parse_args()
    try:
        globals()[args.command](args)
    except Exception as error:
        base.pilot.json_write(args.output_dir / "failure.json", dict(
            error_type=type(error).__name__, error=str(error), command=args.command))
        raise


if __name__ == "__main__":
    main()
