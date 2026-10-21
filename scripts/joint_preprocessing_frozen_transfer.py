"""Confirm the frozen step576 preprocessor on the locked 58 outside sources."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from scripts import joint_preprocessing_fraction_context as base

ROOT = Path(__file__).resolve().parents[1]
PARENT_FINGERPRINT = "d2a497a5d8eba2b58d42001301691e7dceea7b07e57f7876a3ad3b4b10fe8f43"
PARENT_FILES = ("manifest.json", "sources.pt", "training_complete.json", "complete.json", "control.pt", "state.pt")
EXPECTED_COUNTS = dict(train=83, validation=10, evaluation=58)


def specification(args):
    settings = base.residual.settings(args)
    # Preserve all parent dependencies so inference cannot silently use a changed model path.
    for source_path in (Path(__file__), Path(base.__file__), Path(base.prior.__file__),
                        ROOT / "src/tabicl/_hyperspline/statistics_preprocessing.py",
                        ROOT / "scripts/joint_preprocessing_lr_response.py",
                        ROOT / "scripts/run_joint_preprocessing_lr_response.sh",
                        ROOT / "scripts/run_joint_preprocessing_fraction_context.sh",
                        ROOT / "scripts/run_joint_preprocessing_frozen_transfer.sh"):
        settings["code_hashes"][source_path.relative_to(ROOT).as_posix()] = base.residual.objective.canonical_hash(source_path)
    settings.update(source_dir=str(args.source_dir), parent_fingerprint=PARENT_FINGERPRINT,
        checkpoint="control.pt, LR0.0003, absolute step576", model_seed=0,
        arms=["candidate"], panels=["evaluation"], training_updates=0,
        initialization="frozen completed control576; no optimizer or target-specific selection",
        conditioning="unchanged context-only statistics encoder and affine/K20 spline residual heads",
        objective="inference only; aligned native16 versus native8 plus adapted8, temperature0.9",
        primary="equal-dataset mean NLL and W/L/T on58 sources outside current83/10 allocation",
        context_fraction=.8, context_row_cap=20000, query_row_cap=None, splits=2,
        query_chunk_rows=args.query_chunk_rows, evaluation_precision="unchanged FP32 parent execution",
        row_policy="two deterministic80/20 partitions; preserve every query; cap only context at20000",
        validation_selection="none; one checkpoint fixed before outside-source scores",
        selection_qualification="historically inspected development sources; catalog-only freshness unverified",
        hypothesis="the current seven-of-ten validation signal extends to additional dataset groups",
        success_rule="report breadth, magnitudes, harms, uncertainty and largest-contributor sensitivity",
        wall_minutes=args.wall_minutes)
    return settings


def evaluation_entries(entries):
    counts = {panel: sum(entry["panel"] == panel for entry in entries) for panel in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS or sum(counts.values()) != len(entries):
        raise ValueError(f"unexpected locked source allocation: {counts}")
    if len({entry["source_group"] for entry in entries}) != len(entries):
        raise ValueError("source groups overlap between training, validation or evaluation")
    if len({entry["family"] for entry in entries}) != len(entries):
        raise ValueError("source names are ambiguous")
    return sorted((entry for entry in entries if entry["panel"] == "evaluation"),
                  key=lambda entry: entry["source_group"])


def parent_inputs(args, settings):
    parent_manifest = base.verified_json(args.source_dir, "sources.pt")
    if parent_manifest["fingerprint"] != PARENT_FINGERPRINT:
        raise ValueError("wrong completed parent experiment")
    for filename, expected_hash in parent_manifest["settings"]["code_hashes"].items():
        if settings["code_hashes"].get(filename) != expected_hash:
            raise ValueError(f"parent dependency changed: {filename}")
    completed = base.previous.read(args.source_dir / "complete.json")
    training = base.previous.read(args.source_dir / "training_complete.json")
    for record in (completed, training):
        if record["fingerprint"] != PARENT_FINGERPRINT or record["completed_updates_per_arm"] != 256 or not record["reached_target"]:
            raise ValueError("parent did not finish the prescribed256 updates")
    if training["absolute_step"] != 576:
        raise ValueError("wrong parent endpoint")
    checkpoint = torch.load(args.source_dir / "control.pt", weights_only=True, map_location="cpu")
    paired_state = torch.load(args.source_dir / "state.pt", weights_only=True, map_location="cpu")
    if (checkpoint["step"] != 576 or paired_state["fingerprint"] != PARENT_FINGERPRINT
            or paired_state["completed_updates"] != 256):
        raise ValueError("wrong frozen checkpoint or paired endpoint")
    saved_control = paired_state["models"]["control"]
    if checkpoint["model"].keys() != saved_control.keys() or any(
            not torch.equal(value, saved_control[name]) for name, value in checkpoint["model"].items()):
        raise ValueError("exported control differs from the durable paired state")
    entries = torch.load(args.source_dir / "sources.pt", weights_only=True, map_location="cpu")["entries"]
    selected = evaluation_entries(entries)
    lock = dict(parent_fingerprint=PARENT_FINGERPRINT, checkpoint_step=576,
        files={filename: base.pilot.hash_file(args.source_dir / filename) for filename in PARENT_FILES},
        backbone=parent_manifest["backbone"], source_counts=parent_manifest["counts"])
    return selected, checkpoint, lock


def prepare(args):
    settings = specification(args)
    base.residual.diversity.verify_revision(settings)
    entries, checkpoint, source_lock = parent_inputs(args, settings)
    allocation = [{name: entry[name] for name in ("family", "source_group", "full_rows", "cache_path", "cache_sha256", "exposure")}
                  | dict(historical_relation=entry.get("historical_relation", "")) for entry in entries]
    manifest = dict(settings=settings, source_lock=source_lock, allocation=allocation)
    manifest["fingerprint"] = base.previous.digest(manifest)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if base.previous.read(manifest_path) != manifest:
            raise ValueError("frozen experiment inputs, checkpoint or settings changed")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("use an empty result directory for new inference")
        # Check the existing raw tables before loading any GPU model; no bank rebuild.
        source_cache = base.prior.SourceCache(entries, capacity=1)
        for entry in entries:
            source = source_cache.get(entry["family"])
            if len(source["labels"]) != entry["full_rows"]:
                raise ValueError(f"full-table row count changed: {entry['family']}")
        base.pilot.json_write(manifest_path, manifest)
    print(f"LOCKED {len(entries)} outside sources; control576 SHA256={source_lock['files']['control.pt']}; "
          f"fingerprint={manifest['fingerprint']}", flush=True)
    return manifest, entries, checkpoint


def summaries(records):
    result = dict(overall=base.aggregate_records(records, "candidate"))
    for exposure in sorted({record["exposure"] for record in records}):
        result[exposure] = base.aggregate_records([record for record in records if record["exposure"] == exposure], "candidate")
    per_dataset = result["overall"]["per_dataset"]
    largest = max(per_dataset, key=lambda row: row["native_nll"] - row["candidate_nll"])["family"]
    if len(per_dataset) > 1:
        result["without_largest_contributor"] = dict(removed_family=largest,
            **base.aggregate_records([record for record in records if record["family"] != largest], "candidate"))
    # Local Elo is a descriptive conversion of NLL outcomes, never a TabArena rating.
    for summary in result.values():
        adjusted_score = (summary["wins"] + .5 * summary["ties"] + .5) / (summary["datasets"] + 1)
        summary["local_nll_elo_equivalent"] = 400 * math.log10(adjusted_score / (1 - adjusted_score))
    return result


def pipeline(args):
    started = time.monotonic()
    deadline = started + args.wall_minutes * 60
    manifest, entries, checkpoint = prepare(args)
    completed_path = args.output_dir / "complete.json"
    if completed_path.exists():
        if base.previous.read(completed_path)["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("completed report changed")
        print("Frozen evaluation already complete", flush=True)
        return True
    backbone, checkpoint_path, checkpoint_hash = base.pilot.load_frozen(args, torch.device(args.device))
    if dict(path=Path(checkpoint_path).name, sha256=checkpoint_hash) != manifest["source_lock"]["backbone"]:
        raise ValueError("frozen TabICL checkpoint changed")
    base.configure_backbone(backbone)
    model = base.prior.new_model().to(args.device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"])
    source_cache = base.prior.SourceCache(entries, capacity=1)
    records = []
    for source_position, entry in enumerate(entries):
        for split in range(2):
            fingerprint = base.previous.digest(dict(experiment=manifest["fingerprint"],
                source_group=entry["source_group"], split=split))
            folder = args.output_dir / "predictions" / f"source{source_position}_split{split}"
            # Reuse completed scores directly; partial query chunks resume in the parent predictor.
            if (folder / "complete.json").exists():
                record = base.previous.read(folder / "complete.json")
                if record["fingerprint"] != fingerprint:
                    raise ValueError("cached prediction fingerprint changed")
            else:
                record = None
                if time.monotonic() < deadline:
                    episode = base.episode_for(entry, source_cache, split=split, context_row_cap=20000)
                    record = base.prediction_record(args, backbone, dict(candidate=model), episode, folder, fingerprint, deadline)
                if record is None:
                    base.pilot.json_write(args.output_dir / "progress.json", dict(fingerprint=manifest["fingerprint"],
                        completed_episodes=len(records), expected_episodes=2 * len(entries), complete=False))
                    print("Clean inference budget stop; completed query chunks are resumable", flush=True)
                    return False
            exposure = "previously_declared_source" if entry.get("historical_relation") else entry["exposure"]
            records.append(dict(record, source_group=entry["source_group"], exposure=exposure,
                                historical_relation=entry.get("historical_relation", "")))
    # Assert that scoring never changed any candidate weights or accumulated gradients.
    if any(not torch.equal(value.cpu(), checkpoint["model"][name]) for name, value in model.state_dict().items()):
        raise ValueError("candidate weights changed during frozen inference")
    if any(parameter.grad is not None for parameter in (*model.parameters(), *backbone.parameters())):
        raise ValueError("unexpected inference gradients")
    base.pilot.json_write(completed_path, dict(fingerprint=manifest["fingerprint"], datasets=len(entries),
        episodes=len(records), records=records, summaries=summaries(records), source_lock=manifest["source_lock"],
        elapsed_seconds=time.monotonic() - started, training_updates=0,
        qualification="outside current training/validation; historical exposure; no pristine-test claim",
        timing_qualification="combined native/candidate scoring time includes preprocessing and resume overhead"))
    print(f"COMPLETE: {len(entries)} datasets, {len(records)} episodes, no weight updates", flush=True)
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pipeline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=base.residual.DEFAULT_CANDIDATES)
    parser.add_argument("--expected-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-rows", type=int, default=2048)
    parser.add_argument("--wall-minutes", type=float, default=50.)
    # The inherited dependency builder validates this historical endpoint; no training loop uses it.
    parser.set_defaults(steps=576, evaluate_every=1, save_every=1, lr=.0003, continuation_seed=0)
    args = parser.parse_args(argv)
    if args.query_chunk_rows < 1 or not math.isfinite(args.wall_minutes) or args.wall_minutes <= 0:
        parser.error("positive query chunk size and finite wall budget required")
    return args


if __name__ == "__main__":
    options = parse_args()
    globals()[options.command](options)
