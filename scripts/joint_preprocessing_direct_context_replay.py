"""Replay frozen direct maps on identical queries with original and expanded contexts."""
from __future__ import annotations

import argparse
import copy
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_statistics_diagnostic as source

pilot, previous, residual = source.pilot, source.previous, source.residual
SCORE_ATOL, SCORE_RTOL = 1e-6, 1e-5
SOURCE_FINGERPRINT = "fd95d52d80b8e41436338776031e1b2babd1df330bf5a86b3d114858b7ad730d"


def query_episode(entry, context_choice):
    """Keep outer query rows unchanged; substitute only the labelled context."""
    outer_episode, fitting_episode = entry["outer"], entry["fitting"]
    if context_choice not in ("original", "expanded"):
        raise ValueError("unknown context choice")
    # The saved fitting and selection episodes must have the same inner context.
    for context_field in ("x_context", "y_context", "context_missing", "context_indices"):
        if not torch.equal(fitting_episode[context_field], entry["selection"][context_field]):
            raise ValueError("fitting and selection contexts differ")
    outer_query_rows = set(outer_episode["query_indices"].tolist())
    for partition in (outer_episode, fitting_episode, entry["selection"]):
        if outer_query_rows.intersection(partition["context_indices"].tolist()):
            raise ValueError("outer query row in labelled context")
    for partition in (fitting_episode, entry["selection"]):
        if outer_query_rows.intersection(partition["query_indices"].tolist()):
            raise ValueError("outer query row in fitting or selection queries")
    episode = dict(outer_episode)
    chosen_context = fitting_episode if context_choice == "original" else outer_episode
    for context_field in ("x_context", "y_context", "context_missing", "context_indices", "n_context"):
        episode[context_field] = chosen_context[context_field]
    episode.update(stage=f"outer_query_{context_choice}_context",
                   actual_rows=episode["n_context"] + episode["n_query"])
    return episode


def check_score(actual, recorded):
    """Reject replay drift before interpreting tiny differences as gains."""
    for metric in ("nll", "accuracy"):
        if not math.isclose(actual[metric], recorded[metric], abs_tol=SCORE_ATOL, rel_tol=SCORE_RTOL):
            raise ValueError(f"recorded {metric} did not replay: {actual[metric]} vs {recorded[metric]}")
    return abs(actual["nll"] - recorded["nll"])


def checked_inputs(source_dir, expected_fingerprint=SOURCE_FINGERPRINT):
    repository = Path(__file__).resolve().parents[1]
    manifest = previous.read(source_dir / "manifest.json")
    if manifest["fingerprint"] != expected_fingerprint:
        raise ValueError("unexpected source experiment")
    if previous.digest({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("source manifest fingerprint changed")
    # The new replay can run at a newer revision; every original dependency must match.
    for filename, expected_hash in manifest["settings"]["code_hashes"].items():
        if residual.objective.canonical_hash(repository / filename) != expected_hash:
            raise ValueError(f"source dependency changed: {filename}")
    input_hashes = {}
    for filename, expected_hash in (
        ("banks_manifest.json", manifest["banks_manifest_sha256"]),
        ("direct_bank_manifest.json", manifest["direct_bank_manifest_sha256"]),
        ("initial_statistics.pt", manifest["initial_sha256"]["statistics"]),
    ):
        actual_hash = pilot.hash_file(source_dir / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"source input changed: {filename}")
        input_hashes[filename] = actual_hash
    if previous.read(source_dir / "backbone_lock.json") != manifest["backbone"]:
        raise ValueError("source backbone lock changed")
    completion = previous.read(source_dir / "complete.json")
    preflight = previous.read(source_dir / "preflight" / "complete.json")
    if completion["fingerprint"] != manifest["fingerprint"] or preflight["fingerprint"] != manifest["fingerprint"] or not preflight["both_passed"]:
        raise ValueError("source run or GPU preflight incomplete")
    bank_lock = previous.read(source_dir / "direct_bank_manifest.json")
    if bank_lock["specification_fingerprint"] != previous.digest(manifest["settings"]) or bank_lock["protocol"] != manifest["settings"]["direct_reference"]:
        raise ValueError("direct bank protocol changed")
    bank_path = (source_dir / bank_lock["path"]).resolve()
    if source_dir.resolve() not in bank_path.parents or pilot.hash_file(bank_path) != bank_lock["sha256"]:
        raise ValueError("direct bank path or hash changed")
    entries = torch.load(bank_path, map_location="cpu", weights_only=True)["values"]
    if [entry["outer"]["family"] for entry in entries] != bank_lock["families"]:
        raise ValueError("direct source ordering changed")
    input_hashes[bank_lock["path"]] = bank_lock["sha256"]
    for source_position, entry in enumerate(entries):
        folder = source_dir / "direct" / f"source_{source_position:02d}"
        reference = previous.read(folder / "complete.json")
        if reference["fingerprint"] != manifest["fingerprint"] or reference["family"] != entry["outer"]["family"]:
            raise ValueError("direct reference provenance changed")
        if reference["outer_query_used_for_optimization"] or reference["outer_query_used_for_selection"]:
            raise ValueError("outer queries used in direct fitting")
        for filename in ("complete.json", "selected.pt", "state.pt"):
            input_hashes[f"direct/source_{source_position:02d}/{filename}"] = pilot.hash_file(folder / filename)
        if input_hashes[f"direct/source_{source_position:02d}/selected.pt"] != reference["selected_sha256"]:
            raise ValueError("selected map changed")
        query_episode(entry, "original")
    return manifest, entries, input_hashes


def evaluate_entry(backbone, initial, entry, selected_checkpoint, final_checkpoint, reference):
    for checkpoint in (selected_checkpoint, final_checkpoint):
        if checkpoint["fingerprint"] != reference["fingerprint"]:
            raise ValueError("checkpoint fingerprint changed")
    if selected_checkpoint["step"] != reference["selected_step"] or final_checkpoint["step"] != reference["steps"]:
        raise ValueError("checkpoint step changed")
    model = source.direct_model(initial, entry["outer"], next(backbone.parameters()).device)
    results, drift = {}, []
    for context_choice in ("original", "expanded"):
        episode = query_episode(entry, context_choice)
        context_result = dict(n_context=episode["n_context"], n_query=episode["n_query"],
                             ordinary16=source.score_episode(backbone, None, episode))
        for checkpoint_choice, checkpoint in (("selected", selected_checkpoint), ("final", final_checkpoint)):
            model.load_state_dict(checkpoint["model"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            before = copy.deepcopy(model.state_dict())
            context_result[checkpoint_choice] = source.score_episode(backbone, model, episode)
            # Frozen inference must not mutate map weights or buffers between contexts.
            for parameter_name, parameter_value in model.state_dict().items():
                torch.testing.assert_close(parameter_value, before[parameter_name], rtol=0, atol=0)
            if context_choice == "expanded":
                drift.append(check_score(context_result[checkpoint_choice], reference[f"{checkpoint_choice}_outer"]))
            elif checkpoint_choice == "selected":
                selection_score = source.score_episode(backbone, model, entry["selection"])
                if not math.isclose(selection_score["nll"], reference["selected_selection_nll"], abs_tol=SCORE_ATOL, rel_tol=SCORE_RTOL):
                    raise ValueError("inner checkpoint-selection score did not replay")
        if context_choice == "expanded":
            drift.append(check_score(context_result["ordinary16"], reference["ordinary16_outer"]))
        results[context_choice] = context_result
    return dict(family=reference["family"], selected_step=reference["selected_step"],
                contexts=results, maximum_full_context_nll_replay_difference=max(drift))


def run(args):
    if args.source_dir.resolve() == args.output_dir.resolve() or args.source_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError("replay output must be separate from source artifacts")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.expected_revision and revision != args.expected_revision:
        raise ValueError("replay revision differs from requested revision")
    manifest, entries, input_hashes = checked_inputs(args.source_dir)
    device = torch.device(args.device)
    backbone, checkpoint_path, checkpoint_hash = pilot.load_frozen(args, device)
    if checkpoint_hash != manifest["backbone"]["sha256"] or Path(checkpoint_path).name != manifest["backbone"]["checkpoint"] or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("frozen source backbone changed")
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    initial = source.new_model("statistics").to(device)
    initial.load_state_dict(torch.load(args.source_dir / "initial_statistics.pt", map_location="cpu", weights_only=True)["model"])
    specification = dict(source_fingerprint=manifest["fingerprint"], source_dir=str(args.source_dir),
        revision=revision, replay_code_sha256=residual.objective.canonical_hash(__file__), inputs=input_hashes,
        checkpoint_sha256=checkpoint_hash, device=args.device,
        protocol="Same outer queries, matched ordinary16 in each context; selected map primary, final diagnostic; no optimization or selection",
        score_atol=SCORE_ATOL, score_rtol=SCORE_RTOL)
    specification["fingerprint"] = previous.digest(specification)
    lock_path = args.output_dir / "manifest.json"
    if lock_path.exists() and previous.read(lock_path) != specification:
        raise ValueError("locked replay specification changed")
    # Lock all saved models and source artifacts before any new query evaluation.
    pilot.json_write(lock_path, specification)
    results = []
    for source_position, entry in enumerate(entries):
        folder = args.source_dir / "direct" / f"source_{source_position:02d}"
        reference = previous.read(folder / "complete.json")
        selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=True)
        final = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
        result = evaluate_entry(backbone, initial, entry, selected, final, reference)
        results.append(result)
        pilot.json_write(args.output_dir / f"source_{source_position:02d}.json", result)
        print(f"Replayed {source_position + 1}/{len(entries)}: {result['family']}; contexts "
              f"{result['contexts']['original']['n_context']} -> {result['contexts']['expanded']['n_context']}", flush=True)
    summaries = {}
    for context_choice in ("original", "expanded"):
        baseline = np.array([result["contexts"][context_choice]["ordinary16"]["nll"] for result in results])
        for checkpoint_choice in ("selected", "final"):
            candidate = np.array([result["contexts"][context_choice][checkpoint_choice]["nll"] for result in results])
            summaries[f"{context_choice}/{checkpoint_choice}"] = dict(
                versus_ordinary16=residual.objective.comparison(candidate, baseline),
                mean_nll_reduction=1 - float(candidate.mean() / baseline.mean()))
    for relative_path, expected_hash in input_hashes.items():
        if pilot.hash_file(args.source_dir / relative_path) != expected_hash:
            raise ValueError("source artifacts changed during replay")
    pilot.json_write(args.output_dir / "complete.json", dict(fingerprint=specification["fingerprint"],
        results=results, summaries=summaries,
        qualification="Eight previously inspected training-source datasets, one split each; development diagnostic. Context size and refitted native preprocessing change together. Outer scores never choose maps."))
    print(f"Replay complete: {args.output_dir / 'complete.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-revision")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
