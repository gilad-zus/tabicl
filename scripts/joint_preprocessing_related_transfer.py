"""Two matched domain continuations, followed by locked cross-domain evaluation."""
from __future__ import annotations

import argparse
import copy
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_ensemble_objective as objective
from scripts import joint_preprocessing_real_meta_bank as bank

core, pilot, previous = objective.core, objective.pilot, objective.previous
GROUPS = ("clinical", "financial")
DEFAULT_GROUPS = Path(__file__).resolve().parents[1] / "docs/experiments/joint_preprocessing_related_transfer_groups_20261006.json"


def specification(args):
    spec = objective.settings(args)
    repo = Path(__file__).resolve().parents[1]
    spec["code_hashes"][Path(__file__).relative_to(repo).as_posix()] = objective.canonical_hash(__file__)
    spec.update(arms=list(GROUPS), objective="ensemble", group_manifest_sha256=pilot.hash_file(args.group_manifest),
                selection="common two-dataset validation", primary="final equal-budget cross-domain comparison",
                historical_status="development diagnostic; datasets previously inspected",
                matched_rows="shared cap from all eight training datasets")
    return spec


def validate_groups(groups, metadata):
    if groups.get("format_version") != 1 or set(groups["groups"]) != set(GROUPS):
        raise ValueError("expected exactly the clinical and financial groups")
    available = {entry["name"]: entry for entries in metadata["families"].values() for entry in entries}
    names, sources, hashes = set(), set(), set()
    for group in GROUPS:
        for panel, count in (("train", 4), ("validation", 1), ("test", 2)):
            selected = groups["groups"][group][panel]
            if len(selected) != count:
                raise ValueError("group sizes must be 4/1/2")
            for name in selected:
                if name not in available:
                    raise ValueError(f"required dataset unavailable: {name}")
                entry = available[name]
                source = bank.normalized_name(entry["source_group"])
                fingerprint = entry["inputs_sha256"]
                if name in names or source in sources or fingerprint in hashes:
                    raise ValueError(f"duplicate/related source crosses group roles: {name}")
                names.add(name)
                sources.add(source)
                hashes.add(fingerprint)
    return available


def prepare(args):
    spec = specification(args)
    old = previous.read(args.bank_dir / "manifest.json")
    if previous.digest({k: v for k, v in old.items() if k != "fingerprint"}) != old["fingerprint"]:
        raise ValueError("old bank fingerprint changed")
    metadata = previous.read(args.bank_dir / "real_manifest.json")
    if pilot.hash_file(args.bank_dir / "real_manifest.json") != old["real_manifest_sha256"]:
        raise ValueError("old real manifest changed")
    groups = previous.read(args.group_manifest)
    available = validate_groups(groups, metadata)
    source = core.transfer.model_lock(args.source_dir)
    if source != old["source_lock"] or source["arms"]["repeated"]["selected_step"] != 5120:
        raise ValueError("expected original synthetic-only checkpoint at step 5120")
    intent = dict(settings=spec, groups=groups, source_lock=source,
                  old_bank_sha256=pilot.hash_file(args.bank_dir / "manifest.json"),
                  old_real_manifest_sha256=pilot.hash_file(args.bank_dir / "real_manifest.json"))
    preparation = args.output_dir / "preparation.json"
    if preparation.exists():
        if previous.read(preparation) != intent:
            raise ValueError("preparation settings/source/groups changed")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("use a new empty result root")
        pilot.json_write(preparation, intent)
    frozen = args.output_dir / "manifest.json"
    if frozen.exists():
        manifest = checked_manifest(args)
        for panel in manifest["banks"]:
            load_bank(args.output_dir, manifest, panel)
        return
    raw = {f["family"]: f for f in core.load_panel(args.bank_dir, old, "real_train")}
    existing = {}
    for panel in ("real_validation", "real_test"):
        for e in core.load_panel(args.bank_dir, old, panel):
            existing.setdefault(e["family"], []).append(e)

    def episodes(name):
        values = ([bank.sample_real_episode(raw[name], 1024, .7, seed) for seed in (0, 1)]
                  if name in raw else existing[name])
        if len(values) != 2 or {e["split_seed"] for e in values} != {0, 1}:
            raise ValueError(f"expected two fixed evaluation episodes: {name}")
        for e in values:
            if set(e["context_indices"].tolist()) & set(e["query_indices"].tolist()):
                raise ValueError("context/query rows overlap")
            core.real._prepared_views(e, 8)
        return values

    panels, selected_metadata = {}, {}
    all_train = [raw[name] for g in GROUPS for name in groups["groups"][g]["train"]]
    cap = min(len(f["labels"]) for f in all_train)
    for group in GROUPS:
        selection = groups["groups"][group]
        families = [raw[name] for name in selection["train"]]
        for f in families:
            for rows in pilot.LENGTHS:
                for fraction in pilot.FRACTIONS:
                    core.real._prepared_views(bank.sample_real_episode(f, min(rows, cap), fraction, 20261006), 8)
        panels[f"{group}_train"] = families
        panels[f"{group}_probe"] = [bank.sample_real_episode(f, min(1024, cap), .7, 20261006) for f in families]
        selected_metadata[group] = {p: [available[n] for n in selection[p]] for p in ("train", "validation", "test")}
    for role in ("validation", "test"):
        panels[f"real_{role}"] = [e for g in GROUPS for name in groups["groups"][g][role] for e in episodes(name)]
    banks = {}
    for panel, values in panels.items():
        path = args.output_dir / "banks" / f"{panel}.pt"
        pilot.atomic_save(path, dict(values=values))
        banks[panel] = dict(path=path.relative_to(args.output_dir).as_posix(), count=len(values), sha256=pilot.hash_file(path))
    manifest = dict(intent, banks=banks, training_row_cap=cap, selected_metadata=selected_metadata,
                    note="No new target weights; old development bank, not fresh confirmation.")
    manifest["fingerprint"] = previous.digest(manifest)
    pilot.json_write(frozen, manifest)
    print(f"Prepared related-transfer banks: {frozen}; common training cap={cap}", flush=True)


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != specification(args) or previous.digest(
            {k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("settings/code/manifest changed")
    if core.transfer.model_lock(args.source_dir) != manifest["source_lock"]:
        raise ValueError("source weights changed")
    return manifest


def load_bank(root, manifest, panel):
    info = manifest["banks"][panel]
    path = root / info["path"]
    if pilot.hash_file(path) != info["sha256"]:
        raise ValueError(f"{panel} bank hash changed")
    values = torch.load(path, map_location="cpu", weights_only=True)["values"]
    if len(values) != info["count"]:
        raise ValueError("bank count changed")
    return values


def setup(args):
    manifest = checked_manifest(args)
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if backbone_hash != manifest["source_lock"]["backbone_sha256"] or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("frozen original backbone required")
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    initial, _ = previous.selected_model(args.source_dir, "repeated", manifest["source_lock"]["source_fingerprint"], device)
    return backbone, initial, manifest, device


def training_batch(families, cap, step):
    rows, fraction = core.shape_for_step(0, step)
    order = np.random.default_rng(431001 + step - 1).permutation(4)
    values = []
    for position, index in enumerate(order):
        seed = int(np.random.SeedSequence([441001, 0, step, position]).generate_state(1)[0])
        e = bank.sample_real_episode(families[int(index)], min(rows, cap), fraction, seed)
        e.update(domain="real", source_seed=seed, task_id=16_000_000_000 + (step - 1) * 4 + position)
        values.append(e)
    return values


def train(args):
    root, group = args.output_dir, args.group
    if (root / "lock.json").exists() or (root / "test_report/started.json").exists():
        raise ValueError("choices are locked; further training forbidden")
    backbone, initial, manifest, device = setup(args)
    fp = manifest["fingerprint"]
    folder = core.run_dir(root, group, 0)
    runfp = core.run_fingerprint(fp, group, 0)
    if (folder / "complete.json").exists():
        if previous.read(folder / "complete.json")["fingerprint"] != runfp:
            raise ValueError("completed run changed")
        return
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError("run exists; pass --resume")
    families = load_bank(root, manifest, f"{group}_train")
    banks = dict(real_probe=load_bank(root, manifest, f"{group}_probe"),
                 real_validation=load_bank(root, manifest, "real_validation"))
    refs = {p: core.references(backbone, initial, es, root,
             f"{group}_probe" if p == "real_probe" else p, fp) for p, es in banks.items()}
    torch.manual_seed(451001)
    np.random.seed(451001)
    random.seed(451001)
    model = copy.deepcopy(initial).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    state_path = folder / "state.pt"
    step = best_step = clipped = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != runfp:
            raise ValueError("resume fingerprint changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        core.capacity.optimizer_to(optimizer, device)
        step, best_step, clipped = state["step"], state["best_step"], state["clipped"]
        best_score, best_model, last = state["best_score"], state["best_model"], state["last"]
        torch.set_rng_state(state["rng"])
        core.restore_rng(state["other_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for name in objective.LOGS:
            core.capacity.trim_csv(folder / name, step)
    else:
        objective.execution_audit(backbone, initial, training_batch(families, manifest["training_row_cap"], 1)[0], folder, device)
        last = objective.evaluate(backbone, model, banks, refs, folder, 0, device)
        best_score, best_model = last["real_validation"]["validation_score"], pilot.state_cpu(model)
    pilot.json_write(folder / "config.json", dict(fingerprint=runfp, experiment_fingerprint=fp,
        group=group, continuation_seed=0, source_step=5120, settings=manifest["settings"],
        revision=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()))

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=runfp, step=step, best_step=best_step, clipped=clipped,
            best_score=best_score, best_model=best_model, last=last, model=pilot.state_cpu(model),
            optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), other_rng=core.rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))

    save()
    stop = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    if stop < step:
        raise ValueError("stop precedes durable state")
    for step in range(step + 1, stop + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.train()
        backbone.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, e in enumerate(training_batch(families, manifest["training_row_cap"], step)):
            losses.append(float(objective.ensemble_backward(backbone, model, e)))
            pilot.csv_append(folder / "presentations.csv", dict(step=step, position=position, domain=group,
                family=e["family"], source_seed=e["source_seed"], task_id=e["task_id"],
                n_context=e["n_context"], n_query=e["n_query"], n_features=e["x_context"].shape[-1], n_classes=e["n_classes"]))
        norms = core.diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not math.isfinite(norm) or not all(math.isfinite(v) for v in norms.values()):
            raise FloatingPointError("nonfinite gradient")
        clipped += int(norm > 1)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        pilot.csv_append(folder / "training.csv", dict(step=step, seconds=time.perf_counter() - started,
            objective=float(np.mean(losses)), lr=args.lr, real_episodes=4, preclip_gradient_norm=norm,
            clip_factor=min(1., 1 / (norm + 1e-6)), clipped_fraction=clipped / step, **norms))
        evaluated = step % args.evaluate_every == 0 or step == args.steps
        if evaluated:
            last = objective.evaluate(backbone, model, banks, refs, folder, step, device)
            score = last["real_validation"]["validation_score"]
            if score < best_score:
                best_score, best_step, best_model = score, step, pilot.state_cpu(model)
        if evaluated or step % args.save_every == 0 or step == stop:
            save()
            print(f"group={group} seed=0 step={step}/{args.steps} CE={np.mean(losses):.6f} gradient={norm:.4f} "
                  f"clipped={clipped / step:.3f} best={best_score:.6f}@{best_step}", flush=True)
    if step != args.steps:
        return
    for name, state, saved_step in (("selected", best_model, best_step), ("final", pilot.state_cpu(model), step)):
        pilot.atomic_save(folder / f"{name}.pt", dict(fingerprint=runfp, model=state, step=saved_step))
    pilot.json_write(folder / "complete.json", dict(fingerprint=runfp, experiment_fingerprint=fp,
        group=group, steps=step, selected_step=best_step, selected_score=best_score, final=last,
        selected_sha256=pilot.hash_file(folder / "selected.pt"), final_sha256=pilot.hash_file(folder / "final.pt")))


def lock(args):
    manifest = checked_manifest(args)
    runs = {}
    for group in GROUPS:
        folder = core.run_dir(args.output_dir, group, 0)
        run = previous.read(folder / "complete.json")
        if run["fingerprint"] != core.run_fingerprint(manifest["fingerprint"], group, 0) or run["steps"] != args.steps:
            raise ValueError("both full-budget runs required before test")
        for choice in ("selected", "final"):
            if pilot.hash_file(folder / f"{choice}.pt") != run[f"{choice}_sha256"]:
                raise ValueError("completed checkpoint changed")
        runs[group] = run
    result = dict(fingerprint=manifest["fingerprint"], runs=runs, primary="final", seed=0)
    path = args.output_dir / "lock.json"
    if path.exists() and previous.read(path) != result:
        raise ValueError("locked choices changed")
    pilot.json_write(path, result)
    return result


def grouped_scores(rows, key, names):
    values = []
    for name in names:
        episodes = [r for r in rows if r["family"] == name]
        if len(episodes) != 2 or {int(r["split_seed"]) for r in episodes} != {0, 1}:
            raise ValueError("incomplete/duplicated target episodes")
        values.append(np.mean([float(r[key]) for r in episodes]))
    return np.asarray(values)


def test(args):
    locked = lock(args)
    report_dir = args.output_dir / "test_report"
    started = dict(lock_sha256=pilot.hash_file(args.output_dir / "lock.json"))
    if (report_dir / "started.json").exists() and previous.read(report_dir / "started.json") != started:
        raise ValueError("test lock changed")
    pilot.json_write(report_dir / "started.json", started)
    backbone, initial, manifest, device = setup(args)
    episodes = load_bank(args.output_dir, manifest, "real_test")
    refs = core.references(backbone, initial, episodes, args.output_dir, "real_test", manifest["fingerprint"])
    rows_by_choice = {}
    for group in GROUPS:
        for choice in ("final", "selected"):
            folder = report_dir / f"{group}_{choice}"
            done = folder / "complete.json"
            if not done.exists():
                # Restart a partial evaluation from its beginning; inference has no training state.
                for name in ("evaluation.csv", "evaluation_episodes.csv"):
                    if (folder / name).exists():
                        (folder / name).unlink()
                saved = torch.load(core.run_dir(args.output_dir, group, 0) / f"{choice}.pt", map_location=device, weights_only=True)
                if saved["fingerprint"] != locked["runs"][group]["fingerprint"]:
                    raise ValueError("test checkpoint fingerprint changed")
                model = copy.deepcopy(initial)
                model.load_state_dict(saved["model"])
                model.requires_grad_(False)
                summary = core.evaluate(backbone, model, {"real_test": episodes}, {"real_test": refs}, folder, saved["step"], device)
                pilot.json_write(done, dict(lock=started, summary=summary,
                    episodes_sha256=pilot.hash_file(folder / "evaluation_episodes.csv")))
            complete = previous.read(done)
            if complete["lock"] != started or complete["episodes_sha256"] != pilot.hash_file(folder / "evaluation_episodes.csv"):
                raise ValueError("completed test evaluation changed")
            rows_by_choice[(group, choice)] = core.capacity.read_rows(folder / "evaluation_episodes.csv")
    summaries = {}
    for choice in ("final", "selected"):
        related_all, cross_all = [], []
        for target_group in GROUPS:
            other = next(g for g in GROUPS if g != target_group)
            names = manifest["groups"]["groups"][target_group]["test"]
            related = grouped_scores(rows_by_choice[(target_group, choice)], "blend_nll", names)
            cross = grouped_scores(rows_by_choice[(other, choice)], "blend_nll", names)
            ordinary = grouped_scores(rows_by_choice[(target_group, choice)], "ordinary16_nll", names)
            initial_blend = grouped_scores(rows_by_choice[(target_group, choice)], "initial_blend_nll", names)
            learned = grouped_scores(rows_by_choice[(target_group, choice)], "learned_nll", names)
            ordinary8 = grouped_scores(rows_by_choice[(target_group, choice)], "ordinary8_nll", names)
            baseline_other = grouped_scores(rows_by_choice[(other, choice)], "ordinary16_nll", names)
            np.testing.assert_allclose(ordinary, baseline_other, rtol=0, atol=0)
            summaries[f"{choice}/{target_group}"] = dict(names=names,
                related_versus_cross=objective.comparison(related, cross),
                related_versus_ordinary16=objective.comparison(related, ordinary),
                cross_versus_ordinary16=objective.comparison(cross, ordinary),
                related_versus_initial_blend=objective.comparison(related, initial_blend),
                cross_versus_initial_blend=objective.comparison(cross, initial_blend),
                related_learned8_versus_ordinary8=objective.comparison(learned, ordinary8),
                mean_related_nll=float(related.mean()), mean_cross_nll=float(cross.mean()), mean_ordinary16_nll=float(ordinary.mean()),
                per_dataset=[dict(name=n, related_nll=float(a), cross_nll=float(b), ordinary16_nll=float(c))
                             for n, a, b, c in zip(names, related, cross, ordinary, strict=True)])
            related_all.extend(related)
            cross_all.extend(cross)
        summaries[f"{choice}/pooled_related_versus_cross"] = objective.comparison(np.array(related_all), np.array(cross_all))
    result = dict(fingerprint=manifest["fingerprint"], lock=started, summaries=summaries,
                  note="Four previously inspected target datasets; descriptive development diagnostic, not broad transfer proof. Final is primary, selected supplementary.")
    path = args.output_dir / "complete.json"
    if path.exists() and previous.read(path) != result:
        raise ValueError("completed report changed")
    pilot.json_write(path, result)
    print(f"Related transfer complete: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "lock", "test", "pipeline"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--group", choices=GROUPS, default="clinical")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--continuation-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--evaluate-every", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.command != "pipeline":
        globals()[args.command](args)
        return
    prepare(args)
    if (args.output_dir / "complete.json").exists():
        test(args)
        return
    if not (args.output_dir / "lock.json").exists():
        for group in GROUPS:
            args.group, args.resume = group, True
            train(args)
            if args.max_steps is not None and args.max_steps < args.steps:
                return
    test(args)


if __name__ == "__main__":
    main()
