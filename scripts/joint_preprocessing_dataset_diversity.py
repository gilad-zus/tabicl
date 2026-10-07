"""Fresh real-data learning: nested 40/160 source banks, seed zero, equal updates."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import sklearn

from scripts import joint_preprocessing_dataset_diversity_bank as bank
from scripts import joint_preprocessing_dataset_diversity_catalog as catalog
from scripts import joint_preprocessing_ensemble_objective as objective

core, pilot, previous = objective.core, objective.pilot, objective.previous
ARMS = ("small", "large")
PANELS = ("real_probe", "large_only_probe", "real_validation")
DEFAULT_CANDIDATES = Path(__file__).resolve().parents[1] / "docs/experiments/joint_preprocessing_dataset_diversity_candidates_20261007.json"


def settings(args):
    spec = objective.settings(args)
    repo = Path(__file__).resolve().parents[1]
    try:
        candidate_key = args.candidate_manifest.resolve().relative_to(repo).as_posix()
        spec["code_hashes"][candidate_key] = objective.canonical_hash(args.candidate_manifest)
    except ValueError:
        if getattr(args, "expected_revision", None):
            raise ValueError("pinned experiment candidates must be committed inside the repository")
    for m in (bank, catalog):
        spec["code_hashes"][Path(m.__file__).relative_to(repo).as_posix()] = objective.canonical_hash(m.__file__)
    spec["code_hashes"][Path(__file__).relative_to(repo).as_posix()] = objective.canonical_hash(__file__)
    spec.update(arms=list(ARMS), model_seed=0, initialization="fresh JointPreprocessor(joint), CPU seed 0",
        panels=list(PANELS), objective="ordinary8+learned8 CE", validation_selection="minimum arithmetic mean of dataset mean NLL; update zero eligible",
        primary="final equal-update comparison; selected checkpoints are secondary deployment candidates",
        paired_rows="same per-step minimum row cap across both scheduled four-source batches",
        candidate_sha256=objective.canonical_hash(args.candidate_manifest),
        expected_revision=getattr(args, "expected_revision", None),
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
        hypothesis="greater dataset diversity improves unseen-dataset transfer at equal updates; per-source visits differ",
        success_rule=dict(minimum_win_fraction=.7, minimum_mean_nll_reduction=.005,
                          positive_median_gain=True, require_adjacent_passing_checkpoints=True))
    return spec


def verify_revision(spec):
    revision = spec["expected_revision"]
    if revision is None:
        return
    current = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if current != revision:
        raise ValueError("Git revision differs from the explicitly pinned experiment")
    import hashlib
    for path, expected in spec["code_hashes"].items():
        content = subprocess.run(["git", "show", f"{revision}:{path}"], check=True, capture_output=True).stdout
        if hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"experiment dependency has uncommitted changes: {path}")


def new_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return core.JointPreprocessor("joint")


def prepare(args):
    spec = settings(args)
    verify_revision(spec)
    intent = dict(settings=spec)
    path = args.output_dir / "preparation.json"
    if path.exists():
        if previous.read(path) != intent:
            raise ValueError("preparation settings/code/candidates changed")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("use a new empty output root")
        pilot.json_write(path, intent)
    banks = bank.prepare(args.output_dir, args.candidate_manifest, args.cache_dir)
    initial = args.output_dir / "initial.pt"
    if not initial.exists():
        pilot.atomic_save(initial, dict(model=pilot.state_cpu(new_model()), model_seed=0))
    saved = torch.load(initial, map_location="cpu", weights_only=True)
    expected = new_model().state_dict()
    if saved["model_seed"] != 0 or set(saved["model"]) != set(expected):
        raise ValueError("fresh initialization differs")
    for k, v in expected.items():
        torch.testing.assert_close(saved["model"][k], v, rtol=0, atol=0)
    manifest = dict(settings=spec, banks=banks["banks"], counts=banks["counts"],
        banks_manifest_sha256=pilot.hash_file(args.output_dir / "banks_manifest.json"),
        initial_sha256=pilot.hash_file(initial), no_test_bank=True)
    manifest["fingerprint"] = previous.digest(manifest)
    path = args.output_dir / "manifest.json"
    if path.exists() and previous.read(path) != manifest:
        raise ValueError("frozen manifest changed")
    pilot.json_write(path, manifest)
    print(f"Prepared fresh dataset-diversity experiment: {path}", flush=True)


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != settings(args) or previous.digest(
            {k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("settings/code/manifest changed")
    for path, key in (("banks_manifest.json", "banks_manifest_sha256"), ("initial.pt", "initial_sha256")):
        if pilot.hash_file(args.output_dir / path) != manifest[key]:
            raise ValueError("bank selection or initial weights changed")
    verify_revision(manifest["settings"])
    return manifest


def setup(args):
    manifest = checked_manifest(args)
    device = torch.device(args.device)
    backbone, path, backbone_hash = pilot.load_frozen(args, device)
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("TabICL weights must be frozen")
    lock_path = args.output_dir / "backbone_lock.json"
    lock = dict(sha256=backbone_hash, checkpoint=Path(path).name)
    if lock_path.exists() and previous.read(lock_path) != lock:
        raise ValueError("frozen backbone changed")
    pilot.json_write(lock_path, lock)
    backbone.train()
    for m in backbone.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.
    initial = new_model().to(device)
    initial.load_state_dict(torch.load(args.output_dir / "initial.pt", map_location="cpu", weights_only=True)["model"])
    return backbone, initial, manifest, device


def scheduled_sources(families, step):
    def pick(draw):
        epoch, position = divmod(draw, len(families))
        order = np.random.default_rng(431001 + epoch).permutation(len(families))
        return families[int(order[position])]
    return [pick((step - 1) * 4 + i) for i in range(4)]


def training_batch(arm, step, source_banks):
    if arm not in ARMS:
        raise ValueError("unknown training arm")
    rows, fraction = core.shape_for_step(0, step)
    assigned = {k: scheduled_sources(source_banks[k], step) for k in ARMS}
    cap = min(rows, *(len(f["labels"]) for fs in assigned.values() for f in fs))
    values = []
    for i, f in enumerate(assigned[arm]):
        seed = int(np.random.SeedSequence([20261007, 0, step, i]).generate_state(1)[0])
        e = bank.bank.sample_real_episode(f, cap, fraction, seed)
        e.update(domain="real", source_seed=seed, task_id=17_000_000_000 + (step - 1) * 4 + i,
                 scheduled_rows=rows, paired_row_cap=cap)
        values.append(e)
    return values


def grouped(rows, key, expected_count=None):
    names = sorted({r["family"] for r in rows})
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError("incomplete evaluation")
    ids = [(r["family"], int(r["split_seed"])) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicated evaluation rows")
    return names, np.array([np.mean([float(r[key]) for r in rows if r["family"] == name]) for name in names])


def evaluate(backbone, model, panels, refs, folder, step, device):
    result = core.evaluate(backbone, model, panels, refs, folder, step, device)
    all_rows = core.capacity.read_rows(folder / "evaluation_episodes.csv")
    for panel in panels:
        rows = [r for r in all_rows if int(r["step"]) == step and r["panel"] == panel]
        _, blend = grouped(rows, "blend_nll", len(panels[panel]))
        _, initial = grouped(rows, "initial_blend_nll")
        _, baseline = grouped(rows, "ordinary16_nll")
        metrics = objective.comparison(blend, baseline)
        row = dict(step=step, panel=panel, mean_blend_nll=float(blend.mean()),
            mean_initial_nll=float(initial.mean()), mean_ordinary16_nll=float(baseline.mean()),
            mean_reduction_from_initial=1 - float(blend.mean() / initial.mean()),
            mean_reduction_vs_ordinary16=1 - float(blend.mean() / baseline.mean()),
            wins_vs_ordinary16=metrics["material_wins"], losses_vs_ordinary16=metrics["material_losses"],
            median_gain_vs_ordinary16=float(np.median(1 - (blend + 1e-4) / (baseline + 1e-4))))
        pilot.csv_append(folder / "learning.csv", row)
        result[panel].update(row)
        print(f"step={step} {panel}: mean NLL gain vs ordinary16={100 * row['mean_reduction_vs_ordinary16']:.3f}%, "
              f"from initial={100 * row['mean_reduction_from_initial']:.3f}%, "
              f"W/L={row['wins_vs_ordinary16']}/{row['losses_vs_ordinary16']}", flush=True)
    return result


def train(args):
    root, arm = args.output_dir, args.arm
    if (root / "complete.json").exists():
        raise ValueError("completed choices are locked; further training forbidden")
    backbone, initial, manifest, device = setup(args)
    fp = manifest["fingerprint"]
    runfp = core.run_fingerprint(fp, arm, 0)
    folder = core.run_dir(root, arm, 0)
    if (folder / "complete.json").exists():
        if previous.read(folder / "complete.json")["fingerprint"] != runfp:
            raise ValueError("completed run changed")
        return
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError("run exists; pass --resume")
    source_banks = {k: bank.load_bank(root, manifest, f"{k}_train") for k in ARMS}
    panels = {p: bank.load_bank(root, manifest, p) for p in PANELS}
    refs = {p: core.references(backbone, initial, es, root, p, fp) for p, es in panels.items()}
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
        objective.execution_audit(backbone, model, training_batch(arm, 1, source_banks)[0], folder, device)
        last = evaluate(backbone, model, panels, refs, folder, 0, device)
        best_score, best_model = last["real_validation"]["mean_blend_nll"], pilot.state_cpu(model)
    pilot.json_write(folder / "config.json", dict(fingerprint=runfp, experiment_fingerprint=fp,
        arm=arm, model_seed=0, initialization_sha256=manifest["initial_sha256"],
        settings=manifest["settings"], backbone=previous.read(root / "backbone_lock.json"),
        training_sources=len(source_banks[arm]),
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
        for position, e in enumerate(training_batch(arm, step, source_banks)):
            losses.append(float(objective.ensemble_backward(backbone, model, e)))
            pilot.csv_append(folder / "presentations.csv", dict(step=step, position=position, domain=arm,
                family=e["family"], source_seed=e["source_seed"], task_id=e["task_id"],
                n_context=e["n_context"], n_query=e["n_query"], n_features=e["x_context"].shape[-1],
                n_classes=e["n_classes"], scheduled_rows=e["scheduled_rows"], paired_row_cap=e["paired_row_cap"]))
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
            last = evaluate(backbone, model, panels, refs, folder, step, device)
            score = last["real_validation"]["mean_blend_nll"]
            if score < best_score:
                best_score, best_step, best_model = score, step, pilot.state_cpu(model)
        if evaluated or step % args.save_every == 0 or step == stop:
            save()
            print(f"arm={arm} seed=0 step={step}/{args.steps} CE={np.mean(losses):.6f} gradient={norm:.4f} "
                  f"clipped={clipped / step:.3f} best validation mean NLL={best_score:.6f}@{best_step}", flush=True)
    if step != args.steps:
        return
    for name, values, saved_step in (("selected", best_model, best_step), ("final", pilot.state_cpu(model), step)):
        pilot.atomic_save(folder / f"{name}.pt", dict(fingerprint=runfp, model=values, step=saved_step))
    pilot.json_write(folder / "complete.json", dict(fingerprint=runfp, experiment_fingerprint=fp, arm=arm,
        model_seed=0, steps=step, selected_step=best_step, selected_score=best_score, final=last,
        selected_sha256=pilot.hash_file(folder / "selected.pt"), final_sha256=pilot.hash_file(folder / "final.pt"),
        recorded_training_seconds=sum(float(r["seconds"]) for r in core.capacity.read_rows(folder / "training.csv")),
        recorded_evaluation_seconds=sum(float(r["seconds"]) for r in core.capacity.read_rows(folder / "evaluation.csv"))))


def passing(candidate, reference):
    gain = 1 - (candidate + 1e-4) / (reference + 1e-4)
    return bool((candidate < reference - 1e-6).sum() >= math.ceil(.7 * len(candidate))
                and 1 - candidate.mean() / reference.mean() >= .005 and np.median(gain) > 0)


def report(args):
    manifest = checked_manifest(args)
    runs, summaries, scores, decisions = {}, {}, {}, {}
    for arm in ARMS:
        folder = core.run_dir(args.output_dir, arm, 0)
        run = previous.read(folder / "complete.json")
        if run["fingerprint"] != core.run_fingerprint(manifest["fingerprint"], arm, 0) or run["steps"] != args.steps:
            raise ValueError("both full-budget runs required")
        for choice in ("selected", "final"):
            if pilot.hash_file(folder / f"{choice}.pt") != run[f"{choice}_sha256"]:
                raise ValueError("completed checkpoint changed")
        runs[arm] = run
        rows = core.capacity.read_rows(folder / "evaluation_episodes.csv")
        for choice, step in (("initial", 0), ("selected", run["selected_step"]), ("final", args.steps)):
            for panel in PANELS:
                current = [r for r in rows if int(r["step"]) == step and r["panel"] == panel]
                names, blend = grouped(current, "blend_nll", manifest["banks"][panel]["count"])
                _, baseline = grouped(current, "ordinary16_nll")
                _, initial = grouped(current, "initial_blend_nll")
                scores[(arm, choice, panel)] = names, blend
                summaries[f"{arm}/{choice}/{panel}"] = dict(names=names,
                    versus_ordinary16=objective.comparison(blend, baseline),
                    versus_initial=objective.comparison(blend, initial),
                    mean_nll=float(blend.mean()), ordinary16_mean_nll=float(baseline.mean()),
                    mean_nll_reduction=1 - float(blend.mean() / baseline.mean()))
        val = [r for r in rows if r["panel"] == "real_validation"]
        steps = sorted({int(r["step"]) for r in val})
        passes = {}
        for step in steps:
            current = [r for r in val if int(r["step"]) == step]
            _, c = grouped(current, "blend_nll", manifest["banks"]["real_validation"]["count"])
            _, b = grouped(current, "ordinary16_nll")
            passes[step] = passing(c, b)
        selected_index = steps.index(run["selected_step"])
        neighbours = steps[max(0, selected_index - 1):selected_index] + steps[selected_index + 1:selected_index + 2]
        decisions[arm] = dict(passing_steps=[s for s in steps if passes[s]],
            selected_checkpoint_passes=passes[run["selected_step"]],
            worth_fresh_confirmation=run["selected_step"] > 0 and passes[run["selected_step"]]
                and any(passes[s] for s in neighbours))
    paired = {}
    for choice in ("selected", "final"):
        for panel in PANELS:
            a, b = scores[("large", choice, panel)], scores[("small", choice, panel)]
            if a[0] != b[0]:
                raise ValueError("arm evaluation ordering differs")
            paired[f"{choice}/{panel}"] = objective.comparison(a[1], b[1])
    for panel in PANELS:
        a, b = scores[("large", "initial", panel)], scores[("small", "initial", panel)]
        np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-8)
    result = dict(fingerprint=manifest["fingerprint"], runs=runs, summaries=summaries,
        large_versus_small=paired, decisions=decisions,
        qualification="One nested source-bank draw and one seed; equal updates give different per-dataset exposure and may differ in runtime. Validation is development evidence. No confirmation test bank opened.")
    path = args.output_dir / "complete.json"
    if path.exists() and previous.read(path) != result:
        raise ValueError("locked report changed")
    pilot.json_write(path, result)
    print(f"Dataset-diversity experiment complete: {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "train", "report", "pipeline"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--cache-dir", type=Path, default=Path("results/pmlb_cache"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--arm", choices=ARMS, default="small")
    p.add_argument("--continuation-seed", type=int, choices=(0,), default=0)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--evaluate-every", type=int, default=512)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=.0003)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--expected-revision", help="Require this Git HEAD and all hashed dependencies to match its committed contents")
    args = p.parse_args()
    if args.command != "pipeline":
        globals()[args.command](args)
        return
    prepare(args)
    if (args.output_dir / "complete.json").exists():
        report(args)
        return
    for arm in ARMS:
        args.arm, args.resume = arm, True
        train(args)
        if args.max_steps is not None and args.max_steps < args.steps:
            return
    report(args)


if __name__ == "__main__":
    main()
