"""One-seed, objective-only pilot; never opens the previously inspected test banks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts import joint_preprocessing_real_meta_continuation as core
from tabicl._model.inference_config import InferenceConfig
from tabicl._sklearn import preprocessing

pilot, previous, real = core.pilot, core.previous, core.real
ARMS = ("single", "ensemble")
PANELS = ("real_probe", "real_validation")
LOGS = (*core.LOGS, "learning.csv")


def canonical_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def settings(args):
    if min(args.steps, args.evaluate_every, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid update budget or learning rate")
    if args.continuation_seed != 0:
        raise ValueError("this pilot authorizes seed zero only")
    modules = (core, core.pilot, core.real, core.previous, core.diagnostic, core.synthetic,
               core.transfer, core.capacity, preprocessing)
    paths = [Path(__file__), *(Path(m.__file__) for m in modules),
             Path(inspect.getfile(core.JointPreprocessor)),
             Path(core.__file__).with_name("joint_preprocessing_real_meta_bank.py")]
    # Pin model internals too. Normalize line endings for Windows/Linux parity.
    package = Path(inspect.getfile(core.JointPreprocessor)).parents[1]
    for directory in ("_model", "_hyperspline", "_sklearn"):
        paths += sorted((package / directory).glob("*.py"))
    repo = Path(__file__).resolve().parents[1]
    return dict(format_version=1, arms=list(ARMS), continuation_seeds=[0],
                steps=args.steps, evaluate_every=args.evaluate_every, save_every=args.save_every,
                tasks_per_update=4, lr=args.lr, weight_decay=1e-4, betas=[.9, .999],
                eps=1e-8, gradient_clip=1., alpha=.5, temperature=.9,
                lengths=list(pilot.LENGTHS), fractions=list(pilot.FRACTIONS),
                panels=list(PANELS), optimizer_resets="step zero only",
                training_precision="float32; inherited attention backend",
                evaluation_precision="unchanged default inference managers (CUDA AMP)",
                code_hashes={p.relative_to(repo).as_posix(): canonical_hash(p) for p in paths})


def prepare(args):
    bank = previous.read(args.bank_dir / "manifest.json")
    if previous.digest({k: v for k, v in bank.items() if k != "fingerprint"}) != bank["fingerprint"]:
        raise ValueError("source bank manifest fingerprint changed")
    source = core.transfer.model_lock(args.source_dir)
    if source != bank["source_lock"] or source["arms"]["repeated"]["selected_step"] != 5120:
        raise ValueError("starting checkpoint differs from the prior continuation")
    # Reuse the train/validation protocol, without regenerating or deserializing tests.
    for panel in ("real_train", *PANELS):
        info = bank["banks"][panel]
        if pilot.hash_file(args.bank_dir / info["path"]) != info["sha256"]:
            raise ValueError(f"{panel} bank hash changed")
    manifest = dict(settings=settings(args), source_lock=source,
                    bank_manifest_sha256=pilot.hash_file(args.bank_dir / "manifest.json"),
                    bank_fingerprint=bank["fingerprint"],
                    banks={p: bank["banks"][p] for p in ("real_train", *PANELS)},
                    episode_settings=bank["settings"],
                    note="Development pilot on reused validation families; no final-test evidence.")
    manifest["fingerprint"] = previous.digest(manifest)
    path = args.output_dir / "manifest.json"
    if path.exists():
        if previous.read(path) != manifest:
            raise ValueError("pilot settings, code, source weights or banks changed")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use a new empty pilot root")
    pilot.json_write(path, manifest)
    print(f"Prepared one-seed objective pilot: {args.output_dir}", flush=True)


def setup(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != settings(args) or previous.digest(
            {k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("pilot settings/code/manifest changed")
    if pilot.hash_file(args.bank_dir / "manifest.json") != manifest["bank_manifest_sha256"]:
        raise ValueError("bank manifest changed")
    if core.transfer.model_lock(args.source_dir) != manifest["source_lock"]:
        raise ValueError("starting weights changed")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if backbone_hash != manifest["source_lock"]["backbone_sha256"] or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("backbone must match the frozen original")
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    initial, _ = previous.selected_model(args.source_dir, "repeated",
                        manifest["source_lock"]["source_fingerprint"], device)
    return backbone, initial, manifest, manifest["fingerprint"], device


def prepared_episode(e, device):
    generator, members, positions, numerical_keep, keep = real._prepared_views(e, 8)
    xc = e["x_context"][..., keep][..., positions].to(device)
    xq = e["x_query"][..., keep][..., positions].to(device)
    mc = e["context_missing"][None, :, numerical_keep].to(device)
    mq = e["query_missing"][None, :, numerical_keep].to(device)
    yc = e["y_context"].to(device)
    views = []
    for method, (xs, ys) in members.items():
        for index, (features, classes) in enumerate(generator.ensemble_configs_[method]):
            inverse = {int(original): position for position, original in enumerate(features)}
            columns = torch.tensor([inverse[int(original)] for original in positions], device=device)
            views.append(dict(slot=0 if method == "none" else 1, columns=columns, features=list(features),
                x=torch.from_numpy(xs[index:index + 1]).to(device=device, dtype=torch.float32),
                y=torch.from_numpy(ys[index:index + 1]).to(device=device, dtype=torch.float32),
                classes=torch.as_tensor(classes, device=device, dtype=torch.long)))
    if len(views) != 8:
        raise ValueError("expected eight actual ensemble views")
    return (xc, xq, yc, mc, mq), views


def numeric_slots(model, context):
    xc, xq, yc, mc, mq = context
    generated = model.generate(xc, yc, mc)
    return [torch.cat((model.apply(xc, generated, slot, mc),
                       model.apply(xq, generated, slot, mq)), 1) for slot in (0, 1)]


def view_logits(backbone, view, n_classes, numeric=None, inference_config=None):
    x = view["x"] if numeric is None else view["x"].index_copy(-1, view["columns"], numeric)
    backbone.clear_cache()
    kwargs = {} if inference_config is None else dict(inference_config=inference_config,
               feature_shuffles=[view["features"]], return_logits=True)
    raw = backbone(x, view["y"], **kwargs)
    return raw[..., :n_classes][..., view["classes"]]


def ensemble_mean(backbone, views, n_classes, slots=None):
    return torch.stack([view_logits(backbone, v, n_classes,
                         None if slots is None else slots[v["slot"]]) for v in views]).mean(0)


def ensemble_backward(backbone, model, e, scale=.25):
    """Exact deterministic chain rule, with only one backbone activation graph live.

    The detached logit pass computes dCE/d(mean learned logits). Each learned
    view is then replayed to accumulate dCE/d(numerical slots), followed by one
    backward through the small hypernetwork. No ordinary-branch gradient exists.
    """
    device = next(backbone.parameters()).device
    context, views = prepared_episode(e, device)
    slots = numeric_slots(model, context)
    with torch.no_grad():
        ordinary = ensemble_mean(backbone, views, e["n_classes"])
        learned = ensemble_mean(backbone, views, e["n_classes"], [s.detach() for s in slots])
    leaf = learned.detach().requires_grad_(True)
    loss = F.cross_entropy((.5 * (ordinary + leaf) / .9).flatten(0, 1),
                           e["y_query"].flatten().to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite ensemble CE")
    upstream, = torch.autograd.grad(loss, leaf)
    slot_grads = [torch.zeros_like(s) for s in slots]
    for view in views:
        numeric = slots[view["slot"]].detach().requires_grad_(True)
        logits = view_logits(backbone, view, e["n_classes"], numeric)
        grad, = torch.autograd.grad(logits, numeric, grad_outputs=upstream / len(views))
        slot_grads[view["slot"]].add_(grad)
    torch.autograd.backward(slots, [g * scale for g in slot_grads])
    return loss.detach()


def execution_audit(backbone, initial, e, folder, device):
    """Fail early if the differentiable path differs materially from deployment."""
    model = copy.deepcopy(initial).train().requires_grad_(True)
    backbone.train()
    context, views = prepared_episode(e, device)
    with torch.no_grad():
        slots = numeric_slots(model, context)
        learned = ensemble_mean(backbone, views, e["n_classes"], slots)
        ordinary = ensemble_mean(backbone, views, e["n_classes"])
    # Existing deployment managers enable CUDA AMP; compare mathematical paths
    # at matching precision, and separately disclose the usual AMP discrepancy.
    config = InferenceConfig(COL_CONFIG=dict(use_amp=False), ROW_CONFIG=dict(use_amp=False),
                             ICL_CONFIG=dict(use_amp=False))
    with torch.no_grad(), pilot.frozen_inference(backbone):
        deployed_learned = torch.stack([view_logits(backbone, v, e["n_classes"], slots[v["slot"]], config)
                                        for v in views]).mean(0)
        deployed_ordinary = torch.stack([view_logits(backbone, v, e["n_classes"], inference_config=config)
                                         for v in views]).mean(0)
    for actual, expected in ((learned, deployed_learned), (ordinary, deployed_ordinary)):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    expected_ce = F.cross_entropy((.5 * (deployed_learned + deployed_ordinary) / .9).flatten(0, 1),
                                  e["y_query"].flatten().to(device))
    default_learned, _ = core.panel_logits(backbone, model, e, 8)
    default_ordinary, _ = core.panel_logits(backbone, "ordinary", e, 8)
    default_ce = F.cross_entropy((.5 * (default_learned + default_ordinary) / .9).float().flatten(0, 1),
                                e["y_query"].flatten().to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    value = ensemble_backward(backbone, model, e, 1.)
    if device.type == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(value, expected_ce, rtol=2e-4, atol=2e-4)
    norms = core.diagnostic.gradient_norms(model)
    if not all(math.isfinite(v) for v in norms.values()) or not any(v > 0 for v in norms.values()):
        raise FloatingPointError("ensemble objective has no finite nonzero learning gradient")
    if any(p.grad is not None for p in backbone.parameters()):
        raise AssertionError("frozen backbone received parameter gradients")
    report = dict(task_id=e["task_id"], learned_max_absolute_difference=float((learned - deployed_learned).abs().max()),
                  ordinary_max_absolute_difference=float((ordinary - deployed_ordinary).abs().max()),
                  ensemble_ce=float(value), deployed_ce=float(expected_ce),
                  default_amp_deployed_ce=float(default_ce),
                  default_amp_learned_max_difference=float((learned - default_learned).abs().max()),
                  default_amp_ordinary_max_difference=float((ordinary - default_ordinary).abs().max()),
                  seconds=time.perf_counter() - started, gradient_norms=norms,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
    pilot.json_write(folder / "execution_audit.json", report)
    print(f"Ensemble execution audit: {json.dumps(report)}", flush=True)


def evaluate(backbone, model, banks, refs, folder, step, device):
    summaries = core.evaluate(backbone, model, banks, refs, folder, step, device)
    rows = core.capacity.read_rows(folder / "evaluation_episodes.csv")
    for panel in PANELS:
        current = [r for r in rows if r["panel"] == panel and int(r["step"]) == step]
        families = sorted({r["family"] for r in current})
        def grouped(key):
            return np.array([np.mean([float(r[key]) for r in current if r["family"] == family]) for family in families])
        blend, initial = grouped("blend_nll"), grouped("initial_blend_nll")
        raw, baseline = grouped("single_nll"), grouped("ordinary16_nll")
        row = dict(step=step, panel=panel, mean_blend_nll=float(blend.mean()),
                   initial_mean_blend_nll=float(initial.mean()),
                   blend_mean_reduction_from_initial=1 - float(blend.mean() / initial.mean()),
                   blend_geometric_gain_from_initial=1 - float(np.exp(np.log((blend + 1e-4) / (initial + 1e-4)).mean())),
                   median_family_gain_from_initial=float(np.median(1 - (blend + 1e-4) / (initial + 1e-4))),
                   wins_from_initial=int((blend < initial - 1e-6).sum()),
                   losses_from_initial=int((blend > initial + 1e-6).sum()),
                   mean_single_nll=float(raw.mean()), ordinary16_mean_nll=float(baseline.mean()),
                   mean_nll_delta_vs_ordinary16=float((blend - baseline).mean()))
        pilot.csv_append(folder / "learning.csv", row)
        summaries[panel].update(row)
        print(f"step={step} {panel} blend learning from start={100 * row['blend_mean_reduction_from_initial']:.3f}% "
              f"W/L={row['wins_from_initial']}/{row['losses_from_initial']}", flush=True)
    return summaries


def train(args):
    root = args.output_dir
    if (root / "complete.json").exists():
        raise ValueError("pilot choices are locked; further training forbidden")
    backbone, initial, manifest, fp, device = setup(args)
    if args.arm not in ARMS or args.continuation_seed != 0:
        raise ValueError("unknown objective or seed; only seed zero is authorized")
    folder = core.run_dir(root, args.arm, 0)
    runfp = core.run_fingerprint(fp, args.arm, 0)
    if (folder / "complete.json").exists():
        if previous.read(folder / "complete.json")["fingerprint"] != runfp:
            raise ValueError("completed run fingerprint changed")
        return
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError("run exists; pass --resume")
    families = core.load_panel(args.bank_dir, manifest, "real_train")
    banks = {p: core.load_panel(args.bank_dir, manifest, p) for p in PANELS}
    # Shared, checked baseline cache avoids repeating ordinary inference per arm.
    refs = {p: core.references(backbone, initial, episodes, root, p, fp) for p, episodes in banks.items()}
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
        for name in LOGS:
            core.capacity.trim_csv(folder / name, step)
    else:
        if args.arm == "ensemble":
            execution_audit(backbone, initial, core.training_batch("real", 0, 1, families,
                            manifest["episode_settings"])[0], folder, device)
        last = evaluate(backbone, model, banks, refs, folder, 0, device)
        best_score, best_model = last["real_validation"]["validation_score"], pilot.state_cpu(model)
    pilot.json_write(folder / "config.json", dict(experiment_fingerprint=fp, fingerprint=runfp,
        arm=args.arm, continuation_seed=0, source_step=5120, settings=manifest["settings"],
        source_checkpoint_sha256=manifest["source_lock"]["arms"]["repeated"]["checkpoint_sha256"],
        revision=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()))

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=runfp, step=step, best_step=best_step, clipped=clipped,
            best_score=best_score, best_model=best_model, last=last, model=pilot.state_cpu(model),
            optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), other_rng=core.rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))

    save()
    stop = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    if stop < step:
        raise ValueError("requested stop precedes durable state")
    for step in range(step + 1, stop + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        batch = core.training_batch("real", 0, step, families, manifest["episode_settings"])
        model.train()
        backbone.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, e in enumerate(batch):
            if args.arm == "ensemble":
                loss = ensemble_backward(backbone, model, e)
            else:
                loss = core.surrogate_nll(backbone, model, e)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite single-view CE")
                (loss / 4).backward()
            losses.append(float(loss.detach()))
            pilot.csv_append(folder / "presentations.csv", dict(step=step, position=position, domain="real",
                family=e["family"], source_seed=e["source_seed"], task_id=e["task_id"],
                n_context=e["x_context"].shape[1], n_query=e["x_query"].shape[1],
                n_features=e["x_context"].shape[-1], n_classes=e["n_classes"]))
        norms = core.diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not math.isfinite(norm) or not all(math.isfinite(x) for x in norms.values()):
            raise FloatingPointError("nonfinite gradient")
        clipped += int(norm > 1)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        pilot.csv_append(folder / "training.csv", dict(step=step, seconds=time.perf_counter() - started,
            objective=float(np.mean(losses)), real_episodes=4, lr=args.lr,
            preclip_gradient_norm=norm, clip_factor=min(1., 1 / (norm + 1e-6)), clipped_fraction=clipped / step,
            **norms))
        evaluated = step % args.evaluate_every == 0 or step == args.steps
        if evaluated:
            last = evaluate(backbone, model, banks, refs, folder, step, device)
            score = last["real_validation"]["validation_score"]
            if score < best_score:
                best_score, best_step, best_model = score, step, pilot.state_cpu(model)
        if evaluated or step % args.save_every == 0 or step == stop:
            save()
            print(f"arm={args.arm} seed=0 step={step}/{args.steps} CE={np.mean(losses):.6f} "
                  f"gradient={norm:.4f} clipped={clipped / step:.3f} best={best_score:.6f}@{best_step}", flush=True)
    if step != args.steps:
        return
    for name, values, saved_step in (("selected", best_model, best_step), ("final", pilot.state_cpu(model), step)):
        pilot.atomic_save(folder / f"{name}.pt", dict(fingerprint=runfp, model=values, step=saved_step))
    pilot.json_write(folder / "complete.json", dict(experiment_fingerprint=fp, fingerprint=runfp,
        arm=args.arm, continuation_seed=0, steps=step, selected_step=best_step, selected_score=best_score,
        selected_sha256=pilot.hash_file(folder / "selected.pt"), final_sha256=pilot.hash_file(folder / "final.pt"),
        final=last, recorded_training_seconds=sum(float(r["seconds"]) for r in core.capacity.read_rows(folder / "training.csv")),
        recorded_evaluation_seconds=sum(float(r["seconds"]) for r in core.capacity.read_rows(folder / "evaluation.csv"))))


def report(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    runs, panels = {}, {}
    for arm in ARMS:
        folder = core.run_dir(args.output_dir, arm, 0)
        complete = previous.read(folder / "complete.json")
        if complete["fingerprint"] != core.run_fingerprint(manifest["fingerprint"], arm, 0):
            raise ValueError("completion fingerprint changed")
        for name in ("selected", "final"):
            if pilot.hash_file(folder / f"{name}.pt") != complete[f"{name}_sha256"]:
                raise ValueError("completed checkpoint changed")
        runs[arm] = complete
        rows = core.capacity.read_rows(folder / "evaluation_episodes.csv")
        for choice, step in (("initial", 0), ("selected", complete["selected_step"]), ("final", args.steps)):
            for panel in PANELS:
                values = [r for r in rows if int(r["step"]) == step and r["panel"] == panel]
                families = sorted({r["family"] for r in values})
                if not values or len(values) != manifest["banks"][panel]["count"]:
                    raise ValueError("incomplete or duplicated pilot evaluation rows")
                def grouped(key):
                    return np.array([np.mean([float(r[key]) for r in values if r["family"] == f]) for f in families])
                panels[(arm, choice, panel)] = (families, grouped("blend_nll"), grouped("ordinary16_nll"),
                                                grouped("single_nll"))
    summaries = {}
    for (arm, choice, panel), (families, blend, baseline, single) in panels.items():
        start = panels[(arm, "initial", panel)]
        summaries[f"{arm}/{choice}/{panel}"] = dict(families=families,
            versus_ordinary16=comparison(blend, baseline),
            versus_initial=comparison(blend, start[1]),
            single_mean_nll=float(single.mean()), blend_mean_nll=float(blend.mean()),
            single_mean_reduction_from_initial=1 - float(single.mean() / start[3].mean()))
    paired = {}
    for choice in ("selected", "final"):
        for panel in PANELS:
            a, b = panels[("ensemble", choice, panel)], panels[("single", choice, panel)]
            if a[0] != b[0]:
                raise ValueError("arm family ordering differs")
            paired[f"{choice}/{panel}"] = comparison(a[1], b[1])
    result = dict(fingerprint=manifest["fingerprint"], continuation_seeds=[0], runs=runs,
                  summaries=summaries, ensemble_versus_single=paired,
                  note="Reused source/validation diagnostic. No test bank opened; intervals describe family resampling, not seed variability.")
    path = args.output_dir / "complete.json"
    if path.exists() and previous.read(path) != result:
        raise ValueError("locked pilot report changed")
    pilot.json_write(path, result)
    print(f"Pilot complete: {path}", flush=True)


def comparison(candidate, reference):
    result = pilot.comparison_summary(candidate, reference, bootstrap_seed=461001, bootstrap_samples=10000)
    gain = 1 - (candidate + 1e-4) / (reference + 1e-4)
    delta = candidate - reference
    result.update(material_wins=int((delta < -1e-6).sum()), material_losses=int((delta > 1e-6).sum()),
                  gain_percentiles={str(q): float(np.percentile(gain, q)) for q in (0, 10, 25, 50, 75, 90, 100)},
                  harms_over_1pct=int((gain < -.01).sum()), harms_over_5pct_with_floor=int((gain < -.05).sum()),
                  harms_over_10pct=int((gain < -.1).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "report", "pipeline"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=ARMS, default="single")
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
        report(args)
        return
    # Exercise the new objective first; a failed GPU audit must not spend the
    # full control budget before surfacing.
    for arm in ("ensemble", "single"):
        args.arm, args.resume = arm, True
        train(args)
        if args.max_steps is not None and args.max_steps < args.steps:
            return
    report(args)


if __name__ == "__main__":
    main()
