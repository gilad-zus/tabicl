"""Seed-zero raw versus frozen-TabICL conditioning on 160 frozen real sources."""
from __future__ import annotations

import argparse
import copy
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

from scripts import joint_preprocessing_dataset_diversity as diversity
from tabicl._hyperspline.residual_preprocessing import ResidualPreprocessor, frozen_context_features
from tabicl._model.inference_config import InferenceConfig

bank, objective = diversity.bank, diversity.objective
core, pilot, previous = diversity.core, diversity.pilot, diversity.previous
ARMS = ("raw", "backbone")
PANELS = diversity.PANELS
DEFAULT_CANDIDATES = diversity.DEFAULT_CANDIDATES


def settings(args):
    spec = diversity.settings(args)
    repo = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), Path(inspect.getfile(ResidualPreprocessor)),
                 Path(pilot.load_backbone.__code__.co_filename)):
        spec["code_hashes"][path.relative_to(repo).as_posix()] = objective.canonical_hash(path)
    # These regression/checkpoint-export modules have unrelated local edits and
    # are never invoked by this classifier experiment. Do not stage those edits.
    for key in ("src/tabicl/_model/quantile_dist.py", "src/tabicl/_hyperspline/checkpoint.py"):
        del spec["code_hashes"][key]
    spec.update(arms=list(ARMS), initialization="fresh paired transformation heads, CPU seed zero",
        objective="CE of exactly 16 native views, eight unchanged and eight residual-adapted",
        alpha=None, training_sources=160, paired_rows="identical old large-arm sampling schedule in both arms",
        numerical_guard="native scaler unchanged; mask residuals only at originally missing numerical cells",
        hypothesis="richer frozen full-context representations improve generated zero-shot residuals",
        conditioning=dict(raw="original numerical cells and labels",
            backbone="original numerical cells, aligned native column/group embeddings, CLS-averaged full-feature row embeddings and class pooling",
            queries="no query features, labels, masks or statistics enter the conditioner",
            feature_precision="frozen context-only column/row forward in float32 in train and deployment"),
        diagnostic="source-probe correction RMS, common-grid map signatures, parameter and encoder gradient norms",
        selection_qualification="25 previously inspected validation sources, development only; no test bank")
    return spec


def new_model(arm, embedding_dim=128):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return ResidualPreprocessor(arm, embedding_dim=embedding_dim)


def prepare(args):
    spec = settings(args)
    diversity.verify_revision(spec)
    intent = dict(settings=spec)
    path = args.output_dir / "preparation.json"
    if path.exists() and previous.read(path) != intent:
        raise ValueError("preparation intent changed; use a new root")
    if not path.exists() and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use a new empty root")
    pilot.json_write(path, intent)
    if args.reuse_bank_dir is None:
        raise ValueError("this comparison requires the verified frozen 160/25 bank")
    source_manifest = previous.read(args.reuse_bank_dir / "manifest.json")
    if previous.digest({k: v for k, v in source_manifest.items() if k != "fingerprint"}) != source_manifest["fingerprint"]:
        raise ValueError("source manifest fingerprint changed")
    if source_manifest["fingerprint"] != args.source_fingerprint:
        raise ValueError("wrong frozen source bank")
    if pilot.hash_file(args.reuse_bank_dir / "banks_manifest.json") != source_manifest["banks_manifest_sha256"]:
        raise ValueError("source bank manifest changed")
    bank.reuse_frozen_banks(args.output_dir, args.candidate_manifest, args.reuse_bank_dir)
    banks = bank.prepare(args.output_dir, args.candidate_manifest, args.cache_dir)
    if banks["counts"] != dict(small=40, large=160, validation=25):
        raise ValueError("expected existing 160 training / 25 validation source allocation")
    backbone, checkpoint, backbone_hash = pilot.load_frozen(args, torch.device("cpu"))
    embedding_dim = backbone.row_interactor.embed_dim
    backbone_lock = dict(sha256=backbone_hash, checkpoint=Path(checkpoint).name,
        embedding_dim=embedding_dim, num_cls=backbone.row_interactor.num_cls,
        feature_group=backbone.col_embedder.feature_group,
        feature_group_size=backbone.col_embedder.feature_group_size)
    hashes, parameter_counts = {}, {}
    for arm in ARMS:
        model = new_model(arm, embedding_dim)
        path = args.output_dir / f"initial_{arm}.pt"
        if not path.exists():
            pilot.atomic_save(path, dict(model=pilot.state_cpu(model), model_seed=0))
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["model_seed"] != 0 or saved["model"].keys() != model.state_dict().keys():
            raise ValueError("initialization changed")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
        hashes[arm] = pilot.hash_file(path)
        parameter_counts[arm] = sum(p.numel() for p in model.parameters())
    del backbone
    lock_path = args.output_dir / "backbone_lock.json"
    if lock_path.exists() and previous.read(lock_path) != backbone_lock:
        raise ValueError("backbone changed during preparation")
    pilot.json_write(lock_path, backbone_lock)
    manifest = dict(settings=spec, banks=banks["banks"], counts=banks["counts"],
        banks_manifest_sha256=pilot.hash_file(args.output_dir / "banks_manifest.json"),
        initial_sha256=hashes, backbone=backbone_lock, parameter_counts=parameter_counts,
        source_fingerprint=source_manifest["fingerprint"], no_test_bank=True)
    manifest["fingerprint"] = previous.digest(manifest)
    path = args.output_dir / "manifest.json"
    if path.exists() and previous.read(path) != manifest:
        raise ValueError("frozen comparison changed")
    pilot.json_write(path, manifest)
    print(f"Prepared residual-conditioning comparison: {path}", flush=True)


def checked_manifest(args):
    manifest = previous.read(args.output_dir / "manifest.json")
    if manifest["settings"] != settings(args) or previous.digest(
            {k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("settings/code/manifest changed")
    if pilot.hash_file(args.output_dir / "banks_manifest.json") != manifest["banks_manifest_sha256"]:
        raise ValueError("bank manifest changed")
    for arm in ARMS:
        if pilot.hash_file(args.output_dir / f"initial_{arm}.pt") != manifest["initial_sha256"][arm]:
            raise ValueError("initial weights changed")
    if previous.read(args.output_dir / "backbone_lock.json") != manifest["backbone"]:
        raise ValueError("backbone lock changed")
    diversity.verify_revision(manifest["settings"])
    return manifest


def setup(args):
    manifest = checked_manifest(args)
    device = torch.device(args.device)
    backbone, path, sha = pilot.load_frozen(args, device)
    lock = manifest["backbone"]
    if sha != lock["sha256"] or Path(path).name != lock["checkpoint"] or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("frozen backbone changed")
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    initial = new_model(args.arm, lock["embedding_dim"]).to(device)
    initial.load_state_dict(torch.load(args.output_dir / f"initial_{args.arm}.pt",
                           map_location="cpu", weights_only=True)["model"])
    return backbone, initial, manifest, device


def prepared_episode(e, device):
    generator, members, positions, numerical_keep, keep = objective.real._prepared_views(e, 16)
    views = []
    canonical_context = None
    for method, (xs, ys) in members.items():
        configs = generator.ensemble_configs_[method]
        if len(configs) != 8:
            raise ValueError("expected eight native views per normalization method")
        for index, (features, classes) in enumerate(configs):
            inverse = {int(original): position for position, original in enumerate(features)}
            columns = torch.tensor([inverse[int(original)] for original in positions], device=device)
            x = torch.from_numpy(xs[index:index + 1]).to(device=device, dtype=torch.float32)
            if method == "none" and index == 0:
                order = torch.tensor([inverse[i] for i in range(x.shape[-1])], device=device)
                canonical_context = x[:, :e["n_context"]].index_select(-1, order)
            views.append(dict(slot=0 if method == "none" else 1, adapted=index >= 4,
                columns=columns, features=list(features), x=x,
                y=torch.from_numpy(ys[index:index + 1]).to(device=device, dtype=torch.float32),
                classes=torch.as_tensor(classes, device=device, dtype=torch.long)))
    if len(views) != 16 or sum(v["adapted"] for v in views) != 8 or canonical_context is None:
        raise ValueError("expected exactly the sixteen native views")
    context = dict(x=e["x_context"][..., keep][..., positions].to(device),
        y=e["y_context"].to(device), missing=e["context_missing"][None, :, numerical_keep].to(device),
        full=canonical_context, positions=torch.as_tensor(positions, device=device, dtype=torch.long),
        all_missing=torch.cat((e["context_missing"][None, :, numerical_keep],
                               e["query_missing"][None, :, numerical_keep]), 1).to(device))
    return context, views


def generate(model, backbone, context):
    features = frozen_context_features(backbone, context["full"], context["y"], context["positions"]) if model.conditioning == "backbone" else None
    return model.generate(context["x"], context["y"], context["missing"], frozen_features=features)


def numeric_views(model, backbone, context, views):
    parameters = generate(model, backbone, context)
    values = [model.apply_native(v["x"].index_select(-1, v["columns"]), parameters,
                                v["slot"], context["all_missing"]) if v["adapted"] else None for v in views]
    return values, parameters


def mean_logits(backbone, views, n_classes, numeric=None, inference_config=None, deployment=False):
    numeric = [None] * len(views) if numeric is None else numeric
    results = []
    for view, values in zip(views, numeric, strict=True):
        if deployment:
            config = inference_config if inference_config is not None else InferenceConfig()
        else:
            config = None
        output = objective.view_logits(backbone, view, n_classes, values, config)
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite native/residual view logits")
        results.append(output)
    return torch.stack(results).mean(0)


def episode_logits(backbone, model, e):
    context, views = prepared_episode(e, next(backbone.parameters()).device)
    with torch.no_grad():
        numeric = numeric_views(model, backbone, context, views)[0] if model is not None else None
        with pilot.frozen_inference(backbone):
            return mean_logits(backbone, views, e["n_classes"], numeric, deployment=True)


def ensemble_backward(backbone, model, e, scale=.25):
    context, views = prepared_episode(e, next(backbone.parameters()).device)
    values, _ = numeric_views(model, backbone, context, views)
    with torch.no_grad():
        logits = mean_logits(backbone, views, e["n_classes"], [None if v is None else v.detach() for v in values])
    leaf = logits.detach().requires_grad_(True)
    loss = F.cross_entropy((leaf / .9).flatten(0, 1), e["y_query"].flatten().to(leaf.device))
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite native16 residual objective")
    upstream, = torch.autograd.grad(loss, leaf)
    targets, gradients = [], []
    for view, value in zip(views, values, strict=True):
        if value is None:
            continue
        numeric = value.detach().requires_grad_(True)
        logits = objective.view_logits(backbone, view, e["n_classes"], numeric)
        grad, = torch.autograd.grad(logits, numeric, grad_outputs=upstream / 16.)
        targets.append(value)
        gradients.append(grad * scale)
    torch.autograd.backward(targets, gradients)
    return loss.detach()


def transform_diagnostics(model, backbone, e):
    context, views = prepared_episode(e, next(backbone.parameters()).device)
    with torch.no_grad():
        values, parameters = numeric_views(model, backbone, context, views)
        result = {}
        valid = ~context["all_missing"]
        for slot in (0, 1):
            index = next(i for i, v in enumerate(views) if v["adapted"] and v["slot"] == slot)
            view, value = views[index], values[index]
            original = view["x"].index_select(-1, view["columns"])
            result[f"slot{slot}_correction_rms"] = float((value - original)[valid].square().mean().sqrt()) if valid.any() else 0.
            result[f"slot{slot}_spline_gate"] = float(parameters.spline_gate[:, slot].mean())
            result[f"slot{slot}_neural_gate"] = float(parameters.neural_gate[:, slot].mean())
            result[f"slot{slot}_mixing_norm"] = float(parameters.mixing[:, slot].norm(dim=(-2, -1)).mean())
            # A common grid measures emitted functions independent of the source's
            # observed feature distribution. Column-averaged signature across all
            # source episodes quantifies between-dataset conditioning variation.
            grid = torch.linspace(-4., 4., 17, device=value.device)[None, :, None].expand(1, 17, value.shape[-1])
            correction = model.apply_native(grid, parameters, slot, torch.zeros_like(grid, dtype=torch.bool)) - grid
            result.update({f"slot{slot}_map_g{i}": float(v) for i, v in enumerate(correction.mean(-1).flatten())})
        return result


def execution_audit(backbone, initial, e, folder, device):
    model = copy.deepcopy(initial).train().requires_grad_(True)
    backbone.train()
    context, views = prepared_episode(e, device)
    with torch.no_grad():
        numeric, _ = numeric_views(model, backbone, context, views)
        for view, value in zip(views, numeric, strict=True):
            if value is not None:
                torch.testing.assert_close(value, view["x"].index_select(-1, view["columns"]), rtol=0, atol=0)
        raw = mean_logits(backbone, views, e["n_classes"])
        adapted = mean_logits(backbone, views, e["n_classes"], numeric)
        torch.testing.assert_close(raw, adapted, rtol=0, atol=0)
        config = InferenceConfig(COL_CONFIG=dict(use_amp=False), ROW_CONFIG=dict(use_amp=False), ICL_CONFIG=dict(use_amp=False))
        with pilot.frozen_inference(backbone):
            deployed = mean_logits(backbone, views, e["n_classes"], numeric, config, True)
        torch.testing.assert_close(raw, deployed, rtol=2e-4, atol=2e-4)
        amp = episode_logits(backbone, model, e)
        ordinary_amp = episode_logits(backbone, None, e)
        torch.testing.assert_close(amp, ordinary_amp, rtol=0, atol=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    norms = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = ensemble_backward(backbone, model, e, 1.)
        norms.append(core.diagnostic.gradient_norms(model))
        if not all(math.isfinite(v) for v in norms[-1].values()) or not any(v > 0 for v in norms[-1].values()):
            raise FloatingPointError("missing finite learning gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()):
        raise AssertionError("encoder received no gradient after head update")
    if any(p.grad is not None for p in backbone.parameters()):
        raise AssertionError("frozen backbone received gradients")
    with torch.no_grad():
        after = episode_logits(backbone, model, e)
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = dict(conditioning=model.conditioning, family=e["family"], split_seed=e["split_seed"],
        task_id=e.get("task_id"), exact_identity_inputs=True, exact_identity_default_amp_logits=True,
        fp32_deployment_max_difference=float((raw - deployed).abs().max()),
        default_amp_max_difference_from_fp32=float((amp - raw).abs().max()),
        last_ce=float(loss), gradient_norms=norms, post_two_updates_amp_finite=bool(torch.isfinite(after).all()),
        seconds=time.perf_counter() - started, optimizer_updates_to_saved_initial=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
    pilot.json_write(folder / "execution_audit.json", report)
    print(f"Residual execution audit: {json.dumps(report)}", flush=True)
    return report


def preflight(args):
    for arm in ARMS:
        current = copy.copy(args)
        current.arm = arm
        backbone, model, manifest, device = setup(current)
        episodes = bank.load_bank(args.output_dir, manifest, "real_validation")
        chosen = next(e for e in episodes if e["family"] == args.preflight_family and e["split_seed"] == args.preflight_split_seed)
        folder = args.output_dir / "preflight" / arm
        execution_audit(backbone, model, chosen, folder, device)
        pilot.json_write(folder / "complete.json", dict(fingerprint=manifest["fingerprint"], arm=arm,
            no_saved_model_updates=True, default_amp_finite=True, family=chosen["family"]))
        del backbone, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pilot.json_write(args.output_dir / "preflight" / "complete.json", dict(
        fingerprint=manifest["fingerprint"], arms=list(ARMS), both_passed=True))


def training_batch(arm, step, source_banks):
    if arm not in ARMS:
        raise ValueError("unknown conditioning arm")
    return diversity.training_batch("large", step, source_banks)


def references(backbone, initial, episodes, root, panel, fp):
    path = root / "references" / f"{panel}.pt"
    ids = [(e["family"], e["split_seed"]) for e in episodes]
    cache = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else dict(fingerprint=fp, ids=ids, values={})
    if cache["fingerprint"] != fp or cache["ids"] != ids:
        raise ValueError("reference cache changed")
    for i, e in enumerate(episodes):
        if i not in cache["values"]:
            logits = episode_logits(backbone, None, e)
            cache["values"][i] = dict(control=logits.cpu(), initial_blend_nll=previous.score(logits, e["y_query"].to(logits.device))["nll"])
            if i % 20 == 19:
                pilot.atomic_save(path, cache)
                print(f"Cached native16 {panel}: {i+1}/{len(episodes)}", flush=True)
    pilot.atomic_save(path, cache)
    return cache["values"]


grouped = diversity.grouped
passing = diversity.passing


def evaluate(backbone, model, panels, refs, folder, step, device):
    model.eval()
    result = {}
    for panel, episodes in panels.items():
        started, rows = time.perf_counter(), []
        for i, e in enumerate(episodes):
            logits = episode_logits(backbone, model, e)
            labels = e["y_query"].to(device)
            ref = refs[panel][i]
            scores = previous.score(logits, labels)
            baseline = previous.score(ref["control"].to(device), labels)
            row = dict(step=step, panel=panel, family=e["family"], split_seed=e["split_seed"],
                blend_views=16, learned_views=8, unchanged_views=8,
                initial_blend_nll=ref["initial_blend_nll"],
                **{f"blend_{k}": v for k, v in scores.items()},
                **{f"ordinary16_{k}": v for k, v in baseline.items()},
                **transform_diagnostics(model, backbone, e))
            if not all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
                raise FloatingPointError("nonfinite residual diagnostic")
            rows.append(row)
            pilot.csv_append(folder / "evaluation_episodes.csv", row)
        _, candidate = grouped(rows, "blend_nll", len(episodes))
        _, baseline = grouped(rows, "ordinary16_nll")
        metrics = objective.comparison(candidate, baseline)
        summary = dict(step=step, panel=panel, episodes=len(episodes), families=len(candidate),
            seconds=time.perf_counter() - started, mean_blend_nll=float(candidate.mean()),
            mean_initial_nll=float(baseline.mean()), mean_ordinary16_nll=float(baseline.mean()),
            mean_reduction_from_initial=1-float(candidate.mean()/baseline.mean()),
            mean_reduction_vs_ordinary16=1-float(candidate.mean()/baseline.mean()),
            wins_vs_ordinary16=metrics["material_wins"], losses_vs_ordinary16=metrics["material_losses"],
            median_gain_vs_ordinary16=float(np.median(1-(candidate+1e-4)/(baseline+1e-4))),
            harms_over_1pct=int(((candidate+1e-4)/(baseline+1e-4)>1.01).sum()),
            harms_over_5pct=int(((candidate+1e-4)/(baseline+1e-4)>1.05).sum()),
            correction_rms=float(np.mean([r[f"slot{s}_correction_rms"] for r in rows for s in (0,1)])),
            between_episode_map_signature_std=float(np.mean([np.std([r[f"slot{s}_map_g{j}"] for r in rows]) for s in (0,1) for j in range(17)])))
        result[panel] = summary
        pilot.csv_append(folder / "evaluation.csv", summary)
        pilot.csv_append(folder / "learning.csv", summary)
        print(f"step={step} {panel}: native16 mean gain={100*summary['mean_reduction_vs_ordinary16']:.3f}% W/L={summary['wins_vs_ordinary16']}/{summary['losses_vs_ordinary16']}, correction RMS={summary['correction_rms']:.5g}, map variation={summary['between_episode_map_signature_std']:.5g}", flush=True)
    return result


def train(args):
    root, arm = args.output_dir, args.arm
    if (root / "complete.json").exists():
        raise ValueError("completed choices are locked; further training forbidden")
    backbone, initial, manifest, device = setup(args)
    fp = manifest["fingerprint"]
    if device.type == "cuda":
        gate = previous.read(root / "preflight" / "complete.json")
        if gate["fingerprint"] != fp or not gate["both_passed"]:
            raise ValueError("both GPU preflights must pass before training")
    runfp = core.run_fingerprint(fp, arm, 0)
    folder = core.run_dir(root, arm, 0)
    if (folder / "complete.json").exists():
        if previous.read(folder / "complete.json")["fingerprint"] != runfp:
            raise ValueError("completed run changed")
        return
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError("run exists; pass --resume")
    source_banks = {k: bank.load_bank(root, manifest, f"{k}_train") for k in diversity.ARMS}
    panels = {p: bank.load_bank(root, manifest, p) for p in PANELS}
    refs = {p: references(backbone, initial, es, root, p, fp) for p, es in panels.items()}
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
        execution_audit(backbone, model, training_batch(arm, 1, source_banks)[0], folder, device)
        last = evaluate(backbone, model, panels, refs, folder, 0, device)
        best_score, best_model = last["real_validation"]["mean_blend_nll"], pilot.state_cpu(model)
    pilot.json_write(folder / "config.json", dict(fingerprint=runfp, experiment_fingerprint=fp,
        arm=arm, model_seed=0, initialization_sha256=manifest["initial_sha256"][arm],
        settings=manifest["settings"], backbone=previous.read(root / "backbone_lock.json"),
        training_sources=len(source_banks["large"]),
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
            losses.append(float(ensemble_backward(backbone, model, e)))
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
            a, b = scores[("backbone", choice, panel)], scores[("raw", choice, panel)]
            if a[0] != b[0]:
                raise ValueError("arm evaluation ordering differs")
            paired[f"{choice}/{panel}"] = objective.comparison(a[1], b[1])
    for panel in PANELS:
        a, b = scores[("backbone", "initial", panel)], scores[("raw", "initial", panel)]
        np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-8)
    result = dict(fingerprint=manifest["fingerprint"], runs=runs, summaries=summaries,
        backbone_versus_raw=paired, decisions=decisions,
        qualification="Same 160 sources and episodes, paired heads and one seed; encoder parameter counts and compute differ. Validation is development evidence. No confirmation test bank opened.")
    path = args.output_dir / "complete.json"
    if path.exists() and previous.read(path) != result:
        raise ValueError("locked report changed")
    pilot.json_write(path, result)
    print(f"Residual-conditioning experiment complete: {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "train", "report", "pipeline"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--cache-dir", type=Path, default=Path("results/pmlb_cache"))
    p.add_argument("--reuse-bank-dir", type=Path, help="Copy hash-verified frozen data panels into a new result root; never reuse references or learned state")
    p.add_argument("--preflight-family", default="Credit_Risk_Modeling")
    p.add_argument("--preflight-split-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--arm", choices=ARMS, default="raw")
    p.add_argument("--continuation-seed", type=int, choices=(0,), default=0)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--evaluate-every", type=int, default=512)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=.0003)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--expected-revision", help="Require this Git HEAD and all hashed dependencies to match its committed contents")
    p.add_argument("--source-fingerprint", default="7149269ed57b83ce024a07934f293bae6b69b80b0bd41bba1686c997a9f149f1")
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
