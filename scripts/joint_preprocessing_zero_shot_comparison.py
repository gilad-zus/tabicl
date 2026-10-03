"""Compare fresh, repeated and teacher-assisted shared zero-shot preprocessing.

Commands: prepare, teachers, train, lock, test. Test is inaccessible to training
and requires a saved lock of all validation-selected models and blend weights.
"""

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
from torch.nn import functional as F

from scripts import hyperspline_synthetic_train as synthetic
from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor
from tabicl._sklearn.preprocessing import EnsembleGenerator


ARMS = ("fresh", "repeated", "teacher")
TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET = 6_000_000_000, 7_000_000_000, 8_000_000_000
TRAIN_SEED, VAL_SEED, TEST_SEED = 201001, 211001, 221001
ORDER_SEED = 231001
ALPHAS = (0., .25, .5)
LOGS = ("training.csv", "evaluation.csv", "evaluation_tasks.csv", "fresh_tasks.csv")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def settings(args):
    if (min(args.steps, args.repeated_tasks, args.validation_tasks, args.test_tasks,
            args.probe_tasks, args.evaluate_every, args.save_every, args.fit_steps,
            args.fit_evaluate_every, args.bootstrap_samples) <= 0 or
            args.repeated_tasks % 4 or args.steps * 4 % args.repeated_tasks or
            args.teacher_steps * 4 % args.repeated_tasks or
            not 0 < args.teacher_steps < args.steps or args.probe_tasks > args.repeated_tasks or
            not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("invalid experiment budgets")
    paths = [Path(__file__), Path(pilot.__file__), Path(diagnostic.__file__),
             Path(capacity.__file__), Path(synthetic.__file__), Path(inspect.getfile(JointPreprocessor)),
             Path(__file__).parents[1] / "src/tabicl/prior/_dataset.py",
             Path(__file__).with_name("direct_spline_synthetic_headroom.py")]
    return dict(format_version=1, steps=args.steps, repeated_tasks=args.repeated_tasks,
        validation_tasks=args.validation_tasks, test_tasks=args.test_tasks, probe_tasks=args.probe_tasks,
        teacher_steps=args.teacher_steps, evaluate_every=args.evaluate_every, save_every=args.save_every,
        fit_steps=args.fit_steps, fit_evaluate_every=args.fit_evaluate_every,
        fit_lr=.001, lr=args.lr, tasks_per_update=4, weight_decay=1e-4, gradient_clip=1.,
        model_seed=0, order_seed=ORDER_SEED, alphas=ALPHAS,
        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=20261003,
        code_hashes={p.name: pilot.hash_file(p) for p in paths},
        generator=dict(prior_type="mix_scm", min_features=5, max_features=100,
            max_classes=10, prior_n_jobs=1, synthetic_observation_mode="coverage_expanded", train_seed=TRAIN_SEED),
        lengths=pilot.LENGTHS, fractions=pilot.FRACTIONS)


def fresh_seed(step):
    return (TRAIN_SEED + step * 1_000_003) % 2**32


def shape_for_step(step):
    shapes = [(n, f) for n in pilot.LENGTHS for f in pilot.FRACTIONS]
    order = np.random.default_rng(TRAIN_SEED).permutation(len(shapes))
    return shapes[int(order[(step - 1) % len(shapes)])]


def fresh_batch(step, config):
    length, fraction = shape_for_step(step)
    args = argparse.Namespace(**config["generator"], sequence_length=length, context_fraction=fraction)
    return synthetic.generate_episodes(args, 4, source_seed=fresh_seed(step),
        task_offset=TRAIN_OFFSET + (step - 1) * 4, device=torch.device("cpu"))


def seed_audit(config):
    train = {fresh_seed(s) for s in range(1, config["steps"] + 1)}
    val = {VAL_SEED + 10_000_019 * i for i in range(1, 13)}
    test = {TEST_SEED + 10_000_019 * i for i in range(1, 13)}
    old = {(161001 + s * 1_000_003) % 2**32 for s in range(1, 10001)}
    old |= {base + 10_000_019 * i for base in (171001, 172001, 181001, 182001) for i in range(1, 13)}
    if len(train) != config["steps"] or train & val or train & test or val & test or (train | val | test) & old:
        raise ValueError("new seed schedules collide with one another or prior joint experiments")
    if TRAIN_OFFSET + 4 * config["steps"] >= VAL_OFFSET or VAL_OFFSET + config["validation_tasks"] >= TEST_OFFSET:
        raise ValueError("task ID namespaces overlap")
    return dict(train_seeds=sorted(train), validation_seeds=sorted(val), test_seeds=sorted(test),
                historical_schedule="seed-0 40k joint pilot and fixed 32/128-task banks", historical_seeds_checked=len(old))


def episode_hash(e):
    h = hashlib.sha256()
    for value in (e.x_context, e.x_query, e.y_context, e.y_query):
        array = value.detach().cpu().contiguous().numpy()
        h.update(str((array.shape, str(array.dtype))).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def metadata(e):
    return dict(task_id=e.task_id, seed=e.source_seed, n_context=e.x_context.shape[1],
        n_query=e.x_query.shape[1], n_features=e.x_context.shape[-1], n_classes=e.n_classes,
        content_hash=episode_hash(e), training_view=diagnostic.training_view(pilot.filtered_episode(e)))


def prepare(args):
    root, spec = args.output_dir, settings(args)
    intent = dict(settings=spec, fingerprint=digest(spec), seed_audit=seed_audit(spec))
    path = root / "preparation.json"
    if path.exists():
        if read(path)["fingerprint"] != intent["fingerprint"]:
            raise ValueError("preparation fingerprint differs")
        if not args.resume:
            raise FileExistsError("preparation exists; use --resume")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("use an empty experiment directory")
        pilot.json_write(path, intent)
    banks, content_ids = {}, set()
    for panel, count, seed, offset in (("repeated", args.repeated_tasks, TRAIN_SEED, TRAIN_OFFSET),
        ("validation", args.validation_tasks, VAL_SEED, VAL_OFFSET),
        ("test", args.test_tasks, TEST_SEED, TEST_OFFSET)):
        path = root / "banks" / f"{panel}.pt"
        info_path = path.with_suffix(".json")
        if info_path.exists():
            info = read(info_path)
            if pilot.hash_file(path) != info["sha256"]:
                raise ValueError("prepared bank hash differs")
            episodes = synthetic.load_episode_bank(path, expected_seed=seed, expected_count=count,
                device=torch.device("cpu"), expected_observation_mode="coverage_expanded")
        else:
            if panel == "repeated":
                episodes = [e for step in range(1, count // 4 + 1) for e in fresh_batch(step, spec)]
            else:
                episodes = synthetic.generate_scheduled_episodes(argparse.Namespace(**spec["generator"]), count,
                    source_seed=seed, task_offset=offset, device=torch.device("cpu"),
                    sequence_lengths=pilot.LENGTHS, context_fractions=pilot.FRACTIONS,
                    observation_mode="coverage_expanded")
            synthetic.validate_episode_classes(episodes, 10)
            temporary = path.with_suffix(".pt.tmp")
            synthetic.save_episode_bank(temporary, episodes, source_seed=seed)
            temporary.replace(path)
            info = dict(path=path.relative_to(root).as_posix(), source_seed=seed, count=count,
                sha256=pilot.hash_file(path), tasks=[metadata(e) for e in episodes])
            pilot.json_write(info_path, info)
        for e, recorded in zip(episodes, info["tasks"], strict=True):
            if metadata(e) != recorded:
                raise ValueError("bank metadata differs")
            if recorded["content_hash"] in content_ids:
                raise ValueError("duplicate task content within/between new banks")
            content_ids.add(recorded["content_hash"])
        if [e.task_id for e in episodes] != list(range(offset, offset + count)):
            raise ValueError("bank task ID schedule differs")
        banks[panel] = info
    manifest = dict(format_version=1, settings=spec["generator"], experiment=spec,
        intent_fingerprint=intent["fingerprint"], banks=banks, seed_audit=intent["seed_audit"],
        repeated_subset="first repeated_tasks/4 complete fresh-stream steps, chosen before scores")
    pilot.json_write(root / "manifest.json", manifest)


def setup(args):
    root = args.output_dir
    manifest = read(root / "manifest.json")
    spec = settings(args)
    if digest(spec) != manifest["intent_fingerprint"]:
        raise ValueError("experiment settings/code differ from preparation")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("TabICL must be frozen")
    backbone.train()
    fingerprint = digest(dict(settings=spec, manifest_sha256=pilot.hash_file(root / "manifest.json"), backbone_hash=backbone_hash))
    path = root / "config.json"
    if path.exists():
        if read(path)["fingerprint"] != fingerprint:
            raise ValueError("backbone/config fingerprint differs")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        pilot.json_write(path, dict(**spec, fingerprint=fingerprint, backbone_hash=backbone_hash, revision=revision))
    torch.manual_seed(0)
    initial = JointPreprocessor("joint").to(device)
    return backbone, initial, manifest, fingerprint, device


def score(logits, labels):
    logits = logits.float() / .9
    value = float(F.cross_entropy(logits.flatten(0, 1), labels.flatten()))
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite evaluation")
    return dict(nll=value, accuracy=float((logits.argmax(-1).flatten() == labels.flatten()).float().mean()))


def ordinary_logits(backbone, episode, estimators=8):
    """Exact standard none/power preprocessing, aligned mean logits."""
    generator = EnsembleGenerator(classification=True, n_estimators=estimators,
        norm_methods=["none", "power"], feat_shuffle_method="latin", class_shuffle_method="shift", random_state=0)
    generator.fit(episode.x_context.squeeze(0).cpu().numpy(), episode.y_context.squeeze(0).cpu().numpy().astype(int))
    members = generator.transform(episode.x_query.squeeze(0).cpu().numpy(), mode="both")
    device, logits = next(backbone.parameters()).device, []
    with torch.no_grad(), pilot.frozen_inference(backbone):
        for method, (xs, ys) in members.items():
            for index, (features, classes) in enumerate(generator.ensemble_configs_[method]):
                backbone.clear_cache()
                raw = backbone(torch.from_numpy(xs[index:index + 1]).to(device=device, dtype=torch.float32),
                    torch.from_numpy(ys[index:index + 1]).to(device=device, dtype=torch.float32),
                    feature_shuffles=[list(features)], return_logits=True)
                logits.append(raw[..., :episode.n_classes][..., torch.tensor(classes, device=device, dtype=torch.long)])
    return torch.stack(logits).mean(0), len(logits)


def references(backbone, episodes, root, panel, fingerprint, device):
    path = root / "references" / f"{panel}.pt"
    metadata_path = path.with_suffix(".json")
    if path.exists():
        if metadata_path.exists() and read(metadata_path)["sha256"] != pilot.hash_file(path):
            raise ValueError("reference cache hash differs")
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache["fingerprint"] != fingerprint or cache["ids"] != [e.task_id for e in episodes]:
            raise ValueError("reference cache differs")
        values = cache["values"]
    else:
        values = {}
    if set(values) - {e.task_id for e in episodes}:
        raise ValueError("reference cache contains unexpected tasks")
    if len(values) == len(episodes) and metadata_path.exists():
        return values
    for raw in episodes:
        if raw.task_id in values:
            continue
        e = pilot.on_device(raw, device)
        ordinary, views8 = ordinary_logits(backbone, e, 8)
        control, views16 = ordinary_logits(backbone, e, 16)
        with torch.no_grad(), pilot.frozen_inference(backbone):
            identity = pilot.forward_views(backbone, None, e)
        with torch.no_grad():
            single = float(diagnostic.surrogate_nll(backbone, None, pilot.filtered_episode(e)))
        values[e.task_id] = dict(ordinary=ordinary.cpu(), control=control.cpu(), identity=identity.cpu(),
            ordinary_nll=score(ordinary, e.y_query)["nll"], identity_nll=score(identity, e.y_query)["nll"],
            identity_single_nll=single, views8=views8, views16=views16)
        print(f"references {panel} task={e.task_id}", flush=True)
        if len(values) % 25 == 0:
            pilot.atomic_save(path, dict(fingerprint=fingerprint, ids=[e.task_id for e in episodes], values=values))
    pilot.atomic_save(path, dict(fingerprint=fingerprint, ids=[e.task_id for e in episodes], values=values))
    pilot.json_write(metadata_path, dict(fingerprint=fingerprint, sha256=pilot.hash_file(path)))
    return values


def teachers(args):
    backbone, initial, manifest, fp, device = setup(args)
    bank = pilot.load_bank(args.output_dir, manifest, "repeated")
    root = args.output_dir / "teachers"
    if (root / "complete.json").exists():
        if read(root / "complete.json")["fingerprint"] != fp:
            raise ValueError("teacher completion differs")
        return
    if args.max_teachers is not None and args.max_teachers <= 0:
        raise ValueError("max-teachers must be positive")
    finished, new_fits = {}, 0
    started = time.perf_counter()
    for raw in bank:
        folder = root / "fits" / str(raw.task_id)
        already_done = (folder / "complete.json").exists()
        if not already_done and args.max_teachers is not None and new_fits >= args.max_teachers:
            break
        e = pilot.on_device(pilot.filtered_episode(raw), device)
        with torch.no_grad():
            baseline = float(diagnostic.surrogate_nll(backbone, initial, e))
        finished[e.task_id] = capacity.fit_task(backbone, initial, e, baseline, folder, args, .001, fp + f":teacher:{e.task_id}")
        new_fits += int(not already_done)
    seconds = time.perf_counter() - started
    pilot.csv_append(root / "timing.csv", dict(segment=time.time(), newly_completed=new_fits, seconds=seconds))
    if len(finished) != len(bank):
        print("Teacher chunk stopped; resume to finish remaining tasks.", flush=True)
        return
    targets = {}
    for raw in bank:
        e = pilot.on_device(pilot.filtered_episode(raw), device)
        fit = finished[e.task_id]
        teacher = capacity.independent_map(initial, e)
        useful = fit["best_gain"] >= .01
        if useful:
            saved = torch.load(root / "fits" / str(e.task_id) / "selected.pt", map_location="cpu", weights_only=True)
            if saved["fingerprint"] != fp + f":teacher:{e.task_id}":
                raise ValueError("teacher checkpoint differs")
            teacher.load_state_dict(saved["model"])
        with torch.no_grad():
            p = teacher.generate(e.x_context, e.y_context)
            slot = pilot.view_specs(e.x_context.shape[-1], e.n_classes)[diagnostic.training_view(e)][0]
            x = torch.cat((e.x_context, e.x_query), 1)
            transformed = teacher.apply(x, p, slot)
            baseline = (x - p.location[:, None]) / p.scale[:, None]
            replay = float(diagnostic.surrogate_nll(backbone, teacher, e))
        expected = fit["best_nll"] if useful else fit["identity_nll"]
        if abs(replay - expected) > 1e-4 * (1 + expected):
            raise ValueError("teacher replay differs")
        targets[e.task_id] = dict(slot=slot, transformed=transformed.cpu(), useful=useful,
            baseline_mse=float((transformed - baseline).square().mean()), nll=replay)
    path = root / "targets.pt"
    pilot.atomic_save(path, dict(fingerprint=fp, targets=targets))
    pilot.json_write(root / "complete.json", dict(fingerprint=fp, tasks=len(targets),
        useful=sum(t["useful"] for t in targets.values()), neutral=sum(not t["useful"] for t in targets.values()),
        targets_sha256=pilot.hash_file(path), independent_updates=len(bank) * args.fit_steps,
        recorded_segment_seconds=sum(float(r["seconds"]) for r in capacity.read_rows(root / "timing.csv")),
        limitation="fitting targets; recorded segments exclude work lost before timing save"))


def load_targets(root, fingerprint, ids):
    complete = read(root / "teachers/complete.json")
    path = root / "teachers/targets.pt"
    if complete["fingerprint"] != fingerprint or complete["targets_sha256"] != pilot.hash_file(path):
        raise ValueError("teacher cache differs")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache["fingerprint"] != fingerprint or set(cache["targets"]) != set(ids):
        raise ValueError("teacher task IDs differ")
    if any(not torch.isfinite(t["transformed"]).all() or not math.isfinite(t["baseline_mse"]) or t["baseline_mse"] < 0 for t in cache["targets"].values()):
        raise ValueError("nonfinite teacher cache")
    return cache["targets"], complete["targets_sha256"]


def batch_indices(step, count):
    epoch, batch = divmod(step - 1, count // 4)
    return np.random.default_rng(ORDER_SEED + epoch).permutation(count)[4 * batch:4 * batch + 4].tolist()


def evaluate(backbone, model, banks, refs, root, step, device, targets=None):
    model.eval()
    scores = {}
    for panel, episodes in banks.items():
        started = time.perf_counter()
        values, log_ratios = [], []
        for raw in episodes:
            e = pilot.on_device(raw, device)
            with torch.no_grad():
                single = float(diagnostic.surrogate_nll(backbone, model, pilot.filtered_episode(e)))
                branch = diagnostic.transform_diagnostics(model, pilot.filtered_episode(e))
            with torch.no_grad(), pilot.frozen_inference(backbone):
                logits = pilot.forward_views(backbone, model, e)
            result = score(logits, e.y_query)
            ref = refs[panel][e.task_id]
            teacher_fields = dict(function_mse=None, normalized_function_mse=None, teacher_nll=None, teacher_recovery=None)
            if targets and e.task_id in targets:
                target = targets[e.task_id]
                with torch.no_grad():
                    normalized, mse = capacity.function_loss(model, pilot.filtered_episode(e), target)
                teacher_fields.update(function_mse=float(mse), normalized_function_mse=float(normalized), teacher_nll=target["nll"],
                    teacher_recovery=(ref["identity_single_nll"]-single)/(ref["identity_single_nll"]-target["nll"]) if target["useful"] else None)
            row = dict(step=step, panel=panel, task_id=e.task_id, single_nll=single,
                identity_single_nll=ref["identity_single_nll"], ordinary_nll=ref["ordinary_nll"],
                identity_ensemble_nll=ref["identity_nll"], ordinary_views=ref["views8"],
                learned_views=len(pilot.view_specs(pilot.filtered_episode(e).x_context.shape[-1], e.n_classes)),
                **result, **teacher_fields, **branch)
            if not all(math.isfinite(float(v)) for k, v in row.items() if k != "panel" and v is not None):
                raise FloatingPointError("nonfinite panel evaluation")
            pilot.csv_append(root / "evaluation_tasks.csv", row)
            values.append(row)
            log_ratios.append(math.log((result["nll"] + 1e-4) / (ref["ordinary_nll"] + 1e-4)))
        summary = dict(step=step, panel=panel, tasks=len(values), seconds=time.perf_counter()-started,
            mean_nll=float(np.mean([r["nll"] for r in values])),
            mean_single_nll=float(np.mean([r["single_nll"] for r in values])),
            validation_score=float(np.mean(log_ratios)), geometric_gain=1 - math.exp(float(np.mean(log_ratios))),
            wins=sum(r["nll"] < r["ordinary_nll"] - 1e-6 for r in values),
            losses=sum(r["nll"] > r["ordinary_nll"] + 1e-6 for r in values),
            harms_over_5pct=sum(r["nll"] + 1e-4 > 1.05 * (r["ordinary_nll"] + 1e-4) for r in values))
        for name in ("function_mse", "normalized_function_mse", "teacher_recovery"):
            numbers = [r[name] for r in values if r[name] is not None]
            summary[name] = float(np.mean(numbers)) if numbers else None
        pilot.csv_append(root / "evaluation.csv", summary)
        scores[panel] = summary
        print(f"step={step} {panel} nll={summary['mean_nll']:.6f} gain={100*summary['geometric_gain']:.3f}% "
              f"W/L={summary['wins']}/{summary['losses']}", flush=True)
    return scores


def train(args):
    backbone, initial, manifest, fp, device = setup(args)
    root = args.output_dir / args.arm
    if (args.output_dir / "lock.json").exists() or (args.output_dir / "test_report/started.json").exists():
        raise ValueError("experiment is locked; training cannot continue")
    repeated = pilot.load_bank(args.output_dir, manifest, "repeated")
    banks = dict(probe=repeated[:args.probe_tasks], validation=pilot.load_bank(args.output_dir, manifest, "validation"))
    refs = {panel: references(backbone, bank, args.output_dir, panel, fp, device) for panel, bank in banks.items()}
    targets, teacher_hash = (load_targets(args.output_dir, fp, [e.task_id for e in repeated]) if args.arm == "teacher" else ({}, None))
    fingerprint = digest(dict(experiment=fp, arm=args.arm, teacher_hash=teacher_hash))
    complete_path, state_path = root / "complete.json", root / "state.pt"
    if complete_path.exists():
        if read(complete_path)["fingerprint"] != fingerprint:
            raise ValueError("completed arm differs")
        return
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("arm exists; use --resume")
    torch.manual_seed(0)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    step = best_step = clipped = resets = 0
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        capacity.optimizer_to(optimizer, device)
        step, best_step, clipped, resets = saved["step"], saved["best_step"], saved["clipped"], saved["resets"]
        best_score, best_model, last = saved["best_score"], saved["best_model"], saved["last"]
        torch.set_rng_state(saved["rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for name in LOGS:
            capacity.trim_csv(root / name, step)
    else:
        last = evaluate(backbone, model, banks, refs, root, 0, device, targets)
        best_score, best_model = last["validation"]["validation_score"], pilot.state_cpu(model)

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=fingerprint, step=step, best_step=best_step,
            clipped=clipped, resets=resets, best_score=best_score, best_model=best_model, last=last,
            model=pilot.state_cpu(model), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

    save()
    stop = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    if stop < step:
        raise ValueError("stop precedes saved state")
    for step in range(step + 1, stop + 1):
        started = time.perf_counter()
        if step == args.teacher_steps + 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            resets += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.arm == "fresh":
            batch = repeated[4 * (step - 1):4 * step] if step <= len(repeated) // 4 else fresh_batch(step, manifest["experiment"])
            for e in batch:
                pilot.csv_append(root / "fresh_tasks.csv", dict(step=step, **metadata(e)))
        else:
            batch = [repeated[i] for i in batch_indices(step, len(repeated))]
        synthetic.validate_episode_classes(batch, backbone.max_classes)
        losses, function_losses = [], []
        function_stage = args.arm == "teacher" and step <= args.teacher_steps
        for raw in batch:
            e = pilot.on_device(pilot.filtered_episode(raw), device)
            loss = capacity.function_loss(model, e, targets[e.task_id])[0] if function_stage else diagnostic.surrogate_nll(backbone, model, e)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training objective")
            (loss / 4).backward()
            losses.append(float(loss.detach()))
            if args.arm == "teacher":
                if function_stage:
                    function_losses.append(float(loss.detach()))
                else:
                    with torch.no_grad():
                        function_losses.append(float(capacity.function_loss(model, e, targets[e.task_id])[0]))
        norms = diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not math.isfinite(norm) or not all(math.isfinite(v) for v in norms.values()):
            raise FloatingPointError("nonfinite gradient")
        clipped += int(norm > 1)
        optimizer.step()
        pilot.csv_append(root / "training.csv", dict(step=step, stage="function" if function_stage else "prediction",
            seconds=time.perf_counter()-started,
            task_ids="|".join(str(e.task_id) for e in batch), objective=float(np.mean(losses)),
            normalized_function_loss=float(np.mean(function_losses)) if function_losses else None,
            lr=args.lr, optimizer_resets=resets, preclip_gradient_norm=norm,
            clip_factor=min(1., 1 / (norm + 1e-6)), clipped_fraction=clipped / step, **norms))
        evaluated = step % args.evaluate_every == 0 or step in (args.teacher_steps, args.steps)
        if evaluated:
            last = evaluate(backbone, model, banks, refs, root, step, device, targets)
            candidate = last["validation"]["validation_score"]
            if candidate < best_score:
                best_score, best_step, best_model = candidate, step, pilot.state_cpu(model)
        if evaluated or step % args.save_every == 0 or step == stop:
            save()
            print(f"arm={args.arm} step={step}/{args.steps} objective={np.mean(losses):.6f} "
                  f"gradient={norm:.4f} clipped={clipped/step:.3f} best_validation={best_score:.6f}@{best_step}", flush=True)
    if step != args.steps:
        return
    pilot.atomic_save(root / "selected.pt", dict(fingerprint=fingerprint, model=best_model, step=best_step))
    pilot.json_write(complete_path, dict(fingerprint=fingerprint, experiment_fingerprint=fp, arm=args.arm,
        steps=step, selected_step=best_step, selected_score=best_score, optimizer_resets=resets,
        final=last, selected_sha256=pilot.hash_file(root / "selected.pt"), teacher_hash=teacher_hash,
        recorded_training_seconds=sum(float(r["seconds"]) for r in capacity.read_rows(root / "training.csv")),
        recorded_evaluation_seconds=sum(float(r["seconds"]) for r in capacity.read_rows(root / "evaluation.csv")),
        timing_limitation="recorded steps/panels exclude setup, references, and interrupted unsaved work"))


def bootstrap_summary(candidate, baseline, samples):
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if candidate.shape != baseline.shape or not len(candidate) or not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("invalid paired losses")
    ratios = (candidate + 1e-4) / (baseline + 1e-4)
    logs = np.log(ratios)
    rng, draws = np.random.default_rng(20261003), []
    for start in range(0, samples, 256):
        indices = rng.integers(0, len(logs), size=(min(256, samples - start), len(logs)))
        draws.extend(logs[indices].mean(1).tolist())
    delta = candidate - baseline
    return dict(tasks=len(candidate), mean_nll=float(candidate.mean()), mean_nll_delta=float(delta.mean()),
        mean_nll_reduction=1-float(candidate.mean()/baseline.mean()), geometric_gain=1-float(np.exp(logs.mean())),
        gain_ci95=[1-float(np.exp(np.quantile(draws, q))) for q in (.975, .025)],
        median_gain=float(np.median(1-ratios)), wins=int((delta < -1e-6).sum()),
        losses=int((delta > 1e-6).sum()), ties=int((np.abs(delta) <= 1e-6).sum()),
        **{f"harms_over_{p}pct": int((ratios > 1+p/100).sum()) for p in (1,5,10)},
        max_harm=max(0., float(ratios.max()-1)))


def prediction_panel(backbone, model, bank, refs, device):
    model.eval()
    predictions, nlls = {}, {}
    for raw in bank:
        e = pilot.on_device(raw, device)
        with torch.no_grad(), pilot.frozen_inference(backbone):
            logits = pilot.forward_views(backbone, model, e)
        ordinary = refs[e.task_id]["ordinary"].to(device)
        choices = {"learned": logits, **{f"alpha{alpha:g}": (1-alpha)*ordinary+alpha*logits for alpha in ALPHAS}}
        predictions[e.task_id] = dict(logits=logits.cpu(), labels=e.y_query.cpu())
        nlls[e.task_id] = {name: score(value, e.y_query)["nll"] for name, value in choices.items()}
    return predictions, nlls


def select_alpha(nlls, reference):
    return min(ALPHAS, key=lambda a: float(np.mean([math.log((nlls[t][f"alpha{a:g}"]+1e-4)/(reference[t]["ordinary_nll"]+1e-4)) for t in nlls])))


def selected_model(root, arm, fp, device):
    complete = read(root / arm / "complete.json")
    path = root / arm / "selected.pt"
    if complete["experiment_fingerprint"] != fp or complete["selected_sha256"] != pilot.hash_file(path):
        raise ValueError("selected model differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["fingerprint"] != complete["fingerprint"] or state["step"] != complete["selected_step"]:
        raise ValueError("selected checkpoint metadata differs")
    model = JointPreprocessor("joint").to(device)
    model.load_state_dict(state["model"])
    return model, complete


def lock(args):
    backbone, _, manifest, fp, device = setup(args)
    root = args.output_dir
    if (root / "lock.json").exists():
        if read(root / "lock.json")["fingerprint"] != fp:
            raise ValueError("lock differs")
        return
    if (root / "test_report/started.json").exists():
        raise ValueError("test has started; choices cannot be reselected")
    # Require all arms before evaluating or freezing final choices.
    for arm in ARMS:
        read(root / arm / "complete.json")
    bank = pilot.load_bank(root, manifest, "validation")
    refs = references(backbone, bank, root, "validation", fp, device)
    choices = {}
    for arm in ARMS:
        model, complete = selected_model(root, arm, fp, device)
        predictions, nlls = prediction_panel(backbone, model, bank, refs, device)
        replay_score = float(np.mean([math.log((nlls[t]["learned"]+1e-4)/(refs[t]["ordinary_nll"]+1e-4)) for t in nlls]))
        if abs(replay_score - complete["selected_score"]) > 1e-4 * (1 + abs(complete["selected_score"])):
            raise ValueError("selected validation checkpoint replay differs")
        alpha = select_alpha(nlls, refs)
        pilot.atomic_save(root / arm / "validation_predictions.pt", dict(fingerprint=fp, predictions=predictions))
        choices[arm] = dict(alpha=alpha, checkpoint_sha256=complete["selected_sha256"],
            complete_sha256=pilot.hash_file(root / arm / "complete.json"), selected_step=complete["selected_step"],
            validation_nlls=nlls)
    pilot.json_write(root / "lock.json", dict(fingerprint=fp, manifest_sha256=pilot.hash_file(root / "manifest.json"),
        arms=choices, selection="one global validation checkpoint and alpha per arm; test labels never consulted"))


def test(args):
    root = args.output_dir
    if not (root / "lock.json").exists():
        raise ValueError("final testing requires lock of all arms")
    backbone, _, manifest, fp, device = setup(args)
    locked = read(root / "lock.json")
    if locked["fingerprint"] != fp or locked["manifest_sha256"] != pilot.hash_file(root / "manifest.json"):
        raise ValueError("lock/config differs")
    destination = root / "test_report"
    lock_hash = pilot.hash_file(root / "lock.json")
    started_path = destination / "started.json"
    if started_path.exists():
        if read(started_path)["lock_hash"] != lock_hash:
            raise ValueError("test already started under a different lock")
    else:
        pilot.json_write(started_path, dict(fingerprint=fp, lock_hash=lock_hash))
    if (destination / "complete.json").exists():
        if read(destination / "complete.json")["lock_hash"] != lock_hash:
            raise ValueError("test already evaluated under a different lock")
        return
    models = {}
    for arm in ARMS:
        choice = locked["arms"][arm]
        if choice["complete_sha256"] != pilot.hash_file(root / arm / "complete.json") or choice["checkpoint_sha256"] != pilot.hash_file(root / arm / "selected.pt"):
            raise ValueError("locked model changed")
        models[arm] = selected_model(root, arm, fp, device)[0]
    bank = pilot.load_bank(root, manifest, "test")
    refs = references(backbone, bank, root, "test", fp, device)
    ids = [e.task_id for e in bank]
    learned_views = {e.task_id: len(pilot.view_specs(pilot.filtered_episode(e).x_context.shape[-1], e.n_classes)) for e in bank}
    ordinary = [refs[t]["ordinary_nll"] for t in ids]
    control = [score(refs[e.task_id]["control"], e.y_query)["nll"] for e in bank]
    report = dict(fingerprint=fp, lock_hash=lock_hash, ordinary16_vs_ordinary8=bootstrap_summary(control, ordinary, args.bootstrap_samples), arms={}, pairs={})
    results = {}
    table_path = destination / "tasks.csv"
    if table_path.exists():
        table_path.unlink()
    for arm, model in models.items():
        predictions, nlls = prediction_panel(backbone, model, bank, refs, device)
        pilot.atomic_save(destination / f"{arm}_predictions.pt", dict(fingerprint=fp, predictions=predictions))
        alpha = locked["arms"][arm]["alpha"]
        candidates = {"learned": [nlls[t]["learned"] for t in ids], "equal_blend": [nlls[t]["alpha0.5"] for t in ids],
            "selected_blend": [nlls[t][f"alpha{alpha:g}"] for t in ids]}
        results[arm] = candidates
        report["arms"][arm] = dict(alpha=alpha, selected_step=locked["arms"][arm]["selected_step"],
            **{name + "_vs_ordinary8": bootstrap_summary(values, ordinary, args.bootstrap_samples) for name, values in candidates.items()},
            equal_blend_vs_ordinary16=bootstrap_summary(candidates["equal_blend"], control, args.bootstrap_samples),
            selected_blend_vs_ordinary16=bootstrap_summary(candidates["selected_blend"], control, args.bootstrap_samples))
        for t, baseline, ordinary16 in zip(ids, ordinary, control):
            pilot.csv_append(table_path, dict(arm=arm, task_id=t, ordinary8_nll=baseline,
                ordinary16_nll=ordinary16, learned_nll=nlls[t]["learned"], equal_blend_nll=nlls[t]["alpha0.5"],
                selected_alpha=alpha, selected_blend_nll=nlls[t][f"alpha{alpha:g}"],
                ordinary8_views=refs[t]["views8"], ordinary16_views=refs[t]["views16"],
                learned_views=learned_views[t], equal_blend_views=refs[t]["views8"]+learned_views[t],
                selected_blend_deployment_views=refs[t]["views8"]+(learned_views[t] if alpha else 0)))
    for a, b in (("repeated","fresh"),("teacher","repeated"),("teacher","fresh")):
        report["pairs"][a + "_vs_" + b] = {name: bootstrap_summary(results[a][name], results[b][name], args.bootstrap_samples)
            for name in ("learned","equal_blend","selected_blend")}
    pilot.json_write(destination / "complete.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "teachers", "train", "lock", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=10240)
    parser.add_argument("--repeated-tasks", type=int, default=512)
    parser.add_argument("--validation-tasks", type=int, default=512)
    parser.add_argument("--test-tasks", type=int, default=1024)
    parser.add_argument("--probe-tasks", type=int, default=48)
    parser.add_argument("--teacher-steps", type=int, default=2048)
    parser.add_argument("--evaluate-every", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--fit-steps", type=int, default=250)
    parser.add_argument("--fit-evaluate-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-teachers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "train" and args.arm is None:
        parser.error("train requires --arm")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
