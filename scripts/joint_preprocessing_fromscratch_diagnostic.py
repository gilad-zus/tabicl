"""Test full-hypernetwork prediction learning with independent-fit references.

Teachers are replayed for scoring only. Every trained full hypernetwork starts
from the same fresh seed-zero initialization and uses true-label query CE.
All scores reuse fitting rows; this experiment does not measure zero-shot gain.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import validate_episode_classes
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


LOGS = ("training.csv", "evaluation.csv", "evaluation_tasks.csv")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def close_loss(actual, expected, label):
    if not math.isfinite(actual) or abs(actual - expected) > 1e-4 * (1 + abs(expected)):
        raise ValueError(f"{label} does not replay: actual={actual}, expected={expected}")


def select_tasks(bank, count):
    ids = [e.task_id for e in bank]
    if count <= 0 or count > len(bank) or len(ids) != len(set(ids)):
        raise ValueError("invalid task selection")
    return sorted(bank, key=lambda e: e.task_id)[:count]


def task_index(step, count):
    """Exactly one visit per task per epoch, without outcome-dependent ordering."""
    if step < 1 or count < 1:
        raise ValueError("invalid task schedule")
    epoch, position = divmod(step - 1, count)
    return int(np.random.default_rng(183001 + epoch).permutation(count)[position])


def recovery(loss, baseline, teacher):
    # Suppress unstable ratios where the reference has less than 1% benefit.
    if capacity.gain(teacher, baseline) < .01:
        return None
    return (baseline - loss) / (baseline - teacher)


def source_data(args, backbone, backbone_hash, initial, device):
    """Replay independent checkpoints without passing weights to training."""
    bank, references, _, bank_config, _ = capacity.source_data(
        argparse.Namespace(source_dir=args.bank_dir), backbone_hash)
    config = read_json(args.teacher_dir / "config.json")
    complete = read_json(args.teacher_dir / "complete.json")
    if config["backbone_hash"] != backbone_hash:
        raise ValueError("teacher backbone differs")
    if config["runner_hash"] != pilot.hash_file(Path(capacity.__file__)):
        raise ValueError("teacher runner differs")
    if config["model_hash"] != bank_config["model_hash"] or config["inference_hash"] != bank_config["inference_helper_hash"]:
        raise ValueError("teacher model/inference implementation differs")
    if complete["fingerprint"] != config["fingerprint"] or config["model_seed"] != 0:
        raise ValueError("teacher completion differs")
    for name, expected in config["source_hashes"].items():
        if pilot.hash_file(args.bank_dir / name) != expected:
            raise ValueError(f"teacher source hash mismatch: {name}")
    if config["bank_sha256"] != pilot.read_manifest(args.bank_dir)["banks"]["train"]["sha256"]:
        raise ValueError("teacher bank differs")
    if config["fit_lrs"][0] != .001 or config["fit_steps"] != args.visits:
        raise ValueError("primary teacher must have lr .001 and matched visits")
    chosen = select_tasks(bank, args.tasks)
    validate_episode_classes(chosen, backbone.max_classes)
    episodes = [pilot.on_device(pilot.filtered_episode(e), device) for e in chosen]
    rows, refs, hashes = [], {}, {}
    for episode in episodes:
        reference = references[episode.task_id]
        baseline = float(reference["identity_surrogate_nll"])
        if (int(reference["n_context"]), int(reference["n_query"])) != (
                episode.x_context.shape[1], episode.x_query.shape[1]):
            raise ValueError("reference row counts differ")
        with torch.no_grad():
            close_loss(float(diagnostic.surrogate_nll(backbone, None, episode)), baseline, "identity")
            close_loss(float(diagnostic.surrogate_nll(backbone, initial, episode)), baseline, "fresh hypernetwork")
        task_rows = []
        for index, lr in enumerate(config["fit_lrs"]):
            folder = args.teacher_dir / "fits" / str(episode.task_id) / f"lr{index}"
            fit = read_json(folder / "complete.json")
            expected = config["fingerprint"] + f":{episode.task_id}:{index}"
            if (fit["fingerprint"] != expected or fit["task_id"] != episode.task_id or
                    fit["lr"] != lr or fit["visits"] != args.visits or
                    not 0 <= fit["best_step"] <= args.visits):
                raise ValueError("independent fit metadata differs")
            close_loss(fit["identity_nll"], baseline, "teacher identity")
            payload = torch.load(folder / "selected.pt", map_location="cpu", weights_only=True)
            if payload["fingerprint"] != expected:
                raise ValueError("teacher checkpoint fingerprint differs")
            if any(not torch.isfinite(v).all() for v in payload["model"].values()):
                raise ValueError("nonfinite teacher weights")
            teacher = capacity.independent_map(initial, episode)
            teacher.load_state_dict(payload["model"])
            with torch.no_grad():
                replay = float(diagnostic.surrogate_nll(backbone, teacher, episode))
            close_loss(replay, fit["best_nll"], "selected teacher")
            row = dict(task_id=episode.task_id, lr=lr, primary_rate=int(index == 0),
                       visits=args.visits, selected_step=fit["best_step"],
                       n_context=episode.x_context.shape[1], n_query=episode.x_query.shape[1],
                       n_features=episode.x_context.shape[-1], n_classes=episode.n_classes,
                       view_index=diagnostic.training_view(episode),
                       identity_nll=baseline, teacher_nll=replay, final_nll=fit["final_nll"],
                       teacher_gain=capacity.gain(replay, baseline),
                       selected_ensemble_nll=fit["selected_ensemble_nll"],
                       checkpoint_sha256=pilot.hash_file(folder / "selected.pt"))
            rows.append(row)
            task_rows.append(row)
            for name in ("complete.json", "selected.pt"):
                path = folder / name
                hashes[path.relative_to(args.teacher_dir).as_posix()] = pilot.hash_file(path)
            del teacher, payload
        refs[episode.task_id] = dict(identity_nll=baseline,
            ordinary_nll=float(reference["ordinary_ensemble_nll"]),
            identity_ensemble_nll=float(reference["identity_ensemble_nll"]),
            teacher_nll=task_rows[0]["teacher_nll"],
            sweep_teacher_nll=min(r["teacher_nll"] for r in task_rows))
        print(f"verified task={episode.task_id} identity={baseline:.6f} "
              f"teacher={refs[episode.task_id]['teacher_nll']:.6f}", flush=True)
    for name in ("config.json", "complete.json"):
        hashes[name] = pilot.hash_file(args.teacher_dir / name)
    return episodes, refs, rows, hashes, config


def evaluate(backbone, model, episodes, refs, root, step, label):
    model.eval()
    rows = []
    for e in episodes:
        ref = refs[e.task_id]
        with torch.no_grad():
            nll = float(diagnostic.surrogate_nll(backbone, model, e))
            branch = diagnostic.transform_diagnostics(model, e)
        if not math.isfinite(nll):
            raise FloatingPointError("nonfinite fitting loss")
        row = dict(step=step, visits_per_task=step // len(episodes), task_id=e.task_id,
                   nll=nll, gain_vs_identity=capacity.gain(nll, ref["identity_nll"]),
                   **ref, teacher_recovery=recovery(nll, ref["identity_nll"], ref["teacher_nll"]),
                   sweep_teacher_recovery=recovery(nll, ref["identity_nll"], ref["sweep_teacher_nll"]),
                   **branch)
        pilot.csv_append(root / "evaluation_tasks.csv", row)
        rows.append(row)
    metrics = capacity.summarize_losses([r["nll"] for r in rows], [r["identity_nll"] for r in rows])
    recoveries = [r["teacher_recovery"] for r in rows if r["teacher_recovery"] is not None]
    summary = dict(step=step, visits_per_task=step // len(episodes), tasks=len(rows), **metrics,
                   meaningful_teacher_tasks=len(recoveries),
                   teacher_recovery_mean=float(np.mean(recoveries)) if recoveries else None,
                   teacher_recovery_median=float(np.median(recoveries)) if recoveries else None,
                   teachers_half_recovered=sum(r >= .5 for r in recoveries),
                   **{key: float(np.mean([r[key] for r in rows])) for key in branch})
    pilot.csv_append(root / "evaluation.csv", summary)
    print(f"{label} visits={summary['visits_per_task']} nll={summary['mean_nll']:.6f} "
          f"gain={100 * summary['geometric_gain']:.2f}% W/L={summary['wins']}/{summary['losses']} "
          f"teacher_recovery={summary['teacher_recovery_mean']}", flush=True)
    return summary


def ensemble_report(backbone, model, episodes, refs, root, step, state):
    for e in episodes:
        nll = pilot.episode_metrics(backbone, model, e)["nll"]
        if not math.isfinite(nll):
            raise FloatingPointError("nonfinite ensemble loss")
        ref = refs[e.task_id]
        pilot.csv_append(root / "ensemble.csv", dict(state=state, step=step,
            task_id=e.task_id, ensemble_nll=nll, ordinary_nll=ref["ordinary_nll"],
            gain_vs_ordinary=capacity.gain(nll, ref["ordinary_nll"]),
            identity_nll=ref["identity_ensemble_nll"]))


def train_network(backbone, initial, episodes, refs, root, args, lr, fingerprint, label, *, stop_at=None):
    """One task per update; teacher values are used only in evaluation."""
    root.mkdir(parents=True, exist_ok=True)
    state_path, complete_path = root / "state.pt", root / "complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        if complete["fingerprint"] != fingerprint:
            raise ValueError("completed network fingerprint mismatch")
        return complete
    torch.manual_seed(0)
    model = copy.deepcopy(initial)
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("full hypernetwork parameters must all be trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    step = best_step = clipped = 0
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("network resume fingerprint mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        capacity.optimizer_to(optimizer, next(model.parameters()).device)
        step, best_step, clipped = saved["step"], saved["best_step"], saved["clipped"]
        best_nll, best_model = saved["best_nll"], saved["best_model"]
        initial_summary, last = saved["initial_summary"], saved["last_summary"]
        torch.set_rng_state(saved["rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        for name in LOGS:
            capacity.trim_csv(root / name, step)
    else:
        last = initial_summary = evaluate(backbone, model, episodes, refs, root, 0, label)
        for e in episodes:
            with torch.no_grad():
                close_loss(float(diagnostic.surrogate_nll(backbone, model, e)),
                           refs[e.task_id]["identity_nll"], "training initialization")
        best_nll, best_model = last["mean_nll"], pilot.state_cpu(model)

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=fingerprint, step=step,
            best_step=best_step, clipped=clipped, best_nll=best_nll,
            model=pilot.state_cpu(model), optimizer=optimizer.state_dict(), best_model=best_model,
            initial_summary=initial_summary, last_summary=last, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

    if step == 0:
        save()
    total = args.visits * len(episodes)
    stop = total if stop_at is None else min(total, stop_at)
    if stop < step:
        raise ValueError("stop precedes saved state")
    started = time.perf_counter()
    for step in range(step + 1, stop + 1):
        model.train()
        e = episodes[task_index(step, len(episodes))]
        optimizer.zero_grad(set_to_none=True)
        loss = diagnostic.surrogate_nll(backbone, model, e)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        gradients = diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm) or not all(math.isfinite(v) for v in gradients.values()):
            raise FloatingPointError("nonfinite gradients")
        clipped += int(norm > 1)
        optimizer.step()
        pilot.csv_append(root / "training.csv", dict(step=step, task_id=e.task_id,
            epoch=(step - 1) // len(episodes), lr=lr, nll=float(loss.detach()),
            preclip_gradient_norm=norm, clip_factor=min(1., 1 / (norm + 1e-6)),
            clipped_fraction=clipped / step, **gradients))
        evaluated = step % (args.evaluate_every * len(episodes)) == 0 or step == total
        if evaluated:
            last = evaluate(backbone, model, episodes, refs, root, step, label)
            if last["mean_nll"] < best_nll:
                best_nll, best_step, best_model = last["mean_nll"], step, pilot.state_cpu(model)
        if evaluated or step % args.save_every == 0 or step == stop:
            save()
            print(f"{label} step={step}/{total} sampled_nll={float(loss.detach()):.6f} "
                  f"gradient={norm:.4f} clipped={clipped / step:.3f} "
                  f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    if step != total:
        return None
    # Rebuild these reports if interruption occurred during completion.
    ensemble_path = root / "ensemble.csv"
    if ensemble_path.exists():
        ensemble_path.unlink()
    ensemble_report(backbone, model, episodes, refs, root, step, "final")
    model.load_state_dict(best_model)
    selected_root = root / "selected_report"
    for name in ("evaluation.csv", "evaluation_tasks.csv"):
        path = selected_root / name
        if path.exists():
            path.unlink()
    selected = evaluate(backbone, model, episodes, refs, selected_root, best_step, label + " selected")
    ensemble_report(backbone, model, episodes, refs, root, best_step, "selected")
    pilot.atomic_save(root / "selected.pt", dict(model=best_model, fingerprint=fingerprint, step=best_step))
    complete = dict(fingerprint=fingerprint, label=label, lr=lr, steps=step,
        visits_per_task=args.visits, tasks=[e.task_id for e in episodes], selected_step=best_step,
        initial=initial_summary, final=last, selected=selected, clipped_updates=clipped,
        selection="earliest minimum mean fitting NLL, including initialization; final is primary",
        limitation="same fitting queries reused; no teacher supervision or zero-shot evidence")
    pilot.json_write(complete_path, complete)
    return complete


def final_comparison(root, episodes, refs, lrs):
    """Keep separate-final and one shared-selected state distinct from hindsight."""
    destination = root / "comparison.csv"
    if destination.exists():
        destination.unlink()
    summaries = {}
    for index, lr in enumerate(lrs):
        for method in ("separate", "shared"):
            paired = []
            for e in episodes:
                folder = root / f"lr{index}" / method
                if method == "separate":
                    folder = folder / str(e.task_id)
                final = [r for r in capacity.read_rows(folder / "evaluation_tasks.csv")
                         if int(r["task_id"]) == e.task_id][-1]
                selected = [r for r in capacity.read_rows(folder / "selected_report" / "evaluation_tasks.csv")
                            if int(r["task_id"]) == e.task_id][0]
                ref = refs[e.task_id]
                row = dict(method=method, lr=lr, task_id=e.task_id, **ref,
                           final_nll=float(final["nll"]), selected_nll=float(selected["nll"]),
                           selected_step=int(selected["step"]),
                           final_teacher_recovery=recovery(float(final["nll"]), ref["identity_nll"], ref["teacher_nll"]),
                           selected_teacher_recovery=recovery(float(selected["nll"]), ref["identity_nll"], ref["teacher_nll"]))
                pilot.csv_append(destination, row)
                paired.append(row)
            for state in ("final", "selected"):
                metrics = capacity.summarize_losses([r[state + "_nll"] for r in paired],
                                                   [r["identity_nll"] for r in paired])
                values = [r[state + "_teacher_recovery"] for r in paired
                          if r[state + "_teacher_recovery"] is not None]
                summaries[f"{method}:lr{lr:g}:{state}"] = dict(**metrics,
                    mean_nll_reduction=1 - metrics["mean_nll"] / np.mean([r["identity_nll"] for r in paired]),
                    teacher_recovery_mean=float(np.mean(values)) if values else None,
                    teacher_recovery_median=float(np.median(values)) if values else None,
                    meaningful_teacher_tasks=len(values), teachers_half_recovered=sum(v >= .5 for v in values))
    return summaries


def run(args):
    if min(args.tasks, args.visits, args.evaluate_every, args.save_every) <= 0:
        raise ValueError("invalid training schedule")
    if not args.lrs or len(set(args.lrs)) != len(args.lrs) or any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):
        raise ValueError("invalid learning-rate grid")
    root = args.output_dir.resolve()
    for source in (args.bank_dir.resolve(), args.teacher_dir.resolve()):
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("output and source directories must be separate")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("output exists; use --resume")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("TabICL must be frozen")
    backbone.train()
    torch.manual_seed(0)
    initial = JointPreprocessor("joint").to(device)
    episodes, refs, teachers, hashes, teacher_config = source_data(args, backbone, backbone_hash, initial, device)
    settings = dict(format_version=1, backbone_hash=backbone_hash,
        runner_hash=pilot.hash_file(Path(__file__)), model_hash=teacher_config["model_hash"],
        inference_hash=teacher_config["inference_hash"], teacher_source_hashes=hashes,
        bank_source_hashes=teacher_config["source_hashes"],
        bank_sha256=teacher_config["bank_sha256"], task_ids=[e.task_id for e in episodes],
        selection="lowest task IDs before scores", model_seed=0, task_order_seed=183001,
        lrs=args.lrs, visits=args.visits, evaluate_every=args.evaluate_every, save_every=args.save_every,
        weight_decay=1e-4, gradient_clip=1., tasks_per_update=1, device=str(device),
        teacher_role="reference scores only; no weights or targets supplied to network training")
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = root / "config.json"
    if path.exists():
        if read_json(path)["fingerprint"] != fingerprint:
            raise ValueError("root resume fingerprint mismatch")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        pilot.json_write(path, dict(**settings, fingerprint=fingerprint, source_revision=revision,
            bank_dir=str(args.bank_dir), teacher_dir=str(args.teacher_dir)))
    reference_path = root / "teacher_reference.csv"
    if reference_path.exists():
        reference_path.unlink()
    for row in teachers:
        pilot.csv_append(reference_path, row)
    completed = {}
    for rate_index, lr in enumerate(args.lrs):
        rate_root = root / f"lr{rate_index}"
        runs = {}
        for e in episodes:
            label = f"separate task={e.task_id} lr={lr:g}"
            runs[str(e.task_id)] = train_network(backbone, initial, [e], refs,
                rate_root / "separate" / str(e.task_id), args, lr,
                fingerprint + f":{rate_index}:{e.task_id}", label)
        shared = train_network(backbone, initial, episodes, refs, rate_root / "shared",
            args, lr, fingerprint + f":{rate_index}:shared", f"shared lr={lr:g}")
        completed[str(lr)] = dict(separate=runs, shared=shared)
    summaries = final_comparison(root, episodes, refs, args.lrs)
    pilot.json_write(root / "complete.json", dict(fingerprint=fingerprint, rates=completed, summary=summaries,
        primary_teacher_rate=teacher_config["fit_lrs"][0],
        total_new_ce_updates=2 * len(episodes) * args.visits * len(args.lrs),
        teacher_training_cost="reused prior fits; replay and scoring only in this run",
        limitation="fitting diagnostic; no unused query rows or unseen tasks evaluated"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--visits", type=int, default=250)
    parser.add_argument("--lrs", nargs="+", type=float, default=[.001, .0003])
    parser.add_argument("--evaluate-every", type=int, default=25, help="Visits per task")
    parser.add_argument("--save-every", type=int, default=50, help="Optimizer updates")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
