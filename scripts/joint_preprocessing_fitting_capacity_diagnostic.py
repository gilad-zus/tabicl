"""Fit task-specific maps, then test seen-task hypernetwork function learning.

Reuses the completed fixed-task diagnostic bank. Every selection is on fitting
labels, so neither the independent maps nor the student are test evidence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import validate_episode_classes
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


class IgnoredContextEncoder(nn.Module):
    """The independent arm has no trainable conditioner."""

    def __init__(self, features, width):
        super().__init__()
        self.features, self.width = features, width

    def forward(self, x, y, stats, missing):
        if x.shape[-1] != self.features:
            raise ValueError("independent map belongs to a different feature count")
        return x.new_zeros((x.shape[0], self.features, self.width))


class RawTaskHead(nn.Module):
    """Optimize raw head outputs, retaining generate()'s original constraints."""

    def __init__(self, values):
        super().__init__()
        self.raw = nn.Parameter(values.detach().clone())

    def forward(self, tokens):
        return self.raw.expand(tokens.shape[0], -1, -1, -1)


def independent_map(initial, episode):
    """Copy initialization exactly; only raw per-task head tensors can learn."""
    heads = {name: head for name, head in initial.named_children() if name.endswith("_head")}
    captured = {}
    hooks = [head.register_forward_hook(
        lambda module, inputs, output, name=name: captured.__setitem__(name, output.detach().clone()))
        for name, head in heads.items()]
    try:
        with torch.no_grad():
            initial.generate(episode.x_context, episode.y_context)
    finally:
        for hook in hooks:
            hook.remove()
    model = copy.deepcopy(initial)
    model.requires_grad_(False)
    model.encoder = IgnoredContextEncoder(episode.x_context.shape[-1], initial.hidden_dim)
    for name, values in captured.items():
        setattr(model, name, RawTaskHead(values))
    return model


def read_rows(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def trim_csv(path, step):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["step"]) <= step]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def optimizer_to(optimizer, device):
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.to(device)


def gain(candidate, reference):
    return 1 - (candidate + 1e-4) / (reference + 1e-4)


def summarize_losses(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or not len(candidate) or not np.isfinite(candidate).all():
        raise ValueError("invalid paired fitting losses")
    delta = candidate - reference
    return dict(mean_nll=float(candidate.mean()),
                geometric_gain=1 - float(np.exp(np.log((candidate + 1e-4) / (reference + 1e-4)).mean())),
                wins=int((delta < -1e-6).sum()), losses=int((delta > 1e-6).sum()),
                ties=int((np.abs(delta) <= 1e-6).sum()),
                gains_over_1pct=int((candidate + 1e-4 < .99 * (reference + 1e-4)).sum()))


def fit_task(backbone, initial, episode, reference, root, args, lr, fingerprint, *, stop_at=None):
    """Checkpoint one task/rate, including exact interrupted optimizer resume."""
    root.mkdir(parents=True, exist_ok=True)
    complete_path, state_path = root / "complete.json", root / "state.pt"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf8"))
        if complete["fingerprint"] != fingerprint:
            raise ValueError("completed independent fit fingerprint mismatch")
        return complete
    model = independent_map(initial, episode)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    step = best_step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("independent resume fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_to(optimizer, episode.x_context.device)
        step, best_step = state["step"], state["best_step"]
        best_nll, last_nll = state["best_nll"], state["last_nll"]
        best_model = state["best_model"]
        trim_csv(root / "training.csv", step)
        trim_csv(root / "evaluation.csv", step)
    else:
        with torch.no_grad():
            last_nll = best_nll = float(diagnostic.surrogate_nll(backbone, model, episode))
        if abs(best_nll - reference) > 1e-4 * (1 + reference):
            raise ValueError("initial independent loss differs from the source identity reference")
        best_model = pilot.state_cpu(model)

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=fingerprint, step=step, best_step=best_step,
                          best_nll=best_nll, last_nll=last_nll, model=pilot.state_cpu(model),
                          optimizer=optimizer.state_dict(), best_model=best_model))

    def evaluate():
        nonlocal last_nll, best_nll, best_step, best_model
        with torch.no_grad():
            last_nll = float(diagnostic.surrogate_nll(backbone, model, episode))
            branch = diagnostic.transform_diagnostics(model, episode)
        if not math.isfinite(last_nll):
            raise FloatingPointError("nonfinite independent fitting loss")
        if last_nll < best_nll:
            best_nll, best_step, best_model = last_nll, step, pilot.state_cpu(model)
        pilot.csv_append(root / "evaluation.csv", dict(step=step, nll=last_nll,
                         gain_vs_identity=gain(last_nll, reference), **branch))

    if step == 0 and not (root / "evaluation.csv").exists():
        save()
        evaluate()
        save()
    stop = args.fit_steps if stop_at is None else min(args.fit_steps, stop_at)
    if stop < step:
        raise ValueError("stop precedes independent saved state")
    for step in range(step + 1, stop + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = diagnostic.surrogate_nll(backbone, model, episode)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite independent training loss")
        loss.backward()
        gradients = diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite independent gradient")
        optimizer.step()
        pilot.csv_append(root / "training.csv", dict(step=step, task_id=episode.task_id,
                         lr=lr, nll=float(loss.detach()), preclip_gradient_norm=norm, **gradients))
        if step % args.fit_evaluate_every == 0 or step == args.fit_steps or step == stop:
            evaluate()
            save()
            print(f"direct task={episode.task_id} lr={lr:g} visit={step}/{args.fit_steps} "
                  f"nll={last_nll:.6f} best_gain={100 * gain(best_nll, reference):.3f}%", flush=True)
    if step != args.fit_steps:
        return None
    model.load_state_dict(best_model)
    with torch.no_grad():
        best_branch = diagnostic.transform_diagnostics(model, episode)
    ensemble = pilot.episode_metrics(backbone, model, episode)["nll"]
    payload = dict(fingerprint=fingerprint, task_id=episode.task_id, lr=lr, visits=step,
                   best_step=best_step, best_nll=best_nll, final_nll=last_nll,
                   best_gain=gain(best_nll, reference), identity_nll=reference,
                   selected_ensemble_nll=ensemble, trainable_parameters=sum(p.numel() for p in parameters),
                   selection="best recorded fitting loss including initialization; not held out",
                   ensemble_limitation="only the task's fixed view/slot is directly fitted",
                   selected_branch_diagnostics=best_branch)
    pilot.atomic_save(root / "selected.pt", dict(model=best_model, fingerprint=fingerprint))
    pilot.json_write(complete_path, payload)
    return payload


def function_loss(model, episode, target):
    """Match the active-slot function; equalize task scales with a fixed floor."""
    p = model.generate(episode.x_context, episode.y_context)
    x = torch.cat((episode.x_context, episode.x_query), dim=1)
    student = model.apply(x, p, target["slot"])
    teacher = target["transformed"].to(x.device)
    mse = (student - teacher).square().mean()
    return mse / max(target["baseline_mse"], .01), mse


def distill_evaluate(backbone, model, episodes, targets, references, root, step):
    model.eval()
    rows = []
    for episode, target in zip(episodes, targets):
        reference = references[episode.task_id]
        with torch.no_grad():
            normalized, mse = function_loss(model, episode, target)
            single = float(diagnostic.surrogate_nll(backbone, model, episode))
            logits = pilot.forward_views(backbone, model, episode,
                                         view_index=diagnostic.training_view(episode))
            teacher_probability = target["logits"].to(logits.device).float().softmax(dim=-1)
            kl = float(F.kl_div(logits.float().log_softmax(dim=-1), teacher_probability,
                               reduction="sum") / teacher_probability.shape[1])
            branch = diagnostic.transform_diagnostics(model, episode)
        ensemble = pilot.episode_metrics(backbone, model, episode)["nll"]
        row = dict(step=step, task_id=episode.task_id, useful_teacher=int(target["useful"]),
                   teacher_nll=target["nll"], identity_nll=float(reference["identity_surrogate_nll"]),
                   single_nll=single, single_gain=gain(single, float(reference["identity_surrogate_nll"])),
                   ensemble_nll=ensemble, ordinary_nll=float(reference["ordinary_ensemble_nll"]),
                   function_mse=float(mse), normalized_function_mse=float(normalized),
                   relative_function_mse=float(mse) / max(target["baseline_mse"], 1e-12),
                   prediction_kl=kl, **branch)
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError("nonfinite distillation evaluation")
        pilot.csv_append(root / "evaluation_tasks.csv", row)
        rows.append(row)
    useful = [row for row in rows if row["useful_teacher"]]
    single = summarize_losses([r["single_nll"] for r in rows], [r["identity_nll"] for r in rows])
    ordinary = summarize_losses([r["ensemble_nll"] for r in rows], [r["ordinary_nll"] for r in rows])
    summary = dict(step=step, tasks=len(rows), useful_teachers=len(useful),
                   function_mse=float(np.mean([r["function_mse"] for r in rows])),
                   normalized_function_mse=float(np.mean([r["normalized_function_mse"] for r in rows])),
                   useful_relative_function_mse=float(np.mean([r["relative_function_mse"] for r in useful])),
                   useful_gain_recovery=float(np.mean([
                       (r["identity_nll"] - r["single_nll"]) / (r["identity_nll"] - r["teacher_nll"])
                       for r in useful])),
                   **{"single_" + k: v for k, v in single.items()},
                   **{"ordinary_" + k: v for k, v in ordinary.items()})
    pilot.csv_append(root / "evaluation.csv", summary)
    print(f"distill step={step} function_error={summary['normalized_function_mse']:.5f} "
          f"useful_recovery={summary['useful_gain_recovery']:.3f} "
          f"single_W/L={single['wins']}/{single['losses']}", flush=True)
    return summary


def distill(backbone, initial, episodes, targets, references, root, args, fingerprint, *, stop_at=None):
    root.mkdir(parents=True, exist_ok=True)
    complete_path, state_path = root / "complete.json", root / "state.pt"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf8"))
        if complete["fingerprint"] != fingerprint:
            raise ValueError("completed distillation fingerprint mismatch")
        return complete
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.distill_lr, weight_decay=1e-4)
    step = best_step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("distillation resume fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_to(optimizer, episodes[0].x_context.device)
        step, best_step = state["step"], state["best_step"]
        best_score, best_model = state["best_score"], state["best_model"]
        last_summary = state["last_summary"]
        for name in ("training.csv", "evaluation.csv", "evaluation_tasks.csv"):
            trim_csv(root / name, step)
    else:
        last_summary = distill_evaluate(backbone, model, episodes, targets, references, root, 0)
        best_score, best_model = last_summary["normalized_function_mse"], pilot.state_cpu(model)

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=fingerprint, step=step, best_step=best_step,
                          best_score=best_score, best_model=best_model, model=pilot.state_cpu(model),
                          optimizer=optimizer.state_dict(), last_summary=last_summary))

    save()
    stop = args.distill_steps if stop_at is None else min(args.distill_steps, stop_at)
    if stop < step:
        raise ValueError("stop precedes distillation saved state")
    for step in range(step + 1, stop + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in diagnostic.batch_indices(step, len(episodes)):
            loss, _ = function_loss(model, episodes[index], targets[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite distillation loss")
            (loss / 4).backward()
            losses.append(float(loss.detach()))
        gradients = diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite distillation gradient")
        optimizer.step()
        pilot.csv_append(root / "training.csv", dict(step=step, mean_function_loss=float(np.mean(losses)),
                         preclip_gradient_norm=norm, **gradients))
        if step % args.distill_evaluate_every == 0 or step == args.distill_steps or step == stop:
            last_summary = distill_evaluate(backbone, model, episodes, targets, references, root, step)
            score = last_summary["normalized_function_mse"]
            if score < best_score:
                best_score, best_step, best_model = score, step, pilot.state_cpu(model)
            save()
        elif step % 50 == 0:
            save()
            print(f"distill step={step}/{args.distill_steps} fit_mse={np.mean(losses):.5f}", flush=True)
    if step != args.distill_steps:
        return None
    model.load_state_dict(best_model)
    selected_summary = distill_evaluate(backbone, model, episodes, targets, references,
                                        root / "selected_report", best_step)
    complete = dict(fingerprint=fingerprint, steps=step, selected_step=best_step,
                    final=last_summary, selected=selected_summary,
                    selection="fitting function error; no validation or test selection")
    pilot.atomic_save(root / "selected.pt", dict(model=best_model, fingerprint=fingerprint))
    pilot.json_write(complete_path, complete)
    return complete


def source_data(args, backbone_hash):
    source = args.source_dir
    config = json.loads((source / "config.json").read_text(encoding="utf8"))
    complete = json.loads((source / "complete.json").read_text(encoding="utf8"))
    manifest = pilot.read_manifest(source)
    if config["backbone_hash"] != backbone_hash:
        raise ValueError("source backbone differs")
    expected = {"model_hash": Path(inspect.getfile(JointPreprocessor)),
                "runner_hash": Path(diagnostic.__file__), "inference_helper_hash": Path(pilot.__file__)}
    for field, path in expected.items():
        if config[field] != pilot.hash_file(path):
            raise ValueError(f"source {field} differs from current implementation")
    if config["manifest_sha256"] != pilot.hash_file(source / "manifest.json"):
        raise ValueError("source manifest hash mismatch")
    if config["model_seed"] != 0 or complete["steps_completed"] != config["steps"]:
        raise ValueError("source must be a completed seed-zero diagnostic")
    bank = pilot.load_bank(source, manifest, "train")
    if len(bank) < 4 or len(bank) % 4:
        raise ValueError("fitting bank size must be a positive multiple of four")
    references = {int(r["task_id"]): r for r in read_rows(source / "references.csv") if r["panel"] == "train"}
    history = {}
    for row in read_rows(source / "evaluation_tasks.csv"):
        if row["panel"] == "train":
            history.setdefault(int(row["task_id"]), {})[int(row["step"])] = row
    if set(references) != {e.task_id for e in bank} or set(history) != set(references):
        raise ValueError("source task IDs do not match the bank")
    for e in bank:
        filtered = pilot.filtered_episode(e)
        ref = references[e.task_id]
        if int(ref["view_index"]) != diagnostic.training_view(filtered):
            raise ValueError("source view schedule differs")
        if int(ref["n_features"]) != filtered.x_context.shape[-1]:
            raise ValueError("source feature filtering differs")
        if not {0, complete["steps_completed"], complete["selected_step"]} <= set(history[e.task_id]):
            raise ValueError("source is missing comparison checkpoints")
    return bank, references, history, config, complete


def run(args):
    if min(args.fit_steps, args.fit_evaluate_every, args.distill_steps, args.distill_evaluate_every,
           args.min_useful_tasks) <= 0 or not 0 < args.teacher_gain < 1 or args.distill_lr <= 0:
        raise ValueError("invalid diagnostic settings")
    if not args.fit_lrs or min(args.fit_lrs) <= 0 or len(set(args.fit_lrs)) != len(args.fit_lrs):
        raise ValueError("invalid direct learning-rate grid")
    root = args.output_dir
    if root.resolve() == args.source_dir.resolve() or args.source_dir.resolve() in root.resolve().parents:
        raise ValueError("new results must not modify the source run")
    config_path = root / "config.json"
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("output exists; use --resume")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("TabICL must be frozen")
    backbone.train()
    bank, references, history, source_config, source_complete = source_data(args, backbone_hash)
    validate_episode_classes(bank, backbone.max_classes)
    episodes = [pilot.on_device(pilot.filtered_episode(e), device) for e in bank]
    source_files = ("manifest.json", "config.json", "complete.json", "references.csv", "evaluation_tasks.csv")
    settings = dict(format_version=1, source_hashes={name: pilot.hash_file(args.source_dir / name) for name in source_files},
                    bank_sha256=pilot.read_manifest(args.source_dir)["banks"]["train"]["sha256"],
                    backbone_hash=backbone_hash, model_hash=source_config["model_hash"],
                    runner_hash=pilot.hash_file(Path(__file__)), source_runner_hash=source_config["runner_hash"],
                    inference_hash=source_config["inference_helper_hash"], model_seed=0,
                    fit_lrs=args.fit_lrs, fit_steps=args.fit_steps, fit_evaluate_every=args.fit_evaluate_every,
                    distill_steps=args.distill_steps, distill_evaluate_every=args.distill_evaluate_every,
                    distill_lr=args.distill_lr, teacher_gain=args.teacher_gain, min_useful_tasks=args.min_useful_tasks,
                    weight_decay=1e-4, gradient_clip=1.0, function_scale_floor=.01,
                    source_visits_per_task=source_config["steps"] * source_config["tasks_per_update"] / len(bank))
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf8"))["fingerprint"] != fingerprint:
            raise ValueError("root resume fingerprint mismatch")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        pilot.json_write(config_path, dict(**settings, fingerprint=fingerprint, source_revision=revision,
                         source_dir=str(args.source_dir), limitation="seen fitting tasks; no held-out query or task score"))
    torch.manual_seed(0)
    initial = JointPreprocessor("joint").to(device)
    fits, comparison = {}, []
    for episode in episodes:
        ref = references[episode.task_id]
        fits[episode.task_id] = []
        for index, lr in enumerate(args.fit_lrs):
            destination = root / "fits" / str(episode.task_id) / f"lr{index}"
            fit = fit_task(backbone, initial, episode, float(ref["identity_surrogate_nll"]),
                           destination, args, lr, fingerprint + f":{episode.task_id}:{index}")
            fits[episode.task_id].append((fit, destination))
            source_history = history[episode.task_id]
            comparison.append(dict(task_id=episode.task_id, lr=lr, primary_rate=int(index == 0),
                direct_final_nll=fit["final_nll"], direct_best_nll=fit["best_nll"], direct_best_step=fit["best_step"],
                identity_nll=float(ref["identity_surrogate_nll"]),
                hyper_final_nll=float(source_history[source_complete["steps_completed"]]["surrogate_nll"]),
                hyper_best_fitting_nll=min(float(row["surrogate_nll"]) for row in source_history.values()),
                hyper_validation_selected_nll=float(source_history[source_complete["selected_step"]]["surrogate_nll"]),
                direct_selected_ensemble_nll=fit["selected_ensemble_nll"],
                ordinary_nll=float(ref["ordinary_ensemble_nll"])))
    # Rebuild this small derived table from completed fits, avoiding resume duplicates.
    destination = root / "fitting_comparison.csv"
    with destination.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    primary = [row for row in comparison if row["primary_rate"]]
    reference = [row["identity_nll"] for row in primary]
    summary = dict(fingerprint=fingerprint, tasks=len(episodes),
                   primary_rate=args.fit_lrs[0],
                   direct_primary_final=summarize_losses([r["direct_final_nll"] for r in primary], reference),
                   direct_primary_best=summarize_losses([r["direct_best_nll"] for r in primary], reference),
                   hyper_final=summarize_losses([r["hyper_final_nll"] for r in primary], reference),
                   hyper_best_per_task_hindsight=summarize_losses([r["hyper_best_fitting_nll"] for r in primary], reference),
                   hyper_validation_selected=summarize_losses([r["hyper_validation_selected_nll"] for r in primary], reference),
                   selection_warning="best fitting steps/rates use reused labels, not a deployable policy")
    teacher_path = root / "teachers.pt"
    if teacher_path.exists():
        cache = torch.load(teacher_path, map_location="cpu", weights_only=True)
        if cache["fingerprint"] != fingerprint:
            raise ValueError("teacher fingerprint mismatch")
        targets = cache["targets"]
    else:
        targets = []
        for episode in episodes:
            fit, destination = min(fits[episode.task_id], key=lambda pair: pair[0]["best_nll"])
            teacher = independent_map(initial, episode)
            useful = fit["best_gain"] >= args.teacher_gain
            if useful:
                payload = torch.load(destination / "selected.pt", map_location=device, weights_only=True)
                teacher.load_state_dict(payload["model"])
            view_index = diagnostic.training_view(episode)
            slot = pilot.view_specs(episode.x_context.shape[-1], episode.n_classes)[view_index][0]
            with torch.no_grad():
                p = teacher.generate(episode.x_context, episode.y_context)
                x = torch.cat((episode.x_context, episode.x_query), dim=1)
                transformed = teacher.apply(x, p, slot)
                baseline = (x - p.location[:, None]) / p.scale[:, None]
                logits = pilot.forward_views(backbone, teacher, episode, view_index=view_index)
                nll = float(F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten()))
            if useful and abs(nll - fit["best_nll"]) > 1e-4 * (1 + nll):
                raise ValueError("selected teacher does not replay its fitted loss")
            targets.append(dict(task_id=episode.task_id, slot=slot, useful=useful,
                                transformed=transformed.cpu(), logits=logits.cpu(), nll=nll,
                                baseline_mse=float((transformed - baseline).square().mean()),
                                selected_lr=fit["lr"] if useful else None,
                                selected_step=fit["best_step"] if useful else 0,
                                selection="fitting gain >= threshold; otherwise initialized identity"))
        pilot.atomic_save(teacher_path, dict(fingerprint=fingerprint, targets=targets))
    if [t["task_id"] for t in targets] != [e.task_id for e in episodes]:
        raise ValueError("teacher ordering differs from the fitting bank")
    summary["useful_teachers"] = sum(t["useful"] for t in targets)
    summary["teacher_grid_best"] = summarize_losses([t["nll"] for t in targets], reference)
    pilot.json_write(root / "fitting_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["useful_teachers"] >= args.min_useful_tasks:
        distill_fingerprint = fingerprint + ":" + pilot.hash_file(teacher_path)
        teaching = distill(backbone, initial, episodes, targets, references, root / "distillation",
                           args, distill_fingerprint)
    else:
        teaching = dict(skipped=True, reason=f"too few fitting teachers with gain >= {args.teacher_gain:g}",
                        useful_teachers=summary["useful_teachers"], minimum=args.min_useful_tasks)
    pilot.json_write(root / "complete.json", dict(fingerprint=fingerprint, fitting=summary, teaching=teaching,
                     limitation="all selections and scores use fitting examples; no zero-shot claim"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fit-lrs", nargs="+", type=float, default=[.001, .003, .01])
    parser.add_argument("--fit-steps", type=int, default=250)
    parser.add_argument("--fit-evaluate-every", type=int, default=25)
    parser.add_argument("--distill-steps", type=int, default=2000)
    parser.add_argument("--distill-evaluate-every", type=int, default=100)
    parser.add_argument("--distill-lr", type=float, default=.001)
    parser.add_argument("--teacher-gain", type=float, default=.01)
    parser.add_argument("--min-useful-tasks", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
