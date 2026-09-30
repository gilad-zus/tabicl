"""Repeated-task learning diagnostic for one shared joint preprocessing model.

Run with ``python -m scripts.joint_preprocessing_learning_diagnostic``.
Training-panel query labels are reused for fitting; its scores measure fit,
not generalization. Only independent validation tasks select checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import (
    generate_scheduled_episodes, save_episode_bank, validate_episode_classes,
)
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


def prepare(args):
    root = args.output_dir
    if pilot.manifest_path(root).exists():
        return
    if any((root / "banks" / f"{name}.pt").exists() for name in ("train", "validation")):
        raise FileExistsError("partial bank exists; use a fresh directory")
    settings = dict(prior_type="mix_scm", min_features=5, max_features=100,
                    max_classes=10, prior_n_jobs=1,
                    synthetic_observation_mode="coverage_expanded")
    generation = argparse.Namespace(**settings, train_seed=0)
    banks = {}
    for name, count, seed, offset in (("train", args.train_tasks, 181001, 4_000_000_000),
                                      ("validation", args.validation_tasks, 182001, 5_000_000_000)):
        episodes = generate_scheduled_episodes(
            generation, count, source_seed=seed, task_offset=offset,
            device=torch.device("cpu"), sequence_lengths=pilot.LENGTHS,
            context_fractions=pilot.FRACTIONS, observation_mode="coverage_expanded")
        destination = pilot.bank_path(root, name)
        save_episode_bank(destination, episodes, source_seed=seed)
        banks[name] = dict(path=destination.relative_to(root).as_posix(), count=count,
                           source_seed=seed, task_offset=offset, sha256=pilot.hash_file(destination))
    pilot.json_write(pilot.manifest_path(root), dict(
        format_version=1, settings=settings, banks=banks,
        sequence_lengths=pilot.LENGTHS, context_fractions=pilot.FRACTIONS,
        task_order_seed=183001, model_seed=0,
        training_panel="fixed query labels reused for fitting; not generalization evidence"))
    print(f"Prepared independent fixed banks at {root}", flush=True)


def batch_indices(step: int, count: int) -> list[int]:
    if step < 1 or count < 4 or count % 4:
        raise ValueError("training bank must have a positive multiple of four tasks")
    epoch, batch = divmod(step - 1, count // 4)
    order = np.random.default_rng(183001 + epoch).permutation(count)
    return order[4 * batch:4 * batch + 4].tolist()


def training_view(episode) -> int:
    return random.Random(episode.task_id).randrange(
        len(pilot.view_specs(episode.x_context.shape[-1], episode.n_classes)))


def surrogate_nll(backbone, model, episode):
    """Exact original training path: fixed task view, raw logits, no temperature."""
    logits = pilot.forward_views(backbone, model, episode, view_index=training_view(episode))
    return F.cross_entropy(logits.flatten(0, 1), episode.y_query.flatten())


def gradient_norms(model):
    grouped = {}
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        grouped.setdefault(group, 0.0)
        if parameter.grad is not None:
            grouped[group] += float(parameter.grad.detach().square().sum())
    return {"grad_" + group: math.sqrt(value) for group, value in sorted(grouped.items())}


def transform_diagnostics(model, episode):
    """Measure effective corrections from the actual apply() implementation."""
    raw_spline = []
    hook = model.spline_head.register_forward_hook(lambda module, inputs, output: raw_spline.append(output))
    try:
        p = model.generate(episode.x_context, episode.y_context)
    finally:
        hook.remove()
    x = torch.cat((episode.x_context, episode.x_query), dim=1)
    z = (x - p.location[:, None]) / p.scale[:, None]
    output = {}
    for slot in (0, 1):
        affine = model.apply(x, replace(p, spline_controls=None,
                                       neural_first_weight=None, mixing=None), slot)
        spline = model.apply(x, replace(p, neural_first_weight=None, mixing=None), slot)
        neural = model.apply(x, replace(p, spline_controls=None, mixing=None), slot)
        unmixed = model.apply(x, replace(p, mixing=None), slot)
        full = model.apply(x, p, slot)
        prefix = f"slot{slot}_"
        for name, delta in (("affine", affine - z), ("spline", spline - affine),
                            ("neural", neural - affine), ("mixing", full - unmixed),
                            ("total", full - z)):
            output[prefix + name + "_rms"] = float(delta.square().mean().sqrt())
        output[prefix + "shift_bound_fraction"] = float((p.shift[:, slot].abs() > .95).float().mean())
        output[prefix + "log_scale_bound_fraction"] = float((p.log_scale[:, slot].abs() > .95).float().mean())
        output[prefix + "spline_domain_clip_fraction"] = float((affine.abs() >= 4).float().mean())
        output[prefix + "spline_gap_bound_fraction"] = float((raw_spline[0][:, slot].tanh().abs() > .95).float().mean())
        for name, gate in (("spline", p.spline_gate[:, slot]), ("neural", p.neural_gate[:, slot])):
            output[prefix + name + "_gate_mean"] = float(gate.mean())
            output[prefix + name + "_gate_low_fraction"] = float((gate < .05).float().mean())
            output[prefix + name + "_gate_high_fraction"] = float((gate > .95).float().mean())
        hidden = torch.tanh(affine[..., None] * p.neural_first_weight[:, slot, None]
                            + p.neural_first_bias[:, slot, None])
        output[prefix + "neural_hidden_saturation_fraction"] = float((hidden.abs() > .95).float().mean())
        residual = (hidden * p.neural_last_weight[:, slot, None]).sum(-1) + p.neural_last_bias[:, slot, None]
        output[prefix + "neural_output_saturation_fraction"] = float((torch.tanh(residual).abs() > .95).float().mean())
        norm = torch.linalg.matrix_norm(p.mixing[:, slot], ord="fro")
        output[prefix + "mixing_frobenius_norm"] = float(norm.mean())
        output[prefix + "mixing_bound_fraction"] = float((norm > .095).float().mean())
    if not all(math.isfinite(value) for value in output.values()):
        raise FloatingPointError("nonfinite transform diagnostic")
    return output


def references(backbone, banks, device, root):
    result = {}
    for panel, bank in banks.items():
        for source in bank:
            episode = pilot.on_device(pilot.filtered_episode(source), device)
            with torch.no_grad():
                single = float(surrogate_nll(backbone, None, episode))
            identity = pilot.episode_metrics(backbone, None, episode)["nll"]
            ordinary = pilot.ordinary_episode_metrics(backbone, source)["nll"]
            row = dict(panel=panel, task_id=episode.task_id, view_index=training_view(episode),
                       actual_views=len(pilot.view_specs(episode.x_context.shape[-1], episode.n_classes)),
                       identity_surrogate_nll=single, identity_ensemble_nll=identity,
                       ordinary_ensemble_nll=ordinary,
                       n_context=episode.x_context.shape[1], n_query=episode.x_query.shape[1],
                       n_features=episode.x_context.shape[-1], n_classes=episode.n_classes)
            if not all(math.isfinite(value) for value in (single, identity, ordinary)):
                raise FloatingPointError("nonfinite reference loss")
            result[(panel, episode.task_id)] = row
            pilot.csv_append(root / "references.csv", row)
        print(f"Cached {len(bank)} {panel} identity/ordinary references", flush=True)
    return result


def evaluate(backbone, model, banks, refs, device, root, step):
    model.eval()
    summaries = {}
    for panel, bank in banks.items():
        rows = []
        for source in bank:
            episode = pilot.on_device(pilot.filtered_episode(source), device)
            reference = refs[(panel, episode.task_id)]
            with torch.no_grad():
                single = float(surrogate_nll(backbone, model, episode))
                diagnostics = transform_diagnostics(model, episode)
            ensemble = pilot.episode_metrics(backbone, model, episode)["nll"]
            if not all(math.isfinite(value) for value in (single, ensemble)):
                raise FloatingPointError("nonfinite candidate loss")
            row = dict(step=step, panel=panel, task_id=episode.task_id,
                       surrogate_nll=single, ensemble_nll=ensemble,
                       surrogate_log_ratio=math.log((single + 1e-4) / (reference["identity_surrogate_nll"] + 1e-4)),
                       ensemble_log_ratio=math.log((ensemble + 1e-4) / (reference["identity_ensemble_nll"] + 1e-4)),
                       ordinary_log_ratio=math.log((ensemble + 1e-4) / (reference["ordinary_ensemble_nll"] + 1e-4)),
                       identity_ensemble_nll=reference["identity_ensemble_nll"],
                       ordinary_ensemble_nll=reference["ordinary_ensemble_nll"], **diagnostics)
            pilot.csv_append(root / "evaluation_tasks.csv", row)
            rows.append(row)
        summary = dict(step=step, panel=panel, tasks=len(rows))
        for field in ("surrogate_nll", "ensemble_nll", "surrogate_log_ratio",
                      "ensemble_log_ratio", "ordinary_log_ratio", *diagnostics):
            summary[field] = float(np.mean([row[field] for row in rows]))
        for reference_name in ("identity", "ordinary"):
            deltas = np.asarray([row["ensemble_nll"] - row[reference_name + "_ensemble_nll"] for row in rows])
            summary[reference_name + "_wins"] = int((deltas < 0).sum())
            summary[reference_name + "_losses"] = int((deltas > 0).sum())
            summary[reference_name + "_ties"] = int((deltas == 0).sum())
        pilot.csv_append(root / "evaluation.csv", summary)
        summaries[panel] = summary
        print(f"step={step} {panel} single_view_gain={100 * (1 - math.exp(summary['surrogate_log_ratio'])):.3f}% "
              f"ensemble_gain={100 * (1 - math.exp(summary['ensemble_log_ratio'])):.3f}% "
              f"ordinary_W/L={summary['ordinary_wins']}/{summary['ordinary_losses']}", flush=True)
    return summaries


def trim_logs(root, step):
    import csv
    for name in ("training.csv", "evaluation.csv", "evaluation_tasks.csv"):
        path = root / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            retained = [row for row in reader if int(row["step"]) <= step]
        with path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)


def run(args):
    if args.train_tasks < 4 or args.train_tasks % 4 or args.validation_tasks <= 0:
        raise ValueError("invalid task bank sizes")
    if min(args.steps, args.evaluate_every, args.save_every) <= 0 or args.lr <= 0:
        raise ValueError("invalid training schedule")
    root = args.output_dir
    if any((root / name).exists() for name in ("config.json", "references.csv", "state.pt")) and not args.resume:
        raise FileExistsError("run exists; use --resume")
    prepare(args)
    manifest = pilot.read_manifest(root)
    if any(manifest["banks"][name]["count"] != count for name, count in
           (("train", args.train_tasks), ("validation", args.validation_tasks))):
        raise ValueError("requested task count differs from frozen bank")
    banks = {name: pilot.load_bank(root, manifest, name) for name in ("train", "validation")}
    if set(e.task_id for e in banks["train"]) & set(e.task_id for e in banks["validation"]):
        raise ValueError("training/validation task IDs overlap")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("the TabICL backbone must be frozen")
    backbone.train()
    for bank in banks.values():
        validate_episode_classes(bank, backbone.max_classes)
    torch.manual_seed(0)
    model = JointPreprocessor("joint").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                  betas=(.9, .999), eps=1e-8)
    import inspect
    settings = dict(manifest_sha256=pilot.hash_file(pilot.manifest_path(root)),
                    backbone_hash=backbone_hash, steps=args.steps, lr=args.lr,
                    evaluate_every=args.evaluate_every, model_seed=0, tasks_per_update=4,
                    weight_decay=1e-4, gradient_clip=1.0,
                    runner_hash=pilot.hash_file(Path(__file__)),
                    model_hash=pilot.hash_file(Path(inspect.getfile(JointPreprocessor))),
                    inference_helper_hash=pilot.hash_file(Path(pilot.__file__)))
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    state_path = root / "state.pt"
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("resume fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        step, best_score, best_step = state["step"], state["best_score"], state["best_step"]
        best_model, refs, last_summary = state["best_model"], state["references"], state["last_summary"]
        trim_logs(root, step)
    else:
        refs = references(backbone, banks, device, root)
        step = best_step = 0
        last_summary = evaluate(backbone, model, banks, refs, device, root, 0)
        best_score = last_summary["validation"]["ensemble_log_ratio"]
        best_model = pilot.state_cpu(model)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pilot.json_write(root / "config.json", dict(**settings, fingerprint=fingerprint,
                                               source_revision=revision, device=str(device),
                                               train_panel_reuses_fitting_labels=True))

    def save(step):
        pilot.atomic_save(state_path, dict(step=step, fingerprint=fingerprint,
                          model=pilot.state_cpu(model), optimizer=optimizer.state_dict(),
                          best_model=best_model, best_score=best_score, best_step=best_step,
                          references=refs, last_summary=last_summary))

    save(step)
    stop = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    if stop < step or stop < 1:
        raise ValueError("requested stop precedes the saved state or first update")
    started = time.perf_counter()
    for current in range(step + 1, stop + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in batch_indices(current, len(banks["train"])):
            episode = pilot.on_device(pilot.filtered_episode(banks["train"][index]), device)
            loss = surrogate_nll(backbone, model, episode)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at {current}")
            (loss / 4).backward()
            losses.append(float(loss.detach()))
        gradients = gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm) or not all(math.isfinite(v) for v in gradients.values()):
            raise FloatingPointError(f"nonfinite gradient at {current}")
        optimizer.step()
        pilot.csv_append(root / "training.csv", dict(step=current, tasks_seen=4 * current,
                         mean_nll=float(np.mean(losses)), preclip_gradient_norm=norm, **gradients))
        evaluated = current % args.evaluate_every == 0 or current == args.steps
        if evaluated:
            last_summary = evaluate(backbone, model, banks, refs, device, root, current)
            score = last_summary["validation"]["ensemble_log_ratio"]
            if score < best_score:
                best_score, best_step, best_model = score, current, pilot.state_cpu(model)
        if evaluated or current % args.save_every == 0 or current == stop:
            save(current)
        if current % args.save_every == 0 or current == stop:
            print(f"step={current}/{args.steps} train_nll={np.mean(losses):.5f} "
                  f"gradient_norm={norm:.5f} best={best_score:.6f}@{best_step} "
                  f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    if stop == args.steps:
        pilot.atomic_save(root / "selected.pt", dict(model=best_model, selected_step=best_step,
                          validation_score=best_score, fingerprint=fingerprint, **settings))
        pilot.json_write(root / "complete.json", dict(steps_completed=stop, selected_step=best_step,
                         validation_score=best_score, final_panels=last_summary,
                         limitation="repeated training tasks and synthetic validation; no final test"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-tasks", type=int, default=32)
    parser.add_argument("--validation-tasks", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
