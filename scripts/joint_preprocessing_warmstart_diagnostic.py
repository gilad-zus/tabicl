"""Compare continued function teaching with downstream-loss fine-tuning.

Both arms start from one completed taught checkpoint. Training reuses fitting
labels; the separate, previously inspected synthetic bank is development
validation. Neither panel is untouched final-test evidence.
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
from torch.nn import functional as F

from scripts import joint_preprocessing_fitting_capacity_diagnostic as capacity
from scripts import joint_preprocessing_learning_diagnostic as diagnostic
from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts.hyperspline_synthetic_train import validate_episode_classes
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor


ARMS = ("imitation", "prediction")
LOGS = ("training.csv", "evaluation.csv", "evaluation_tasks.csv")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def source_data(args, backbone_hash):
    """Verify the exact teacher run, checkpoint, banks and cached references."""
    source = args.source_dir
    config = read_json(source / "config.json")
    complete = read_json(source / "complete.json")
    if config["backbone_hash"] != backbone_hash:
        raise ValueError("teacher run backbone differs")
    if config["runner_hash"] != pilot.hash_file(Path(capacity.__file__)):
        raise ValueError("teacher runner differs from the saved implementation")
    if complete["fingerprint"] != config["fingerprint"] or complete["teaching"].get("skipped"):
        raise ValueError("a completed teaching run is required")
    for name, expected in config["source_hashes"].items():
        if pilot.hash_file(args.bank_dir / name) != expected:
            raise ValueError(f"teacher source hash mismatch: {name}")
    train, _, _, bank_config, _ = capacity.source_data(
        argparse.Namespace(source_dir=args.bank_dir), backbone_hash)
    manifest = pilot.read_manifest(args.bank_dir)
    validation = pilot.load_bank(args.bank_dir, manifest, "validation")
    banks = {"train": train, "validation": validation}
    ids = {panel: {e.task_id for e in bank} for panel, bank in banks.items()}
    if not validation or ids["train"] & ids["validation"]:
        raise ValueError("training and validation task IDs must be separate")
    refs = {}
    for row in capacity.read_rows(args.bank_dir / "references.csv"):
        key = (row["panel"], int(row["task_id"]))
        if key in refs:
            raise ValueError("duplicate reference task")
        refs[key] = {name: float(value) for name, value in row.items() if name != "panel"}
    if set(refs) != {(panel, e.task_id) for panel, bank in banks.items() for e in bank}:
        raise ValueError("reference task IDs differ from the banks")
    for panel, bank in banks.items():
        for e in bank:
            filtered = pilot.filtered_episode(e)
            ref = refs[(panel, e.task_id)]
            if int(ref["view_index"]) != diagnostic.training_view(filtered):
                raise ValueError("reference view schedule differs")
            if (int(ref["n_features"]), int(ref["n_context"]), int(ref["n_query"])) != (
                filtered.x_context.shape[-1], filtered.x_context.shape[1], filtered.x_query.shape[1]):
                raise ValueError("reference episode dimensions differ")
            if not all(math.isfinite(v) for v in ref.values()):
                raise ValueError("nonfinite source reference")
    teacher_path = source / "teachers.pt"
    cache = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if cache["fingerprint"] != config["fingerprint"]:
        raise ValueError("teacher cache fingerprint mismatch")
    targets = cache["targets"]
    if [t["task_id"] for t in targets] != [e.task_id for e in train]:
        raise ValueError("teacher ordering differs from the fitting bank")
    for e, target in zip(train, targets):
        filtered = pilot.filtered_episode(e)
        view = diagnostic.training_view(filtered)
        slot = pilot.view_specs(filtered.x_context.shape[-1], e.n_classes)[view][0]
        shape = (1, e.x_context.shape[1] + e.x_query.shape[1], filtered.x_context.shape[-1])
        if target["slot"] != slot or tuple(target["transformed"].shape) != shape:
            raise ValueError("teacher function target differs from its task/view")
        if tuple(target["logits"].shape) != (1, e.x_query.shape[1], e.n_classes):
            raise ValueError("teacher logits differ from their task")
        if not all(torch.isfinite(target[name]).all() for name in ("transformed", "logits")):
            raise ValueError("nonfinite teacher target")
        if not math.isfinite(target["nll"]) or not math.isfinite(target["baseline_mse"]) or target["baseline_mse"] < 0:
            raise ValueError("invalid teacher scalar")
        if target["useful"] and target["nll"] >= refs[("train", e.task_id)]["identity_surrogate_nll"]:
            raise ValueError("useful teacher has no fitting benefit")
    taught = read_json(source / "distillation" / "complete.json")
    expected_fingerprint = config["fingerprint"] + ":" + pilot.hash_file(teacher_path)
    selected = torch.load(source / "distillation" / "selected.pt", map_location="cpu", weights_only=True)
    if taught != complete["teaching"] or taught["fingerprint"] != expected_fingerprint or selected["fingerprint"] != expected_fingerprint:
        raise ValueError("taught checkpoint fingerprint mismatch")
    if taught["steps"] != config["distill_steps"] or not 0 <= taught["selected_step"] <= taught["steps"]:
        raise ValueError("teaching run is incomplete")
    if any(not torch.isfinite(value).all() for value in selected["model"].values()):
        raise ValueError("nonfinite taught weights")
    return banks, refs, targets, selected["model"], config, bank_config, taught


def evaluate(backbone, model, banks, targets, refs, root, step, arm):
    model.eval()
    target_by_id = {t["task_id"]: t for t in targets}
    summaries = {}
    for panel, bank in banks.items():
        rows = []
        for episode in bank:
            ref = refs[(panel, episode.task_id)]
            with torch.no_grad():
                single = float(diagnostic.surrogate_nll(backbone, model, episode))
                branch = diagnostic.transform_diagnostics(model, episode)
                teacher_fields = dict(function_mse=None, normalized_function_mse=None,
                                      teacher_nll=None, teacher_gain_recovery=None,
                                      prediction_kl=None, useful_teacher=None)
                if panel == "train":
                    target = target_by_id[episode.task_id]
                    normalized, mse = capacity.function_loss(model, episode, target)
                    logits = pilot.forward_views(backbone, model, episode,
                                                 view_index=diagnostic.training_view(episode))
                    probability = target["logits"].to(logits.device).float().softmax(-1)
                    kl = float(F.kl_div(logits.float().log_softmax(-1), probability,
                                        reduction="sum") / probability.shape[1])
                    recovery = ((ref["identity_surrogate_nll"] - single) /
                                (ref["identity_surrogate_nll"] - target["nll"])) if target["useful"] else None
                    teacher_fields.update(function_mse=float(mse), normalized_function_mse=float(normalized),
                                          teacher_nll=target["nll"], teacher_gain_recovery=recovery,
                                          prediction_kl=kl, useful_teacher=int(target["useful"]))
            ensemble = pilot.episode_metrics(backbone, model, episode)["nll"]
            row = dict(step=step, arm=arm, panel=panel, task_id=episode.task_id,
                       single_nll=single, identity_single_nll=ref["identity_surrogate_nll"],
                       ensemble_nll=ensemble, identity_ensemble_nll=ref["identity_ensemble_nll"],
                       ordinary_nll=ref["ordinary_ensemble_nll"], **teacher_fields, **branch)
            if not all(math.isfinite(v) for k, v in row.items() if k not in ("arm", "panel") and v is not None):
                raise FloatingPointError("nonfinite evaluation")
            pilot.csv_append(root / "evaluation_tasks.csv", row)
            rows.append(row)
        single = capacity.summarize_losses([r["single_nll"] for r in rows], [r["identity_single_nll"] for r in rows])
        ordinary = capacity.summarize_losses([r["ensemble_nll"] for r in rows], [r["ordinary_nll"] for r in rows])
        identity = capacity.summarize_losses([r["ensemble_nll"] for r in rows], [r["identity_ensemble_nll"] for r in rows])
        ratios = [(r["ensemble_nll"] + 1e-4) / (r["ordinary_nll"] + 1e-4) for r in rows]
        useful = [r for r in rows if r["useful_teacher"]]
        summary = dict(step=step, arm=arm, panel=panel, tasks=len(rows),
                       **{"single_" + k: v for k, v in single.items()},
                       **{"ordinary_" + k: v for k, v in ordinary.items()},
                       **{"identity_ensemble_" + k: v for k, v in identity.items()},
                       ordinary_median_gain=float(np.median([1 - r for r in ratios])),
                       ordinary_max_harm=max(0., max(ratios) - 1),
                       **{f"ordinary_harms_over_{pct}pct": sum(r > 1 + pct / 100 for r in ratios)
                          for pct in (1, 5, 10)},
                       function_mse=float(np.mean([r["function_mse"] for r in rows])) if panel == "train" else None,
                       normalized_function_mse=float(np.mean([r["normalized_function_mse"] for r in rows])) if panel == "train" else None,
                       prediction_kl=float(np.mean([r["prediction_kl"] for r in rows])) if panel == "train" else None,
                       useful_teachers=len(useful),
                       teacher_gain_recovery_mean=float(np.mean([r["teacher_gain_recovery"] for r in useful])) if useful else None,
                       teacher_gain_recovery_median=float(np.median([r["teacher_gain_recovery"] for r in useful])) if useful else None,
                       teachers_half_recovered=sum(r["teacher_gain_recovery"] >= .5 for r in useful),
                       **{k: float(np.mean([r[k] for r in rows])) for k in branch})
        pilot.csv_append(root / "evaluation.csv", summary)
        summaries[panel] = summary
        print(f"arm={arm} step={step} {panel} single_nll={single['mean_nll']:.6f} "
              f"ordinary_gain={100 * ordinary['geometric_gain']:.3f}% "
              f"W/L={ordinary['wins']}/{ordinary['losses']} "
              f"harms>5%={summary['ordinary_harms_over_5pct']} "
              f"function_mse={summary['normalized_function_mse']} "
              f"teacher_recovery={summary['teacher_gain_recovery_mean']}", flush=True)
    return summaries


def selection_score(summary):
    # Minimize the mean log ratio against ordinary TabICL, equal task weighting.
    return math.log1p(-summary["validation"]["ordinary_geometric_gain"])


def train_arm(backbone, initial, banks, targets, refs, root, args, fingerprint,
              arm, source_step, *, stop_at=None, expected_initial=None):
    if arm not in ARMS:
        raise ValueError("unknown objective")
    root.mkdir(parents=True, exist_ok=True)
    complete_path, state_path = root / "complete.json", root / "state.pt"
    if complete_path.exists():
        complete = read_json(complete_path)
        if complete["fingerprint"] != fingerprint:
            raise ValueError("completed arm fingerprint mismatch")
        return complete
    torch.manual_seed(0)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                  betas=(.9, .999), eps=1e-8)
    step = best_step = clipped = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("arm resume fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        capacity.optimizer_to(optimizer, next(model.parameters()).device)
        step, best_step, clipped = state["step"], state["best_step"], state["clipped_updates"]
        best_score, best_model, last = state["best_score"], state["best_model"], state["last_summary"]
        initial_summary = state["initial_summary"]
        torch.set_rng_state(state["rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for name in LOGS:
            capacity.trim_csv(root / name, step)
    else:
        last = initial_summary = evaluate(backbone, model, banks, targets, refs, root, 0, arm)
        if expected_initial is not None:
            for name in ("single_mean_nll", "normalized_function_mse"):
                if not math.isclose(last["train"][name], expected_initial[name], rel_tol=1e-4, abs_tol=1e-5):
                    raise ValueError(f"taught checkpoint does not replay source {name}")
        best_score, best_model = selection_score(last), pilot.state_cpu(model)

    def save():
        pilot.atomic_save(state_path, dict(fingerprint=fingerprint, step=step, best_step=best_step,
                          clipped_updates=clipped, model=pilot.state_cpu(model), optimizer=optimizer.state_dict(),
                          best_model=best_model, best_score=best_score, last_summary=last,
                          initial_summary=initial_summary, rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []))

    save()
    stop = args.steps if stop_at is None else min(args.steps, stop_at)
    if stop < step:
        raise ValueError("stop precedes saved state")
    started = time.perf_counter()
    for step in range(step + 1, stop + 1):
        model.train()
        backbone.train()
        optimizer.zero_grad(set_to_none=True)
        losses, function_losses = [], []
        for index in diagnostic.batch_indices(source_step + step, len(banks["train"])):
            episode = banks["train"][index]
            if arm == "imitation":
                loss, _ = capacity.function_loss(model, episode, targets[index])
                function_loss = float(loss.detach())
            else:
                loss = diagnostic.surrogate_nll(backbone, model, episode)
                with torch.no_grad():
                    function_loss = float(capacity.function_loss(model, episode, targets[index])[0])
            if not torch.isfinite(loss) or not math.isfinite(function_loss):
                raise FloatingPointError("nonfinite training loss")
            (loss / 4).backward()
            losses.append(float(loss.detach()))
            function_losses.append(function_loss)
        gradients = diagnostic.gradient_norms(model)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm) or not all(math.isfinite(v) for v in gradients.values()):
            raise FloatingPointError("nonfinite training gradient")
        clipped += int(norm > 1)
        optimizer.step()
        pilot.csv_append(root / "training.csv", dict(step=step, source_step=source_step + step,
                         arm=arm, tasks_seen=step * 4, mean_objective=float(np.mean(losses)),
                         mean_function_loss=float(np.mean(function_losses)), preclip_gradient_norm=norm,
                         clip_factor=min(1., 1 / (norm + 1e-6)), clipped_fraction=clipped / step,
                         lr=optimizer.param_groups[0]["lr"], **gradients))
        evaluated = step % args.evaluate_every == 0 or step == args.steps
        if evaluated:
            last = evaluate(backbone, model, banks, targets, refs, root, step, arm)
            score = selection_score(last)
            if score < best_score:
                best_score, best_step, best_model = score, step, pilot.state_cpu(model)
        if evaluated or step % args.save_every == 0 or step == stop:
            save()
        if step % args.save_every == 0 or step == stop:
            print(f"arm={arm} step={step}/{args.steps} objective={np.mean(losses):.6f} "
                  f"gradient_norm={norm:.4f} clipped_fraction={clipped / step:.3f} "
                  f"best_validation_log_ratio={best_score:.6f}@{best_step} "
                  f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    if step != args.steps:
        return None
    model.load_state_dict(best_model)
    selected = evaluate(backbone, model, banks, targets, refs, root / "selected_report", best_step, arm)
    complete = dict(fingerprint=fingerprint, arm=arm, steps_completed=step, source_step=source_step,
                    selected_step=best_step, validation_log_ratio=best_score, initial=initial_summary,
                    final=last, selected=selected, clipped_updates=clipped,
                    selection="earliest best validation mean ensemble log ratio vs ordinary, including step zero",
                    limitation="training labels reused; validation tasks previously inspected; no final test")
    pilot.atomic_save(root / "selected.pt", dict(model=best_model, selected_step=best_step,
                      validation_log_ratio=best_score, fingerprint=fingerprint))
    pilot.json_write(complete_path, complete)
    return complete


def run(args):
    if min(args.steps, args.evaluate_every, args.save_every) <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid training schedule")
    root = args.output_dir.resolve()
    for source in (args.source_dir.resolve(), args.bank_dir.resolve()):
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("output and source directories must be separate")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("output exists; use --resume")
    device = torch.device(args.device)
    backbone, _, backbone_hash = pilot.load_frozen(args, device)
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("TabICL must be frozen")
    backbone.train()
    banks, refs, targets, weights, config, bank_config, taught = source_data(args, backbone_hash)
    for panel, bank in banks.items():
        validate_episode_classes(bank, backbone.max_classes)
        banks[panel] = [pilot.on_device(pilot.filtered_episode(e), device) for e in bank]
    files = ("config.json", "complete.json", "teachers.pt", "distillation/complete.json", "distillation/selected.pt")
    settings = dict(format_version=1, arms=ARMS, source_hashes={name: pilot.hash_file(args.source_dir / name) for name in files},
                    bank_source_hashes=config["source_hashes"],
                    bank_hashes={name: info["sha256"] for name, info in pilot.read_manifest(args.bank_dir)["banks"].items()},
                    model_hash=config["model_hash"], capacity_runner_hash=config["runner_hash"],
                    diagnostic_runner_hash=bank_config["runner_hash"], inference_hash=config["inference_hash"],
                    runner_hash=pilot.hash_file(Path(__file__)), backbone_hash=backbone_hash,
                    source_selected_step=taught["selected_step"], source_teaching_updates=taught["steps"],
                    steps=args.steps, evaluate_every=args.evaluate_every, save_every=args.save_every,
                    lr=args.lr, weight_decay=1e-4, gradient_clip=1., tasks_per_update=4,
                    optimizer_reset=True, model_seed=0, device=str(device), function_scale_floor=.01,
                    training_tasks=len(banks["train"]), validation_tasks=len(banks["validation"]))
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    config_path = root / "config.json"
    if config_path.exists():
        if read_json(config_path)["fingerprint"] != fingerprint:
            raise ValueError("root resume fingerprint mismatch")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        pilot.json_write(config_path, dict(**settings, fingerprint=fingerprint, source_revision=revision,
                         source_dir=str(args.source_dir), bank_dir=str(args.bank_dir)))
    torch.manual_seed(0)
    initial = JointPreprocessor("joint").to(device)
    initial.load_state_dict(weights)
    completed = {}
    for arm in ARMS:
        completed[arm] = train_arm(backbone, initial, banks, targets, refs, root / arm, args,
                                   fingerprint + ":" + arm, arm, taught["selected_step"],
                                   expected_initial=taught["selected"])
    pilot.json_write(root / "complete.json", dict(fingerprint=fingerprint, arms=completed,
                     limitation="repeated fitting tasks and inspected development validation; no untouched test"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
