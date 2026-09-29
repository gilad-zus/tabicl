"""Synthetic-first joint preprocessing pilot for frozen TabICLv2.

Prepare fixed validation/test banks, train one arm/seed per resumable run, then
report on the test bank only after all checkpoint choices are frozen.
Run as ``python -m scripts.joint_preprocessing_synthetic_pilot``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from scripts.direct_spline_multidataset_headroom import load_backbone
from scripts.hyperspline_synthetic_train import (
    SyntheticEpisode, generate_scheduled_episodes, load_episode_bank,
    save_episode_bank, validate_episode_classes,
)
from tabicl._hyperspline.joint_preprocessing import JointPreprocessor
from tabicl._sklearn.preprocessing import EnsembleGenerator

ARMS = ("restricted", "joint", "no_spline")
LENGTHS = (128, 256, 512, 1024)
FRACTIONS = (0.5, 0.7, 0.85)
FORMAT_VERSION = 1


def json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")
    os.replace(temporary, path)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def csv_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def bank_path(root: Path, name: str) -> Path:
    return root / "banks" / f"{name}.pt"


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def prepare(args: argparse.Namespace) -> None:
    path = manifest_path(args.output_dir)
    if path.exists() or bank_path(args.output_dir, "validation").exists() or bank_path(args.output_dir, "test").exists():
        raise FileExistsError("experiment bank or manifest exists; use a fresh output directory")
    if args.validation_seed == args.test_seed or args.validation_tasks <= 0 or args.test_tasks <= 0:
        raise ValueError("bank seeds must differ and counts must be positive")
    settings = dict(prior_type="mix_scm", min_features=5, max_features=100,
                    max_classes=10, prior_n_jobs=1,
                    synthetic_observation_mode="coverage_expanded")
    generation = argparse.Namespace(**settings, train_seed=0)
    banks = {}
    for name, count, seed, offset in (("validation", args.validation_tasks, args.validation_seed, 2_000_000_000),
                                      ("test", args.test_tasks, args.test_seed, 3_000_000_000)):
        episodes = generate_scheduled_episodes(
            generation, count, source_seed=seed, task_offset=offset, device=torch.device("cpu"),
            sequence_lengths=LENGTHS, context_fractions=FRACTIONS,
            observation_mode=settings["synthetic_observation_mode"],
        )
        destination = bank_path(args.output_dir, name)
        save_episode_bank(destination, episodes, source_seed=seed)
        banks[name] = dict(path=destination.relative_to(args.output_dir).as_posix(), count=count,
                           source_seed=seed, task_offset=offset, sha256=hash_file(destination),
                           context_query_sizes=sorted({f"{e.x_context.shape[1]}/{e.x_query.shape[1]}" for e in episodes}))
    json_write(path, dict(format_version=FORMAT_VERSION, settings=settings,
                          sequence_lengths=LENGTHS, context_fractions=FRACTIONS,
                          banks=banks, train_stream_seeds=(161001, 162001),
                          selection="mean task log((candidate_nll+1e-4)/(matched_identity_nll+1e-4))",
                          reference="frozen tabicl-classifier-v2-20260212.ckpt"))
    print(f"Prepared frozen banks: {path}", flush=True)


def read_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path(root).read_text(encoding="utf8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported manifest version")
    return manifest


def load_bank(root: Path, manifest: dict[str, Any], name: str) -> list[SyntheticEpisode]:
    info = manifest["banks"][name]
    path = root / info["path"]
    if hash_file(path) != info["sha256"]:
        raise ValueError(f"{name} bank hash mismatch")
    return load_episode_bank(path, expected_seed=info["source_seed"], expected_count=info["count"],
                             expected_observation_mode=manifest["settings"]["synthetic_observation_mode"],
                             device=torch.device("cpu"))


@lru_cache(maxsize=2048)
def view_specs(n_features: int, n_classes: int) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    """Use the standard TabICL eight-view shuffle schedule for two numeric slots."""
    generator = EnsembleGenerator(classification=True, n_estimators=8,
                                  norm_methods=["none", "power"], feat_shuffle_method="latin",
                                  class_shuffle_method="shift", random_state=0)
    generator.n_features_in_ = n_features
    generator.n_classes_ = n_classes
    generator.norm_methods_ = ["none", "power"]
    generator.rng_ = random.Random(0)
    configs, _, _ = generator._generate_ensemble()
    return tuple((0 if method == "none" else 1, tuple(map(int, features)), tuple(map(int, labels)))
                 for method, members in configs.items() for features, labels in members)


def filtered_episode(episode: SyntheticEpisode) -> SyntheticEpisode:
    # The baseline's UniqueFeatureFilter removes context-constant columns.
    # Synthetic episodes are finite; the range check is equivalent to >1 unique value.
    x = episode.x_context
    if not torch.isfinite(x).all() or not torch.isfinite(episode.x_query).all():
        raise ValueError("synthetic episode unexpectedly contains nonfinite features")
    keep = x.amax(dim=1).squeeze(0) != x.amin(dim=1).squeeze(0)
    if not bool(keep.any()):
        raise ValueError(f"task {episode.task_id} has no varying context feature")
    from dataclasses import replace
    return replace(episode, x_context=x[..., keep], x_query=episode.x_query[..., keep])


def on_device(e: SyntheticEpisode, device: torch.device) -> SyntheticEpisode:
    from dataclasses import replace
    return replace(e, x_context=e.x_context.to(device), x_query=e.x_query.to(device),
                   y_context=e.y_context.to(device), y_query=e.y_query.to(device))


def _view_logits(backbone, e: SyntheticEpisode, transformed: torch.Tensor,
                 view: tuple[int, tuple[int, ...], tuple[int, ...]]) -> torch.Tensor:
    _, features, labels = view
    feature_index = torch.tensor(features, device=transformed.device)
    class_index = torch.tensor(labels, device=transformed.device)
    backbone.clear_cache()
    if getattr(backbone, "training", True):
        logits = backbone(transformed[..., feature_index], class_index[e.y_context.long()])
    else:
        logits = backbone(transformed[..., feature_index], class_index[e.y_context.long()],
                          feature_shuffles=[list(features)], return_logits=True)
    # Standard TabICL unshuffles with output[..., class_shuffle].
    return logits[..., : e.n_classes][..., class_index]


def forward_views(backbone, model: JointPreprocessor | None, episode: SyntheticEpisode,
                  *, view_index: int | None = None) -> torch.Tensor:
    e = filtered_episode(episode)
    specs = view_specs(e.x_context.shape[-1], e.n_classes)
    chosen = range(len(specs)) if view_index is None else (view_index % len(specs),)
    parameters = model.generate(e.x_context, e.y_context) if model is not None else None
    logits = []
    for index in chosen:
        slot = specs[index][0]
        if model is None:
            context = (e.x_context - e.x_context.mean(dim=1, keepdim=True)) / e.x_context.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            query = (e.x_query - e.x_context.mean(dim=1, keepdim=True)) / e.x_context.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        else:
            context = model.apply(e.x_context, parameters, slot)
            query = model.apply(e.x_query, parameters, slot)
        transformed = torch.cat((context, query), dim=1)
        logits.append(_view_logits(backbone, e, transformed, specs[index]))
    return torch.stack(logits).mean(dim=0) if view_index is None else logits[0]


@contextmanager
def frozen_inference(backbone):
    flags = [(module, module.training) for module in backbone.modules()]
    backbone.eval()
    try:
        yield
    finally:
        for module, was_training in flags:
            module.training = was_training


def episode_metrics(backbone, model: JointPreprocessor | None, episode: SyntheticEpisode) -> dict[str, float]:
    with torch.no_grad(), frozen_inference(backbone):
        logits = forward_views(backbone, model, episode)
        log_prob = F.log_softmax(logits.float() / 0.9, dim=-1)
        labels = episode.y_query.flatten()
        nll = float(F.nll_loss(log_prob.flatten(0, 1), labels))
        accuracy = float((log_prob.argmax(dim=-1).flatten() == labels).float().mean())
    if not math.isfinite(nll):
        raise FloatingPointError(f"nonfinite NLL for task {episode.task_id}")
    auc = float("nan")
    if episode.n_classes == 2 and torch.unique(labels).numel() == 2:
        auc = float(roc_auc_score(labels.cpu().numpy(), log_prob.exp().flatten(0, 1)[:, 1].cpu().numpy()))
    return dict(nll=nll, accuracy=accuracy, auc=auc)


def ordinary_episode_metrics(backbone, episode: SyntheticEpisode) -> dict[str, float]:
    """Use TabICL's standard numeric none/power preprocessors and view schedule."""
    x_context = episode.x_context.squeeze(0).cpu().numpy()
    x_query = episode.x_query.squeeze(0).cpu().numpy()
    y_context = episode.y_context.squeeze(0).cpu().numpy().astype(int)
    generator = EnsembleGenerator(classification=True, n_estimators=8,
                                  norm_methods=["none", "power"], feat_shuffle_method="latin",
                                  class_shuffle_method="shift", random_state=0).fit(x_context, y_context)
    members = generator.transform(x_query, mode="both")
    device = next(backbone.parameters()).device
    logits = []
    with torch.no_grad(), frozen_inference(backbone):
        for method, (xs, ys) in members.items():
            for index, (feature_shuffle, class_shuffle) in enumerate(generator.ensemble_configs_[method]):
                backbone.clear_cache()
                raw = backbone(torch.from_numpy(xs[index:index + 1]).to(device=device, dtype=torch.float32),
                               torch.from_numpy(ys[index:index + 1]).to(device=device, dtype=torch.float32),
                               feature_shuffles=[list(feature_shuffle)], return_logits=True)
                class_index = torch.as_tensor(class_shuffle, device=device, dtype=torch.long)
                logits.append(raw[..., :episode.n_classes][..., class_index])
    combined = torch.stack(logits).mean(dim=0)
    labels = episode.y_query.to(device)
    log_prob = F.log_softmax(combined.float() / 0.9, dim=-1)
    nll = float(F.nll_loss(log_prob.flatten(0, 1), labels))
    accuracy = float((log_prob.argmax(dim=-1).flatten() == labels).float().mean())
    if not math.isfinite(nll):
        raise FloatingPointError(f"nonfinite ordinary TabICL NLL for task {episode.task_id}")
    auc = float("nan")
    if episode.n_classes == 2 and torch.unique(labels).numel() == 2:
        auc = float(roc_auc_score(labels.cpu().numpy(), log_prob.exp().flatten(0, 1)[:, 1].cpu().numpy()))
    return dict(nll=nll, accuracy=accuracy, auc=auc)


def comparison_summary(candidate: np.ndarray, reference: np.ndarray,
                       *, bootstrap_seed: int = 9001, bootstrap_samples: int = 2000) -> dict[str, Any]:
    if candidate.ndim != 1 or candidate.shape != reference.shape or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("comparison requires matched finite task-level NLL vectors")
    delta = candidate - reference
    log_ratio = np.log((candidate + 1e-4) / (reference + 1e-4))
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, candidate.size, size=(bootstrap_samples, candidate.size))
    sampled = log_ratio[draws].mean(axis=1)
    return dict(tasks=int(candidate.size), mean_nll_delta=float(delta.mean()),
                median_nll_delta=float(np.median(delta)),
                geometric_mean_nll_ratio=float(np.exp(log_ratio.mean())),
                ratio_ci95=[float(np.exp(np.quantile(sampled, q))) for q in (0.025, 0.975)],
                wins=int((delta < 0).sum()), losses=int((delta > 0).sum()), ties=int((delta == 0).sum()),
                harms_over_5pct=int((candidate > 1.05 * reference).sum()))


def validation_score(backbone, model: JointPreprocessor,
                     episodes: list[SyntheticEpisode], identity_nll: list[float],
                     device: torch.device) -> tuple[float, float]:
    model.eval()
    scores, losses = [], []
    for source, reference in zip(episodes, identity_nll, strict=True):
        e = on_device(source, device)
        row = episode_metrics(backbone, model, e)
        losses.append(row["nll"])
        scores.append(math.log((row["nll"] + 1e-4) / (reference + 1e-4)))
    return float(np.mean(scores)), float(np.mean(losses))


def state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def training_seed(model_seed: int, step: int) -> int:
    if step < 1:
        raise ValueError("step must be positive")
    return (161001 + model_seed * 1000 + step * 1_000_003) % (2**32)


def train_episodes(args: argparse.Namespace, step: int, device: torch.device) -> list[SyntheticEpisode]:
    from scripts.hyperspline_synthetic_train import generate_episodes
    choices = [(length, fraction) for length in LENGTHS for fraction in FRACTIONS]
    order = np.random.default_rng(161001 + args.model_seed * 1000).permutation(len(choices))
    length, fraction = choices[int(order[(step - 1) % len(choices)])]
    generation = argparse.Namespace(prior_type="mix_scm", min_features=5, max_features=100,
                                    max_classes=10, prior_n_jobs=1, batch_size_per_gp=1,
                                    sequence_length=length, context_fraction=fraction,
                                    synthetic_observation_mode="coverage_expanded",
                                    train_seed=161001 + args.model_seed * 1000)
    return generate_episodes(generation, 4, source_seed=training_seed(args.model_seed, step),
                             task_offset=1_000_000_000 + step * 4, device=device)


def load_frozen(args: argparse.Namespace, device: torch.device):
    backbone, path = load_backbone(argparse.Namespace(checkpoint=args.checkpoint,
                                                       checkpoint_version="tabicl-classifier-v2-20260212.ckpt"), device)
    if backbone.max_classes < 10:
        raise ValueError("backbone must support at least ten classes")
    return backbone, path, hash_file(path)


def config_fingerprint(args: argparse.Namespace, manifest: dict[str, Any], backbone_hash: str) -> str:
    payload = dict(version=FORMAT_VERSION, manifest_sha256=hash_file(manifest_path(args.output_dir)),
                   backbone_hash=backbone_hash, arm=args.arm, model_seed=args.model_seed,
                   lr=args.lr, steps=args.steps, validate_every=args.validate_every,
                   weight_decay=1e-4, task_seed=161001 + args.model_seed * 1000)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def train(args: argparse.Namespace) -> None:
    if args.arm not in ARMS or args.model_seed not in (0, 1) or args.steps <= 0 or args.validate_every <= 0 or args.save_every <= 0:
        raise ValueError("invalid arm, seed or training schedule")
    manifest = read_manifest(args.output_dir)
    device = torch.device(args.device)
    backbone, _, backbone_hash = load_frozen(args, device)
    validation = load_bank(args.output_dir, manifest, "validation")
    validate_episode_classes(validation, backbone.max_classes)
    run_dir = args.output_dir / "runs" / f"{args.arm}_seed{args.model_seed}"
    state_path = run_dir / "state.pt"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run exists; pass --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.model_seed)
    model = JointPreprocessor(args.arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                  betas=(0.9, 0.999), eps=1e-8)
    fingerprint = config_fingerprint(args, manifest, backbone_hash)
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
        step, best_score, best_step, best_model = state["step"], state["best_score"], state["best_step"], state["best_model"]
        identity_nll = state["identity_nll"]
        for name in ("training.csv", "validation.csv"):
            log = run_dir / name
            if log.exists():
                with log.open(newline="", encoding="utf8") as handle:
                    reader = csv.DictReader(handle)
                    retained = [row for row in reader if int(row["step"]) <= step]
                    fields = reader.fieldnames
                with log.open("w", newline="", encoding="utf8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(retained)
    else:
        identity_nll = [episode_metrics(backbone, None, on_device(e, device))["nll"] for e in validation]
        step, best_score, best_step, best_model = 0, 0.0, 0, state_cpu(model)
        initial_score, initial_nll = validation_score(backbone, model, validation, identity_nll, device)
        if initial_score < best_score:
            best_score = initial_score
        csv_append(run_dir / "validation.csv", dict(step=0, tasks_seen=0, score=initial_score,
                                                     mean_nll=initial_nll, selected=True))
        print(f"{args.arm} seed={args.model_seed} step=0 val_nll={initial_nll:.5f} "
              f"val_score={initial_score:.6f} best={best_score:.6f}@0", flush=True)
    json_write(run_dir / "config.json", dict(arm=args.arm, model_seed=args.model_seed,
                                              fingerprint=fingerprint, backbone_hash=backbone_hash,
                                              steps=args.steps, learning_rate=args.lr))

    def save_state(current_step: int) -> None:
        atomic_save(state_path, dict(format_version=FORMAT_VERSION, fingerprint=fingerprint,
                                     step=current_step, model=state_cpu(model), optimizer=optimizer.state_dict(),
                                     best_model=best_model, best_score=best_score, best_step=best_step,
                                     identity_nll=identity_nll))

    if not args.resume:
        save_state(0)
    stop = args.steps if args.max_steps is None else min(args.steps, args.max_steps)
    recent_losses: list[float] = []
    recent_norms: list[float] = []
    last_progress_time = time.perf_counter()
    for current_step in range(step + 1, stop + 1):
        model.train()
        episodes = train_episodes(args, current_step, device)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for e in episodes:
            e = filtered_episode(e)
            index = random.Random(e.task_id + 7919 * args.model_seed).randrange(len(view_specs(e.x_context.shape[-1], e.n_classes)))
            logits = forward_views(backbone, model, e, view_index=index)
            # The unshuffled logits align with original query labels.
            loss = F.cross_entropy(logits.flatten(0, 1), e.y_query.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at step {current_step}, task {e.task_id}")
            (loss / 4).backward()
            losses.append(float(loss.detach()))
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at step {current_step}")
        optimizer.step()
        mean_loss = float(np.mean(losses))
        recent_losses.append(mean_loss)
        recent_norms.append(norm)
        if len(recent_losses) > args.save_every:
            recent_losses.pop(0)
            recent_norms.pop(0)
        csv_append(run_dir / "training.csv", dict(step=current_step, tasks_seen=4 * current_step,
                                                   mean_nll=mean_loss, preclip_gradient_norm=norm))
        validated = current_step % args.validate_every == 0 or current_step == args.steps
        if validated:
            score, mean_nll = validation_score(backbone, model, validation, identity_nll, device)
            selected = score < best_score
            if selected:
                best_score, best_step, best_model = score, current_step, state_cpu(model)
            csv_append(run_dir / "validation.csv", dict(step=current_step, tasks_seen=4 * current_step,
                                                         score=score, mean_nll=mean_nll, selected=selected))
            print(f"{args.arm} seed={args.model_seed} step={current_step} train_nll={mean_loss:.5f} "
                  f"val_nll={mean_nll:.5f} val_score={score:.6f} "
                  f"best={best_score:.6f}@{best_step}", flush=True)
        if current_step % args.save_every == 0 or current_step == stop:
            now = time.perf_counter()
            print(f"{args.arm} seed={args.model_seed} step={current_step}/{args.steps} "
                  f"train_nll_recent={np.mean(recent_losses):.5f} "
                  f"grad_norm_recent={np.mean(recent_norms):.5f} "
                  f"elapsed_since_progress_s={now - last_progress_time:.1f}", flush=True)
            last_progress_time = now
        if validated or current_step % args.save_every == 0 or current_step == stop:
            save_state(current_step)
    if stop == args.steps:
        atomic_save(run_dir / "selected.pt", dict(format_version=FORMAT_VERSION, arm=args.arm,
                                                   model_seed=args.model_seed, model=best_model,
                                                   selected_step=best_step, validation_score=best_score,
                                                   fingerprint=fingerprint, backbone_hash=backbone_hash,
                                                   manifest_sha256=hash_file(manifest_path(args.output_dir))))
        json_write(run_dir / "complete.json", dict(arm=args.arm, model_seed=args.model_seed,
                                                    selected_step=best_step, validation_score=best_score,
                                                    steps_completed=stop))


def report(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.output_dir)
    seeds = tuple(args.model_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (0, 1) for seed in seeds):
        raise ValueError("model seeds must be distinct and selected from 0, 1")
    runs = [(arm, seed, args.output_dir / "runs" / f"{arm}_seed{seed}") for arm in ARMS for seed in seeds]
    if any(not (path / "complete.json").is_file() for _, _, path in runs):
        raise FileNotFoundError("all requested arm/seed runs must complete before opening the test bank")
    device = torch.device(args.device)
    backbone, _, backbone_hash = load_frozen(args, device)
    selected = {}
    for arm, seed, path in runs:
        payload = torch.load(path / "selected.pt", map_location="cpu", weights_only=True)
        if payload["backbone_hash"] != backbone_hash or payload["manifest_sha256"] != hash_file(manifest_path(args.output_dir)):
            raise ValueError("checkpoint backbone or manifest mismatch")
        model = JointPreprocessor(arm).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        selected[(arm, seed)] = model
    test = load_bank(args.output_dir, manifest, "test")
    rows = []
    for i, source in enumerate(test):
        e = on_device(source, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        identity = episode_metrics(backbone, None, e)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        identity_seconds = time.perf_counter() - start
        start = time.perf_counter()
        ordinary = ordinary_episode_metrics(backbone, source)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ordinary_seconds = time.perf_counter() - start
        for (arm, seed), model in selected.items():
            start = time.perf_counter()
            candidate = episode_metrics(backbone, model, e)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidate_seconds = time.perf_counter() - start
            rows.append(dict(task_id=e.task_id, arm=arm, model_seed=seed,
                             identity_nll=identity["nll"], candidate_nll=candidate["nll"],
                             identity_accuracy=identity["accuracy"], candidate_accuracy=candidate["accuracy"],
                             identity_auc=identity["auc"], candidate_auc=candidate["auc"],
                             ordinary_nll=ordinary["nll"], ordinary_accuracy=ordinary["accuracy"],
                             ordinary_auc=ordinary["auc"], identity_seconds=identity_seconds,
                             ordinary_seconds=ordinary_seconds, candidate_seconds=candidate_seconds,
                             n_context=e.x_context.shape[1], n_query=e.x_query.shape[1],
                             n_features=e.x_context.shape[2], n_classes=e.n_classes))
        if (i + 1) % 32 == 0:
            print(f"reported {i + 1}/{len(test)} synthetic tasks", flush=True)
    output = args.output_dir / ("report" if seeds == (0, 1) else
                                "report_seed" + "_".join(map(str, seeds)))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "synthetic_tasks.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    task_vectors = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        per_task = {}
        for row in subset:
            per_task.setdefault(row["task_id"], []).append(row)
        task_ids = sorted(per_task)
        candidate = np.asarray([np.mean([item["candidate_nll"] for item in per_task[task]]) for task in task_ids])
        ordinary = np.asarray([per_task[task][0]["ordinary_nll"] for task in task_ids])
        identity = np.asarray([per_task[task][0]["identity_nll"] for task in task_ids])
        task_vectors[arm] = candidate
        summaries[arm] = dict(vs_ordinary=comparison_summary(candidate, ordinary),
                              vs_identity=comparison_summary(candidate, identity),
                              mean_accuracy_delta_vs_ordinary=float(np.mean([
                                  np.mean([item["candidate_accuracy"] - item["ordinary_accuracy"] for item in group])
                                  for group in per_task.values()])),
                              mean_candidate_seconds=float(np.mean([item["candidate_seconds"] for item in subset])),
                              model_seeds=len(seeds))
    pairwise = {f"{left}_vs_{right}": comparison_summary(task_vectors[left], task_vectors[right])
                for left, right in (("joint", "restricted"), ("no_spline", "restricted"), ("joint", "no_spline"))}
    json_write(output / "synthetic_summary.json", dict(arms=summaries, pairwise=pairwise,
                                                        model_seed_ids=list(seeds),
                                                        bank_sha256=manifest["banks"]["test"]["sha256"],
                                                        limitation="same-generator synthetic transfer only"))
    print(json.dumps(summaries, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--validation-tasks", type=int, default=512)
    prep.add_argument("--test-tasks", type=int, default=1024)
    prep.add_argument("--validation-seed", type=int, default=171001)
    prep.add_argument("--test-seed", type=int, default=172001)
    training = sub.add_parser("train")
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--arm", choices=ARMS, required=True)
    training.add_argument("--model-seed", type=int, choices=(0, 1), required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--checkpoint", type=Path, default=None)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--steps", type=int, default=10000)
    training.add_argument("--validate-every", type=int, default=1000)
    training.add_argument("--save-every", type=int, default=50, help="Resume state interval; does not select checkpoints")
    training.add_argument("--max-steps", type=int, default=None, help="Pause after this absolute step; resume later")
    training.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("report")
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--checkpoint", type=Path, default=None)
    evaluation.add_argument("--model-seeds", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
