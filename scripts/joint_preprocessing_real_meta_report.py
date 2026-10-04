"""Lock and evaluate shared preprocessing continuation on unseen families.

Checkpoint choices are immutable before either test bank is loaded. Prediction
caches are resumable per panel/episode; continuation seeds remain paired within
each real family rather than becoming additional independent datasets.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from scripts import joint_preprocessing_synthetic_pilot as pilot
from scripts import joint_preprocessing_zero_shot_comparison as synthetic
from scripts import joint_preprocessing_zero_shot_real_transfer as transfer

ALPHA = 0.5
TEMPERATURE = transfer.TEMPERATURE
FLOOR = transfer.FLOOR
PANELS = ("real_test", "synthetic_test")


def _runner():
    # The training runner imports these entry points lazily as well.
    return importlib.import_module("scripts.joint_preprocessing_real_meta_continuation")


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _choices(root, fingerprint, runner, expected_steps=None):
    choices = {}
    for arm in runner.ARMS:
        for seed in runner.SEEDS:
            name = f"{arm}_seed{seed}"
            directory = runner.run_dir(root, arm, seed)
            complete = synthetic.read(directory / "complete.json")
            if complete["experiment_fingerprint"] != fingerprint:
                raise ValueError(f"{name}: completion experiment fingerprint differs")
            if (hasattr(runner, "run_fingerprint")
                    and complete["fingerprint"] != runner.run_fingerprint(fingerprint, arm, seed)):
                raise ValueError(f"{name}: completion arm/seed fingerprint differs")
            if complete.get("arm", arm) != arm or complete.get("continuation_seed", seed) != seed:
                raise ValueError(f"{name}: completion arm/seed identity differs")
            steps, selected_step = int(complete["steps"]), int(complete["selected_step"])
            if (steps < 1 or not 0 <= selected_step <= steps
                    or (expected_steps is not None and steps != expected_steps)
                    or not math.isfinite(float(complete["selected_score"]))):
                raise ValueError(f"{name}: completion schedule/score differs")
            checks = {}
            for state, step in (("selected", selected_step), ("final", steps)):
                path = directory / f"{state}.pt"
                actual = pilot.hash_file(path)
                if actual != complete[f"{state}_sha256"]:
                    raise ValueError(f"{name}: {state} checkpoint hash differs")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["fingerprint"] != complete["fingerprint"] or payload["step"] != step:
                    raise ValueError(f"{name}: {state} checkpoint metadata differs")
                checks[f"{state}_sha256"] = actual
            choices[name] = dict(arm=arm, seed=int(seed), alpha=ALPHA,
                fingerprint=complete["fingerprint"], selected_step=selected_step,
                steps=steps, selected_score=float(complete["selected_score"]),
                complete_sha256=pilot.hash_file(directory / "complete.json"), **checks)
    return choices


def _lock_from_setup(args, initial, manifest, fingerprint, runner):
    root = args.output_dir
    samples = int(args.bootstrap_samples)
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    locked = dict(format_version=1, fingerprint=fingerprint,
        manifest_sha256=pilot.hash_file(root / "manifest.json"),
        common_initial_state_sha256=_state_hash(initial), alpha=ALPHA,
        temperature=TEMPERATURE, nll_floor=FLOOR, bootstrap_samples=samples,
        choices=_choices(root, fingerprint, runner, getattr(args, "steps", None)),
        selection="One global real-validation checkpoint per arm/continuation seed; fixed alpha=0.5; final states are secondary diagnostics.",
        seed_scope="Two continuation seeds share the same synthetic-pretrained starting model.")
    path = root / "lock.json"
    if path.exists():
        if synthetic.read(path) != locked:
            raise ValueError("continuation lock/checkpoint choices changed")
    else:
        if (root / "test_report/started.json").exists():
            raise ValueError("test has started; checkpoint choices cannot be relocked")
        pilot.json_write(path, locked)
    return locked


def lock(args):
    runner = _runner()
    _, initial, manifest, fingerprint, _ = runner.setup(args)
    return _lock_from_setup(args, initial, manifest, fingerprint, runner)


def metrics(logits, labels, classes):
    logits = logits.float()
    if logits.shape[-1] != classes:
        raise ValueError("prediction class count differs")
    log_prob = F.log_softmax(logits / TEMPERATURE, dim=-1).reshape(-1, classes)
    labels = labels.flatten().to(log_prob.device).long()
    if len(labels) != len(log_prob) or not torch.isfinite(log_prob).all():
        raise ValueError("invalid prediction shape or nonfinite logits")
    auc = None
    if classes == 2 and torch.unique(labels).numel() == 2:
        auc = float(roc_auc_score(labels.cpu().numpy(), log_prob[:, 1].exp().cpu().numpy()))
    return dict(nll=float(F.nll_loss(log_prob, labels)),
        accuracy=float((log_prob.argmax(-1) == labels).float().mean()), auc=auc)


def _timed_prediction(runner, backbone, model, episode, estimators, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), pilot.frozen_inference(backbone):
        logits, views = runner.panel_logits(backbone, model, episode, estimators=estimators)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if views != estimators or not math.isfinite(seconds):
        raise ValueError("invalid prediction timing/view count")
    return dict(logits=logits.detach().cpu(), views=int(views), seconds=seconds)


def _methods(runner, include_final):
    return ("ordinary8", "ordinary16", "initial_learned", "initial_blend",
        *[f"{arm}_seed{seed}_{state}_{kind}"
          for arm in runner.ARMS for seed in runner.SEEDS
          for state in (("selected", "final") if include_final else ("selected",))
          for kind in ("learned", "blend")])


def _episode_identity(panel, episode):
    family = (str(episode["family"]) if panel == "real_test"
              else f"synthetic_task_{int(episode['task_id'])}")
    return family, int(episode.get("split_seed", 0))


def _mean_optional(values):
    return float(np.mean(values)) if all(value is not None for value in values) else None


def _summary(candidate, baseline, samples):
    c, b = np.asarray([r["nll"] for r in candidate]), np.asarray([r["nll"] for r in baseline])
    if not len(c) or c.shape != b.shape or not np.isfinite(c).all() or not np.isfinite(b).all():
        raise ValueError("invalid paired family losses")
    result = transfer.family_summary(c, b, samples=samples)
    common_auc = [(left["auc"], right["auc"]) for left, right in zip(candidate, baseline, strict=True)
                  if left["auc"] is not None and right["auc"] is not None]
    candidate_seconds = float(np.mean([r["seconds"] for r in candidate]))
    baseline_seconds = float(np.mean([r["seconds"] for r in baseline]))
    result.update(mean_accuracy=float(np.mean([r["accuracy"] for r in candidate])),
        mean_accuracy_delta=float(np.mean([r["accuracy"]-s["accuracy"] for r, s in zip(candidate, baseline, strict=True)])),
        auc_families=len(common_auc), mean_auc_delta=float(np.mean([c-b for c, b in common_auc])) if common_auc else None,
        mean_seconds=candidate_seconds, baseline_mean_seconds=baseline_seconds,
        mean_runtime_ratio=candidate_seconds/baseline_seconds if baseline_seconds else None,
        mean_views=float(np.mean([r["views"] for r in candidate])),
        baseline_mean_views=float(np.mean([r["views"] for r in baseline])))
    return result


def _seed_consistency(candidates, baselines):
    deltas = np.stack([[c["nll"]-b["nll"] for c, b in zip(cs, bs, strict=True)]
                       for cs, bs in zip(candidates, baselines, strict=True)])
    ratios = np.stack([[(c["nll"]+FLOOR)/(b["nll"]+FLOOR)
                       for c, b in zip(cs, bs, strict=True)]
                       for cs, bs in zip(candidates, baselines, strict=True)])
    both_wins = (deltas < -1e-6).all(0)
    both_losses = (deltas > 1e-6).all(0)
    return dict(both_seed_wins=int(both_wins.sum()), both_seed_losses=int(both_losses.sum()),
        other_seed_direction=int((~(both_wins | both_losses)).sum()),
        per_seed_geometric_gain=[1-float(np.exp(np.log(r).mean())) for r in ratios],
        **{f"both_seed_harms_over_{p}pct": int((ratios > 1+p/100).all(0).sum()) for p in (1, 5, 10)})


def aggregate(rows, expected, methods, runner, samples, include_final=True):
    """One bootstrap unit per family/task; average splits and paired seeds first."""
    family_rows, vectors = [], {}
    index = {}
    for row in rows:
        key = (row["family"], row["split_seed"], row["method"])
        if key in index:
            raise ValueError("duplicate episode/method row")
        index[key] = row
    wanted = {(f, s, m) for f, splits in expected.items() for s in splits for m in methods}
    if set(index) != wanted:
        raise ValueError("incomplete or unexpected split/method coverage")
    for method in methods:
        vectors[method] = []
        for family, splits in expected.items():
            group = [index[(family, split, method)] for split in sorted(splits)]
            result = dict(panel=group[0]["panel"], family=family, method=method, splits=len(group),
                **{key: float(np.mean([r[key] for r in group])) for key in ("nll", "accuracy", "seconds", "views")},
                auc=_mean_optional([r["auc"] for r in group]))
            vectors[method].append(result)
            family_rows.append(result)
    comparisons = {}
    def compare(name, candidate, baseline, seed_sets=None):
        result = _summary(vectors[candidate], vectors[baseline], samples)
        if seed_sets is not None:
            result["seed_consistency"] = _seed_consistency(*seed_sets)
        comparisons[name] = result
    for method in methods[2:]:
        compare(f"{method}_vs_ordinary8", method, "ordinary8")
        compare(f"{method}_vs_ordinary16", method, "ordinary16")
    compare("ordinary16_vs_ordinary8", "ordinary16", "ordinary8")
    for state in (("selected", "final") if include_final else ("selected",)):
        for kind in ("learned", "blend"):
            for arm in runner.ARMS:
                name = f"{arm}_paired_seed_mean_{state}_{kind}"
                groups = [vectors[f"{arm}_seed{seed}_{state}_{kind}"] for seed in runner.SEEDS]
                vectors[name] = []
                for records in zip(*groups, strict=True):
                    record = dict(records[0], method=name,
                        **{key: float(np.mean([r[key] for r in records])) for key in ("nll", "accuracy", "seconds", "views")},
                        auc=_mean_optional([r["auc"] for r in records]))
                    vectors[name].append(record)
                    family_rows.append(record)
                compare(f"{name}_vs_ordinary16", name, "ordinary16",
                    (groups, [vectors["ordinary16"] for _ in runner.SEEDS]))
                compare(f"{name}_vs_ordinary8", name, "ordinary8",
                    (groups, [vectors["ordinary8"] for _ in runner.SEEDS]))
                compare(f"{name}_vs_initial_{kind}", name, f"initial_{kind}",
                    (groups, [vectors[f"initial_{kind}"] for _ in runner.SEEDS]))
            for arm in ("real", "mixed"):
                for seed in runner.SEEDS:
                    left, right = f"{arm}_seed{seed}_{state}_{kind}", f"synthetic_seed{seed}_{state}_{kind}"
                    compare(f"{left}_vs_{right}", left, right)
                left, right = f"{arm}_paired_seed_mean_{state}_{kind}", f"synthetic_paired_seed_mean_{state}_{kind}"
                compare(f"{left}_vs_{right}", left, right,
                    ([vectors[f"{arm}_seed{seed}_{state}_{kind}"] for seed in runner.SEEDS],
                     [vectors[f"synthetic_seed{seed}_{state}_{kind}"] for seed in runner.SEEDS]))
    return family_rows, comparisons


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def report(args):
    runner = _runner()
    backbone, initial, manifest, fingerprint, device = runner.setup(args)
    locked = _lock_from_setup(args, initial, manifest, fingerprint, runner)
    include_final = bool(getattr(args, "include_final", True))
    destination = args.output_dir / "test_report"
    identity = dict(experiment_fingerprint=fingerprint, lock_hash=pilot.hash_file(args.output_dir / "lock.json"),
        manifest_sha256=locked["manifest_sha256"], include_final=include_final,
        code_sha256={Path(__file__).name: pilot.hash_file(Path(__file__)),
                     Path(runner.__file__).name: pilot.hash_file(Path(runner.__file__))})
    report_fingerprint = synthetic.digest(identity)
    started = destination / "started.json"
    if started.exists():
        if synthetic.read(started) != dict(fingerprint=report_fingerprint, **identity):
            raise ValueError("test already started under different lock/protocol/code")
    else:
        pilot.json_write(started, dict(fingerprint=report_fingerprint, **identity))
    completed = destination / "complete.json"
    if completed.exists():
        if synthetic.read(completed)["fingerprint"] != report_fingerprint:
            raise ValueError("completed test report differs")
        print("Real/synthetic continuation test report already complete.", flush=True)
        return
    models = {"initial_learned": initial.eval().requires_grad_(False)}
    for name, choice in locked["choices"].items():
        directory = runner.run_dir(args.output_dir, choice["arm"], choice["seed"])
        for state in (("selected", "final") if include_final else ("selected",)):
            model = copy.deepcopy(initial)
            payload = torch.load(directory / f"{state}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model"], strict=True)
            models[f"{name}_{state}_learned"] = model.to(device).eval().requires_grad_(False)
    model_hashes = {name: _state_hash(model) for name, model in models.items()}
    methods = _methods(runner, include_final)
    all_rows, all_family_rows, panel_reports = [], [], {}
    for panel in PANELS:
        episodes = runner.load_panel(args.output_dir, manifest, panel)
        expected = {}
        for episode in episodes:
            family, split = _episode_identity(panel, episode)
            if split in expected.setdefault(family, set()):
                raise ValueError("duplicate test episode identity")
            expected[family].add(split)
        if not expected or (panel == "real_test" and any(s != {0, 1} for s in expected.values())):
            raise ValueError("empty panel or incomplete real two-split family coverage")
        panel_rows = []
        for index, episode in enumerate(episodes):
            family, split = _episode_identity(panel, episode)
            cache = destination / "predictions" / panel / f"episode{index:06d}.pt"
            episode_identity = dict(panel=panel, family=family, split_seed=split,
                task_id=int(episode.get("task_id", index)))
            if cache.exists():
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                if (saved["fingerprint"] != report_fingerprint or saved["identity"] != episode_identity
                        or set(saved["predictions"]) != set(methods)
                        or not torch.equal(saved["labels"], episode["y_query"].cpu())):
                    raise ValueError("prediction cache identity/methods/labels differ")
            else:
                try:
                    predictions = {f"ordinary{n}": _timed_prediction(runner, backbone, "ordinary", episode, n, device) for n in (8, 16)}
                    state_sources = {}
                    for name, model in models.items():
                        state_hash = model_hashes[name]
                        if state_hash in state_sources:
                            source_name = state_sources[state_hash]
                            predictions[name] = dict(predictions[source_name],
                                prediction_source_method=source_name, inference_reused=True)
                        else:
                            source_name = name
                            predictions[name] = dict(_timed_prediction(runner, backbone, model, episode, 8, device),
                                prediction_source_method=name, inference_reused=False)
                            state_sources[state_hash] = name
                        ordinary, learned = predictions["ordinary8"], predictions[name]
                        predictions[name.removesuffix("learned")+"blend"] = dict(
                            logits=(1-ALPHA)*ordinary["logits"]+ALPHA*learned["logits"],
                            views=ordinary["views"]+learned["views"],
                            seconds=ordinary["seconds"]+learned["seconds"],
                            prediction_source_method=source_name.removesuffix("learned")+"blend",
                            inference_reused=learned["inference_reused"])
                    saved = dict(fingerprint=report_fingerprint, identity=episode_identity,
                        predictions=predictions, labels=episode["y_query"].detach().cpu())
                    pilot.atomic_save(cache, saved)
                except Exception as error:
                    pilot.json_write(destination / "failure.json", dict(**episode_identity,
                        error=f"{type(error).__name__}: {error}"))
                    raise
            for method in methods:
                value = saved["predictions"][method]
                panel_rows.append(dict(panel=panel, family=family, split_seed=split,
                    task_id=episode_identity["task_id"], method=method,
                    n_context=episode["x_context"].shape[1], n_query=episode["x_query"].shape[1],
                    n_features=episode["x_context"].shape[-1], n_numerical=int(episode["numerical_mask"].sum()),
                    n_classes=int(episode["n_classes"]), **metrics(value["logits"], episode["y_query"], episode["n_classes"]),
                    views=value["views"], seconds=value["seconds"],
                    prediction_source_method=value.get("prediction_source_method", method),
                    inference_reused=value.get("inference_reused", False)))
            _write_csv(destination / "episodes.csv", all_rows+panel_rows)
            if index == 0 or (index+1) % 16 == 0 or index+1 == len(episodes):
                print(f"Scored {panel} {index+1}/{len(episodes)}: {family} split {split} ({len(methods)} methods)", flush=True)
        family_rows, comparisons = aggregate(panel_rows, expected, methods, runner,
            args.bootstrap_samples, include_final=include_final)
        all_rows.extend(panel_rows)
        all_family_rows.extend(family_rows)
        _write_csv(destination / "families.csv", all_family_rows)
        panel_reports[panel] = dict(families=len(expected), episodes=len(episodes), comparisons=comparisons,
            bootstrap_unit="dataset family after averaging its two row splits" if panel == "real_test" else "independent synthetic task")
        pilot.json_write(destination / f"{panel}.json", dict(fingerprint=report_fingerprint, **panel_reports[panel]))
    pilot.json_write(completed, dict(fingerprint=report_fingerprint, **identity, alpha=ALPHA,
        choices=locked["choices"], panels=panel_reports,
        seed_aggregation="Average each model's split-level metrics within family, then average the two paired continuation seeds before family bootstrap; these are mean seed scores, not an ensemble across seeds.",
        timing_note="Synchronized preprocessing plus inference, excluding initial raw encoding. Ordinary8 is cached once per episode and reused in all blends. Identical model states share inference, with prediction_source_method recorded and the original deployment timing retained. Blend deployment seconds sum ordinary8 and learned8. Cold starts affect early episodes.",
        conditioning="Generated numerical maps use labeled context and its numerical missing masks only; no target-specific parameter updates.",
        uncertainty="Continuation seeds share one synthetic-pretrained initialization; final states are secondary and never select checkpoints."))
    print(json.dumps({p: {"families": v["families"], "episodes": v["episodes"]} for p, v in panel_reports.items()}), flush=True)


test = report
