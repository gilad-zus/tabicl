"""Paired learned-line versus free-cubic appended numerical features.

Training uses T-only episodes, separate A/B checkpoint selection, and frozen
context expansion at deployment. Main scores never select an identity blend.
Final-state measurements are diagnostics, never checkpoint-selection inputs.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from direct_spline_openml_support_audit import _find_source_cases, _load_json, _sha256, _write_json
from direct_spline_openml_crossfit_blend import (
    _crossfit_validation_halves, _load_source_task, _source_standard_prediction, _validate_case,
)
from tabicl._experiments.direct_spline_openml import (
    _bag_splits, _cpu_state_dict, _metric_bundle, _safe_name, _seed, load_frozen_backbone,
)
from tabicl._experiments.direct_spline_openml_standard import (
    _AdapterSet, _classification_training_objective_from_logits, _cosine_scheduler,
    _fit_standard_bag, _identity_view_parity, _normal_prediction,
    _normal_prediction_with_appended_context, _training_logits,
)
from tabicl._experiments.direct_spline_protocol import deployment_error, sample_episode_indices
from tabicl._hyperspline.feature_expansion import SplineFeatureExpansion


ARMS = {"line": None, "spline8": 8, "spline20": 20}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True, action="append", type=int)
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--training-seed", type=int, default=20260920)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--smoothness-weight", type=float, default=1e-4)
    parser.add_argument("--audit-episodes", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--regressor-checkpoint", type=Path)
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-equivalent-hardware-resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_id)) != len(args.task_id) or args.bags < 2:
        parser.error("task IDs must be unique and bags >= 2")
    if min(args.steps, args.validation_interval, args.audit_episodes) < 1:
        parser.error("steps, interval, and audit episodes must be positive")
    if args.learning_rate <= 0 or args.smoothness_weight < 0:
        parser.error("invalid learning rate or regularization")
    if args.allow_equivalent_hardware_resume and not args.resume:
        parser.error("equivalent-hardware resume requires --resume")
    return args


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_torch_save(value, path):
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _runtime(device):
    stable = {
        "torch": str(torch.__version__), "numpy": np.__version__,
        "python": platform.python_version(), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "device_type": device.type,
    }
    if device.type == "cuda":
        stable.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    return {"stable": stable, "hostname": platform.node(), "device": str(device)}


def _prepare(args, device):
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "src/tabicl/_hyperspline/feature_expansion.py",
             root / "src/tabicl/_hyperspline/bspline.py",
             root / "src/tabicl/_experiments/direct_spline_openml_standard.py",
             root / "src/tabicl/_experiments/tabarena_direct_spline_protocol.py",
             root / "scripts/direct_spline_openml_crossfit_blend.py"]
    semantic = {
        "experiment": "appended numerical line versus free cubic features v1",
        "source_dir": str(args.source_dir),
        "source_manifest_sha256": _sha256(args.source_dir / "experiment_manifest.json"),
        "task_ids": sorted(args.task_id), "protocol_seed": args.protocol_seed,
        "training_seed": args.training_seed, "bags": args.bags, "steps": args.steps,
        "validation_interval": args.validation_interval, "learning_rate": args.learning_rate,
        "smoothness_weight": args.smoothness_weight, "audit_episodes": args.audit_episodes,
        "arms": ARMS, "query_fraction_range": [0.05, 0.2],
        "optimizer": "Adam; betas=(0.9,0.999); eps=1e-8; weight_decay=0",
        "cosine_min_lr_ratio": 0.1, "gradient_clip_norm": 2.0,
        "coordinate_mapping": "u=2/pi*atan(pi*z/8); initial added feature=4*u",
        "selection": "Independent best A/B checkpoints; no blend/identity selection",
        "test_context": "T+A+B; preprocessing and learned features remain frozen from T",
        "feature_shuffles": "Extend original permutations; each extra feature follows its source column",
        "implementation_sha256": {str(path.relative_to(root)): _sha256(path) for path in paths},
        "backbone_overrides": {
            key: None if getattr(args, key) is None else _sha256(getattr(args, key))
            for key in ("classifier_checkpoint", "regressor_checkpoint")
        },
    }
    path = args.output_dir / "experiment_manifest.json"
    runtime = _runtime(device)
    if path.exists():
        old = _load_json(path, label="feature-expansion manifest")
        if not args.resume or _canonical(old["semantic"]) != _canonical(semantic):
            raise ValueError("resume requires the same experiment semantics and source files")
        if old["runtime"] != runtime and not (
            args.allow_equivalent_hardware_resume and old["runtime"]["stable"] == runtime["stable"]
        ):
            raise ValueError("runtime differs; equivalent stable hardware needs --allow-equivalent-hardware-resume")
    else:
        _write_json(path, {"semantic": semantic, "runtime": runtime})
    return hashlib.sha256(_canonical(semantic).encode()).hexdigest()


def _adapters(bundle, capacity, device):
    return _AdapterSet(OrderedDict(
        (method, SplineFeatureExpansion(len(bundle.numerical_indices), n_control_points=capacity).to(device))
        for method in bundle.estimator.ensemble_generator_.preprocessors_
    ))


def _smoothness(adapters):
    return torch.stack([adapter.smoothness() for adapter in adapters.adapters.values()]).mean()


def _linearized_features(bundle, adapters, device):
    """Project each learned feature onto a line using T coordinates only.

    This diagnostic preserves the original columns and shared arctan view,
    while removing learned curvature without fitting against any held-out label.
    """
    lines = _adapters(bundle, None, device)
    with torch.no_grad():
        for method, preprocessor in bundle.estimator.ensemble_generator_.preprocessors_.items():
            values = torch.as_tensor(preprocessor.X_transformed_[:, bundle.numerical_indices], dtype=torch.float32, device=device).unsqueeze(0)
            u = (2.0 / torch.pi) * torch.atan(torch.pi * values / 8.0)
            output = adapters.for_method(method).transform(values)
            mean_u, mean_output = u.mean(dim=1), output.mean(dim=1)
            centered = u - mean_u.unsqueeze(1)
            variance = centered.square().mean(dim=1)
            covariance = (centered * (output - mean_output.unsqueeze(1))).mean(dim=1)
            slope = torch.where(variance > 1e-12, covariance / variance.clamp_min(1e-12), torch.zeros_like(variance))
            bias = mean_output - slope * mean_u
            lines.for_method(method).coefficients[..., 0].copy_(slope - 4.0)
            lines.for_method(method).coefficients[..., 1].copy_(bias)
    return lines


def _objective(task, bundle, adapters, context, query, device):
    output = _training_logits(bundle=bundle, adapters=adapters, context_indices=context, query_indices=query, device=device)
    target = torch.as_tensor(bundle.fit_labels[query], device=device)
    if task.problem_type == "regression":
        return F.mse_loss(output.flatten(), target.float().flatten())
    return _classification_training_objective_from_logits(
        logits=output, target=target, problem_type=task.problem_type, n_classes=task.n_classes,
        softmax_temperature=float(bundle.estimator.softmax_temperature), config={},
    )


def _sample(task, bundle, rng):
    return sample_episode_indices(
        bundle.fit_labels, problem_type=task.problem_type,
        context_rows=max(1, len(bundle.fit_labels) - 1), query_rows=256,
        query_fraction_range=(0.05, 0.2), rng=rng,
    )


def _train_audit(task, bundle, adapters, episodes, device):
    with torch.no_grad():
        losses = [float(_objective(task, bundle, adapters, c, q, device)) for c, q in episodes]
    return {"mean": float(np.mean(losses)), "episode_losses": losses}


def _predict(task, bundle, adapters, rows, appended, device, *, test=False):
    query_x = task.x_test if test else task.x_train.iloc[rows].reset_index(drop=True)
    if len(appended):
        return _normal_prediction_with_appended_context(
            bundle=bundle, query_x=query_x, context_indices=bundle.support_indices,
            appended_context_x=task.x_train.iloc[appended].reset_index(drop=True),
            appended_context_y=task.y_train[appended], adapters=adapters, device=device,
        )
    return _normal_prediction(bundle=bundle, query_x=query_x, context_indices=bundle.support_indices, adapters=adapters, device=device)


def _error(task, rows, prediction):
    result = float(deployment_error(task.problem_type, task.y_train[rows], prediction, n_classes=task.n_classes))
    if not np.isfinite(result):
        raise FloatingPointError("nonfinite validation error")
    return result


def _save_npz(path, metadata, **arrays):
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, metadata=np.asarray(_canonical(metadata)), **arrays)
    os.replace(temp, path)


def _read_npz(path, fingerprint, fit, a, b):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata["fingerprint"] != fingerprint:
            raise ValueError(f"fingerprint mismatch: {path}")
        for key, expected in (("fit", fit), ("a", a), ("b", b)):
            if not np.array_equal(payload[key], expected):
                raise ValueError(f"split mismatch: {path} {key}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata"}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"nonfinite artifact: {path}")
    return {"metadata": metadata, **arrays}


def _fit_arm(args, task, bundle, bag, fit, a, b, arm, device, fingerprint, task_dir):
    artifact = task_dir / f"bag_{bag}.{arm}.npz"
    completed = _read_npz(artifact, fingerprint, fit, a, b)
    if completed is not None:
        return completed
    capacity = ARMS[arm]
    adapters = _adapters(bundle, capacity, device)
    optimizer = torch.optim.Adam(adapters.parameters(), lr=args.learning_rate)
    scheduler = _cosine_scheduler(optimizer, total_steps=args.steps, min_lr_ratio=0.1)
    rng = np.random.default_rng(_seed(args.training_seed, task.task_id, bag, 203))
    audit_rng = np.random.default_rng(_seed(args.training_seed, task.task_id, bag, 990))
    episodes = [_sample(task, bundle, audit_rng) for _ in range(args.audit_episodes)]
    initial = _cpu_state_dict(adapters)
    records = []
    best = {}
    start_step = 0
    elapsed = 0.0
    progress_path = task_dir / f"bag_{bag}.{arm}.progress.pt"
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("training progress fingerprint mismatch")
        adapters.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["episode_rng"]
        start_step, best, records = saved["step"], saved["best"], saved["records"]
        elapsed = saved["elapsed"]
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def evaluate(step):
        train = _train_audit(task, bundle, adapters, episodes, device)
        errors = {name: _error(task, rows, _predict(task, bundle, adapters, rows, [], device)) for name, rows in (("a", a), ("b", b))}
        if not np.isfinite(train["mean"]):
            raise FloatingPointError("nonfinite diagnostic training loss")
        for name, error in errors.items():
            if name not in best or error < best[name]["error"]:
                best[name] = {"step": step, "error": error, "state": _cpu_state_dict(adapters)}
        records.append({"step": step, "training": train, "validation": errors,
                        "smoothness": float(_smoothness(adapters).detach()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        print(f"task {task.task_id} bag {bag} {arm} step {step}: train={train['mean']:.6g} A={errors['a']:.6g} B={errors['b']:.6g}", flush=True)

    if not best:
        evaluate(0)
    for step in range(start_step + 1, args.steps + 1):
        context, query = _sample(task, bundle, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(task, bundle, adapters, context, query, device)
        objective = loss + args.smoothness_weight * _smoothness(adapters)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"nonfinite objective: task {task.task_id} {arm} step {step}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 2.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        del loss, objective
        if step % args.validation_interval == 0 or step == args.steps:
            evaluate(step)
            _atomic_torch_save({
                "fingerprint": fingerprint, "step": step, "state": _cpu_state_dict(adapters),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "episode_rng": rng.bit_generator.state, "best": best, "records": records,
                "elapsed": elapsed + time.perf_counter() - started,
            }, progress_path)
    final = _cpu_state_dict(adapters)
    del optimizer, scheduler
    state_diagnostics = {}
    # Selected and final checkpoints use exactly the same fixed train episodes.
    for name, state in (("final", final), ("selected_a", best["a"]["state"]), ("selected_b", best["b"]["state"])):
        adapters.load_state_dict(state)
        state_diagnostics[name] = {
            "training": _train_audit(task, bundle, adapters, episodes, device),
            "normalized_curvature": float(_smoothness(adapters).detach()),
        }
    validation = np.sort(np.concatenate((a, b)))
    adapters.load_state_dict(final)
    final_test = _predict(task, bundle, adapters, None, validation, device, test=True)
    arrays = {"fit": fit, "a": a, "b": b, "final_test": final_test}
    for selected_on, rows, appended in (("a", b, a), ("b", a, b)):
        adapters.load_state_dict(best[selected_on]["state"])
        arrays[f"oof_{'b' if selected_on == 'a' else 'a'}"] = _predict(task, bundle, adapters, rows, appended, device)
        arrays[f"test_{selected_on}"] = _predict(task, bundle, adapters, None, validation, device, test=True)
        if capacity is not None:
            lines = _linearized_features(bundle, adapters, device)
            arrays[f"linearized_test_{selected_on}"] = _predict(task, bundle, lines, None, validation, device, test=True)
    metadata = {
        "fingerprint": fingerprint, "arm": arm, "bag": bag,
        "selected_steps": {key: int(value["step"]) for key, value in best.items()},
        "checkpoints": records, "state_diagnostics": state_diagnostics,
        "trainable_parameters": sum(p.numel() for p in adapters.parameters() if p.requires_grad),
        "n_numeric": len(bundle.numerical_indices),
        "fit_rows": len(fit), "training_seconds": elapsed + time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
    }
    _atomic_torch_save({
        "fingerprint": fingerprint, "task_id": task.task_id, "outer_split_hash": task.outer_split_hash,
        "bag": bag, "arm": arm, "initial": initial, "final": final,
        "selected_a": best["a"], "selected_b": best["b"],
    }, task_dir / f"bag_{bag}.{arm}.adapters.pt")
    _save_npz(artifact, metadata, **arrays)
    return {"metadata": metadata, **arrays}


def _baselines(task, bundle, bag, fit, a, b, device, fingerprint, task_dir):
    path = task_dir / f"bag_{bag}.baselines.npz"
    result = _read_npz(path, fingerprint, fit, a, b)
    if result is not None:
        return result
    _identity_view_parity(bundle=bundle, adapters=None, query_x=task.x_train.iloc[a].reset_index(drop=True),
                         device=device, progress=None, task_id=task.task_id, bag=bag, split="baseline")
    initial = _adapters(bundle, None, device)
    arrays = {"fit": fit, "a": a, "b": b}
    for name, adapters in (("identity", None), ("initial", initial)):
        arrays[f"{name}_a"] = _predict(task, bundle, adapters, a, b, device)
        arrays[f"{name}_b"] = _predict(task, bundle, adapters, b, a, device)
        arrays[f"{name}_test"] = _predict(task, bundle, adapters, None, np.sort(np.concatenate((a, b))), device, test=True)
    metadata = {"fingerprint": fingerprint, "bag": bag}
    _save_npz(path, metadata, **arrays)
    return {"metadata": metadata, **arrays}


def _assemble(task, bags, prefix=None):
    first = bags[0]["oof_a" if prefix is None else f"{prefix}_a"]
    shape = (len(task.y_train), *first.shape[1:])
    oof = np.full(shape, np.nan)
    tests = []
    for bag in bags:
        oof[bag["a"]] = bag["oof_a" if prefix is None else f"{prefix}_a"]
        oof[bag["b"]] = bag["oof_b" if prefix is None else f"{prefix}_b"]
        tests.append(0.5 * (bag["test_a"] + bag["test_b"]) if prefix is None else bag[f"{prefix}_test"])
    if not np.isfinite(oof).all():
        raise ValueError("incomplete OOF assembly")
    return oof, np.mean(tests, axis=0)


def _run_task(args, case, immutable, device, fingerprint):
    task = _load_source_task(case=case, immutable_run=immutable)
    full = _source_standard_prediction(source_dir=case.source_dir, task=task)
    if full is None:
        raise FileNotFoundError(f"task {task.task_id}: cached ordinary full-context TabICLv2 baseline is required")
    task_dir = args.output_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"
    task_dir.mkdir(parents=True, exist_ok=True)
    config = {**case.config, "max_context_rows": None, "train_context_rows": None,
              "row_interaction_chunk_rows": 256}
    backbone, _, provenance = load_frozen_backbone(
        problem_type=task.problem_type, device=device,
        classifier_checkpoint=args.classifier_checkpoint, regressor_checkpoint=args.regressor_checkpoint,
    )
    provenance_path = task_dir / "task_provenance.json"
    task_provenance = {"task_id": task.task_id, "outer_split_hash": task.outer_split_hash,
                       "backbone": provenance, "fingerprint": fingerprint}
    if provenance_path.exists():
        previous = _load_json(provenance_path, label="task provenance")
        if previous != task_provenance:
            raise ValueError("task/backbone provenance changed")
    else:
        _write_json(provenance_path, task_provenance)
    all_bags = {arm: [] for arm in ARMS}
    baselines = []
    splits = list(_bag_splits(task, requested_bags=args.bags, seed=_seed(args.protocol_seed, task.task_id, 0)))
    for bag, (fit, heldout) in enumerate(splits):
        fit = np.asarray(fit, dtype=int)
        a, b = _crossfit_validation_halves(task=task, validation_indices=heldout, seed=_seed(args.protocol_seed, task.task_id, bag, 405))
        bundle = _fit_standard_bag(task=task, fit_indices=fit, config=config, protocol_seed=args.protocol_seed, bag=bag, backbone=backbone, device=device)
        if not len(bundle.numerical_indices):
            raise ValueError("feature expansion requires a numerical column")
        baselines.append(_baselines(task, bundle, bag, fit, a, b, device, fingerprint, task_dir))
        for arm in ARMS:
            all_bags[arm].append(_fit_arm(args, task, bundle, bag, fit, a, b, arm, device, fingerprint, task_dir))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del bundle
    records = {}
    arrays = {}
    for name in ("identity", "initial", *ARMS):
        oof, prediction = _assemble(task, baselines, name) if name in ("identity", "initial") else _assemble(task, all_bags[name])
        arrays[f"{name}_oof"], arrays[f"{name}_test"] = oof, prediction
        records[name] = {"oof": _metric_bundle(task.problem_type, task.y_train, oof, task.n_classes),
                         "test": _metric_bundle(task.problem_type, task.y_test, prediction, task.n_classes)}
        if name in ARMS:
            final_prediction = np.mean([bag["final_test"] for bag in all_bags[name]], axis=0)
            arrays[f"{name}_final_test"] = final_prediction
            records[name]["final_test_diagnostic"] = _metric_bundle(task.problem_type, task.y_test, final_prediction, task.n_classes)
            records[name]["bags"] = [bag["metadata"] for bag in all_bags[name]]
            if ARMS[name] is not None:
                linearized = np.mean([0.5 * (bag["linearized_test_a"] + bag["linearized_test_b"]) for bag in all_bags[name]], axis=0)
                arrays[f"{name}_linearized_test"] = linearized
                records[name]["linearized_selected_test_diagnostic"] = _metric_bundle(task.problem_type, task.y_test, linearized, task.n_classes)
    records["ordinary_full_tabiclv2"] = {"test": _metric_bundle(task.problem_type, task.y_test, full, task.n_classes)}
    arrays["ordinary_full_tabiclv2_test"] = full
    result = {"task_id": task.task_id, "dataset_name": task.dataset_name, "problem_type": task.problem_type,
              "outer_split_hash": task.outer_split_hash, "methods": records}
    _write_json(task_dir / "summary.json", result)
    _save_npz(task_dir / "task_predictions.npz", {"fingerprint": fingerprint}, **arrays)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _summary(rows):
    comparisons = {}
    for candidate in ("spline8", "spline20"):
        for reference in ("line", "identity", "initial", "ordinary_full_tabiclv2"):
            paired = [row for row in rows if reference in row["methods"]]
            gains = []
            for row in paired:
                baseline = float(row["methods"][reference]["test"]["benchmark_error"])
                error = float(row["methods"][candidate]["test"]["benchmark_error"])
                gains.append((baseline - error) / baseline if baseline else 0.0)
            if gains:
                comparisons[f"{candidate}_vs_{reference}"] = {
                    "wins": sum(x > 1e-12 for x in gains), "ties": sum(abs(x) <= 1e-12 for x in gains),
                    "losses": sum(x < -1e-12 for x in gains), "mean_relative_gain": float(np.mean(gains)),
                    "median_relative_gain": float(np.median(gains)), "n_tasks": len(paired),
                }
    return {"n_tasks": len(rows), "comparisons": comparisons, "task_results": rows,
            "interpretation": "All capacities are predeclared separate arms. No test-based winner selection, no identity fallback. Final-state test scores are diagnostic only."}


def main():
    args = _parse_args()
    args.source_dir, args.output_dir = args.source_dir.resolve(), args.output_dir.resolve()
    if args.openml_cache_dir:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    manifest = _load_json(args.source_dir / "experiment_manifest.json", label="source manifest")
    cases = _find_source_cases(source_dir=args.source_dir, manifest=manifest, config_label="D", requested_task_ids=set(args.task_id))
    if {case.task_id for case in cases} != set(args.task_id):
        raise ValueError("missing source tasks")
    for case in cases:
        _validate_case(case)
        if case.problem_type not in {"multiclass", "regression"}:
            raise ValueError("pilot supports multiclass and regression only")
    fingerprint = _prepare(args, device)
    rows = []
    for case in cases:
        rows.append(_run_task(args, case, manifest["immutable_run"], device, fingerprint))
        _write_json(args.output_dir / "feature_expansion_summary.json", _summary(rows))
    print(json.dumps(_summary(rows)["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
