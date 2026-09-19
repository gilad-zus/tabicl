"""Audit whether selected spline curvature fits training rows but fails to generalize.

This is an inference-only audit of a completed staged-curvature experiment.  It
loads the exact selected continued-line and continued-spline checkpoints and
scores them, plus unadapted TabICLv2, on matched deterministic episodes sampled
only from each bag's adapter-training rows.  Existing cross-fitted OOF and
outer-test results are copied from the completed staged experiment, so no
checkpoint is selected and no parameter is updated here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from direct_spline_openml_support_audit import _find_source_cases, _load_json, _write_json
from direct_spline_openml_crossfit_blend import _load_source_task, _validate_case
from tabicl._experiments.direct_spline_openml import _bag_splits, _safe_name, _seed, load_frozen_backbone
from tabicl._experiments.direct_spline_openml_standard import (
    _classification_training_objective_from_logits,
    _fit_standard_bag,
    _make_adapters,
    _training_logits,
)
from tabicl._experiments.direct_spline_protocol import sample_episode_indices


ARMS = ("continued_line", "continued_spline")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", required=True)
    parser.add_argument("--config-label", default="D")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--audit-seed", type=int, default=20260919)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--regressor-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_id)) != len(args.task_id):
        raise ValueError("--task-id values must be unique")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    return args


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(args: argparse.Namespace) -> None:
    manifests = {
        arm: args.staged_dir / arm / "experiment_manifest.json"
        for arm in ARMS
    }
    manifest = {
        "experiment": "selected-checkpoint train/OOF/test generalization audit",
        "source_dir": str(args.source_dir.resolve()),
        "staged_dir": str(args.staged_dir.resolve()),
        "staged_manifest_sha256": {arm: _sha256(path) for arm, path in manifests.items()},
        "task_ids": sorted(args.task_id),
        "config_label": args.config_label,
        "protocol_seed": args.protocol_seed,
        "audit_seed": args.audit_seed,
        "episodes_per_bag": args.episodes,
        "training_evaluation": (
            "Matched deterministic context/query episodes sampled from each bag's T rows; "
            "query labels are targets only and never enter their episode context."
        ),
        "no_training": True,
        "implementation_sha256": _sha256(Path(__file__)),
    }
    path = args.output_dir / "experiment_manifest.json"
    if path.exists():
        if _canonical(_load_json(path, label="audit manifest")) != _canonical(manifest):
            raise ValueError("output directory belongs to a different audit")
        if not args.resume:
            raise ValueError("output directory exists; pass --resume")
    else:
        _write_json(path, manifest)


def _task_dir(root: Path, task_id: int, dataset_name: str) -> Path:
    return root / "raw" / f"task_{task_id}_{_safe_name(dataset_name)}"


def _objective(*, task: Any, bundle: Any, adapters: Any, context: np.ndarray, query: np.ndarray, device: torch.device, config: Mapping[str, Any]) -> float:
    with torch.no_grad():
        output = _training_logits(
            bundle=bundle, adapters=adapters, context_indices=context,
            query_indices=query, device=device,
        )
        target = torch.as_tensor(bundle.fit_labels[query], device=device)
        if task.problem_type == "regression":
            loss = F.mse_loss(output.flatten(), target.float().flatten())
        else:
            loss = _classification_training_objective_from_logits(
                logits=output,
                target=target,
                problem_type=task.problem_type,
                n_classes=task.n_classes,
                softmax_temperature=float(bundle.estimator.softmax_temperature),
                config=dict(config),
            )
    return float(loss)


def _load_checkpoint(path: Path, *, task: Any, bag: int, protocol_seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing selected adapter checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    provenance = payload.get("provenance", {})
    expected = (int(task.task_id), int(bag), str(task.outer_split_hash), int(protocol_seed))
    actual = (
        int(provenance.get("task_id", -1)), int(provenance.get("bag", -1)),
        str(provenance.get("outer_split_hash", "")), int(provenance.get("protocol_seed", -1)),
    )
    if actual != expected:
        raise ValueError(f"checkpoint provenance mismatch for {path}: {actual} != {expected}")
    if set(payload.get("checkpoints", {})) != {"original_a", "original_b"}:
        raise ValueError(f"checkpoint lacks the two selected states: {path}")
    return payload


def _arm_existing_metrics(staged_summary: Mapping[str, Any], task_id: int) -> dict[str, float]:
    row = next(item for item in staged_summary["task_results"] if int(item["task_id"]) == task_id)
    return {
        "oof_raw_relative_curvature_gain": float(row["oof_raw_spline_relative_curvature_gain"]),
        "outer_test_raw_relative_curvature_gain": float(row["outer_test_raw_spline_relative_curvature_gain"]),
        "outer_test_selected_relative_curvature_gain": float(row["outer_test_selected_blend_relative_curvature_gain"]),
    }


def _run_task(
    *,
    args: argparse.Namespace,
    task: Any,
    requested_bags: int,
    staged_summary: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    splits = list(
        _bag_splits(
            task,
            requested_bags=requested_bags,
            seed=_seed(args.protocol_seed, task.task_id, 0),
        )
    )
    backbone, _path, _metadata = load_frozen_backbone(
        problem_type=task.problem_type, device=device,
        classifier_checkpoint=args.classifier_checkpoint,
        regressor_checkpoint=args.regressor_checkpoint,
    )
    losses = {"identity": [], "continued_line": [], "continued_spline": []}
    selected_steps = {arm: [] for arm in ARMS}
    episode_sizes: list[dict[str, int]] = []
    for bag, (fit_indices, _validation_indices) in enumerate(splits):
        checkpoints = {
            arm: _load_checkpoint(
                _task_dir(args.staged_dir / arm, task.task_id, task.dataset_name) / f"bag_{bag}.adapters.pt",
                task=task, bag=bag, protocol_seed=args.protocol_seed,
            )
            for arm in ARMS
        }
        line_config = dict(checkpoints["continued_line"]["config"])
        spline_config = dict(checkpoints["continued_spline"]["config"])
        structural_fields = (
            "adapter_architecture", "n_control_points", "coordinate_mapping",
            "query_fraction_min", "query_fraction_max", "random_state",
            "cross_column_mixing_rank", "cross_column_mixing_bound",
        )
        if any(line_config.get(name) != spline_config.get(name) for name in structural_fields):
            raise ValueError(f"line/spline checkpoint configs are not matched for task {task.task_id} bag {bag}")
        bundle = _fit_standard_bag(
            task=task, fit_indices=np.asarray(fit_indices, dtype=int), config=line_config,
            protocol_seed=args.protocol_seed, bag=bag, backbone=backbone, device=device,
        )
        adapters = {}
        for arm, payload in checkpoints.items():
            adapters[arm] = _make_adapters(bundle, dict(payload["config"]), device)
            if adapters[arm] is None:
                raise ValueError(f"task {task.task_id} bag {bag} has no numerical adapter")
        rng = np.random.default_rng(_seed(args.audit_seed, task.task_id, bag, 901))
        episodes = []
        for _ in range(args.episodes):
            context_limit = max(1, bundle.fit_labels.size - 1)
            context, query = sample_episode_indices(
                bundle.fit_labels, problem_type=task.problem_type,
                context_rows=context_limit,
                query_rows=int(line_config["query_batch_rows"]), rng=rng,
                query_fraction_range=(
                    float(line_config["query_fraction_min"]),
                    float(line_config["query_fraction_max"]),
                ),
            )
            episodes.append((context, query))
            episode_sizes.append({"context_rows": int(context.size), "query_rows": int(query.size)})
        for checkpoint_name in ("original_a", "original_b"):
            for arm in ARMS:
                record = checkpoints[arm]["checkpoints"][checkpoint_name]
                adapters[arm].load_state_dict(record["state_dict"], strict=True)
                selected_steps[arm].append(int(record["step"]))
            for context, query in episodes:
                losses["identity"].append(_objective(
                    task=task, bundle=bundle, adapters=None, context=context, query=query,
                    device=device, config=line_config,
                ))
                for arm in ARMS:
                    losses[arm].append(_objective(
                        task=task, bundle=bundle, adapters=adapters[arm], context=context,
                        query=query, device=device, config=checkpoints[arm]["config"],
                    ))
        del adapters, bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del backbone
    means = {name: float(np.mean(values)) for name, values in losses.items()}
    result = {
        "task_id": int(task.task_id), "dataset_name": task.dataset_name,
        "problem_type": task.problem_type, "bags": len(splits),
        "episodes_per_checkpoint": args.episodes,
        "checkpoint_evaluations_per_arm": len(losses["continued_line"]),
        "training_objective": "mse_on_scaled_target" if task.problem_type == "regression" else "cross_entropy",
        "training_episode_loss": means,
        "training_relative_curvature_gain": (
            (means["continued_line"] - means["continued_spline"]) / means["continued_line"]
            if means["continued_line"] else 0.0
        ),
        "training_relative_line_gain_vs_identity": (
            (means["identity"] - means["continued_line"]) / means["identity"] if means["identity"] else 0.0
        ),
        "training_relative_spline_gain_vs_identity": (
            (means["identity"] - means["continued_spline"]) / means["identity"] if means["identity"] else 0.0
        ),
        "selected_step_zero_counts": {
            arm: sum(step == 0 for step in steps) for arm, steps in selected_steps.items()
        },
        "episode_context_rows": {
            "min": min(item["context_rows"] for item in episode_sizes),
            "max": max(item["context_rows"] for item in episode_sizes),
        },
        "episode_query_rows": {
            "min": min(item["query_rows"] for item in episode_sizes),
            "max": max(item["query_rows"] for item in episode_sizes),
        },
        **_arm_existing_metrics(staged_summary, int(task.task_id)),
    }
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "training_relative_curvature_gain", "oof_raw_relative_curvature_gain",
        "outer_test_raw_relative_curvature_gain", "outer_test_selected_relative_curvature_gain",
    )
    comparisons = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows])
        comparisons[field] = {
            "wins": int(np.sum(values > 0)), "ties": int(np.sum(values == 0)),
            "losses": int(np.sum(values < 0)), "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
    return {
        "n_tasks": len(rows), "comparisons": comparisons,
        "train_win_test_loss_datasets": [
            row["dataset_name"] for row in rows
            if row["training_relative_curvature_gain"] > 0
            and row["outer_test_raw_relative_curvature_gain"] < 0
        ],
        "task_results": rows,
        "interpretation": (
            "Training uses matched fixed train-only episodes on the exact selected checkpoints. "
            "OOF and test values are copied from the completed staged experiment."
        ),
    }


def main() -> None:
    args = _parse_args()
    args.source_dir = args.source_dir.resolve()
    args.staged_dir = args.staged_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    _prepare(args)
    staged_summary = _load_json(
        args.staged_dir / "staged_curvature_ablation_summary.json", label="staged summary"
    )
    staged_manifest = _load_json(
        args.staged_dir / "experiment_manifest.json", label="staged manifest"
    )
    requested_bags = int(staged_manifest["bags"])
    source_manifest = _load_json(args.source_dir / "experiment_manifest.json", label="source manifest")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest has no immutable_run")
    cases = _find_source_cases(
        source_dir=args.source_dir, manifest=source_manifest, config_label=args.config_label,
        requested_task_ids=set(args.task_id),
    )
    if {case.task_id for case in cases} != set(args.task_id):
        raise ValueError("one or more task IDs are absent from the source run")
    for case in cases:
        _validate_case(case)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested as {args.device!r}, but it is unavailable")
    rows = []
    for position, case in enumerate(cases, 1):
        print(f"[{position}/{len(cases)}] task {case.task_id} {case.dataset_name}", flush=True)
        task = _load_source_task(case=case, immutable_run=immutable_run)
        rows.append(_run_task(
            args=args,
            task=task,
            requested_bags=requested_bags,
            staged_summary=staged_summary,
            device=device,
        ))
    rows.sort(key=lambda item: int(item["task_id"]))
    result = _aggregate(rows)
    _write_json(args.output_dir / "generalization_audit_summary.json", result)
    with (args.output_dir / "generalization_audit_results.csv").open("w", newline="", encoding="utf-8") as handle:
        flat_rows = [
            {key: value for key, value in row.items() if not isinstance(value, dict)}
            for row in rows
        ]
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat_rows for key in row}))
        writer.writeheader(); writer.writerows(flat_rows)
    print(json.dumps(result["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
