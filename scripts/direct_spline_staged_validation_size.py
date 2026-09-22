"""Compare staged-curvature checkpoint selection with 12.5% and 25% held-out rows.

Replays the existing four-bag staged K20 continuation from its saved A/B-selected
line states. Each trajectory is trained on its original T rows. The checkpoint
selected on its original A or B half is compared with the checkpoint selected
on the full A+B fold of the *same* trajectory. Both are deployed unchanged with
the frozen T preprocessor and full T+A+B labelled ICL context. This is a
development diagnostic for checkpoint selection; it does not create new data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from direct_spline_openml_crossfit_blend import _crossfit_validation_halves, _load_source_task, _validate_case
from direct_spline_openml_crossfit_context_expansion import _append_prediction, _load_staged_line_state, _task_dir
from direct_spline_openml_support_audit import _find_source_cases, _load_json, _write_json
from tabicl._experiments.direct_spline_openml import _bag_splits, _cpu_state_dict, _metric_bundle, _seed, load_frozen_backbone
from tabicl._experiments.direct_spline_openml_standard import (
    _candidate_deployment_error, _classification_training_objective_from_logits,
    _cosine_scheduler, _fit_standard_bag, _make_adapters, _normal_prediction,
    _optimizer, _training_logits,
)
from tabicl._experiments.direct_spline_protocol import sample_episode_indices


DEFAULT_TASK_IDS = (4602, 75158, 167186, 361539)
SOURCE_DIR = Path("/home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828")
INITIAL_DIR = Path("/home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_direct_arctan_ablation/dev8_seed20260915_v2/multiclass/direct_arctan_line")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--initial-line-dir", type=Path, default=INITIAL_DIR)
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument("--protocol-seed", type=int, default=20260915)
    parser.add_argument("--bags", type=int, default=4)
    parser.add_argument("--continuation-steps", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--openml-cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.task_id = list(DEFAULT_TASK_IDS if args.task_id is None else args.task_id)
    if len(set(args.task_id)) != len(args.task_id) or args.bags < 2 or args.continuation_steps < 1:
        raise ValueError("task IDs must be unique; bags >= 2; continuation steps >= 1")
    return args


def _matched_config(case_config: Mapping[str, Any], *, steps: int) -> dict[str, Any]:
    config = dict(case_config)
    if str(config.get("adapter_architecture", "fixed_cubic")) != "fixed_cubic" or int(config.get("n_control_points", 20)) != 20:
        raise ValueError("staged replay requires the source fixed cubic K20 arm")
    config.update(
        trainable_shape=True, direct_spline_output=True, trainable_location_scale=False,
        coordinate_mapping="arctan", query_fraction_min=0.05, query_fraction_max=0.20,
        adapter_steps=int(steps), adapter_patience=None,
    )
    if config.get("cosine_schedule_steps") is not None:
        config["cosine_schedule_steps"] = int(steps)
    return config


def _select_best(
    records: list[dict[str, Any]], *, size: str
) -> int:
    """Choose the earliest checkpoint at minimum valid error."""
    candidates = [(float(item[f"{size}_error"]), int(item["step"])) for item in records]
    return min(candidates, key=lambda pair: (pair[0], pair[1]))[1]


def _bag_run(
    *, task: Any, bag: int, fit_indices: np.ndarray, validation_indices: np.ndarray,
    config: dict[str, Any], initial_dir: Path, protocol_seed: int, backbone: Any,
    device: torch.device, output_path: Path,
) -> dict[str, Any]:
    a_indices, b_indices = _crossfit_validation_halves(
        task=task, validation_indices=validation_indices,
        seed=_seed(protocol_seed, task.task_id, bag, 405),
    )
    bundle = _fit_standard_bag(
        task=task, fit_indices=fit_indices, config=config, protocol_seed=protocol_seed,
        bag=bag, backbone=backbone, device=device,
    )
    adapter_seed = _seed(int(config["random_state"]), task.task_id, bag, 202)
    torch.manual_seed(adapter_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(adapter_seed)
    adapters = _make_adapters(bundle, config, device)
    if adapters is None:
        raise ValueError(f"task {task.task_id} has no trainable numerical adapter")
    initial_path = _task_dir(initial_dir, task) / f"bag_{bag}.adapters.pt"
    payload = torch.load(initial_path, map_location="cpu", weights_only=True)
    provenance = payload.get("provenance", {})
    for name, expected in (
        ("task_id", task.task_id), ("bag", bag), ("outer_split_hash", task.outer_split_hash),
        ("protocol_seed", protocol_seed),
    ):
        if provenance.get(name) != expected:
            raise ValueError(f"initial line checkpoint {initial_path} has wrong {name}")
    source_config = payload.get("config", {})
    if not source_config.get("direct_spline_output") or source_config.get("trainable_shape"):
        raise ValueError("initial checkpoint must be the matched direct-line arm")
    for name in (
        "adapter_architecture", "coordinate_mapping", "query_fraction_min",
        "query_fraction_max", "random_state", "cross_column_mixing_rank",
        "cross_column_mixing_bound",
    ):
        if source_config.get(name) != config.get(name):
            raise ValueError(f"staged replay changed {name} from the initial line")
    labels = {"a": np.asarray(task.y_train[a_indices]), "b": np.asarray(task.y_train[b_indices])}
    x = {"a": task.x_train.iloc[a_indices].reset_index(drop=True), "b": task.x_train.iloc[b_indices].reset_index(drop=True)}
    both_x = task.x_train.iloc[validation_indices].reset_index(drop=True)
    both_y = np.asarray(task.y_train[validation_indices])
    state_payload: dict[str, Any] = {}
    bag_result: dict[str, Any] = {
        "task_id": task.task_id, "dataset_name": task.dataset_name, "problem_type": task.problem_type,
        "bag": bag, "outer_split_hash": task.outer_split_hash,
        "fit_rows": int(fit_indices.size), "a_rows": int(a_indices.size),
        "b_rows": int(b_indices.size), "large_rows": int(validation_indices.size),
        "fit_indices_sha256": hashlib.sha256(np.asarray(fit_indices, dtype=np.int64).tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(np.asarray(validation_indices, dtype=np.int64).tobytes()).hexdigest(),
        "initial_line_checkpoint_sha256": _hash(initial_path),
        "trajectories": {},
    }
    identity_test = _append_prediction(
        bundle=bundle, task=task, query_x=task.x_test,
        appended_indices=validation_indices, adapters=None, device=device,
    )
    prediction_payload: dict[str, np.ndarray] = {"identity_test": identity_test}
    for half in ("a", "b"):
        source_name = f"original_{half}"
        if source_name not in payload.get("checkpoints", {}):
            raise ValueError(f"initial checkpoint lacks {source_name}")
        _load_staged_line_state(adapters, payload["checkpoints"][source_name]["state_dict"])
        prediction_payload[f"{half}_line_test"] = _append_prediction(
            bundle=bundle, task=task, query_x=task.x_test,
            appended_indices=validation_indices, adapters=adapters, device=device,
        )
        optimizer = _optimizer(adapters, config)
        scheduler = None if config.get("cosine_schedule_steps") is None else _cosine_scheduler(
            optimizer, total_steps=int(config["cosine_schedule_steps"]),
            min_lr_ratio=float(config["cosine_min_lr_ratio"]),
        )
        rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 203))
        best_errors = {"small": float("inf"), "large": float("inf")}
        best_states: dict[str, Any] = {}
        records: list[dict[str, Any]] = []
        for step in range(int(config["adapter_steps"]) + 1):
            if step:
                configured_context = config.get("train_context_rows")
                context_limit = (
                    max(1, bundle.fit_labels.size - 1)
                    if configured_context is None else int(configured_context)
                )
                context_rows, query_rows = sample_episode_indices(
                    bundle.fit_labels, problem_type=task.problem_type,
                    context_rows=context_limit, query_rows=int(config["query_batch_rows"]), rng=rng,
                    query_fraction_range=(float(config["query_fraction_min"]), float(config["query_fraction_max"])),
                )
                optimizer.zero_grad(set_to_none=True)
                output = _training_logits(
                    bundle=bundle, adapters=adapters,
                    context_indices=context_rows, query_indices=query_rows, device=device,
                )
                target = torch.as_tensor(bundle.fit_labels[query_rows], device=device)
                objective = _classification_training_objective_from_logits(
                    logits=output, target=target, problem_type=task.problem_type,
                    n_classes=task.n_classes,
                    softmax_temperature=float(bundle.estimator.softmax_temperature), config=config,
                )
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"nonfinite staged objective task {task.task_id} bag {bag} step {step}")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(adapters.parameters(), float(config["grad_clip"]))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                del output, target, objective
            if step and step % int(config["validation_interval"]) != 0 and step != int(config["adapter_steps"]):
                continue
            small_prediction = _normal_prediction(
                bundle=bundle, query_x=x[half], context_indices=bundle.support_indices,
                adapters=adapters, device=device,
            )
            large_prediction = _normal_prediction(
                bundle=bundle, query_x=both_x, context_indices=bundle.support_indices,
                adapters=adapters, device=device,
            )
            errors = {
                "small": _candidate_deployment_error(task.problem_type, labels[half], small_prediction, n_classes=task.n_classes),
                "large": _candidate_deployment_error(task.problem_type, both_y, large_prediction, n_classes=task.n_classes),
            }
            records.append({"step": step, "small_error": float(errors["small"]), "large_error": float(errors["large"])})
            for size in ("small", "large"):
                if errors[size] < best_errors[size]:
                    best_errors[size] = float(errors[size])
                    best_states[size] = _cpu_state_dict(adapters)
            print(f"task {task.task_id} bag {bag} {half} checkpoint {step}: small={errors['small']:.6g} large={errors['large']:.6g}", flush=True)
        del optimizer, scheduler
        if set(best_states) != {"small", "large"}:
            raise RuntimeError("no finite small/large selection checkpoint")
        chosen = {size: _select_best(records, size=size) for size in ("small", "large")}
        bag_result["trajectories"][half] = {"checkpoints": records, "selected_steps": chosen}
        for size in ("small", "large"):
            adapters.load_state_dict(best_states[size], strict=True)
            prediction_payload[f"{half}_{size}_test"] = _append_prediction(
                bundle=bundle, task=task, query_x=task.x_test,
                appended_indices=validation_indices, adapters=adapters, device=device,
            )
            state_payload[f"{half}_{size}"] = best_states[size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **prediction_payload)
    adapter_path = output_path.with_suffix(".adapters.pt")
    torch.save({"states": state_payload, "config": config, "task_id": task.task_id,
                "bag": bag, "outer_split_hash": task.outer_split_hash}, output_path.with_suffix(".adapters.pt"))
    bag_result["prediction_artifact_sha256"] = _hash(output_path)
    bag_result["adapter_artifact_sha256"] = _hash(adapter_path)
    del bundle, adapters
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bag_result


def _score_task(*, task: Any, bag_records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    identity_members = []
    members: dict[str, list[np.ndarray]] = {"small": [], "large": [], "line": []}
    for record in bag_records:
        path = output_dir / "raw" / f"task_{task.task_id}" / f"bag_{record['bag']}.npz"
        with np.load(path, allow_pickle=False) as saved:
            identity_members.append(np.asarray(saved["identity_test"]))
            members["line"].append(np.asarray(saved["a_line_test"]))
            members["line"].append(np.asarray(saved["b_line_test"]))
            for size in ("small", "large"):
                members[size].extend((np.asarray(saved[f"a_{size}_test"]), np.asarray(saved[f"b_{size}_test"])))
    identity = np.mean(np.stack(identity_members), axis=0)
    predictions = {size: np.mean(np.stack(values), axis=0) for size, values in members.items() if values}
    return {
        "task_id": task.task_id, "dataset_name": task.dataset_name,
        "problem_type": task.problem_type, "outer_split_hash": task.outer_split_hash,
        "identity": _metric_bundle(task.problem_type, task.y_test, identity, task.n_classes),
        "line": _metric_bundle(task.problem_type, task.y_test, predictions["line"], task.n_classes),
        "small": _metric_bundle(task.problem_type, task.y_test, predictions["small"], task.n_classes),
        "large": _metric_bundle(task.problem_type, task.y_test, predictions["large"], task.n_classes),
        "n_bags": len(bag_records),
        "changed_checkpoint_count": sum(
            int(trajectory["selected_steps"]["small"] != trajectory["selected_steps"]["large"])
            for record in bag_records for trajectory in record["trajectories"].values()
        ),
    }


def main() -> None:
    args = _parse()
    if args.openml_cache_dir is not None:
        os.environ["OPENML_CACHE_DIR"] = str(args.openml_cache_dir.resolve())
    source_manifest = _load_json(args.source_dir / "experiment_manifest.json", label="source manifest")
    initial_manifest = _load_json(args.initial_line_dir / "experiment_manifest.json", label="initial line manifest")
    if int(initial_manifest.get("protocol_seed", -1)) != args.protocol_seed:
        raise ValueError("initial line protocol seed differs")
    if str(initial_manifest.get("source_dir")) != str(args.source_dir.resolve()):
        raise ValueError("initial line and source directory differ")
    immutable_run = source_manifest.get("immutable_run")
    if not isinstance(immutable_run, Mapping):
        raise ValueError("source manifest lacks immutable run")
    cases = _find_source_cases(
        source_dir=args.source_dir, manifest=source_manifest,
        config_label="D", requested_task_ids=set(args.task_id),
    )
    if {case.task_id for case in cases} != set(args.task_id):
        raise ValueError("requested task absent from source run")
    for case in cases:
        _validate_case(case)
        if case.problem_type != "multiclass":
            raise ValueError("this diagnostic is scoped to multiclass")
    manifest = {
        "experiment": "staged K20 checkpoint selection size on fixed 75% fit bags",
        "task_ids": sorted(args.task_id), "bags": args.bags,
        "protocol_seed": args.protocol_seed, "continuation_steps": args.continuation_steps,
        "source_manifest_sha256": _hash(args.source_dir / "experiment_manifest.json"),
        "initial_line_manifest_sha256": _hash(args.initial_line_dir / "experiment_manifest.json"),
        "source_dir": str(args.source_dir.resolve()),
        "initial_line_dir": str(args.initial_line_dir.resolve()),
        "implementation_sha256": _hash(Path(__file__)),
        "training_semantics_sha256": _hash(Path(__file__).resolve().parents[1] / "src/tabicl/_experiments/direct_spline_openml_standard.py"),
        "selection": "A or B (~12.5%) versus A+B (~25%) on each same staged continuation trajectory; earliest minimum, including inherited line state",
        "deployment": "unchanged adapter, frozen T preprocessor, full T+A+B ICL context",
    }
    manifest_path = args.output_dir / "experiment_manifest.json"
    if manifest_path.exists():
        if _load_json(manifest_path, label="existing manifest") != manifest or not args.resume:
            raise ValueError("output directory belongs to a different run, or requires --resume")
    elif args.resume:
        raise FileNotFoundError("cannot resume without experiment manifest")
    else:
        _write_json(manifest_path, manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summaries = []
    for case in cases:
        task = _load_source_task(case=case, immutable_run=immutable_run)
        config = _matched_config(case.config, steps=args.continuation_steps)
        splits = list(_bag_splits(task, requested_bags=args.bags, seed=_seed(args.protocol_seed, task.task_id, 0)))
        if len(splits) != args.bags:
            raise ValueError(f"task {task.task_id} cannot support all {args.bags} bags")
        bag_records = []
        missing = [
            (bag, np.asarray(fit, dtype=int), np.asarray(heldout, dtype=int))
            for bag, (fit, heldout) in enumerate(splits)
            if not (args.resume and (args.output_dir / "raw" / f"task_{task.task_id}" / f"bag_{bag}.json").is_file()
                    and (args.output_dir / "raw" / f"task_{task.task_id}" / f"bag_{bag}.npz").is_file()
                    and (args.output_dir / "raw" / f"task_{task.task_id}" / f"bag_{bag}.adapters.pt").is_file())
        ]
        backbone = None
        if missing:
            backbone, _path, _metadata = load_frozen_backbone(
                problem_type=task.problem_type, device=device,
                classifier_checkpoint=args.classifier_checkpoint, regressor_checkpoint=None,
            )
        for bag, fit, heldout in enumerate(splits):
            destination = args.output_dir / "raw" / f"task_{task.task_id}" / f"bag_{bag}.npz"
            record_path = destination.with_suffix(".json")
            if any(item[0] == bag for item in missing):
                print(f"task {task.task_id} bag {bag}: replay staged continuation", flush=True)
                record = _bag_run(
                    task=task, bag=bag, fit_indices=np.asarray(fit, dtype=int),
                    validation_indices=np.asarray(heldout, dtype=int), config=config,
                    initial_dir=args.initial_line_dir, protocol_seed=args.protocol_seed,
                    backbone=backbone, device=device, output_path=destination,
                )
                _write_json(record_path, record)
            else:
                record = _load_json(record_path, label="completed bag")
                if record["outer_split_hash"] != task.outer_split_hash or record["fit_indices_sha256"] != hashlib.sha256(np.asarray(fit, dtype=np.int64).tobytes()).hexdigest() or record["validation_indices_sha256"] != hashlib.sha256(np.asarray(heldout, dtype=np.int64).tobytes()).hexdigest():
                    raise ValueError("completed bag has different immutable split")
                if record.get("prediction_artifact_sha256") != _hash(destination) or record.get("adapter_artifact_sha256") != _hash(destination.with_suffix(".adapters.pt")):
                    raise ValueError("completed bag artifact checksum mismatch")
            bag_records.append(record)
        del backbone
        summary = _score_task(task=task, bag_records=bag_records, output_dir=args.output_dir)
        _write_json(args.output_dir / "task_summaries" / f"task_{task.task_id}.json", summary)
        summaries.append(summary)
        _write_json(args.output_dir / "summary.json", {"task_results": summaries, "n_completed": len(summaries), "n_requested": len(cases)})
    print(json.dumps({"n_completed": len(summaries), "task_results": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
