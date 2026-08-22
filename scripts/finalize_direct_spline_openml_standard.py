"""CPU-only finalizer for an interrupted standard DirectSpline experiment.

This command never constructs a TabICL estimator, loads a model checkpoint, or
fits an adapter.  It verifies the immutable manifest and every saved bag,
rebuilds the small per-configuration aggregation artifacts from those bags,
then writes the ordinary task and experiment reports.  It is intended for a
run that has finished its bags but was interrupted while writing summaries.

It refuses to claim a final result if an unskipped task is missing a bag.  In
that case the printed finalization report names the remaining task/config/bag
artifacts; normal GPU training is still required for those files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from tabicl._experiments.direct_spline_openml import (
    BagPredictions,
    OpenMLTaskData,
    _bag_splits,
    _config_dir,
    _json_dump,
    _json_load,
    _load_bag,
    _metric_bundle,
    _prediction_shape,
    _safe_name,
    _seed,
    load_tabarena_openml_task,
    summarize_experiment,
    summarize_task_tuning,
)
from tabicl._experiments.direct_spline_openml_standard import _candidate_deployment_error
from tabicl._experiments.direct_spline_protocol import choose_identity_guard, deployment_error


class MissingTrainingArtifacts(RuntimeError):
    """Raised when report reconstruction would require a new model fit."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when saved artifacts do not safely belong to this immutable run."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = _json_load(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"{description} is unreadable: {path} ({type(error).__name__}: {error})") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"{description} is not a JSON object: {path}")
    return payload


def _quarantine_malformed_config_summary(path: Path) -> Path | None:
    """Preserve a malformed final JSON file before replacing it from bags."""

    if not path.is_file():
        return None
    try:
        payload = _json_load(path)
        if isinstance(payload, dict):
            return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for counter in range(10_000):
        suffix = "" if counter == 0 else f"-{counter}"
        target = path.with_name(f"{path.name}.interrupted-{timestamp}{suffix}")
        if not target.exists():
            path.replace(target)
            return target
    raise RuntimeError(f"could not choose a quarantine path for {path}")


def _expected_checkpoint(manifest_run: dict[str, Any], task: OpenMLTaskData) -> dict[str, Any]:
    kind = "regressor" if task.problem_type == "regression" else "classifier"
    fingerprints = manifest_run.get("checkpoint_fingerprints")
    if not isinstance(fingerprints, dict) or not isinstance(fingerprints.get(kind), dict):
        raise ArtifactIntegrityError(f"manifest has no immutable {kind} checkpoint fingerprint")
    return fingerprints[kind]


def _validate_bag(
    *,
    result: BagPredictions,
    path: Path,
    expected_bag: int,
    expected_validation_indices: np.ndarray,
    expected_test_shape: tuple[int, ...],
    run_fingerprint_hash: str,
) -> None:
    metadata = result.metadata
    if metadata.get("run_fingerprint_hash") != run_fingerprint_hash:
        raise ArtifactIntegrityError(f"{path} belongs to a different immutable run fingerprint")
    if int(metadata.get("bag", -1)) != expected_bag:
        raise ArtifactIntegrityError(f"{path} has bag={metadata.get('bag')!r}; expected {expected_bag}")
    if not np.array_equal(result.validation_indices, expected_validation_indices):
        raise ArtifactIntegrityError(f"{path} has validation indices different from the frozen bag split")
    expected_validation_shape = (len(expected_validation_indices), *expected_test_shape[1:])
    for name in ("identity_validation", "adapted_validation", "guarded_validation"):
        value = getattr(result, name)
        if value.shape != expected_validation_shape or not np.isfinite(value).all():
            raise ArtifactIntegrityError(f"{path} has invalid {name} predictions")
    for name in ("identity_test", "adapted_test", "guarded_test"):
        value = getattr(result, name)
        if value.shape != expected_test_shape:
            raise ArtifactIntegrityError(f"{path} has {name} shape {value.shape}; expected {expected_test_shape}")


def rebuild_standard_config_from_bags(
    *,
    task: OpenMLTaskData,
    label: str,
    config: dict[str, Any],
    output_dir: Path,
    requested_bags: int,
    protocol_seed: int,
    run_fingerprint_hash: str,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Recreate one standard config's derived files using only persisted bags."""

    config_dir = _config_dir(output_dir, task, label)
    bag_results: list[BagPredictions] = []
    expected_test_shape = _prediction_shape(len(task.y_test), task.problem_type, task.n_classes)
    splits = list(_bag_splits(task, requested_bags=requested_bags, seed=_seed(protocol_seed, task.task_id, 0)))
    for bag, (_, validation_indices) in enumerate(splits):
        bag_path = config_dir / f"bag_{bag}.npz"
        if not bag_path.is_file():
            raise MissingTrainingArtifacts(f"missing trained bag: task={task.task_id} config={label} bag={bag} ({bag_path})")
        try:
            result = _load_bag(bag_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(f"cannot read {bag_path}: {type(error).__name__}: {error}") from error
        _validate_bag(
            result=result,
            path=bag_path,
            expected_bag=bag,
            expected_validation_indices=validation_indices,
            expected_test_shape=expected_test_shape,
            run_fingerprint_hash=run_fingerprint_hash,
        )
        bag_results.append(result)

    train_shape = _prediction_shape(len(task.y_train), task.problem_type, task.n_classes)
    identity_validation = np.full(train_shape, np.nan, dtype=np.float64)
    adapted_validation = np.full(train_shape, np.nan, dtype=np.float64)
    guarded_validation = np.full(train_shape, np.nan, dtype=np.float64)
    for result in bag_results:
        identity_validation[result.validation_indices] = result.identity_validation
        adapted_validation[result.validation_indices] = result.adapted_validation
        guarded_validation[result.validation_indices] = result.guarded_validation
    if not (
        np.isfinite(identity_validation).all()
        and np.isfinite(adapted_validation).all()
        and np.isfinite(guarded_validation).all()
    ):
        raise ArtifactIntegrityError(f"task {task.task_id} config {label} has incomplete OOF predictions")

    identity_test = np.mean([result.identity_test for result in bag_results], axis=0)
    adapted_test = np.mean([result.adapted_test for result in bag_results], axis=0)
    guarded_test = np.mean([result.guarded_test for result in bag_results], axis=0)
    identity_oof_error = deployment_error(
        task.problem_type, task.y_train, identity_validation, n_classes=task.n_classes
    )
    adapted_oof_error = _candidate_deployment_error(
        task.problem_type, task.y_train, adapted_validation, n_classes=task.n_classes
    )
    config_guard = choose_identity_guard(
        identity_error=identity_oof_error,
        adapted_error=adapted_oof_error,
        required_relative_improvement=float(config["guard_relative_improvement"]),
    )
    summary_path = config_dir / "config_summary.json"
    quarantined = _quarantine_malformed_config_summary(summary_path)
    if quarantined is not None:
        print(f"Quarantined malformed final artifact: {quarantined}", flush=True)
    np.savez_compressed(
        config_dir / "config_predictions.npz",
        identity_validation=identity_validation,
        adapted_validation=adapted_validation,
        guarded_validation=guarded_validation,
        identity_test=identity_test,
        adapted_test=adapted_test,
        guarded_test=guarded_test,
    )
    summary = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "config_label": label,
        "config": config,
        "checkpoint": checkpoint_metadata,
        "run_fingerprint_hash": run_fingerprint_hash,
        "requested_bags": requested_bags,
        "effective_bags": len(bag_results),
        "pipeline": "standard_ensemble",
        "identity_definition": (
            "Normal TabICLv2 fitted independently on each inner-bag fit partition, with the same preprocessing "
            "views, class/feature shuffles, temperature, and logit aggregation as the adapted path."
        ),
        "context_policy": (
            "all_fit_rows" if config.get("max_context_rows") is None else "stratified_context_cap_when_needed"
        ),
        "validation": {
            "identity": _metric_bundle(task.problem_type, task.y_train, identity_validation, task.n_classes),
            "adapted": _metric_bundle(task.problem_type, task.y_train, adapted_validation, task.n_classes),
            "guarded": _metric_bundle(task.problem_type, task.y_train, guarded_validation, task.n_classes),
        },
        "test_metrics_deferred_to_task_summary": True,
        "guard_protocol": "retouche_per_bag_validation_guard_then_test_ensemble",
        "guard_selected_adapted_fraction": float(
            np.mean([bool(result.metadata["guard_selected_adapted"]) for result in bag_results])
        ),
        "adapter_valid_learned_checkpoint_fraction": float(
            np.mean([bool(result.metadata["adapter_has_valid_learned_checkpoint"]) for result in bag_results])
        ),
        "global_oof_guard_selected_adapted_diagnostic": bool(config_guard.use_adapted),
        "global_oof_guard_relative_improvement_diagnostic": float(config_guard.relative_improvement),
        "mean_train_seconds_per_bag": float(np.mean([float(result.metadata["train_seconds"]) for result in bag_results])),
        "max_peak_allocated_gib": float(np.max([float(result.metadata["peak_allocated_gib"]) for result in bag_results])),
        "max_identity_parity_abs": float(
            np.max(
                [
                    max(
                        float(result.metadata["identity_parity_max_abs_validation"]),
                        float(result.metadata["identity_parity_max_abs_test"]),
                    )
                    for result in bag_results
                ]
            )
        ),
        "bag_metadata": [result.metadata for result in bag_results],
    }
    _json_dump(summary_path, summary)
    return summary


def _recorded_skips(output_dir: Path) -> dict[int, dict[str, Any]]:
    progress_path = output_dir / "run_progress.json"
    if not progress_path.is_file():
        return {}
    payload = _mapping(progress_path, description="run progress")
    raw_skips = payload.get("skipped_tasks", [])
    if not isinstance(raw_skips, list):
        raise ArtifactIntegrityError(f"{progress_path} has invalid skipped_tasks")
    return {int(item["task_id"]): item for item in raw_skips if isinstance(item, dict) and "task_id" in item}


def _standard_baseline_oom_skips(output_dir: Path) -> set[int]:
    """Find normal-baseline OOMs that still permit a DirectSpline task result."""

    progress_path = output_dir / "progress.jsonl"
    if not progress_path.is_file():
        return set()
    skipped: set[int] = set()
    for line_number, line in enumerate(progress_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactIntegrityError(f"invalid progress event at {progress_path}:{line_number}") from error
        if (
            isinstance(event, dict)
            and event.get("event") == "standard_baseline_skipped"
            and event.get("reason") == "cuda_out_of_memory"
            and "task_id" in event
        ):
            skipped.add(int(event["task_id"]))
    return skipped


def _standard_baseline(
    *,
    output_dir: Path,
    task: OpenMLTaskData,
    run_fingerprint_hash: str,
    required: bool,
) -> dict[str, Any] | None:
    if not required:
        return None
    directory = output_dir / "standard_tabarena_baseline" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"
    summary_path = directory / "summary.json"
    predictions_path = directory / "predictions.npz"
    if not summary_path.is_file() or not predictions_path.is_file():
        raise MissingTrainingArtifacts(f"missing normal TabICLv2 baseline for task={task.task_id} ({directory})")
    summary = _mapping(summary_path, description="standard baseline summary")
    if summary.get("run_fingerprint_hash") != run_fingerprint_hash:
        raise ArtifactIntegrityError(f"{summary_path} belongs to a different immutable run fingerprint")
    try:
        with np.load(predictions_path, allow_pickle=False) as payload:
            prediction = payload["prediction"]
    except (OSError, ValueError, KeyError) as error:
        raise ArtifactIntegrityError(f"cannot read {predictions_path}: {type(error).__name__}: {error}") from error
    expected_shape = _prediction_shape(len(task.y_test), task.problem_type, task.n_classes)
    if prediction.shape != expected_shape:
        raise ArtifactIntegrityError(f"{predictions_path} shape {prediction.shape}; expected {expected_shape}")
    return summary


def main() -> None:
    args = _parse_args()
    manifest_path = args.output_dir / "experiment_manifest.json"
    manifest = _mapping(manifest_path, description="experiment manifest")
    run = manifest.get("immutable_run")
    if not isinstance(run, dict):
        raise ArtifactIntegrityError(f"{manifest_path} has no immutable_run object")
    if run.get("pipeline") != "standard":
        raise ValueError("this finalizer supports only the standard DirectSpline pipeline")
    run_fingerprint_hash = manifest.get("run_fingerprint_sha256")
    if not isinstance(run_fingerprint_hash, str) or not run_fingerprint_hash:
        raise ArtifactIntegrityError(f"{manifest_path} has no run_fingerprint_sha256")
    data_source = run.get("data_source")
    if not isinstance(data_source, dict) or not isinstance(data_source.get("task_ids"), list):
        raise ArtifactIntegrityError(f"{manifest_path} has no frozen task list")
    outer_split = data_source.get("outer_split")
    if not isinstance(outer_split, dict):
        raise ArtifactIntegrityError(f"{manifest_path} has no frozen outer split")
    labels = run.get("config_labels")
    configs = run.get("configs")
    if not isinstance(labels, list) or not isinstance(configs, list) or len(labels) != len(configs) or not labels:
        raise ArtifactIntegrityError(f"{manifest_path} has invalid frozen DirectSpline configurations")
    if not all(isinstance(label, str) and isinstance(config, dict) for label, config in zip(labels, configs, strict=True)):
        raise ArtifactIntegrityError(f"{manifest_path} has invalid config label/config records")
    requested_bags = int(run.get("inner_bags", 0))
    protocol_seed = int(run.get("protocol_seed", 0))
    if requested_bags < 2:
        raise ArtifactIntegrityError(f"{manifest_path} has invalid inner_bags")
    task_ids = [int(task_id) for task_id in data_source["task_ids"]]
    skips = _recorded_skips(args.output_dir)
    baseline_oom_skips = _standard_baseline_oom_skips(args.output_dir)
    effective_skips = dict(skips)
    completed: list[dict[str, Any]] = []
    incomplete: list[str] = []
    baseline_required = run.get("standard_tabarena_baseline") is not None
    print("CPU-only finalization: verifying saved bags; no TabICL model or spline training will run.", flush=True)
    for task_id in task_ids:
        raw_matches = list((args.output_dir / "raw").glob(f"task_{task_id}_*"))
        if task_id in skips and not raw_matches:
            continue
        # A stale OOM record can remain if an earlier resume completed the
        # task but was interrupted before its next progress snapshot. Saved
        # bags are stronger evidence, so recover that task and remove the
        # obsolete skip from the final report.
        effective_skips.pop(task_id, None)
        if not raw_matches:
            incomplete.append(f"task={task_id}: no DirectSpline bag directory")
            continue
        task = load_tabarena_openml_task(
            task_id,
            outer_repeat=int(outer_split["repeat"]),
            outer_fold=int(outer_split["fold"]),
            outer_sample=int(outer_split["sample"]),
        )
        checkpoint_metadata = _expected_checkpoint(run, task)
        try:
            for label, config in zip(labels, configs, strict=True):
                summary = rebuild_standard_config_from_bags(
                    task=task,
                    label=label,
                    config=config,
                    output_dir=args.output_dir,
                    requested_bags=requested_bags,
                    protocol_seed=protocol_seed,
                    run_fingerprint_hash=run_fingerprint_hash,
                    checkpoint_metadata=checkpoint_metadata,
                )
                print(
                    f"[task={task_id} config={label}] rebuilt final aggregation from {summary['effective_bags']} saved bags",
                    flush=True,
                )
            standard_tabarena = _standard_baseline(
                output_dir=args.output_dir,
                task=task,
                run_fingerprint_hash=run_fingerprint_hash,
                required=baseline_required and task_id not in baseline_oom_skips,
            )
        except MissingTrainingArtifacts as error:
            incomplete.append(str(error))
            continue
        task_summary = summarize_task_tuning(
            task=task,
            config_labels=labels,
            output_dir=args.output_dir,
            ensemble_rounds=int(run["ensemble_rounds"]),
            standard_tabarena=standard_tabarena,
        )
        completed.append(task_summary)

    finalization = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "cpu_only_saved_bag_finalization",
        "run_fingerprint_sha256": run_fingerprint_hash,
        "completed_task_ids": [item["task_id"] for item in completed],
        "recorded_skipped_task_ids": sorted(effective_skips),
        "incomplete": incomplete,
    }
    _json_dump(args.output_dir / "finalization_report.json", finalization)
    if incomplete:
        print(json.dumps(finalization, indent=2), flush=True)
        raise MissingTrainingArtifacts(
            "finalization stopped without training; unfinished artifacts are listed in finalization_report.json"
        )
    if not completed:
        raise MissingTrainingArtifacts("no completed tasks were available to finalize")
    summary = summarize_experiment(
        task_summaries=completed,
        output_dir=args.output_dir,
        bootstrap_rounds=int(run["bootstrap_rounds"]),
        bootstrap_seed=protocol_seed,
        skipped_tasks=list(effective_skips.values()),
        task_eligibility=run.get("task_eligibility"),
    )
    finalization["status"] = "complete"
    finalization["summary_path"] = str(args.output_dir / "summary.json")
    _json_dump(args.output_dir / "finalization_report.json", finalization)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
