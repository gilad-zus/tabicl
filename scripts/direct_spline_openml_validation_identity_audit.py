"""Audit validation decisions that selected the DirectSpline identity path.

The validation-selected/refit experiment persists the selected adapter state.
When validation selects identity, that state is necessarily the identity state,
so the prior artifact contains no outer-test prediction for the *best observed
non-identity checkpoint*.  This runner fills exactly that gap without changing
the original run:

* read completed task summaries from ``--source-dir``;
* keep only variants for which the original validation guard chose identity;
* reload the saved inner train/validation indices and replay the original
  training trajectory;
* force only the final post-validation branch to freeze the best spline
  candidate's outer-test prediction; and
* score that already-frozen prediction against its matched identity control.

The force happens after the validation trajectory and its best candidate have
already been determined.  It does not change optimisation, checkpoint choice,
or use outer-test labels.  Test labels are read only after both replayed test
prediction arrays are frozen.

The output answers the limited, important question for identity choices:
was rejecting the validation-best spline a correct rejection (the spline loses
on test), a false negative (it wins), or a test tie?  It is not a full router
evaluation: spline-accepted cases are deliberately out of scope here.

Use a new output directory.  ``--resume`` safely reuses completed per-variant
counterfactual records from the same source experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

# Keep this consistent with the original launcher unless the caller already
# supplied a more specific allocator setting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from tabicl._experiments.direct_spline_openml import (
    _resolve_device,
    _safe_name,
    load_frozen_backbone,
    load_tabarena_openml_task,
)
from tabicl._experiments import direct_spline_openml_standard as standard_runner


AUDIT_SCHEMA_VERSION = 1
_REPRODUCTION_RTOL = 1e-7
_REPRODUCTION_ATOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Completed or partially completed validation-selected/refit result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for counterfactual predictions and the selector audit.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        type=int,
        help="Audit only this OpenML task ID. Repeat for a short pilot; omit for all completed source tasks.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=None,
        help="Override the classifier checkpoint recorded in the source manifest.",
    )
    parser.add_argument(
        "--regressor-checkpoint",
        type=Path,
        default=None,
        help="Override the regressor checkpoint recorded in the source manifest.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device; normally use cuda.")
    parser.add_argument(
        "--test-relative-tie-tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance on relative test improvement used to call a counterfactual win/loss/tie.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching per-variant audit records already present in --output-dir.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_sha256(fit_indices: np.ndarray, validation_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(fit_indices, dtype=np.int64).tobytes())
    digest.update(b"|")
    digest.update(np.asarray(validation_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _relative_improvement(identity_error: float, candidate_error: float) -> float | None:
    if not np.isfinite(candidate_error):
        return None
    return float((identity_error - candidate_error) / max(abs(identity_error), 1e-12))


def _array_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    payload: dict[str, Any] = {
        "left_shape": list(left_array.shape),
        "right_shape": list(right_array.shape),
        "shape_match": left_array.shape == right_array.shape,
    }
    if left_array.shape != right_array.shape:
        return payload
    difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
    finite = difference[np.isfinite(difference)]
    payload["nonfinite_differences"] = int(difference.size - finite.size)
    payload["max_abs"] = None if not finite.size else float(np.max(finite))
    payload["mean_abs"] = None if not finite.size else float(np.mean(finite))
    return payload


def _close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return bool(np.isclose(float(left), float(right), rtol=_REPRODUCTION_RTOL, atol=_REPRODUCTION_ATOL))
    except (TypeError, ValueError):
        return False


def _source_raw_summary_path(source_dir: Path, task_id: int, dataset_name: str, label: str) -> Path:
    expected = (
        source_dir
        / "raw"
        / f"task_{task_id}_{_safe_name(dataset_name)}"
        / f"config_{label}"
        / "validation_selected_refit"
        / "summary.json"
    )
    if expected.is_file():
        return expected
    matches = sorted(
        source_dir.glob(
            f"raw/task_{task_id}_*/config_{label}/validation_selected_refit/summary.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"could not uniquely locate the raw source summary for task {task_id}, variant {label}: {matches}"
        )
    return matches[0]


def _identity_cases(
    source_dir: Path,
    requested_task_ids: set[int] | None,
) -> Iterator[dict[str, Any]]:
    task_summary_dir = source_dir / "validation_selected_refit_task_summaries"
    if not task_summary_dir.is_dir():
        raise FileNotFoundError(f"source directory has no completed task summaries: {task_summary_dir}")
    for task_summary_path in sorted(task_summary_dir.glob("task_*.json")):
        task_summary = _load_json(task_summary_path)
        task_id = int(task_summary["task_id"])
        if requested_task_ids is not None and task_id not in requested_task_ids:
            continue
        dataset_name = str(task_summary["dataset_name"])
        variants = task_summary.get("variants")
        if not isinstance(variants, dict):
            raise ValueError(f"source task summary has no variants mapping: {task_summary_path}")
        for label, variant in variants.items():
            if not isinstance(variant, dict):
                raise ValueError(f"invalid variant {label!r} in {task_summary_path}")
            selection = variant.get("validation_selection")
            if not isinstance(selection, dict):
                raise ValueError(f"variant {label!r} has no validation selection metadata")
            if bool(selection.get("selected_use_adapted")):
                continue
            raw_summary_path = _source_raw_summary_path(source_dir, task_id, dataset_name, str(label))
            raw_summary = _load_json(raw_summary_path)
            yield {
                "task_summary_path": task_summary_path,
                "task_summary": task_summary,
                "raw_summary_path": raw_summary_path,
                "raw_summary": raw_summary,
                "task_id": task_id,
                "dataset_name": dataset_name,
                "label": str(label),
                "config": variant["config"],
                "selection": selection,
            }


def _load_source_split(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Path]:
    raw_summary = case["raw_summary"]
    split_info = raw_summary.get("validation_split")
    if not isinstance(split_info, dict):
        raise ValueError(f"source raw summary lacks validation split: {case['raw_summary_path']}")
    split_name = split_info.get("artifact")
    if not isinstance(split_name, str):
        raise ValueError(f"source raw summary lacks the persisted split artifact name: {case['raw_summary_path']}")
    split_path = Path(case["raw_summary_path"]).parent / split_name
    if not split_path.is_file():
        raise FileNotFoundError(f"missing persisted inner split: {split_path}")
    with np.load(split_path, allow_pickle=False) as artifact:
        fit_indices = np.asarray(artifact["inner_train_indices"], dtype=int)
        validation_indices = np.asarray(artifact["validation_indices"], dtype=int)
    if not fit_indices.size or not validation_indices.size or np.intersect1d(fit_indices, validation_indices).size:
        raise ValueError(f"invalid persisted inner split: {split_path}")
    expected_hash = split_info.get("sha256")
    actual_hash = _split_sha256(fit_indices, validation_indices)
    if expected_hash != actual_hash:
        raise ValueError(f"persisted inner split hash mismatch for {split_path}: {actual_hash} != {expected_hash}")
    return fit_indices, validation_indices, split_path


def _source_identity_prediction(case: dict[str, Any]) -> tuple[np.ndarray, Path]:
    raw_summary_path = Path(case["raw_summary_path"])
    prediction_path = raw_summary_path.parent / "predictions.npz"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"missing source prediction artifact: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as artifact:
        return np.asarray(artifact["inner_identity_test"]), prediction_path


def _force_best_spline_selection() -> tuple[Any, Any]:
    """Replace only the terminal guard used by the existing selection routine."""

    original = standard_runner.choose_identity_guard

    def force_candidate(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(use_adapted=True)

    standard_runner.choose_identity_guard = force_candidate
    return original, force_candidate


def _replay_best_spline(
    *,
    task: Any,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict[str, Any],
    protocol_seed: int,
    device: torch.device,
    classifier_checkpoint: Path | str | None,
    regressor_checkpoint: Path | str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Replay training, then freeze the validation-best spline's test output.

    The underlying routine has the original training and checkpoint-selection
    implementation.  The temporary guard replacement occurs only after that
    trajectory has finished, when its selected candidate state is already
    fixed.  Its state is restored even if fitting fails.
    """

    backbone, checkpoint_path, checkpoint_metadata = load_frozen_backbone(
        problem_type=task.problem_type,
        device=device,
        classifier_checkpoint=classifier_checkpoint,
        regressor_checkpoint=regressor_checkpoint,
    )
    original_guard, _forced_guard = _force_best_spline_selection()
    try:
        predictions, metadata, _state = standard_runner._fit_validation_selected_checkpoint_standard(
            task=task,
            fit_indices=fit_indices,
            validation_indices=validation_indices,
            config=config,
            protocol_seed=protocol_seed,
            backbone=backbone,
            device=device,
            progress=None,
        )
    finally:
        standard_runner.choose_identity_guard = original_guard
        del backbone, checkpoint_path
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions, metadata, checkpoint_metadata


def _reproduction_diagnostics(
    source_selection: dict[str, Any], replay_selection: dict[str, Any], checkpoint_matches: bool
) -> dict[str, Any]:
    checks = {
        "checkpoint_matches_source": bool(checkpoint_matches),
        "best_spline_checkpoint_step": (
            source_selection.get("best_spline_checkpoint_step")
            == replay_selection.get("best_spline_checkpoint_step")
        ),
        "identity_validation_error": _close(
            source_selection.get("identity_validation_error"), replay_selection.get("identity_validation_error")
        ),
        "best_spline_validation_error": _close(
            source_selection.get("best_spline_validation_error"), replay_selection.get("best_spline_validation_error")
        ),
        "replay_has_candidate": replay_selection.get("best_spline_checkpoint_step") is not None,
        "replay_forced_candidate_path": bool(replay_selection.get("selected_use_adapted")),
    }
    return {
        "matches_source": bool(all(checks.values())),
        "checks": checks,
        "source": {
            "best_spline_checkpoint_step": source_selection.get("best_spline_checkpoint_step"),
            "identity_validation_error": source_selection.get("identity_validation_error"),
            "best_spline_validation_error": source_selection.get("best_spline_validation_error"),
        },
        "replay": {
            "best_spline_checkpoint_step": replay_selection.get("best_spline_checkpoint_step"),
            "identity_validation_error": replay_selection.get("identity_validation_error"),
            "best_spline_validation_error": replay_selection.get("best_spline_validation_error"),
        },
    }


def _test_outcome(relative_improvement: float | None, tolerance: float) -> str:
    if relative_improvement is None or not np.isfinite(relative_improvement):
        return "loss_or_invalid"
    if relative_improvement > tolerance:
        return "win"
    if relative_improvement < -tolerance:
        return "loss"
    return "tie"


def _record_path(output_dir: Path, case: dict[str, Any]) -> Path:
    return output_dir / "identity_selection_records" / (
        f"task_{case['task_id']}_{_safe_name(case['dataset_name'])}_{_safe_name(case['label'])}.json"
    )


def _record_is_reusable(record_path: Path, case: dict[str, Any], source_manifest_sha256: str) -> bool:
    if not record_path.is_file():
        return False
    try:
        record = _load_json(record_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        record.get("audit_schema_version") == AUDIT_SCHEMA_VERSION
        and record.get("source_manifest_sha256") == source_manifest_sha256
        and record.get("source_raw_summary_sha256") == _sha256(Path(case["raw_summary_path"]))
    )


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task_id",
        "dataset_name",
        "problem_type",
        "config_label",
        "status",
        "source_best_spline_step",
        "source_validation_relative_improvement_vs_identity",
        "replay_matches_source",
        "candidate_test_outcome",
        "identity_test_deployment_error",
        "candidate_test_deployment_error",
        "candidate_test_relative_improvement_vs_identity",
        "selector_assessment",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            test_pair = record.get("counterfactual_test_pair", {})
            identity = test_pair.get("identity", {}) if isinstance(test_pair, dict) else {}
            candidate = test_pair.get("selected", {}) if isinstance(test_pair, dict) else {}
            writer.writerow(
                {
                    "task_id": record.get("task_id"),
                    "dataset_name": record.get("dataset_name"),
                    "problem_type": record.get("problem_type"),
                    "config_label": record.get("config_label"),
                    "status": record.get("status"),
                    "source_best_spline_step": record.get("source_best_spline_step"),
                    "source_validation_relative_improvement_vs_identity": record.get(
                        "source_validation_relative_improvement_vs_identity"
                    ),
                    "replay_matches_source": record.get("reproduction", {}).get("matches_source"),
                    "candidate_test_outcome": record.get("candidate_test_outcome"),
                    "identity_test_deployment_error": identity.get("deployment_error"),
                    "candidate_test_deployment_error": candidate.get("deployment_error"),
                    "candidate_test_relative_improvement_vs_identity": test_pair.get(
                        "selected_deployment_relative_improvement_vs_identity"
                    )
                    if isinstance(test_pair, dict)
                    else None,
                    "selector_assessment": record.get("selector_assessment"),
                }
            )


def _selector_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    identity_selected = len(records)
    structural_identity = sum(record.get("status") == "not_auditable_no_spline_candidate" for record in records)
    completed = [record for record in records if record.get("status") == "completed"]
    matched = [record for record in completed if record.get("reproduction", {}).get("matches_source")]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        outcomes = [str(record.get("candidate_test_outcome")) for record in group]
        correct = outcomes.count("loss") + outcomes.count("loss_or_invalid")
        false_negative = outcomes.count("win")
        ties = outcomes.count("tie")
        decisive = correct + false_negative
        relative = [
            float(record["counterfactual_test_pair"]["selected_deployment_relative_improvement_vs_identity"])
            for record in group
            if record.get("counterfactual_test_pair", {}).get("selected_deployment_relative_improvement_vs_identity")
            is not None
        ]
        return {
            "n": len(group),
            "correct_rejections_spline_loses": correct,
            "false_negatives_spline_wins": false_negative,
            "test_ties": ties,
            "selector_accuracy_on_decisive_cases": None if not decisive else float(correct / decisive),
            "mean_candidate_test_relative_improvement": None if not relative else float(np.mean(relative)),
            "median_candidate_test_relative_improvement": None if not relative else float(np.median(relative)),
        }

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for record in matched:
        by_variant.setdefault(str(record["config_label"]), []).append(record)
    return {
        "identity_selected_variants_in_completed_source_tasks": identity_selected,
        "structural_identity_only_no_spline_candidate": structural_identity,
        "counterfactuals_completed": len(completed),
        "replays_matching_source_candidate": len(matched),
        "replay_mismatches_excluded_from_selector_conclusion": len(completed) - len(matched),
        "selector_assessment_reproduced_cases": summarize(matched),
        "selector_assessment_by_variant_reproduced_cases": {
            label: summarize(group) for label, group in sorted(by_variant.items())
        },
        "interpretation": (
            "A correct rejection means the validation-best non-identity spline loses to its matched identity "
            "prediction on the outer test set. A false negative means it wins. Ties are not counted as either. "
            "This is a correctness audit of identity selections, not a model win for the deployed identity path."
        ),
    }


def _validate_source_manifest(source_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = source_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing source experiment manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    immutable = manifest.get("immutable_run")
    if not isinstance(immutable, dict):
        raise ValueError("source manifest lacks immutable_run")
    if immutable.get("pipeline") != "standard" or immutable.get("validation_selected_refit") is not True:
        raise ValueError("source is not a standard validation-selected/refit DirectSpline experiment")
    return manifest, _sha256(manifest_path)


def main() -> None:
    args = parse_args()
    if args.test_relative_tie_tolerance < 0.0:
        raise ValueError("--test-relative-tie-tolerance must be non-negative")
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("--output-dir must differ from --source-dir; this runner never modifies the source experiment")
    manifest, source_manifest_sha256 = _validate_source_manifest(source_dir)
    immutable = manifest["immutable_run"]
    requested = None if args.task_id is None else set(args.task_id)
    cases = list(_identity_cases(source_dir, requested))
    if not cases:
        raise RuntimeError("no completed source variants selected identity for the requested task scope")

    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is non-empty; use a new path or pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path = output_dir / "audit_manifest.json"
    if args.resume and output_manifest_path.is_file():
        existing = _load_json(output_manifest_path)
        if existing.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError("--resume output belongs to a different source experiment manifest")

    classifier_checkpoint: Path | str | None = (
        args.classifier_checkpoint
        if args.classifier_checkpoint is not None
        else immutable.get("classifier_checkpoint_argument")
    )
    regressor_checkpoint: Path | str | None = (
        args.regressor_checkpoint
        if args.regressor_checkpoint is not None
        else immutable.get("regressor_checkpoint_argument")
    )
    device = _resolve_device(args.device)
    outer_split = immutable.get("data_source", {}).get("outer_split", {})
    if not isinstance(outer_split, dict):
        raise ValueError("source manifest has no usable outer split definition")
    protocol_seed = int(immutable["protocol_seed"])
    audit_manifest = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "validation_identity_counterfactual",
        "source_dir": str(source_dir),
        "source_manifest_sha256": source_manifest_sha256,
        "source_run_fingerprint_sha256": manifest.get("run_fingerprint_sha256"),
        "requested_task_ids": None if requested is None else sorted(requested),
        "device": str(device),
        "classifier_checkpoint_argument": None if classifier_checkpoint is None else str(classifier_checkpoint),
        "regressor_checkpoint_argument": None if regressor_checkpoint is None else str(regressor_checkpoint),
        "test_relative_tie_tolerance": float(args.test_relative_tie_tolerance),
        "counterfactual_contract": {
            "source_inner_split_reused": True,
            "training_and_checkpoint_selection_replayed_from_source_config": True,
            "only_terminal_identity_guard_forced_to_best_spline": True,
            "outer_test_labels_used_only_after_identity_and_candidate_predictions_frozen": True,
            "source_experiment_is_read_only": True,
        },
    }
    _write_json(output_manifest_path, audit_manifest)

    print(f"Found {len(cases)} validation-identity variants across completed source tasks.", flush=True)
    for index, case in enumerate(cases, start=1):
        record_path = _record_path(output_dir, case)
        if args.resume and _record_is_reusable(record_path, case, source_manifest_sha256):
            print(f"[{index}/{len(cases)}] reuse task {case['task_id']} {case['label']}", flush=True)
            continue

        selection = case["selection"]
        source_best_step = selection.get("best_spline_checkpoint_step")
        source_identity_error = float(selection["identity_validation_error"])
        source_best_error = selection.get("best_spline_validation_error")
        source_validation_relative = (
            None
            if source_best_error is None
            else _relative_improvement(source_identity_error, float(source_best_error))
        )
        common = {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "source_manifest_sha256": source_manifest_sha256,
            "source_raw_summary_path": str(case["raw_summary_path"]),
            "source_raw_summary_sha256": _sha256(Path(case["raw_summary_path"])),
            "source_task_summary_path": str(case["task_summary_path"]),
            "task_id": int(case["task_id"]),
            "dataset_name": str(case["dataset_name"]),
            "problem_type": case["task_summary"]["problem_type"],
            "config_label": str(case["label"]),
            "source_best_spline_step": source_best_step,
            "source_validation_relative_improvement_vs_identity": source_validation_relative,
            "source_validation_guard_selected_identity": True,
        }
        if source_best_step is None:
            common.update(
                {
                    "status": "not_auditable_no_spline_candidate",
                    "reason": "The original trajectory had no finite non-identity spline candidate to score.",
                }
            )
            _write_json(record_path, common)
            print(f"[{index}/{len(cases)}] task {case['task_id']} {case['label']}: no spline candidate", flush=True)
            continue

        print(
            f"[{index}/{len(cases)}] task {case['task_id']} {case['dataset_name']} {case['label']}: "
            f"replay validation-best step {source_best_step}",
            flush=True,
        )
        fit_indices, validation_indices, split_path = _load_source_split(case)
        task = load_tabarena_openml_task(
            int(case["task_id"]),
            outer_repeat=int(outer_split["repeat"]),
            outer_fold=int(outer_split["fold"]),
            outer_sample=int(outer_split["sample"]),
        )
        if task.outer_split_hash != case["task_summary"].get("outer_split_hash"):
            raise RuntimeError(f"OpenML outer split changed for task {task.task_id}; refusing to mix it with source results")
        source_identity_test, source_prediction_path = _source_identity_prediction(case)
        predictions, replay_selection, replay_checkpoint = _replay_best_spline(
            task=task,
            fit_indices=fit_indices,
            validation_indices=validation_indices,
            config=dict(case["config"]),
            protocol_seed=protocol_seed,
            device=device,
            classifier_checkpoint=classifier_checkpoint,
            regressor_checkpoint=regressor_checkpoint,
        )
        source_checkpoint = case["raw_summary"].get("selection_checkpoint", {})
        checkpoint_matches = (
            isinstance(source_checkpoint, dict)
            and replay_checkpoint.get("sha256") == source_checkpoint.get("sha256")
        )
        reproduction = _reproduction_diagnostics(selection, replay_selection, checkpoint_matches)

        # From here onward the two test prediction arrays are frozen.  This is
        # the first operation in this runner that reads task.y_test.
        test_pair = standard_runner._selected_pair_summary(
            task=task,
            identity_prediction=predictions["identity_test"],
            selected_prediction=predictions["selected_test"],
        )
        candidate_relative = test_pair["selected_deployment_relative_improvement_vs_identity"]
        outcome = _test_outcome(candidate_relative, float(args.test_relative_tie_tolerance))
        selector_assessment = {
            "win": "false_negative_identity_rejected_a_test_winning_spline",
            "loss": "correct_rejection_spline_loses_on_test",
            "loss_or_invalid": "correct_rejection_spline_invalid_or_loses_on_test",
            "tie": "test_tie_no_decisive_selector_evidence",
        }[outcome]
        prediction_path = record_path.with_suffix(".predictions.npz")
        np.savez_compressed(
            prediction_path,
            replay_identity_test=predictions["identity_test"],
            validation_best_spline_test=predictions["selected_test"],
        )
        common.update(
            {
                "status": "completed",
                "source_split_path": str(split_path),
                "source_split_sha256": _sha256(split_path),
                "source_prediction_path": str(source_prediction_path),
                "replay_checkpoint": replay_checkpoint,
                "reproduction": reproduction,
                "source_vs_replay_identity_test": _array_difference(
                    source_identity_test, predictions["identity_test"]
                ),
                "counterfactual_prediction_path": str(prediction_path),
                "counterfactual_prediction_sha256": _sha256(prediction_path),
                "counterfactual_test_pair": test_pair,
                "candidate_test_outcome": outcome,
                "selector_assessment": selector_assessment,
            }
        )
        _write_json(record_path, common)
        print(
            f"[{index}/{len(cases)}] frozen: test outcome={outcome}, "
            f"replay_matches_source={reproduction['matches_source']}",
            flush=True,
        )
        del task, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    records = [_load_json(path) for path in sorted((output_dir / "identity_selection_records").glob("*.json"))]
    summary = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "source_dir": str(source_dir),
        "test_relative_tie_tolerance": float(args.test_relative_tie_tolerance),
        "selector_summary": _selector_summary(records),
        "records": [str(path) for path in sorted((output_dir / "identity_selection_records").glob("*.json"))],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(records, output_dir / "identity_selection_audit.csv")
    print(
        "Complete. Read summary.json for correct-rejection / false-negative counts and "
        "identity_selection_audit.csv for every identity-selected variant.",
        flush=True,
    )


if __name__ == "__main__":
    main()
