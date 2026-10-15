"""Verify immutable cross-arm artifacts before reusing their controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_FINGERPRINT = "59eab331e9cc3afbaed45d54b928aad936a10124654e6dc1dff49209ea864503"
COMMON_TASKS = (363614,363616,363618,363619,363620,363621,363623,363624,
    363626,363627,363628,363629,363630,363632,363671,363673,363674,363676,
    363679,363681,363682,363683,363684,363685,363691,363694,363696,363699,
    363700,363704,363706,363711,363712)


def reference_folder(args, task_id, fold_number=None):
    reference_root = getattr(args, "reference_dir", None)
    if reference_root is None:
        return None
    task_folder = reference_root / "tasks" / str(task_id)
    return task_folder if fold_number is None else task_folder / f"fold_{fold_number}"


def checked_reference(args, specification):
    reference_root = getattr(args, "reference_dir", None)
    if reference_root is None:
        return None
    if getattr(args, "adapter", "cross") != "spline":
        raise ValueError("The matched reference requires the spline arm")
    if reference_root.resolve() == args.output_dir.resolve():
        raise ValueError("Reference results must remain read-only")
    manifest = json.loads((reference_root / "manifest.json").read_text())
    declared_hash = hashlib.sha256(json.dumps(manifest["specification"], sort_keys=True).encode()).hexdigest()
    if manifest["fingerprint"] != REFERENCE_FINGERPRINT or declared_hash != REFERENCE_FINGERPRINT:
        raise ValueError("Unexpected cross reference manifest")
    source_settings = manifest["specification"]
    for setting_name in ("outer_split", "seed", "folds", "epochs", "patience", "context_rows",
                         "query_rows", "prediction_context_rows", "query_chunk_rows", "checkpoint_sha256", "defaults"):
        if source_settings[setting_name] != specification[setting_name]:
            raise ValueError(f"Shared protocol setting differs: {setting_name}")
    if tuple(specification["task_ids"]) != COMMON_TASKS:
        raise ValueError("Matched experiment must retain all 33 common datasets in order")
    # The runner adds only adapter construction and artifact reuse; its cross-path
    # equivalence is tested locally against the pinned original runner revision.
    repository = Path(__file__).resolve().parents[1]
    for relative_path, expected_hash in source_settings["code_hashes"].items():
        if relative_path == "scripts/retouche_benchmark.py":
            continue
        actual_hash = hashlib.sha256((repository / relative_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Cross reference dependency changed: {relative_path}")
    completion = json.loads((reference_root / "complete.json").read_text())
    if completion["fingerprint"] != REFERENCE_FINGERPRINT or len(completion["results"]) != 38:
        raise ValueError("Cross reference is incomplete")
    artifact_hashes = {}
    for task_id in COMMON_TASKS:
        task_folder = reference_folder(args, task_id)
        task_record = json.loads((task_folder / "task.json").read_text())
        relative_files = ["task.json", "complete.json", "native_full_predictions.npz", "predictions.npz"]
        for fold_number in range(task_record["effective_folds"]):
            fold_prefix = f"fold_{fold_number}/"
            relative_files.extend(fold_prefix + name for name in
                ("data.json", "lock.json", "baseline_validation.npz", "test_predictions.npz", "training.csv"))
            lock = json.loads((task_folder / fold_prefix / "lock.json").read_text())
            if lock["fingerprint"] != REFERENCE_FINGERPRINT or lock["test_labels_used_for_training"] or lock["test_labels_used_for_selection"]:
                raise ValueError("Cross reference fold is not safely locked")
        for relative_file in relative_files:
            artifact_path = task_folder / relative_file
            artifact_hashes[artifact_path.relative_to(reference_root).as_posix()] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return dict(root=reference_root.as_posix(), fingerprint=REFERENCE_FINGERPRINT,
                artifact_hashes=artifact_hashes, reused_cross_fits=True)


def check_task_reference(args, task_record):
    task_folder = reference_folder(args, task_record["task_id"])
    if task_folder is None:
        return
    source_record = json.loads((task_folder / "task.json").read_text())
    for field_name, field_value in task_record.items():
        if field_name != "fingerprint" and source_record[field_name] != field_value:
            raise ValueError(f"Cross-arm dataset differs: {field_name}")


def check_fold_reference(args, task_id, fold_number, fitting_positions, validation_positions,
                         support_positions, preprocessor):
    fold_folder = reference_folder(args, task_id, fold_number)
    if fold_folder is None:
        return
    source = json.loads((fold_folder / "data.json").read_text())
    for field_name, positions in (("fitting_positions", fitting_positions),
                                  ("validation_positions", validation_positions),
                                  ("support_positions", support_positions)):
        if source[field_name] != positions.tolist():
            raise ValueError(f"Cross-arm row allocation differs: {field_name}")
    # JSON converts column tuples to lists; compare their canonical serialization.
    if json.dumps(source["preprocessing"], sort_keys=True) != json.dumps(preprocessor.__dict__, sort_keys=True):
        raise ValueError("Cross-arm input preprocessing differs")


def reference_predictions(args, task_id, fold_number):
    fold_folder = reference_folder(args, task_id, fold_number)
    if fold_folder is None:
        return None
    with np.load(fold_folder / "test_predictions.npz") as predictions:
        return dict(predictions)
