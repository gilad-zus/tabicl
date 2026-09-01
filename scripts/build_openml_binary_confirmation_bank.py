"""Build a reviewed, TabArena-disjoint OpenML binary confirmation bank.

This freezes the task-level development data for the DirectSpline binary-loss
experiment.  Candidate selection uses published OpenML metadata and a
structural split audit only: no TabICL, spline, validation, or outer-test
metric is consulted.  Every selected task has two classes, enough examples of
each class for the ordinary eight-bag protocol, and at least one trainable
numerical feature.

Example
-------
Build a 20-task bank that excludes the TabArena-Lite task *and dataset* IDs::

    python scripts/build_openml_binary_confirmation_bank.py \
      --output results/openml_direct_spline_binary_objective/binary_bank_v1.json \
      --exclude-task-id-file results/openml_direct_spline_standard/full_D_500_big_gpu_v1/experiment_manifest.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:  # Support both ``python scripts/...`` and pytest imports.
    from scripts.build_openml_regression_confirmation_bank import (
        DEFAULT_CANDIDATE_MULTIPLIER,
        DEFAULT_MAX_FEATURES,
        DEFAULT_MAX_OUTER_TRAIN_ROWS,
        DEFAULT_MAX_TOTAL_ROWS,
        DEFAULT_MIN_OUTER_TEST_ROWS,
        DEFAULT_MIN_OUTER_TRAIN_ROWS,
        DEFAULT_MIN_TOTAL_ROWS,
        _atomic_json_dump,
        _count_trainable_numerical_columns,
        _dataset_ids_for_task_ids,
        _file_provenance,
        _openml_task_listing,
        _sha256_json,
        _task_ids_from_exclusion_file,
        select_distinct_openml_candidates,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation.
    from build_openml_regression_confirmation_bank import (
        DEFAULT_CANDIDATE_MULTIPLIER,
        DEFAULT_MAX_FEATURES,
        DEFAULT_MAX_OUTER_TRAIN_ROWS,
        DEFAULT_MAX_TOTAL_ROWS,
        DEFAULT_MIN_OUTER_TEST_ROWS,
        DEFAULT_MIN_OUTER_TRAIN_ROWS,
        DEFAULT_MIN_TOTAL_ROWS,
        _atomic_json_dump,
        _count_trainable_numerical_columns,
        _dataset_ids_for_task_ids,
        _file_provenance,
        _openml_task_listing,
        _sha256_json,
        _task_ids_from_exclusion_file,
        select_distinct_openml_candidates,
    )
from tabicl._experiments.direct_spline_openml import (
    TABARENA_V0PT1_OPENML_SUITE_ID,
    load_tabarena_openml_task,
    tabarena_v0pt1_task_ids,
)


BINARY_CONFIRMATION_BANK_VERSION = 1
SELECTION_NAMESPACE = "direct-spline-openml-binary-confirmation-v1"
DEFAULT_SELECTION_SEED = 20_260_901
DEFAULT_TASK_COUNT = 20
DEFAULT_MIN_OUTER_TRAIN_CLASS_ROWS = 8


def _source_sha256() -> str:
    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def audit_binary_candidate_task(
    candidate: dict[str, Any],
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_sample: int,
    min_outer_train_rows: int,
    min_outer_test_rows: int,
    max_outer_train_rows: int,
    max_features: int,
    min_outer_train_class_rows: int,
    task_loader: Callable[..., Any] = load_tabarena_openml_task,
) -> dict[str, Any]:
    """Validate binary structure without evaluating any learned prediction."""

    task = task_loader(
        int(candidate["task_id"]),
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        outer_sample=outer_sample,
    )
    if task.problem_type != "binary":
        raise ValueError(f"published task type resolved to {task.problem_type}, not binary")
    n_classes = int(task.n_classes or 0)
    if n_classes != 2:
        raise ValueError(f"binary task must expose exactly two classes, received {n_classes}")
    n_train = int(len(task.y_train))
    n_test = int(len(task.y_test))
    n_features = int(task.x_train.shape[1])
    if n_train < min_outer_train_rows:
        raise ValueError(f"outer train rows {n_train} < required {min_outer_train_rows}")
    if n_train > max_outer_train_rows:
        raise ValueError(f"outer train rows {n_train} > cap {max_outer_train_rows}")
    if n_test < min_outer_test_rows:
        raise ValueError(f"outer test rows {n_test} < required {min_outer_test_rows}")
    if n_features <= 0 or n_features > max_features:
        raise ValueError(f"input features {n_features} outside 1..{max_features}")
    labels = np.asarray(task.y_train, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.shape != (2,) or int(class_counts.min()) < min_outer_train_class_rows:
        raise ValueError(
            f"smallest outer-training class has {int(class_counts.min())} rows; "
            f"requires at least {min_outer_train_class_rows} for eight bags"
        )
    n_numeric, numeric_columns = _count_trainable_numerical_columns(task.x_train)
    if n_numeric == 0:
        raise ValueError("outer training table has no non-constant numerical input column")
    return {
        "task_id": int(task.task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": str(task.dataset_name),
        "problem_type": str(task.problem_type),
        "n_classes": 2,
        "outer_train_class_counts": [int(value) for value in class_counts],
        "min_outer_train_class_count": int(class_counts.min()),
        "outer_train_rows": n_train,
        "outer_test_rows": n_test,
        "n_features": n_features,
        "n_trainable_raw_numerical_features": n_numeric,
        "trainable_raw_numerical_columns": numeric_columns,
        "outer_split_hash": str(task.outer_split_hash),
        "selection_key": candidate["selection_key"],
        "listed_metadata": {
            key: candidate[key]
            for key in (
                "listed_dataset_name",
                "listed_status",
                "listed_total_rows",
                "listed_features",
                "listed_numerical_features",
                "listed_categorical_features",
                "listed_classes",
            )
        },
    }


def _openml_classification_listing() -> list[dict[str, Any]]:
    return _openml_task_listing(
        task_type_name="SUPERVISED_CLASSIFICATION",
        fallback_task_type=1,
        problem_description="supervised-classification",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Frozen task-bank JSON to create.")
    parser.add_argument(
        "--exclude-task-id-file",
        type=Path,
        default=None,
        help=(
            "Prior DirectSpline experiment_manifest.json or task-bank JSON whose task and dataset IDs must "
            "be excluded. Use the completed TabArena manifest to avoid a fresh suite lookup."
        ),
    )
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    parser.add_argument("--candidate-multiplier", type=int, default=DEFAULT_CANDIDATE_MULTIPLIER)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--min-total-rows", type=int, default=DEFAULT_MIN_TOTAL_ROWS)
    parser.add_argument("--max-total-rows", type=int, default=DEFAULT_MAX_TOTAL_ROWS)
    parser.add_argument("--min-outer-train-rows", type=int, default=DEFAULT_MIN_OUTER_TRAIN_ROWS)
    parser.add_argument("--min-outer-test-rows", type=int, default=DEFAULT_MIN_OUTER_TEST_ROWS)
    parser.add_argument("--max-outer-train-rows", type=int, default=DEFAULT_MAX_OUTER_TRAIN_ROWS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--min-outer-train-class-rows", type=int, default=DEFAULT_MIN_OUTER_TRAIN_CLASS_ROWS)
    parser.add_argument("--outer-repeat", type=int, default=0)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--outer-sample", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--task-count": args.task_count,
        "--candidate-multiplier": args.candidate_multiplier,
        "--min-total-rows": args.min_total_rows,
        "--max-total-rows": args.max_total_rows,
        "--min-outer-train-rows": args.min_outer_train_rows,
        "--min-outer-test-rows": args.min_outer_test_rows,
        "--max-outer-train-rows": args.max_outer_train_rows,
        "--max-features": args.max_features,
        "--min-outer-train-class-rows": args.min_outer_train_class_rows,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError(f"these values must be positive: {', '.join(bad)}")
    if args.min_total_rows > args.max_total_rows:
        raise ValueError("--min-total-rows cannot exceed --max-total-rows")
    if args.min_outer_train_rows > args.max_outer_train_rows:
        raise ValueError("--min-outer-train-rows cannot exceed --max-outer-train-rows")
    if min(args.outer_repeat, args.outer_fold, args.outer_sample) < 0:
        raise ValueError("outer repeat/fold/sample values must be non-negative")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing bank: {args.output}; pass --overwrite to replace it")

    listed_records = _openml_classification_listing()
    if args.exclude_task_id_file is None:
        suite_task_ids = set(tabarena_v0pt1_task_ids())
        exclusion_source = {
            "kind": "OpenML TabArena-v0.1 suite lookup",
            "suite_id": TABARENA_V0PT1_OPENML_SUITE_ID,
        }
    else:
        suite_task_ids = set(_task_ids_from_exclusion_file(args.exclude_task_id_file))
        exclusion_source = {"kind": "reviewed task-ID file", **_file_provenance(args.exclude_task_id_file)}
    suite_dataset_ids = _dataset_ids_for_task_ids(suite_task_ids)
    candidates, metadata_rejections = select_distinct_openml_candidates(
        listed_records,
        excluded_task_ids=suite_task_ids,
        excluded_dataset_ids=suite_dataset_ids,
        min_total_rows=args.min_total_rows,
        max_total_rows=args.max_total_rows,
        max_features=args.max_features,
        selection_seed=args.selection_seed,
        selection_namespace=SELECTION_NAMESPACE,
        min_listed_classes=2,
        max_listed_classes=2,
    )
    candidate_universe = [
        {"task_id": item["task_id"], "dataset_id": item["dataset_id"], "selection_key": item["selection_key"]}
        for item in candidates
    ]
    probe_limit = min(len(candidates), args.task_count * args.candidate_multiplier)
    accepted: list[dict[str, Any]] = []
    split_rejections: list[dict[str, Any]] = []
    for candidate in candidates[:probe_limit]:
        try:
            audited = audit_binary_candidate_task(
                candidate,
                outer_repeat=args.outer_repeat,
                outer_fold=args.outer_fold,
                outer_sample=args.outer_sample,
                min_outer_train_rows=args.min_outer_train_rows,
                min_outer_test_rows=args.min_outer_test_rows,
                max_outer_train_rows=args.max_outer_train_rows,
                max_features=args.max_features,
                min_outer_train_class_rows=args.min_outer_train_class_rows,
            )
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            split_rejections.append(
                {
                    "task_id": candidate["task_id"],
                    "dataset_id": candidate["dataset_id"],
                    "listed_dataset_name": candidate["listed_dataset_name"],
                    "selection_key": candidate["selection_key"],
                    "reason": reason,
                }
            )
            print(f"[task={candidate['task_id']} dataset={candidate['listed_dataset_name']}] rejected: {reason}", flush=True)
            continue
        accepted.append(audited)
        print(
            f"[task={audited['task_id']} dataset={audited['dataset_name']}] accepted: "
            f"min-class={audited['min_outer_train_class_count']} train={audited['outer_train_rows']} "
            f"test={audited['outer_test_rows']} features={audited['n_features']} "
            f"numerical={audited['n_trainable_raw_numerical_features']}",
            flush=True,
        )
        if len(accepted) == args.task_count:
            break

    payload = {
        "format_version": BINARY_CONFIRMATION_BANK_VERSION,
        "experiment": "DirectSpline unseen OpenML binary objective confirmation task bank",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_source_sha256": _source_sha256(),
        "shared_selector_source_sha256": _file_provenance(
            Path(__file__).with_name("build_openml_regression_confirmation_bank.py")
        )["sha256"],
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": args.selection_seed,
        "selection_rule": (
            "Published OpenML supervised-classification metadata only; retain exactly two-class candidates; "
            "exclude TabArena-v0.1 task and dataset IDs; then use deterministic hash rank, one task per "
            "dataset, and structural split audit. No outer-test metric participates in selection."
        ),
        "tabarena_exclusion": {
            "suite_id": TABARENA_V0PT1_OPENML_SUITE_ID,
            "source": exclusion_source,
            "task_ids": sorted(suite_task_ids),
            "dataset_ids": sorted(suite_dataset_ids),
        },
        "outer_split": {"repeat": args.outer_repeat, "fold": args.outer_fold, "sample": args.outer_sample},
        "eligibility": {
            "min_total_rows": args.min_total_rows,
            "max_total_rows": args.max_total_rows,
            "min_outer_train_rows": args.min_outer_train_rows,
            "min_outer_test_rows": args.min_outer_test_rows,
            "max_outer_train_rows": args.max_outer_train_rows,
            "max_features": args.max_features,
            "n_classes": 2,
            "min_outer_train_class_rows": args.min_outer_train_class_rows,
            "requires_nonconstant_raw_numerical_feature": True,
        },
        "candidate_listing": {
            "n_openml_classification_tasks": len(listed_records),
            "n_distinct_metadata_eligible_datasets": len(candidates),
            "metadata_rejections": metadata_rejections,
            "candidate_universe_sha256": _sha256_json(candidate_universe),
            "candidate_probe_limit": probe_limit,
        },
        "selected_task_ids": [item["task_id"] for item in accepted],
        "selected_tasks": accepted,
        "split_audit_rejections": split_rejections,
        "is_complete": len(accepted) == args.task_count,
        "requested_task_count": args.task_count,
    }
    _atomic_json_dump(args.output, payload, overwrite=args.overwrite)
    audit_path = args.output.with_name(args.output.stem + "_audit.json")
    _atomic_json_dump(audit_path, payload, overwrite=args.overwrite)
    print(f"Wrote binary task bank with {len(accepted)}/{args.task_count} tasks: {args.output}", flush=True)
    print(f"Wrote selection and split audit: {audit_path}", flush=True)
    if len(accepted) != args.task_count:
        raise RuntimeError(
            f"only {len(accepted)} eligible binary datasets were found after auditing {probe_limit} candidates; "
            "increase --candidate-multiplier or relax a documented eligibility cap, then create a new bank file"
        )


if __name__ == "__main__":
    main()
