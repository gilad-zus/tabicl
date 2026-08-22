"""Build a reviewed, dataset-disjoint OpenML regression confirmation bank.

This is the dataset-selection phase for the next DirectSpline experiment.  It
does not construct TabICL, fit a spline, score an outer test set, or require a
GPU.  It asks OpenML for supervised-regression task metadata, excludes the
TabArena-v0.1 regression datasets already used by the first experiment, and
then checks each candidate's *published* outer split before freezing a bank.

The resulting JSON file is passed unchanged to
``direct_spline_openml_standard.py --task-id-file``.  The training launcher
copies the exact ordered task IDs into its immutable run manifest, so a later
resume never re-runs discovery.

The defaults deliberately target a confirmation cohort rather than a
throughput benchmark: 30 new datasets; 600--18,000 total rows; at most 200
input columns; and at most 12,000 outer-training rows.  The final cap excludes
the known full-context OOM region while preserving normal, uncapped context for
every accepted task.  A task that still OOMs is recorded by the training
launcher and can be retried later on larger hardware.

Examples
--------
Build and inspect the bank (CPU/network only)::

    python scripts/build_openml_regression_confirmation_bank.py \
      --output results/openml_direct_spline_regression_confirmation/regression_bank_v1.json \
      --exclude-task-id-file results/openml_direct_spline_standard/full_D_500_big_gpu_v1/experiment_manifest.json

Then run the exact approved bank with the ordinary standard runner::

    python scripts/direct_spline_openml_standard.py \
      --output-dir results/openml_direct_spline_regression_confirmation/full_D_500_v1 \
      --task-id-file results/openml_direct_spline_regression_confirmation/regression_bank_v1.json \
      --adapter-steps 500 --adapter-patience 10 --validation-interval 10 \
      --bootstrap-rounds 10000 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from tabicl._experiments.direct_spline_openml import (
    TABARENA_V0PT1_OPENML_SUITE_ID,
    load_tabarena_openml_task,
    tabarena_v0pt1_task_ids,
)


REGRESSION_CONFIRMATION_BANK_VERSION = 1
SELECTION_NAMESPACE = "direct-spline-openml-regression-confirmation-v1"
DEFAULT_SELECTION_SEED = 20_260_822
DEFAULT_TASK_COUNT = 30
DEFAULT_CANDIDATE_MULTIPLIER = 5
DEFAULT_MIN_TOTAL_ROWS = 600
DEFAULT_MAX_TOTAL_ROWS = 18_000
DEFAULT_MIN_OUTER_TRAIN_ROWS = 400
DEFAULT_MIN_OUTER_TEST_ROWS = 200
DEFAULT_MAX_OUTER_TRAIN_ROWS = 12_000
DEFAULT_MAX_FEATURES = 200


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _source_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _task_ids_from_exclusion_file(path: Path) -> list[int]:
    """Read either a prior DirectSpline manifest or a frozen task-bank file."""

    if not path.is_file():
        raise FileNotFoundError(f"--exclude-task-id-file does not exist or is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"--exclude-task-id-file must be a valid UTF-8 JSON file: {path}") from error
    raw_ids: Any = payload.get("selected_task_ids", payload.get("task_ids")) if isinstance(payload, dict) else None
    if raw_ids is None and isinstance(payload, dict):
        raw_ids = payload.get("immutable_run", {}).get("data_source", {}).get("task_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(
            f"--exclude-task-id-file must be a task bank or DirectSpline experiment manifest with task IDs: {path}"
        )
    try:
        task_ids = [int(task_id) for task_id in raw_ids]
    except (TypeError, ValueError) as error:
        raise ValueError(f"--exclude-task-id-file contains a non-integer task ID: {path}") from error
    if any(task_id <= 0 for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError(f"--exclude-task-id-file must contain unique positive task IDs: {path}")
    return task_ids


def _file_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if not math.isfinite(float(value)) or float(value) != float(converted):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return converted


def _normalise_task_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise current and older OpenML ``list_tasks`` column spellings."""

    task_id = _as_int(_first_present(record, ("tid", "task_id", "task")))
    dataset_id = _as_int(_first_present(record, ("did", "dataset_id", "data_id")))
    if task_id is None or task_id <= 0 or dataset_id is None or dataset_id <= 0:
        return None
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "listed_dataset_name": str(_first_present(record, ("name", "dataset_name")) or f"OpenML-{dataset_id}"),
        "listed_status": _first_present(record, ("status", "Status")),
        "listed_total_rows": _as_int(_first_present(record, ("NumberOfInstances", "number_of_instances"))),
        "listed_features": _as_int(_first_present(record, ("NumberOfFeatures", "number_of_features"))),
        "listed_numerical_features": _as_int(
            _first_present(record, ("NumberOfNumericFeatures", "number_of_numeric_features"))
        ),
        "listed_categorical_features": _as_int(
            _first_present(record, ("NumberOfSymbolicFeatures", "number_of_symbolic_features"))
        ),
    }


def _selection_key(candidate: dict[str, Any], *, seed: int) -> str:
    return hashlib.sha256(
        f"{SELECTION_NAMESPACE}:{seed}:{candidate['task_id']}:{candidate['dataset_id']}".encode("utf-8")
    ).hexdigest()


def select_distinct_regression_candidates(
    listed_records: Iterable[dict[str, Any]],
    *,
    excluded_task_ids: set[int],
    excluded_dataset_ids: set[int],
    min_total_rows: int,
    max_total_rows: int,
    max_features: int,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter public metadata, then rank one task per unseen dataset deterministically.

    No performance result appears in this function.  The hash rank exists only
    to avoid selecting the earliest OpenML task IDs, which would make the
    confirmation cohort depend on task publication date.
    """

    rejected: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for raw in listed_records:
        candidate = _normalise_task_record(raw)
        if candidate is None:
            rejected["missing_task_or_dataset_id"] += 1
            continue
        if candidate["task_id"] in excluded_task_ids:
            rejected["tabarena_task"] += 1
            continue
        if candidate["dataset_id"] in excluded_dataset_ids:
            rejected["tabarena_dataset"] += 1
            continue
        status = candidate["listed_status"]
        if status is not None and str(status).lower() != "active":
            rejected["inactive_task"] += 1
            continue
        rows = candidate["listed_total_rows"]
        if rows is None:
            rejected["missing_total_rows"] += 1
            continue
        if rows < min_total_rows:
            rejected["too_few_total_rows"] += 1
            continue
        if rows > max_total_rows:
            rejected["too_many_total_rows"] += 1
            continue
        features = candidate["listed_features"]
        if features is None or features <= 0:
            rejected["missing_or_empty_features"] += 1
            continue
        if features > max_features:
            rejected["too_many_features"] += 1
            continue
        # Keep unknown numeric metadata for the split audit, but reject a
        # published zero immediately: DirectSpline has nothing to optimise.
        if candidate["listed_numerical_features"] == 0:
            rejected["listed_no_numerical_features"] += 1
            continue
        candidate["selection_key"] = _selection_key(candidate, seed=selection_seed)
        eligible.append(candidate)

    # Different OpenML tasks can point at the same dataset with another split.
    # The experiment is a new-*dataset* confirmation, so retain only the
    # deterministic first task for each underlying dataset ID.
    selected: list[dict[str, Any]] = []
    seen_dataset_ids: set[int] = set()
    for candidate in sorted(eligible, key=lambda item: (item["selection_key"], item["task_id"])):
        if candidate["dataset_id"] in seen_dataset_ids:
            rejected["duplicate_dataset_task"] += 1
            continue
        seen_dataset_ids.add(candidate["dataset_id"])
        selected.append(candidate)
    return selected, dict(sorted(rejected.items()))


def _count_trainable_numerical_columns(frame: Any) -> tuple[int, list[str]]:
    """Count non-constant numeric raw columns without fitting the public estimator."""

    try:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype
    except ModuleNotFoundError as error:  # pragma: no cover - OpenML itself requires pandas.
        raise ModuleNotFoundError("OpenML dataframe task auditing requires pandas") from error
    names: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
            continue
        # A non-constant numerical column survives the first necessary
        # condition of TabICL's unique-feature filter.  The runner still
        # records any exceptional identity-only bag after normal preprocessing.
        if int(series.nunique(dropna=False)) > 1:
            names.append(str(column))
    return len(names), names


def audit_candidate_task(
    candidate: dict[str, Any],
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_sample: int,
    min_outer_train_rows: int,
    min_outer_test_rows: int,
    max_outer_train_rows: int,
    max_features: int,
    task_loader: Callable[..., Any] = load_tabarena_openml_task,
) -> dict[str, Any]:
    """Validate only structural eligibility of a candidate's public outer split.

    The loader necessarily downloads the task target to construct its split,
    but this audit never computes a test metric, reads test target values for
    selection, fits preprocessing, or invokes TabICL.
    """

    task = task_loader(
        int(candidate["task_id"]),
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        outer_sample=outer_sample,
    )
    if task.problem_type != "regression":
        raise ValueError(f"published task type resolved to {task.problem_type}, not regression")
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
    y_train = np.asarray(task.y_train, dtype=np.float64)
    if not np.isfinite(y_train).all():
        raise ValueError("outer training target has non-finite values")
    if np.unique(y_train).size < 2:
        raise ValueError("outer training target is constant")
    n_numeric, numeric_columns = _count_trainable_numerical_columns(task.x_train)
    if n_numeric == 0:
        raise ValueError("outer training table has no non-constant numerical input column")
    return {
        "task_id": int(task.task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": str(task.dataset_name),
        "problem_type": str(task.problem_type),
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
            )
        },
    }


def _openml_regression_listing() -> list[dict[str, Any]]:
    try:
        import openml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Install the lightweight OpenML client before building the regression bank: python -m pip install openml"
        ) from error
    try:
        from openml.tasks.task import TaskType

        task_type: Any = TaskType.SUPERVISED_REGRESSION
    except (ImportError, AttributeError):  # compatible with older OpenML clients.
        task_type = 2  # OpenML's documented SUPERVISED_REGRESSION value.
    table = openml.tasks.list_tasks(task_type=task_type, output_format="dataframe")
    try:
        rows = table.to_dict(orient="records")
    except AttributeError as error:
        raise RuntimeError("OpenML list_tasks did not return the requested dataframe") from error
    if not rows:
        raise RuntimeError("OpenML returned no supervised-regression tasks")
    return [dict(row) for row in rows]


def _tabarena_regression_dataset_ids(listed_records: Iterable[dict[str, Any]], suite_task_ids: set[int]) -> set[int]:
    """Recover IDs from the same regression listing without downloading 51 datasets."""

    result: set[int] = set()
    for raw in listed_records:
        candidate = _normalise_task_record(raw)
        if candidate is not None and candidate["task_id"] in suite_task_ids:
            result.add(candidate["dataset_id"])
    return result


def _atomic_json_dump(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite an existing bank: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Frozen task-bank JSON to create.")
    parser.add_argument(
        "--exclude-task-id-file",
        type=Path,
        default=None,
        help=(
            "Prior DirectSpline experiment_manifest.json or task-bank JSON whose task IDs must be excluded. "
            "Use the completed 51-task manifest to avoid a fresh OpenML suite lookup."
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
    }
    if any(value <= 0 for value in positive.values()):
        bad = [name for name, value in positive.items() if value <= 0]
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
        raise FileExistsError(f"refusing to overwrite an existing bank: {args.output}; pass --overwrite to replace it")

    listed_records = _openml_regression_listing()
    if args.exclude_task_id_file is None:
        suite_task_ids = set(tabarena_v0pt1_task_ids())
        exclusion_source = {
            "kind": "OpenML TabArena-v0.1 suite lookup",
            "suite_id": TABARENA_V0PT1_OPENML_SUITE_ID,
        }
    else:
        suite_task_ids = set(_task_ids_from_exclusion_file(args.exclude_task_id_file))
        exclusion_source = {
            "kind": "reviewed task-ID file",
            **_file_provenance(args.exclude_task_id_file),
        }
    suite_dataset_ids = _tabarena_regression_dataset_ids(listed_records, suite_task_ids)
    candidates, metadata_rejections = select_distinct_regression_candidates(
        listed_records,
        excluded_task_ids=suite_task_ids,
        excluded_dataset_ids=suite_dataset_ids,
        min_total_rows=args.min_total_rows,
        max_total_rows=args.max_total_rows,
        max_features=args.max_features,
        selection_seed=args.selection_seed,
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
            audited = audit_candidate_task(
                candidate,
                outer_repeat=args.outer_repeat,
                outer_fold=args.outer_fold,
                outer_sample=args.outer_sample,
                min_outer_train_rows=args.min_outer_train_rows,
                min_outer_test_rows=args.min_outer_test_rows,
                max_outer_train_rows=args.max_outer_train_rows,
                max_features=args.max_features,
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
            f"train={audited['outer_train_rows']} test={audited['outer_test_rows']} "
            f"features={audited['n_features']} numerical={audited['n_trainable_raw_numerical_features']}",
            flush=True,
        )
        if len(accepted) == args.task_count:
            break

    payload = {
        "format_version": REGRESSION_CONFIRMATION_BANK_VERSION,
        "experiment": "DirectSpline unseen OpenML regression confirmation task bank",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_source_sha256": _source_sha256(),
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_seed": args.selection_seed,
        "selection_rule": (
            "Published OpenML supervised-regression metadata only; exclude TabArena-v0.1 task and dataset IDs; "
            "then use a deterministic hash rank, one task per dataset, and structural split audit in that order. "
            "No outer-test metric participates in selection."
        ),
        "tabarena_exclusion": {
            "suite_id": TABARENA_V0PT1_OPENML_SUITE_ID,
            "source": exclusion_source,
            "task_ids": sorted(suite_task_ids),
            "regression_dataset_ids_from_listing": sorted(suite_dataset_ids),
        },
        "outer_split": {"repeat": args.outer_repeat, "fold": args.outer_fold, "sample": args.outer_sample},
        "eligibility": {
            "min_total_rows": args.min_total_rows,
            "max_total_rows": args.max_total_rows,
            "min_outer_train_rows": args.min_outer_train_rows,
            "min_outer_test_rows": args.min_outer_test_rows,
            "max_outer_train_rows": args.max_outer_train_rows,
            "max_features": args.max_features,
            "requires_nonconstant_raw_numerical_feature": True,
        },
        "candidate_listing": {
            "n_openml_regression_tasks": len(listed_records),
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
    print(f"Wrote regression task bank with {len(accepted)}/{args.task_count} tasks: {args.output}", flush=True)
    print(f"Wrote selection and split audit: {audit_path}", flush=True)
    if len(accepted) != args.task_count:
        raise RuntimeError(
            f"only {len(accepted)} eligible regression datasets were found after auditing {probe_limit} candidates; "
            "increase --candidate-multiplier or relax a documented eligibility cap, then create a new bank file"
        )


if __name__ == "__main__":
    main()
