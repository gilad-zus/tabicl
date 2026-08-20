"""Direct OpenML implementation of the guarded DirectSpline benchmark.

This module intentionally uses only the small :mod:`openml` client to obtain
the public TabArena-v0.1 task suite.  It does **not** import TabArena,
AutoGluon, Ray, or any other benchmark model package.  OpenML supplies the
published outer split; this module owns up to eight valid inner bags, frozen
TabICL calls, hyperparameter selection, prediction artifacts, and paired Elo
report.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.checkpoint import checkpoint

from tabicl._experiments.direct_spline_protocol import (
    FoldPreprocessor,
    ProblemType,
    benchmark_error,
    bootstrap_paired_elo,
    choose_identity_guard,
    deployment_error,
    paired_elo_delta,
    sample_episode_indices,
    sample_prediction_context,
)
from tabicl._hyperspline import DirectSplineTransform
from tabicl._model.tabicl import TabICL


TABARENA_V0PT1_OPENML_SUITE_ID = 457
CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"

# This is deliberately the ordinary public TabICLv2 inference configuration,
# not a reimplementation of the raw one-context path used by DirectSpline.
# Keeping it explicit makes the additional baseline auditable in every run.
STANDARD_TABICL_CONFIG: dict[str, Any] = {
    "n_estimators": 8,
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "support_many_classes": True,
    "batch_size": 8,
    "kv_cache": False,
    "numerical_preprocessing": "existing",
    "random_state": 0,
}

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class OpenMLTaskData:
    """One public task restricted to its published outer split."""

    task_id: int
    dataset_id: int
    dataset_name: str
    problem_type: ProblemType
    n_classes: int | None
    x_train: Any
    y_train: np.ndarray
    x_test: Any
    y_test: np.ndarray
    outer_split_hash: str


@dataclass
class BagPredictions:
    """Predictions and metadata persisted after each completed inner bag."""

    validation_indices: np.ndarray
    identity_validation: np.ndarray
    adapted_validation: np.ndarray
    guarded_validation: np.ndarray
    identity_test: np.ndarray
    adapted_test: np.ndarray
    guarded_test: np.ndarray
    metadata: dict[str, Any]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _seed(*parts: int) -> int:
    """Stable 32-bit seed without relying on Python's randomized hash."""
    encoded = ":".join(str(int(part)) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.blake2s(encoded, digest_size=4).digest(), "little")


def _split_hash(train_indices: np.ndarray, test_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    digest.update(b"|")
    digest.update(np.asarray(test_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    """Hash a local checkpoint once so resumed results name the exact weights."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _emit(progress: ProgressCallback | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {device}, but CUDA is unavailable")
        if device.index is not None:
            torch.cuda.set_device(device)
    return device


def _resolve_checkpoint(path: str | Path | None, version: str) -> Path:
    if path is not None:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"TabICL checkpoint does not exist: {resolved}")
        return resolved
    return Path(hf_hub_download(repo_id="jingang/TabICL", filename=version))


def _set_frozen_autograd_routing(backbone: TabICL) -> None:
    """Select TabICL's differentiable math without enabling stochastic layers.

    CUDA inference managers intentionally run under ``no_grad``.  Upstream
    spline optimisation therefore needs the direct mathematical route in all
    four top-level stages.  Recursively calling ``train()`` is not equivalent:
    it also enables attention/dropout behavior that regular prediction does
    not use.
    """

    backbone.eval()
    for module_name in ("col_embedder", "row_interactor", "icl_predictor"):
        module = getattr(backbone, module_name, None)
        if module is not None:
            module.training = True
    backbone.training = True


def load_frozen_backbone(
    *,
    problem_type: ProblemType,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
) -> tuple[TabICL, Path, dict[str, Any]]:
    """Load the required TabICLv2 head once and keep all backbone weights frozen."""
    path = _resolve_checkpoint(
        regressor_checkpoint if problem_type == "regression" else classifier_checkpoint,
        REGRESSOR_CHECKPOINT if problem_type == "regression" else CLASSIFIER_CHECKPOINT,
    )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"invalid TabICL checkpoint: {path}")
    backbone = TabICL(**checkpoint["config"])
    backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    if problem_type == "regression" and backbone.max_classes != 0:
        raise ValueError(f"regression checkpoint {path.name} is not a regression backbone")
    if problem_type != "regression" and backbone.max_classes <= 0:
        raise ValueError(f"classification checkpoint {path.name} is not a classifier backbone")
    backbone.to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    _set_frozen_autograd_routing(backbone)
    return backbone, path, _checkpoint_metadata(path)


def load_tabarena_openml_task(
    task_id: int,
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_sample: int,
) -> OpenMLTaskData:
    """Download one task and use OpenML's published outer split verbatim."""
    try:
        import openml
    except ModuleNotFoundError as error:  # pragma: no cover - tested by launcher diagnosis.
        raise ModuleNotFoundError(
            "This experiment needs only the lightweight OpenML client. Install it with "
            "`python -m pip install openml`, then rerun."
        ) from error
    # Keep this compatible with both current OpenML and older server installs:
    # data and split files are fetched lazily by the methods below.
    task = openml.tasks.get_task(task_id)
    x, y = task.get_X_and_y(dataset_format="dataframe")
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=outer_repeat, fold=outer_fold, sample=outer_sample
    )
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if not train_indices.size or not test_indices.size or np.intersect1d(train_indices, test_indices).size:
        raise ValueError(f"OpenML task {task_id} returned an invalid outer split")
    x = x.reset_index(drop=True)
    y = np.asarray(y)
    x_train, x_test = x.iloc[train_indices].reset_index(drop=True), x.iloc[test_indices].reset_index(drop=True)
    y_train_raw, y_test_raw = y[train_indices], y[test_indices]
    task_type = str(task.task_type).lower()
    if "regression" in task_type:
        problem_type: ProblemType = "regression"
        y_train = np.asarray(y_train_raw, dtype=np.float64)
        y_test = np.asarray(y_test_raw, dtype=np.float64)
        n_classes: int | None = None
    else:
        encoder = LabelEncoder().fit(y_train_raw)
        y_train = encoder.transform(y_train_raw).astype(np.int64)
        try:
            y_test = encoder.transform(y_test_raw).astype(np.int64)
        except ValueError as error:
            raise ValueError(f"OpenML task {task_id} has an outer-test class absent from outer training") from error
        n_classes = int(encoder.classes_.size)
        if n_classes < 2:
            raise ValueError(f"OpenML task {task_id} has fewer than two training classes")
        problem_type = "binary" if n_classes == 2 else "multiclass"
    dataset = task.get_dataset()
    return OpenMLTaskData(
        task_id=int(task_id),
        dataset_id=int(getattr(task, "dataset_id", getattr(task, "data_set_id", dataset.dataset_id))),
        dataset_name=str(dataset.name),
        problem_type=problem_type,
        n_classes=n_classes,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        outer_split_hash=_split_hash(train_indices, test_indices),
    )


def tabarena_v0pt1_task_ids(*, attempts: int = 5, initial_retry_seconds: float = 5.0) -> list[int]:
    """Return the canonical 51 public task IDs from OpenML suite 457.

    OpenML's study endpoint occasionally returns a transient gateway error even
    while individual task downloads are healthy.  Retrying here is safe: the
    resulting list is checked for the exact published cardinality before it is
    ever used to create a new manifest.  Resumed runs do not call this helper;
    they recover their already-frozen task IDs from ``experiment_manifest``.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if initial_retry_seconds < 0:
        raise ValueError("initial_retry_seconds must be non-negative")
    try:
        import openml
    except ModuleNotFoundError as error:  # pragma: no cover - launcher diagnosis.
        raise ModuleNotFoundError("Install `openml` to obtain the public task suite.") from error
    for attempt in range(1, attempts + 1):
        try:
            suite = openml.study.get_suite(TABARENA_V0PT1_OPENML_SUITE_ID)
            task_ids = [int(task_id) for task_id in suite.tasks]
            if len(task_ids) != 51:
                raise RuntimeError(
                    f"OpenML suite {TABARENA_V0PT1_OPENML_SUITE_ID} returned {len(task_ids)} tasks, expected 51. "
                    "Refuse to run against a silently changed benchmark suite."
                )
            return task_ids
        except RuntimeError:
            # A successful response with a changed suite is not transient and
            # must never be retried into a different benchmark definition.
            raise
        except Exception as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"could not retrieve OpenML suite {TABARENA_V0PT1_OPENML_SUITE_ID} after {attempts} attempts; "
                    "start a new run later, or use --resume for an output directory that already has an "
                    "experiment_manifest.json"
                ) from error
            delay_seconds = initial_retry_seconds * (2 ** (attempt - 1))
            print(
                f"OpenML suite {TABARENA_V0PT1_OPENML_SUITE_ID} request failed "
                f"({type(error).__name__}: {error}); retrying in {delay_seconds:g}s "
                f"({attempt}/{attempts})",
                flush=True,
            )
            time.sleep(delay_seconds)


def _make_adapter(
    support_features: np.ndarray,
    numerical_indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> DirectSplineTransform | None:
    if numerical_indices.size == 0:
        return None
    support = torch.as_tensor(
        support_features[:, numerical_indices], dtype=torch.float32, device=device
    ).unsqueeze(0)
    adapter = DirectSplineTransform(
        support,
        n_control_points=int(config["n_control_points"]),
        trainable_shape=True,
        trainable_location_scale=bool(config["trainable_location_scale"]),
        knot_placement="uniform",
        control_mode="monotone",
        cross_column_mixing_rank=int(config["cross_column_mixing_rank"]),
        cross_column_mixing_bound=float(config["cross_column_mixing_bound"]),
    ).to(device)
    # FoldPreprocessor puts numerical columns into a standardised coordinate
    # system.  This makes the DirectSpline identity guard literal rather than
    # merely approximately identity around the support subset.
    with torch.no_grad():
        adapter.location.zero_()
        adapter.scale.fill_(1.0)
        if not torch.allclose(adapter.transform(support), support, atol=2e-5, rtol=2e-5):
            raise RuntimeError("DirectSpline did not initialise to the identity")
    return adapter


def _transform(
    features: torch.Tensor,
    adapter: DirectSplineTransform | None,
    numerical_indices: np.ndarray,
) -> torch.Tensor:
    if adapter is None:
        return features
    indices = torch.as_tensor(numerical_indices, dtype=torch.long, device=features.device)
    output = features.clone()
    output[..., indices] = adapter.transform(features.index_select(-1, indices))
    return output


def _forward(
    backbone: TabICL,
    adapter: DirectSplineTransform | None,
    numerical_indices: np.ndarray,
    context_features: torch.Tensor,
    context_labels: torch.Tensor,
    query_features: torch.Tensor,
    *,
    checkpoint_backbone_activations: bool = False,
) -> torch.Tensor:
    features = torch.cat(
        (
            _transform(context_features, adapter, numerical_indices),
            _transform(query_features, adapter, numerical_indices),
        ),
        dim=1,
    )

    def run_backbone(input_features: torch.Tensor) -> torch.Tensor:
        backbone.clear_cache()
        return backbone(input_features, context_labels)

    if checkpoint_backbone_activations:
        # The backbone is frozen but gradients still need to pass through it
        # to the DirectSpline adapter.  Checkpointing trades a second frozen
        # forward call during backward for substantially lower activation
        # memory, which lets wide OpenML tasks fit on a 10-GiB GPU without
        # changing an episode's rows or its numerical precision.
        return checkpoint(run_backbone, features, use_reentrant=False)
    return run_backbone(features)


def _optimizer(adapter: DirectSplineTransform, config: dict[str, Any]) -> torch.optim.Optimizer:
    regular: list[torch.nn.Parameter] = []
    gated: list[torch.nn.Parameter] = []
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in ("gate", "offset", "log_scale")):
            gated.append(parameter)
        else:
            regular.append(parameter)
    groups: list[dict[str, Any]] = []
    if regular:
        groups.append(
            {
                "params": regular,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )
    if gated:
        groups.append(
            {
                "params": gated,
                "lr": float(config["learning_rate"]) * float(config["gate_learning_rate_factor"]),
                "weight_decay": 0.0,
            }
        )
    if not groups:
        raise RuntimeError("DirectSpline has no trainable parameters")
    return torch.optim.AdamW(groups)


@torch.no_grad()
def _predict(
    *,
    backbone: TabICL,
    adapter: DirectSplineTransform | None,
    numerical_indices: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    problem_type: ProblemType,
    n_classes: int | None,
    query_chunk_rows: int,
    target_mean: float,
    target_scale: float,
) -> np.ndarray:
    device = next(backbone.parameters()).device
    context_features = torch.as_tensor(support_features, dtype=torch.float32, device=device).unsqueeze(0)
    context_labels = torch.as_tensor(support_labels, dtype=torch.float32, device=device).unsqueeze(0)
    chunks: list[np.ndarray] = []
    for start in range(0, query_features.shape[0], query_chunk_rows):
        query = torch.as_tensor(
            query_features[start : start + query_chunk_rows], dtype=torch.float32, device=device
        ).unsqueeze(0)
        raw = _forward(backbone, adapter, numerical_indices, context_features, context_labels, query)
        if problem_type == "regression":
            prediction = backbone.quantile_dist(raw).quantiles.mean(dim=-1).squeeze(0)
            chunks.append((prediction.cpu().numpy() * target_scale + target_mean).astype(np.float64))
        else:
            logits = _classification_logits(raw, n_classes)
            chunks.append(logits.softmax(dim=-1).squeeze(0).cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def _classification_logits(raw: torch.Tensor, n_classes: int | None) -> torch.Tensor:
    """Restrict the foundation-model classifier head to this task's labels.

    TabICL's classifier checkpoint exposes ``max_classes`` logits (typically
    ten), while an OpenML task can be binary or have fewer classes.  The loss
    and probabilities must use the same leading task-specific logits; otherwise
    binary prediction arrays have the checkpoint width rather than ``(n, 2)``.
    """
    if n_classes is None or n_classes < 2:
        raise ValueError(f"classification requires at least two task classes, got {n_classes}")
    if raw.ndim < 1 or raw.shape[-1] < n_classes:
        raise ValueError(
            f"TabICL returned {tuple(raw.shape)} logits, insufficient for {n_classes} task classes"
        )
    return raw[..., :n_classes]


def _prediction_shape(n_rows: int, problem_type: ProblemType, n_classes: int | None) -> tuple[int, ...]:
    return (n_rows,) if problem_type == "regression" else (n_rows, int(n_classes))


def _save_bag(path: Path, result: BagPredictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        validation_indices=result.validation_indices,
        identity_validation=result.identity_validation,
        adapted_validation=result.adapted_validation,
        guarded_validation=result.guarded_validation,
        identity_test=result.identity_test,
        adapted_test=result.adapted_test,
        guarded_test=result.guarded_test,
        metadata=np.asarray(json.dumps(result.metadata, sort_keys=True)),
    )


def _load_bag(path: Path) -> BagPredictions:
    with np.load(path, allow_pickle=False) as payload:
        return BagPredictions(
            validation_indices=payload["validation_indices"],
            identity_validation=payload["identity_validation"],
            adapted_validation=payload["adapted_validation"],
            guarded_validation=payload["guarded_validation"],
            identity_test=payload["identity_test"],
            adapted_test=payload["adapted_test"],
            guarded_test=payload["guarded_test"],
            metadata=json.loads(str(payload["metadata"].item())),
        )


def _metric_bundle(
    problem_type: ProblemType, labels: np.ndarray, prediction: np.ndarray, n_classes: int | None
) -> dict[str, float]:
    deploy = deployment_error(problem_type, labels, prediction, n_classes=n_classes)
    bench = benchmark_error(problem_type, labels, prediction, n_classes=n_classes)
    return {"deployment_error": float(deploy), "benchmark_error": float(bench)}


def _candidate_metric_bundle(
    problem_type: ProblemType,
    labels: np.ndarray,
    prediction: np.ndarray,
    n_classes: int | None,
) -> dict[str, float | bool | str | None]:
    """Score an adapted outer-test prediction without aborting the whole run.

    A learned checkpoint that was finite on its validation fold can still be
    invalid on a particular test table.  Preserve that result as an explicit
    failed candidate so a single numerical failure cannot hide the completed
    task or terminate the remaining benchmark.
    """

    try:
        return {**_metric_bundle(problem_type, labels, prediction, n_classes), "prediction_valid": True}
    except (TypeError, ValueError, FloatingPointError) as error:
        return {
            "deployment_error": float("inf"),
            "benchmark_error": float("inf"),
            "prediction_valid": False,
            "invalid_prediction_error": f"{type(error).__name__}: {error}",
        }


def effective_inner_bag_count(task: OpenMLTaskData, *, requested_bags: int) -> int:
    """Return the largest valid stratified OOF bag count up to the request.

    Retouche-style adaptation needs every fitting fold to retain at least two
    rows from every class: one labelled context row and one episode query row.
    A fixed 8-fold split is therefore impossible for a task such as ``anneal``
    whose rarest outer-training class has five rows.  Reducing *only that
    task* to five folds keeps stratification, produces full OOF predictions,
    and is preferable to silently dropping its rare class or leaking labels.
    """
    if requested_bags < 2:
        raise ValueError("requested_bags must be at least two")
    n_rows = len(task.y_train)
    if task.problem_type == "regression":
        if n_rows < requested_bags:
            raise ValueError(
                f"task {task.task_id} has {n_rows} outer-training rows; cannot form {requested_bags} regression bags"
            )
        return requested_bags
    counts = np.bincount(task.y_train)
    smallest_class = int(counts.min())
    if smallest_class < 3:
        raise ValueError(
            f"task {task.task_id} has a training class with {smallest_class} rows; "
            "DirectSpline needs at least three rows per class to retain two fitting rows"
        )
    return min(requested_bags, smallest_class)


def _bag_splits(
    task: OpenMLTaskData, *, requested_bags: int, seed: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    effective_bags = effective_inner_bag_count(task, requested_bags=requested_bags)
    if task.problem_type == "regression":
        yield from KFold(n_splits=effective_bags, shuffle=True, random_state=seed).split(task.x_train)
        return
    yield from StratifiedKFold(n_splits=effective_bags, shuffle=True, random_state=seed).split(task.x_train, task.y_train)


def _fit_one_bag(
    *,
    task: OpenMLTaskData,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    bag: int,
    config: dict[str, Any],
    protocol_seed: int,
    backbone: TabICL,
    device: torch.device,
    run_fingerprint_hash: str | None = None,
    progress: ProgressCallback | None = None,
    requested_bags: int | None = None,
    effective_bags: int | None = None,
) -> BagPredictions:
    """Fit one child adapter without reading the OpenML outer-test labels."""
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit_x = task.x_train.iloc[fit_indices].reset_index(drop=True)
    validation_x = task.x_train.iloc[validation_indices].reset_index(drop=True)
    preprocessor = FoldPreprocessor.fit(fit_x)
    fit_features = preprocessor.transform(fit_x)
    validation_features = preprocessor.transform(validation_x)
    test_features = preprocessor.transform(task.x_test)
    numerical_indices = preprocessor.numerical_indices
    raw_fit_y = task.y_train[fit_indices]
    raw_validation_y = task.y_train[validation_indices]
    if task.problem_type == "regression":
        target_mean = float(np.mean(raw_fit_y))
        target_scale = float(np.std(raw_fit_y))
        if target_scale < 1e-12:
            target_scale = 1.0
        fit_labels = ((raw_fit_y - target_mean) / target_scale).astype(np.float32)
    else:
        target_mean, target_scale = 0.0, 1.0
        fit_labels = raw_fit_y.astype(np.float32)
    support_rng = np.random.default_rng(_seed(protocol_seed, task.task_id, bag, 1))
    support_indices = sample_prediction_context(
        fit_labels,
        problem_type=task.problem_type,
        max_context_rows=int(config["max_context_rows"]),
        rng=support_rng,
    )
    support_features = fit_features[support_indices]
    support_labels = fit_labels[support_indices]
    _emit(
        progress,
        event="bag_started",
        task_id=task.task_id,
        bag=bag,
        fit_rows=int(fit_indices.size),
        validation_rows=int(validation_indices.size),
        support_rows=int(support_indices.size),
        requested_bags=requested_bags,
        effective_bags=effective_bags,
    )
    torch.manual_seed(_seed(int(config["random_state"]), task.task_id, bag, 2))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(_seed(int(config["random_state"]), task.task_id, bag, 2))
    adapter = _make_adapter(support_features, numerical_indices, config, device)
    identity_state = None if adapter is None else _cpu_state_dict(adapter)
    if adapter is None:
        identity_validation = _predict(
            backbone=backbone, adapter=None, numerical_indices=numerical_indices,
            support_features=support_features, support_labels=support_labels,
            query_features=validation_features, problem_type=task.problem_type,
            n_classes=task.n_classes,
            query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
            target_mean=target_mean, target_scale=target_scale,
        )
        adapted_validation = identity_validation.copy()
        adapted_state = None
        first_objective = final_objective = float("nan")
        executed_steps = 0
    else:
        optimizer = _optimizer(adapter, config)
        episode_rng = np.random.default_rng(_seed(int(config["random_state"]), task.task_id, bag, 3))
        train_features = torch.as_tensor(fit_features, dtype=torch.float32, device=device)
        train_labels = torch.as_tensor(fit_labels, dtype=torch.float32, device=device)
        best_state = _cpu_state_dict(adapter)
        best_error = float("inf")
        stale = 0
        first_objective = final_objective = float("nan")
        executed_steps = 0
        for step in range(1, int(config["adapter_steps"]) + 1):
            context_rows, query_rows = sample_episode_indices(
                fit_labels,
                problem_type=task.problem_type,
                context_rows=int(config["train_context_rows"]),
                query_rows=int(config["query_batch_rows"]),
                rng=episode_rng,
            )
            context_features = train_features[context_rows].unsqueeze(0)
            context_labels = train_labels[context_rows].unsqueeze(0)
            query_features = train_features[query_rows].unsqueeze(0)
            query_labels = train_labels[query_rows].unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            raw = _forward(
                backbone,
                adapter,
                numerical_indices,
                context_features,
                context_labels,
                query_features,
                checkpoint_backbone_activations=True,
            )
            if task.problem_type == "regression":
                objective = F.mse_loss(backbone.quantile_dist(raw).quantiles.mean(dim=-1).flatten(), query_labels.flatten())
            else:
                objective = F.cross_entropy(
                    _classification_logits(raw, task.n_classes).flatten(0, 1), query_labels.long().flatten()
                )
            if step == 1:
                first_objective = float(objective.detach())
            objective.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(config["grad_clip"]))
            optimizer.step()
            final_objective = float(objective.detach())
            executed_steps = step
            if step % int(config["validation_interval"]) == 0 or step == int(config["adapter_steps"]):
                candidate = _predict(
                    backbone=backbone, adapter=adapter, numerical_indices=numerical_indices,
                    support_features=support_features, support_labels=support_labels,
                    query_features=validation_features, problem_type=task.problem_type,
                    n_classes=task.n_classes,
                    query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
                    target_mean=target_mean, target_scale=target_scale,
                )
                candidate_error = deployment_error(
                    task.problem_type, raw_validation_y, candidate, n_classes=task.n_classes
                )
                if candidate_error < best_error:
                    best_error = candidate_error
                    best_state = _cpu_state_dict(adapter)
                    stale = 0
                else:
                    stale += 1
                    if stale >= int(config["adapter_patience"]):
                        break
                _emit(
                    progress,
                    event="adapter_validation",
                    task_id=task.task_id,
                    bag=bag,
                    step=step,
                    validation_error=float(candidate_error),
                    best_validation_error=float(best_error),
                    stale_validations=stale,
                    elapsed_seconds=float(time.perf_counter() - started),
                )
        adapter.load_state_dict(best_state, strict=True)
        adapted_state = _cpu_state_dict(adapter)
        identity_adapter = _make_adapter(support_features, numerical_indices, config, device)
        assert identity_adapter is not None and identity_state is not None
        identity_adapter.load_state_dict(identity_state, strict=True)
        identity_validation = _predict(
            backbone=backbone, adapter=identity_adapter, numerical_indices=numerical_indices,
            support_features=support_features, support_labels=support_labels,
            query_features=validation_features, problem_type=task.problem_type,
            n_classes=task.n_classes,
            query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
            target_mean=target_mean, target_scale=target_scale,
        )
        adapted_validation = _predict(
            backbone=backbone, adapter=adapter, numerical_indices=numerical_indices,
            support_features=support_features, support_labels=support_labels,
            query_features=validation_features, problem_type=task.problem_type,
            n_classes=task.n_classes,
            query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
            target_mean=target_mean, target_scale=target_scale,
        )
        del optimizer, train_features, train_labels, identity_adapter
    identity_error = deployment_error(task.problem_type, raw_validation_y, identity_validation, n_classes=task.n_classes)
    adapted_error = deployment_error(task.problem_type, raw_validation_y, adapted_validation, n_classes=task.n_classes)
    decision = choose_identity_guard(
        identity_error=identity_error,
        adapted_error=adapted_error,
        required_relative_improvement=float(config["guard_relative_improvement"]),
    )
    guarded_validation = adapted_validation if decision.use_adapted else identity_validation
    if adapter is not None:
        adapter.load_state_dict(adapted_state if decision.use_adapted else identity_state, strict=True)
    if adapter is None:
        identity_test = _predict(
            backbone=backbone, adapter=None, numerical_indices=numerical_indices,
            support_features=support_features, support_labels=support_labels, query_features=test_features,
            problem_type=task.problem_type, query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
            n_classes=task.n_classes,
            target_mean=target_mean, target_scale=target_scale,
        )
    else:
        identity_adapter = _make_adapter(support_features, numerical_indices, config, device)
        assert identity_adapter is not None and identity_state is not None
        identity_adapter.load_state_dict(identity_state, strict=True)
        identity_test = _predict(
            backbone=backbone, adapter=identity_adapter, numerical_indices=numerical_indices,
            support_features=support_features, support_labels=support_labels, query_features=test_features,
            problem_type=task.problem_type, query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
            n_classes=task.n_classes,
            target_mean=target_mean, target_scale=target_scale,
        )
        adapter.load_state_dict(adapted_state, strict=True)
    adapted_test = _predict(
        backbone=backbone, adapter=adapter, numerical_indices=numerical_indices,
        support_features=support_features, support_labels=support_labels, query_features=test_features,
        problem_type=task.problem_type, query_chunk_rows=int(config["evaluation_query_chunk_rows"]),
        n_classes=task.n_classes,
        target_mean=target_mean, target_scale=target_scale,
    )
    guarded_test = adapted_test if decision.use_adapted else identity_test
    peak_gib = 0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
    metadata = {
        "bag": int(bag),
        "fit_rows": int(fit_indices.size),
        "validation_rows": int(validation_indices.size),
        "requested_bags": requested_bags,
        "effective_bags": effective_bags,
        "support_rows": int(support_indices.size),
        "n_features": int(fit_features.shape[1]),
        "n_numerical_features": int(numerical_indices.size),
        "identity_error": float(identity_error),
        "adapted_error": float(adapted_error),
        "relative_improvement": float(decision.relative_improvement),
        "guard_selected_adapted": bool(decision.use_adapted),
        "adapter_first_objective": first_objective,
        "adapter_final_objective": final_objective,
        "adapter_steps_executed": int(executed_steps),
        "backbone_activation_checkpointing": adapter is not None,
        "train_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(peak_gib),
        "run_fingerprint_hash": run_fingerprint_hash,
    }
    _emit(progress, event="bag_completed", task_id=task.task_id, **metadata)
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return BagPredictions(
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        identity_validation=identity_validation,
        adapted_validation=adapted_validation,
        guarded_validation=guarded_validation,
        identity_test=identity_test,
        adapted_test=adapted_test,
        guarded_test=guarded_test,
        metadata=metadata,
    )


def _config_dir(output_dir: Path, task: OpenMLTaskData, label: str) -> Path:
    return output_dir / "raw" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}" / f"config_{label}"


def _standard_baseline_dir(output_dir: Path, task: OpenMLTaskData) -> Path:
    return output_dir / "standard_tabarena_baseline" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"


def run_standard_tabarena_baseline(
    *,
    task: OpenMLTaskData,
    output_dir: Path,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
    resume: bool,
    run_fingerprint_hash: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run normal public TabICLv2 inference on the full outer-training split.

    This does not take part in DirectSpline guarding, HPO, or ensembling.  It
    is intentionally stored separately because it fits one ordinary public
    eight-estimator TabICLv2 model on every outer-training row.  DirectSpline
    is evaluated through inner-bag fits and then averages their predictions,
    so this baseline is an end-to-end reference rather than the matched
    identity control that isolates the spline change.
    """
    directory = _standard_baseline_dir(output_dir, task)
    summary_path = directory / "summary.json"
    predictions_path = directory / "predictions.npz"
    if resume and summary_path.is_file() and predictions_path.is_file():
        summary = _json_load(summary_path)
        if summary.get("run_fingerprint_hash") != run_fingerprint_hash:
            raise RuntimeError(
                f"refusing to resume {directory}: standard-baseline artifacts have a different run fingerprint"
            )
        _emit(progress, event="standard_baseline_reused", task_id=task.task_id)
        return summary

    from tabicl import TabICLClassifier, TabICLRegressor

    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    shared = {
        "n_estimators": int(STANDARD_TABICL_CONFIG["n_estimators"]),
        "norm_methods": list(STANDARD_TABICL_CONFIG["norm_methods"]),
        "feat_shuffle_method": str(STANDARD_TABICL_CONFIG["feat_shuffle_method"]),
        "outlier_threshold": float(STANDARD_TABICL_CONFIG["outlier_threshold"]),
        "batch_size": int(STANDARD_TABICL_CONFIG["batch_size"]),
        "kv_cache": bool(STANDARD_TABICL_CONFIG["kv_cache"]),
        "numerical_preprocessing": str(STANDARD_TABICL_CONFIG["numerical_preprocessing"]),
        "random_state": int(STANDARD_TABICL_CONFIG["random_state"]),
        "device": str(device),
        "verbose": False,
    }
    _emit(progress, event="standard_baseline_started", task_id=task.task_id, **shared)
    if task.problem_type == "regression":
        estimator = TabICLRegressor(
            **shared,
            model_path=str(regressor_checkpoint) if regressor_checkpoint is not None else None,
            checkpoint_version=REGRESSOR_CHECKPOINT,
        )
    else:
        estimator = TabICLClassifier(
            **shared,
            class_shuffle_method=str(STANDARD_TABICL_CONFIG["class_shuffle_method"]),
            softmax_temperature=float(STANDARD_TABICL_CONFIG["softmax_temperature"]),
            average_logits=bool(STANDARD_TABICL_CONFIG["average_logits"]),
            support_many_classes=bool(STANDARD_TABICL_CONFIG["support_many_classes"]),
            model_path=str(classifier_checkpoint) if classifier_checkpoint is not None else None,
            checkpoint_version=CLASSIFIER_CHECKPOINT,
        )
    estimator.fit(task.x_train, task.y_train)
    # Public predict deliberately fills all-NaN query columns in-place before
    # applying its feature mask.  Keep the task object immutable because the
    # DirectSpline path consumes the same outer-test data afterwards.
    prediction_input = task.x_test.copy()
    prediction = (
        estimator.predict(prediction_input)
        if task.problem_type == "regression"
        else estimator.predict_proba(prediction_input)
    )
    checkpoint_path = Path(estimator.model_path_)
    metadata = {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "standard_tabarena_config": STANDARD_TABICL_CONFIG,
        "checkpoint": _checkpoint_metadata(checkpoint_path),
        "run_fingerprint_hash": run_fingerprint_hash,
        "test_metrics_deferred_to_task_summary": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_allocated_gib": float(
            0.0 if device.type != "cuda" else torch.cuda.max_memory_allocated(device) / 2**30
        ),
    }
    np.savez_compressed(predictions_path, prediction=prediction)
    _json_dump(summary_path, metadata)
    _emit(progress, event="standard_baseline_completed", **metadata)
    del estimator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def run_task_config(
    *,
    task: OpenMLTaskData,
    label: str,
    config: dict[str, Any],
    output_dir: Path,
    bags: int,
    protocol_seed: int,
    device: torch.device,
    classifier_checkpoint: str | Path | None,
    regressor_checkpoint: str | Path | None,
    resume: bool,
    run_fingerprint_hash: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run/recover all eight child fits for one task/configuration pair."""
    config_dir = _config_dir(output_dir, task, label)
    summary_path = config_dir / "config_summary.json"
    predictions_path = config_dir / "config_predictions.npz"
    effective_bags = effective_inner_bag_count(task, requested_bags=bags)
    if resume and summary_path.is_file() and predictions_path.is_file():
        summary = _json_load(summary_path)
        if summary.get("run_fingerprint_hash") != run_fingerprint_hash:
            raise RuntimeError(
                f"refusing to resume {config_dir}: artifacts were made by a different immutable run fingerprint"
            )
        _emit(progress, event="config_reused", task_id=task.task_id, config_label=label)
        return summary
    backbone, checkpoint, checkpoint_metadata = load_frozen_backbone(
        problem_type=task.problem_type,
        device=device,
        classifier_checkpoint=classifier_checkpoint,
        regressor_checkpoint=regressor_checkpoint,
    )
    _emit(
        progress,
        event="config_started",
        task_id=task.task_id,
        config_label=label,
        requested_bags=bags,
        effective_bags=effective_bags,
    )
    bag_results: list[BagPredictions] = []
    for bag, (fit_indices, validation_indices) in enumerate(
        _bag_splits(task, requested_bags=bags, seed=_seed(protocol_seed, task.task_id, 0))
    ):
        bag_path = config_dir / f"bag_{bag}.npz"
        if resume and bag_path.is_file():
            bag_result = _load_bag(bag_path)
            if bag_result.metadata.get("run_fingerprint_hash") != run_fingerprint_hash:
                raise RuntimeError(
                    f"refusing to resume {bag_path}: it was made by a different immutable run fingerprint"
                )
            _emit(progress, event="bag_reused", task_id=task.task_id, config_label=label, bag=bag)
        else:
            bag_result = _fit_one_bag(
                task=task,
                fit_indices=fit_indices,
                validation_indices=validation_indices,
                bag=bag,
                config=config,
                protocol_seed=protocol_seed,
                backbone=backbone,
                device=device,
                run_fingerprint_hash=run_fingerprint_hash,
                progress=progress,
                requested_bags=bags,
                effective_bags=effective_bags,
            )
            _save_bag(bag_path, bag_result)
        bag_results.append(bag_result)
    shape = _prediction_shape(len(task.y_train), task.problem_type, task.n_classes)
    identity_validation = np.full(shape, np.nan, dtype=np.float64)
    adapted_validation = np.full(shape, np.nan, dtype=np.float64)
    guarded_validation = np.full(shape, np.nan, dtype=np.float64)
    for bag_result in bag_results:
        identity_validation[bag_result.validation_indices] = bag_result.identity_validation
        adapted_validation[bag_result.validation_indices] = bag_result.adapted_validation
        guarded_validation[bag_result.validation_indices] = bag_result.guarded_validation
    if not np.isfinite(guarded_validation).all():
        raise RuntimeError(f"task {task.task_id} config {label} did not produce complete OOF predictions")
    identity_test = np.mean([result.identity_test for result in bag_results], axis=0)
    adapted_test = np.mean([result.adapted_test for result in bag_results], axis=0)
    guarded_test = np.mean([result.guarded_test for result in bag_results], axis=0)
    np.savez_compressed(
        predictions_path,
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
        "requested_bags": bags,
        "effective_bags": effective_bags,
        "inner_bag_note": (
            "Uses the requested number of bags except classification tasks whose rarest outer-training "
            "class is smaller; those use the largest valid stratified count while retaining two fitting rows/class."
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
        "mean_train_seconds_per_bag": float(np.mean([float(result.metadata["train_seconds"]) for result in bag_results])),
        "max_peak_allocated_gib": float(np.max([float(result.metadata["peak_allocated_gib"]) for result in bag_results])),
        "bag_metadata": [result.metadata for result in bag_results],
    }
    _json_dump(summary_path, summary)
    _emit(
        progress,
        event="config_completed",
        task_id=task.task_id,
        config_label=label,
        guarded_validation_error=summary["validation"]["guarded"]["deployment_error"],
    )
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _load_config_prediction(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        return payload[key]


def greedy_validation_ensemble(
    *,
    predictions: list[np.ndarray],
    labels: np.ndarray,
    problem_type: ProblemType,
    n_classes: int | None,
    rounds: int,
) -> tuple[list[int], np.ndarray]:
    """Caruana-style repeated greedy selection using only validation OOF rows."""
    if not predictions or rounds <= 0:
        raise ValueError("ensemble needs predictions and a positive number of rounds")
    current = np.zeros_like(predictions[0], dtype=np.float64)
    selected: list[int] = []
    for round_index in range(rounds):
        errors = []
        for candidate in predictions:
            proposed = (current * round_index + candidate) / (round_index + 1)
            errors.append(deployment_error(problem_type, labels, proposed, n_classes=n_classes))
        best = int(np.argmin(errors))
        current = (current * round_index + predictions[best]) / (round_index + 1)
        selected.append(best)
    return selected, current


def summarize_task_tuning(
    *,
    task: OpenMLTaskData,
    config_labels: list[str],
    output_dir: Path,
    ensemble_rounds: int,
    standard_tabarena: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use OOF validation only to create raw and guarded outer-test outputs.

    The historical ``default``, ``tuned``, and ``tuned_ensemble`` keys are
    retained as aliases of the guarded deployment arms.  Raw adapted outputs
    are reported separately so the identity guard cannot hide either spline
    gains or spline regressions when measuring adaptation headroom.
    """
    if not config_labels:
        raise ValueError("task tuning needs at least one DirectSpline configuration")
    summaries = [_json_load(_config_dir(output_dir, task, label) / "config_summary.json") for label in config_labels]
    prediction_paths = [_config_dir(output_dir, task, label) / "config_predictions.npz" for label in config_labels]
    guarded_validation_predictions = [
        _load_config_prediction(path, "guarded_validation") for path in prediction_paths
    ]
    guarded_test_predictions = [_load_config_prediction(path, "guarded_test") for path in prediction_paths]
    raw_adapted_test_predictions = [
        _load_config_prediction(path, "adapted_test") for path in prediction_paths
    ]
    guarded_selected_index = int(
        np.argmin([float(summary["validation"]["guarded"]["deployment_error"]) for summary in summaries])
    )
    raw_selected_index = int(
        np.argmin([float(summary["validation"]["adapted"]["deployment_error"]) for summary in summaries])
    )
    only_default = len(config_labels) == 1
    if only_default:
        # Repeating the sole D prediction for every greedy round produces the
        # exact same arm.  Record an explicit alias instead of presenting it
        # as extra tuning or ensemble evidence.
        ensemble_indices = [0]
        ensemble_validation = guarded_validation_predictions[0]
        ensemble_test = guarded_test_predictions[0]
    else:
        ensemble_indices, ensemble_validation = greedy_validation_ensemble(
            predictions=guarded_validation_predictions,
            labels=task.y_train,
            problem_type=task.problem_type,
            n_classes=task.n_classes,
            rounds=ensemble_rounds,
        )
        ensemble_test = np.mean([guarded_test_predictions[index] for index in ensemble_indices], axis=0)
    default = summaries[0]
    identity_test = _load_config_prediction(prediction_paths[0], "identity_test")
    # Config choice is now frozen. Only from this point onward may the outer
    # test labels be used to score the saved predictions.
    raw_adapted_default = _candidate_metric_bundle(
        task.problem_type, task.y_test, raw_adapted_test_predictions[0], task.n_classes
    )
    raw_adapted_tuned = _candidate_metric_bundle(
        task.problem_type,
        task.y_test,
        raw_adapted_test_predictions[raw_selected_index],
        task.n_classes,
    )
    guarded_default = _candidate_metric_bundle(
        task.problem_type, task.y_test, guarded_test_predictions[0], task.n_classes
    )
    guarded_tuned = _candidate_metric_bundle(
        task.problem_type,
        task.y_test,
        guarded_test_predictions[guarded_selected_index],
        task.n_classes,
    )
    guarded_tuned_ensemble = _candidate_metric_bundle(
        task.problem_type, task.y_test, ensemble_test, task.n_classes
    )
    if standard_tabarena is None:
        standard_tabarena_test = None
    else:
        standard_prediction = _load_config_prediction(
            _standard_baseline_dir(output_dir, task) / "predictions.npz", "prediction"
        )
        standard_tabarena_test = _metric_bundle(
            task.problem_type, task.y_test, standard_prediction, task.n_classes
        )
    reported_arm_aliases = {
        # Backward-compatible artifact names.
        "default": "guarded_default",
        "tuned": "guarded_default" if only_default else "guarded_tuned",
        "tuned_ensemble": "guarded_default" if only_default else "guarded_tuned_ensemble",
    }
    if only_default:
        reported_arm_aliases.update(
            {
                "raw_adapted_tuned": "raw_adapted_default",
                "guarded_tuned": "guarded_default",
                "guarded_tuned_ensemble": "guarded_default",
            }
        )
    distinct_reported_arms = ["raw_adapted_default", "guarded_default"]
    if not only_default:
        distinct_reported_arms.extend(
            ["raw_adapted_tuned", "guarded_tuned", "guarded_tuned_ensemble"]
        )
    result = {
        "report_schema_version": 2,
        "outer_test_scored_after_validation_selection": True,
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "dataset_name": task.dataset_name,
        "problem_type": task.problem_type,
        "n_classes": task.n_classes,
        "outer_split_hash": task.outer_split_hash,
        "pipeline": default.get("pipeline", "lite_raw"),
        "identity_definition": default.get(
            "identity_definition",
            "Raw frozen TabICL identity path used by the legacy lite DirectSpline runner.",
        ),
        "identity": _metric_bundle(task.problem_type, task.y_test, identity_test, task.n_classes),
        "raw_adapted_default": raw_adapted_default,
        "raw_adapted_tuned": raw_adapted_tuned,
        "raw_adapted_tuned_config_label": config_labels[raw_selected_index],
        "raw_adapted_tuned_validation_deployment_error": summaries[raw_selected_index]["validation"][
            "adapted"
        ]["deployment_error"],
        "guarded_default": guarded_default,
        "guarded_tuned": guarded_tuned,
        "guarded_tuned_config_label": config_labels[guarded_selected_index],
        "guarded_tuned_validation_deployment_error": summaries[guarded_selected_index]["validation"][
            "guarded"
        ]["deployment_error"],
        "guarded_tuned_ensemble": guarded_tuned_ensemble,
        "guarded_tuned_ensemble_validation": _metric_bundle(
            task.problem_type, task.y_train, ensemble_validation, task.n_classes
        ),
        "guarded_tuned_ensemble_selected_config_labels": [
            config_labels[index] for index in ensemble_indices
        ],
        "reported_arm_aliases": reported_arm_aliases,
        "distinct_reported_arms": distinct_reported_arms,
        # Historical names remain guarded-arm aliases for artifact consumers.
        "default": guarded_default,
        "tuned": guarded_tuned,
        "tuned_config_label": config_labels[guarded_selected_index],
        "tuned_validation_deployment_error": summaries[guarded_selected_index]["validation"]["guarded"][
            "deployment_error"
        ],
        "tuned_ensemble": guarded_tuned_ensemble,
        "tuned_ensemble_validation": _metric_bundle(
            task.problem_type, task.y_train, ensemble_validation, task.n_classes
        ),
        "tuned_ensemble_selected_config_labels": [config_labels[index] for index in ensemble_indices],
        "default_guard_selected_adapted_fraction": default["guard_selected_adapted_fraction"],
        "default_adapter_valid_learned_checkpoint_fraction": default.get(
            "adapter_valid_learned_checkpoint_fraction"
        ),
        "guard_protocol": default.get(
            "guard_protocol", "retouche_per_bag_validation_guard_then_test_ensemble"
        ),
        "direct_spline_requested_bags": default["requested_bags"],
        "direct_spline_effective_bags": default["effective_bags"],
        "standard_tabarena": standard_tabarena_test,
        "standard_tabarena_note": (
            "Normal TabICLv2 fitted once on every outer-training row. DirectSpline uses inner-bag "
            "fits and averages their predictions, so this is an end-to-end pipeline reference, "
            "not the matched identity control that isolates the spline change."
        ),
    }
    _json_dump(output_dir / "task_summaries" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}.json", result)
    return result


_EXPLICIT_REPORT_ARM_ORDER = (
    "raw_adapted_default",
    "raw_adapted_tuned",
    "guarded_default",
    "guarded_tuned",
    "guarded_tuned_ensemble",
)
_LEGACY_GUARDED_ARM_ALIASES = {
    "default": "guarded_default",
    "tuned": "guarded_tuned",
    "tuned_ensemble": "guarded_tuned_ensemble",
}


def _relative_error_change(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return (candidate - reference) / np.maximum(np.abs(reference), 1e-12)


def _paired_comparison_summary(
    *,
    reference: np.ndarray,
    candidate: np.ndarray,
    problem_types: np.ndarray,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Summarize paired outcomes without averaging incomparable raw errors."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    problem_types = np.asarray(problem_types, dtype=object)
    valid_reference = np.isfinite(reference)
    if not np.any(valid_reference):
        raise ValueError("at least one finite reference result is required")
    invalid_candidate = valid_reference & ~np.isfinite(candidate)
    n_invalid_reference = int(np.sum(~valid_reference))
    n_invalid_candidate = int(np.sum(invalid_candidate))
    reference = reference[valid_reference]
    candidate = candidate[valid_reference]
    problem_types = problem_types[valid_reference]
    if n_invalid_candidate:
        # A malformed candidate prediction is an observed loss, not a missing
        # comparison. Use a finite, task-relative penalty solely for paired
        # win/loss and bootstrap arithmetic; the per-task metric remains inf
        # and carries the explicit invalid-prediction diagnostic.
        candidate[~np.isfinite(candidate)] = (
            reference[~np.isfinite(candidate)]
            + np.maximum(np.abs(reference[~np.isfinite(candidate)]), 1.0) * 1_000_000.0
        )
    relative = _relative_error_change(reference, candidate)
    by_problem_type: dict[str, dict[str, Any]] = {}
    for problem_type in sorted(set(str(value) for value in problem_types)):
        mask = problem_types == problem_type
        grouped_reference = reference[mask]
        grouped_candidate = candidate[mask]
        grouped = paired_elo_delta(grouped_reference, grouped_candidate)
        grouped["median_relative_error_change"] = float(
            np.median(_relative_error_change(grouped_reference, grouped_candidate))
        )
        by_problem_type[problem_type] = grouped
    return {
        **paired_elo_delta(reference, candidate),
        "bootstrap": bootstrap_paired_elo(
            reference,
            candidate,
            rounds=bootstrap_rounds,
            seed=bootstrap_seed,
        ),
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "n_invalid_candidate_predictions": n_invalid_candidate,
        "n_invalid_reference_predictions": n_invalid_reference,
        "median_relative_error_change": float(np.median(relative)),
        "by_problem_type": by_problem_type,
    }


def _copy_paired_alias(result: dict[str, Any], *, alias_of: str) -> dict[str, Any]:
    return {**result, "alias_of": alias_of}


def summarize_experiment(
    *,
    task_summaries: list[dict[str, Any]],
    output_dir: Path,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    skipped_tasks: list[dict[str, Any]] | None = None,
    task_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write task metrics and paired-Elo deltas without a TabArena install.

    Absolute Elo ratings from the Retouche paper use a much larger published
    comparison pool.  This local report therefore makes the valid, directly
    interpretable quantity explicit: the paired DirectSpline-vs-identity Elo
    difference on exactly the tasks we ran.
    """
    if not task_summaries:
        raise ValueError("cannot summarise an empty experiment")
    skipped_tasks = [] if skipped_tasks is None else skipped_tasks
    has_explicit_arms = all(
        all(arm in item for arm in _EXPLICIT_REPORT_ARM_ORDER) for item in task_summaries
    )
    if has_explicit_arms:
        candidate_arms = list(_EXPLICIT_REPORT_ARM_ORDER)
        global_aliases: dict[str, str] = {}
        for arm in candidate_arms:
            targets = {
                str(item.get("reported_arm_aliases", {}).get(arm, arm)) for item in task_summaries
            }
            if len(targets) == 1:
                target = next(iter(targets))
                if target != arm:
                    global_aliases[arm] = target
        distinct_arms = []
        for arm in candidate_arms:
            target = global_aliases.get(arm, arm)
            if target not in distinct_arms:
                distinct_arms.append(target)
    else:
        # Old task-summary artifacts remain readable. They did not retain raw
        # adapted metrics, so only their historical guarded arms can be shown.
        candidate_arms = ["default", "tuned", "tuned_ensemble"]
        global_aliases = {}
        distinct_arms = list(candidate_arms)
    identity = np.asarray(
        [float(item["identity"]["benchmark_error"]) for item in task_summaries], dtype=float
    )
    problem_types = np.asarray([str(item["problem_type"]) for item in task_summaries], dtype=object)
    paired: dict[str, dict[str, Any]] = {}
    for offset, method in enumerate(distinct_arms):
        candidate = np.asarray(
            [float(item[method]["benchmark_error"]) for item in task_summaries], dtype=float
        )
        paired[method] = _paired_comparison_summary(
            reference=identity,
            candidate=candidate,
            problem_types=problem_types,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed + offset,
            reference_label="matched_inner_bag_identity",
            candidate_label=method,
        )
    for method, target in global_aliases.items():
        paired[method] = _copy_paired_alias(paired[target], alias_of=target)
    paired_result_aliases = dict(global_aliases)
    if has_explicit_arms:
        for legacy, explicit in _LEGACY_GUARDED_ARM_ALIASES.items():
            target = global_aliases.get(explicit, explicit)
            paired_result_aliases[legacy] = target
            paired[legacy] = _copy_paired_alias(paired[target], alias_of=target)
    rows: list[dict[str, Any]] = []
    for item in task_summaries:
        row: dict[str, Any] = {
            "task_id": item["task_id"],
            "dataset_id": item["dataset_id"],
            "dataset_name": item["dataset_name"],
            "problem_type": item["problem_type"],
            "outer_split_hash": item["outer_split_hash"],
            "pipeline": item.get("pipeline", "lite_raw"),
            "identity_benchmark_error": item["identity"]["benchmark_error"],
            "identity_deployment_error": item["identity"]["deployment_error"],
            "tuned_config_label": item["tuned_config_label"],
            "tuned_validation_deployment_error": item["tuned_validation_deployment_error"],
            "default_guard_selected_adapted_fraction": item["default_guard_selected_adapted_fraction"],
            "default_adapter_valid_learned_checkpoint_fraction": item.get(
                "default_adapter_valid_learned_checkpoint_fraction"
            ),
            "guard_protocol": item.get("guard_protocol"),
            "direct_spline_requested_bags": item["direct_spline_requested_bags"],
            "direct_spline_effective_bags": item["direct_spline_effective_bags"],
            "tuned_ensemble_selected_config_labels": ";".join(item["tuned_ensemble_selected_config_labels"]),
        }
        if has_explicit_arms:
            row.update(
                {
                    "raw_adapted_tuned_config_label": item["raw_adapted_tuned_config_label"],
                    "raw_adapted_tuned_validation_deployment_error": item[
                        "raw_adapted_tuned_validation_deployment_error"
                    ],
                    "guarded_tuned_config_label": item["guarded_tuned_config_label"],
                    "guarded_tuned_validation_deployment_error": item[
                        "guarded_tuned_validation_deployment_error"
                    ],
                    "reported_arm_aliases": json.dumps(item["reported_arm_aliases"], sort_keys=True),
                    "distinct_reported_arms": ";".join(item["distinct_reported_arms"]),
                }
            )
        if item["standard_tabarena"] is not None:
            row["standard_tabarena_benchmark_error"] = item["standard_tabarena"]["benchmark_error"]
            row["standard_tabarena_deployment_error"] = item["standard_tabarena"]["deployment_error"]
        csv_methods = list(dict.fromkeys(["default", "tuned", "tuned_ensemble", *candidate_arms]))
        for method in csv_methods:
            row[f"{method}_benchmark_error"] = item[method]["benchmark_error"]
            row[f"{method}_deployment_error"] = item[method]["deployment_error"]
            row[f"{method}_minus_identity_benchmark_error"] = (
                item[method]["benchmark_error"] - item["identity"]["benchmark_error"]
            )
            row[f"{method}_relative_error_change_vs_identity"] = float(
                _relative_error_change(
                    np.asarray([item["identity"]["benchmark_error"]], dtype=float),
                    np.asarray([item[method]["benchmark_error"]], dtype=float),
                )[0]
            )
            if item["standard_tabarena"] is not None:
                row[f"{method}_minus_standard_tabarena_benchmark_error"] = (
                    item[method]["benchmark_error"] - item["standard_tabarena"]["benchmark_error"]
                )
                row[f"{method}_relative_error_change_vs_standard_tabarena"] = float(
                    _relative_error_change(
                        np.asarray([item["standard_tabarena"]["benchmark_error"]], dtype=float),
                        np.asarray([item[method]["benchmark_error"]], dtype=float),
                    )[0]
                )
        rows.append(row)
    csv_path = output_dir / "task_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    standard_available = [item["standard_tabarena"] is not None for item in task_summaries]
    end_to_end_results: dict[str, dict[str, Any]] = {}
    for offset, method in enumerate(distinct_arms):
        selected_items = [item for item in task_summaries if item["standard_tabarena"] is not None]
        if not selected_items:
            break
        standard_errors = np.asarray(
            [float(item["standard_tabarena"]["benchmark_error"]) for item in selected_items],
            dtype=float,
        )
        candidate_errors = np.asarray(
            [float(item[method]["benchmark_error"]) for item in selected_items],
            dtype=float,
        )
        selected_problem_types = np.asarray(
            [str(item["problem_type"]) for item in selected_items], dtype=object
        )
        end_to_end_results[method] = _paired_comparison_summary(
            reference=standard_errors,
            candidate=candidate_errors,
            problem_types=selected_problem_types,
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_seed=bootstrap_seed + 10_000 + offset,
            reference_label="full_outer_training_standard_tabicl",
            candidate_label=method,
        )
    summary = {
        "n_tasks": len(task_summaries),
        "n_skipped_tasks": len(skipped_tasks),
        "task_eligibility": task_eligibility,
        "skipped_tasks": skipped_tasks,
        "paired_elo_note": (
            "Elo deltas are paired DirectSpline-versus-identity values on this run only; "
            "they are not absolute ratings on Retouche's published multi-method pool. Raw-adapted "
            "arms show spline headroom without the identity guard; guarded arms show deployment behavior. "
            "If a bag never produces a finite learned checkpoint, its raw prediction conservatively falls "
            "back to identity and the per-task valid-checkpoint fraction records that fallback."
        ),
        "metric_note": "Binary: 1-AUC; multiclass: log loss; regression: RMSE for comparison, MSE for guard/HPO.",
        "guard_protocol": "retouche_per_bag_validation_guard_then_test_ensemble",
        "paired_results": paired,
        "distinct_paired_result_keys": distinct_arms,
        "paired_result_aliases": paired_result_aliases,
        "standard_tabarena": {
            "available": all(standard_available),
            "n_tasks_available": int(sum(standard_available)),
            "n_tasks_missing": int(len(standard_available) - sum(standard_available)),
            "note": (
                "This is normal TabICLv2 fitted once on every outer-training row. Cross-task raw error "
                "is intentionally not averaged because binary 1-AUC, multiclass log loss, and regression "
                "RMSE have incomparable scales."
            ),
        },
        "end_to_end_vs_standard_tabarena": {
            "available": bool(end_to_end_results),
            "n_tasks_available": int(sum(standard_available)),
            "causal_interpretation": "none_end_to_end_protocols_are_confounded",
            "primary_spline_effect_comparison": "paired_results_vs_matched_inner_bag_identity",
            "note": (
                "These dataset-paired results compare complete pipelines, not an isolated spline effect: "
                "the standard model fits once on all outer-training rows, while DirectSpline fits inner "
                "bags, uses OOF guard/tuning decisions, and averages bag predictions. Use the matched "
                "identity paired results above to attribute changes specifically to the spline."
            ),
            "results": end_to_end_results,
        },
        "task_results_csv": str(csv_path),
    }
    _json_dump(output_dir / "summary.json", summary)
    return summary
