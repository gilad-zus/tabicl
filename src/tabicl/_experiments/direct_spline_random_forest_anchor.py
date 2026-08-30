"""Local Random-Forest anchor experiment for saved DirectSpline results.

This module deliberately has a narrow purpose.  Given a completed
DirectSpline experiment, it reuses its frozen OpenML outer splits, trains one
deterministic full-outer-training sklearn Random Forest per task, and fits a
three-method Bradley--Terry board over:

* ``RandomForest_D`` (the anchor, rebased to 1000 Elo),
* ``TabICLv2_D`` (the saved full-outer-training standard baseline), and
* ``DirectSpline_D`` (one saved DirectSpline deployment arm).

It is a local calibration experiment, *not* a reproduction of the published
TabArena / Retouche leaderboard: its Random Forest is sklearn-based and the
method/dataset pool is intentionally much smaller.  In particular, this
module never calibrates Random Forest probabilities and never uses a
validation or outer-test label to choose Random Forest hyperparameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tabicl._experiments.direct_spline_openml import OpenMLTaskData, _json_dump, _safe_name
from tabicl._experiments.direct_spline_protocol import FoldPreprocessor, benchmark_error, deployment_error


METHOD_RANDOM_FOREST = "RandomForest_D"
METHOD_TABICL = "TabICLv2_D"
METHOD_DIRECT_SPLINE = "DirectSpline_D"
DEFAULT_METHOD_ORDER = (METHOD_RANDOM_FOREST, METHOD_TABICL, METHOD_DIRECT_SPLINE)
DEFAULT_RANDOM_FOREST_CONFIG: dict[str, Any] = {
    "implementation": "sklearn.ensemble.RandomForest",
    "n_estimators": 100,
    "classification_criterion": "gini",
    "regression_criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features_classifier": "sqrt",
    "max_features_regressor": 1.0,
    "bootstrap": True,
    "random_state": 0,
}
DEFAULT_BT_CONFIG: dict[str, Any] = {
    "model": "Bradley-Terry logistic maximum-likelihood with anchor-fixed ridge stabilization",
    "anchor_method": METHOD_RANDOM_FOREST,
    "anchor_elo": 1000.0,
    "tie_atol": 1e-12,
    "ridge_strength": 1e-6,
    "maxiter": 10_000,
}

# ``model`` is persisted as descriptive metadata, rather than a numerical
# option accepted by either Bradley--Terry fitting routine.
BRADLEY_TERRY_FIT_CONFIG_KEYS = frozenset(
    {"anchor_method", "anchor_elo", "tie_atol", "ridge_strength", "maxiter"}
)


@dataclass(frozen=True)
class RandomForestTaskResult:
    """Persisted metrics and prediction for one full-outer-training RF fit."""

    metadata: dict[str, Any]
    prediction: np.ndarray


def bradley_terry_fit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fitting options from the persisted Bradley--Terry metadata."""

    unknown = set(config) - (set(BRADLEY_TERRY_FIT_CONFIG_KEYS) | {"model"})
    if unknown:
        raise ValueError(f"unknown Bradley--Terry configuration fields: {sorted(unknown)}")
    return {key: config[key] for key in BRADLEY_TERRY_FIT_CONFIG_KEYS if key in config}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_random_forest_config(config: Mapping[str, Any] | None, *, n_jobs: int) -> dict[str, Any]:
    """Merge a small explicit override with the frozen reference configuration."""

    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    result = dict(DEFAULT_RANDOM_FOREST_CONFIG)
    if config is not None:
        unknown = sorted(set(config).difference({*result, "n_jobs"}))
        if unknown:
            raise ValueError(f"unknown Random Forest config keys: {unknown}")
        if "n_jobs" in config and int(config["n_jobs"]) != int(n_jobs):
            raise ValueError("Random Forest config n_jobs conflicts with the n_jobs argument")
        result.update(config)
    result["n_jobs"] = int(n_jobs)
    if int(result["n_estimators"]) < 1:
        raise ValueError("n_estimators must be positive")
    if int(result["min_samples_split"]) < 2 or int(result["min_samples_leaf"]) < 1:
        raise ValueError("Random Forest minimum-sample settings are invalid")
    return result


def _classification_prediction(model: RandomForestClassifier, features: np.ndarray, n_classes: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    expected = np.arange(n_classes, dtype=np.int64)
    if probabilities.shape != (features.shape[0], n_classes) or not np.array_equal(classes, expected):
        raise RuntimeError(
            "Random Forest did not retain exactly the outer-training label classes; "
            f"classes={classes.tolist()}, expected={expected.tolist()}"
        )
    return probabilities


def fit_random_forest_task(
    *,
    task: OpenMLTaskData,
    config: Mapping[str, Any] | None = None,
    n_jobs: int = 1,
) -> RandomForestTaskResult:
    """Fit a predeclared full-outer-training Random Forest and score its test prediction.

    ``FoldPreprocessor`` is fitted only on the outer-training rows.  It keeps
    categorical encoding and missing-value handling deterministic while
    preventing outer-test feature statistics from influencing the model.
    """

    resolved = resolve_random_forest_config(config, n_jobs=n_jobs)
    preprocessor = FoldPreprocessor.fit(task.x_train)
    train_features = preprocessor.transform(task.x_train)
    test_features = preprocessor.transform(task.x_test)
    common = {
        "n_estimators": int(resolved["n_estimators"]),
        "max_depth": resolved["max_depth"],
        "min_samples_split": int(resolved["min_samples_split"]),
        "min_samples_leaf": int(resolved["min_samples_leaf"]),
        "bootstrap": bool(resolved["bootstrap"]),
        "random_state": int(resolved["random_state"]),
        "n_jobs": int(resolved["n_jobs"]),
    }
    if task.problem_type == "regression":
        model = RandomForestRegressor(
            **common,
            criterion=str(resolved["regression_criterion"]),
            max_features=resolved["max_features_regressor"],
        )
        prediction = np.asarray(model.fit(train_features, task.y_train).predict(test_features), dtype=np.float64)
    else:
        if task.n_classes is None:
            raise ValueError("classification task has no class count")
        model = RandomForestClassifier(
            **common,
            criterion=str(resolved["classification_criterion"]),
            max_features=resolved["max_features_classifier"],
        )
        model.fit(train_features, task.y_train)
        prediction = _classification_prediction(model, test_features, task.n_classes)
    metadata = {
        "method": METHOD_RANDOM_FOREST,
        "task_id": int(task.task_id),
        "dataset_id": int(task.dataset_id),
        "dataset_name": str(task.dataset_name),
        "problem_type": str(task.problem_type),
        "n_classes": task.n_classes,
        "outer_split_hash": str(task.outer_split_hash),
        "preprocessor": "FoldPreprocessor.fit(outer_training_rows_only)",
        "random_forest_config": resolved,
        "benchmark_error": float(benchmark_error(task.problem_type, task.y_test, prediction, n_classes=task.n_classes)),
        "deployment_error": float(deployment_error(task.problem_type, task.y_test, prediction, n_classes=task.n_classes)),
        "outer_test_scored_after_fit": True,
    }
    return RandomForestTaskResult(metadata=metadata, prediction=prediction)


def save_random_forest_task_result(*, output_dir: Path, result: RandomForestTaskResult) -> Path:
    """Persist a completed RF task in a self-contained, resume-safe directory."""

    metadata = result.metadata
    directory = output_dir / "random_forest" / f"task_{metadata['task_id']}_{_safe_name(str(metadata['dataset_name']))}"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "predictions.npz", prediction=np.asarray(result.prediction, dtype=np.float64))
    _json_dump(directory / "summary.json", metadata)
    return directory


def load_random_forest_task_result(*, output_dir: Path, task: OpenMLTaskData, expected_config: Mapping[str, Any]) -> RandomForestTaskResult:
    """Load and validate one previously saved RF task result without refitting it."""

    directory = output_dir / "random_forest" / f"task_{task.task_id}_{_safe_name(task.dataset_name)}"
    summary_path = directory / "summary.json"
    prediction_path = directory / "predictions.npz"
    if not summary_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"missing Random Forest artifacts for task {task.task_id}: {directory}")
    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        metadata.get("method") != METHOD_RANDOM_FOREST
        or int(metadata.get("task_id", -1)) != task.task_id
        or metadata.get("outer_split_hash") != task.outer_split_hash
        or metadata.get("random_forest_config") != dict(expected_config)
    ):
        raise ValueError(f"saved Random Forest artifacts do not match the immutable experiment for task {task.task_id}")
    with np.load(prediction_path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
    expected_shape = (len(task.y_test),) if task.problem_type == "regression" else (len(task.y_test), int(task.n_classes))
    if prediction.shape != expected_shape or not np.isfinite(prediction).all():
        raise ValueError(f"saved Random Forest prediction has invalid shape or values for task {task.task_id}")
    return RandomForestTaskResult(metadata=metadata, prediction=prediction)


def _battle_data(
    errors: np.ndarray,
    *,
    tie_atol: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Turn per-task errors into Bradley--Terry pairwise logistic examples."""

    if errors.ndim != 2 or errors.shape[0] < 1 or errors.shape[1] < 2 or not np.isfinite(errors).all():
        raise ValueError("errors must be a finite two-dimensional task-by-method array")
    features: list[np.ndarray] = []
    outcomes: list[int] = []
    records: list[dict[str, int]] = []
    for task_index, row in enumerate(errors):
        for left in range(errors.shape[1]):
            for right in range(left + 1, errors.shape[1]):
                feature = np.zeros(errors.shape[1], dtype=np.float64)
                feature[left] = 1.0
                feature[right] = -1.0
                if row[left] < row[right] - tie_atol:
                    features.append(feature)
                    outcomes.append(1)
                    records.append({"task_index": task_index, "winner": left, "loser": right, "tie": 0})
                elif row[left] > row[right] + tie_atol:
                    features.append(feature)
                    outcomes.append(0)
                    records.append({"task_index": task_index, "winner": right, "loser": left, "tie": 0})
                else:
                    # A tie contributes half a win to each method.
                    features.extend((feature, feature))
                    outcomes.extend((1, 0))
                    records.append({"task_index": task_index, "winner": left, "loser": right, "tie": 1})
    return np.asarray(features, dtype=np.float64), np.asarray(outcomes, dtype=np.float64), records


def fit_bradley_terry_elo(
    *,
    errors_by_method: Mapping[str, Sequence[float] | np.ndarray],
    anchor_method: str = METHOD_RANDOM_FOREST,
    anchor_elo: float = 1000.0,
    tie_atol: float = 1e-12,
    ridge_strength: float = 1e-6,
    maxiter: int = 10_000,
) -> dict[str, Any]:
    """Fit a small, anchored, tie-aware Bradley--Terry Elo board.

    Ratings are estimated in natural-logistic coordinates while keeping the
    anchor's coordinate fixed at zero.  A tiny ridge term gives finite,
    deterministic ratings even if a small local board contains a perfect
    head-to-head record.  The returned Elo scale follows the conventional
    400 points per 10:1 expected win-odds convention.
    """

    names = list(errors_by_method)
    if len(names) < 2 or len(names) != len(set(names)) or anchor_method not in names:
        raise ValueError("need at least two uniquely named methods including the anchor")
    arrays = [np.asarray(errors_by_method[name], dtype=np.float64) for name in names]
    if any(values.ndim != 1 for values in arrays) or len({values.size for values in arrays}) != 1:
        raise ValueError("every method needs a one-dimensional error vector of the same length")
    errors = np.column_stack(arrays)
    if not np.isfinite(errors).all():
        raise ValueError("Bradley--Terry input errors must be finite")
    if not np.isfinite(anchor_elo) or not np.isfinite(tie_atol) or tie_atol < 0.0:
        raise ValueError("anchor_elo and tie_atol must be finite, with non-negative tie_atol")
    if not np.isfinite(ridge_strength) or ridge_strength <= 0.0 or maxiter < 1:
        raise ValueError("ridge_strength and maxiter must be positive")
    features, outcomes, records = _battle_data(errors, tie_atol=tie_atol)
    anchor_index = names.index(anchor_method)
    free_indices = np.asarray([index for index in range(len(names)) if index != anchor_index], dtype=np.int64)
    reduced_features = features[:, free_indices]

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = reduced_features @ coefficients
        # log(1 + exp(logit)) - y * logit is stable through logaddexp.
        loss = float(np.logaddexp(0.0, logits).sum() - outcomes @ logits)
        loss += float(0.5 * ridge_strength * np.dot(coefficients, coefficients))
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))
        gradient = reduced_features.T @ (probabilities - outcomes) + ridge_strength * coefficients
        return loss, np.asarray(gradient, dtype=np.float64)

    result = minimize(
        fun=lambda coefficients: objective(coefficients)[0],
        x0=np.zeros(free_indices.size, dtype=np.float64),
        jac=lambda coefficients: objective(coefficients)[1],
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Bradley--Terry fit did not converge: {result.message}")
    coefficients = np.zeros(len(names), dtype=np.float64)
    coefficients[free_indices] = result.x
    elo = float(anchor_elo) + coefficients * (400.0 / np.log(10.0))
    wins = np.zeros(len(names), dtype=float)
    ties = np.zeros(len(names), dtype=float)
    losses = np.zeros(len(names), dtype=float)
    for record in records:
        left, right = int(record["winner"]), int(record["loser"])
        if record["tie"]:
            ties[left] += 1.0
            ties[right] += 1.0
        else:
            wins[left] += 1.0
            losses[right] += 1.0
    return {
        "rating_kind": "local_anchored_bradley_terry_elo",
        "method_order": names,
        "anchor_method": anchor_method,
        "anchor_elo": float(anchor_elo),
        "n_tasks": int(errors.shape[0]),
        "n_pairwise_battles": int(len(records)),
        "ridge_strength": float(ridge_strength),
        "tie_atol": float(tie_atol),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": True,
            "iterations": int(result.nit),
            "objective": float(result.fun),
        },
        "ratings": {
            name: {
                "elo": float(elo[index]),
                "logistic_strength": float(coefficients[index]),
                "wins": float(wins[index]),
                "ties": float(ties[index]),
                "losses": float(losses[index]),
            }
            for index, name in enumerate(names)
        },
    }


def bootstrap_bradley_terry_elo(
    *,
    errors_by_method: Mapping[str, Sequence[float] | np.ndarray],
    rounds: int,
    seed: int,
    anchor_method: str = METHOD_RANDOM_FOREST,
    anchor_elo: float = 1000.0,
    tie_atol: float = 1e-12,
    ridge_strength: float = 1e-6,
    maxiter: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap whole tasks, preserving all method outcomes within a task."""

    if rounds < 1:
        raise ValueError("bootstrap rounds must be positive")
    names = list(errors_by_method)
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in errors_by_method.items()}
    n_tasks = next(iter(arrays.values())).size
    if n_tasks < 1 or any(values.shape != (n_tasks,) for values in arrays.values()):
        raise ValueError("bootstrap requires equally sized non-empty error vectors")
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(rounds, dtype=np.float64) for name in names}
    for draw_index in range(rounds):
        indices = rng.integers(0, n_tasks, size=n_tasks)
        board = fit_bradley_terry_elo(
            errors_by_method={name: values[indices] for name, values in arrays.items()},
            anchor_method=anchor_method,
            anchor_elo=anchor_elo,
            tie_atol=tie_atol,
            ridge_strength=ridge_strength,
            maxiter=maxiter,
        )
        for name in names:
            draws[name][draw_index] = float(board["ratings"][name]["elo"])
    return {
        "rounds": int(rounds),
        "seed": int(seed),
        "ratings": {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for name, values in draws.items()
        },
    }


def make_anchor_manifest(
    *,
    source_dir: Path,
    source_manifest: Mapping[str, Any],
    direct_spline_arm: str,
    random_forest_config: Mapping[str, Any],
    bt_config: Mapping[str, Any],
    task_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the immutable provenance record for a local RF-anchor run."""

    source_manifest_path = source_dir / "experiment_manifest.json"
    source_summary_path = source_dir / "summary.json"
    if not source_manifest_path.is_file() or not source_summary_path.is_file():
        raise FileNotFoundError("source directory must contain experiment_manifest.json and summary.json")
    return {
        "schema_version": 1,
        "experiment": "direct_spline_openml_random_forest_anchor",
        "scope": (
            "Local anchored three-method Bradley--Terry board. It is not a published TabArena/Retouche "
            "absolute leaderboard because the dataset and method pools differ."
        ),
        "source": {
            "directory": str(source_dir),
            "experiment_manifest_sha256": _sha256_file(source_manifest_path),
            "summary_sha256": _sha256_file(source_summary_path),
            "source_run_fingerprint_sha256": source_manifest.get("run_fingerprint_sha256"),
        },
        "methods": {
            "random_forest": METHOD_RANDOM_FOREST,
            "standard_tabicl": METHOD_TABICL,
            "direct_spline_arm": direct_spline_arm,
            "direct_spline_label": METHOD_DIRECT_SPLINE,
        },
        "random_forest_config": dict(random_forest_config),
        "bradley_terry_config": dict(bt_config),
        "tasks": [dict(record) for record in task_records],
    }
