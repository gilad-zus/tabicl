"""Leakage-safe building blocks for the DirectSpline benchmark experiments.

The direct OpenML runner uses these functions for fold-local preprocessing,
episode sampling, identity guarding, and paired-Elo summaries.  The module has
no TabArena, AutoGluon, Ray, or OpenML import, so those protocol decisions can
be tested independently of benchmark downloads and model checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score


ProblemType = Literal["binary", "multiclass", "regression"]


def _as_frame(frame: Any):
    """Return a pandas frame while delaying the optional pandas import."""
    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - TabArena requires pandas.
        raise ModuleNotFoundError(
            "The TabArena DirectSpline experiment requires pandas. "
            "Install TabArena with its benchmark extra first."
        ) from error
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return pd.DataFrame(frame)


@dataclass(frozen=True)
class GuardDecision:
    """Validation-only decision to deploy identity or the learned adapter."""

    use_adapted: bool
    identity_error: float
    adapted_error: float
    relative_improvement: float
    required_relative_improvement: float


def deployment_error(
    problem_type: ProblemType,
    labels: np.ndarray,
    prediction: np.ndarray,
    *,
    n_classes: int | None = None,
) -> float:
    """Return the TFM-Retouche deployment error for one validation prediction.

    The paper uses ``1 - AUC`` for binary tasks, log loss for multiclass tasks,
    and MSE for regression.  Lower is always better, which makes it safe to
    use in the same identity guard for every problem type.
    """
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if problem_type == "binary":
        if prediction.ndim != 2 or prediction.shape[1] != 2:
            raise ValueError("binary predictions must have shape (n_rows, 2)")
        if np.unique(labels).size != 2:
            raise ValueError("binary validation labels must contain both classes for ROC-AUC")
        return float(1.0 - roc_auc_score(labels, prediction[:, 1]))
    if problem_type == "multiclass":
        if prediction.ndim != 2:
            raise ValueError("multiclass predictions must have shape (n_rows, n_classes)")
        classes = np.arange(n_classes if n_classes is not None else prediction.shape[1])
        # TabICL produces float32 softmax values.  Casting them to NumPy does
        # not make their row sums exactly one, and sklearn's log_loss rightly
        # warns when the accumulated float32 rounding error is visible.  Make
        # the metric's probability contract explicit here.  This is also what
        # sklearn does internally after issuing that warning, so it does not
        # change the intended loss; it merely makes the input valid and keeps
        # a long benchmark log readable.
        probabilities = _normalise_probability_rows(prediction)
        return float(log_loss(labels, probabilities, labels=classes))
    if problem_type == "regression":
        if prediction.ndim != 1:
            prediction = prediction.reshape(-1)
        return float(mean_squared_error(labels, prediction))
    raise ValueError(f"unknown problem type: {problem_type!r}")


def _normalise_probability_rows(prediction: np.ndarray) -> np.ndarray:
    """Return finite non-negative class probabilities with unit row sums."""
    probabilities = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("multiclass predictions must be finite probabilities")
    if np.any(probabilities < 0.0):
        raise ValueError("multiclass predictions must be non-negative probabilities")
    row_sums = probabilities.sum(axis=1, keepdims=True, dtype=np.float64)
    if np.any(row_sums <= 0.0):
        raise ValueError("each multiclass prediction row must have positive probability mass")
    return probabilities / row_sums


def benchmark_error(
    problem_type: ProblemType,
    labels: np.ndarray,
    prediction: np.ndarray,
    *,
    n_classes: int | None = None,
) -> float:
    """Return the TabArena comparison error for an outer-test prediction.

    The identity guard deliberately follows Retouche and uses MSE for
    regression.  TabArena's cross-dataset comparison instead uses RMSE there.
    They are monotonic for a fixed task, but recording this distinction avoids
    silently mixing the deployment and leaderboard metrics.
    """
    if problem_type == "regression":
        return float(np.sqrt(deployment_error(problem_type, labels, prediction, n_classes=n_classes)))
    return deployment_error(problem_type, labels, prediction, n_classes=n_classes)


def paired_elo_delta(
    identity_errors: np.ndarray,
    candidate_errors: np.ndarray,
    *,
    tie_atol: float = 1e-12,
) -> dict[str, float | int]:
    """Compute a transparent two-method Elo-equivalent difference.

    This is intentionally not the absolute Elo value on TabArena's published
    multi-method pool.  It converts paired task wins/ties/losses into the
    conventional 400-point Elo log-odds scale, with a Jeffreys 0.5 pseudocount
    so an all-win pilot remains finite.
    """
    identity = np.asarray(identity_errors, dtype=float)
    candidate = np.asarray(candidate_errors, dtype=float)
    if identity.shape != candidate.shape or identity.ndim != 1:
        raise ValueError("paired error arrays must be one-dimensional with equal shape")
    valid = np.isfinite(identity) & np.isfinite(candidate)
    identity = identity[valid]
    candidate = candidate[valid]
    if identity.size == 0:
        raise ValueError("at least one finite paired result is required")
    candidate_wins = int(np.sum(candidate < identity - tie_atol))
    identity_wins = int(np.sum(candidate > identity + tie_atol))
    ties = int(identity.size - candidate_wins - identity_wins)
    score = (candidate_wins + 0.5 * ties + 0.5) / (identity.size + 1.0)
    delta = 400.0 * np.log10(score / (1.0 - score))
    return {
        "n_tasks": int(identity.size),
        "candidate_wins": candidate_wins,
        "identity_wins": identity_wins,
        "ties": ties,
        "candidate_score": float(score),
        "paired_elo_delta": float(delta),
    }


def bootstrap_paired_elo(
    identity_errors: np.ndarray,
    candidate_errors: np.ndarray,
    *,
    rounds: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap tasks to provide a simple uncertainty interval for paired Elo."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    identity = np.asarray(identity_errors, dtype=float)
    candidate = np.asarray(candidate_errors, dtype=float)
    valid = np.isfinite(identity) & np.isfinite(candidate)
    identity, candidate = identity[valid], candidate[valid]
    if identity.size == 0:
        raise ValueError("at least one finite paired result is required")
    rng = np.random.default_rng(seed)
    values = np.empty(rounds, dtype=float)
    for index in range(rounds):
        sampled = rng.integers(0, identity.size, size=identity.size)
        values[index] = float(paired_elo_delta(identity[sampled], candidate[sampled])["paired_elo_delta"])
    return {
        "rounds": float(rounds),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def choose_identity_guard(
    *,
    identity_error: float,
    adapted_error: float,
    required_relative_improvement: float = 0.005,
) -> GuardDecision:
    """Apply the paper's tolerance without inspecting the outer test fold.

    ``required_relative_improvement=0.005`` means that the adapted validation
    error must be at least 0.5% lower than identity.  A non-finite score, or a
    zero identity error, conservatively keeps the identity path.
    """
    if not 0.0 <= required_relative_improvement < 1.0:
        raise ValueError("required_relative_improvement must lie in [0, 1)")
    if not np.isfinite(identity_error) or not np.isfinite(adapted_error) or identity_error <= 0.0:
        relative_improvement = float("nan")
        use_adapted = False
    else:
        relative_improvement = (identity_error - adapted_error) / identity_error
        use_adapted = relative_improvement >= required_relative_improvement
    return GuardDecision(
        use_adapted=use_adapted,
        identity_error=float(identity_error),
        adapted_error=float(adapted_error),
        relative_improvement=float(relative_improvement),
        required_relative_improvement=float(required_relative_improvement),
    )


def _stratified_sample(
    labels: np.ndarray, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample every class deterministically, then distribute remaining rows."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if n_rows < classes.size or n_rows > labels.size:
        raise ValueError("stratified sample must contain every class and fit in the source")
    expected = counts.astype(float) * n_rows / labels.size
    taken = np.floor(expected).astype(int)
    taken = np.maximum(taken, 1)
    taken = np.minimum(taken, counts)
    while taken.sum() < n_rows:
        candidates = np.flatnonzero(taken < counts)
        index = candidates[np.argmax(expected[candidates] - taken[candidates])]
        taken[index] += 1
    while taken.sum() > n_rows:
        candidates = np.flatnonzero(taken > 1)
        index = candidates[np.argmax(taken[candidates] - expected[candidates])]
        taken[index] -= 1
    selected = [rng.permutation(np.flatnonzero(labels == label))[:count] for label, count in zip(classes, taken)]
    return rng.permutation(np.concatenate(selected))


def sample_episode_indices(
    labels: np.ndarray,
    *,
    problem_type: ProblemType,
    context_rows: int,
    query_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw disjoint train-only context/query rows for one adapter update."""
    labels = np.asarray(labels)
    if context_rows <= 0 or query_rows <= 0:
        raise ValueError("context_rows and query_rows must be positive")
    if labels.size < 2:
        raise ValueError("at least two fitting rows are required")
    if problem_type == "regression":
        total = min(labels.size, context_rows + query_rows)
        if total < 2:
            raise ValueError("regression episode needs context and query rows")
        context_size = min(context_rows, total - 1)
        query_size = min(query_rows, labels.size - context_size)
        permutation = rng.permutation(labels.size)
        return permutation[:context_size], permutation[context_size : context_size + query_size]

    classes, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("each classification class needs at least two fitting rows")
    # Reserve one row per class for the labelled query episode before choosing
    # the context.  This prevents accidental class disappearance in a step.
    context_size = min(context_rows, labels.size - classes.size)
    context_size = max(context_size, classes.size)
    context = _stratified_sample(labels, context_size, rng)
    remaining_mask = np.ones(labels.size, dtype=bool)
    remaining_mask[context] = False
    remaining = np.flatnonzero(remaining_mask)
    remaining_labels = labels[remaining]
    query_size = min(query_rows, remaining.size)
    query_size = max(query_size, classes.size)
    query_relative = _stratified_sample(remaining_labels, query_size, rng)
    return context, remaining[query_relative]


def sample_prediction_context(
    labels: np.ndarray,
    *,
    problem_type: ProblemType,
    max_context_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose the fixed, train-only context reused for guard and test calls."""
    if max_context_rows <= 0:
        raise ValueError("max_context_rows must be positive")
    labels = np.asarray(labels)
    n_rows = min(max_context_rows, labels.size)
    if problem_type == "regression":
        return rng.permutation(labels.size)[:n_rows]
    return _stratified_sample(labels, n_rows, rng)


@dataclass
class FoldPreprocessor:
    """Fold-local numeric/categorical conversion used by the adapter model.

    Numeric columns are median-imputed then standardized.  Categorical columns
    are ordinal encoded from *training only*, with an explicit unknown code,
    then standardized.  The stored metadata makes it possible to verify that
    no validation or outer-test rows influenced preprocessing.
    """

    columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numerical_medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    category_maps: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, frame: Any) -> "FoldPreprocessor":
        from pandas.api.types import is_numeric_dtype

        data = _as_frame(frame)
        numerical_columns: list[str] = []
        categorical_columns: list[str] = []
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        category_maps: dict[str, dict[str, int]] = {}
        for column in data.columns:
            name = str(column)
            series = data[column]
            if is_numeric_dtype(series.dtype):
                numerical_columns.append(name)
                values = np.asarray(series, dtype=np.float64).copy()
                values[~np.isfinite(values)] = np.nan
                median = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
                filled = np.where(np.isfinite(values), values, median)
            else:
                categorical_columns.append(name)
                tokens = _category_tokens(series)
                mapping = {token: index for index, token in enumerate(sorted(set(tokens)))}
                category_maps[name] = mapping
                filled = np.asarray([mapping[token] for token in tokens], dtype=np.float64)
                median = 0.0
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            medians[name] = median
            means[name] = mean
            scales[name] = scale
        return cls(
            columns=tuple(str(column) for column in data.columns),
            numerical_columns=tuple(numerical_columns),
            categorical_columns=tuple(categorical_columns),
            numerical_medians=medians,
            means=means,
            scales=scales,
            category_maps=category_maps,
        )

    @property
    def numerical_indices(self) -> np.ndarray:
        numerical = set(self.numerical_columns)
        return np.asarray([index for index, name in enumerate(self.columns) if name in numerical], dtype=np.int64)

    def transform(self, frame: Any) -> np.ndarray:
        data = _as_frame(frame)
        received = tuple(str(column) for column in data.columns)
        if received != self.columns:
            raise ValueError("prediction columns do not exactly match the fitting columns")
        output = np.empty((len(data), len(self.columns)), dtype=np.float32)
        numerical = set(self.numerical_columns)
        for index, name in enumerate(self.columns):
            series = data.iloc[:, index]
            if name in numerical:
                values = np.asarray(series, dtype=np.float64).copy()
                values[~np.isfinite(values)] = self.numerical_medians[name]
            else:
                mapping = self.category_maps[name]
                values = np.asarray([mapping.get(token, -1) for token in _category_tokens(series)], dtype=np.float64)
            output[:, index] = ((values - self.means[name]) / self.scales[name]).astype(np.float32)
        return output


def _category_tokens(series: Any) -> list[str]:
    """Create stable category tokens while keeping missingness distinct."""
    import pandas as pd

    values = series.astype("string")
    return ["<MISSING>" if pd.isna(value) else f"<VALUE>{value}" for value in values.tolist()]


DEFAULT_DIRECT_SPLINE_CONFIG: dict[str, Any] = {
    "adapter_steps": 150,
    "adapter_patience": 10,
    "validation_interval": 10,
    "max_context_rows": 512,
    "train_context_rows": 384,
    "query_batch_rows": 256,
    "evaluation_query_chunk_rows": 256,
    "n_control_points": 20,
    "learning_rate": 0.005,
    "weight_decay": 0.003,
    "grad_clip": 2.0,
    "gate_learning_rate_factor": 3.0,
    "trainable_location_scale": True,
    "cross_column_mixing_rank": 4,
    "cross_column_mixing_bound": 0.10,
    "guard_relative_improvement": 0.005,
    "random_state": 0,
}


def shared_random_direct_spline_configs(n_configs: int, *, seed: int) -> list[dict[str, Any]]:
    """Create the fixed, shared tuning configurations for every TabArena task."""
    if n_configs < 0:
        raise ValueError("n_configs must be non-negative")
    rng = np.random.default_rng(seed)

    def log_uniform(low: float, high: float) -> float:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    configs: list[dict[str, Any]] = []
    for index in range(n_configs):
        config = dict(DEFAULT_DIRECT_SPLINE_CONFIG)
        config.update(
            {
                "adapter_steps": int(rng.integers(100, 201)),
                "adapter_patience": int(rng.integers(10, 16)),
                "n_control_points": int(rng.choice([12, 16, 20, 24])),
                "learning_rate": log_uniform(1e-3, 1.5e-2),
                "weight_decay": log_uniform(1e-3, 5e-2),
                "grad_clip": log_uniform(1.0, 5.0),
                "gate_learning_rate_factor": log_uniform(2.0, 10.0),
                "cross_column_mixing_rank": int(rng.choice([0, 2, 4, 8])),
                "cross_column_mixing_bound": log_uniform(0.02, 0.20),
                "random_state": int(seed + index + 1),
            }
        )
        configs.append(config)
    return configs
