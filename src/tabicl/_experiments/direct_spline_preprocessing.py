"""Numerical preprocessing intervention for the DirectSpline development pilot."""

from __future__ import annotations

import numpy as np


class MinimalNumericalPreprocessor:
    """Replace numerical normalization while preserving each branch's categories.

    Inputs are the ordinary T-fitted encoder's outputs, so its missing-value
    imputation and categorical encoding are shared by both experimental routes.
    Numerical columns receive only a T-fitted mean/std transform. No power
    transform, outlier clipping, or extra post-spline normalization is applied.
    """

    def __init__(self, original, fit_values: np.ndarray, numerical_indices: np.ndarray):
        self.original = original
        self.numerical_indices = np.asarray(numerical_indices, dtype=int)
        fit_values = np.asarray(fit_values, dtype=np.float64)
        if fit_values.ndim != 2 or not np.isfinite(fit_values).all():
            raise ValueError("minimal preprocessing requires finite encoded fitting values")
        self.n_features_in_ = fit_values.shape[1]
        numeric = fit_values[:, self.numerical_indices]
        self.mean_ = numeric.mean(axis=0)
        self.scale_ = numeric.std(axis=0)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        self._category_probe = fit_values[0].copy()
        self.X_transformed_ = self.transform(fit_values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("minimal preprocessing received a different feature layout")
        if not np.isfinite(values).all():
            raise ValueError("minimal preprocessing received nonfinite encoded values")
        # Numerical values are irrelevant to categorical transforms. Substitute
        # a valid training probe so an extreme query cannot trigger a power-
        # transform fallback that unnecessarily changes categorical values.
        categorical_input = values.copy()
        categorical_input[:, self.numerical_indices] = self._category_probe[self.numerical_indices]
        output = np.asarray(self.original.transform(categorical_input)).copy()
        output[:, self.numerical_indices] = (
            values[:, self.numerical_indices] - self.mean_
        ) / self.scale_
        return output


def install_minimal_numerical_preprocessing(bundle) -> None:
    """Keep the original ensemble slots/permutations and replace their numerics."""
    generator = bundle.estimator.ensemble_generator_
    generator.preprocessors_ = {
        method: MinimalNumericalPreprocessor(
            original, generator.X_, bundle.numerical_indices
        )
        for method, original in generator.preprocessors_.items()
    }
