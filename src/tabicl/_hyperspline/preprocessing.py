"""Typed ensemble assembly for HyperSpline estimator inference."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from tabicl._sklearn.preprocessing import EncodedTable, EnsembleGenerator


class HyperSplineEnsembleGenerator:
    """Reuse TabICL shuffles while deferring only true numerical columns.

    Categorical ordinal columns use the same per-column sklearn preprocessing as
    the baseline. Numerical columns remain in typed, imputed form until the
    tensor-native HyperSpline is evaluated.
    """

    def __init__(
        self,
        *,
        classification: bool,
        n_estimators: int,
        norm_methods,
        feat_shuffle_method: str,
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    @staticmethod
    def _merge(parts: EncodedTable) -> np.ndarray:
        return np.concatenate((parts.categorical, parts.numerical), axis=1)

    def fit(self, parts: EncodedTable, y: np.ndarray) -> "HyperSplineEnsembleGenerator":
        merged = self._merge(parts)
        # The existing generator remains the authoritative source for feature
        # filtering, class permutations, and feature permutations. Its numeric
        # preprocessors are deliberately never used below.
        self.ensemble_ = EnsembleGenerator(
            classification=self.classification,
            n_estimators=self.n_estimators,
            norm_methods=self.norm_methods,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        ).fit(merged, y)
        keep = self.ensemble_.unique_filter_.features_to_keep_
        n_cat = parts.categorical.shape[1]
        self.categorical_keep_ = keep[:n_cat]
        self.numerical_keep_ = keep[n_cat:]
        self.context_categorical_ = parts.categorical[:, self.categorical_keep_]
        self.context_numerical_ = parts.numerical[:, self.numerical_keep_]
        self.context_missing_ = parts.numerical_missing[:, self.numerical_keep_]
        self.y_ = np.asarray(y)
        return self

    @property
    def feature_shuffles_(self):
        return self.ensemble_.feature_shuffles_

    @property
    def class_shuffles_(self):
        return self.ensemble_.class_shuffles_

    @property
    def n_numerical_features_(self) -> int:
        return self.context_numerical_.shape[1]

    def query_numerical(self, parts: EncodedTable) -> tuple[np.ndarray, np.ndarray]:
        return parts.numerical[:, self.numerical_keep_], parts.numerical_missing[:, self.numerical_keep_]

    def build(
        self,
        query_parts: EncodedTable,
        context_numerical: torch.Tensor,
        query_numerical: torch.Tensor,
    ) -> OrderedDict:
        """Return baseline-shaped ensemble arrays after tensor HyperSpline output.

        ``context_numerical`` and ``query_numerical`` are canonical-order
        tensors shaped ``(1, N_C, D_N)`` and ``(1, N_Q, D_N)``.  They are
        transformed before shuffling, so generated parameters remain aligned.
        """
        if context_numerical.shape[0] != 1 or query_numerical.shape[0] != 1:
            raise ValueError("estimator inference expects one context dataset")
        if query_parts.categorical.shape[1] != self.categorical_keep_.shape[0]:
            raise ValueError("query categorical columns do not match fitted input")
        num_context = context_numerical.squeeze(0).detach().cpu().numpy()
        num_query = query_numerical.squeeze(0).detach().cpu().numpy()
        result = OrderedDict()
        for method, configs in self.ensemble_.ensemble_configs_.items():
            if self.context_categorical_.shape[1]:
                # Use the existing full preprocessing pipeline and retain only
                # its categorical output. This keeps categorical treatment
                # exactly identical to the baseline, including its randomness.
                baseline_preprocessor = self.ensemble_.preprocessors_[method]
                cat_context = baseline_preprocessor.X_transformed_[:, : self.context_categorical_.shape[1]]
                query_merged = self._merge(query_parts)
                query_filtered = self.ensemble_.unique_filter_.transform(query_merged)
                cat_query = baseline_preprocessor.transform(query_filtered)[:, : self.context_categorical_.shape[1]]
                canonical = np.concatenate((cat_context, num_context), axis=1)
                query_canonical = np.concatenate((cat_query, num_query), axis=1)
            else:
                canonical, query_canonical = num_context, num_query
            rows = np.concatenate((canonical, query_canonical), axis=0)
            x_variants, y_variants = [], []
            for feature_shuffle, class_shuffle in configs:
                x_variants.append(rows[:, feature_shuffle])
                if self.classification:
                    y_variants.append(np.asarray(class_shuffle)[self.y_.astype(int)])
                else:
                    y_variants.append(self.y_)
            result[method] = (np.stack(x_variants), np.stack(y_variants))
        return result
