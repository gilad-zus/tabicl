"""Paper-based Retouche cross adapter and explicit experimental choices.

Reference: Nguyen et al., arXiv:2605.06047v1, sections 3.2--3.5 and table 2.
The paper does not specify small-normal scale, BatchNorm placement, or the
gate constraint. Those choices are declared here rather than inferred as fact.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class RetoucheCrossLayer(nn.Module):
    def __init__(self, feature_count: int, rank_ratio: float = .25):
        super().__init__()
        rank = max(1, int(rank_ratio * feature_count))
        self.input_projection = nn.Linear(feature_count, rank, bias=False)
        self.output_projection = nn.Linear(rank, feature_count, bias=False)
        self.cross_bias = nn.Parameter(torch.zeros(feature_count))
        self.normalization = nn.BatchNorm1d(feature_count)
        # The paper says small-normal without a scale. Use 0.01 for both factors.
        nn.init.normal_(self.input_projection.weight, std=.01)
        nn.init.normal_(self.output_projection.weight, std=.01)

    def forward(self, original_features, current_features):
        interactions = self.output_projection(self.input_projection(current_features)) + self.cross_bias
        # Normalize the correction branch before adding the layer residual.
        correction = original_features * interactions
        correction = self.normalization(correction.reshape(-1, correction.shape[-1])).reshape_as(correction)
        return current_features + correction


class RetoucheAdapter(nn.Module):
    def __init__(self, feature_count: int, *, feature_cap: int = 500):
        super().__init__()
        if feature_count < 1:
            raise ValueError("Retouche requires at least one input feature")
        self.layers = nn.ModuleList([RetoucheCrossLayer(feature_count) for _ in range(2)])
        self.gate = nn.Parameter(torch.full((feature_count,), .02))
        # High-dimensional tasks use the paper's learned orthogonal projection.
        self.projection = nn.Linear(feature_count, feature_cap, bias=False) if feature_count > feature_cap else nn.Identity()
        if isinstance(self.projection, nn.Linear):
            nn.init.orthogonal_(self.projection.weight)

    def forward(self, features):
        adapted_features = features
        for layer in self.layers:
            adapted_features = layer(features, adapted_features)
        # Directly optimize a gate constrained to [0,1] after each optimizer step.
        blended_features = features + self.gate * (adapted_features - features)
        return self.projection(blended_features)

    def project_identity(self, features):
        return self.projection(features)

    def optimizer(self):
        matrices, biases_and_normalizations = [], []
        for name, parameter in self.named_parameters():
            if name == "gate":
                continue
            (matrices if parameter.ndim == 2 else biases_and_normalizations).append(parameter)
        return torch.optim.AdamW([
            dict(params=matrices, lr=.005, weight_decay=.003, group_name="matrices"),
            dict(params=biases_and_normalizations, lr=.005, weight_decay=0., group_name="biases_and_normalizations"),
            dict(params=[self.gate], lr=.015, weight_decay=0., group_name="gate"),
        ], betas=(.9, .97))


def coslog4(progress: float) -> float:
    """RealMLP coslog4 schedule, as defined in pytabkit scheduling.py."""
    if not 0 <= progress <= 1:
        raise ValueError("schedule progress must be in [0,1]")
    return .5 * (1 - math.cos(2 * math.pi * math.log2(1 + 15 * progress)))


def epoch_episodes(labels, *, context_rows: int, query_rows: int, seed: int):
    """Visit the full fitting pool in shuffled mini-tables each epoch.

    One random example/class is reserved as shared context for this epoch.
    Remaining examples appear once as an ordinary context or query. Anchors
    are redrawn each epoch, so every row can receive query supervision later.
    This finite-memory heuristic is an explicit paper-reconstruction choice.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if context_rows < len(classes) or query_rows < 1 or len(labels) <= len(classes):
        raise ValueError("episode budgets cannot retain class-complete context and queries")
    random_generator = np.random.default_rng(seed)
    anchors = np.array([random_generator.choice(np.flatnonzero(labels == label)) for label in classes])
    available_mask = np.ones(len(labels), dtype=bool)
    available_mask[anchors] = False
    remaining_positions = random_generator.permutation(np.flatnonzero(available_mask))
    ordinary_context_rows = context_rows - len(anchors)
    episode_size = ordinary_context_rows + query_rows
    for start in range(0, len(remaining_positions), episode_size):
        episode_positions = remaining_positions[start:start + episode_size]
        # Allocate a query even when the last mini-table contains only one row.
        query_count = min(query_rows, max(1, round(len(episode_positions) * query_rows / episode_size)))
        query_positions = episode_positions[:query_count]
        context_positions = np.concatenate((anchors, episode_positions[query_count:]))
        yield context_positions, query_positions
