"""Small class-distribution inputs for the existing spline residual generator."""
from __future__ import annotations

import torch
from torch import nn

from .statistics_preprocessing import StatisticsContextEncoder, StatisticsSplineResidualPreprocessor


class ClassProfileContextEncoder(StatisticsContextEncoder):
    """Encode each complete ordered class histogram before pooling over classes."""

    def __init__(self, conditional_profiles: bool):
        super().__init__()
        self.conditional_profiles = conditional_profiles
        # Seventeen boundaries define sixteen interior intervals and two tails.
        self.register_buffer("histogram_boundaries", torch.linspace(-4., 4., 17))
        self.class_profile_mlp = nn.Sequential(nn.Linear(20, 32), nn.GELU(),
                                               nn.Linear(32, 32), nn.GELU())
        self.profile_projection = nn.Linear(32, 64, bias=False)
        nn.init.zeros_(self.profile_projection.weight)
        self.last_profile_rms = 0.

    @torch.no_grad()
    def profiles(self, context_values, context_labels, statistics, context_missing):
        """Return one (classes, columns, 20) tensor for each batch member."""
        valid_values = torch.isfinite(context_values) & ~context_missing
        standardized_values = (context_values.float() - statistics.location[:, None]) / statistics.scale[:, None]
        histogram_bins = torch.bucketize(standardized_values.contiguous(), self.histogram_boundaries)
        # Imputed cells occupy a separate category, including all-missing columns.
        histogram_bins = histogram_bins.masked_fill(~valid_values, 18)
        batch_profiles = []
        for batch_position in range(context_values.shape[0]):
            labels = context_labels[batch_position]
            classes, class_counts = torch.unique(labels, return_counts=True)
            frequencies = class_counts.float() / labels.numel()
            column_count = context_values.shape[-1]
            column_bins = histogram_bins[batch_position].transpose(0, 1)
            marginal_counts = context_values.new_zeros((column_count, 19), dtype=torch.float32)
            marginal_counts.scatter_add_(1, column_bins, torch.ones_like(column_bins, dtype=torch.float32))
            marginal_probabilities = marginal_counts / labels.numel()
            class_profiles = []
            for class_position, class_label in enumerate(classes):
                if self.conditional_profiles:
                    selected_bins = column_bins[:, labels == class_label]
                    class_histogram = torch.zeros_like(marginal_counts)
                    class_histogram.scatter_add_(1, selected_bins, torch.ones_like(selected_bins, dtype=torch.float32))
                    probabilities = class_histogram / class_counts[class_position]
                else:
                    # This control preserves marginal information and model capacity.
                    probabilities = marginal_probabilities
                class_frequency = frequencies[class_position].expand(column_count, 1)
                class_profiles.append(torch.cat((probabilities, class_frequency), dim=-1))
            batch_profiles.append((torch.stack(class_profiles), frequencies))
        return batch_profiles

    def forward(self, context_values, context_labels, statistics, context_missing):
        summary_tokens = super().forward(context_values, context_labels, statistics, context_missing)
        pooled_profiles = []
        for profiles, frequencies in self.profiles(context_values, context_labels, statistics, context_missing):
            class_embeddings = self.class_profile_mlp(profiles)
            # Class frequencies give a symmetric pool without numeric label identities.
            pooled_profiles.append((class_embeddings * frequencies[:, None, None]).sum(dim=0))
        profile_tokens = self.profile_projection(torch.stack(pooled_profiles))
        self.last_profile_rms = float(profile_tokens.detach().square().mean().sqrt())
        return summary_tokens + profile_tokens


class ClassProfileSplinePreprocessor(StatisticsSplineResidualPreprocessor):
    """Change only conditioning inputs; retain native residuals and affine/spline heads."""

    def __init__(self, conditional_profiles: bool):
        super().__init__()
        self.encoder = ClassProfileContextEncoder(conditional_profiles)
