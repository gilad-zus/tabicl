"""Context-only numerical column summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ColumnStatistics:
    """Context-derived information for one summary vector per numerical column."""

    summary: torch.Tensor  # (B, D, S)
    location: torch.Tensor  # (B, D)
    scale: torch.Tensor  # (B, D)
    all_missing: torch.Tensor  # (B, D), bool


# The first 23 entries are entirely distributional.  The final entries are a
# class-permutation-invariant block, populated only when context labels exist.
SUPERVISED_SUMMARY_DIM = 8
UNSUPERVISED_SUMMARY_DIM = 23
SUMMARY_DIM = UNSUPERVISED_SUMMARY_DIM + SUPERVISED_SUMMARY_DIM


def summarize_context(
    x_context: torch.Tensor,
    missing: Optional[torch.Tensor] = None,
    y_context: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
) -> ColumnStatistics:
    """Compute fixed-size summaries using only context rows.

    ``missing`` identifies values that were imputed by the external pipeline.
    Their values are excluded from distribution summaries but the missing rate is
    retained.  When context labels are supplied, a fixed-size, symmetric
    class-conditional block is appended.  It deliberately never uses the
    numeric values of class identifiers, so relabeling classes cannot alter a
    summary.
    """
    if x_context.ndim != 3:
        raise ValueError("x_context must have shape (B, N, D)")
    x = x_context.float()
    b, n, d = x.shape
    if missing is None:
        missing = ~torch.isfinite(x)
    if missing.shape != x.shape:
        raise ValueError("missing must align with x_context")
    valid = (~missing) & torch.isfinite(x)
    masked = x.masked_fill(~valid, float("nan")).transpose(1, 2)  # (B, D, N)
    count = valid.sum(dim=1).float()
    all_missing = count == 0
    location = torch.nanmean(masked, dim=-1).nan_to_num(0.0)
    centered = masked - location.unsqueeze(-1)
    variance = torch.nanmean(centered.square(), dim=-1).nan_to_num(0.0)
    scale = variance.sqrt().clamp_min(eps)
    z = centered / scale.unsqueeze(-1)
    quantiles = torch.nanquantile(z, torch.tensor([.01, .05, .10, .25, .50, .75, .90, .95, .99], device=x.device), dim=-1)
    quantiles = quantiles.permute(1, 2, 0).nan_to_num(0.0)
    z_mean = torch.nanmean(z, dim=-1).nan_to_num(0.0)
    z_std = torch.nanmean((z - z_mean.unsqueeze(-1)).square(), dim=-1).sqrt().nan_to_num(0.0)
    median = quantiles[..., 4]
    mad = torch.nanmean((z - median.unsqueeze(-1)).abs(), dim=-1).nan_to_num(0.0)
    iqr = quantiles[..., 5] - quantiles[..., 3]
    skew = torch.nanmean(z.pow(3), dim=-1).nan_to_num(0.0).clamp(-10, 10)
    kurtosis = (torch.nanmean(z.pow(4), dim=-1).nan_to_num(0.0) - 3).clamp(-10, 10)
    tails = torch.stack([torch.nanmean((z.abs() > t).float().masked_fill(~valid.transpose(1, 2), float("nan")), dim=-1).nan_to_num(0.0) for t in (1.0, 2.0, 4.0)], dim=-1)
    missing_fraction = missing.float().mean(dim=1)
    unique_fraction = torch.zeros((b, d), dtype=x.dtype, device=x.device)
    for batch_idx in range(b):
        for feature_idx in range(d):
            values = x[batch_idx, valid[batch_idx, :, feature_idx], feature_idx]
            unique_fraction[batch_idx, feature_idx] = values.unique().numel() / max(values.numel(), 1)

    supervised = torch.zeros((b, d, SUPERVISED_SUMMARY_DIM), dtype=x.dtype, device=x.device)
    if y_context is not None:
        y = y_context
        if y.shape != (b, n):
            raise ValueError("y_context must have shape (B, N)")
        for batch_idx in range(b):
            for feature_idx in range(d):
                labels = y[batch_idx]
                keep = valid[batch_idx, :, feature_idx] & torch.isfinite(labels.float())
                if not keep.any():
                    continue
                values = z[batch_idx, feature_idx, keep]
                labels = labels[keep]
                classes, inverse, class_counts = torch.unique(
                    labels, sorted=True, return_inverse=True, return_counts=True
                )
                n_classes = classes.numel()
                frequencies = class_counts.float() / values.numel()
                class_means = torch.stack(
                    [values[inverse == class_idx].mean() for class_idx in range(n_classes)]
                )
                class_variances = torch.stack(
                    [values[inverse == class_idx].var(unbiased=False) for class_idx in range(n_classes)]
                )
                between_variance = (frequencies * (class_means - values.mean()).square()).sum()
                within_variance = (frequencies * class_variances).sum()
                if n_classes > 1:
                    normalized_entropy = -(frequencies * frequencies.clamp_min(eps).log()).sum() / torch.log(
                        torch.tensor(float(n_classes), device=x.device)
                    )
                    max_mean_separation = torch.pdist(class_means.unsqueeze(-1)).max()
                else:
                    normalized_entropy = values.new_zeros(())
                    max_mean_separation = values.new_zeros(())
                class_iqrs = torch.stack(
                    [
                        torch.quantile(values[inverse == class_idx], .75)
                        - torch.quantile(values[inverse == class_idx], .25)
                        for class_idx in range(n_classes)
                    ]
                )
                supervised[batch_idx, feature_idx] = torch.stack(
                    (
                        values.new_tensor(n_classes / max(n, 1)),
                        normalized_entropy,
                        frequencies.min(),
                        between_variance.sqrt(),
                        within_variance.sqrt(),
                        between_variance / within_variance.clamp_min(eps),
                        max_mean_separation,
                        (frequencies * class_iqrs).sum(),
                    )
                )

    summary = torch.cat(
        (
            (count / max(n, 1)).unsqueeze(-1),
            missing_fraction.unsqueeze(-1),
            unique_fraction.unsqueeze(-1),
            quantiles,
            z_mean.unsqueeze(-1),
            z_std.unsqueeze(-1),
            mad.unsqueeze(-1),
            iqr.unsqueeze(-1),
            skew.unsqueeze(-1),
            kurtosis.unsqueeze(-1),
            tails,
            (count <= 1).float().unsqueeze(-1),
            all_missing.float().unsqueeze(-1),
            supervised,
        ),
        dim=-1,
    )
    assert summary.shape[-1] == SUMMARY_DIM
    return ColumnStatistics(summary, location, scale, all_missing)
