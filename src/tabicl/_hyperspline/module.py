"""Shared, context-conditioned monotone B-spline numerical transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .bspline import evaluate_bspline, greville_abscissae, uniform_augmented_knots
from .statistics import ColumnStatistics, SUMMARY_DIM, summarize_context


@dataclass(frozen=True)
class HyperSplineParameters:
    control_points: torch.Tensor  # (B, D, K)
    gate: torch.Tensor  # (B, D)
    location: torch.Tensor  # (B, D)
    scale: torch.Tensor  # (B, D)


class HyperSplineTransform(nn.Module):
    """Generate one monotone scalar spline for each context numerical column."""

    def __init__(
        self,
        n_control_points: int = 10,
        degree: int = 3,
        hidden_dim: int = 64,
        standardized_range: float = 4.0,
        generate_location: bool = False,
        generate_scale: bool = False,
        gate_initial_probability: float = 0.01,
        location_bound: float = 1.0,
        log_scale_bound: float = 1.0,
        gap_adjustment_bound: float = 2.0,
        target_aware: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if degree != 3:
            raise ValueError("the first HyperSpline version fixes cubic degree=3")
        if n_control_points <= degree:
            raise ValueError("n_control_points must exceed degree")
        if standardized_range <= 0:
            raise ValueError("standardized_range must be positive")
        self.n_control_points = n_control_points
        self.degree = degree
        self.standardized_range = standardized_range
        self.generate_location = generate_location
        self.generate_scale = generate_scale
        self.location_bound = location_bound
        self.log_scale_bound = log_scale_bound
        self.gap_adjustment_bound = gap_adjustment_bound
        self.target_aware = target_aware
        self.eps = eps
        knots = uniform_augmented_knots(n_control_points, degree)
        identity = greville_abscissae(knots, degree, n_control_points)
        self.register_buffer("knots", knots)
        self.register_buffer("identity_control_points", identity)
        self.register_buffer("identity_gaps", identity[1:] - identity[:-1])
        output_dim = (n_control_points - 1) + 3
        self.mlp = nn.Sequential(
            nn.LayerNorm(SUMMARY_DIM),
            nn.Linear(SUMMARY_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        gate_bias = torch.logit(torch.tensor(gate_initial_probability)).item()
        last.bias.data[n_control_points - 1] = gate_bias

    def generate_parameters(self, statistics: ColumnStatistics) -> HyperSplineParameters:
        raw = self.mlp(statistics.summary)
        gap_raw = raw[..., : self.n_control_points - 1]
        gap = self.identity_gaps * torch.exp(self.gap_adjustment_bound * torch.tanh(gap_raw))
        gap = 2.0 * gap / gap.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        control_points = torch.cat((torch.full_like(gap[..., :1], -1.0), -1.0 + gap.cumsum(dim=-1)), dim=-1)
        gate = torch.sigmoid(raw[..., self.n_control_points - 1])
        loc_raw = raw[..., self.n_control_points]
        scale_raw = raw[..., self.n_control_points + 1]
        location = statistics.location
        scale = statistics.scale
        if self.generate_location:
            location = location + scale * self.location_bound * torch.tanh(loc_raw)
        if self.generate_scale:
            scale = scale * torch.exp(self.log_scale_bound * torch.tanh(scale_raw))
        return HyperSplineParameters(control_points, gate, location, scale.clamp_min(self.eps))

    def apply_transform(
        self, x: torch.Tensor, parameters: HyperSplineParameters, missing: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.float()
        z = (x_float - parameters.location.unsqueeze(1)) / parameters.scale.unsqueeze(1)
        u = (z / self.standardized_range).clamp(-1.0, 1.0)
        spline = evaluate_bspline(u, parameters.control_points.float(), self.knots, self.degree)
        transformed = z + parameters.gate.unsqueeze(1) * self.standardized_range * (spline - u)
        if missing is not None:
            transformed = transformed.masked_fill(missing, 0.0)
        return transformed.to(original_dtype)

    def forward(
        self,
        x_context: torch.Tensor,
        x_query: torch.Tensor,
        context_missing: Optional[torch.Tensor] = None,
        query_missing: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        *,
        return_parameters: bool = False,
    ):
        statistics = summarize_context(
            x_context, context_missing, y_context if self.target_aware else None, eps=self.eps
        )
        parameters = self.generate_parameters(statistics)
        context_out = self.apply_transform(x_context, parameters, context_missing)
        query_out = self.apply_transform(x_query, parameters, query_missing)
        if return_parameters:
            return context_out, query_out, parameters
        return context_out, query_out


class DirectSplineTransform(nn.Module):
    """Per-dataset trainable spline used only for the headroom experiment."""

    def __init__(self, x_context: torch.Tensor, n_control_points: int = 10, degree: int = 3, standardized_range: float = 4.0, eps: float = 1e-6) -> None:
        super().__init__()
        if x_context.ndim != 3:
            raise ValueError("x_context must have shape (B, N, D)")
        if degree != 3 or n_control_points <= degree:
            raise ValueError("DirectSplineTransform requires valid fixed cubic splines")
        self.degree = degree
        self.standardized_range = standardized_range
        self.eps = eps
        statistics = summarize_context(x_context, eps=eps)
        knots = uniform_augmented_knots(n_control_points, degree)
        identity = greville_abscissae(knots, degree, n_control_points)
        self.register_buffer("knots", knots)
        self.register_buffer("identity_gaps", identity[1:] - identity[:-1])
        self.register_buffer("location", statistics.location)
        self.register_buffer("scale", statistics.scale)
        self.gap_logits = nn.Parameter(torch.zeros(x_context.shape[0], x_context.shape[2], n_control_points - 1))
        self.gate_logits = nn.Parameter(torch.full((x_context.shape[0], x_context.shape[2]), torch.logit(torch.tensor(0.01))))

    def parameters_for_transform(self) -> HyperSplineParameters:
        gaps = self.identity_gaps * torch.exp(torch.tanh(self.gap_logits))
        gaps = 2.0 * gaps / gaps.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        controls = torch.cat((torch.full_like(gaps[..., :1], -1.0), -1.0 + gaps.cumsum(dim=-1)), dim=-1)
        return HyperSplineParameters(controls, torch.sigmoid(self.gate_logits), self.location, self.scale)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        params = self.parameters_for_transform()
        z = (x.float() - params.location.unsqueeze(1)) / params.scale.unsqueeze(1)
        u = (z / self.standardized_range).clamp(-1.0, 1.0)
        spline = evaluate_bspline(u, params.control_points, self.knots, self.degree)
        return (z + params.gate.unsqueeze(1) * self.standardized_range * (spline - u)).to(x.dtype)


class FrozenTabICLHyperSpline(nn.Module):
    """Apply HyperSpline only to numerical columns before a frozen TabICL call.

    The adapter is intentionally tensor-native.  External categorical encoding
    and its existing preprocessing happen before this boundary; categorical
    columns are copied through unchanged here.
    """

    def __init__(self, backbone: nn.Module, hyperspline: HyperSplineTransform) -> None:
        super().__init__()
        self.backbone = backbone
        self.hyperspline = hyperspline
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        """Train HyperSpline while permanently retaining frozen backbone inference mode."""
        self.hyperspline.train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        """Return the only parameters an optimizer may update."""
        return self.hyperspline.parameters()

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Expose only HyperSpline weights to optimizers built from this adapter.

        The frozen backbone remains a registered module so device transfers and
        state-dict traversal stay correct, but it cannot be accidentally added
        to a training optimizer through ``adapter.parameters()``.
        """
        name_prefix = f"{prefix}." if prefix else ""
        yield from self.hyperspline.named_parameters(
            prefix=f"{name_prefix}hyperspline",
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def forward(
        self,
        x_context: torch.Tensor,
        x_query: torch.Tensor,
        y_context: torch.Tensor,
        numerical_mask: torch.Tensor,
        context_missing: Optional[torch.Tensor] = None,
        query_missing: Optional[torch.Tensor] = None,
        *,
        return_parameters: bool = False,
        **backbone_kwargs,
    ):
        if numerical_mask.ndim != 1 or numerical_mask.dtype != torch.bool:
            raise ValueError("numerical_mask must be a bool tensor with shape (D,)")
        if x_context.shape[-1] != numerical_mask.numel() or x_query.shape[-1] != numerical_mask.numel():
            raise ValueError("numerical_mask must align with model feature dimension")
        if not numerical_mask.any():
            output = self.backbone(torch.cat((x_context, x_query), dim=1), y_context, **backbone_kwargs)
            return (output, None) if return_parameters else output
        num_context = x_context[..., numerical_mask]
        num_query = x_query[..., numerical_mask]
        num_context_missing = None if context_missing is None else context_missing[..., numerical_mask]
        num_query_missing = None if query_missing is None else query_missing[..., numerical_mask]
        transformed_context, transformed_query, parameters = self.hyperspline(
            num_context,
            num_query,
            num_context_missing,
            num_query_missing,
            y_context if self.hyperspline.target_aware else None,
            return_parameters=True,
        )
        merged_context = x_context.clone()
        merged_query = x_query.clone()
        merged_context[..., numerical_mask] = transformed_context
        merged_query[..., numerical_mask] = transformed_query
        output = self.backbone(torch.cat((merged_context, merged_query), dim=1), y_context, **backbone_kwargs)
        if return_parameters:
            return output, parameters
        return output
