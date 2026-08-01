"""Shared, context-conditioned monotone B-spline numerical transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .bspline import evaluate_bspline, greville_abscissae, uniform_augmented_knots
from .statistics import (
    ColumnStatistics,
    SUMMARY_DIM,
    SUPERVISED_SUMMARY_DIM,
    UNSUPERVISED_SUMMARY_DIM,
    summarize_context,
)


@dataclass(frozen=True)
class HyperSplineParameters:
    control_points: torch.Tensor  # (B, D, K)
    gate: torch.Tensor  # (B, D)
    location: torch.Tensor  # (B, D)
    scale: torch.Tensor  # (B, D)
    supervised_residual_gate: Optional[torch.Tensor] = None  # (B, D) when a supervised residual is active


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
        supervised_residual: bool = False,
        supervised_residual_gate_initial_probability: float = 0.01,
        cross_column_residual: bool = False,
        cross_column_num_heads: int = 4,
        cross_column_residual_bound: float = 0.1,
        cross_column_gate_initial_probability: float = 0.01,
        raw_context_residual: bool = False,
        raw_context_num_heads: int = 4,
        raw_context_residual_bound: float = 0.5,
        raw_context_gate_initial_probability: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if degree != 3:
            raise ValueError("the first HyperSpline version fixes cubic degree=3")
        if n_control_points <= degree:
            raise ValueError("n_control_points must exceed degree")
        if standardized_range <= 0:
            raise ValueError("standardized_range must be positive")
        residual_variants = supervised_residual + cross_column_residual + raw_context_residual
        if residual_variants and not target_aware:
            raise ValueError("supervised residuals require target_aware=True")
        if residual_variants > 1:
            raise ValueError("supervised residual variants are mutually exclusive")
        if not 0 < supervised_residual_gate_initial_probability < 1:
            raise ValueError("supervised_residual_gate_initial_probability must be in (0, 1)")
        if cross_column_num_heads <= 0 or hidden_dim % cross_column_num_heads:
            raise ValueError("cross_column_num_heads must divide hidden_dim")
        if cross_column_residual_bound <= 0:
            raise ValueError("cross_column_residual_bound must be positive")
        if not 0 < cross_column_gate_initial_probability < 1:
            raise ValueError("cross_column_gate_initial_probability must be in (0, 1)")
        if raw_context_num_heads <= 0 or hidden_dim % raw_context_num_heads:
            raise ValueError("raw_context_num_heads must divide hidden_dim")
        if raw_context_residual_bound <= 0:
            raise ValueError("raw_context_residual_bound must be positive")
        if not 0 < raw_context_gate_initial_probability < 1:
            raise ValueError("raw_context_gate_initial_probability must be in (0, 1)")
        self.n_control_points = n_control_points
        self.degree = degree
        self.standardized_range = standardized_range
        self.generate_location = generate_location
        self.generate_scale = generate_scale
        self.location_bound = location_bound
        self.log_scale_bound = log_scale_bound
        self.gap_adjustment_bound = gap_adjustment_bound
        self.target_aware = target_aware
        self.supervised_residual = supervised_residual
        self.cross_column_residual = cross_column_residual
        self.cross_column_residual_bound = cross_column_residual_bound
        self.raw_context_residual = raw_context_residual
        self.raw_context_residual_bound = raw_context_residual_bound
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
        if supervised_residual:
            self.supervised_residual_mlp = nn.Sequential(
                nn.LayerNorm(SUPERVISED_SUMMARY_DIM),
                nn.Linear(SUPERVISED_SUMMARY_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.supervised_residual_mlp[-1].weight)
            nn.init.zeros_(self.supervised_residual_mlp[-1].bias)
            self.supervised_residual_gate_logit = nn.Parameter(
                torch.tensor(torch.logit(torch.tensor(supervised_residual_gate_initial_probability)).item())
            )
        if cross_column_residual:
            # Every operation is shared over the column-token axis and has no
            # positional encoding, making this block permutation equivariant.
            self.supervised_token_encoder = nn.Sequential(
                nn.LayerNorm(SUPERVISED_SUMMARY_DIM),
                nn.Linear(SUPERVISED_SUMMARY_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.cross_column_attention = nn.MultiheadAttention(
                hidden_dim, cross_column_num_heads, batch_first=True
            )
            self.cross_column_attention_norm = nn.LayerNorm(hidden_dim)
            self.cross_column_feedforward = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.cross_column_feedforward_norm = nn.LayerNorm(hidden_dim)
            self.cross_column_residual_head = nn.Linear(hidden_dim, output_dim)
            self.cross_column_gate_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.cross_column_residual_head.weight)
            nn.init.zeros_(self.cross_column_residual_head.bias)
            nn.init.zeros_(self.cross_column_gate_head.weight)
            self.cross_column_gate_head.bias.data.fill_(
                torch.logit(torch.tensor(cross_column_gate_initial_probability)).item()
            )
        if raw_context_residual:
            # Cell tokens contain only standardized values, missingness, and a
            # shared encoding of the column's distributional summary.  There
            # is no feature or class positional embedding, so all subsequent
            # attention and pooling operations are permutation equivariant.
            self.raw_context_column_encoder = nn.Sequential(
                nn.LayerNorm(UNSUPERVISED_SUMMARY_DIM),
                nn.Linear(UNSUPERVISED_SUMMARY_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.raw_context_cell_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.raw_context_row_attention = nn.MultiheadAttention(
                hidden_dim, raw_context_num_heads, batch_first=True
            )
            self.raw_context_row_norm = nn.LayerNorm(hidden_dim)
            self.raw_context_class_encoder = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.raw_context_class_attention = nn.MultiheadAttention(
                hidden_dim, raw_context_num_heads, batch_first=True
            )
            self.raw_context_class_norm = nn.LayerNorm(hidden_dim)
            self.raw_context_column_attention = nn.MultiheadAttention(
                hidden_dim, raw_context_num_heads, batch_first=True
            )
            self.raw_context_column_norm = nn.LayerNorm(hidden_dim)
            self.raw_context_feedforward = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.raw_context_feedforward_norm = nn.LayerNorm(hidden_dim)
            self.raw_context_residual_head = nn.Linear(hidden_dim, output_dim)
            self.raw_context_gate_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.raw_context_residual_head.weight)
            nn.init.zeros_(self.raw_context_residual_head.bias)
            nn.init.zeros_(self.raw_context_gate_head.weight)
            self.raw_context_gate_head.bias.data.fill_(
                torch.logit(torch.tensor(raw_context_gate_initial_probability)).item()
            )

    @property
    def has_supervised_residual(self) -> bool:
        return self.supervised_residual or self.cross_column_residual or self.raw_context_residual

    def freeze_marginal_policy(self) -> None:
        """Prevent the copied marginal MLP from receiving future gradients."""
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)

    def unfreeze_marginal_policy(self) -> None:
        """Allow controlled real-meta fine-tuning of the copied marginal MLP."""
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(True)

    def initialize_supervised_residual_from(self, marginal: "HyperSplineTransform") -> None:
        """Copy and freeze a trained marginal policy for supervised-residual training."""
        if not self.has_supervised_residual:
            raise ValueError("initialize_supervised_residual_from requires a supervised residual")
        if marginal.target_aware or marginal.has_supervised_residual:
            raise ValueError("the residual base must be a marginal-only HyperSpline")
        if self.n_control_points != marginal.n_control_points or self.degree != marginal.degree:
            raise ValueError("the residual and marginal HyperSpline architectures must match")
        self.mlp.load_state_dict(marginal.mlp.state_dict())
        self.freeze_marginal_policy()

    @property
    def supervised_residual_gate(self) -> torch.Tensor:
        if not self.supervised_residual:
            return self.knots.new_zeros(())
        return torch.sigmoid(self.supervised_residual_gate_logit)

    def _supervised_residual(
        self, supervised_statistics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a bounded raw correction and its per-column gate."""
        if self.supervised_residual:
            gate = self.supervised_residual_gate.expand(supervised_statistics.shape[:-1])
            return gate.unsqueeze(-1) * self.supervised_residual_mlp(supervised_statistics), gate
        if self.cross_column_residual:
            tokens = self.supervised_token_encoder(supervised_statistics)
            attended, _ = self.cross_column_attention(tokens, tokens, tokens, need_weights=False)
            tokens = self.cross_column_attention_norm(tokens + attended)
            tokens = self.cross_column_feedforward_norm(tokens + self.cross_column_feedforward(tokens))
            gate = torch.sigmoid(self.cross_column_gate_head(tokens)).squeeze(-1)
            residual = self.cross_column_residual_bound * torch.tanh(self.cross_column_residual_head(tokens))
            return gate.unsqueeze(-1) * residual, gate
        raise RuntimeError("_supervised_residual called without a supervised residual architecture")

    def _raw_context_residual(
        self,
        x_context: torch.Tensor,
        statistics: ColumnStatistics,
        y_context: torch.Tensor,
        context_missing: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode labelled context rows without relying on class-ID ordering.

        Labels only choose which rows are aggregated together.  Classes are
        subsequently processed with shared weights and mean pooled, making the
        branch invariant to arbitrary relabeling.
        """
        if context_missing is None:
            context_missing = ~torch.isfinite(x_context)
        valid = (~context_missing) & torch.isfinite(x_context)
        z = (x_context.float() - statistics.location.unsqueeze(1)) / statistics.scale.unsqueeze(1)
        z = z.masked_fill(~valid, 0.0).clamp(-8.0, 8.0)
        cell_features = torch.stack((z, z.square(), z.abs(), valid.float()), dim=-1)
        column_tokens = self.raw_context_column_encoder(
            statistics.summary[..., :UNSUPERVISED_SUMMARY_DIM]
        ).unsqueeze(1)
        cells = self.raw_context_cell_encoder(cell_features) + column_tokens
        b, n, d, h = cells.shape
        row_tokens = cells.reshape(b * n, d, h)
        attended, _ = self.raw_context_row_attention(row_tokens, row_tokens, row_tokens, need_weights=False)
        cells = self.raw_context_row_norm((row_tokens + attended).reshape(b, n, d, h))

        pooled_batches = []
        for batch_idx in range(b):
            labels = y_context[batch_idx]
            finite_labels = torch.isfinite(labels.float())
            classes, inverse = torch.unique(labels[finite_labels], sorted=True, return_inverse=True)
            class_tokens = []
            for class_idx in range(classes.numel()):
                rows = torch.where(finite_labels)[0][inverse == class_idx]
                class_cells = cells[batch_idx, rows]  # (N_class, D, H)
                mean = class_cells.mean(dim=0)
                # A singleton class (or a constant class token) has exactly
                # zero variance.  ``sqrt(0)`` has an infinite derivative,
                # which can turn the zero gradient from the initialised
                # residual head into NaN during the first backward pass.
                spread = (class_cells - mean).square().mean(dim=0).clamp_min(self.eps).sqrt()
                frequency = mean.new_full((d, 1), rows.numel() / max(n, 1)).log1p()
                class_tokens.append(self.raw_context_class_encoder(torch.cat((mean, spread, frequency), dim=-1)))
            if not class_tokens:
                pooled_batches.append(cells.new_zeros((d, h)))
                continue
            class_tokens_tensor = torch.stack(class_tokens, dim=0).transpose(0, 1)  # (D, C, H)
            class_attended, _ = self.raw_context_class_attention(
                class_tokens_tensor, class_tokens_tensor, class_tokens_tensor, need_weights=False
            )
            class_tokens_tensor = self.raw_context_class_norm(class_tokens_tensor + class_attended)
            pooled_batches.append(class_tokens_tensor.mean(dim=1))
        tokens = torch.stack(pooled_batches, dim=0)
        attended, _ = self.raw_context_column_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.raw_context_column_norm(tokens + attended)
        tokens = self.raw_context_feedforward_norm(tokens + self.raw_context_feedforward(tokens))
        gate = torch.sigmoid(self.raw_context_gate_head(tokens)).squeeze(-1)
        residual = self.raw_context_residual_bound * torch.tanh(self.raw_context_residual_head(tokens))
        return gate.unsqueeze(-1) * residual, gate

    def generate_parameters(
        self,
        statistics: ColumnStatistics,
        *,
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        context_missing: Optional[torch.Tensor] = None,
    ) -> HyperSplineParameters:
        if self.has_supervised_residual:
            # The frozen marginal policy always receives exactly the summary it
            # saw during marginal training: its 23 distributional entries plus
            # zeroed label entries.  Labels can affect only the residual path.
            marginal_summary = torch.cat(
                (
                    statistics.summary[..., :UNSUPERVISED_SUMMARY_DIM],
                    torch.zeros_like(statistics.summary[..., -SUPERVISED_SUMMARY_DIM:]),
                ),
                dim=-1,
            )
            raw = self.mlp(marginal_summary)
            if self.raw_context_residual:
                if x_context is None or y_context is None:
                    raise ValueError("raw_context_residual requires x_context and y_context")
                residual_raw, residual_gate = self._raw_context_residual(
                    x_context, statistics, y_context, context_missing
                )
            else:
                residual_raw, residual_gate = self._supervised_residual(
                    statistics.summary[..., -SUPERVISED_SUMMARY_DIM:]
                )
            raw = raw + residual_raw
        else:
            raw = self.mlp(statistics.summary)
            residual_gate = None
        return self._parameters_from_raw(raw, statistics, residual_gate)

    def generate_marginal_parameters(self, statistics: ColumnStatistics) -> HyperSplineParameters:
        """Generate the frozen marginal reference used by residual stabilizers."""
        if not self.has_supervised_residual:
            return self._parameters_from_raw(self.mlp(statistics.summary), statistics, None)
        marginal_summary = torch.cat(
            (
                statistics.summary[..., :UNSUPERVISED_SUMMARY_DIM],
                torch.zeros_like(statistics.summary[..., -SUPERVISED_SUMMARY_DIM:]),
            ),
            dim=-1,
        )
        return self._parameters_from_raw(self.mlp(marginal_summary), statistics, None)

    def _parameters_from_raw(
        self,
        raw: torch.Tensor,
        statistics: ColumnStatistics,
        residual_gate: Optional[torch.Tensor],
    ) -> HyperSplineParameters:
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
        return HyperSplineParameters(control_points, gate, location, scale.clamp_min(self.eps), residual_gate)

    def grid_deformation_penalty(
        self,
        parameters: HyperSplineParameters,
        grid_size: int = 33,
        reference_parameters: Optional[HyperSplineParameters] = None,
    ) -> torch.Tensor:
        """Mean squared transform difference on a standardized grid.

        Residual training supplies the frozen marginal parameters as the
        reference.  Without one, this retains the useful identity-relative
        diagnostic used by marginal-only models.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        grid = torch.linspace(
            -self.standardized_range, self.standardized_range, grid_size,
            dtype=parameters.location.dtype, device=parameters.location.device,
        ).view(1, grid_size, 1)
        grid = grid.expand(parameters.location.shape[0], -1, parameters.location.shape[1])
        raw_grid = parameters.location.unsqueeze(1) + grid * parameters.scale.unsqueeze(1)
        transformed = self.apply_transform(raw_grid, parameters)
        reference = (
            grid
            if reference_parameters is None
            else self.apply_transform(raw_grid, reference_parameters).float()
        )
        return (transformed.float() - reference).square().mean()

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
        parameters = self.generate_parameters(
            statistics,
            x_context=x_context,
            y_context=y_context if self.target_aware else None,
            context_missing=context_missing,
        )
        context_out = self.apply_transform(x_context, parameters, context_missing)
        query_out = self.apply_transform(x_query, parameters, query_missing)
        if return_parameters:
            return context_out, query_out, parameters
        return context_out, query_out


class DirectSplineTransform(nn.Module):
    """Per-dataset trainable spline used only for DirectSpline headroom tests.

    Optional basis freedoms are intentionally kept here first.  They are
    ablated before adding matching outputs to HyperSpline, so the amortizer is
    not made more complex without evidence that a freedom improves held-out
    rows.
    """

    def __init__(
        self,
        x_context: torch.Tensor,
        n_control_points: int = 10,
        degree: int = 3,
        standardized_range: float = 4.0,
        eps: float = 1e-6,
        *,
        trainable_range: bool = False,
        trainable_location_scale: bool = False,
        range_min: float = 1.0,
        range_max: float = 8.0,
        location_adjustment_bound: float = 1.0,
        scale_adjustment_bound: float = 2.0,
    ) -> None:
        super().__init__()
        if x_context.ndim != 3:
            raise ValueError("x_context must have shape (B, N, D)")
        if degree != 3 or n_control_points <= degree:
            raise ValueError("DirectSplineTransform requires valid fixed cubic splines")
        self.degree = degree
        self.standardized_range = standardized_range
        self.eps = eps
        if not 0 < range_min < standardized_range < range_max:
            raise ValueError("range_min < standardized_range < range_max is required")
        self.trainable_range = trainable_range
        self.trainable_location_scale = trainable_location_scale
        self.range_min = range_min
        self.range_max = range_max
        self.location_adjustment_bound = location_adjustment_bound
        self.scale_adjustment_bound = scale_adjustment_bound
        statistics = summarize_context(x_context, eps=eps)
        knots = uniform_augmented_knots(n_control_points, degree)
        identity = greville_abscissae(knots, degree, n_control_points)
        self.register_buffer("knots", knots)
        self.register_buffer("identity_gaps", identity[1:] - identity[:-1])
        self.register_buffer("location", statistics.location)
        self.register_buffer("scale", statistics.scale)
        self.gap_logits = nn.Parameter(torch.zeros(x_context.shape[0], x_context.shape[2], n_control_points - 1))
        self.gate_logits = nn.Parameter(torch.full((x_context.shape[0], x_context.shape[2]), torch.logit(torch.tensor(0.01))))
        self.location_offsets = nn.Parameter(torch.zeros_like(statistics.location), requires_grad=trainable_location_scale)
        self.log_scale_offsets = nn.Parameter(torch.zeros_like(statistics.scale), requires_grad=trainable_location_scale)
        initial_range_fraction = (standardized_range - range_min) / (range_max - range_min)
        self.range_logits = nn.Parameter(
            torch.full_like(statistics.location, torch.logit(torch.tensor(initial_range_fraction))),
            requires_grad=trainable_range,
        )

    def _location_scale_range(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        location = self.location + self.location_adjustment_bound * torch.tanh(self.location_offsets) * self.scale
        scale = self.scale * torch.exp(torch.log(torch.as_tensor(self.scale_adjustment_bound, device=self.scale.device)) * torch.tanh(self.log_scale_offsets))
        standardized_range = self.range_min + (self.range_max - self.range_min) * torch.sigmoid(self.range_logits)
        return location, scale, standardized_range

    def parameters_for_transform(self) -> HyperSplineParameters:
        gaps = self.identity_gaps * torch.exp(torch.tanh(self.gap_logits))
        gaps = 2.0 * gaps / gaps.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        controls = torch.cat((torch.full_like(gaps[..., :1], -1.0), -1.0 + gaps.cumsum(dim=-1)), dim=-1)
        location, scale, _ = self._location_scale_range()
        return HyperSplineParameters(controls, torch.sigmoid(self.gate_logits), location, scale)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        params = self.parameters_for_transform()
        _, _, standardized_range = self._location_scale_range()
        z = (x.float() - params.location.unsqueeze(1)) / params.scale.unsqueeze(1)
        u = (z / standardized_range.unsqueeze(1)).clamp(-1.0, 1.0)
        spline = evaluate_bspline(u, params.control_points, self.knots, self.degree)
        return (z + params.gate.unsqueeze(1) * standardized_range.unsqueeze(1) * (spline - u)).to(x.dtype)


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
