"""Small context-statistics conditioner and a matched directly fitted reference."""
from __future__ import annotations

import math

import torch
from torch import nn

from .joint_preprocessing import JointParameters
from .residual_preprocessing import ResidualPreprocessor
from .statistics import SUMMARY_DIM, summarize_context


class StatisticsContextEncoder(nn.Module):
    """Share one small MLP across columns; fixed summaries replace all attention."""

    def __init__(self, hidden_dim=64, bottleneck_dim=32):
        super().__init__()
        input_dim = 2 * SUMMARY_DIM + 3
        self.summary_mlp = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(), nn.Linear(bottleneck_dim, hidden_dim), nn.GELU(),
        )

    def forward(self, context_values, context_labels, statistics, context_missing):
        column_summaries = statistics.summary
        batch_size, column_count, _ = column_summaries.shape
        # A fixed average supplies table context without a learned interaction layer.
        dataset_summaries = column_summaries.mean(dim=1, keepdim=True).expand_as(column_summaries)
        class_counts = context_labels.new_tensor([
            torch.unique(labels).numel() for labels in context_labels
        ]).log1p()
        size_summaries = torch.stack((
            class_counts.new_full((batch_size,), math.log1p(context_values.shape[1])),
            class_counts.new_full((batch_size,), math.log1p(column_count)),
            class_counts,
        ), dim=-1)[:, None].expand(batch_size, column_count, 3)
        conditioner_inputs = torch.cat((column_summaries, dataset_summaries, size_summaries), dim=-1)
        if not torch.isfinite(conditioner_inputs).all():
            raise FloatingPointError("nonfinite context summary")
        return self.summary_mlp(conditioner_inputs)


class StatisticsResidualPreprocessor(ResidualPreprocessor):
    """Keep all original transformation heads, bounds, slots, and native residuals."""

    def __init__(self):
        super().__init__("raw")
        self.encoder = StatisticsContextEncoder(self.hidden_dim)
        self.conditioning = "statistics"


class DirectResidualPreprocessor(nn.Module):
    """Optimize per-column raw head outputs, with the shared model's constraints."""

    apply_native = ResidualPreprocessor.apply_native

    def __init__(self, initial, numerical_context, context_labels, context_missing):
        super().__init__()
        self.conditioning, self.eps, self.rank = "direct", initial.eps, initial.rank
        for buffer_name in ("knots", "identity_gaps", "native_identity_controls"):
            self.register_buffer(buffer_name, getattr(initial, buffer_name).detach().clone())
        # Capture raw outputs once. There is no trainable encoder or shared head here.
        with torch.no_grad():
            statistics = summarize_context(numerical_context, context_missing, context_labels, eps=self.eps)
            column_tokens = initial.encoder(numerical_context, context_labels, statistics, context_missing)
            slot_tokens = column_tokens[:, None] + initial.slot_embeddings[None, :, None]
            self.raw_outputs = nn.ParameterDict({
                head_name: nn.Parameter(head(slot_tokens).detach().clone())
                for head_name, head in initial.named_children() if head_name.endswith("_head")
            })

    def generate(self, numerical_context, context_labels, context_missing=None, *, numerical_column_positions):
        if numerical_context.shape[0] != 1 or numerical_context.shape[-1] != len(numerical_column_positions):
            raise ValueError("direct map requires one dataset and aligned numerical columns")
        raw_outputs = {
            name: values.index_select(2, numerical_column_positions)
            for name, values in self.raw_outputs.items()
        }
        affine_outputs = raw_outputs["affine_head"]
        column_shifts = affine_outputs[..., 0].tanh()
        column_log_scales = affine_outputs[..., 1].tanh()

        # Positive gaps preserve monotone spline controls and the fixed [-1, 1] range.
        spline_gaps = self.identity_gaps * torch.exp(2 * raw_outputs["spline_head"].tanh())
        spline_gaps = 2 * spline_gaps / spline_gaps.sum(-1, keepdim=True).clamp_min(self.eps)
        spline_controls = torch.cat((torch.full_like(spline_gaps[..., :1], -1.),
                                     -1. + spline_gaps.cumsum(-1)), dim=-1)
        spline_strength = raw_outputs["spline_gate_head"].sigmoid().squeeze(-1)
        neural_first = raw_outputs["neural_first_head"]
        neural_last = raw_outputs["neural_last_head"]
        neural_strength = raw_outputs["neural_gate_head"].sigmoid().squeeze(-1)

        # Use the same rank and Frobenius bound as the generated mixing matrix.
        effective_rank = min(self.rank, numerical_context.shape[-1])
        mixing_left = raw_outputs["mix_left_head"][..., :effective_rank]
        mixing_right = raw_outputs["mix_right_head"][..., :effective_rank]
        raw_mixing = mixing_left @ mixing_right.transpose(-1, -2) / effective_rank
        bounded_mixing = .1 * raw_mixing / raw_mixing.norm(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        mixing_strength = raw_outputs["mix_gate_head"].mean(dim=-2).tanh()[..., None]
        with torch.no_grad():
            statistics = summarize_context(numerical_context, context_missing, context_labels, eps=self.eps)
        return JointParameters(
            statistics.location, statistics.scale, column_shifts, column_log_scales,
            spline_controls, spline_strength, neural_first[..., :8], neural_first[..., 8:],
            neural_last[..., :8], neural_last[..., 8], neural_strength,
            mixing_strength * bounded_mixing,
        )
