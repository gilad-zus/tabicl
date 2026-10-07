"""Context-conditioned numerical preprocessing for the synthetic-first pilot.

All generated parameters depend on the labelled context alone.  The same map
is applied to context and any later query batch.  Columns and class labels have
no positional embedding, so the conditioner is feature equivariant and class
label invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .bspline import evaluate_bspline, greville_abscissae, uniform_augmented_knots
from .statistics import UNSUPERVISED_SUMMARY_DIM, ColumnStatistics, summarize_context


@dataclass(frozen=True)
class JointParameters:
    location: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor                 # (B, 2, D)
    log_scale: torch.Tensor             # (B, 2, D)
    spline_controls: torch.Tensor | None
    spline_gate: torch.Tensor | None
    neural_first_weight: torch.Tensor | None  # (B, 2, D, 8)
    neural_first_bias: torch.Tensor | None
    neural_last_weight: torch.Tensor | None
    neural_last_bias: torch.Tensor | None
    neural_gate: torch.Tensor | None
    mixing: torch.Tensor | None         # (B, 2, D, D), spectral norm <= 0.1


class RawContextEncoder(nn.Module):
    """Pool within rows, then classes, then columns using shared attention."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("num_heads must divide hidden_dim")
        self.column_encoder = nn.Sequential(nn.LayerNorm(UNSUPERVISED_SUMMARY_DIM), nn.Linear(UNSUPERVISED_SUMMARY_DIM, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cell_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.row_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.row_norm = nn.LayerNorm(hidden_dim)
        self.class_encoder = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.class_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.class_norm = nn.LayerNorm(hidden_dim)
        self.column_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.column_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, stats: ColumnStatistics, missing: torch.Tensor) -> torch.Tensor:
        valid = (~missing) & torch.isfinite(x)
        z = ((x.float() - stats.location[:, None]) / stats.scale[:, None]).masked_fill(~valid, 0).clamp(-8, 8)
        features = torch.stack((z, z.square(), z.abs(), valid.float()), dim=-1)
        cells = self.cell_encoder(features) + self.column_encoder(stats.summary[..., :UNSUPERVISED_SUMMARY_DIM])[:, None]
        b, n, d, h = cells.shape
        rows = cells.reshape(b * n, d, h)
        attended, _ = self.row_attention(rows, rows, rows, need_weights=False)
        cells = self.row_norm(rows + attended).reshape(b, n, d, h)
        pooled = []
        for batch in range(b):
            labels = y[batch]
            present = torch.unique(labels[torch.isfinite(labels.float())])
            class_tokens = []
            for label in present:
                group = cells[batch, labels == label]
                mean = group.mean(dim=0)
                spread = (group - mean).square().mean(dim=0).clamp_min(1e-6).sqrt()
                frequency = mean.new_full((d, 1), math.log1p(group.shape[0] / n))
                class_tokens.append(self.class_encoder(torch.cat((mean, spread, frequency), dim=-1)))
            if not class_tokens:
                raise ValueError("context must contain at least one finite class")
            classes = torch.stack(class_tokens, dim=1)
            attended, _ = self.class_attention(classes, classes, classes, need_weights=False)
            pooled.append(self.class_norm(classes + attended).mean(dim=1))
        tokens = torch.stack(pooled)
        attended, _ = self.column_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.column_norm(tokens + attended)
        return self.output_norm(tokens + self.feedforward(tokens))


class JointPreprocessor(nn.Module):
    """Three matched arms: ``restricted``, ``joint``, ``no_spline``."""

    def __init__(self, arm: str, *, hidden_dim: int = 64, num_heads: int = 4,
                 n_control_points: int = 20, rank: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        if arm not in {"restricted", "joint", "no_spline"}:
            raise ValueError(f"unknown arm: {arm}")
        if n_control_points <= 3 or rank <= 0:
            raise ValueError("invalid spline or mixing size")
        self.arm, self.eps, self.rank = arm, eps, rank
        self.hidden_dim = hidden_dim
        self.n_control_points = n_control_points
        self.encoder = RawContextEncoder(hidden_dim, num_heads)
        self.slot_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.affine_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.affine_head.bias)
        if arm != "no_spline":
            knots = uniform_augmented_knots(n_control_points, 3)
            identity = greville_abscissae(knots, 3, n_control_points)
            self.register_buffer("knots", knots)
            self.register_buffer("identity_gaps", identity[1:] - identity[:-1])
            self.spline_head = nn.Linear(hidden_dim, n_control_points - 1)
            self.spline_gate_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.spline_head.weight)
            nn.init.zeros_(self.spline_head.bias)
            nn.init.zeros_(self.spline_gate_head.weight)
            nn.init.constant_(self.spline_gate_head.bias, torch.logit(torch.tensor(0.1)).item())
        if arm != "restricted":
            self.neural_first_head = nn.Linear(hidden_dim, 16)
            self.neural_last_head = nn.Linear(hidden_dim, 9)
            self.neural_gate_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.neural_last_head.weight)
            nn.init.zeros_(self.neural_last_head.bias)
            nn.init.zeros_(self.neural_gate_head.weight)
            nn.init.constant_(self.neural_gate_head.bias, torch.logit(torch.tensor(0.1)).item())
            self.mix_left_head = nn.Linear(hidden_dim, rank)
            self.mix_right_head = nn.Linear(hidden_dim, rank)
            self.mix_gate_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.mix_gate_head.weight)
            nn.init.zeros_(self.mix_gate_head.bias)

    def generate(self, x_context: torch.Tensor, y_context: torch.Tensor,
                 context_missing: torch.Tensor | None = None) -> JointParameters:
        if x_context.ndim != 3 or y_context.shape != x_context.shape[:2] or x_context.shape[-1] == 0:
            raise ValueError("expected nonempty (B, N, D) numerical context and (B, N) labels")
        if context_missing is None:
            context_missing = ~torch.isfinite(x_context)
        # Summary statistics have no trainable weights.  Keep raw cell encoding
        # differentiable while excluding the costly quantile graph.
        with torch.no_grad():
            stats = summarize_context(x_context, context_missing, y_context, eps=self.eps)
        tokens = self.encoder(x_context, y_context, stats, context_missing)[:, None] + self.slot_embeddings[None, :, None]
        affine = self.affine_head(tokens)
        shift, log_scale = affine[..., 0].tanh(), affine[..., 1].tanh()
        controls = spline_gate = None
        if self.arm != "no_spline":
            raw = self.spline_head(tokens)
            gaps = self.identity_gaps * torch.exp(2.0 * torch.tanh(raw))
            gaps = 2.0 * gaps / gaps.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            controls = torch.cat((torch.full_like(gaps[..., :1], -1.0), -1.0 + gaps.cumsum(dim=-1)), dim=-1)
            spline_gate = self.spline_gate_head(tokens).sigmoid().squeeze(-1)
        first_weight = first_bias = last_weight = last_bias = neural_gate = mixing = None
        if self.arm != "restricted":
            first = self.neural_first_head(tokens)
            last = self.neural_last_head(tokens)
            first_weight, first_bias = first[..., :8], first[..., 8:]
            last_weight, last_bias = last[..., :8], last[..., 8]
            neural_gate = self.neural_gate_head(tokens).sigmoid().squeeze(-1)
            left = self.mix_left_head(tokens)
            right = self.mix_right_head(tokens)
            rank = min(self.rank, x_context.shape[-1])
            raw_mix = left[..., :rank] @ right[..., :rank].transpose(-1, -2) / rank
            # Frobenius norm bounds spectral norm, including with varying D.
            bounded = 0.1 * raw_mix / raw_mix.norm(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            gate = self.mix_gate_head(tokens).mean(dim=-2).tanh()[..., None]
            mixing = gate * bounded
        return JointParameters(stats.location, stats.scale, shift, log_scale,
                               controls, spline_gate, first_weight, first_bias,
                               last_weight, last_bias, neural_gate, mixing)

    def apply(self, x: torch.Tensor, p: JointParameters, slot: int,
              missing: torch.Tensor | None = None) -> torch.Tensor:
        if slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        if missing is None:
            missing = ~torch.isfinite(x)
        # Match the ordinary scaler's fixed range guard. A feature with tiny
        # context variance can otherwise turn a finite query outlier into an
        # input beyond float16 range in the frozen backbone's CUDA AMP path.
        # This bound depends on no query statistics and is shared by train/eval.
        z = ((x.float() - p.location[:, None]) / p.scale[:, None]).masked_fill(missing, 0).clamp(-100, 100)
        a = p.log_scale[:, slot, None].exp() * z + p.shift[:, slot, None]
        result = a
        if p.spline_controls is not None:
            u = (a / 4.0).clamp(-1, 1)
            spline = evaluate_bspline(u, p.spline_controls[:, slot].float(), self.knots, 3)
            result = result + p.spline_gate[:, slot, None] * 4.0 * (spline - u)
        if p.neural_first_weight is not None:
            h = torch.tanh(a[..., None] * p.neural_first_weight[:, slot, None] + p.neural_first_bias[:, slot, None])
            residual = (h * p.neural_last_weight[:, slot, None]).sum(dim=-1) + p.neural_last_bias[:, slot, None]
            result = result + p.neural_gate[:, slot, None] * torch.tanh(residual)
        if p.mixing is not None:
            result = result + result @ p.mixing[:, slot]
        return result.masked_fill(missing, 0)

    def forward(self, x_context: torch.Tensor, x_query: torch.Tensor,
                y_context: torch.Tensor, *, slot: int = 0,
                context_missing: torch.Tensor | None = None,
                query_missing: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.generate(x_context, y_context, context_missing)
        return self.apply(x_context, p, slot, context_missing), self.apply(x_query, p, slot, query_missing)
