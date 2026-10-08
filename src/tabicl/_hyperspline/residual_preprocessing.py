"""Zero-shot residuals on native TabICL numerical views, with exact identity."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .bspline import evaluate_bspline
from .joint_preprocessing import JointParameters, JointPreprocessor, RawContextEncoder
from .statistics import UNSUPERVISED_SUMMARY_DIM, summarize_context


@contextmanager
def frozen_context_mode(backbone):
    """Extract the same FP32 context features during training and deployment."""
    modes = [(module, module.training) for module in backbone.modules()]
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("context feature extractor must be frozen")
    if any(isinstance(m, nn.Dropout) and m.p for m in backbone.modules()):
        raise ValueError("context feature extractor must have zero dropout")
    try:
        backbone.train()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def frozen_context_features(backbone, full_context, labels, numerical_positions):
    """No queries: align column/group embeddings, plus full-feature row summaries.

    In 'same' grouping, token (i-1) has feature i in its first channel (the
    upstream circular grouping offsets start at +1). 'valid' uses contiguous padded
    groups. CLS tokens precede feature tokens. Average the row's CLS vectors to
    preserve a checkpoint-derived embedding width independent of CLS count.
    """
    with frozen_context_mode(backbone):
        backbone.clear_cache()
        columns = backbone.col_embedder(full_context, labels, embed_with_test=False)
        grouping = backbone.col_embedder.feature_group
        indices = numerical_positions
        if grouping in (True, "same"):
            indices = (indices - 1) % full_context.shape[-1]
        elif grouping:
            indices = indices // backbone.col_embedder.feature_group_size
        indices = indices + backbone.col_embedder.reserve_cls_tokens
        aligned = columns.index_select(-2, indices).clone()
        rows = backbone.row_interactor(columns)
        rows = rows.reshape(*rows.shape[:2], backbone.row_interactor.num_cls,
                            backbone.row_interactor.embed_dim).mean(-2)
        if not torch.isfinite(aligned).all() or not torch.isfinite(rows).all():
            raise FloatingPointError("nonfinite frozen context representations")
        return aligned.detach(), rows.detach()


class TabICLContextEncoder(RawContextEncoder):
    """Keep row/class/column pooling, enrich cells with frozen full-table features."""

    def __init__(self, embedding_dim, hidden_dim=64, num_heads=4):
        super().__init__(hidden_dim, num_heads)
        self.embedding_dim = embedding_dim
        self.cell_encoder = nn.Sequential(nn.Linear(4 + 2 * embedding_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.column_feature_norm = nn.LayerNorm(embedding_dim)
        self.row_feature_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, y, stats, missing, frozen_features):
        valid = (~missing) & torch.isfinite(x)
        z = ((x.float() - stats.location[:, None]) / stats.scale[:, None]).masked_fill(~valid, 0).clamp(-8, 8)
        cells = torch.stack((z, z.square(), z.abs(), valid.float()), -1)
        columns, rows = frozen_features
        if columns.shape[:3] != x.shape or columns.shape[-1] != self.embedding_dim:
            raise ValueError("frozen column features are not aligned with numeric context")
        row_features = self.row_feature_norm(rows)[:, :, None].expand_as(columns)
        inputs = torch.cat((cells, self.column_feature_norm(columns), row_features), -1)
        cells = self.cell_encoder(inputs) + self.column_encoder(stats.summary[..., :UNSUPERVISED_SUMMARY_DIM])[:, None]
        return self.pool_cells(cells, y)


class ResidualPreprocessor(JointPreprocessor):
    """Generate bounded corrections; never replace native preprocessing."""

    def __init__(self, conditioning, *, embedding_dim=128, hidden_dim=64, num_heads=4):
        if conditioning not in ("raw", "backbone"):
            raise ValueError("unknown conditioning arm")
        super().__init__("joint", hidden_dim=hidden_dim, num_heads=num_heads)
        self.conditioning = conditioning
        if conditioning == "backbone":
            original = self.encoder.state_dict()
            self.encoder = TabICLContextEncoder(embedding_dim, hidden_dim, num_heads)
            matching = {k: v for k, v in original.items() if not k.startswith("cell_encoder.")}
            self.encoder.load_state_dict(matching, strict=False)
        # Evaluate the identity spline by the identical operations, so subtracting
        # it is exactly zero while derivatives of the learned controls remain.
        gaps = 2.0 * self.identity_gaps / self.identity_gaps.sum().clamp_min(self.eps)
        controls = torch.cat((gaps.new_full((1,), -1.), -1. + gaps.cumsum(-1)))
        self.register_buffer("native_identity_controls", controls)

    def generate(self, x_context, y_context, context_missing=None, *, frozen_features=None):
        if x_context.ndim != 3 or y_context.shape != x_context.shape[:2] or x_context.shape[-1] == 0:
            raise ValueError("expected nonempty numeric context and aligned labels")
        if context_missing is None:
            context_missing = ~torch.isfinite(x_context)
        with torch.no_grad():
            stats = summarize_context(x_context, context_missing, y_context, eps=self.eps)
        if self.conditioning == "backbone":
            if frozen_features is None:
                raise ValueError("backbone arm requires frozen context-only features")
            tokens = self.encoder(x_context, y_context, stats, context_missing, frozen_features)
        else:
            tokens = self.encoder(x_context, y_context, stats, context_missing)
        return self.parameters_from_tokens(tokens[:, None] + self.slot_embeddings[None, :, None], stats)

    def apply_native(self, native: torch.Tensor, p: JointParameters, slot: int,
                     missing: torch.Tensor) -> torch.Tensor:
        if slot not in (0, 1) or missing.shape != native.shape or not torch.isfinite(native).all():
            raise ValueError("expected finite native numeric values and aligned missing mask")
        # expm1(0) and every other delta are exactly zero; native imputed values
        # survive unchanged. Mask corrections, including cross-column mixing.
        delta = torch.expm1(p.log_scale[:, slot, None]) * native + p.shift[:, slot, None]
        a = native + delta
        u = (a / 4.).clamp(-1, 1)
        spline = evaluate_bspline(u, p.spline_controls[:, slot].float(), self.knots, 3)
        identity = self.native_identity_controls.expand_as(p.spline_controls[:, slot])
        anchor = evaluate_bspline(u, identity, self.knots, 3)
        delta = delta + p.spline_gate[:, slot, None] * 4. * (spline - anchor)
        hidden = torch.tanh(a[..., None] * p.neural_first_weight[:, slot, None] + p.neural_first_bias[:, slot, None])
        residual = (hidden * p.neural_last_weight[:, slot, None]).sum(-1) + p.neural_last_bias[:, slot, None]
        delta = delta + p.neural_gate[:, slot, None] * torch.tanh(residual)
        delta = delta + (native + delta) @ p.mixing[:, slot]
        return native + delta.masked_fill(missing, 0)
