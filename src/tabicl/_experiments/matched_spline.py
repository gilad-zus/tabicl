"""Restricted affine/K20 spline on the same inputs as the Retouche adapter."""
from __future__ import annotations

import copy

import torch
from torch import nn

from tabicl._hyperspline.bspline import evaluate_bspline, greville_abscissae, uniform_augmented_knots
from .retouche import RetoucheAdapter


class MatchedSplineAdapter(nn.Module):
    def __init__(self, reference_adapter: RetoucheAdapter):
        super().__init__()
        feature_count = len(reference_adapter.gate)
        self.affine_shift = nn.Parameter(torch.zeros(feature_count))
        self.affine_log_scale = nn.Parameter(torch.zeros(feature_count))
        self.spline_gap_logits = nn.Parameter(torch.zeros(feature_count, 19))
        self.gate = nn.Parameter(torch.full((feature_count,), .1))
        knots = uniform_augmented_knots(20, 3)
        identity_controls = greville_abscissae(knots, 3, 20)
        identity_gaps = identity_controls.diff()
        identity_gaps = 2 * identity_gaps / identity_gaps.sum()
        # Use identical arithmetic for learned and identity curves at initialization.
        normalized_identity_gaps = 2 * identity_gaps / identity_gaps.sum()
        identity_controls = torch.cat((torch.tensor([-1.]), -1 + normalized_identity_gaps.cumsum(-1)))
        self.register_buffer("knots", knots)
        self.register_buffer("identity_gaps", identity_gaps)
        self.register_buffer("identity_controls", identity_controls)
        # Preserve the cross arm's exact initial projection, including its RNG draw.
        self.projection = copy.deepcopy(reference_adapter.projection)

    def forward(self, features):
        affine_corrections = torch.expm1(self.affine_log_scale.tanh()) * features + self.affine_shift.tanh()
        affine_features = features + affine_corrections
        spline_coordinates = (affine_features / 4).clamp(-1, 1)[None]
        positive_gaps = self.identity_gaps * torch.exp(2 * self.spline_gap_logits.tanh())
        normalized_gaps = 2 * positive_gaps / positive_gaps.sum(-1, keepdim=True)
        spline_controls = torch.cat((normalized_gaps.new_full((len(self.gate), 1), -1.),
                                     -1 + normalized_gaps.cumsum(-1)), dim=-1)[None]
        learned_curve = evaluate_bspline(spline_coordinates, spline_controls, self.knots, 3)[0]
        identity_curve = evaluate_bspline(spline_coordinates,
            self.identity_controls.expand_as(spline_controls), self.knots, 3)[0]
        # Each encoded coordinate has its own monotone correction; no cross interactions.
        corrected_features = affine_features + 4 * self.gate * (learned_curve - identity_curve)
        return self.projection(corrected_features)

    def optimizer(self):
        matrices, scalar_parameters = [], []
        for parameter_name, parameter in self.named_parameters():
            if parameter_name == "gate":
                continue
            (matrices if parameter.ndim == 2 else scalar_parameters).append(parameter)
        return torch.optim.AdamW([
            dict(params=matrices, lr=.005, weight_decay=.003, group_name="matrices"),
            dict(params=scalar_parameters, lr=.005, weight_decay=0., group_name="biases_and_normalizations"),
            dict(params=[self.gate], lr=.015, weight_decay=0., group_name="gate"),
        ], betas=(.9, .97))


def make_adapter(feature_count, adapter_kind="cross"):
    # Construct the cross initializer in both arms to share projection initialization.
    reference_adapter = RetoucheAdapter(feature_count)
    if adapter_kind == "cross":
        return reference_adapter
    if adapter_kind == "spline":
        return MatchedSplineAdapter(reference_adapter)
    raise ValueError(f"Unknown adapter kind: {adapter_kind}")
