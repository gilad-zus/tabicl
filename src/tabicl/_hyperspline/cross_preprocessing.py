"""Small directly fitted multiplicative corrections to native numerical views."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .statistics_preprocessing import DirectResidualPreprocessor


@dataclass
class CrossParameters:
    shift: torch.Tensor
    log_scale: torch.Tensor
    input_factors: torch.Tensor
    output_factors: torch.Tensor
    cross_bias: torch.Tensor
    cross_strength: torch.Tensor


class DirectCrossPreprocessor(nn.Module):
    """One affine map plus a rank-four cross layer, with two native-view slots."""

    conditioning = "direct"

    def __init__(self, initial, numerical_context, context_labels, context_missing):
        super().__init__()
        numerical_column_count = numerical_context.shape[-1]
        self.rank = min(4, numerical_column_count)
        if numerical_context.shape[0] != 1 or self.rank < 1:
            raise ValueError("cross fitting requires one nonempty numerical dataset")
        affine_initialization = DirectResidualPreprocessor(
            initial, numerical_context, context_labels, context_missing, restricted=True)
        affine_outputs = affine_initialization.raw_outputs["affine_head"].detach().clone()
        factor_shape = (1, 2, numerical_column_count, self.rank)
        # A nonzero input factor and zero output factor preserve exact identity
        # while allowing the output factor to receive gradients on the first step.
        random_generator = torch.Generator(device="cpu").manual_seed(0)
        input_factors = torch.randn(factor_shape, generator=random_generator) / math.sqrt(numerical_column_count)
        input_factors = input_factors.to(affine_outputs)
        scalar_shape = (1, 2, numerical_column_count, 1)
        self.raw_outputs = nn.ParameterDict({
            "affine_head": nn.Parameter(affine_outputs),
            "cross_input_factors": nn.Parameter(input_factors),
            "cross_output_factors": nn.Parameter(torch.zeros_like(input_factors)),
            "cross_bias": nn.Parameter(affine_outputs.new_zeros(scalar_shape)),
            "cross_gate": nn.Parameter(affine_outputs.new_full(scalar_shape, math.log(.02 / .98))),
        })

    def generate(self, numerical_context, context_labels, context_missing=None, *, numerical_column_positions):
        if numerical_context.shape[0] != 1 or numerical_context.shape[-1] != len(numerical_column_positions):
            raise ValueError("cross map requires aligned original numerical-column positions")
        # Inner episodes may remove constant columns; gather both factors in
        # the surviving original-column ordering, rather than renumbering them.
        aligned_outputs = {name: values.index_select(2, numerical_column_positions)
                           for name, values in self.raw_outputs.items()}
        affine_outputs = aligned_outputs["affine_head"]
        return CrossParameters(
            affine_outputs[..., 0].tanh(), affine_outputs[..., 1].tanh(),
            aligned_outputs["cross_input_factors"], aligned_outputs["cross_output_factors"],
            aligned_outputs["cross_bias"].squeeze(-1), aligned_outputs["cross_gate"].sigmoid().squeeze(-1))

    def apply_native(self, native_preprocessed_values, parameters, slot, missing):
        if slot not in (0, 1) or missing.shape != native_preprocessed_values.shape:
            raise ValueError("expected aligned missing mask and one of two native slots")
        if not torch.isfinite(native_preprocessed_values).all():
            raise ValueError("cross map requires finite native preprocessed values")
        affine_corrections = (torch.expm1(parameters.log_scale[:, slot, None]) * native_preprocessed_values
                              + parameters.shift[:, slot, None])
        affine_values = native_preprocessed_values + affine_corrections
        # Multiplying each feature by a learned combination of other features
        # introduces quadratic interactions without changing the column count.
        cross_combinations = ((affine_values @ parameters.input_factors[:, slot])
                              @ parameters.output_factors[:, slot].transpose(-1, -2))
        cross_corrections = (parameters.cross_strength[:, slot, None] * affine_values
                             * (cross_combinations + parameters.cross_bias[:, slot, None]))
        numerical_corrections = (affine_corrections + cross_corrections).masked_fill(missing, 0)
        return native_preprocessed_values + numerical_corrections
