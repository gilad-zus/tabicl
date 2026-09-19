"""Small learned numerical features appended to an unchanged input table."""

from __future__ import annotations

import torch
from torch import nn

from .bspline import evaluate_bspline, greville_abscissae, uniform_augmented_knots


class SplineFeatureExpansion(nn.Module):
    """One line or free cubic feature per numerical column.

    Parameters are offsets from the same fixed line, R*u, solely to make the
    initial function bit exact across capacities. All spline controls,
    including endpoints, are free; this is not an identity-gated transform.
    """

    expands_features = True

    def __init__(self, n_features: int, *, n_control_points: int | None, scale: float = 4.0):
        super().__init__()
        if n_features < 1 or scale <= 0:
            raise ValueError("positive feature count and scale required")
        self.n_features = n_features
        self.n_control_points = n_control_points
        self.scale = float(scale)
        if n_control_points is None:
            self.coefficients = nn.Parameter(torch.zeros(1, n_features, 2))
            self.register_buffer("knots", torch.empty(0))
        else:
            self.register_buffer("knots", uniform_augmented_knots(n_control_points, 3))
            self.coefficients = nn.Parameter(torch.zeros(1, n_features, n_control_points))
        # Fixed function-space grid, shared by all capacities. No row or label
        # statistics are used in the regularizer.
        self.register_buffer("penalty_grid", torch.linspace(-1.0, 1.0, 66)[1:-1])

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        u = (2.0 / torch.pi) * torch.atan(torch.pi * values / (2.0 * self.scale))
        if self.n_control_points is None:
            delta = self.coefficients[..., 0].unsqueeze(1) * u + self.coefficients[..., 1].unsqueeze(1)
        else:
            delta = evaluate_bspline(u, self.coefficients, self.knots, 3)
        return self.scale * u + delta

    def smoothness(self) -> torch.Tensor:
        """Mean squared second derivative of s(u)/R, not coefficient decay."""
        if self.n_control_points is None:
            return self.coefficients.sum() * 0.0
        controls = self.coefficients / self.scale
        knots = self.knots
        for degree in (3, 2):
            count = controls.shape[-1]
            denominator = knots[degree + 1:degree + count] - knots[1:count]
            controls = degree * (controls[..., 1:] - controls[..., :-1]) / denominator
            knots = knots[1:-1]
        grid = self.penalty_grid.view(1, -1, 1).expand(1, -1, self.n_features)
        return evaluate_bspline(grid, controls, knots, 1).square().mean()

    def curve_diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            return {"mean_squared_normalized_curvature": float(self.smoothness())}

    def full_control_points(self) -> torch.Tensor:
        if self.n_control_points is None:
            raise ValueError("the line arm has no spline controls")
        identity = greville_abscissae(self.knots, 3, self.n_control_points)
        return self.scale * identity + self.coefficients
