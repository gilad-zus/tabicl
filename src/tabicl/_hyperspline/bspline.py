"""Small differentiable B-spline evaluator used by :mod:`tabicl._hyperspline`.

The evaluator intentionally has no learnable state: HyperSpline generates the
per-column control points.  Knots are fixed buffers on the owning module.
"""

from __future__ import annotations

import torch


def uniform_augmented_knots(n_control_points: int, degree: int, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """Return an open-uniform, clamped knot vector on ``[-1, 1]``."""
    if n_control_points <= degree:
        raise ValueError("n_control_points must be greater than degree")
    head = torch.full((degree + 1,), -1.0, dtype=dtype, device=device)
    tail = torch.full((degree + 1,), 1.0, dtype=dtype, device=device)
    n_internal = n_control_points - degree - 1
    internal = (
        torch.linspace(-1.0, 1.0, n_internal + 2, dtype=dtype, device=device)[1:-1]
        if n_internal
        else torch.empty(0, dtype=dtype, device=device)
    )
    return torch.cat((head, internal, tail))


def greville_abscissae(knots: torch.Tensor, degree: int, n_control_points: int) -> torch.Tensor:
    """Return control points that represent the identity curve exactly."""
    if degree == 0:
        return knots[:n_control_points]
    return torch.stack([knots[i + 1 : i + degree + 1].mean() for i in range(n_control_points)])


def _local_basis(u: torch.Tensor, knots: torch.Tensor, spans: torch.Tensor, degree: int) -> torch.Tensor:
    """Cox--de Boor basis values local to each ``span``.

    ``u`` and ``spans`` have shape ``(N, M)``.  The returned basis has shape
    ``(N, M, degree + 1)`` and remains differentiable with respect to ``u``.
    """
    shape = u.shape
    basis = [torch.ones(shape, dtype=u.dtype, device=u.device)]
    left = [torch.zeros(shape, dtype=u.dtype, device=u.device)]
    right = [torch.zeros(shape, dtype=u.dtype, device=u.device)]
    eps = torch.finfo(u.dtype).eps
    for j in range(1, degree + 1):
        left.append(u - knots[(spans + 1 - j).clamp(0, knots.numel() - 1)])
        right.append(knots[(spans + j).clamp(0, knots.numel() - 1)] - u)
        saved = torch.zeros(shape, dtype=u.dtype, device=u.device)
        next_basis = []
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = torch.where(denom.abs() > eps, basis[r] / denom, torch.zeros_like(denom))
            next_basis.append(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
        next_basis.append(saved)
        basis = next_basis
    return torch.stack(basis, dim=-1)


def evaluate_bspline(u: torch.Tensor, control_points: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate generated scalar B-splines.

    Args:
        u: Coordinates ``(B, N, D)`` in the fixed knot domain.
        control_points: Scalar controls ``(B, D, K)``.
        knots: Shared one-dimensional knot vector of length ``K + degree + 1``.

    Returns:
        Scalar values with shape ``(B, N, D)``.
    """
    if u.ndim != 3 or control_points.ndim != 3:
        raise ValueError("u and control_points must have shapes (B, N, D) and (B, D, K)")
    batch, n_rows, n_features = u.shape
    if control_points.shape[:2] != (batch, n_features):
        raise ValueError("control points must align with u batch and feature dimensions")
    n_control_points = control_points.shape[-1]
    if knots.ndim != 1 or knots.numel() != n_control_points + degree + 1:
        raise ValueError("invalid shared knot vector")

    # Treat every (table, column) pair as a separate curve while retaining a
    # common row dimension.  This avoids fixed feature-count module parameters.
    flat_u = u.permute(1, 0, 2).reshape(n_rows, batch * n_features).contiguous()
    flat_controls = control_points.reshape(batch * n_features, n_control_points)
    spans = torch.searchsorted(knots, flat_u.detach(), right=True) - 1
    spans = spans.clamp(min=degree, max=n_control_points - 1)
    basis = _local_basis(flat_u, knots, spans, degree)
    offsets = torch.arange(degree + 1, device=u.device).view(1, 1, -1)
    indices = (spans.unsqueeze(-1) - degree + offsets).clamp(0, n_control_points - 1)
    curve_indices = torch.arange(batch * n_features, device=u.device).view(1, -1, 1)
    controls = flat_controls[curve_indices, indices]
    result = (basis * controls).sum(dim=-1)
    return result.reshape(n_rows, batch, n_features).permute(1, 0, 2)
