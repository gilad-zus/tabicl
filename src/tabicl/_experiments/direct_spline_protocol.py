"""Public import path for DirectSpline benchmark protocol utilities.

The implementation remains in the original module temporarily so existing
local experiment notebooks keep working.  New code should import from here;
it has no dependency on the TabArena package.
"""

from .tabarena_direct_spline_protocol import (
    DEFAULT_DIRECT_SPLINE_CONFIG,
    FoldPreprocessor,
    GuardDecision,
    ProblemType,
    benchmark_error,
    bootstrap_paired_elo,
    choose_identity_guard,
    deployment_error,
    paired_elo_delta,
    sample_episode_indices,
    sample_prediction_context,
    shared_random_direct_spline_configs,
)

__all__ = [
    "DEFAULT_DIRECT_SPLINE_CONFIG",
    "FoldPreprocessor",
    "GuardDecision",
    "ProblemType",
    "benchmark_error",
    "bootstrap_paired_elo",
    "choose_identity_guard",
    "deployment_error",
    "paired_elo_delta",
    "sample_episode_indices",
    "sample_prediction_context",
    "shared_random_direct_spline_configs",
]
