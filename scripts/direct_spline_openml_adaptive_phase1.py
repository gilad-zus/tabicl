"""Phase 1 DirectSpline architecture comparison on the standard OpenML suite.

This is the deliberately small development experiment for the question that
the original fixed cubic, 20-control-point spline could not answer: whether
different numerical columns need different univariate capacities, and whether
a bounded row-conditional cross-column interaction makes those transforms
useful together.

It runs three predeclared arms on identical persisted train/validation/test
splits: the existing fixed cubic-20 DirectSpline, an adaptive per-column
mixture of (degree 1, 4 points), (degree 2, 8 points), and (degree 3, 20
points), then that same adaptive mixture with a rank-4 conditional amplitude.
The inner validation split selects step zero (identity) or a training
checkpoint; test remains an evaluation only.  There is no query-label use,
identity penalty, hyperparameter sweep, or architecture selection from test.
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        validation_selected_refit=True,
        adaptive_phase1=True,
        description=__doc__,
    )
