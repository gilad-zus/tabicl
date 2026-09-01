"""Run the frozen unseen binary DirectSpline loss-objective confirmation.

This uses the ordinary Retouche-style eight preserved train/validation bags,
but holds architecture and optimizer fixed at the cubic-20 DirectSpline
default. It compares three predeclared training losses: cross entropy,
pairwise logistic AUC, and their equal-weight hybrid. A reviewed frozen
binary task-bank file is required, so the run cannot silently fall back to
the TabArena suite or tune on its outer-test outcomes.
"""

from __future__ import annotations

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        adaptive_retouche=True,
        binary_auc_objectives=True,
        description=__doc__,
    )
