"""Run the all-outer-training-row DirectSpline refit experiment.

This is the decisive companion to ``direct_spline_openml_standard.py``.  On
every dataset it fits one fresh DirectSpline on all outer-training rows using
the sole configuration and full step budget frozen in the manifest.  There is
no bag training, OOF checkpoint selection, or deployment guard: the spline is
evaluated unconditionally against the exact identity prediction of the same
normal eight-estimator TabICLv2 pipeline.

An earlier guarded-bag run may be supplied with ``--oof-source-dir``.  Those
artifacts are opened only after each all-row spline prediction is frozen and
are used solely to report whether the old validation signal correlates with
the actual full-context outcome.
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        full_context_refit=True,
    )
