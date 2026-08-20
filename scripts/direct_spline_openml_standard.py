"""Run DirectSpline through the normal TabICLv2 preprocessing ensemble.

This is the launcher to use for the corrected spline-only headroom experiment.
It is intentionally a thin wrapper around ``direct_spline_openml_lite`` so
both launchers retain the same frozen manifests, OpenML split handling,
guarded selection, and paired-Elo reporting.

Unlike the legacy ``direct_spline_openml_lite.py`` default, this launcher
creates the ordinary eight-view TabICLv2 preprocessing ensemble inside every
inner bag for both arms.  A freshly initialised spline must reproduce the
normal fitted estimator before adapter optimisation begins.  Thus the reported
Elo isolates the learned spline rather than a change in inference pipeline.

Examples
--------
One task, full-context parity smoke test::

    python scripts/direct_spline_openml_standard.py \
      --output-dir results/openml_direct_spline_standard/smoke \
      --task-id 363621 --device cuda:1

Memory-constrained diagnostic (not exact public-estimator parity)::

    python scripts/direct_spline_openml_standard.py \
      --output-dir results/openml_direct_spline_standard/capped_512 \
      --task-id 363621 --context-cap 512 --device cuda:1
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(default_pipeline="standard", required_pipeline="standard")
