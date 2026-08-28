"""Run the final preserved-fold DirectSpline experiment on the OpenML suite.

This is the matched TabICLv2 experiment for the question: can a learned
input-space spline improve the normal eight-bag TabICLv2 predictor on a
dataset, using only train/validation labels and without a full-data refit?

Every predeclared arm is trained in each deterministic inner bag on that
bag's fitting rows. It runs a 500-step maximum cosine trajectory with
checkpoints every 25 steps, stopping only after 12 consecutive non-improving
validation checks. It retains the validation-best *trained* checkpoint and
only then applies a Retouche-style 0.5% post-training identity guard. The
saved outer-test prediction is the mean of those eight guarded members. No
outer-test label selects a checkpoint, a guard, an architecture, or an
ensemble member.

The three arms are:

* D: fixed degree-3, 20-control-point DirectSpline;
* adaptive_columns: per-column soft routing over (degree, control-points)
  experts (1, 4), (2, 8), and (3, 20);
* conditional_adaptive_columns: the same adaptive basis plus a bounded
  rank-4 row-dependent residual amplitude.

The task summary reports D, T (one arm selected from guarded OOF validation),
and T+E (a greedy guarded-OOF ensemble). The matched bagged identity path is
the primary head-to-head baseline. A separate normal TabICLv2 run is retained
for public-estimator parity diagnostics.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_adaptive_retouche.py \
      --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_retouche/multiclass_seed20260828_patience12 \
      --task-id-file /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_adaptive_phase1/multiclass_seed20260826/experiment_manifest.json \
      --device cuda:0

To preserve work already completed by the earlier full-500-step version while
using patience 12 only for unfinished bags/configurations, point to the old
output directory and add::

      --resume --allow-retouche-efficiency-resume --allow-equivalent-hardware-resume
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        adaptive_retouche=True,
        description=__doc__,
    )
