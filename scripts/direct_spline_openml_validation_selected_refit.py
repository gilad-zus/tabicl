"""Evaluate validation-selected DirectSpline checkpoints and full-data refits.

For every OpenML task, this experiment makes one deterministic split within
the published outer-training rows.  It trains DirectSpline for the complete
fixed cosine horizon, chooses the best inner-validation checkpoint (including
step-0 identity), then freezes two test predictions:

1. the selected checkpoint fitted on the inner-training subset; and
2. a fresh fit on every outer-training row for the selected duration, using
   the same cosine learning-rate prefix.

It runs two fixed arms with the same split and adapter seed: cosine scheduling
alone, and the same schedule plus an explicit function-space identity penalty.
Outer-test labels are used only after all four prediction pairs are frozen.

Example
-------

    /home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
      scripts/direct_spline_openml_validation_selected_refit.py \
      --output-dir results/openml_direct_spline/validation_selected_refit_seed20260826 \
      --task-id-file results/openml_direct_spline/evaluation_task_ids.json \
      --adapter-steps 500 --split-seed 20260826 --adapter-seed 20260826 --device cuda
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        validation_selected_refit=True,
        description=__doc__,
    )
