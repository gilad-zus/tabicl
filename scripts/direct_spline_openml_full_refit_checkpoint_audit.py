"""Diagnose full-context DirectSpline learning dynamics on development tasks.

This launcher fits the same all-outer-training-row DirectSpline as the
unconditional full-refit experiment, but freezes a prediction and adapter
state at each requested training step.  Only after the whole checkpoint curve
has been frozen are the published outer-test labels used to score the curve.

It is intentionally a development diagnostic, not a deployment procedure:
the resulting outer-test curve must not choose a checkpoint, regularisation
strength, configuration, or identity guard.  Freeze any resulting policy on a
separate evaluation bank.

Example
-------

    python scripts/direct_spline_openml_full_refit_checkpoint_audit.py \
        --output-dir results/openml_direct_spline/checkpoint_audit_dev \
        --task-id-file results/openml_direct_spline/development_task_ids.json \
        --adapter-steps 500 --checkpoint-steps 0,25,50,100,200,300,500 \
        --device cuda
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        full_context_refit=True,
        checkpoint_audit=True,
        description=__doc__,
    )
