"""Run the D-only DirectSpline evaluation on the full TabArena-Lite suite.

This is the decisive broad evaluation of the current preserved-fold
DirectSpline recipe. It runs only D, the fixed degree-3, 20-control-point
spline, because D was the strongest arm in the preceding multiclass result.
The adapter trains inside eight deterministic inner bags for at most 500
steps on the cosine schedule, retains the validation-best learned checkpoint,
and applies the 0.5% per-bag identity guard before the outer-test ensemble.

The default task selection is OpenML suite 457 (the 51-task TabArena-Lite
suite), with its published outer split. The normal full-outer-training
TabICLv2 estimator remains the end-to-end reference and matched bagged
identity remains the spline-isolation control. No outer-test label is used
for training, checkpoint selection, or guard selection.
"""

from direct_spline_openml_lite import main


if __name__ == "__main__":
    main(
        default_pipeline="standard",
        required_pipeline="standard",
        adaptive_retouche=True,
        adaptive_retouche_d_only=True,
        description=__doc__,
    )
