# Saved-checkpoint column-curvature audit

Agreed 2026-09-20. Inference only, development diagnostics, no retraining.

## Question and closest prior work

Does a fitted staged-replacement spline contain helpful curvature in some columns
and harmful curvature in others? Earlier adaptive/sparse columns and the K4/K20
factorial explored related ideas. This intervention instead removes **only one
column's curvature from the current saved K20 model**, preserving the other curves
and its mixer. It does not infer that a retrained low-capacity model would do better.

## Data and frozen model

Use all eight existing development tasks: multiclass 4602, 75158, 167186, 361539;
regression 4999, 5042, 362096, 362343. No new TabArena evaluation and no test-selected
dataset subset. Sources are the `continued_spline` artifacts under
`openml_direct_spline_staged_curvature_ablation/dev8_seed20260915/{type}`.
Both selected A/B checkpoints in each of four bags are reused. K12 is a separate
experiment, not a prerequisite for this diagnostic and not modified by it.

## Intervention

For each normalization branch and column j, compute the saved adapter's arctan
coordinate u and unmixed output s(u) on its fitting rows T. Fit the least-squares
line `l_j(u) = a_j*u + b_j` using those coordinates/outputs only, without labels.
For each j separately, replace `s_j` by `l_j`, keep every other column unchanged,
then apply the original mixer. This preserves the empirical mean and best linear
component on T, not the spline's endpoint values or exact variance. The original
center/span/mixer parameters are not updated. Already-straight columns are exact
no-ops. A column removed by a bag's constant-feature filter is a no-op in that bag.
Original input-column positions identify interventions across bags and encodings.

Include unchanged spline and all-columns projected-line diagnostics. The latter
is not the separately trained continued-line control. Report that cached control
and ordinary TabICLv2 separately.

## Validation/test separation

- A-selected checkpoint predicts B with context T+A; B-selected predicts A with T+B.
- Every outer-test prediction uses T+A+B, with frozen T-fitted preprocessing.
- Each intervention produces complete cross-fitted OOF predictions and an average
  of all eight selected-state test predictions.
- Report each column's OOF and test error difference and relative difference.
  Positive removal effect means removing curvature helped, not that curvature helped.
- Choose at most **one column or unchanged spline** by minimum pooled OOF error.
  Ties favor unchanged. All-columns projection is excluded from this choice.
- Persist that decision before scoring any test labels. Test effects are diagnostic,
  never a mask-selection signal. OOF is the selection score, not an unbiased
  post-selection performance estimate.
- No joint mask or combinatorial search. Single-column effects need not add up;
  fitted curves and the mixer can be co-adapted. Columns are not independent datasets.

## Integrity and cost

Verify source split, adapter and backbone provenance. Reconstruct T preprocessing,
then replay each unmodified checkpoint's saved expanded OOF and test predictions.
Fail on mismatches (default absolute/relative tolerance 2e-5 each); record maximum
differences. Do not silently loosen tolerances or count mismatched results.

Freeze every learned parameter; no optimizer. Use the original ensemble, row/context
counts and precision path. Work scales linearly in numerical columns: for d columns,
`8 * (d+2) * 2` inference calls per task (OOF and test), not new training runs.
This is GPU inference, not a CPU-only analysis of existing predictions.

Atomic per-state/per-column prediction caches support fine-grained resume. Their
fingerprint binds inference code and source artifacts. Both `--resume` and
`--allow-equivalent-hardware-resume` are supported; the latter permits allocation
changes only when GPU/software/precision properties still match.

## Command and outputs

```bash
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
  /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_column_curvature_audit.py \
  --results-root /home/dsi/zusmang/TabICL/tabicl/results \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_column_curvature_audit/dev8_20260920 \
  --device cuda:0 --resume --allow-equivalent-hardware-resume
```

The one command processes multiclass then regression. Each type writes
`column_curvature_summary.json`, updated after each complete task. Under `raw/`,
each task retains its full per-column table, OOF selection, projection coefficients,
replay differences and resumable predictions. No backbone copies are written.

Decision: mixed helpful/harmful curvature supports a subsequent selective-curvature
training test; predominantly harmful removals favor retaining current curves;
OOF/test disagreement points to selection transfer. None alone establishes a final
deployable method or confirms the thesis on untouched datasets.
