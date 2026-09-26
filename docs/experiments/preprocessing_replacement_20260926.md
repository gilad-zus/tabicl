# Numerical preprocessing replacement pilot — 26 September 2026

Goal: test whether spline curvature is more useful when it learns numerical
reshaping directly than when it corrects TabICL's existing numerical preprocessing.
Closest prior experiments: the legacy lite runner changed context, prediction,
ensemble and spline formulation together; the input-preserving studies kept
ordinary preprocessing. This pilot controls those surrounding choices.

## Model arms and data

| Arm | Numerical representation before adapter | Adapter |
|---|---|---|
| standard_line | Standard `none` / `power` branches including outlier handling | Frozen S(u)=u; train center/span and mixer |
| standard_spline | Same | Train monotone cubic K20 shape, center/span and mixer |
| minimal_line | T-fitted mean/std, without power/outlier transformation | Same line control |
| minimal_spline | Same | Same K20 spline |

The ordinary T-fitted encoder's imputation and categorical encoding are retained.
This means the existing numerical imputation rule is shared, rather than adding
a new median-versus-mean difference. Original branch-specific transformations of
categorical coordinates are retained. Only numerical output columns are replaced.
Both original adapter slots, eight views, feature/class permutations, temperature,
and aggregation remain. The two minimal slots receive identical numerical
coordinates initially and learn separate adapters. Retaining view count does
not retain preprocessing diversity. Per-slot OOF/test results are saved.

Development multiclass tasks: 4602, 75158, 167186, 361539.
Development regression tasks: 4999, 5042, 362096, 362343.
These previously inspected tasks test a mechanism, not final generalization.

## Training, selection and diagnostics

- Four bags; fitting T is about 75% of outer-training rows; held-out A/B halves
  are about 12.5% each. Published outer test remains outside fitting/selection.
- Protocol seed 20260915; training RNG 20260828; 500 steps; validation every 25.
- Same train-only 5–20% query sampler and all remaining T rows as context.
- Input-preserving formula `z+c+(s-4)*u+s*(S(u)-u)` with
  `u=2/pi*atan(pi*z/8)`. It initializes exactly to each route's z. K20 means
  twenty controls / nineteen trainable gap logits, cubic degree, fixed knots.
- AdamW LR .005; gate group 3x LR; weight decay .003 on regular group, zero
  on gate group; gradient clip 2; rank-four mixer bounded by .1.
- Each arm receives constant LR and cosine to 1% of initial LR. Each family
  independently chooses its schedule by pooled raw OOF error; constant wins
  exact ties. Also report both fixed-schedule comparisons. No primary identity
  blend; its selected alpha is secondary diagnostic information.
- Fit only T. Select checkpoints independently on A and B with T-only context,
  evaluate the opposite half, append opposite-half labels for expanded OOF,
  append all A+B for test without refitting. Average the eight selected states
  across four bags.
- Four fixed training episodes at initialization, every validation checkpoint,
  final state and selected states. Save adapters, traces, per-slot predictions,
  original/expanded identity and raw predictions, source provenance and timing.

Explicitly override any inherited source schedule before each constant arm, so
the old cosine source cannot accidentally make both candidates cosine.
Source LR/optimizer/mixing values are checked before execution. Both routes are
run under the same executable revision; older cached pilots have different
diagnostic requirements and are not reused as silently equivalent arms.

## Interpretation

Primary comparisons: minimal spline vs minimal line; standard spline vs standard
line; minimal spline vs standard spline; minimal spline vs ordinary full TabICLv2.
Report OOF and outer-test counts, mean/median relative gains, absolute errors and
the difference in incremental curvature gains between routes. Multiclass uses
log loss; regression reports RMSE and trains MSE.

A larger minimal spline-versus-line gain is insufficient if removing preprocessing
only weakened the minimal line. The spline must remain competitive with the
standard spline and ordinary full TabICLv2. Changing the preprocessing block
jointly removes numerical power/outlier transformations and representation diversity;
this pilot cannot attribute an outcome solely to the power transform. Per-slot
diagnostics help identify whether a result depends on averaging representations.

## Execution

One sequential SLURM job covers both task types and all eight variants per type.
Completed bags resume; a partially completed bag restarts. Equivalent-hardware
resume checks stable GPU/software/precision while allowing a new physical GPU
allocation. Source/code semantic hashes must stay unchanged.

```bash
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
  /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_preprocessing_replacement.py \
  --results-root /home/dsi/zusmang/TabICL/tabicl/results \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_preprocessing_replacement/dev8_seed20260926_v1 \
  --device cuda:0 --resume --allow-equivalent-hardware-resume
```

The source runs are `openml_direct_spline_adaptive_retouche/multiclass_seed20260828`
and `openml_direct_spline_regression_confirmation/full_D_500_v1`.
`--preflight-only` checks required source artifacts without training; the local
`--metadata-only-preflight --preflight-only` variant checks metadata when prediction
arrays were not downloaded. `--summarize-only` rebuilds reports on CPU.

Local validation: broader suite 57 passed / 2 GPU-only skips, plus the added
regression case passed. Complete minimal-route bags on tiny real frozen multiclass
and regression TabICL backbones cover numerical/categorical features, training
gradients, appended-context inference and both slot predictions.
Submission status will be recorded after the approved server operation.
