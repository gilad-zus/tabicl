# Lower learning rate versus multiplicative preprocessing

Authorized 2026-10-13: one comparison with two new conditions, seed zero only.
Reuse the completed full and restricted benchmarks as controls. The goal is to
decide whether gentler optimization or a different transformation can produce
useful held-out improvements before another shared zero-shot generator run.

## Conditions and controls

- `spline_low_lr`: the unchanged restricted affine/K20 monotone cubic spline,
  with constant AdamW learning rate 0.0003 instead of 0.001.
- `cross_low_lr`: the same affine correction followed by one rank-four
  multiplicative cross layer, also at constant learning rate 0.0003. It contains
  no spline, per-column neural correction, additional linear mixer or attention.

The existing restricted constant-0.001 selected/final scores answer the LR
comparison. The new cross-versus-spline comparison has matched LR and data,
although capacities and parameterizations differ. Ordinary16 remains the main
reference. Completed full-map scores are retained as contextual controls.
Do not choose a per-dataset condition using its test score.

## Frozen data and protocol

Use the exact 33 accepted datasets (28 binary/five multiclass), original first
official outer split, encoded bank bytes, exclusions and deterministic resampled
row assignments from `hyperspline_joint_benchmark_sampling/v1_seed20261011`.
No new data-bank preparation or download is required. Full source fingerprint:
`60e0745f4221d102fc84e563b99f8cb5b2866afd446a1536a7686edf402ae04f`;
complete-report SHA256:
`49f293b701f631aa55e5bbdb56a05863422dc400db140528f76cf664dff304db`.

Restricted reference: `hyperspline_joint_benchmark_restricted/v1_seed20261013`;
fingerprint `a118acfde02d0f05476cf6917091364c5b2cfd157403bad1f1f07e5441dfa11d`;
complete-report SHA256:
`0732f859bef2a497fcdfc6bb93558506987a74637fe6277f0e31499f68bb8799`.
Verify both manifests/reports, original bank hashes and selected/final map locks.
Historical test-score contents are read only after every new fit is locked.

At the cap: 600 fitting rows, 150 external-validation rows, and resampled
450-context/150-query training episodes. Smaller datasets keep their original
counts. Validate/test with the same full fitting context; score all 256,856
official test rows in chunks of 256. Encoding uses fitting rows only; native
preprocessing is fitted on each episode's context. Keep the exact sixteen native
views, eight unchanged/eight adapted, two map slots and unchanged categories.

Both conditions run 250 updates, AdamW weight decay 0.0001 on all parameters,
gradient clip 1, temperature 0.9, and the existing FP32 two-pass input gradient.
Frozen classifier checkpoint SHA256:
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
Evaluate at zero, one, every 25 updates and final; select minimum external
validation NLL with initial identity eligible. Keep fixed fitting probes,
query-exposure records, optimizer resume and immutable selected/final maps.

## Cross-layer specification

For native numerical values, compute the existing bounded affine correction.
Calling its output `affine_values`, the new numerical values are:

`affine_values + gate * affine_values * ((affine_values @ input_factors) @ output_factors.T + cross_bias)`.

This formula describes `src/tabicl/_hyperspline/cross_preprocessing.py`, not a
claim of faithful Retouche reproduction. Each slot has factors with effective
rank `min(4, original numerical column count)`, a per-column bias and sigmoid
gate. Initialize the input factors with CPU seed zero, normal standard deviation
`1/sqrt(original numerical columns)`; output factors/bias/affine outputs are zero
and gates start at 0.02. A zero output factor gives exact native16 identity while
the nonzero input factor permits cross-output gradients at the first update.
Later updates can train both factors and the gate. Use the same LR/weight decay
as other parameters; Retouche's gate LR multiplier and full tuning/guard/bagging
pipeline are outside this bounded comparison.

Trainable scalar count is `2 * columns * (4 + 2 * effective_rank)`: 24 per
numerical column when rank four is available, versus 44 for the restricted
spline. When an inner context drops constant columns, gather both factors and
affine/gate parameters by their original numerical-column positions. Each
missing output cell retains its native imputed value; finite imputed input cells
can participate in cross combinations, as with the earlier mixing branch.
The cross layer is not constrained to be monotone. Inputs and diagnostics must
remain finite; logits/gradient checks fail the run on numerical errors.

Training CSV additionally records constant LR and separate affine/non-affine
gradient norms. Validation CSV records per-slot correction RMS on a fixed
fitting episode; cross logs also record both factor norms and gate mean. These
measure whether the new branch learns and how strongly it changes inputs.

## Reporting and limits

Lock all 66 selected/final new fits before loading test banks. Recompute
ordinary16, require NLL/row-count parity with both historical reports, then reuse
their scores. Report each new condition against ordinary16, lower-LR spline
against original-LR spline, and cross against spline at matched LR. Include
selected/final NLL, W/L/ties, mean/median gains, percentiles, bootstrap intervals,
harms, selected steps and binary/multiclass breakdowns during analysis.

Retain the practical development gate: aggregate mean NLL gain at least 0.5%,
positive median and at least 24/33 wins, with harm/uncertainty review before
another shared-model run. These already inspected datasets cannot supply fresh
confirmation. Fixed steps with lower LR reduce optimization travel; failure does
not establish convergence or rule out more fitting data. Direct fitting is not
a meta-learning upper bound. Positive results identify a candidate for a shared
generator tested on different datasets without target fitting/configuration HPO.

Closest work: the completed full/restricted October 11--13 benchmark, September
24 schedule comparison, and earlier cross-column mixing studies. What changes
is smaller LR for the exact current spline, and explicit multiplicative feature
interactions for the new family. Retouche inspiration:
https://arxiv.org/html/2605.06047v1, sections 3.3--3.5.

## Execution

Use the existing `scripts/joint_preprocessing_benchmark_sampling.py` runner,
`--transformation comparison --lr 0.0003`, both source directories and the new
root `results/hyperspline_joint_benchmark_cross_comparison/v1_seed20261013`.
Submit a serial two-element SLURM array (`0-1%1`), mapping zero to
`spline_low_lr` and one to `cross_low_lr`, through `submit_gpu`. Each array task
receives one GPU/four CPUs/32G/eight hours, uriofir/dsiuriofir01, email enabled.
At most one array task runs at a time. Each task resumes its named condition;
whichever completes the last required fit generates the combined test report.
If an allocation expires, resume that condition/root/revision rather than
restarting completed fitting. The runner prepares only references to frozen
banks inside the GPU job; no separate CPU preparation job is launched.

Local validation completed: 43 targeted CPU checks passed in 214.48 seconds,
covering identity, cross interactions/column alignment, first and later cross
gradients, full-autograd/two-pass parity, missing values, both condition resumes,
fixed row pairing, component/transform logs, immutable source reuse, historical
report protection and all-map test locking, plus existing benchmark/residual/
statistics regressions. CLI help and the scoped diff were reviewed. The dependency
inventory contains 49 paths; the only changed tracked dependency is this runner,
with the cross module added as a new pinned dependency.
