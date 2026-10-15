# Full-data matched affine/spline versus reconstructed Retouche cross

Authorized 2026-10-15: implement and launch the comparison proposed after the
historical fairness audit. One seed zero, the common 33 suite-457 datasets
(28 binary/five multiclass), 261 effective inner folds, official outer split
0/0/0. Complete official test partitions, previously 256,856 rows. No HPO.

## Question and closest evidence

The October 13--14 capped-row spline/cross comparison changed architecture at
the same small budget, but the October 14--15 Retouche reconstruction changed
the whole protocol. The unanswered question is whether restricted affine/K20
spline preprocessing helps when given the cross arm's full fitting pools,
prediction context and training/evaluation rules. This is development evidence
for per-dataset fitting; shared zero-shot transfer still requires a separate
dataset-disjoint experiment. Direct fits are not an upper bound on meta-learning.

## Shared protocol

Use `scripts/retouche_benchmark.py` for both architectures. Reuse the completed
cross fits from `results/retouche_default/v1_seed20261014`, fingerprint
`59eab331e9cc3afbaed45d54b928aad936a10124654e6dc1dff49209ea864503`.
Verify all original runtime dependency hashes except the runner, which has
explicit adapter/reuse hooks. A local equivalence test executes the original
runner from revision 53ff309 and the modified cross path: weights, training and
validation logs, metadata, locks and predictions must agree exactly.

Same frozen classifier checkpoint SHA256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`;
same stratified inner splits (eight, five for anneal); all fold-fitting rows;
same fold-local imputation/ordinal encoding/standardization and [-100,100]
input clamp. Check dataset hashes, row order, fitting/validation/support
positions and preprocessing metadata against the reference before fitting.

Same epoch sampler and task/fold/epoch seeds, at most 1024 context/256 query
rows per episode, full pool visited each epoch with fresh class anchors.
Same full fitting context at validation/test prediction, query chunks 256,
temperature .9, FP32 differentiable frozen backbone and public inference policy.
Each adapted fold uses the same single direct backbone prediction path; native
fallback uses its saved eight-view prediction. Average across the same folds.

Same AdamW base LR .005, gate LR .015, betas (.9,.97), weight decay .003 on
matrix parameters and zero on vectors, coslog4 over 150 maximum epochs,
label smoothing .15, clip two, patience ten, validation every epoch. Separate
early stopping may yield different executed epochs/updates; report both and
runtime, rather than claiming equal actual FLOPs. No architecture-specific
learning-rate search is performed.

Same minimum-validation binary 1-AUC/multiclass NLL checkpoint selection and
0.5% relative-error improvement requirement against the exact same native
fold baseline. Test evaluation happens after every fold of that task is locked.
Previously inspected tests make this a development comparison.

## Spline arm and explicit departures from its earlier use

`src/tabicl/_experiments/matched_spline.py` retains the restricted map's
tanh-bounded affine shift/log-scale and monotone cubic B-spline with 20
controls. Gaps use normalized identity gaps times exp(2*tanh(raw)); the
residual uses the same [-4,4] coordinate range and exact subtractive identity
anchor. There are no neural corrections, cross-feature mixing, or conditioner.

Apply the map to ALL standardized coordinates, including ordinal categorical
coordinates, so both arms receive and can modify the same features. This is
an adaptation of our restricted spline family to the common interface, not
an unchanged replay of the numerical-only historical arm. Both use one map
per fold, rather than the earlier spline's two native-view slots. Missing
values are imputed by the common preprocessor before either adapter.

The spline gate starts at effective .1, retaining its historical effective
initial value; it is directly optimized and clamped to [0,1], following the
common gate policy, rather than parameterized through a sigmoid. Cross gates
retain their original .02 initialization. Spline affine/gap parameters start
at identity. Matrix weight decay acts on gap logits and the projection; vector
parameters are not decayed. These architectural initialization/parameterization
differences are declared, not optimized after inspecting test scores.

Above 500 input features, copy the exact initial projection drawn by the cross
initializer; both arms learn their projection during fitting. The initial
fixed identity/projection control must replay the saved cross control within
atol 1e-6/rtol 1e-5 on each fold and on the averaged prediction. Failure stops
the experiment; it does not silently relax the comparison.

## Reuse, outputs and execution

Fingerprint the reference JSON/CSV/prediction artifacts at startup; changes
invalidate resume. Reuse native validation/test and full-training predictions;
never retrain completed cross models. The spline's final task report includes
`adapted`, `guarded`, `standardized_identity`, `native_bagged`, `native_full`,
`cross_adapted`, and `cross_guarded`. Preserve full prediction arrays.
Report both adapters versus the shared identity/native baselines and directly
against each other, guarded and unguarded, on the identical 33 datasets.
Use primary binary AUC/multiclass NLL plus accuracy/F1 and harm percentiles;
do not average AUC and NLL changes into one arithmetic score.

Save loss, gradients, learning rates, gate, query exposure, actual episode
sizes, per-update elapsed seconds and episode-row hashes; save per-epoch
validation/probe metrics and selected epochs. Preserve atomic resume every
ten updates/epoch/budget exit. Existing checkpoint resume/fallback tests also
cover the spline and its cached-control reuse.

Result root: `results/matched_preprocessing/v1_seed20261015`. Submit uriofir,
one GPU/four CPUs/32G/eight hours, notifications enabled, stop/save at 465
minutes. Queue a second allocation with an after-success dependency to resume
the same revision/settings/root. If already complete, it exits before loading
the model. Additional allocations, if necessary, require fresh exact-command
approval. No separate bank build or GPU runnohup.

Local checks: 15 targeted CPU tests passed in 9.90 seconds, including actual
TabICL gradients/inference, exact original cross-runner parity, monotonicity,
identity/projection, interrupted spline resume and exact cached-control reuse.
CLI import/help and scoped diff reviewed before submission.
