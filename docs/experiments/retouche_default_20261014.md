# Retouche default reconstruction, seed zero

Authorized 2026-10-14: implement Retouche and launch an experiment. The closest
previous run is the October 13--14 numerical-only rank-four cross diagnostic,
whose selected test NLL was +0.102% aggregate gain (14/15/4) and binary AUC was
18/8/2 with +0.052 percentage points mean gain. It was a simplified custom arm.
The unanswered question is whether the paper's fuller adaptation recipe yields
useful gains under our frozen TabICLv2. This remains per-target adaptation;
the eventual shared zero-shot generator is a separate experiment.

## Source and scope

Paper: https://arxiv.org/html/2605.06047v1, sections 3.2--3.5, 4.1 and appendices
G/H. Official code was not verified in the current search; the paper promises
a release. This implementation is a documented paper reconstruction and cannot
be described as an exact reproduction of unpublished code.

Run all 38 classification tasks from suite 457, official repeat/fold/sample
0/0/0, including the five categorical-only tasks previously excluded. Regression
is deferred: this initial run uses the already verified classifier checkpoint.
Use one seed zero and one default configuration, no HPO. Eight stratified inner
folds; reduce only where the rarest class cannot support eight folds, recording
the effective count (for example anneal). Every fold uses all its fitting rows;
there is no 600-row fitting-pool cap. Outer test rows never participate in
preprocessing fitting, gradient updates, checkpoint selection or guarding.

## Adapter and defaults

Two DCNv2 cross layers, rank max(1,floor(0.25 * original feature count)), no inner
activation, BatchNorm, per-feature gate initialized to 0.02. Median imputation
and standard scaling for numerical columns; fold-local ordinal encoding and
standard scaling for categorical columns. All coordinates are adapted. Inputs
above 500 features get a learned orthogonally initialized projection to 500
features after the adapter, as described in appendix G. Backbone weights frozen;
stochastic backbone layers disabled.

AdamW base LR 0.005, gate LR 0.015, beta2 0.97, matrix-only weight decay 0.003,
zero weight decay for biases/BatchNorm/gate, gradient clipping 2, CE label
smoothing 0.15. Maximum 150 epochs, validation each epoch, patience 10; preserve
the minimum deployment-error checkpoint. Binary selection uses 1-AUC and
multiclass uses log loss. Guard requires 0.5% relative validation-error improvement
over the native raw-input TabICLv2 model fitted on that same fold's fitting rows.
Otherwise deploy its exact native prediction.

The coslog4 multiplier follows the primary pytabkit definition:
https://raw.githubusercontent.com/dholzmueller/pytabkit/main/pytabkit/models/training/scheduling.py
Use four cycles, progress measured across the declared full 150-epoch budget,
including within-epoch progress. Early stopping can shorten the schedule.

## Explicit reconstruction choices and limits

The paper does not specify small-normal scale, BatchNorm placement, gate
constraint, or concrete batching/reshuffling implementation. We use normal
standard deviation 0.01 for both factors, BatchNorm on each correction before
its residual addition, and a directly optimized gate clamped to [0,1] after
updates. These choices may matter; a failure would not invalidate the paper.

Each epoch permutes the complete fitting pool into mini-tables with at most
1,024 context and 256 query rows. One newly sampled context anchor per class
is reused within that epoch to maintain class coverage. Every other fitting
row appears once as context or query; anchors and roles change every epoch.
Queries never appear in their own context. This makes all fitting rows available
without differentiating through an entire large dataset at once. It is not
equivalent to the former one-sampled-episode-per-update capped bank.

Validation/test use the complete fold-fitting context, fixed across chunks,
with no context expansion or refit after selection. Query chunks have 256 rows.
Training uses FP32 with activation checkpointing; evaluation uses the public
TabICL precision policy. A fixed standardized-input guard clips to [-100,100],
carried over from the earlier demonstrated AMP-overflow problem. The decoder
uses the native 0.9 classification temperature. These computational/encoding
choices are explicitly part of this reconstruction.

## Controls, metrics and logs

For each fold retain native raw-input TabICLv2 predictions (eight native views),
selected unguarded adapter predictions, and guarded predictions. Also retain
the standardized-input identity path with its initial fixed orthogonal
projection where needed. The latter separates the learned correction from
the representation change; it is not the guard's raw native baseline.

After all folds of a task are locked, average each path across its folds and
score the complete official outer test. Include a native raw-input estimator
fitted to the complete outer-training partition as the end-to-end reference.
Early task results are descriptive; they cannot alter the protocol for later
tasks. Since this default has no HPO, there is no per-task configuration search.

Save labels/probabilities to allow NLL, accuracy, balanced accuracy, macro F1,
binary AUC, average precision and Brier score. Primary benchmark deployment
metrics remain binary 1-AUC/multiclass NLL. Compare guarded and unguarded against
both the native bagged and full-training references; avoid pooling AUC and NLL
as one arithmetic score. Save per-update loss, LR, gate LR, gradient norm, gate
mean, actual context/query sizes and query coverage; per-epoch external metrics,
fixed fitting-probe NLL/accuracy, best epoch and stale-validation count.

Atomic state every ten updates and every epoch/clean budget exit; deterministic
sampler plus logged-position truncation make resume reproduce uninterrupted
training. Check actual classifier gradients on eight- and 500-feature synthetic
tables inside the GPU allocation before expensive fits. Lock selected model
hashes and guard choices before outer-test scoring. Pin revision and relevant
classification dependency hashes. Regression-only quantile helper changes do
not participate in this classification experiment.

## Execution

Runner: `scripts/retouche_benchmark.py`; model: `src/tabicl/_experiments/retouche.py`.
Output: `results/retouche_default/v1_seed20261014`. Initial allocation: uriofir,
one GPU/four CPUs/32G/eight hours, email enabled, serial tasks/folds. The runner
stops and saves state after 465 minutes, reserving time for clean exit. The
complete panel may require subsequent allocations with the same revision/root;
do not mistake a budget exit for complete benchmark results. No runnohup GPU
work and no separate bank-building job. Existing OpenML caches supply the raw
tasks through the normal loader.
