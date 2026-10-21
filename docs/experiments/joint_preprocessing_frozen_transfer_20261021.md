# Frozen control576 breadth evaluation

Authorized October 21, 2026: the user's "Do it" approves evaluating the current
lower-LR checkpoint on the 58 existing sources outside the current 83 training
and ten development-validation groups. This is inference only, one seed-zero
checkpoint, with no target-specific optimization, guard, or checkpoint choice.

Closest precedent: October17 all-catalog inference found approximately neutral
58-source transfer from earlier context-size models. This run changes the frozen
checkpoint to the completed step576 model after the large-context/LR lineage;
it tests whether its current 7/3 validation signal extends to additional sources.
The catalog is historically inspected development data, not a pristine final test.

Parent: `results/hyperspline_lr_response/v1_seed20261019`, fingerprint
`d2a497a5d8eba2b58d42001301691e7dceea7b07e57f7876a3ad3b4b10fe8f43`.
Use only `control.pt`, LR0.0003, absolute576. Verify its tensors match the durable
paired state and that all256 prescribed additional updates completed. Pin all
parent metadata/input/checkpoint file hashes and preserve all50 parent code
dependencies. Model/AdamW training is never invoked by this runner.

Use the existing locked full-table caches and source allocation. Select exactly
the58 evaluation groups, in source-group order, before scoring; reject overlap
or changed allocation, cache hashes, or row counts. No source-bank reconstruction,
new downloads, or performance-based eligibility filtering. Every source is scored
on two deterministic80/20 context/query partitions, with complete query coverage.
Cap only context at20000; process queries in2048-row chunks. The uncapped80%
context can exceed this cap, leaving some context-pool rows unused. Encoding,
imputation and generated maps use only the selected labeled context. Native16
and candidate share identical contexts, queries, labels and view draws.

Reuse the unchanged FP32 predictor from the completed parent, including the
eight unchanged/eight adapted views and temperature0.9. One generated map serves
all query chunks of an episode. Save partial predictions for resume, fingerprint
completed episodes, skip already completed predictions, and verify candidate
weights/gradient state remain unchanged after inference.

Average the two partition scores within each dataset, then weight datasets
equally. Report NLL means/reduction, W/L/T, median and percentiles, material
harms, dataset-bootstrap uncertainty, accuracy, binary AUC and macroF1.
Report exposure strata, retain the historical KDD relation, and repeat the
summary excluding the largest absolute NLL contributor. Local NLL Elo-equivalent
is a descriptive corrected win-score conversion, not a TabArena rating.

Expected workload from the locked allocation:116 episodes and about204396 query
predictions, approximately192 query chunks (class-stratified split counts can
vary slightly). The prior full-query validation took a substantial share of
the completed LR job. Request one uriofir GPU, four CPUs,32G and two hours to
cover this full-query workload; runner budget110 minutes, resumable on timeout.
Mail remains enabled. No separate GPU profiling or CPU preparation job.

Result root: `results/hyperspline_frozen_transfer/v1_seed20261021`.
Implementation: `scripts/joint_preprocessing_frozen_transfer.py` and
`scripts/run_joint_preprocessing_frozen_transfer.sh`.

Local validation: seven new checks pass, including actual tiny-TabICL inference
with complete queries and frozen weights, exact resumed/uninterrupted scores,
completed-result skipping, checkpoint byte/tensor tampering, source exclusion,
history labels and equal-dataset aggregation.21 existing fraction-context checks
also passed. An initial inherited-settings error (zero metadata step budget)
was fixed before the seven final checks passed. No remote operation yet.
