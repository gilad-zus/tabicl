# Full 80/20 row episodes for shared zero-shot preprocessing

Authorized October 18, 2026. One seed-zero shared model. Implement locally,
validate correctness, and measure actual uncapped GPU resource use before
committing the full training budget. This replaces the proposed 512-context
LR continuations. No target-specific fitting or teacher construction.

## Research question and predecessors

Does giving both the hypernetwork and TabICL a substantial fraction of each
dataset, with prediction supervision on every remaining query row, improve
unseen-source performance? This changes context and query exposure jointly;
it is a fuller-data recipe comparison, not a one-factor causal attribution.

Closest shared predecessor: October 16--17 compact statistics spline model,
83 real training sources, 512/2048 contexts and 256 queries. Broad October 17
inference was near neutral outside training/validation. Training sampled full
pools but did not expose the complete pool in a single episode. Earlier
variable-size shared training used small episodes and different model heads.
The October 15 matched direct spline/Retouche study used full deployment
contexts but capped training episodes and independently fitted maps; it did
not answer this shared-learning question.

## Data, rows and labels

Reuse the exact full-table 83 train / 10 validation allocation and permanently
reserved 256 training-source probe rows from
`hyperspline_context_size/v1_seed20261016`, fingerprint
`f28b919a88dac747f4a4c998edf9ce996cd374bf0c8b307ff01af6fd4ea2c427`.
Keep the 58 catalog sources outside that allocation as final evaluation,
from `hyperspline_catalog_inference/v1_seed20261017`, fingerprint
`4aae70bd8544a962a8c83448b5b00ea6779632b575a2bb9d027c7d0c781a8405`.
These 58 have historical research exposure; they are not a pristine final
test. Preserve the KDD2009 historical-relation annotation and source labels.

Each training visit partitions ALL fitting rows into floor(0.8*N) labelled
context and the remaining query rows. Context and query are disjoint; their
union exhausts the fitting pool. Reserve class support in context. New seeded
partitions on later visits change row roles. Permanently reserved probe rows
never enter either training role. Query labels supervise loss but never enter
native preprocessing, the hypernetwork or TabICL's prediction inputs.

Validation uses three predetermined 80/20 partitions per source. Final
evaluation uses two. All query rows are scored, with no 1024-row cap.
Training-source final diagnostics keep their original 256 held-out queries
and use 80% of the fitting rows as labelled context; label these separately.
Context-only raw encoding/imputation and native16 preprocessing are fitted
once per episode. No context/query cap, dataset substitution or automatic
resource-based exclusion is allowed.

## Model, optimization and exact query chunking

Fresh seed-zero StatisticsSplineResidualPreprocessor, 5912 parameters:
65-input statistics MLP, affine/K20 monotone spline heads and two native-view
slots. Eight native16 views remain unchanged, eight receive residual numerical
corrections. Initialization is exact native16 identity. TabICL is frozen.

2048 target optimizer updates, four equally weighted dataset episodes/update.
Sources use balanced shuffled schedules. AdamW LR0.0003, weight decay0.0001,
betas0.9/0.999, eps1e-8, clip norm1, temperature0.9. No label smoothing,
schedule/HPO, teacher or synthetic mix. Full-query mean CE is averaged within
episode, then across four episodes, independently of dataset row count.

Default query chunk2048 is an execution size, not a sampling cap. The complete
context and one generated map are reused across every chunk. Each chunk's
gradient has weight chunk_rows/episode_queries/4. No optimizer update occurs
between chunks or between the four source episodes. Native preprocessing is
fitted once, with all query features retained on CPU until needed.

Use one adapted-view backward graph at a time, FP32 with existing encoder
activation recomputation. Gradients pass through detached generated-map
outputs and then the shared generator by the exact chain rule. CPU tests
compare against the original full-query gradient. FP32 train-mode forward
also supplies evaluation predictions, with frozen weights and zero dropout;
all compared models/native receive the same execution policy and rows.

Durable state saves accumulated parameter gradients, optimizer, completed
update/source position, query cursor, partial weighted losses and boolean row
coverage. Resume reconstructs the deterministic episode/map at unchanged
weights. Evaluation also persists query predictions/cursors. Both can span
8-hour allocations without discarding a completed episode's query work.

## Validation, logging and final inference

Evaluate every256 completed optimizer updates and at0/final. Average query
loss within partition, three partitions within dataset, then equal-weight
the ten datasets. Select ONE shared checkpoint by minimum mean NLL, with
identity eligible. Validation query labels cause no gradients. Final inference
requires the completed training lock and reports final, selected and all four
old final/selected checkpoints, with exact old step-zero identity reuse.

Record full episode CE/update, first-query-chunk native/candidate paired NLL
per episode, finite gradients by model group, affine/spline magnitudes, actual
sizes, all query cursors, row hashes, unique role coverage and timings.
First-chunk native diagnostics are explicitly a subset; every training query
still contributes to optimization. Final metrics include NLL, accuracy, AUC
where applicable and macroF1; summaries average partitions within datasets
before equal-weight comparisons, with W/L/T, gain percentiles and harms.

The confirmed primary evaluation uses the full80/20 recipe. Earlier proposed
50%/100% secondary context sweeps are deferred until resource profiling;
never describe an unexecuted sweep as a completed result.

## Resource preflight and launch

Actual contexts are approximately12K at median training-source size, up to
119,506 in training and159,618 in validation. Profile the median training
table, longest actual context and maximum context-row-by-column count,
deduplicating coincident sources. Each profile computes native16 ensemble
loss and a complete adapted backward on up to2048 actual query rows, using
the entire uncapped context and perturbed nonidentity heads. Report peak GPU
allocation, preparation time, chunk time, projected all-query episode time
and finite nonzero generator gradients. Do not update saved initial weights.

Profile first in a SLURM allocation: uriofir, dsiuriofir01, one GPU,4 CPUs,
32G host RAM, one hour initially, email enabled. A failed memory/runtime profile does
not justify silently changing row fractions or source membership. Review
resource measurements before launching the full training budget. Training
has explicit resume support; do not queue an unbounded continuation chain.

Result root:
`results/hyperspline_fraction_context/v1_seed20261018`.
CPU preparation helper log: `jp-fraction-bank-261018.log` in the remote repo.
Runner: `scripts/joint_preprocessing_fraction_context.py`.
Wrapper: `scripts/run_joint_preprocessing_fraction_context.sh`.
Local checks: all10 new tests passed after final functional edits. Twenty existing
context/catalog tests also passed. Coverage includes actual tiny-backbone full
versus chunked gradients with recomputation, full-query prediction parity,
one-map generation/query independence, complete 80/20 row use, equal-dataset
aggregation, raw-cache manifest replay, and interrupted/resumed gradients,
weights, optimizer, row coverage and evaluation queries. The actual resource
profile path also passed with a tiny CPU backbone without changing initial
weights. No remote operation yet.
