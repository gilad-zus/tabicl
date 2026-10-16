# Paired context-size experiment, October 16

Authorized October 16: two seed-zero shared spline hypernetworks, trained on
real datasets with 512 versus 2,048 context rows and exactly 256 query rows.
Each receives 2,048 AdamW updates with four source episodes per update.
No teacher construction or target-specific optimization/validation fallback.

## Question and nearest predecessors

The October 10 statistics-only experiment used a small attention-free
conditioner but retained affine/spline/neural/mixing heads, and variable
128--1,024-total-row episodes. It gave tiny development gains. Restricted K20
hypernetworks were already tested on synthetic tasks; the October 15 full-data
direct-fit spline/cross comparison supports learning within their standardized
path but does not show broad native improvement. That comparison changed many
settings and cannot identify row effects.

This experiment isolates context size **between its two paired models**. The
new compact output family and full-pool sampling are common to both arms.
Comparisons with older runs cannot isolate simplification or pool coverage.
Context size changes information available to both the conditioner and TabICL;
this experiment does not separately attribute these two effects.

## Data and pairing

Retain the original 160 training / 25 development-validation source-group
assignment from `hyperspline_joint_dataset_diversity/v4_seed20261007`, fingerprint
`7149269ed57b83ce024a07934f293bae6b69b80b0bd41bba1686c997a9f149f1`.
Restore full cached tables, replaying their original complete-input content
fingerprints and locking raw-cache hashes. No downloads, new allocation,
performance-based source selection, or silent fallback to 16,384-row banks.
Full caches may reside in the original v1/v2 preparation directories.

Training sources need at least 2,560 original rows: reserve 256 fixed probe
queries and retain at least 2,304 fitting rows. Each class needs at least three
rows to support this reservation and later context/query separation. Validation
sources need at least 2,304 rows. Keep the original 5--100 encoded-feature
interface with at least one numerical feature. Exclusions are recorded before
training. Metadata predicts 83 training / 10 validation sources before class
and preprocessing checks; actual counts must be read from eligibility.json.

Every step schedules the same four sources in both arms. A deterministic seed
selects the same 256 fitting queries, then a label-complete 2,048-row context
disjoint from them; the 512-row context is a label-complete subset. Sampling
spans the complete fitting pool, without a row cap. Fixed probe query rows never
enter shared training. Sampling does not guarantee exhaustive coverage; log
unique context/query/combined row coverage and every episode's row-ID hashes.
Context-only encoding/imputation is fitted separately for each actual context.
Thus numerical/categorical estimates may change as a consequence of context size.

Choose 16 source-probe families by a seeded ordering, before observing scores.
Evaluate these held-out training-source queries and every eligible independent
validation source at both context sizes, with nested contexts and common queries.
The source probes diagnose fresh-row learning on known datasets; only validation
sources measure unseen-dataset transfer. Previously inspected validation remains
development evidence, requiring fresh dataset-level confirmation after success.

## Model, objective and optimizer

Use StatisticsSplineResidualPreprocessor (5,912 trainable parameters): the existing 31 label-aware column
summaries, their table mean, and three size summaries feed the shared
LayerNorm(65) / Linear(65,32) / GELU / Linear(32,64) / GELU conditioner.
Two slot embeddings and shared heads generate bounded affine shifts/log-scales,
19 positive normalized gaps for a K20 monotone cubic spline, and a sigmoid
spline gate initialized to 0.1. No attention, neural correction, learned
projection, or cross-feature mixing is included in the hypernetwork.

Use the existing native16 residual path: eight native none/power views remain
unchanged, eight receive the corrections. Native feature/category handling is
preserved; corrections at originally missing numerical cells are masked.
Initialization is exactly the matched native16 prediction. The frozen backbone
retains its original architecture. This native16 control uses the established
aligned-logit ensemble; no new claim of sklearn-estimator probability parity.

Optimize CE of the mean aligned native16 logits divided by temperature 0.9.
Staged input gradients reproduce the full ensemble gradient while keeping one
backbone graph live. AdamW LR 0.0003, weight decay 0.0001, betas 0.9/0.999,
epsilon 1e-8, gradient clipping 1. Equal updates/query counts across arms;
large contexts cost more GPU compute. No HPO or synthetic mixing.

## Measurements and decisions

Evaluate at zero and every 256 updates, also at the final update. Both arms use
the same evaluation sources/queries at both context sizes; native16 receives
identical rows. Select a secondary shared checkpoint by minimum mean dataset
validation NLL at 2,048 context, with update zero eligible. Primary comparison
is final versus final at the same 2,048-row evaluation context.

Log per-step CE, group/encoder gradients and clipping; actual episode sizes,
query hashes, row coverage; per-dataset NLL/accuracy/binary AUC/macro F1,
correction RMS and spline gates. Summaries include wins/losses/ties, gain
percentiles, >1%/>5% harms, and a descriptive paired dataset-bootstrap interval.
The continuation gate is >=0.5% mean NLL reduction versus native, >=70% wins,
positive median and an adjacent passing checkpoint. With ten validation
datasets this is preliminary evidence, not a thesis confirmation.

Large-context source and validation benefits justify scaling row budgets.
Source-only benefit points to transfer; neither benefiting weakens this recipe.
Do not interpret failing this bounded run as disproving all preprocessing.

## Execution and reproducibility

Local meaningful tests cover exact identity, staged/full gradient agreement,
nested actual-size sampling, fixed-query leakage prevention, context-only
encoding, complete-source loading/hash failures, preparation/allocation locks,
and uninterrupted/resumed state equivalence. GPU preflight checks both actual
context sizes against the real frozen checkpoint before training.

Prepare CPU inputs with runnohup on dsiofir01; no model training there. GPU
pipeline uses uriofir / dsiuriofir01, one GPU, four CPUs, 32G and at most eight
hours per allocation, email enabled. Clean stop after 450 minutes and resume
model, optimizer, RNG and row-coverage state. It runs small then large serially.
If a continuation is queued, set/verify its explicit SLURM afterok dependency.

Result root: `results/hyperspline_context_size/v1_seed20261016`.
CPU preparation log: `jp-context-bank-261016.log` in the remote repository.
Status: implementation complete. Thirty-four targeted CPU checks passed
(133.46 seconds); a final pairing/report/resume check also passed after the
completion audit was added (34.37 seconds). All 47 code dependencies were
reviewed; changes are confined to the declared model/runner/wrapper. Remote
preparation/submission remains pending.
