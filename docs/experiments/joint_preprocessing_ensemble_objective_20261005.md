# Joint preprocessing: ensemble-objective pilot

Authorized 2026-10-05. **Seed 0 only**, explicitly requested by the user to avoid
spending compute on a second seed. Implementation is complete; **51 distinct
targeted local checks passed**, including the final logging/precision checks.
Submission pending.
Entry point: `scripts/joint_preprocessing_ensemble_objective.py`.

Closest previous experiment is the completed October 4 real-meta continuation,
whose real-only selected models achieve +0.323% geometric gain and 16/14 wins/losses
versus ordinary16 on 30 families, with a confidence interval crossing zero.
Source learning is also modest. This tests the ensemble-loss idea deferred on
September 30: does training on the deployed blend improve useful learning and
unseen-family validation compared with raw single-view training? It does not
change the conditioner, transform family, source coverage or deployment gate.

## Fixed comparison

| Arm | Loss per real episode | Updates | Episodes/update | Seed |
|---|---|---:|---:|---:|
| single | Raw single-view query CE, existing task-deterministic view selection | 1,024 | 4 | 0 |
| ensemble | Query CE of class-aligned mean ordinary8+learned8 logits, temperature 0.9 | 1,024 | 4 | 0 |

The ensemble logits are `(ordinary8 + learned8) / 2`. Each branch first averages
its eight class-aligned raw logits. Ordinary weights/logits are detached;
gradients pass through learned numerical preprocessing and frozen TabICL input
operations. No teacher, per-dataset fitting or new routing/gating is introduced.
Both runs start from the original repeated teacher-free checkpoint at step 5,120:
SHA256 `30f1255c740385894903d7d1311d9017ce648ea9fd4366528649700ed531a206`.
They do not start from the real-continuation selected models.

Keep the joint model, context-only numerical/missingness/label conditioner,
ordinary categorical values, two numerical slots and feature/class permutations.
Disable backbone dropout and freeze all backbone parameters. Use float32 AdamW,
lr 0.0003, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8, gradient clipping
1.0. Reset optimizer at step zero only; average four episode losses equally.

## Data and sizes

Reuse the frozen bank from `hyperspline_joint_real_meta_continuation/v1_seed20261004`:
40 training families, 40 fixed source-probe episodes and 20 validation episodes
from **10 different families**, with two fixed partitions per validation family.
No source downloads, synthetic generation or bank regeneration are needed.
The source probe is a seen-family learning diagnostic. Validation is unseen
relative to training, but previously inspected; this is a development pilot.
Neither real nor synthetic final-test files are deserialized by this runner.

Each arm uses exactly the old real seed-0 schedule for the first 1,024 updates:
four shuffled source families per update, deterministic resampled rows and
matched episode IDs/seeds. Cycle total rows 128/256/512/1,024 and context fractions
0.5/0.7/0.85 in the same 12-shape order. Cap all four episodes to the smallest
available source-table size in that batch. Query size is `ceil(rows*(1-fraction))`.
Nominal context/query sizes (before batch-size caps):

| Total | 50% context | 70% context | 85% context |
|---:|---:|---:|---:|
| 128 | 64 / 64 | 89 / 39 | 108 / 20 |
| 256 | 128 / 128 | 179 / 77 | 217 / 39 |
| 512 | 256 / 256 | 358 / 154 | 435 / 77 |
| 1,024 | 512 / 512 | 716 / 308 | 870 / 154 |

Constrained stratification preserves every class in context and query.
Fit categorical vocabularies/imputation on each context only. Keep the October 5
context-empty-column repair. Query labels enter only loss/scoring.

## Compute, checks and logs

Equal update/task counts **do not mean equal compute**. Single-view training uses
one learned backbone call per episode. The memory-bounded ensemble loss uses
eight ordinary calls, eight detached learned calls, then replays the eight learned
views one at a time to accumulate numerical-input gradients. Finally it performs
one backward through the small shared hypernetwork. This is the exact chain rule
for deterministic forward execution, with one backbone activation graph live.
Local tests compare every parameter gradient with naive full-graph autograd and
check deployment parity with tiny actual TabICL, including grouped features.
The GPU runner repeats deployment-logit/loss/finite-gradient parity on its first
actual source episode and writes `runs/ensemble_seed0/execution_audit.json`.
Training uses float32 and the inherited attention backend. Existing deployment
inference managers enable CUDA AMP; validation preserves those defaults. The
execution audit compares train/inference logits at matching float32 precision
and separately logs the default mixed-precision logit/CE discrepancy. Thus
"exact ensemble objective" specifies views, class alignment, aggregation and
temperature; it does not claim bitwise equality across arithmetic precisions.

Evaluate step zero and every 256 updates, including 1,024. Log both raw single-view
and deployed blend NLL on source and validation, ordinary8/16 references, family
means, geometric gain, median gain, material wins/losses, harm counts and transform
correction/gate/mixing/saturation diagnostics. `training.csv` records objective,
synchronized step seconds, gradients, clipping and LR. `presentations.csv` retains
the paired family IDs, seeds and actual context/query sizes. Per-episode evaluation
rows permit auditing outlier families and near-zero baseline ratios.
`learning.csv` and stdout explicitly compare fixed source/validation blend NLL
with the common start, alongside raw single-view learning and ordinary16 gains.

Select one global checkpoint per objective by the unchanged real-validation
mean family log NLL ratio of blend versus ordinary16, using floor 1e-4 and keeping
step zero eligible. Final step 1,024 is also reported. Save resume state every
25 updates and after evaluation; restore model, AdamW and Python/NumPy/Torch/CUDA
RNG, trimming rows newer than the durable step. Shared ordinary reference caches
avoid repeating baseline inference between arms. Pin bank/source hashes and
line-ending-normalized code hashes. Never resume across a changed protocol.

After both arms finish, `complete.json` locks selected/final checkpoint hashes
and reports paired ensemble versus single results, comparisons with ordinary16
and the common start, gain percentiles, raw NLL, medians, harms and family-bootstrap
intervals. One seed cannot quantify continuation-seed variability. Ten validation
families do not establish broad transfer or thesis certainty.

## Decision and execution

Require useful source learning plus validation gains that are not driven solely
by extreme relative ratios, with acceptable loss tails, before scaling this
recipe. Report both selected and final outcomes and actual compute. If both
objectives remain flat, investigate conditioning/representation rather than
buying more of the same training. Positive development results need a later
frozen confirmation on fresh families.

Run one serial GPU allocation, **uriofir / p_uriofir / ug_uri_ofir**, one GPU,
32G host RAM, four CPUs, at most eight hours, email enabled. Do not reserve a
second allocation speculatively. A timeout can resume the durable current arm.
Run the new ensemble objective first, so its GPU parity/gradient check fails
early if necessary; run the single-view control next in the same allocation.
Result root: `results/hyperspline_joint_ensemble_objective/v1_seed20261005`.
Source root: `results/hyperspline_joint_zero_shot_comparison/v1_seed20261003`.
Bank root: `results/hyperspline_joint_real_meta_continuation/v1_seed20261004`.

Evidence: [decision and learning curves](joint_preprocessing_next_decision_20261005.md),
[previous results](joint_preprocessing_real_meta_continuation_results_20261005.md),
[previous sampling protocol](joint_preprocessing_real_meta_continuation_20261004.md).
