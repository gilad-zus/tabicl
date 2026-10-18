# Large-context continuation with sampled training queries

Authorized October 18, 2026. One seed-zero continuation of the capped run.
Its first allocation completed only 46/2048 updates in 7h30m. Repeated query
chunks consumed 80.74% of measured training-chunk time. This revision changes
the amount of supervision at each visit, while preserving the large contexts.

## Protocol

- Reuse the same 83 training, 10 validation and 58 outside datasets, raw-cache
  hashes, original fitting pools and 256 reserved training-probe rows.
- Each training visit first draws the existing 80/20 context/query partition.
  Preserve exactly the same context draw, capped at 20000 with class support.
  Uniformly sample without replacement up to 2048 rows from that visit's full
  query pool. Redraw on subsequent visits. Small query pools remain complete.
  Unselected query rows do not become context rows on that visit.
- One training query chunk per source visit; four equally weighted source
  means per AdamW update. Uniform query sampling estimates the full pool's
  mean loss conditional on the context and query pool. It increases stochastic
  gradient noise; it is not exact all-query gradient accumulation.
- Same compact 5912-parameter hypernetwork, frozen TabICL, FP32, native16
  ensemble, activation recomputation, per-view backward replay, LR0.0003,
  weight decay0.0001, gradient clip1, seed0. Total target remains 2048 completed
  updates, including the 46 inherited updates, rather than 2048 additional ones.
- Import the audited completed-step46 model and AdamW moments. The parent's
  unfinished step47 has one accumulated source gradient but no weight update.
  Discard those unfinished gradients/loss counters and restart all four source
  visits of step47 under the new policy. Preserve historical exposure coverage;
  new chunk/update logs start separately and identify the parent run.
- Validate imported step46 before any new optimizer update. Then validate at
  absolute steps64,96,...,2048 (every32), retaining exact identity as eligible.
  All 20% query rows and all three fixed validation partitions remain intact.
  Select one shared checkpoint by mean dataset NLL; no per-target fitting.
- Final inference retains all query rows, the same two fixed partitions and
  previous checkpoints/native comparisons. The 58 outside sources have historical
  research exposure and are not a pristine test bank.

## Continuation safety and execution

The parent manifest fingerprint is
`8e406f6e46e0550b01c089b846753a3570f40dbabb467d28a48d34a82760cf10`.
Its exact imported state SHA256 is
`930fa1981520f689e665242ac7fc278bdf223536ad7d985bbf2c92af16c21b1e`.
Only query sampling, validation cadence, runner/wrapper code and continuation
metadata may change. Other settings and model/dependency hashes must match.
The parent's saved state is read-only and stays available for the original recipe.

Reuse the passed parent GPU shape check: context cap20000 and execution chunk2048
are unchanged. No new profiling allocation or full-source-bank rebuild is needed.
Preparation only creates a new manifest/allocation using the locked bank. Resume
supports partial validation, sampled episodes and allocation boundaries.

Submit one uriofir allocation on dsiuriofir01, 1GPU/4CPUs/32G/8h, email enabled.
The pipeline validates step46 first, then trains with remaining wall time; no
automatic chain of future allocations. Result root:
`results/hyperspline_fraction_context/v3_cap20000_query2048_seed20261018`.

Timings suggest roughly 4-5x faster training, not a measured speedup for this
new run. Full validation cost remains; no useful-learning or transfer claim yet.

Implementation: `scripts/joint_preprocessing_fraction_context.py`.
Wrapper: `scripts/run_joint_preprocessing_fraction_context.sh`.
Checks: `tests/test_joint_preprocessing_fraction_context.py`.
All 17 targeted CPU checks passed locally, including uniform-subset membership,
unchanged contexts, complete evaluation queries, small-pool behavior, gradient
equivalence, validation-before-update continuation, AdamW preservation, sampled
episode resume, dependency-change rejection and reuse of the passed GPU profile.
Predecessor/evidence: [capped progress](joint_preprocessing_fraction_context_cap20000_progress_20261018.md).
Launch ID/revision/logs will be recorded after submission.
