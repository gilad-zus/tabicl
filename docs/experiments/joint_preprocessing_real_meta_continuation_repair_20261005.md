# Real-meta continuation: missing-column crash repair

Diagnosed and repaired locally on **2026-10-05**. Original array 32073418 failed:
task 0 after 04:41:51, task 1 after 00:01:14, both exit 1:0. Scheduler and logs
were checked at 07:55:29 IDT. This was an indexing exception, not a resource limit.

Both synthetic continuations completed 4,096 updates. The real seed-zero run
completed update 1,477 and failed while constructing update **1,478**. Its most
recent scheduled durable checkpoint is update **1,450**. Resume replays the
later unsaved updates and trims their non-durable log rows.

## Cause and behavior

The exact failing episode is **Titanic**, sample seed 3813224753, 128 total rows,
context fraction 0.5. A numerical feature (`column_11`) has no observed value in
the sampled 64-row context. Default mean imputation drops it. The merged ordinary
matrix has 12 columns, but `transform_parts` still indexes the pre-imputation
13-column layout and attempts to read column 12.

The fix derives retained numerical input positions from the fitted imputer's
non-NaN statistics and recomputes numerical output positions. Missingness masks
use those same retained input positions. Query-only observations do not restore
a feature absent from context. Ordinary `transform` keeps its existing behavior;
typed results now match its merged output. Arrays, mixed DataFrames, categorical
only inputs, fully missing numerical inputs and encoder refitting are covered.

## Verification and recovery

- The unmodified code reproduced the exact exception locally on the downloaded,
  hash-verified training bank. Final-test banks were not downloaded or opened.
- The repaired exact batch produces aligned finite `(1, 64, 12)` context/query
  tensors. CPU forward/backward through an actual small TabICL produces finite
  loss and nonzero learned-preprocessor gradients, with the backbone frozen.
- 69 distinct targeted checks passed across the continuation suite and new
  encoder regressions; 13 checks were added for this repair.
- All 1,638 planned real draws from `Titanic` and `horse_colic_outcome`, the two
  families with a numerical column observed on fewer than half of source rows,
  pass encoding, shape and finiteness checks across both 4,096-update streams.
  Mixed real draws are subsets of those paired real schedules.

Reuse the same result directory, source weights, optimizer/RNG checkpoint,
completed synthetic runs and frozen banks. Do not regenerate the manifest or
replace the failed episode. The pinned continuation modules are unchanged.
The original experiment manifest does not include the shared sklearn encoder
in its nine code hashes; this separate repair record captures its old/new hashes
and the synchronized revision. This is a recorded bug fix, not a relaxation of
the settings/hash checks.

Resume command uses the existing `pipeline --device cuda --resume` with one
GPU, 32G RAM and four CPUs on uriofir. Keep two serial eight-hour slots because
roughly 6–7 hours of remaining training/diagnostics plus locked final reporting
may exceed one allocation. Finished runs are skipped; a later slot exits if
the pipeline has already completed. Mail remains enabled.

Result root:
`/home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_real_meta_continuation/v1_seed20261004`.
Exact repair hashes, CPU proof and replacement submission metadata are saved in
[the repair record](joint_preprocessing_real_meta_continuation_repair_20261005.json).
