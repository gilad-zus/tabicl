# Direct native preprocessing: fixed versus resampled training on TabArena-Lite

Authorized 2026-10-11: expand the proposed eight-source comparison to benchmark
datasets. Implementation complete; seven targeted CPU checks passed. Remote
preparation/submission pending. One model seed, zero. This is a per-dataset
adaptation diagnostic for the transformation/training path; the research target
remains zero-shot generation on unseen datasets.

## Dataset panel and scope

Freeze all 51 task IDs from OpenML suite 457 using the existing
`openml_direct_spline_standard/full_D_500_big_gpu_v1/experiment_manifest.json`.
Use official repeat 0/fold 0/sample 0. Consider every classification task;
regression requires a different backbone/objective and is outside this diagnostic.
Record every exclusion, including tasks with more than ten classes, more than
2,000 features, insufficient rows of a class, or no varying numerical feature.
Preparation must accept at least 25 distinct dataset IDs before GPU work starts.
Selection uses structural data properties, never prediction quality. Download
failures stop preparation rather than silently reducing the panel. GPU failures
stop the run rather than silently removing difficult tasks.

Stratify at most 750 rows from each official outer-training partition. Reserve
20% for external validation V; the remaining 80% form fitting pool T. At the cap,
T=600 and V=150. Smaller tables use their available rows, preserving class coverage.
Full official outer-test E is retained; evaluation queries are batched at 256 rows.
Unused outer-training rows are unused in both arms and baseline. This is a capped
training-data diagnostic on benchmark datasets, not a full official leaderboard
replication. It is one fold/seed and includes benchmark datasets previously used
in development, so success is not fresh confirmation of zero-shot transfer.

Fit categorical encoding/numerical imputation on T only. Training, V and E use
that same raw encoder. Native none/power transforms fit each episode's context.
At V/E inference, both arms and ordinary16 use the identical complete T as context;
V is not added to the final context. Stored row IDs explicitly use outer-training
positions plus offset outer-test positions; the original outer split hash is saved.

## The paired comparison

For each dataset, fit two direct maps, initialized identically to ordinary16:

- Fixed: the same stratified 75%/25% context/query partition of T each step.
- Resampled: fresh stratified 75%/25% roles within the same T each step.

At the row cap, both use 450 context and 150 fitting-query rows per update. Both
parts contain every class, have fixed equal sizes across arms, and share no rows
within an episode. Step-derived seeds make resumption deterministic. Resampling
adds learning situations, not new independent observations.

Keep the current bounded affine, 20-control cubic spline, 8-unit neural residual
and rank-four mixing outputs, two native slots; no trainable encoder. Frozen
TabICLv2 classifier checkpoint SHA256:
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
Exactly sixteen native views: eight unchanged, eight residual-adapted. Mean-logit
cross-entropy at temperature 0.9; AdamW lr 0.001, weight decay 0.0001, clip 1;
250 updates per arm/dataset. No teacher or new architecture/regularizer.

Validate at zero, one (early GPU learning-path check), every 25 updates and final.
Select minimum V NLL, with identity eligible. Finish the full update budget even
if identity remains selected. Save each evaluated map, optimizer, exposure counts,
selected/final map hashes and immutable data/code/settings manifest.

Logs per dataset/arm:

- `training.csv`: fitting NLL, gradient norm, clipping, actual context/query rows,
  update time.
- `validation.csv`: external V NLL and ordinary16 gain, mean NLL/gain on three
  fixed training-pool probe episodes, selected step, min/max query exposure and
  fraction of fitting rows ever used as queries. These probes use T only.
- `checkpoints/step_*.pt`, `state.pt`, `selected.pt`, `final.pt`,
  `fit_complete.json`: durable states and locks.

No E score is computed until both arms on every accepted dataset have locked
their selected and final maps. Report all selected/final maps versus ordinary16,
and resampled versus fixed, including paired W/L/ties, arithmetic aggregate NLL
gain, mean/median dataset-relative gain, gain percentiles and dataset-bootstrap
intervals. Record accuracy and binary AUC; primary selection/comparison is NLL.
Use an NLL denominator floor of 0.0001 for relative summaries and disclose it.

## Relation to previous work

Closest work: October 10's eight-source fixed direct reference (selected outer
3/3/2) and October 11's context replay (original context 4/2/2, mean gain +0.087%).
Both controls are rerun because encoding is now T-only and V/E share full T as
context. September's sampler/cross-fitting studies already resampled rows with
other representations; this is not a new training principle. Its unanswered
question is whether the current native residual maps learn useful held-out-row
transformations under that protocol across a broader panel. The old TabArena
context-expansion artifact and research history section 6.3 were reviewed: its
16 completed tasks were a feasible subset and its matched identity gains limit
attribution. No historical scores are reused as this run's matched baseline.

Literature and rationale: [ICL adapter training review](icl_adapter_training_review_20261011.md).
Runner: `scripts/joint_preprocessing_benchmark_sampling.py`.
Local tests: `tests/test_joint_preprocessing_benchmark_sampling.py`.

## Execution

Planned root: `results/hyperspline_joint_benchmark_sampling/v1_seed20261011`.
CPU preparation uses dsiofir01 and `runnohup`; GPU fitting/evaluation uses SLURM
uriofir, one GPU, four CPUs, 32G host RAM, at most eight hours per submission,
email notifications enabled. Resume continues saved states and trims log rows
after the last saved update. If eight hours are insufficient, submit a resume
under the same immutable protocol; do not lower update budgets or drop tasks.
