# Joint preprocessing: synthetic / real / mixed continuation

Authorized **2026-10-04**. Implementation is complete; local combined checks and
launch preparation are underway. No GPU outcome exists at this snapshot.

Question: does exposure to real source families improve frozen zero-shot
preprocessing beyond the same additional synthetic training? Closest predecessors
are the September 28 mixed-training proposal, older real-meta code without an
identified completed local outcome, and the completed October 4 synthetic-only
comparison plus frozen 20-family transfer. The new intervention is the source of
shared continuation training; deployment still performs no target weight updates.

## Fixed design

| Arm | Episodes per update | Updates | Continuation seeds |
|---|---|---:|---|
| synthetic | Four fresh synthetic | 4,096 | 0, 1 |
| real | Four resampled real-source episodes | 4,096 | 0, 1 |
| mixed | Two of each, alternating positions | 4,096 | 0, 1 |

All six runs start from the **same repeated teacher-free synthetic checkpoint**,
step 5,120, hash
`30f1255c740385894903d7d1311d9017ce648ea9fd4366528649700ed531a206`.
Source result root: `hyperspline_joint_zero_shot_comparison/v1_seed20261003`.
Verify the original model/test lock and source/backbone hashes. Reset AdamW at
continuation step zero: lr 0.0003, weight decay 0.0001, betas (0.9,0.999), epsilon
1e-8, gradient clipping 1.0, float32. The earlier teacher-transition optimizer
reset is not part of this experiment. Backbone parameters remain frozen.

Each arm receives 16,384 episode presentations; mixed receives exactly 8,192
real and 8,192 synthetic. The real-source shuffle is shared across paired arms;
mixed takes two positions per batch. Fresh synthetic tasks are paired with the
synthetic control using the same generation seed. Continuation seeds vary data
order/task draws, not pretrained model initialization. They do not measure
pretraining-seed robustness.

## Family banks and sampling

Target **40 real training families / 10 validation / 30 final-test families**.
Freeze predeclared candidate and fallback order before model scoring. Accept
replacements only for source availability/eligibility, with reasons recorded in
`availability.json`; require all counts before any GPU experiment. Disjoint
source groups and aliases cover related datasets. Content hashes and shared-row
checks additionally detect copies or subsamples. This is a documented overlap
audit, not proof that every unknown common-source relationship is absent.

Final-test candidates exclude the current inspected 20-family panel and other
historically examined names/source groups. Real pools exclude known generated
benchmarks. In particular, `waveform_21` in the older nominally real panel is
generated data, as documented by [UCI](https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1).
The existing 20-family result remains the predeclared benchmark result; its
strict real-world provenance is therefore qualified. This continuation uses a
stricter source inventory rather than selecting datasets by observed gains.

Eligibility: at least 256 usable source rows, 2–10 classes, 5–100 context-encoded
features, at least one retained varying numerical feature and enough examples
to place each class in context/query. Retain at most 16,384 source rows via
deterministic constrained stratified sampling, recording original row indices.
Fit encoding, missing-value imputation and category vocabularies on **each
episode's context only**.

Training samples the 12 combinations of total rows 128/256/512/1,024 and context
fraction 0.5/0.7/0.85. Select four real families using a uniform shuffled source
schedule, then cap the entire paired batch to the smallest available pool among
those four. Synthetic and mixed controls follow the same cap so row size is not
silently coupled to data source. Constrained class quotas preserve every class
in context and query, with actual/requested sizes recorded. Numerical-only
synthetics retain `mix_scm` and marginal-only `coverage_expanded` observation
transforms from the preceding experiment.

Validation/test real episodes use two fixed partitions, seeds 0 and 1, up to
1,024 rows, 70% context / 30% queries. These are repeated frozen evaluations,
not training on one partition and testing on another. The actual locked manifest
lists selected families and provenance.

Banks are separate files. Training deserializes `real_train`, `real_probe`,
`real_validation`, `synthetic_probe` and `synthetic_validation`; final test files
are first deserialized after all six model choices are locked. Synthetic
validation has 128 tasks for diagnostic tracking; synthetic final test has
1,024 new tasks. New seed schedules are checked against the previous pilot and
October comparison. The fixed synthetic probe contains the first 48 seed-zero
fresh-stream tasks; its seen status differs across arms/seeds and is diagnostic.

## Objective and monitoring

Keep teacher-free raw-logit single-view query CE through frozen TabICL, equally
averaging four task losses. For real data, ordinary preprocessing is retained
for categorical cells; the generated numerical maps replace numerical cells in
the corresponding view. The conditioner sees numerical context, its missingness
mask and context labels. It does not newly condition on categorical values or
query features. No teacher is fitted.

The frozen backbone uses its differentiable training execution path for input
gradients, with dropout disabled. Generated maps remain in the graph. Preserve
feature/class shuffle alignment; query labels enter only the CE/scoring loss.
Raw training logits have no temperature. Evaluation averages class-aligned
logits, then uses temperature 0.9.

Evaluate step zero and every 512 updates. Log learned8, ordinary8, ordinary16,
and fixed 50/50 ordinary8+learned8 blend scores; average two partitions within
each real family before W/L and normalized metrics. Select one global checkpoint
per arm/seed by mean real-validation family log NLL ratio of the blend against
ordinary16. Include step zero. Keep alpha 0.5 without target-specific selection.
Final step-4,096 states are additional diagnostics, not test-selected candidates.

`training.csv` records CE, LR, step seconds, gradient norms/clipping and source
counts. `presentations.csv` records family/task identity and actual shapes.
`evaluation.csv` records source-family versus new-family loss/gain/WL/harms.
`evaluation_episodes.csv` retains per-task values and source-probe gate,
correction, mixing and neural-saturation diagnostics. Fixed real probes use seen
training families; improvements there never count as unseen-family evidence.
Heartbeat output and atomic checkpoints occur every 50 updates. Resume restores
model, AdamW, Python/NumPy/Torch/CUDA RNG, trims non-durable log rows and reuses
verified reference caches.

## Final comparisons

After all six runs complete, freeze `lock.json` before opening either test bank.
Report ordinary8/16, unchanged starting learned/blended models, and selected and
final learned/blended models. Save resumable class-aligned prediction caches,
episode CSV, family CSV and complete summaries. Report per seed and paired seed
means; two continuation seeds do not double the independent family count.

Primary comparisons: each selected blend versus ordinary16; real/mixed versus
paired synthetic continuation; all arms versus unchanged starting blend. Report
geometric/median NLL gain, 10,000 paired family-bootstrap intervals, accuracy,
binary AUC, gain percentiles, >1/5/10% harm counts and seed-consistent outcomes.
Synthetic tasks are reported separately. Inference timing synchronizes CUDA,
excludes common context encoding and estimates blend time by summing the two
calls; no polished end-to-end latency claim.

If real/mixed gains transfer across new families and continuation seeds, develop
real-source coverage and subsequently test ensemble-aware training. If source
probes improve while new-family validation stagnates, investigate transfer/task
coverage. If source learning remains weak, investigate optimization and
conditioning. These patterns guide diagnostics but do not uniquely identify a
cause. Conditioner necessity and another pretraining seed remain later controls.

## Execution

Entry point: `scripts/joint_preprocessing_real_meta_continuation.py`.
CPU preparation uses `runnohup` on dsiofir01. GPU training/testing uses uriofir
SLURM, one GPU, 32G RAM and four CPUs per allocation, email enabled. Prior measured
training time implies 62–83 minutes per 4,096-update arm before setup/evaluation.
Launch two serial eight-hour array slots, concurrency one, each running the same
resumable pipeline. Slots skip finished arms, resume the current one, lock all
six models, then resume final reporting. This uses one GPU at a time and respects
the eight-hour maximum per allocation. About 6–8.3 training hours plus diagnostic
evaluation/setup and up to roughly three reporting hours justify more than one
slot. Later slots exit promptly if the pipeline is already complete. Exact
approved commands, job IDs and logs will be appended after submission.

Result root: `results/hyperspline_joint_real_meta_continuation/v1_seed20261004`.
