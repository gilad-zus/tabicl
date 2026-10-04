# Frozen real-dataset transfer of the current zero-shot models

Authorized 2026-10-04. Status: implementation complete; 31 relevant local checks passed;
remote preparation and GPU submission have not yet occurred.

Question: do the small unseen-synthetic-task gains carry over to real datasets
without changing the shared hypernetwork weights or choosing settings on real labels?
Closest predecessors are the older seven-family HyperSpline transfer studies and
the unexecuted joint-pilot real panel. This run uses the current fresh/repeated/
teacher models and ordinary-8/16 ensemble controls. The panel has some historically
examined datasets; treat it as development transfer evidence, not an untouched
final benchmark. It does not isolate conditioner necessity or spline merit.

Source experiment: `hyperspline_joint_zero_shot_comparison/v1_seed20261003`.
Training revision: `a94e06d2753e480918eee9acd175c5e1cf2aafb9`.
Reuse `lock.json` unchanged: steps fresh/repeated/teacher = 3,072/5,120/9,216;
all three global learned-view weights are 0.5. Check source manifest, fingerprint,
backbone and selected checkpoint hashes before preparation and reporting.
Checkpoints are selected only from synthetic validation and remain frozen.

Use `joint_preprocessing_real_transfer_manifest_20260928.json` unchanged:
20 classification families, split seeds 0 and 1, at most 1,024 rows, stratified
70% labeled context / 30% scoring queries; minimum 256 usable rows, 2-10 classes,
5-100 encoded features and at least one numerical feature. Query labels are
used only to construct stratified evaluation partitions and compute scores;
they do not enter preprocessing fitting, generated parameters or predictions.
The two partitions are repeated evaluations of the same frozen models, never
hypernetwork training on one partition and testing on the other.

Family list: breast_cancer, digits, adult v2, credit-g v1, bank-marketing v1,
allbp, bupa, coil2000, dermatology, ecoli, ionosphere, magic, page_blocks,
phoneme, satimage, spambase, spectf, vehicle, waveform_21 and yeast.
Resolve unavailable/ineligible families before any learned-model scoring.
Preparation records failures and refuses to freeze an incomplete panel;
families cannot be replaced or excluded based on prediction performance.

Context-fitted numerical/categorical encoding is shared by all methods. Learned
preprocessing acts on numerical columns and preserves the ordinary categorical
views. PMLB columns are already encoded numerically; the three OpenML sources
exercise explicitly typed categorical data. Missingness summaries use the
context missingness mask. Query features are transformed by context-generated
parameters; they do not condition the hypernetwork.

Per episode, infer ordinary TabICL with 8 and 16 actual views, each of three
learned-only models with 8 views, and each model's fixed 8 ordinary + 8 learned
blend. Align class logits before averaging; temperature 0.9 matches the source
synthetic report. No gradients, optimizer, teacher fitting, per-dataset gating,
checkpoint selection or real-data blend-weight selection are performed.

Save class-aligned logits and query labels in one atomic cache per episode;
resume unfinished reporting with the same model/bank/code fingerprint. Preserve
actual view counts and fail on nonfinite predictions or unmatched counts rather
than quietly dropping outcomes. Stream `episodes.csv` after each scored episode.
Aggregate NLL, accuracy, binary AUC and time over the two splits within each
family before dataset win/loss counts and paired 20-family bootstrapping.
The primary ensemble comparison is against ordinary16; ordinary8 and learned-only
comparisons are additional references. Show all three arms, gains/harms >1/5/10%,
gain percentiles, and 10,000-draw paired family-bootstrap intervals (seed 20261004,
NLL floor 1e-4). Bootstrap uncertainty does not include training-seed variation.

Timing: synchronize CUDA around each full preprocessing-plus-inference call;
exclude shared context encoding; estimate blend time as ordinary8 plus learned8.
The first episode includes cold-start effects. This is a single-pass timing
diagnostic, not a repeated latency benchmark or an equal-runtime claim.

Implementation: `scripts/joint_preprocessing_zero_shot_real_transfer.py` reuses
the existing context-only real encoder and exact ensemble schedule. Its
preparation needs CPUs/network; reporting needs the frozen TabICL checkpoint/GPU.
Local tests exercise ordinary8/16 parity, learned-path parity, missing/mixed
features, withheld query labels, model/weight lock integrity, resume and dataset
aggregation.

Checks completed locally: nine real-transfer checks and 22 existing
synthetic/comparison regression checks passed. The latter include an actual
tiny TabICL forward/backward/inference smoke test. The downloaded source
experiment fingerprint was independently reproduced from its manifest/config.
One pytest cache-path warning occurred; regression checks disabled caching.

Result root:
`results/hyperspline_joint_zero_shot_real_transfer/v1_seed20261004/real_transfer/`.
Useful files: `model_lock.json`, `availability.json`, `manifest.json`,
`report/episodes.csv`, `report/families.csv`, `report/complete.json` and failures.
Bank/prediction tensors remain remote unless needed for a later audit.

Planned CPU preparation on dsiofir01 uses `runnohup jp-real-bank-s0-261004`;
combined log: `/home/dsi/zusmang/TabICL/tabicl/jp-real-bank-s0-261004.log`.
Planned GPU submission uses uriofir / p_uriofir / ug_uri_ofir targeting
dsiuriofir01, one GPU, four CPUs, 32G RAM, one hour, email BEGIN/END/FAIL.
The workload is 40 real episodes and 48 backbone views per episode, much smaller
than the completed 1,024-task synthetic report. Request the default minimum
profile resources. Log names and job ID will be recorded after submission.

Exact commands (run from the committed repository revision):

```sh
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_zero_shot_real_transfer prepare --source-dir results/hyperspline_joint_zero_shot_comparison/v1_seed20261003 --output-dir results/hyperspline_joint_zero_shot_real_transfer/v1_seed20261004
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_zero_shot_real_transfer report --source-dir results/hyperspline_joint_zero_shot_comparison/v1_seed20261003 --output-dir results/hyperspline_joint_zero_shot_real_transfer/v1_seed20261004 --device cuda
```
