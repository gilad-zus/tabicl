# Restricted direct native preprocessing on the 33-dataset benchmark

Authorized 2026-10-13. Train one additional resampled arm, seed zero, retaining
per-column affine corrections and the existing K20 monotone cubic spline.
Remove all neural correction and rank-four mixing parameters and computations.
Retrain from exact native16 identity; do not edit the completed fitted maps.
This jointly removes two branches and cannot identify their individual effects.
The research target remains a shared generator deployed zero-shot on new datasets.

## Frozen controls and data

Reuse the completed October 11 benchmark's 33 accepted classification datasets
(28 binary, five multiclass), official repeat 0/fold 0/sample 0, exclusions,
encoded bank bytes and row IDs. Source fingerprint:
`60e0745f4221d102fc84e563b99f8cb5b2866afd446a1536a7686edf402ae04f`.
Source complete-report SHA256:
`49f293b701f631aa55e5bbdb56a05863422dc400db140528f76cf664dff304db`.

Source directory: `results/hyperspline_joint_benchmark_sampling/v1_seed20261011`.
New directory: `results/hyperspline_joint_benchmark_restricted/v1_seed20261013`.
Read banks directly from the source; write all new logs, maps and reports into
the new directory. Verify source fingerprints, bank bytes, completed control
fits and selected/final map hashes before fitting. No OpenML download, new data
bank, full-map retraining, or fixed-episode arm is needed.

At the cap use 600 fitting rows, 150 external validation rows, and resampled
450-context/150-query training episodes. Use the identical step-derived seeds
as the completed full-map resampled arm. Smaller datasets retain their original
sizes. Encoding/imputation was fitted on the fitting pool only. Both V and full
official test E receive the same complete fitting context. Query chunk size 256.
The completed reference scored 256,856 official test rows.

## Optimization and selection

Freeze the same classifier checkpoint, SHA256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
Use exactly sixteen native views, eight unchanged and eight residual-adapted,
with two generated-map slots. The restricted direct map has 44 trainable scalars
per numerical column, compared with 114 in the completed full map. No conditioner
or backbone parameters are trained. Retained raw outputs initialize identically
to their full-map counterparts; absent branches have no optimizer state.

Use 250 updates, AdamW lr 0.001, weight decay 0.0001, gradient clipping 1,
mean-logit CE at temperature 0.9 and the existing FP32 two-pass input gradient.
Validate at zero, one, each 25 updates and final. Select minimum external V NLL
with identity eligible. Finish every update budget. Preserve checkpoint/optimizer
resume, query exposure, fitting loss/gradient logs and three fixed training probes.

Lock all 33 selected/final maps before opening test banks or reading historical
test-score values. Report selected/final maps against ordinary16 and the existing
full-map resampled counterparts. Recompute ordinary16 and require numerical
parity with the historical baseline (NLL absolute tolerance 1e-6, relative 1e-5)
before reusing its control scores. Report W/L/ties, mean/median gains, percentiles,
bootstrap intervals and harms, with binary/multiclass summaries during analysis.

## Decision and evidence boundary

Use the existing practical gate as a development decision: selected aggregate
mean NLL gain at least 0.5%, positive median gain and at least 70% dataset wins
(24/33). Also review losses above 5% and uncertainty before investing in a new
shared-model run. Lower harms alone do not establish useful preprocessing.
This gate is a research resource decision, not a significance test or a fresh
confirmation claim. A negative result motivates shelving this recipe; it does
not disprove all zero-shot preprocessing. A shared model can regularize differently
from direct fitting, so direct results are not an upper bound on meta-learning.

Closest experiments: the completed full-map real benchmark and September 29's
synthetic restricted/joint comparison. The latter's broad generator had no clear
advantage; old capacity and regularization studies were mixed. What changes here
is the removal of both branches within the current exact-native real-data recipe.
All 33 datasets are already inspected development evidence. Any later zero-shot
claim needs dataset-disjoint evaluation.

## Execution

Use the existing `scripts/joint_preprocessing_benchmark_sampling.py` runner with
`--transformation restricted --source-dir
results/hyperspline_joint_benchmark_sampling/v1_seed20261011`. Manifest preparation
only references and hashes existing files, and runs as part of the GPU job.
Submit to uriofir/dsiuriofir01, one GPU, four CPUs, 32G host RAM, eight hours,
email enabled; resume the same immutable run if the allocation expires.

Implementation checked locally: 37 targeted CPU tests passed. These cover
restricted parameter removal/counts, retained-head initialization, exact native16
identity, monotonicity/missing-value preservation, two-pass gradient parity,
restart equivalence, immutable source-bank reuse and test locking, plus the
existing full-model and statistics-runner regression checks. CLI arguments and
the scoped diff were reviewed. A pytest cache-write warning did not affect tests.
