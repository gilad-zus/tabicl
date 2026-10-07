# Fresh real-data learning with 40 versus 160 datasets

Authorized 2026-10-07. Status: implementation complete; **51 distinct targeted
checks passed** (22 new-runner/bank/catalog checks, ten inherited ensemble-gradient
checks, 19 real-bank checks). CLI and source compilation passed. No remote
preparation or GPU submission at local-verification time. Seed **0 only** for both runs.

Dispatch update: revision `26c5ec5` committed/pushed and synchronized. CPU bank
preparation launched on dsiofir01, PID **3160688**, log
`/home/dsi/zusmang/TabICL/tabicl/jp-dd-bank-261007.log`. Completion not yet
verified; no GPU submission yet. Live execution details:
[submission record](joint_preprocessing_dataset_diversity_submission_20261007.json).

Preparation repair: PMLB numerical-only loaders can return NumPy arrays; coverage
descriptors now normalize those arrays to DataFrames and copy numeric values
before imputation so read-only pandas buffers are supported. The new regression
checks pass. An explicit unfinished-preparation repair archives the old intent
and reuses source caches; it refuses repair after any bank/initial-weight lock.
Repair committed/pushed as `6d5f593`, synchronized and restarted with existing
source caches. Active PID **3161872**, log
`/home/dsi/zusmang/TabICL/tabicl/jp-dd-bank-fix-261007.log`.
That preparation completed with 185 declared groups, manifest SHA256
`3159c1975e047a9954d9cc3a12f681bb21e6827bd9600ab37909abc5f18d8290`.
It is retained as an **unused preparation audit**, not a training bank: subsequent
provenance review identified Adult/ADA and F16 target variants needing grouping.
No GPU time used.

The corrected catalog holds `adult`, `ada_prior` and anonymous AutoML `ada` in one
source group. The prior-knowledge ADA description explicitly names Adult as its
raw data ([OpenML 1037](https://www.openml.org/search?type=data&id=1037)); grouping
anonymous `ada` is conservative rather than established identity proof.
Aircraft targets `ailerons`, `elevators`, `delta_ailerons`, `delta_elevators` are
also held together because their descriptions identify the same F16 control
domain ([OpenML 198](https://www.openml.org/search?type=data&id=198),
[OpenML 216](https://www.openml.org/search?type=data&id=216)). This avoids treating
alternative targets/feature tables as independent transfer datasets.
Rebuild all allocations in a new v2 result root, using the prior raw cache only
for identical data-loading fields. Provenance/group annotation changes are allowed;
data IDs, targets, names, aliases and other loading fields must match. The old
raw cache is read-only and all duplicate/eligibility audits rerun.

Full cached-description review additionally excluded `Loan_Approval_Status`
(explicit synthetic version), `Dynamically-Generated-Hate-Speech-Dataset`
(explicit synthetic content), and `students_scores` (collection provenance
contradicts its name). Description screening now also runs on reused raw caches.
Grouped source copies include Satellite/satimage, spam/spambase, Wine Quality's
benchmark version named wine, re-encoded credit risk tables for China/modeling,
Give Me Some Credit variants and a conservative CreditCardSubset/fraud grouping.
Catalog inventories and class-count metadata support the financial copy grouping;
not every shared-source inference is an established record-level identity.

Final preparation dispatched from committed revision `52799ce` on dsiofir01,
PID **3168727**, exact log
`/home/dsi/zusmang/TabICL/tabicl/jp-dd-bank-final-261007.log`.
The earlier v2 preparation was stopped before freezing any bank. Its intent is
archived; identical raw data are reused under the corrected rules. Preparation
completed. All 185 distinct declared groups, unique input hashes, nested panels,
train/validation group separation and 42 committed dependency hashes were
verified locally after downloading the four JSON metadata artifacts.
Final manifest SHA256:
`0e7295768e6392573650f21af5f75edd918b006617543d384853082d6d9d19ee`.

Submitted SLURM array **33062914**, job name **jp-dd-s0-261007**, October 7 19:49.
Tasks 0=small, 1=large; `ArrayTaskThrottle=1` verified. Pending **Resources** at the
status check; no training results yet. Per task: profile uriofir, partition
p_uriofir (sole node dsiuriofir01), account ug_uri_ofir, one GPU, four CPUs, 32G RAM,
six-hour limit and email BEGIN/END/FAIL enabled. Result root is v2 as below.

Exact logs under `/home/dsi/zusmang/TabICL/tabicl/slurm_logs/`:

- `slurm-jp-dd-s0-261007-33062914_0.out`
- `slurm-jp-dd-s0-261007-33062914_0.err`
- `slurm-jp-dd-s0-261007-33062914_1.out`
- `slurm-jp-dd-s0-261007-33062914_1.err`

Coverage is now observable: median feature counts small/large/validation are
18.5/16/16, and median absolute skew is 0.783/1.039/1.049. Two validation sources
have class counts outside the small-bank range, zero outside the large-bank range;
skew is outside range for three versus one. These descriptors support broader
coverage, not equivalence of optimal transformations. Full coverage in the
submission record and downloaded banks_manifest.json.

October 7 failure update: both tasks failed before optimizer updates while
building validation references. [Numerical repair](joint_preprocessing_dataset_diversity_numerical_repair_20261007.md)
adds a fixed [-100,100] standardized-input guard to train/inference and reuses
the exact frozen data panels in v3. Neither failed startup provides a learning
result. Replacement submission and GPU preflight are tracked in the repair note.

## Question and closest predecessor

Does broader real-data experience improve zero-shot preprocessing at a fixed
update budget? The October 6 ensemble-objective pilot trained on 40 real sources
from the synthetic repeated checkpoint, and the October 7 related-domain result
continued that same checkpoint on four sources per domain. Both showed weak
source learning and no reliable baseline benefit. This run changes initialization
to a fresh shared JointPreprocessor and compares nested training-source counts.
Both arms use ensemble CE, so it is not another objective comparison.

Working assumption: real coverage may be inadequate and synthetic initialization
may restrict useful real learning. The two fresh arms isolate the dataset-count
comparison under this recipe; comparison with historical continuation runs does
not cleanly isolate initialization because their banks/selection/budgets differ.

## Frozen protocol

| Setting | Small bank | Large bank |
|---|---:|---:|
| Training source groups | 40 | 160 |
| Shared parameter initialization | CPU torch seed 0 | Identical saved weights |
| Updates | 4,096 | 4,096 |
| Episodes per update | 4 | 4 |
| Total episode presentations | 16,384 | 16,384 |
| Presentations per source | 409 or 410 | 102 or 103 |
| Learning rate | 0.0003 | 0.0003 |
| AdamW weight decay | 0.0001 | 0.0001 |
| AdamW betas / epsilon | 0.9, 0.999 / 1e-8 | Identical |
| Gradient clipping | Global norm 1 | Identical |
| Evaluation interval | 512 updates, also zero/final | Identical |
| Durable resume interval | 25 updates | Identical |

TabICL classifier v2 20260212 weights are frozen. The architecture is the existing
joint generator: shared raw-context encoder (hidden 64, four attention heads),
two preprocessing slots, generated affine parameters, 20-control-point monotone
splines, neural residuals, rank-four bounded feature mixing and existing gates.
Generate parameters using labelled context only, then apply the same map to
context/query. Categorical features retain the standard preprocessing path.
The available query labels supply meta-training CE; validation labels never
update weights. Deployment has no dataset-specific optimization or query-label
conditioning. There are no teacher targets or teacher fitting jobs.

Optimize the existing memory-bounded exact gradient of ordinary8+learned8
ensemble CE: class-aligned logits, learned/ordinary weights 0.5/0.5, temperature
0.9. Training uses FP32; inference retains the default CUDA AMP managers. Run the
existing numerical train/deployment-gradient audit before the first update of
each arm. Ordinary16 is the deployment reference, with ordinary8/learned8 and the
common fresh initial blend additionally logged.

## Dataset construction and overlap protection

Candidates are frozen in
`joint_preprocessing_dataset_diversity_candidates_20261007.json` before any model
scoring. The public OpenML metadata snapshot yields reviewed candidate IDs;
known prior PMLB/sklearn/OpenML candidates supply alternate availability sources.
The catalog builder records snapshot hashes and explicit source-group aliases:
371 candidate entries represent 209 declared groups before actual-source checks.
Many entries are versions or alternative feature/target tables and do not count
as independent datasets. Source-name review is incomplete provenance evidence.

Exclude known generated/simulated feature families and descriptions explicitly
identifying synthetic/artificial/simulated data. Real-observation classification
tables may include older OpenML versions with discretized targets. This broader
classification scope is disclosed; do not claim all 160 have original natural
classification targets or exhaustively verified collection provenance.

Require 185 accepted source groups before GPU use: 160 train plus 25 common
validation. At least 256 usable rows, 2–10 classes with at least two rows per class,
5–100 context-encoded features and a retained varying numerical feature. Retain
up to 16,384 rows with recorded class-constrained source indices. Audit declared
source aliases, feature-only hashes invariant to row/column ordering and >=80%
shared-row copies. Remove detected copies before panel allocation, recording
reasons. No model gains influence eligibility, replacements or grouping.

Allocate the 25 validation datasets and the nested 40-source subset by a fixed
metadata-balanced rule: binary/multiclass, feature-count cells and categorical
fraction. Record class balance, missingness, skew, absolute correlation, rows and
features, including percentiles and validation values outside training ranges.
These properties describe coverage; they do not establish similarity between
useful transformations. The validation group is excluded from both source banks.

All selection/availability decisions and packed source data are cached for resume.
If fewer than 185 eligible independent sources survive, preparation fails and
no GPU run starts. Counts are never silently reduced.

## Episodes and learning diagnostics

Each arm cycles uniform independently shuffled source schedules. Request total
rows 128/256/512/1,024 and context fractions 0.5/0.7/0.85 in the same balanced
12-shape schedule. Each update caps rows to the smallest available pool among
both arms' scheduled eight sources; this pairs actual context/query sizes between
arms. Resample rows every visit, preserve all classes in context/query, and fit
encoding/imputation/category vocabularies on each context only.

For example, 1,024 rows at context fraction 0.7 yields 716 context and 308 query
rows; fractions 0.5/0.85 yield 512/512 and 870/154. At 128 rows these are 89/39,
64/64 and 108/20, respectively. Actual capped sizes are logged per presentation.
Equal updates/row schedules do not promise equal runtime because feature counts
and class counts differ; report actual compute separately.

Fixed evaluation panels:

- 40 common source datasets: one fixed episode each, seen by both arms.
- 40 sources seen only by the large arm: one fixed episode each; source evidence
  for the large arm and unseen-dataset evidence for the small arm. These are
  development probes, not the common validation selection panel.
- 25 common validation source groups: two fixed partitions each, up to 1,024 rows
  and 70% context. Average partitions within each dataset before aggregation.

Log stochastic training CE, encoder/head gradient norms, clipping, actual shapes
and dataset visit counts. Fixed probes report gain from the identical start,
baseline wins/losses, mean NLL, medians, harm tails, transform magnitudes/gates and
saturation on the common-source panel. Fresh training episodes and fixed source
probes differ, so flat probe results do not prove failure to fit actual batches.

## Selection and decision

Select one global checkpoint per arm by minimum **arithmetic mean dataset NLL**
on common validation, with update zero eligible. This prospectively changes the
prior mean log-ratio selection, which was sensitive to near-perfect baselines.
Primary diversity comparison uses final equal-update models. Selected-model
comparisons are secondary and may have unequal selected training budgets.

Practical development gate: the selected checkpoint must be later than zero,
win at least 70% of validation datasets (18/25), reduce mean NLL by at least 0.5%
versus ordinary16, have a positive median relative gain, and have an adjacent
evaluated checkpoint also satisfying those performance thresholds. Report all
metrics even if the gate fails; this heuristic is not a significance test.

Use paired dataset bootstrap intervals and gain percentiles. One seed and one
nested source-bank draw do not establish seed/subset robustness. A larger bank
winning despite fewer visits supports paying for more diversity. A negative result
cannot rule out a larger bank needing more optimization. Validation may include
previously inspected project datasets and remains development evidence.

No final-test bank is created or opened in this run. Fresh confirmation would
require model choices frozen first and independent uninspected source groups.
Training/report entry points reject changed code, settings, banks or checkpoints.

## Execution specification

Result root:
`results/hyperspline_joint_dataset_diversity/v2_seed20261007`.
Remote repository: `/home/dsi/zusmang/TabICL/tabicl`.
Python: `/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python`.

CPU preparation on dsiofir01, after reproducible Git synchronization:
`runnohup jp-dd-bank-final-261007 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <python> -u -m scripts.joint_preprocessing_dataset_diversity prepare --output-dir <root> --reuse-source-cache <v1-root>/source_cache --repair-unfinished-preparation --device cpu --expected-revision 52799ceff19d2171cdc4ce03b8cd29b3f226735c`.
Exact log: `/home/dsi/zusmang/TabICL/tabicl/jp-dd-bank-final-261007.log`.

After bank completion/coverage verification, submit a two-task SLURM array limited
to one running task (`0-1%1`): task zero trains small, task one trains large.
Scheduler task order is not assumed. Each task conditionally builds the paired
report after both completion files exist; the second successful finisher reports.
Profile uriofir, partition p_uriofir, account ug_uri_ofir, one GPU, 32G host RAM,
four CPUs, six hours each, email notifications enabled. The measured previous
ensemble run needed about one training hour per 1,024 updates; 4,096 updates plus
larger evaluation panels motivate this limit. Resume durable state if needed.

Full commands are recorded with exact revision, job IDs, output/error names and
approvals in the submission record after dispatch. No GPU allocation waits for
CPU preparation. The array concurrency limit prevents simultaneous writes to
the shared reference caches; one seed is used throughout.
Launches pass `--expected-revision` and reject dirty experiment dependencies or
candidate metadata; fingerprints also record Python/Torch/NumPy/sklearn versions.
