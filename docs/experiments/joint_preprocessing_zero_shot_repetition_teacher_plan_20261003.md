# Zero-shot learning: task diversity, repetition and teacher supervision

Status 2026-10-03: implementation completed locally; execution now authorized by
"so continue, run it". The three-arm runner has local regression checks and a
native-prior CPU bank-preparation smoke check. Submission is being prepared;
job IDs and logs will be recorded below once confirmed.
This replaces the initialization-only comparison as the recommended next design.

Post-diagnostic interpretation, 2026-10-03: the eight-task from-scratch control
has completed. Repeated fitting queries permit memorization, so its gains
establish improved fitting rather than a transferable preprocessing rule.
Lower shared lr .0003 improved fitting relative to .001, motivating the
already planned .0003 setting; it is not a demonstrated optimum for the
four-task batch regime here. Return to this new-task validation/test comparison
as the recommended next experiment; another extension of eight-task fitting
is not a prerequisite. The current request authorizes implementation.
The 80-visit bank budget is an initial resource choice, not a claim that it
converges: interpret fitting and new-task validation trajectories together.
Weak fitting at that budget cannot establish that repetition or teacher
supervision is incapable of helping. Final test remains locked until global
checkpoint and ensemble choices are fixed on validation.

## Why this comparison

The original pilot used 40,000 fresh tasks once each. The recent diagnostics
used 32 fixed tasks repeatedly and showed strong fitting after teaching plus
prediction fine-tuning, but weak new-task validation. These results do not
establish that one visit per task is optimal, or that all finite task banks
overfit. The missing regime is a substantially larger repeated bank.

Revisiting training datasets is compatible with zero-shot deployment. Every
visit updates one shared hypernetwork. Evaluation tasks remain disjoint and
receive no gradient updates, fitted teacher, target-specific checkpoint or
query-label-based selection. Independent teachers are training assets only.

The previous initialization-only proposal inherited teaching on just 32 tasks;
it would not directly test teacher supervision across the current training
bank. This design does. Teacher/conditioner learning has earlier precedents
(research history section 3); the changed question is unseen-task performance
of the current joint network after learning on a 512-task teacher bank,
compared with direct learning on exactly those tasks.

## Three arms

| Arm | Distinct training tasks | Shared training | Presentations per task |
| --- | ---: | --- | ---: |
| A: fresh/direct | 40,960 | 10,240 updates of query CE | 1 |
| B: repeated/direct | 512 | 10,240 updates of query CE | 80 |
| C: repeated/teacher then direct | The same 512 as B | 2,048 updates of function teaching, then 8,192 updates of query CE | 16 teacher + 64 direct |

Each update averages four task losses. Thus each arm gets 40,960 shared-model
task presentations and 10,240 shared optimizer updates. The round budget is
2.4% larger than the former 40,000-task proposal so the finite bank has exactly
80 passes. The repeated bank is a deterministic, size-balanced 512-task subset
of A's task stream, chosen before any model scores. B and C share its context,
queries, labels, per-task training view and epoch permutations exactly.

All three hypernetworks start from the same fresh seed-zero model. C does not
inherit the old 32-task checkpoint: its teaching is on the current 512 tasks.
The teacher-only stage-end model is evaluated as a learning diagnostic; it has
only 2,048 updates and is not an equal-budget standalone fourth arm.

A versus B tests the breadth/repetition allocation at fixed task presentations.
B versus C tests the staged teacher-training recipe on a matched task bank
and shared-update budget. It does not isolate teacher benefit at matched
downstream CE visits: C replaces its first 2,048 CE updates with imitation.
The additional teacher construction cost must also be reported separately.

## Teacher construction and student supervision

Fit independent, bounded raw transformation parameters from the current joint
family on each of the 512 training tasks. Retain the same fixed labelled
context and active single view used by the direct arms. Use one predeclared
AdamW rate, 0.001, for 250 updates, weight decay 0.0001 and clipping at 1.0.
This is the primary rate from the fitting-capacity diagnostic; there is no
three-rate teacher sweep here. Record checkpoints every 25 updates and select
each teacher by its fitting NLL, including initialized standardized identity.
This selection uses training data and is not a target-deployment procedure.

If the selected teacher has less than 1% stabilized fitting gain over identity,
use the initialized identity function as its target. Retain the task in the
bank. Such a label describes this fitting procedure; it does not prove the
task cannot benefit from preprocessing. Report positive/neutral target counts.

As in the completed diagnostic, teach the active slot's transformed values on
context and fitting query features. Normalize function MSE by
max(teacher-versus-identity MSE, 0.01). The hypernetwork's conditioning input
still contains only context features and labels. Use actual query-label CE
through frozen TabICL in the second phase. Both phases train the shared model.
Only one slot/view per task is directly supervised, matching the current
diagnostic; full ensemble effects are measured separately.

We choose teaching followed by prediction loss because continued function
imitation alone recovered little prediction benefit in the last experiment.
This is a concrete, already implemented form of teacher supervision, not a
claim that function MSE is the best distillation objective. Log function error
and prediction benefit separately throughout both phases.

Teacher generation adds 512 x 250 = 128,000 independent optimization updates,
plus scoring. It is material extra compute. Save resumable per-task results
and record teacher-building GPU hours separately from shared training. Equal
shared update counts are not equal total compute, and imitation updates are
cheaper than CE updates through TabICL. The inherited 32-task teacher results
are evidence motivating the design, not extra training data for any arm.

## Matching, data and evaluation

Use the current joint model, frozen TabICLv2, synthetic generator, class/feature
ranges and 12 context/query size combinations from the previous detailed
plan. Use fixed episodes/views for B/C in this first comparison; new resplits
are an additional intervention and are not silently added. Shuffle all 512
task IDs each epoch, with four distinct tasks per minibatch. Do not perform
80 consecutive updates on one dataset. One-task gradients affect the shared
network used on every task.

Shared AdamW lr 0.0003, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8,
gradient clip 1.0, float32, seed 0. Reset optimizer moments at update 2,048
in all arms, controlling for C's loss transition. No architecture or generator
change is introduced. Fixed-bank repetition can still encourage memorization;
new-task validation is what decides whether it helps.

Use new, fixed 512-task validation and 1,024-task final-test banks, disjoint
from all training and teacher tasks and from the previously inspected test
banks. Select one global checkpoint per arm using validation ensemble mean
log NLL ratio against ordinary TabICL. Evaluate step zero, every 1,024 updates
and final 10,240; this includes the teaching transition at 2,048. Never select
on training gain or teacher recovery. Freeze all choices before final testing.

Retain gradient/branch diagnostics, per-task predictions, learning curves,
paired task bootstrap intervals, median/geometric gains, W/L/T and material
harm counts from the earlier plan. Score a predeclared common training probe
without extra optimization to compare fitting with new-task behavior. The
secondary ordinary-plus-learned ensemble comparison, global validation-only
weighting and ordinary-16 compute control remain applicable; teacher fitting
and gating never occur on validation/test datasets.

Interpret A>B as evidence favoring diversity at this budget; B>A as evidence
favoring repetition for this bank size; C>B as evidence for this teacher-plus-
prediction recipe on unseen tasks, with its extra construction cost stated.
Improved bank fitting alone is not success. No result would prove a universal
optimal task count. Real-family transfer remains a subsequent frozen-model
evaluation after a promising synthetic result.

Implementation must lock the repeated-bank subset, all seed schedules,
source/model hashes and exact teacher/source artifacts before launch. The
previous initialization-only plan is retained as a superseded proposal, not
a second active run. No additional teacher-only experiment is a prerequisite.

## Implemented data protocol

Runner: `scripts/joint_preprocessing_zero_shot_comparison.py`. All generated
episodes initially reside on CPU. The existing native TabICL `mix_scm` prior
draws MLP/tree SCMs with probabilities 0.7/0.3, 5-100 features and 2-10 classes,
one task per prior group/subgroup, one generation thread and one prior worker.
`coverage_expanded` applies label-independent monotone marginal distortions
before splitting context/query: signed powers, exponential responses,
saturation, log compression, asymmetric tails, censoring and tail stretching,
with optional rounding/resolution floors. It does not enable the newer
structural modes. The context must contain every generated class; the existing
generator's class checks/remapping are retained.

| Total rows | 50% context: context/query | 70% context: context/query | 85% context: context/query |
| ---: | ---: | ---: | ---: |
| 128 | 64/64 | 89/39 | 108/20 |
| 256 | 128/128 | 179/77 | 217/39 |
| 512 | 256/256 | 358/154 | 435/77 |
| 1,024 | 512/512 | 716/308 | 870/154 |

The fresh stream cycles a seeded permutation of these 12 shapes, generating
four independent tasks per update. The repeated bank uses its first 128 complete
updates, before outcomes exist: each shape receives 40 or 44 of the 512 tasks.
Validation/test coverage differs by at most one task per shape. Each B/C epoch
shuffles the whole bank and presents four distinct tasks per update. The common
training probe is the first 48 bank tasks, covering one full shape cycle; its
scores do not select checkpoints.

Task-ID namespaces start at 6/7/8 billion for training/validation/test. Fresh
update seeds are `(201001 + step * 1000003) mod 2**32`; validation/test base
seeds are 211001/221001 and their shape groups use the existing helper's
`base + 10000019 * (group_index + 1)` schedule. Epoch ordering uses
`231001 + epoch`. Preparation checks seed separation against the recent joint
pilot and fixed-bank schedules, and checks content hashes within/across the new
banks. This audit does not claim exhaustive comparison to every historical
synthetic episode. Source, bank, backbone and teacher hashes are recorded and
must match on resume. The fresh stream records every task's content hash,
actual dimensions, source seed and fixed training view as it is visited.

## Selection, logs and reports

Training evaluates only the probe and validation panels. Hypernetwork
conditioning uses context features/labels; query labels contribute only to
training loss or held-out scoring. Evaluation uses frozen weights without
dataset-specific optimization. The global checkpoint criterion is the mean
task log ratio `(learned NLL + 1e-4)/(ordinary8 NLL + 1e-4)`, including step zero.
Inference aligns class-shuffled logits, averages them and applies temperature
0.9, matching the existing ordinary baseline. Fitting CE retains the previous
single-view training path and its temperature convention.

After all three arms complete, `lock` replays the selected validation scores
and freezes each checkpoint hash and one global blend alpha from
`{0, 0.25, 0.5}`. Smaller alpha wins exact ties. The fixed 50/50 addition is also
reported. Blending averages aligned logits with learned weight alpha; alpha zero
is ordinary TabICL. There is no per-dataset gate fit on validation or test.
`test` requires this lock, records its hash before opening the test bank and
rejects changed choices/checkpoints. Training and re-selection are blocked
after test starts, including after an interrupted test.

Each arm records:

- `training.csv`: objective and phase, task IDs, branch gradient norms,
  clipping, optimizer-reset count, per-update seconds; C also records normalized
  function error during both imitation and CE.
- `evaluation.csv` / `evaluation_tasks.csv`: fixed-probe and new-task validation
  NLL, W/L, geometric gain and harms; per-task branch/gate/saturation diagnostics.
  C's probe also includes teacher function error and useful-teacher NLL recovery.
- `state.pt`: exact model/AdamW/RNG resume state; `selected.pt` / `complete.json`:
  the one globally selected checkpoint and recorded training/evaluation seconds.

Teacher fits retain per-task losses/checkpoints, replay selected outputs and
write useful/neutral counts, target hashes and `teachers/timing.csv`. Recorded
timings exclude work lost before save; arm step timings also exclude setup and
reference computation. Scheduler elapsed time should accompany these timings
for total compute comparisons.

`test_report/complete.json` contains learned, equal-blend and selected-blend
results against ordinary8/ordinary16, the ordinary16-versus-ordinary8 control,
and paired B-versus-A, C-versus-B, C-versus-A comparisons. Reports include mean
NLL/delta, median/geometric gains, 95% paired task-bootstrap gain intervals
(10,000 draws, seed 20261003), W/L/T (absolute NLL tolerance 1e-6), harms over
1/5/10% and worst harm. Stabilized relative gains use the same 1e-4 floor.
`tasks.csv` records actual ordinary/learned/blend view counts, including the
ordinary-only deployment cost when the selected alpha is zero. CPU prediction files preserve
query logits/labels so the final report can be audited locally.

Read fitting and validation together: improved probe loss with stagnant/worse
validation suggests a transfer/generalization limitation; weak probe loss with
continuing improvement suggests more optimization may help. High saturation,
clipping or vanishing branch gradients are learning diagnostics, not proof of
one cause. A negative final result at 80 visits does not prove the architecture
cannot learn. Same-generator synthetic gains still require later real-family
transfer evaluation.

## Commands and resume

Intended result root: `results/hyperspline_joint_zero_shot_comparison/v1_seed20261003`.
These module commands are the experiment payloads; remote execution must follow
the repository's separate exact-command approval and SLURM workflow. Preparation
is CPU-only; the remaining commands need the frozen TabICLv2 checkpoint and GPU
resources. The 512-teacher workload must be chunked/resumed within job limits.

```bash
PY=/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python
ROOT=results/hyperspline_joint_zero_shot_comparison/v1_seed20261003
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison prepare --output-dir "$ROOT" --device cpu
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison teachers --output-dir "$ROOT" --max-teachers 32 --resume
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison train --output-dir "$ROOT" --arm fresh --resume
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison train --output-dir "$ROOT" --arm repeated --resume
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison train --output-dir "$ROOT" --arm teacher --resume
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison lock --output-dir "$ROOT"
"$PY" -m scripts.joint_preprocessing_zero_shot_comparison test --output-dir "$ROOT"
```

Repeat the teacher command until `teachers/complete.json` exists: the cap counts
newly completed fits, so each invocation advances past finished tasks. Every fit
resumes its saved AdamW state at the latest 25-step checkpoint. Use `train
--max-steps N` for an absolute shared-step cap, followed by `--resume` to continue;
the default saves every 100 shared updates and every evaluation. All budgets
and hashes remain unchanged across chunks. Serialize commands that write shared
root/config/reference files. A/B may train before teaching completes; C cannot.
The full default budgets are encoded in the runner, so no inherited 32-task
checkpoints or alternate hyperparameters enter these commands.

Local verification: 29 targeted CPU tests passed, covering synthetic-subset matching, seed/budget checks,
progressive teacher chunks and identity fallback, baseline parity, the three
arms, validation-only selection, exact interrupted resume through the common
optimizer reset, constant-feature filtering in single-view diagnostics, and
lock/test integrity. A native-prior CPU smoke run prepared
four repeated, two validation and two test tasks. These checks do not establish
GPU performance or generalization; experiment outcomes remain pending.

## Execution authorization, 2026-10-03

Prepare banks on dsiofir01 with `runnohup`. Execute the GPU phases serially:
fresh/direct, repeated/direct, independent teachers, teacher-then-direct,
validation lock and final test. This puts the teacher-free comparison first
and avoids concurrent writes to shared configuration/reference caches.
Use the uriofir profile on dsiuriofir01, one GPU, 32G host RAM and four CPUs,
with up to eight hours per submission and default BEGIN/END/FAIL email.
Previous 10k-update pilot arms requested eight hours; the smaller 8k-update
fitting diagnostic took 41m40s, and the 32-task three-rate fitting/distillation
run took 2h14m07s. Those support resume support and chunking, not a precise ETA
for this larger bank. Bank/fitting/validation workload and generation cost differ.
Sequential continuation jobs may resume after a preceding wall-time expiry;
completed phases return immediately, and saved settings/checkpoints stay fixed.
