# Independent fitting and hypernetwork learning diagnostic, 2026-09-30

Status: user authorized execution. Implemented locally in
`scripts/joint_preprocessing_fitting_capacity_diagnostic.py`; 21 targeted tests
passed across the new runner and its existing diagnostic/pilot dependencies.
This includes an actual tiny frozen TabICL backward pass, matching initial
generated parameters, preserved bounds/rank, independent task parameters,
optimizer resume equality, conditional teaching, and source integrity checks.
Remote synchronization/submission is pending at this document's creation.

## Question and scope

Can the current numerical transform family find useful fitting changes when
its parameters are optimized independently per task? If so, can the current
shared hypernetwork learn those functions when supplied explicit targets?

Closest prior work: the 32-task shared learning diagnostic (`31595974`, commit
`df5b8ee`), and earlier DirectSpline headroom/teacher/conditioner experiments
(research history section 3). This matches the current joint transform family,
raw labelled-context encoder, same synthetic task bank, initialization and
training view; earlier per-dataset teachers used different formulations.
The earlier descriptor conditioner failed to predict held-out teacher curves,
which does not settle seen-task learning by this encoder. The cost is justified
by the unanswered fitting question, not by a search for a new target-dataset
pipeline. No learned deployment gate is introduced.

All scores/choices here use reused fitting examples. They are capacity and
optimization diagnostics, not unseen-query or zero-shot performance evidence.
Within-task query generalization and unseen datasets remain later questions.

## Frozen source and references

Read only:
`results/hyperspline_joint_learning_diagnostic/v1_seed20260930/`.
The source is completed: 32 fitting tasks, 2,000 updates, four tasks/update,
250 exposures/task, seed 0; selected step 1,500. Read only its train bank and
source manifest/config/completion/reference/evaluation CSVs. The 128 validation
tasks and old synthetic test banks are not opened by this runner. No bank is
generated or modified. Source bank hash:
`4ca00f5d79810f43d4db13702284e15b5b6874f99555789e2275fb3796e32272`.
The backbone is frozen TabICLv2, hash
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.

The source uses mix_scm/coverage_expanded, 5–100 original features, 2–10 classes,
and context/query splits from 128/256/512/1,024 total rows and .5/.7/.85 context
fractions. Reuse context-constant filtering, actual feature/class shuffles and
the fixed training view for each task. Compare against the source's matched
single-view standardized identity. Ordinary TabICL's none/power eight-view
ensemble is retained only as a secondary deployment reference; neutralizing
the learned map is not the same as restoring that ordinary pipeline.

Source/backbone/model/inference/runner hashes, bank IDs and view schedule are
validated before fitting. New source or settings require a fresh result root.

## Stage 1: independent numerical parameters

Start from exactly the raw outputs of the untrained seed-0 joint generator for
each task. Replace its encoder with a parameter-free placeholder and each head
with trainable tensors of raw per-column/per-slot outputs. Keep the production
`generate()` bounding maps and `apply()` code. Affine bounds, monotone K20 cubic
spline, gated 1–8–1 scalar neural residual and rank-at-most-four mixing are
unchanged. No unconstrained full mixing matrix or new function family is used.
No parameters are shared across tasks. Initial parameters must exactly match
the generator; initial loss must agree with the source identity within numerical
tolerance.

For each of the 32 tasks, fit 250 visits separately at learning rates 0.001,
0.003 and 0.01. AdamW: weight decay 1e-4, betas .9/.999, epsilon 1e-8; global
gradient clip 1.0. The 0.001 arm is the primary exposure-matched comparison.
The other rates are a bounded optimizer sanity check. The three-rate search
costs three times the fitting exposures, and best-rate choices are hindsight
diagnostics, not a fair equal-budget deployment claim. Matching numerical rates
does not imply equal effective step sizes in raw outputs and network weights.

Log training loss and gradient norms every visit. Evaluate fitting loss,
effective branch magnitudes, gates and saturation at initialization and every
25 visits; save resumable optimizer/model/best state at those points. Step zero
is eligible; earlier ties are retained. Save both final and best recorded fitting
loss. Full-ensemble evaluation is performed for each selected task/rate map.
Only the fixed view/slot receives direct loss supervision; ensemble scores must
therefore not be used as the sole fitting-capability metric.

`fitting_comparison.csv` compares independent final/best losses with the shared
model at final step, source validation-selected step, and each task's hindsight
best recorded shared checkpoint. These answer different selection questions.
`fitting_summary.json` reports mean/geometric losses, wins/losses/ties (absolute
NLL tolerance 1e-6), and task gains over 1%. A weak direct fit is not proof that
no beneficial transformation exists; the optimizer is not a global oracle.

## Stage 2: learn demonstrated functions

For each task, select its lowest recorded fitting loss across the three rates,
including initialization. Use that map as a teacher only if the paired fitting
NLL reduction is at least 1%, with the same 1e-4 stabilizer as the source scores.
Otherwise use the initialized standardized identity function. These usefulness
labels use fitting data and cannot define deployment routing or population task
usefulness. Cache targets, decisions and selected rate/visit in `teachers.pt`.

If fewer than four tasks have 1% teachers, save an explicit skipped-stage result.
Otherwise train a fresh seed-0 joint hypernetwork on all 32 contexts, including
the neutral targets. Targets are the teacher's transformed values on the fixed
context plus fitting-query feature rows, for the task's supervised slot. Parameter
generation still reads context features/labels only. Different parameterizations
can realize similar functions, so raw parameter matching is avoided.

Loss per task: mean squared difference between generated and teacher-transformed
values, divided by max(teacher-versus-standardized-identity MSE, 0.01). The floor
corresponds to RMS 0.1 in standardized feature units, controls the weighting of
small changes, and is fixed before the run. Average four task losses per update.
Train 2,000 updates using the source's deterministic epoch schedule, AdamW lr
0.001, weight decay 1e-4 and clipping at 1.0. No gradients go through TabICL in
this stage's training objective; it is used only for scoring. Cache detached
teacher logits for prediction KL diagnostics.

Every 100 updates report function error, relative error on beneficial targets,
prediction KL, fitting NLL and gains, ensemble NLL, gain recovery and branch
diagnostics for all 32 tasks. Save state every 50 updates. Select earliest best
mean normalized fitting function error, including step zero; report both final
and selected scores. Gain recovery is (identity NLL - student NLL) / (identity
NLL - teacher NLL) on beneficial targets, descriptive and not clipped to [0,1].
Log mean and per-task values so a good average cannot hide poorly learned tasks.

Successful function and prediction recovery would demonstrate learning of these
seen targets, supporting the current encoder/model's capacity for them. A gap
between this and downstream-loss training motivates investigating that training
procedure. Failure leaves conditioning, capacity and optimizer explanations
unresolved; it does not prove the family cannot learn. No conclusions about
unseen tasks or real transfer follow from either result.

## Artifacts and execution

New result root:
`hyperspline_joint_fitting_capacity_diagnostic/v1_seed20260930/`.
Root config/source hashes, fitting comparison/summary, teachers and completion;
`fits/<task_id>/lr<index>/` contains traces, selected weights, resume state and
completion; `distillation/` contains training/evaluation curves, per-task rows,
resume state, selected weights and a separate selected report. Resume validates
fingerprints, skips completed fits and trims rows after the last saved update.
Download small CSV/JSON summaries first; banks, states and target arrays remain
remote unless needed.

Use uriofir, one GPU, 32G RAM, four CPUs, three-hour wall time, email enabled.
Evidence for exceeding one hour: the source took 1:19:58 for 8,000 differentiable
task visits plus evaluations. This run uses 24,000 direct visits plus conditional
function learning and about 21 x 32 full-ensemble evaluations. Removing the encoder
from direct fitting and the backbone from teaching lowers cost, but one hour
cannot be justified. Resume is available if three hours is insufficient.

Exact experiment command:

```bash
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -m scripts.joint_preprocessing_fitting_capacity_diagnostic --source-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_learning_diagnostic/v1_seed20260930 --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_fitting_capacity_diagnostic/v1_seed20260930 --device cuda --fit-lrs 0.001 0.003 0.01 --fit-steps 250 --fit-evaluate-every 25 --distill-steps 2000 --distill-evaluate-every 100 --distill-lr 0.001 --teacher-gain 0.01 --min-useful-tasks 4
```

Every remote operation uses the exact-command built-in approval required by
AGENTS.md. No independent remote development, direct login-node GPU work, or
additional seed is needed.
