# Full-network learning from scratch with teacher references

Status 2026-10-03: user authorized implementation and GPU execution. Local
verification passed; submission is pending. This is the next diagnostic; the
broader diversity/repetition/teaching plans remain unexecuted.

## Question and closest prior experiments

Can the complete current context-conditioned hypernetwork discover beneficial
transforms from true-label prediction loss alone? The 32-task direct diagnostic
reduced mean fitting NLL only 2.07%. Independent raw-parameter fitting reduced
it 45.78% at primary lr .001, but bypassed the encoder. Subsequent prediction
fine-tuning reduced mean NLL 27.94%, starting from taught weights; that does
not establish learning from scratch. This diagnostic adds separate full
hypernetworks per task and a shared full hypernetwork, all freshly initialized.
Earlier single-dataset conditioner studies are related (research history
section 3); the change is this current joint family, raw-context encoder and
exact frozen-TabICL prediction path.

## Fixed tasks and teacher controls

Reuse the completed bank in
`hyperspline_joint_learning_diagnostic/v1_seed20260930/` and independently
fitted maps in `hyperspline_joint_fitting_capacity_diagnostic/v1_seed20260930/`.
Choose the eight lowest training task IDs before inspecting their scores.
No generation changes or task selection by teacher benefit.

| Task ID | Context rows | Fitting query rows | Filtered features | Classes | Fixed view |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4000000000 | 435 | 77 | 79 | 2 | 6 |
| 4000000001 | 89 | 39 | 24 | 2 | 1 |
| 4000000002 | 179 | 77 | 27 | 9 | 6 |
| 4000000003 | 128 | 128 | 80 | 8 | 0 |
| 4000000004 | 870 | 154 | 78 | 2 | 2 |
| 4000000005 | 108 | 20 | 63 | 2 | 2 |
| 4000000006 | 358 | 154 | 63 | 4 | 6 |
| 4000000007 | 217 | 39 | 36 | 6 | 2 |

The bank uses the original mix_scm prior (70% MLP, 30% tree), 5--100 raw
features, 2--10 classes and coverage_expanded observation transformations.
Its training source seed is 181001. This observation mode adds nonlinear
feature distortions, rounding and dead zones, without the later structural
noise/missingness changes. Rows, labels, filtering, bounds and views stay fixed.

Replay all 24 saved selected independent maps (eight tasks x lr
.001/.003/.01) through the current backbone. Verify source bank, code and
reference hashes, checkpoint fingerprints, fresh identity loss and teacher
loss within absolute tolerance 1e-4*(1+abs(expected)). Failure stops the run.
Export scores and checkpoint hashes to `teacher_reference.csv`.

Primary reference: lr .001, 250 visits, original bounded family and initial
outputs. Report the best-fitting .003/.01 sweep separately as a stronger
fitting reference with its extra historical search cost. These references are
empirical achieved losses, not global optima. Reuse saves teacher refitting;
new work in this phase is replay and scoring. No taught checkpoint, teacher
function or teacher logits are supplied to the training function.

## New network training

For each lr in {.001, .0003}, train:

- Eight separate full JointPreprocessor("joint") networks, one per task,
  each for 250 updates.
- One full shared network on the same eight tasks, for 2,000 updates,
  exactly 250 visits per task.

All 18 networks copy identical fresh seed-zero weights. The encoder and every
head remain trainable, including in the separate condition. No source student
weights are loaded. Context features and labels alone generate parameters.
The fixed-view query CE on true labels, temperature 1, sends gradients through
the frozen TabICL training path. Teacher scores never affect gradients, task
order, stopping or checkpoint selection.

One task per update in both conditions avoids a batch-averaging difference.
Shared epochs use permutation seed 183001 + epoch, visiting every task once
before the next epoch. AdamW, constant LR, weight decay 1e-4, betas (.9,.999),
epsilon 1e-8 and global clipping 1.0. Total new CE updates/task presentations:
8,000. Matching numerical LR does not match effective steps between free
parameters and network weights. Separate optimizers and shared moments also
differ; failures do not isolate a single cause.

## Measurements and interpretation

Evaluate initialization and every 25 visits per task (25 optimizer updates
for separate networks, 200 for shared). Save per-task NLL, identity gain,
primary/sweep teacher benefit recovery, effective transform changes, gates,
bound use and saturation. Recovery is (identity NLL - network NLL) divided by
(identity NLL - teacher NLL); report it only where the teacher gives at least
1% relative gain. Do not clip negative or greater-than-one recovery values.
Aggregate mean NLL, geometric gain, wins/losses/ties (1e-6 tolerance), recovery
mean/median and number recovering at least half the primary teacher benefit.
Per-update logs include task ID, CE, all component gradient norms, preclip
norm, clip factor and cumulative clipping fraction.

Final checkpoints are primary. Also save the earliest best mean fitting-NLL
checkpoint, including initialization, as a descriptive fit diagnostic. For
shared training it is one global checkpoint, never a separate best checkpoint
per task. Report all learning rates rather than claim a fitting-selected
rate is validated. Full learned eight-view inference (temperature .9) versus
ordinary eight-view is secondary, at final and best-fitting states only;
training supervises one fixed view/slot per task.

Strong independent references plus weak separate full networks motivate
investigating full-network optimization/parameterization. Strong separate
and weak shared fits implicate sharing/conditioning/shared optimization as
candidates, without proving gradient conflict. Weak references and weak
networks are inconclusive about attainable benefit. Strong fitting in both
network conditions supports proceeding to new-task transfer experiments.

All query labels are reused for fitting. This permits memorization and cannot
establish new-row or zero-shot benefit. There is no validation/test checkpoint
selection in this diagnostic. The previously reserved real tasks remain out
of this experiment. No 1,000-visit extension is included in this first run.

## Execution and outputs

Runner: `scripts/joint_preprocessing_fromscratch_diagnostic.py`.
CPU checks: `tests/test_joint_preprocessing_fromscratch_diagnostic.py` plus
existing fitting and training-path tests. Resume saves model, optimizer,
CPU/CUDA RNG and best state every 50 optimizer updates and evaluations;
trims uncommitted CSV rows and skips completed networks. Hashes/settings
must match on resume.

Result root:
`hyperspline_joint_fromscratch_diagnostic/v1_seed20261003/`.
Root config, `teacher_reference.csv`, completion; `lr0`/`lr1` contain
`separate/<task_id>/` and `shared/`, each with training/evaluation CSVs,
state and selected checkpoints, selected reports, ensemble CSV and completion.
Root `comparison.csv` and completion summaries combine separate/shared final
and best-fitting scores without selecting a task-specific shared checkpoint.

Local verification: 23 targeted checks passed across the new diagnostic and
existing fitting-capacity, fixed-task learning and warm-start diagnostics.
Coverage includes teacher replay/integrity, matched task exposure, exact
interrupted resume, preserving initial weights, score-only teacher isolation,
and fresh-head/encoder gradients through an actual tiny frozen TabICL.

Use uriofir, target dsiuriofir01, one GPU, 32G RAM, four CPUs, two hours,
BEGIN/END/FAIL email. The earlier diagnostic with 8,000 CE task presentations
and broader evaluation took 1:19:58; this requests a bounded margin for the
same training exposure, teacher replay and serialization. Resume is supported.

Exact experiment command:

```bash
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_fromscratch_diagnostic --bank-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_learning_diagnostic/v1_seed20260930 --teacher-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_fitting_capacity_diagnostic/v1_seed20260930 --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_fromscratch_diagnostic/v1_seed20261003 --device cuda --tasks 8 --visits 250 --lrs 0.001 0.0003 --evaluate-every 25 --save-every 50
```

Synchronize only reviewed experiment files on the current branch. All remote
operations use exact-command built-in approval under AGENTS.md. Submission
will record the revision, job ID and exact stdout/stderr paths here.
