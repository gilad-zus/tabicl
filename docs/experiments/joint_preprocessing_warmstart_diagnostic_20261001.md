# Taught initialization: imitation versus prediction fine-tuning

Agreed 2026-10-01 after the user requested the proposed experiment. Local
implementation is complete: seven new CPU checks and 21 existing targeted
checks passed. Synchronization/submission is pending; no new result is claimed.

## Question and closest evidence

The completed fitting-capacity diagnostic reduced normalized function error
80.60%, but recovered only 5.38% mean per-task teacher prediction benefit.
Function error was still falling at update 2,000 while prediction recovery
remained about 5–6% late in training. Its fitting-only scores cannot establish
that the teachers transfer to fresh rows. See
[verified results](joint_preprocessing_fitting_capacity_results_20261001.md).

This experiment tests whether optimizing prediction loss from the taught
initialization converts its partially learned functions into better predictions.
The earlier fixed-task shared model optimized prediction loss from identity
initialization. The previous function-teaching model optimized transformed-value
MSE. The new comparison changes the objective from the same taught weights,
with an identical additional budget and learning rate for both arms. Historical
teacher prediction/conditioning and split-specificity studies are summarized
in research history section 3; this is a current-family training diagnostic,
not a claim that teacher learning is a new research idea.

## Frozen data and initialization

Reuse the original completed fixed-task diagnostic bank:
`hyperspline_joint_learning_diagnostic/v1_seed20260930/`.
It has 32 repeated fitting tasks and 128 separate synthetic validation tasks,
generated with `mix_scm`, `coverage_expanded` observations, 5–100 features,
2–10 classes, sequence lengths 128/256/512/1024 and context fractions
0.5/0.7/0.85. Query size is total sequence length minus context size; use the
saved episode dimensions without regenerating data. Source seeds are 181001
and 182001, with disjoint task ID ranges. Validation has previously been
inspected and is development validation, not a final test.

Reuse the saved targets and selected taught checkpoint from
`hyperspline_joint_fitting_capacity_diagnostic/v1_seed20260930/`.
The taught selected update is 2,000. All 32 teachers qualified on fitting loss;
their raw task-specific transforms selected lr 0.01/update 250. Each task's
fixed view and active slot are inherited exactly. The other slot is not
directly supervised for that task, as in the source diagnostic.

Both arms use the same current joint hypernetwork and frozen TabICLv2
checkpoint. Parameters are generated using context features and context labels
only. Query labels enter training losses, never parameter generation. No
per-target optimization occurs during validation. Initial weights, teacher
arrays, code, banks and references are checked and hashed. The source data and
source run are never modified.

## Two matched branches

| Arm | Training objective |
| --- | --- |
| `imitation` | Mean squared transformed-value error divided by max(teacher-versus-identity MSE, 0.01), matching the prior teaching objective. |
| `prediction` | Frozen TabICL single-view query cross-entropy using the inherited fitting view, raw logits and temperature 1. |

Each arm adds 2,000 updates, four tasks per update, AdamW lr 0.0003,
weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8 and global gradient
clipping at 1.0. Reset optimizer moments in both arms. Use constant LR for
both, so objective is the intended difference. The lower common LR is a
controlled continuation choice; this experiment does not isolate LR's effect
against the preceding lr 0.001 teaching run.

Both arms continue the same deterministic epoch schedule after source update
2,000 (shuffle seed 183001 + epoch). Each task receives 250 further visits.
The two arms run sequentially on one GPU. The historical prediction-from-
identity run is contextual evidence, not an equal-total-compute control for
the two-stage recipe.

## Measurements and selection

Evaluate initialization and every 100 updates on all 32 fitting and 128
validation tasks. Log per-task and aggregate single-view NLL versus matched
standardized identity; full learned-eight-view ensemble NLL versus ordinary
eight-view TabICL and matched identity; geometric and median gains;
wins/losses/ties with absolute NLL tolerance 1e-6; harm counts over 1%, 5%,
10% and maximum harm. Ensemble inference retains temperature 0.9.

On fitting tasks also log function MSE, normalized function MSE, teacher
prediction KL, mean/median teacher-benefit recovery, and number recovering at
least half the teacher gain. These use reused fitting labels. Log effective
affine/spline/neural/mixing changes, gates, bound use and saturation on both
panels. Every update logs objective, sampled function error, per-component
gradient norms, preclip norm, clip factor and cumulative clipped fraction.
Sampled losses vary with minibatches; use the full fixed-panel curves to
assess learning.

For each arm select the earliest checkpoint minimizing mean validation
ensemble log ratio against ordinary TabICL, including update zero. Save and
report initialization, final and selected states separately. Both checkpoint
selection and arm comparison use development validation. No untouched final
test or real dataset evaluation is part of this diagnostic.

Save optimizer/model/RNG state every 50 updates and at evaluations. Resume
requires matching hashes and settings, trims log entries beyond the saved
update and skips completed branches. Selected reports are separate from the
chronological curves.

If prediction fine-tuning improves fitting more than continued imitation,
it supports objective mismatch or a useful optimization route. If imitation
catches up, precision/time contributed. Better fitting with weak validation
leaves transfer unresolved. Weak fitting in both leaves conditioning,
capacity and optimization explanations open. Neither outcome proves a
capacity limit or that every task has useful preprocessing headroom.
Check independently fitted teachers on unused query rows before scaling
teacher-based training; no existing fitting query can become a held-out row
retroactively. The proposed standard-plus-learned ensemble audit remains a
separate, unexecuted direction.

## Execution and artifacts

Runner: `scripts/joint_preprocessing_warmstart_diagnostic.py`.
Result root: `hyperspline_joint_warmstart_diagnostic/v1_seed20261001/`.
Root files include source hashes/config and completion. Each arm contains
`training.csv`, `evaluation.csv`, `evaluation_tasks.csv`, `state.pt`,
`selected.pt`, `complete.json` and `selected_report/`.

Request uriofir targeting `dsiuriofir01`, one GPU, 32G RAM, four CPUs and
three hours, with email enabled. Evidence for exceeding one hour: the prior
prediction-loss diagnostic took 1:19:58 for 2,000 updates and scoring both
banks; this run repeats full-panel scoring for two branches and has resume
support. All code and CPU checks run locally.

Exact experiment command:

```bash
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -m scripts.joint_preprocessing_warmstart_diagnostic --source-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_fitting_capacity_diagnostic/v1_seed20260930 --bank-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_learning_diagnostic/v1_seed20260930 --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_warmstart_diagnostic/v1_seed20261001 --device cuda --steps 2000 --evaluate-every 100 --save-every 50 --lr 0.0003
```

Every remote synchronization, inspection and launch goes through exact-command
built-in approval, as required by AGENTS.md. Submission identifiers and exact
stdout/stderr filenames will be recorded after successful submission.
