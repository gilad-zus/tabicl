# Joint preprocessing: architecture and learning diagnosis, 2026-09-30

Status: local review of the completed seed-0 runs. Full training/validation
CSVs were downloaded to their mirrored paths under the local results root.
The user authorized the diagnostic on 2026-09-30. Implementation and local
checks are complete; remote submission is pending below.
Computed curve summaries and parameter counts are in
`joint_preprocessing_learning_diagnostic_20260930.json`.

## What the model generates

`src/tabicl/_hyperspline/joint_preprocessing.py` implements one shared
hypernetwork trained across datasets. At deployment it consumes labelled
context features and labels, generates transformation parameters, and applies
the same parameters to context and query rows. Query features do not condition
parameter generation. Query labels are used only for meta-training/scoring.

The encoder is shared by restricted, joint, and no-spline arms. It embeds four
per-cell features (standardized value, square, absolute value, validity) and
23 distribution summaries per column into width 64. Four-head attention mixes
columns within each row. Per-class means and spreads of those cell embeddings,
plus class frequency, are encoded and pooled using attention across classes.
Attention then mixes the resulting column tokens. There is one final
64-dimensional token per column, informed by the whole labelled context.
There is no row-to-row attention layer; rows are pooled within classes.

Shared output heads generate per-column transformations for two numerical
ensemble slots. Restricted generates affine corrections and a gated monotone
cubic K20 spline. Joint additionally generates a scalar 1-8-1 tanh network for
each column/slot and a rank-at-most-four residual mixing matrix. No-spline
retains affine, scalar neural residual, and mixing, but omits the spline.
The restricted transform is univariate when applied, but its parameters are
still dataset-conditioned; this experiment does not compare independent
column encoders with a dataset encoder.

Affine shifts and log-scales are bounded to [-1,1]. The scalar neural residual
is bounded to one standardized unit and has a generated gate. Mixing has norm
at most 0.1. Spline and neural residuals are evaluated from the same affine
input before mixing. Imputation/encoding and operation order are fixed.
This is a constrained numerical transform family, not generation of arbitrary
preprocessing programs. Both slots initialize at matched context standardization.

Trainable parameter counts: joint 85,287; restricted 83,012; no-spline 83,987.
All three contain the same 81,454-parameter encoder. Thus similar arm results
do not establish that increasing encoder capacity would be ineffective.
Frozen TabICLv2 supplies the query loss; gradients update the hypernetwork,
not the TabICL weights.

## What the completed learning curves show

Validation reductions below are geometric NLL reductions versus matched
identity on the same 512-task bank (positive is better).

| Arm | Best validation step | Best reduction | Reduction at step 10,000 | Mean training NLL, first/last 1,000 steps |
| --- | ---: | ---: | ---: | ---: |
| Joint | 1,000 | 0.351% | -1.197% | 0.90653 / 0.90199 |
| Restricted | 3,000 | 0.448% | -0.752% | 0.90713 / 0.90085 |
| No spline | 8,000 | 0.308% | 0.221% | 0.90958 / 0.89806 |

Preclip gradient norms exceeded the clip threshold in 18.21%, 26.61%, and
18.42% of steps, respectively. This is an observation, not evidence by itself
that the learning rate is unstable or that gradients reach every branch.

Each run used 40,000 freshly generated tasks, four per update, with constant
AdamW learning rate 1e-3. This is not repeated training over a fixed small set
of datasets. More data volume could still help, but longer training under the
current joint recipe demonstrably did not improve its monitored metric.
Generator diversity and task count are separate questions.

Current logs cannot diagnose a conventional train/validation gap:

- Training averages raw query cross-entropy on a sampled ensemble view of
  newly generated tasks, without the inference temperature adjustment.
- Validation averages task log ratios to identity, using all available
  ensemble views, logit averaging, and temperature 0.9 on fixed tasks.
- Absolute NLL, mean task log ratios, and single-view versus ensemble loss
  are different objectives. A decrease in one need not improve the others.
- Total gradient norms were logged, but per-branch gradient/transform norms,
  gate saturation, and a fixed seen-task learning curve were not.

The held-out test still has a small gain versus ordinary TabICL: joint 0.350%
geometric NLL reduction; its task-bootstrap interval corresponds to about
0.142% to 0.556% reduction. This quantifies uncertainty over tasks from the
tested generator, not variability across training seeds or real datasets.
It neither identifies overfitting/underfitting nor proves the dataset
conditioner is responsible for the gain.

## Recommended next diagnostic

Closest precedent: the earlier selected-checkpoint DirectSpline
generalization audit (research history section 9). Its original multiclass
and regression `generalization_audit_summary.json` files were rechecked.
Those compared selected per-dataset spline/line adapters on train episodes
and recorded OOF/test outcomes; they did not isolate this shared hypernetwork's
optimization and its single-view/ensemble objective difference.

Run a small learning sanity experiment with the current joint architecture:
32 fixed synthetic meta-training tasks, 128 independent fixed validation
tasks, model seed 0, four tasks per update, at most 2,000 updates, and
diagnostics every 100 updates. Use one shared hypernetwork across all tasks,
without per-dataset optimization. Keep current transform bounds and optimizer
for the first diagnostic so the experiment tests the existing formulation.

On both panels, log the actual training-view objective and the deployed
ensemble NLL/log-ratio under identical evaluation settings. Record identity
and ordinary references once. Also log per-module gradients, effective
spline/neural corrections, mixing norm, and gate/bound saturation. Training
panel scores explicitly reuse fitting query labels and are not generalization
evidence. Do not require near-zero loss: the backbone is frozen and the
transform family may have limited attainable headroom.

- Strong seen-task improvement with poor independent-task improvement supports
  investigating generalization and task diversity.
- Better single-view loss but worse deployed ensemble loss supports a controlled
  objective-alignment experiment.
- Little progress even on repeated tasks motivates checking optimization,
  gates/constraints, representational capacity, and attainable transform
  headroom; it does not uniquely prove that the MLP is too small.

Then change one axis supported by the diagnosis: encoder width 64 versus 128,
generated scalar MLP width 8 versus 32, or task-pool size/diversity at matched
update budgets. Do not change all of them together. Keep the already-inspected
1,024-task test bank out of these decisions; a revised model needs a new
untouched confirmation bank. Real-family transfer remains a separate question.

## Locked execution protocol

Runner: `scripts/joint_preprocessing_learning_diagnostic.py`. Model: unchanged
`JointPreprocessor("joint")`, width 64, four attention heads, K20 cubic spline,
generated 1-8-1 scalar networks, rank-four mixing, seed 0, identity initialization.
AdamW: learning rate 1e-3, weight decay 1e-4, betas (0.9,0.999), epsilon 1e-8,
gradient clipping at norm 1. Train 2,000 updates, four tasks per update.

The new fixed banks use `mix_scm` / `coverage_expanded`, 5–100 features,
2–10 present classes, total rows 128/256/512/1024, and context fractions
0.50/0.70/0.85. The 12 size/fraction combinations are covered nearly equally.
Train bank: 32 tasks, seed 181001, task IDs starting at 4,000,000,000.
Validation bank: 128 tasks, seed 182001, IDs starting at 5,000,000,000.
Banks are generated once on CPU within the experiment, hashed, and cached.
The prior pilot's test bank is not read.

Each training epoch shuffles all 32 tasks using seed 183001 + epoch and
consumes eight batches of four; 2,000 updates are 250 passes over the task bank.
The chosen training view is fixed per task using the original seed-0 view rule.
Its labels and view are repeated intentionally to test the current objective's
ability to fit. Step 0 and every 100 updates evaluate both panels using the
same single-view training path and the same deployed ensemble path. Checkpoint
selection uses only validation ensemble log ratio to matched identity, with
step 0 eligible and earliest ties retained. No early stopping is applied.

Artifacts relative to the remote/local results roots:
`hyperspline_joint_learning_diagnostic/v1_seed20260930/`. `references.csv`
caches matched single-view identity, ensemble identity and ordinary TabICL.
`training.csv` records every update's loss, total preclip gradient norm and
gradient norms by module. `evaluation.csv` contains paired-panel curves;
`evaluation_tasks.csv` contains per-task NLL, ratios and actual branch/gate/bound
diagnostics. The runner also writes bank hashes, source revision, config,
resumable optimizer/model state every 50 steps, selected weights, and completion
metadata. Resuming trims logs after the saved state before replaying updates.

Local validation: 15 targeted tests passed, including tiny actual frozen TabICL
forward/backward, bank separation, effective identity corrections, no validation
gradient use, and exact interrupted/resumed model equality. A subsequent targeted
check passed after adding explicit frozen-backbone and saturation guards.
Use SLURM uriofir, one GPU, 32G host RAM, four CPUs and two-hour maximum,
with email enabled. The earlier 1,024-task five-method report took 34 minutes;
21 passes over 160 tasks plus 2,000 differentiable updates justify exceeding
the default one-hour allocation. Resume is available if needed.
