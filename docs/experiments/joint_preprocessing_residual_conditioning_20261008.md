# Native residual conditioning comparison, 2026-10-08

Authorized by the user on 2026-10-08. Implementation and local CPU validation
completed; remote preparation/submission pending. One seed, two arms, serial GPU
execution. This tests richer conditioning within a shared zero-shot residual
preprocessor; there is no target-dataset optimization or teacher supervision.

Closest evidence: the 2026-10-07/08 fresh 40/160 study, whose large arm learned
only +0.146% mean validation NLL from its initial blend. September 23/24 input
preservation optimized separate target adapters. Here the generator is shared,
its inputs are context only, and deployment requires no gradients. See
[joint_preprocessing_weak_link_review_20261008.md](joint_preprocessing_weak_link_review_20261008.md)
for the diagnosis and qualifications; no weak-link hypothesis is yet proven.

## Data and update budget

Reuse all five hash-verified panels from
`results/hyperspline_joint_dataset_diversity/v4_seed20261007`, fingerprint
`7149269ed57b83ce024a07934f293bae6b69b80b0bd41bba1686c997a9f149f1`.
Both arms train on the same 160 real source groups, with exactly the old large
arm's source order, episode seeds, row caps, context/query partitions and labels.
The small 40-source bank is retained only to reproduce the old paired row-cap
schedule. It does not create a third training arm. Never reuse old model states
or model-dependent reference predictions.

- Seed 0, 4,096 optimizer updates, four fresh episodes/update: 16,384 episodes.
- Scheduled total rows: 128/256/512/1024, context fractions 0.5/0.7/0.85.
  Queries are the remaining rows; integer counts and source-dependent row caps
  are unchanged from the prior sampler, not new independent query draws.
- AdamW, LR 0.0003, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8,
  gradient-norm clipping 1.0. Mean four-episode CE, temperature 0.9, no LR schedule.
- Evaluate at update zero and every 512 updates; durable resume state every 25.
- Source probes: 40 common-training and 40 additional-training source episodes.
  Validation: the existing 25 disjoint source groups, two fixed episodes each.
  This is reused development validation, not independent confirmation.

## Views, heads, and conditioning

Fit the ordinary 16-view EnsembleGenerator on context only, with its normal
none/power methods, feature permutations, class permutations, random seed zero,
unique filtering, and missing-value handling. Per method, keep views 0-3
unchanged and residual-adapt views 4-7. Eight untouched + eight adapted = the
same sixteen native views. Align numerical columns and output classes through
each permutation. Never replace native categorical values. Mask the numerical
correction at originally missing cells, preserving their native imputed values.

The generator uses the inherited 64-wide, four-head row/class/column attention
encoder, two view-slot embeddings, affine heads, 20-control-point cubic monotone
spline, eight-unit neural residual and rank-four bounded mixing. Generate maps
once from context and apply them after each adapted view's native preprocessing.
Do not restandardize native numerical inputs by raw-context moments. Subtract
an identically evaluated spline anchor so zero heads yield exactly zero
correction; spline/neural gates initialize to 0.1, mixing gate to zero. Gates and
outputs are not both zero. Baseline identity must hold in actual deployment.

Raw arm: the existing encoder's original numerical cells and context labels.
Backbone arm: enrich each numeric cell's four raw descriptors with its aligned
frozen TabICL column/group embedding and its full-feature row embedding, including
categorical inputs. Full context is the unshuffled native 'none' representation.
Extract only context rows with original context labels. For 'same' circular
grouping, token (column-1) contains that column in its first group channel; for
'valid' grouping use floor(column/group_size). Account for the leading CLS tokens.
Average the row's CLS vectors, normalize column and row representations, and
project the enriched cells to 64. Then use the same class-label pooling and
column attention as the raw arm. The embedding width is read from the frozen
checkpoint, not guessed. Head, slot and compatible encoder weights are paired;
record both parameter counts and runtime because encoder size/compute differs.

TabICL weights remain frozen. Extract context features in FP32 in training and
deployment. No query rows, labels, missing masks or query statistics condition
the generator. Native preprocessing itself still transforms query inputs using
its context-fitted state. Prediction may of course depend on query features.
Train through each adapted frozen-backbone view with an exact staged chain rule
matching full autograd; one backbone activation graph is live at a time.

## Logs, selection, and operational gate

Log CE, update timing, clipping, all parameter-group gradient norms and exact
source/episode presentations. Every evaluation logs dataset/episode NLL,
accuracy, wins/losses, mean and median gains, >1%/>5% harms, correction RMS,
spline/neural gates, mixing norm and emitted maps on a common 17-point grid.
Report variation of these signatures across episodes (not proof that variation
is useful). Source probes measure transfer across episodes of training sources,
not loss on the optimized batch. Baseline-only model references are shared and
fingerprint-locked. The initial model equals ordinary16, so improvement from
initialization also equals improvement over that baseline.

Select minimum arithmetic mean dataset validation NLL, with update zero eligible.
The final equal-update comparison is primary. For a selected model to justify
fresh confirmation, require >=0.5% reduction in mean NLL, >=18/25 wins, positive
median gain and a neighbouring checkpoint that also passes. Record gain/loss
distributions and uncertainty; one seed and reused validation cannot establish
the thesis with certainty. A/B isolates the richer encoder in this new residual
formulation. Comparing with old results also changes the view formulation.

CPU checks passed: 52 distinct targeted checks across the new residual runner,
ensemble-objective and dataset-diversity regressions. Covers actual tiny TabICL
in all three grouping modes, exact native16 identity with missing/category cells
and a finite extreme query, full/staged gradient equivalence, query isolation,
categorical conditioner sensitivity, head pairing, bit-exact resume, shared
baseline references, completed report locks and preparation integrity. A local
Windows atomic-file replacement failure passed on retry; production Linux code
was not changed to accommodate it.

Before any real updates, a GPU preflight audits both arms on the known
Credit_Risk_Modeling validation episode at split zero, comparing FP32 training
and deployment, exact default-AMP baseline identity, finite gradients and encoder
gradients after two disposable diagnostic updates. Saved initial models receive
zero updates. Also audit the first sampled training episode before each run.
A failed preflight must prevent training and the next array arm from proceeding.

Result root: `results/hyperspline_joint_residual_conditioning/v1_seed20261008`.
SLURM: uriofir / p_uriofir / ug_uri_ofir, node dsiuriofir01, 1 GPU, 4 CPUs, 32G RAM,
6 hours per arm; serial array 0-1%1, mail notifications enabled. Same old view
forward/replay count with added context feature extraction; prior large run took
4:09:54. Resume if the allotted time proves insufficient. No confirmation-test
bank or synthetic mixture is opened in this experiment.
