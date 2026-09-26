# Research memory: what we have already tried

Last reviewed: **2026-09-26**. Read before proposing the next experiment.
Purpose: avoid rediscovering old ideas, not enumerate every result or permanently rule
out approaches. Details and source paths: [research history](research_history_20260920.md).

Meeting reference, 2026-09-26: `thesis_meeting_brief_20260927.md` consolidates recent
results, matched-control qualifications, training protocols and formula changes.
Implementation check: recent crossfit replacement runs use AdamW weight decay
and structural constraints, not an explicit identity/curvature penalty; the
feature-expansion experiment separately uses a second-derivative penalty. The
input-preserving line is straight in arctan coordinates, not globally affine in
the original input. No new experiment was run or approved for this synthesis.

Implementation caveats/fixes are recorded separately in the
[2026-09-20 code review](code_review_20260920.md); these are not new experiment results.

## Keep the actual target fixed

Show that **learned spline curvature** helps per-dataset adaptation of frozen TabICLv2:
both versus ordinary TabICLv2 and versus the same pipeline without curvature.
Gains entirely from affine maps, mixing, context, ensembling, or identity fallback do
not satisfy this target. Those remain useful controls, not the desired replacement thesis.
Do not assume an expressive spline must beat a line under finite training.

HyperSpline is the separate zero-shot direction. Synthetic average improvement exists;
robustness and transfer are unresolved. Do not turn it into per-dataset optimization
without explicitly acknowledging the different goal.

## Already tried — check these before calling an idea new

Agreed 2026-09-26: controlled numerical preprocessing replacement pilot on the
shared eight development tasks. Four line/spline × standard/minimal arms, with
constant/cosine OOF selection in each family. Retains original categories, slots,
permutations and full context, unlike legacy lite. Minimal replaces numerical
power/outlier transformations with T-fitted standardization. Status: implemented,
local tests passed; submission pending, no results yet. See `preprocessing_replacement_20260926.md`.

W/L/T below means dataset/task wins, losses, ties; counts are not pooled across rows.
History section numbers point to the linked document above.

| Idea | What was already done / learned | History |
|---|---|---|
| Establish per-dataset spline headroom | Positive small teacher/headroom studies; transformations partly split-specific. Finding an oracle teacher is not learning a deployable selector. | §3.1 |
| Average teachers or predict a low-rank curve | Consensus, stability, descriptor sufficiency, and low-rank teacher projections were tested. Compressing known teachers worked better than predicting useful unseen-table teachers. | §3 |
| Add cross-column information / mixing | Cross-column and mixing-capacity studies, plus conditional adaptive columns, already exist. Mixing can help, but does not establish curvature benefit. | §3.1, §4.3 |
| Early stopping, lower/decaying LR, regularization | Checkpoint audit and cosine-scheduled validation selection, with/without identity regularization. Selection did not guarantee transfer to test or a restarted full-context fit. Do not propose “add validation” as missing. | §4.1 |
| Refit on all training rows after choosing duration | Already tested, including unconditional full refit. More labelled context does not guarantee the same optimization outcome. Distinguish refitting from frozen-adapter context expansion. | §4, §6.3 |
| Check whether validation correctly chose identity | Counterfactual replay audit exists, but all 13 cases were excluded for replay mismatch. It does **not** answer guard correctness; a valid replay would be a repair, not the first such idea. | §4.2 |
| OOF / A-B selection / more than one validation | Nested independent-selection audit, A/B cross-fitting, hard gates, blends, ensemble-selection and repeat-selection audits already exist. Selection optimism is real; guards sometimes discard useful gains. | §5.2, §6 |
| More bags or aggregate repeated fits | Bagging, repeat fits, pooled selection, and CPU bag-count audits were explored. Subsetting existing predictions is not equivalent to retraining with different folds. | §6 |
| Append validation rows as context without training on them | Already done with frozen adapters/preprocessing and a matched identity control. Often helped, especially regression. This is not a new proposed fix. | §6.3 |
| Confirm on more datasets / another seed / TabArena | Multiple confirmations and a feasible TabArena subset replication exist. Name the previous set/seed and new uncertainty; do not pool revisited datasets as fresh evidence. | §4.3, §6.3 |
| Use random forest at 1000 Elo | Already tried. RF lost every comparison; complete separation makes absolute rating distances regularization-sensitive. This does not calibrate our local Elo to published TabArena/Retouche Elo. | §4.3 |
| Binary loss versus AUC mismatch | Identified and motivated AUC-focused work. No broad completed positive binary result established here; check partial artifacts before rerunning or declaring binary impossible. | §4.3 |
| Unsupported query regions / bad interpolation | Support, hypothetical basis-leverage, and row/cell audits exist. Some multiclass association, not a universal regression explanation. Changing geometric support alone has not proved a predictive fix. | §5.1 |
| Uniform versus quantile/learned knots; degree/shape constraints | Early capacity/knot-placement/refinement and monotonicity studies exist. No general “fewer/more/learned knots always wins” conclusion. Check exact tested degrees/configurations before claiming coverage. | §3.1 |
| Different complexity for different columns | Adaptive-column models and exhaustive K4/K20 assignments on three-column visualizing_soil already tested. OOF chose [20,4,4], beating both uniform capacities there. Generality remains open; K4 cubic is not a line. | §4.3, §8.1 |
| Improve the training query sampler | Revised to variable 5–20% query fractions, preserving rare-class context without forcing balanced query labels. Memory chunking is not a fixed total-query cap. Caching/small adapter saving were also added. | §7 |
| Remove affine/mixing gains from the attribution | Matched affine/mixing ablations already done: raw spline vs control was MC 1/3/0, regression 3/1/0 on four tasks each. Pipeline wins alone are insufficient. | §7 |
| Arctan/Cauchy-style bounded coordinates | Arctan pipeline already tested. A line in arctan coordinates is itself nonlinear in original input; compare against that matched line, not only identity. | §8.2 |
| Free endpoints / direct spline output instead of identity correction | Already tested in direct-output arctan. Raw spline vs line: MC 1/3/0, regression 4/0/0 on development tasks. This is not an untried simplification. | §8.2 |
| Learn a line first, then add curvature | Staged matched continuations already tested. Later set: MC 5/4/1, regression 8/2/0; raw mean relative gains about 1.07% and 0.79%. Selected blending can reverse the regression aggregate benefit. | §8.2 |
| Audit training performance to diagnose overfitting | Saved selected-checkpoint train/OOF/test audit exists. Not a universal train-win/test-loss pattern; one such MC case and none in four regression tasks. Does not rule out overfitting or optimization difficulty. | §9 |
| Zero-shot HyperSpline and safety routing | Teacher prediction, query descriptors, synthetic coverage, and router studies exist. Synthetic frozen test 548/468/8; router validation predictability weak (~0.54 AUC). Real-data transfer and better representations remain open. | §3.2 |

## Latest pilot: results reviewed 2026-09-20

See [results and interpretation](representation_pilot_results_20260920.md) and
[specification](spline_representation_pilot_20260920.md).

- **Feature expansion completed, four tasks per type:** keep originals and append
  line/K8/K20 features, with no learned mixer or primary identity blend. Both spline
  capacities have 0 wins / 4 losses against ordinary TabICLv2 on multiclass. Regression K20
  wins 3/4 against line (+0.70% mean relative gain) and 3/4 against ordinary TabICLv2,
  but the latter mean gain is -0.15%. Do not propose this exact recipe as untried.
- **The initial augmentation itself hurts:** all four multiclass tasks and notably
  visualizing_soil (+8.69% error) are worse before learning. K20 has clear incremental
  curvature benefits on Seoul bike and IEEE80211aa-GATS, not a universal benefit.
  Training/OOF/test traces also show curvature generalization problems on cardiotocography
  and steel, so neither augmentation cost nor overfitting explains everything alone.
- **Multiclass K12 replacement completed:** raw K12 vs line 3/1, mean gain +0.58%;
  vs K20 2/2, mean gain -1.98%. Smaller is not established as better. Guarded 4/4
  versus ordinary includes an identity selection and must not be called four spline wins.
- **Regression K12 completed:** raw K12 is 2 / 2 versus its continued line (+0.26%
  mean), but 1 / 3 versus K20 (-0.61% mean). Lower uniform capacity is not established
  as better; see the representation-pilot update for the matched-control details.

Result directory: `openml_direct_spline_representation_pilot/dev8_20260920`.

## Agreed diagnostic, 2026-09-20: per-column curvature removal

[Specification](column_curvature_audit_20260920.md): reuse all eight development
tasks' staged K20 checkpoints, project one column at a time onto its T-fitted line
before the unchanged mixer, and compare cross-fitted OOF/test effects. No training,
no joint mask and no test-selected column choice. This follows earlier adaptive/
factorial work but isolates curvature removal from the current fitted model.
Implemented in commit `c03132f`; 22 local tests passed. Completed 2026-09-21 as job
**30954048**, `ds-column-curvature-260920`. OOF-selected removal was MC 1 / 3 and
regression 1 / 2 / 1 on test, with negligible-to-negative mean effects. Mixed individual
effects exist but OOF has substantial sign-transfer failures, often for tiny changes;
this does not show validation is inherently incapable of selecting useful curvature.
Removing all curvature worsens all four regression tasks (+0.06% to +0.57% error
reduction from retaining curves, measured relative to the projected-line errors),
agreeing with their separately trained-line comparisons. Eye_movements also has a
large curvature benefit (+10.58% versus projection). These are attribution checks
on existing models/splits, not fresh replication. Do not propose this exact
audit as an untried route to a deployable per-column selector. See
[audit results](column_curvature_audit_20260920.md).

Capacity interpretation clarified 2026-09-21: the earlier jointly trained soil
assignment [20,4,4], uniform K12/K20 staged continuations, and frozen-model
projection onto a line test different interventions. K4 retains cubic curvature;
projection removes it without retraining. The latter two results do not refute
column-specific capacity benefits, which remain supported only narrowly.

Results: `openml_direct_spline_column_curvature_audit/dev8_20260920`.
Remote log stem under `/home/dsi/zusmang/TabICL/tabicl/slurm_logs/`:
`slurm-ds-column-curvature-260920-30954048` (`.out` and `.err`).

## Before recommending more compute

Portfolio coverage checked 2026-09-21: among seven existing raw spline variants on
the eight shared development tasks, hindsight selection finds an own-control win
on 8/8, and a candidate beating both its own control and ordinary TabICLv2 on 7/8
(MC 3/4, regression 4/4). Cardiotocography remains uncovered. This is test-oracle
coverage, not a validation-selected method, best-no-spline-portfolio comparison,
or an evaluated prediction ensemble. Some wins are tiny. See
[coverage scope and artifacts](spline_portfolio_coverage_20260921.md).

Portfolio selection audit completed 2026-09-22: among seven raw spline candidates
and five distinct no-spline controls (seven entries, two duplicates), select each
family independently by absolute cross-fitted
OOF deployment error; ordinary TabICLv2 is additionally available as a shared
fallback. On the eight shared development tasks, selected spline versus selected
no-spline is 4 / 4 overall (MC 1 / 3, regression 3 / 1), with a negative median
gain (-0.02%) and a mean dominated by eye_movements. Both selected families beat
ordinary TabICLv2 on 6 / 8 tasks; including the ordinary fallback gives 4 / 3 / 1
for spline-or-base versus no-spline-or-base. This does not establish a spline HPO
method: the variants came from separate non-nested runs and each had internal
checkpoint selection. It does show a bounded common grid plus independent
evaluation would be the right formulation if continuing this direction. See
`spline_portfolio_selector_results_20260922.json` and its producing script.
Review correction 2026-09-22: equal search budgets were not established by duplicate
control entries. Follow-up compatibility check found no discrepancies: all candidates'
manifests agree on seed 20260915, four bags, source-manifest hash and outer split per
task; 256 available standard bag artifacts agree on saved validation/A/B indices
and recorded context counts (224 comparisons against 32 reference bags). K12 and
feature-expansion raw bags are absent locally; their compatibility is supported by
manifests and shared splitting/context code, not direct array comparison. Counts
remain 4/4. Different model families and checkpoint selection do not themselves
invalidate HPO; this remains development evidence, not equal-budget confirmation.

Headroom/selection follow-up 2026-09-22: OOF chooses the test-best spline on 4/8.
Best-test-spline versus best-test-no-spline wins 6/8 retrospectively, but MC gains
are eye +11.04%, theorem +0.055%, steel +0.147%, cardio -0.228% (log loss).
Selection is not the sole bottleneck. Final configuration selection pools all
outer-training OOF rows; the much smaller A/B halves select checkpoints. See the
selector results report for exact sizes and the OOF-versus-test ensemble/context
difference. A current-family selection-size/stability diagnostic is proposed, not
run; it would not substitute for retraining or fresh-dataset evaluation.

Pending controlled checkpoint-selection-size diagnostic, 2026-09-22: the prior
20% inner-validation/refit run changes both the fitted rows and selection rows,
so it cannot isolate whether selection evidence is too small. The new fixed-fit
experiment trains one K20 DirectSpline trajectory on 60% of each multiclass
outer-training split, then uses nested 20% and 40% labelled pools to select a
checkpoint or identity from that *same* trajectory and predicts the outer test
with the same 60%-fit context. It runs two deterministic split seeds on the
four prior multiclass development tasks. There is no full-context refit by
design. It can answer whether more selection labels alter checkpoint decisions
and improve the unchanged trajectory; it cannot establish the best normal
80/20 protocol, explain all configuration-family selection issues, or provide
fresh-dataset confirmation. Implementation:
`scripts/direct_spline_openml_fixed_fit_validation_size.py`.

Correction, 2026-09-22: the above 60%/20%/40% K20 job 30968626 was cancelled
while pending and produced no experiment result. It did not match the current
staged-curvature pipeline. Its replacement is
`scripts/direct_spline_staged_validation_size.py`: replay the four-bag K20
staged continuation from the saved A/B-selected direct-line states, with T
fixed at approximately 75% of outer training. On each trajectory, compare
checkpoint choice from its original A or B half (~12.5%) with choice from
the whole held-out A+B fold (~25%). Test both unchanged states with the same
T+A+B context. Keep the inherited line state eligible at step zero; report
matched line, identity, and both selected spline errors. This isolates the
effect of extra labels on the *continuation checkpoint* choice conditional
on the previously selected line states; it does not test the line-selection
stage or final configuration search. Four inspected multiclass development tasks
only. First job 30968845 failed before training due to a split-loop unpacking
error. Corrected job 30969291 completed 2026-09-22; results are in
`results/openml_direct_spline_staged_validation_size/mc_dev4_seed20260915_v2`.
The larger pool changed 13/32 continuation checkpoint choices. On outer test,
larger versus smaller selection improved steel and cardiotocography, but worsened
theorem proving and eye movements. Against the matched line, both choices helped
eye movements (~14.4% relative log-loss reduction) and only marginally helped
theorem proving; larger selection turned cardiotocography from a 2.86% loss to
a 1.04% gain over line, yet both remained worse than ordinary identity there.
Steel remained worse than line and identity. Thus selection-half noise contributes
on some trajectories, but doubling these labels is not a general fix. This is a
four-task development diagnostic conditioned on the previously A/B-selected line
states, not a fresh confirmation or a clean test of the line-selection stage.

Literature/claim boundary checked 2026-09-21: [TFM-Retouche](https://arxiv.org/html/2605.06047v1)
already learns nonlinear input adaptation (default multiplicative DCNv2 cross blocks;
residual MLP alternative), not merely affine scaling/mixing. [BETA](https://proceedings.mlr.press/v267/liu25cn.html)
also precedes this work with learned input encoders for TabPFN. Our matched-line
comparisons establish incremental curvature effects in our pipeline, not superiority
to those nonlinear adapters. Regression is a promising empirical subgroup, and
eye_movements a strong observed case; neither is a validated train/validation-only
rule predicting spline benefit on unseen datasets. Do not frame generic successful
input adaptation or adding nonlinearity as the novel spline contribution.

State briefly: **closest previous experiment → what changes → what unresolved question
the new result can answer**. A new seed can be worthwhile replication, but call it that
and explain why it is preferable to a different experiment.

Initialization clarification, 2026-09-22: standard-pipeline linear-coordinate
DirectSpline already initialized as identity after TabICL preprocessing. In
`DirectSplineTransform.unmixed_transform`, however, both arctan variants use
R*u as the base, with u=(2/pi)*atan(pi*z/(2R)); zero curvature is not identity
in z. Staged curvature preserves the selected arctan-line function at step zero,
not ordinary TabICLv2. This makes initialization/preprocessing mismatch a plausible
issue for the arctan variants, not an established explanation for all spline
failures. A residual on the unchanged z with arctan used only for basis evaluation
would be a distinct intervention from the current arctan residual. Retouche's
near-identity adapter likewise needs whole-pipeline baseline parity checked; its
paper's separate preprocessing and raw-input guard mean adapter identity alone
does not prove equality with ordinary TabICLv2. A faithful Retouche comparison
is still blocked on an official/public implementation. The initialization check
completed 2026-09-23; results are summarized below.

Input-preserving direct-arctan ablation agreed 2026-09-23: on the eight shared
development tasks, retain the existing completed direct-arctan line/spline
trajectories as read-only matched controls and train only two new arms. They use
``g(z)=z+c+sS(u(z))-R u(z)`` with ``u(z)=2/pi*atan(pi*z/(2R))``, so the fresh
transform is exactly TabICLv2's ordinary standardized feature ``z``. The line
freezes ``S(u)=u``; the cubic arm alone learns curvature. This differs from the
previous direct-arctan experiment, whose initial feature was ``R*u(z)``. The
primary result will be raw preserved cubic versus raw preserved line; comparisons
to the existing compressed arms separate native-input initialization from
curvature. The first jobs (30983634/30983635) failed before training because
the launcher imported an uncommitted local metrics helper. Revision 7768869
makes its reporting self-contained. Replacement jobs 30987410/30987412 then
failed at startup because the algebraically zero arctan correction was subtracted
after adding the original feature, causing GPU rounding drift against exact
identity parity. Revision 5568d56 forms each zero-initialized residual term
before adding the feature. CUDA smoke job 30987566 completed with 2 passing
adapter initialization tests. Full replacement jobs 30987589 (multiclass) and
30987592 (regression) were submitted 2026-09-23 into
`openml_direct_spline_input_preserving_ablation/dev8_seed20260923_v2` and
completed successfully. Raw preserved cubic versus matched preserved line won
4/4 multiclass tasks (mean relative log-loss reduction +5.94%, median +0.80%,
strongly driven by eye_movements +21.95%) and 2/4 regression tasks (mean
relative RMSE reduction -0.62%). Preserved cubic beat the earlier compressed
cubic on 4/4 multiclass tasks but lost on 4/4 regression tasks. Against the
source full-training TabICLv2, raw preserved cubic won 3/4 multiclass and 4/4
regression tasks; the regression wins do not establish curvature merit because
the matched preserved line also wins. On pooled OOF, preserved cubic beat its
matched line on only 1/4 multiclass tasks: the test-only 4/4 result is not a
validated selector. The validation-selected blend uses alpha=0 on
cardiotocography, so its guarded win there is an identity fallback, while raw
cubic still loses to TabICLv2. These are revisited development datasets, not
fresh confirmation. Full evidence: each problem type's
`input_preserving_ablation_summary.json` and four `task_summaries.json` files
under the result directory. Do not call a hand-written approximation Retouche.

Input-preserving multiclass breadth check agreed 2026-09-23: extend the
four-task formulation (K20 cubic, four bags, 500 steps, 5–20% train-only query
sampling, arctan basis with native TabICL input, frozen-context expansion) and
train its matched line and curvature arms on the ten datasets in the existing
`openml_direct_spline_heldout_confirmation/multiclass_bank_v4.json`. Compare
raw spline versus line and full TabICLv2; report selected blends separately.
This reuses an already inspected cohort with completed source artifacts, so it
tests breadth of the new formulation but is not untouched final evidence.
Implementation: `scripts/direct_spline_openml_input_preserving_confirmation.py`
(commit `1bdc126`). Completed as uriofir job `30990166`; results are in
`results/openml_direct_spline_input_preserving_confirmation/heldout10_seed20260915_v1`.
Raw spline versus matched line won 2/10 (mean gain -1.34%, median -0.52%); OOF
favored spline on 5/10 and its per-task sign disagreed with test on 3/10. Raw
spline beat full TabICLv2 on 6/10, but the median gain was only +0.37% and the
mean +5.37% was strongly affected by MiceProtein (+42.7%). Selected spline blend
versus selected line was 3/6/1 with median -0.12%; two spline blends selected
alpha zero. The matched line and raw spline beat full TabICLv2 on exactly the
same 6/10 tasks. Important protocol qualification found 2026-09-24: the
cross-fit runner inherits source training settings. Four-task development had
a 500-step cosine LR schedule, checkpoint interval 25, and training random
state 20260828; the ten-task source had constant LR, interval 10, and state 0.
The runner disables inherited patience in both. Thus this result is negative
for the ten-task configuration but is **not a strict replication of the full
four-task training recipe**. The model implementation, backbone checkpoint,
K20, four bags, 500 steps, and query-fraction settings match. These are not
fresh benchmark datasets.
Per-bag training diagnostic reviewed 2026-09-24 from
`heldout10_seed20260915_v1/training_diagnostics_npz.tar.gz`: all 40 spline
bags completed 500 steps, but fixed validation error was worse at step 500
than step 10 in 28/40 bags. Across the 80 A/B checkpoint choices, 41 were
at or before step 100 and 65 at or before step 250. Only first/last stochastic
training-episode objectives were saved, not a full fixed-train curve. Thus
high constant LR is plausible but unproven; early validation peaks also mean
a 500-step cosine schedule changes LR little for many selected checkpoints.

Schedule-only follow-up completed 2026-09-24 as uriofir job `31041873`
(`ds-preserve-mc10-cos-260924`); local results are at
`results/openml_direct_spline_input_preserving_confirmation/heldout10_seed20260915_cosine_v1`.
It reuses the same ten tasks, source, training seed 0, checkpoint interval 10,
500 steps, and matched line/spline arms; only cosine decay to 1% of the initial
LR is added. Raw spline versus matched line improved from 2/8 to 7/3 wins/losses,
but the median relative gain is only +0.14% (mean +0.56%). Against full TabICLv2,
raw spline remains 6/4, as in the constant-LR run; it beat both the matched
line and full TabICLv2 on 4/10 tasks. Cosine improved the spline
arm against its constant-LR version on 6/10 tasks (median +0.30%); it worsened
the line arm on 8/10 (median -0.56%). Selected spline versus selected line is
5/4/1 with median +0.02% and one alpha-zero spline choice. Per-bag diagnostics
show the spline's best fixed-validation error improved on 44/80 A/B halves,
with median gain +0.008%; the median selected step moved from 100 to 110.
First training objective matched exactly in every bag; on the final sampled
episode cosine had lower spline objective in only 16/40 bags. The schedule
changes the relative spline/line result, but these records do not establish
that failed training under constant LR caused the earlier loss. Evidence:
`confirmation_summary.json`, `confirmation_results.csv`, and the 80 local
`raw/*/bag_*.npz` records. This tests the schedule difference only, not the
other differences from the four-task pilot.

Schedule-selection audit, 2026-09-24: independently choosing constant/cosine by
saved raw OOF loss in each family gives spline versus line 4/6, median -0.14%.
Even retrospectively taking each family's best test schedule gives 5/5; the
headroom within this two-schedule set is limited. This is a narrower repeat of
the earlier portfolio audit on the current input-preserving formulation, not
new benchmark evidence. See [schedule review](input_preserving_schedule_review_20260924.md).

2026-09-26 clarification: independently OOF-selected schedule families are 3/7
on selection OOF (median -0.142%) and 4/6 on outer test (median -0.141%). This is
already the best-of-two-schedules comparison, not a forced shared schedule.

2026-09-25 synthesis: the proposed frozen-line experiment was withdrawn as the
recommended next run; interference between curvature and other parameters was
not established. Regression's 8/2 staged result also needs a stronger-control
qualification: versus the retained initial line it is 7/3 (mean -0.145%, median
+0.024%), and versus an OOF choice of initial/continued line it is 6/4 (mean
-0.134%). Both continued line and spline beat full TabICLv2 on all ten tasks.
Forty of 80 spline-arm selected states are the inherited step-zero line, though
every dataset includes positive-step states. Recommended, not agreed/submitted:
a fixed staged-regression confirmation on new families and two outer splits,
retaining both line baselines and reporting actual curvature use and effect size.
See [research state and proposed confirmation](research_state_20260925.md).

Check existing predictions/adapters first for a cheap audit. Separate “tested and
negative,” “mixed/limited,” “invalid replay,” and “not yet analyzed.” Preserve both
ordinary-TabICLv2 and matched non-curved comparisons. Do not reuse test-selected choices
as validation-selected success, or treat already-inspected datasets as untouched tests.

Update this guide when new findings change an entry; keep detailed tables in the history.
