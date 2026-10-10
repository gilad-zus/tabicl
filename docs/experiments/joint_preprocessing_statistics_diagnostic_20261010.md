# Statistics-only generator with a real direct-fitting reference

Authorized 2026-10-10. One seed-zero shared run, plus eight bounded per-source
direct-fitting references. No synthetic mixture or teacher targets. Status:
implemented; 25 targeted CPU checks passed; remote launch pending.

The closest predecessor is the October 8 raw/backbone residual comparison on
160 training and 25 validation sources. Older summary-based HyperSpline models
have synthetic gains but weak real transfer; the October 3 direct fitting study
showed synthetic fitting gains without scoring new rows. This revisits summary
conditioning inside the current exact-native16 residual formulation and adds a
real held-out-row capability reference. It does not establish novelty by itself.

## Shared model and training

Keep all current output heads and constraints: affine shifts/log scales bounded
by tanh, 20-control cubic monotone spline with positive normalized gaps and a
sigmoid gate, generated eight-unit neural residual with a sigmoid gate, rank-four
mixing bounded in Frobenius norm by 0.1 with a signed gate. Two normalization
slots share each generated map across four adapted views. Eight of sixteen
native none/power views remain ordinary; the other eight receive residuals after
the unchanged native preprocessing. Missing-cell corrections are masked and
categories receive no correction. Initialization must exactly equal ordinary16.

Replace only the conditioner with 31 fixed label-aware context summaries per
numerical column, their arithmetic mean across numerical columns, and log1p
context-row, numerical-column and class counts. The generator has 8,187 trainable parameters versus the prior raw model's 85,287. The shared 65-to-32-to-64 MLP
uses input LayerNorm and GELU after each linear layer. No trainable attention,
cell encoder, frozen-context extraction or feature-group alignment remains in
the conditioner. The frozen TabICL predictor retains its original architecture.

Fresh model seed zero. Train 4096 AdamW updates with four real episodes per
update, lr 0.0003, weight decay 0.0001, betas 0.9/0.999, epsilon 1e-8, gradient
clip 1. Exact staged input gradients optimize CE of mean aligned logits divided
by temperature 0.9. Use the existing balanced 160-source scheduling order and
12 requested length/fraction combinations: lengths 128/256/512/1024, context
fractions 0.5/0.7/0.85. Cap each episode independently by its own retained pool.
No query features, masks, labels or statistics condition the generator.

Evaluate at zero and every 512 updates on the same 25 independent development
validation sources (50 episodes) and fixed common/additional training-source
probes. Save every 25 updates and resume model, optimizer and RNG states.
Select minimum arithmetic mean dataset validation NLL, with identity eligible.
Report final and selected models, gain percentiles, dataset bootstrap intervals,
W/L, median gain, >1% and >5% harms, gradients, clipping, correction RMS and maps.
The existing practical gate remains >=0.5% mean reduction, >=18/25 wins, positive
median and an adjacent passing checkpoint. Validation is reused development
evidence, and no confirmation test bank is created or opened.

## Eight direct-fitting references

Before scoring, permute the 160 source families with NumPy seed 20261010 and
take the first eight whose sampled outer context supports all classes in three
inner partitions. Record skipped sources and reasons; no outcome-based choice.
Draw up to 1024 source rows, with 75% outer context and 25% outer query. Require
at least three outer-context rows per class. Within outer context, use about 60%
as the fixed inner context, 20% fitting queries and 20% checkpoint-selection
queries, with stratified quotas and at least one example of each class per part.
All source-row IDs and partitions are saved and hash-locked before GPU work.

Table encoding/imputation is fitted on outer context only. Inner episodes slice
those already encoded rows; each native preprocessing view is fitted on its
inner context. At outer prediction, all methods receive the same full outer
context, and native preprocessing is refitted on that context. Thus a selected
residual must also survive the increased context and its fitted native scales.

Optimize raw per-dataset transformation head outputs directly, preserving the
same bounds, identity initialization and rank as the shared model. Original
numerical-column IDs align parameters if inner/outer native constant-column
filters retain different subsets. Two slots receive the same native16 ensemble
loss. There is no trainable conditioner. AdamW lr 0.001, weight decay 0.0001,
clip 1, 250 steps per source; evaluate inner selection every 25 steps, retaining
identity. Lock selected weights before scoring outer query labels. Report
selected and final outer scores against ordinary16 and fitting/selection curves.
There is no LR sweep, teacher imitation or shared-model warm start.

Remove each reference's outer query source-row IDs from that source's shared
meta-training pool. The shared model still learns from the eight source datasets,
but neither direct nor shared training uses these held-out labels. Shared-model
selection uses the separate 25-source validation panel. Compare selected/final
shared predictions to direct references on the eight outer queries after shared
training. These are training-source held-out-row diagnostics; only the separate
25-source evaluation measures dataset-disjoint zero-shot transfer. A bounded
direct optimization failure is inconclusive about all possible preprocessing.

## Row-cap audit and comparison limits

Downloaded October 8 raw-arm presentation logs contain 16384 episodes. 45.752%
were below requested length. Mean requested/actual lengths were 480/286.628;
actual rows were 59.714% of requested rows. At requested lengths 512 and 1024,
87.207%/95.801% were reduced, with actual means 358.780/403.730. The inherited
minimum across eight scheduled sources included four from the unused old small
arm. The new sampler removes that cross-source minimum. Audit artifact:
[row-cap JSON](joint_preprocessing_row_cap_audit_20261010.json).

The new design also reserves reference queries from meta-training. Comparison
with the old attention model is therefore a comparison of recipes, not a clean
causal attention ablation. Negative results do not isolate information loss,
optimizer budget, headroom, or insufficient training-source diversity.

## Execution

Result root: `results/hyperspline_joint_statistics_diagnostic/v1_seed20261010`.
Reuse only verified frozen panels from
`results/hyperspline_joint_dataset_diversity/v4_seed20261007`, fingerprint
`7149269ed57b83ce024a07934f293bae6b69b80b0bd41bba1686c997a9f149f1`.
CPU preparation uses dsiofir01 runnohup, log `jp-stats-bank-261010.log`.
GPU pipeline runs preflight, eight direct fits, shared training and report serially.
Use uriofir/p_uriofir/ug_uri_ofir, node dsiuriofir01, one GPU, four CPUs, 32G RAM,
eight hours, email enabled. Prior shared runs needed about 4.2 hours; independent
row caps increase the actual row budget and direct fitting adds 2000 updates,
justifying a larger wall limit. Resume durable states if necessary. Pin HEAD,
code hashes, data/initial/backbone hashes and direct-bank hashes throughout.
