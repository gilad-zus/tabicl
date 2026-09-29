# Joint preprocessing hypernetwork: proposed pilot, 2026-09-28

Status, 2026-09-28: user requested execution. The synthetic-first model and
runner have been implemented locally, and targeted CPU tests passed. Commit
`4162107` is synchronized to the university checkout. After VPN reconnection,
the 512-task validation and 1,024-task test banks were generated and frozen.
GPU smoke job `31333979` completed on 2026-09-29 (exit 0). Its step-0
validation NLL was 0.9188; its one training update had NLL 0.7883 and finite
preclip gradient norm 1.1602. This checks execution, not learning. Twenty real
family source IDs are reserved in `joint_preprocessing_real_transfer_manifest_20260928.json`;
eligibility and source hashes are not yet verified, and no real data have been scored.
The CPU bank log is `/home/dsi/zusmang/TabICL/tabicl/joint-preprocessing-bank-260928.log`.
Smoke stdout/stderr are `slurm_logs/slurm-jp-smoke-j0-260928-31333979.out`
and `.err` in the remote repository. Its requested resources are one uriofir
GPU, 32G RAM, four CPUs and one hour, with default SLURM email enabled.
The full trainer writes per-step training NLL and gradient norm to `training.csv`,
prints a 50-step training heartbeat to stdout, and records and prints the fixed
512-task validation score and NLL every 1,000 steps. Sustained negative
validation scores against matched identity, rather than falling training loss
on newly generated tasks alone, are the learning signal.

Current recommendation following the synthetic-first question, 2026-09-28:
train and select checkpoints using synthetic tasks only, then evaluate frozen
models on synthetic tests and subsequently on real dataset families. The earlier
50/50 real/synthetic training and validation proposal is superseded. Real-data
meta-training is a possible follow-up, not a prerequisite. The user authorized
the synthetic-first experiment; the smoke job above has completed.

## Question and relation to previous work

Can a shared hypernetwork generate a useful numerical preprocessing policy for
an unseen dataset, improving frozen TabICLv2 without target-dataset optimization?
Does jointly generating a broader transform help compared with column-wise
HyperSpline, and is a spline branch necessary within the broader family?

Closest previous work: HyperSpline shape/location/scale generation; synthetic
`stats_elo_shape` and `stats_elo_locscale` runs; raw-context and cross-column
conditioners under `label_aware_ablation_paired`; and per-dataset DirectSpline
mixing. Real/synthetic meta-training code also already exists. The intervention
is joint generation of the broader transformation, including a non-spline
residual and cross-feature mixing. It is not the first direct downstream-loss
training, use of labels, richer encoder, normalization generation, or mixing.

The new restricted control is retrained with the same encoder, data and budget.
Comparison with old checkpoints alone would confound the new data/encoder with
the additional generated components. Separately optimized pipeline wins are
neither a prerequisite nor training targets.

## Three learned arms and two fixed references

| Arm | Generated preprocessing |
|---|---|
| A: column-wise HyperSpline | Per-column shift/scale, monotone cubic K20 residual and bypass gates |
| B: joint generator (primary) | A plus a small non-spline scalar neural residual and rank-four cross-feature mixing, with generated activation gates |
| C: joint without splines | Same as B, with the spline branch disabled throughout training and inference |
| Matched identity | Same preprocessing shell and inference views, with all learned corrections disabled |
| Ordinary TabICLv2 | Standard numerical preprocessing and standard eight-view inference on the same context/query rows |

A, B and C share the same encoder architecture, information access, optimization
settings, episode order and evaluation rules. Each trains its own encoder weights.
Common encoder initialization is paired within a seed. Disabled heads in A/C are
excluded from optimization. Report active parameter counts and runtime; equal
episode budgets are not a claim of equal active capacity or FLOPs. B versus C
measures the practical value of including splines at fixed remaining architecture,
not a parameter-count-matched expressive-capacity theorem.

## Generator and preprocessing shell

Use labelled context only to generate parameters once per dataset. No query
features or labels enter the conditioner in this pilot. This supports consistent
processing of arbitrary future query batches. All arms use the same context-only
choice; it is not an exact replication of earlier query-conditioned checkpoints.

Reuse the existing raw-context encoder's class-ID-invariant pooling and attention
across features; expose its per-column hidden states to new output heads. Hidden
width 64 and four attention heads. This encoder can observe same-row feature
relationships, which marginal column summaries alone cannot supply for mixing.
Handle variable column counts with shared column heads, not a fixed-width output.
Verify row permutation invariance, column permutation equivariance, class-ID
invariance, and sensitivity to cross-column dependence with equal marginals.

Fit TabICL's existing `TransformToNumerical` on context: numerical columns use
its mean imputation, and categorical columns use ordinal encoding with `-1`
for unknown/missing categories. Preserve the original numerical missingness
mask for the generator's context statistics even after imputation. Fit the
initial numerical mean/std only on context. This typed encoding is shared
with ordinary TabICLv2; the learned arms replace its subsequent numerical
normalization choices. Synthetic episodes enter as fully numerical tensors.
The generator then emits:

1. Bounded shift and log-scale corrections per column.
2. K20 cubic residual parameters and a scalar neural residual (1-8-1 tanh MLP),
   each with a generated per-column gate. The neural residual can express
   non-monotone reshaping; its output is bounded in standardized coordinates.
3. A low-rank residual mixing matrix, rank min(4, number of numerical columns),
   with norm bounded by 0.1 and a generated bypass gate.

Apply affine corrections, then gated nonlinear residuals, then residual mixing.
Use the identical generated transformation for context and queries. Identity,
affine-only and univariate-only transformations are included by bypassing blocks.
This is a defined family of numerical transformations, not arbitrary pipeline
program synthesis or a search over every operator ordering. Imputation and
categorical encoding are fixed in this first pilot; do not describe them as learned.

Preserve the existing categorical treatment, feature/class permutations and
eight-view aggregation. Generate two numerical views using a slot embedding
with shared weights, rather than silently duplicating a single numerical map
into both former none/power slots. Every learned arm has the same two-view setup.
The numeric maps in both slots start as context standardization; this matches
the identity reference but not necessarily ordinary TabICLv2's power branch.
Do not claim ordinary-baseline parity from zero generated corrections alone.

Initialize all corrections to zero and maps exactly at matched identity. Keep
nonzero hidden features/factors behind zero residual output heads so useful
gradients exist; never initialize both factors of a bilinear branch to zero.
Freeze and hash the `tabicl-classifier-v2-20260212.ckpt` backbone.

### Proposed numerical parameterization

Let `z = (x - mean_context) / max(std_context, 1e-6)` after context-fitted
imputation. Emit a per-column shift `b` in [-1, 1] and log-scale `l` in [-1, 1],
giving `a = exp(l) * z + b`. Let `u = clamp(a / 4, -1, 1)`. The K20 cubic branch
adds `g_s * 4 * (S(u) - u)`, where S is a monotone spline with 20 controls,
identity controls at initialization, gap-adjustment bound 2, and gate `g_s`
initialized at 0.1. The non-spline branch adds `g_n * tanh(N(a))`, where N is a
generated 1-8-1 tanh MLP and `g_n` also starts at 0.1. The tanh bounds the
non-spline correction to one standardized unit. Arms A/B/C enable the branches
listed in the table above. Feed the resulting numerical features through a
generated rank-four residual mixing matrix with spectral norm at most 0.1,
initialized at zero effective strength. Each arm emits parameters for both
numerical slots; the two slot embeddings distinguish them.

The generated neural transform's first-layer features must be nonzero at
identity while its final coefficients start at zero, so gradients can reach
the final head. Likewise, mixing factors must be nonzero behind a zero effective
mixing gate. Use shared, feature-permutation-equivariant heads; fixed per-index
factor templates would break the intended invariance. At initialization all
arms equal the matched standardized-input identity in both slots exactly.

## Meta-training data and budget

No real training or validation bank is required for the initial phase. Target 20
real final-evaluation families for a subsequent frozen-model transfer evaluation;
final families must be reserved before looking at their outcomes.
Group alternate versions, derived tasks and closely related datasets together.
Lock source IDs, versions, hashes, family assignments and exclusion reasons in
a manifest before training. These counts are design targets, not a claim that a
verified bank already exists. If eligibility cannot support the counts, revise
the manifest/protocol before training rather than replacing failed tasks by score.

Initial scope: binary/multiclass classification, 2-10 classes, 5-100 total encoded
features, at least one numerical column, and 256+ usable rows. Preserve original
feature types. Use at most 1,024 rows per episode/evaluation split. Regression
requires a separate compatible head/loss and is outside this pilot.

Train entirely on fresh synthetic episodes. Use the existing
`mix_scm` / `coverage_expanded` generator. Draw
episode lengths from 128/256/512/1024 when feasible and context fractions from
0.50/0.70/0.85. Ensure each query class has labelled context without artificially
balancing query class frequencies. The same episode stream is replayed for A/B/C.

### Exact episode sizes

For synthetic tasks, `PriorDataset` uses `min_seq_len=N, max_seq_len=N+1`
and equal minimum/maximum context fraction, so context rows are `floor(N*f)`;
the remaining rows are queries. The 12 scheduled combinations are:

| Total N | 50% context/query | 70% context/query | 85% context/query |
|---:|---:|---:|---:|
| 128 | 64 / 64 | 89 / 39 | 108 / 20 |
| 256 | 128 / 128 | 179 / 77 | 217 / 39 |
| 512 | 256 / 256 | 358 / 154 | 435 / 77 |
| 1,024 | 512 / 512 | 716 / 308 | 870 / 154 |

The synthetic training stream cycles through these 12 combinations in a
seeded shuffled order, then generates four fresh synthetic tasks with that
combination at each step. Thus 40,000 synthetic training tasks are distributed
nearly evenly across all combinations. Frozen validation and test banks use
the existing `generate_scheduled_episodes` helper, which balances the same
12 combinations as evenly as possible: 42-43 validation tasks and 85-86 test
tasks per combination. Store the actual context/query sizes in each bank manifest.

There are no real training/validation episodes in this phase. Subsequent real test episodes
use `N = min(1,024, eligible rows)` and a 70/30 stratified split. At the
eligibility minimum N=256 this is 179 context / 77 query; at N=1,024 it is
716 / 308. Use split seeds 0 and 1, with all preprocessing fitted on context.

### Synthetic task generation

Use `PriorDataset` through `scripts/hyperspline_synthetic_train.py` with
`prior_type=mix_scm`, `min_features=5`, `max_features=100`,
`max_classes=10`, `prior_n_jobs=1`, and `batch_size_per_gp=1` as in the
repository helper. The mixture samples MLP causal models about 70% and tree
causal models about 30%. Feature count is sampled between 5 and 100; half
the tasks choose binary classification directly and the remainder sample
2-10 classes (so binary also occurs in the second branch). The native prior
creates features and labels, with every class represented in both context
and query after its sanity check. These episodes are numerical tensors; the
prior can produce category-like values but supplies no categorical type mask.

Set `synthetic_observation_mode=coverage_expanded`. Before the context/query
split, each feature independently receives one of seven label-independent,
non-decreasing observation transforms: signed power, asymmetric exponential,
saturation, log compression, asymmetric piecewise scaling, censoring, or tail
stretching. Optional rounding and near-zero dead zones model measurement
resolution. The same generated feature mapping therefore applies to both
context and query. This mode does not inject missing values, class imbalance,
label noise, or extra redundant/nuisance features. It retains the task's label
relationship except where censoring/rounding loses information. Do not call
this synthetic validation a test of those absent mechanisms or of categorical
preprocessing. The real bank covers those separate challenges as present.

Prepare 512 fixed synthetic validation
tasks and 1,024 fixed synthetic final-test tasks from the same declared generator
mixture and size/context schedule as synthetic training. Use independent task
draws and disjoint seed/task namespaces from training and from each other;
record generation configuration, task IDs, seeds and content hashes. All arms
and both model seeds share these evaluation tasks. Do not regenerate a validation
bank at each checkpoint, and do not choose synthetic tasks by observed difficulty
or whether the proposed preprocessing helps them.

Use synthetic bank source seeds 171001 (validation) and 172001 (test), with task-ID
offsets 2,000,000,000 and 3,000,000,000. Training stream seeds are 161001 and
162001 for model seeds 0 and 1, with task IDs starting at 1,000,000,000. The
implementation must validate disjoint task IDs, source seeds and saved bank
hashes. The choice of these numbers is for reproducibility, not for tuning.

The earlier expanded synthetic study already used 512 validation / 1,024 test
tasks. Reuse its bank machinery, not its previously inspected test bank as fresh
confirmation. Generate and freeze new banks under this experiment's output root.
These banks test new tasks within the training generator distribution, not unseen
generator families. Report strata for generator type, task size, feature count,
class count and observation regimes when recorded; a held-out generator-family
stress test would require a separately locked design and is outside this pilot.

Two model seeds (0, 1), each paired across the three arms: six training runs.
Budget 40,000 fresh synthetic tasks per run,
four episodes accumulated per optimizer step, 10,000 steps total.
This is 240,000 training episodes in total. It simplifies data preparation;
at equal episode budget it does not promise lower GPU cost than mixed training.
AdamW, LR 1e-3, weight decay 1e-4, gradient clip 1, fixed LR for this pilot.
No per-arm learning-rate search, teacher fitting or per-dataset optimization.

Use exactly four fresh synthetic tasks per optimizer step.
Accumulate four losses and make one optimizer update. Set
AdamW betas (0.9, 0.999), epsilon 1e-8, no scheduler, and no warmup. For a
given model seed, A/B/C receive exactly the same synthetic tasks and
view choices. Save RNG and optimizer state to resume
without changing the stream. Use float32 for generated numerical operations,
frozen TabICL forward and final NLL. If the smoke check shows this cannot fit
the requested hardware, revise the precision protocol for every arm and
reference before a full run.

Optimize mean query cross-entropy through frozen TabICL into the generator.
For tractable training, sample one of the eight inference views uniformly per
episode, with the view index paired across arms. This optimizes expected per-view
loss, not the full ensemble loss; validate and deploy using all eight views.
Apply no additional teacher, identity-regret or curvature penalty in this pilot;
the bounded parameterization and weight decay are shared stabilization choices.

The eight inference views follow the same fixed feature/class permutation
configuration for all learned arms, matched identity and ordinary TabICLv2.
Each learned arm replaces the numeric `none`/`power` slots with its two generated
numeric slots while keeping the categorical branch and class/feature shuffles.
Where the existing ensemble generator emits fewer than eight distinct views,
record and use its actual count consistently for all comparisons. Sample one
view per training episode with a seeded view index; validation/test aggregate
all views by class-unshuffling, averaging logits, then softmax with TabICL's
temperature 0.9. This preserves the standard inference aggregation. Validation
and test losses use ensemble predictions, so the training surrogate differs
in a declared way.

Validate every 1,000 optimizer steps on all 512 fixed synthetic validation tasks.
Choose one global checkpoint
per arm/seed; step zero remains eligible. For each task compute the log ratio of
candidate to matched-identity NLL, with 1e-4 added to both errors to limit dominance
by near-zero-loss tasks. Average synthetic task scores to get S and minimize S,
with earliest checkpoint winning exact ties. Save S at every checkpoint.
Real datasets play no role in checkpoint, hyperparameter or model-seed selection.
Fix every rule before training. Never choose separate deployment checkpoints by
test domain or select settings/checkpoints on final datasets.

## Final evaluation and interpretation

The first phase ends with frozen synthetic test evaluation. Subsequently use
two fixed, stratified 70/30 context/query splits per real final family, seeds 0/1,
under the same declared row cap. Evaluate all three selected models for both
model seeds, matched identity, and ordinary TabICLv2. All methods see the same
labelled context and query rows. Use context-fitted encoders/imputers; query
labels are scoring-only. Save finite float32 probabilities restricted to present
classes, with explicit diagnostics if logits/probabilities are invalid.

After all checkpoint choices are frozen, evaluate those same checkpoints and both
references on the 1,024 synthetic final-test tasks. These synthetic tasks retain
their predeclared context fractions and sizes; the real 70/30 split rule does not
replace them. Test banks are scoring-only and are not opened for interim learning
curve decisions, early stopping, hyperparameter changes or model-seed selection.

Primary comparison: B versus ordinary TabICLv2. Paired secondary comparisons:
B versus A, C versus ordinary, C versus A, B versus C, and all versus identity.
Average losses across splits and model seeds within a family before aggregate
analysis; also show each seed's results. This is not prediction ensembling across
model seeds. Bootstrap families, not rows, splits or model seeds, for uncertainty.
For synthetic results, average model-seed metrics within task and bootstrap
independent generated tasks (or parent generator-instance groups if tasks share
one). Report seed results as well. Never pool the 1,024 synthetic tasks with the
20 real families into one win count, mean improvement or confidence interval.

Report NLL as primary, binary AUC and accuracy as secondary. Include per-family
absolute/relative NLL, win/loss/tie counts, median gain, geometric mean error
ratio, worst losses, failures, preprocessing/inference latency and peak memory.
Save generated gates, transform magnitudes and mixing norms; variation in gates
alone is not evidence that dataset conditioning is causally useful.

Interpret the two test panels jointly without substituting one for the other:
benefit on both supports transfer within the evaluated scope; synthetic benefit
without real benefit suggests a transfer/distribution issue; real benefit without
synthetic benefit indicates narrower empirical usefulness; benefit on neither
does not isolate representation, optimization and data as possible causes.
Synthetic success alone cannot satisfy the real-data continuation criteria below.

Predeclared practical continuation criterion for the primary model: at least 1%
geometric-mean NLL reduction versus ordinary TabICLv2, positive median gain,
13/20 family wins, aggregate improvement in both model seeds, and at most two
families with over 5% NLL degradation. These are proposed pilot decision rules,
not statistical significance thresholds; report uncertainty even if they pass.
Positive median B-versus-A gain is additionally needed to credit the broader
generator over the restricted family. Apply the same descriptive criteria to C,
but a promising secondary arm requires a new locked confirmation before a broad
claim. If neither joint arm helps, conclude only that this formulation/budget
failed; the experiment does not reject all hypernetwork preprocessing.

## Implementation, artifacts and execution boundary

Before GPU work: implement the joint model and runner locally; check exact
matched-identity initialization, nonzero gradients for enabled branches, frozen
backbone weights, permutation behavior, no query-label dependence, deterministic
episode replay/resume, family separation and finite metrics on tiny CPU cases.
Also verify train/validation/test task separation, frozen bank hashes, synthetic-only
checkpoint selection, and test-bank access only in final reporting.
Lock manifest, source revision, configurations and exact commands after those
checks. No remote launch command is claimed ready by this design document.

Output directory relative to remote/local results roots:
`hyperspline_joint_preprocessing_pilot/v1_seed20260928/`.
Save family/episode manifests, model/backbone hashes, selected checkpoints,
training/validation traces, test predictions and per-family summary tables.
Save synthetic validation/test summaries and bank manifests, then separate
`real` test summaries for the later transfer evaluation. Validation adds inference cost;
include that cost in the throughput estimate.

Use SLURM uriofir for this substantial workload: one GPU, 32G RAM, four CPUs;
at most eight hours per submission, with resume and email enabled. Measure
throughput/memory before estimating total runtime; no evidence currently supports
larger resources or a completion-time claim. Every remote command needs the exact
built-in approval required by AGENTS.md. This specification does not launch a job.

Implementation: `src/tabicl/_hyperspline/joint_preprocessing.py` contains the
three generated numerical transforms. `scripts/joint_preprocessing_synthetic_pilot.py`
prepares independent frozen banks, trains each arm/seed with deterministic fresh
episodes and resumable optimizer state, and opens the test bank only in `report`.
It reuses TabICL's feature/class view schedule, uses its standard none/power
preprocessing for the ordinary reference, and evaluates all methods through the
frozen inference path. Local checks: 30 targeted tests passed, including a tiny
real TabICL forward/backward/inference smoke test and end-to-end one-step runner
test; compilation passed. `ruff` is unavailable in the selected local venv.
The local interpreter is `C:/Users/Gilad/Documents/Cursor/.venv/Scripts/python.exe`.

The later real panel is fixed in
`joint_preprocessing_real_transfer_manifest_20260928.json`. Its preparation and
evaluation implementation is `scripts/joint_preprocessing_real_transfer.py`.
PMLB sources are already numerical encodings and may not preserve original
categorical types; the three OpenML families exercise typed categorical
encoding. Prior research has inspected some of these families, so the panel is
reserved against selection in this pilot but is not an untouched historical
benchmark. Only synthetic validation determines checkpoints.

Methodological references: [HyperFast](https://ojs.aaai.org/index.php/AAAI/article/view/28988)
establishes dataset-to-model-weight generation; [TabICL](https://proceedings.mlr.press/v267/qu25d.html)
provides the frozen in-context prediction setting. Neither establishes that this
particular preprocessing generator improves TabICLv2 or is itself novel.
