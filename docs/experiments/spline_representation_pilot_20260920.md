# Spline representation and capacity pilot

The research target is a positive contribution from learned spline curvature,
both relative to ordinary TabICLv2 and relative to a matched line-only control.
Fallback-to-identity wins do not count as evidence for splines.

## Separate questions

1. Existing replacement pipeline: does cubic K12 learn a better transformation
   than the existing K20 continuation? Start from the same saved A/B-selected
   direct-line states, preserving center/span and mixer. Run the same 250-step
   continuation, sampler, splits, optimizer, and selection protocol. Capacity
   changes, not initialization. Compare with cached K20 and continued-line
   predictions from `openml_direct_spline_staged_curvature_ablation/dev8_seed20260915`.
2. New representation: retain every original preprocessed column and append
   one learned feature for each numerical column. Compare line, free cubic K8,
   and free cubic K20. K12 is NOT another feature-expansion arm.

## Feature-expansion specification

- Fixed coordinate: `u = (2/pi) atan(pi*z/8)`. All added features initially equal
  `4*u` exactly. The original preprocessed `z` remains unchanged.
- Lines learn two parameters per numeric column per normalization branch.
  Splines learn K free controls, including endpoints; curves may be nonmonotone.
- Two normal preprocessing branches (`none`, `power`) have separate functions;
  ordinary ensemble members within each branch share them.
- Degree 3, open-uniform fixed knots; no learned mixer, gates, or knot locations.
- Existing feature permutations are extended so each extra feature follows its
  source column. Line and spline views have identical widths and permutations.
- Adam LR 0.001, cosine decay to 0.0001, 500 steps, gradient norm clip 2,
  weight decay zero. Smoothness weight 0.0001 on mean squared second derivative
  of `s(u)/4` over 64 common interior points. Straight lines have zero penalty.
- Multiclass cross-entropy; regression MSE in the frozen T-fitted target scale.
- Four bags, protocol seed 20260915; feature-training seed 20260920.
- Each step samples 5–20% of T as query; all remaining T rows form context.
  Query labels are loss targets only. No training/context row caps.
- Checkpoints scored every 25 steps, including step zero. Independent A/B best
  states. Opposite-half OOF scoring; test uses frozen transformations with full
  T+A+B labelled context. Main predictions average both states across four bags.
- No identity blending or test-based choice of capacity in the primary report.

## Data and controls

Development tasks only: multiclass 4602, 75158, 167186, 361539; regression 4999,
5042, 362096, 362343. These are not a fresh confirmatory benchmark.

Report cached ordinary full-context TabICLv2, matched expanded-context identity,
untrained augmented features, learned line, and each spline capacity. The matched
identity and untrained augmentation are computed once per bag and cached.
Both reference predictions and source split provenance are retained.

For each selected spline checkpoint, also replace its added features with their
least-squares straight-line projections on T coordinates (no held-out labels).
Score the frozen model again. This secondary diagnostic checks whether its
predictions benefit from the learned curvature, beyond the separate trained-line
control. It never chooses a checkpoint or changes the deployed primary prediction.

## Optimization versus generalization

Feature runs retain initial, final, and both selected adapter states. Four fixed
train-only episodes are evaluated at every validation checkpoint. Record raw
training loss separately from smoothness and validation loss. Final test scores
are diagnostic and never select a checkpoint. The K12 replacement run additionally
saves both final trajectory states and eight fixed training-episode evaluations
using the previous training-audit seed, 20260919.

Smaller capacity improving both train and test supports a training-efficiency
hypothesis; larger capacity improving train but hurting test supports a
generalization hypothesis. Neither pattern is a causal proof. Uniform K8/K20
knot grids are not nested, and raw optimization budgets do not establish global
optima. The K12/K20 replacement comparison also remains a development comparison.

## Execution and resume

One sequential launcher runs feature expansion and replacement K12 for multiclass,
then regression. Feature training saves optimizer, scheduler, sampler position,
best states, and trace every 25 steps. Its equivalent-hardware option permits
allocation/ordinal changes only when stable GPU/software/precision properties match.
Replacement K12 uses the existing completed-bag resume; an unfinished bag restarts.

```bash
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python \
  /home/dsi/zusmang/TabICL/tabicl/scripts/direct_spline_openml_representation_pilot.py \
  --results-root /home/dsi/zusmang/TabICL/tabicl/results \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/openml_direct_spline_representation_pilot/dev8_20260920 \
  --device cuda:0 --resume --allow-equivalent-hardware-resume
```

Only small adapter/optimizer states are persisted; no duplicate backbone weights.
The extra columns increase computation despite the small number of trainable
parameters. Row-interaction chunks are 256 in the feature run to bound activation
memory without removing data. This is a new GPU training experiment, not an audit.
