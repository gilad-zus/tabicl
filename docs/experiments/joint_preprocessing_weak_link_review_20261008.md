# Post-diversity diagnosis and proposed comparison, 2026-10-08

Authorized on 2026-10-08. Residual-conditioning implementation and 52 targeted
CPU checks completed; remote preparation/submission pending. Operational plan:
[joint_preprocessing_residual_conditioning_20261008.md](joint_preprocessing_residual_conditioning_20261008.md).

Evidence: the verified 40/160-source results show only +0.146% learned reduction
in mean validation NLL from the initial blend for the large model. Its common
source-probe mean gain from initialization is +0.383%; its large-only source
probe changes -0.008%. These are source probes, not optimized training-episode
losses; weak probe gain does not prove underfitting or absent real headroom.

Read-only audit of downloaded training.csv: both arms have 4,096 finite loss and
gradient records. Encoder gradients are nonzero on 4,095 updates (zero-head
initialization explains update one). Final clipped-update fraction is 2.124%
small / 7.324% large. Large median encoder norm is 0.00701 overall, 0.00386 over
the last 1,024 updates. This disfavors a disconnected gradient path; it does not
establish useful conditioning, sufficient optimization or absence of interference.

Code-derived hypotheses, not causal results:
- numeric_context/prepared_episode pass only numeric columns and context labels
  to the conditioner. Categorical values remain in TabICL but are unavailable to
  the generator. RawContextEncoder does consume raw numeric cells plus summaries,
  with row attention, class pooling and column attention; it is not a stats-only MLP.
- The spline evaluates clamp(a/4,-1,1), with endpoint-preserving controls. Its
  nonlinear tail correction vanishes outside that range. The neural correction
  is bounded by one and affine scaling is exp(tanh), with modest linear mixing.
  Thus this family is not arbitrary preprocessing and may struggle with useful
  tail compression. This is an architectural restriction, not a measured cause.
- Learned views replace native numerical preprocessing with generated maps.
  Standardized identity of the adapter is not exact identity of ordinary16.
- Prior large teacher gains optimize synthetic fitting queries, often against
  single-view standardized identity. They do not establish similarly large,
  attainable zero-shot improvement against ordinary16 on these real sources.

Closest prior experiments: September 23-24 input-preserving per-target adapters;
October 3 separate/shared synthetic fitting; October 4-5 synthetic/real/mixed
continuation; October 6 ensemble-objective ablation; October 7-8 fresh 40/160.
Mixed continuation's real-test geometric gains were -0.232% (14/16), versus
real-only +0.323% (16/14), with intervals crossing zero. A new 160-source mixed
run would change bank size, fresh initialization/objective relative to that
study; it is untested but not the highest-priority diagnostic. At fixed total
updates, a 50/50 mixture also halves real episode exposure.

Proposed next comparison: two shared, frozen-at-deployment preprocessing models,
seed zero, the existing 160 training sources / 25 development-validation sources,
same 4,096 updates and sampling schedule. Both start from the exact ordinary16
view ensemble: keep eight views unchanged and add context-conditioned residuals
to the other eight after their normal preprocessing. Zero correction must replay
the same 16 native views exactly, not duplicate eight views. Retain transform
heads initially to avoid conflating spline removal with the encoder change.
Avoid zeroing both a multiplicative gate and its residual output at initialization.

Arm A uses the current numerical-context encoder. Arm B conditions on frozen
TabICL representations of the full labelled context, including categorical
features; representation extraction and per-column alignment require design and
parity checks before implementation. No query labels feed the conditioner.
TabICL weights stay frozen and there is no target-specific optimization.
Compare source probes, validation mean/median NLL, W/L, tails and emitted-map
variation; use the existing current-model report as historical context.
A vs B isolates the benefit of richer conditioning within the revised residual
formulation. Comparison with the old run changes the view/fallback formulation
and is not a clean one-factor causal test. Benefits of a learned gate are not
guarantees against unseen-dataset harm.

This revisits input preservation in a shared zero-shot generator, rather than
repeating per-target teacher fitting. If this produces reliable learned benefit,
then test added synthetic experience while accounting for real exposures/compute.
If it remains flat, the design and training-signal assumptions remain unresolved;
do not declare the general hypernetwork thesis impossible.

Primary background: TabICL uses column-then-row embeddings
https://arxiv.org/abs/2502.05564 ; using them as conditioner inputs here is our
proposal, not a result established by that paper. MotherNet demonstrates the
adjacent possibility of dataset-to-network hypernetworks, trained at much larger
synthetic task scale: https://arxiv.org/abs/2312.08598 .
