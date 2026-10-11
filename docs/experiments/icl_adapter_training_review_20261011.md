# Training an input adapter through frozen tabular ICL

Reviewed 2026-10-11. The proposal below was subsequently authorized with a
broader benchmark panel; see the [implemented protocol](joint_preprocessing_benchmark_sampling_20261011.md).
The target remains a shared zero-shot preprocessor.
Direct per-dataset fitting is a capability diagnostic.

## What the sources actually support

- [TFM-Retouche, May 2026 preprint, sections 3.4–3.5 and appendix G](https://arxiv.org/html/2605.06047v1):
  fits input adapters through frozen TabICLv2 and reshuffles context/query roles
  each epoch. An external held-out validation split decides the identity guard.
  The broader recipe includes regularization, model selection and bagging, so its
  reported success does not isolate resampling or validate our transformation.
- [TabForestPFN study, 2024 version, section 4.2](https://arxiv.org/html/2405.13396v1):
  draws fresh 80/20 support/query splits within fitting data at each gradient step;
  early-stopping query rows come from a separate partition. Step zero is eligible.
  This updates backbone weights, unlike our adapter-only setting. Its reported
  fine-tuning win rate was 54% on datasets below 1,000 observations, illustrating
  that such training does not guarantee broad gains on small tables.
- [Official PriorLabs TabPFN finetuned_base.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_base.py)
  and [finetuned_classifier.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/finetuning/finetuned_classifier.py),
  main branch inspected 2026-10-11: reserve validation before constructing training
  episodes; regenerate context/query splits using an epoch-dependent seed; select
  using validation. The classifier evaluates validation predictions conditioned on
  X_train/y_train. These are mutable upstream references, not pinned dependencies.
- [BETA / TabPFN Unleashed, sections 2.3 and 3 and appendix B](https://arxiv.org/html/2502.02527v1):
  reports a variance/overfitting cost from fine-tuning, especially with small
  training sets; learns input encoders with a frozen TabPFN and uses encoder
  ensembling/bagging. Its external training/validation/test separation and
  validation selection are relevant, but its first-generation TabPFN and learned
  representation differ from our native residual maps.

Hugging Face markdown paper endpoints failed during retrieval; original arXiv
HTML and official upstream code were used instead. No source establishes a
universally optimal split, learning rate, adapter, or guarantee against harm.

## Interpretation for the user's concern

Only preprocessing parameters receive gradient updates in our current experiment;
the TabICL backbone is frozen. Those preprocessing parameters can nevertheless
overfit training rows. Reusing those labelled training rows as context for unseen
validation/test queries is a normal prediction protocol. Their presence does not
make the query labels available to the predictor or prevent detection of poor
generalization to new rows. Rows may alternate between context and query roles
across training steps; the query's label must be absent from its own forward-pass
context, and validation/test labels must stay outside optimization.

Resampling creates more prediction situations, not more independent observations.
It can reduce dependence on one fixed split but cannot rule out memorizing the
whole fitting pool. External validation and locked outer evaluation are still
needed. A direct per-dataset map is allowed to specialize to that dataset's
labelled context; transfer to other datasets is the later shared model's separate
requirement. For the eventual zero-shot claim, entire dataset families must be
held out from shared training and development decisions.

The current direct runner optimizes the same `entry["fitting"]` every step in
`scripts/joint_preprocessing_statistics_diagnostic.py`. The shared hypernetwork
already used resampled training episodes; this omission applies to the latest
direct reference, not to every previous run. The October 11 replay showed that
holding context fixed at evaluation does not rescue its aggregate held-out gain.
September's sampler and cross-fitting studies already investigated resampled
per-dataset fitting with other representations. See research history section 7
and the original `openml_direct_spline_sampler_ablation/pilot4_seed20260915/`
artifacts. This proposal revisits that training practice for the current maps.

## Revised controlled proposal

Keep the same eight development sources and outer row IDs. For illustration,
1,000 sampled rows become T=600 fitting rows, V=150 checkpoint-selection rows and
E=250 outer evaluation rows. Actual source sizes are preserved from the existing
bank; the 1,024-row maximum is a diagnostic budget, not a claim that larger tables
have been fully used. T is the union of old fitting-context and fitting-query rows.

Two paired arms, model seed zero, 250 updates each:

1. Fixed assignment: the same approximately 450 context / 150 query rows from T
   at every update.
2. Resampled assignment: new stratified context/query membership within T each
   update, with exactly the same respective sizes and class-coverage constraints.

Both directly optimize the current affine/spline/neural/mixing outputs through
frozen TabICL16. Preserve identity initialization, AdamW lr 0.001, weight decay
0.0001, gradient clipping 1, loss/temperature and native views. Do not add new
regularizers or architecture changes in this comparison. Raw encoding/imputation
must be fitted using T only. Fit native episode preprocessing on each episode's
context, as in the existing prediction path.

At step zero and every 25 updates, predict V using the same complete T as labelled
context. Select minimum validation NLL, allowing identity. At final evaluation,
predict E with exactly that same T and T-fitted native preprocessing. Ordinary16
receives the identical context and preprocessing fit. Do not append V after
selection in this diagnostic. This aligns validation and test conditions while
letting training use the usual context/query split inside T.

This refines the earlier recommendation: rerun the fixed arm under the same new
validation/test protocol. The existing 3/3/2 record is historical context, not a
fully matched control for this revised design. Its checkpoint was selected with
a smaller context and raw encoding/imputation used the outer context.

Log fitting loss, a fixed panel of training-pool probe episodes, external V loss,
native-baseline-relative changes, gradients and per-row query exposure. Fixed
training probes diagnose fitting of already observed rows; they are not additional
validation data. Save every evaluated map. Lock selected and final maps before E
scoring, and never select the context, checkpoint or regularization using E.
Report both arms against matched ordinary16 and against each other, including
mean/median gains, W/L/ties and harms. Continued train-probe improvement with flat
or worsening V performance is evidence consistent with overfitting; it is not an
isolated causal diagnosis.

The eight outer panels have already informed development. This comparison can
guide the next decision, but success needs fresh data/family confirmation before
a zero-shot/generalization claim. Another weak result would justify deprioritizing
this recipe for budget reasons; it would not prove every preprocessing family or
training procedure incapable of improvement.
