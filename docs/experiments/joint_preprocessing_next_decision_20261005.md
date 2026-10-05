# Decision after real-meta continuation

Reviewed 2026-10-05. User authorized the objective pilot, then restricted it to
**seed 0 only**. Implementation complete; 51 distinct local checks passed.
Submission pending.

## Evidence

The best selected arm has only 16/14 real-family wins/losses, +0.323% geometric
NLL gain, median +0.037% and a confidence interval crossing zero against ordinary16.
Raw mean NLL slightly worsens. These results do not establish useful, broad real
zero-shot improvement. Smaller extreme harms are an encouraging secondary
observation, insufficient to call the primary objective achieved.

Downloaded all six `runs/*/complete.json` and `evaluation.csv` files under
`../results/hyperspline_joint_real_meta_continuation/v1_seed20261004/`.
Every completion-file hash matches its entry in the frozen model-selection lock.

| Run | Selected step | Source-probe single-view NLL reduction from start | Source-probe blend gain vs ordinary16 | New-family validation blend gain vs ordinary16 |
|---|---:|---:|---:|---:|
| real_seed0 | 4096 | +0.983% | +0.283% | -9.567% |
| real_seed1 | 1024 | +0.727% | -0.032% | -11.255% |
| mixed_seed0 | 2560 | +0.910% | -0.062% | -12.899% |
| mixed_seed1 | 1024 | +0.574% | +0.193% | -7.018% |

Final real seed1 source-probe single-view NLL worsens 2.436% from its start.
Fixed probes contain source-family rows, with possible overlap with resampled
training episodes. They are learning diagnostics, not unseen-family evidence.
Their modest gains do not by themselves prove underfitting: available useful
headroom under this sampling/representation is not established. Neither do the
results establish a clean strong-source-fit/weak-transfer explanation.

Validation-relative ratios vary substantially while raw mean NLL changes little;
the two selected real models each win on only 1/10 validation families under the
logged absolute-NLL tolerance. This warrants reporting raw NLL, medians and harms
alongside the selected mean log-ratio score. Do not silently change the reported
primary metric or retrospectively select a checkpoint using the test panel.

## Recommended next experiment

Run one bounded objective ablation before increasing model/data scale. This is
the ensemble-loss idea proposed on September 30 and explicitly deferred during
the October 4 data-source comparison; it is not a new idea or another per-dataset
teacher-fitting experiment.

Compare the current raw single-view CE objective against CE on the deployed
fixed ordinary8+learned8 ensemble, using its exact class-aligned logit aggregation
and temperature. Detach the ordinary branch, freeze TabICL weights, backpropagate
through learned preprocessing. Keep the generator, transform family, alpha=0.5,
common starting checkpoint, real source bank, episode schedule and optimizer
identical. Do not add a new gate or change the conditioner in this ablation.
The ensemble objective costs more model evaluations; disclose actual compute
alongside equal task/update counts rather than claiming equal compute.

A first 1,024-update, **one-continuation-seed (0)** pilot uses the existing 40 training
and 10 validation families. Run one current-objective control and one ensemble-loss
arm with matched episode schedules. Log source and validation scores
under both the training and deployed ensemble objectives, as well as medians,
material wins/losses and harm tails. This remains one frozen shared network at
deployment, with no target-dataset optimization.

Decision: require meaningful, consistent source and new-family validation benefit
before paying for a larger confirmation. If both remain flat, stop scaling this
recipe and investigate representation/conditioning. An objective mismatch is a
testable hypothesis, not a proven cause or promise that this ablation will work.
Any positive claim about generalization needs a subsequently frozen evaluation
on fresh families; the inspected 30-family test is now development evidence.
Conditioner necessity versus a learned global recipe remains a later attribution
control if useful transfer appears.

Implementation protocol: [one-seed objective pilot](joint_preprocessing_ensemble_objective_20261005.md).
Evidence: [final results](joint_preprocessing_real_meta_continuation_results_20261005.md),
[protocol](joint_preprocessing_real_meta_continuation_20261004.md),
`scripts/joint_preprocessing_real_meta_continuation.py` (`surrogate_logits`,
`surrogate_nll`, `evaluate`) and downloaded run-level curves.
