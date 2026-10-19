# Learning-rate response experiment after bounded320

October 19, 2026. Authorized by the user's "Do it" after the revised smaller-row
proposal. Implemented locally; submission pending. The earlier64-update/20k-row
proposal was superseded and never launched.

Stop extending the unchanged large-context recipe. Step320 validation is
+0.10283% mean NLL reduction, 5/5, essentially the step64 result. Paired online
training benefit is also tiny; those changing-episode observations do not prove
convergence or distinguish fitting from transfer. See
[completed results](joint_preprocessing_bounded320_results_20261019.md).

Start both seed-zero arms from the same step320 model and AdamW state. Use
identical subsequent dataset/row/view draws,256 additional updates per arm,
four sources per update, training context cap4096 and training query cap512.
Only constant learning rate differs between the arms:

| Arm | Learning rate |
|---|---:|
| Current-rate control | 0.0003 |
| Larger-update intervention | 0.003 |

Preserve model, objective, ensemble, gates, clipping, weight decay and83/10/58
source allocation. Resample from full fitting pools using the original80/20
split policy before row caps. Permanent256-row training-source probes remain
excluded. Both arms use the same absolute steps321-576 and native view draws.
Log paired native/candidate loss, parameter update norms,
correction magnitude, gate values, clipping and nonfinite values. Compare both
final checkpoints with their common starting checkpoint. Replay the first eight
actually optimized continuation episodes at the4096/512 training sizes, verifying
logged row hashes, to measure fitting without a context-size confound.
Separately replay a small fixed
panel of eight previously fitted parent episodes, selected by ID without scores,
at the start/end; label it fitting evidence. Replay their original20k/2048 row
limits and verify context/query hashes against actual logged presentations.
Evaluate the common starting checkpoint and both final arms together, sharing
native predictions. Use the existing complete ten-source,three-partition
validation, context cap20000/all queries/chunks2048. There is no per-target
choice, checkpoint selection or test access. This is a learning-response
diagnostic, not a convergence test. Comparing arms isolates LR at4096-row
training; historical comparisons also change training row caps.

Bound the entire comparison to one uriofir8h allocation,1GPU/4CPUs/32G/mail on.
The runner has a450-minute deadline and reserves its final120 minutes for
evaluation. Actual paired-update timers and remaining-training estimates are
printed from the first update; no speedup factor is assumed. Train the two
arms on each shared episode and commit both optimizer steps in one atomic
checkpoint after every four-source update. Resume discards any log records
beyond that durable paired boundary. A crash during a pair replays the pair.

If the training deadline arrives early, lock the same completed update count
for both arms and report incomplete target coverage. Evaluate that matched
endpoint; do not automatically chain jobs or silently increase the budget.
Once evaluation begins, resume preserves the locked endpoint and checkpoint
bytes and continues full-query evaluation rather than retraining.

Closest precedents: the [October3 LR comparison](joint_preprocessing_fromscratch_results_20261003.md)
used eight synthetic fitting tasks and a different85k-parameter joint model;
0.0003 outperformed0.001 there. It does not support a general higher-LR claim.
The [October17 LR proposal](joint_preprocessing_catalog_interpretation_20261017.md)
was superseded, unexecuted. This comparison instead tests whether the current
5912-parameter residual ensemble and real sources respond to larger optimizer
updates. Earlier2048-row training already gave weak results; reduced rows are
an execution tradeoff, not a new claimed solution. The higher-LR hypothesis
is unestablished; tenfold LR is not equivalent to tenfold training updates.

If fitting improves substantially but validation does not, stronger evidence
points toward generalization/conditioning. If neither improves, this fails to
justify further extensions of this particular recipe; it does not prove that
all learning rates, preprocessing families or zero-shot hypernetworks fail.
Only broad, materially larger validation improvement would justify a fresh
dataset-level confirmation, not a tiny improvement or W/L count alone.

Parent: `results/hyperspline_fraction_context/v4_bounded320_seed20261018`.
Actual parent state downloaded/audited locally: step320,clean0/0 cursor,no
pending validation or gradients,SHA256
`4df5679950a0ce437dd051dd698635d4b7a1345d9c156c73b25847bbd7c262c7`.
Both imported models and AdamW moment tensors match the parent exactly;
all optimizer step counters remain320. All48 parent code dependencies match
the new50-dependency specification before synchronization.

Result root: `results/hyperspline_lr_response/v1_seed20261019`.
Implementation: `scripts/joint_preprocessing_lr_response.py`.
Launch wrapper: `scripts/run_joint_preprocessing_lr_response.sh`.
Checks: `tests/test_joint_preprocessing_lr_response.py` plus the unchanged
fraction-context runner's regression tests. No separate CPU bank construction
or GPU profiling allocation; pipeline verifies/copies the locked source bank
and starts actual paired training.
