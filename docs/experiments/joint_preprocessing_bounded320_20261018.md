# Bounded continuation from update142 to320

Authorized October18,2026: one seed-zero eight-hour allocation to decide whether
the current large-context shared preprocessing recipe deserves more training.
Predecessor job35667323 stopped cleanly after142 updates with weak validation
gains. This continuation changes only the stopping and validation schedule.

## Locked learning protocol

Use the same83 training/10 validation/58 historically inspected outside source
groups and original raw-cache hashes/reserved training probes. Keep context
cap20000, uniform sampled training queries up to2048, one chunk/source visit,
four equal-weight source means per optimizer update, LR0.0003, AdamW moments,
weight decay0.0001, clip1, frozen TabICL/native16, compact5912-parameter model
and seed0. Resume the deterministic source/row schedule at update143.

Import exactly the completed-step142 state of
`results/hyperspline_fraction_context/v3_cap20000_query2048_seed20261018`.
Parent fingerprint:
`facd59803d5ae333b3c712d498db20cc41f45d2a175c3061d5115a7dae9cff0b`.
Parent state SHA256:
`807e0643a7534b88e722ad191dff842f2e486cc09f982f1297405dbb2ce11d17`.
Local audit confirms cursor142/episode0/query0, no pending validation, no
accumulated gradients. Preserve weights, optimizer state, historical row
coverage, best validation score and selected step64 checkpoint. No partial
training gradient is discarded. New logs identify the inherited142 updates.

## Budget and evaluation

Train178 additional updates, stopping at320 total. Do not run an initial
validation or intermediate sweeps. Run one full validation at320: all ten
validation datasets, all three existing fixed80/20 partitions, all query rows,
context cap20000, matched native16. Preserve the historical best checkpoint
when comparing the new validation result. Save final/selected checkpoints and
stop after validation; final catalog inference is deferred until the decision.

Reuse the passed parent GPU shape check and original locked bank; preparation
does not rebuild data or run a new profiling experiment. Check compatibility
of all model/dependency/settings hashes, permitting only runner/wrapper code,
revision/provenance, target update count, validation cadence and deferred
evaluation timing changes. All other training fields must equal the parent.

Submit one uriofir allocation,1GPU/4CPU/32G/8h,email enabled. The runner uses a
450-minute resumable deadline; it may stop before320 or during validation if
runtime exceeds the estimate. No automatic allocation chain. Observed training
rate implies5.66h for178 updates, plus full validation and other overhead;
completion within one allocation is an estimate.

New result root:
`results/hyperspline_fraction_context/v4_bounded320_seed20261018`.
Use existing paired online training/native diagnostics and the full validation
result to make the next compute decision. Tiny changes without a clear increase
in useful learning do not justify repeatedly extending this exact recipe. A
negative outcome does not prove the broader zero-shot preprocessing idea fails.

Implementation/wrapper: `scripts/joint_preprocessing_fraction_context.py`,
`scripts/run_joint_preprocessing_fraction_context.sh`.
Targeted checks: `tests/test_joint_preprocessing_fraction_context.py`.
All21 targeted CPU checks passed, including exact imported weights/moments,
sampled-query interruption/resume for both continuation policies, final-only
validation, settings/dependency rejection, and stopping before catalog inference.
Launch revision,ID and log paths are recorded after submission.
