# Direct-map context replay

Authorized 2026-10-11. Evaluate existing frozen direct maps; no training.
Source: `results/hyperspline_joint_statistics_diagnostic/v1_seed20261010`,
fingerprint `fd95d52d80b8e41436338776031e1b2babd1df330bf5a86b3d114858b7ad730d`.
Output: `results/hyperspline_joint_direct_context_replay/v1_seed20261011`.

Question: do the current directly optimized maps help new rows with the original
inner context, and does the effect change under the expanded deployment context?
This revisits the September context-expansion audits for the current residual
implementation. The October 10 reference changed context and native preprocessing
at once; its selected outer results were 3 wins / 3 losses / 2 identity ties.

For all eight saved sources, keep the same outer query values, labels, masks and
row IDs. Evaluate with (1) the exact fitting/selection context and (2) the full
outer context. Each gets a separately matched ordinary16 baseline. The native
preprocessing is fitted on the respective context, using the existing prediction
path. Raw table encoding/imputation remains as saved in the original bank. The
selected checkpoint is primary; final checkpoints are diagnostics. Query results
never select a checkpoint, duration, blend, or context policy.

Require unchanged original code hashes, backbone, bank, initialization, selected
weights, source fingerprints and source-row partitions. Lock final state hashes
in the new manifest before replay. Reproduce selected inner-selection NLL and
ordinary/selected/final expanded-context NLL and accuracy, with absolute tolerance
1e-6 and relative tolerance 1e-5; reject drift before reporting a conclusion.
Check inference does not mutate map parameters. Preserve original artifacts and
write the replay to a separate result directory.

Report per-source and aggregate selected/final gains relative to each matched
baseline, W/L/ties, median, percentiles, harms, and replay drift. Weak results
under both contexts leave transformation/fitting/selection generalization
unresolved. Improvement with only original context implicates the combined
context/native-refitting change, without identifying which component caused it.
This is reused development evidence on eight training-source datasets, one split
each; it cannot establish dataset-disjoint zero-shot transfer.

Runner: `scripts/joint_preprocessing_direct_context_replay.py`.
Use the existing remote Python interpreter and cached backbone; one GPU, explicit
revision pin, one CPU thread for numerical libraries. Launch and result transfer
use exact-command approvals per AGENTS.md. Status: implemented; CPU checks cover
query-row isolation, identical query tensors, frozen inference with a real tiny
TabICL, score replay tolerances, and saved-map hash tampering.
