# Class-profile input comparison, October 21

Authorized by the user's "do it" after the paired-input proposal. Implementation
uses the existing paired training, frozen backbone gradient replay, native
preprocessing, row allocator and resumable prediction helpers. No bank rebuild.

Question: does retaining each class's ordered feature distribution before
pooling improve useful affine/spline generation beyond a matched marginal
profile branch? The current checkpoint has a small positive breadth signal
(30/25/3 across58 groups, mean NLL+0.010376%) and weak fitting diagnostics.
Earlier CDF/raw/backbone encoders already exist; this is a specific representation
comparison in the compact residual model, not a first richer-input experiment.
See [proposal and precedents](joint_preprocessing_input_proposal_20261020.md).

Both arms retain the31 column summaries and table/size summaries. Histograms
use context mean/std,16 equal-width intervals in[-4,4], two tail intervals,
and one missing category. Append class frequency. Each20-vector passes through
20->32 GELU->32 GELU; class embeddings are averaged with class-frequency weights,
then projected32->64 and added to the existing column token. Projection starts
at exactly zero. Numeric class identifiers do not enter the network.

- `marginal_profiles`: repeat the overall column histogram for every class.
- `conditional_profiles`: supply the actual histogram within each class.

Each model has9688 trainable parameters versus5912 in the frozen starting model.
Both begin with identical parameters and exactly reproduce control576 generated
maps. Existing AdamW moments/counters are copied by parameter name; added
parameters start without moments. All model parameters remain trainable.
Affine/K20 monotone spline heads, bounds, two slots, missing masks and native
eight plus adapted eight views are unchanged. No teacher or target-specific fit.

Parent: `results/hyperspline_lr_response/v1_seed20261019`, fingerprint
`d2a497a5d8eba2b58d42001301691e7dceea7b07e57f7876a3ad3b4b10fe8f43`.
Control SHA256`bc26f03af59526180e811248d1eb405bcc45cda6df4f3771ea5dbc43776dceef`;
paired state SHA256`9aeba9a41782b8c54cef145350e40f68fa182eabbb67c0916e73fea308faa615`.
All50 parent dependencies remain unchanged;55 runtime dependencies are pinned.

Seed0, LR0.0003,384 additional updates per arm, absolute577 through960.
Four equal-weight datasets per update; context cap4096/query cap512, resample
on revisits, identical draws between arms. FP32, frozen TabICL/dropout0,
mean-logit CE at temperature0.9. One atomic state commits both arms after
each complete update; interruption replays an unfinished pair. No automatic chain.

Evaluate both fixed final arms and frozen starting576 versus matched native16:

1. Replay the first16 actual fitting episodes, with exact row hashes.
2. All83 training datasets' permanently reserved queries; context cap4096.
3. Ten validation datasets, partition0 at4096 context, to compare with the same
   partition/query pool at the larger context size.
4. Ten validation datasets x3 fixed partitions, context cap20000, all queries.
5.58 outside groups x2 fixed partitions, context cap20000, all queries.

Queries are chunked at2048 only for evaluation. Reserved rows are excluded from
training. All validation/outside sources have historical development exposure;
these are not a pristine test set. No checkpoint selection or per-source guard.
Report NLL magnitude/W/L/T/percentiles/bootstrap intervals and accuracy/AUC/F1,
paired candidate/control and candidate/start differences, exposure strata and
largest-contributor sensitivity. Log paired native-relative training losses,
gradient/update norms, profile-branch gradient/use, and correction sizes.

Output: `results/hyperspline_class_profiles/v1_seed20261021`.
Command: `bash scripts/run_joint_preprocessing_class_profiles.sh REVISION`.
Submission: uriofir/p_uriofir/ug_uri_ofir,1GPU,4CPU,32G,08:00:00,mail enabled.
Runner450min, reserving150min for final evaluation. Prior measured paired
forward/backward time41.3s implies4.4h for384 updates before overhead; the
eight-hour request follows that measurement. If training reaches its time
boundary early, lock/report the same completed-update count for both arms;
do not label it a complete384-update comparison. Partial evaluation resumes.

Useful fresh-row/outside gains over control and native justify confirmation.
Fitting-only gains suggest remaining generalization limits. A flat result
reduces the case for scaling this input variant; it is not convergence proof.
