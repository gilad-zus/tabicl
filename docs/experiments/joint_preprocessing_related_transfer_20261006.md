# Related versus cross-domain preprocessing transfer

Agreed **2026-10-06**, seed **0 only**. Two offline continuations from the same
synthetic repeated checkpoint at step 5,120; frozen TabICL; no target weight fitting.
This tests whether a shared preprocessing generator transfers better between tasks
in the same declared domain than between the two domains. Domain membership is a
hypothesis based on task descriptions, not a computed distance or known shared
optimal transformation. No model receives domain labels or feature names.

Closest previous experiments: the broad 40-source synthetic/real/mixed continuation
and the subsequent single/ensemble objective pilot. Real-source fitting was modest
and useful broad baseline improvement was unestablished. The material change is
four sources per model and controlled domain specialization, with a cross-domain
model evaluated on exactly the same targets at the same training budget. This
does not separately estimate the effect of reducing source count versus the old
40-source experiment. Older within-dataset diagnostics did not test this transfer.

## Exact datasets

| Domain | Four training datasets | Validation | Two held-out targets |
|---|---|---|---|
| Clinical | breast_cancer, bupa, diabetes, heart_disease_cleveland | maternal_health_risk | mammography, ilpd |
| Financial | credit-g, credit_card_clients_default, heloc, GiveMeSomeCredit | credit_approval_australia | polish_companies_bankruptcy, taiwanese_bankruptcy_prediction |

Both models use **the same two validation datasets**, with equal dataset weights,
so selection is not another difference between the arms. One dataset has two
evaluation partitions, averaged into one dataset result. Final checkpoints at
update 1,024 are the primary comparison; validation-selected checkpoints, including
update zero, are supplementary.

Membership is locked in [the group manifest](joint_preprocessing_related_transfer_groups_20261006.json).
All fourteen sources have different declared source groups and input hashes in
the existing frozen bank. No alternate thyroid targets, heart aliases, repeated
Polish forecasting horizons, or row splits are treated as different training/test
datasets. These datasets have appeared in earlier research: this is explicitly an
exploratory development diagnostic, not a fresh confirmatory benchmark. The two
new models start from synthetic-only weights, and their targets are excluded from
their training and checkpoint selection.

The financial group is deliberately broad: consumer credit versus company
bankruptcy is a remaining task mismatch. The clinical group also has different
measurement types and diseases. A negative result cannot rule out transfer between
more closely matched independent cohorts. Two targets per domain cannot establish
broad generality. The source descriptions support the task labels, not a claim
that their preprocessing must match: [ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset),
[Polish bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
[Taiwanese bankruptcy](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction).

## Training and evaluation

- Two serial runs: clinical, then financial. One seed and one GPU allocation.
- 1,024 AdamW updates per model, four episodes per update: one from each source.
  Each dataset therefore supplies 1,024 presentations, approximately ten times the
  per-dataset allocation of the previous 40-source pilot at the same update count.
- Same fixed original checkpoint/architecture; full hypernetwork trainable,
  TabICL parameters frozen. No teacher, per-target fitted gate, or new architecture.
- Ensemble CE on ordinary8+learned8, logit blend weight 0.5, temperature 0.9.
  The existing memory-bounded exact gradient and GPU numerical parity audit are reused.
- Learning rate 0.0003, weight decay 0.0001, betas (0.9,0.999), epsilon 1e-8,
  gradient norm clip 1, float32 training; existing CUDA AMP evaluation.
- Requested lengths 128/256/512/1024 and context fractions 0.5/0.7/0.85 follow the
  same schedule. A **shared cap of 303 rows** from the smallest source applies
  to both arms: actual lengths 128/256/303/303. Queries number ceil(N*(1-f)),
  approximately 20–152 per episode; both partitions retain every class.
- Evaluation every 256 updates, durable saves every 25. Source probe has four
  fixed episodes per arm, capped at 303 rows; common validation has four episodes.
- Targets have two fixed seeds (0/1), 70% context, up to 1,024 rows. Context/query
  indices are disjoint within each episode. Partitions of a target are never
  training datasets. Financial targets and mammography use 1,024 rows (~308 query),
  ilpd uses 583 (~175 query); both models see exactly the same target episodes.
- Both full runs and selected/final checkpoint hashes must be locked before the
  training pipeline can deserialize its separate target bank for evaluation.
- Compare related-trained versus cross-trained models on each target, ordinary16,
  and the unchanged starting blend. Also retain learned8 versus ordinary8 results.
  Report NLL deltas, relative gains with floor 1e-4, medians, wins/losses and harms.
  Four-dataset bootstrap intervals are descriptive only.
- Log training CE, gradients by head/encoder, clipping, source/validation learning
  relative to initialization, transformation diagnostics, episode sizes and timing.
  Weak source fitting makes a negative transfer result inconclusive about transfer.

## Execution

Runner: `scripts/joint_preprocessing_related_transfer.py`.
Commands: `prepare`, `train`, `lock`, `test`, resumable `pipeline`.
Result root: `results/hyperspline_joint_related_transfer/v1_seed20261006`.
CPU preparation subsets the existing frozen bank; no data download or synthetic
generation is needed. GPU estimate: roughly 2–3 hours including evaluation, with
dataset-dependent variability. Request one GPU, 32G RAM, four CPUs, four hours on
the default uriofir profile, email enabled.

Status **2026-10-06**: implementation complete; **19 distinct targeted tests passed**
(new runner and inherited ensemble gradient/deployment checks), CLI verified,
and local metadata audit confirms all 14 sources and the 303-row cap. Not yet
submitted. Exact revision, preparation log and SLURM job/log paths will be
recorded after launch. No additional seeds or automatic scale-up are authorized.
