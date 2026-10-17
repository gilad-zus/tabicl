# All-catalog frozen inference, October 17

Authorized: evaluate all eligible datasets, labeling exposure instead of limiting
the run to a fresh confirmation subset. No hypernetwork training, target fitting,
teacher construction, checkpoint reselection, or validation fallback.

Use the complete existing 371-entry real-data candidate catalog. Try original
185 allocated representatives first, then remaining catalog entries. Accept
one eligible full table per declared source group. Eligibility: at least768
rows,2-10 classes,at least2 examples per class,5-100 encoded columns including
one nonconstant numerical column,finite preprocessing. Recheck descriptions for
known generated data and whole-table duplicate fingerprints. Record every
availability/eligibility rejection. Metadata/provenance choices precede scores.
This catalog remains a finite reviewed inventory, not every OpenML dataset.

Predict with native16 and all four frozen context-size checkpoints:
small_final2048,small_selected1792,large_final2048,large_selected0.
Final small model at512 context on groups absent from current training and
validation is the primary candidate; other comparisons are secondary.
Verify source completion,checkpoint hashes,full-table reservation hashes,
original allocation and all runner/model dependencies. Reuse native logits for
the verified step-zero checkpoint and unchanged eight views of learned models.
No checkpoint is picked per dataset.

For each dataset use two fixed row splits,512 context,plus2048 when enough rows
exist. Contexts are nested and queries identical across contexts/checkpoints.
Current training sources retain their permanently excluded256 probe queries;
two splits vary contexts and share those held-out queries. All other sources
use up to1024 queries,with at least256; datasets below2304 rows have only512
context. Encoding/imputation is fitted on each actual context. No row cap on
source tables; retain only sampled encoded inference episodes after preparation.

Exposure labels:
- training_source: source group used in the current83-source hypernetwork fit;
  held-out queries measure fresh rows from known datasets.
- validation_source: current ten development-validation groups,with new row
  sampling; selection already used these source groups.
- previously_declared_source: conservatively covered by prior185 allocation,
  older real-meta candidate groups,or historical-name inventory. Some were
  training/validation/test sources in older experiments; declaration alone is
  not proof every source had performance inspected.
- catalog_only_source: absent from that history registry. This is metadata
  evidence,not a guarantee that no concealed copy or unrecorded inspection exists.

Report all labels separately; also all sources,all groups absent from current
training,and groups absent from both current training and validation. The last
panel still includes previously declared sources and is qualified accordingly.
Average two splits within dataset before equal-weight NLL/win/loss/bootstrap
aggregation. Report NLL reduction,20k-draw paired dataset bootstrap intervals,
median/percentiles,harms,accuracy,binaryAUC and macroF1. Source-label summaries
must accompany aggregate results. Do not rename previously inspected groups as
fresh independent confirmation.

CPU preparation uses runnohup on dsiofir01,logs named explicitly. GPU inference
uses submit_gpu uriofir,1GPU/4CPU/32G,at most8h with mail enabled. Per-episode
JSON caches permit resume without recomputing completed predictions. No weights
or full source tables need downloading for result analysis.

Result root: results/hyperspline_catalog_inference/v1_seed20261017.
Local checks:10 tests passed,including actual tiny-TabICL parity of cached-view
inference against the original predictor,identity reuse,no parameter changes,
exact/nested sizes,reserved query protection,exposure separation,dataset-level
aggregation,checkpoint-tamper rejection,cached preparation,and resume skipping.
