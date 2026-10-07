# Dataset-diversity startup failure and numerical repair

Original array 33062914 failed on both tasks, October 7 2026. Small ran for
6m01s, large for 2m15s. Both failed in initial reference construction with
`FloatingPointError: nonfinite evaluation`; neither reached an optimizer update.
The error followed the first 20 of 50 validation episodes.

Local analysis of the unchanged frozen validation bank identifies episode 20:
`Credit_Risk_Modeling`, split 0. Its features are finite, and the generated
preprocessing parameters are finite. A query value divided by very small context
variance produces an input around 257,587 standardized units. FP16 cannot
represent that value. The ordinary preprocessing path has a fixed scaler range
guard, while the learned path previously replaced it with an unbounded map.

Repair: clamp context-standardized inputs to [-100, 100] before generated
affine/spline/residual/mixing operations. Apply the same guard during training and
deployment, for every arm and both slots. It uses only context-derived location
and scale, with a fixed bound; query labels/statistics do not configure it.
Moderate inputs keep the existing behavior. This is a numerical-policy change,
not evidence that the network learned a beneficial transformation.

Checked locally:

- All 50 actual validation episodes have finite fresh transforms after repair.
- The failing episode's largest transformed input becomes 100.
- 52 distinct targeted tests pass: dataset-diversity lifecycle, frozen-bank reuse,
  default/gradient checks, context invariance, finite outlier gradients and source-
  attributed failures. CLI inspection and diff checks pass.

Create `results/hyperspline_joint_dataset_diversity/v3_seed20261007` with exactly
the same five data-panel hashes, source allocation, row partitions and counts as
v2. Copy only verified frozen data panels and their descriptive availability audit.
Start from the same CPU seed-zero fresh parameter values. Keep v2 as a failed
startup artifact. Recompute all model-dependent references under the repaired code;
do not import old references, optimizer states or model checkpoints.

Before resubmission, run a small GPU preflight on the failing episode: unbounded
initial-standardization reconstruction on default AMP, followed by repaired
default AMP, FP32 train/inference parity and finite nonzero learning gradients.
No optimizer updates occur in preflight. GPU verification and replacement job IDs
are pending at this document's initial status.

The replacement experiment retains one seed, two serial 40/160-source arms,
4,096 updates each and the common 25-source development validation. No confirmation
test bank is opened. All source fingerprints and data-panel hashes stay locked.
