#!/usr/bin/env bash
set -euo pipefail

# The submitter supplies the tested revision; every phase verifies the same lock.
expected_revision="$1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_statistics_diagnostic pipeline \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_statistics_diagnostic/v1_seed20261010 \
  --device cuda --expected-revision "$expected_revision" --resume
