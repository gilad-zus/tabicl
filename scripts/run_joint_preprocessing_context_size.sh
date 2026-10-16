#!/usr/bin/env bash
set -euo pipefail

expected_revision="$1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_context_size pipeline \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_context_size/v1_seed20261016 \
  --reuse-bank-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_dataset_diversity/v4_seed20261007 \
  --source-cache-dirs /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_dataset_diversity/v1_seed20261007/source_cache /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_dataset_diversity/v2_seed20261007/source_cache \
  --device cuda --expected-revision "$expected_revision" --resume --wall-minutes 450
