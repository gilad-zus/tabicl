#!/usr/bin/env bash
set -euo pipefail
expected_revision="$1"
command="$2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_catalog_inference "$command" \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_catalog_inference/v1_seed20261017 \
  --source-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_context_size/v1_seed20261016 \
  --original-bank-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_dataset_diversity/v4_seed20261007 \
  --source-cache-dirs /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_dataset_diversity/v1_seed20261007/source_cache /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_joint_dataset_diversity/v2_seed20261007/source_cache \
  --cache-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_catalog_inference/data_cache \
  --device cuda --expected-revision "$expected_revision" --wall-minutes 450
