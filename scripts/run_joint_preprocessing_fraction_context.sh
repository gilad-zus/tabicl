#!/usr/bin/env bash
set -euo pipefail
expected_revision="$1"
command="$2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_fraction_context "$command" \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_fraction_context/v3_cap20000_query2048_seed20261018 \
  --reuse-prepared-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_fraction_context/v1_seed20261018 \
  --continue-from-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_fraction_context/v2_cap20000_seed20261018 \
  --context-row-cap 20000 --training-query-row-cap 2048 --query-chunk-rows 2048 --evaluate-every 32 \
  --source-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_context_size/v1_seed20261016 \
  --catalog-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_catalog_inference/v1_seed20261017 \
  --device cuda --expected-revision "$expected_revision" --resume --wall-minutes 450
