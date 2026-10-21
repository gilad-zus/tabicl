#!/usr/bin/env bash
set -euo pipefail
expected_revision="$1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_frozen_transfer pipeline \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_frozen_transfer/v1_seed20261021 \
  --source-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_lr_response/v1_seed20261019 \
  --device cuda --expected-revision "$expected_revision" --query-chunk-rows 2048 --wall-minutes 110
