#!/usr/bin/env bash
set -euo pipefail
expected_revision="$1"
command="$2"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
/home/eng/zusmang/try_micormamba/.venv_311_ticl/bin/python -u -m scripts.joint_preprocessing_lr_response "$command" \
  --output-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_lr_response/v1_seed20261019 \
  --parent-dir /home/dsi/zusmang/TabICL/tabicl/results/hyperspline_fraction_context/v4_bounded320_seed20261018 \
  --parent-state-sha256 4df5679950a0ce437dd051dd698635d4b7a1345d9c156c73b25847bbd7c262c7 \
  --additional-updates 256 --wall-minutes 450 --evaluation-reserve-minutes 120 \
  --device cuda --expected-revision "$expected_revision" --resume
