#!/bin/bash
set -euo pipefail

LOG_DIR="${HOME}/logs/DeLTA/tracking"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_tracking_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Logging to: $LOG_FILE"

source "${HOME}/.bashrc"
conda activate "${HOME}/MaxWork/conda_envs/delta_tf_blackwell"

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

REPO_DIR="${HOME}/MaxWork/DeLTA_TF_New"
TEST_ROOT="${REPO_DIR}/data/My2D/CTC/test"
MODEL_PATH="${REPO_DIR}/models/My2D/track/my2d_track_model.keras"
MODEL_TAG="$(basename "${MODEL_PATH}" .keras)"

cd "$REPO_DIR"

echo "======================================"
echo "Running tracking on test sequences"
echo "======================================"

for seq in "${TEST_ROOT}"/[0-9][0-9]; do

    [ -d "$seq" ] || continue

    echo
    echo "--------------------------------------"
    echo "Sequence: $(basename "$seq")"
    echo "--------------------------------------"

    if [ ! -d "${seq}/segmentation" ]; then
        echo "Skipping ${seq} (no segmentation folder found)"
        continue
    fi
    python scripts/tracking.py \
        --inputs_folder "$seq" \
        --model-path "${MODEL_PATH}" \
        --model-tag "${MODEL_TAG}" \
        --frame-metrics-csv "${seq}/tracking/metrics/${MODEL_TAG}/frame_metrics.csv"
done

echo
echo "Finished at: $(date)"
echo "Done."
