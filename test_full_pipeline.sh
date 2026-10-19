#!/bin/bash
set -euo pipefail

LOG_DIR="${HOME}/logs/DeLTA/pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Logging to: $LOG_FILE"

source "${HOME}/.bashrc"
conda activate "${HOME}/MaxWork/conda_envs/delta_tf_blackwell"

REPO_DIR="${HOME}/MaxWork/DeLTA_TF_New"
CTC_ROOT="${REPO_DIR}/data/My2D/CTC"

export LD_LIBRARY_PATH="/home/max/MaxWork/conda_envs/delta_tf_blackwell/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/home/max/MaxWork/conda_envs/delta_tf_blackwell/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/max/MaxWork/conda_envs/delta_tf_blackwell/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/max/MaxWork/conda_envs/delta_tf_blackwell/lib/python3.10/site-packages/nvidia/cublas/lib:/home/max/MaxWork/conda_envs/delta_tf_blackwell/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO_DIR"

for split in test; do
    echo
    echo "======================================"
    echo "Processing ${split}"
    echo "======================================"

    python scripts/run_pipeline.py --split "$split"

done

echo
echo "Finished all pipeline inference."