#!/bin/bash
set -euo pipefail

LOG_DIR="${HOME}/logs/DeLTA/inference"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_val_masks_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Logging to: $LOG_FILE"

source "${HOME}/.bashrc"
conda activate "${HOME}/MaxWork/conda_envs/delta_tf_blackwell"

REPO_DIR="${HOME}/MaxWork/DeLTA_TF_New"
CTC_ROOT="${REPO_DIR}/data/My2D/CTC"

cd "$REPO_DIR"

for split in test; do

    echo
    echo "======================================"
    echo "Processing ${split}"
    echo "======================================"

    for seq in "${CTC_ROOT}/${split}"/[0-9][0-9]; do
        [ -d "$seq" ] || continue

        echo
        echo "--------------------------------------"
        echo "Sequence: $(basename "$seq")"
        echo "--------------------------------------"

        python scripts/segmentation.py "$seq"

    done

done

echo
echo "Finished all segmentation inference."
