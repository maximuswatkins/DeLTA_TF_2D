#! /bin/bash
set -euo pipefail

# Put logs somewhere easy to find
LOG_DIR="${HOME}/logs/DeLTA"
mkdir -p "$LOG_DIR"

# Timestamped log file
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# Send everything to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Logging to: $LOG_FILE"

# Make conda available
source "${HOME}/.bashrc"

# Activate DeLTA TensorFlow environment
conda activate "${HOME}/MaxWork/conda_envs/delta_tf_blackwell"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

# ---- EDIT THESE LINES -------
DATASET_NAME="My2D"
REPO_DIR="${HOME}/MaxWork/DeLTA_TF_New"
CTC_DIR="${REPO_DIR}/data/${DATASET_NAME}/CTC"
# ------------------------------

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_DIR}"

echo "Repo directory: $(pwd)"
echo "CTC dataset root: ${CTC_DIR}"

echo "Checking dataset structure..."
test -d "${CTC_DIR}/train"
test -d "${CTC_DIR}/val"

find "${CTC_DIR}" -maxdepth 3 -type d | sort | head -80

echo "Checking CTC tracking annotations..."
find "${CTC_DIR}/train" -name "man_track.txt" | head
find "${CTC_DIR}/train" -name "man_track*.tif" | head
find "${CTC_DIR}/train" -name "man_seg*.tif" | head

echo "Checking Python environment..."
which python
python --version

echo "Checking TensorFlow..."
python - <<'PY'
import tensorflow as tf
import delta

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("DeLTA:", delta.__file__)
PY

echo
echo "===== Starting DeLTA training ====="

TOTAL_START=$(date +%s)

############################################
# Segmentation
############################################

SEG_START=$(date +%s)

echo
echo "===== Training segmentation model ====="

python scripts/train_seg.py

SEG_END=$(date +%s)

echo
echo "===== Segmentation training finished ====="
echo "Segmentation time:"
echo "  Seconds : $((SEG_END - SEG_START))"
echo "  Minutes : $(((SEG_END - SEG_START) / 60))"

############################################
# Tracking
############################################

TRACK_START=$(date +%s)

echo
echo "===== Training tracking model ====="

python scripts/train_track.py

TRACK_END=$(date +%s)

echo
echo "===== Tracking training finished ====="
echo "Tracking time:"
echo "  Seconds : $((TRACK_END - TRACK_START))"
echo "  Minutes : $(((TRACK_END - TRACK_START) / 60))"

############################################
# Total
############################################

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo
echo "=========================================="
echo "Finished at: $(date)"
echo
echo "Training summary"
echo "----------------"
echo "Segmentation : $(((SEG_END - SEG_START) / 60)) min"
echo "Tracking     : $(((TRACK_END - TRACK_START) / 60)) min"
echo "Total        : ${TOTAL_TIME} sec"
echo "               $((TOTAL_TIME / 60)) min"
echo "               $((TOTAL_TIME / 3600)) h"
echo "=========================================="
