#!/usr/bin/env bash
#SBATCH -J mg_men_seg
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# MENINGIOMA SEGMENTATION — SLURM WORKER
#
# Runs the BraTS 2025 Meningioma Segmentation 1st-place algorithm
# (Yu Haitao et al.) via Singularity on an A100 GPU, then calls
# mengrowth-segment postprocess to remap outputs.
#
# Expected env vars (exported by meningioma_seg.sh launcher):
#   CONDA_ENV_NAME, REPO_SRC, CONFIG_FILE,
#   SIF_PATH, WORK_DIR
#
# Reference:
#   BraTS Orchestrator — Kofler et al. (2025), arXiv:2506.13807
#   Container interface: /input (ro), /output (rw)
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "================================================="
echo "MENINGIOMA SEGMENTATION — WORKER"
echo "================================================="
echo "Time:     $(date)"
echo "Hostname: $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================

# Load singularity module
module load singularity 2>/dev/null || true

# Load conda
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail "$m" 2>&1 | grep -qi "${m}"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "=========================================="
echo "ENVIRONMENT VERIFICATION"
echo "=========================================="
echo "[singularity] $(singularity --version || echo 'NOT FOUND')"
echo "[python]      $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

# Verify SIF exists
if [ -f "${SIF_PATH}" ]; then
    echo "[OK]   SIF image: ${SIF_PATH} ($(du -h "${SIF_PATH}" | cut -f1))"
else
    echo "[FAIL] SIF image not found: ${SIF_PATH}"
    echo "       Run the launcher from the login node first."
    exit 1
fi

# Verify work directory
if [ -d "${WORK_DIR}" ]; then
    echo "[OK]   Work dir: ${WORK_DIR}"
else
    echo "[FAIL] Work dir not found: ${WORK_DIR}"
    exit 1
fi

# Verify GPU is accessible via Singularity --nv
singularity exec --nv "${SIF_PATH}" nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1 && \
    echo "[OK]   Singularity --nv GPU passthrough" || \
    echo "[WARN] Singularity --nv GPU check failed; inference may fall back to CPU or error"

echo "Pre-flight checks PASSED"
echo ""

# ========================================================================
# RUN SINGULARITY INFERENCE
# ========================================================================
echo "=========================================="
echo "RUNNING INFERENCE"
echo "=========================================="

BRATS_INPUT="${WORK_DIR}/input"
BRATS_OUTPUT="${WORK_DIR}/output"

echo "SIF:    ${SIF_PATH}"
echo "Input:  ${BRATS_INPUT}"
echo "Output: ${BRATS_OUTPUT}"
echo ""

INFER_START=$(date +%s)

set +e
singularity run \
    --nv \
    --cleanenv \
    --no-home \
    --writable-tmpfs \
    --bind "${BRATS_INPUT}:/input:ro" \
    --bind "${BRATS_OUTPUT}:/output:rw" \
    "${SIF_PATH}"

INFER_EXIT=$?
set -e

INFER_EXIT=$?
INFER_END=$(date +%s)
INFER_ELAPSED=$((INFER_END - INFER_START))

echo ""
echo "Inference exit code: ${INFER_EXIT}"
echo "Inference duration:  $(($INFER_ELAPSED / 60))m $(($INFER_ELAPSED % 60))s"

if [ "${INFER_EXIT}" -ne 0 ]; then
    echo "ERROR: Singularity inference failed with exit code ${INFER_EXIT}"
    echo ""
    echo "Debugging tips:"
    echo "  1. Inspect container interactively:"
    echo "     singularity shell --nv ${SIF_PATH}"
    echo "  2. Check container's runscript:"
    echo "     singularity inspect --runscript ${SIF_PATH}"
    rm -rf "${WORK_DIR}"
    exit "${INFER_EXIT}"
fi

# ========================================================================
# POST-PROCESS: REMAP OUTPUTS TO STUDY DIRECTORIES
# ========================================================================
echo ""
echo "=========================================="
echo "POST-PROCESSING"
echo "=========================================="

mengrowth-segment postprocess \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR}" \
    --verbose

POST_EXIT=$?

# ========================================================================
# CLEANUP
# ========================================================================
echo ""
echo "Cleaning up work directory: ${WORK_DIR}"
rm -rf "${WORK_DIR}"

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "WORKER COMPLETED"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo ""

if [ "${INFER_EXIT}" -eq 0 ] && [ "${POST_EXIT}" -eq 0 ]; then
    echo "Segmentation completed successfully."
    exit 0
else
    echo "Segmentation FAILED (inference=${INFER_EXIT}, postproc=${POST_EXIT})"
    exit 1
fi
