#!/usr/bin/env bash
#SBATCH -J mg_men_seg
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# MENINGIOMA SEGMENTATION — SLURM WORKER
#
# Runs the full segmentation pipeline on the compute node:
#   1. Prepare BraTS-format input (mengrowth-segment prepare)
#   2. Run inference via Singularity
#   3. Post-process outputs back to study directories
#
# Prepare runs here (not on the login node) so that --depends-on works
# correctly: the data must be fully preprocessed before prepare discovers it.
#
# Expected env vars (exported by meningioma_seg.sh launcher):
#   CONDA_ENV_NAME, CONFIG_FILE, SIF_PATH, PATIENT_ARG
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
# STEP 1: PREPARE BRATS INPUT
# ========================================================================
echo "=========================================="
echo "STEP 1: PREPARE BRATS INPUT"
echo "=========================================="

PREPARE_CMD="mengrowth-segment prepare --config ${CONFIG_FILE} --verbose"
if [ -n "${PATIENT_ARG:-}" ]; then
    PREPARE_CMD="${PREPARE_CMD} --patient ${PATIENT_ARG}"
fi

echo "Running: ${PREPARE_CMD}"
set +e
PREPARE_OUTPUT=$(${PREPARE_CMD})
PREPARE_EXIT=$?
set -e

if [ "${PREPARE_EXIT}" -ne 0 ]; then
    echo "ERROR: Prepare step failed with exit code ${PREPARE_EXIT}"
    echo "${PREPARE_OUTPUT}"
    exit "${PREPARE_EXIT}"
fi

# Capture WORK_DIR from prepare output
WORK_DIR=$(echo "${PREPARE_OUTPUT}" | grep "^WORK_DIR=" | tail -1 | cut -d= -f2-)

if [ -z "${WORK_DIR}" ] || [ ! -d "${WORK_DIR}" ]; then
    echo "ERROR: Failed to capture WORK_DIR from prepare output."
    echo "Prepare output:"
    echo "${PREPARE_OUTPUT}"
    exit 1
fi

echo "Work directory: ${WORK_DIR}"
echo ""

# ========================================================================
# STEP 2: PRE-FLIGHT CHECKS
# ========================================================================
echo "=========================================="
echo "STEP 2: PRE-FLIGHT CHECKS"
echo "=========================================="

# Verify SIF exists
if [ -f "${SIF_PATH}" ]; then
    echo "[OK]   SIF image: ${SIF_PATH} ($(du -h "${SIF_PATH}" | cut -f1))"
else
    echo "[FAIL] SIF image not found: ${SIF_PATH}"
    echo "       Run the launcher from the login node first."
    exit 1
fi

# Verify GPU is accessible via Singularity --nv
singularity exec --nv "${SIF_PATH}" nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1 && \
    echo "[OK]   Singularity --nv GPU passthrough" || \
    echo "[WARN] Singularity --nv GPU check failed; inference may fall back to CPU or error"

# Detect container WORKDIR (inference.py location)
CONTAINER_PWD=$(singularity exec --nv "${SIF_PATH}" \
    find / -maxdepth 3 -name inference.py -printf '%h\n' -quit 2>/dev/null || true)
if [ -z "${CONTAINER_PWD}" ]; then
    CONTAINER_PWD="/app"
    echo "[WARN] Could not detect inference.py location, defaulting to ${CONTAINER_PWD}"
else
    echo "[OK]   inference.py found in: ${CONTAINER_PWD}"
fi

echo "Pre-flight checks PASSED"
echo ""

# ========================================================================
# STEP 3: RUN SINGULARITY INFERENCE
# ========================================================================
echo "=========================================="
echo "STEP 3: RUNNING INFERENCE"
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
    --pwd "${CONTAINER_PWD}" \
    --bind "${BRATS_INPUT}:/input:ro" \
    --bind "${BRATS_OUTPUT}:/output:rw" \
    "${SIF_PATH}"

INFER_EXIT=$?
set -e

INFER_END=$(date +%s)
INFER_ELAPSED=$((INFER_END - INFER_START))

echo ""
echo "Inference exit code: ${INFER_EXIT}"
echo "Inference duration:  $(($INFER_ELAPSED / 60))m $(($INFER_ELAPSED % 60))s"

if [ "${INFER_EXIT}" -ne 0 ]; then
    echo "ERROR: Singularity inference failed with exit code ${INFER_EXIT}"
    echo ""
    echo "Debugging tips:"
    echo "  1. Inspect work dir: ${WORK_DIR}"
    echo "  2. Inspect container interactively:"
    echo "     singularity shell --nv ${SIF_PATH}"
    echo "  3. Check container's runscript:"
    echo "     singularity inspect --runscript ${SIF_PATH}"
    echo ""
    echo "Work directory preserved for debugging."
    exit "${INFER_EXIT}"
fi

# ========================================================================
# STEP 4: POST-PROCESS
# ========================================================================
echo ""
echo "=========================================="
echo "STEP 4: POST-PROCESSING"
echo "=========================================="

mengrowth-segment postprocess \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR}" \
    --verbose

POST_EXIT=$?

# ========================================================================
# CLEANUP & COMPLETION
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
    echo "Cleaning up work directory: ${WORK_DIR}"
    rm -rf "${WORK_DIR}"
    exit 0
else
    echo "Segmentation FAILED (inference=${INFER_EXIT}, postproc=${POST_EXIT})"
    echo "Work directory preserved for debugging: ${WORK_DIR}"
    exit 1
fi
