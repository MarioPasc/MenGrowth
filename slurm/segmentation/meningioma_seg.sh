#!/usr/bin/env bash
# =============================================================================
# MENINGIOMA SEGMENTATION — SLURM LAUNCHER
#
# Login-node script that:
#   1. Pulls the BraTS 2025 Meningioma 1st-place Docker image as Singularity SIF
#      (compute nodes have no internet)
#   2. Runs mengrowth-segment prepare to create BraTS-format input
#   3. Submits a SLURM job to run inference on the prepared data
#
# The container is the BraTS 2025 Meningioma Segmentation 1st place winner
# by Yu Haitao et al., distributed via BrainLesion BraTS Orchestrator.
#   Docker image: brainles/brats25_men_qing:latest
#
# Input data must be BraTS-preprocessed: co-registered, skull-stripped,
# registered to SRI24 atlas, with shape (240, 240, 155).
# Required modalities: T1c, T1n, T2f, T2w.
#
# Reference:
#   BraTS Orchestrator — Kofler et al. (2025), arXiv:2506.13807
#   BraTS 2023 Meningioma Challenge — LaBella et al. (2023), arXiv:2305.07642
#   BraTS Meningioma Dataset — LaBella et al. (2024), Scientific Data 11:496
#
# Usage (from Picasso login node):
#   bash slurm/segmentation/meningioma_seg.sh
#   bash slurm/segmentation/meningioma_seg.sh --patient MenGrowth-0015
#   bash slurm/segmentation/meningioma_seg.sh --depends-on 123456
#   bash slurm/segmentation/meningioma_seg.sh --config configs/picasso/segmentation.yaml
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "================================================="
echo "MENINGIOMA SEGMENTATION — LAUNCHER"
echo "================================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="mengrowth"

# Defaults
CONFIG_FILE="${REPO_ROOT}/configs/picasso/segmentation.yaml"
PATIENT_ARG=""
WALL_TIME="0-01:00:00"
DEPENDS_ON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --patient)
            PATIENT_ARG="$2"
            shift 2
            ;;
        --wall-time)
            WALL_TIME="$2"
            shift 2
            ;;
        --depends-on)
            DEPENDS_ON="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash meningioma_seg.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Path to segmentation YAML config (default: configs/picasso/segmentation.yaml)"
            echo "  --patient ID        Process only this patient (e.g., MenGrowth-0015)"
            echo "  --wall-time T       SLURM wall time (default: 0-01:00:00)"
            echo "  --depends-on JID    SLURM job ID to depend on (afterok dependency)"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

export CONFIG_FILE

# Extract SIF path, docker image, and log dir from config
# Uses grep+awk to avoid depending on Python/pyyaml before conda activation
SIF_PATH=$(grep 'sif_path:' "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
DOCKER_IMAGE=$(grep 'docker_image:' "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
LOG_DIR=$(grep 'log_dir:' "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')

export SIF_PATH

echo "Configuration:"
echo "  Config file: ${CONFIG_FILE}"
echo "  Docker image: ${DOCKER_IMAGE}"
echo "  SIF path:    ${SIF_PATH}"
if [ -n "${PATIENT_ARG}" ]; then
    echo "  Patient:     ${PATIENT_ARG}"
fi
echo "  Wall time:   ${WALL_TIME}"
if [ -n "${DEPENDS_ON}" ]; then
    echo "  Depends on:  ${DEPENDS_ON} (afterok)"
fi
echo ""

# ========================================================================
# STEP 1: PULL SINGULARITY IMAGE (login node has internet)
# ========================================================================
echo "================================================="
echo "STEP 1: Singularity image"
echo "================================================="

SIF_DIR="$(dirname "${SIF_PATH}")"
mkdir -p "${SIF_DIR}"

if [ -f "${SIF_PATH}" ]; then
    echo "  SIF already exists: ${SIF_PATH}"
    echo "  Size: $(du -h "${SIF_PATH}" | cut -f1)"
    echo "  (Delete and re-run to force re-pull)"
else
    echo "  Pulling Docker image -> Singularity SIF..."
    echo "  This may take 10-30 min depending on image size and network speed."
    echo ""

    module load singularity 2>/dev/null || true

    singularity pull "${SIF_PATH}" "docker://${DOCKER_IMAGE}"

    if [ $? -eq 0 ]; then
        echo ""
        echo "  SIF created successfully: ${SIF_PATH}"
        echo "  Size: $(du -h "${SIF_PATH}" | cut -f1)"
    else
        echo ""
        echo "  ERROR: Singularity pull failed."
        echo "  Try manually: singularity pull ${SIF_PATH} docker://${DOCKER_IMAGE}"
        exit 1
    fi
fi
echo ""

# ========================================================================
# STEP 2: PREPARE BRATS INPUT (login node, uses mengrowth-segment CLI)
# ========================================================================
echo "================================================="
echo "STEP 2: Preparing BraTS input"
echo "================================================="

# Activate conda for the prepare step
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail "$m" 2>&1 | grep -qi "${m}"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

# Build prepare command
PREPARE_CMD="mengrowth-segment prepare --config ${CONFIG_FILE} --verbose"
if [ -n "${PATIENT_ARG}" ]; then
    PREPARE_CMD="${PREPARE_CMD} --patient ${PATIENT_ARG}"
fi

echo "Running: ${PREPARE_CMD}"
set +e
PREPARE_OUTPUT=$(${PREPARE_CMD})
PREPARE_EXIT=$?
set -e

if [ "${PREPARE_EXIT}" -ne 0 ]; then
    echo "ERROR: Prepare step failed with exit code ${PREPARE_EXIT}"
    exit "${PREPARE_EXIT}"
fi

# Capture WORK_DIR from the last line of prepare output
WORK_DIR=$(echo "${PREPARE_OUTPUT}" | grep "^WORK_DIR=" | tail -1 | cut -d= -f2-)

if [ -z "${WORK_DIR}" ] || [ ! -d "${WORK_DIR}" ]; then
    echo "ERROR: Failed to capture WORK_DIR from prepare output."
    echo "Prepare output:"
    echo "${PREPARE_OUTPUT}"
    exit 1
fi

export WORK_DIR
echo ""
echo "Work directory: ${WORK_DIR}"
echo ""

# ========================================================================
# STEP 3: SUBMIT SLURM JOB
# ========================================================================
echo "================================================="
echo "STEP 3: Submitting SLURM job"
echo "================================================="

mkdir -p "${LOG_DIR}"

# Build sbatch command
SBATCH_ARGS=(
    --parsable
    --job-name="mg_men_seg"
    --time="${WALL_TIME}"
    --ntasks=1
    --cpus-per-task=8
    --mem=32G
    --constraint=dgx
    --gres=gpu:1
    --output="${LOG_DIR}/men_seg_%j.out"
    --error="${LOG_DIR}/men_seg_%j.err"
    --export=ALL
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/meningioma_seg_worker.sh")

echo ""
echo "================================================="
echo "JOB SUBMITTED"
echo "================================================="
echo "Job ID:     ${JOB_ID}"
echo "Monitor:    squeue -j ${JOB_ID}"
echo "Logs:       ${LOG_DIR}/men_seg_${JOB_ID}.{out,err}"
echo "Work dir:   ${WORK_DIR}"
echo ""
