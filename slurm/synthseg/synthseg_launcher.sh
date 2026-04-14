#!/usr/bin/env bash
# =============================================================================
# MenGrowth SynthSeg — SLURM LAUNCHER
#
# Login-node script that:
#   1. Discovers patient directories under the configured input_root.
#   2. Writes a one-patient-per-line manifest to log_dir.
#   3. Submits a SLURM array job where each task processes all studies of
#      one patient via mengrowth-synthseg run-patient.
#
# Usage (from Picasso login node):
#   bash slurm/synthseg/synthseg_launcher.sh
#   bash slurm/synthseg/synthseg_launcher.sh --patient MenGrowth-0017
#   bash slurm/synthseg/synthseg_launcher.sh --depends-on 123456
#   bash slurm/synthseg/synthseg_launcher.sh --config configs/picasso/synthseg.yaml
#   bash slurm/synthseg/synthseg_launcher.sh --max-concurrent 4
#
# Reference:
#   Billot et al., "SynthSeg: Segmentation of brain MRI scans of any
#   contrast and resolution without retraining", MedIA 2023.
#   Billot et al., "Robust machine learning segmentation for large-scale
#   analysis of heterogeneous clinical brain MRI datasets", PNAS 2023.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "================================================="
echo "MENGROWTH SYNTHSEG — LAUNCHER"
echo "================================================="
echo "Time: $(date)"
echo ""

# -----------------------------------------------------------------------------
# Defaults (conda env hosting mengrowth CLI — NOT the SynthSeg env; the worker
# switches to the SynthSeg env before running SynthSeg itself)
# -----------------------------------------------------------------------------
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-mengrowth}"

CONFIG_FILE="${REPO_ROOT}/configs/picasso/synthseg.yaml"
PATIENT_ARG=""
WALL_TIME="0-02:00:00"
DEPENDS_ON=""
MAX_CONCURRENT=8

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)          CONFIG_FILE="$2";     shift 2 ;;
        --patient)         PATIENT_ARG="$2";     shift 2 ;;
        --wall-time)       WALL_TIME="$2";       shift 2 ;;
        --depends-on)      DEPENDS_ON="$2";      shift 2 ;;
        --max-concurrent)  MAX_CONCURRENT="$2";  shift 2 ;;
        --help|-h)
            cat <<'EOF'
Usage: bash synthseg_launcher.sh [OPTIONS]

Options:
  --config PATH          Path to synthseg YAML (default: configs/picasso/synthseg.yaml)
  --patient ID           Run a single-patient array job (array size = 1)
  --wall-time T          SLURM wall time per array task (default: 0-02:00:00)
  --depends-on JID       Chain to a prior job (afterok)
  --max-concurrent N     Cap concurrent array tasks (default: 8)
  --help, -h             Show this help
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Extract log_dir from YAML (grep/awk to avoid Python before conda activation)
# -----------------------------------------------------------------------------
LOG_DIR=$(grep '^\s*log_dir:' "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
if [ -z "${LOG_DIR}" ]; then
    echo "ERROR: log_dir not set in ${CONFIG_FILE}"
    exit 1
fi
mkdir -p "${LOG_DIR}"

echo "Configuration:"
echo "  Config file:     ${CONFIG_FILE}"
echo "  Log dir:         ${LOG_DIR}"
echo "  Wall time:       ${WALL_TIME}"
echo "  Max concurrent:  ${MAX_CONCURRENT}"
if [ -n "${PATIENT_ARG}" ]; then
    echo "  Patient:         ${PATIENT_ARG}  (array size will be 1)"
fi
if [ -n "${DEPENDS_ON}" ]; then
    echo "  Depends on:      ${DEPENDS_ON} (afterok)"
fi
echo ""

# -----------------------------------------------------------------------------
# STEP 1: Build patient manifest
#
# We run `mengrowth-synthseg discover` on the login node (no SLURM queue
# wait), which requires the mengrowth conda env to already be active (or
# on PATH). The launcher activates it via the same idiom as the worker.
# -----------------------------------------------------------------------------
echo "================================================="
echo "STEP 1: Discovering patients"
echo "================================================="

# Activate conda (best effort on login node)
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    module load "$m" 2>/dev/null && break || true
done
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}" || true
fi

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
PATIENT_LIST="${LOG_DIR}/patients_${STAMP}.txt"

if [ -n "${PATIENT_ARG}" ]; then
    echo "${PATIENT_ARG}" > "${PATIENT_LIST}"
else
    mengrowth-synthseg discover --config "${CONFIG_FILE}" --output "${PATIENT_LIST}"
fi

if [ ! -s "${PATIENT_LIST}" ]; then
    echo "ERROR: Patient manifest is empty: ${PATIENT_LIST}"
    exit 1
fi

N_PATIENTS=$(wc -l < "${PATIENT_LIST}")
echo "  Manifest:     ${PATIENT_LIST}"
echo "  N patients:   ${N_PATIENTS}"
echo ""

# -----------------------------------------------------------------------------
# STEP 2: Submit the array job
# -----------------------------------------------------------------------------
echo "================================================="
echo "STEP 2: Submitting SLURM array job"
echo "================================================="

ARRAY_SPEC="0-$((N_PATIENTS - 1))%${MAX_CONCURRENT}"

SBATCH_ARGS=(
    --parsable
    --job-name="mg_synthseg"
    --time="${WALL_TIME}"
    --array="${ARRAY_SPEC}"
    --output="${LOG_DIR}/synthseg_%A_%a.out"
    --error="${LOG_DIR}/synthseg_%A_%a.err"
    --export="ALL,CONFIG_FILE=${CONFIG_FILE},PATIENT_LIST=${PATIENT_LIST},CONDA_ENV_NAME=${CONDA_ENV_NAME}"
)

if [ -n "${DEPENDS_ON}" ]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDS_ON}")
fi

JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/synthseg_worker.sh")

echo ""
echo "================================================="
echo "JOB SUBMITTED"
echo "================================================="
echo "Job ID:       ${JOB_ID}"
echo "Array spec:   ${ARRAY_SPEC}"
echo "Manifest:     ${PATIENT_LIST}"
echo "Monitor:      squeue -j ${JOB_ID}"
echo "Logs:         ${LOG_DIR}/synthseg_${JOB_ID}_*.{out,err}"
echo "Verify:       mengrowth-synthseg verify --config ${CONFIG_FILE}"
echo ""
