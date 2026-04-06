#!/usr/bin/env bash
# =============================================================================
# MENGROWTH PREPROCESSING — ECLARE SLURM LAUNCHER
#
# Same as preprocess.sh but with resources tuned for composite resampling
# (BSpline + ECLARE deep-learning super-resolution for thick-slice volumes).
#
# Differences from preprocess.sh:
#   - Walltime: 4h (vs 2h) — ECLARE adds ~1-2 min per thick-slice volume
#   - Memory:  96G (vs 64G) — ECLARE spawns 16 data-loader workers (~1GB each)
#   - Config:  preprocessing.yaml with method=composite
#   - Pre-downloads ECLARE model weights on login node
#
# Usage (from login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth
#
#   # Full dataset:
#   bash slurm/preprocessing/preprocess_eclare.sh
#
#   # Specific patients:
#   bash slurm/preprocessing/preprocess_eclare.sh --patients MenGrowth-0028,MenGrowth-0023
#
#   # Limit concurrency:
#   bash slurm/preprocessing/preprocess_eclare.sh --max-concurrent 16
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "MENGROWTH PREPROCESSING LAUNCHER (ECLARE)"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION
# ========================================================================
export CONDA_ENV_NAME="mengrowth"
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth"
export CONFIG_FILE="${REPO_SRC}/configs/picasso/preprocessing.yaml"
export DATASET_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/dataset/MenGrowth-2025"
export LOG_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/mengrowth-dataset/logs"

# ECLARE conda env (may differ from main pipeline env)
export ECLARE_CONDA_ENV="mengrowth"

# Defaults
MAX_CONCURRENT=32
PATIENTS_OVERRIDE=""

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --patients)
            PATIENTS_OVERRIDE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash slurm/preprocessing/preprocess_eclare.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-concurrent N    Max concurrent SLURM array tasks (default: 32)"
            echo "  --patients P1,P2,...  Comma-separated patient IDs (default: all)"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ========================================================================
# BUILD PATIENT LIST
# ========================================================================
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PATIENT_LIST="${LOG_DIR}/patient_list_eclare_${TIMESTAMP}.txt"

if [ -n "${PATIENTS_OVERRIDE}" ]; then
    echo "${PATIENTS_OVERRIDE}" | tr ',' '\n' > "${PATIENT_LIST}"
    echo "Patient source: command-line override"
else
    if [ ! -d "${DATASET_ROOT}" ]; then
        echo "ERROR: Dataset root not found: ${DATASET_ROOT}"
        exit 1
    fi
    find "${DATASET_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'MenGrowth-*' \
        | xargs -I{} basename {} \
        | sort \
        > "${PATIENT_LIST}"
    echo "Patient source: auto-discovered from ${DATASET_ROOT}"
fi

NUM_PATIENTS=$(wc -l < "${PATIENT_LIST}")

if [ "${NUM_PATIENTS}" -eq 0 ]; then
    echo "ERROR: No patients found."
    exit 1
fi

echo ""
echo "Configuration (ECLARE):"
echo "  Repo:            ${REPO_SRC}"
echo "  Config:          ${CONFIG_FILE}"
echo "  Dataset:         ${DATASET_ROOT}"
echo "  Conda env:       ${CONDA_ENV_NAME}"
echo "  ECLARE env:      ${ECLARE_CONDA_ENV}"
echo "  Patients:        ${NUM_PATIENTS}"
echo "  Max concurrent:  ${MAX_CONCURRENT}"
echo "  Patient list:    ${PATIENT_LIST}"
echo ""
echo "  Resources per task:"
echo "    Walltime:  4h  (vs 2h standard)"
echo "    Memory:    96G (vs 64G standard)"
echo "    CPUs:      8"
echo "    GPU:       1"
echo ""

# Show first/last patients
echo "Patients to process:"
if [ "${NUM_PATIENTS}" -le 10 ]; then
    cat "${PATIENT_LIST}" | sed 's/^/  /'
else
    head -3 "${PATIENT_LIST}" | sed 's/^/  /'
    echo "  ... (${NUM_PATIENTS} total)"
    tail -3 "${PATIENT_LIST}" | sed 's/^/  /'
fi
echo ""

# ========================================================================
# PRE-DOWNLOAD: HD-BET + ECLARE weights (compute nodes have no internet)
# ========================================================================
echo "Checking model weights..."
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# HD-BET weights
python -c "
from brainles_hd_bet.utils import get_params_fname, maybe_download_parameters
missing = [i for i in range(5) if not get_params_fname(i).exists()]
if missing:
    print(f'  Downloading HD-BET weights for folds: {missing}')
    for fold in missing:
        maybe_download_parameters(fold)
    print('  HD-BET weights: download complete')
else:
    print('  HD-BET weights: all 5 folds already present')
" || {
    echo "WARNING: Could not verify HD-BET weights."
}

# ECLARE weights — trigger a dry-run import so weights get cached
conda run -n "${CONDA_ENV_NAME}" python -c "
try:
    from eclare.model import ECLARE
    print('  ECLARE model: import OK (weights will be cached on first run)')
except ImportError:
    print('  WARNING: ECLARE not importable in env ${CONDA_ENV_NAME}')
" 2>/dev/null || echo "  WARNING: Could not verify ECLARE installation"

echo ""

# ========================================================================
# SUBMIT ARRAY JOB
# ========================================================================
ARRAY_MAX=$((NUM_PATIENTS - 1))

JOB_ID=$(sbatch --parsable \
    --job-name="mgpp-eclare" \
    --time=0-04:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=96G \
    --constraint=dgx \
    --gres=gpu:1 \
    --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
    --output="${LOG_DIR}/preprocess_eclare_%A_%a.out" \
    --error="${LOG_DIR}/preprocess_eclare_%A_%a.err" \
    --export=ALL,PATIENT_LIST="${PATIENT_LIST}" \
    "${SCRIPT_DIR}/preprocess_eclare_worker.sh")

echo "=========================================="
echo "ECLARE JOB ARRAY SUBMITTED"
echo "=========================================="
echo "Job ID:         ${JOB_ID}"
echo "Array range:    0-${ARRAY_MAX} (max ${MAX_CONCURRENT} concurrent)"
echo "Patient list:   ${PATIENT_LIST}"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB_ID}                                # Running tasks"
echo "  sacct -j ${JOB_ID} --format=JobID,State,Elapsed,MaxRSS,ExitCode  # Completed"
echo ""
echo "Logs:"
echo "  ${LOG_DIR}/preprocess_eclare_${JOB_ID}_<task>.out   # stdout per patient"
echo "  ${LOG_DIR}/preprocess_eclare_${JOB_ID}_<task>.err   # stderr per patient"
echo ""
echo "Cancel:"
echo "  scancel ${JOB_ID}"
echo ""
echo "Resume after failure:"
echo "  bash slurm/preprocessing/preprocess_eclare.sh       # Checkpoints auto-skip completed steps"
