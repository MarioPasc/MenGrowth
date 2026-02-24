#!/usr/bin/env bash
#SBATCH -J mgpp
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=preprocess_%A_%a.out
#SBATCH --error=preprocess_%A_%a.err

# =============================================================================
# MENGROWTH PREPROCESSING WORKER
#
# Processes a single patient from the SLURM array. Each array task reads its
# patient ID from the PATIENT_LIST file using SLURM_ARRAY_TASK_ID as line index.
#
# Expected env vars (exported by preprocess.sh launcher):
#   REPO_SRC, CONFIG_FILE, DATASET_ROOT, CONDA_ENV_NAME, PATIENT_LIST
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "MenGrowth preprocessing worker started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"

# ========================================================================
# RESOLVE PATIENT ID
# ========================================================================
if [ -z "${PATIENT_LIST:-}" ]; then
    echo "ERROR: PATIENT_LIST not set. Run via preprocess.sh launcher."
    exit 1
fi

if [ ! -f "${PATIENT_LIST}" ]; then
    echo "ERROR: Patient list not found: ${PATIENT_LIST}"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
# Lines are 0-indexed: task 0 = line 1, task 1 = line 2, etc.
LINE_NUM=$((TASK_ID + 1))
PATIENT_ID=$(sed -n "${LINE_NUM}p" "${PATIENT_LIST}")

if [ -z "${PATIENT_ID}" ]; then
    echo "ERROR: No patient at line ${LINE_NUM} in ${PATIENT_LIST}"
    exit 1
fi

echo "Patient ID: ${PATIENT_ID}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
  source activate "${CONDA_ENV_NAME}"
fi

# ========================================================================
# THREADING CONFIGURATION
# ========================================================================
# ANTs/ITK multi-threading: use all allocated CPUs for registration
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# Deterministic ANTs registration
export ANTS_RANDOM_SEED=42

echo "=========================================="
echo "ENVIRONMENT"
echo "=========================================="
echo "[python]    $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
echo "[threading] ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS}"
echo "[threading] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[threading] ANTS_RANDOM_SEED=${ANTS_RANDOM_SEED}"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "[warn] nvidia-smi not available"
echo ""

# ========================================================================
# PRE-FLIGHT CHECKS
# ========================================================================
echo "=========================================="
echo "PRE-FLIGHT CHECKS"
echo "=========================================="

cd "${REPO_SRC}"

# Check config file
if [ -f "${CONFIG_FILE}" ]; then
    echo "[OK]   Config: ${CONFIG_FILE}"
else
    echo "[FAIL] Config not found: ${CONFIG_FILE}"
    exit 1
fi

# Check patient data directory
PATIENT_DIR="${DATASET_ROOT}/${PATIENT_ID}"
if [ -d "${PATIENT_DIR}" ]; then
    NUM_STUDIES=$(find "${PATIENT_DIR}" -maxdepth 1 -mindepth 1 -type d | wc -l)
    echo "[OK]   Patient data: ${PATIENT_DIR} (${NUM_STUDIES} studies)"
else
    echo "[FAIL] Patient directory not found: ${PATIENT_DIR}"
    exit 1
fi

# Quick import check
python -c "
from mengrowth.preprocessing.src.config import load_preprocessing_pipeline_config
from mengrowth.preprocessing.src.preprocess import run_preprocessing
print('[OK]   MenGrowth imports')
"

# HD-BET weights check (no internet on compute nodes)
python -c "
from brainles_hd_bet.utils import get_params_fname
missing = [i for i in range(5) if not get_params_fname(i).exists()]
if missing:
    print(f'[FAIL] HD-BET weights missing for folds: {missing}')
    print('       Run on login node: python -c \"from brainles_hd_bet.utils import maybe_download_parameters; [maybe_download_parameters(i) for i in range(5)]\"')
    exit(1)
else:
    print('[OK]   HD-BET weights: all 5 folds present')
"

# CUDA check
python -c "
import torch
if torch.cuda.is_available():
    print(f'[OK]   CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)')
else:
    print('[warn] CUDA not available — skull stripping will be slow')
" 2>/dev/null || echo "[warn] PyTorch CUDA check failed"

echo ""

# ========================================================================
# RUN PREPROCESSING
# ========================================================================
echo "=========================================="
echo "PREPROCESSING: ${PATIENT_ID}"
echo "=========================================="
echo "Config:  ${CONFIG_FILE}"
echo "Patient: ${PATIENT_ID}"
echo ""

mengrowth-preprocess \
    --config "${CONFIG_FILE}" \
    --patient "${PATIENT_ID}" \
    --verbose

PREPROCESS_EXIT=$?

# ========================================================================
# POST-FLIGHT SUMMARY
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT SUMMARY: ${PATIENT_ID}"
echo "=========================================="

# Count output files
OUTPUT_ROOT=$(python -c "
from mengrowth.preprocessing.src.config import load_preprocessing_pipeline_config
cfg = load_preprocessing_pipeline_config('${CONFIG_FILE}')
print(cfg.output_root)
" 2>/dev/null || echo "")

if [ -n "${OUTPUT_ROOT}" ] && [ -d "${OUTPUT_ROOT}/${PATIENT_ID}" ]; then
    NIFTI_COUNT=$(find "${OUTPUT_ROOT}/${PATIENT_ID}" -name '*.nii.gz' | wc -l)
    echo "Output NIfTI files: ${NIFTI_COUNT}"
    find "${OUTPUT_ROOT}/${PATIENT_ID}" -name '*.nii.gz' | sort | sed 's/^/  /'
else
    echo "Output directory not found (may use pipeline mode with in-place writes)"
fi

# ========================================================================
# COMPLETION
# ========================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "PREPROCESSING COMPLETED: ${PATIENT_ID}"
echo "=========================================="
echo "Finished:   $(date)"
echo "Duration:   $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Exit code:  ${PREPROCESS_EXIT}"

if [ "$PREPROCESS_EXIT" -eq 0 ]; then
    echo "Patient ${PATIENT_ID} preprocessed successfully."
else
    echo "Patient ${PATIENT_ID} preprocessing FAILED with exit code ${PREPROCESS_EXIT}."
    exit "${PREPROCESS_EXIT}"
fi
