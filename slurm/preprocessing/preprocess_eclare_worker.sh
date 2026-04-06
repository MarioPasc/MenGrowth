#!/usr/bin/env bash
#SBATCH -J mgpp-eclare
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=preprocess_eclare_%A_%a.out
#SBATCH --error=preprocess_eclare_%A_%a.err

# =============================================================================
# MENGROWTH PREPROCESSING WORKER (ECLARE)
#
# Same as preprocess_worker.sh but with resources and env tuned for composite
# resampling (BSpline + ECLARE deep-learning super-resolution).
#
# Differences from preprocess_worker.sh:
#   - 4h walltime, 96G RAM (ECLARE spawns ~16 data-loader workers)
#   - Verifies ECLARE is importable from its conda env before starting
#   - Sets ECLARE_CONDA_ENV for subprocess calls
#
# Expected env vars (exported by preprocess_eclare.sh launcher):
#   REPO_SRC, CONFIG_FILE, DATASET_ROOT, CONDA_ENV_NAME, PATIENT_LIST
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "MenGrowth ECLARE preprocessing worker started at: $(date)"
echo "Hostname: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"

# ========================================================================
# RESOLVE PATIENT ID
# ========================================================================
if [ -z "${PATIENT_LIST:-}" ]; then
    echo "ERROR: PATIENT_LIST not set. Run via preprocess_eclare.sh launcher."
    exit 1
fi

if [ ! -f "${PATIENT_LIST}" ]; then
    echo "ERROR: Patient list not found: ${PATIENT_LIST}"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
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
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
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

# Config file
if [ -f "${CONFIG_FILE}" ]; then
    echo "[OK]   Config: ${CONFIG_FILE}"
else
    echo "[FAIL] Config not found: ${CONFIG_FILE}"
    exit 1
fi

# Patient data directory
PATIENT_DIR="${DATASET_ROOT}/${PATIENT_ID}"
if [ -d "${PATIENT_DIR}" ]; then
    NUM_STUDIES=$(find "${PATIENT_DIR}" -maxdepth 1 -mindepth 1 -type d | wc -l)
    echo "[OK]   Patient data: ${PATIENT_DIR} (${NUM_STUDIES} studies)"
else
    echo "[FAIL] Patient directory not found: ${PATIENT_DIR}"
    exit 1
fi

# MenGrowth import check
python -c "
from mengrowth.preprocessing.src.config import load_preprocessing_pipeline_config
from mengrowth.preprocessing.src.preprocess import run_preprocessing
print('[OK]   MenGrowth imports')
"

# HD-BET weights
python -c "
from brainles_hd_bet.utils import get_params_fname
missing = [i for i in range(5) if not get_params_fname(i).exists()]
if missing:
    print(f'[FAIL] HD-BET weights missing for folds: {missing}')
    exit(1)
else:
    print('[OK]   HD-BET weights: all 5 folds present')
"

# ECLARE availability (eclare is in the same env, installed with --no-deps)
python -c "
from eclare.model import ECLARE
print('[OK]   ECLARE model importable')
" || {
    echo "[FAIL] ECLARE not importable. Install: pip install --no-deps eclare"
    exit 1
}

# CUDA check
python -c "
import torch
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'[OK]   CUDA: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)')
else:
    print('[warn] CUDA not available — skull stripping and ECLARE will be slow')
" 2>/dev/null || echo "[warn] PyTorch CUDA check failed"

echo ""

# ========================================================================
# RUN PREPROCESSING
# ========================================================================
echo "=========================================="
echo "PREPROCESSING (ECLARE): ${PATIENT_ID}"
echo "=========================================="
echo "Config:  ${CONFIG_FILE}"
echo "Patient: ${PATIENT_ID}"
echo ""

set +e
mengrowth-preprocess \
    --config "${CONFIG_FILE}" \
    --patient "${PATIENT_ID}" \
    --verbose
PREPROCESS_EXIT=$?
set -e

# ========================================================================
# POST-FLIGHT SUMMARY
# ========================================================================
echo ""
echo "=========================================="
echo "OUTPUT SUMMARY: ${PATIENT_ID}"
echo "=========================================="

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

# Check audit report
ARTIFACTS_ROOT=$(python -c "
from mengrowth.preprocessing.src.config import load_preprocessing_pipeline_config
cfg = load_preprocessing_pipeline_config('${CONFIG_FILE}')
print(cfg.preprocessing_artifacts_path)
" 2>/dev/null || echo "")

AUDIT_REPORT="${ARTIFACTS_ROOT}/${PATIENT_ID}/mask_volume_audit.json"
if [ -f "${AUDIT_REPORT}" ]; then
    FLAGGED=$(python -c "import json; r=json.load(open('${AUDIT_REPORT}')); print(r.get('n_flagged', 0))")
    if [ "${FLAGGED}" -gt 0 ]; then
        echo ""
        echo "*** AUDIT: ${FLAGGED} study/modality pair(s) flagged for review ***"
        python -c "
import json
r = json.load(open('${AUDIT_REPORT}'))
for f in r['flagged_studies']:
    print(f'  {f[\"study_id\"]} {f[\"modality\"]}: deficit={f[\"deficit_pct\"]}%')
"
    else
        echo "Mask volume audit: no concerns"
    fi
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
