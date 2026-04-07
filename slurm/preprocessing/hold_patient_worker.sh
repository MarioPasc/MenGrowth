#!/usr/bin/env bash
#SBATCH -J mgpp-fix
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# HOLD PATIENT WORKER — single-patient preprocessing with ECLARE
#
# Expected env vars (set by launcher or sbatch --export):
#   PATIENT_ID   — e.g. MenGrowth-0049
#   CONFIG_FILE  — path to patient-specific YAML config
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "HOLD PATIENT FIX: ${PATIENT_ID}"
echo "=========================================="
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Config:  ${CONFIG_FILE}"
echo ""

# ── Conda setup ──
CONDA_ENV_NAME="mengrowth"
ECLARE_ENV="mengrowth-eclare"

module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] No conda module; assuming conda in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate "${CONDA_ENV_NAME}" 2>/dev/null || . activate "${CONDA_ENV_NAME}" 2>/dev/null || true
fi

# ── Threading ──
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export ANTS_RANDOM_SEED=42
export ECLARE_CONDA_ENV="${ECLARE_ENV}"

echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "[warn] no nvidia-smi"
echo ""

# ── Pre-flight ──
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

python -c "
from mengrowth.preprocessing.src.config import load_preprocessing_pipeline_config
from mengrowth.preprocessing.src.preprocess import run_preprocessing
print('[OK] MenGrowth imports')
"

# ── Run ──
echo "=========================================="
echo "PREPROCESSING: ${PATIENT_ID}"
echo "=========================================="

set +e
mengrowth-preprocess \
    --config "${CONFIG_FILE}" \
    --patient "${PATIENT_ID}" \
    --verbose
EXIT_CODE=$?
set -e

echo ""
echo "=========================================="
echo "COMPLETED: ${PATIENT_ID}"
echo "Exit code: ${EXIT_CODE}"
echo "Time: $(date)"
echo "=========================================="

exit ${EXIT_CODE}
