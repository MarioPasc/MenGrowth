#!/usr/bin/env bash
#SBATCH -J mg_synthseg
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

# =============================================================================
# MenGrowth SynthSeg — SLURM WORKER (array task)
#
# Processes all studies of one patient via `mengrowth-synthseg run-patient`.
#
# Array task index is mapped to a patient ID by reading line
# ($SLURM_ARRAY_TASK_ID + 1) of $PATIENT_LIST.
#
# Expected env vars (exported by synthseg_launcher.sh):
#   CONFIG_FILE       Path to synthseg YAML
#   PATIENT_LIST      Path to patient manifest (one ID per line)
#   CONDA_ENV_NAME    Conda env hosting the mengrowth CLI
#
# The worker first activates the mengrowth env for the CLI, then derives the
# SynthSeg conda env path and Python interpreter from the YAML. The runner
# itself resolves `python_bin` from (a) config, (b) {conda_env_path}/bin/python,
# (c) sys.executable — so we pass the SynthSeg env via config, not a switch.
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)
echo "================================================="
echo "MENGROWTH SYNTHSEG — WORKER"
echo "================================================="
echo "Time:     $(date)"
echo "Hostname: $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-local}"
echo "Array:    ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"
echo ""

# -----------------------------------------------------------------------------
# ENV SETUP
# -----------------------------------------------------------------------------
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
echo "[python]  $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
echo "[mengrowth-synthseg] $(which mengrowth-synthseg || echo 'NOT FOUND')"

# GPU info (SynthSeg uses TF/Keras; GPU is picked up via CUDA env)
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
echo ""

# -----------------------------------------------------------------------------
# RESOLVE PATIENT ID FROM ARRAY INDEX
# -----------------------------------------------------------------------------
if [ -z "${PATIENT_LIST:-}" ] || [ ! -f "${PATIENT_LIST}" ]; then
    echo "ERROR: PATIENT_LIST not set or missing: ${PATIENT_LIST:-<unset>}"
    exit 1
fi

TASK_IDX="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NO=$((TASK_IDX + 1))
PATIENT_ID=$(sed -n "${LINE_NO}p" "${PATIENT_LIST}" | tr -d '[:space:]')

if [ -z "${PATIENT_ID}" ]; then
    echo "ERROR: No patient at line ${LINE_NO} of ${PATIENT_LIST}"
    exit 1
fi

echo "=========================================="
echo "PATIENT: ${PATIENT_ID}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Extract SynthSeg env path and expose its python to the runner.
# The runner's _resolve_python_bin() picks {conda_env_path}/bin/python if
# python_bin is unset in YAML. We additionally PATH-prepend it so SynthSeg's
# own auxiliary scripts (e.g. nibabel, tensorflow) resolve correctly.
# -----------------------------------------------------------------------------
SYNTHSEG_ENV=$(grep '^\s*conda_env_path:' "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
if [ -n "${SYNTHSEG_ENV}" ] && [ -d "${SYNTHSEG_ENV}" ]; then
    export PATH="${SYNTHSEG_ENV}/bin:${PATH}"
    echo "[env] Prepended SynthSeg env to PATH: ${SYNTHSEG_ENV}/bin"
else
    echo "[env] WARNING: SynthSeg env not found at '${SYNTHSEG_ENV}'; runner will fall back to sys.executable"
fi
echo ""

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
set +e
mengrowth-synthseg run-patient \
    --config "${CONFIG_FILE}" \
    --patient "${PATIENT_ID}" \
    --verbose
RUN_EXIT=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "WORKER COMPLETED"
echo "=========================================="
echo "Patient:    ${PATIENT_ID}"
echo "Exit code:  ${RUN_EXIT}"
echo "Duration:   $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Finished:   $(date)"

# Exit code semantics (mirror PatientResult.exit_code):
#   0 = success (or all already done)
#   2 = one or more study failures
#   3 = no studies with the required input modality
exit "${RUN_EXIT}"
