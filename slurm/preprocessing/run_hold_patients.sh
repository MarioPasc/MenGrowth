#!/usr/bin/env bash
# =============================================================================
# Submit SLURM jobs for the 4 hold patients that need Picasso re-execution.
#
# Each patient runs as an independent SLURM job using the ECLARE worker
# with patient-specific configs (composite resampling, single-patient mode).
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth
#   bash slurm/preprocessing/run_hold_patients.sh
#
# All jobs use:
#   - 4h walltime, 96G RAM, 1 GPU (DGX), 8 CPUs
#   - Composite resampling (BSpline + ECLARE)
#   - overwrite=true (replaces previous v4_eclare results)
#   - QC metrics disabled (speed)
# =============================================================================

set -euo pipefail

REPO="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth"
LOG_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/mengrowth-dataset/logs/hold_patients"
CONDA_ENV="mengrowth"
ECLARE_ENV="mengrowth-eclare"

mkdir -p "${LOG_DIR}"

PATIENTS=(
    "MenGrowth-0003"
    "MenGrowth-0011"
    "MenGrowth-0025"
    "MenGrowth-0049"
)

echo "=========================================="
echo "SUBMITTING HOLD PATIENT JOBS"
echo "=========================================="
echo "Time: $(date)"
echo ""

for PATIENT in "${PATIENTS[@]}"; do
    CONFIG="${REPO}/configs/local/patient_specific_files/picasso_${PATIENT}.yaml"

    if [ ! -f "${CONFIG}" ]; then
        echo "[SKIP] Config not found: ${CONFIG}"
        continue
    fi

    JOB_NAME="mgpp-fix-${PATIENT##*-}"

    # Submit via sbatch with inline script
    JOB_ID=$(sbatch \
        --job-name="${JOB_NAME}" \
        --time=0-04:00:00 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=96G \
        --constraint=dgx \
        --gres=gpu:1 \
        --output="${LOG_DIR}/${PATIENT}_%j.out" \
        --error="${LOG_DIR}/${PATIENT}_%j.err" \
        --parsable \
        --wrap="
            # Conda setup
            module_loaded=0
            for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
                if module avail 2>/dev/null | grep -qi \"^\${m}[[:space:]]\"; then
                    module load \"\$m\" && module_loaded=1 && break
                fi
            done
            if [ \"\$module_loaded\" -eq 0 ]; then
                echo '[env] No conda module; assuming conda in PATH'
            fi
            if command -v conda >/dev/null 2>&1; then
                source \"\$(conda info --base)/etc/profile.d/conda.sh\" || true
                conda activate ${CONDA_ENV} 2>/dev/null || source activate ${CONDA_ENV}
            else
                source activate ${CONDA_ENV}
            fi

            # Threading
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=\${SLURM_CPUS_PER_TASK:-8}
            export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
            export ANTS_RANDOM_SEED=42
            export ECLARE_CONDA_ENV=${ECLARE_ENV}

            echo '=========================================='
            echo 'PATIENT: ${PATIENT}'
            echo 'CONFIG:  ${CONFIG}'
            echo '=========================================='
            python -c \"import sys; print('Python', sys.version.split()[0])\"
            nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

            cd ${REPO}
            mengrowth-preprocess --config ${CONFIG} --patient ${PATIENT} --verbose
        ")

    echo "[OK] ${PATIENT} -> Job ${JOB_ID} (${JOB_NAME})"
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$(whoami) -n mgpp-fix"
echo "  tail -f ${LOG_DIR}/*.out"
