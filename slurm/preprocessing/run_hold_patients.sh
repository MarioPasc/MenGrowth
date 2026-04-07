#!/usr/bin/env bash
# =============================================================================
# Submit SLURM jobs for hold patients needing Picasso re-execution.
#
# Usage (from Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth
#   bash slurm/preprocessing/run_hold_patients.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER="${SCRIPT_DIR}/hold_patient_worker.sh"
LOG_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/mengrowth-dataset/logs/hold_patients"

mkdir -p "${LOG_DIR}"

# If arguments given, use them; otherwise run all hold patients
if [ $# -gt 0 ]; then
    PATIENTS=("$@")
else
    PATIENTS=(
        "MenGrowth-0003"
        "MenGrowth-0011"
        "MenGrowth-0025"
        "MenGrowth-0049"
    )
fi

echo "=========================================="
echo "SUBMITTING HOLD PATIENT JOBS"
echo "=========================================="
echo "Time: $(date)"
echo "Repo: ${REPO}"
echo ""

for PATIENT in "${PATIENTS[@]}"; do
    CONFIG="${REPO}/configs/local/patient_specific_files/picasso_${PATIENT}.yaml"

    if [ ! -f "${CONFIG}" ]; then
        echo "[SKIP] Config not found: ${CONFIG}"
        continue
    fi

    SHORT="${PATIENT##*-}"
    JOB_ID=$(sbatch \
        --job-name="mgpp-fix-${SHORT}" \
        --output="${LOG_DIR}/${PATIENT}_%j.out" \
        --error="${LOG_DIR}/${PATIENT}_%j.err" \
        --export="PATIENT_ID=${PATIENT},CONFIG_FILE=${CONFIG}" \
        --parsable \
        "${WORKER}")

    echo "[OK] ${PATIENT} -> Job ${JOB_ID}"
done

echo ""
echo "Monitor:"
echo "  squeue -u \$(whoami) | grep mgpp-fix"
echo "  tail -f ${LOG_DIR}/*.out"
