#!/bin/bash
# ------------------------------------------------------------------
# SLURM job - Phase 2 role-based server (teacher | student_mid | student_small)
# ------------------------------------------------------------------
# Required export variables:
#   ROLE=teacher|student_mid|student_small
# Optional:
#   PROJECT_DIR=/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
#
# Publishes endpoint URL to:
#   results/routing/endpoints/${ROLE}.url
# ------------------------------------------------------------------
#SBATCH --job-name=vllm-role-p2
#SBATCH --output=logs/vllm-role-p2-%j.out
#SBATCH --error=logs/vllm-role-p2-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=15:00:00
#SBATCH --exclusive

set -euo pipefail

ROLE="${ROLE:-}"
if [[ -z "${ROLE}" ]]; then
  echo "ERROR: ROLE is required (teacher|student_mid|student_small)."
  echo "Example: sbatch --export=ALL,ROLE=teacher slurm/server_role_phase2.sbatch"
  exit 1
fi

case "${ROLE}" in
  teacher|student_mid|student_small) ;;
  *)
    echo "ERROR: invalid ROLE='${ROLE}'."
    exit 1
    ;;
esac

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs results/routing/endpoints

HOSTNAME_FQDN=$(hostname -f 2>/dev/null || hostname)
BASE_URL="http://${HOSTNAME_FQDN}:8000"
ENDPOINT_FILE="results/routing/endpoints/${ROLE}.url"

echo "=========================================="
echo " Phase 2 role server"
echo "=========================================="
echo "ROLE:         ${ROLE}"
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Node:         ${HOSTNAME_FQDN}"
echo "Endpoint:     ${BASE_URL}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

# Publish endpoint URL for the evaluator job.
TMP_ENDPOINT_FILE="${ENDPOINT_FILE}.tmp.${SLURM_JOB_ID}"
printf '%s\n' "${BASE_URL}" > "${TMP_ENDPOINT_FILE}"
mv "${TMP_ENDPOINT_FILE}" "${ENDPOINT_FILE}"

echo "Launching role server process..."
python -m serving.start_server \
  --config configs/serving.yaml \
  --models configs/models.yaml \
  --role "${ROLE}"
