#!/bin/bash
# ------------------------------------------------------------------
# SLURM job - Phase 2 cascade evaluation (online + offline quality)
# ------------------------------------------------------------------
# Requires three running OpenAI-compatible endpoints published by
# slurm/server_role_phase2.sbatch into:
#   results/routing/endpoints/{teacher,student_mid,student_small}.url
#
# This job does NOT launch servers. It validates health checks and then
# runs:
#   1) routing.router with cascade_three_tier policy (online/service metrics)
#   2) routing.cascade_quality (offline quality metrics)
# ------------------------------------------------------------------
#SBATCH --job-name=eval-cascade-p2
#SBATCH --output=logs/eval-cascade-p2-%j.out
#SBATCH --error=logs/eval-cascade-p2-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
source env/setup_env.sh
mkdir -p logs results/routing results/quality

ENDPOINT_DIR="results/routing/endpoints"
RUNTIME_CFG="results/routing/routing_phase2_runtime-${SLURM_JOB_ID}.yaml"

echo "=========================================="
echo " Phase 2 - Cascade evaluation"
echo "=========================================="
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

echo "Waiting for endpoint publication files..."
for role in teacher student_mid student_small; do
  file="${ENDPOINT_DIR}/${role}.url"
  for attempt in $(seq 1 120); do
    if [[ -s "${file}" ]]; then
      break
    fi
    sleep 5
  done
  if [[ ! -s "${file}" ]]; then
    echo "ERROR: endpoint file not found or empty: ${file}"
    exit 1
  fi
done

TEACHER_URL=$(cat "${ENDPOINT_DIR}/teacher.url")
STUDENT_MID_URL=$(cat "${ENDPOINT_DIR}/student_mid.url")
STUDENT_SMALL_URL=$(cat "${ENDPOINT_DIR}/student_small.url")

for url in "${TEACHER_URL}" "${STUDENT_MID_URL}" "${STUDENT_SMALL_URL}"; do
  if [[ ! "${url}" =~ ^http://[^[:space:]]+$ ]]; then
    echo "ERROR: malformed endpoint URL '${url}'"
    exit 1
  fi
done

echo "Resolved endpoints:"
echo "  teacher      -> ${TEACHER_URL}"
echo "  student_mid  -> ${STUDENT_MID_URL}"
echo "  student_small-> ${STUDENT_SMALL_URL}"

# Health checks for all three endpoints
python -m serving.healthcheck --url "${TEACHER_URL}" --retries 120 --interval 5
python -m serving.healthcheck --url "${STUDENT_MID_URL}" --retries 120 --interval 5
python -m serving.healthcheck --url "${STUDENT_SMALL_URL}" --retries 120 --interval 5

python - << EOF
from utils.config_loader import load_yaml, save_yaml

cfg = load_yaml("configs/routing_phase2.yaml")
cfg["endpoints"]["teacher"]["base_url"] = "${TEACHER_URL}"
cfg["endpoints"]["student_mid"]["base_url"] = "${STUDENT_MID_URL}"
cfg["endpoints"]["student_small"]["base_url"] = "${STUDENT_SMALL_URL}"
cfg["endpoints"]["student"]["base_url"] = "${STUDENT_MID_URL}"
save_yaml(cfg, "${RUNTIME_CFG}")
print(f"Runtime routing config written to: ${RUNTIME_CFG}")
EOF

echo "[1/2] Online routing/service evaluation"
python -m routing.router --config "${RUNTIME_CFG}"

echo "[2/2] Offline quality evaluation with the same cascade policy"
python -m routing.cascade_quality \
  --eval-config configs/eval.yaml \
  --routing-config "${RUNTIME_CFG}" \
  --role cascade_phase2

echo "=========================================="
echo " Cascade phase-2 evaluation complete"
echo " Results:"
echo "   results/routing/cascade_three_tier-*"
echo "   results/quality/quality-cascade_phase2-*"
echo "=========================================="
