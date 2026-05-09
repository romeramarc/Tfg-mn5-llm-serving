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
#SBATCH --time=24:00:00

set -euo pipefail

ROLE="${ROLE:-}"
if [[ -z "${ROLE}" ]]; then
  echo "ERROR: ROLE is required (teacher|student_mid|student_q3b|student_small|student_tiny)."
  echo "Example: sbatch --export=ALL,ROLE=teacher slurm/server_role_phase2.sbatch"
  exit 1
fi

case "${ROLE}" in
  teacher|student_mid|student_q3b|student_small|student_tiny) ;;
  *)
    echo "ERROR: invalid ROLE='${ROLE}'."
    exit 1
    ;;
esac

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs results/routing/endpoints results/phase_a/gpu_samples_by_role

# Pick a free port inside a role-specific range so jobs can safely share nodes.
case "${ROLE}" in
  teacher)        PORT_BASE=8000 ;;
  student_mid)    PORT_BASE=8100 ;;
  student_q3b)    PORT_BASE=8150 ;;
  student_small)  PORT_BASE=8200 ;;
  student_tiny)   PORT_BASE=8250 ;;
esac
PORT_SPAN=100
PORT_END=$((PORT_BASE + PORT_SPAN - 1))

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  PORT_START=$((PORT_BASE + (SLURM_JOB_ID % PORT_SPAN)))
else
  PORT_START="${PORT_BASE}"
fi

pick_free_port() {
  local start_port="$1"
  local end_port="$2"
  local port

  for port in $(seq "${start_port}" "${end_port}"); do
    if python - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
    then
      echo "${port}"
      return 0
    fi
  done

  return 1
}

if ! SERVER_PORT="$(pick_free_port "${PORT_START}" "${PORT_END}")"; then
  if (( PORT_START > PORT_BASE )); then
    if ! SERVER_PORT="$(pick_free_port "${PORT_BASE}" "$((PORT_START - 1))")"; then
      echo "ERROR: could not find a free TCP port for ROLE=${ROLE} in range ${PORT_BASE}-${PORT_END}."
      exit 1
    fi
  else
    echo "ERROR: could not find a free TCP port for ROLE=${ROLE} in range ${PORT_BASE}-${PORT_END}."
    exit 1
  fi
fi

RUNTIME_SERVING_CFG="results/routing/serving-${ROLE}-${SLURM_JOB_ID}.yaml"

HOSTNAME_FQDN=$(hostname -f 2>/dev/null || hostname)
BASE_URL="http://${HOSTNAME_FQDN}:${SERVER_PORT}"
ENDPOINT_FILE="results/routing/endpoints/${ROLE}.url"

echo "=========================================="
echo " Phase 2 role server"
echo "=========================================="
echo "ROLE:         ${ROLE}"
echo "Date:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Node:         ${HOSTNAME_FQDN}"
echo "Port:         ${SERVER_PORT}"
echo "Endpoint:     ${BASE_URL}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

# Publish endpoint URL for the evaluator job.
TMP_ENDPOINT_FILE="${ENDPOINT_FILE}.tmp.${SLURM_JOB_ID}"
printf '%s\n' "${BASE_URL}" > "${TMP_ENDPOINT_FILE}"
mv "${TMP_ENDPOINT_FILE}" "${ENDPOINT_FILE}"

python - << EOF
from utils.config_loader import load_yaml, save_yaml

cfg = load_yaml("configs/serving.yaml")
cfg.setdefault("server", {})["port"] = int("${SERVER_PORT}")
save_yaml(cfg, "${RUNTIME_SERVING_CFG}")
print(f"Runtime serving config written to: ${RUNTIME_SERVING_CFG}")
EOF

# ── GPU sampler sidecar (runs ON THE SERVER NODE, sees the right GPU) ──
# Captures will read from this path instead of running their own sampler,
# so GPU telemetry is always for the GPU that is actually serving the model.
GPU_SAMPLES_DIR="results/phase_a/gpu_samples_by_role"
GPU_SAMPLES_PATH="${GPU_SAMPLES_DIR}/${ROLE}-${SLURM_JOB_ID}.jsonl"
GPU_SAMPLES_PUBLISH="results/routing/endpoints/${ROLE}.gpu"

# Publish gpu samples path BEFORE starting vLLM so captures can pick it up.
TMP_GPU_PUBLISH="${GPU_SAMPLES_PUBLISH}.tmp.${SLURM_JOB_ID}"
printf '%s\n' "${PROJECT_DIR}/${GPU_SAMPLES_PATH}" > "${TMP_GPU_PUBLISH}"
mv "${TMP_GPU_PUBLISH}" "${GPU_SAMPLES_PUBLISH}"

# Start the sampler in background, kill on exit.
python -m bench.gpu_sampler --output "${GPU_SAMPLES_PATH}" --interval 0.2 \
  > "logs/gpu-sampler-${ROLE}-${SLURM_JOB_ID}.out" \
  2> "logs/gpu-sampler-${ROLE}-${SLURM_JOB_ID}.err" &
GPU_SAMPLER_PID=$!
echo "GPU sampler started: pid=${GPU_SAMPLER_PID}, output=${GPU_SAMPLES_PATH}"

cleanup() {
  if [[ -n "${GPU_SAMPLER_PID:-}" ]] && kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${GPU_SAMPLER_PID}" || true
    sleep 1
    kill -KILL "${GPU_SAMPLER_PID}" 2>/dev/null || true
  fi
  rm -f "${GPU_SAMPLES_PUBLISH}" "${ENDPOINT_FILE}"
}
trap cleanup EXIT

echo "Launching role server process..."
python -m serving.start_server \
  --config "${RUNTIME_SERVING_CFG}" \
  --models configs/models.yaml \
  --role "${ROLE}"
