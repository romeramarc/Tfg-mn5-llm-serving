#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Phase A load capture client
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
# Required exported variables:
#   ROLE                   teacher|student_mid|student_small (server role)
# Optional:
#   PROJECT_DIR            repo root on shared FS (default: $PWD when sbatch'ed)
#   SERVER_BASE_URL        full http://host:port (default: read from
#                          results/routing/endpoints/${ROLE}.url written by
#                          slurm/server_role_phase2.sbatch)
#   PHASE_A_CONFIG         path to YAML (default: configs/phase_a.yaml)
#   BENCHMARK_LABEL        label written into the trace (default: phase_a_workload)
#
# This job is intentionally CPU-only — the GPU work runs inside the role
# server job, which this client targets via HTTP. Submit with
# --nodelist=<server-node> (the launcher does this automatically) so that
# in-host networking minimises noise on TTFT and queue measurements.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=phase-a-capture
#SBATCH --output=logs/phase-a-capture-%j.out
#SBATCH --error=logs/phase-a-capture-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail

ROLE="${ROLE:-}"
if [[ -z "${ROLE}" ]]; then
  echo "ERROR: ROLE is required (teacher|student_mid|student_q3b|student_small|student_small_base|student_tiny)."
  echo "Submit example:"
  echo "  sbatch --nodelist=\$SERVER_NODE --export=ALL,ROLE=teacher,PROJECT_DIR=\$PWD slurm/phase_a_capture.sbatch"
  exit 1
fi

case "${ROLE}" in
  teacher|student_mid|student_q3b|student_small|student_small_base|student_tiny) ;;
  *)
    echo "ERROR: invalid ROLE='${ROLE}'."
    exit 1
    ;;
esac

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PHASE_A_CONFIG="${PHASE_A_CONFIG:-configs/phase_a.yaml}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-phase_a_workload}"

cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs results/phase_a/captures results/routing/endpoints

# ── Resolve endpoint ───────────────────────────────────────
ENDPOINT_FILE="results/routing/endpoints/${ROLE}.url"
if [[ -z "${SERVER_BASE_URL:-}" ]]; then
  if [[ -f "${ENDPOINT_FILE}" ]]; then
    SERVER_BASE_URL="$(cat "${ENDPOINT_FILE}")"
  else
    echo "ERROR: SERVER_BASE_URL is empty and ${ENDPOINT_FILE} does not exist."
    echo "       Either set SERVER_BASE_URL or launch via slurm/launch_phase_a.sh"
    exit 1
  fi
fi

echo "=========================================="
echo " Phase A load capture client"
echo "=========================================="
echo "ROLE:           ${ROLE}"
echo "Date:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID:-interactive}"
echo "Node:           $(hostname)"
echo "Server URL:     ${SERVER_BASE_URL}"
echo "Config:         ${PHASE_A_CONFIG}"
echo "Benchmark tag:  ${BENCHMARK_LABEL}"
echo "Git commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

# ── Wait for server to be ready ────────────────────────────
python -m serving.healthcheck --url "${SERVER_BASE_URL}" --retries 60 --interval 5

# ── Run the capture ────────────────────────────────────────
python -m bench.run_load_capture \
  --config "${PHASE_A_CONFIG}" \
  --role "${ROLE}" \
  --base-url "${SERVER_BASE_URL}" \
  --benchmark-label "${BENCHMARK_LABEL}"

echo "=== PHASE A CAPTURE DONE — role=${ROLE} ==="
