#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SLURM job — Phase B quality capture client
# Target cluster: MareNostrum 5   (BSC)
# ─────────────────────────────────────────────────────────────
# Required exported variables:
#   ROLE                   teacher|student_mid|student_q3b|student_small|student_tiny
# Optional:
#   PROJECT_DIR            repo root on shared FS (default: SLURM_SUBMIT_DIR)
#   SERVER_BASE_URL        full http://host:port (default: read from
#                          results/routing/endpoints/${ROLE}.url written by
#                          slurm/server_role_phase2.sbatch)
#   PHASE_B_CONFIG         path to YAML (default: configs/phase_b.yaml)
#
# Issues GSM8K + MATH-500 problems via Poisson arrivals against the role's
# vLLM endpoint, capturing logprobs and scoring against gold. Emits
# results/phase_b/captures/capture-<role>-<TS>/trace.jsonl in the
# ModelExecutionTrace schema. CPU-only — the GPU work runs on the role
# server. Submit with --nodelist=<server-node> when COLOC_CAPTURE=1.
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=phase-b-capture
#SBATCH --output=logs/phase-b-capture-%j.out
#SBATCH --error=logs/phase-b-capture-%j.err
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
  echo "  sbatch --export=ALL,ROLE=teacher,PROJECT_DIR=\$PWD slurm/phase_b_capture.sbatch"
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
PHASE_B_CONFIG="${PHASE_B_CONFIG:-configs/phase_b.yaml}"

cd "${PROJECT_DIR}"
source env/setup_env.sh

mkdir -p logs results/phase_b/captures results/routing/endpoints

# ── Resolve endpoint ───────────────────────────────────────
ENDPOINT_FILE="results/routing/endpoints/${ROLE}.url"
if [[ -z "${SERVER_BASE_URL:-}" ]]; then
  if [[ -f "${ENDPOINT_FILE}" ]]; then
    SERVER_BASE_URL="$(cat "${ENDPOINT_FILE}")"
  else
    echo "ERROR: SERVER_BASE_URL is empty and ${ENDPOINT_FILE} does not exist."
    echo "       Either set SERVER_BASE_URL or launch via slurm/launch_phase_b.sh"
    exit 1
  fi
fi

echo "=========================================="
echo " Phase B quality capture client"
echo "=========================================="
echo "ROLE:           ${ROLE}"
echo "Date:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID:-interactive}"
echo "Node:           $(hostname)"
echo "Server URL:     ${SERVER_BASE_URL}"
echo "Config:         ${PHASE_B_CONFIG}"
echo "Git commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================="

# ── Wait for server to be ready ────────────────────────────
python -m serving.healthcheck --url "${SERVER_BASE_URL}" --retries 60 --interval 5

# ── Run the quality capture ────────────────────────────────
python -m bench.run_quality_capture \
  --config "${PHASE_B_CONFIG}" \
  --role "${ROLE}" \
  --base-url "${SERVER_BASE_URL}"

echo "=== PHASE B QUALITY CAPTURE DONE — role=${ROLE} ==="
