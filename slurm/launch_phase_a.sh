#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Launch script — Phase A on BSC (servers + capture + predictor)
# ─────────────────────────────────────────────────────────────
# What this script does:
#
#   1. Cleans stale endpoint files left by previous runs.
#   2. Submits a vLLM server job per role declared in
#      configs/phase_a.yaml → capture.roles using
#      slurm/server_role_phase2.sbatch.
#   3. Submits one capture client per role with
#      slurm/phase_a_capture.sbatch (waits for the server endpoint).
#   4. Submits the predictor training job (phase_a_train.sbatch)
#      with --dependency=afterok on every capture job, so it runs
#      automatically once captures have finished.
#
# Usage
# -----
#   bash slurm/launch_phase_a.sh
#
# Env overrides
# -------------
#   PHASE_A_CONFIG     path to YAML (default: configs/phase_a.yaml)
#   ROLES              space-separated role list (default: read from YAML)
#   BENCHMARK_LABEL    label written into the trace (default: phase_a_workload)
#
# After captures complete, the launcher prints the scancel command for
# the long-running server jobs.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd -P)"

PHASE_A_CONFIG="${PHASE_A_CONFIG:-configs/phase_a.yaml}"
BENCHMARK_LABEL="${BENCHMARK_LABEL:-phase_a_workload}"

if [[ ! -f "${PHASE_A_CONFIG}" ]]; then
  echo "ERROR: ${PHASE_A_CONFIG} not found."
  exit 1
fi

# Resolve role list either from $ROLES or by reading the YAML.
if [[ -z "${ROLES:-}" ]]; then
  ROLES=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${PHASE_A_CONFIG}")
roles = [r["name"] for r in (cfg.get("capture", {}).get("roles") or [])]
print(" ".join(roles))
EOF
  )
fi

if [[ -z "${ROLES// /}" ]]; then
  echo "ERROR: no roles resolved (configs/phase_a.yaml → capture.roles)."
  exit 1
fi

mkdir -p logs results/routing/endpoints results/phase_a/captures

echo "=========================================="
echo " Launching Phase A (BSC)"
echo "=========================================="
echo "Project dir:       ${PROJECT_DIR}"
echo "Config:            ${PHASE_A_CONFIG}"
echo "Roles:             ${ROLES}"
echo "Benchmark label:   ${BENCHMARK_LABEL}"
echo "=========================================="

# ── 1) Clean stale endpoint files ─────────────────────────
for role in ${ROLES}; do
  rm -f "results/routing/endpoints/${role}.url"
done

# ── 2) Submit one server per role ─────────────────────────
declare -A SERVER_JOB
SERVER_JOBS_LIST=()
for role in ${ROLES}; do
  jid=$(sbatch --parsable --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=${role}" \
              slurm/server_role_phase2.sbatch)
  SERVER_JOB[${role}]="${jid}"
  SERVER_JOBS_LIST+=("${jid}")
  echo "server ${role}: ${jid}"
done

# ── 3) Submit one capture per role, co-located on its server node ──
# We start the capture job with --dependency=after:<server_jobid> so
# the scheduler only places it once the server allocation is RUNNING.
declare -a CAPTURE_JOBS_LIST
for role in ${ROLES}; do
  srv_jid="${SERVER_JOB[${role}]}"
  cap_jid=$(sbatch --parsable \
                   --dependency=after:${srv_jid} \
                   --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=${role},PHASE_A_CONFIG=${PHASE_A_CONFIG},BENCHMARK_LABEL=${BENCHMARK_LABEL}" \
                   slurm/phase_a_capture.sbatch)
  CAPTURE_JOBS_LIST+=("${cap_jid}")
  echo "capture ${role}: ${cap_jid} (depends on server ${srv_jid})"
done

CAPTURE_JOBS_DEP=$(IFS=:; echo "${CAPTURE_JOBS_LIST[*]}")

# ── 4) Train predictor after every capture has finished OK ─
JOB_TRAIN=$(sbatch --parsable \
                   --dependency=afterok:${CAPTURE_JOBS_DEP} \
                   --export="ALL,PROJECT_DIR=${PROJECT_DIR},PHASE_A_CONFIG=${PHASE_A_CONFIG}" \
                   slurm/phase_a_train.sbatch)
echo "train predictor: ${JOB_TRAIN} (afterok of captures)"

# ── 5) Auto-cleanup of long-running servers when captures end ─
SERVER_JOBS_SCANCEL_ARGS="${SERVER_JOBS_LIST[*]}"
JOB_CLEANUP=$(sbatch --parsable \
                     --dependency=afterany:${CAPTURE_JOBS_DEP} \
                     --partition=gpp --account=bsc98 --qos=gp_bsccs \
                     --time=00:05:00 --ntasks=1 --cpus-per-task=1 \
                     --output=logs/phase-a-cleanup-%j.out \
                     --error=logs/phase-a-cleanup-%j.err \
                     --wrap="scancel ${SERVER_JOBS_SCANCEL_ARGS} || true")
echo "auto-cleanup:     ${JOB_CLEANUP} (scancels servers when captures end)"

echo ""
echo "Submitted jobs:"
for role in ${ROLES}; do
  echo "  server  ${role}:        ${SERVER_JOB[${role}]}"
done
for jid in "${CAPTURE_JOBS_LIST[@]}"; do
  echo "  capture:                 ${jid}"
done
echo "  train predictor:         ${JOB_TRAIN}"
echo "  auto-cleanup (scancel):  ${JOB_CLEANUP}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/phase-a-capture-*.out"
echo "  tail -f logs/phase-a-train-${JOB_TRAIN}.out"
echo ""
echo "Manual stop (if you must):"
echo "  scancel ${SERVER_JOBS_SCANCEL_ARGS}"
