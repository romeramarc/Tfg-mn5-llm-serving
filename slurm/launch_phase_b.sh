#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Launch script — Phase B on BSC (servers + quality capture + train)
# ─────────────────────────────────────────────────────────────
# What this script does:
#
#   1. Cleans stale endpoint files left by previous runs.
#   2. Submits a vLLM server job per role declared in
#      configs/phase_b.yaml → capture.roles using
#      slurm/server_role_phase2.sbatch (same as Phase A — generic role
#      server with a server-side gpu_sampler sidecar).
#   3. WAITS until each server allocation is RUNNING and extracts the
#      compute node it landed on.
#   4. Submits one quality-capture client per role with
#      slurm/phase_b_capture.sbatch. Co-location with the server is
#      opt-in via COLOC_CAPTURE=1 because the GPU sampler runs on the
#      server side and publishes its samples path.
#   5. Submits the predictor training job (phase_b_train.sbatch) with
#      --dependency=afterok on every capture, so it auto-runs once
#      captures have finished successfully.
#   6. Submits a small auto-cleanup job that scancels the long-running
#      server jobs once the captures end (whether OK or failed).
#
# Usage
# -----
#   bash slurm/launch_phase_b.sh
#
# Env overrides
# -------------
#   PHASE_B_CONFIG     path to YAML (default: configs/phase_b.yaml)
#   ROLES              space-separated role list (default: read from YAML)
#   SERVER_WAIT_S      max seconds to wait for each server to RUN (default: 1800)
#   COLOC_CAPTURE      "1" → submit captures with --nodelist=<server-node>
# ─────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd -P)"

PHASE_B_CONFIG="${PHASE_B_CONFIG:-configs/phase_b.yaml}"
SERVER_WAIT_S="${SERVER_WAIT_S:-1800}"

if [[ ! -f "${PHASE_B_CONFIG}" ]]; then
  echo "ERROR: ${PHASE_B_CONFIG} not found."
  exit 1
fi

# Resolve role list either from $ROLES or by reading the YAML.
if [[ -z "${ROLES:-}" ]]; then
  ROLES=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${PHASE_B_CONFIG}")
roles = [r["name"] for r in (cfg.get("capture", {}).get("roles") or [])]
print(" ".join(roles))
EOF
  )
fi

if [[ -z "${ROLES// /}" ]]; then
  echo "ERROR: no roles resolved (configs/phase_b.yaml → capture.roles)."
  exit 1
fi

mkdir -p logs results/routing/endpoints results/phase_b/captures

echo "=========================================="
echo " Launching Phase B (BSC) — quality + routing"
echo "=========================================="
echo "Project dir:       ${PROJECT_DIR}"
echo "Config:            ${PHASE_B_CONFIG}"
echo "Roles:             ${ROLES}"
echo "Server wait (s):   ${SERVER_WAIT_S}"
echo "Co-locate capture: ${COLOC_CAPTURE:-0}"
echo "=========================================="

# ── 1) Clean stale endpoint files ─────────────────────────
for role in ${ROLES}; do
  rm -f "results/routing/endpoints/${role}.url"
  rm -f "results/routing/endpoints/${role}.gpu"
done

# ── 2) Submit one server per role ─────────────────────────
declare -A SERVER_JOB
declare -A SERVER_NODE
SERVER_JOBS_LIST=()
for role in ${ROLES}; do
  jid=$(sbatch --parsable --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=${role}" \
              slurm/server_role_phase2.sbatch)
  SERVER_JOB[${role}]="${jid}"
  SERVER_JOBS_LIST+=("${jid}")
  echo "server ${role}: ${jid} (queued)"
done

# ── 3) Wait for each server to be RUNNING and extract its node ──
wait_for_running() {
  local jid="$1"
  local timeout="$2"
  local elapsed=0
  while (( elapsed < timeout )); do
    local state node
    local show_o
    show_o=$(scontrol show job "${jid}" -o 2>/dev/null || echo "")
    # Portable parsing (do not use grep -P — absent or broken on some login nodes).
    state=$(echo "${show_o}" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')
    case "${state}" in
      RUNNING)
        node=$(echo "${show_o}" | sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' | cut -d'+' -f1 | tr -d '()')
        if [[ -n "${node}" && "${node}" != "null" ]]; then
          echo "${node}"
          return 0
        fi
        ;;
      FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY)
        echo "ERROR: server job ${jid} ended in state '${state}' before allocation." >&2
        return 1
        ;;
    esac
    sleep 10
    (( elapsed += 10 ))
  done
  echo "ERROR: server job ${jid} did not reach RUNNING within ${timeout}s." >&2
  return 1
}

echo ""
echo "Waiting for servers to start (max ${SERVER_WAIT_S}s each)..."
for role in ${ROLES}; do
  jid="${SERVER_JOB[${role}]}"
  if ! node=$(wait_for_running "${jid}" "${SERVER_WAIT_S}"); then
    echo "FATAL: cancelling all submitted servers because ${role} (${jid}) did not start." >&2
    scancel "${SERVER_JOBS_LIST[@]}" || true
    exit 1
  fi
  SERVER_NODE[${role}]="${node}"
  echo "  ${role}: server ${jid} RUNNING on ${node}"
done

# ── 4) Submit one capture per role ────────────────────────
declare -a CAPTURE_JOBS_LIST
for role in ${ROLES}; do
  srv_jid="${SERVER_JOB[${role}]}"
  srv_node="${SERVER_NODE[${role}]}"
  if [[ "${COLOC_CAPTURE:-0}" == "1" ]]; then
    cap_jid=$(sbatch --parsable \
                     --dependency=after:${srv_jid} \
                     --nodelist="${srv_node}" \
                     --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=${role},PHASE_B_CONFIG=${PHASE_B_CONFIG}" \
                     slurm/phase_b_capture.sbatch)
    echo "capture ${role}: ${cap_jid} on ${srv_node} (co-located, depends on server ${srv_jid})"
  else
    cap_jid=$(sbatch --parsable \
                     --dependency=after:${srv_jid} \
                     --export="ALL,PROJECT_DIR=${PROJECT_DIR},ROLE=${role},PHASE_B_CONFIG=${PHASE_B_CONFIG}" \
                     slurm/phase_b_capture.sbatch)
    echo "capture ${role}: ${cap_jid} (depends on server ${srv_jid}; GPU samples from ${srv_node})"
  fi
  CAPTURE_JOBS_LIST+=("${cap_jid}")
done

CAPTURE_JOBS_DEP=$(IFS=:; echo "${CAPTURE_JOBS_LIST[*]}")

# ── 5) Train predictors after every capture has finished OK ─
JOB_TRAIN=$(sbatch --parsable \
                   --dependency=afterok:${CAPTURE_JOBS_DEP} \
                   --export="ALL,PROJECT_DIR=${PROJECT_DIR},PHASE_B_CONFIG=${PHASE_B_CONFIG}" \
                   slurm/phase_b_train.sbatch)
echo "train predictors: ${JOB_TRAIN} (afterok of captures)"

# ── 6) Auto-cleanup of long-running servers when captures end ─
SERVER_JOBS_SCANCEL_ARGS="${SERVER_JOBS_LIST[*]}"
JOB_CLEANUP=$(sbatch --parsable \
                     --dependency=afterany:${CAPTURE_JOBS_DEP} \
                     --partition=gpp --account=bsc98 --qos=gp_bsccs \
                     --time=00:05:00 --ntasks=1 --cpus-per-task=1 \
                     --output=logs/phase-b-cleanup-%j.out \
                     --error=logs/phase-b-cleanup-%j.err \
                     --wrap="scancel ${SERVER_JOBS_SCANCEL_ARGS} || true")
echo "auto-cleanup:     ${JOB_CLEANUP} (scancels servers when captures end)"

echo ""
echo "Submitted jobs:"
for role in ${ROLES}; do
  echo "  server  ${role}:        ${SERVER_JOB[${role}]} (node=${SERVER_NODE[${role}]})"
done
for jid in "${CAPTURE_JOBS_LIST[@]}"; do
  echo "  capture:                 ${jid}"
done
echo "  train predictors:        ${JOB_TRAIN}"
echo "  auto-cleanup (scancel):  ${JOB_CLEANUP}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -o '%.18i %.20j %.2t %.10M %.20R'"
echo "  tail -f logs/phase-b-capture-*.out"
echo "  tail -f logs/phase-b-train-${JOB_TRAIN}.out"
echo ""
echo "Manual stop (if you must):"
echo "  scancel ${SERVER_JOBS_SCANCEL_ARGS}"
