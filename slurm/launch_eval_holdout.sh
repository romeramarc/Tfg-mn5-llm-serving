#!/usr/bin/env bash
# ------------------------------------------------------------------
# Queue holdout evaluation on MN5.
#
# Usage (from login node, inside tmux/screen):
#   bash slurm/launch_eval_holdout.sh servers     # vLLM servers for configured roles
#   bash slurm/launch_eval_holdout.sh clients     # eval jobs for configured systems
#   bash slurm/launch_eval_holdout.sh all          # servers + evals (no sbatch deps)
#
# Eval jobs do NOT use sbatch --dependency on BSC (afterbegin on server job IDs
# is rejected: "Job dependency problem"). Each eval job polls endpoint .url files
# inside slurm/eval_holdout.sbatch (ENDPOINT_WAIT_S, default 7200s).
#
# Environment:
#   PROJECT_DIR, EVAL_CONFIG, PROMPT_POOL, SERVER_WAIT_S (default 7200)
# ------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
mkdir -p logs results/routing/endpoints results/routing_eval_holdout

# Ensure repo root is importable for the inline Python heredocs below (read_config_*).
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${1:-all}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/routing_eval_holdout.yaml}"
PROMPT_POOL="${PROMPT_POOL:-results/routing_eval_holdout/prompt_pool.jsonl}"
SERVER_WAIT_S="${SERVER_WAIT_S:-7200}"

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: EVAL_CONFIG not found: ${EVAL_CONFIG}" >&2
  exit 1
fi
if [[ ! -s "${PROMPT_POOL}" ]]; then
  echo "[INFO] Prompt pool missing or empty; will be built from ${EVAL_CONFIG}" >&2
fi

read_config_roles_and_systems() {
  python - <<'PY'
from utils.config_loader import load_yaml
import json
import os

cfg = load_yaml(os.environ.get("EVAL_CONFIG", "configs/routing_eval_holdout.yaml"))
systems = cfg.get("systems", []) or []

all_roles = []
system_ids = []
for s in systems:
    sid = s.get("id")
    roles = list(s.get("roles") or [])
    if sid:
        system_ids.append(str(sid))
    for r in roles:
        if r not in all_roles:
            all_roles.append(r)

print(json.dumps({"roles": all_roles, "systems": system_ids}))
PY
}

CFG_JSON="$(read_config_roles_and_systems)"
ALL_ROLES=($(python - <<PY
import json
cfg = json.loads(r'''${CFG_JSON}''')
print(" ".join(cfg["roles"]))
PY
))
SYSTEMS=($(python - <<PY
import json
cfg = json.loads(r'''${CFG_JSON}''')
print(" ".join(cfg["systems"]))
PY
))

submit_servers() {
  local -a job_ids=()
  for role in "${ALL_ROLES[@]}"; do
    rm -f "results/routing/endpoints/${role}.url"
    jid=$(sbatch --parsable \
      --job-name="vllm-${role}" \
      --export=ALL,ROLE="${role}",PROJECT_DIR="${PROJECT_DIR}" \
      slurm/server_role_phase2.sbatch)
    echo "Server ${role}: job ${jid}" >&2
    job_ids+=("${jid}")
  done
  printf '%s\n' "${job_ids[@]}"
}

wait_for_endpoints() {
  local deadline=$((SECONDS + SERVER_WAIT_S))
  for role in "${ALL_ROLES[@]}"; do
    file="results/routing/endpoints/${role}.url"
    echo "Waiting for ${file} (up to ${SERVER_WAIT_S}s)..."
    while (( SECONDS < deadline )); do
      if [[ -s "${file}" ]]; then
        echo "  OK ${role}"
        break
      fi
      sleep 10
    done
    if [[ ! -s "${file}" ]]; then
      echo "ERROR: timeout waiting for ${file}"
      exit 1
    fi
  done
}

build_prompt_pool() {
  if [[ ! -s "${PROMPT_POOL}" ]]; then
    echo "Building prompt pool..."
    python -m bench.holdout_pool --config "${EVAL_CONFIG}" --output "${PROMPT_POOL}"
  else
    echo "Prompt pool exists: ${PROMPT_POOL}"
  fi
}

submit_clients() {
  if [[ ${#SYSTEMS[@]} -eq 0 ]]; then
    echo "ERROR: no systems found in ${EVAL_CONFIG}" >&2
    exit 1
  fi
  for sys in "${SYSTEMS[@]}"; do
    case "${sys}" in
      sysB_only_tiny)        tlimit="04:00:00" ;;
      sysA_only_teacher)     tlimit="08:00:00" ;;
      sysC_routing_distilled) tlimit="12:00:00" ;;
      # v2 — eval ~15 min; walltime includes wait for vLLM endpoints (no sbatch deps).
      sysC_l*)               tlimit="24:00:00" ;;
      sysD_cascade4)         tlimit="24:00:00" ;;
      sysE_l*)               tlimit="24:00:00" ;;
      *)                     tlimit="14:00:00" ;;
    esac
    jid=$(sbatch --parsable \
      --job-name="eval-${sys}" \
      --time="${tlimit}" \
      --export=ALL,SYSTEM_ID="${sys}",PROJECT_DIR="${PROJECT_DIR}",EVAL_CONFIG="${EVAL_CONFIG}",PROMPT_POOL="${PROMPT_POOL}",ENDPOINT_WAIT_S="${SERVER_WAIT_S:-7200}" \
      slurm/eval_holdout.sbatch)
    echo "Eval ${sys}: job ${jid} (time=${tlimit})" >&2
  done
}

case "${MODE}" in
  servers)
    submit_servers > logs/launch-eval-holdout-servers.ids
    echo "Servers submitted. When all .url files exist, run: bash slurm/launch_eval_holdout.sh clients"
    ;;
  clients)
    build_prompt_pool
    submit_clients
    ;;
  all)
    mapfile -t SIDS < <(submit_servers)
    dep=$(IFS=:; echo "${SIDS[*]}")
    echo "Server job IDs: ${dep}" >&2
    build_prompt_pool >&2
    submit_clients
    echo "Eval jobs poll endpoint .url files (ENDPOINT_WAIT_S=${SERVER_WAIT_S:-7200}); no sbatch --dependency on BSC." >&2
    echo "You can disconnect; check later: squeue -u \$USER" >&2
    echo "Config: ${EVAL_CONFIG}" >&2
    ;;
  *)
    echo "Usage: bash slurm/launch_eval_holdout.sh {servers|clients|all}"
    exit 1
    ;;
esac
