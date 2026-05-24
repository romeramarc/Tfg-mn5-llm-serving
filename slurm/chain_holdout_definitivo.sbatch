#!/bin/bash
# ------------------------------------------------------------------
# Cadena definitiva holdout v2 (login node, fire-and-forget):
#
#   Fase 1 — Barrido λ realista (distilado): routing_eval_holdout_v2_routing_real.yaml
#            5× sysE + 4× sysC opcionales, 4 servidores vLLM destilados.
#   Fase 2 — Comparación destilado vs base (Opción A): no_distill, mismo λ=1e-4.
#            student_small sirve Qwen2.5-1.5B-Instruct vía models_no_distill.yaml.
#
# Antes de lanzar: cancela jobs viejos (ver slurm/submit_holdout_definitivo.sh).
#
#   sbatch slurm/chain_holdout_definitivo.sbatch
# ------------------------------------------------------------------
#SBATCH --job-name=chain-holdout-def
#SBATCH --output=logs/chain-holdout-def-%j.out
#SBATCH --error=logs/chain-holdout-def-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}}"
cd "${PROJECT_DIR}"
source env/setup_env.sh
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PROMPT_POOL="results/routing_eval_holdout/prompt_pool.jsonl"
CFG_REAL="configs/routing_eval_holdout_v2_routing_real.yaml"
CFG_ND="configs/routing_eval_holdout_v2_no_distill.yaml"
SERVER_WAIT_S="${SERVER_WAIT_S:-7200}"
ENDPOINT_WAIT_S="${ENDPOINT_WAIT_S:-7200}"

wait_job() {
  local jid="$1"
  local label="${2:-job}"
  echo "Waiting for ${label} (${jid})..."
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    sleep 30
  done
  if sacct -j "${jid}" --format=State -n -P 2>/dev/null | grep -qE 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL'; then
    echo "ERROR: ${label} ${jid} failed"
    sacct -j "${jid}" --format=JobID,State,ExitCode -n || true
    exit 1
  fi
  echo "OK ${label} (${jid})"
}

wait_jobs_file() {
  local f="$1"
  local label="$2"
  [[ -s "${f}" ]] || { echo "ERROR: missing job id file ${f}"; exit 1; }
  while read -r jid; do
    [[ -n "${jid}" ]] || continue
    wait_job "${jid}" "${label}-${jid}"
  done < "${f}"
}

launch_holdout_wave() {
  local cfg="$1"
  local models_config="${2:-configs/models.yaml}"
  local tag="$3"
  local id_file="logs/chain-holdout-def-${SLURM_JOB_ID}-${tag}.ids"

  echo "=========================================="
  echo " Wave: ${tag}"
  echo " Config: ${cfg}"
  echo " MODELS_CONFIG (student_small): ${models_config}"
  echo "=========================================="

  python scripts/preflight_eval_holdout.py --config "${cfg}"

  if [[ ! -s "${PROMPT_POOL}" ]]; then
    python -m bench.holdout_pool --config "${cfg}" --output "${PROMPT_POOL}"
  fi

  local cfg_json
  cfg_json="$(WAVE_CFG="${cfg}" python - <<'PY'
import json, os
from utils.config_loader import load_yaml
cfg = load_yaml(os.environ["WAVE_CFG"])
roles, systems = [], []
for s in cfg.get("systems") or []:
    sid = s.get("id")
    if sid:
        systems.append(str(sid))
    for r in s.get("roles") or []:
        if r not in roles:
            roles.append(r)
print(json.dumps({"roles": roles, "systems": systems}))
PY
)"

  local -a roles systems
  mapfile -t roles < <(python -c "import json,sys; d=json.loads(sys.argv[1]); print('\n'.join(d['roles']))" "${cfg_json}")
  mapfile -t systems < <(python -c "import json,sys; d=json.loads(sys.argv[1]); print('\n'.join(d['systems']))" "${cfg_json}")

  : > "${id_file}"
  for role in "${roles[@]}"; do
    rm -f "results/routing/endpoints/${role}.url"
    if [[ "${role}" == "student_small" && "${models_config}" != "configs/models.yaml" ]]; then
      jid=$(sbatch --parsable --job-name="vllm-${role}-${tag}" \
        --export=ALL,ROLE="${role}",PROJECT_DIR="${PROJECT_DIR}",MODELS_CONFIG="${models_config}" \
        slurm/server_role_phase2.sbatch)
    else
      jid=$(sbatch --parsable --job-name="vllm-${role}-${tag}" \
        --export=ALL,ROLE="${role}",PROJECT_DIR="${PROJECT_DIR}" \
        slurm/server_role_phase2.sbatch)
    fi
    echo "${jid}" >> "${id_file}"
    echo "Server ${role}: ${jid}"
  done

  local deadline=$((SECONDS + SERVER_WAIT_S))
  for role in "${roles[@]}"; do
    file="results/routing/endpoints/${role}.url"
    echo "Waiting for ${file}..."
    while (( SECONDS < deadline )); do
      [[ -s "${file}" ]] && break
      sleep 10
    done
    [[ -s "${file}" ]] || { echo "ERROR: timeout ${file}"; exit 1; }
  done

  for sys in "${systems[@]}"; do
  case "${sys}" in
    sysB_only_tiny)        tlimit="04:00:00" ;;
    sysA_only_teacher)     tlimit="08:00:00" ;;
    sysC_routing_distilled) tlimit="12:00:00" ;;
    sysC_l*|sysD_cascade4|sysE_l*|sysE_floor*) tlimit="24:00:00" ;;
    *)                     tlimit="14:00:00" ;;
  esac
    jid=$(sbatch --parsable --time="${tlimit}" \
      --job-name="eval-${sys}-${tag}" \
      --export=ALL,SYSTEM_ID="${sys}",PROJECT_DIR="${PROJECT_DIR}",EVAL_CONFIG="${cfg}",PROMPT_POOL="${PROMPT_POOL}",ENDPOINT_WAIT_S="${ENDPOINT_WAIT_S}" \
      slurm/eval_holdout.sbatch)
    echo "${jid}" >> "${id_file}"
    echo "Eval ${sys}: ${jid}"
  done

  # Wait eval jobs only (skip server lines: same file has servers first — filter eval-*)
  while read -r jid; do
    [[ -n "${jid}" ]] || continue
    jname=$(sacct -j "${jid}" --format=JobName -n -P 2>/dev/null | head -1 || true)
    if [[ "${jname}" == eval-* ]]; then
      wait_job "${jid}" "eval-${jname}"
    fi
  done < "${id_file}"

  echo "Wave ${tag} complete."
}

# models_no_distill.yaml for Fase 2
if [[ ! -f configs/models_no_distill.yaml ]]; then
  python - <<'PY'
import yaml, pathlib
src, dst = pathlib.Path("configs/models.yaml"), pathlib.Path("configs/models_no_distill.yaml")
d = yaml.safe_load(src.read_text())
d["student_small"]["name"] = "Qwen/Qwen2.5-1.5B-Instruct"
d["student_small"]["revision"] = "main"
dst.write_text(yaml.safe_dump(d, sort_keys=False))
print("Wrote configs/models_no_distill.yaml")
PY
fi

echo "=========================================="
echo " HOLDOUT DEFINITIVO — chain start"
echo " Job: ${SLURM_JOB_ID}  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

launch_holdout_wave "${CFG_REAL}" "configs/models.yaml" "routing-real"
launch_holdout_wave "${CFG_ND}" "configs/models_no_distill.yaml" "no-distill"

echo "=========================================="
echo " CHAIN COMPLETE"
echo "  Fase 1: results/routing_eval_holdout_v2_routing_real/"
echo "  Fase 2: results/routing_eval_holdout_v2_no_distill/"
echo "  Opción B (predictores en base): bash slurm/submit_option_b_base.sh"
echo "=========================================="
