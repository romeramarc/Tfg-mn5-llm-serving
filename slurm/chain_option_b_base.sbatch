#!/bin/bash
# ------------------------------------------------------------------
# SLURM orchestrator — Option B (base 1.5B ladder):
#   capture A+B (if missing) → train A+B ladder_base → refine predictors
#   → holdout sysE_l100 with retrained bundles (4 servers + 1 eval).
#
# Submit from login (fire-and-forget):
#   sbatch slurm/chain_option_b_base.sbatch
# ------------------------------------------------------------------
#SBATCH --job-name=chain-optB-base
#SBATCH --output=logs/chain-optB-base-%j.out
#SBATCH --error=logs/chain-optB-base-%j.err
#SBATCH --partition=gpp
#SBATCH --account=bsc98
#SBATCH --qos=gp_bsccs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}}"
cd "${PROJECT_DIR}"
source env/setup_env.sh
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ROLE="student_small_base"
PHASE_A_CAP_CFG="configs/phase_a_capture_base_1p5b.yaml"
PHASE_B_CAP_CFG="configs/phase_b_capture_base_1p5b.yaml"
PHASE_A_TRAIN_CFG="configs/phase_a_train_ladder_base_1p5b.yaml"
PHASE_B_TRAIN_CFG="configs/phase_b_train_ladder_base_1p5b.yaml"
DS_A="results/phase_a/datasets/service_cost_phase_a_ladder_base.jsonl"
DS_B_EX="results/phase_b/datasets/quality_ex_ante_phase_b_ladder_base.jsonl"
DS_B_PH="results/phase_b/datasets/quality_post_hoc_phase_b_ladder_base.jsonl"
EVAL_CFG_OUT="configs/routing_eval_holdout_v2_retrained_base.yaml"
EVAL_SYSTEM="sysE_l100_retrained_base"
PROMPT_POOL="results/routing_eval_holdout/prompt_pool.jsonl"

wait_job() {
  local jid="$1"
  local label="${2:-job}"
  echo "Waiting for ${label} (${jid})..."
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    sleep 30
  done
  if sacct -j "${jid}" --format=State -n -P 2>/dev/null | grep -qE 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL'; then
    echo "ERROR: ${label} ${jid} failed (sacct)"
    sacct -j "${jid}" --format=JobID,State,ExitCode -n || true
    exit 1
  fi
  echo "OK ${label} (${jid}) finished"
}

has_capture_a() {
  compgen -G "results/phase_a/captures/capture-${ROLE}-*/trace.jsonl" > /dev/null
}
has_capture_b() {
  compgen -G "results/phase_b/captures/capture-${ROLE}-*/trace.jsonl" > /dev/null
}
datasets_ready() {
  [[ -s "${DS_A}" && -s "${DS_B_EX}" && -s "${DS_B_PH}" ]]
}

echo "=========================================="
echo " Option B chain (base 1.5B ladder)"
echo " Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " Job:  ${SLURM_JOB_ID}"
echo "=========================================="

# models_no_distill.yaml (student_small = HF 1.5B) for final holdout servers
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
grep -q 'MODELS_CONFIG' slurm/server_role_phase2.sbatch || \
  sed -i 's#--models configs/models.yaml#--models "${MODELS_CONFIG:-configs/models.yaml}"#' slurm/server_role_phase2.sbatch

# --- Captures (skip if traces already on scratch) ---
if ! has_capture_a; then
  echo ">>> Phase A capture (${ROLE})"
  export ROLES="${ROLE}" SKIP_TRAIN=1 PHASE_A_CONFIG="${PHASE_A_CAP_CFG}" BENCHMARK_LABEL=phase_a_workload_base
  bash slurm/launch_phase_a.sh
  # shellcheck source=/dev/null
  source "${PROJECT_DIR}/logs/last_phase_a_capture.env"
  cap_var="CAPTURE_JOB_${ROLE}"
  CAP_A="${!cap_var:-}"
  [[ -n "${CAP_A}" ]] || { echo "ERROR: missing ${cap_var} in last_phase_a_capture.env"; exit 1; }
  wait_job "${CAP_A}" "phase-a-capture"
else
  echo "SKIP Phase A capture (trace exists)"
fi

if ! has_capture_b; then
  echo ">>> Phase B capture (${ROLE})"
  export ROLES="${ROLE}" SKIP_TRAIN=1 PHASE_B_CONFIG="${PHASE_B_CAP_CFG}"
  bash slurm/launch_phase_b.sh
  # shellcheck source=/dev/null
  source "${PROJECT_DIR}/logs/last_phase_b_capture.env"
  cap_var="CAPTURE_JOB_${ROLE}"
  CAP_B="${!cap_var:-}"
  [[ -n "${CAP_B}" ]] || { echo "ERROR: missing ${cap_var} in last_phase_b_capture.env"; exit 1; }
  wait_job "${CAP_B}" "phase-b-capture"
else
  echo "SKIP Phase B capture (trace exists)"
fi

# --- Train ladder_base datasets ---
if ! datasets_ready; then
  echo ">>> Phase A train (ladder_base)"
  jid=$(sbatch --parsable --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PHASE_A_CONFIG="${PHASE_A_TRAIN_CFG}" slurm/phase_a_train.sbatch)
  wait_job "${jid}" "phase-a-train"

  echo ">>> Phase B train (ladder_base)"
  jid=$(sbatch --parsable --export=ALL,PROJECT_DIR="${PROJECT_DIR}",PHASE_B_CONFIG="${PHASE_B_TRAIN_CFG}" slurm/phase_b_train.sbatch)
  wait_job "${jid}" "phase-b-train"
else
  echo "SKIP train (ladder_base datasets exist)"
fi

for f in "${DS_A}" "${DS_B_EX}" "${DS_B_PH}"; do
  [[ -s "${f}" ]] || { echo "ERROR: missing ${f}"; exit 1; }
done

# --- Refine predictors (RF + thresholds) on ladder_base ---
echo ">>> Retrain predictors (model_tier, ladder_base)"
mkdir -p results/predictors_model_tier_base/datasets results/predictors_model_tier_base/phase_a results/predictors_model_tier_base/phase_b
jid=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",\
PHASE_A_SOURCE="${DS_A}",\
PHASE_B_EX_ANTE_SOURCE="${DS_B_EX}",\
PHASE_B_POST_HOC_SOURCE="${DS_B_PH}",\
DATASET_DIR=results/predictors_model_tier_base/datasets,\
PHASE_A_CONFIG=configs/refine_phase_a_model_tier_base.yaml,\
PHASE_B_CONFIG=configs/refine_phase_b_model_tier_base.yaml,\
BASE_ROUTING_CONFIG=configs/routing_eval_holdout_v2_no_distill.yaml,\
OUTPUT_ROUTING_CONFIG="${EVAL_CFG_OUT}" \
  slurm/retrain_predictors_model_tier.sbatch)
wait_job "${jid}" "pred-tier-base"

[[ -f "${EVAL_CFG_OUT}" ]] || { echo "ERROR: ${EVAL_CFG_OUT} not written"; exit 1; }

python - <<PY
import yaml, pathlib
p = pathlib.Path("${EVAL_CFG_OUT}")
cfg = yaml.safe_load(p.read_text())
cfg["common"]["results_base_dir"] = "results/routing_eval_holdout_v2_retrained_base"
for s in cfg.get("systems") or []:
    if s.get("id") == "sysE_l100_no_distill":
        s["id"] = "${EVAL_SYSTEM}"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Patched", p, "system=", [x["id"] for x in cfg["systems"]])
PY

python scripts/preflight_eval_holdout.py --config "${EVAL_CFG_OUT}"

# --- Holdout eval: 4 servers (student_small = base) + 1 client ---
echo ">>> Holdout ${EVAL_SYSTEM}"
rm -f results/routing/endpoints/{teacher,student_mid,student_q3b,student_small}.url

J_TEACHER=$(sbatch --parsable --export=ALL,ROLE=teacher,PROJECT_DIR="${PROJECT_DIR}" slurm/server_role_phase2.sbatch)
J_MID=$(sbatch --parsable --export=ALL,ROLE=student_mid,PROJECT_DIR="${PROJECT_DIR}" slurm/server_role_phase2.sbatch)
J_Q3B=$(sbatch --parsable --export=ALL,ROLE=student_q3b,PROJECT_DIR="${PROJECT_DIR}" slurm/server_role_phase2.sbatch)
J_SMALL=$(sbatch --parsable --export=ALL,ROLE=student_small,MODELS_CONFIG=configs/models_no_distill.yaml,PROJECT_DIR="${PROJECT_DIR}" slurm/server_role_phase2.sbatch)
echo "Servers: teacher=${J_TEACHER} mid=${J_MID} q3b=${J_Q3B} small=${J_SMALL}"

J_EVAL=$(sbatch --parsable --time=12:00:00 \
  --job-name="eval-${EVAL_SYSTEM}" \
  --export=ALL,SYSTEM_ID="${EVAL_SYSTEM}",PROJECT_DIR="${PROJECT_DIR}",EVAL_CONFIG="${EVAL_CFG_OUT}",PROMPT_POOL="${PROMPT_POOL}",ENDPOINT_WAIT_S=7200 \
  slurm/eval_holdout.sbatch)
echo "Eval submitted: ${J_EVAL} (polls endpoints inside job)"
wait_job "${J_EVAL}" "holdout-eval"

echo "=========================================="
echo " OPTION B CHAIN COMPLETE"
echo " Results: results/routing_eval_holdout_v2_retrained_base/${EVAL_SYSTEM}-*/"
echo " Bundles: results/predictors_model_tier_base/phase_*/refined-*"
echo "=========================================="
