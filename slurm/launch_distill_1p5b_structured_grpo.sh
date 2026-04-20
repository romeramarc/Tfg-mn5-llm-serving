#!/bin/bash
# ------------------------------------------------------------------
# Launch structured-reasoning + GRPO-like 1.5B line.
#
# Pipeline:
#   distill_generate -> distill_train_1.5b -> distill_grpo_1.5b -> posteval_1.5b_structured_grpo
#
# Usage:
#   bash slurm/launch_distill_1p5b_structured_grpo.sh
#   DISTILL_CONFIG=configs/distill_1p5b_structured_grpo.yaml bash slurm/launch_distill_1p5b_structured_grpo.sh
# ------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."
DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_structured_grpo.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
SKIP_PERF_BENCH="${SKIP_PERF_BENCH:-0}"
NUM_REPS="${NUM_REPS:-3}"

if [[ ! -f "${DISTILL_CONFIG}" ]]; then
  echo "ERROR: Distillation config not found: ${DISTILL_CONFIG}"
  exit 1
fi

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: Evaluation config not found: ${EVAL_CONFIG}"
  exit 1
fi

GRPO_ENABLED=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print("1" if cfg.get("grpo_refinement", {}).get("enabled", False) else "0")
EOF
)

echo "=========================================="
echo " Launch 1.5B structured + GRPO line"
echo "=========================================="
echo "UTC time:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git HEAD:       $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "Config:         ${DISTILL_CONFIG}"
echo "Eval config:    ${EVAL_CONFIG}"
echo "Pipeline:       generate -> train_1.5b -> [optional grpo_1.5b] -> posteval"
echo "=========================================="

if [[ "${GRPO_ENABLED}" == "1" ]]; then
  python -m distill.preflight --step config --simulate-offline --config "${DISTILL_CONFIG}" --require-grpo
else
  python -m distill.preflight --step config --simulate-offline --config "${DISTILL_CONFIG}"
fi
python -m distill.preflight --step 1 --simulate-offline --config "${DISTILL_CONFIG}"

JOB_DG=$(sbatch --parsable --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_generate.sbatch)
echo "[1/4] distill-generate             : ${JOB_DG}"

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_train_1.5b.sbatch)
echo "[2/4] train-1.5b                   : ${JOB_T1} (after ${JOB_DG})"

if [[ "${GRPO_ENABLED}" == "1" ]]; then
  JOB_G3=$(sbatch --parsable --dependency=afterok:${JOB_T1} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG}" slurm/distill_grpo_1.5b.sbatch)
  echo "[3/4] grpo-like-refinement-1.5b    : ${JOB_G3} (after ${JOB_T1})"

  JOB_P4=$(sbatch --parsable --dependency=afterok:${JOB_G3} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG},EVAL_CONFIG=${EVAL_CONFIG},SKIP_PERF_BENCH=${SKIP_PERF_BENCH},NUM_REPS=${NUM_REPS}" slurm/posteval_1.5b_structured_grpo.sbatch)
  echo "[4/4] posteval-1.5b-structured-grpo: ${JOB_P4} (after ${JOB_G3})"
else
  DISTILL_ADAPTER_GLOB=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
student = cfg.get("training", {}).get("student_model", "Qwen/Qwen2.5-1.5B-Instruct")
model_short = student.split("/")[-1].lower().replace("-instruct", "")
exp = str(cfg.get("training", {}).get("experiment_tag", "")).strip()
tag = f"sft-{model_short}"
if exp:
    tag = f"{tag}-{exp}"
print(f"results/distill/{tag}-*/final_adapter")
EOF
)

  DISTILL_MERGED=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
student = cfg.get("training", {}).get("student_model", "Qwen/Qwen2.5-1.5B-Instruct")
model_short = student.split("/")[-1].lower().replace("-instruct", "")
exp = str(cfg.get("training", {}).get("experiment_tag", "")).strip()
suffix = f"-{exp}" if exp else ""
print(f"results/distill/merged-{model_short}{suffix}")
EOF
)

  DISTILL_EVAL_ROLE=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
exp = str(cfg.get("training", {}).get("experiment_tag", "")).strip().replace("-", "_")
print(f"distilled_student_small_{exp}" if exp else "distilled_student_small")
EOF
)

  JOB_G3="SKIPPED"
  echo "[3/4] grpo-like-refinement-1.5b    : skipped (grpo_refinement.enabled=false)"

  JOB_P4=$(sbatch --parsable --dependency=afterok:${JOB_T1} --export="ALL,ADAPTER_GLOB=${DISTILL_ADAPTER_GLOB},MERGED=${DISTILL_MERGED},EVAL_ROLE=${DISTILL_EVAL_ROLE},EVAL_CONFIG=${EVAL_CONFIG},SKIP_PERF_BENCH=${SKIP_PERF_BENCH},NUM_REPS=${NUM_REPS}" slurm/posteval_1.5b.sbatch)
  echo "[4/4] posteval-1.5b (distill-only) : ${JOB_P4} (after ${JOB_T1})"
fi

echo ""
echo "Submitted jobs:"
echo "  ${JOB_DG} distill-generate"
echo "  ${JOB_T1} train-1.5b"
echo "  ${JOB_G3} grpo-like-refinement-1.5b"
echo "  ${JOB_P4} posteval"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/distill-gen-${JOB_DG}.out"
echo "  tail -f logs/distill-1.5b-${JOB_T1}.out"
echo "  tail -f logs/distill-grpo1.5b-${JOB_G3}.out"
