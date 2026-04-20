#!/bin/bash
# ------------------------------------------------------------------
# Launch smoke integration run for structured-reasoning + GRPO-like 1.5B.
#
# Pipeline:
#   preflight(config+step1) -> distill_generate -> distill_train_1.5b
#   -> posteval_1.5b (distill-only smoke)
#   -> distill_grpo_1.5b
#   -> posteval_1.5b_structured_grpo (post-RL smoke)
# ------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_structured_grpo_smoke.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_smoke.yaml}"
SKIP_PERF_BENCH="${SKIP_PERF_BENCH:-1}"
NUM_REPS="${NUM_REPS:-1}"
MIN_VALID_SAMPLES="${MIN_VALID_SAMPLES:-}"
TEACHER_MAX_SAMPLES="${TEACHER_MAX_SAMPLES:-}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-}"
GRPO_MAX_PROMPTS="${GRPO_MAX_PROMPTS:-}"
GRPO_MAX_TRAIN_SAMPLES="${GRPO_MAX_TRAIN_SAMPLES:-}"

if [[ ! -f "${DISTILL_CONFIG}" ]]; then
  echo "ERROR: Distillation smoke config not found: ${DISTILL_CONFIG}"
  exit 1
fi

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: Eval smoke config not found: ${EVAL_CONFIG}"
  exit 1
fi

BASE_MODEL=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("training", {}).get("student_model", "Qwen/Qwen2.5-1.5B-Instruct"))
EOF
)

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

echo "=========================================================="
echo " Structured + GRPO 1.5B smoke launch"
echo "=========================================================="
echo "UTC time:            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git HEAD:            $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "Distill config:      ${DISTILL_CONFIG}"
echo "Eval config:         ${EVAL_CONFIG}"
echo "Base model:          ${BASE_MODEL}"
echo "Distill adapter glob:${DISTILL_ADAPTER_GLOB}"
echo "Distill merged path: ${DISTILL_MERGED}"
echo "Distill eval role:   ${DISTILL_EVAL_ROLE}"
echo "Skip perf benches:   ${SKIP_PERF_BENCH}"
echo "Num reps:            ${NUM_REPS}"
if [[ -n "${TEACHER_MAX_SAMPLES}" ]]; then
  echo "Teacher max samples: ${TEACHER_MAX_SAMPLES}"
fi
if [[ -n "${SFT_MAX_SAMPLES}" ]]; then
  echo "SFT max samples:     ${SFT_MAX_SAMPLES}"
fi
if [[ -n "${GRPO_MAX_PROMPTS}" ]]; then
  echo "GRPO max prompts:    ${GRPO_MAX_PROMPTS}"
fi
if [[ -n "${GRPO_MAX_TRAIN_SAMPLES}" ]]; then
  echo "GRPO max train samp: ${GRPO_MAX_TRAIN_SAMPLES}"
fi
echo "=========================================================="

python -m distill.preflight --step config --simulate-offline --config "${DISTILL_CONFIG}" --require-grpo --smoke
python -m distill.preflight --step 1 --simulate-offline --config "${DISTILL_CONFIG}"

JOB_DG=$(sbatch --parsable --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG},MIN_VALID_SAMPLES=${MIN_VALID_SAMPLES},TEACHER_MAX_SAMPLES=${TEACHER_MAX_SAMPLES}" slurm/distill_generate.sbatch)
echo "[1/5] distill-generate (smoke)          : ${JOB_DG}"

JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG},MIN_VALID_SAMPLES=${MIN_VALID_SAMPLES},SFT_MAX_SAMPLES=${SFT_MAX_SAMPLES}" slurm/distill_train_1.5b.sbatch)
echo "[2/5] train-1.5b (smoke)                : ${JOB_T1} (after ${JOB_DG})"

JOB_E2=$(sbatch --parsable --dependency=afterok:${JOB_T1} --export="ALL,BASE_MODEL=${BASE_MODEL},ADAPTER_GLOB=${DISTILL_ADAPTER_GLOB},MERGED=${DISTILL_MERGED},EVAL_ROLE=${DISTILL_EVAL_ROLE},EVAL_CONFIG=${EVAL_CONFIG},SKIP_PERF_BENCH=${SKIP_PERF_BENCH},NUM_REPS=${NUM_REPS}" slurm/posteval_1.5b.sbatch)
echo "[3/5] posteval distill-only (smoke)     : ${JOB_E2} (after ${JOB_T1})"

JOB_G3=$(sbatch --parsable --dependency=afterok:${JOB_E2} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG},MIN_VALID_SAMPLES=${MIN_VALID_SAMPLES},GRPO_MAX_PROMPTS=${GRPO_MAX_PROMPTS},GRPO_MAX_TRAIN_SAMPLES=${GRPO_MAX_TRAIN_SAMPLES}" slurm/distill_grpo_1.5b.sbatch)
echo "[4/5] grpo-like-refinement-1.5b (smoke) : ${JOB_G3} (after ${JOB_E2})"

JOB_P4=$(sbatch --parsable --dependency=afterok:${JOB_G3} --export="ALL,DISTILL_CONFIG=${DISTILL_CONFIG},EVAL_CONFIG=${EVAL_CONFIG},SKIP_PERF_BENCH=${SKIP_PERF_BENCH},NUM_REPS=${NUM_REPS}" slurm/posteval_1.5b_structured_grpo.sbatch)
echo "[5/5] posteval post-RL (smoke)          : ${JOB_P4} (after ${JOB_G3})"

echo ""
echo "Submitted smoke jobs in strict order:"
echo "  ${JOB_DG} distill-generate"
echo "  ${JOB_T1} train-1.5b"
echo "  ${JOB_E2} posteval-distill-only"
echo "  ${JOB_G3} grpo-like-refinement"
echo "  ${JOB_P4} posteval-post-rl"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/distill-gen-${JOB_DG}.out"
echo "  tail -f logs/distill-1.5b-${JOB_T1}.out"
echo "  tail -f logs/posteval-1.5b-${JOB_E2}.out"
echo "  tail -f logs/distill-grpo1.5b-${JOB_G3}.out"
echo "  tail -f logs/posteval-1.5b-grpo-${JOB_P4}.out"
