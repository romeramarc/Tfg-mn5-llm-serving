#!/bin/bash
# ------------------------------------------------------------------
# SLURM job - Distillation Step 3 (optional): GRPO-like refinement
# for Qwen2.5-1.5B using weighted completion-only SFT.
#
# Pipeline inside this job:
#   1) Merge latest structured SFT adapter into a temporary policy model
#   2) Start policy vLLM endpoint
#   3) Sample + reward + group-advantage weighting
#   4) Train weighted LoRA adapter (GRPO-like)
# ------------------------------------------------------------------
#SBATCH --job-name=distill-grpo1.5b
#SBATCH --output=logs/distill-grpo1.5b-%j.out
#SBATCH --error=logs/distill-grpo1.5b-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/distill

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_structured_grpo.yaml}"
MIN_VALID_SAMPLES="${MIN_VALID_SAMPLES:-}"
GRPO_MAX_PROMPTS="${GRPO_MAX_PROMPTS:-}"
GRPO_MAX_TRAIN_SAMPLES="${GRPO_MAX_TRAIN_SAMPLES:-}"

BASE_MODEL=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("student_model", cfg.get("training", {}).get("student_model", "Qwen/Qwen2.5-1.5B-Instruct")))
EOF
)

INIT_PATTERN=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("init_adapter_path", "results/distill/sft-qwen2.5-1.5b-structured-*/final_adapter"))
EOF
)

POLICY_MODEL_PATH=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("policy_model_path", "results/distill/merged-qwen2.5-1.5b-structured-policy"))
EOF
)

POLICY_BASE_URL=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("base_url", "http://localhost:8002"))
EOF
)

POLICY_PORT=$(python - <<EOF
from urllib.parse import urlparse
url = urlparse("${POLICY_BASE_URL}")
print(url.port or 8002)
EOF
)

OUTPUT_GLOB=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("output_adapter_glob", "results/distill/sft-qwen2.5-1.5b-structured-grpo-*/final_adapter"))
EOF
)

ADAPTER=$(python - <<EOF
import glob
pattern = "${INIT_PATTERN}"
matches = sorted(glob.glob(pattern), reverse=True)
print(matches[0] if matches else "")
EOF
)

if [[ -z "${ADAPTER}" ]]; then
    echo "ERROR: Could not resolve stage1 adapter with pattern: ${INIT_PATTERN}"
    exit 1
fi

_SERVER_PID=""
cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "========================================="
echo " Distillation Step 3: GRPO-like 1.5B"
echo " Config:             ${DISTILL_CONFIG}"
echo " Base model:         ${BASE_MODEL}"
echo " Init adapter:       ${ADAPTER}"
echo " Policy model path:  ${POLICY_MODEL_PATH}"
echo " Policy URL:         ${POLICY_BASE_URL}"
echo " Job ID:             ${SLURM_JOB_ID}"
echo " Node:               $(hostname)"
if [[ -n "${GRPO_MAX_PROMPTS}" ]]; then
    echo " GRPO max prompts:   ${GRPO_MAX_PROMPTS}"
fi
if [[ -n "${GRPO_MAX_TRAIN_SAMPLES}" ]]; then
    echo " GRPO max train:     ${GRPO_MAX_TRAIN_SAMPLES}"
fi
echo "========================================="

python -m distill.preflight --step config --config "${DISTILL_CONFIG}" --require-grpo --simulate-offline

PREFLIGHT_STEP2_ARGS=(--step 2 --config "${DISTILL_CONFIG}")
if [[ -n "${MIN_VALID_SAMPLES}" ]]; then
    PREFLIGHT_STEP2_ARGS+=(--min-valid-samples "${MIN_VALID_SAMPLES}")
fi
python -m distill.preflight "${PREFLIGHT_STEP2_ARGS[@]}"

export BASE_MODEL ADAPTER POLICY_MODEL_PATH
python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil
import torch

base_model = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER"]
out_dir = os.environ["POLICY_MODEL_PATH"]

base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter)
merged = model.merge_and_unload()
shutil.rmtree(out_dir, ignore_errors=True)
merged.save_pretrained(out_dir)
AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
print(f"Merged policy model saved to: {out_dir}")
PY

python -m vllm.entrypoints.openai.api_server \
    --model "${POLICY_MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${POLICY_PORT}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --seed 42 &
_SERVER_PID=$!

python -m serving.healthcheck --url "${POLICY_BASE_URL}" --retries 120 --interval 5

# Let refinement process stop the policy server after sampling to avoid
# GPU memory contention during adapter merge and weighted SFT training.
export GRPO_POLICY_SERVER_PID="${_SERVER_PID}"

GRPO_ARGS=(--config "${DISTILL_CONFIG}")
if [[ -n "${GRPO_MAX_PROMPTS}" ]]; then
    GRPO_ARGS+=(--max-prompts "${GRPO_MAX_PROMPTS}")
fi
if [[ -n "${GRPO_MAX_TRAIN_SAMPLES}" ]]; then
    GRPO_ARGS+=(--max-train-samples "${GRPO_MAX_TRAIN_SAMPLES}")
fi
python -m distill.refine_student_grpo "${GRPO_ARGS[@]}"

OUTPUT_ADAPTER=$(python - <<EOF
import glob
pattern = "${OUTPUT_GLOB}"
matches = sorted(glob.glob(pattern), reverse=True)
print(matches[0] if matches else "")
EOF
)

if [[ -z "${OUTPUT_ADAPTER}" ]]; then
    echo "ERROR: No GRPO adapter found with pattern: ${OUTPUT_GLOB}"
    exit 1
fi

echo ""
echo "========================================="
echo " GRPO-like refinement complete"
echo " Adapter: ${OUTPUT_ADAPTER}"
echo "========================================="
