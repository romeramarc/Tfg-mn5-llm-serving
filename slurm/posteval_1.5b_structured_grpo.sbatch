#!/bin/bash
#SBATCH --job-name=posteval-1.5b-grpo
#SBATCH --output=logs/posteval-1.5b-grpo-%j.out
#SBATCH --error=logs/posteval-1.5b-grpo-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh

mkdir -p logs results/quality results/throughput results/online results/distill

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_1p5b_structured_grpo.yaml}"
SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS="${NUM_REPS:-3}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
SKIP_PERF_BENCH="${SKIP_PERF_BENCH:-0}"

BASE_MODEL=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("student_model", cfg.get("training", {}).get("student_model", "Qwen/Qwen2.5-1.5B-Instruct")))
EOF
)

ADAPTER_GLOB=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("output_adapter_glob", "results/distill/sft-qwen2.5-1.5b-structured-grpo-*/final_adapter"))
EOF
)

MERGED_PATH=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("merged_output_path", "results/distill/merged-qwen2.5-1.5b-structured-grpo"))
EOF
)

EVAL_ROLE=$(python - <<EOF
from utils.config_loader import load_yaml
cfg = load_yaml("${DISTILL_CONFIG}")
print(cfg.get("grpo_refinement", {}).get("eval_role", "distilled_student_small_structured_grpo"))
EOF
)

cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ADAPTER=$(python - <<EOF
import glob
pattern = "${ADAPTER_GLOB}"
matches = sorted(glob.glob(pattern), reverse=True)
print(matches[0] if matches else "")
EOF
)

if [[ -z "${ADAPTER}" ]]; then
    echo "ERROR: No GRPO adapter found with pattern: ${ADAPTER_GLOB}"
    exit 1
fi

echo "POSTEVAL 1.5B GRPO: Merge + Eval"
date

echo "Config:       ${DISTILL_CONFIG}"
echo "Base model:   ${BASE_MODEL}"
echo "Adapter:      ${ADAPTER}"
echo "Merged path:  ${MERGED_PATH}"
echo "Eval role:    ${EVAL_ROLE}"
echo "Eval config:  ${EVAL_CONFIG}"

export BASE_MODEL ADAPTER MERGED_PATH
python - <<'MERGE'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import shutil
import torch

base = AutoModelForCausalLM.from_pretrained(os.environ["BASE_MODEL"], torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, os.environ["ADAPTER"])
merged = model.merge_and_unload()
shutil.rmtree(os.environ["MERGED_PATH"], ignore_errors=True)
merged.save_pretrained(os.environ["MERGED_PATH"])
AutoTokenizer.from_pretrained(os.environ["BASE_MODEL"]).save_pretrained(os.environ["MERGED_PATH"])
MERGE

python -m vllm.entrypoints.openai.api_server --model "${MERGED_PATH}" --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto --seed 42 &
_SERVER_PID=$!
python -m serving.healthcheck --url "${SERVER_URL}" --retries 120 --interval 5

python -m eval.run_quality --config "${EVAL_CONFIG}" --model "${MERGED_PATH}" --role "${EVAL_ROLE}"

if [[ "${SKIP_PERF_BENCH}" == "1" ]]; then
    echo "Skipping throughput/online benchmarks (SKIP_PERF_BENCH=1)"
else
    for REP in $(seq 1 ${NUM_REPS}); do
        python -m bench.run_throughput --config configs/benchmark.yaml --model "${MERGED_PATH}" --role "${EVAL_ROLE}"
        python -m bench.run_online_load --config configs/benchmark.yaml --model "${MERGED_PATH}" --role "${EVAL_ROLE}"
    done
fi

echo "posteval-1.5b-structured-grpo done"
