#!/bin/bash
#SBATCH --job-name=posteval-1.5b
#SBATCH --output=logs/posteval-1.5b-%j.out
#SBATCH --error=logs/posteval-1.5b-%j.err
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

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS="${NUM_REPS:-3}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval.yaml}"
SKIP_PERF_BENCH="${SKIP_PERF_BENCH:-0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER_GLOB="${ADAPTER_GLOB:-results/distill/sft-qwen2.5-1.5b-*/final_adapter}"
MERGED="${MERGED:-results/distill/merged-qwen2.5-1.5b}"
EVAL_ROLE="${EVAL_ROLE:-distilled_student_small}"

echo "POSTEVAL 1.5B: Merge + Eval"
date

cleanup() { [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

ADAPTER=$(ls -dt ${ADAPTER_GLOB} 2>/dev/null | head -1)
[[ -z "$ADAPTER" ]] && echo "ERROR: No 1.5B adapter" && exit 1

export BASE_MODEL ADAPTER MERGED
python3 << 'MERGE'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained(os.environ['BASE_MODEL'], torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, os.environ['ADAPTER'])
merged = model.merge_and_unload()
shutil.rmtree(os.environ['MERGED'], ignore_errors=True)
merged.save_pretrained(os.environ['MERGED'])
AutoTokenizer.from_pretrained(os.environ['BASE_MODEL']).save_pretrained(os.environ['MERGED'])
MERGE

python -m vllm.entrypoints.openai.api_server --model "$MERGED" --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto --seed 42 &
_SERVER_PID=$!
python -m serving.healthcheck --url "$SERVER_URL" --retries 120 --interval 5

python -m eval.run_quality --config "${EVAL_CONFIG}" --model "$MERGED" --role "${EVAL_ROLE}"

if [[ "${SKIP_PERF_BENCH}" == "1" ]]; then
    echo "Skipping throughput/online benchmarks (SKIP_PERF_BENCH=1)"
else
    for REP in $(seq 1 ${NUM_REPS}); do
        python -m bench.run_throughput --config configs/benchmark.yaml --model "$MERGED" --role "${EVAL_ROLE}"
        python -m bench.run_online_load --config configs/benchmark.yaml --model "$MERGED" --role "${EVAL_ROLE}"
    done
fi

echo "posteval-1.5b done"
