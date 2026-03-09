#!/bin/bash
#SBATCH --job-name=posteval-7b
#SBATCH --output=logs/posteval-7b-%j.out
#SBATCH --error=logs/posteval-7b-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=18:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh

mkdir -p logs results/quality results/throughput results/online results/distill

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS=3

echo "POSTEVAL 7B: Merge + Eval"
date

cleanup() { [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

ADAPTER=$(ls -dt results/distill/sft-qwen2.5-7b-*/final_adapter 2>/dev/null | head -1)
[[ -z "$ADAPTER" ]] && echo "ERROR: No 7B adapter" && exit 1

export ADAPTER MERGED="results/distill/merged-qwen2.5-7b"
python3 << 'MERGE'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, shutil, os

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, os.environ['ADAPTER'])
merged = model.merge_and_unload()
shutil.rmtree(os.environ['MERGED'], ignore_errors=True)
merged.save_pretrained(os.environ['MERGED'])
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct').save_pretrained(os.environ['MERGED'])
MERGE

python -m vllm.entrypoints.openai.api_server --model "$MERGED" --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto --seed 42 &
_SERVER_PID=$!
python -m serving.healthcheck --url "$SERVER_URL" --retries 120 --interval 5

python -m eval.run_quality --config configs/eval.yaml --model "$MERGED" --role "distilled_student_mid"

for REP in $(seq 1 ${NUM_REPS}); do
    python -m bench.run_throughput --config configs/benchmark.yaml --model "$MERGED" --role "distilled_student_mid"
    python -m bench.run_online_load --config configs/benchmark.yaml --model "$MERGED" --role "distilled_student_mid"
done

echo "✅ posteval-7b done"
