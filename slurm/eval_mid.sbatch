#!/bin/bash
#SBATCH --job-name=eval-mid
#SBATCH --output=logs/eval-mid-%j.out
#SBATCH --error=logs/eval-mid-%j.err
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
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE
mkdir -p logs results/quality results/throughput results/online

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS=3

echo "EVAL STUDENT MID:"
date

cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_server() {
    python -m serving.start_server --config configs/serving.yaml --models configs/models.yaml --role "$1" &
    _SERVER_PID=$!
    python -m serving.healthcheck --url "$SERVER_URL" --retries 120 --interval 5
}

stop_server() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
        _SERVER_PID=""
        sleep 5
    fi
}

start_server "student_mid"
python -m eval.run_quality --config configs/eval.yaml --model "Qwen/Qwen2.5-7B-Instruct" --role "student_mid"

for REP in $(seq 1 ${NUM_REPS}); do
    python -m bench.run_throughput --config configs/benchmark.yaml --model "Qwen/Qwen2.5-7B-Instruct" --role "student_mid"
    python -m bench.run_online_load --config configs/benchmark.yaml --model "Qwen/Qwen2.5-7B-Instruct" --role "student_mid"
done

stop_server
echo "✅ eval-mid done"
