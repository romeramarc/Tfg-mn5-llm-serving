#!/bin/bash
#SBATCH --job-name=eval-teacher
#SBATCH --output=logs/eval-teacher-%j.out
#SBATCH --error=logs/eval-teacher-%j.err
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
mkdir -p logs results/quality results/throughput results/online results/routing

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS=3

echo "══════════════════════════════════════════════════════════"
echo "  EVAL TEACHER (baseline): Quality + Routing + Efficiency"
echo "══════════════════════════════════════════════════════════"
date

cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
    nvidia-smi
}
trap cleanup EXIT

start_server() {
    local ROLE=$1
    echo "Starting server for ${ROLE}..."
    python -m serving.start_server \
        --config configs/serving.yaml \
        --models configs/models.yaml \
        --role "${ROLE}" &
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

start_server "teacher"

echo "Quality eval..."
python -m eval.run_quality --config configs/eval.yaml --model "Qwen/Qwen2.5-14B-Instruct" --role "teacher"

echo "Routing eval..."
python -m routing.run_routing --config configs/routing.yaml || true

for REP in $(seq 1 ${NUM_REPS}); do
    echo "Efficiency rep ${REP}..."
    python -m bench.run_throughput --config configs/benchmark.yaml --model "Qwen/Qwen2.5-14B-Instruct" --role "teacher"
    python -m bench.run_online_load --config configs/benchmark.yaml --model "Qwen/Qwen2.5-14B-Instruct" --role "teacher"
done

stop_server
echo "✅ eval-teacher done"
