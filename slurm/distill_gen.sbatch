#!/bin/bash
#SBATCH --job-name=distill-gen
#SBATCH --output=logs/distill-gen-%j.out
#SBATCH --error=logs/distill-gen-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00

set -euo pipefail

cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE
mkdir -p logs results/distill

SERVER_URL="http://localhost:8000"
_SERVER_PID=""

echo "DISTILL GEN: Generate teacher outputs"
date

cleanup() { [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

start_server() {
    python -m serving.start_server --config configs/serving.yaml --models configs/models.yaml --role "$1" &
    _SERVER_PID=$!
    python -m serving.healthcheck --url "$SERVER_URL" --retries 120 --interval 5
}

stop_server() {
    [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true
    _SERVER_PID=""
    sleep 5
}

start_server "teacher"
python -m distill.generate_teacher_outputs --config configs/distill.yaml
stop_server
echo "✅ distill-gen done"
