#!/bin/bash
# ─────────────────────────────────────────────────────────────
# FASE 2: KNOWLEDGE DISTILLATION
# ─────────────────────────────────────────────────────────────
# 2.1 Generate teacher outputs (TRAIN splits — no data leakage)
# 2.2 Train 7B student (LoRA + 15% validation + early stopping)
# 2.3 Train 1.5B student (LoRA + 15% validation + early stopping)
#
# Chain:
#   JOB1=$(sbatch --parsable slurm/fase1_baselines.sbatch)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/fase2_distill.sbatch)
#   sbatch --dependency=afterok:$JOB2 slurm/fase3_posteval.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=fase2-distill
#SBATCH --output=logs/fase2-distill-%j.out
#SBATCH --error=logs/fase2-distill-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/distill

SERVER_URL="http://localhost:8000"
_SERVER_PID=""

# ── Header ─────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  FASE 2: KNOWLEDGE DISTILLATION"
echo "══════════════════════════════════════════════════════════"
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "══════════════════════════════════════════════════════════"

# ── Cleanup trap ───────────────────────────────────────────
cleanup() {
    if [[ -n "${_SERVER_PID}" ]]; then
        echo "--- Stopping server (PID ${_SERVER_PID}) ---"
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
    fi
    echo ""
    echo "--- Final GPU state ---"
    nvidia-smi
    echo "--- FASE 2 finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
}
trap cleanup EXIT

# ── Helpers ────────────────────────────────────────────────
start_server() {
    local ROLE=$1
    echo "Starting server for ${ROLE}..."
    python -m serving.start_server \
        --config configs/serving.yaml \
        --models configs/models.yaml \
        --role "${ROLE}" &
    _SERVER_PID=$!
    echo "Server PID: ${_SERVER_PID}"
    python -m serving.healthcheck \
        --url "${SERVER_URL}" \
        --retries 120 --interval 5
    echo "Server healthy."
}

stop_server() {
    if [[ -n "${_SERVER_PID}" ]]; then
        kill "${_SERVER_PID}" 2>/dev/null || true
        wait "${_SERVER_PID}" 2>/dev/null || true
        _SERVER_PID=""
        sleep 5
    fi
}

# ═══════════════════════════════════════════════════════════
# 2.1 Generate teacher outputs (TRAIN splits)
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 2.1 Generate teacher outputs (TRAIN splits)"
echo "════════════════════════════════════════"

start_server "teacher"

python -m distill.generate_teacher_outputs \
    --config configs/distill.yaml
echo "Teacher outputs generated (train splits, no leakage)."

stop_server

# ═══════════════════════════════════════════════════════════
# 2.2 Train 7B student (LoRA SFT + validation)
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 2.2 Train 7B student (LoRA + validation + early stop)"
echo "════════════════════════════════════════"

python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "Qwen/Qwen2.5-7B-Instruct"
echo "7B SFT complete."

# ═══════════════════════════════════════════════════════════
# 2.3 Train 1.5B student (LoRA SFT + validation)
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 2.3 Train 1.5B student (LoRA + validation + early stop)"
echo "════════════════════════════════════════"

python -m distill.train_student_sft \
    --config configs/distill.yaml \
    --student "Qwen/Qwen2.5-1.5B-Instruct"
echo "1.5B SFT complete."

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✅ FASE 2 COMPLETE — Distillation done"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  results/distill/teacher_outputs.jsonl"
echo "  results/distill/sft-qwen2.5-7b-*/final_adapter/"
echo "  results/distill/sft-qwen2.5-1.5b-*/final_adapter/"
