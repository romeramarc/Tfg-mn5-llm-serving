#!/bin/bash
# ─────────────────────────────────────────────────────────────
# FASE 1: PRE-DISTILLATION BASELINES
# ─────────────────────────────────────────────────────────────
# Teacher (14B):  Quality + Routing + Throughput×3 + Online×3
# Student 7B:     Quality + Throughput×3 + Online×3
# Student 1.5B:   Quality + Throughput×3 + Online×3
#
# Usage:
#   JOB1=$(sbatch --parsable slurm/fase1_baselines.sbatch)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/fase2_distill.sbatch)
#   sbatch --dependency=afterok:$JOB2 slurm/fase3_posteval.sbatch
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=fase1-baselines
#SBATCH --output=logs/fase1-baselines-%j.out
#SBATCH --error=logs/fase1-baselines-%j.err
#SBATCH --partition=acc
#SBATCH --account=bsc98
#SBATCH --qos=acc_bsccs
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --exclusive
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ────────────────────────────────────────────
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving
source env/setup_env.sh
mkdir -p logs results/quality results/throughput results/online results/routing

SERVER_URL="http://localhost:8000"
_SERVER_PID=""
NUM_REPS=3

# ── Header ─────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  FASE 1: PRE-DISTILLATION BASELINES"
echo "══════════════════════════════════════════════════════════"
echo "Date:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
pip freeze 2>/dev/null | head -60
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
    echo "--- FASE 1 finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
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

run_quality() {
    local MODEL=$1
    local ROLE=$2
    echo "Quality eval for ${ROLE}..."
    python -m eval.run_quality \
        --config configs/eval.yaml \
        --model  "${MODEL}" \
        --role   "${ROLE}"
    echo "Quality eval done for ${ROLE}."
}

run_efficiency() {
    local MODEL=$1
    local ROLE=$2
    for REP in $(seq 1 ${NUM_REPS}); do
        echo "── Efficiency rep ${REP}/${NUM_REPS} for ${ROLE} ──"
        echo "[${REP}a] Throughput..."
        python -m bench.run_throughput \
            --config configs/benchmark.yaml \
            --model "${MODEL}" \
            --role  "${ROLE}"
        echo "[${REP}b] Online load..."
        python -m bench.run_online_load \
            --config configs/benchmark.yaml \
            --model "${MODEL}" \
            --role  "${ROLE}"
        echo "── Rep ${REP} done ──"
    done
}

# ═══════════════════════════════════════════════════════════
# 1.1 TEACHER (14B): Quality + Routing + Efficiency ×3
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 1.1 TEACHER: Quality + Routing + Efficiency"
echo "════════════════════════════════════════"

start_server "teacher"

# Quality eval (GSM8K + MATH-500)
run_quality "Qwen/Qwen2.5-14B-Instruct" "teacher"

# Routing eval
echo "Routing eval for teacher..."
if python -c "import routing.run_routing" 2>/dev/null; then
    python -m routing.run_routing --config configs/routing.yaml
    echo "Routing eval done."
else
    python -m eval.run_quality \
        --config configs/eval.yaml \
        --model  "Qwen/Qwen2.5-14B-Instruct" \
        --role   "teacher_routing"
    echo "Routing fallback done."
fi

# Efficiency ×3 reps
run_efficiency "Qwen/Qwen2.5-14B-Instruct" "teacher"

stop_server
echo "✅ Teacher baseline complete."

# ═══════════════════════════════════════════════════════════
# 1.2 STUDENT MID (7B): Quality + Efficiency ×3
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 1.2 STUDENT MID (7B): Quality + Efficiency"
echo "════════════════════════════════════════"

start_server "student_mid"

run_quality "Qwen/Qwen2.5-7B-Instruct" "student_mid"
run_efficiency "Qwen/Qwen2.5-7B-Instruct" "student_mid"

stop_server
echo "✅ Student mid baseline complete."

# ═══════════════════════════════════════════════════════════
# 1.3 STUDENT SMALL (1.5B): Quality + Efficiency ×3
# ═══════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " 1.3 STUDENT SMALL (1.5B): Quality + Efficiency"
echo "════════════════════════════════════════"

start_server "student_small"

run_quality "Qwen/Qwen2.5-1.5B-Instruct" "student_small"
run_efficiency "Qwen/Qwen2.5-1.5B-Instruct" "student_small"

stop_server
echo "✅ Student small baseline complete."

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✅ FASE 1 COMPLETE — All pre-distillation baselines done"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Results:"
echo "  results/quality/quality-teacher-*/"
echo "  results/quality/quality-teacher_routing-*/"
echo "  results/quality/quality-student_mid-*/"
echo "  results/quality/quality-student_small-*/"
echo "  results/throughput/throughput-{teacher,student_mid,student_small}-*/ (×3 each)"
echo "  results/online/online-{teacher,student_mid,student_small}-*/       (×3 each)"
