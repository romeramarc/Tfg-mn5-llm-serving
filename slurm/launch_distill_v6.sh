#!/bin/bash
# =============================================================================
#  EXPERIMENT 6 — Pure-Distillation · Maximum Envelope (Sprint 1 pipeline)
# =============================================================================
#  Objetivo: apretar al máximo la calidad del 1.5B con distillation pura.
#   Sprint 1 (este script):
#     1) distill-gen-v6   : RFT ­— N samples/problema, guardar hasta K correctos
#        con diversidad (MATH + GSM8K train). Config: configs/distill_v6.yaml.
#     2) distill-1.5b-v6  : Full-parameter FT del 1.5B (no LoRA) sobre el JSONL
#        expandido. Config: configs/distill_v6.yaml  (training.mode=full).
#     3) posteval-1.5b-v6 : Quality (GSM8K + MATH) + throughput/online. Sin
#        paso de merge (el modelo guardado es ya completo).
#
#  Sprints posteriores (se añaden sobre esta misma config, no rompen nada):
#     · Sprint 2: persistir top-20 logprobs y entrenar con CE + KL.
#     · Sprint 3: GKD on-policy (student samples, teacher supervisa).
#     · Sprint 4: RFT iteration sobre errores del student + polish + multi-seed.
#
#  Dependency graph:
#
#    distill-gen-v6 (≤20h) ── distill-1.5b-v6 (≤16h) ── posteval-1.5b-v6 (≤12h)
#
#  Critical path ≈ 48h wall-clock worst-case (normalmente bastante menos).
#
#  Uso (en login node del BSC, desde la raíz del repo):
#     bash slurm/launch_distill_v6.sh            # modo SAFE (espera a gen OK)
#     bash slurm/launch_distill_v6.sh --async    # encadenado con afterok
#
#  Overrides útiles vía env (antes de `bash …`):
#     DISTILL_CONFIG=configs/distill_v6.yaml
#     TEACHER_MAX_SAMPLES=40        # smoke test de generación
#     SFT_MAX_SAMPLES=200           # smoke test de SFT
#     SKIP_PERF_BENCH=1             # solo quality en posteval
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f env/setup_env.sh ]]; then
    echo "ERROR: env/setup_env.sh not found"
    exit 1
fi

echo "Bootstrapping environment for launcher preflight..."
if ! source env/setup_env.sh; then
    echo "WARN: env/setup_env.sh returned non-zero on login node; continuing with current shell environment"
fi

WAIT_STAGE1=1
if [[ "${1:-}" == "--async" ]]; then
    WAIT_STAGE1=0
fi

DISTILL_CONFIG="${DISTILL_CONFIG:-configs/distill_v6.yaml}"

echo "══════════════════════════════════════════════════════════"
echo "  KD pipeline — Exp. 6 (Pure-Distillation · Max Envelope, Sprint 1)"
echo "══════════════════════════════════════════════════════════"
echo "  UTC time:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Git HEAD:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "  Config:     ${DISTILL_CONFIG}"
echo "  Jobs:       gen-v6 → train-1.5b-v6 → posteval-1.5b-v6"
if [[ ${WAIT_STAGE1} -eq 1 ]]; then
    echo "  Mode:       SAFE (wait for distill-gen-v6 success before submitting train/posteval)"
else
    echo "  Mode:       ASYNC (full dependency chain submission)"
fi
echo "══════════════════════════════════════════════════════════"
echo ""

echo "Running preflight checks (step 1)..."
python -m distill.preflight --step 1 --simulate-offline --config "${DISTILL_CONFIG}" || {
    echo "WARN: preflight step 1 on login node failed (ok if offline); review manually before async mode";
}

export DISTILL_CONFIG

if [[ ${WAIT_STAGE1} -eq 1 ]]; then
    JOB_DG=$(sbatch --parsable --wait slurm/distill_generate_v6.sbatch)
    echo "  [1/3] distill-gen-v6   : Job ${JOB_DG}  [20h]  (completed OK)"

    JOB_T1=$(sbatch --parsable slurm/distill_train_1.5b_v6.sbatch)
    echo "  [2/3] distill-1.5b-v6  : Job ${JOB_T1}  [16h]"
else
    JOB_DG=$(sbatch --parsable slurm/distill_generate_v6.sbatch)
    echo "  [1/3] distill-gen-v6   : Job ${JOB_DG}  [20h]  (immediate)"

    JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/distill_train_1.5b_v6.sbatch)
    echo "  [2/3] distill-1.5b-v6  : Job ${JOB_T1}  [16h]  (after ${JOB_DG})"
fi

JOB_P1=$(sbatch --parsable --dependency=afterok:${JOB_T1} slurm/posteval_1.5b_v6.sbatch)
echo "  [3/3] posteval-1.5b-v6 : Job ${JOB_P1}  [12h]  (after ${JOB_T1})"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  3 jobs submitted."
echo ""
echo "  ${JOB_DG} distill-gen-v6"
echo "    └── ${JOB_T1} distill-1.5b-v6 → ${JOB_P1} posteval-1.5b-v6"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:"
echo "    tail -f logs/distill-gen-v6-${JOB_DG}.out"
echo "    tail -f logs/distill-1.5b-v6-${JOB_T1}.out"
echo "    tail -f logs/posteval-1.5b-v6-${JOB_P1}.out"
if [[ ${WAIT_STAGE1} -eq 1 ]]; then
    echo "  Note:     SAFE mode used; distill-gen-v6 already validated before stage 2/3 submit"
else
    echo "  Note:     ASYNC mode used; if distill-gen-v6 fails, downstream jobs stay unsatisfied"
fi
echo "══════════════════════════════════════════════════════════"
