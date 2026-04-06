#!/bin/bash
# =============================================================================
#  EXPERIMENT 4 — Distillation pipeline (quality-focused iteration)
# =============================================================================
#  Lanzar también con:  bash slurm/launch.sh   (mismo contenido)
#
#  Prioridad (profe): GSM8K — hay margen respecto al teacher (92,41 %)
#  en los students; esta receta fuerza más señal GSM8K en el TRAIN (oversample)
#  y SFT más conservador en 7B para reducir regresión. No garantiza cifras,
#  pero sí apunta el entrenamiento a ese objetivo.
#
#  model_overrides en configs/distill.yaml:
#   7B:   lr=3e-5, LoRA r=48, oversample gsm8k:2 math:1
#   1.5B: 5 ep, LoRA r=96, oversample gsm8k:2 math:2
#
# Dependency graph (same SLURM jobs as v2):
#
#   distill_gen (8h) ─┬─ train_7b  (12h) ─── posteval_7b  (18h)
#                     └─ train_1.5b(14h) ─── posteval_1.5b(12h)
#
# Critical path: gen → train_7b → posteval_7b  ≈  38h wall-clock
#
# BEFORE SUBMITTING:
#   1. git pull on BSC (this repo + scripts above)
#   2. Edit cd=... in each *.sbatch if your clone path differs
#   3. (Optional) regenerate teacher only if you changed prompts / generation
#
# AFTER JOBS FINISH:
#   python3 scripts/collect_all_results.py
#   python3 scripts/plot_results.py --from-summary results/summary_all_models.csv
#
# Usage (BSC login node, desde la raíz del repo):
#   bash slurm/launch.sh
#   # o:  bash slurm/launch_distill_v4.sh
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

echo "══════════════════════════════════════════════════════════"
echo "  KD pipeline — Exp. 4 (prioridad GSM8K + calidad, feedback profe)"
echo "══════════════════════════════════════════════════════════"
echo "  UTC time:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Git HEAD:   $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo "  Config:     configs/distill.yaml → model_overrides (7B/1.5B)"
echo "  Jobs:       gen → train 7B + 1.5B → posteval cada uno"
if [[ ${WAIT_STAGE1} -eq 1 ]]; then
	echo "  Mode:       SAFE (wait for distill-gen success before submitting train/posteval)"
else
	echo "  Mode:       ASYNC (full dependency chain submission)"
fi
echo "══════════════════════════════════════════════════════════"
echo ""

echo "Running preflight checks (step 1)..."
python -m distill.preflight --step 1 --simulate-offline --config configs/distill.yaml

if [[ ${WAIT_STAGE1} -eq 1 ]]; then
		JOB_DG=$(sbatch --parsable --wait slurm/distill_generate.sbatch)
		echo "  [1/5] distill-gen   : Job ${JOB_DG}  [16h]  (completed OK)"

		JOB_T7=$(sbatch --parsable slurm/distill_train_7b.sbatch)
		echo "  [2/5] train-7b      : Job ${JOB_T7}  [12h]"

		JOB_T1=$(sbatch --parsable slurm/distill_train_1.5b.sbatch)
		echo "  [3/5] train-1.5b    : Job ${JOB_T1}  [14h]"
else
		JOB_DG=$(sbatch --parsable slurm/distill_generate.sbatch)
		echo "  [1/5] distill-gen   : Job ${JOB_DG}  [16h]  (immediate)"

		JOB_T7=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/distill_train_7b.sbatch)
		echo "  [2/5] train-7b      : Job ${JOB_T7}  [12h]  (after ${JOB_DG})"

		JOB_T1=$(sbatch --parsable --dependency=afterok:${JOB_DG} slurm/distill_train_1.5b.sbatch)
		echo "  [3/5] train-1.5b    : Job ${JOB_T1}  [14h]  (after ${JOB_DG})"
fi

JOB_P7=$(sbatch --parsable --dependency=afterok:${JOB_T7} slurm/posteval_7b.sbatch)
echo "  [4/5] posteval-7b   : Job ${JOB_P7}  [18h]  (after ${JOB_T7})"

JOB_P1=$(sbatch --parsable --dependency=afterok:${JOB_T1} slurm/posteval_1.5b.sbatch)
echo "  [5/5] posteval-1.5b : Job ${JOB_P1}  [12h]  (after ${JOB_T1})"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  5 jobs submitted."
echo ""
echo "  ${JOB_DG} distill-gen"
echo "    ├── ${JOB_T7} distill-7b → ${JOB_P7} posteval-7b"
echo "    └── ${JOB_T1} distill-1.5b → ${JOB_P1} posteval-1.5b"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:     tail -f logs/distill-gen-${JOB_DG}.out"
if [[ ${WAIT_STAGE1} -eq 1 ]]; then
	echo "  Note:     SAFE mode used; distill-gen already validated before stage 2/3 submit"
else
	echo "  Note:     ASYNC mode used; if distill-gen fails, downstream jobs stay unsatisfied"
fi
echo "══════════════════════════════════════════════════════════"
