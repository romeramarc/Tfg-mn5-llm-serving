#!/usr/bin/env bash
# ------------------------------------------------------------------
# Lanzamiento definitivo holdout v2 en MN5 (glogin4, dentro de tmux/screen).
#
# Qué hace la cadena (chain_holdout_definitivo.sbatch):
#   1) Cancela evals/servidores viejos si los indicas (manual, abajo).
#   2) Barrido λ realista con 1.5B destilado (routing_real: 5 sysE + 4 sysC).
#   3) sysE_l1e4_no_distill con 1.5B base HF (mismo λ, predictores actuales).
#
# Opción B (cadena larga capture→train→retrain→holdout) es independiente:
#   bash slurm/submit_option_b_base.sh
#   (no la mezcles con esta cadena: compiten por student_small en GPU).
# ------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "=== Jobs actuales ==="
squeue -u "${USER}" || true
echo ""
echo "Para CANCELAR todo lo tuyo en MN5 (revisa la lista arriba):"
echo "  scancel -u \${USER}"
echo ""
read -r -p "¿Has cancelado los jobs viejos (Option A/B, holdout v2 viejo, etc.)? [y/N] " ans
if [[ "${ans}" != [yY] ]]; then
  echo "Abortado. Cancela primero: scancel -u \${USER}"
  exit 1
fi

echo "=== Sync repo (git pull en scratch) ==="
git pull || echo "[WARN] git pull failed — asegúrate de tener routing_real + chain en scratch"

python scripts/preflight_eval_holdout.py --config configs/routing_eval_holdout_v2_routing_real.yaml
python scripts/preflight_eval_holdout.py --config configs/routing_eval_holdout_v2_no_distill.yaml

J=$(sbatch --parsable --export=ALL,PROJECT_DIR="${PROJECT_DIR}" slurm/chain_holdout_definitivo.sbatch)

echo ""
echo "Cadena definitiva enviada: job ${J}"
echo "  tail -f logs/chain-holdout-def-${J}.out"
echo "  squeue -u \${USER}"
echo ""
echo "Solo barrido routing (sin cadena, manual):"
echo "  EVAL_CONFIG=configs/routing_eval_holdout_v2_routing_real.yaml \\"
echo "  PROMPT_POOL=results/routing_eval_holdout/prompt_pool.jsonl \\"
echo "  bash slurm/launch_eval_holdout.sh all"
