#!/usr/bin/env bash
# Fase 2 holdout: 1.5B base (student_small con models_no_distill.yaml)
#   - sysB_only_1p5b      (always_student_small con 1.5B instruct, baseline suelo)
#   - sysE_l1e4_no_distill (routing+cascada λ=1e-4, mismo punto que sysE_l1e4 destilado)
#
# El sysE_l1e4 DESTILADO no se relanza: está en routing_eval_holdout_v2_routing_real/.
#
# IMPORTANTE: este script borra results/routing/endpoints/{role}.url para forzar
# que los servidores nuevos publiquen sus endpoints reales (no reutilizar los
# del run destilado anterior, que servían el modelo SFT v6 y daban 404 con base).
#
# Uso (login MN5, tras cancelar servidores destilados huérfanos):
#   cd $PROJECT_DIR && source env/setup_env.sh
#   export PROJECT_DIR=$PWD
#   bash slurm/submit_holdout_phase2.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving}"
cd "${PROJECT_DIR}"
source env/setup_env.sh
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_CONFIG="${EVAL_CONFIG:-configs/routing_eval_holdout_v2_phase2.yaml}"
PROMPT_POOL="${PROMPT_POOL:-results/routing_eval_holdout/prompt_pool.jsonl}"
MODELS_NO_DISTILL="${MODELS_NO_DISTILL:-configs/models_no_distill.yaml}"

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: missing ${EVAL_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${MODELS_NO_DISTILL}" ]]; then
  echo "ERROR: missing ${MODELS_NO_DISTILL} (1.5B base en student_small)" >&2
  exit 1
fi

# Comprobar que el eval YAML está pinned a models_no_distill.yaml para el cliente.
# Si falta, el cliente cae al default configs/models.yaml (destilado) y vuelve a fallar.
if ! grep -q 'models_no_distill\.yaml' "${EVAL_CONFIG}"; then
  echo "ERROR: ${EVAL_CONFIG} no referencia configs/models_no_distill.yaml" >&2
  echo "       (el cliente del eval pediría el modelo destilado y daría 404)" >&2
  exit 1
fi

# Limpiar endpoint files antiguos para asegurar que los nuevos servidores publican fresh.
echo "=== Limpiando endpoints antiguos ==="
for role in teacher student_mid student_q3b student_small; do
  rm -f "results/routing/endpoints/${role}.url" "results/routing/endpoints/${role}.gpu" || true
done

# Preflight estático (sin ping; los endpoints aún no existen).
python scripts/preflight_holdout_phase2.py --config "${EVAL_CONFIG}" || {
  echo "WARN: preflight estático con avisos (endpoints sin arrancar; normal)" >&2
}

echo ""
echo "=== Fase 2: servidores (student_small = 1.5B instruct) ==="
for role in teacher student_mid student_q3b; do
  jid=$(sbatch --parsable \
    --job-name="vllm-${role}-p2" \
    --export=ALL,ROLE="${role}",PROJECT_DIR="${PROJECT_DIR}" \
    slurm/server_role_phase2.sbatch)
  echo "  ${role}: ${jid}"
done

jid_small=$(sbatch --parsable \
  --job-name="vllm-student_small-p2" \
  --export=ALL,ROLE=student_small,MODELS_CONFIG="${MODELS_NO_DISTILL}",PROJECT_DIR="${PROJECT_DIR}" \
  slurm/server_role_phase2.sbatch)
echo "  student_small (no_distill): ${jid_small}"

echo ""
echo "Cuando los .url estén publicados, ejecuta el preflight con ping:"
echo "  python scripts/preflight_holdout_phase2.py --config ${EVAL_CONFIG} --ping"
echo "Si OK, lanzar los evals:"
echo "  EVAL_CONFIG=${EVAL_CONFIG} PROMPT_POOL=${PROMPT_POOL} PROJECT_DIR=${PROJECT_DIR} \\"
echo "    bash slurm/launch_eval_holdout.sh clients"
echo ""
echo "Comparar después con:"
echo "  Destilado: results/routing_eval_holdout_v2_routing_real/sysE_l1e4-*"
echo "  Base:      results/routing_eval_holdout_v2_phase2/sysE_l1e4_no_distill-*"
echo "  Baseline:  results/routing_eval_holdout_v2_phase2/sysB_only_1p5b-*"
