#!/usr/bin/env bash
# Same pipeline as slurm/retrain_predictors_model_tier.sbatch, for laptop/WSL.
# Requires: results/predictors/traces/iter2_real_multimodel_trace.jsonl (copy from BSC).
set -euo pipefail

cd "$(dirname "$0")/.."
TRACE_PATTERN="${TRACE_PATTERN:-results/predictors/traces/iter2_real_multimodel_trace.jsonl}"
DATASET_DIR="${DATASET_DIR:-results/predictors_model_tier/datasets}"

if [[ ! -f "${TRACE_PATTERN}" ]]; then
  echo "ERROR: missing trace file: ${TRACE_PATTERN}"
  echo "Copy from BSC, e.g.:"
  echo "  scp bsc381408@glogin4.bsc.es:/gpfs/scratch/.../Tfg-mn5-llm-serving/${TRACE_PATTERN} ."
  exit 1
fi

mkdir -p "${DATASET_DIR}" results/predictors_model_tier/phase_a results/predictors_model_tier/phase_b logs

python -m predictors.builders.build_ex_ante_dataset \
  --input "${TRACE_PATTERN}" \
  --output-dir "${DATASET_DIR}" \
  --dataset-name quality_ex_ante_model_tier

python -m predictors.builders.build_post_hoc_dataset \
  --input "${TRACE_PATTERN}" \
  --output-dir "${DATASET_DIR}" \
  --dataset-name quality_post_hoc_model_tier

python -m predictors.builders.build_cost_dataset \
  --input "${TRACE_PATTERN}" \
  --output-dir "${DATASET_DIR}" \
  --dataset-name service_cost_model_tier \
  --feature-policy strict_ex_ante \
  --cost-mode latency_ms

python scripts/refine_predictors.py --config configs/refine_phase_a_model_tier.yaml
python scripts/refine_predictors.py --config configs/refine_phase_b_model_tier.yaml

python scripts/write_routing_config_from_refinements.py \
  --base-config configs/routing_eval_holdout_v2.yaml \
  --phase-a-selection results/predictors_model_tier/phase_a/REFINEMENT_SELECTION.json \
  --phase-b-selection results/predictors_model_tier/phase_b/REFINEMENT_SELECTION.json \
  --output-config configs/routing_eval_holdout_v2_retrained.yaml \
  --project-root .

echo "Done. Config: configs/routing_eval_holdout_v2_retrained.yaml"
echo "Preflight: python scripts/preflight_eval_holdout.py --config configs/routing_eval_holdout_v2_retrained.yaml"
