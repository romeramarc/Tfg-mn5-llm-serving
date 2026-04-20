# Offline Predictors (Consolidated)

This package implements the offline predictor stack used for the cascade/routing preparation stage.

## Scope

Current predictors:

- `quality_ex_ante` (classification)
- `quality_post_hoc` (classification)
- `service_cost` (regression)

The current phase is intentionally offline and tabular-only. No conceptual redesign and no online router deployment logic are introduced here.

## Package Layout

- `predictors/schemas.py`: canonical trace dataclasses and JSONL loading helpers.
- `predictors/trace_logging.py`: adapters from legacy quality outputs to canonical traces.
- `predictors/builders/build_real_trace_pool.py`: multi-model real trace pool with telemetry alignment.
- `predictors/builders/build_ex_ante_dataset.py`: quality ex-ante dataset builder.
- `predictors/builders/build_post_hoc_dataset.py`: quality post-hoc dataset builder.
- `predictors/builders/build_cost_dataset.py`: service-cost dataset builder with explicit feature policy.
- `predictors/feature_registry.py`: feature-level availability and leakage rules.
- `predictors/audit/feature_audit.py`: auditable feature table generation and leakage warnings.
- `predictors/training/*.py`: train/eval/select scripts for the three predictors.
- `predictors/inference.py`: bundle loading and inference API for offline smoke checks.
- `predictors/smoke_inference.py`: end-to-end smoke inference command.

## Leakage Guardrails

### Important fix included

`build_real_trace_pool.py` no longer derives `resources.gpu_seconds` from request `latency_ms`. That derivation leaked the cost target into features.

### Cost predictor policy

Use `--feature-policy strict_ex_ante` in `build_cost_dataset.py` to keep only request-time available features. This policy excludes direct/near-direct post-hoc target proxies.

## End-to-End Command

From repository root:

```bash
python scripts/predictors_hardened_pipeline.py \
  --project-root . \
  --manifest configs/predictors_final_rerun_manifest.json \
  --tag final_rerun
```

Optional overrides (when needed):

```bash
python scripts/predictors_hardened_pipeline.py \
  --project-root . \
  --manifest configs/predictors_final_rerun_manifest.json \
  --trace-pattern results/predictors/traces/<trace_tag>.jsonl \
  --tag <run_tag> \
  --output-root results/pred_reruns \
  --cost-policy strict_ex_ante \
  --cost-mode latency_ms \
  --smoke-max-rows 30
```

This generates:

- Run-scoped artifacts in `results/pred_reruns/<run_tag>/`
- Datasets in `results/pred_reruns/<run_tag>/datasets/`
- Trained model runs in `results/pred_reruns/<run_tag>/models/` (short model-dir aliases: `qea`, `qph`, `sc`)
- Selection outputs in `results/pred_reruns/<run_tag>/reports/winner_bundles.json` and `winner_bundles.md`
- Feature audit outputs in `results/pred_reruns/<run_tag>/reports/feature_audit.*`
- Trace preflight in `results/pred_reruns/<run_tag>/reports/trace_preflight.*`
- Dataset contract checks in `results/pred_reruns/<run_tag>/reports/dataset_contract_*.json|md`
- Smoke inference output in `results/pred_reruns/<run_tag>/reports/smoke_inference.json`
- Final report in `results/pred_reruns/<run_tag>/reports/final_rerun_report.json|md`
- Pipeline summary in `results/pred_reruns/<run_tag>/reports/pipeline_summary.json`
- Resolved manifest snapshot in `results/pred_reruns/<run_tag>/manifest/resolved_manifest.json`

Important validation behavior:

- The run fails fast if trace preflight checks fail (for example, duplicate rows or missing expected model coverage).
- The run fails fast if dataset contracts, feature-audit gates, or required-winner checks fail.

## Manual Command Sequence

Use this only for debugging individual stages; official reruns should use the manifest-driven command above.

1. Build datasets

```bash
python -m predictors.builders.build_ex_ante_dataset --input <trace.jsonl> --output-dir <run_root>/datasets --dataset-name quality_ex_ante_dataset
python -m predictors.builders.build_post_hoc_dataset --input <trace.jsonl> --output-dir <run_root>/datasets --dataset-name quality_post_hoc_dataset
python -m predictors.builders.build_cost_dataset --input <trace.jsonl> --output-dir <run_root>/datasets --dataset-name service_cost_dataset --feature-policy strict_ex_ante --cost-mode latency_ms
```

2. Train families

```bash
python -m predictors.training.train_ex_ante --dataset <run_root>/datasets/quality_ex_ante_dataset.jsonl --meta <run_root>/datasets/quality_ex_ante_dataset_meta.json --family linear --output-root <run_root>/models
python -m predictors.training.train_ex_ante --dataset <run_root>/datasets/quality_ex_ante_dataset.jsonl --meta <run_root>/datasets/quality_ex_ante_dataset_meta.json --family random_forest --output-root <run_root>/models
python -m predictors.training.train_ex_ante --dataset <run_root>/datasets/quality_ex_ante_dataset.jsonl --meta <run_root>/datasets/quality_ex_ante_dataset_meta.json --family gradient_boosting --output-root <run_root>/models

python -m predictors.training.train_post_hoc --dataset <run_root>/datasets/quality_post_hoc_dataset.jsonl --meta <run_root>/datasets/quality_post_hoc_dataset_meta.json --family linear --output-root <run_root>/models
python -m predictors.training.train_post_hoc --dataset <run_root>/datasets/quality_post_hoc_dataset.jsonl --meta <run_root>/datasets/quality_post_hoc_dataset_meta.json --family random_forest --output-root <run_root>/models
python -m predictors.training.train_post_hoc --dataset <run_root>/datasets/quality_post_hoc_dataset.jsonl --meta <run_root>/datasets/quality_post_hoc_dataset_meta.json --family gradient_boosting --output-root <run_root>/models

python -m predictors.training.train_cost --dataset <run_root>/datasets/service_cost_dataset.jsonl --meta <run_root>/datasets/service_cost_dataset_meta.json --family linear --output-root <run_root>/models
python -m predictors.training.train_cost --dataset <run_root>/datasets/service_cost_dataset.jsonl --meta <run_root>/datasets/service_cost_dataset_meta.json --family random_forest --output-root <run_root>/models
python -m predictors.training.train_cost --dataset <run_root>/datasets/service_cost_dataset.jsonl --meta <run_root>/datasets/service_cost_dataset_meta.json --family gradient_boosting --output-root <run_root>/models
```

3. Select winners and audit

```bash
python -m predictors.training.select_best_bundles --models-root <run_root>/models --output-json <run_root>/reports/winner_bundles.json --output-md <run_root>/reports/winner_bundles.md
python -m predictors.audit.feature_audit --datasets-dir <run_root>/datasets --cost-policy strict_ex_ante --dataset-name quality_ex_ante=quality_ex_ante_dataset --dataset-name quality_post_hoc=quality_post_hoc_dataset --dataset-name service_cost=service_cost_dataset --output-csv <run_root>/reports/feature_audit.csv --output-md <run_root>/reports/feature_audit.md --output-json <run_root>/reports/feature_audit.json
```

4. Smoke inference

```bash
python -m predictors.smoke_inference --trace-pattern <trace.jsonl> --selection-json <run_root>/reports/winner_bundles.json --output-json <run_root>/reports/smoke_inference.json --max-rows 30
```
