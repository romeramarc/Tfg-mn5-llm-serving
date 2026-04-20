# Predictors Consolidation Runbook (Strict, No Redesign)

## Objective

Consolidate offline predictors with strict methodological controls:

- No conceptual redesign.
- No online router implementation in this phase.
- No unnecessary deep-learning additions.
- Explicit leakage controls and auditable artifacts.

## Inputs Required

1. A canonical multi-model trace file produced by:

```bash
python -m predictors.builders.build_real_trace_pool \
  --quality-root results/quality \
  --online-root results/online \
  --throughput-root results/throughput \
  --logs-dir logs \
  --output-trace results/predictors/traces/<trace_tag>.jsonl \
  --output-report results/predictors/traces/<trace_tag>_report.json
```

2. Existing Python environment with dependencies from `requirements.txt`.

## One-Command Consolidation (Manifest-Driven)

```bash
python scripts/predictors_hardened_pipeline.py \
  --project-root . \
  --manifest configs/predictors_final_rerun_manifest.json \
  --tag <run_tag>
```

Optional overrides:

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

## What This Produces

Under `results/pred_reruns/<run_tag>/`:

- `datasets/quality_ex_ante_dataset.*`
- `datasets/quality_post_hoc_dataset.*`
- `datasets/service_cost_dataset.*`
- `models/<predictor_alias>-<family_alias>-<timestamp>/...`
- `reports/trace_preflight.json`
- `reports/trace_preflight.md`
- `reports/dataset_contract_quality_ex_ante.json`
- `reports/dataset_contract_quality_post_hoc.json`
- `reports/dataset_contract_service_cost.json`
- `reports/winner_bundles.json`
- `reports/winner_bundles.md`
- `reports/feature_audit.csv`
- `reports/feature_audit.md`
- `reports/feature_audit.json`
- `reports/smoke_inference.json`
- `reports/final_rerun_report.json`
- `reports/final_rerun_report.md`
- `reports/pipeline_summary.json`
- `manifest/resolved_manifest.json`

## Mandatory Validation Checklist

1. `trace_preflight.json` passes all checks:
  - no duplicate `(query_id, benchmark, model_tier)` rows,
  - expected models/benchmarks present,
  - minimum benchmark coverage reached,
  - alignment ratio above configured threshold.
2. Every dataset-contract report passes (`dataset_contract_*.json`) with exact target and feature-set match.
3. `feature_audit.csv` has no `drop` or `review` actions in strict mode.
4. `feature_audit.json` has no `high_correlation_warnings` when strict fail-gate is enabled.
5. `winner_bundles.json` includes winners for all three required predictors.
6. `smoke_inference.json` contains scored rows without runtime errors.
7. Model metrics include:
   - Global metrics
   - `by_benchmark`
   - `by_model`
   - `by_benchmark_model`
   - Calibration bins and ECE for classifiers
   - Feature importance CSV per model run

## When New 1.5B Arrives

Only update data/manifests and rerun. Keep methodology unchanged.

1. Re-run quality/online/throughput experiments with the new 1.5B baseline.
2. Rebuild trace pool (`build_real_trace_pool`).
3. Update `configs/predictors_final_rerun_manifest.json` inputs (trace pattern/model registry values if needed).
4. Run `scripts/predictors_hardened_pipeline.py` with a new `--tag`.
5. If preflight fails on duplicates, regenerate/clean trace input before rerunning.
6. Compare current and previous run using `final_rerun_report.json` and `pipeline_summary.json`.
7. If performance shifts materially, inspect per-group metrics before selecting deployment candidates.

## What Must Not Change in Consolidation Phase

- Do not introduce router online logic in these scripts.
- Do not relax strict cost feature policy for official comparisons.
- Do not add post-hoc target proxies into cost predictor features.
- Do not change split logic (query-level deterministic split) without explicit methodological approval.
