from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_MANIFEST_PATH = "configs/predictors_final_rerun_manifest.json"
PREDICTOR_IDS: Tuple[str, ...] = (
    "quality_ex_ante",
    "quality_post_hoc",
    "service_cost",
)
TRAIN_MODULE_BY_PREDICTOR = {
    "quality_ex_ante": "predictors.training.train_ex_ante",
    "quality_post_hoc": "predictors.training.train_post_hoc",
    "service_cost": "predictors.training.train_cost",
}


def run_cmd(cmd: List[str], *, cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_windows_path_lengths(paths: Sequence[Path], *, limit: int = 260) -> None:
    if os.name != "nt":
        return

    too_long = [(str(path), len(str(path))) for path in paths if len(str(path)) >= limit]
    if not too_long:
        return

    details = "\n".join([f"  - [{length}] {path}" for path, length in too_long[:8]])
    raise RuntimeError(
        "Windows path-length limit exceeded for planned artifacts. "
        "Use a shorter --tag or --output-root.\n"
        f"Failing paths:\n{details}"
    )


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _normalize_patterns(value: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_list(value: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _assert_manifest_shape(manifest: Dict[str, Any]) -> None:
    required_top = [
        "manifest_version",
        "inputs",
        "pipeline",
        "predictor_contracts",
        "outputs",
        "validations",
    ]
    for key in required_top:
        if key not in manifest:
            raise ValueError(f"Manifest missing required key: {key}")

    for predictor_id in PREDICTOR_IDS:
        if predictor_id not in manifest["predictor_contracts"]:
            raise ValueError(f"Manifest missing predictor contract: {predictor_id}")


def _resolve_runtime_config(
    *,
    project_root: Path,
    manifest: Dict[str, Any],
    trace_patterns_override: Sequence[str],
    tag_override: str | None,
    output_root_override: str | None,
    cost_policy_override: str | None,
    cost_mode_override: str | None,
    smoke_rows_override: int | None,
) -> Dict[str, Any]:
    inputs = manifest["inputs"]
    pipeline = manifest["pipeline"]
    outputs = manifest["outputs"]

    trace_patterns = _normalize_patterns(trace_patterns_override)
    if not trace_patterns:
        trace_patterns = _normalize_patterns(inputs.get("trace_patterns", []))
    if not trace_patterns:
        raise ValueError("No trace patterns provided via CLI or manifest")

    tag = str(tag_override or pipeline.get("default_tag") or "predictors_final_rerun").strip()
    if not tag:
        raise ValueError("Resolved run tag is empty")
    if "/" in tag or "\\" in tag:
        raise ValueError("Run tag must not contain path separators")
    cost_policy = str(cost_policy_override or pipeline.get("cost_policy") or "strict_ex_ante")
    cost_mode = str(cost_mode_override or pipeline.get("cost_mode") or "latency_ms")
    smoke_max_rows = int(smoke_rows_override if smoke_rows_override is not None else pipeline.get("smoke_max_rows", 30))

    output_root_str = str(output_root_override or outputs.get("base_output_root") or "results/predictors/reruns")
    output_root = _resolve_path(project_root, output_root_str)
    run_root = output_root / tag

    run_dirs = {
        "run_root": run_root,
        "datasets": run_root / str(outputs.get("datasets_dir_name", "datasets")),
        "models": run_root / str(outputs.get("models_dir_name", "models")),
        "reports": run_root / str(outputs.get("reports_dir_name", "reports")),
        "manifest": run_root / str(outputs.get("manifest_dir_name", "manifest")),
    }

    expected_models = _normalize_list(inputs.get("expected_model_tiers", []))
    expected_benchmarks = _normalize_list(inputs.get("expected_benchmarks", []))
    min_rows_per_benchmark = {
        str(k): int(v)
        for k, v in (inputs.get("min_rows_per_benchmark") or {}).items()
    }

    predictor_families = {
        predictor_id: _normalize_list((pipeline.get("predictor_families") or {}).get(predictor_id, []))
        for predictor_id in PREDICTOR_IDS
    }
    for predictor_id in PREDICTOR_IDS:
        if not predictor_families[predictor_id]:
            raise ValueError(f"No candidate families configured for predictor: {predictor_id}")

    selection_cfg = pipeline.get("selection") or {}
    required_predictors = _normalize_list(selection_cfg.get("required_predictors", list(PREDICTOR_IDS)))

    runtime = {
        "manifest_version": str(manifest.get("manifest_version")),
        "tag": tag,
        "trace_patterns": trace_patterns,
        "trace_model_field": str(inputs.get("trace_model_field", "model_tier")),
        "expected_models": expected_models,
        "expected_benchmarks": expected_benchmarks,
        "min_rows_per_benchmark": min_rows_per_benchmark,
        "min_alignment_ratio": float(inputs.get("min_alignment_ratio", 1.0)),
        "predictor_families": predictor_families,
        "cost_policy": cost_policy,
        "cost_mode": cost_mode,
        "selection": {
            "required_predictors": required_predictors,
            "class_maximize": _normalize_list((selection_cfg.get("classification") or {}).get("maximize", [])),
            "class_minimize": _normalize_list((selection_cfg.get("classification") or {}).get("minimize", [])),
            "reg_minimize": _normalize_list((selection_cfg.get("regression") or {}).get("minimize", [])),
            "reg_maximize": _normalize_list((selection_cfg.get("regression") or {}).get("maximize", [])),
        },
        "smoke_max_rows": smoke_max_rows,
        "run_dirs": {name: str(path) for name, path in run_dirs.items()},
        "validations": manifest.get("validations") or {},
        "predictor_contracts": manifest.get("predictor_contracts") or {},
    }
    return runtime


def _resolve_trace_files(project_root: Path, trace_patterns: Sequence[str]) -> Tuple[List[Path], List[str]]:
    files: List[Path] = []
    missing_patterns: List[str] = []

    for pattern in trace_patterns:
        pattern_path = _resolve_path(project_root, pattern)
        matches = sorted(Path(path) for path in glob.glob(str(pattern_path)))
        if not matches:
            missing_patterns.append(pattern)
            continue
        files.extend(matches)

    unique_files: List[Path] = []
    seen = set()
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(path)
    return unique_files, missing_patterns


def run_trace_preflight(
    *,
    project_root: Path,
    trace_patterns: Sequence[str],
    trace_model_field: str,
    expected_models: Sequence[str],
    expected_benchmarks: Sequence[str],
    min_rows_per_benchmark: Dict[str, int],
    min_alignment_ratio: float,
) -> Dict[str, Any]:
    trace_files, missing_patterns = _resolve_trace_files(project_root, trace_patterns)

    expected_model_set = set(expected_models)
    expected_benchmark_set = set(expected_benchmarks)

    rows_total = 0
    parse_errors = 0
    benchmark_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    combo_counts: Counter[Tuple[str, str, str]] = Counter()
    query_models: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    for path in trace_files:
        with path.open("r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                rows_total += 1
                benchmark = str(row.get("benchmark") or "")
                query_id = str(row.get("query_id") or "")
                model_value = row.get(trace_model_field)
                if model_value is None:
                    model_value = row.get("model_tier") if trace_model_field != "model_tier" else row.get("model_name")
                model = str(model_value or "")

                benchmark_counts[benchmark] += 1
                model_counts[model] += 1
                combo_counts[(benchmark, query_id, model)] += 1
                query_models[(benchmark, query_id)].add(model)

    bench_query_counts: Counter[str] = Counter()
    bench_aligned_counts: Counter[str] = Counter()
    total_query_groups = 0
    aligned_query_groups = 0

    for (benchmark, _query_id), models in query_models.items():
        total_query_groups += 1
        bench_query_counts[benchmark] += 1
        if models == expected_model_set:
            aligned_query_groups += 1
            bench_aligned_counts[benchmark] += 1

    alignment_ratio = (aligned_query_groups / total_query_groups) if total_query_groups > 0 else 0.0
    duplicate_rows = sum(count - 1 for count in combo_counts.values() if count > 1)

    missing_expected_models = sorted(expected_model_set - set(model_counts.keys()))
    missing_expected_benchmarks = sorted(expected_benchmark_set - set(benchmark_counts.keys()))

    benchmark_checks = {}
    for benchmark in expected_benchmarks:
        observed_rows = int(benchmark_counts.get(benchmark, 0))
        min_required = int(min_rows_per_benchmark.get(benchmark, 1))
        bench_total_queries = int(bench_query_counts.get(benchmark, 0))
        bench_aligned_queries = int(bench_aligned_counts.get(benchmark, 0))
        bench_alignment_ratio = (
            float(bench_aligned_queries / bench_total_queries)
            if bench_total_queries > 0
            else 0.0
        )
        benchmark_checks[benchmark] = {
            "observed_rows": observed_rows,
            "min_required_rows": min_required,
            "rows_ok": observed_rows >= min_required,
            "query_groups": bench_total_queries,
            "aligned_query_groups": bench_aligned_queries,
            "alignment_ratio": bench_alignment_ratio,
            "alignment_ok": bench_alignment_ratio >= min_alignment_ratio,
        }

    checks: List[Dict[str, Any]] = []
    checks.append(
        {
            "id": "trace_patterns_resolved",
            "passed": len(missing_patterns) == 0 and len(trace_files) > 0,
            "details": {
                "trace_files_found": len(trace_files),
                "missing_patterns": missing_patterns,
            },
        }
    )
    checks.append(
        {
            "id": "json_parse_errors",
            "passed": parse_errors == 0,
            "details": {"parse_errors": parse_errors},
        }
    )
    checks.append(
        {
            "id": "expected_models_present",
            "passed": len(missing_expected_models) == 0,
            "details": {
                "missing_expected_models": missing_expected_models,
                "model_counts": dict(model_counts),
            },
        }
    )
    checks.append(
        {
            "id": "expected_benchmarks_present",
            "passed": len(missing_expected_benchmarks) == 0,
            "details": {
                "missing_expected_benchmarks": missing_expected_benchmarks,
                "benchmark_counts": dict(benchmark_counts),
            },
        }
    )
    checks.append(
        {
            "id": "no_duplicate_trace_rows",
            "passed": duplicate_rows == 0,
            "details": {"duplicate_rows": duplicate_rows},
        }
    )
    checks.append(
        {
            "id": "global_alignment_ratio",
            "passed": alignment_ratio >= min_alignment_ratio,
            "details": {
                "alignment_ratio": alignment_ratio,
                "min_alignment_ratio": min_alignment_ratio,
                "aligned_query_groups": aligned_query_groups,
                "total_query_groups": total_query_groups,
            },
        }
    )

    for benchmark, node in benchmark_checks.items():
        checks.append(
            {
                "id": f"benchmark_{benchmark}_rows",
                "passed": bool(node["rows_ok"]),
                "details": node,
            }
        )
        checks.append(
            {
                "id": f"benchmark_{benchmark}_alignment",
                "passed": bool(node["alignment_ok"]),
                "details": node,
            }
        )

    is_valid = all(bool(check["passed"]) for check in checks)
    return {
        "generated_at_utc": _now_utc_iso(),
        "trace_patterns": list(trace_patterns),
        "trace_files": [str(path) for path in trace_files],
        "rows_total": rows_total,
        "expected_models": list(expected_models),
        "expected_benchmarks": list(expected_benchmarks),
        "model_counts": dict(model_counts),
        "benchmark_counts": dict(benchmark_counts),
        "alignment_ratio": alignment_ratio,
        "min_alignment_ratio": min_alignment_ratio,
        "query_group_total": total_query_groups,
        "query_group_aligned": aligned_query_groups,
        "checks": checks,
        "is_valid": is_valid,
    }


def _render_preflight_md(preflight: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Trace Preflight")
    lines.append("")
    lines.append(f"- generated_at_utc: {preflight.get('generated_at_utc')}")
    lines.append(f"- rows_total: {preflight.get('rows_total')}")
    lines.append(f"- alignment_ratio: {preflight.get('alignment_ratio')}")
    lines.append(f"- min_alignment_ratio: {preflight.get('min_alignment_ratio')}")
    lines.append(f"- is_valid: {preflight.get('is_valid')}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for check in preflight.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- [{status}] {check.get('id')}")
        lines.append(f"  details: `{json.dumps(check.get('details', {}), ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


def _contract_for_predictor(
    *,
    predictor_id: str,
    contracts: Dict[str, Any],
    cost_policy: str,
) -> Dict[str, Any]:
    node = contracts.get(predictor_id) or {}
    if predictor_id != "service_cost":
        return {
            "target": str(node.get("target", "")),
            "features": [str(x) for x in (node.get("features") or [])],
        }

    features_by_policy = node.get("features_by_cost_policy") or {}
    selected_features = [str(x) for x in (features_by_policy.get(cost_policy) or [])]
    return {
        "target": str(node.get("target", "")),
        "features": selected_features,
    }


def validate_dataset_contract(
    *,
    dataset_meta_path: Path,
    predictor_id: str,
    expected_target: str,
    expected_features: Sequence[str],
) -> Dict[str, Any]:
    meta = _read_json(dataset_meta_path)
    observed_target = str(meta.get("target_column", ""))
    observed_features = [str(x) for x in (meta.get("feature_columns") or [])]

    expected_feature_set = set(expected_features)
    observed_feature_set = set(observed_features)
    missing_features = sorted(expected_feature_set - observed_feature_set)
    unexpected_features = sorted(observed_feature_set - expected_feature_set)

    checks = [
        {
            "id": "target_matches",
            "passed": observed_target == expected_target,
            "details": {
                "expected_target": expected_target,
                "observed_target": observed_target,
            },
        },
        {
            "id": "no_missing_features",
            "passed": len(missing_features) == 0,
            "details": {
                "missing_features": missing_features,
            },
        },
        {
            "id": "no_unexpected_features",
            "passed": len(unexpected_features) == 0,
            "details": {
                "unexpected_features": unexpected_features,
            },
        },
        {
            "id": "non_empty_dataset",
            "passed": int(meta.get("row_count", 0)) > 0,
            "details": {
                "row_count": int(meta.get("row_count", 0)),
            },
        },
    ]

    is_valid = all(bool(check["passed"]) for check in checks)
    return {
        "predictor_id": predictor_id,
        "dataset_meta_path": str(dataset_meta_path),
        "checks": checks,
        "is_valid": is_valid,
        "feature_count_expected": len(expected_features),
        "feature_count_observed": len(observed_features),
    }


def validate_audit_report(
    *,
    audit_payload: Dict[str, Any],
    validations_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fail_on_drop = bool(validations_cfg.get("fail_on_drop_features", True))
    fail_on_review = bool(validations_cfg.get("fail_on_review_features", True))
    fail_on_missing_expected = bool(validations_cfg.get("fail_on_missing_expected_features", True))
    fail_on_high_corr = bool(validations_cfg.get("fail_on_high_correlation_warnings", True))

    summary = audit_payload.get("summary") or {}
    checks: List[Dict[str, Any]] = []

    for predictor_id in PREDICTOR_IDS:
        node = summary.get(predictor_id) or {}
        action_counts = node.get("action_counts") or {}
        corr_warnings = node.get("high_correlation_warnings") or []

        drop_count = int(action_counts.get("drop", 0))
        review_count = int(action_counts.get("review", 0))
        missing_expected_count = int(action_counts.get("missing_expected", 0))
        corr_count = len(corr_warnings)

        checks.append(
            {
                "id": f"{predictor_id}_drop_features",
                "passed": (drop_count == 0) if fail_on_drop else True,
                "details": {"drop_count": drop_count},
            }
        )
        checks.append(
            {
                "id": f"{predictor_id}_review_features",
                "passed": (review_count == 0) if fail_on_review else True,
                "details": {"review_count": review_count},
            }
        )
        checks.append(
            {
                "id": f"{predictor_id}_missing_expected_features",
                "passed": (missing_expected_count == 0) if fail_on_missing_expected else True,
                "details": {"missing_expected_count": missing_expected_count},
            }
        )
        checks.append(
            {
                "id": f"{predictor_id}_high_correlation_warnings",
                "passed": (corr_count == 0) if fail_on_high_corr else True,
                "details": {"high_correlation_warning_count": corr_count},
            }
        )

    is_valid = all(bool(check["passed"]) for check in checks)
    return {
        "checks": checks,
        "is_valid": is_valid,
    }


def _validate_check_block(name: str, block: Dict[str, Any]) -> None:
    if block.get("is_valid"):
        return

    failed = [check for check in block.get("checks", []) if not check.get("passed")]
    formatted = [f"{check.get('id')}: {json.dumps(check.get('details', {}), ensure_ascii=False)}" for check in failed]
    detail = "\n".join(formatted)
    raise RuntimeError(f"Validation block '{name}' failed:\n{detail}")


def _render_final_report_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Predictors Final Rerun Report")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload.get('generated_at_utc')}")
    lines.append(f"- manifest_version: {payload.get('manifest_version')}")
    lines.append(f"- tag: {payload.get('tag')}")
    lines.append(f"- status: {payload.get('status')}")
    lines.append("")

    lines.append("## Validation Summary")
    lines.append("")
    for name, block in (payload.get("validation") or {}).items():
        status = "PASS" if block.get("is_valid") else "FAIL"
        lines.append(f"- {name}: {status}")
    lines.append("")

    lines.append("## Winners")
    lines.append("")
    winners = ((payload.get("selection") or {}).get("winners") or {})
    for predictor_id in PREDICTOR_IDS:
        winner = winners.get(predictor_id)
        if not isinstance(winner, dict):
            lines.append(f"- {predictor_id}: MISSING")
            continue
        lines.append(f"- {predictor_id}: {winner.get('model_family')} ({winner.get('model_dir')})")
        lines.append(f"  test_global: `{json.dumps(winner.get('test_global', {}), ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for key, value in (payload.get("artifacts") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def run_pipeline(
    *,
    project_root: Path,
    python_bin: str,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    run_dirs = {name: Path(path) for name, path in (runtime["run_dirs"] or {}).items()}
    for path in run_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    datasets_dir = run_dirs["datasets"]
    models_dir = run_dirs["models"]
    reports_dir = run_dirs["reports"]
    manifest_dir = run_dirs["manifest"]

    tag = str(runtime["tag"])
    cost_policy = str(runtime["cost_policy"])
    cost_mode = str(runtime["cost_mode"])
    trace_patterns = [str(x) for x in runtime["trace_patterns"]]

    preflight_json = reports_dir / "trace_preflight.json"
    preflight_md = reports_dir / "trace_preflight.md"

    ds_names = {
        "quality_ex_ante": "quality_ex_ante_dataset",
        "quality_post_hoc": "quality_post_hoc_dataset",
        "service_cost": "service_cost_dataset",
    }

    planned_paths: List[Path] = [
        preflight_json,
        preflight_md,
        reports_dir / "winner_bundles.json",
        reports_dir / "winner_bundles.md",
        reports_dir / "feature_audit.csv",
        reports_dir / "feature_audit.md",
        reports_dir / "feature_audit.json",
        reports_dir / "smoke_inference.json",
        reports_dir / "final_rerun_report.json",
        reports_dir / "final_rerun_report.md",
        reports_dir / "pipeline_summary.json",
        manifest_dir / "resolved_manifest.json",
    ]
    for predictor_id in PREDICTOR_IDS:
        planned_paths.append(reports_dir / f"dataset_contract_{predictor_id}.json")
        planned_paths.append(reports_dir / f"dataset_contract_{predictor_id}.md")
        planned_paths.append(datasets_dir / f"{ds_names[predictor_id]}.jsonl")
        planned_paths.append(datasets_dir / f"{ds_names[predictor_id]}_meta.json")
    _validate_windows_path_lengths(planned_paths)

    preflight = run_trace_preflight(
        project_root=project_root,
        trace_patterns=trace_patterns,
        trace_model_field=str(runtime["trace_model_field"]),
        expected_models=[str(x) for x in runtime["expected_models"]],
        expected_benchmarks=[str(x) for x in runtime["expected_benchmarks"]],
        min_rows_per_benchmark={str(k): int(v) for k, v in (runtime["min_rows_per_benchmark"] or {}).items()},
        min_alignment_ratio=float(runtime["min_alignment_ratio"]),
    )
    _write_json(preflight_json, preflight)
    _write_text(preflight_md, _render_preflight_md(preflight))
    _validate_check_block("trace_preflight", preflight)

    ex_cmd = [
        python_bin,
        "-m",
        "predictors.builders.build_ex_ante_dataset",
        "--output-dir",
        str(datasets_dir),
        "--dataset-name",
        ds_names["quality_ex_ante"],
    ]
    for pattern in trace_patterns:
        ex_cmd.extend(["--input", pattern])
    run_cmd(ex_cmd, cwd=project_root)

    ph_cmd = [
        python_bin,
        "-m",
        "predictors.builders.build_post_hoc_dataset",
        "--output-dir",
        str(datasets_dir),
        "--dataset-name",
        ds_names["quality_post_hoc"],
    ]
    for pattern in trace_patterns:
        ph_cmd.extend(["--input", pattern])
    run_cmd(ph_cmd, cwd=project_root)

    cost_cmd = [
        python_bin,
        "-m",
        "predictors.builders.build_cost_dataset",
        "--output-dir",
        str(datasets_dir),
        "--dataset-name",
        ds_names["service_cost"],
        "--feature-policy",
        cost_policy,
        "--cost-mode",
        cost_mode,
    ]
    for pattern in trace_patterns:
        cost_cmd.extend(["--input", pattern])
    run_cmd(cost_cmd, cwd=project_root)

    contract_reports: Dict[str, Dict[str, Any]] = {}
    predictor_contracts = runtime["predictor_contracts"] or {}
    for predictor_id in PREDICTOR_IDS:
        contract = _contract_for_predictor(
            predictor_id=predictor_id,
            contracts=predictor_contracts,
            cost_policy=cost_policy,
        )
        meta_path = datasets_dir / f"{ds_names[predictor_id]}_meta.json"
        report = validate_dataset_contract(
            dataset_meta_path=meta_path,
            predictor_id=predictor_id,
            expected_target=str(contract["target"]),
            expected_features=[str(x) for x in contract["features"]],
        )
        contract_reports[predictor_id] = report
        _validate_check_block(f"dataset_contract_{predictor_id}", report)

    predictor_families = runtime["predictor_families"] or {}
    for predictor_id in PREDICTOR_IDS:
        families = [str(x) for x in predictor_families.get(predictor_id, [])]
        module_name = TRAIN_MODULE_BY_PREDICTOR[predictor_id]
        dataset_json = datasets_dir / f"{ds_names[predictor_id]}.jsonl"
        meta_json = datasets_dir / f"{ds_names[predictor_id]}_meta.json"

        for family in families:
            run_cmd(
                [
                    python_bin,
                    "-m",
                    module_name,
                    "--dataset",
                    str(dataset_json),
                    "--meta",
                    str(meta_json),
                    "--family",
                    family,
                    "--output-root",
                    str(models_dir),
                ],
                cwd=project_root,
            )

    selection_json = reports_dir / "winner_bundles.json"
    selection_md = reports_dir / "winner_bundles.md"
    selection_cfg = runtime["selection"] or {}
    run_cmd(
        [
            python_bin,
            "-m",
            "predictors.training.select_best_bundles",
            "--models-root",
            str(models_dir),
            "--required-predictors",
            ",".join([str(x) for x in selection_cfg.get("required_predictors", list(PREDICTOR_IDS))]),
            "--class-maximize",
            ",".join([str(x) for x in selection_cfg.get("class_maximize", [])]),
            "--class-minimize",
            ",".join([str(x) for x in selection_cfg.get("class_minimize", [])]),
            "--reg-minimize",
            ",".join([str(x) for x in selection_cfg.get("reg_minimize", [])]),
            "--reg-maximize",
            ",".join([str(x) for x in selection_cfg.get("reg_maximize", [])]),
            "--output-json",
            str(selection_json),
            "--output-md",
            str(selection_md),
        ],
        cwd=project_root,
    )
    selection_payload = _read_json(selection_json)

    audit_csv = reports_dir / "feature_audit.csv"
    audit_md = reports_dir / "feature_audit.md"
    audit_json = reports_dir / "feature_audit.json"
    run_cmd(
        [
            python_bin,
            "-m",
            "predictors.audit.feature_audit",
            "--datasets-dir",
            str(datasets_dir),
            "--cost-policy",
            cost_policy,
            "--dataset-name",
            f"quality_ex_ante={ds_names['quality_ex_ante']}",
            "--dataset-name",
            f"quality_post_hoc={ds_names['quality_post_hoc']}",
            "--dataset-name",
            f"service_cost={ds_names['service_cost']}",
            "--output-csv",
            str(audit_csv),
            "--output-md",
            str(audit_md),
            "--output-json",
            str(audit_json),
        ],
        cwd=project_root,
    )
    audit_payload = _read_json(audit_json)
    audit_validation = validate_audit_report(
        audit_payload=audit_payload,
        validations_cfg=runtime.get("validations") or {},
    )
    _validate_check_block("feature_audit", audit_validation)

    smoke_json = reports_dir / "smoke_inference.json"
    run_cmd(
        [
            python_bin,
            "-m",
            "predictors.smoke_inference",
            "--trace-pattern",
            trace_patterns[0],
            "--selection-json",
            str(selection_json),
            "--output-json",
            str(smoke_json),
            "--max-rows",
            str(int(runtime.get("smoke_max_rows", 30))),
        ],
        cwd=project_root,
    )
    smoke_payload = _read_json(smoke_json)

    manifest_snapshot = manifest_dir / "resolved_manifest.json"
    _write_json(manifest_snapshot, runtime)

    final_report_payload = {
        "generated_at_utc": _now_utc_iso(),
        "status": "valid",
        "manifest_version": runtime.get("manifest_version"),
        "tag": tag,
        "resolved_runtime": runtime,
        "validation": {
            "trace_preflight": preflight,
            "dataset_contracts": contract_reports,
            "feature_audit": audit_validation,
        },
        "selection": selection_payload,
        "smoke": {
            "rows_scored": len(smoke_payload) if isinstance(smoke_payload, list) else None,
            "artifact": str(smoke_json),
        },
        "artifacts": {
            "run_root": str(run_dirs["run_root"]),
            "datasets_dir": str(datasets_dir),
            "models_dir": str(models_dir),
            "reports_dir": str(reports_dir),
            "manifest_snapshot": str(manifest_snapshot),
            "trace_preflight_json": str(preflight_json),
            "trace_preflight_md": str(preflight_md),
            "selection_json": str(selection_json),
            "selection_md": str(selection_md),
            "audit_json": str(audit_json),
            "audit_md": str(audit_md),
            "audit_csv": str(audit_csv),
            "smoke_json": str(smoke_json),
        },
    }

    final_report_json = reports_dir / "final_rerun_report.json"
    final_report_md = reports_dir / "final_rerun_report.md"
    _write_json(final_report_json, final_report_payload)
    _write_text(final_report_md, _render_final_report_md(final_report_payload))

    summary_payload = {
        "run_root": str(run_dirs["run_root"]),
        "datasets_dir": str(datasets_dir),
        "models_dir": str(models_dir),
        "reports_dir": str(reports_dir),
        "selection_json": str(selection_json),
        "audit_json": str(audit_json),
        "smoke_json": str(smoke_json),
        "final_report_json": str(final_report_json),
        "final_report_md": str(final_report_md),
        "manifest_snapshot": str(manifest_snapshot),
    }
    summary_json = reports_dir / "pipeline_summary.json"
    _write_json(summary_json, summary_payload)
    summary_payload["pipeline_summary_json"] = str(summary_json)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final turnkey offline predictor rerun pipeline")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root path",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help="Path to final rerun manifest JSON",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use",
    )
    parser.add_argument(
        "--trace-pattern",
        action="append",
        default=[],
        dest="trace_patterns",
        help="Input trace JSONL glob pattern. Repeatable.",
    )
    parser.add_argument("--tag", default=None, help="Override run tag from manifest")
    parser.add_argument("--output-root", default=None, help="Override base output root from manifest")
    parser.add_argument(
        "--cost-policy",
        choices=["strict_ex_ante", "extended_operational"],
        default=None,
        help="Override cost policy from manifest",
    )
    parser.add_argument(
        "--cost-mode",
        choices=["latency_ms", "composite"],
        default=None,
        help="Override cost mode from manifest",
    )
    parser.add_argument("--smoke-max-rows", type=int, default=None, help="Override smoke rows from manifest")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = _resolve_path(project_root, args.manifest)
    manifest = _read_json(manifest_path)
    _assert_manifest_shape(manifest)

    runtime = _resolve_runtime_config(
        project_root=project_root,
        manifest=manifest,
        trace_patterns_override=list(args.trace_patterns),
        tag_override=args.tag,
        output_root_override=args.output_root,
        cost_policy_override=args.cost_policy,
        cost_mode_override=args.cost_mode,
        smoke_rows_override=args.smoke_max_rows,
    )

    summary = run_pipeline(
        project_root=project_root,
        python_bin=args.python,
        runtime=runtime,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
