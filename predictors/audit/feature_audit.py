from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from distill.dataset_utils import read_jsonl
from predictors.feature_registry import allowed_for_predictor, rule_map


DEFAULT_DATASETS = {
    "quality_ex_ante": "quality_ex_ante_iter2_real_hardened",
    "quality_post_hoc": "quality_post_hoc_iter2_real_hardened",
    "service_cost": "service_cost_iter2_real_hardened",
}


def run_feature_audit(
    *,
    datasets_dir: Path,
    output_csv: Path,
    output_md: Path,
    output_json: Path,
    cost_policy: str,
    dataset_names: Dict[str, str],
) -> Dict[str, Any]:
    rules = rule_map()
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for predictor_id in ["quality_ex_ante", "quality_post_hoc", "service_cost"]:
        dataset_name = dataset_names[predictor_id]
        meta_path = datasets_dir / f"{dataset_name}_meta.json"
        jsonl_path = datasets_dir / f"{dataset_name}.jsonl"

        if not meta_path.exists() or not jsonl_path.exists():
            raise FileNotFoundError(f"Missing dataset artifacts for {predictor_id}: {meta_path} / {jsonl_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_columns = [str(x) for x in meta.get("feature_columns", [])]
        target_column = str(meta.get("target_column"))

        allowed = set(allowed_for_predictor(predictor_id, cost_policy=cost_policy))
        universe = sorted(set(feature_columns) | set(allowed) | set(rules.keys()))

        predictor_rows: List[Dict[str, Any]] = []
        for feature in universe:
            rule = rules.get(feature)
            present = feature in feature_columns
            is_allowed = feature in allowed

            if rule is None and present:
                status = "review"
            elif is_allowed and present:
                status = "keep"
            elif (not is_allowed) and present:
                status = "drop"
            elif is_allowed and (not present):
                status = "missing_expected"
            else:
                status = "not_used"

            predictor_rows.append(
                {
                    "predictor_id": predictor_id,
                    "dataset_name": dataset_name,
                    "feature": feature,
                    "source": rule.source if rule else "unknown",
                    "availability": rule.availability if rule else "unknown",
                    "allowed": "yes" if is_allowed else "no",
                    "present_in_dataset": "yes" if present else "no",
                    "leakage_risk": rule.leakage_risk if rule else "unknown",
                    "policy_recommendation": rule.recommendation if rule else "review",
                    "action": status,
                }
            )

        leakage_scan = correlation_scan(
            rows=read_jsonl(jsonl_path),
            feature_columns=feature_columns,
            target_column=target_column,
        )

        summary[predictor_id] = {
            "dataset_name": dataset_name,
            "feature_count": len(feature_columns),
            "rows_in_audit": len(predictor_rows),
            "action_counts": _count_by(predictor_rows, "action"),
            "high_correlation_warnings": leakage_scan,
        }
        all_rows.extend(predictor_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, all_rows)

    markdown = render_markdown(summary=summary, rows=all_rows)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    payload = {
        "summary": summary,
        "artifacts": {
            "csv": str(output_csv),
            "markdown": str(output_md),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def correlation_scan(
    *,
    rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
    target_column: str,
    threshold: float = 0.995,
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    y = _to_float_array([row.get(target_column) for row in rows])
    if y is None:
        return []

    warnings: List[Dict[str, Any]] = []
    for feature in feature_columns:
        x = _to_float_array([row.get(feature) for row in rows])
        if x is None:
            continue
        corr = _safe_corr(x, y)
        if corr is None:
            continue
        if abs(corr) >= threshold:
            warnings.append(
                {
                    "feature": feature,
                    "abs_corr": float(abs(corr)),
                    "corr": float(corr),
                }
            )

    warnings.sort(key=lambda w: w["abs_corr"], reverse=True)
    return warnings


def _to_float_array(values: Sequence[Any]) -> Optional[np.ndarray]:
    buf: List[float] = []
    for value in values:
        try:
            if value is None or value == "":
                return None
            buf.append(float(value))
        except (TypeError, ValueError):
            return None
    return np.asarray(buf, dtype=float)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    vx = np.var(x)
    vy = np.var(y)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(*, summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Predictor Feature Audit")
    lines.append("")

    for predictor_id in ["quality_ex_ante", "quality_post_hoc", "service_cost"]:
        node = summary.get(predictor_id, {})
        lines.append(f"## {predictor_id}")
        lines.append("")
        lines.append(f"- dataset_name: {node.get('dataset_name')}")
        lines.append(f"- feature_count: {node.get('feature_count')}")
        lines.append(f"- action_counts: `{json.dumps(node.get('action_counts', {}), ensure_ascii=False)}`")
        warnings = node.get("high_correlation_warnings", []) or []
        lines.append(f"- high_correlation_warnings: {len(warnings)}")
        for warning in warnings[:15]:
            lines.append(
                f"  - {warning.get('feature')}: abs_corr={warning.get('abs_corr'):.6f}"
            )
        lines.append("")

    lines.append("## Policy Legend")
    lines.append("")
    lines.append("- keep: allowed and currently present")
    lines.append("- drop: disallowed but currently present")
    lines.append("- missing_expected: allowed by policy but not present")
    lines.append("- not_used: neither allowed nor present")
    lines.append("")

    return "\n".join(lines)


def _count_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out


def parse_dataset_mapping(values: Sequence[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_DATASETS)
    for value in values:
        if "=" not in value:
            raise ValueError("Dataset mapping must follow predictor_id=dataset_name")
        predictor_id, dataset_name = value.split("=", 1)
        predictor_id = predictor_id.strip()
        dataset_name = dataset_name.strip()
        if predictor_id not in mapping:
            raise ValueError(f"Unsupported predictor in mapping: {predictor_id}")
        if not dataset_name:
            raise ValueError("dataset_name cannot be empty")
        mapping[predictor_id] = dataset_name
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature audit tables and leakage warnings")
    parser.add_argument("--datasets-dir", default="results/predictors/datasets")
    parser.add_argument("--cost-policy", choices=["strict_ex_ante", "extended_operational"], default="strict_ex_ante")
    parser.add_argument(
        "--dataset-name",
        action="append",
        default=[],
        help="Override dataset mapping with predictor_id=dataset_name (repeatable)",
    )
    parser.add_argument("--output-csv", default="results/predictors/reports/hardened_feature_audit.csv")
    parser.add_argument("--output-md", default="results/predictors/reports/hardened_feature_audit.md")
    parser.add_argument("--output-json", default="results/predictors/reports/hardened_feature_audit.json")
    args = parser.parse_args()

    mapping = parse_dataset_mapping(args.dataset_name)
    result = run_feature_audit(
        datasets_dir=Path(args.datasets_dir),
        output_csv=Path(args.output_csv),
        output_md=Path(args.output_md),
        output_json=Path(args.output_json),
        cost_policy=args.cost_policy,
        dataset_names=mapping,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
