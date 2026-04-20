from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_REQUIRED_PREDICTORS: Tuple[str, ...] = (
    "quality_ex_ante",
    "quality_post_hoc",
    "service_cost",
)


DEFAULT_POLICY: Dict[str, Dict[str, List[str]]] = {
    "classification": {
        "maximize": ["roc_auc", "average_precision"],
        "minimize": ["ece_abs", "brier"],
    },
    "regression": {
        "minimize": ["mae", "rmse"],
        "maximize": ["r2"],
    },
}


def select_best_bundles(
    models_root: Path,
    *,
    dataset_filter: str | None = None,
    required_predictors: Sequence[str] = DEFAULT_REQUIRED_PREDICTORS,
    selection_policy: Dict[str, Dict[str, List[str]]] | None = None,
    strict_required: bool = True,
) -> Dict[str, Any]:
    policy = selection_policy or DEFAULT_POLICY
    candidates_by_predictor: Dict[str, List[Dict[str, Any]]] = {}

    for model_dir in sorted(models_root.glob("*")):
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.json"
        bundle_path = model_dir / "model_bundle.joblib"
        if not metrics_path.exists() or not bundle_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)

        # Keep only the hardened/new metrics format.
        if not isinstance(metrics.get("test"), dict):
            continue

        dataset_jsonl = str(((metrics.get("dataset") or {}).get("jsonl") or ""))
        if dataset_filter and dataset_filter not in dataset_jsonl:
            continue

        predictor_id = str(metrics.get("predictor_id", ""))
        if not predictor_id:
            continue

        entry = {
            "model_dir": str(model_dir),
            "metrics_json": str(metrics_path),
            "bundle": str(bundle_path),
            "predictor_id": predictor_id,
            "model_family": metrics.get("model_family"),
            "task": metrics.get("task"),
            "metrics": metrics,
        }
        candidates_by_predictor.setdefault(predictor_id, []).append(entry)

    winners: Dict[str, Any] = {}
    rankings: Dict[str, List[Dict[str, Any]]] = {}

    for predictor_id, candidates in candidates_by_predictor.items():
        ranked = sorted(candidates, key=lambda c: _ranking_key(c, policy), reverse=False)
        rankings[predictor_id] = [_summary_row(c, policy) for c in ranked]
        if ranked:
            winners[predictor_id] = _summary_row(ranked[0], policy)

    required_list = [str(x) for x in required_predictors]
    missing_predictors = [predictor for predictor in required_list if predictor not in winners]
    if strict_required and missing_predictors:
        raise RuntimeError(
            "Missing winner bundles for required predictors: " + ", ".join(missing_predictors)
        )

    return {
        "selection_policy": policy,
        "required_predictors": required_list,
        "missing_predictors": missing_predictors,
        "winners": winners,
        "rankings": rankings,
    }


def _summary_row(candidate: Dict[str, Any], selection_policy: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    metrics = candidate["metrics"]
    test_global = ((metrics.get("test") or {}).get("global") or {})
    ranking_key = _ranking_key(candidate, selection_policy)
    return {
        "predictor_id": candidate.get("predictor_id"),
        "task": candidate.get("task"),
        "model_family": candidate.get("model_family"),
        "model_dir": candidate.get("model_dir"),
        "bundle": candidate.get("bundle"),
        "metrics_json": candidate.get("metrics_json"),
        "test_global": test_global,
        "ranking_key": list(ranking_key),
    }


def _ranking_key(candidate: Dict[str, Any], selection_policy: Dict[str, Dict[str, List[str]]]) -> Tuple:
    metrics = candidate["metrics"]
    task = str(candidate.get("task", ""))
    global_metrics = ((metrics.get("test") or {}).get("global") or {})
    model_family = str(candidate.get("model_family") or "")
    model_dir = str(candidate.get("model_dir") or "")

    if task == "classification":
        policy = selection_policy.get("classification", {})
        maximize = [str(x) for x in policy.get("maximize", [])]
        minimize = [str(x) for x in policy.get("minimize", [])]
        key_parts: List[Any] = []
        for metric in maximize:
            key_parts.append(-_num(global_metrics.get(metric), default=-1e18))
        for metric in minimize:
            key_parts.append(_num(global_metrics.get(metric), default=1e18))
        key_parts.append(model_family)
        key_parts.append(model_dir)
        return tuple(key_parts)

    policy = selection_policy.get("regression", {})
    minimize = [str(x) for x in policy.get("minimize", [])]
    maximize = [str(x) for x in policy.get("maximize", [])]
    key_parts = []
    for metric in minimize:
        key_parts.append(_num(global_metrics.get(metric), default=1e18))
    for metric in maximize:
        key_parts.append(-_num(global_metrics.get(metric), default=-1e18))
    key_parts.append(model_family)
    key_parts.append(model_dir)
    return tuple(key_parts)


def _num(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Predictor Model Selection")
    lines.append("")
    lines.append("## Selection Policy")
    lines.append("")
    lines.append(f"- required_predictors: `{json.dumps(report.get('required_predictors', []), ensure_ascii=False)}`")
    lines.append(f"- selection_policy: `{json.dumps(report.get('selection_policy', {}), ensure_ascii=False)}`")
    lines.append(f"- missing_predictors: `{json.dumps(report.get('missing_predictors', []), ensure_ascii=False)}`")
    lines.append("")

    winners = report.get("winners", {})
    if not winners:
        lines.append("No model candidates were found.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Winners")
    lines.append("")
    for predictor_id in sorted(winners.keys()):
        winner = winners[predictor_id]
        lines.append(f"### {predictor_id}")
        lines.append("")
        lines.append(f"- model_family: {winner.get('model_family')}")
        lines.append(f"- model_dir: {winner.get('model_dir')}")
        lines.append(f"- bundle: {winner.get('bundle')}")
        lines.append(f"- metrics_json: {winner.get('metrics_json')}")
        lines.append(f"- test_global: `{json.dumps(winner.get('test_global', {}), ensure_ascii=False)}`")
        lines.append(f"- ranking_key: `{json.dumps(winner.get('ranking_key', []), ensure_ascii=False)}`")
        lines.append("")

    lines.append("## Full Rankings")
    lines.append("")
    rankings = report.get("rankings", {})
    for predictor_id in sorted(rankings.keys()):
        lines.append(f"### {predictor_id}")
        lines.append("")
        for idx, row in enumerate(rankings[predictor_id], start=1):
            lines.append(f"{idx}. {row.get('model_family')} - {row.get('model_dir')}")
            lines.append(f"   test_global: `{json.dumps(row.get('test_global', {}), ensure_ascii=False)}`")
            lines.append(f"   ranking_key: `{json.dumps(row.get('ranking_key', []), ensure_ascii=False)}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best predictor model bundles from metrics")
    parser.add_argument("--models-root", default="results/predictors/models")
    parser.add_argument("--dataset-filter", default=None, help="Optional substring that must appear in metrics.dataset.jsonl")
    parser.add_argument(
        "--required-predictors",
        default=",".join(DEFAULT_REQUIRED_PREDICTORS),
        help="Comma-separated predictor IDs that must have winners.",
    )
    parser.add_argument(
        "--class-maximize",
        default=",".join(DEFAULT_POLICY["classification"]["maximize"]),
        help="Comma-separated classification metrics to maximize in order.",
    )
    parser.add_argument(
        "--class-minimize",
        default=",".join(DEFAULT_POLICY["classification"]["minimize"]),
        help="Comma-separated classification metrics to minimize in order.",
    )
    parser.add_argument(
        "--reg-minimize",
        default=",".join(DEFAULT_POLICY["regression"]["minimize"]),
        help="Comma-separated regression metrics to minimize in order.",
    )
    parser.add_argument(
        "--reg-maximize",
        default=",".join(DEFAULT_POLICY["regression"]["maximize"]),
        help="Comma-separated regression metrics to maximize in order.",
    )
    parser.add_argument(
        "--allow-missing-required",
        action="store_true",
        help="Do not fail if required predictors do not have winners.",
    )
    parser.add_argument("--output-json", default="results/predictors/reports/hardened_selected_bundles.json")
    parser.add_argument("--output-md", default="results/predictors/reports/hardened_selected_bundles.md")
    args = parser.parse_args()

    required_predictors = _parse_csv_list(args.required_predictors)
    policy = {
        "classification": {
            "maximize": _parse_csv_list(args.class_maximize),
            "minimize": _parse_csv_list(args.class_minimize),
        },
        "regression": {
            "minimize": _parse_csv_list(args.reg_minimize),
            "maximize": _parse_csv_list(args.reg_maximize),
        },
    }

    report = select_best_bundles(
        Path(args.models_root),
        dataset_filter=args.dataset_filter,
        required_predictors=required_predictors,
        selection_policy=policy,
        strict_required=not args.allow_missing_required,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))


def _parse_csv_list(value: str) -> List[str]:
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",")]
    return [item for item in items if item]


if __name__ == "__main__":
    main()
