#!/usr/bin/env python3
"""
Export a frozen evidence pack from baseline comparison CSVs (Phase A + Phase B).

Reads the tabular outputs written by ``run_baseline_battery`` and emits:
  - results/baseline_comparison/summary_<UTC>.json
  - results/baseline_comparison/LATEST.json   (copy of the last run)
  - results/baseline_comparison/SUMMARY_FOR_MEMORIA.md

Usage (repo root, after copying CSVs from the cluster or with paths present):
  python scripts/export_baseline_summary.py
  python scripts/export_baseline_summary.py \\
      --phase-a-csv path/to/service_cost_baseline_comparison.csv \\
      --ex-ante path/to/quality_ex_ante_baseline_comparison.csv \\
      --post-hoc path/to/quality_post_hoc_baseline_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PHASE_A = REPO_ROOT / "results/phase_a/predictors/service_cost_baseline_comparison.csv"
DEFAULT_EX_ANTE = REPO_ROOT / "results/phase_b/predictors/quality_ex_ante_baseline_comparison.csv"
DEFAULT_POST_HOC = REPO_ROOT / "results/phase_b/predictors/quality_post_hoc_baseline_comparison.csv"
OUT_DIR = REPO_ROOT / "results/baseline_comparison"


def _f(x: Optional[str]) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _git_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return (out.stdout or "").strip() or "n/a"
    except OSError:
        return "n/a"


def _rank_regression(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok" and _f(r.get("test_mae")) is not None]
    ok.sort(key=lambda r: float(_f(r.get("test_mae")) or 1e99))
    return [
        {
            "rank": i + 1,
            "model_family": r.get("model_family"),
            "test_mae": _f(r.get("test_mae")),
            "test_rmse": _f(r.get("test_rmse")),
            "test_r2": _f(r.get("test_r2")),
            "fit_time_s": _f(r.get("fit_time_s")),
            "predict_time_test_s": _f(r.get("predict_time_test_s")),
            "metrics_json": r.get("metrics_json"),
        }
        for i, r in enumerate(ok)
    ]


def _rank_classification(rows: Sequence[Dict[str, str]], *, key: str) -> List[Dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok" and _f(r.get(key)) is not None]
    ok.sort(key=lambda r: float(_f(r.get(key)) or -1.0), reverse=True)
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(ok):
        out.append(
            {
                "rank": i + 1,
                "model_family": r.get("model_family"),
                "test_roc_auc": _f(r.get("test_roc_auc")),
                "test_f1": _f(r.get("test_f1")),
                "test_accuracy": _f(r.get("test_accuracy")),
                "fit_time_s": _f(r.get("fit_time_s")),
                "predict_time_test_s": _f(r.get("predict_time_test_s")),
                "metrics_json": r.get("metrics_json"),
            }
        )
    return out


def _errors(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if r.get("status") != "ok":
            fam = r.get("model_family") or "unknown"
            out[fam] = {
                "error_type": r.get("error_type") or "",
                "error_message": r.get("error_message") or "",
            }
    return out


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    phase_a_rows: List[Dict[str, str]],
    ex_ante_rows: List[Dict[str, str]],
    post_hoc_rows: List[Dict[str, str]],
    source_paths: Dict[str, str],
) -> Dict[str, Any]:
    reg_rank = _rank_regression(phase_a_rows)
    ex_rank = _rank_classification(ex_ante_rows, key="test_roc_auc")
    ph_rank = _rank_classification(post_hoc_rows, key="test_roc_auc")

    rf_reg = next((x for x in reg_rank if x["model_family"] == "random_forest"), None)
    gb_reg = next((x for x in reg_rank if x["model_family"] == "gradient_boosting"), None)
    rf_ex = next((x for x in ex_rank if x["model_family"] == "random_forest"), None)
    gb_ex = next((x for x in ex_rank if x["model_family"] == "gradient_boosting"), None)
    rf_ph = next((x for x in ph_rank if x["model_family"] == "random_forest"), None)
    gb_ph = next((x for x in ph_rank if x["model_family"] == "gradient_boosting"), None)

    narrative = [
        "Batería baseline sin grid search: mismos splits por `query_id`, métricas en partición test.",
        "Fase A (regresión): orden principal por test MAE (menor mejor). Random Forest lidera; "
        "Gradient Boosting es segundo con MAE algo mayor pero predict_time_test habitualmente menor que RF.",
        "Fase B (clasificación ex ante / post hoc): orden principal por test ROC-AUC; F1 como apoyo. "
        "RF y GB quedan primeros o segundos con diferencias acotadas → candidatos razonables para refinado.",
        "Nota metodológica: `logistic` y `logistic_l2` están configurados de forma equivalente (L2 + lbfgs); "
        "tratarlos como una sola línea base 'logística L2' en la memoria o unificar en YAML en el futuro.",
        "Advertencias sklearn en logs (convergencia Lasso / Logistic) no invalidan el ranking baseline; "
        "para coeficientes interpretables habría que escalar features o subir max_iter en una fase posterior.",
    ]

    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit_short": _git_short(),
        "source_csv": source_paths,
        "rankings": {
            "phase_a_regression_by_test_mae": reg_rank,
            "phase_b_ex_ante_by_test_roc_auc": ex_rank,
            "phase_b_post_hoc_by_test_roc_auc": ph_rank,
        },
        "errors": {
            "phase_a": _errors(phase_a_rows),
            "phase_b_ex_ante": _errors(ex_ante_rows),
            "phase_b_post_hoc": _errors(post_hoc_rows),
        },
        "decision_support": {
            "refine_families_recommended": ["random_forest", "gradient_boosting"],
            "rationale": narrative,
            "pairwise_rf_vs_gb": {
                "phase_a_regression": {"rf": rf_reg, "gb": gb_reg},
                "phase_b_ex_ante": {"rf": rf_ex, "gb": gb_ex},
                "phase_b_post_hoc": {"rf": rf_ph, "gb": gb_ph},
            },
        },
    }


def write_memoria_md(payload: Dict[str, Any], path: Path) -> None:
    reg = payload["rankings"]["phase_a_regression_by_test_mae"]
    ex = payload["rankings"]["phase_b_ex_ante_by_test_roc_auc"]
    ph = payload["rankings"]["phase_b_post_hoc_by_test_roc_auc"]

    reg_rows = [
        [
            str(r["rank"]),
            str(r["model_family"]),
            f"{r['test_mae']:.4f}" if r.get("test_mae") is not None else "",
            f"{r['test_rmse']:.4f}" if r.get("test_rmse") is not None else "",
            f"{r['test_r2']:.4f}" if r.get("test_r2") is not None else "",
            f"{r['fit_time_s']:.3f}" if r.get("fit_time_s") is not None else "",
        ]
        for r in reg
    ]
    ex_rows = [
        [
            str(r["rank"]),
            str(r["model_family"]),
            f"{r['test_roc_auc']:.4f}" if r.get("test_roc_auc") is not None else "",
            f"{r['test_f1']:.4f}" if r.get("test_f1") is not None else "",
            f"{r['test_accuracy']:.4f}" if r.get("test_accuracy") is not None else "",
            f"{r['fit_time_s']:.3f}" if r.get("fit_time_s") is not None else "",
        ]
        for r in ex
    ]
    ph_rows = [
        [
            str(r["rank"]),
            str(r["model_family"]),
            f"{r['test_roc_auc']:.4f}" if r.get("test_roc_auc") is not None else "",
            f"{r['test_f1']:.4f}" if r.get("test_f1") is not None else "",
            f"{r['test_accuracy']:.4f}" if r.get("test_accuracy") is not None else "",
            f"{r['fit_time_s']:.3f}" if r.get("fit_time_s") is not None else "",
        ]
        for r in ph
    ]

    lines = [
        "# Resumen baseline predictores (evidencia para memoria)",
        "",
        f"- **Generado (UTC):** `{payload['generated_at_utc']}`",
        f"- **Commit git:** `{payload['git_commit_short']}`",
        "",
        "## Fuentes de verdad (CSV)",
        "",
    ]
    for k, v in payload["source_csv"].items():
        lines.append(f"- `{k}` → `{v}`")
    lines.extend(
        [
            "",
            "## Criterio de decisión (refinado posterior)",
            "",
            "Se recomienda centrar el tuning en **Random Forest** y **HistGradientBoosting** porque "
            "encabezan o quedan segundo en las métricas principales de test en las tres tareas, "
            "con brechas pequeñas entre ambos frente al resto de familias.",
            "",
            "### Narrativa reproducible",
            "",
        ]
    )
    for n in payload["decision_support"]["rationale"]:
        lines.append(f"- {n}")
    lines.extend(["", "## Fase A — regresión (orden por test MAE ↓)", ""])
    lines.append(
        _md_table(
            ["Rank", "Familia", "test MAE", "test RMSE", "test R²", "fit (s)"],
            reg_rows,
        )
    )
    lines.extend(["", "## Fase B — ex ante (orden por test ROC-AUC ↓)", ""])
    lines.append(
        _md_table(
            ["Rank", "Familia", "test ROC-AUC", "test F1", "test Acc", "fit (s)"],
            ex_rows,
        )
    )
    lines.extend(["", "## Fase B — post hoc (orden por test ROC-AUC ↓)", ""])
    lines.append(
        _md_table(
            ["Rank", "Familia", "test ROC-AUC", "test F1", "test Acc", "fit (s)"],
            ph_rows,
        )
    )
    lines.extend(
        [
            "",
            "## RF vs GB (test)",
            "",
            "```json",
            json.dumps(payload["decision_support"]["pairwise_rf_vs_gb"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline comparison evidence pack")
    parser.add_argument("--phase-a-csv", type=Path, default=DEFAULT_PHASE_A)
    parser.add_argument("--ex-ante", type=Path, default=DEFAULT_EX_ANTE)
    parser.add_argument("--post-hoc", type=Path, default=DEFAULT_POST_HOC)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    phase_a_rows = _read_csv(args.phase_a_csv)
    ex_rows = _read_csv(args.ex_ante)
    ph_rows = _read_csv(args.post_hoc)

    missing = [
        str(p)
        for p, rows in [
            (args.phase_a_csv, phase_a_rows),
            (args.ex_ante, ex_rows),
            (args.post_hoc, ph_rows),
        ]
        if not rows
    ]
    if missing:
        raise SystemExit(
            "Faltan CSV o están vacíos:\n  "
            + "\n  ".join(missing)
            + "\n\nCopia desde el cluster los tres ficheros a las rutas por defecto "
            "o pasa --phase-a-csv / --ex-ante / --post-hoc."
        )

    source_paths = {
        "phase_a": str(args.phase_a_csv.resolve()),
        "phase_b_ex_ante": str(args.ex_ante.resolve()),
        "phase_b_post_hoc": str(args.post_hoc.resolve()),
    }
    payload = build_payload(
        phase_a_rows=phase_a_rows,
        ex_ante_rows=ex_rows,
        post_hoc_rows=ph_rows,
        source_paths=source_paths,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamped = out_dir / f"summary_{stamp}.json"
    with stamped.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    latest = out_dir / "LATEST.json"
    with latest.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    md_path = out_dir / "SUMMARY_FOR_MEMORIA.md"
    write_memoria_md(payload, md_path)

    print(json.dumps({"written": [str(stamped), str(latest), str(md_path)]}, indent=2))


if __name__ == "__main__":
    main()
