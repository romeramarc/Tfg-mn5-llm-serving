"""
eval/rescore_gsm8k_results.py
=============================
Re-score existing GSM8K result JSONL files with the robust numeric extractor.

Useful for fair pre/post comparisons when historical runs used weaker parsing.

Usage
-----
    python -m eval.rescore_gsm8k_results \
        --input results/quality/quality-student_small-*/gsm8k/gsm8k_results.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from eval.scoring import compute_accuracy, extract_numeric_answer, numeric_match


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run(input_path: str, output_dir: str | None = None, pattern: str | None = None) -> Dict[str, Any]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    rows = _read_jsonl(p)
    answer_pattern = pattern or r"####\s*([\-\d,\.]+)"

    rescored: List[Dict[str, Any]] = []
    for row in rows:
        response = str(row.get("model_response") or "")
        reference = row.get("reference_answer")
        error = row.get("error")

        predicted = None
        correct = False
        scorable = False

        if error is None:
            predicted = extract_numeric_answer(response, answer_pattern)
            scorable = predicted is not None
            if predicted is not None and reference is not None:
                correct = numeric_match(predicted, str(reference))

        out = dict(row)
        out["predicted_answer_robust"] = predicted
        out["scorable_robust"] = scorable
        out["correct_robust"] = correct
        rescored.append(out)

    metrics_input = [
        {
            "correct": bool(r.get("correct_robust", False)),
            "scorable": bool(r.get("scorable_robust", False)),
        }
        for r in rescored
    ]
    metrics = compute_accuracy(metrics_input)
    metrics["benchmark"] = "gsm8k"
    if rescored:
        metrics["model"] = rescored[0].get("model", "")

    out_dir = Path(output_dir) if output_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rescored_path = out_dir / "gsm8k_results_rescored.jsonl"
    with rescored_path.open("w", encoding="utf-8") as fh:
        for row in rescored:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_json = out_dir / "gsm8k_metrics_rescored.json"
    metrics_csv = out_dir / "gsm8k_metrics_rescored.csv"
    with metrics_json.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    return {
        "rescored_path": str(rescored_path),
        "metrics_json": str(metrics_json),
        "metrics_csv": str(metrics_csv),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score GSM8K result JSONL with robust parser")
    parser.add_argument("--input", required=True, help="Path to gsm8k_results.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: input dir)")
    parser.add_argument(
        "--answer-pattern",
        default=None,
        help="Optional extraction regex override (default: #### numeric pattern)",
    )
    args = parser.parse_args()

    result = run(args.input, output_dir=args.output_dir, pattern=args.answer_pattern)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
