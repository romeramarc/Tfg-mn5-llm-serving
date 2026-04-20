from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from predictors.inference import suite_from_selection_json
from predictors.schemas import load_traces_from_patterns


def run_smoke(
    *,
    trace_pattern: str,
    selection_json: Path,
    output_json: Path,
    max_rows: int,
) -> Dict[str, Any]:
    traces = load_traces_from_patterns([trace_pattern])
    if not traces:
        raise ValueError(f"No traces found for pattern: {trace_pattern}")

    suite = suite_from_selection_json(selection_json)

    rows: List[Dict[str, Any]] = []
    for trace in traces[:max_rows]:
        rows.append(suite.predict_from_trace(trace))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    return {
        "rows_scored": len(rows),
        "output_json": str(output_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke inference with selected predictor bundles")
    parser.add_argument("--trace-pattern", default="results/predictors/traces/iter2_real_multimodel_trace.jsonl")
    parser.add_argument("--selection-json", default="results/predictors/reports/hardened_selected_bundles.json")
    parser.add_argument("--output-json", default="results/predictors/reports/hardened_smoke_inference.json")
    parser.add_argument("--max-rows", type=int, default=20)
    args = parser.parse_args()

    result = run_smoke(
        trace_pattern=args.trace_pattern,
        selection_json=Path(args.selection_json),
        output_json=Path(args.output_json),
        max_rows=args.max_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
