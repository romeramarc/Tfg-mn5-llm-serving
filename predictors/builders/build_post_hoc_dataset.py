from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from predictors.dataset_common import (
    metadata_from_trace,
    post_hoc_feature_row,
    target_correct,
    write_dataset_artifacts,
)
from predictors.schemas import load_traces_from_patterns


def build_post_hoc_dataset(
    *,
    input_patterns: Sequence[str],
    output_dir: Path,
    dataset_name: str,
    report_path: Path,
) -> Dict[str, Any]:
    traces = load_traces_from_patterns(input_patterns)

    rows: List[Dict[str, Any]] = []
    dropped_no_target = 0

    for trace in traces:
        y = target_correct(trace)
        if y is None:
            dropped_no_target += 1
            continue

        row = metadata_from_trace(trace)
        row.update(post_hoc_feature_row(trace))
        row["target_correct"] = y
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("benchmark")), str(r.get("query_id")), str(r.get("model_name"))))

    artifacts = write_dataset_artifacts(
        rows=rows,
        dataset_name=dataset_name,
        target_column="target_correct",
        output_dir=output_dir,
    )

    report = {
        "dataset_name": dataset_name,
        "input_patterns": list(input_patterns),
        "input_trace_rows": len(traces),
        "kept_rows": len(rows),
        "dropped_rows_no_target": dropped_no_target,
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality post-hoc predictor dataset")
    parser.add_argument(
        "--input",
        action="append",
        dest="input_patterns",
        default=[],
        help="Glob pattern with trace JSONL files. Repeatable.",
    )
    parser.add_argument("--output-dir", default="results/predictors/datasets")
    parser.add_argument("--dataset-name", default="quality_post_hoc_iter2_real_hardened")
    parser.add_argument("--report", default=None, help="Optional explicit path for build report JSON")
    args = parser.parse_args()

    report_path = Path(args.report) if args.report else Path(args.output_dir) / f"{args.dataset_name}_build_report.json"

    input_patterns = args.input_patterns or ["results/predictors/traces/iter2_real_multimodel_trace.jsonl"]

    result = build_post_hoc_dataset(
        input_patterns=input_patterns,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        report_path=report_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
