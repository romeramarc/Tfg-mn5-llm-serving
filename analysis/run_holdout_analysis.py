from __future__ import annotations

import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - notebook machines may differ
    sns = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRS = [
    ROOT / "results" / "routing_eval_holdout",
    ROOT / "results" / "routing_eval_holdout_v2",
]
OUT_DIR = ROOT / "analysis" / "holdout_analysis_outputs"
FIG_DIR = OUT_DIR / "figures"

SYSTEM_LABELS = {
    "sysA_only_teacher": "A: teacher",
    "sysB_only_tiny": "B: tiny",
    "sysC_routing_distilled": "C: routing (5 rungs)",
    "sysD_cascade_distilled": "D: cascade (5 rungs)",
    "sysE_routing_cascade_distilled": "E: routing+cascade (5 rungs)",
    "sysC_routing4": "C4: routing (4 rungs)",
    "sysD_cascade4": "D4: cascade (4 rungs)",
    "sysE_l010": "E4 λ=0.010",
    "sysE_l025": "E4 λ=0.025",
    "sysE_l050": "E4 λ=0.050",
    "sysE_l100": "E4 λ=0.100",
}

SYSTEM_ORDER = [
    "sysA_only_teacher",
    "sysB_only_tiny",
    "sysC_routing_distilled",
    "sysD_cascade_distilled",
    "sysE_routing_cascade_distilled",
    "sysC_routing4",
    "sysD_cascade4",
    "sysE_l010",
    "sysE_l025",
    "sysE_l050",
    "sysE_l100",
]

MODEL_LABELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5B tiny",
    "results/distill/sft-full-qwen2.5-1.5b-v6-20260423T121313Z/final_model": "1.5B distilled",
    "Qwen/Qwen2.5-3B-Instruct": "3B",
    "Qwen/Qwen2.5-7B-Instruct": "7B",
    "Qwen/Qwen2.5-14B-Instruct": "14B teacher",
}


@dataclass(frozen=True)
class RunFile:
    path: Path
    run_dir: str
    run_date: str
    system_id: str
    policy: str


def parse_run_file(path: Path) -> RunFile:
    run_dir = path.parent.name
    match = re.match(r"(?P<system>.+)-(?P<policy>[^-]+)-(?P<ts>\d{8})T", run_dir)
    if not match:
        raise ValueError(f"Unexpected run directory name: {run_dir}")
    return RunFile(
        path=path,
        run_dir=run_dir,
        run_date=match.group("ts"),
        system_id=match.group("system"),
        policy=match.group("policy"),
    )


def safe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() == "nan":
        return None
    if text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def pct(value: float) -> float:
    return float(value) * 100.0


def q(series: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.quantile(quantile))


def read_all_requests() -> pd.DataFrame:
    files: list[Path] = []
    for base in RESULTS_DIRS:
        if not base.exists():
            continue
        files.extend(sorted(base.glob("*/per_request.csv")))
    files = sorted(set(files))
    if not files:
        searched = ", ".join(str(p) for p in RESULTS_DIRS)
        raise SystemExit(f"No per_request.csv files under {searched}")

    frames: list[pd.DataFrame] = []
    for path in files:
        rf = parse_run_file(path)
        df = pd.read_csv(path)
        df["run_dir"] = rf.run_dir
        df["run_date"] = rf.run_date
        df["system_id_from_dir"] = rf.system_id
        df["policy_from_dir"] = rf.policy
        if "system_id" not in df.columns:
            df["system_id"] = rf.system_id
        if "policy" not in df.columns:
            df["policy"] = rf.policy
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df["system_label"] = all_df["system_id"].map(SYSTEM_LABELS).fillna(all_df["system_id"])
    all_df["model_label"] = all_df["selected_model"].map(MODEL_LABELS).fillna(all_df["selected_model"].astype(str))
    all_df["used_teacher"] = all_df["selected_model"].astype(str).str.contains("14B", na=False)
    all_df["is_scorable"] = all_df["scorable"].astype(bool)
    all_df["is_correct"] = all_df["correct"].astype(bool)
    all_df["client_wall_s"] = pd.to_numeric(all_df["client_wall_ms"], errors="coerce") / 1000.0
    all_df["latency_s"] = pd.to_numeric(all_df["latency_ms"], errors="coerce") / 1000.0
    all_df["num_attempts"] = pd.to_numeric(all_df.get("num_attempts"), errors="coerce")
    if all_df["num_attempts"].isna().all() and "attempts" in all_df.columns:
        all_df["num_attempts"] = all_df["attempts"].map(lambda x: len(safe_json(x) or []))
    all_df["total_output_tokens"] = pd.to_numeric(all_df.get("total_output_tokens"), errors="coerce")
    all_df["confidence"] = pd.to_numeric(all_df.get("confidence"), errors="coerce")
    all_df["post_hoc_probability"] = pd.to_numeric(all_df.get("post_hoc_probability"), errors="coerce")
    all_df["segment"] = all_df["benchmark"].astype(str) + "/" + all_df["length_bucket"].astype(str)
    return all_df


def summarize_core(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_date, system_id, run_dir), g in df.groupby(["run_date", "system_id", "run_dir"], sort=True):
        scorable = g[g["is_scorable"]]
        rows.append(
            {
                "run_date": run_date,
                "system_id": system_id,
                "system_label": SYSTEM_LABELS.get(system_id, system_id),
                "policy": g["policy"].iloc[0],
                "run_dir": run_dir,
                "n": len(g),
                "scorable_rate_pct": pct(g["is_scorable"].mean()),
                "accuracy_total_pct": pct(g["is_correct"].mean()),
                "accuracy_scorable_pct": pct(scorable["is_correct"].mean()) if len(scorable) else math.nan,
                "correct_requests": int(g["is_correct"].sum()),
                "scorable_requests": int(g["is_scorable"].sum()),
                "teacher_share_pct": pct(g["used_teacher"].mean()),
                "mean_attempts": float(g["num_attempts"].mean()),
                "p50_attempts": q(g["num_attempts"], 0.50),
                "p95_attempts": q(g["num_attempts"], 0.95),
                "client_wall_mean_s": float(g["client_wall_s"].mean()),
                "client_wall_p50_s": q(g["client_wall_s"], 0.50),
                "client_wall_p90_s": q(g["client_wall_s"], 0.90),
                "client_wall_p95_s": q(g["client_wall_s"], 0.95),
                "client_wall_p99_s": q(g["client_wall_s"], 0.99),
                "tail_p95_over_p50": q(g["client_wall_s"], 0.95) / q(g["client_wall_s"], 0.50),
                "mean_output_tokens": float(g["total_output_tokens"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["system_rank"] = out["system_id"].map({s: i for i, s in enumerate(SYSTEM_ORDER)})
    return out.sort_values(["run_date", "system_rank"]).drop(columns=["system_rank"])


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    dims = [
        ["run_date", "system_id", "benchmark"],
        ["run_date", "system_id", "length_bucket"],
        ["run_date", "system_id", "benchmark", "length_bucket"],
    ]
    rows: list[dict[str, Any]] = []
    for dim in dims:
        for keys, g in df.groupby(dim, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            item = dict(zip(dim, keys))
            scorable = g[g["is_scorable"]]
            item.update(
                {
                    "slice": "+".join([d.replace("run_date", "run") for d in dim[2:]]) or dim[-1],
                    "system_label": SYSTEM_LABELS.get(item["system_id"], item["system_id"]),
                    "n": len(g),
                    "accuracy_total_pct": pct(g["is_correct"].mean()),
                    "accuracy_scorable_pct": pct(scorable["is_correct"].mean()) if len(scorable) else math.nan,
                    "scorable_rate_pct": pct(g["is_scorable"].mean()),
                    "teacher_share_pct": pct(g["used_teacher"].mean()),
                    "mean_attempts": float(g["num_attempts"].mean()),
                    "p50_s": q(g["client_wall_s"], 0.50),
                    "p95_s": q(g["client_wall_s"], 0.95),
                    "mean_tokens": float(g["total_output_tokens"].mean()),
                }
            )
            rows.append(item)
    return pd.DataFrame(rows)


def summarize_model_mix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_date, system_id, model_label), g in df.groupby(["run_date", "system_id", "model_label"], sort=True):
        rows.append(
            {
                "run_date": run_date,
                "system_id": system_id,
                "system_label": SYSTEM_LABELS.get(system_id, system_id),
                "model_label": model_label,
                "count": len(g),
                "share_pct": pct(len(g) / len(df[(df["run_date"] == run_date) & (df["system_id"] == system_id)])),
                "accuracy_total_pct_within_model": pct(g["is_correct"].mean()),
                "scorable_rate_pct": pct(g["is_scorable"].mean()),
                "p50_s": q(g["client_wall_s"], 0.50),
                "p95_s": q(g["client_wall_s"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def pairwise_against_sys_e(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key = ["run_date", "benchmark", "example_id"]
    cols = key + ["system_id", "is_correct", "is_scorable", "client_wall_s", "used_teacher", "num_attempts", "model_label", "reason"]
    slim = df[cols].copy()
    for run_date, run_df in slim.groupby("run_date"):
        target = run_df[run_df["system_id"] == "sysE_routing_cascade_distilled"].copy()
        if target.empty:
            continue
        target = target.rename(columns={c: f"e_{c}" for c in slim.columns if c not in key})
        for baseline in [s for s in SYSTEM_ORDER if s != "sysE_routing_cascade_distilled"]:
            base = run_df[run_df["system_id"] == baseline].copy()
            if base.empty:
                continue
            base = base.rename(columns={c: f"b_{c}" for c in slim.columns if c not in key})
            merged = target.merge(base, on=key, how="inner")
            same = merged["e_is_correct"] == merged["b_is_correct"]
            rows.append(
                {
                    "run_date": run_date,
                    "target_system": "sysE_routing_cascade_distilled",
                    "baseline_system": baseline,
                    "baseline_label": SYSTEM_LABELS.get(baseline, baseline),
                    "matched_prompts": len(merged),
                    "target_more_correct": int((merged["e_is_correct"] & ~merged["b_is_correct"]).sum()),
                    "baseline_more_correct": int((~merged["e_is_correct"] & merged["b_is_correct"]).sum()),
                    "same_correctness": int(same.sum()),
                    "target_faster_when_same_correctness": int((same & (merged["e_client_wall_s"] < merged["b_client_wall_s"])).sum()),
                    "target_slower_when_same_correctness": int((same & (merged["e_client_wall_s"] > merged["b_client_wall_s"])).sum()),
                    "mean_delta_latency_s": float((merged["e_client_wall_s"] - merged["b_client_wall_s"]).mean()),
                    "p50_delta_latency_s": q(merged["e_client_wall_s"] - merged["b_client_wall_s"], 0.50),
                    "p95_delta_latency_s": q(merged["e_client_wall_s"] - merged["b_client_wall_s"], 0.95),
                    "target_uses_teacher_not_baseline": int((merged["e_used_teacher"] & ~merged["b_used_teacher"]).sum()),
                    "baseline_uses_teacher_not_target": int((~merged["e_used_teacher"] & merged["b_used_teacher"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def prompt_pair_sys_e_vs_sys_d(df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    key = ["benchmark", "example_id", "pool_index", "length_bucket"]
    cols = key + ["is_correct", "is_scorable", "client_wall_s", "used_teacher", "num_attempts", "model_label", "reason", "confidence", "post_hoc_probability", "route_path"]
    d = df[(df["run_date"] == run_date) & (df["system_id"] == "sysD_cascade_distilled")][cols].copy()
    e = df[(df["run_date"] == run_date) & (df["system_id"] == "sysE_routing_cascade_distilled")][cols].copy()
    d = d.rename(columns={c: f"d_{c}" for c in cols if c not in key})
    e = e.rename(columns={c: f"e_{c}" for c in cols if c not in key})
    merged = d.merge(e, on=key, how="inner")
    merged["delta_latency_s"] = merged["e_client_wall_s"] - merged["d_client_wall_s"]
    merged["delta_correct"] = merged["e_is_correct"].astype(int) - merged["d_is_correct"].astype(int)
    merged["teacher_change"] = merged["e_used_teacher"].astype(int) - merged["d_used_teacher"].astype(int)
    return merged.sort_values("delta_latency_s", ascending=False)


def routing_score_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    routing_df = df[df["system_id"].isin(["sysC_routing_distilled", "sysE_routing_cascade_distilled"])].copy()
    for (run_date, system_id), g in routing_df.groupby(["run_date", "system_id"], sort=True):
        parsed = g["routing_scores"].map(safe_json)
        n_with_scores = int(parsed.notna().sum())
        all_equal = 0
        max_margin_values = []
        selected_counts: dict[str, int] = {}
        for scores, selected in zip(parsed, g["route_path"], strict=False):
            if not isinstance(scores, dict) or not scores:
                continue
            utilities = []
            for rung, payload in scores.items():
                if isinstance(payload, dict) and "utility" in payload:
                    utilities.append((rung, float(payload["utility"])))
            if utilities:
                vals = [round(v, 9) for _, v in utilities]
                if len(set(vals)) == 1:
                    all_equal += 1
                ordered = sorted(utilities, key=lambda x: x[1], reverse=True)
                if len(ordered) > 1:
                    max_margin_values.append(ordered[0][1] - ordered[1][1])
            if isinstance(selected, str):
                selected_counts[selected] = selected_counts.get(selected, 0) + 1
        rows.append(
            {
                "run_date": run_date,
                "system_id": system_id,
                "n": len(g),
                "rows_with_scores": n_with_scores,
                "all_candidate_utilities_equal_rows": all_equal,
                "all_candidate_utilities_equal_pct_of_scored": pct(all_equal / n_with_scores) if n_with_scores else math.nan,
                "median_top_utility_margin": float(pd.Series(max_margin_values).median()) if max_margin_values else math.nan,
                "route_path_counts": json.dumps(selected_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def prompt_consensus(df: pd.DataFrame, latest_run: str) -> pd.DataFrame:
    latest = df[df["run_date"] == latest_run].copy()
    key = ["benchmark", "example_id", "pool_index", "length_bucket"]
    pivot = latest.pivot_table(index=key, columns="system_id", values="is_correct", aggfunc="first").reset_index()
    systems = [s for s in SYSTEM_ORDER if s in pivot.columns]
    pivot["n_correct_systems"] = pivot[systems].sum(axis=1)
    pivot["all_wrong"] = pivot["n_correct_systems"] == 0
    pivot["only_teacher_correct"] = (pivot["sysA_only_teacher"] == True) & (pivot["n_correct_systems"] == 1) if "sysA_only_teacher" in pivot.columns else False
    pivot["sysE_unique_win"] = (pivot["sysE_routing_cascade_distilled"] == True) & (pivot["n_correct_systems"] == 1) if "sysE_routing_cascade_distilled" in pivot.columns else False
    return pivot.sort_values(["n_correct_systems", "benchmark", "example_id"])


def is_pareto_efficient(points: pd.DataFrame) -> pd.Series:
    # Maximize accuracy, minimize p95 latency and teacher share.
    efficient = []
    for _, row in points.iterrows():
        dominated = False
        for _, other in points.iterrows():
            if other.name == row.name:
                continue
            better_or_equal = (
                other["accuracy_total_pct"] >= row["accuracy_total_pct"]
                and other["client_wall_p95_s"] <= row["client_wall_p95_s"]
                and other["teacher_share_pct"] <= row["teacher_share_pct"]
            )
            strictly_better = (
                other["accuracy_total_pct"] > row["accuracy_total_pct"]
                or other["client_wall_p95_s"] < row["client_wall_p95_s"]
                or other["teacher_share_pct"] < row["teacher_share_pct"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        efficient.append(not dominated)
    return pd.Series(efficient, index=points.index)


def plot_core(core: pd.DataFrame, latest_run: str) -> None:
    latest = core[core["run_date"] == latest_run].copy()
    latest["system_label"] = pd.Categorical(latest["system_label"], [SYSTEM_LABELS[s] for s in SYSTEM_ORDER], ordered=True)
    latest = latest.sort_values("system_label")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(latest["system_label"], latest["accuracy_total_pct"], color="#4C78A8", alpha=0.85, label="Accuracy total (%)")
    ax2.plot(latest["system_label"], latest["client_wall_p95_s"], color="#F58518", marker="o", linewidth=2.5, label="p95 latency (s)")
    ax1.set_ylabel("Accuracy total (%)")
    ax2.set_ylabel("p95 latency (s)")
    ax1.set_xlabel("")
    ax1.set_title(f"Quality vs tail latency, latest run ({latest_run})")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"quality_vs_latency_{latest_run}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        latest["client_wall_p95_s"],
        latest["accuracy_total_pct"],
        s=80 + latest["teacher_share_pct"] * 3,
        c=latest["teacher_share_pct"],
        cmap="viridis",
        edgecolor="black",
    )
    for _, row in latest.iterrows():
        ax.annotate(row["system_label"], (row["client_wall_p95_s"], row["accuracy_total_pct"]), xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("p95 client latency (s)")
    ax.set_ylabel("Accuracy total (%)")
    ax.set_title("Pareto view: quality, latency, teacher usage")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Teacher usage (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pareto_accuracy_latency_teacher_{latest_run}.png", dpi=180)
    plt.close(fig)


def plot_segment_heatmaps(segment: pd.DataFrame, latest_run: str) -> None:
    latest = segment[(segment["run_date"] == latest_run) & segment["benchmark"].notna() & segment["length_bucket"].notna()].copy()
    systems = [s for s in SYSTEM_ORDER if s in latest["system_id"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [
        (axes[0], "teacher_share_pct", "Teacher share (%)"),
        (axes[1], "p95_s", "p95 latency (s)"),
    ]:
        pivot = latest.pivot_table(index=["benchmark", "length_bucket"], columns="system_id", values=metric, aggfunc="first")
        pivot = pivot[[s for s in systems if s in pivot.columns]]
        pivot.columns = [SYSTEM_LABELS.get(c, c) for c in pivot.columns]
        if sns:
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="mako_r" if metric == "p95_s" else "rocket_r", ax=ax)
        else:
            ax.imshow(pivot.values)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)), ["/".join(idx) for idx in pivot.index])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("benchmark / bucket")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"segment_heatmaps_{latest_run}.png", dpi=180)
    plt.close(fig)


def plot_model_mix(model_mix: pd.DataFrame, latest_run: str) -> None:
    latest = model_mix[model_mix["run_date"] == latest_run].copy()
    pivot = latest.pivot_table(index="system_id", columns="model_label", values="share_pct", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([s for s in SYSTEM_ORDER if s in pivot.index])
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="tab20")
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in pivot.index], rotation=20, ha="right")
    ax.set_ylabel("Selected model share (%)")
    ax.set_title(f"Model mix by policy ({latest_run})")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"model_mix_{latest_run}.png", dpi=180)
    plt.close()


def plot_run_to_run(core: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [
        (axes[0], "accuracy_total_pct", "Accuracy total (%)"),
        (axes[1], "client_wall_p95_s", "p95 client latency (s)"),
    ]:
        pivot = core.pivot_table(index="system_id", columns="run_date", values=metric, aggfunc="first")
        pivot = pivot.reindex([s for s in SYSTEM_ORDER if s in pivot.index])
        pivot.index = [SYSTEM_LABELS.get(s, s) for s in pivot.index]
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "run_to_run_comparison.png", dpi=180)
    plt.close(fig)


def write_findings(core: pd.DataFrame, pairwise: pd.DataFrame, routing_diag: pd.DataFrame, latest_run: str) -> None:
    latest = core[core["run_date"] == latest_run].set_index("system_id")
    lines: list[str] = []
    lines.append("# Holdout Routing Analysis - Preliminary Findings\n")
    lines.append(f"Latest run analysed: `{latest_run}`.\n")
    lines.append("## Executive Claims\n")
    if {"sysA_only_teacher", "sysE_routing_cascade_distilled", "sysD_cascade_distilled", "sysB_only_tiny", "sysC_routing_distilled"}.issubset(latest.index):
        a = latest.loc["sysA_only_teacher"]
        b = latest.loc["sysB_only_tiny"]
        c = latest.loc["sysC_routing_distilled"]
        d = latest.loc["sysD_cascade_distilled"]
        e = latest.loc["sysE_routing_cascade_distilled"]
        lines.append(
            f"- The teacher baseline remains the quality ceiling: {a.accuracy_total_pct:.2f}% total accuracy, "
            f"but it uses the 14B model for 100% of requests and has p95 latency {a.client_wall_p95_s:.2f}s.\n"
        )
        lines.append(
            f"- The cheap policies (tiny and routing-only in the latest run) are service-fast but quality-poor: "
            f"sysB {b.accuracy_total_pct:.2f}% at p95 {b.client_wall_p95_s:.2f}s; "
            f"sysC {c.accuracy_total_pct:.2f}% at p95 {c.client_wall_p95_s:.2f}s with {c.teacher_share_pct:.1f}% teacher usage.\n"
        )
        lines.append(
            f"- Cascade-only is the strongest balanced candidate in the latest run if p95 latency matters: "
            f"{d.accuracy_total_pct:.2f}% total accuracy, {d.teacher_share_pct:.1f}% teacher usage, p95 {d.client_wall_p95_s:.2f}s.\n"
        )
        lines.append(
            f"- Routing+cascade recovers similar quality ({e.accuracy_total_pct:.2f}%) but pays more tail latency "
            f"(p95 {e.client_wall_p95_s:.2f}s) and more teacher usage ({e.teacher_share_pct:.1f}%) than cascade-only in the latest run.\n"
        )
    lines.append("\n## Pairwise SysE vs Baselines\n")
    for _, row in pairwise[pairwise["run_date"] == latest_run].iterrows():
        lines.append(
            f"- vs {row['baseline_label']}: sysE is more correct on {int(row.target_more_correct)} prompts, "
            f"less correct on {int(row.baseline_more_correct)} prompts, and has median latency delta "
            f"{row.p50_delta_latency_s:.2f}s.\n"
        )
    lines.append("\n## Routing Diagnostics\n")
    for _, row in routing_diag.iterrows():
        lines.append(
            f"- {row.run_date} {row.system_id}: {row.all_candidate_utilities_equal_pct_of_scored:.1f}% of scored rows "
            f"have equal utilities across candidates; route path counts = {row.route_path_counts}.\n"
        )
    (OUT_DIR / "findings.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = read_all_requests()
    latest_run = sorted(df["run_date"].unique())[-1]

    core = summarize_core(df)
    segment = summarize_segments(df)
    model_mix = summarize_model_mix(df)
    pairwise = pairwise_against_sys_e(df)
    routing_diag = routing_score_diagnostics(df)
    prompt_consensus_latest = prompt_consensus(df, latest_run)
    pareto_latest = core[core["run_date"] == latest_run].copy()
    pareto_latest["pareto_efficient"] = is_pareto_efficient(pareto_latest)

    core.to_csv(OUT_DIR / "core_metrics.csv", index=False)
    segment.to_csv(OUT_DIR / "segment_metrics.csv", index=False)
    model_mix.to_csv(OUT_DIR / "model_mix.csv", index=False)
    pairwise.to_csv(OUT_DIR / "pairwise_sysE_vs_baselines.csv", index=False)
    routing_diag.to_csv(OUT_DIR / "routing_score_diagnostics.csv", index=False)
    prompt_consensus_latest.to_csv(OUT_DIR / "prompt_consensus_latest.csv", index=False)
    pareto_latest.to_csv(OUT_DIR / "pareto_latest.csv", index=False)

    for run_date in sorted(df["run_date"].unique()):
        pair = prompt_pair_sys_e_vs_sys_d(df, run_date)
        pair.to_csv(OUT_DIR / f"prompt_pair_sysE_vs_sysD_{run_date}.csv", index=False)

    plot_core(core, latest_run)
    plot_segment_heatmaps(segment, latest_run)
    plot_model_mix(model_mix, latest_run)
    plot_run_to_run(core)
    write_findings(core, pairwise, routing_diag, latest_run)

    print(f"Loaded {len(df)} per-request rows from {df['run_dir'].nunique()} runs")
    print(f"Wrote outputs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Latest run: {latest_run}")
    print(core[core["run_date"] == latest_run][["system_label", "accuracy_total_pct", "client_wall_p95_s", "teacher_share_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
