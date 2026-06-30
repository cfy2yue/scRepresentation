#!/usr/bin/env python3
"""Design gate for a SciPlex pharmacogenomic training axis.

Consumes the strict source admission output and tries to form a leakage-safe
high/low train-only design from external drug-cell-line sensitivity. This is a
CPU/report-only gate. It does not train, infer, select checkpoints, read
canonical multi for selection, read Track C query, or use GPU.

Passing this script still does not authorize GPU by itself; it only prepares a
bounded candidate that must be paired with a predeclared launcher, RUN_STATUS,
resource audit, and posthoc no-harm/dual-baseline gates.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
ADMISSION_DIR = ROOT / "reports" / "sciplex_pharmacogenomic_source_admission_gate_20260630"
ADMISSION_JSON = ADMISSION_DIR / "sciplex_pharmacogenomic_source_admission_gate_20260630.json"
ADMISSION_ROWS = ADMISSION_DIR / "sciplex_pharmacogenomic_source_admission_rows_20260630.csv"
OUT_DIR = ROOT / "reports" / "sciplex_pharmacogenomic_training_design_gate_20260630"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt((float(np.var(x)) + float(np.var(y))) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(x)) - float(np.mean(y))) / pooled)


def max_share(values: list[str]) -> float:
    if not values:
        return float("nan")
    counts = Counter(values)
    return max(counts.values()) / len(values)


def pair_within_dataset(df: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    pairs: list[dict[str, Any]] = []
    for dataset, sub in df.groupby("dataset", sort=True):
        sub = sub.sort_values("mean_sensitivity_z")
        lo_cut = sub["mean_sensitivity_z"].quantile(q_low)
        hi_cut = sub["mean_sensitivity_z"].quantile(q_high)
        low = sub[sub["mean_sensitivity_z"] <= lo_cut].copy()
        high = sub[sub["mean_sensitivity_z"] >= hi_cut].copy()
        n = min(len(low), len(high))
        if n == 0:
            continue
        low = low.head(n).reset_index(drop=True)
        high = high.tail(n).sort_values("mean_sensitivity_z", ascending=False).reset_index(drop=True)
        for i in range(n):
            h = high.iloc[i]
            l = low.iloc[i]
            pairs.append(
                {
                    "dataset": dataset,
                    "high_condition": h["sciplex_condition"],
                    "low_condition": l["sciplex_condition"],
                    "high_score": h["mean_sensitivity_z"],
                    "low_score": l["mean_sensitivity_z"],
                    "score_delta": h["mean_sensitivity_z"] - l["mean_sensitivity_z"],
                    "high_source_count": h["source_count"],
                    "low_source_count": l["source_count"],
                    "high_mean_auc": h.get("mean_auc", np.nan),
                    "low_mean_auc": l.get("mean_auc", np.nan),
                }
            )
    return pd.DataFrame(pairs)


def summarize_pairs(name: str, pairs: pd.DataFrame) -> dict[str, Any]:
    if pairs.empty:
        return {
            "design": name,
            "n_pairs": 0,
            "n_datasets": 0,
            "status": "fail_no_gpu",
            "reasons": "empty_pairs",
        }
    high_source = pd.to_numeric(pairs["high_source_count"], errors="coerce")
    low_source = pd.to_numeric(pairs["low_source_count"], errors="coerce")
    reasons: list[str] = []
    n_pairs = int(len(pairs))
    n_datasets = int(pairs["dataset"].nunique())
    min_pairs_per_dataset = int(pairs.groupby("dataset").size().min())
    top_dataset_fraction = max_share(pairs["dataset"].astype(str).tolist())
    mean_delta = float(pd.to_numeric(pairs["score_delta"], errors="coerce").mean())
    source_count_smd = smd(high_source, low_source)
    if n_pairs < 60:
        reasons.append("pairs_below_60")
    if n_datasets < 3:
        reasons.append("datasets_below_3")
    if min_pairs_per_dataset < 15:
        reasons.append("min_pairs_per_dataset_below_15")
    if top_dataset_fraction > 0.45:
        reasons.append("top_dataset_fraction_gt_0p45")
    if mean_delta <= 0.75:
        reasons.append("mean_source_score_delta_le_0p75")
    if math.isfinite(source_count_smd) and abs(source_count_smd) > 0.15:
        reasons.append("source_count_smd_gt_0p15")
    status = "design_pass_prepare_launcher_review_no_gpu" if not reasons else "fail_no_gpu"
    return {
        "design": name,
        "n_pairs": n_pairs,
        "n_datasets": n_datasets,
        "min_pairs_per_dataset": min_pairs_per_dataset,
        "top_dataset_fraction": top_dataset_fraction,
        "mean_score_delta": mean_delta,
        "source_count_smd": source_count_smd,
        "status": status,
        "reasons": ";".join(reasons) if reasons else "none",
    }


def main() -> int:
    admission = read_json(ADMISSION_JSON)
    reasons: list[str] = []
    if not ADMISSION_JSON.exists():
        reasons.append("admission_json_missing")
    if not ADMISSION_ROWS.exists():
        reasons.append("admission_rows_missing")
    if admission.get("status") != "sciplex_pharmacogenomic_source_admission_gate_pass_outcome_gate_next_no_gpu":
        reasons.append("admission_not_passed")

    status = "sciplex_pharmacogenomic_training_design_gate_blocked_no_gpu"
    design_rows = pd.DataFrame()
    selected_pairs = pd.DataFrame()
    best: dict[str, Any] = {}

    if not reasons:
        df = pd.read_csv(ADMISSION_ROWS)
        df = df[pd.to_numeric(df["source_count"], errors="coerce") >= 2].copy()
        df = df[pd.to_numeric(df["mean_sensitivity_z"], errors="coerce").notna()].copy()
        rows = []
        pair_tables: dict[str, pd.DataFrame] = {}
        for name, ql, qh in [
            ("tertile_q33_q67", 0.33, 0.67),
            ("q40_q60", 0.40, 0.60),
            ("q25_q75", 0.25, 0.75),
        ]:
            pairs = pair_within_dataset(df, ql, qh)
            pair_tables[name] = pairs
            rows.append(summarize_pairs(name, pairs))
        design_rows = pd.DataFrame(rows).sort_values(
            ["status", "n_pairs", "mean_score_delta"], ascending=[True, False, False]
        )
        pass_rows = design_rows[design_rows["status"] == "design_pass_prepare_launcher_review_no_gpu"]
        if pass_rows.empty:
            reasons.extend(sorted({r for val in design_rows["reasons"] for r in str(val).split(";") if r != "none"}))
        else:
            best = pass_rows.iloc[0].to_dict()
            selected_pairs = pair_tables[str(best["design"])]
            status = "sciplex_pharmacogenomic_training_design_gate_pass_prepare_launcher_review_no_gpu"

    if reasons:
        status = "sciplex_pharmacogenomic_training_design_gate_blocked_no_gpu"

    design_csv = OUT_DIR / "sciplex_pharmacogenomic_training_design_rows_20260630.csv"
    pairs_csv = OUT_DIR / "sciplex_pharmacogenomic_training_design_selected_pairs_20260630.csv"
    json_path = OUT_DIR / "sciplex_pharmacogenomic_training_design_gate_20260630.json"
    md_path = OUT_DIR / "LATENTFM_SCIPLEX_PHARMACOGENOMIC_TRAINING_DESIGN_GATE_20260630.md"
    design_rows.to_csv(design_csv, index=False)
    selected_pairs.to_csv(pairs_csv, index=False)

    result = {
        "timestamp": now(),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "uses_gpu": False,
        },
        "blocked_reasons": reasons,
        "best_design": best,
        "admission_json": str(ADMISSION_JSON),
        "admission_rows": str(ADMISSION_ROWS),
        "outputs": {"design_rows": str(design_csv), "selected_pairs": str(pairs_csv), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# SciPlex Pharmacogenomic Training Design Gate",
        "",
        f"Created: `{result['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only high/low train design gate from external pharmacogenomic sensitivity.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- Passing this gate only prepares launcher review; it is not model evidence.",
        "",
        "## Decision",
        "",
    ]
    if status.endswith("pass_prepare_launcher_review_no_gpu"):
        lines.append("A balanced high/low design exists. Review selected pairs and write a bounded launcher/RUN_STATUS before any GPU use.")
    else:
        lines.append("Blocked or failed. Do not launch GPU from this source route.")
    lines += [
        "",
        "## Reasons",
        "",
    ]
    if reasons:
        for reason in reasons:
            lines.append(f"- `{reason}`")
    else:
        lines.append("- `none`")
    lines += [
        "",
        "## Design Rows",
        "",
        "| design | pairs | datasets | min pairs/dataset | mean score delta | source-count SMD | status | reasons |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if not design_rows.empty:
        for _, row in design_rows.iterrows():
            lines.append(
                f"| `{row['design']}` | {int(row['n_pairs'])} | {int(row['n_datasets'])} | "
                f"{int(row['min_pairs_per_dataset'])} | {float(row['mean_score_delta']):.4f} | "
                f"{float(row['source_count_smd']):.4f} | `{row['status']}` | {row['reasons']} |"
            )
    lines += [
        "",
        "## Outputs",
        "",
        f"- design rows: `{design_csv}`",
        f"- selected pairs: `{pairs_csv}`",
        f"- JSON: `{json_path}`",
        f"- Markdown: `{md_path}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
