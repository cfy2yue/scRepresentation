#!/usr/bin/env python3
"""Combine primary and supplemental exact response-information coverage dirs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def read_required(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def budget_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cols, level in [
        (["group", "dataset", "budget"], "dataset"),
        (["group", "budget"], "group"),
        (["budget"], "all"),
    ]:
        grouped = frame.groupby(cols, sort=True)
        for keys, part in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(cols, keys))
            rows.append(
                {
                    "level": level,
                    "group": key_map.get("group", "__ALL__"),
                    "dataset": key_map.get("dataset", "__ALL__"),
                    "budget": int(key_map["budget"]),
                    "condition_rows": int(part.shape[0]),
                    "hvg_share_mean": float(part["hvg_share"].mean()),
                    "abundance_share_mean": float(part["abundance_share"].mean()),
                    "hvg_minus_abundance_mean": float(part["hvg_minus_abundance"].mean()),
                    "hvg_abundance_overlap_fraction_mean": float(part["hvg_abundance_overlap_fraction"].mean()),
                    "budget_clipped_fraction": float(part["budget_clipped"].mean()),
                }
            )
    return pd.DataFrame(rows)


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cols, level in [(["group", "dataset"], "dataset"), (["group"], "group")]:
        for keys, part in frame.groupby(cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(cols, keys))
            rows.append(
                {
                    "level": level,
                    "group": key_map.get("group", "__ALL__"),
                    "dataset": key_map.get("dataset", "__ALL__"),
                    "condition_rows": int(part.shape[0]),
                    "hvg_k80_median": float(part["hvg_k80"].median()),
                    "hvg_k90_median": float(part["hvg_k90"].median()),
                    "abundance_k80_median": float(part["abundance_k80"].median()),
                    "abundance_k90_median": float(part["abundance_k90"].median()),
                    "response_energy_mean": float(part["response_energy"].mean()),
                }
            )
    rows.append(
        {
            "level": "all",
            "group": "__ALL__",
            "dataset": "__ALL__",
            "condition_rows": int(frame.shape[0]),
            "hvg_k80_median": float(frame["hvg_k80"].median()),
            "hvg_k90_median": float(frame["hvg_k90"].median()),
            "abundance_k80_median": float(frame["abundance_k80"].median()),
            "abundance_k90_median": float(frame["abundance_k90"].median()),
            "response_energy_mean": float(frame["response_energy"].mean()),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    condition = pd.concat(
        [
            read_required(args.primary_dir, "exact_response_information_condition_rows.csv"),
            read_required(args.supplement_dir, "exact_response_information_condition_rows.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["group", "dataset", "condition"], keep="last")
    budget = pd.concat(
        [
            read_required(args.primary_dir, "exact_response_information_budget_rows.csv"),
            read_required(args.supplement_dir, "exact_response_information_budget_rows.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["group", "dataset", "condition", "budget"], keep="last")
    meta = pd.concat(
        [
            read_required(args.primary_dir, "exact_response_information_dataset_meta.csv"),
            read_required(args.supplement_dir, "exact_response_information_dataset_meta.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["group", "dataset"], keep="last")

    condition_csv = args.out_dir / "exact_response_information_condition_rows.csv"
    budget_csv = args.out_dir / "exact_response_information_budget_rows.csv"
    budget_summary_csv = args.out_dir / "exact_response_information_budget_summary.csv"
    condition_summary_csv = args.out_dir / "exact_response_information_condition_summary.csv"
    meta_csv = args.out_dir / "exact_response_information_dataset_meta.csv"
    json_path = args.out_dir / "latentfm_exact_response_information_coverage_20260628.json"
    report_md = args.out_dir / "LATENTFM_EXACT_RESPONSE_INFORMATION_COVERAGE_20260628.md"
    condition.to_csv(condition_csv, index=False)
    budget.to_csv(budget_csv, index=False)
    budget_sum = budget_summary(budget)
    condition_sum = condition_summary(condition)
    budget_sum.to_csv(budget_summary_csv, index=False)
    condition_sum.to_csv(condition_summary_csv, index=False)
    meta.to_csv(meta_csv, index=False)

    overall1000 = budget_sum[(budget_sum["level"] == "all") & (budget_sum["budget"] == 1000)]
    overall1000_row = overall1000.iloc[0].to_dict() if not overall1000.empty else {}
    all_cond = condition_sum[condition_sum["level"] == "all"].iloc[0].to_dict()
    payload = {
        "created_at": now_cst(),
        "status": "exact_response_information_combined_coverage_ready_no_gpu",
        "primary_dir": str(args.primary_dir),
        "supplement_dir": str(args.supplement_dir),
        "condition_rows": int(condition.shape[0]),
        "budget_rows": int(budget.shape[0]),
        "datasets": int(meta.shape[0]),
        "condition_csv": str(condition_csv),
        "budget_csv": str(budget_csv),
        "budget_summary_csv": str(budget_summary_csv),
        "condition_summary_csv": str(condition_summary_csv),
        "meta_csv": str(meta_csv),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Exact Response-Information Combined Coverage",
        "",
        f"Created: {payload['created_at']}",
        "",
        "Status: `exact_response_information_combined_coverage_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only concatenation of primary and supplemental exact coverage outputs.",
        "* No train/infer/GPU/canonical multi/Track C query/checkpoint selection.",
        "",
        "## Summary",
        "",
        f"* Datasets: `{payload['datasets']}`.",
        f"* Condition rows: `{payload['condition_rows']}`; budget rows: `{payload['budget_rows']}`.",
    ]
    if overall1000_row:
        lines.append(
            f"* Overall top-1000 HVG share `{fmt_float(overall1000_row.get('hvg_share_mean'))}`, "
            f"abundance share `{fmt_float(overall1000_row.get('abundance_share_mean'))}`."
        )
    lines.append(
        f"* Overall median abundance k80/k90: `{fmt_float(all_cond.get('abundance_k80_median'))}`/"
        f"`{fmt_float(all_cond.get('abundance_k90_median'))}` genes."
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* Condition rows: `{condition_csv}`",
            f"* Budget rows: `{budget_csv}`",
            f"* Budget summary: `{budget_summary_csv}`",
            f"* Condition summary: `{condition_summary_csv}`",
            f"* Dataset meta: `{meta_csv}`",
            f"* JSON: `{json_path}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_md}")
    print("status exact_response_information_combined_coverage_ready_no_gpu")


if __name__ == "__main__":
    main()
