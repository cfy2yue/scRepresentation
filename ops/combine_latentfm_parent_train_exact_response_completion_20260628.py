#!/usr/bin/env python3
"""Combine existing exact coverage with parent-train completion rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
BASE_COVERAGE = ROOT / "reports/exact_response_information_combined_coverage_20260628"
COMPLETION = ROOT / "reports/parent_train_exact_response_completion_20260628"
OUT_DIR = ROOT / "reports/exact_response_information_parent_train_complete_20260628"


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def budget_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cols, level in [
        (["group", "dataset", "budget"], "dataset"),
        (["group", "budget"], "group"),
        (["budget"], "all"),
    ]:
        for keys, part in frame.groupby(cols, sort=True):
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
    rows: list[dict[str, Any]] = []
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=BASE_COVERAGE)
    parser.add_argument("--completion-dir", type=Path, default=COMPLETION)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_condition = read_required(args.base_dir / "exact_response_information_condition_rows.csv")
    base_budget = read_required(args.base_dir / "exact_response_information_budget_rows.csv")
    base_meta = read_required(args.base_dir / "exact_response_information_dataset_meta.csv")
    comp_condition = read_required(args.completion_dir / "parent_train_exact_response_completion_condition_rows.csv")
    comp_budget = read_required(args.completion_dir / "parent_train_exact_response_completion_budget_rows.csv")
    comp_meta = read_required(args.completion_dir / "parent_train_exact_response_completion_dataset_meta.csv")

    condition = pd.concat([base_condition, comp_condition], ignore_index=True).drop_duplicates(
        ["group", "dataset", "condition"], keep="last"
    )
    budget = pd.concat([base_budget, comp_budget], ignore_index=True).drop_duplicates(
        ["group", "dataset", "condition", "budget"], keep="last"
    )
    meta = pd.concat([base_meta, comp_meta], ignore_index=True)

    condition_csv = args.out_dir / "exact_response_information_condition_rows.csv"
    budget_csv = args.out_dir / "exact_response_information_budget_rows.csv"
    budget_summary_csv = args.out_dir / "exact_response_information_budget_summary.csv"
    condition_summary_csv = args.out_dir / "exact_response_information_condition_summary.csv"
    meta_csv = args.out_dir / "exact_response_information_dataset_meta.csv"
    json_path = args.out_dir / "latentfm_exact_response_information_parent_train_complete_20260628.json"
    report_md = args.out_dir / "LATENTFM_EXACT_RESPONSE_INFORMATION_PARENT_TRAIN_COMPLETE_20260628.md"

    condition.to_csv(condition_csv, index=False)
    budget.to_csv(budget_csv, index=False)
    budget_sum = budget_summary(budget)
    condition_sum = condition_summary(condition)
    budget_sum.to_csv(budget_summary_csv, index=False)
    condition_sum.to_csv(condition_summary_csv, index=False)
    meta.to_csv(meta_csv, index=False)

    new_rows = len(condition) - len(base_condition.drop_duplicates(["group", "dataset", "condition"]))
    overall1000 = budget_sum[(budget_sum["level"] == "all") & (budget_sum["budget"] == 1000)]
    overall1000_row = overall1000.iloc[0].to_dict() if not overall1000.empty else {}
    all_cond = condition_sum[condition_sum["level"] == "all"].iloc[0].to_dict()
    payload = {
        "timestamp": now_cst(),
        "status": "exact_response_information_parent_train_complete_ready_no_gpu",
        "gpu_authorized_next": False,
        "base_dir": str(args.base_dir),
        "completion_dir": str(args.completion_dir),
        "condition_rows": int(condition.shape[0]),
        "budget_rows": int(budget.shape[0]),
        "new_unique_condition_rows": int(new_rows),
        "outputs": {
            "condition_csv": str(condition_csv),
            "budget_csv": str(budget_csv),
            "budget_summary_csv": str(budget_summary_csv),
            "condition_summary_csv": str(condition_summary_csv),
            "meta_csv": str(meta_csv),
        },
    }
    write_json(json_path, payload)

    lines = [
        "# LatentFM Exact Response-Information Parent-Train Complete Coverage",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only union of existing combined exact coverage and parent-train completion rows.",
        "- No training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query.",
        "",
        "## Summary",
        "",
        f"- condition rows: `{payload['condition_rows']}`",
        f"- budget rows: `{payload['budget_rows']}`",
        f"- new unique condition rows vs base: `{payload['new_unique_condition_rows']}`",
    ]
    if overall1000_row:
        lines.append(
            f"- overall top-1000 HVG share `{fmt_float(overall1000_row.get('hvg_share_mean'))}`, "
            f"abundance share `{fmt_float(overall1000_row.get('abundance_share_mean'))}`"
        )
    lines.append(
        f"- overall median abundance k80/k90: `{fmt_float(all_cond.get('abundance_k80_median'))}`/"
        f"`{fmt_float(all_cond.get('abundance_k90_median'))}`"
    )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Condition rows: `{condition_csv}`",
            f"- Budget rows: `{budget_csv}`",
            f"- Budget summary: `{budget_summary_csv}`",
            f"- Condition summary: `{condition_summary_csv}`",
            f"- Dataset meta: `{meta_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
