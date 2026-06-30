#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(data: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not data:
        return default
    return data.get(key, default)


def decide(fixed: dict[str, Any] | None, placebo: dict[str, Any] | None) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    fixed_status = scalar(fixed, "status", "missing")
    placebo_status = scalar(placebo, "status", "missing")
    if fixed_status == "missing":
        return "zscape_periderm_branch_waiting_fixedcell_no_gpu", ["fixedcell_missing"], "wait_for_fixedcell"

    if fixed_status != "zscape_bioinformation_fixedcell_periderm_partial_pass_no_gpu":
        reasons.append(f"fixedcell_not_pass:{fixed_status}")
        return "zscape_periderm_branch_close_model_enabling_no_gpu", reasons, "close_model_enabling_branch"

    if placebo_status == "missing":
        reasons.append("fixedcell_pass_waiting_placebo")
        return "zscape_periderm_branch_waiting_placebo_no_gpu", reasons, "launch_placebo_control"

    if placebo_status == "zscape_periderm_placebo_control_pass_no_gpu":
        return "zscape_periderm_branch_design_review_candidate_no_gpu", [], "write_bounded_design_review"

    reasons.append(f"placebo_not_pass:{placebo_status}")
    return "zscape_periderm_branch_close_model_enabling_no_gpu", reasons, "close_model_enabling_branch"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixedcell-json", type=Path, required=True)
    parser.add_argument("--placebo-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fixed = load_json(args.fixedcell_json)
    placebo = load_json(args.placebo_json)
    status, reasons, next_action = decide(fixed, placebo)

    fixed_summary_csv = None
    fixed_row_csv = None
    if fixed:
        fixed_summary_csv = fixed.get("summary_csv")
        fixed_row_csv = fixed.get("row_csv")
    fixed_summary = pd.read_csv(fixed_summary_csv) if fixed_summary_csv and Path(fixed_summary_csv).exists() else pd.DataFrame()
    fixed_rows = pd.read_csv(fixed_row_csv) if fixed_row_csv and Path(fixed_row_csv).exists() else pd.DataFrame()

    out_json = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "fixedcell_json": str(args.fixedcell_json),
        "placebo_json": str(args.placebo_json) if args.placebo_json else None,
        "fixedcell_status": scalar(fixed, "status", "missing"),
        "placebo_status": scalar(placebo, "status", "missing"),
        "reasons": reasons,
        "next_action": next_action,
    }
    json_path = args.out_dir / "zscape_periderm_branch_decision_20260628.json"
    json_path.write_text(json.dumps(out_json, indent=2, sort_keys=True), encoding="utf-8")

    md_path = args.out_dir / "LATENTFM_ZSCAPE_PERIDERM_BRANCH_DECISION_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Periderm Branch Decision",
        "",
        f"Timestamp: `{utc_now()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Report-only synthesis of fixed-cell and optional placebo gates.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, read Track C query, or authorize GPU.",
        "",
        "## Gate Status",
        "",
        f"- fixed-cell status: `{scalar(fixed, 'status', 'missing')}`",
        f"- placebo status: `{scalar(placebo, 'status', 'missing')}`",
        f"- next action: `{next_action}`",
        "",
        "## Decision Reasons",
        "",
        *(f"- {reason}" for reason in reasons),
    ]
    if not fixed_summary.empty:
        lines.extend(
            [
                "",
                "## Fixed-Cell Subset Summary",
                "",
                "| subset | rows | pass | pass frac | mean ratio | max JSD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in fixed_summary.iterrows():
            lines.append(
                f"| {row.get('subset', '')} | {int(row.get('n_evaluated_rows', 0))} | "
                f"{int(row.get('n_pass_rows', 0))} | {float(row.get('pass_fraction', float('nan'))):.3f} | "
                f"{float(row.get('mean_effect_ratio', float('nan'))):.3f} | "
                f"{float(row.get('max_matched_subtype_jsd', float('nan'))):.3f} |"
            )
    if not fixed_rows.empty:
        periderm = fixed_rows[fixed_rows.get("cell_type_broad", pd.Series(dtype=str)).astype(str) == "periderm"]
        if not periderm.empty:
            lines.extend(
                [
                    "",
                    "## Periderm Row Results",
                    "",
                    "| row_id | target | time | ratio | p_cc | p_label | JSD | lib SMD | gate |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in periderm.iterrows():
                lines.append(
                    f"| {row.get('row_id', '')} | {row.get('gene_target', '')} | {row.get('timepoint', '')} | "
                    f"{float(row.get('effect_ratio_vs_max_null_p95', float('nan'))):.3f} | "
                    f"{float(row.get('p_observed_le_matched_cc_null', float('nan'))):.4f} | "
                    f"{float(row.get('p_observed_le_matched_label_null', float('nan'))):.4f} | "
                    f"{float(row.get('matched_subtype_jsd', float('nan'))):.3f} | "
                    f"{float(row.get('expression_library_smd', float('nan'))):.3f} | "
                    f"{bool(row.get('strict_row_gate', False))} |"
                )
    lines.extend(
        [
            "",
            "## Claim Guard",
            "",
            "- A positive decision only authorizes a bounded design review.",
            "- No direct GPU, model promotion, broad ZSCAPE mechanism, mature-fast-muscle strict claim, canonical multi selection, or Track C query use is authorized.",
            "",
            "## Output Files",
            "",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
