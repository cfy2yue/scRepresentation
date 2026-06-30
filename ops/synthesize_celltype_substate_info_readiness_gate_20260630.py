#!/usr/bin/env python3
"""Readiness gate for cell-type/substate information density x.

CPU/report-only. This checks whether current LatentFM local artifacts contain
enough condition-level state/substate metadata to build Beauvoir's
cell-type/substate information-density training axis. It does not train,
infer, select checkpoints, read canonical multi, or read Track C query.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
STATE_MATRIX = ROOT / "reports/state_context_condition_matrix_20260628/state_context_condition_matrix.csv"
ANCHOR = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json"
CANDIDATE = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json"
OUT_DIR = ROOT / "reports/celltype_substate_info_readiness_gate_20260630"
OUT_JSON = OUT_DIR / "celltype_substate_info_readiness_gate_20260630.json"
OUT_MD = OUT_DIR / "LATENTFM_CELLTYPE_SUBSTATE_INFO_READINESS_GATE_20260630.md"
OUT_JOIN = OUT_DIR / "celltype_substate_info_outcome_join_20260630.csv"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_outcomes(anchor_path: Path, candidate_path: Path) -> pd.DataFrame:
    anchor = load_json(anchor_path)
    cand = load_json(candidate_path)
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        a_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in ((anchor.get("groups") or {}).get(group) or {}).get("condition_metrics", [])
        }
        c_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in ((cand.get("groups") or {}).get(group) or {}).get("condition_metrics", [])
        }
        for key in sorted(set(a_rows) & set(c_rows)):
            a = a_rows[key]
            c = c_rows[key]
            rows.append(
                {
                    "group": group,
                    "dataset": key[0],
                    "condition": key[1],
                    "anchor_pearson_pert": float(a["pearson_pert"]),
                    "candidate_pearson_pert": float(c["pearson_pert"]),
                    "pp_delta": float(c["pearson_pert"]) - float(a["pearson_pert"]),
                    "mmd_delta": float(c["test_mmd"]) - float(a["test_mmd"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_group(joined: pd.DataFrame, group: str) -> dict[str, Any]:
    part = joined[joined["group"] == group].copy()
    if part.empty:
        return {"group": group, "n": 0}
    state = part[part["max_state_entropy"] > 0]
    no_state = part[part["max_state_entropy"] <= 0]
    return {
        "group": group,
        "n": int(len(part)),
        "state_n": int(len(state)),
        "state_datasets": int(state["dataset"].nunique()) if len(state) else 0,
        "state_mean_pp_delta": float(state["pp_delta"].mean()) if len(state) else None,
        "no_state_mean_pp_delta": float(no_state["pp_delta"].mean()) if len(no_state) else None,
        "state_minus_no_state_pp_delta": float(state["pp_delta"].mean() - no_state["pp_delta"].mean()) if len(state) and len(no_state) else None,
        "state_mean_mmd_delta": float(state["mmd_delta"].mean()) if len(state) else None,
        "state_dataset_min_pp_delta": float(state.groupby("dataset")["pp_delta"].mean().min()) if len(state) else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = pd.read_csv(STATE_MATRIX)
    unique = state.drop_duplicates(["dataset", "condition"]).copy()
    unique["has_state_signal"] = pd.to_numeric(unique["max_state_entropy"], errors="coerce").fillna(0.0) > 0
    state_rows = unique[unique["has_state_signal"]].copy()
    outcomes = load_outcomes(ANCHOR, CANDIDATE)
    joined = outcomes.merge(unique, on=["dataset", "condition"], how="left")
    joined.to_csv(OUT_JOIN, index=False)

    state_dataset_counts = state_rows["dataset"].astype(str).value_counts()
    top_dataset_fraction = float(state_dataset_counts.iloc[0] / state_dataset_counts.sum()) if len(state_dataset_counts) else 0.0
    group_summaries = [summarize_group(joined, group) for group in GROUPS]
    reasons: list[str] = []
    if int(state_rows["dataset"].nunique()) < 8:
        reasons.append("state_signal_datasets_lt_8")
    if int(len(state_rows)) < 200:
        reasons.append("state_signal_conditions_lt_200")
    if top_dataset_fraction > 0.25:
        reasons.append("top_state_dataset_fraction_gt_0p25")
    for row in group_summaries:
        if row.get("state_minus_no_state_pp_delta") is None:
            reasons.append(f"{row['group']}_state_contrast_missing")
        elif float(row["state_minus_no_state_pp_delta"]) <= 0.0:
            reasons.append(f"{row['group']}_state_minus_no_state_pp_not_positive")
        if row.get("state_dataset_min_pp_delta") is not None and float(row["state_dataset_min_pp_delta"]) < -0.02:
            reasons.append(f"{row['group']}_state_dataset_min_lt_neg0p02")
    status = "celltype_substate_info_readiness_fail_no_gpu" if reasons else "celltype_substate_info_readiness_pass_prepare_cpu_pair_gate"

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "inputs": {
            "state_matrix": str(STATE_MATRIX),
            "anchor_internal": str(ANCHOR),
            "candidate_internal": str(CANDIDATE),
        },
        "n_conditions": int(len(unique)),
        "state_signal_conditions": int(len(state_rows)),
        "state_signal_datasets": int(state_rows["dataset"].nunique()),
        "top_state_dataset_fraction": top_dataset_fraction,
        "state_dataset_counts": state_dataset_counts.to_dict(),
        "group_summaries": group_summaries,
        "reasons": sorted(set(reasons)),
        "outputs": {"join": str(OUT_JOIN), "report": str(OUT_MD)},
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# LatentFM Cell-Type/Substate Information Readiness Gate 20260630",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only readiness check for Beauvoir's cell-type/substate information-density x.",
        "- Uses existing state/context condition matrix and internal cap120 candidate-vs-anchor condition outcomes.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Coverage",
        "",
        f"- condition rows: `{payload['n_conditions']}`",
        f"- state-signal conditions: `{payload['state_signal_conditions']}`",
        f"- state-signal datasets: `{payload['state_signal_datasets']}`",
        f"- top state dataset fraction: `{fmt(payload['top_state_dataset_fraction'])}`",
        "",
        "State-signal datasets:",
    ]
    for dataset, count in state_dataset_counts.items():
        lines.append(f"- `{dataset}`: `{int(count)}`")
    lines.extend(
        [
            "",
            "## Internal Outcome Contrast",
            "",
            "| group | n | state n | state datasets | state pp | no-state pp | state minus no-state | state MMD | state dataset min |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in group_summaries:
        lines.append(
            f"| `{row['group']}` | {row.get('n', 0)} | {row.get('state_n', 0)} | {row.get('state_datasets', 0)} | "
            f"{fmt(row.get('state_mean_pp_delta'))} | {fmt(row.get('no_state_mean_pp_delta'))} | "
            f"{fmt(row.get('state_minus_no_state_pp_delta'))} | {fmt(row.get('state_mean_mmd_delta'))} | "
            f"{fmt(row.get('state_dataset_min_pp_delta'))} |"
        )
    lines.extend(["", "## Decision", ""])
    if payload["reasons"]:
        lines.append("Current local LatentFM artifacts do not support a GPU-ready cell-type/substate information axis.")
        lines.extend(f"- reason: `{reason}`" for reason in payload["reasons"])
    else:
        lines.append("Readiness passed; next step is a stricter matched-pair CPU gate before any GPU.")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- join rows: `{OUT_JOIN}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": payload["reasons"], "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
