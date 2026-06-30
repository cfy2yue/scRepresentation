#!/usr/bin/env python3
"""Readiness gate for observable gene-budget stability as a training axis."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
BUDGET_ROWS = ROOT / "reports/hvg_meanmatched_expanded_controls_20260629/condition_negative_control_rows.csv"
ANCHOR = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json"
CANDIDATE = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json"
OUT_DIR = ROOT / "reports/observable_gene_budget_stability_readiness_gate_20260630"
OUT_JSON = OUT_DIR / "observable_gene_budget_stability_readiness_gate_20260630.json"
OUT_MD = OUT_DIR / "LATENTFM_OBSERVABLE_GENE_BUDGET_STABILITY_READINESS_GATE_20260630.md"
OUT_JOIN = OUT_DIR / "observable_gene_budget_stability_outcome_join_20260630.csv"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
FEATURES = (
    "split_half_jaccard",
    "response_energy_over_shuffled_mean",
    "hvg_minus_mean_matched_mean",
    "hvg_minus_shuffled_label_hvg_mean",
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


def load_outcomes() -> pd.DataFrame:
    anchor = load_json(ANCHOR)
    cand = load_json(CANDIDATE)
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
                    "eval_group": group,
                    "dataset": key[0],
                    "condition": key[1],
                    "pp_delta": float(c["pearson_pert"]) - float(a["pearson_pert"]),
                    "mmd_delta": float(c["test_mmd"]) - float(a["test_mmd"]),
                }
            )
    return pd.DataFrame(rows)


def corr_safe(df: pd.DataFrame, feature: str) -> float | None:
    part = df[[feature, "pp_delta"]].dropna()
    if len(part) < 10 or part[feature].nunique() < 3:
        return None
    return float(part[feature].corr(part["pp_delta"], method="spearman"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = pd.read_csv(BUDGET_ROWS)
    budget = budget[budget["budget"].isin([500, 1000])].copy()
    outcomes = load_outcomes()
    joined = outcomes.merge(budget, on=["dataset", "condition"], how="inner")
    joined.to_csv(OUT_JOIN, index=False)

    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for budget_value, bpart in joined.groupby("budget", sort=True):
        for eval_group, part in bpart.groupby("eval_group", sort=True):
            item = {
                "budget": int(budget_value),
                "eval_group": str(eval_group),
                "rows": int(len(part)),
                "datasets": int(part["dataset"].nunique()),
                "mean_pp_delta": float(part["pp_delta"].mean()) if len(part) else None,
                "mean_mmd_delta": float(part["mmd_delta"].mean()) if len(part) else None,
                "dataset_min_pp_delta": float(part.groupby("dataset")["pp_delta"].mean().min()) if len(part) else None,
                "feature_spearman": {feature: corr_safe(part, feature) for feature in FEATURES},
            }
            rows.append(item)
            if item["rows"] < 100:
                reasons.append(f"budget{budget_value}_{eval_group}_overlap_rows_lt_100")
            if item["datasets"] < 6:
                reasons.append(f"budget{budget_value}_{eval_group}_datasets_lt_6")
            if item["mean_pp_delta"] is None or item["mean_pp_delta"] <= 0.0:
                reasons.append(f"budget{budget_value}_{eval_group}_mean_pp_not_positive")
            if item["dataset_min_pp_delta"] is not None and item["dataset_min_pp_delta"] < -0.02:
                reasons.append(f"budget{budget_value}_{eval_group}_dataset_min_lt_neg0p02")

    status = "observable_gene_budget_stability_readiness_fail_no_gpu" if reasons else "observable_gene_budget_stability_readiness_pass_prepare_cpu_pair_gate"
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
            "budget_rows": str(BUDGET_ROWS),
            "anchor_internal": str(ANCHOR),
            "candidate_internal": str(CANDIDATE),
        },
        "joined_rows": int(len(joined)),
        "joined_datasets": int(joined["dataset"].nunique()) if len(joined) else 0,
        "rows": rows,
        "reasons": sorted(set(reasons)),
        "outputs": {"join": str(OUT_JOIN), "report": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Observable Gene-Budget Stability Readiness Gate 20260630",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only readiness check for observable gene-budget stability.",
        "- Uses mean-matched/split-half gene-budget controls and internal cap120 candidate-vs-anchor condition outcomes.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Joined Outcome Rows",
        "",
        f"- joined rows: `{payload['joined_rows']}`",
        f"- joined datasets: `{payload['joined_datasets']}`",
        "",
        "| budget | eval group | rows | datasets | mean pp | mean MMD | dataset min | split-half rho | energy/shuffle rho | HVG-mean rho |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        spearman = row["feature_spearman"]
        lines.append(
            f"| {row['budget']} | `{row['eval_group']}` | {row['rows']} | {row['datasets']} | "
            f"{fmt(row['mean_pp_delta'])} | {fmt(row['mean_mmd_delta'])} | {fmt(row['dataset_min_pp_delta'])} | "
            f"{fmt(spearman.get('split_half_jaccard'))} | {fmt(spearman.get('response_energy_over_shuffled_mean'))} | "
            f"{fmt(spearman.get('hvg_minus_mean_matched_mean'))} |"
        )
    lines.extend(["", "## Decision", ""])
    if payload["reasons"]:
        lines.append("Current artifacts do not support a GPU-ready observable gene-budget stability route.")
        lines.extend(f"- reason: `{reason}`" for reason in payload["reasons"])
    else:
        lines.append("Readiness passed; next step would be a stricter matched-pair CPU gate before GPU.")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- join rows: `{OUT_JOIN}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": payload["reasons"], "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
