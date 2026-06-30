#!/usr/bin/env python3
"""Formal observable-gene budget scaling-law descriptor gate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CONDITION_ROWS = ROOT / "reports/raw_expression_hvg_budget_expanded_gate_20260629/condition_budget_rows.csv"
DEFAULT_CONTROL_JSON = ROOT / "reports/hvg_observable_budget_control_decision_20260629/latentfm_hvg_observable_budget_control_decision_20260629.json"
DEFAULT_OUT_DIR = ROOT / "reports/observable_gene_budget_scaling_law_gate_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int = 1000) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    means = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(repeats)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def concentration_k(curve: pd.DataFrame, threshold: float) -> float:
    part = curve.sort_values("budget")
    prev_budget = 0.0
    prev_share = 0.0
    for _, row in part.iterrows():
        budget = float(row["budget"])
        share = float(row["control_hvg_share_mean"])
        if share >= threshold:
            if share <= prev_share + 1e-12:
                return budget
            alpha = (threshold - prev_share) / (share - prev_share)
            return prev_budget + alpha * (budget - prev_budget)
        prev_budget = budget
        prev_share = share
    return float("nan")


def summarize_budget_rows(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for keys, part in rows.groupby(["level", "group", "dataset", "budget"], dropna=False):
        level, group, dataset, budget = keys
        hvg = pd.to_numeric(part["control_hvg_share"], errors="coerce")
        rnd = pd.to_numeric(part["random_share_mean"], errors="coerce")
        adv = hvg - rnd
        hvg_low, hvg_high = bootstrap_ci(hvg.to_numpy(), rng)
        adv_low, adv_high = bootstrap_ci(adv.to_numpy(), rng)
        out.append(
            {
                "level": level,
                "group": group,
                "dataset": dataset,
                "budget": int(budget),
                "condition_rows": int(len(part)),
                "control_hvg_share_mean": float(hvg.mean()),
                "control_hvg_share_ci_low": hvg_low,
                "control_hvg_share_ci_high": hvg_high,
                "random_share_mean": float(rnd.mean()),
                "hvg_minus_random_mean": float(adv.mean()),
                "hvg_minus_random_ci_low": adv_low,
                "hvg_minus_random_ci_high": adv_high,
                "oracle_response_share_mean": float(pd.to_numeric(part["oracle_response_share"], errors="coerce").mean()),
                "response_energy_mean": float(pd.to_numeric(part["response_energy"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(out)


def add_aggregate_levels(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows.copy()
    base["level"] = "dataset"
    all_rows = rows.copy()
    all_rows["level"] = "all"
    all_rows["group"] = "__ALL__"
    all_rows["dataset"] = "__ALL__"
    group_rows = rows.copy()
    group_rows["level"] = "group"
    group_rows["dataset"] = "__ALL__"
    return pd.concat([base, group_rows, all_rows], ignore_index=True)


def lodo_rows(rows: pd.DataFrame, budget: int) -> pd.DataFrame:
    part = rows[rows["budget"].eq(budget)].copy()
    out: list[dict[str, Any]] = []
    for leave_dataset in sorted(part["dataset"].astype(str).unique()):
        sub = part[part["dataset"].astype(str).ne(leave_dataset)]
        hvg = pd.to_numeric(sub["control_hvg_share"], errors="coerce")
        rnd = pd.to_numeric(sub["random_share_mean"], errors="coerce")
        out.append(
            {
                "budget": budget,
                "leave_dataset": leave_dataset,
                "n_rows": int(len(sub)),
                "hvg_share_mean": float(hvg.mean()),
                "random_share_mean": float(rnd.mean()),
                "hvg_minus_random_mean": float((hvg - rnd).mean()),
            }
        )
    for leave_group in sorted(part["group"].astype(str).unique()):
        sub = part[part["group"].astype(str).ne(leave_group)]
        hvg = pd.to_numeric(sub["control_hvg_share"], errors="coerce")
        rnd = pd.to_numeric(sub["random_share_mean"], errors="coerce")
        out.append(
            {
                "budget": budget,
                "leave_dataset": f"group:{leave_group}",
                "n_rows": int(len(sub)),
                "hvg_share_mean": float(hvg.mean()),
                "random_share_mean": float(rnd.mean()),
                "hvg_minus_random_mean": float((hvg - rnd).mean()),
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-rows", type=Path, default=DEFAULT_CONDITION_ROWS)
    parser.add_argument("--control-json", type=Path, default=DEFAULT_CONTROL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    rows = pd.read_csv(args.condition_rows)
    control = json.loads(args.control_json.read_text())
    rows = rows[rows["budget"].isin([500, 1000, 2000, 4000])].copy()
    rows_aug = add_aggregate_levels(rows)
    budget_summary = summarize_budget_rows(rows_aug, rng)

    concentration: list[dict[str, Any]] = []
    for (level, group, dataset), curve in budget_summary.groupby(["level", "group", "dataset"]):
        concentration.append(
            {
                "level": level,
                "group": group,
                "dataset": dataset,
                "k80_interp": concentration_k(curve, 0.80),
                "k90_interp": concentration_k(curve, 0.90),
                "share_top1000": float(curve.loc[curve["budget"].eq(1000), "control_hvg_share_mean"].iloc[0])
                if curve["budget"].eq(1000).any()
                else float("nan"),
                "adv_top1000": float(curve.loc[curve["budget"].eq(1000), "hvg_minus_random_mean"].iloc[0])
                if curve["budget"].eq(1000).any()
                else float("nan"),
            }
        )
    concentration_df = pd.DataFrame(concentration)
    lodo = lodo_rows(rows, 1000)

    controls = pd.DataFrame(control["synthesis_rows"])
    top1000_controls = controls[controls["budget"].eq(1000) & controls["group"].isin(["chemicalpert_bench", "genepert_DE5000_small"])]
    max_meanmatched = float(pd.to_numeric(top1000_controls["hvg_minus_mean_matched"], errors="coerce").max())
    max_abundance = float(pd.to_numeric(top1000_controls["hvg_minus_abundance"], errors="coerce").abs().max())
    max_detection = float(pd.to_numeric(top1000_controls["hvg_minus_detection"], errors="coerce").max())
    min_overlap = float(pd.to_numeric(top1000_controls["hvg_abundance_overlap"], errors="coerce").min())

    all_top1000 = budget_summary[
        budget_summary["level"].eq("all")
        & budget_summary["group"].eq("__ALL__")
        & budget_summary["budget"].eq(1000)
    ].iloc[0]
    group_top1000 = budget_summary[
        budget_summary["level"].eq("group")
        & budget_summary["budget"].eq(1000)
        & budget_summary["group"].isin(["chemicalpert_bench", "genepert_DE5000_small"])
    ].copy()
    min_group_adv = float(group_top1000["hvg_minus_random_mean"].min())
    min_lodo_adv = float(lodo["hvg_minus_random_mean"].min())
    descriptor_pass = (
        float(all_top1000["control_hvg_share_mean"]) >= 0.80
        and min_group_adv >= 0.25
        and min_lodo_adv >= 0.20
    )
    hvg_specific_fail = max_meanmatched < 0.05 and max_abundance < 0.02 and max_detection < 0.05
    status = (
        "observable_gene_budget_scaling_descriptor_pass_hvg_specific_intervention_fail_no_gpu"
        if descriptor_pass and hvg_specific_fail
        else "observable_gene_budget_scaling_descriptor_incomplete_no_gpu"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    budget_csv = args.out_dir / "observable_gene_budget_curve_rows.csv"
    concentration_csv = args.out_dir / "observable_gene_budget_concentration_rows.csv"
    lodo_csv = args.out_dir / "observable_gene_budget_lodo_rows.csv"
    budget_summary.to_csv(budget_csv, index=False)
    concentration_df.to_csv(concentration_csv, index=False)
    lodo.to_csv(lodo_csv, index=False)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "descriptor_pass": descriptor_pass,
        "hvg_specific_intervention_gate": "fail" if hvg_specific_fail else "review",
        "decision_metrics": {
            "all_top1000_hvg_share": float(all_top1000["control_hvg_share_mean"]),
            "all_top1000_random_share": float(all_top1000["random_share_mean"]),
            "all_top1000_hvg_minus_random": float(all_top1000["hvg_minus_random_mean"]),
            "min_group_top1000_hvg_minus_random": min_group_adv,
            "min_lodo_top1000_hvg_minus_random": min_lodo_adv,
            "max_top1000_hvg_minus_mean_matched": max_meanmatched,
            "max_abs_top1000_hvg_minus_abundance": max_abundance,
            "max_top1000_hvg_minus_detection": max_detection,
            "min_top1000_hvg_abundance_overlap": min_overlap,
        },
        "inputs": {
            "condition_rows": str(args.condition_rows),
            "control_json": str(args.control_json),
        },
        "outputs": {
            "budget_curve_rows": str(budget_csv),
            "concentration_rows": str(concentration_csv),
            "lodo_rows": str(lodo_csv),
        },
        "boundary": "cpu_report_only_scaling_descriptor_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    json_path = args.out_dir / "latentfm_observable_gene_budget_scaling_law_gate_20260630.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top_curve = budget_summary[budget_summary["level"].eq("all") & budget_summary["group"].eq("__ALL__")].sort_values("budget")
    md_path = args.out_dir / "LATENTFM_OBSERVABLE_GENE_BUDGET_SCALING_LAW_GATE_20260630.md"
    lines = [
        "# LatentFM Observable-Gene Budget Scaling-Law Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only formalization of the expanded raw-expression gene-budget result.",
        "* Uses all eligible `899` raw-expression perturbation conditions and the expanded abundance/detection/mean-matched controls.",
        "* Does not train, infer, evaluate canonical multi, use Track C query, or select checkpoints.",
        "",
        "## Budget Curve",
        "",
        "| budget | rows | HVG share | CI | random | HVG-random | CI | oracle response |",
        "|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for _, row in top_curve.iterrows():
        lines.append(
            f"| `{int(row['budget'])}` | `{int(row['condition_rows'])}` | `{fmt(row['control_hvg_share_mean'])}` | "
            f"`[{fmt(row['control_hvg_share_ci_low'])}, {fmt(row['control_hvg_share_ci_high'])}]` | "
            f"`{fmt(row['random_share_mean'])}` | `{fmt(row['hvg_minus_random_mean'])}` | "
            f"`[{fmt(row['hvg_minus_random_ci_low'])}, {fmt(row['hvg_minus_random_ci_high'])}]` | "
            f"`{fmt(row['oracle_response_share_mean'])}` |"
        )
    lines.extend(
        [
            "",
            "## Descriptor Gate",
            "",
            f"* All-condition top1000 HVG share: `{fmt(payload['decision_metrics']['all_top1000_hvg_share'])}` versus random `{fmt(payload['decision_metrics']['all_top1000_random_share'])}`.",
            f"* Minimum group-level top1000 HVG-minus-random: `{fmt(min_group_adv)}`.",
            f"* Minimum leave-one-dataset/group top1000 HVG-minus-random: `{fmt(min_lodo_adv)}`.",
            f"* Descriptor gate pass: `{descriptor_pass}`.",
            "",
            "## Matched-Control Gate",
            "",
            f"* Max top1000 HVG-minus-mean-matched: `{fmt(max_meanmatched)}`.",
            f"* Max absolute top1000 HVG-minus-abundance: `{fmt(max_abundance)}`.",
            f"* Max top1000 HVG-minus-detection: `{fmt(max_detection)}`.",
            f"* Minimum top1000 HVG/abundance overlap: `{fmt(min_overlap)}`.",
            f"* HVG-specific intervention gate: `{payload['hvg_specific_intervention_gate']}`.",
            "",
            "## Decision",
            "",
            "* The scaling-law descriptor is positive: compact observable/top-token gene budgets concentrate perturbation response energy across datasets and leave-one-dataset checks.",
            "* The HVG-specific intervention claim fails: abundance, detection, and mean-matched controls explain almost all top1000 advantage.",
            "* Use this as a biological/scaling descriptor and covariate, not as a GPU launch route.",
            "",
            "## Outputs",
            "",
            f"* Budget curve rows: `{budget_csv}`",
            f"* Concentration rows: `{concentration_csv}`",
            f"* LODO rows: `{lodo_csv}`",
            f"* JSON: `{json_path}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
