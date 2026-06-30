#!/usr/bin/env python3
"""Build claim-controlled figures for exact response-information scaling."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
JOIN_CSV = ROOT / "reports/exact_response_information_clustered_ci_20260628/exact_response_information_outcome_join_rows.csv"
ASSOC_CSV = ROOT / "reports/exact_response_information_clustered_ci_20260628/exact_response_information_clustered_association_rows.csv"
LODO_CSV = ROOT / "reports/exact_response_information_clustered_ci_20260628/exact_response_information_lodo_rows.csv"
OUT_DIR = ROOT / "reports/exact_response_information_claim_report_20260628"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_CLAIM_REPORT_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_exact_response_information_claim_report_20260628.json"


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


def scatter_panel(df: pd.DataFrame, x: str, y: str, out_path: Path) -> None:
    markers = {
        "count_smokes": "o",
        "protocol_matrix": "s",
        "truecell_budget_curve": "^",
    }
    colors = {
        "condition_exposure": "#2c7fb8",
        "true_cell_count": "#d95f0e",
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=160)
    for (source, axis), part in df.groupby(["source_family", "axis_family"], sort=True):
        ax.scatter(
            part[x],
            part[y],
            s=48,
            marker=markers.get(source, "o"),
            c=colors.get(axis, "#636363"),
            edgecolor="black",
            linewidth=0.45,
            label=f"{source} / {axis}",
            alpha=0.86,
        )
    clean = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.shape[0] >= 3:
        coef = np.polyfit(clean[x].to_numpy(dtype=float), clean[y].to_numpy(dtype=float), deg=1)
        xs = np.linspace(float(clean[x].min()), float(clean[x].max()), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#252525", linewidth=1.3, alpha=0.8)
    ax.axhline(0, color="#969696", linewidth=0.8, linestyle="--")
    ax.set_xlabel("exact train-condition coverage fraction")
    ax.set_ylabel(y.replace("_", " "))
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = pd.read_csv(JOIN_CSV)
    assoc = pd.read_csv(ASSOC_CSV)
    lodo = pd.read_csv(LODO_CSV)

    fig_mmd = OUT_DIR / "exact_condition_fraction_vs_family_mmd_delta.png"
    fig_tail = OUT_DIR / "exact_condition_fraction_vs_tail_score.png"
    scatter_panel(joined, "exact_condition_fraction", "family_mmd_delta", fig_mmd)
    scatter_panel(joined, "exact_condition_fraction", "tail_score", fig_tail)

    def assoc_row(predictor: str, outcome: str) -> dict[str, Any]:
        row = assoc[(assoc["predictor"] == predictor) & (assoc["outcome"] == outcome)]
        return row.iloc[0].to_dict() if not row.empty else {}

    primary_mmd = assoc_row("exact_condition_fraction", "family_mmd_delta")
    primary_tail = assoc_row("exact_condition_fraction", "tail_score")
    primary_lodo = lodo[
        (lodo["predictor"] == "exact_condition_fraction")
        & (lodo["outcome"] == "family_mmd_delta")
    ]
    same_sign_rate = float(primary_lodo["same_sign"].mean()) if not primary_lodo.empty else float("nan")

    allowed_claims = [
        "Exact raw-expression train-condition coverage is now measurable for a broad subset of existing split designs.",
        "In the frozen downstream outcome table, higher exact coverage is associated with lower tail/MMD risk under split-cluster bootstrap.",
        "The effect is a scaling-law candidate and mechanism hypothesis, not a final model-improvement proof.",
    ]
    prohibited_claims = [
        "Do not claim HVG-specific superiority: abundance-ranked genes are nearly equivalent to HVG-ranked genes.",
        "Do not claim that top-1000 response coverage itself robustly predicts better downstream metrics; its clustered CI crosses zero.",
        "Do not launch GPU or select checkpoints from this evidence alone.",
        "Do not use oracle response ranking for training or feature selection.",
    ]
    next_gates = [
        "Add clustered/LODO figure generation to the scaling-law manuscript evidence folder.",
        "Design a matched split family where exact coverage varies while source/background and condition count are controlled.",
        "Only after an external audit approves a leakage-safe launcher should any GPU smoke be considered.",
    ]

    payload = {
        "created_at": now_cst(),
        "status": "exact_response_information_claim_report_ready_no_gpu",
        "figures": [str(fig_mmd), str(fig_tail)],
        "primary_mmd": primary_mmd,
        "primary_tail": primary_tail,
        "primary_lodo_same_sign_rate": same_sign_rate,
        "allowed_claims": allowed_claims,
        "prohibited_claims": prohibited_claims,
        "next_gates": next_gates,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Exact Response-Information Claim Report",
        "",
        f"Created: {payload['created_at']}",
        "",
        "Status: `exact_response_information_claim_report_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only claim-control artifact over completed exact coverage and clustered CI outputs.",
        "* No train/infer/GPU/canonical multi/Track C query/checkpoint selection.",
        "",
        "## Key Evidence",
        "",
        f"* `exact_condition_fraction -> family_mmd_delta`: rho `{fmt_float(primary_mmd.get('rho'))}`, clustered CI `["
        f"{fmt_float(primary_mmd.get('cluster_boot_ci95_low'))}, {fmt_float(primary_mmd.get('cluster_boot_ci95_high'))}]`.",
        f"* `exact_condition_fraction -> tail_score`: rho `{fmt_float(primary_tail.get('rho'))}`, clustered CI `["
        f"{fmt_float(primary_tail.get('cluster_boot_ci95_low'))}, {fmt_float(primary_tail.get('cluster_boot_ci95_high'))}]`.",
        f"* Primary family-MMD LODO same-sign rate: `{fmt_float(same_sign_rate)}`.",
        "",
        "## Allowed Claims",
        "",
    ]
    lines.extend([f"* {claim}" for claim in allowed_claims])
    lines.extend(["", "## Prohibited Claims", ""])
    lines.extend([f"* {claim}" for claim in prohibited_claims])
    lines.extend(["", "## Next Gates", ""])
    lines.extend([f"* {gate}" for gate in next_gates])
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"* `{fig_mmd}`",
            f"* `{fig_tail}`",
            "",
            "## Inputs",
            "",
            f"* `{JOIN_CSV}`",
            f"* `{ASSOC_CSV}`",
            f"* `{LODO_CSV}`",
            "",
            "## JSON",
            "",
            f"* `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print("status exact_response_information_claim_report_ready_no_gpu")


if __name__ == "__main__":
    main()
