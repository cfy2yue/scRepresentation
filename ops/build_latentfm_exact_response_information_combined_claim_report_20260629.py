#!/usr/bin/env python3
"""Build claim-controlled figures for combined exact response-information scaling."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
JOIN_CSV = (
    ROOT
    / "reports/exact_response_information_clustered_ci_combined_20260628/"
    / "exact_response_information_outcome_join_rows.csv"
)
ASSOC_CSV = (
    ROOT
    / "reports/exact_response_information_clustered_ci_combined_20260628/"
    / "exact_response_information_clustered_association_rows.csv"
)
LODO_CSV = (
    ROOT
    / "reports/exact_response_information_clustered_ci_combined_20260628/"
    / "exact_response_information_lodo_rows.csv"
)
IPW_MD = (
    ROOT
    / "reports/exact_response_information_ipw_missingness_combined_20260628/"
    / "LATENTFM_EXACT_RESPONSE_INFORMATION_IPW_MISSINGNESS_20260628.md"
)
MATCHED_MD = (
    ROOT
    / "reports/exact_coverage_strict_matched_draft_splits_combined_20260628/"
    / "LATENTFM_EXACT_COVERAGE_STRICT_MATCHED_DRAFT_SPLITS_20260628.md"
)
OUT_DIR = ROOT / "reports/exact_response_information_combined_claim_report_20260629"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_COMBINED_CLAIM_REPORT_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_exact_response_information_combined_claim_report_20260629.json"


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


def assoc_row(assoc: pd.DataFrame, predictor: str, outcome: str) -> dict[str, Any]:
    row = assoc[(assoc["predictor"] == predictor) & (assoc["outcome"] == outcome)]
    return row.iloc[0].to_dict() if not row.empty else {}


def scatter_panel(df: pd.DataFrame, x: str, y: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=170)
    markers = {
        "count_smokes": "o",
        "protocol_matrix": "s",
        "truecell_budget_curve": "^",
        "gene_only_budget": "D",
    }
    colors = {
        "condition_exposure": "#2c7fb8",
        "true_cell_count": "#d95f0e",
        "gene_budget": "#31a354",
    }
    label_seen: set[str] = set()
    group_cols = [col for col in ["source_family", "axis_family"] if col in df.columns]
    if len(group_cols) == 2:
        iterator = df.groupby(group_cols, sort=True)
    else:
        iterator = [("all", df)]
    for key, part in iterator:
        if isinstance(key, tuple):
            source, axis = key
            label = f"{source} / {axis}"
        else:
            source, axis, label = "all", "all", "all"
        ax.scatter(
            part[x],
            part[y],
            s=48,
            marker=markers.get(str(source), "o"),
            c=colors.get(str(axis), "#636363"),
            edgecolor="black",
            linewidth=0.45,
            alpha=0.86,
            label=label if label not in label_seen else None,
        )
        label_seen.add(label)
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = pd.read_csv(JOIN_CSV)
    assoc = pd.read_csv(ASSOC_CSV)
    lodo = pd.read_csv(LODO_CSV)

    fig_mmd = OUT_DIR / "combined_exact_condition_fraction_vs_family_mmd_delta.png"
    fig_tail = OUT_DIR / "combined_exact_condition_fraction_vs_tail_score.png"
    scatter_panel(joined, "exact_condition_fraction", "family_mmd_delta", fig_mmd)
    scatter_panel(joined, "exact_condition_fraction", "tail_score", fig_tail)

    primary_mmd = assoc_row(assoc, "exact_condition_fraction", "family_mmd_delta")
    primary_tail = assoc_row(assoc, "exact_condition_fraction", "tail_score")
    primary_pp = assoc_row(assoc, "exact_condition_fraction", "family_pp_delta")
    primary_lodo = lodo[
        (lodo["predictor"] == "exact_condition_fraction")
        & (lodo["outcome"] == "family_mmd_delta")
    ]
    same_sign_rate = float(primary_lodo["same_sign"].mean()) if not primary_lodo.empty else float("nan")

    allowed_claims = [
        "Exact raw-expression train-condition coverage is measurable across the combined 22-dataset coverage panel.",
        "In frozen downstream outcome summaries, higher exact coverage is associated with lower family-MMD and tail risk under split-cluster bootstrap.",
        "This is currently a scaling-law descriptor and mechanism hypothesis, not a model-improvement proof.",
    ]
    caveats = [
        "The IPW/missingness report says residual coverage keeps the family-MMD signal but tail becomes weaker and dataset-stratified permutations are not significant.",
        "The strict matched draft has only 165 matched pairs over 12 datasets, below the >=300 feasibility gate.",
        "The signal should not be used as an oracle feature selector for training; it can guide matched split design and reporting.",
    ]
    prohibited_claims = [
        "Do not claim monotonic dataset-size scaling.",
        "Do not claim HVG-specific superiority over abundance-matched genes from this exact-coverage result.",
        "Do not launch GPU or select checkpoints from this evidence alone.",
        "Do not use canonical multi or Track C query for selection.",
    ]
    next_gates = [
        "Build a strict matched split family or expand covered/uncovered matched pairs until the >=300-pair feasibility gate is met.",
        "Add exact coverage as a preregistered x variable in the scaling-law manuscript package with IPW and missingness caveats shown.",
        "Test whether ZSCAPE-derived state/OT variables add independent explanatory value beyond exact coverage and background/source controls.",
    ]

    payload = {
        "created_at": now_cst(),
        "status": "exact_response_information_combined_claim_report_ready_no_gpu",
        "n_join_rows": int(joined.shape[0]),
        "figures": [str(fig_mmd), str(fig_tail)],
        "primary_mmd": primary_mmd,
        "primary_tail": primary_tail,
        "primary_family_pp": primary_pp,
        "primary_lodo_same_sign_rate": same_sign_rate,
        "allowed_claims": allowed_claims,
        "caveats": caveats,
        "prohibited_claims": prohibited_claims,
        "next_gates": next_gates,
        "inputs": {
            "join_csv": str(JOIN_CSV),
            "assoc_csv": str(ASSOC_CSV),
            "lodo_csv": str(LODO_CSV),
            "ipw_md": str(IPW_MD),
            "matched_md": str(MATCHED_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Exact Response-Information Combined Claim Report",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        "Status: `exact_response_information_combined_claim_report_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only claim-control artifact over combined exact-coverage and clustered-CI outputs.",
        "- No training, inference, GPU, canonical multi selection, Track C query, or checkpoint selection.",
        "",
        "## Key Evidence",
        "",
        f"- Joined rows: `{payload['n_join_rows']}`.",
        f"- `exact_condition_fraction -> family_mmd_delta`: rho `{fmt_float(primary_mmd.get('rho'))}`, clustered CI `[{fmt_float(primary_mmd.get('cluster_boot_ci95_low'))}, {fmt_float(primary_mmd.get('cluster_boot_ci95_high'))}]`.",
        f"- `exact_condition_fraction -> tail_score`: rho `{fmt_float(primary_tail.get('rho'))}`, clustered CI `[{fmt_float(primary_tail.get('cluster_boot_ci95_low'))}, {fmt_float(primary_tail.get('cluster_boot_ci95_high'))}]`.",
        f"- `exact_condition_fraction -> family_pp_delta`: rho `{fmt_float(primary_pp.get('rho'))}`, clustered CI `[{fmt_float(primary_pp.get('cluster_boot_ci95_low'))}, {fmt_float(primary_pp.get('cluster_boot_ci95_high'))}]`.",
        f"- Primary family-MMD LODO same-sign rate: `{fmt_float(same_sign_rate)}`.",
        "",
        "## Allowed Claims",
        "",
    ]
    lines.extend([f"- {claim}" for claim in allowed_claims])
    lines.extend(["", "## Caveats", ""])
    lines.extend([f"- {caveat}" for caveat in caveats])
    lines.extend(["", "## Prohibited Claims", ""])
    lines.extend([f"- {claim}" for claim in prohibited_claims])
    lines.extend(["", "## Next Gates", ""])
    lines.extend([f"- {gate}" for gate in next_gates])
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- `{fig_mmd}`",
            f"- `{fig_tail}`",
            "",
            "## Inputs",
            "",
            f"- Join rows: `{JOIN_CSV}`",
            f"- Association rows: `{ASSOC_CSV}`",
            f"- LODO rows: `{LODO_CSV}`",
            f"- IPW/missingness caveat: `{IPW_MD}`",
            f"- Matched split feasibility caveat: `{MATCHED_MD}`",
            "",
            "## JSON",
            "",
            f"- `{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(OUT_MD), "figures": payload["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
