#!/usr/bin/env python3
"""Synthesize ZSCAPE dynamic-response laws from frozen report artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/zscape_response_law_panel_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strict = _read(
        ROOT / "reports/zscape_strict_ot_decomposition_gate_20260628/zscape_strict_ot_decomposition_rows.csv"
    )
    query = _read(
        ROOT
        / "reports/zscape_embryo_heldout_dynamic_specificity_gate_20260628/zscape_embryo_heldout_dynamic_specificity_query_rows.csv"
    )
    celltype_corr = _read(
        ROOT
        / "reports/zscape_celltype_information_axis_20260628/zscape_celltype_information_axis_correlations.csv"
    )
    celltype_rows = _read(
        ROOT / "reports/zscape_celltype_information_axis_20260628/zscape_celltype_information_axis_rows.csv"
    )

    n_rows = int(len(strict))
    dynamic_pass = strict[strict.get("dynamic_response_gate", False).astype(bool)] if not strict.empty else pd.DataFrame()
    embryo_pass = strict[strict.get("embryo_vector_gate", False).astype(bool)] if not strict.empty else pd.DataFrame()
    specificity_pass = (
        query[query.get("specificity_gate", False).astype(bool)]
        if not query.empty
        else pd.DataFrame()
    )
    effect_pass = (
        query[query.get("effect_gate", False).astype(bool)]
        if not query.empty
        else pd.DataFrame()
    )
    confounded = (
        strict[strict.get("strict_dynamic_class", "").astype(str).str.contains("confounded", na=False)]
        if not strict.empty
        else pd.DataFrame()
    )

    hvg_dynamic = {}
    if not dynamic_pass.empty:
        for col in ["hvg1000_response_energy_share", "hvg2000_response_energy_share", "hvg4000_response_energy_share"]:
            hvg_dynamic[col] = float(pd.to_numeric(dynamic_pass[col], errors="coerce").mean())
    entropy_rho = ""
    entropy_p = ""
    if not celltype_corr.empty:
        hit = celltype_corr[
            (celltype_corr["x"] == "subtype_entropy_norm")
            & (celltype_corr["y"] == "mean_response_energy_total")
        ]
        if not hit.empty:
            entropy_rho = f"{float(hit.iloc[0]['spearman_rho']):.4f}"
            entropy_p = f"{float(hit.iloc[0]['p']):.4g}"

    laws = [
        {
            "law": "state_preserved_response",
            "status": "supported_but_specificity_blocked",
            "evidence": (
                f"{len(dynamic_pass)}/{n_rows} strict rows pass dynamic response; "
                f"passing rows: {', '.join(dynamic_pass.get('row_id', pd.Series(dtype=str)).astype(str).tolist())}"
            ),
            "blocker": "module/pathway specificity still fails; no model loss.",
            "next_gate": "transfer within-state OT margin into a residualized table with wrong-time/lineage/target controls.",
            "model_translation": "diagnostic or sampling covariate only until specificity and no-harm pass.",
        },
        {
            "law": "reliability_is_necessary_not_specific",
            "status": "supported_as_filter_only",
            "evidence": f"{len(embryo_pass)}/{n_rows} rows pass embryo-vector reliability.",
            "blocker": "broad reliability is non-discriminative because confounded comparator rows also pass.",
            "next_gate": "combine embryo reliability with specificity margin and composition/time residualization.",
            "model_translation": "minimum quality filter, not standalone scaling x or training weight.",
        },
        {
            "law": "specific_module_response",
            "status": "failed_current_modules",
            "evidence": f"{len(effect_pass)}/{len(query)} query effects positive, but {len(specificity_pass)}/{len(query)} specificity gates pass.",
            "blocker": "wrong-target/time/lineage thresholds catch up to heldout module effects.",
            "next_gate": "rediscover modules in train embryos only, then heldout embryos plus wrong controls.",
            "model_translation": "no pathway/program regularizer from current modules.",
        },
        {
            "law": "magnitude_is_not_information",
            "status": "supported_by_negative_controls",
            "evidence": f"{len(confounded)} high-effect rows are classified as composition/time-confounded comparators.",
            "blocker": "response magnitude alone mixes composition, time, and QC effects.",
            "next_gate": "residualize response norm against composition fraction, time tangent, lineage, and library/QC.",
            "model_translation": "split evaluation into composition, within-state expression response, and no-harm.",
        },
        {
            "law": "compact_observable_response",
            "status": "hypothesis_only_controls_required",
            "evidence": (
                "dynamic-row mean HVG shares: "
                + ", ".join(f"{k}={v:.4f}" for k, v in hvg_dynamic.items())
            ),
            "blocker": "HVG/response concentration is entangled with abundance, variance, detection, and RawFM MMD harm.",
            "next_gate": "abundance/variance/detection-matched gene-budget controls plus no-harm.",
            "model_translation": "RawFM mask/curriculum ingredient only after matched controls pass.",
        },
        {
            "law": "substate_information_density",
            "status": "weak_biological_hint_report_only",
            "evidence": (
                f"celltype panel n={len(celltype_rows)}; subtype entropy vs response energy rho={entropy_rho}, p={entropy_p}."
            ),
            "blocker": "n is small and lineage/source/QC confounding is unresolved.",
            "next_gate": "state/support matrix with source LODO, cell-count matching, and metadata-availability placebo.",
            "model_translation": "candidate sampling principle, not launch axis.",
        },
    ]

    law_df = pd.DataFrame(laws)
    csv_path = OUT_DIR / "zscape_response_law_panel_rows.csv"
    law_df.to_csv(csv_path, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": "zscape_response_law_panel_complete_no_gpu",
        "gpu_authorized": False,
        "rows_csv": str(csv_path),
        "strict_rows": n_rows,
        "dynamic_pass_rows": int(len(dynamic_pass)),
        "embryo_reliability_pass_rows": int(len(embryo_pass)),
        "module_specificity_pass_queries": int(len(specificity_pass)),
    }
    json_path = OUT_DIR / "zscape_response_law_panel_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# ZSCAPE Response-Law Panel",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis over frozen ZSCAPE report artifacts.",
        "- No training, no inference, no new OT pairing, no GPU, no checkpoint selection.",
        "- Purpose: make the ab initio dynamic-response laws falsifiable and model-facing.",
        "",
        "## Law Panel",
        "",
        "| law | status | evidence | blocker | next gate | model translation |",
        "|---|---|---|---|---|---|",
    ]
    for row in laws:
        lines.append(
            f"| {row['law']} | {row['status']} | {row['evidence']} | "
            f"{row['blocker']} | {row['next_gate']} | {row['model_translation']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No ZSCAPE law is currently ready to become a LatentFM/RawFM loss.",
            "The strongest reusable insight is within-state response plus specificity/no-harm gates; reliability and magnitude alone are insufficient.",
            "",
            "## Outputs",
            "",
            f"- rows: `{csv_path}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    (OUT_DIR / "LATENTFM_ZSCAPE_RESPONSE_LAW_PANEL_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
