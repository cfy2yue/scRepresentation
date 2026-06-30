#!/usr/bin/env python3
"""Reconcile ZSCAPE trajectory-time and preprocessing proxy-latent gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_TRAJECTORY_ALIGNMENT = (
    ROOT
    / "runs/zscape_expression_trajectory_time_gate_20260628"
    / "zscape_expression_trajectory_time_gate_20260628_084025"
    / "outputs/zscape_expression_trajectory_time_perturb_alignment.csv"
)
DEFAULT_PREPROCESSING_ALIGNMENT = (
    ROOT
    / "runs/zscape_latent_preprocessing_sensitivity_20260628"
    / "zscape_latent_preprocessing_sensitivity_20260628_133904"
    / "outputs/zscape_latent_preprocessing_sensitivity_alignment_rows.csv"
)
DEFAULT_PREPROCESSING_SUMMARY = (
    ROOT
    / "runs/zscape_latent_preprocessing_sensitivity_20260628"
    / "zscape_latent_preprocessing_sensitivity_20260628_133904"
    / "outputs/zscape_latent_preprocessing_sensitivity_summary.csv"
)
DEFAULT_OUT = ROOT / "reports/zscape_latent_proxy_reconciliation_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-alignment", type=Path, default=DEFAULT_TRAJECTORY_ALIGNMENT)
    parser.add_argument("--preprocessing-alignment", type=Path, default=DEFAULT_PREPROCESSING_ALIGNMENT)
    parser.add_argument("--preprocessing-summary", type=Path, default=DEFAULT_PREPROCESSING_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference-variant", default="ref_noqc_log1p_hvg2000")
    parser.add_argument("--min-sign-agreement-frac", type=float, default=0.80)
    parser.add_argument("--min-gate-agreement-frac", type=float, default=0.80)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(args.trajectory_alignment)
    prep = pd.read_csv(args.preprocessing_alignment)
    prep_ref = prep[prep["variant"].astype(str) == args.reference_variant].copy()
    summary = pd.read_csv(args.preprocessing_summary)

    rows: list[dict[str, Any]] = []
    for _, trow in traj.iterrows():
        row_id = str(trow["row_id"])
        prow = prep_ref[prep_ref["row_id"].astype(str) == row_id]
        if prow.empty:
            continue
        p = prow.iloc[0]
        t_cos = float(trow.get("cosine_to_lineage_time_vector", np.nan))
        p_cos = float(p.get("cosine_to_lineage_time_vector", np.nan))
        rows.append(
            {
                "row_id": row_id,
                "lineage": trow.get("lineage", ""),
                "target": trow.get("gene_target", trow.get("target", "")),
                "timepoint": trow.get("timepoint", ""),
                "trajectory_method": "matched_embryo_balanced_displacement_n128",
                "preprocessing_method": "all_row_centroid_displacement",
                "trajectory_cosine": t_cos,
                "preprocessing_cosine": p_cos,
                "cosine_delta_preprocessing_minus_trajectory": p_cos - t_cos,
                "sign_agreement": bool(np.sign(t_cos) == np.sign(p_cos)),
                "trajectory_margin": float(trow.get("cosine_margin_vs_wrong_lineage", np.nan)),
                "preprocessing_margin": float(p.get("cosine_margin_vs_wrong_lineage", np.nan)),
                "trajectory_gate": boolish(trow.get("alignment_gate", False)),
                "preprocessing_gate": boolish(p.get("alignment_gate", False)),
                "gate_agreement": boolish(trow.get("alignment_gate", False)) == boolish(p.get("alignment_gate", False)),
            }
        )
    out = pd.DataFrame(rows)
    primary = out[out["lineage"].isin(["mature fast muscle", "periderm"])].copy()
    periderm = out[out["lineage"].eq("periderm")].copy()
    sign_agree = float(primary["sign_agreement"].mean()) if len(primary) else float("nan")
    gate_agree = float(primary["gate_agreement"].mean()) if len(primary) else float("nan")
    periderm_gate_agree = float(periderm["gate_agreement"].mean()) if len(periderm) else float("nan")
    trajectory_periderm_pass = int(periderm["trajectory_gate"].sum()) if len(periderm) else 0
    preprocessing_periderm_pass = int(periderm["preprocessing_gate"].sum()) if len(periderm) else 0

    prep_summary_ref = summary[summary["variant"].astype(str) == args.reference_variant].iloc[0].to_dict()
    hvg_stable = float(summary["signature_corr_vs_ref_primary"].dropna().min()) >= 0.85
    qc_row = summary[summary["variant"].astype(str).eq("qc_log1p_hvg2000")]
    qc_stable = (not qc_row.empty) and float(qc_row.iloc[0]["signature_corr_vs_ref_primary"]) >= 0.99

    reconciled = (
        sign_agree >= args.min_sign_agreement_frac
        and gate_agree >= args.min_gate_agreement_frac
        and periderm_gate_agree >= args.min_gate_agreement_frac
    )
    status = (
        "zscape_latent_proxy_reconciliation_pass_no_gpu"
        if reconciled
        else "zscape_latent_proxy_reconciliation_fail_no_gpu"
    )

    row_csv = args.out_dir / "zscape_latent_proxy_reconciliation_rows.csv"
    json_path = args.out_dir / "zscape_latent_proxy_reconciliation_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_LATENT_PROXY_RECONCILIATION_20260628.md"
    out.to_csv(row_csv, index=False)
    result = {
        "timestamp_cst": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "reference_variant": args.reference_variant,
        "primary_sign_agreement_fraction": sign_agree,
        "primary_gate_agreement_fraction": gate_agree,
        "periderm_gate_agreement_fraction": periderm_gate_agree,
        "trajectory_periderm_alignment_pass": trajectory_periderm_pass,
        "preprocessing_periderm_alignment_pass": preprocessing_periderm_pass,
        "qc_signature_stable": bool(qc_stable),
        "hvg_signature_stable": bool(hvg_stable),
        "preprocessing_reference_summary": prep_summary_ref,
        "rows": str(row_csv),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Latent Proxy Reconciliation",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only reconciliation of existing ZSCAPE proxy-latent outputs.",
        "- No training, no inference, no true scFM embedding extraction, no canonical multi, and no Track C query.",
        "",
        "## Summary",
        "",
        f"- primary sign agreement: `{sign_agree:.4f}`.",
        f"- primary gate agreement: `{gate_agree:.4f}`.",
        f"- periderm gate agreement: `{periderm_gate_agree:.4f}`.",
        f"- trajectory periderm alignment pass: `{trajectory_periderm_pass}/{len(periderm)}`.",
        f"- preprocessing/all-centroid periderm alignment pass: `{preprocessing_periderm_pass}/{len(periderm)}`.",
        f"- QC signature stable: `{qc_stable}`; HVG signature stable: `{hvg_stable}`.",
        "",
        "## Interpretation",
        "",
    ]
    if reconciled:
        lines.append(
            "The two proxy-latent definitions are sufficiently concordant for a bounded design review, while still not replacing true scFM latent evidence."
        )
    else:
        lines.append(
            "The two proxy-latent definitions are not reconciled. Treat ZSCAPE as expression-space biology/information-axis evidence only; do not claim a latent temporal tangent or flow-matching constraint from current proxy latents."
        )
    lines.extend(
        [
            "",
            "## Row Comparison",
            "",
            "| row | lineage | target | traj cos | prep cos | traj gate | prep gate | sign agree | gate agree |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in out.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(row["lineage"]),
                    str(row["target"]),
                    f"{float(row['trajectory_cosine']):.4f}",
                    f"{float(row['preprocessing_cosine']):.4f}",
                    str(bool(row["trajectory_gate"])),
                    str(bool(row["preprocessing_gate"])),
                    str(bool(row["sign_agreement"])),
                    str(bool(row["gate_agreement"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- rows: `{row_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
