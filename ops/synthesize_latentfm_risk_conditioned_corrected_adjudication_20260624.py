#!/usr/bin/env python3
"""Corrected adjudication for the risk-conditioned LatentFM portfolio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_JSON = ROOT / "reports/latentfm_risk_conditioned_general_exposure_smoke_decision_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_risk_conditioned_corrected_adjudication_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_CONDITIONED_CORRECTED_ADJUDICATION_20260624.md"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def main() -> int:
    payload = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    rows = payload["rows"]
    best = max(
        rows,
        key=lambda row: (
            row["metrics"].get("cross_pp_delta_vs_anchor") or -999.0,
            row["metrics"].get("family_gene_pp_delta_vs_anchor") or -999.0,
            -(row["metrics"].get("family_gene_mmd_delta_vs_anchor") or 999.0),
        ),
    )
    best_m = best["metrics"]
    failed = {item["name"]: item["reasons"] for item in payload["decision"].get("failed", [])}
    tian_norman_reasons = failed.get("xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42", [])
    status = "mutate_not_promote"
    canonical_allowed = False
    next_action = "cpu_only_risk_stratified_gate_before_any_canonical_noharm"

    out = {
        "status": status,
        "source_decision_status": payload["status"],
        "source_json": str(SRC_JSON),
        "canonical_allowed": canonical_allowed,
        "next_action": next_action,
        "bug_fix": {
            "issue": "summarizer previously treated zero target_dataset_mmd_harm_rows as missing via `or 999`",
            "fixed": True,
            "remaining_tian_norman_fail_reasons": tian_norman_reasons,
        },
        "best_mechanism_signal": {
            "name": best["name"],
            "cross_pp_delta_vs_anchor": best_m.get("cross_pp_delta_vs_anchor"),
            "family_gene_pp_delta_vs_anchor": best_m.get("family_gene_pp_delta_vs_anchor"),
            "family_gene_mmd_delta_vs_anchor": best_m.get("family_gene_mmd_delta_vs_anchor"),
            "target_dataset_mean_mmd_delta": best_m.get("target_dataset_mean_mmd_delta"),
            "target_dataset_mean_pp_delta": best_m.get("target_dataset_mean_pp_delta"),
            "target_dataset_mmd_harm_rows": best_m.get("target_dataset_mmd_harm_rows"),
            "risk_dataset_harm_count": best_m.get("risk_dataset_harm_count"),
        },
        "decision": {
            "recommendation": "do_not_promote_or_run_canonical_now",
            "reason": "tian-norman has strong aggregate/internal target signal but still exceeds predeclared broad risk-dataset harm gate",
            "allowed_next_gate": "CPU-only corrected risk-stratified adjudication using existing train-only internal outputs; no canonical/multi/query",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Conditioned Corrected Adjudication",
        "",
        f"Status: `{status}`",
        f"Next action: `{next_action}`",
        f"Canonical allowed now: `{canonical_allowed}`",
        "",
        "## Boundary",
        "",
        "- CPU-only adjudication over completed train-only internal outputs.",
        "- No canonical metrics, canonical multi, or Track C query were read.",
        "- Source decision JSON was generated after fixing the zero-as-missing gate bug.",
        "",
        "## Corrected Gate Note",
        "",
        "- Previous summarizer logic used falsy `or` defaults, so `target_dataset_mmd_harm_rows = 0` could be treated as missing.",
        "- The summarizer now distinguishes `None` from zero.",
        "- After correction, the tian-norman arm no longer fails target-harm rows; it still fails `too_many_risk_datasets_harmed`.",
        "",
        "## Best Mechanism Signal",
        "",
        "| run | cross pp | family pp | family MMD | Tian MMD | Tian pp | Tian harm rows | risk harm count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{best['name']}` | {fmt(best_m.get('cross_pp_delta_vs_anchor'))} | "
            f"{fmt(best_m.get('family_gene_pp_delta_vs_anchor'))} | "
            f"{fmt(best_m.get('family_gene_mmd_delta_vs_anchor'))} | "
            f"{fmt(best_m.get('target_dataset_mean_mmd_delta'))} | "
            f"{fmt(best_m.get('target_dataset_mean_pp_delta'))} | "
            f"{best_m.get('target_dataset_mmd_harm_rows')} | {best_m.get('risk_dataset_harm_count')} |"
        ),
        "",
        "## Decision",
        "",
        "- Do not promote the risk-conditioned branch.",
        "- Do not run frozen canonical no-harm yet; the predeclared broad tail-risk gate still fails.",
        "- Keep the tian-norman arm as positive mechanism evidence, not a checkpoint-selection candidate.",
        "- Next work, if any, should be a CPU-only risk-stratified gate that separates target-risk success from non-target risk-dataset harms and predeclares an acceptable severity criterion.",
        "",
        "## Sources",
        "",
        f"- Source decision JSON: `{SRC_JSON}`",
        f"- Source decision report: `{ROOT / 'reports/LATENTFM_RISK_CONDITIONED_GENERAL_EXPOSURE_SMOKE_DECISION_20260624.md'}`",
        f"- Output JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
