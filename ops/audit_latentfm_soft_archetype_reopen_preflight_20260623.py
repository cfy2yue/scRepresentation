#!/usr/bin/env python3
"""CPU-only preflight for reopening archetype as a soft state-prior branch.

This is a short read-only synthesis over existing archetype audits.  It does
not fit new archetypes and does not authorize GPU work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RESID_JSON = ROOT / "reports/latentfm_residualized_archetype_cpu_audit_light_20260621.json"
CONS_JSON = ROOT / "reports/latentfm_archetype_consensus_cpu_audit_20260621.json"
BG_JSON = ROOT / "reports/latentfm_xverse_background_state_residual_consensus_gate_20260622.json"
OUT_JSON = ROOT / "reports/latentfm_soft_archetype_reopen_preflight_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_REOPEN_PREFLIGHT_20260623.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def fmt(value: Any) -> str:
    value = fnum(value)
    return "NA" if value is None else f"{value:+.6f}"


def best_residualized(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results") or []
    return max(
        rows,
        key=lambda row: (
            fnum(row.get("focus_entropy_min")) or -999.0,
            -(fnum(row.get("focus_max_cluster_fraction_max")) or 999.0),
            fnum(row.get("seed_ari_mean")) or -999.0,
        ),
    )


def best_consensus(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results") or []
    return max(rows, key=lambda row: fnum(row.get("median_ari")) or -999.0)


def bg_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    deltas = payload.get("paired_deltas") or []
    interesting = [
        row
        for row in deltas
        if row.get("candidate") == "background_gene_interact_ridge"
        and row.get("baseline") in {"dataset_mean", "gene_raw_mean", "background_only_ridge"}
    ]
    return {
        "status": (payload.get("decision") or {}).get("status"),
        "reasons": (payload.get("decision") or {}).get("reasons") or [],
        "key_deltas": interesting,
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    resid = payload["best_residualized"]
    cons = payload["best_consensus"]
    bg = payload["background_state_gate"]
    if (fnum(resid.get("focus_entropy_min")) or 0.0) < 0.50:
        reasons.append("residualized_focus_coverage_still_weak")
    if (fnum(resid.get("seed_ari_mean")) or 0.0) < 0.35:
        reasons.append("residualized_seed_stability_too_low_for_hard_labels")
    if (fnum(cons.get("median_ari")) or 0.0) < 0.35:
        reasons.append("consensus_hard_label_stability_too_low")
    if bg.get("status") != "cpu_gate_pass":
        reasons.append("background_state_predictive_gate_failed")
    status = "soft_archetype_reopen_preflight_no_gpu" if reasons else "soft_archetype_reopen_preflight_unexpected_gpu_review"
    return {
        "status": status,
        "gpu_authorization": "none",
        "reasons": reasons,
        "next_action": "implement_soft_assignment_predictive_cpu_gate",
    }


def render(payload: dict[str, Any]) -> str:
    resid = payload["best_residualized"]
    cons = payload["best_consensus"]
    bg = payload["background_state_gate"]
    lines = [
        "# LatentFM Soft Archetype Reopen Preflight",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Inputs",
        "",
        f"- residualized audit: `{payload['inputs']['residualized_json']}`",
        f"- consensus audit: `{payload['inputs']['consensus_json']}`",
        f"- background-state gate: `{payload['inputs']['background_state_json']}`",
        "",
        "## Evidence",
        "",
        f"- best residualized row: residualization `{resid.get('residualization')}`, K `{resid.get('k')}`, "
        f"focus entropy min `{fmt(resid.get('focus_entropy_min'))}`, "
        f"focus max fraction `{fmt(resid.get('focus_max_cluster_fraction_max'))}`, "
        f"seed ARI `{fmt(resid.get('seed_ari_mean'))}`, gate `{resid.get('gate_status')}`",
        f"- best hard-label consensus: K `{cons.get('k')}`, median ARI `{fmt(cons.get('median_ari'))}`, "
        f"p10 ARI `{fmt(cons.get('p10_ari'))}`, gate `{cons.get('gate_status')}`",
        f"- background-state predictive gate: `{bg.get('status')}`",
        "",
        "## Decision Reasons",
        "",
    ]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(
        [
            "",
            "## Consequence",
            "",
            "Old hard-archetype GPU adapters remain closed. The only allowed next",
            "archetype work is a new CPU gate over soft assignments or continuous",
            "state features with a shuffled-feature ablation and internal proxy",
            "baseline comparisons.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = {
        "inputs": {
            "residualized_json": str(RESID_JSON),
            "consensus_json": str(CONS_JSON),
            "background_state_json": str(BG_JSON),
        },
        "best_residualized": best_residualized(load(RESID_JSON)),
        "best_consensus": best_consensus(load(CONS_JSON)),
        "background_state_gate": bg_gate_summary(load(BG_JSON)),
    }
    payload["decision"] = decide(payload)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
