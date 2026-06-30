#!/usr/bin/env python3
"""Aggregate tail/no-harm reopenability audit for LatentFM.

CPU-only synthesis. Reads completed gate reports for tail protection, fallback,
richer priors, response-program projection, and scaling v2 gates. It decides
whether any non-duplicate tail/no-harm mechanism currently authorizes GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_tail_noharm_reopenability_audit_20260625.json"
OUT_MD = REPORTS / "LATENTFM_TAIL_NOHARM_REOPENABILITY_AUDIT_20260625.md"

GATES = [
    {
        "name": "uncertainty_nonnoop",
        "path": REPORTS / "latentfm_uncertainty_gated_anchor_fallback_nonnoop_gate_20260625.json",
        "family": "fallback",
    },
    {
        "name": "truecell_stratum",
        "path": REPORTS / "latentfm_truecell_stratum_tail_protection_gate_20260625.json",
        "family": "fallback",
    },
    {
        "name": "recurrent_gene_sentinel",
        "path": REPORTS / "latentfm_recurrent_gene_harm_sentinel_gate_20260625.json",
        "family": "sentinel",
    },
    {
        "name": "lodo_domain_conflict",
        "path": REPORTS / "latentfm_lodo_domain_conflict_gate_20260625.json",
        "family": "domain_consensus",
    },
    {
        "name": "truecell_riskrow_complementarity",
        "path": REPORTS / "latentfm_truecell_riskrow_complementarity_gate_20260625.json",
        "family": "risk_row",
    },
    {
        "name": "multiprior_tailrisk_mask",
        "path": REPORTS / "latentfm_multiprior_tailrisk_mask_gate_20260625.json",
        "family": "richer_prior",
    },
    {
        "name": "background_target_actionability",
        "path": REPORTS / "latentfm_background_target_actionability_gate_20260625.json",
        "family": "richer_prior",
    },
    {
        "name": "response_program_projection",
        "path": REPORTS / "latentfm_response_program_projection_gate_20260625.json",
        "family": "response_program",
    },
    {
        "name": "scaling_nested_condition_exposure_v2",
        "path": REPORTS / "latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json",
        "family": "scaling",
    },
    {
        "name": "scaling_source_resolved_estimand_v2",
        "path": REPORTS / "latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json",
        "family": "scaling",
    },
]


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "status": "missing", "gpu_authorized": False, "reasons": ["missing_report"]}
    with path.open() as f:
        data = json.load(f)
    data.setdefault("gpu_authorized", False)
    data.setdefault("reasons", [])
    return data


def short_metric(name: str, data: dict[str, Any]) -> str:
    s = data.get("summary") or {}
    if name == "uncertainty_nonnoop":
        return "enabled canonical footprint 0"
    if name == "truecell_stratum":
        rows = data.get("canonical_rows") or []
        enabled = sum(int(r.get("enabled_n") or r.get("enabled") or 0) for r in rows)
        return f"canonical_enabled={enabled}"
    if name == "recurrent_gene_sentinel":
        return f"AUROC={s.get('auc', 'NA')}, footprint={s.get('canonical_metadata_footprint', 'NA')}"
    if name == "lodo_domain_conflict":
        return f"pp={s.get('overall_pp_mean', data.get('overall', {}).get('pp_mean', 'NA'))}"
    if name == "truecell_riskrow_complementarity":
        return "protect_frac test/family below gate"
    if name == "multiprior_tailrisk_mask":
        primary = s.get("primary") or {}
        return f"LODO_AUROC={primary.get('auc', 'NA')}, pp={primary.get('sim_pp_mean', 'NA')}"
    if name == "background_target_actionability":
        high = data.get("high_actionability") or {}
        return f"high_pp={high.get('pp_mean', 'NA')}, min={high.get('dataset_min_pp', 'NA')}"
    if name == "response_program_projection":
        high = s.get("high_supported") or {}
        return f"pp={high.get('pp_mean', 'NA')}, min={high.get('dataset_min_pp', 'NA')}"
    if name == "scaling_nested_condition_exposure_v2":
        return f"cap120-cap30={s.get('cap120_minus_cap30_cross_pp', 'NA')}, lodo_min={s.get('mixed_lodo_dataset_min_pp', 'NA')}"
    if name == "scaling_source_resolved_estimand_v2":
        return f"pp={s.get('pp_delta_mean', 'NA')}, min={s.get('dataset_min_pp', 'NA')}"
    return ""


def main() -> int:
    rows = []
    any_authorized = False
    missing = []
    for gate in GATES:
        data = load(gate["path"])
        authorized = bool(data.get("gpu_authorized"))
        any_authorized = any_authorized or authorized
        if data.get("_missing"):
            missing.append(gate["name"])
        rows.append(
            {
                "name": gate["name"],
                "family": gate["family"],
                "path": str(gate["path"]),
                "status": data.get("status", "missing"),
                "gpu_authorized": authorized,
                "metric": short_metric(gate["name"], data),
                "reasons": data.get("reasons", []),
                "next_action": data.get("next_action") or data.get("next_actions"),
            }
        )

    closed_families = sorted({r["family"] for r in rows if not r["gpu_authorized"] and r["status"] != "missing"})
    reasons = []
    if any_authorized:
        reasons.append("at_least_one_tail_noharm_gate_authorized_gpu")
    else:
        reasons.append("no_tail_noharm_gate_authorizes_gpu")
    if missing:
        reasons.append("some_expected_reports_missing")

    status = (
        "tail_noharm_reopenability_gpu_candidate_present"
        if any_authorized
        else "tail_noharm_reopenability_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": any_authorized,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "rows": rows,
        "summary": {
            "n_gates": len(rows),
            "n_gpu_authorized": sum(1 for r in rows if r["gpu_authorized"]),
            "closed_families": closed_families,
            "missing": missing,
        },
        "reasons": reasons,
        "next_action": (
            "launch_only_the_authorized_gate_after_external_review_and_resource_audit"
            if any_authorized
            else "no tail/no-harm GPU launch; chemical V2 remains separate ACK-gated route; otherwise build final scaling/failure-map package or invent a materially new CPU-first mechanism"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Tail / No-Harm Reopenability Audit",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{any_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed gate reports.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Gate Rows",
        "",
        "| gate | family | status | gpu | metric |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['family']}` | `{row['status']}` | `{row['gpu_authorized']}` | {row['metric']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- closed families: `{closed_families}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
