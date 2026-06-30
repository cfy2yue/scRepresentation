#!/usr/bin/env python3
"""CPU-only preflight for adaptive count-knee scaling.

This script checks whether existing scaling evidence is enough to authorize a
new GPU run, or whether a fresh CPU split/proxy gate is required first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SPLITS_JSON = ROOT / "reports/latentfm_xverse_scaling_splits_v2_20260624.json"
DECISION_JSON = ROOT / "reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json"
CANONICAL_JSON = ROOT / "reports/latentfm_xverse_scaling_canonical_noharm_decision_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_adaptive_count_knee_preflight_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_ADAPTIVE_COUNT_KNEE_PREFLIGHT_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> None:
    split_payload = load_json(SPLITS_JSON)
    decision = load_json(DECISION_JSON)
    canonical = load_json(CANONICAL_JSON)
    rows = decision.get("rows") or []
    by_run = {row.get("name"): row for row in rows}
    cap120 = by_run.get("xverse_scaling_cap120_all_3k_seed42") or {}
    cap30 = by_run.get("xverse_scaling_cap30_all_3k_seed42") or {}
    full = by_run.get("xverse_scaling_full_trainonly_3k_seed42") or {}
    type_bal = by_run.get("xverse_scaling_type_balanced_cap120_3k_seed42") or {}

    reasons = []
    if (decision.get("decision") or {}).get("status") != "count_scaling_internal_pass":
        reasons.append("cap120_internal_count_gate_not_passed")
    if (decision.get("full_extension_decision") or {}).get("status") == "full_trainonly_extension_fail":
        reasons.append("full_trainonly_extension_underperforms_cap120")
    if (decision.get("type_balance_extension_decision") or {}).get("status") == "type_balanced_extension_fail":
        reasons.append("type_balanced_extension_underperforms_cap120")
    if canonical and (canonical.get("status") or (canonical.get("decision") or {}).get("status")):
        cstatus = canonical.get("status") or (canonical.get("decision") or {}).get("status")
        if "fail" in str(cstatus):
            reasons.append("cap120_family_canonical_noharm_failed")
    # Current artifacts expose aggregate cap30/cap120/full outcomes, but not
    # enough per-dataset count-response observations to learn a deployable knee.
    reasons.extend(
        [
            "existing_scaling_evidence_has_only_aggregate_cap30_cap120_full_points",
            "no_nested_leave_one_dataset_knee_selection_artifact",
            "no_random_or_equal_count_knee_control_artifact",
        ]
    )

    status = "adaptive_count_knee_preflight_no_gpu_cpu_gate_required"
    result = {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "cpu_split_builder_and_trainonly_proxy_gate_only",
        "decision_reasons": reasons,
        "boundary": {
            "cpu_only_existing_reports": True,
            "canonical_metrics_used_for_noharm_context_not_selection": True,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_launch": False,
        },
        "inputs": {
            "splits": str(SPLITS_JSON),
            "decision": str(DECISION_JSON),
            "canonical_noharm": str(CANONICAL_JSON),
        },
        "key_rows": {
            "cap30": cap30,
            "cap120": cap120,
            "full": full,
            "type_balanced": type_bal,
        },
        "required_next_gate": {
            "name": "adaptive_count_knee_cpu_split_proxy_gate",
            "must_not_be": "another naked cap sweep",
            "requirements": [
                "build per-dataset/per-type caps from train-only counts and internal reliability only",
                "nested leave-one-dataset selection",
                "cross-bg pp >= cap120 + 0.003 on internal proxy",
                "family pp >= cap120 - 0.002 and family MMD <= cap120",
                "leave-one-dataset minimum delta >= -0.02",
                "random/equal-count knee controls must collapse",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM xverse Adaptive Count-Knee Preflight",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only aggregation of existing scaling reports.",
        "- Canonical no-harm is used only as context for why cap120 is not promotable, not as a new selection signal.",
        "- No held-out query, active logs, or GPU artifacts are read.",
        "",
        "## Existing Scaling Evidence",
        "",
        "| arm | cross-bg pp | cross-bg delta vs anchor | family pp | family MMD delta vs anchor |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in [("cap30", cap30), ("cap120", cap120), ("full", full), ("type_balanced", type_bal)]:
        cross = (row.get("groups") or {}).get("internal_val_cross_background_seen_gene_proxy") or {}
        family = (row.get("groups") or {}).get("internal_val_family_gene_proxy") or {}
        cross_cand = (cross.get("candidate") or {}).get("pearson_pert")
        family_cand = (family.get("candidate") or {}).get("pearson_pert")
        lines.append(
            f"| `{label}` | {fmt(cross_cand)} | {fmt(cross.get('delta_pearson_pert'))} | "
            f"{fmt(family_cand)} | {fmt(family.get('delta_mmd'))} |"
        )
    lines.extend([
        "",
        "## Decision Reasons",
        "",
    ])
    lines.extend([f"- `{reason}`" for reason in reasons])
    lines.extend([
        "",
        "## Required Next Gate",
        "",
        "- Build adaptive per-dataset/per-type caps from train-only counts and internal reliability only.",
        "- Use nested leave-one-dataset selection plus random/equal-count knee controls.",
        "- Require cross-bg pp `>= cap120 + 0.003`, family pp `>= cap120 - 0.002`, family MMD `<= cap120`, and leave-one-dataset min delta `>= -0.02`.",
        "- A pass may authorize one capped Track A smoke; this preflight itself authorizes no GPU.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
