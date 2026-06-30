#!/usr/bin/env python3
"""Summarize active LatentFM candidate pool closure status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_active_candidate_pool_closure_20260624.json"
OUT_MD = REPORTS / "LATENTFM_ACTIVE_CANDIDATE_POOL_CLOSURE_20260624.md"


INPUTS = {
    "stale_gpu_passes": "latentfm_stale_gpu_pass_consumption_audit_20260624.json",
    "trackc_row_reliability_v2": "latentfm_trackc_row_reliability_v2_gate_20260624.json",
    "trackc_exogenous_row_qa_v3": "latentfm_trackc_exogenous_row_qa_v3_gate_20260624.json",
    "global_noharm_positive_class": "latentfm_global_noharm_positive_class_inventory_20260624.json",
    "scaling_source_resolved": "latentfm_scaling_source_resolved_matched_estimand_gate_20260624.json",
    "scaling_noharm_surrogate_v2": "latentfm_scaling_noharm_surrogate_v2_gate_20260624.json",
    "scaling_provenance_tail_sentinel": "latentfm_scaling_provenance_tail_sentinel_gate_20260624.json",
    "stale_pending_cleanup": "latentfm_stale_pending_cleanup_20260624.json",
    "post_locke_portfolio": "latentfm_post_locke_portfolio_decision_20260624.json",
}


def load(name: str) -> dict[str, Any]:
    path = REPORTS / name
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def status_of(obj: dict[str, Any]) -> str:
    return str(obj.get("status") or (obj.get("decision") or {}).get("status") or "unknown")


def main() -> int:
    objs = {key: load(name) for key, name in INPUTS.items()}
    rows = [
        {
            "branch": "stale_gpu_passes",
            "status": status_of(objs["stale_gpu_passes"]),
            "gpu_ready": False,
            "evidence": "old pass artifacts were consumed by later fail/no-gpu decisions",
        },
        {
            "branch": "trackc_support_row_quality",
            "status": status_of(objs["trackc_exogenous_row_qa_v3"]),
            "gpu_ready": False,
            "evidence": "V2 and V3 row gates have no tail-safe rule-spec; V3 best still min pp -0.079082",
        },
        {
            "branch": "global_noharm_surrogate",
            "status": status_of(objs["global_noharm_positive_class"]),
            "gpu_ready": False,
            "evidence": "0 nontrivial positive no-harm rows in historical frozen no-harm inventory",
        },
        {
            "branch": "scaling_matched_estimand",
            "status": status_of(objs["scaling_source_resolved"]),
            "gpu_ready": False,
            "evidence": "background/type NMI 0.570487, dataset-min pp -0.231049, no tail sentinel pass",
        },
        {
            "branch": "pending_placeholders",
            "status": status_of(objs["stale_pending_cleanup"]),
            "gpu_ready": False,
            "evidence": "soft-exposure seed robustness and cap60 seed44 confirmation are stale placeholders",
        },
    ]
    immediate_gpu_candidates = [row for row in rows if row["gpu_ready"]]
    status = "active_candidate_pool_closed_no_immediate_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_existing_reports_only": True,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "rows": rows,
        "n_immediate_gpu_candidates": len(immediate_gpu_candidates),
        "decision": {
            "current_best_default": "xverse_8k_anchor",
            "next_action": "new mechanism CPU gate or manuscript/failure-map consolidation",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Active Candidate Pool Closure",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of existing closure reports.",
        "- Does not train, infer, launch GPU, read canonical multi, or read held-out Track C query.",
        "",
        "## Branch Rows",
        "",
        "| branch | status | GPU ready | evidence |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['branch']}` | `{row['status']}` | `{row['gpu_ready']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- immediate GPU candidates: `{len(immediate_gpu_candidates)}`",
            "- current deployable/default: `xverse_8k_anchor`",
            "- next valid GPU launch requires a materially new train-only gate with explicit negative controls and tail safety.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
