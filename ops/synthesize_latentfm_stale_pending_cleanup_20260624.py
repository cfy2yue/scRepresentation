#!/usr/bin/env python3
"""CPU-only cleanup audit for stale pending LatentFM reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_stale_pending_cleanup_20260624.json"
OUT_MD = REPORTS / "LATENTFM_STALE_PENDING_CLEANUP_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    soft_pending = load_json(REPORTS / "latentfm_xverse_soft_exposure_seed_robustness_decision_20260624.json")
    soft_canon = load_json(REPORTS / "latentfm_xverse_soft_exposure_canonical_noharm_decision_20260624.json")
    cap60_pending = load_json(REPORTS / "latentfm_scaling_cap60_seed44_confirmation_decision_20260624.json")
    seed_gate = load_json(REPORTS / "latentfm_scaling_seed_matched_micro_matrix_gate_20260624.json")
    stale_audit = load_json(REPORTS / "latentfm_stale_gpu_pass_consumption_audit_20260624.json")
    noharm_surrogate = load_json(REPORTS / "latentfm_scaling_noharm_surrogate_v2_gate_20260624.json")

    rows: list[dict[str, Any]] = []

    soft_status = (soft_pending.get("decision") or {}).get("status") or soft_pending.get("status")
    soft_canon_status = (soft_canon.get("decision") or {}).get("status") or soft_canon.get("status")
    soft_reasons = []
    if soft_status == "pending":
        soft_reasons.append("pending_report_exists")
    if soft_canon_status != "soft_exposure_canonical_noharm_pass":
        soft_reasons.append(f"precondition_canonical_pass_not_met:{soft_canon_status}")
    rows.append(
        {
            "name": "soft_exposure_seed_robustness",
            "pending_status": soft_status,
            "current_resolution": "stale_pending_closed_no_launch",
            "precondition_status": soft_canon_status,
            "reasons": soft_reasons,
            "action": "do not wait for seed43/44; seed robustness launcher was guarded by a canonical pass that failed",
        }
    )

    cap60_status = (cap60_pending.get("decision") or {}).get("status") or cap60_pending.get("status")
    seed_gate_status = seed_gate.get("status")
    cap60_reasons = []
    if cap60_status == "pending":
        cap60_reasons.append("pending_report_exists")
    if seed_gate_status and seed_gate_status.endswith("fail_no_gpu"):
        cap60_reasons.append(f"seed_matched_gate_failed:{seed_gate_status}")
    if noharm_surrogate.get("status", "").endswith("fail_no_gpu"):
        cap60_reasons.append(f"noharm_surrogate_failed:{noharm_surrogate.get('status')}")
    rows.append(
        {
            "name": "cap60_seed44_confirmation",
            "pending_status": cap60_status,
            "current_resolution": "stale_pending_closed_no_launch",
            "precondition_status": seed_gate_status,
            "reasons": cap60_reasons,
            "action": "do not wait for seed44; cap/count seed expansion is closed by seed43 sign flip and canonical no-harm failure",
        }
    )

    status = "stale_pending_cleanup_complete_no_gpu"
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
        "stale_gpu_pass_audit_status": stale_audit.get("status"),
        "decision": {
            "next_action": "do not poll stale pending reports; use latest closure gates and next-mechanism slate",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Stale Pending Cleanup",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only cleanup audit over existing reports.",
        "- Does not modify old pending files, launch jobs, read canonical multi, or read held-out Track C query.",
        "",
        "## Rows",
        "",
        "| item | old pending status | current resolution | precondition/current gate | action | reasons |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['pending_status']}` | `{row['current_resolution']}` | `{row['precondition_status']}` | {row['action']} | `{','.join(row['reasons'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- GPU authorized: `False`",
            "- These pending reports are historical stale placeholders, not active waits.",
            "- Do not launch or wait on their listed seed runs unless a future, materially new gate reopens the family.",
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
