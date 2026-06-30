#!/usr/bin/env python3
"""Decision helper after the support-set task summary gate.

This helper is read-only and safe to run before or after the delayed checker.
It does not inspect active job logs or RUN_STATUS files. It only reads the
formal summary gate report, if present, and the predeclared MMD/no-harm
protocol, then emits the next branch decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SUMMARY_JSON = ROOT / "reports/latentfm_trackc_support_set_task_summary_gate_20260623.json"
MMD_PROTOCOL_JSON = ROOT / "reports/latentfm_trackc_support_set_task_mmd_noharm_protocol_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_task_after_summary_decision_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_TASK_AFTER_SUMMARY_DECISION_20260623.md"

PASS_STATUS = "trackc_support_set_task_summary_gate_pass_posthoc_mmd_gate_next_no_gpu"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> dict[str, Any]:
    summary = load_json(SUMMARY_JSON)
    protocol = load_json(MMD_PROTOCOL_JSON)
    summary_status = None
    if isinstance(summary, dict):
        summary_status = ((summary.get("decision") or {}).get("status") or summary.get("status"))

    if summary is None:
        status = "support_set_task_after_summary_waiting_for_formal_summary_gate"
        next_action = "wait_for_delayed_checker_or_run_guarded_checker_after_window"
        reasons = ["formal_summary_gate_json_missing"]
    elif summary_status == PASS_STATUS:
        status = "support_set_task_after_summary_prepare_query_free_mmd_noharm"
        next_action = "implement_or_run_query_free_mmd_noharm_posthoc_after_fresh_resource_audit"
        reasons = []
    else:
        status = "support_set_task_after_summary_close_summary_rule_no_gpu"
        next_action = "record_negative_evidence_and_pivot_to_materially_new_cpu_gate_or_reporting"
        reasons = (summary.get("decision") or {}).get("reasons") or ["summary_gate_status_not_pass"]

    return {
        "status": status,
        "summary_gate_status": summary_status,
        "gpu_authorization": "none",
        "next_action": next_action,
        "decision_reasons": reasons,
        "inputs": {
            "summary_json": str(SUMMARY_JSON),
            "summary_json_exists": SUMMARY_JSON.is_file(),
            "mmd_protocol_json": str(MMD_PROTOCOL_JSON),
            "mmd_protocol_exists": MMD_PROTOCOL_JSON.is_file(),
            "mmd_protocol_status": None if protocol is None else protocol.get("status"),
        },
        "boundary": [
            "No held-out query authorization is granted by this helper.",
            "A summary-gate pass only enables query-free MMD/no-harm posthoc implementation/launch after fresh resource audit.",
            "A summary-gate failure closes this support-set summary rule; do not retune alpha/checkpoint on support_val or query.",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Set Task After-Summary Decision",
        "",
        f"Status: `{payload['status']}`",
        f"Summary gate status: `{payload['summary_gate_status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next action: `{payload['next_action']}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in payload.get("decision_reasons") or [])
    lines.extend(["", "## Boundary", ""])
    lines.extend(f"- {item}" for item in payload["boundary"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
