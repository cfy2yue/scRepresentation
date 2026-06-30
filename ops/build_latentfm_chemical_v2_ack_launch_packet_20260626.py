#!/usr/bin/env python3
"""Build an ACK-gated launch packet for chemical scaffold V2 controls.

Short CPU/report task. Reads existing protocol/inventory artifacts only. It
does not train, infer, read canonical multi, read Track C query, or use GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_chemical_v2_ack_launch_packet_20260626.json"
OUT_MD = REPORTS / "LATENTFM_CHEMICAL_V2_ACK_LAUNCH_PACKET_20260626.md"

PROTOCOL_JSON = REPORTS / "latentfm_chemical_v2_fixedstep_launcher_protocol_audit_20260625.json"
INVENTORY_JSON = REPORTS / "latentfm_current_gpu_candidate_inventory_20260625.json"
CPU_UNLOCK_JSON = REPORTS / "latentfm_chemical_unseen_scaffold_v2_cpu_unlock_20260625.json"
DECISION_JSON = REPORTS / "latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_decision_20260625.json"

ACK = "launch_v2_fixedstep_controls_after_protocol_review"
COMMAND = f"""LATENTFM_CHEM_V2_ARMS=real_morgan512:43,real_morgan512:44 \\
LATENTFM_CHEM_V2_FIXEDSTEP_ACK={ACK} \\
bash ops/launch_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.sh"""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_row(payload: dict[str, Any], branch: str) -> dict[str, Any]:
    for row in payload.get("rows", []):
        if row.get("branch") == branch:
            return row
    return {}


def main() -> int:
    protocol = read_json(PROTOCOL_JSON)
    inventory = read_json(INVENTORY_JSON)
    cpu_unlock = read_json(CPU_UNLOCK_JSON)
    decision = read_json(DECISION_JSON)
    chem_row = inventory_row(inventory, "chemical_unseen_scaffold_v2_fixedstep_controls")

    checks = protocol.get("checks", [])
    all_protocol_checks_pass = bool(checks) and all(bool(row.get("pass")) for row in checks)
    seeds = sorted(int(row["split_seed"]) for row in cpu_unlock.get("rows", []) if row.get("status") == "ok")
    cache_names = ["real_morgan512"] + [row.get("name") for row in cpu_unlock.get("control_caches", [])]
    cache_names = sorted({str(x) for x in cache_names if x})
    packet_ready = (
        protocol.get("status") == "chemical_v2_fixedstep_launcher_protocol_safe_ack_still_required"
        and all_protocol_checks_pass
        and chem_row.get("state") == "protocol_safe_ack_required"
        and inventory.get("immediate_gpu_candidate_count") == 0
        and set(seeds) >= {43, 44}
    )
    payload = {
        "status": "chemical_v2_ack_launch_packet_ready_ack_required" if packet_ready else "chemical_v2_ack_launch_packet_not_ready",
        "gpu_authorized": False,
        "ack_required": ACK,
        "command": COMMAND,
        "resource_plan": {
            "current_cap_physical_gpus": 2,
            "first_batch_arms": ["real_morgan512:43", "real_morgan512:44"],
            "launcher_need_gpus": 2,
            "launcher_jobs_per_gpu": 1,
            "threads_per_arm": 3,
            "fresh_audit_required_at_launch": True,
        },
        "protocol": {
            "status": protocol.get("status"),
            "all_checks_pass": all_protocol_checks_pass,
            "check_count": len(checks),
        },
        "inventory": {
            "status": inventory.get("status"),
            "immediate_gpu_candidate_count": inventory.get("immediate_gpu_candidate_count"),
            "chemical_v2_state": chem_row.get("state"),
            "chemical_v2_latest_status": chem_row.get("latest_status"),
        },
        "cpu_unlock": {
            "status": cpu_unlock.get("status"),
            "seeds": seeds,
            "descriptor_caches": cache_names,
        },
        "decision_summary": {
            "status": decision.get("status"),
            "action": decision.get("action"),
        },
        "gate": {
            "after_first_real_batch": "run fixed-step summarizer; if real seed43/44 fail, close branch",
            "promotion": "only if real descriptors pass, launch shuffled/random controls; mechanism claim requires real-vs-control margin and external review",
            "forbidden": [
                "canonical multi selection",
                "Track C query use",
                "candidate best.pt adjudication",
                "deployable or NM scaling-law claim from first real batch alone",
            ],
        },
        "sources": {
            "protocol_audit": str(PROTOCOL_JSON),
            "inventory": str(INVENTORY_JSON),
            "cpu_unlock": str(CPU_UNLOCK_JSON),
            "pending_decision": str(DECISION_JSON),
            "launcher": str(ROOT / "ops/launch_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.sh"),
            "summarizer": str(ROOT / "ops/summarize_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.py"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Chemical V2 ACK Launch Packet",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized now: `{payload['gpu_authorized']}`",
        "",
        "## Purpose",
        "",
        "Launch the first ACK-gated chemical unseen-scaffold V2 fixed-step control batch only if the user explicitly approves the reviewed protocol.",
        "",
        "## Why This Is The Only Prepared GPU Route",
        "",
        f"- Current inventory status: `{payload['inventory']['status']}`.",
        f"- Immediate non-ACK GPU candidate count: `{payload['inventory']['immediate_gpu_candidate_count']}`.",
        f"- Chemical V2 inventory state: `{payload['inventory']['chemical_v2_state']}`.",
        f"- Protocol audit: `{payload['protocol']['status']}`, checks pass `{payload['protocol']['all_checks_pass']}` ({payload['protocol']['check_count']} checks).",
        "",
        "## Exact ACK Command",
        "",
        "```bash",
        COMMAND,
        "```",
        "",
        "## Resource Plan",
        "",
        "- Fresh 3-sample GPU audit at launch is mandatory.",
        "- Current cap: at most 2 physical GPUs for this block.",
        "- First batch: `real_morgan512:43` and `real_morgan512:44`.",
        "- Launcher asks for 2 GPU slots and uses 1 training process per GPU.",
        "- Each arm sets 3 CPU threads; total launcher CPU use is within the 24-core temporary cap.",
        "",
        "## Evaluation Boundary",
        "",
        "- Independent V2 seed43/44 unseen-scaffold splits with zero drug/scaffold overlap.",
        "- Fixed candidate checkpoint policy: `latest.pt` only; `best.pt` is not used for adjudication.",
        "- `TRAIN_EVAL_ENABLED=0`; train-time IID eval does not select checkpoints.",
        "- No canonical multi and no Track C query.",
        "",
        "## Gate",
        "",
        "- First real batch is only a mechanism screen.",
        "- If real seed43/44 fail the fixed-step summarizer, close the branch.",
        "- If real seed43/44 pass, launch shuffled/random Morgan512 controls before any chemical-scaling claim.",
        "- Any claim after controls still needs external review and no-harm/failure-case reporting.",
        "",
        "## Sources",
        "",
    ]
    for key, value in payload["sources"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
