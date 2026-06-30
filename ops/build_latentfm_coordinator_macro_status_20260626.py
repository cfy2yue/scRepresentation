#!/usr/bin/env python3
"""Build a coordinator macro-status report for active LatentFM work.

Short CPU/report task. It reads completed decision reports and takes a
lightweight tmux/GPU/RAM snapshot. It does not train, infer, inspect
checkpoints, read canonical multi, read Track C query outputs, or use GPU.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
OUT_MD = REPORTS / "LATENTFM_COORDINATOR_MACRO_STATUS_20260626.md"
OUT_JSON = REPORTS / "latentfm_coordinator_macro_status_20260626.json"
OUT_DIR = REPORTS / "coordinator_macro_status_20260626"
OUT_BRANCH_CSV = OUT_DIR / "branch_status.csv"
OUT_NEXT_CSV = OUT_DIR / "next_gate_queue.csv"

INVENTORY_JSON = REPORTS / "latentfm_current_gpu_candidate_inventory_20260625.json"
SLATE_JSON = REPORTS / "latentfm_next_action_slate_20260626.json"
SCALING_JSON = REPORTS / "latentfm_scaling_final_package_index_20260626.json"
CHEM_ACK_JSON = REPORTS / "latentfm_chemical_v2_ack_launch_packet_20260626.json"
TRACKC_DECISION_MD = REPORTS / "LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_route_condprior_w05_replay1_2k_seed42.md"
TRACKC_RUN_STATUS = RUNS / "latentfm_xverse_trackc_routed_distill_20260622/xverse_trackc_route_condprior_w05_replay1_2k_seed42/RUN_STATUS.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_text(path: Path, limit: int = 20000) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[:limit]


def run_cmd(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # pragma: no cover - defensive for environment variance
        return 999, str(exc)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def status_contains(text: str, needle: str) -> bool:
    return needle in text


def main() -> None:
    inventory = read_json(INVENTORY_JSON)
    slate = read_json(SLATE_JSON)
    scaling = read_json(SCALING_JSON)
    chem_ack = read_json(CHEM_ACK_JSON)
    trackc_text = read_text(TRACKC_DECISION_MD)
    trackc_status_text = read_text(TRACKC_RUN_STATUS)
    scout = read_json(REPORTS / "latentfm_condition_level_reliability_source_scout_20260626.json")
    acquisition = read_json(REPORTS / "latentfm_external_condition_artifact_acquisition_slate_20260626.json")
    gwt = read_json(REPORTS / "latentfm_gwt_condition_reliability_artifact_preflight_20260627.json")

    tmux_code, tmux_out = run_cmd(["tmux", "ls"])
    gpu_code, gpu_out = run_cmd(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    )
    free_code, free_out = run_cmd(["free", "-h"])

    no_tmux = tmux_code != 0 and "no server running" in tmux_out
    trackc_closed = status_contains(trackc_text, "trackc_smoke_fail_canonical_harm_close_branch")

    branch_rows = [
        {
            "branch": "track_a_default_model",
            "status": "default_best_active",
            "gpu_now": "false",
            "evidence": "Current inventory and scaling package keep xverse_8k_anchor as default/deployable.",
            "next_gate": "new non-duplicate CPU mechanism or external artifact before GPU",
        },
        {
            "branch": "track_c_routed_distill",
            "status": "closed_before_query" if trackc_closed else "needs_manual_audit",
            "gpu_now": "false",
            "evidence": "support gain below gate and canonical no-harm failed; held-out query not evaluated",
            "next_gate": "materially new support-only mechanism on safe trainselect split",
        },
        {
            "branch": "scaling_package",
            "status": scaling.get("status", "unknown"),
            "gpu_now": "false",
            "evidence": f"{scaling.get('artifact_count', 'unknown')} artifacts indexed; missing {scaling.get('missing_artifact_count', 'unknown')}; ready as mechanism/failure-map not checkpoint promotion",
            "next_gate": "manuscript polish or genuinely new condition-level external artifact from source-scout",
        },
        {
            "branch": "chemical_v2_fixedstep",
            "status": chem_ack.get("status", "ack_required"),
            "gpu_now": "false",
            "evidence": "only prepared GPU route, but exact protocol ACK required",
            "next_gate": "exact ACK then real_morgan512 seed43/44 before shuffled/random controls",
        },
        {
            "branch": "external_artifact_v3",
            "status": "gwt_failed_sciplex_or_condition_level_source_next"
            if str(gwt.get("status", "")).endswith("fail_no_gpu")
            else acquisition.get("status") or scout.get("status", "source_leads_only_no_condition_level_artifact"),
            "gpu_now": "false",
            "evidence": "condition-level source scout reports zero local-ready candidates; acquisition slate defines P0/P1 leads; GWT P0 was materialized but failed strict tail/MMD preflight",
            "next_gate": "materialize SciPlex dose/time or another truly condition-level small-table artifact, then strict CPU preflight before GPU",
        },
    ]

    next_rows = []
    for action in slate.get("actions", []):
        next_rows.append(
            {
                "priority": action.get("priority"),
                "action": action.get("name"),
                "type": action.get("type"),
                "gpu_now": str(bool(action.get("gpu_authorized_now", False))).lower(),
                "next_gate": action.get("next_gate"),
                "stop_rule": action.get("fail_close"),
            }
        )
    next_rows.append(
        {
            "priority": "R",
            "action": "scaling_manuscript_polish_from_final_package",
            "type": "reporting",
            "gpu_now": "false",
            "next_gate": "use LATENTFM_SCALING_FINAL_PACKAGE_INDEX_20260626.md as entry point",
            "stop_rule": "do not promote checkpoint or claim monotonic law",
        }
    )

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": "latentfm_coordinator_macro_status_no_immediate_gpu",
        "current_default_model": scaling.get("current_default_model", "xverse_8k_anchor"),
        "inventory_status": inventory.get("status"),
        "immediate_gpu_candidate_count": inventory.get("immediate_gpu_candidate_count", 0),
        "gpu_authorized": False,
        "active_tmux": not no_tmux,
        "tmux_snapshot": tmux_out,
        "gpu_snapshot": gpu_out,
        "free_snapshot": free_out,
        "trackc_closed_before_query": trackc_closed,
        "trackc_run_status_has_finished": "Finished; smoke gate failed" in trackc_status_text,
        "branch_rows": branch_rows,
        "next_gate_queue": next_rows,
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
    }

    write_csv(OUT_BRANCH_CSV, branch_rows, ["branch", "status", "gpu_now", "evidence", "next_gate"])
    write_csv(OUT_NEXT_CSV, next_rows, ["priority", "action", "type", "gpu_now", "next_gate", "stop_rule"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Coordinator Macro Status",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Default/deployable model: `{payload['current_default_model']}`",
        "",
        "Immediate GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only macro checkpoint.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- Resource snapshot is for scheduling awareness only, not long-job log polling.",
        "",
        "## Resource Snapshot",
        "",
        f"- active tmux: `{not no_tmux}`",
        "- GPU snapshot:",
        "",
        "```text",
        gpu_out,
        "```",
        "",
        "## Branch Status",
        "",
        "| branch | status | GPU now | next gate |",
        "|---|---|---|---|",
    ]
    for row in branch_rows:
        lines.append(
            f"| `{row['branch']}` | `{row['status']}` | `{row['gpu_now']}` | {row['next_gate']} |"
        )

    lines += [
        "",
        "## Next Gate Queue",
        "",
        "| priority | action | type | GPU now | next gate |",
        "|---:|---|---|---|---|",
    ]
    for row in next_rows:
        lines.append(
            f"| {row['priority']} | `{row['action']}` | `{row['type']}` | `{row['gpu_now']}` | {row['next_gate']} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "- No immediate non-ACK GPU launch is legal from current evidence, despite available GPU capacity.",
        "- Track C routed-distill is closed before query; do not evaluate held-out query for that checkpoint.",
        "- Scaling is report-ready as mechanism/failure-map; use the final package index as entry point.",
        "- The only prepared GPU route is chemical V2 fixed-step after exact ACK; otherwise acquire a genuinely new condition-level artifact and run strict CPU gates.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Branch status CSV: `{OUT_BRANCH_CSV}`",
        f"- Next gate queue CSV: `{OUT_NEXT_CSV}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
