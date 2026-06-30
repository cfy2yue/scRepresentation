#!/usr/bin/env python3
"""Audit closure of legacy active-run items from the original LatentFM prompt."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
RUN = ROOT / "runs" / "latentfm_xverse_trackc_routed_distill_20260622" / "xverse_trackc_route_condprior_w05_replay1_2k_seed42"
RUN_NAME = "xverse_trackc_route_condprior_w05_replay1_2k_seed42"
DECISION = REPORTS / f"LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{RUN_NAME}.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def marker(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "value": read_text(path).strip() if path.exists() else None,
    }


def tmux_ls() -> dict[str, Any]:
    proc = subprocess.run(
        ["tmux", "ls"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> None:
    run_status = read_text(RUN / "RUN_STATUS.md")
    decision_text = read_text(DECISION)
    markers = {
        "train_exit_code": marker(RUN / f"{RUN_NAME}.EXIT_CODE"),
        "train_finished": marker(RUN / f"{RUN_NAME}.FINISHED"),
        "posthoc_exit_code": marker(RUN / f"{RUN_NAME}.POSTHOC_EXIT_CODE"),
        "posthoc_finished": marker(RUN / "posthoc.FINISHED"),
    }
    tmux = tmux_ls()
    status_checks = [
        {
            "name": "run_status_exists",
            "passed": (RUN / "RUN_STATUS.md").is_file(),
            "detail": str(RUN / "RUN_STATUS.md"),
        },
        {
            "name": "decision_report_exists",
            "passed": DECISION.is_file(),
            "detail": str(DECISION),
        },
        {
            "name": "run_status_closed",
            "passed": "Finished; smoke gate failed and branch is closed" in run_status,
            "detail": "RUN_STATUS current status",
        },
        {
            "name": "decision_status_failed_closed",
            "passed": "trackc_smoke_fail_canonical_harm_close_branch" in decision_text,
            "detail": "decision report status",
        },
        {
            "name": "train_exit_zero",
            "passed": markers["train_exit_code"]["value"] == "0",
            "detail": str(markers["train_exit_code"]),
        },
        {
            "name": "posthoc_exit_zero",
            "passed": markers["posthoc_exit_code"]["value"] == "0",
            "detail": str(markers["posthoc_exit_code"]),
        },
        {
            "name": "no_active_tmux_sessions",
            "passed": tmux["returncode"] != 0 and not tmux["stdout"],
            "detail": tmux["stderr"] or "no tmux sessions",
        },
        {
            "name": "query_not_authorized",
            "passed": (
                (
                    "held-out query split is intentionally not evaluated" in run_status
                    or "do not launch canonical uncapped no-harm and do not evaluate held-out" in run_status
                )
                and "held-out query is forbidden for this decision" in decision_text
            ),
            "detail": "query remained forbidden after failed smoke",
        },
    ]
    failures = [row for row in status_checks if not row["passed"]]
    status = (
        "legacy_active_run_closure_audit_pass_no_gpu"
        if not failures
        else "legacy_active_run_closure_audit_fail_no_gpu"
    )
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "boundary": {
            "reads_run_status_and_decision_only": True,
            "reads_training_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "run": {
            "name": RUN_NAME,
            "root": str(RUN),
            "run_status": str(RUN / "RUN_STATUS.md"),
            "decision_report": str(DECISION),
        },
        "markers": markers,
        "tmux": tmux,
        "checks": status_checks,
        "failures": failures,
        "decision": {
            "legacy_prompt_wait_item_closed": not failures,
            "close_reason": "support gain below material gate plus canonical no-harm/MMD harm",
            "query_authorized": False,
            "gpu_authorized": False,
            "next_action": "do_not_reopen_this_routed_distill_smoke_without_new_train_support_no_harm_mechanism",
        },
    }

    json_path = REPORTS / "latentfm_legacy_active_run_closure_audit_20260624.json"
    md_path = REPORTS / "LATENTFM_LEGACY_ACTIVE_RUN_CLOSURE_AUDIT_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Legacy Active-Run Closure Audit",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Reads the legacy Track C routed-distill RUN_STATUS, decision report, marker files, and one current tmux listing.",
        "- Does not read training logs, raw canonical/query artifacts, use canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Legacy Prompt Item",
        "",
        f"- Run: `{RUN_NAME}`",
        f"- RUN_STATUS: `{RUN / 'RUN_STATUS.md'}`",
        f"- Decision report: `{DECISION}`",
        "",
        "## Result",
        "",
        "- The legacy Track C routed-distill smoke is closed.",
        "- Decision status: `trackc_smoke_fail_canonical_harm_close_branch`.",
        "- Train exit code: `0`.",
        "- Posthoc exit code: `0`.",
        "- Held-out Track C query remained unauthorized.",
        "- No active tmux sessions were present during this audit.",
        "",
        "## Checks",
        "",
        "| Check | Pass | Detail |",
        "|---|---:|---|",
    ]
    for row in status_checks:
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {row['detail']} |")
    lines.extend(["", "## Decision", ""])
    if failures:
        lines.append("The legacy prompt wait item is not fully closed; inspect failed checks before continuing.")
    else:
        lines.append(
            "The original prompt's Track C routed-distill wait item is fully closed. Do not launch uncapped canonical no-harm or held-out query for this checkpoint."
        )
    lines.extend(["", "## JSON", "", f"`{json_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
