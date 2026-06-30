#!/usr/bin/env python3
"""Summarize active LatentFM smoke decision artifacts.

This is a lightweight artifact integrator.  It does not inspect active logs,
tmux sessions, GPUs, or EXIT_CODE markers.  It reads only frozen decision/gate
artifacts when they exist and reports pending files otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_MD = ROOT / "reports/LATENTFM_ACTIVE_POSTHOC_DECISION_SUMMARY_20260623.md"
OUT_JSON = ROOT / "reports/latentfm_active_posthoc_decision_summary_20260623.json"


RUNS = [
    {
        "name": "trackc_residual_retry1",
        "track": "Track C",
        "decision_md": ROOT
        / "reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42_retry1.md",
        "gate_json": ROOT
        / "reports/latentfm_trackc_residual_operator_route_gap_gate_xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42_retry1.json",
        "promotion": "If support and canonical gates pass, proceed only to uncapped canonical no-harm; no query yet.",
        "fail_close": "If support/canonical gates fail, close residual retry and use the alternative support-conditioning CPU gate protocol.",
    },
    {
        "name": "tracka_scf_jiang_lowcount",
        "track": "Track A",
        "decision_md": ROOT
        / "reports/LATENTFM_TRACKA_SCF_JIANG_LOWCOUNT_ADAPTER_scfoundation_tracka_gene_shrink_k2_jiang_lowcount_adapter_2k_seed42_DECISION_20260623.md",
        "gate_json": ROOT
        / "reports/latentfm_tracka_scf_jiang_lowcount_adapter_scfoundation_tracka_gene_shrink_k2_jiang_lowcount_adapter_2k_seed42_gate_20260623.json",
        "promotion": "If crossbg pp >= +0.02 with no all-single/family/Jiang harm, compare against broad fallback before seed expansion.",
        "fail_close": "If crossbg gain is below gate or no-harm fails, close narrow Jiang-lowcount branch.",
    },
    {
        "name": "tracka_scf_dataset_negative",
        "track": "Track A",
        "decision_md": ROOT
        / "reports/LATENTFM_TRACKA_SCF_DATASET_NEGATIVE_ADAPTER_scfoundation_tracka_gene_shrink_k2_dataset_negative_adapter_2k_seed42_DECISION_20260623.md",
        "gate_json": ROOT
        / "reports/latentfm_tracka_scf_dataset_negative_adapter_scfoundation_tracka_gene_shrink_k2_dataset_negative_adapter_2k_seed42_gate_20260623.json",
        "promotion": "If broad fallback passes, require comparison against narrow Jiang-lowcount and fallback-dataset harm audit before promotion.",
        "fail_close": "If no-harm fails or fallback datasets show material harm, demote behind narrow branch or close.",
    },
]


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def first_heading(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def compact_gate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("status", "decision", "result", "gate_status"):
        if key in payload:
            out[key] = payload[key]
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None
    if metrics:
        out["metric_keys"] = sorted(metrics)[:12]
    return out


def terminal_gate_status(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    if isinstance(decision, dict):
        status = str(decision.get("status") or "")
        if status.endswith("_fail_close_branch") or status.endswith("_pass") or "gate_fail" in status:
            return status
    status = str(payload.get("status") or "")
    if status.endswith("_fail_close_branch") or status.endswith("_pass") or "gate_fail" in status:
        return status
    return None


def summarize() -> dict[str, Any]:
    rows = []
    for run in RUNS:
        decision_md = Path(run["decision_md"])
        gate_json = Path(run["gate_json"])
        gate_payload = load_json(gate_json)
        terminal_status = terminal_gate_status(gate_payload)
        if decision_md.exists():
            status = "decision_ready"
        elif terminal_status is not None:
            status = "terminal_gate_ready"
        elif gate_payload is not None:
            status = "gate_json_ready_decision_md_missing"
        else:
            status = "pending"
        rows.append(
            {
                "name": run["name"],
                "track": run["track"],
                "status": status,
                "decision_md": str(decision_md),
                "gate_json": str(gate_json),
                "decision_title": first_heading(decision_md),
                "gate_compact": compact_gate(gate_payload),
                "terminal_gate_status": terminal_status,
                "promotion": run["promotion"],
                "fail_close": run["fail_close"],
            }
        )
    ready = sum(1 for row in rows if row["status"] in {"decision_ready", "terminal_gate_ready"})
    return {
        "status": "all_decisions_ready" if ready == len(rows) else "posthoc_decisions_pending",
        "ready_decisions": ready,
        "total_runs": len(rows),
        "rows": rows,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Active Posthoc Decision Summary",
        "",
        f"Status: `{payload['status']}`",
        f"Ready decisions: `{payload['ready_decisions']}/{payload['total_runs']}`",
        "",
        "| run | track | status | decision | gate json |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['name']}` | {row['track']} | `{row['status']}` | "
            f"`{row['decision_md']}` | `{row['gate_json']}` |"
        )
    lines += ["", "## Next Action Rules", ""]
    for row in payload["rows"]:
        lines += [
            f"### {row['name']}",
            "",
            f"- promotion: {row['promotion']}",
            f"- fail-close: {row['fail_close']}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    payload = summarize()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ready": payload["ready_decisions"], "total": payload["total_runs"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
