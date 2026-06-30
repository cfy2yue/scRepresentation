#!/usr/bin/env python3
"""Summarize Track C support-context capped smokes without reading query data.

This script reads only run status markers and the support/canonical decision
JSONs produced by the per-run posthoc launcher. It does not inspect live logs
or held-out Track C query artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_trackc_support_context_20260622"
REPORT_DIR = ROOT / "reports"
RUN_NAMES = (
    "xverse_trackc_ctx_bridge_fm_2k_seed42",
    "xverse_trackc_ctx_bridge_ep025_2k_seed42",
    "xverse_trackc_ctx_bridge_ep050_2k_seed42",
)
PASS_STATUS = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
ROUTE_GAP_PASS_STATUS = "route_gap_gate_pass"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def f(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def indexed_rows(payload: dict[str, Any], table: str) -> dict[tuple[str, str], dict[str, Any]]:
    body = (payload.get("tables") or {}).get(table) or {}
    if isinstance(body, dict) and isinstance(body.get("rows"), list):
        rows = body.get("rows") or []
    elif isinstance(body, dict):
        rows = [row for row in body.values() if isinstance(row, dict)]
    else:
        rows = []
    return {(str(row.get("group")), str(row.get("metric"))): row for row in rows}


def pick_metric(payload: dict[str, Any], table: str, groups: tuple[str, ...], metric: str) -> dict[str, Any] | None:
    rows = indexed_rows(payload, table)
    for group in groups:
        row = rows.get((group, metric))
        if row and row.get("status") == "ok":
            return row
    for group in groups:
        row = rows.get((group, metric))
        if row:
            return row
    return None


def summarize_run(run_name: str) -> dict[str, Any]:
    run_root = RUN_ROOT / run_name
    decision_json = REPORT_DIR / f"latentfm_trackc_routed_distill_smoke_decision_{run_name}.json"
    decision_md = REPORT_DIR / f"LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{run_name}.md"
    route_gap_json = REPORT_DIR / f"latentfm_trackc_support_context_route_gap_gate_{run_name}.json"
    route_gap_md = REPORT_DIR / f"LATENTFM_TRACKC_SUPPORT_CONTEXT_ROUTE_GAP_GATE_{run_name}.md"
    train_exit = read_text(run_root / f"{run_name}.EXIT_CODE")
    posthoc_exit = read_text(run_root / f"{run_name}.POSTHOC_EXIT_CODE")

    row: dict[str, Any] = {
        "run": run_name,
        "run_root": str(run_root),
        "run_status": str(run_root / "RUN_STATUS.md"),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "decision_json": str(decision_json),
        "decision_md": str(decision_md),
        "decision_exists": decision_json.is_file(),
        "route_gap_json": str(route_gap_json),
        "route_gap_md": str(route_gap_md),
        "route_gap_exists": route_gap_json.is_file(),
        "base_status": None,
        "route_gap_status": None,
        "status": "decision_pending",
        "action": None,
        "reasons": [],
        "support_pp_delta": None,
        "support_pp_p_improvement": None,
        "support_mmd_p_harm": None,
        "canonical_single_pp_p_harm": None,
        "canonical_family_pp_p_harm": None,
        "wessels_support_pp_delta": None,
        "wessels_route_gap_closure": None,
        "norman_support_pp_delta": None,
        "norman_route_gap_closure": None,
    }
    if train_exit is not None and train_exit != "0":
        row["status"] = "train_failed"
        row["reasons"] = [f"train_exit_{train_exit}"]
        return row
    if posthoc_exit is not None and posthoc_exit != "0":
        row["status"] = "posthoc_failed"
        row["reasons"] = [f"posthoc_exit_{posthoc_exit}"]
        return row
    if not decision_json.is_file():
        if train_exit is None:
            row["status"] = "training_or_waiting"
        elif posthoc_exit is None:
            row["status"] = "posthoc_or_waiting"
        return row

    payload = load_json(decision_json)
    decision = payload.get("decision") or {}
    base_status = decision.get("status") or "decision_missing_status"
    row["base_status"] = base_status
    row["status"] = base_status
    row["action"] = decision.get("action")
    row["reasons"] = list(decision.get("reasons") or [])
    support_pp = pick_metric(payload, "support_split", ("test_multi", "test"), "pearson_pert")
    support_mmd = pick_metric(payload, "support_split", ("test_multi", "test"), "test_mmd_clamped")
    canonical_single_pp = pick_metric(payload, "canonical_split", ("test_single",), "pearson_pert")
    canonical_family_pp = pick_metric(payload, "canonical_family", ("family_gene",), "pearson_pert")
    if support_pp:
        row["support_pp_delta"] = support_pp.get("delta_mean")
        row["support_pp_p_improvement"] = support_pp.get("p_improvement")
    if support_mmd:
        row["support_mmd_p_harm"] = support_mmd.get("p_harm")
    if canonical_single_pp:
        row["canonical_single_pp_p_harm"] = canonical_single_pp.get("p_harm")
    if canonical_family_pp:
        row["canonical_family_pp_p_harm"] = canonical_family_pp.get("p_harm")
    if route_gap_json.is_file():
        route_payload = load_json(route_gap_json)
        route_decision = route_payload.get("decision") or {}
        row["route_gap_status"] = route_decision.get("status") or "route_gap_decision_missing_status"
        for summary_row in route_payload.get("summary") or []:
            dataset = summary_row.get("dataset")
            if dataset == "Wessels":
                row["wessels_support_pp_delta"] = summary_row.get("mean_delta_pp")
                row["wessels_route_gap_closure"] = summary_row.get("weighted_route_gap_closure")
            elif dataset == "NormanWeissman2019_filtered":
                row["norman_support_pp_delta"] = summary_row.get("mean_delta_pp")
                row["norman_route_gap_closure"] = summary_row.get("weighted_route_gap_closure")
    if base_status == PASS_STATUS:
        if not route_gap_json.is_file():
            row["status"] = "route_gap_gate_missing_pending"
            row["action"] = "run_support_context_route_gap_gate_before_any_noharm_launch"
            row["reasons"].append("missing_route_gap_gate_sidecar")
            return row
        route_payload = load_json(route_gap_json)
        route_decision = route_payload.get("decision") or {}
        route_status = row["route_gap_status"]
        if route_status != ROUTE_GAP_PASS_STATUS:
            row["status"] = "trackc_smoke_fail_route_gap_close_branch"
            row["action"] = "close_support_context_smoke_or_redesign_context_mechanism"
            row["reasons"].extend(route_decision.get("reasons") or [route_status])
    return row


def render_md(payload: dict[str, Any]) -> str:
    rows = payload["runs"]
    lines = [
        "# LatentFM Track C Support-Context Smoke Summary",
        "",
        "## Scope",
        "",
        "This summary reads run marker files and per-run support/canonical decision JSONs only.",
        "Held-out Track C query artifacts are not read.",
        "",
        "## Overall Status",
        "",
        f"- status: `{payload['overall_status']}`",
        f"- passing runs: `{len(payload['passing_runs'])}`",
        f"- pending runs: `{len(payload['pending_runs'])}`",
        f"- failed/closed runs: `{len(payload['failed_runs'])}`",
        f"- route-gap gate required: `{payload['route_gap_gate_required']}`",
        "",
        "## Runs",
        "",
        "| run | status | train exit | posthoc exit | support pp delta | support p improve | Wessels delta | Wessels route closure | Norman delta | Norman route closure | canonical single pp harm | canonical family pp harm | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reasons = ", ".join(str(x) for x in row.get("reasons") or []) or "NA"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run']}`",
                    f"`{row['status']}`",
                    f"`{row.get('train_exit') if row.get('train_exit') is not None else 'NA'}`",
                    f"`{row.get('posthoc_exit') if row.get('posthoc_exit') is not None else 'NA'}`",
                    f(row.get("support_pp_delta")),
                    f(row.get("support_pp_p_improvement")),
                    f(row.get("wessels_support_pp_delta")),
                    f(row.get("wessels_route_gap_closure")),
                    f(row.get("norman_support_pp_delta")),
                    f(row.get("norman_route_gap_closure")),
                    f(row.get("canonical_single_pp_p_harm")),
                    f(row.get("canonical_family_pp_p_harm")),
                    reasons,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "A passing capped smoke may proceed only to uncapped canonical no-harm.",
            "A pass requires both the base support/canonical decision and the explicit route-gap gate sidecar.",
            "It does not authorize held-out query access or a formal multi-capability claim.",
            "A failed capped smoke is closed as negative evidence for that branch.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=REPORT_DIR / "latentfm_trackc_support_context_smoke_summary_20260622.json")
    ap.add_argument("--out-md", type=Path, default=REPORT_DIR / "LATENTFM_TRACKC_SUPPORT_CONTEXT_SMOKE_SUMMARY_20260622.md")
    args = ap.parse_args()

    rows = [summarize_run(run) for run in RUN_NAMES]
    passing = [row["run"] for row in rows if row.get("status") == PASS_STATUS]
    pending = [
        row["run"]
        for row in rows
        if row.get("status") in {"training_or_waiting", "posthoc_or_waiting", "decision_pending"}
        or row.get("status") == "route_gap_gate_missing_pending"
    ]
    failed = [
        row["run"]
        for row in rows
        if row["run"] not in set(passing) | set(pending)
    ]
    if pending:
        overall = "support_context_smokes_pending"
    elif passing:
        overall = "support_context_smoke_pass_needs_uncapped_noharm"
    else:
        overall = "support_context_smokes_all_failed_close_branch"
    payload = {
        "overall_status": overall,
        "passing_runs": passing,
        "pending_runs": pending,
        "failed_runs": failed,
        "query_read": False,
        "route_gap_gate_required": True,
        "runs": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"overall_status": overall, "out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
