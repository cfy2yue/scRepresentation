#!/usr/bin/env python3
"""Summarize frozen canonical no-harm for cap60 response-normalized repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_cap60_response_canonical_noharm_20260624"
INTERNAL_JSON = ROOT / "reports/latentfm_scaling_cap60_response_repair_decision_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_scaling_cap60_response_canonical_noharm_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_CAP60_RESPONSE_CANONICAL_NOHARM_DECISION_20260624.md"


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def passed_internal_names() -> list[str]:
    obj = load_json(INTERNAL_JSON) or {}
    return [str(x) for x in ((obj.get("decision") or {}).get("passed") or [])]


def find_delta(gate: dict[str, Any], stratum: str, metric: str) -> dict[str, Any]:
    for row in gate.get("paired_deltas", []) or []:
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return {}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    gate = load_json(gate_json)
    row: dict[str, Any] = {
        "run": run_name,
        "run_dir": str(run_dir),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "posthoc_finished": read_text(run_dir / "POSTHOC_FINISHED"),
        "gate_json": str(gate_json),
        "status": "pending",
        "gate_status": None,
        "gate_reasons": [],
        "metrics": {},
    }
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    if not gate:
        return row
    row["status"] = "done"
    row["gate_status"] = (gate.get("gate") or {}).get("status")
    row["gate_reasons"] = (gate.get("gate") or {}).get("reasons") or []
    for stratum, metric in [
        ("cross_background_seen_gene", "pearson_pert"),
        ("all_test_single", "pearson_pert"),
        ("all_test_single", "test_mmd_clamped"),
        ("family_gene", "pearson_pert"),
        ("family_gene", "test_mmd_clamped"),
    ]:
        item = find_delta(gate, stratum, metric)
        row["metrics"][f"{stratum}:{metric}"] = {
            "delta_mean": item.get("delta_mean"),
            "p_harm": item.get("p_harm"),
            "ci95": item.get("ci95"),
        }
    return row


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "no_internal_pass", "action": "wait_for_internal_pass"}
    if any(r["status"] == "pending" for r in rows):
        return {"status": "pending", "action": "wait_1800s_or_work_on_parallel_branches"}
    failed_posthoc = [r["run"] for r in rows if r["status"] == "posthoc_failed"]
    if failed_posthoc:
        return {"status": "posthoc_failed", "action": "inspect_failed_logs_once", "failed_runs": failed_posthoc}
    passed = [r["run"] for r in rows if r.get("gate_status") == "candidate_gate_pass"]
    near = [
        r["run"]
        for r in rows
        if r.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    if passed:
        return {
            "status": "canonical_noharm_pass",
            "action": "external_review_before_uncapped_or_seed_confirmation",
            "passed_runs": passed,
        }
    if near:
        return {
            "status": "canonical_noharm_near_miss",
            "action": "inspect_failure_cases_before_one_targeted_repair",
            "near_runs": near,
        }
    return {
        "status": "canonical_noharm_fail",
        "action": "close_or_mutate_response_repair_after_failure_case_review",
    }


def main() -> int:
    rows = [summarize_run(name) for name in passed_internal_names()]
    decision = decide(rows)
    payload = {
        "decision": decision,
        "internal_decision_json": str(INTERNAL_JSON),
        "boundary": {
            "canonical_metrics_read": True,
            "canonical_metrics_role": "frozen_noharm_only_not_checkpoint_selection",
            "canonical_multi_selection": False,
            "canonical_multi_eval": False,
            "trackc_query_read": False,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Scaling Cap60 Response Canonical No-Harm Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Frozen posthoc only for response-normalized cap60 repair candidates that passed train-only internal validation.",
        "- Canonical metrics are no-harm evidence only, not checkpoint selection.",
        "- Canonical multi is neither selected nor evaluated here.",
        "- Held-out Track C query is not read.",
        "",
        "## Rows",
        "",
        "| run | status | gate | cross-bg pp delta | all-single pp p_harm | family-gene pp p_harm | family-gene MMD p_harm | reasons |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        fam_pp = metrics.get("family_gene:pearson_pert", {})
        fam_mmd = metrics.get("family_gene:test_mmd_clamped", {})
        lines.append(
            f"| `{row['run']}` | `{row['status']}` | `{row.get('gate_status')}` | "
            f"{fmt(cross.get('delta_mean'))} | {fmt(all_single.get('p_harm'))} | "
            f"{fmt(fam_pp.get('p_harm'))} | {fmt(fam_mmd.get('p_harm'))} | "
            f"{', '.join(row.get('gate_reasons') or [])} |"
        )
    lines.extend(["", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
