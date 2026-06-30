#!/usr/bin/env python3
"""Summarize frozen canonical no-harm for high-throughput scaling smokes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_CANONICAL_RUN_ROOT",
        str(ROOT / "runs/latentfm_scaling_highthroughput_canonical_noharm_20260624"),
    )
)
INTERNAL_JSON = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_INTERNAL_JSON",
        str(ROOT / "reports/latentfm_scaling_highthroughput_smokes_decision_20260624.json"),
    )
)
OUT_JSON = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_CANONICAL_DECISION_JSON",
        str(ROOT / "reports/latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json"),
    )
)
OUT_MD = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_CANONICAL_DECISION_MD",
        str(ROOT / "reports/LATENTFM_SCALING_HIGH_THROUGHPUT_CANONICAL_NOHARM_DECISION_20260624.md"),
    )
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return str(x)


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    gate = load_json(gate_json)
    row = {
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
        return {"status": "not_authorized_or_no_internal_pass", "action": "wait_for_internal_pass"}
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
            "action": "consider_seed_or_uncapped_confirmation_for_passed_candidate",
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
        "action": "close_highthroughput_scaling_candidates_and_prioritize_repair_branches",
    }


def main() -> int:
    rows = [summarize_run(name) for name in passed_internal_names()]
    decision = decide(rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "decision": decision,
                "internal_decision_json": str(INTERNAL_JSON),
                "boundary": {
                    "canonical_metrics_read": True,
                    "canonical_multi_selection": False,
                    "canonical_multi_eval": False,
                    "trackc_query_read": False,
                    "gpu_launched_by_this_script": False,
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = [
        "# LatentFM Scaling High-Throughput Canonical No-Harm Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Canonical posthoc only for high-throughput smokes that passed train-only internal validation.",
        "- Canonical metrics are not used for checkpoint selection.",
        "- Canonical multi is neither selected nor evaluated here.",
        "- Held-out Track C query is not read.",
        "",
        "## Rows",
        "",
        "| run | status | gate | cross-bg pp delta | all-single p_harm | family-gene p_harm | reasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row.get("metrics") or {}
        cross = m.get("cross_background_seen_gene:pearson_pert", {})
        all_single = m.get("all_test_single:pearson_pert", {})
        fam = m.get("family_gene:pearson_pert", {})
        lines.append(
            f"| `{row['run']}` | `{row['status']}` | `{row.get('gate_status')}` | "
            f"{fmt(cross.get('delta_mean'))} | {fmt(all_single.get('p_harm'))} | "
            f"{fmt(fam.get('p_harm'))} | {', '.join(row.get('gate_reasons') or [])} |"
        )
    lines.extend(["", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
