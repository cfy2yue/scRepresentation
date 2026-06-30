#!/usr/bin/env python3
"""Summarize frozen canonical no-harm for soft-exposure candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_soft_exposure_canonical_noharm_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_soft_exposure_canonical_noharm_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_CANONICAL_NOHARM_DECISION_20260624.md"


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_delta(gate: dict[str, Any], stratum: str, metric: str) -> dict[str, Any] | None:
    for row in gate.get("paired_deltas", []):
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    row: dict[str, Any] = {
        "run": run_name,
        "run_dir": str(run_dir),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "status": "pending",
        "gate_status": None,
        "gate_reasons": [],
        "metrics": {},
    }
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    gate = load_json(gate_json)
    if gate is None:
        return row
    row["status"] = "done"
    row["gate_status"] = (gate.get("gate") or {}).get("status")
    row["gate_reasons"] = (gate.get("gate") or {}).get("reasons", [])
    metrics: dict[str, Any] = {}
    for stratum, metric in [
        ("cross_background_seen_gene", "pearson_pert"),
        ("all_test_single", "pearson_pert"),
        ("all_test_single", "test_mmd_clamped"),
        ("family_gene", "pearson_pert"),
        ("family_gene", "test_mmd_clamped"),
        ("family_drug", "pearson_pert"),
    ]:
        item = _find_delta(gate, stratum, metric)
        if item is not None:
            metrics[f"{stratum}:{metric}"] = {
                "delta_mean": item.get("delta_mean"),
                "p_harm": item.get("p_harm"),
                "p_improve": item.get("p_improve"),
                "ci95": item.get("ci95"),
                "status": item.get("status"),
            }
    row["metrics"] = metrics
    return row


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [r["run"] for r in rows if r["status"] == "pending"]
    failed = [r["run"] for r in rows if r["status"] == "posthoc_failed"]
    passed = [r["run"] for r in rows if r.get("gate_status") == "candidate_gate_pass"]
    near = [
        r["run"]
        for r in rows
        if r.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    if pending:
        return {"status": "pending", "action": "wait_without_polling", "pending_runs": pending}
    if failed:
        return {"status": "posthoc_failed", "action": "read_failed_posthoc_logs", "failed_runs": failed}
    if passed:
        return {"status": "soft_exposure_canonical_noharm_pass", "action": "consider_seed_or_uncapped_confirmation", "passed_runs": passed}
    if near:
        return {"status": "soft_exposure_canonical_noharm_near_miss", "action": "inspect_failure_modes_before_any_followup", "near_runs": near}
    return {"status": "soft_exposure_canonical_noharm_fail", "action": "do_not_promote_soft_exposure", "near_runs": near}


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return str(x)


def main() -> int:
    runs = (
        sorted(
            p.name
            for p in RUN_ROOT.iterdir()
            if p.is_dir() and (p / "RUN_STATUS.md").is_file()
        )
        if RUN_ROOT.is_dir()
        else []
    )
    rows = [summarize_run(run) for run in runs]
    decision = decide(rows)
    payload = {"decision": decision, "rows": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM xverse Soft-Exposure Canonical No-Harm Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes frozen canonical no-harm posthoc for soft-exposure candidates.",
        "- Canonical metrics are post-freeze diagnostics, not training selection.",
        "- Canonical multi groups remain diagnostic only.",
        "",
        "## Rows",
        "",
        "| run | status | gate | cross-bg pp delta | all-single pp p_harm | family-gene pp p_harm | reasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        family = metrics.get("family_gene:pearson_pert", {})
        lines.append(
            f"| {row['run']} | {row['status']} | {row.get('gate_status') or ''} | "
            f"{_fmt(cross.get('delta_mean'))} | {_fmt(all_single.get('p_harm'))} | "
            f"{_fmt(family.get('p_harm'))} | {', '.join(row.get('gate_reasons') or [])} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
