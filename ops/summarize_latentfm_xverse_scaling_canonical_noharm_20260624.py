#!/usr/bin/env python3
"""Summarize canonical no-harm posthoc for frozen xverse scaling candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_scaling_canonical_noharm_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md"

BASE_RUNS = [
    "xverse_scaling_cap120_all_3k_seed42",
    "xverse_scaling_gene_cap120_allbg_3k_seed42",
    "xverse_scaling_gene_cap120_k562bg_3k_seed42",
]
OPTIONAL_RUNS = [
    "xverse_scaling_full_trainonly_3k_seed42",
    "xverse_scaling_type_balanced_cap120_3k_seed42",
    "xverse_scaling_jiang_exposure_capped_3k_seed42",
    "xverse_scaling_general_exposure_cap_v2_3k_seed42",
]
PRIMARY_RUN = "xverse_scaling_cap120_all_3k_seed42"
SEPARATE_CANDIDATE_RUNS = set(OPTIONAL_RUNS)
DIAGNOSTIC_RUNS = {run for run in [*BASE_RUNS, *OPTIONAL_RUNS] if run != PRIMARY_RUN}

PRIMARY_ROWS = [
    ("cross_background_seen_gene", "pearson_pert"),
    ("all_test_single", "pearson_pert"),
    ("all_test_single", "test_mmd_clamped"),
    ("family_gene", "pearson_pert"),
    ("family_gene", "test_mmd_clamped"),
    ("family_drug", "pearson_pert"),
]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_delta(gate: dict[str, Any], stratum: str, metric: str) -> dict[str, Any] | None:
    for row in gate.get("paired_deltas", []):
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def family_raw_delta(run_dir: Path, group: str, metric: str) -> dict[str, Any] | None:
    eval_dir = run_dir / "posthoc_eval_canonical"
    anchor = load_json(eval_dir / "condition_family_eval_anchor_ode20_canonical.json")
    cand = load_json(eval_dir / "condition_family_eval_candidate_ode20_canonical.json")
    if not anchor or not cand:
        return None
    a = ((anchor.get("groups") or {}).get(group) or {}).get(metric)
    c = ((cand.get("groups") or {}).get(group) or {}).get(metric)
    if a is None or c is None:
        return None
    return {"anchor": a, "candidate": c, "delta_mean": float(c) - float(a), "status": "raw_eval_delta"}


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
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
    if not gate_json.is_file():
        return row
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    row["status"] = "done"
    row["gate_status"] = (gate.get("gate") or {}).get("status")
    row["gate_reasons"] = (gate.get("gate") or {}).get("reasons", [])
    metrics = {}
    for stratum, metric in PRIMARY_ROWS:
        item = find_delta(gate, stratum, metric)
        if item is not None:
            metrics[f"{stratum}:{metric}"] = {
                "delta_mean": item.get("delta_mean"),
                "ci95": item.get("ci95"),
                "p_improve": item.get("p_improve"),
                "p_harm": item.get("p_harm"),
                "status": item.get("status"),
            }
    drug_pp = family_raw_delta(run_dir, "family_drug", "pearson_pert")
    if drug_pp is not None:
        metrics["family_drug:pearson_pert"] = drug_pp
    drug_mmd = family_raw_delta(run_dir, "family_drug", "test_mmd")
    if drug_mmd is not None:
        metrics["family_drug:test_mmd"] = drug_mmd
    row["metrics"] = metrics
    return row


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [r["run"] for r in rows if r["status"] == "pending"]
    failed = [r["run"] for r in rows if r["status"] == "posthoc_failed"]
    primary = next((r for r in rows if r["run"] == PRIMARY_RUN), None)
    diagnostics = [r for r in rows if r["run"] in DIAGNOSTIC_RUNS]
    separate_candidates = [r for r in rows if r["run"] in SEPARATE_CANDIDATE_RUNS]
    diagnostic_passed = [r["run"] for r in diagnostics if r.get("gate_status") == "candidate_gate_pass"]
    diagnostic_near = [
        r["run"]
        for r in diagnostics
        if r.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    near = [
        PRIMARY_RUN
        if primary is not None
        and primary.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
        else None
    ]
    near = [x for x in near if x is not None]
    separate_passed = [r["run"] for r in separate_candidates if r.get("gate_status") == "candidate_gate_pass"]
    separate_near = [
        r["run"]
        for r in separate_candidates
        if r.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    if pending:
        return {"status": "pending", "action": "wait_without_polling", "pending_runs": pending}
    if primary is not None and primary.get("gate_status") == "candidate_gate_pass":
        return {
            "status": "canonical_noharm_pass",
            "action": "promote_primary_cap120_to_bootstrap_or_seed_check",
            "primary_run": PRIMARY_RUN,
            "diagnostic_passed_runs": diagnostic_passed,
            "diagnostic_near_runs": diagnostic_near,
        }
    if separate_passed:
        return {
            "status": "canonical_noharm_separate_candidate_pass",
            "action": "consider_seed_or_targeted_followup_for_separate_candidate_only",
            "primary_run": PRIMARY_RUN,
            "separate_candidate_passed_runs": separate_passed,
            "separate_candidate_near_runs": separate_near,
            "diagnostic_passed_runs": diagnostic_passed,
            "diagnostic_near_runs": diagnostic_near,
        }
    if near:
        return {
            "status": "canonical_noharm_near_miss",
            "action": "inspect_primary_failure_modes_before_one_targeted_followup",
            "near_runs": near,
            "diagnostic_passed_runs": diagnostic_passed,
            "diagnostic_near_runs": diagnostic_near,
        }
    if failed:
        return {"status": "posthoc_failed", "action": "read_failed_posthoc_logs", "failed_runs": failed}
    return {
        "status": "canonical_noharm_fail",
        "action": "do_not_promote_scaling_candidate_without_new_mechanism",
        "primary_run": PRIMARY_RUN,
        "separate_candidate_near_runs": separate_near,
        "diagnostic_passed_runs": diagnostic_passed,
        "diagnostic_near_runs": diagnostic_near,
    }


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def md_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | status | gate | cross-bg pp delta | all-single pp p_harm | family-gene pp p_harm | family-drug pp delta | family-drug MMD delta | reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        family = metrics.get("family_gene:pearson_pert", {})
        drug = metrics.get("family_drug:pearson_pert", {})
        drug_mmd = metrics.get("family_drug:test_mmd", {})
        lines.append(
            "| {run} | {status} | {gate} | {cross} | {allh} | {famh} | {drug} | {drug_mmd} | {reasons} |".format(
                run=row["run"],
                status=row["status"],
                gate=row.get("gate_status") or "",
                cross="" if "delta_mean" not in cross else f"{float(cross['delta_mean']):+.6f}",
                allh=_fmt(all_single.get("p_harm")),
                famh=_fmt(family.get("p_harm")),
                drug="" if "delta_mean" not in drug else f"{float(drug['delta_mean']):+.6f}",
                drug_mmd="" if "delta_mean" not in drug_mmd else f"{float(drug_mmd['delta_mean']):+.6f}",
                reasons=", ".join(row.get("gate_reasons") or []),
            )
        )
    return "\n".join(lines)


def main() -> int:
    runs = list(BASE_RUNS)
    for run in OPTIONAL_RUNS:
        if (RUN_ROOT / run).exists():
            runs.append(run)
    rows = [summarize_run(run) for run in runs]
    decision = decide(rows)
    payload = {"decision": decision, "rows": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(
        f"""# LatentFM xverse Scaling Canonical No-Harm Decision

Status: `{decision['status']}`
Action: `{decision['action']}`

## Boundary

- Summarizes frozen scaling checkpoints after internal train-only route selection.
- Canonical metrics are post-freeze no-harm diagnostics, not training selection.
- The only primary promotion candidate is `{PRIMARY_RUN}`; gene/background arms
  are diagnostics unless separately predeclared and frozen by a train-only gate.
- Optional full/type-balanced/Jiang/general-exposure arms are separate candidates only if their
  train-only extension gates passed before canonical launch.
- Canonical multi rows remain diagnostic only.

## Rows

{md_table(rows)}

## Decision JSON

`{OUT_JSON}`
""",
        encoding="utf-8",
    )
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
