#!/usr/bin/env python3
"""Summarize xverse OT pair-mode smoke decisions after posthoc completes."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(
    os.environ.get(
        "LATENTFM_XVERSE_OTPAIR_RUN_ROOT",
        str(ROOT / "runs/latentfm_xverse_ot_pairmode_smokes_20260624"),
    )
)
OUT_JSON = Path(
    os.environ.get(
        "LATENTFM_XVERSE_OTPAIR_DECISION_JSON",
        str(ROOT / "reports/latentfm_xverse_ot_pairmode_smokes_decision_20260624.json"),
    )
)
OUT_MD = Path(
    os.environ.get(
        "LATENTFM_XVERSE_OTPAIR_DECISION_MD",
        str(ROOT / "reports/LATENTFM_XVERSE_OT_PAIRMODE_SMOKES_DECISION_20260624.md"),
    )
)

DEFAULT_RUNS = [
    "xverse_otpair_multinomial_2k_seed42",
    "xverse_otpair_assignment_2k_seed42",
    "xverse_otpair_random_2k_seed42",
]
RUNS = [
    item.strip()
    for item in os.environ.get("LATENTFM_XVERSE_OTPAIR_RUNS", ",".join(DEFAULT_RUNS)).split(",")
    if item.strip()
]

PRIMARY_ROWS = [
    ("cross_background_seen_gene", "pearson_pert"),
    ("all_test_single", "pearson_pert"),
    ("all_test_single", "test_mmd_clamped"),
    ("family_gene", "pearson_pert"),
    ("family_gene", "test_mmd_clamped"),
    ("globally_unseen_gene", "pearson_pert"),
]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def find_delta(gate: dict, stratum: str, metric: str) -> dict | None:
    for row in gate.get("paired_deltas", []):
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def summarize_run(run_name: str) -> dict:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    row = {
        "run": run_name,
        "run_dir": str(run_dir),
        "train_exit_code": read_text(run_dir / f"{run_name}.EXIT_CODE"),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "finished": read_text(run_dir / f"{run_name}.FINISHED"),
        "posthoc_finished": read_text(run_dir / "POSTHOC_FINISHED"),
        "gate_json": str(gate_json),
        "status": "pending",
        "gate_status": None,
        "gate_reasons": [],
        "metrics": {},
    }
    if row["train_exit_code"] not in (None, "0"):
        if row["train_exit_code"] == "130" and "random" in run_name:
            row["status"] = "policy_stopped_control_deferred"
        else:
            row["status"] = "train_failed"
        return row
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    if not gate_json.is_file():
        return row
    gate = json.loads(gate_json.read_text())
    row["status"] = "done"
    row["gate_status"] = (gate.get("gate") or {}).get("status")
    row["gate_reasons"] = (gate.get("gate") or {}).get("reasons", [])
    for stratum, metric in PRIMARY_ROWS:
        item = find_delta(gate, stratum, metric)
        if item is not None:
            row["metrics"][f"{stratum}:{metric}"] = {
                "delta_mean": item.get("delta_mean"),
                "ci95": item.get("ci95"),
                "p_improve": item.get("p_improve"),
                "p_harm": item.get("p_harm"),
                "status": item.get("status"),
            }
    return row


def decide(rows: list[dict]) -> dict:
    pending = [r["run"] for r in rows if r["status"] == "pending"]
    failed = [r["run"] for r in rows if r["status"] in {"train_failed", "posthoc_failed"}]
    deferred = [r["run"] for r in rows if r["status"] == "policy_stopped_control_deferred"]
    done = [r for r in rows if r["status"] == "done"]
    passed = [r["run"] for r in done if r.get("gate_status") == "candidate_gate_pass"]
    near = [
        r["run"]
        for r in done
        if r.get("gate_status")
        in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    if pending:
        status = "pending"
        action = "wait_without_polling"
    elif passed:
        status = "pass_candidate_available"
        action = "compare_pass_against_multinomial_then_seed_or_uncapped_noharm"
    elif near:
        status = "near_miss_candidate_available"
        action = "audit_one_targeted_followup_or_seed_check"
    elif failed:
        status = "failed_or_incomplete"
        action = "read_failed_logs_then_close_or_fix"
    elif deferred:
        status = "primary_pairmode_done_control_deferred"
        action = "compare_multinomial_vs_assignment_then_relaunch_random_only_if_needed"
    else:
        status = "all_done_no_pass"
        action = "do_not_change_default_ot_without_new_evidence"
    return {
        "status": status,
        "action": action,
        "pending_runs": pending,
        "failed_runs": failed,
        "deferred_runs": deferred,
        "passed_runs": passed,
        "near_runs": near,
    }


def md_table(rows: list[dict]) -> str:
    lines = [
        "| run | status | gate | cross-bg pp delta | all-single pp p_harm | family-gene pp p_harm | reasons |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        family = metrics.get("family_gene:pearson_pert", {})
        lines.append(
            "| {run} | {status} | {gate} | {cross} | {allh} | {famh} | {reasons} |".format(
                run=row["run"],
                status=row["status"],
                gate=row.get("gate_status") or "",
                cross="" if "delta_mean" not in cross else f"{cross['delta_mean']:+.6f}",
                allh="" if "p_harm" not in all_single else f"{all_single['p_harm']:.3f}",
                famh="" if "p_harm" not in family else f"{family['p_harm']:.3f}",
                reasons=", ".join(row.get("gate_reasons") or []),
            )
        )
    return "\n".join(lines)


def main() -> None:
    rows = [summarize_run(run) for run in RUNS]
    decision = decide(rows)
    payload = {"decision": decision, "rows": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    OUT_MD.write_text(
        f"""# LatentFM xverse OT Pair-Mode Smokes Decision

Status: `{decision['status']}`
Action: `{decision['action']}`

## Boundary

- Summarizes predeclared OT pair-mode smokes only.
- Reads posthoc gate JSONs after runs finish.
- Does not select checkpoints from canonical metrics during training.
- Canonical multi rows are diagnostic only.

## Rows

{md_table(rows)}

## Decision JSON

`{OUT_JSON}`
"""
    )
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
