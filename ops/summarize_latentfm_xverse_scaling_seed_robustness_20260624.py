#!/usr/bin/env python3
"""Summarize guarded seed robustness for xverse scaling cap120_all."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_seed_robustness_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_scaling_seed_robustness_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SCALING_SEED_ROBUSTNESS_DECISION_20260624.md"

DEFAULT_RUNS = [
    "xverse_scaling_cap120_all_3k_seed43",
    "xverse_scaling_cap120_all_3k_seed44",
]

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


def discover_runs() -> list[str]:
    runs = [run for run in DEFAULT_RUNS if (RUN_ROOT / run).exists()]
    if runs:
        return runs
    if not RUN_ROOT.exists():
        return DEFAULT_RUNS
    found = sorted(p.name for p in RUN_ROOT.iterdir() if p.is_dir() and p.name.startswith("xverse_scaling_cap120_all_3k_seed"))
    return found or DEFAULT_RUNS


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    row = {
        "run": run_name,
        "run_dir": str(run_dir),
        "train_exit_code": read_text(run_dir / f"{run_name}.EXIT_CODE"),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "posthoc_finished": read_text(run_dir / "POSTHOC_FINISHED"),
        "gate_json": str(gate_json),
        "status": "pending",
        "gate_status": None,
        "gate_reasons": [],
        "metrics": {},
    }
    if row["train_exit_code"] not in (None, "0"):
        row["status"] = "train_failed"
        return row
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    if not gate_json.is_file():
        return row
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    row["status"] = "done"
    row["gate_status"] = (gate.get("gate") or {}).get("status")
    row["gate_reasons"] = (gate.get("gate") or {}).get("reasons", [])
    metrics: dict[str, Any] = {}
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
    row["metrics"] = metrics
    return row


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [r["run"] for r in rows if r["status"] == "pending"]
    failed = [r["run"] for r in rows if r["status"] in {"train_failed", "posthoc_failed"}]
    done = [r for r in rows if r["status"] == "done"]
    passed = [r["run"] for r in done if r.get("gate_status") == "candidate_gate_pass"]
    near = [
        r["run"]
        for r in done
        if r.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    failed_gate = [r["run"] for r in done if r.get("gate_status") not in {"candidate_gate_pass", "near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}]
    if pending:
        return {"status": "pending", "action": "wait_without_polling", "pending_runs": pending}
    if failed:
        return {"status": "failed_or_incomplete", "action": "read_failed_logs_then_close_or_fix", "failed_runs": failed}
    if done and len(passed) == len(rows):
        return {"status": "seed_robustness_pass", "action": "prepare_reporting_bootstrap_and_failure_cases", "passed_runs": passed}
    if near and not failed_gate:
        return {"status": "seed_robustness_near_miss", "action": "inspect_failure_modes_before_any_extra_seed", "passed_runs": passed, "near_runs": near}
    return {"status": "seed_robustness_fail", "action": "do_not_promote_scaling_candidate_as_seed_robust", "passed_runs": passed, "near_runs": near, "failed_gate_runs": failed_gate}


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def md_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | status | gate | cross-bg pp delta | all-single pp p_harm | family-gene pp p_harm | family-drug pp delta | reasons |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        family = metrics.get("family_gene:pearson_pert", {})
        drug = metrics.get("family_drug:pearson_pert", {})
        lines.append(
            "| {run} | {status} | {gate} | {cross} | {allh} | {famh} | {drug} | {reasons} |".format(
                run=row["run"],
                status=row["status"],
                gate=row.get("gate_status") or "",
                cross="" if "delta_mean" not in cross else f"{float(cross['delta_mean']):+.6f}",
                allh=_fmt(all_single.get("p_harm")),
                famh=_fmt(family.get("p_harm")),
                drug="" if "delta_mean" not in drug else f"{float(drug['delta_mean']):+.6f}",
                reasons=", ".join(row.get("gate_reasons") or []),
            )
        )
    return "\n".join(lines)


def main() -> int:
    rows = [summarize_run(run) for run in discover_runs()]
    decision = decide(rows)
    payload = {"decision": decision, "rows": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(
        f"""# LatentFM xverse Scaling Seed Robustness Decision

Status: `{decision['status']}`
Action: `{decision['action']}`

## Boundary

- Summarizes guarded cap120_all seed robustness only.
- Requires seed42 cap120_all canonical no-harm pass before launch.
- Uses the same nested-v2 train-only cap120 split; canonical split is post-training no-harm only.
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
