#!/usr/bin/env python3
"""Summarize guarded seed robustness for xverse softvisit p085."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_soft_exposure_seed_robustness_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_soft_exposure_seed_robustness_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_SEED_ROBUSTNESS_DECISION_20260624.md"
CAP120_REF = (
    ROOT
    / "runs/latentfm_xverse_scaling_count_smokes_20260624"
    / "xverse_scaling_cap120_all_3k_seed42"
    / "posthoc_eval_internal"
)

DEFAULT_RUNS = [
    "xverse_softvisit_p085_no_cap_3k_seed43",
    "xverse_softvisit_p085_no_cap_3k_seed44",
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


def group(path: Path, key: str) -> dict[str, Any] | None:
    obj = load_json(path)
    if not obj:
        return None
    return (obj.get("groups") or {}).get(key)


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
    found = sorted(p.name for p in RUN_ROOT.iterdir() if p.is_dir() and p.name.startswith("xverse_softvisit_p085_no_cap_3k_seed"))
    return found or DEFAULT_RUNS


def summarize_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    row: dict[str, Any] = {
        "run": run_name,
        "run_dir": str(run_dir),
        "train_exit_code": read_text(run_dir / f"{run_name}.EXIT_CODE"),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "posthoc_finished": read_text(run_dir / "POSTHOC_FINISHED"),
        "gate_json": str(gate_json),
        "status": "pending",
        "gate_status": None,
        "gate_reasons": [],
        "internal_status": "pending",
        "internal_reasons": [],
        "internal_metrics": {},
        "metrics": {},
    }
    if row["train_exit_code"] not in (None, "0"):
        row["status"] = "train_failed"
        return row
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    internal_dir = run_dir / "posthoc_eval_internal"
    cand_split = internal_dir / "split_group_eval_candidate_internal_ode20.json"
    cand_family = internal_dir / "condition_family_eval_candidate_internal_ode20.json"
    anchor_split = internal_dir / "split_group_eval_anchor_internal_ode20.json"
    anchor_family = internal_dir / "condition_family_eval_anchor_internal_ode20.json"
    if cand_split.is_file() and cand_family.is_file() and anchor_split.is_file() and anchor_family.is_file():
        cs = group(cand_split, "internal_val_cross_background_seen_gene_proxy")
        cf = group(cand_family, "family_gene")
        ans = group(anchor_split, "internal_val_cross_background_seen_gene_proxy")
        anf = group(anchor_family, "family_gene")
        cap120_cs = group(
            CAP120_REF / "split_group_eval_candidate_internal_ode20.json",
            "internal_val_cross_background_seen_gene_proxy",
        )
        if cs and cf and ans and anf and cap120_cs:
            im = {
                "cross_pp": float(cs["pearson_pert"]),
                "cross_pp_minus_anchor": float(cs["pearson_pert"]) - float(ans["pearson_pert"]),
                "cross_pp_minus_cap120": float(cs["pearson_pert"]) - float(cap120_cs["pearson_pert"]),
                "family_pp": float(cf["pearson_pert"]),
                "family_pp_minus_anchor": float(cf["pearson_pert"]) - float(anf["pearson_pert"]),
                "family_mmd": float(cf["test_mmd"]),
                "family_mmd_minus_anchor": float(cf["test_mmd"]) - float(anf["test_mmd"]),
            }
            reasons = []
            if im["cross_pp_minus_cap120"] < -0.003:
                reasons.append("internal_cross_pp_too_far_below_cap120")
            if im["cross_pp_minus_anchor"] < 0.010:
                reasons.append("internal_cross_pp_not_material_vs_anchor")
            if im["family_pp_minus_anchor"] < 0.0:
                reasons.append("internal_family_pp_below_anchor")
            if im["family_mmd_minus_anchor"] > 0.001:
                reasons.append("internal_family_mmd_harm")
            row["internal_metrics"] = im
            row["internal_reasons"] = reasons
            row["internal_status"] = "done_pass" if not reasons else "done_fail"
    gate = load_json(gate_json)
    if gate is None:
        return row
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
    passed = [
        r["run"]
        for r in done
        if r.get("gate_status") == "candidate_gate_pass" and r.get("internal_status") == "done_pass"
    ]
    near = [
        r["run"]
        for r in done
        if r.get("gate_status") in {"near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
        and r.get("internal_status") == "done_pass"
    ]
    failed_gate = [
        r["run"]
        for r in done
        if r.get("internal_status") != "done_pass"
        or r.get("gate_status")
        not in {"candidate_gate_pass", "near_miss_one_targeted_followup_allowed", "candidate_gate_near_miss"}
    ]
    if pending:
        return {"status": "pending", "action": "wait_without_polling", "pending_runs": pending}
    if failed:
        return {"status": "failed_or_incomplete", "action": "read_failed_logs_then_close_or_fix", "failed_runs": failed}
    if done and len(passed) == len(rows):
        return {"status": "seed_robustness_pass", "action": "prepare_uncapped_or_reporting_confirmation", "passed_runs": passed}
    if near and not failed_gate:
        return {"status": "seed_robustness_near_miss", "action": "inspect_failure_modes_before_any_extra_seed", "passed_runs": passed, "near_runs": near}
    return {
        "status": "seed_robustness_fail",
        "action": "do_not_promote_soft_exposure_as_seed_robust",
        "passed_runs": passed,
        "near_runs": near,
        "failed_gate_runs": failed_gate,
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
        "| run | status | internal | canonical gate | internal cross vs cap120 | canonical cross delta | all-single pp p_harm | family-gene pp p_harm | reasons |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        internal = row.get("internal_metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        family = metrics.get("family_gene:pearson_pert", {})
        lines.append(
            "| {run} | {status} | {internal_status} | {gate} | {icross} | {cross} | {allh} | {famh} | {reasons} |".format(
                run=row["run"],
                status=row["status"],
                internal_status=row.get("internal_status") or "",
                gate=row.get("gate_status") or "",
                icross="" if "cross_pp_minus_cap120" not in internal else f"{float(internal['cross_pp_minus_cap120']):+.6f}",
                cross="" if "delta_mean" not in cross else f"{float(cross['delta_mean']):+.6f}",
                allh=_fmt(all_single.get("p_harm")),
                famh=_fmt(family.get("p_harm")),
                reasons=", ".join((row.get("internal_reasons") or []) + (row.get("gate_reasons") or [])),
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
        f"""# LatentFM xverse Soft-Exposure Seed Robustness Decision

Status: `{decision['status']}`
Action: `{decision['action']}`

## Boundary

- Summarizes guarded p085 seed robustness only.
- Requires seed42 p085 canonical no-harm pass before launch.
- Uses the same train-only cap120 split and fixed `condition_visit_power=0.85`.
- Canonical split is post-training no-harm confirmation only; canonical multi rows remain diagnostic only.

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
