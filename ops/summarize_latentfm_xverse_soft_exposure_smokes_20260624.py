#!/usr/bin/env python3
"""Summarize xverse cap120 soft-exposure smokes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_soft_exposure_smokes_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_soft_exposure_smokes_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_SMOKES_DECISION_20260624.md"
CAP120_REF = (
    ROOT
    / "runs/latentfm_xverse_scaling_count_smokes_20260624"
    / "xverse_scaling_cap120_all_3k_seed42"
    / "posthoc_eval_internal"
)
RUNS = [
    "xverse_softvisit_p090_no_cap_3k_seed42",
    "xverse_softvisit_p085_no_cap_3k_seed42",
]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def group(path: Path, key: str) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["groups"][key]


def summarize_run(run: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run
    eval_dir = run_dir / "posthoc_eval_internal"
    row: dict[str, Any] = {
        "run": run,
        "run_dir": str(run_dir),
        "train_exit_code": read_text(run_dir / f"{run}.EXIT_CODE"),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "status": "pending",
        "metrics": {},
        "reasons": [],
    }
    if row["train_exit_code"] not in (None, "0"):
        row["status"] = "train_failed"
        return row
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    cand_split = eval_dir / "split_group_eval_candidate_internal_ode20.json"
    cand_family = eval_dir / "condition_family_eval_candidate_internal_ode20.json"
    anchor_split = eval_dir / "split_group_eval_anchor_internal_ode20.json"
    anchor_family = eval_dir / "condition_family_eval_anchor_internal_ode20.json"
    if not cand_split.is_file() or not cand_family.is_file():
        return row

    cs = group(cand_split, "internal_val_cross_background_seen_gene_proxy")
    cf = group(cand_family, "family_gene")
    ans = group(anchor_split, "internal_val_cross_background_seen_gene_proxy")
    anf = group(anchor_family, "family_gene")
    cap120_cs = group(
        CAP120_REF / "split_group_eval_candidate_internal_ode20.json",
        "internal_val_cross_background_seen_gene_proxy",
    )
    row["metrics"] = {
        "cross_pp": float(cs["pearson_pert"]),
        "cross_pp_minus_anchor": float(cs["pearson_pert"]) - float(ans["pearson_pert"]),
        "cross_pp_minus_cap120": float(cs["pearson_pert"]) - float(cap120_cs["pearson_pert"]),
        "family_pp": float(cf["pearson_pert"]),
        "family_pp_minus_anchor": float(cf["pearson_pert"]) - float(anf["pearson_pert"]),
        "family_mmd": float(cf["test_mmd"]),
        "family_mmd_minus_anchor": float(cf["test_mmd"]) - float(anf["test_mmd"]),
    }
    reasons = []
    if row["metrics"]["cross_pp_minus_cap120"] < -0.003:
        reasons.append("cross_pp_too_far_below_cap120")
    if row["metrics"]["cross_pp_minus_anchor"] < 0.010:
        reasons.append("cross_pp_not_material_vs_anchor")
    if row["metrics"]["family_pp_minus_anchor"] < 0.0:
        reasons.append("family_pp_below_anchor")
    if row["metrics"]["family_mmd_minus_anchor"] > 0.001:
        reasons.append("family_mmd_harm")
    row["reasons"] = reasons
    row["status"] = "done_pass" if not reasons else "done_fail"
    return row


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pending = [r["run"] for r in rows if r["status"] == "pending"]
    failed = [r["run"] for r in rows if r["status"] in {"train_failed", "posthoc_failed"}]
    passed = [r for r in rows if r["status"] == "done_pass"]
    if pending:
        return {"status": "pending", "action": "wait_without_polling", "pending_runs": pending}
    if failed:
        return {"status": "failed_or_incomplete", "action": "read_failed_logs", "failed_runs": failed}
    if passed:
        best = max(passed, key=lambda r: r["metrics"]["cross_pp_minus_anchor"])
        return {
            "status": "soft_exposure_internal_pass",
            "action": "consider_frozen_canonical_noharm_for_best_soft_exposure",
            "best_run": best["run"],
            "passed_runs": [r["run"] for r in passed],
        }
    return {
        "status": "soft_exposure_internal_fail",
        "action": "close_soft_exposure_without_new_gate",
    }


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return "NA"


def main() -> None:
    rows = [summarize_run(run) for run in RUNS]
    decision = decide(rows)
    payload = {"decision": decision, "rows": rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM xverse Soft-Exposure Smokes Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes train-only internal posthoc for cap120 soft-exposure smokes.",
        "- Does not read canonical split or Track C query.",
        "- Canonical no-harm is authorized only if this internal gate passes.",
        "",
        "## Rows",
        "",
        "| run | status | cross pp vs cap120 | cross pp vs anchor | family pp vs anchor | family MMD vs anchor | reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row.get("metrics") or {}
        lines.append(
            f"| {row['run']} | {row['status']} | {_fmt(m.get('cross_pp_minus_cap120'))} | "
            f"{_fmt(m.get('cross_pp_minus_anchor'))} | {_fmt(m.get('family_pp_minus_anchor'))} | "
            f"{_fmt(m.get('family_mmd_minus_anchor'))} | {', '.join(row.get('reasons') or [])} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
