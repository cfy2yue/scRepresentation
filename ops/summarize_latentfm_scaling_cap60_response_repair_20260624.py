#!/usr/bin/env python3
"""Summarize cap60 response-normalized no-harm repair smokes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_cap60_response_repair_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_cap60_response_repair_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_CAP60_RESPONSE_REPAIR_DECISION_20260624.md"

RUNS = [
    {"name": "xverse_scaling_cap60_resp010_replay05_4k_seed42", "response_weight": 0.10},
    {"name": "xverse_scaling_cap60_resp025_replay05_4k_seed42", "response_weight": 0.25},
]
THRESHOLDS = {
    "cross_pp_delta_vs_anchor_min": 0.010,
    "internal_family_pp_delta_vs_anchor_min": 0.008,
    "family_gene_pp_delta_floor": -0.005,
    "family_gene_mmd_delta_ceiling": 0.0005,
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def group(payload: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not payload:
        return {}
    return dict(((payload.get("groups") or {}).get(name) or {}))


def delta(cand: dict[str, Any], anchor: dict[str, Any], metric: str) -> float | None:
    if cand.get(metric) is None or anchor.get(metric) is None:
        return None
    return float(cand[metric]) - float(anchor[metric])


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:+.6f}"
    return str(x)


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        run_dir = RUN_ROOT / spec["name"]
        eval_dir = run_dir / "posthoc_eval_internal"
        split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
        fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
        fam_cand = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
        train_exit = read_exit(run_dir / f"{spec['name']}.EXIT_CODE")
        posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
        status = "done" if train_exit == 0 and posthoc_exit == 0 else "pending_or_failed"
        if train_exit not in (None, 0) or posthoc_exit not in (None, 0):
            status = "failed"
        cross_a = group(split_anchor, "internal_val_cross_background_seen_gene_proxy")
        cross_c = group(split_cand, "internal_val_cross_background_seen_gene_proxy")
        internal_family_a = group(split_anchor, "internal_val_family_gene_proxy")
        internal_family_c = group(split_cand, "internal_val_family_gene_proxy")
        family_gene_a = group(fam_anchor, "family_gene")
        family_gene_c = group(fam_cand, "family_gene")
        rows.append(
            {
                **spec,
                "run_dir": str(run_dir),
                "status": status,
                "train_exit": train_exit,
                "posthoc_exit": posthoc_exit,
                "metrics": {
                    "cross_pp_delta_vs_anchor": delta(cross_c, cross_a, "pearson_pert"),
                    "internal_family_pp_delta_vs_anchor": delta(
                        internal_family_c, internal_family_a, "pearson_pert"
                    ),
                    "family_gene_pp_delta_vs_anchor": delta(family_gene_c, family_gene_a, "pearson_pert"),
                    "family_gene_mmd_delta_vs_anchor": delta(family_gene_c, family_gene_a, "test_mmd"),
                },
            }
        )
    return rows


def gate_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    if row["status"] != "done":
        return False, [row["status"]]
    m = row["metrics"]
    reasons = []
    if (m.get("cross_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["cross_pp_delta_vs_anchor_min"]:
        reasons.append("cross_pp_delta_vs_anchor_lt_0p010")
    if (m.get("internal_family_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["internal_family_pp_delta_vs_anchor_min"]:
        reasons.append("internal_family_pp_delta_vs_anchor_lt_0p008")
    if (m.get("family_gene_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["family_gene_pp_delta_floor"]:
        reasons.append("family_gene_pp_hard_harm")
    if (m.get("family_gene_mmd_delta_vs_anchor") or 999.0) > THRESHOLDS["family_gene_mmd_delta_ceiling"]:
        reasons.append("family_gene_mmd_hard_harm")
    return not reasons, reasons


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r["status"] == "failed" for r in rows):
        return {"status": "failed", "action": "inspect_failed_logs_once"}
    if any(r["status"] != "done" for r in rows):
        return {"status": "pending", "action": "wait_1800s_or_work_on_parallel_branches"}
    passed, failed = [], []
    for row in rows:
        ok, reasons = gate_row(row)
        if ok:
            passed.append(row["name"])
        else:
            failed.append({"name": row["name"], "reasons": reasons})
    if passed:
        return {"status": "internal_pass", "action": "review_against_canonical_noharm_queue", "passed": passed, "failed": failed}
    return {"status": "internal_fail", "action": "close_response_repair_or_mutate_general_exposure_mmd_guard", "passed": [], "failed": failed}


def main() -> int:
    rows = collect_rows()
    decision = decide(rows)
    payload = {
        "status": decision["status"],
        "decision": decision,
        "thresholds": THRESHOLDS,
        "boundary": {
            "train_selection": "train_only_internal",
            "canonical_metrics_read": False,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Scaling Cap60 Response Repair Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Train-only internal validation only.",
        "- Uses cap60-specific train-only response normalizer.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "",
        "## Rows",
        "",
        "| run | status | response weight | cross pp delta | internal family pp delta | family pp delta | family MMD delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | {row['response_weight']:.2f} | "
            f"{fmt(m.get('cross_pp_delta_vs_anchor'))} | {fmt(m.get('internal_family_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('family_gene_pp_delta_vs_anchor'))} | {fmt(m.get('family_gene_mmd_delta_vs_anchor'))} |"
        )
    lines.extend(["", "## Gate", "", f"- passed: `{decision.get('passed')}`", f"- failed: `{decision.get('failed')}`", "", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
