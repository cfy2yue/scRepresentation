#!/usr/bin/env python3
"""Summarize LatentFM scaling protocol matrix smokes on train-only gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_protocol_matrix_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_protocol_matrix_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_PROTOCOL_MATRIX_DECISION_20260624.md"

RUNS = [
    {
        "name": "xverse_scaling_protocol_cap60_primary19_3k_seed42",
        "arm": "cap60_primary19",
        "role": "condition_count_midpoint",
    },
    {
        "name": "xverse_scaling_protocol_breadth_few_deep_4ds_3k_seed42",
        "arm": "breadth_few_deep_4ds_cap120_budget480",
        "role": "dataset_breadth_few_deep",
    },
    {
        "name": "xverse_scaling_protocol_breadth_mid_8ds_3k_seed42",
        "arm": "breadth_mid_8ds_cap60_budget480",
        "role": "dataset_breadth_mid",
    },
    {
        "name": "xverse_scaling_protocol_breadth_many_shallow_19ds_3k_seed42",
        "arm": "breadth_many_shallow_19ds_cap30_budget480",
        "role": "dataset_breadth_many_shallow",
    },
]


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


def fam(payload: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not payload:
        return {}
    return dict(((payload.get("families") or {}).get(name) or (payload.get("groups") or {}).get(name) or {}))


def delta(cand: dict[str, Any], anchor: dict[str, Any], metric: str) -> float | None:
    if cand.get(metric) is None or anchor.get(metric) is None:
        return None
    return float(cand[metric]) - float(anchor[metric])


def fmt(x: float | None) -> str:
    return "NA" if x is None else f"{x:+.6f}"


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
        cross_a = group(split_anchor, "internal_val_cross_background_seen_gene_proxy")
        cross_c = group(split_cand, "internal_val_cross_background_seen_gene_proxy")
        internal_family_a = group(split_anchor, "internal_val_family_gene_proxy")
        internal_family_c = group(split_cand, "internal_val_family_gene_proxy")
        family_gene_a = fam(fam_anchor, "family_gene")
        family_gene_c = fam(fam_cand, "family_gene")
        row = {
            **spec,
            "run_dir": str(run_dir),
            "status": "done" if train_exit == 0 and posthoc_exit == 0 else "pending_or_failed",
            "train_exit": train_exit,
            "posthoc_exit": posthoc_exit,
            "metrics": {
                "cross_pp_delta_vs_anchor": delta(cross_c, cross_a, "pearson_pert"),
                "cross_candidate_pp": cross_c.get("pearson_pert"),
                "cross_anchor_pp": cross_a.get("pearson_pert"),
                "internal_family_pp_delta_vs_anchor": delta(internal_family_c, internal_family_a, "pearson_pert"),
                "internal_family_mmd_delta_vs_anchor": delta(internal_family_c, internal_family_a, "test_mmd"),
                "family_gene_pp_delta_vs_anchor": delta(family_gene_c, family_gene_a, "pearson_pert"),
                "family_gene_mmd_delta_vs_anchor": delta(family_gene_c, family_gene_a, "test_mmd"),
            },
        }
        rows.append(row)
    return rows


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r["status"] != "done" for r in rows):
        return {"status": "pending", "action": "wait_without_polling", "passed": [], "failed": []}
    passed = []
    failed = []
    by_arm = {r["arm"]: r for r in rows}
    few = by_arm.get("breadth_few_deep_4ds_cap120_budget480", {})
    many = by_arm.get("breadth_many_shallow_19ds_cap30_budget480", {})
    few_cross = (few.get("metrics") or {}).get("cross_candidate_pp")
    many_cross = (many.get("metrics") or {}).get("cross_candidate_pp")
    many_minus_few = None if few_cross is None or many_cross is None else float(many_cross) - float(few_cross)
    for r in rows:
        m = r["metrics"]
        reasons = []
        if (m.get("cross_pp_delta_vs_anchor") or -999.0) < 0.010:
            reasons.append("cross_pp_delta_vs_anchor_lt_0p010")
        if (m.get("family_gene_pp_delta_vs_anchor") or 0.0) < -0.005:
            reasons.append("family_gene_pp_hard_harm")
        if (m.get("family_gene_mmd_delta_vs_anchor") or 0.0) > 0.001:
            reasons.append("family_gene_mmd_hard_harm")
        if r["arm"] == "breadth_many_shallow_19ds_cap30_budget480":
            if many_minus_few is None or many_minus_few < 0.003:
                reasons.append("many_shallow_not_better_than_few_deep_by_0p003")
        if reasons:
            failed.append({"arm": r["arm"], "reasons": reasons})
        else:
            passed.append(r["arm"])
    status = "protocol_matrix_internal_pass" if passed else "protocol_matrix_internal_fail"
    action = (
        "freeze_passed_arm_for_canonical_noharm_only_after_review"
        if passed
        else "close_protocol_matrix_or_design_new_cpu_gate"
    )
    return {
        "status": status,
        "action": action,
        "passed": passed,
        "failed": failed,
        "many_minus_few_cross_candidate_pp": many_minus_few,
        "thresholds": {
            "cross_pp_delta_vs_anchor": "+0.010",
            "family_gene_pp_delta_floor": "-0.005",
            "family_gene_mmd_delta_ceiling": "+0.001",
            "many_shallow_minus_few_deep_cross_candidate_pp": "+0.003",
        },
    }


def main() -> int:
    rows = collect_rows()
    decision = decide(rows)
    payload = {
        "status": decision["status"],
        "decision": decision,
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
        "# LatentFM Scaling Protocol Matrix Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes protocol matrix smokes on train-only internal validation only.",
        "- Does not read canonical metrics, canonical multi for selection, or Track C query.",
        "",
        "## Rows",
        "",
        "| arm | status | cross pp delta vs anchor | family gene pp delta | family gene MMD delta |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| `{r['arm']}` | `{r['status']}` | {fmt(m.get('cross_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('family_gene_pp_delta_vs_anchor'))} | {fmt(m.get('family_gene_mmd_delta_vs_anchor'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- passed arms: `{decision.get('passed')}`",
            f"- failed arms: `{decision.get('failed')}`",
            f"- many-shallow minus few-deep cross candidate pp: `{fmt(decision.get('many_minus_few_cross_candidate_pp'))}`",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
