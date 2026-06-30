#!/usr/bin/env python3
"""Summarize cap60 response-normalized seed-robustness internal smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_cap60_response_seed_robustness_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_cap60_response_seed_robustness_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_CAP60_RESPONSE_SEED_ROBUSTNESS_DECISION_20260624.md"

RUNS = [
    {
        "name": "xverse_scaling_cap60_resp025_replay05_4k_seed43",
        "response_weight": 0.25,
        "seed": 43,
    },
]
THRESHOLDS = {
    "cross_pp_delta_vs_anchor_min": 0.005,
    "internal_family_pp_delta_vs_anchor_min": 0.005,
    "family_gene_pp_delta_floor": -0.005,
    "family_gene_mmd_delta_ceiling": 0.0005,
}


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


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


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


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
    metrics = row["metrics"]
    reasons = []
    if (metrics.get("cross_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["cross_pp_delta_vs_anchor_min"]:
        reasons.append("cross_pp_delta_vs_anchor_lt_0p005")
    if (metrics.get("internal_family_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["internal_family_pp_delta_vs_anchor_min"]:
        reasons.append("internal_family_pp_delta_vs_anchor_lt_0p005")
    if (metrics.get("family_gene_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["family_gene_pp_delta_floor"]:
        reasons.append("family_gene_pp_hard_harm")
    if (metrics.get("family_gene_mmd_delta_vs_anchor") or 999.0) > THRESHOLDS["family_gene_mmd_delta_ceiling"]:
        reasons.append("family_gene_mmd_hard_harm")
    return not reasons, reasons


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["status"] == "failed" for row in rows):
        return {"status": "failed", "action": "inspect_failed_logs_once"}
    if any(row["status"] != "done" for row in rows):
        return {"status": "pending", "action": "wait_1800s_or_work_on_parallel_branches"}
    passed, failed = [], []
    for row in rows:
        ok, reasons = gate_row(row)
        if ok:
            passed.append(row["name"])
        else:
            failed.append({"name": row["name"], "reasons": reasons})
    if passed:
        return {
            "status": "seed_robust_internal_pass",
            "action": "integrate_with_frozen_canonical_noharm_result_before_any_extension",
            "passed": passed,
            "failed": failed,
        }
    return {
        "status": "seed_robust_internal_fail",
        "action": "treat_response_repair_as_seed_sensitive_even_if_seed42_canonical_survives",
        "passed": [],
        "failed": failed,
    }


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
        "# LatentFM Scaling Cap60 Response Seed Robustness Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Train-only internal validation only.",
        "- Uses the cap60 train-only response normalizer fitted before seed43 launch.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "",
        "## Rows",
        "",
        "| run | status | seed | response weight | cross pp delta | internal family pp delta | family pp delta | family MMD delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | {row['seed']} | {row['response_weight']:.2f} | "
            f"{fmt(metrics.get('cross_pp_delta_vs_anchor'))} | {fmt(metrics.get('internal_family_pp_delta_vs_anchor'))} | "
            f"{fmt(metrics.get('family_gene_pp_delta_vs_anchor'))} | {fmt(metrics.get('family_gene_mmd_delta_vs_anchor'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- passed: `{decision.get('passed')}`",
            f"- failed: `{decision.get('failed')}`",
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
