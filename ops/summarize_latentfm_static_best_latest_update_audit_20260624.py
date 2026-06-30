#!/usr/bin/env python3
"""Summarize static best-vs-latest update audit for recent repair branches."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_static_best_latest_update_audit_20260624"
OUT_JSON = ROOT / "reports/latentfm_static_best_latest_update_audit_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_STATIC_BEST_LATEST_UPDATE_AUDIT_20260624.md"

SPECS = [
    {
        "name": "cap60_resp010_seed42",
        "source_run": "xverse_scaling_cap60_resp010_replay05_4k_seed42",
        "source_root": ROOT / "runs/latentfm_scaling_cap60_response_repair_20260624/xverse_scaling_cap60_resp010_replay05_4k_seed42/posthoc_eval_internal",
    },
    {
        "name": "cap60_resp025_seed42",
        "source_run": "xverse_scaling_cap60_resp025_replay05_4k_seed42",
        "source_root": ROOT / "runs/latentfm_scaling_cap60_response_repair_20260624/xverse_scaling_cap60_resp025_replay05_4k_seed42/posthoc_eval_internal",
    },
    {
        "name": "general_exposure_mmdguard",
        "source_run": "xverse_general_exposure_mmdguard_replay05_mmd05_3k_seed42",
        "source_root": ROOT / "runs/latentfm_general_exposure_mmdguard_repair_20260624/xverse_general_exposure_mmdguard_replay05_mmd05_3k_seed42/posthoc_eval_internal",
    },
]

THRESHOLDS = {
    "latest_cross_pp_delta_vs_anchor_min": 0.006,
    "latest_family_pp_delta_vs_anchor_min": 0.006,
    "latest_family_mmd_delta_vs_anchor_max": 0.0,
    "latest_worst_dataset_mmd_delta_vs_best_max": 0.0,
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


def delta(cand: dict[str, Any], base: dict[str, Any], metric: str) -> float | None:
    if cand.get(metric) is None or base.get(metric) is None:
        return None
    return float(cand[metric]) - float(base[metric])


def condition_rows(payload: dict[str, Any] | None, group_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in (group(payload, group_name).get("condition_metrics") or []):
        if isinstance(row, dict) and row.get("dataset") and row.get("condition"):
            rows[(str(row["dataset"]), str(row["condition"]))] = row
    return rows


def max_dataset_mmd_delta(cand: dict[str, Any] | None, base: dict[str, Any] | None) -> float | None:
    crows = condition_rows(cand, "family_gene")
    brows = condition_rows(base, "family_gene")
    by_ds: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(crows) & set(brows)):
        ds, _ = key
        cv = crows[key].get("test_mmd_clamped")
        bv = brows[key].get("test_mmd_clamped")
        if cv is None or bv is None:
            cv = crows[key].get("test_mmd")
            bv = brows[key].get("test_mmd")
        if cv is None or bv is None:
            continue
        by_ds[ds].append(float(cv) - float(bv))
    means = [sum(vals) / len(vals) for vals in by_ds.values() if vals]
    return max(means) if means else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def collect() -> list[dict[str, Any]]:
    rows = []
    for spec in SPECS:
        latest_root = RUN_ROOT / spec["name"] / "posthoc_eval_internal_latest"
        source_root = Path(spec["source_root"])
        anchor_split = load_json(source_root / "split_group_eval_anchor_internal_ode20.json")
        anchor_family = load_json(source_root / "condition_family_eval_anchor_internal_ode20.json")
        best_split = load_json(source_root / "split_group_eval_candidate_internal_ode20.json")
        best_family = load_json(source_root / "condition_family_eval_candidate_internal_ode20.json")
        latest_split = load_json(latest_root / "split_group_eval_latest_internal_ode20.json")
        latest_family = load_json(latest_root / "condition_family_eval_latest_internal_ode20.json")
        exit_code = read_exit(RUN_ROOT / spec["name"] / "POSTHOC_EXIT_CODE")
        status = "done" if exit_code == 0 and latest_split and latest_family else "pending_or_failed"
        if exit_code not in (None, 0):
            status = "failed"
        best_cross = delta(
            group(best_split, "internal_val_cross_background_seen_gene_proxy"),
            group(anchor_split, "internal_val_cross_background_seen_gene_proxy"),
            "pearson_pert",
        )
        latest_cross = delta(
            group(latest_split, "internal_val_cross_background_seen_gene_proxy"),
            group(anchor_split, "internal_val_cross_background_seen_gene_proxy"),
            "pearson_pert",
        )
        best_family_pp = delta(group(best_family, "family_gene"), group(anchor_family, "family_gene"), "pearson_pert")
        latest_family_pp = delta(group(latest_family, "family_gene"), group(anchor_family, "family_gene"), "pearson_pert")
        best_family_mmd = delta(group(best_family, "family_gene"), group(anchor_family, "family_gene"), "test_mmd")
        latest_family_mmd = delta(group(latest_family, "family_gene"), group(anchor_family, "family_gene"), "test_mmd")
        rows.append(
            {
                "name": spec["name"],
                "source_run": spec["source_run"],
                "source_root": str(spec["source_root"]),
                "status": status,
                "posthoc_exit": exit_code,
                "metrics": {
                    "best_cross_pp_delta_vs_anchor": best_cross,
                    "latest_cross_pp_delta_vs_anchor": latest_cross,
                    "latest_minus_best_cross_pp": None if best_cross is None or latest_cross is None else latest_cross - best_cross,
                    "best_family_pp_delta_vs_anchor": best_family_pp,
                    "latest_family_pp_delta_vs_anchor": latest_family_pp,
                    "latest_minus_best_family_pp": None if best_family_pp is None or latest_family_pp is None else latest_family_pp - best_family_pp,
                    "best_family_mmd_delta_vs_anchor": best_family_mmd,
                    "latest_family_mmd_delta_vs_anchor": latest_family_mmd,
                    "latest_minus_best_family_mmd": None if best_family_mmd is None or latest_family_mmd is None else latest_family_mmd - best_family_mmd,
                    "latest_worst_dataset_mmd_delta_vs_best": max_dataset_mmd_delta(latest_family, best_family),
                },
            }
        )
    return rows


def gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    if row["status"] != "done":
        return False, [row["status"]]
    metrics = row["metrics"]
    reasons = []
    if (metrics.get("latest_cross_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["latest_cross_pp_delta_vs_anchor_min"]:
        reasons.append("latest_cross_pp_too_low")
    if (metrics.get("latest_family_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["latest_family_pp_delta_vs_anchor_min"]:
        reasons.append("latest_family_pp_too_low")
    if (metrics.get("latest_family_mmd_delta_vs_anchor") or 999.0) > THRESHOLDS["latest_family_mmd_delta_vs_anchor_max"]:
        reasons.append("latest_family_mmd_positive")
    if (metrics.get("latest_worst_dataset_mmd_delta_vs_best") or 999.0) > THRESHOLDS["latest_worst_dataset_mmd_delta_vs_best_max"]:
        reasons.append("latest_worst_dataset_mmd_not_reduced_vs_best")
    return not reasons, reasons


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["status"] == "failed" for row in rows):
        return {"status": "failed", "action": "inspect_failed_logs_once"}
    if any(row["status"] != "done" for row in rows):
        return {"status": "pending", "action": "wait_1800s_or_work_on_parallel_branches"}
    passed, failed = [], []
    for row in rows:
        ok, reasons = gate(row)
        if ok:
            passed.append(row["name"])
        else:
            failed.append({"name": row["name"], "reasons": reasons})
    if passed:
        return {
            "status": "latest_internal_candidate_found",
            "action": "external_review_before_frozen_canonical_noharm_for_latest",
            "passed": passed,
            "failed": failed,
        }
    return {
        "status": "no_latest_rescue",
        "action": "close_best_vs_latest_update_magnitude_line",
        "passed": [],
        "failed": failed,
    }


def main() -> int:
    rows = collect()
    decision = decide(rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "status": decision["status"],
                "decision": decision,
                "thresholds": THRESHOLDS,
                "boundary": {
                    "train_selection": "train_only_internal",
                    "canonical_metrics_read": False,
                    "canonical_multi_read": False,
                    "trackc_query_read": False,
                },
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = [
        "# LatentFM Static Best-vs-Latest Update Audit",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Train-only internal posthoc only.",
        "- Compares completed `best.pt` metrics against newly evaluated `latest.pt` metrics.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "",
        "## Rows",
        "",
        "| run | status | best cross | latest cross | latest-best cross | best family pp | latest family pp | latest family MMD | latest worst MMD vs best |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | {fmt(metrics.get('best_cross_pp_delta_vs_anchor'))} | "
            f"{fmt(metrics.get('latest_cross_pp_delta_vs_anchor'))} | {fmt(metrics.get('latest_minus_best_cross_pp'))} | "
            f"{fmt(metrics.get('best_family_pp_delta_vs_anchor'))} | {fmt(metrics.get('latest_family_pp_delta_vs_anchor'))} | "
            f"{fmt(metrics.get('latest_family_mmd_delta_vs_anchor'))} | {fmt(metrics.get('latest_worst_dataset_mmd_delta_vs_best'))} |"
        )
    lines.extend(["", "## Gate", "", f"- passed: `{decision.get('passed')}`", f"- failed: `{decision.get('failed')}`", "", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
