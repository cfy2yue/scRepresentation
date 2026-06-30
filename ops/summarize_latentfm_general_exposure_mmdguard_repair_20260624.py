#!/usr/bin/env python3
"""Summarize general exposure-cap v2 MMD-guarded repair smokes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_general_exposure_mmdguard_repair_20260624"
OUT_JSON = ROOT / "reports/latentfm_general_exposure_mmdguard_repair_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_GENERAL_EXPOSURE_MMDGUARD_REPAIR_DECISION_20260624.md"

RUNS = [
    {"name": "xverse_general_exposure_mmdguard_replay05_mmd05_3k_seed42", "anchor_replay": 0.5},
]
WORST_DATASETS = {
    "TianActivation",
    "NormanWeissman2019_filtered",
    "Nadig_jurket",
    "ReplogleWeissman2022_K562_gwps",
    "Nadig_hepg2",
    "Replogle_RPE1essential",
}
THRESHOLDS = {
    "cross_pp_delta_vs_anchor_min": 0.003,
    "family_gene_mmd_delta_ceiling": 0.001,
    "worst_dataset_mean_mmd_delta_ceiling": 0.005,
    "worst_dataset_harm_count_ceiling": 2,
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


def condition_rows(payload: dict[str, Any] | None, group_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    g = group(payload, group_name)
    rows = {}
    for row in g.get("condition_metrics") or []:
        if isinstance(row, dict) and row.get("dataset") and row.get("condition"):
            rows[(str(row["dataset"]), str(row["condition"]))] = row
    return rows


def worst_dataset_summary(anchor: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    arows = condition_rows(anchor, "family_gene")
    crows = condition_rows(cand, "family_gene")
    by_ds: dict[str, list[dict[str, float]]] = defaultdict(list)
    for key in sorted(set(arows) & set(crows)):
        ds, _ = key
        if ds not in WORST_DATASETS:
            continue
        av = arows[key]
        cv = crows[key]
        if av.get("test_mmd") is None or cv.get("test_mmd") is None:
            continue
        pp_delta = None
        if av.get("pearson_pert") is not None and cv.get("pearson_pert") is not None:
            pp_delta = float(cv["pearson_pert"]) - float(av["pearson_pert"])
        by_ds[ds].append(
            {
                "mmd_delta": float(cv["test_mmd"]) - float(av["test_mmd"]),
                "pp_delta": pp_delta if pp_delta is not None else 0.0,
            }
        )
    rows = []
    for ds, vals in sorted(by_ds.items()):
        rows.append(
            {
                "dataset": ds,
                "n": len(vals),
                "mean_mmd_delta": sum(v["mmd_delta"] for v in vals) / len(vals),
                "mean_pp_delta": sum(v["pp_delta"] for v in vals) / len(vals),
                "mmd_harm_rows": sum(1 for v in vals if v["mmd_delta"] > 0.001),
            }
        )
    max_mean = max((r["mean_mmd_delta"] for r in rows), default=None)
    harm_count = sum(1 for r in rows if r["mean_mmd_delta"] > 0.005 or r["mmd_harm_rows"] > 0)
    return {"rows": rows, "max_worst_dataset_mean_mmd_delta": max_mean, "worst_dataset_harm_count": harm_count}


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
        family_gene_a = group(fam_anchor, "family_gene")
        family_gene_c = group(fam_cand, "family_gene")
        worst = worst_dataset_summary(fam_anchor, fam_cand)
        rows.append(
            {
                **spec,
                "run_dir": str(run_dir),
                "status": status,
                "train_exit": train_exit,
                "posthoc_exit": posthoc_exit,
                "metrics": {
                    "cross_pp_delta_vs_anchor": delta(cross_c, cross_a, "pearson_pert"),
                    "family_gene_pp_delta_vs_anchor": delta(family_gene_c, family_gene_a, "pearson_pert"),
                    "family_gene_mmd_delta_vs_anchor": delta(family_gene_c, family_gene_a, "test_mmd"),
                    **worst,
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
        reasons.append("cross_pp_delta_vs_anchor_lt_0p003")
    if (m.get("family_gene_mmd_delta_vs_anchor") or 999.0) > THRESHOLDS["family_gene_mmd_delta_ceiling"]:
        reasons.append("family_gene_mmd_hard_harm")
    if (m.get("max_worst_dataset_mean_mmd_delta") or 999.0) > THRESHOLDS["worst_dataset_mean_mmd_delta_ceiling"]:
        reasons.append("worst_dataset_mean_mmd_hard_harm")
    if int(m.get("worst_dataset_harm_count") or 0) > THRESHOLDS["worst_dataset_harm_count_ceiling"]:
        reasons.append("too_many_worst_datasets_with_mmd_harm")
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
        return {"status": "mmdguard_internal_pass", "action": "external_review_before_any_canonical_noharm", "passed": passed, "failed": failed}
    return {"status": "mmdguard_internal_fail", "action": "close_general_exposure_mmdguard_repair", "passed": [], "failed": failed}


def main() -> int:
    rows = collect_rows()
    decision = decide(rows)
    payload = {
        "status": decision["status"],
        "decision": decision,
        "thresholds": THRESHOLDS,
        "worst_datasets": sorted(WORST_DATASETS),
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
        "# LatentFM General Exposure MMD-Guarded Repair Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Train-only internal validation only.",
        "- General exposure-cap v2 repair, not a blind rerun.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "",
        "## Rows",
        "",
        "| run | status | replay | cross pp delta | family pp delta | family MMD delta | worst max MMD | worst harm count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | {row['anchor_replay']:.1f} | "
            f"{fmt(m.get('cross_pp_delta_vs_anchor'))} | {fmt(m.get('family_gene_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('family_gene_mmd_delta_vs_anchor'))} | {fmt(m.get('max_worst_dataset_mean_mmd_delta'))} | "
            f"{m.get('worst_dataset_harm_count')} |"
        )
    lines.extend(["", "## Gate", "", f"- passed: `{decision.get('passed')}`", f"- failed: `{decision.get('failed')}`", "", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
