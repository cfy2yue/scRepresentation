#!/usr/bin/env python3
"""Summarize TianActivation-targeted general-exposure risk smoke."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_risk_conditioned_general_exposure_smoke_20260624"
OUT_JSON = ROOT / "reports/latentfm_risk_conditioned_general_exposure_smoke_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_CONDITIONED_GENERAL_EXPOSURE_SMOKE_DECISION_20260624.md"

RUNS = [
    {
        "name": "xverse_general_exposure_tian_mmd20_replayall_3k_seed42",
        "target_dataset": "TianActivation",
        "expected_mmd_filter": "TianActivation",
        "expected_replay_filter": "",
    },
    {
        "name": "xverse_general_exposure_tian_mmd20_replaytian_3k_seed42",
        "target_dataset": "TianActivation",
        "expected_mmd_filter": "TianActivation",
        "expected_replay_filter": "TianActivation",
    },
    {
        "name": "xverse_general_exposure_tian_mmd20_noreplay_3k_seed42",
        "target_dataset": "TianActivation",
        "expected_mmd_filter": "TianActivation",
        "expected_replay_filter": "",
    },
    {
        "name": "xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42",
        "target_dataset": "TianActivation",
        "expected_mmd_filter": "TianActivation,NormanWeissman2019_filtered",
        "expected_replay_filter": "",
    },
]

RISK_DATASETS = {
    "TianActivation",
    "NormanWeissman2019_filtered",
    "Nadig_jurket",
    "ReplogleWeissman2022_K562_gwps",
    "Nadig_hepg2",
    "Replogle_RPE1essential",
}

THRESHOLDS = {
    "cross_pp_delta_vs_anchor_min": 0.003,
    "family_gene_pp_delta_vs_anchor_min": 0.0,
    "family_gene_mmd_delta_vs_anchor_max": 0.001,
    "target_dataset_mean_mmd_delta_max": 0.020,
    "target_dataset_harm_rows_max": 2,
    "risk_dataset_harm_count_max": 2,
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


def condition_rows(payload: dict[str, Any] | None, group_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in (group(payload, group_name).get("condition_metrics") or []):
        if isinstance(row, dict) and row.get("dataset") and row.get("condition"):
            rows[(str(row["dataset"]), str(row["condition"]))] = row
    return rows


def dataset_mmd_summary(anchor: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    arows = condition_rows(anchor, "family_gene")
    crows = condition_rows(cand, "family_gene")
    by_ds: dict[str, list[dict[str, float]]] = defaultdict(list)
    for key in sorted(set(arows) & set(crows)):
        ds, _cond = key
        av = arows[key]
        cv = crows[key]
        av_mmd = av.get("test_mmd_clamped", av.get("test_mmd"))
        cv_mmd = cv.get("test_mmd_clamped", cv.get("test_mmd"))
        if av_mmd is None or cv_mmd is None:
            continue
        pp_delta = 0.0
        if av.get("pearson_pert") is not None and cv.get("pearson_pert") is not None:
            pp_delta = float(cv["pearson_pert"]) - float(av["pearson_pert"])
        by_ds[ds].append({"mmd_delta": float(cv_mmd) - float(av_mmd), "pp_delta": pp_delta})
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
    return {"rows": rows}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def value_or(value: Any, default: Any) -> Any:
    return default if value is None else value


def collect() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        run_dir = RUN_ROOT / spec["name"]
        eval_dir = run_dir / "posthoc_eval_internal"
        out_dir = ROOT / "CoupledFM/output/latentfm_runs/risk_conditioned_general_exposure_20260624" / spec["name"]
        cfg = load_json(out_dir / "config.json") or {}
        split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
        fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
        fam_cand = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
        train_exit = read_exit(run_dir / f"{spec['name']}.EXIT_CODE")
        posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
        status = "done" if train_exit == 0 and posthoc_exit == 0 and split_cand and fam_cand else "pending_or_failed"
        if train_exit not in (None, 0) or posthoc_exit not in (None, 0):
            status = "failed"

        cross_delta = delta(
            group(split_cand, "internal_val_cross_background_seen_gene_proxy"),
            group(split_anchor, "internal_val_cross_background_seen_gene_proxy"),
            "pearson_pert",
        )
        fam_pp = delta(group(fam_cand, "family_gene"), group(fam_anchor, "family_gene"), "pearson_pert")
        fam_mmd = delta(group(fam_cand, "family_gene"), group(fam_anchor, "family_gene"), "test_mmd")
        ds_summary = dataset_mmd_summary(fam_anchor, fam_cand)
        target_rows = [r for r in ds_summary["rows"] if r["dataset"] == spec["target_dataset"]]
        risk_rows = [r for r in ds_summary["rows"] if r["dataset"] in RISK_DATASETS]
        target = target_rows[0] if target_rows else {}
        risk_harm_count = sum(1 for r in risk_rows if r["mean_mmd_delta"] > 0.005 or r["mmd_harm_rows"] > 0)

        rows.append(
            {
                **spec,
                "status": status,
                "train_exit": train_exit,
                "posthoc_exit": posthoc_exit,
                "config_filters": {
                    "mmd_dataset_filter": str(cfg.get("mmd_dataset_filter", "")),
                    "anchor_replay_dataset_filter": str(cfg.get("anchor_replay_dataset_filter", "")),
                },
                "metrics": {
                    "cross_pp_delta_vs_anchor": cross_delta,
                    "family_gene_pp_delta_vs_anchor": fam_pp,
                    "family_gene_mmd_delta_vs_anchor": fam_mmd,
                    "target_dataset_mean_mmd_delta": target.get("mean_mmd_delta"),
                    "target_dataset_mean_pp_delta": target.get("mean_pp_delta"),
                    "target_dataset_mmd_harm_rows": target.get("mmd_harm_rows"),
                    "risk_dataset_harm_count": risk_harm_count,
                    "dataset_rows": ds_summary["rows"],
                },
            }
        )
    return rows


def gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    if row["status"] != "done":
        return False, [row["status"]]
    reasons = []
    m = row["metrics"]
    filters = row["config_filters"]
    if filters["mmd_dataset_filter"] != row["expected_mmd_filter"]:
        reasons.append("mmd_dataset_filter_mismatch")
    if filters["anchor_replay_dataset_filter"] != row["expected_replay_filter"]:
        reasons.append("anchor_replay_dataset_filter_mismatch")
    if value_or(m.get("cross_pp_delta_vs_anchor"), -999.0) < THRESHOLDS["cross_pp_delta_vs_anchor_min"]:
        reasons.append("cross_pp_too_low")
    if value_or(m.get("family_gene_pp_delta_vs_anchor"), -999.0) < THRESHOLDS["family_gene_pp_delta_vs_anchor_min"]:
        reasons.append("family_pp_harm")
    if value_or(m.get("family_gene_mmd_delta_vs_anchor"), 999.0) > THRESHOLDS["family_gene_mmd_delta_vs_anchor_max"]:
        reasons.append("family_mmd_harm")
    if value_or(m.get("target_dataset_mean_mmd_delta"), 999.0) > THRESHOLDS["target_dataset_mean_mmd_delta_max"]:
        reasons.append("target_mmd_not_controlled")
    if int(value_or(m.get("target_dataset_mmd_harm_rows"), 999)) > THRESHOLDS["target_dataset_harm_rows_max"]:
        reasons.append("target_harm_rows_too_many")
    if int(value_or(m.get("risk_dataset_harm_count"), 999)) > THRESHOLDS["risk_dataset_harm_count_max"]:
        reasons.append("too_many_risk_datasets_harmed")
    return not reasons, reasons


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r["status"] == "failed" for r in rows):
        return {"status": "failed", "action": "inspect_failed_logs_once"}
    if any(r["status"] != "done" for r in rows):
        return {"status": "pending", "action": "wait_1800s_or_work_on_parallel_branches"}
    passed, failed = [], []
    for row in rows:
        ok, reasons = gate(row)
        if ok:
            passed.append(row["name"])
        else:
            failed.append({"name": row["name"], "reasons": reasons})
    if passed:
        return {"status": "risk_conditioned_internal_pass", "action": "external_review_before_frozen_canonical_noharm", "passed": passed, "failed": failed}
    return {"status": "risk_conditioned_internal_fail", "action": "close_or_mutate_only_with_new_gate", "passed": [], "failed": failed}


def main() -> int:
    rows = collect()
    decision = decide(rows)
    payload = {
        "status": decision["status"],
        "decision": decision,
        "thresholds": THRESHOLDS,
        "risk_datasets": sorted(RISK_DATASETS),
        "boundary": {
            "train_selection": "train_only_internal",
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Risk-Conditioned General Exposure Smoke Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Train-only internal posthoc only.",
        "- Tests dataset-targeted MMD on `TianActivation`; no canonical metrics, canonical multi, or Track C query.",
        "",
        "## Rows",
        "",
        "| run | status | filters | cross pp | family pp | family MMD | Tian MMD | Tian pp | Tian harm rows | risk harm count |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        filters = row["config_filters"]
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | "
            f"MMD={filters['mmd_dataset_filter'] or 'all'}; replay={filters['anchor_replay_dataset_filter'] or 'all'} | "
            f"{fmt(m.get('cross_pp_delta_vs_anchor'))} | {fmt(m.get('family_gene_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('family_gene_mmd_delta_vs_anchor'))} | {fmt(m.get('target_dataset_mean_mmd_delta'))} | "
            f"{fmt(m.get('target_dataset_mean_pp_delta'))} | {m.get('target_dataset_mmd_harm_rows')} | "
            f"{m.get('risk_dataset_harm_count')} |"
        )
    lines.extend(["", "## Gate", "", f"- passed: `{decision.get('passed')}`", f"- failed: `{decision.get('failed')}`", "", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
