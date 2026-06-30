#!/usr/bin/env python3
"""Decision summary for true cell-count capped-data GPU smokes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_SUMMARY_RUN_ROOT", ROOT / "runs/latentfm_true_cell_count_smokes_20260624"))
OUT_JSON = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_SUMMARY_OUT_JSON", ROOT / "reports/latentfm_true_cell_count_smoke_decision_20260624.json"))
OUT_MD = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_SUMMARY_OUT_MD", ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_SMOKE_DECISION_20260624.md"))


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


def metric(payload: dict[str, Any] | None, group: str, key: str) -> float | None:
    if not payload:
        return None
    value = ((payload.get("groups") or {}).get(group) or {}).get(key)
    return None if value is None else float(value)


def delta(candidate: float | None, anchor: float | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    return float(candidate) - float(anchor)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    eval_dir = run_dir / "posthoc_eval_internal"
    split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
    split_candidate = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    family_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    family_candidate = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    train_exit = read_exit(run_dir / "EXIT_CODE")
    posthoc_exit = read_exit(run_dir / "POSTHOC_RERUN_EXIT_CODE")
    posthoc_exit_source = "POSTHOC_RERUN_EXIT_CODE"
    if posthoc_exit is None:
        posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
        posthoc_exit_source = "POSTHOC_EXIT_CODE"
    groups = {}
    for family, anchor, candidate, names in [
        ("split", split_anchor, split_candidate, ["test", "test_single", "internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy"]),
        ("condition_family", family_anchor, family_candidate, ["test_all", "family_gene", "family_drug", "test_single"]),
    ]:
        for group in names:
            key = f"{family}:{group}"
            groups[key] = {
                "anchor_pearson_pert": metric(anchor, group, "pearson_pert"),
                "candidate_pearson_pert": metric(candidate, group, "pearson_pert"),
                "delta_pearson_pert": delta(metric(candidate, group, "pearson_pert"), metric(anchor, group, "pearson_pert")),
                "anchor_mmd": metric(anchor, group, "test_mmd"),
                "candidate_mmd": metric(candidate, group, "test_mmd"),
                "delta_mmd": delta(metric(candidate, group, "test_mmd"), metric(anchor, group, "test_mmd")),
                "n_conds": ((candidate or {}).get("groups") or {}).get(group, {}).get("n_conds"),
            }
    reasons = []
    cross = groups["split:internal_val_cross_background_seen_gene_proxy"]
    family = groups["condition_family:family_gene"]
    test_single = groups["condition_family:test_single"]
    if train_exit != 0 or posthoc_exit != 0:
        reasons.append("train_or_posthoc_not_complete")
    if cross["delta_pearson_pert"] is None or cross["delta_pearson_pert"] < 0.010:
        reasons.append("cross_background_pp_delta_lt_0p010")
    if family["delta_pearson_pert"] is None or family["delta_pearson_pert"] < 0.0:
        reasons.append("family_gene_pp_negative")
    if family["delta_mmd"] is None or family["delta_mmd"] > 0.001:
        reasons.append("family_gene_mmd_delta_gt_0p001")
    if test_single["delta_pearson_pert"] is None or test_single["delta_pearson_pert"] < -0.005:
        reasons.append("test_single_pp_hard_harm")
    status = "pending_or_failed"
    action = "wait_without_polling_or_debug_failure"
    if train_exit == 0 and posthoc_exit == 0:
        if reasons:
            status = "true_cell_count_smoke_fail_close_or_mutate"
            action = "close_or_mutate_before_more_gpu"
        else:
            status = "true_cell_count_smoke_internal_pass_preliminary"
            action = "run seed/budget matrix and no-harm calibration before canonical no-harm"
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "posthoc_exit_source": posthoc_exit_source,
        "status": status,
        "action": action,
        "reasons": reasons,
        "groups": groups,
        "gate": {
            "cross_background_pp_delta_min": 0.010,
            "family_gene_pp_delta_min": 0.0,
            "family_gene_mmd_delta_max": 0.001,
            "test_single_pp_hard_harm_floor": -0.005,
            "canonical_multi_or_trackc_query_used": False,
        },
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM True Cell-Count Smoke Decision",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes capped-data true cell-count smokes only.",
        "- Uses train-only/internal capped split posthoc outputs.",
        "- Does not read canonical multi or Track C query.",
        "- Does not authorize deployable claims or final scaling-law claims.",
        "",
        "## Runs",
        "",
        "| run | status | cross pp delta | family pp delta | family MMD delta | reasons |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        cross = row["groups"]["split:internal_val_cross_background_seen_gene_proxy"]
        fam = row["groups"]["condition_family:family_gene"]
        def fmt(x: Any) -> str:
            return "NA" if x is None else f"{float(x):+.6f}"
        lines.append(
            f"| `{row['run_name']}` | `{row['status']}` | {fmt(cross['delta_pearson_pert'])} | {fmt(fam['delta_pearson_pert'])} | {fmt(fam['delta_mmd'])} | {', '.join(row['reasons']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- action: `{payload['action']}`",
            f"- GPU authorized by this report: `{payload['gpu_authorized']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="", help="optional run name under true cell-count smoke run root")
    args = ap.parse_args()
    if args.run_name:
        run_dirs = [RUN_ROOT / args.run_name]
    else:
        run_dirs = sorted(
            p for p in RUN_ROOT.iterdir()
            if p.is_dir() and (p / "RUN_STATUS.md").is_file()
        ) if RUN_ROOT.exists() else []
    rows = [summarize_run(p) for p in run_dirs]
    if not rows:
        status = "true_cell_count_smoke_decision_not_ready"
        action = "wait_for_smoke_outputs"
    elif any(row["status"] == "true_cell_count_smoke_internal_pass_preliminary" for row in rows):
        status = "true_cell_count_has_preliminary_internal_pass"
        action = "run designed seed/budget matrix plus controls before canonical no-harm"
    elif all(row["status"] == "true_cell_count_smoke_fail_close_or_mutate" for row in rows):
        status = "true_cell_count_smokes_fail_close"
        action = "close_or_mutate_true_cell_count_branch"
    else:
        status = "true_cell_count_smokes_pending_or_failed"
        action = "wait_without_polling_or_debug_failure"
    payload = {
        "status": status,
        "rows": rows,
        "action": action,
        "gpu_authorized": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
