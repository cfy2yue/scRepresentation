#!/usr/bin/env python3
"""Decision summary for all-modality dose-aware LatentFM smokes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(os.environ.get("LATENTFM_ALLMODALITY_DOSEAWARE_SUMMARY_RUN_ROOT", ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625"))
OUT_JSON = Path(os.environ.get("LATENTFM_ALLMODALITY_DOSEAWARE_SUMMARY_OUT_JSON", ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_smoke_decision_20260625.json"))
OUT_MD = Path(os.environ.get("LATENTFM_ALLMODALITY_DOSEAWARE_SUMMARY_OUT_MD", ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_SMOKE_DECISION_20260625.md"))


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


def n_conds(payload: dict[str, Any] | None, group: str) -> int | None:
    if not payload:
        return None
    value = ((payload.get("groups") or {}).get(group) or {}).get("n_conds")
    return None if value is None else int(value)


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
    posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
    groups: dict[str, dict[str, Any]] = {}
    specs = [
        ("split:test", split_anchor, split_candidate, "test"),
        ("family:test_all", family_anchor, family_candidate, "test_all"),
        ("family:family_gene", family_anchor, family_candidate, "family_gene"),
        ("family:family_drug", family_anchor, family_candidate, "family_drug"),
        ("family:type_drug", family_anchor, family_candidate, "type_drug"),
    ]
    for out_key, anchor, candidate, group in specs:
        groups[out_key] = {
            "anchor_pearson_pert": metric(anchor, group, "pearson_pert"),
            "candidate_pearson_pert": metric(candidate, group, "pearson_pert"),
            "delta_pearson_pert": delta(metric(candidate, group, "pearson_pert"), metric(anchor, group, "pearson_pert")),
            "anchor_mmd": metric(anchor, group, "test_mmd"),
            "candidate_mmd": metric(candidate, group, "test_mmd"),
            "delta_mmd": delta(metric(candidate, group, "test_mmd"), metric(anchor, group, "test_mmd")),
            "n_conds": n_conds(candidate, group),
        }
    reasons: list[str] = []
    if train_exit != 0 or posthoc_exit != 0:
        reasons.append("train_or_posthoc_not_complete")
    all_group = groups["family:test_all"]
    gene = groups["family:family_gene"]
    drug = groups["family:family_drug"]
    if all_group["delta_pearson_pert"] is None or all_group["delta_pearson_pert"] < 0.005:
        reasons.append("test_all_pp_delta_lt_0p005")
    if all_group["delta_mmd"] is None or all_group["delta_mmd"] > 0.002:
        reasons.append("test_all_mmd_delta_gt_0p002")
    if gene["n_conds"] and (gene["delta_pearson_pert"] is None or gene["delta_pearson_pert"] < -0.005):
        reasons.append("family_gene_pp_hard_harm")
    if drug["n_conds"] and (drug["delta_pearson_pert"] is None or drug["delta_pearson_pert"] < 0.005):
        reasons.append("family_drug_pp_delta_lt_0p005")
    if drug["n_conds"] and (drug["delta_mmd"] is None or drug["delta_mmd"] > 0.002):
        reasons.append("family_drug_mmd_delta_gt_0p002")
    status = "pending_or_failed"
    action = "wait_without_polling_or_debug_failure"
    if train_exit == 0 and posthoc_exit == 0:
        if reasons:
            status = "allmodality_doseaware_smoke_fail_close_or_mutate"
            action = "close_or_mutate_before_more_gpu"
        else:
            status = "allmodality_doseaware_smoke_internal_pass_preliminary"
            action = "run seed/budget matrix and controls before canonical no-harm"
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "status": status,
        "action": action,
        "reasons": reasons,
        "groups": groups,
        "gate": {
            "test_all_pp_delta_min": 0.005,
            "test_all_mmd_delta_max": 0.002,
            "family_gene_pp_hard_harm_floor": -0.005,
            "family_drug_pp_delta_min": 0.005,
            "family_drug_mmd_delta_max": 0.002,
            "canonical_multi_or_trackc_query_used": False,
        },
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Smoke Decision",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes dose-aware all-modality bounded smokes only.",
        "- Uses train-only/internal loader split derived from `internal_val_allmodality_doseaware`.",
        "- Does not read canonical multi or Track C query.",
        "- Does not authorize deployable claims or final scaling-law claims.",
        "",
        "## Runs",
        "",
        "| run | status | all pp | gene pp | drug pp | drug MMD | reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        all_g = row["groups"]["family:test_all"]
        gene = row["groups"]["family:family_gene"]
        drug = row["groups"]["family:family_drug"]
        lines.append(
            f"| `{row['run_name']}` | `{row['status']}` | {fmt(all_g['delta_pearson_pert'])} | {fmt(gene['delta_pearson_pert'])} | {fmt(drug['delta_pearson_pert'])} | {fmt(drug['delta_mmd'])} | {', '.join(row['reasons']) or 'none'} |"
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
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    if args.run_name:
        run_dirs = [RUN_ROOT / args.run_name]
    else:
        run_dirs = sorted(p for p in RUN_ROOT.iterdir() if p.is_dir() and (p / "RUN_STATUS.md").is_file()) if RUN_ROOT.exists() else []
    rows = [summarize_run(p) for p in run_dirs]
    if not rows:
        status = "allmodality_doseaware_smoke_decision_not_ready"
        action = "wait_for_smoke_outputs"
    elif any(row["status"] == "allmodality_doseaware_smoke_internal_pass_preliminary" for row in rows):
        status = "allmodality_doseaware_has_preliminary_internal_pass"
        action = "run designed seed/budget matrix plus controls before canonical no-harm"
    elif all(row["status"] == "allmodality_doseaware_smoke_fail_close_or_mutate" for row in rows):
        status = "allmodality_doseaware_smokes_fail_close"
        action = "close_or_mutate_allmodality_branch"
    else:
        status = "allmodality_doseaware_smokes_pending_or_failed"
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
