#!/usr/bin/env python3
"""Decision summary for chemical unseen-drug/scaffold LatentFM smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_chemical_unseen_drug_scaffold_smokes_20260625"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_drug_scaffold_smoke_decision_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_DRUG_SCAFFOLD_SMOKE_DECISION_20260625.md"


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
    return (payload.get("groups") or {}).get(name) or {}


def metric(payload: dict[str, Any] | None, name: str, key: str) -> float | None:
    value = group(payload, name).get(key)
    return None if value is None else float(value)


def n_conds(payload: dict[str, Any] | None, name: str) -> int | None:
    value = group(payload, name).get("n_conds")
    return None if value is None else int(value)


def delta(candidate: float | None, anchor: float | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    return candidate - anchor


def summarize_run(run_dir: Path) -> dict[str, Any]:
    eval_dir = run_dir / "posthoc_eval_internal"
    split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
    split_candidate = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    family_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    family_candidate = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    train_exit = read_exit(run_dir / "EXIT_CODE")
    posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
    specs = [
        ("split:test", split_anchor, split_candidate, "test"),
        ("family:test_all", family_anchor, family_candidate, "test_all"),
        ("family:family_gene", family_anchor, family_candidate, "family_gene"),
        ("family:family_drug", family_anchor, family_candidate, "family_drug"),
        ("family:type_drug", family_anchor, family_candidate, "type_drug"),
    ]
    groups: dict[str, dict[str, Any]] = {}
    for out_key, anchor, candidate, group_name in specs:
        groups[out_key] = {
            "anchor_pearson_pert": metric(anchor, group_name, "pearson_pert"),
            "candidate_pearson_pert": metric(candidate, group_name, "pearson_pert"),
            "delta_pearson_pert": delta(metric(candidate, group_name, "pearson_pert"), metric(anchor, group_name, "pearson_pert")),
            "anchor_mmd": metric(anchor, group_name, "test_mmd"),
            "candidate_mmd": metric(candidate, group_name, "test_mmd"),
            "delta_mmd": delta(metric(candidate, group_name, "test_mmd"), metric(anchor, group_name, "test_mmd")),
            "n_conds": n_conds(candidate, group_name),
        }
    reasons: list[str] = []
    if train_exit != 0 or posthoc_exit != 0:
        reasons.append("train_or_posthoc_not_complete")
    test_all = groups["family:test_all"]
    gene = groups["family:family_gene"]
    drug = groups["family:family_drug"]
    type_drug = groups["family:type_drug"]
    if test_all["delta_pearson_pert"] is None or test_all["delta_pearson_pert"] < 0.005:
        reasons.append("test_all_pp_delta_lt_0p005")
    if test_all["delta_mmd"] is None or test_all["delta_mmd"] > 0.002:
        reasons.append("test_all_mmd_delta_gt_0p002")
    if drug["n_conds"] and (drug["delta_pearson_pert"] is None or drug["delta_pearson_pert"] < 0.005):
        reasons.append("family_drug_pp_delta_lt_0p005")
    if drug["n_conds"] and (drug["delta_mmd"] is None or drug["delta_mmd"] > 0.002):
        reasons.append("family_drug_mmd_delta_gt_0p002")
    if type_drug["n_conds"] and (type_drug["delta_pearson_pert"] is None or type_drug["delta_pearson_pert"] < 0.005):
        reasons.append("type_drug_pp_delta_lt_0p005")
    if gene["n_conds"] and (gene["delta_pearson_pert"] is None or gene["delta_pearson_pert"] < -0.005):
        reasons.append("family_gene_pp_hard_harm")
    if gene["n_conds"] and (gene["delta_mmd"] is None or gene["delta_mmd"] > 0.002):
        reasons.append("family_gene_mmd_hard_harm")
    if train_exit == 0 and posthoc_exit == 0 and not reasons:
        status = "chemical_unseen_smoke_internal_pass_preliminary"
        action = "run seed/step controls and then frozen canonical no-harm veto if mainline use is desired"
    elif train_exit == 0 and posthoc_exit == 0:
        status = "chemical_unseen_smoke_fail_close_or_mutate"
        action = "close or mutate chemical scaling before more GPU"
    else:
        status = "chemical_unseen_smoke_pending_or_failed"
        action = "wait_without_polling_or_debug_failure"
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "status": status,
        "action": action,
        "reasons": reasons,
        "groups": groups,
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Chemical Unseen-Drug/Scaffold Smoke Decision",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes train-only/internal chemical unseen-drug/scaffold smokes.",
        "- Compares candidate checkpoints to `xverse_8k_anchor` on the same split.",
        "- Does not read canonical multi or Track C query.",
        "- Does not authorize deployable claims or final scaling-law claims.",
        "",
        "## Runs",
        "",
        "| run | status | all pp | drug pp | type-drug pp | gene pp | drug MMD | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        all_g = row["groups"]["family:test_all"]
        drug = row["groups"]["family:family_drug"]
        type_drug = row["groups"]["family:type_drug"]
        gene = row["groups"]["family:family_gene"]
        lines.append(
            f"| `{row['run_name']}` | `{row['status']}` | {fmt(all_g['delta_pearson_pert'])} | "
            f"{fmt(drug['delta_pearson_pert'])} | {fmt(type_drug['delta_pearson_pert'])} | "
            f"{fmt(gene['delta_pearson_pert'])} | {fmt(drug['delta_mmd'])} | {', '.join(row['reasons']) or 'none'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- action: `{payload['action']}`",
        f"- GPU authorized by this report: `{payload['gpu_authorized']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    return "\n".join(lines)


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
        status = "chemical_unseen_smoke_decision_not_ready"
        action = "wait_for_smoke_outputs"
    elif any(row["status"] == "chemical_unseen_smoke_internal_pass_preliminary" for row in rows):
        status = "chemical_unseen_has_preliminary_internal_pass"
        action = "run seed/step controls before any broader claim"
    elif all(row["status"] == "chemical_unseen_smoke_fail_close_or_mutate" for row in rows):
        status = "chemical_unseen_smokes_fail_close"
        action = "close_or_mutate_chemical_scaling_branch"
    else:
        status = "chemical_unseen_smokes_pending_or_failed"
        action = "wait_without_polling_or_debug_failure"
    payload = {"status": status, "rows": rows, "action": action, "gpu_authorized": False}
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
