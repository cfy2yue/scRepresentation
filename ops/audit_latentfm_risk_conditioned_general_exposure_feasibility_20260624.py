#!/usr/bin/env python3
"""CPU feasibility gate for risk-conditioned general-exposure no-harm repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json"
FAILURE_REVIEW = ROOT / "reports/latentfm_highthroughput_repair_failure_review_20260624.json"
TRAIN_SCRIPT = ROOT / "CoupledFM/model/latent/train.py"
LAUNCHER = ROOT / "CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh"
OUT_JSON = ROOT / "reports/latentfm_risk_conditioned_general_exposure_feasibility_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_CONDITIONED_GENERAL_EXPOSURE_FEASIBILITY_20260624.md"

SUPPORT_THRESHOLDS = {
    "min_train_conditions_per_risk_dataset": 12,
    "min_internal_family_conditions_per_risk_dataset": 1,
    "risk_dataset_mean_mmd_delta_min": 0.005,
    "risk_condition_mmd_delta_min": 0.005,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_family(cond: str) -> str:
    if "+" in cond:
        return "multi_gene"
    return "single_or_drug"


def current_hook_support() -> dict[str, Any]:
    train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    return {
        "anchor_replay_filter_present": "anchor_replay_condition_filter" in train_text,
        "anchor_replay_dataset_filter_present": "anchor_replay_dataset" in train_text.lower(),
        "mmd_dataset_filter_present": "mmd_dataset" in train_text.lower() or "risk_dataset" in train_text.lower(),
        "launcher_anchor_replay_filter_present": "ANCHOR_REPLAY_CONDITION_FILTER" in launcher_text,
        "launcher_dataset_risk_filter_present": "RISK_DATASET" in launcher_text or "ANCHOR_REPLAY_DATASET" in launcher_text,
        "supported_anchor_replay_filters": ["all", "non_gene_multi"],
    }


def main() -> int:
    split = load_json(SPLIT)
    failure = load_json(FAILURE_REVIEW)
    general = failure["general_exposure_internal"]
    dataset_mmd = {row["dataset"]: float(row["mean"]) for row in general["family_gene_dataset_mmd"]}
    risk_dataset_rows = [
        {
            "dataset": ds,
            "mean_mmd_delta": val,
            "train_conditions": len((split.get(ds) or {}).get("train") or []),
            "internal_family_conditions": len((split.get(ds) or {}).get("internal_val_family_gene_proxy") or []),
            "test_single_conditions": len((split.get(ds) or {}).get("test_single") or []),
        }
        for ds, val in sorted(dataset_mmd.items(), key=lambda item: item[1], reverse=True)
        if val >= SUPPORT_THRESHOLDS["risk_dataset_mean_mmd_delta_min"]
    ]
    top_conditions = [
        row
        for row in general["top_family_mmd_harm"]
        if float(row.get("test_mmd_clamped_delta") or 0.0)
        >= SUPPORT_THRESHOLDS["risk_condition_mmd_delta_min"]
    ]
    condition_rows = []
    for row in top_conditions:
        ds = row["dataset"]
        cond = row["condition"]
        groups = split.get(ds) or {}
        where = [name for name, conds in groups.items() if cond in (conds or [])]
        condition_rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "family": condition_family(cond),
                "mmd_delta": float(row["test_mmd_clamped_delta"]),
                "pp_delta": float(row["pearson_pert_delta"]),
                "split_membership": where,
                "risk_dataset_train_conditions": len(groups.get("train") or []),
                "risk_dataset_internal_family_conditions": len(groups.get("internal_val_family_gene_proxy") or []),
            }
        )
    unsupported = [
        row
        for row in risk_dataset_rows
        if row["train_conditions"] < SUPPORT_THRESHOLDS["min_train_conditions_per_risk_dataset"]
        or row["internal_family_conditions"] < SUPPORT_THRESHOLDS["min_internal_family_conditions_per_risk_dataset"]
    ]
    hooks = current_hook_support()
    has_direct_hook = bool(hooks["anchor_replay_dataset_filter_present"] or hooks["mmd_dataset_filter_present"])
    enough_support = bool(risk_dataset_rows) and not unsupported
    exact_worst_in_train = [row for row in condition_rows if "train" in row["split_membership"]]
    decision: dict[str, Any]
    if not risk_dataset_rows:
        decision = {
            "status": "risk_conditioned_gate_fail_no_risk_datasets",
            "action": "do_not_launch_gpu",
            "reasons": ["no_dataset_mean_mmd_delta_above_threshold"],
        }
    elif not enough_support:
        decision = {
            "status": "risk_conditioned_gate_fail_insufficient_support",
            "action": "do_not_launch_gpu",
            "reasons": ["risk_dataset_has_too_few_train_or_internal_family_conditions"],
        }
    elif not has_direct_hook:
        decision = {
            "status": "risk_conditioned_gate_feasible_but_requires_hook",
            "action": "implement_or_audit_dataset_risk_filter_hook_before_gpu",
            "reasons": ["current_training_code_lacks_dataset_specific_mmd_or_replay_filter"],
        }
    else:
        decision = {
            "status": "risk_conditioned_gate_pass_gpu_authorized",
            "action": "launch_one_bounded_risk_conditioned_smoke_after_resource_audit",
            "reasons": [],
        }
    payload = {
        "status": decision["status"],
        "decision": decision,
        "thresholds": SUPPORT_THRESHOLDS,
        "boundary": {
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "new_gpu_launched": False,
            "split": str(SPLIT),
            "failure_review": str(FAILURE_REVIEW),
        },
        "risk_dataset_rows": risk_dataset_rows,
        "unsupported_risk_datasets": unsupported,
        "top_risk_conditions": condition_rows,
        "top_risk_conditions_exactly_in_train": exact_worst_in_train,
        "current_hook_support": hooks,
        "predeclared_candidate_rule": {
            "risk_dataset_rule": "dataset mean family-gene MMD delta vs anchor >= 0.005 in completed train-only internal failure review",
            "risk_condition_rule": "condition family-gene MMD delta vs anchor >= 0.005 for reporting only; do not select individual held-out conditions for training",
            "mechanism_required": "dataset-risk-conditioned MMD/replay filter over train batches, not scalar gamma/replay sweep",
            "gpu_smoke_gate_if_hook_exists": [
                "cross/family pp delta vs anchor >= +0.005",
                "family MMD delta vs anchor <= 0",
                "worst max dataset MMD materially reduced",
                "worst harm dataset count <= 2",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Conditioned General Exposure Feasibility Gate",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only feasibility gate.",
        "- Reads train-only general-exposure split and completed internal failure review only.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "- Does not launch GPU work.",
        "",
        "## Risk Datasets",
        "",
        "| dataset | mean MMD delta | train conds | internal family conds | test-single conds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in risk_dataset_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['mean_mmd_delta']:+.6f} | {row['train_conditions']} | "
            f"{row['internal_family_conditions']} | {row['test_single_conditions']} |"
        )
    lines.extend(
        [
            "",
            "## Top Risk Conditions",
            "",
            "| dataset | condition | MMD delta | pp delta | split membership |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in condition_rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | {row['mmd_delta']:+.6f} | "
            f"{row['pp_delta']:+.6f} | `{row['split_membership']}` |"
        )
    lines.extend(
        [
            "",
            "## Hook Audit",
            "",
            f"- current anchor replay filters: `{hooks['supported_anchor_replay_filters']}`",
            f"- dataset-specific anchor replay filter present: `{hooks['anchor_replay_dataset_filter_present']}`",
            f"- dataset-specific MMD/risk filter present: `{hooks['mmd_dataset_filter_present']}`",
            "",
            "## Decision",
            "",
            f"- reasons: `{decision['reasons']}`",
            "- Do not launch another scalar MMD/replay smoke from this evidence.",
            "- If this branch continues, first implement or audit a dataset-risk-conditioned hook and run a code-level smoke test.",
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
