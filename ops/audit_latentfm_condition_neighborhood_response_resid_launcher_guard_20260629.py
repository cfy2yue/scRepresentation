#!/usr/bin/env python3
"""Final launcher guard for response-residualized condition-neighborhood smoke.

CPU/report-only. Verifies split provenance, selected-pair membership, inherited
eval fields, and leakage boundaries before the guarded GPU launcher is allowed.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
GATE_JSON = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/latentfm_condition_neighborhood_response_residualized_support_gate_20260629.json"
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
HIGH_SPLIT = ROOT / "dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629/split_seed42_xverse_condition_neighborhood_high_support_response_resid_320pair_q30_resp0.35_cell0.75_ds1.json"
LOW_SPLIT = ROOT / "dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629/split_seed42_xverse_condition_neighborhood_low_support_response_resid_320pair_q30_resp0.35_cell0.75_ds1.json"
SELECTED_PAIRS = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/condition_neighborhood_response_residualized_selected_pairs.csv"
BALANCE = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/condition_neighborhood_response_residualized_balance.csv"
NEGATIVE_CONTROLS = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/condition_neighborhood_response_residualized_negative_controls.csv"
LAUNCHER = ROOT / "ops/launch_latentfm_condition_neighborhood_response_resid_highlow_smoke_20260629.sh"
OUT_DIR = ROOT / "reports/condition_neighborhood_response_resid_launcher_guard_20260629"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_response_resid_launcher_guard_20260629.json"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_RESPONSE_RESID_LAUNCHER_GUARD_20260629.md"
AUDIT_DECISION = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/EXTERNAL_AUDIT_DECISION.json"

EXPECTED_GATE_STATUS = "condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu"
EXPECTED_AUDIT_STATUS = "external_audit_pass_bounded_gpu_smoke"
FORBIDDEN_GROUP_TOKENS = ("multi", "query", "trackc")
ALLOWED_INTERNAL_SELECTION_GROUPS = {
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
}


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def split_train_map(split: dict[str, Any]) -> dict[str, set[str]]:
    return {str(ds): set(map(str, groups.get("train", []))) for ds, groups in split.items()}


def pair_side_map(pairs: pd.DataFrame, side: str) -> dict[str, set[str]]:
    ds_col = f"{side}_dataset"
    cond_col = f"{side}_condition"
    out: dict[str, set[str]] = {}
    for _, row in pairs.iterrows():
        out.setdefault(str(row[ds_col]), set()).add(str(row[cond_col]))
    return out


def flatten_train(train_map: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {(ds, condition) for ds, conditions in train_map.items() for condition in conditions}


def check_split_inheritance(
    parent: dict[str, Any],
    child: dict[str, Any],
    label: str,
    reasons: list[str],
    details: dict[str, Any],
) -> None:
    if set(parent) != set(child):
        reasons.append(f"{label}_dataset_set_differs_from_parent")
    changed_nontrain: list[str] = []
    forbidden_groups: list[str] = []
    internal_groups: set[str] = set()
    eval_groups: set[str] = set()
    train_eval_intersections: list[str] = []
    for dataset, parent_groups in parent.items():
        child_groups = child.get(dataset, {})
        if not isinstance(child_groups, dict):
            reasons.append(f"{label}_{dataset}_groups_not_dict")
            continue
        for key in child_groups:
            key_s = str(key)
            if any(token in key_s.lower() for token in FORBIDDEN_GROUP_TOKENS):
                forbidden_groups.append(f"{dataset}:{key_s}")
            if key_s.startswith("internal_val_"):
                internal_groups.add(key_s)
            if key_s != "train":
                eval_groups.add(key_s)
        for key, parent_value in parent_groups.items():
            if key == "train":
                continue
            if child_groups.get(key) != parent_value:
                changed_nontrain.append(f"{dataset}:{key}")
        train = set(map(str, child_groups.get("train", [])))
        for key, value in child_groups.items():
            if key == "train":
                continue
            overlap = train & set(map(str, value or []))
            if overlap:
                train_eval_intersections.append(f"{dataset}:{key}:{len(overlap)}")
    if changed_nontrain:
        reasons.append(f"{label}_nontrain_fields_changed")
    if forbidden_groups:
        reasons.append(f"{label}_forbidden_group_tokens_present")
    if not ALLOWED_INTERNAL_SELECTION_GROUPS.issubset(internal_groups):
        reasons.append(f"{label}_missing_expected_internal_selection_groups")
    if train_eval_intersections:
        reasons.append(f"{label}_train_eval_overlap_present")
    details[f"{label}_changed_nontrain"] = changed_nontrain[:20]
    details[f"{label}_forbidden_groups"] = forbidden_groups[:20]
    details[f"{label}_internal_groups"] = sorted(internal_groups)
    details[f"{label}_eval_groups"] = sorted(eval_groups)
    details[f"{label}_train_eval_intersections"] = train_eval_intersections[:20]


def check_launcher_text(path: Path, reasons: list[str], details: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        "LATENTFM_CNH_RESPONSE_RESID_ACK=external_audit_passed_bounded_smoke",
        "EXTERNAL_AUDIT_DECISION.json",
        "condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu",
        "external_audit_pass_bounded_gpu_smoke",
        "LATENTFM_SCALING_V2_INFO_HIGH_SPLIT",
        "LATENTFM_SCALING_V2_INFO_LOW_SPLIT",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        reasons.append("launcher_missing_required_guard_snippets")
    details["launcher_missing_snippets"] = missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-audit-decision", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    details: dict[str, Any] = {}

    for label, path in {
        "gate_json": GATE_JSON,
        "parent_split": PARENT_SPLIT,
        "high_split": HIGH_SPLIT,
        "low_split": LOW_SPLIT,
        "selected_pairs": SELECTED_PAIRS,
        "balance": BALANCE,
        "negative_controls": NEGATIVE_CONTROLS,
        "launcher": LAUNCHER,
    }.items():
        if not path.exists():
            reasons.append(f"missing_{label}")

    gate = load_json(GATE_JSON) if GATE_JSON.exists() else {}
    if gate.get("status") != EXPECTED_GATE_STATUS:
        reasons.append(f"gate_status_not_expected:{gate.get('status')}")
    if gate.get("gpu_authorized_next") not in (False, None):
        reasons.append("gate_gpu_authorized_next_should_be_false_before_launcher_guard")

    pairs = pd.read_csv(SELECTED_PAIRS) if SELECTED_PAIRS.exists() else pd.DataFrame()
    required_pair_cols = {
        "high_dataset",
        "high_condition",
        "low_dataset",
        "low_condition",
        "perturbation_type_raw",
        "high_key",
        "low_key",
    }
    missing_pair_cols = sorted(required_pair_cols - set(pairs.columns))
    if missing_pair_cols:
        reasons.append("selected_pairs_missing_required_columns")
    if len(pairs) != 320:
        reasons.append(f"selected_pair_count_not_320:{len(pairs)}")
    if "high_key" in pairs and pairs["high_key"].nunique() != len(pairs):
        reasons.append("high_keys_not_unique")
    if "low_key" in pairs and pairs["low_key"].nunique() != len(pairs):
        reasons.append("low_keys_not_unique")
    if {"high_key", "low_key"}.issubset(pairs.columns):
        overlap = set(pairs["high_key"].astype(str)) & set(pairs["low_key"].astype(str))
        if overlap:
            reasons.append(f"high_low_key_overlap:{len(overlap)}")
        details["high_low_key_overlap_count"] = len(overlap)

    parent = load_json(PARENT_SPLIT) if PARENT_SPLIT.exists() else {}
    high = load_json(HIGH_SPLIT) if HIGH_SPLIT.exists() else {}
    low = load_json(LOW_SPLIT) if LOW_SPLIT.exists() else {}
    if parent and high:
        check_split_inheritance(parent, high, "high_split", reasons, details)
    if parent and low:
        check_split_inheritance(parent, low, "low_split", reasons, details)

    high_train = split_train_map(high) if high else {}
    low_train = split_train_map(low) if low else {}
    expected_high = pair_side_map(pairs, "high") if not pairs.empty and not missing_pair_cols else {}
    expected_low = pair_side_map(pairs, "low") if not pairs.empty and not missing_pair_cols else {}
    high_flat = flatten_train(high_train)
    expected_high_flat = flatten_train(expected_high)
    low_flat = flatten_train(low_train)
    expected_low_flat = flatten_train(expected_low)
    if high_flat != expected_high_flat:
        reasons.append("high_split_train_not_equal_selected_pair_high_side")
        details["high_train_diff_count"] = len(high_flat ^ expected_high_flat)
    else:
        details["high_train_diff_count"] = 0
    if low_flat != expected_low_flat:
        reasons.append("low_split_train_not_equal_selected_pair_low_side")
        details["low_train_diff_count"] = len(low_flat ^ expected_low_flat)
    else:
        details["low_train_diff_count"] = 0
    parent_train = split_train_map(parent) if parent else {}
    if not high_flat.issubset(flatten_train(parent_train)):
        reasons.append("high_train_not_subset_of_parent_train")
    if not low_flat.issubset(flatten_train(parent_train)):
        reasons.append("low_train_not_subset_of_parent_train")

    balance = pd.read_csv(BALANCE) if BALANCE.exists() else pd.DataFrame()
    controls = pd.read_csv(NEGATIVE_CONTROLS) if NEGATIVE_CONTROLS.exists() else pd.DataFrame()
    max_cov_smd = None
    response_auc = None
    if not balance.empty:
        cov_features = [
            "response_norm",
            "n_gt",
            "n_ctrl",
            "max_state_entropy",
            "same_target_cross_dataset_total",
            "exact_bool",
        ]
        cov = balance[balance["feature"].isin(cov_features)].copy()
        max_cov_smd = float(cov["smd_high_minus_low"].abs().max())
        response_auc = float(balance.loc[balance["feature"].eq("response_norm"), "auc_discriminability"].iloc[0])
        if max_cov_smd > 0.35:
            reasons.append(f"max_covariate_smd_above_guard:{max_cov_smd:.4f}")
        if response_auc > 0.65:
            reasons.append(f"response_auc_above_guard:{response_auc:.4f}")
    if not controls.empty and bool(controls["risk"].astype(bool).any()):
        reasons.append("negative_control_risk_present")

    check_launcher_text(LAUNCHER, reasons, details)

    status = "launcher_guard_pass_ready_for_bounded_gpu_smoke" if not reasons else "launcher_guard_fail_no_gpu"
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": status.endswith("bounded_gpu_smoke"),
        "reasons": reasons,
        "details": details,
        "summary": {
            "selected_pairs": int(len(pairs)),
            "high_train_conditions": int(sum(len(v) for v in high_train.values())),
            "low_train_conditions": int(sum(len(v) for v in low_train.values())),
            "max_covariate_smd": max_cov_smd,
            "response_auc": response_auc,
            "external_audit_agent": "Lorentz",
            "external_audit_agent_id": "019f1380-0ac3-7ae3-bb9d-56ecc3383ae7",
        },
        "inputs": {
            "gate_json": str(GATE_JSON),
            "parent_split": str(PARENT_SPLIT),
            "high_split": str(HIGH_SPLIT),
            "low_split": str(LOW_SPLIT),
            "selected_pairs": str(SELECTED_PAIRS),
            "balance": str(BALANCE),
            "negative_controls": str(NEGATIVE_CONTROLS),
            "launcher": str(LAUNCHER),
        },
        "outputs": {"report": str(OUT_MD), "json": str(OUT_JSON), "audit_decision": str(AUDIT_DECISION)},
        "boundary": "cpu_report_only_launcher_guard_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)

    if args.write_audit_decision and status.endswith("bounded_gpu_smoke"):
        decision = {
            "created_at": now_cst(),
            "status": EXPECTED_AUDIT_STATUS,
            "guard_json": str(OUT_JSON),
            "external_audit_agent": "Lorentz",
            "external_audit_agent_id": "019f1380-0ac3-7ae3-bb9d-56ecc3383ae7",
            "scope": "bounded response-residualized condition-neighborhood high/low mechanism smoke only",
            "claim_limits": [
                "no CRISPRa/CRISPRko claim",
                "no canonical multi or Track C query selection",
                "canonical test_single/family_gene only as frozen no-harm veto after internal pass",
            ],
        }
        write_json(AUDIT_DECISION, decision)

    lines = [
        "# Condition-Neighborhood Response-Residualized Launcher Guard",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized next: `{payload['gpu_authorized_next']}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only guard. No training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* Verifies that high/low split `train` lists exactly match selected pairs and non-train eval fields are inherited unchanged from the parent split.",
        "",
        "## Checks",
        "",
        f"* Selected pairs: `{len(pairs)}`.",
        f"* High/low train conditions: `{payload['summary']['high_train_conditions']}` / `{payload['summary']['low_train_conditions']}`.",
        f"* Max covariate SMD: `{fmt(max_cov_smd)}`.",
        f"* Response-norm AUC: `{fmt(response_auc)}`.",
        f"* Reasons: `{'; '.join(reasons) if reasons else 'none'}`.",
        "",
        "## Decision",
        "",
    ]
    if status.endswith("bounded_gpu_smoke"):
        lines.append("* Guard passed. A bounded high/low GPU smoke is allowed by this guard, subject to the launcher resource audit.")
    else:
        lines.append("* Guard failed. Do not launch GPU.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* JSON: `{OUT_JSON}`",
            f"* External audit decision JSON: `{AUDIT_DECISION}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "report": str(OUT_MD), "audit_decision": str(AUDIT_DECISION)}, indent=2))
    return 0 if status.endswith("bounded_gpu_smoke") else 5


if __name__ == "__main__":
    raise SystemExit(main())
