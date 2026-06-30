#!/usr/bin/env python3
"""CPU-only gate audit for Wessels global-prior stablecaps diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = ("test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")
MMD_GATE_KEYS = ("test_mmd_clamped", "test_mmd_biased", "test_mmd")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group(payload: dict[str, Any], name: str) -> dict[str, Any]:
    item = payload.get("groups", {}).get(name, {})
    return item if isinstance(item, dict) else {}


def selected_keys(payload: dict[str, Any], group_name: str) -> list[str]:
    keys: list[str] = []
    for row in group(payload, group_name).get("selected_conditions", []) or []:
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset") or "")
        condition = str(row.get("condition") or "")
        if dataset and condition:
            keys.append(f"{dataset}\t{condition}")
    return sorted(keys)


def metric_delta(base_group: dict[str, Any], run_group: dict[str, Any], key: str) -> dict[str, Any]:
    base = fnum(base_group.get(key))
    value = fnum(run_group.get(key))
    return {
        "baseline": base,
        "value": value,
        "delta": None if base is None or value is None else value - base,
    }


def common_mmd_key(base_group: dict[str, Any], run_group: dict[str, Any]) -> str:
    for key in MMD_GATE_KEYS:
        if base_group.get(key) is not None and run_group.get(key) is not None:
            return key
    return "test_mmd"


def compare_selected(base_payload: dict[str, Any], run_payload: dict[str, Any], group_name: str) -> dict[str, Any]:
    base_keys = selected_keys(base_payload, group_name)
    run_keys = selected_keys(run_payload, group_name)
    base_set = set(base_keys)
    run_set = set(run_keys)
    return {
        "group": group_name,
        "equal": base_keys == run_keys,
        "baseline_n": len(base_keys),
        "run_n": len(run_keys),
        "baseline_only_n": len(base_set - run_set),
        "run_only_n": len(run_set - base_set),
        "baseline_only_examples": sorted(base_set - run_set)[:10],
        "run_only_examples": sorted(run_set - base_set)[:10],
    }


def audit_run(baseline_payload: dict[str, Any], run_name: str, run_json: Path) -> dict[str, Any]:
    run_payload = load_json(run_json)
    selection = [compare_selected(baseline_payload, run_payload, name) for name in GROUPS]
    selection_ok = all(row["equal"] for row in selection)
    groups: dict[str, Any] = {}
    for name in GROUPS:
        bg = group(baseline_payload, name)
        rg = group(run_payload, name)
        mmd_key = common_mmd_key(bg, rg)
        base_mmd = fnum(bg.get(mmd_key))
        run_mmd = fnum(rg.get(mmd_key))
        groups[name] = {
            "n_conds": rg.get("n_conds"),
            "pearson_pert": metric_delta(bg, rg, "pearson_pert"),
            "pearson_ctrl": metric_delta(bg, rg, "pearson_ctrl"),
            "direct_pearson": metric_delta(bg, rg, "direct_pearson"),
            "mmd_gate_metric": mmd_key,
            "mmd_gate": {
                "baseline": base_mmd,
                "value": run_mmd,
                "ratio": None if base_mmd is None or run_mmd is None else run_mmd / max(base_mmd, 1e-12),
            },
        }
    unseen2_pp_delta = (groups["test_multi_unseen2"]["pearson_pert"] or {}).get("delta")
    test_mmd_ratio = (groups["test"]["mmd_gate"] or {}).get("ratio")
    status = "invalid_selection_mismatch"
    if selection_ok:
        status = (
            "pass"
            if unseen2_pp_delta is not None
            and unseen2_pp_delta >= 0.05
            and test_mmd_ratio is not None
            and test_mmd_ratio <= 1.15
            else "fail"
        )
    return {
        "run": run_name,
        "run_json": str(run_json),
        "status": status,
        "selection": selection,
        "groups": groups,
        "unseen2_pp_delta": unseen2_pp_delta,
        "test_mmd_ratio": test_mmd_ratio,
    }


def audit(baseline_name: str, baseline_json: Path, runs: list[tuple[str, Path]]) -> dict[str, Any]:
    baseline_payload = load_json(baseline_json)
    rows = [audit_run(baseline_payload, name, path) for name, path in runs]
    passed = [row for row in rows if row["status"] == "pass"]
    valid = [row for row in rows if row["status"] != "invalid_selection_mismatch"]
    if passed:
        best = sorted(
            passed,
            key=lambda row: (
                fnum(row.get("unseen2_pp_delta")) or -999.0,
                -(fnum(row.get("test_mmd_ratio")) or 999.0),
            ),
            reverse=True,
        )[0]
        next_action = "promote_best_global_prior_to_all_split_diagnostic"
        decision_status = "pass"
        reason = "at least one Wessels global-prior diagnostic passed the strict stablecaps gate"
    elif len(valid) != len(rows):
        best = {}
        next_action = "rerun_or_reaudit_global_prior_selection_mismatch"
        decision_status = "invalid_selection_mismatch"
        reason = "one or more runs selected a different stablecaps condition set than the baseline"
    else:
        best = sorted(
            valid,
            key=lambda row: (
                fnum(row.get("unseen2_pp_delta")) if fnum(row.get("unseen2_pp_delta")) is not None else -999.0,
                -(fnum(row.get("test_mmd_ratio")) if fnum(row.get("test_mmd_ratio")) is not None else 999.0),
            ),
            reverse=True,
        )[0] if valid else {}
        next_action = "design_context_conditioned_prior_or_interaction_residual"
        decision_status = "fail"
        reason = "global train-only gene-mean prior did not pass Wessels unseen2/MMD gate"
    return {
        "baseline": baseline_name,
        "baseline_json": str(baseline_json),
        "runs": rows,
        "decision": {
            "status": decision_status,
            "next_action": next_action,
            "best_run": best.get("run"),
            "reason": reason,
        },
        "gate_rule": {
            "unseen2_pp_delta_min": 0.05,
            "test_mmd_ratio_max": 1.15,
            "requires_identical_selected_conditions": True,
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels Global Prior Gate Audit",
        "",
        f"Baseline: `{payload['baseline']}`",
        f"Baseline JSON: `{payload['baseline_json']}`",
        "",
        f"Decision: `{payload['decision']['status']}`",
        f"Next action: `{payload['decision']['next_action']}`",
        f"Reason: {payload['decision']['reason']}",
        "",
        "This CPU-only audit checks baseline-vs-candidate selected-condition identity before interpreting stablecaps deltas.",
        "",
        "## Gate Summary",
        "",
        "| run | status | mismatch groups | unseen2 pp delta | test MMD ratio | MMD metric |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["runs"]:
        mismatches = [item["group"] for item in row["selection"] if not item["equal"]]
        mmd_metric = row["groups"]["test"]["mmd_gate_metric"]
        lines.append(
            f"| `{row['run']}` | {row['status']} | {len(mismatches)} | "
            f"{fmt(row['unseen2_pp_delta'])} | {fmt(row['test_mmd_ratio'])} | `{mmd_metric}` |"
        )
    lines.extend(
        [
            "",
            "## Group Details",
            "",
            "| run | group | n | pp baseline | pp | delta pp | pc baseline | pc | delta pc | MMD baseline | MMD | MMD ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["runs"]:
        for group_name in GROUPS:
            item = row["groups"][group_name]
            pp = item["pearson_pert"]
            pc = item["pearson_ctrl"]
            mmd = item["mmd_gate"]
            lines.append(
                f"| `{row['run']}` | `{group_name}` | {item['n_conds']} | "
                f"{fmt(pp['baseline'])} | {fmt(pp['value'])} | {fmt(pp['delta'])} | "
                f"{fmt(pc['baseline'])} | {fmt(pc['value'])} | {fmt(pc['delta'])} | "
                f"{fmt(mmd['baseline'])} | {fmt(mmd['value'])} | {fmt(mmd['ratio'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These Wessels-only stablecaps results are diagnostic. Promotion still requires all-split retraining, condition-uncapped posthoc, paired condition bootstrap/CI, and explicit leakage/provenance audit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-name", default="scf_prior010_upperbound_wessels_4k")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path(
            "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/dataset_upper_bound_20260620/"
            "scf_prior010_upperbound_wessels_4k/posthoc_eval_upperbound/"
            "split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
        ),
    )
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_global_prior_gate_audit_20260620.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_GATE_AUDIT_20260620.md"),
    )
    args = parser.parse_args()

    payload = audit(args.baseline_name, args.baseline_json, [(name, Path(path)) for name, path in args.run])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "decision": payload["decision"]}, indent=2))
    return 0 if payload["decision"]["status"] != "invalid_selection_mismatch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
