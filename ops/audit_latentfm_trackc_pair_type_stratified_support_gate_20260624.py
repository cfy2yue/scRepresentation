#!/usr/bin/env python3
"""CPU gate for Track C pair-type stratified support protocol.

This gate reads only the safe trainselect split and already-frozen support-val
posthoc JSONs from the primary support-context v2 run. It does not read held-out
query, canonical metrics, active logs, or launch GPU work.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
RUN_ROOT = (
    ROOT
    / "runs/latentfm_xverse_trackc_support_context_v2_20260623"
    / "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"
)
POSTHOC = RUN_ROOT / "posthoc_eval"
REPORT_JSON = ROOT / "reports/latentfm_trackc_pair_type_stratified_support_gate_20260624.json"
REPORT_MD = ROOT / "reports/LATENTFM_TRACKC_PAIR_TYPE_STRATIFIED_SUPPORT_GATE_20260624.md"

FILES = {
    "anchor": POSTHOC / "support_anchor_split_ode20.json",
    "actual": POSTHOC / "support_candidate_split_ode20.json",
    "zero": POSTHOC / "support_zero_candidate_split_ode20.json",
    "shuffle": POSTHOC / "support_shuffle_condition_candidate_split_ode20.json",
    "absent": POSTHOC / "support_absent_support_candidate_split_ode20.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_genes(condition: str) -> tuple[str, str] | None:
    parts = [x.strip() for x in condition.split("+") if x.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def condition_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    groups = payload.get("groups") or {}
    rows = (groups.get("test_multi") or groups.get("test") or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def split_strata(split: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for ds, groups in split.items():
        train_single = set(str(x) for x in groups.get("train_single") or [])
        train_multi_pairs = [pair_genes(str(x)) for x in groups.get("train_multi") or []]
        train_multi_genes = {g for pair in train_multi_pairs if pair for g in pair}
        for cond in groups.get("support_val_multi") or []:
            cond_s = str(cond)
            pair = pair_genes(cond_s)
            if not pair:
                continue
            single_cov = sum(g in train_single for g in pair)
            multi_gene_cov = sum(g in train_multi_genes for g in pair)
            if single_cov == 2:
                single_label = "both_train_single"
            elif single_cov == 1:
                single_label = "one_train_single"
            else:
                single_label = "none_train_single"
            if multi_gene_cov == 2:
                multi_label = "both_train_multi_gene"
            elif multi_gene_cov == 1:
                multi_label = "one_train_multi_gene"
            else:
                multi_label = "none_train_multi_gene"
            out[(str(ds), cond_s)] = {
                "dataset": str(ds),
                "condition": cond_s,
                "genes": pair,
                "single_cov": single_cov,
                "multi_gene_cov": multi_gene_cov,
                "single_label": single_label,
                "multi_label": multi_label,
                "joint_label": f"{single_label}|{multi_label}",
            }
    return out


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def stratified_delta(
    *,
    anchor: dict[tuple[str, str], dict[str, Any]],
    candidate: dict[tuple[str, str], dict[str, Any]],
    strata: dict[tuple[str, str], dict[str, Any]],
    label_key: str,
    metric: str,
) -> dict[str, Any]:
    buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, meta in strata.items():
        if key not in anchor or key not in candidate:
            continue
        a = fnum(anchor[key].get(metric))
        c = fnum(candidate[key].get(metric))
        if a is None or c is None:
            continue
        buckets[str(meta[label_key])].append((str(meta["dataset"]), c - a))
    out: dict[str, Any] = {}
    for label, rows in sorted(buckets.items()):
        by_ds: dict[str, list[float]] = defaultdict(list)
        for ds, delta in rows:
            by_ds[ds].append(delta)
        ds_means = {ds: sum(vals) / len(vals) for ds, vals in by_ds.items() if vals}
        out[label] = {
            "n_conditions": len(rows),
            "n_datasets": len(ds_means),
            "dataset_means": ds_means,
            "equal_dataset_mean_delta": sum(ds_means.values()) / max(1, len(ds_means)),
            "min_dataset_delta": min(ds_means.values()) if ds_means else None,
        }
    return out


def gate_table(
    *,
    actual_pp: dict[str, Any],
    actual_mmd: dict[str, Any],
    control_pp: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    pass_labels: list[str] = []
    labels = sorted(actual_pp)
    for label in labels:
        app = actual_pp.get(label) or {}
        amm = actual_mmd.get(label) or {}
        controls = {name: (table.get(label) or {}) for name, table in control_pp.items()}
        reasons: list[str] = []
        n = int(app.get("n_conditions") or 0)
        nd = int(app.get("n_datasets") or 0)
        app_delta = float(app.get("equal_dataset_mean_delta") or 0.0)
        amm_delta = float(amm.get("equal_dataset_mean_delta") or 999.0)
        min_ds = app.get("min_dataset_delta")
        if n < 4:
            reasons.append("n_conditions_lt_4")
        if nd < 2:
            reasons.append("n_datasets_lt_2")
        if app_delta < 0.04:
            reasons.append("actual_pp_delta_lt_0p04")
        if amm_delta > 0.0:
            reasons.append("actual_mmd_positive")
        if min_ds is None or float(min_ds) < -0.01:
            reasons.append("dataset_tail_pp_harm")
        for name, row in controls.items():
            c_delta = row.get("equal_dataset_mean_delta")
            if c_delta is None:
                reasons.append(f"{name}_missing")
                continue
            c_delta = float(c_delta)
            if c_delta > 0.02:
                reasons.append(f"{name}_control_pp_gt_0p02")
            if app_delta - c_delta < 0.02:
                reasons.append(f"{name}_not_0p02_below_actual")
        status = "pass_candidate_stratum" if not reasons else "fail"
        if not reasons:
            pass_labels.append(label)
        rows.append(
            {
                "label": label,
                "status": status,
                "reasons": reasons,
                "n_conditions": n,
                "n_datasets": nd,
                "actual_pp_delta": app_delta,
                "actual_mmd_delta": amm_delta,
                "actual_min_dataset_pp_delta": min_ds,
                "control_pp_delta": {
                    name: row.get("equal_dataset_mean_delta") for name, row in controls.items()
                },
                "dataset_means": app.get("dataset_means"),
            }
        )
    return rows, pass_labels


def main() -> int:
    missing = [str(path) for path in [SPLIT, *FILES.values()] if not path.exists()]
    reasons: list[str] = []
    if missing:
        reasons.append("missing_required_input")
    split = load(SPLIT) if not missing else {}
    strata = split_strata(split)
    payloads = {name: load(path) for name, path in FILES.items() if path.exists()}
    rows_by_file = {name: condition_rows(obj) for name, obj in payloads.items()}
    anchor = rows_by_file.get("anchor", {})

    results: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    passed: list[str] = []
    for label_key in ["single_label", "multi_label", "joint_label"]:
        actual_pp = stratified_delta(
            anchor=anchor, candidate=rows_by_file.get("actual", {}), strata=strata, label_key=label_key, metric="pearson_pert"
        )
        actual_mmd = stratified_delta(
            anchor=anchor, candidate=rows_by_file.get("actual", {}), strata=strata, label_key=label_key, metric="test_mmd_clamped"
        )
        control_pp = {
            name: stratified_delta(
                anchor=anchor, candidate=rows_by_file.get(name, {}), strata=strata, label_key=label_key, metric="pearson_pert"
            )
            for name in ["zero", "shuffle", "absent"]
        }
        rows, pass_labels = gate_table(actual_pp=actual_pp, actual_mmd=actual_mmd, control_pp=control_pp)
        results[label_key] = {
            "actual_pp": actual_pp,
            "actual_mmd": actual_mmd,
            "control_pp": control_pp,
            "gate_rows": rows,
            "passed_labels": pass_labels,
        }
        gate_rows.extend({**row, "label_key": label_key} for row in rows)
        passed.extend(f"{label_key}:{label}" for label in pass_labels)

    status = (
        "trackc_pair_type_stratified_support_gate_pass_gpu_protocol_design_next"
        if passed and not reasons
        else "trackc_pair_type_stratified_support_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized_by_this_script": False,
        "missing_inputs": missing,
        "reasons": reasons,
        "passed_strata": passed,
        "boundary": {
            "split": str(SPLIT),
            "run_root": str(RUN_ROOT),
            "reads_heldout_query": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "launches_gpu": False,
        },
        "gate_rules": {
            "n_conditions_min": 4,
            "n_datasets_min": 2,
            "actual_pp_delta_min": 0.04,
            "actual_mmd_delta_max": 0.0,
            "actual_min_dataset_pp_delta_floor": -0.01,
            "control_pp_delta_max": 0.02,
            "actual_minus_control_pp_delta_min": 0.02,
        },
        "gate_rows": gate_rows,
        "results": results,
        "next_action": (
            "design default-off pair-type mask/route support-only launcher"
            if status.endswith("design_next")
            else "do not launch pair-type GPU; consider coverage-floor CPU gate only"
        ),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:+.6f}"
        return str(x)

    lines = [
        "# Track C Pair-Type Stratified Support Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only; reads safe trainselect split and frozen support-val posthoc/control JSONs.",
        "- Does not read held-out query, canonical metrics, canonical multi, active logs, or launch GPU.",
        "",
        "## Gate Rows",
        "",
        "| label key | label | status | n | datasets | actual pp | actual MMD | min dataset pp | zero pp | shuffle pp | absent pp | reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate_rows:
        controls = row.get("control_pp_delta") or {}
        lines.append(
            f"| `{row['label_key']}` | `{row['label']}` | `{row['status']}` | "
            f"{row['n_conditions']} | {row['n_datasets']} | "
            f"{fmt(row['actual_pp_delta'])} | {fmt(row['actual_mmd_delta'])} | "
            f"{fmt(row['actual_min_dataset_pp_delta'])} | "
            f"{fmt(controls.get('zero'))} | {fmt(controls.get('shuffle'))} | {fmt(controls.get('absent'))} | "
            f"`{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- passed strata: `{passed}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{REPORT_JSON}`",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "passed_strata": passed, "out_md": str(REPORT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
