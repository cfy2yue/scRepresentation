#!/usr/bin/env python3
"""CPU gate for Track C single-gene support coverage-floor protocol.

This is a narrow follow-up to the pair-type gate. It asks whether support
context should be enabled only for multi-gene support rows whose two genes are
both covered by train-single conditions. It reads only the safe trainselect
split and frozen seed42 support-val posthoc/control JSONs through the
pair-type stratified gate output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
PAIR_GATE_JSON = ROOT / "reports/latentfm_trackc_pair_type_stratified_support_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_single_gene_coverage_floor_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SINGLE_GENE_COVERAGE_FLOOR_GATE_20260624.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    missing = [] if PAIR_GATE_JSON.exists() else [str(PAIR_GATE_JSON)]
    pair = load(PAIR_GATE_JSON) if not missing else {}
    rows = [
        row
        for row in pair.get("gate_rows", [])
        if row.get("label_key") == "single_label"
    ]
    by_label = {str(row.get("label")): row for row in rows}
    target = by_label.get("both_train_single") or {}
    reasons: list[str] = []
    if missing:
        reasons.append("missing_pair_type_gate_json")
    if not target:
        reasons.append("missing_both_train_single_row")
    else:
        if int(target.get("n_conditions") or 0) < 6:
            reasons.append("both_train_single_n_conditions_lt_6")
        if int(target.get("n_datasets") or 0) < 2:
            reasons.append("both_train_single_n_datasets_lt_2")
        if float(target.get("actual_pp_delta") or 0.0) < 0.03:
            reasons.append("both_train_single_pp_delta_lt_0p03")
        if float(target.get("actual_mmd_delta") or 999.0) > 0.0:
            reasons.append("both_train_single_mmd_positive")
        if target.get("actual_min_dataset_pp_delta") is None or float(target.get("actual_min_dataset_pp_delta")) < -0.01:
            reasons.append("both_train_single_dataset_tail_pp_harm")
        controls = target.get("control_pp_delta") or {}
        for name in ["zero", "shuffle", "absent"]:
            val = controls.get(name)
            if val is None:
                reasons.append(f"{name}_control_missing")
                continue
            if float(val) > 0.02:
                reasons.append(f"{name}_control_pp_gt_0p02")
            if float(target.get("actual_pp_delta") or 0.0) - float(val) < 0.02:
                reasons.append(f"{name}_not_0p02_below_actual")

    status = (
        "trackc_single_gene_coverage_floor_gate_pass_gpu_protocol_design_next"
        if not reasons
        else "trackc_single_gene_coverage_floor_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized_by_this_script": False,
        "missing_inputs": missing,
        "reasons": reasons,
        "target_row": target,
        "all_single_label_rows": rows,
        "boundary": {
            "source_gate": str(PAIR_GATE_JSON),
            "reads_heldout_query": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "launches_gpu": False,
        },
        "gate_rules": {
            "target": "single_label:both_train_single",
            "n_conditions_min": 6,
            "n_datasets_min": 2,
            "pp_delta_min": 0.03,
            "mmd_delta_max": 0.0,
            "min_dataset_pp_delta_floor": -0.01,
            "control_pp_delta_max": 0.02,
            "actual_minus_control_pp_delta_min": 0.02,
        },
        "next_action": (
            "design default-off both-train-single support mask"
            if status.endswith("design_next")
            else "do not launch coverage-floor GPU; pair-type mask remains the only support-stratified GPU branch"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:+.6f}"
        return str(x)

    controls = target.get("control_pp_delta") or {}
    lines = [
        "# Track C Single-Gene Coverage-Floor Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate derived from the pair-type support stratification report.",
        "- Does not read held-out query, canonical metrics, canonical multi, active logs, or launch GPU.",
        "",
        "## Target Row",
        "",
        "| label | status | n | datasets | actual pp | actual MMD | min dataset pp | zero pp | shuffle pp | absent pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `both_train_single` | `{target.get('status', 'missing')}` | "
            f"{target.get('n_conditions', 0)} | {target.get('n_datasets', 0)} | "
            f"{fmt(target.get('actual_pp_delta'))} | {fmt(target.get('actual_mmd_delta'))} | "
            f"{fmt(target.get('actual_min_dataset_pp_delta'))} | "
            f"{fmt(controls.get('zero'))} | {fmt(controls.get('shuffle'))} | {fmt(controls.get('absent'))} |"
        ),
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
