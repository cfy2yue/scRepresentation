#!/usr/bin/env python3
"""Evaluate Track C residual-operator route-gap closure on support-val only.

Inputs are support trainselect posthoc condition metrics plus the frozen CPU
residual-operator gate. Held-out Track C query and canonical Track A selection
artifacts are not read.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CPU_GATE_JSON = ROOT / "reports/latentfm_trackc_residual_operator_cpu_gate_20260623.json"
PASS_STATUS = "residual_route_gap_gate_pass"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: Any) -> str:
    value = to_float(value)
    return "NA" if value is None else f"{value:+.6f}"


def support_rows(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    groups = payload.get("groups") or {}
    for group in ("test_multi", "test"):
        rows = ((groups.get(group) or {}).get("condition_metrics") or [])
        if rows:
            return group, rows
    return "missing", []


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset")), str(row.get("condition"))


def target_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    decision = payload.get("decision") or {}
    if decision.get("status") != "residual_operator_cpu_gate_pass_authorize_one_capped_gpu_smoke":
        raise RuntimeError(f"CPU residual gate is not pass: {decision.get('status')}")
    return {
        row_key(row): row
        for row in payload.get("eval_rows") or []
        if isinstance(row, dict)
    }


def paired_condition_rows(
    *,
    anchor_json: Path,
    candidate_json: Path,
    cpu_gate_json: Path,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    anchor_group, anchor_rows = support_rows(load_json(anchor_json))
    candidate_group, candidate_rows = support_rows(load_json(candidate_json))
    if anchor_group != candidate_group:
        reasons.append(f"support_group_mismatch_{anchor_group}_vs_{candidate_group}")
    if not anchor_rows:
        reasons.append("missing_anchor_support_condition_rows")
    if not candidate_rows:
        reasons.append("missing_candidate_support_condition_rows")

    anchors = {row_key(row): row for row in anchor_rows}
    candidates = {row_key(row): row for row in candidate_rows}
    targets = target_index(cpu_gate_json)
    expected = {
        key
        for key in targets
        if key[0] in {"NormanWeissman2019_filtered", "Wessels"}
    }
    missing_anchor = sorted(expected - set(anchors))
    missing_candidate = sorted(expected - set(candidates))
    if missing_anchor:
        reasons.append(f"missing_anchor_support_conditions_{len(missing_anchor)}")
    if missing_candidate:
        reasons.append(f"missing_candidate_support_conditions_{len(missing_candidate)}")

    rows = []
    for key in sorted(expected):
        if key not in anchors or key not in candidates:
            continue
        ds, cond = key
        anchor_pp = to_float(anchors[key].get("pearson_pert"))
        candidate_pp = to_float(candidates[key].get("pearson_pert"))
        target_pp = to_float(targets[key].get("candidate"))
        if anchor_pp is None or candidate_pp is None or target_pp is None:
            continue
        route_gap = target_pp - anchor_pp
        delta = candidate_pp - anchor_pp
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "support_group": candidate_group,
                "anchor_pp": anchor_pp,
                "candidate_pp": candidate_pp,
                "cpu_residual_target_pp": target_pp,
                "delta_pp": delta,
                "route_gap_from_anchor_pp": route_gap,
                "route_gap_closed_fraction": None if abs(route_gap) <= 1e-12 else delta / route_gap,
                "candidate_above_cpu_target": candidate_pp > target_pp,
            }
        )
    if not rows and not reasons:
        reasons.append("no_matched_residual_route_condition_rows")
    return candidate_group, rows, reasons


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
    out = []
    for ds, items in sorted(by_ds.items()):
        deltas = [to_float(row.get("delta_pp")) for row in items]
        gaps = [to_float(row.get("route_gap_from_anchor_pp")) for row in items]
        deltas = [x for x in deltas if x is not None]
        gaps = [x for x in gaps if x is not None]
        positive_pairs = [
            (to_float(row.get("delta_pp")), to_float(row.get("route_gap_from_anchor_pp")))
            for row in items
        ]
        positive_pairs = [
            (delta, gap)
            for delta, gap in positive_pairs
            if delta is not None and gap is not None and gap > 0
        ]
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(items),
                "mean_delta_pp": mean(deltas) if deltas else None,
                "mean_route_gap_from_anchor_pp": mean(gaps) if gaps else None,
                "weighted_route_gap_closure": None
                if not positive_pairs
                else sum(delta for delta, _ in positive_pairs) / sum(gap for _, gap in positive_pairs),
                "candidate_above_cpu_target_count": sum(1 for row in items if row.get("candidate_above_cpu_target")),
            }
        )
    return out


def by_dataset(summary: list[dict[str, Any]], dataset: str) -> dict[str, Any] | None:
    for row in summary:
        if row.get("dataset") == dataset:
            return row
    return None


def evaluate_gate(summary: list[dict[str, Any]], prior_reasons: list[str]) -> dict[str, Any]:
    reasons = list(prior_reasons)
    wessels = by_dataset(summary, "Wessels")
    norman = by_dataset(summary, "NormanWeissman2019_filtered")
    if not wessels:
        reasons.append("missing_wessels_residual_route_rows")
    if not norman:
        reasons.append("missing_norman_residual_route_rows")
    if not reasons:
        wessels_delta = to_float(wessels.get("mean_delta_pp"))
        wessels_closure = to_float(wessels.get("weighted_route_gap_closure"))
        norman_delta = to_float(norman.get("mean_delta_pp"))
        if wessels_delta is None or wessels_delta < 0.02:
            reasons.append("wessels_support_pp_delta_below_0p02")
        if wessels_closure is None or wessels_closure < 0.05:
            reasons.append("wessels_residual_route_gap_closure_below_0p05")
        if norman_delta is None or norman_delta < -0.02:
            reasons.append("norman_material_pp_loss")
    if not reasons:
        status = PASS_STATUS
        action = "residual_route_gap_pass_allow_smoke_summary_to_consider_uncapped_noharm"
    elif any(reason.startswith("missing_") or reason.startswith("support_group_") for reason in reasons):
        status = "residual_route_gap_gate_missing_required_metrics"
        action = "fail_closed_and_audit_support_posthoc_coverage"
    else:
        status = "residual_route_gap_gate_fail_close_branch"
        action = "close_residual_operator_smoke_or_redesign_cpu_gate"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "Wessels support-val mean pearson_pert delta >= +0.02",
            "Wessels weighted residual route-gap closure >= +0.05",
            "Norman mean pearson_pert delta >= -0.02",
            "inputs are support trainselect posthoc plus frozen CPU residual gate only; held-out query is forbidden",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Residual-Operator Route-Gap Gate",
        "",
        f"Run: `{payload['run_name']}`",
        f"Status: `{payload['decision']['status']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Scope",
        "",
        "This gate reads support trainselect posthoc condition metrics and the frozen CPU residual-operator gate only.",
        "Held-out Track C query artifacts are not read.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | n | mean delta | residual route gap | weighted closure | candidate > CPU target |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary") or []:
        lines.append(
            f"| {row.get('dataset')} | {row.get('n_conditions')} | {fmt(row.get('mean_delta_pp'))} | "
            f"{fmt(row.get('mean_route_gap_from_anchor_pp'))} | {fmt(row.get('weighted_route_gap_closure'))} | "
            f"{row.get('candidate_above_cpu_target_count')} |"
        )
    lines.extend(["", "## Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(["", "## Rules", ""])
    lines.extend(f"- {rule}" for rule in payload["decision"].get("rules") or [])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--anchor-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--cpu-gate-json", type=Path, default=CPU_GATE_JSON)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    group, rows, reasons = paired_condition_rows(
        anchor_json=args.anchor_json,
        candidate_json=args.candidate_json,
        cpu_gate_json=args.cpu_gate_json,
    )
    summary = summarize(rows)
    decision = evaluate_gate(summary, reasons)
    payload = {
        "run_name": args.run_name,
        "support_group": group,
        "inputs": {
            "anchor_json": str(args.anchor_json),
            "candidate_json": str(args.candidate_json),
            "cpu_gate_json": str(args.cpu_gate_json),
        },
        "heldout_query_used": False,
        "canonical_outputs_used": False,
        "n_rows": len(rows),
        "rows": rows,
        "summary": summary,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(args.out_md)}, indent=2))
    return 0 if decision["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
