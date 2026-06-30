#!/usr/bin/env python3
"""Evaluate Track C support-FiLM route-gap closure on support-val only.

Inputs are support trainselect posthoc condition metrics plus the frozen
alternative support-conditioning CPU gate. Held-out Track C query and canonical
Track A selection artifacts are not read.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CPU_GATE_JSON = ROOT / "reports/latentfm_trackc_alternative_support_conditioning_cpu_gate_20260623.json"
PASS_STATUS = "support_film_route_gap_gate_pass"


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


def target_by_dataset(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    top_status = payload.get("status")
    decision = ((payload.get("real") or {}).get("decision") or {})
    if top_status != "trackc_alternative_support_conditioning_cpu_gate_pass_authorize_one_capped_gpu_smoke":
        raise RuntimeError(f"alternative CPU gate is not pass: {top_status}")
    if decision.get("gpu_authorization") != "one_capped_trackc_support_only_smoke":
        raise RuntimeError(f"alternative CPU gate has wrong authorization: {decision.get('gpu_authorization')}")
    rows = (payload.get("real") or {}).get("dataset_breakdown") or []
    return {str(row.get("dataset")): row for row in rows if isinstance(row, dict)}


def paired_condition_rows(
    *,
    anchor_json: Path,
    candidate_json: Path,
    cpu_gate_json: Path,
) -> tuple[str, list[dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
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
    targets = target_by_dataset(cpu_gate_json)
    expected_datasets = {"NormanWeissman2019_filtered", "Wessels"}
    missing_targets = sorted(expected_datasets - set(targets))
    if missing_targets:
        reasons.append(f"missing_cpu_gate_dataset_targets_{len(missing_targets)}")

    expected = {
        key
        for key in set(anchors) | set(candidates)
        if key[0] in expected_datasets
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
        if anchor_pp is None or candidate_pp is None:
            continue
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "support_group": candidate_group,
                "anchor_pp": anchor_pp,
                "candidate_pp": candidate_pp,
                "delta_pp": candidate_pp - anchor_pp,
            }
        )
    if not rows and not reasons:
        reasons.append("no_matched_support_film_route_condition_rows")
    return candidate_group, rows, reasons, targets


def summarize(rows: list[dict[str, Any]], targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
    out = []
    for ds, items in sorted(by_ds.items()):
        target = targets.get(ds) or {}
        deltas = [to_float(row.get("delta_pp")) for row in items]
        anchors = [to_float(row.get("anchor_pp")) for row in items]
        candidates = [to_float(row.get("candidate_pp")) for row in items]
        deltas = [x for x in deltas if x is not None]
        anchors = [x for x in anchors if x is not None]
        candidates = [x for x in candidates if x is not None]
        cpu_route = to_float(target.get("support_selected_route"))
        cpu_candidate = to_float(target.get("candidate"))
        cpu_gap = None if cpu_route is None or cpu_candidate is None else cpu_candidate - cpu_route
        mean_delta = mean(deltas) if deltas else None
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(items),
                "mean_anchor_pp": mean(anchors) if anchors else None,
                "mean_candidate_pp": mean(candidates) if candidates else None,
                "mean_delta_pp": mean_delta,
                "cpu_gate_route_pp": cpu_route,
                "cpu_gate_target_pp": cpu_candidate,
                "cpu_gate_route_gap_pp": cpu_gap,
                "route_gap_closed_fraction": None
                if mean_delta is None or cpu_gap is None or abs(cpu_gap) <= 1e-12
                else mean_delta / cpu_gap,
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
        reasons.append("missing_wessels_support_film_route_rows")
    if not norman:
        reasons.append("missing_norman_support_film_route_rows")
    if not reasons:
        wessels_delta = to_float(wessels.get("mean_delta_pp"))
        wessels_closure = to_float(wessels.get("route_gap_closed_fraction"))
        norman_delta = to_float(norman.get("mean_delta_pp"))
        if wessels_delta is None or wessels_delta < 0.02:
            reasons.append("wessels_support_pp_delta_below_0p02")
        if wessels_closure is None or wessels_closure < 0.05:
            reasons.append("wessels_support_film_route_gap_closure_below_0p05")
        if norman_delta is None or norman_delta < -0.02:
            reasons.append("norman_material_pp_loss")
    if not reasons:
        status = PASS_STATUS
        action = "support_film_route_gap_pass_allow_smoke_summary_to_consider_uncapped_noharm"
    elif any(reason.startswith("missing_") or reason.startswith("support_group_") for reason in reasons):
        status = "support_film_route_gap_gate_missing_required_metrics"
        action = "fail_closed_and_audit_support_posthoc_coverage"
    else:
        status = "support_film_route_gap_gate_fail_close_branch"
        action = "close_support_film_smoke_or_redesign_model_facing_support_operator"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "Wessels support-val mean pearson_pert delta >= +0.02",
            "Wessels support-FiLM route-gap closure against frozen CPU gate >= +0.05",
            "Norman mean pearson_pert delta >= -0.02",
            "inputs are support trainselect posthoc plus frozen alternative CPU gate only; held-out query is forbidden",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-FiLM Route-Gap Gate",
        "",
        f"Run: `{payload['run_name']}`",
        f"Status: `{payload['decision']['status']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Scope",
        "",
        "This gate reads support trainselect posthoc condition metrics and the frozen alternative support-conditioning CPU gate only.",
        "Held-out Track C query artifacts are not read.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | n | mean delta | CPU route gap | closure | anchor pp | candidate pp | CPU target pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary") or []:
        lines.append(
            f"| {row.get('dataset')} | {row.get('n_conditions')} | {fmt(row.get('mean_delta_pp'))} | "
            f"{fmt(row.get('cpu_gate_route_gap_pp'))} | {fmt(row.get('route_gap_closed_fraction'))} | "
            f"{fmt(row.get('mean_anchor_pp'))} | {fmt(row.get('mean_candidate_pp'))} | "
            f"{fmt(row.get('cpu_gate_target_pp'))} |"
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

    group, rows, reasons, targets = paired_condition_rows(
        anchor_json=args.anchor_json,
        candidate_json=args.candidate_json,
        cpu_gate_json=args.cpu_gate_json,
    )
    summary = summarize(rows, targets)
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
