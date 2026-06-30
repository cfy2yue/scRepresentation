#!/usr/bin/env python3
"""Evaluate Track C support-context route-gap closure on support trainselect only.

Inputs are per-condition support-val posthoc JSONs plus the frozen support
route readout artifact. Held-out Track C query artifacts are not read.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_trackc_support_context_20260622"
REPORT_DIR = ROOT / "reports"
READOUT_JSON = REPORT_DIR / "latentfm_trackc_support_memory_readout_gate_20260622.json"
PASS_STATUS = "route_gap_gate_pass"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def support_condition_rows(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    groups = payload.get("groups") or {}
    for group in ("test_multi", "test"):
        body = groups.get(group) or {}
        rows = body.get("condition_metrics") or []
        if rows:
            return group, rows
    return "missing", []


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset")), str(row.get("condition"))


def readout_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    return {
        row_key(row): row
        for row in payload.get("condition_rows") or []
        if isinstance(row, dict)
    }


def paired_condition_rows(
    *,
    anchor_json: Path,
    candidate_json: Path,
    readout_json: Path,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    anchor_payload = load_json(anchor_json)
    candidate_payload = load_json(candidate_json)
    anchor_group, anchor_rows = support_condition_rows(anchor_payload)
    candidate_group, candidate_rows = support_condition_rows(candidate_payload)
    if anchor_group != candidate_group:
        reasons.append(f"support_group_mismatch_{anchor_group}_vs_{candidate_group}")
    if not anchor_rows:
        reasons.append("missing_anchor_support_condition_rows")
    if not candidate_rows:
        reasons.append("missing_candidate_support_condition_rows")

    anchors = {row_key(row): row for row in anchor_rows}
    candidates = {row_key(row): row for row in candidate_rows}
    readout = readout_index(readout_json)
    target_datasets = {"NormanWeissman2019_filtered", "Wessels"}
    expected_keys = {key for key in readout if key[0] in target_datasets}
    missing_anchor = sorted(expected_keys - set(anchors))
    missing_candidate = sorted(expected_keys - set(candidates))
    if missing_anchor:
        examples = ",".join(f"{ds}:{cond}" for ds, cond in missing_anchor[:3])
        reasons.append(f"missing_anchor_support_conditions_{len(missing_anchor)}_examples_{examples}")
    if missing_candidate:
        examples = ",".join(f"{ds}:{cond}" for ds, cond in missing_candidate[:3])
        reasons.append(f"missing_candidate_support_conditions_{len(missing_candidate)}_examples_{examples}")
    rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        if key not in anchors or key not in candidates:
            continue
        ds, cond = key
        anchor_pp = to_float(anchors[key].get("pearson_pert"))
        candidate_pp = to_float(candidates[key].get("pearson_pert"))
        route_pp = to_float(readout[key].get("support_selected_route"))
        if anchor_pp is None or candidate_pp is None or route_pp is None:
            continue
        route_gap = route_pp - anchor_pp
        delta = candidate_pp - anchor_pp
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "support_group": candidate_group,
                "anchor_pp": anchor_pp,
                "candidate_pp": candidate_pp,
                "support_selected_route_pp": route_pp,
                "delta_pp": delta,
                "route_gap_from_anchor_pp": route_gap,
                "route_gap_closed_fraction": (delta / route_gap) if abs(route_gap) > 1e-12 else None,
                "candidate_above_anchor": candidate_pp > anchor_pp,
                "candidate_above_route": candidate_pp > route_pp,
            }
        )
    if not rows and not reasons:
        reasons.append("no_matched_support_route_condition_rows")
    return candidate_group, rows, reasons


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    out: list[dict[str, Any]] = []
    for ds, items in sorted(grouped.items()):
        deltas = [to_float(row.get("delta_pp")) for row in items]
        deltas = [x for x in deltas if x is not None]
        route_gaps = [to_float(row.get("route_gap_from_anchor_pp")) for row in items]
        route_gaps = [x for x in route_gaps if x is not None]
        positive_pairs = [
            (to_float(row.get("delta_pp")), to_float(row.get("route_gap_from_anchor_pp")))
            for row in items
        ]
        positive_pairs = [
            (delta, gap)
            for delta, gap in positive_pairs
            if delta is not None and gap is not None and gap > 0
        ]
        weighted = None
        if positive_pairs:
            weighted = sum(delta for delta, _ in positive_pairs) / sum(gap for _, gap in positive_pairs)
        closures = [to_float(row.get("route_gap_closed_fraction")) for row in items]
        closures = [x for x in closures if x is not None]
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(items),
                "mean_delta_pp": mean(deltas) if deltas else None,
                "median_delta_pp": median(deltas) if deltas else None,
                "mean_route_gap_from_anchor_pp": mean(route_gaps) if route_gaps else None,
                "weighted_route_gap_closure": weighted,
                "mean_route_gap_closure": mean(closures) if closures else None,
                "candidate_above_route_count": sum(1 for row in items if row.get("candidate_above_route") is True),
            }
        )
    return out


def dataset_summary(summary: list[dict[str, Any]], dataset: str) -> dict[str, Any] | None:
    for row in summary:
        if row.get("dataset") == dataset:
            return row
    return None


def evaluate_gate(summary: list[dict[str, Any]], prior_reasons: list[str]) -> dict[str, Any]:
    reasons = list(prior_reasons)
    wessels = dataset_summary(summary, "Wessels")
    norman = dataset_summary(summary, "NormanWeissman2019_filtered")
    if not wessels:
        reasons.append("missing_wessels_route_gap_rows")
    if not norman:
        reasons.append("missing_norman_route_gap_rows")
    if not reasons:
        wessels_delta = to_float(wessels.get("mean_delta_pp"))
        wessels_closure = to_float(wessels.get("weighted_route_gap_closure"))
        norman_delta = to_float(norman.get("mean_delta_pp"))
        if wessels_delta is None:
            reasons.append("missing_wessels_support_pp_delta")
        elif wessels_delta < 0.02:
            reasons.append("wessels_support_pp_delta_below_0p02")
        if wessels_closure is None:
            reasons.append("missing_wessels_route_gap_closure")
        elif wessels_closure < 0.05:
            reasons.append("wessels_route_gap_closure_below_0p05")
        if norman_delta is None:
            reasons.append("missing_norman_support_pp_delta")
        elif norman_delta < -0.01:
            reasons.append("norman_positive_signal_loss_delta_below_minus_0p01")
        norman_closure = to_float(norman.get("weighted_route_gap_closure"))
        if norman_closure is not None and norman_closure < -0.05:
            reasons.append("norman_positive_signal_loss_route_closure_below_minus_0p05")

    if not reasons:
        status = PASS_STATUS
        action = "route_gap_gate_passed_allow_summary_to_consider_uncapped_noharm"
    elif any(reason.startswith("missing_") or reason.startswith("support_group_") for reason in reasons):
        status = "route_gap_gate_missing_required_metrics"
        action = "fail_closed_and_audit_support_posthoc_or_readout_coverage"
    else:
        status = "route_gap_gate_fail_close_branch"
        action = "close_support_context_smoke_or_redesign_context_mechanism"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "Wessels support-val mean pearson_pert delta >= +0.02",
            "Wessels weighted route-gap closure >= +0.05, using positive route gaps only",
            "Norman mean pearson_pert delta must not fall below -0.01",
            "Norman weighted route-gap closure must not fall below -0.05 when defined",
            "inputs are support trainselect posthoc plus frozen support route readout only; held-out query is forbidden",
        ],
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context Route-Gap Gate",
        "",
        f"Run: `{payload['run_name']}`",
        f"Status: `{payload['decision']['status']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Scope",
        "",
        "This gate reads support trainselect posthoc condition metrics and the frozen support route readout only.",
        "Held-out Track C query artifacts are not read.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | n | mean delta | route gap | weighted route-gap closure | candidate > route |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset")),
                    str(row.get("n_conditions")),
                    fmt(row.get("mean_delta_pp")),
                    fmt(row.get("mean_route_gap_from_anchor_pp")),
                    fmt(row.get("weighted_route_gap_closure")),
                    str(row.get("candidate_above_route_count")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Inputs", ""])
    for key, value in payload["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--readout-json", type=Path, default=READOUT_JSON)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    run_root = args.run_root or (RUN_ROOT / args.run_name)
    posthoc = run_root / "posthoc_eval"
    anchor_json = posthoc / "support_anchor_split_ode20.json"
    candidate_json = posthoc / "support_candidate_split_ode20.json"
    out_json = args.out_json or (REPORT_DIR / f"latentfm_trackc_support_context_route_gap_gate_{args.run_name}.json")
    out_md = args.out_md or (REPORT_DIR / f"LATENTFM_TRACKC_SUPPORT_CONTEXT_ROUTE_GAP_GATE_{args.run_name}.md")

    required = {
        "support_anchor_split": anchor_json,
        "support_candidate_split": candidate_json,
        "support_route_readout": args.readout_json,
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required route-gap gate inputs: " + ", ".join(missing))

    support_group, rows, prior_reasons = paired_condition_rows(
        anchor_json=anchor_json,
        candidate_json=candidate_json,
        readout_json=args.readout_json,
    )
    summary = summarize(rows)
    decision = evaluate_gate(summary, prior_reasons)
    payload = {
        "run_name": args.run_name,
        "run_root": str(run_root),
        "support_group": support_group,
        "heldout_query_used": False,
        "inputs": {name: str(path) for name, path in required.items()},
        "n_condition_rows": len(rows),
        "summary": summary,
        "decision": decision,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
