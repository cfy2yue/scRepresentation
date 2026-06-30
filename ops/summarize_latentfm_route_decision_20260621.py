#!/usr/bin/env python3
"""Summarize LatentFM routed-expert bootstrap results into a gate decision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PP_METRIC = "pearson_pert"
MMD_METRIC = "test_mmd_clamped"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def route_provenance(route: str) -> dict[str, Any]:
    """Best-effort route legality metadata from the route name.

    The route bootstrap file contains metric rows but not the upstream route
    construction history. Keep this conservative: diagnostic routes and
    split/focus/dataset-specific names are never deployable promotion rules.
    """

    lower = route.lower()
    uses_split_label = "unseen" in lower or "test_multi" in lower
    uses_dataset_id = "focus" in lower or "wessels" in lower or "norman" in lower or "gasperini" in lower
    diagnostic = lower.endswith("_diagnostic") or "diagnostic" in lower or uses_split_label or uses_dataset_id
    return {
        "route_feature_deployable": not diagnostic,
        "route_selected_predeclared": False,
        "uses_split_label": uses_split_label,
        "uses_dataset_id": uses_dataset_id,
        "uses_heldout_outcome_for_selection": True,
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["comparison"]), str(row["route"]), str(row["group"])


def index_rows(payloads: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for payload in payloads:
        for row in payload.get("rows", []):
            out[(str(row["comparison"]), str(row["route"]), str(row["group"]), str(row["metric"]))] = row
    return out


def all_comparison_routes(payloads: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            pairs.add((str(row["comparison"]), str(row["route"])))
    return sorted(pairs)


def assess_pair(rows: dict[tuple[str, str, str, str], dict[str, Any]], comparison: str, route: str) -> dict[str, Any]:
    def get(group: str, metric: str) -> dict[str, Any] | None:
        return rows.get((comparison, route, group, metric))

    required = {
        "test_pp": get("test", PP_METRIC),
        "test_mmd": get("test", MMD_METRIC),
        "family_gene_pp": get("family_gene", PP_METRIC),
        "family_gene_mmd": get("family_gene", MMD_METRIC),
        "unseen2_pp": get("test_multi_unseen2", PP_METRIC),
        "unseen2_mmd": get("test_multi_unseen2", MMD_METRIC),
        "family_drug_pp": get("family_drug", PP_METRIC),
        "single_pp": get("structure_single", PP_METRIC),
    }
    missing = [name for name, row in required.items() if row is None]
    if missing:
        return {
            "comparison": comparison,
            "route": route,
            "status": "incomplete",
            "reasons": [f"missing {','.join(missing)}"],
            "required": required,
        }

    reasons: list[str] = []
    warnings: list[str] = []
    provenance = route_provenance(route)
    if not provenance["route_feature_deployable"]:
        warnings.append("route_not_deployable_feature_rule")
    # Core route gate: aggregate and family pp positive; MMD not deterministically harmful.
    if float(required["test_pp"]["p_improve"]) < 0.90 or float(required["test_pp"]["delta"]) <= 0:
        reasons.append("test_pp_not_supported")
    if float(required["family_gene_pp"]["p_improve"]) < 0.90 or float(required["family_gene_pp"]["delta"]) <= 0:
        reasons.append("family_gene_pp_not_supported")
    if float(required["unseen2_pp"]["p_improve"]) < 0.80 or float(required["unseen2_pp"]["delta"]) <= 0:
        warnings.append("unseen2_pp_weak")
    if float(required["test_mmd"]["p_harm"]) > 0.80:
        reasons.append("test_mmd_harm")
    if float(required["family_gene_mmd"]["p_harm"]) > 0.80:
        reasons.append("family_gene_mmd_harm")
    unseen2_mmd_ci = required["unseen2_mmd"].get("ci95") or []
    unseen2_mmd_ci_low = float(unseen2_mmd_ci[0]) if len(unseen2_mmd_ci) >= 1 else None
    if float(required["unseen2_mmd"]["p_harm"]) > 0.80 or (
        unseen2_mmd_ci_low is not None and unseen2_mmd_ci_low > 0.0
    ):
        reasons.append("unseen2_mmd_hard_harm")
    # Routes that leave drug/single untouched produce zero rows; treat non-zero negative pp as warning.
    for name in ("family_drug_pp", "single_pp"):
        row = required[name]
        if abs(float(row["delta"])) > 1e-12 and float(row["p_harm"]) > 0.80:
            reasons.append(f"{name}_harm")

    if reasons:
        status = "fail"
    elif warnings:
        status = "diagnostic_pass_with_warnings"
    else:
        status = "route_gate_pass"
    return {
        "comparison": comparison,
        "route": route,
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "required": required,
        "provenance": provenance,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Routed Expert Decision Summary",
        "",
        f"Status: `{payload['overall_status']}`",
        "",
        "Inputs:",
    ]
    for path in payload["input_jsons"]:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## Gate Summary",
        "",
        "| comparison | route | deployable | status | reasons | warnings | test pp | test MMD | family pp | family MMD | unseen2 pp | unseen2 MMD |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["decisions"]:
        req = item.get("required", {})
        provenance = item.get("provenance") or {}
        lines.append(
            "| {comp} | {route} | {deployable} | {status} | {reasons} | {warnings} | {test_pp} | {test_mmd} | {fam_pp} | {fam_mmd} | {u2_pp} | {u2_mmd} |".format(
                comp=item["comparison"],
                route=item["route"],
                deployable="yes" if provenance.get("route_feature_deployable") else "diagnostic",
                status=item["status"],
                reasons=", ".join(item.get("reasons") or []) or "-",
                warnings=", ".join(item.get("warnings") or []) or "-",
                test_pp=fmt(req.get("test_pp", {}).get("delta")),
                test_mmd=fmt(req.get("test_mmd", {}).get("delta")),
                fam_pp=fmt(req.get("family_gene_pp", {}).get("delta")),
                fam_mmd=fmt(req.get("family_gene_mmd", {}).get("delta")),
                u2_pp=fmt(req.get("unseen2_pp", {}).get("delta")),
                u2_mmd=fmt(req.get("unseen2_mmd", {}).get("delta")),
            )
        )
    lines += [
        "",
        "## Decision Rule",
        "",
        "- `route_gate_pass`: test and family_gene pp are positive with `p_improve >= 0.90`; test/family MMD `p_harm <= 0.80`; and `test_multi_unseen2` MMD is not harmful (`p_harm <= 0.80` and CI not all positive).",
        "- `diagnostic_pass_with_warnings`: core gate passes but route legality/provenance or non-hard secondary warnings remain.",
        "- `fail`: aggregate/family pp is unsupported, aggregate/family MMD is harmful, or unseen2 MMD has hard harm.",
        "- Route rules selected after inspecting held-out outcomes remain diagnostic until a train-only/predeclared router reproduces them.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-bootstrap-json", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    payloads = [load_json(path) for path in args.route_bootstrap_json]
    rows = index_rows(payloads)
    decisions = [assess_pair(rows, comparison, route) for comparison, route in all_comparison_routes(payloads)]
    strict_pass = [d for d in decisions if d["status"] == "route_gate_pass"]
    warning_only = [d for d in decisions if d["status"] == "diagnostic_pass_with_warnings"]
    if strict_pass:
        overall = "route_gate_pass_found"
    elif warning_only:
        overall = "diagnostic_candidate_only"
    else:
        overall = "no_route_candidate"
    payload = {
        "input_jsons": [str(path) for path in args.route_bootstrap_json],
        "overall_status": overall,
        "decisions": decisions,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "status": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
