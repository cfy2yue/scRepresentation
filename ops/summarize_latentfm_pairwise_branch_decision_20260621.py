#!/usr/bin/env python3
"""Summarize LatentFM pairwise-family branch bootstrap decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BOOTSTRAP_INDEXES = [
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_condition_smoke_bootstrap_20260621/bootstrap_index.json"),
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_adapter_only_smoke_bootstrap_20260621/bootstrap_index.json"),
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_vs_refinetune_bootstrap_20260621/bootstrap_index.json"),
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_condition_adapter_smoke_bootstrap_20260621/bootstrap_index.json"),
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_global_gene_mean_prior_smoke_bootstrap_20260621/bootstrap_index.json"),
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_global_gene_mean_prior002_smoke_bootstrap_20260621/bootstrap_index.json"),
    Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_global_gene_mean_prior002_replay_smoke_bootstrap_20260621/bootstrap_index.json"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def metric_rows(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.is_file():
        return [{"index": str(index_path), "status": "pending"}]
    index = load_json(index_path)
    out: list[dict[str, Any]] = []
    for item in index.get("outputs", []):
        jpath = Path(item.get("json", ""))
        if not jpath.is_file():
            out.append(
                {
                    "index": str(index_path),
                    "run_name": item.get("run_name"),
                    "kind": item.get("kind"),
                    "status": "pending",
                }
            )
            continue
        payload = load_json(jpath)
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "index": str(index_path),
                    "run_name": item.get("run_name"),
                    "kind": item.get("kind"),
                    "group": row.get("group"),
                    "metric": row.get("metric"),
                    "direction": row.get("direction"),
                    "delta": row.get("delta_mean"),
                    "ci95": [row.get("ci95_low"), row.get("ci95_high")],
                    "p_improve": row.get("p_improvement"),
                    "p_harm": row.get("p_harm"),
                    "selected_match": row.get("selected_match"),
                    "status": row.get("status", "ok"),
                    "n_conditions": row.get("n_matched_conditions"),
                    "n_datasets": row.get("n_matched_datasets"),
                }
            )
    return out


def by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "pending":
            continue
        key = (str(row.get("group")), str(row.get("metric")), str(row.get("kind")))
        out[key] = row
    return out


def branch_decision(run_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_class = "control_comparison" if "vs_canonical_refinetune" in run_name else "anchor_relative"
    if any(row.get("status") == "pending" for row in rows) and not any(row.get("group") for row in rows):
        return {
            "run_name": run_name,
            "comparison_class": comparison_class,
            "status": "pending",
            "reasons": ["bootstrap_missing"],
            "warnings": [],
        }
    k = by_key(rows)

    test_pp = k.get(("test", "pearson_pert", "split"))
    test_mmd = k.get(("test", "test_mmd_clamped", "split"))
    unseen_pp = k.get(("test_multi_unseen2", "pearson_pert", "split"))
    unseen_mmd = k.get(("test_multi_unseen2", "test_mmd_clamped", "split"))
    family_pp = k.get(("family_gene", "pearson_pert", "family"))
    family_mmd = k.get(("family_gene", "test_mmd_clamped", "family"))

    reasons: list[str] = []
    warnings: list[str] = []

    required = {
        "test_pp": test_pp,
        "test_mmd": test_mmd,
        "unseen_pp": unseen_pp,
        "unseen_mmd": unseen_mmd,
        "family_pp": family_pp,
        "family_mmd": family_mmd,
    }
    for name, row in required.items():
        if row is None:
            reasons.append(f"missing_{name}")
        elif row.get("selected_match") is False:
            reasons.append(f"selected_mismatch_{name}")

    def p(row: dict[str, Any] | None, key: str) -> float | None:
        if row is None:
            return None
        value = row.get(key)
        return None if value is None else float(value)

    if p(test_pp, "p_improve") is not None and p(test_pp, "p_improve") < 0.90:
        reasons.append("test_pp_not_supported")
    if p(family_pp, "p_improve") is not None and p(family_pp, "p_improve") < 0.90:
        reasons.append("family_gene_pp_not_supported")
    if p(test_mmd, "p_harm") is not None and p(test_mmd, "p_harm") > 0.80:
        reasons.append("test_mmd_harm")
    if p(family_mmd, "p_harm") is not None and p(family_mmd, "p_harm") > 0.80:
        reasons.append("family_gene_mmd_harm")

    if p(unseen_pp, "p_improve") is not None and p(unseen_pp, "p_improve") < 0.90:
        warnings.append("unseen2_pp_weak")
    if p(unseen_mmd, "p_harm") is not None and p(unseen_mmd, "p_harm") > 0.80:
        warnings.append("unseen2_mmd_harm")

    if reasons:
        status = "fail"
    elif warnings:
        status = "diagnostic_pass_with_warnings"
    else:
        status = "branch_gate_pass"

    return {
        "run_name": run_name,
        "comparison_class": comparison_class,
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "required": required,
    }


def render_md(decisions: list[dict[str, Any]], indexes: list[Path]) -> str:
    lines = [
        "# LatentFM Pairwise Branch Decision Summary",
        "",
        "Inputs:",
        *[f"- `{path}`" for path in indexes],
        "",
        "## Gate Summary",
        "",
        "| run | class | status | reasons | warnings | test pp | test MMD | family pp | family MMD | unseen2 pp | unseen2 MMD |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dec in decisions:
        req = dec.get("required") or {}
        lines.append(
            "| {run} | {klass} | {status} | {reasons} | {warnings} | {test_pp} | {test_mmd} | {family_pp} | {family_mmd} | {unseen_pp} | {unseen_mmd} |".format(
                run=f"`{dec.get('run_name')}`",
                klass=f"`{dec.get('comparison_class')}`",
                status=f"`{dec.get('status')}`",
                reasons=", ".join(dec.get("reasons") or []) or "-",
                warnings=", ".join(dec.get("warnings") or []) or "-",
                test_pp=fmt((req.get("test_pp") or {}).get("delta")),
                test_mmd=fmt((req.get("test_mmd") or {}).get("delta")),
                family_pp=fmt((req.get("family_pp") or {}).get("delta")),
                family_mmd=fmt((req.get("family_mmd") or {}).get("delta")),
                unseen_pp=fmt((req.get("unseen_pp") or {}).get("delta")),
                unseen_mmd=fmt((req.get("unseen_mmd") or {}).get("delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- `branch_gate_pass`: test and family_gene pp have `p_improve >= 0.90`, and test/family MMD have `p_harm <= 0.80`, with no unseen2 warnings.",
            "- `control_comparison` pass means a mechanism improves over a matched refinetune baseline; it is not an anchor-relative promotion candidate.",
            "- `diagnostic_pass_with_warnings`: aggregate/family gate passes but unseen2 pp/MMD is weak or harmful.",
            "- `fail`: aggregate/family pp is unsupported, aggregate/family MMD is harmful, selected conditions mismatch, or required rows are missing.",
            "- Capped branch pass only permits uncapped posthoc/bootstrap; it is not a paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-index", type=Path, action="append", default=None)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_branch_decision_20260621.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_PAIRWISE_BRANCH_DECISION_20260621.md"),
    )
    args = parser.parse_args()

    indexes = args.bootstrap_index or DEFAULT_BOOTSTRAP_INDEXES
    grouped: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for index in indexes:
        rows = metric_rows(index)
        all_rows.extend(rows)
        for row in rows:
            run_name = str(row.get("run_name") or index.parent.name)
            grouped.setdefault(run_name, []).append(row)

    decisions = [branch_decision(run_name, rows) for run_name, rows in sorted(grouped.items())]
    anchor_gate = any(
        dec["status"] == "branch_gate_pass" and dec.get("comparison_class") == "anchor_relative"
        for dec in decisions
    )
    control_gate = any(
        dec["status"] == "branch_gate_pass" and dec.get("comparison_class") == "control_comparison"
        for dec in decisions
    )
    pending_or_diag = any(dec["status"] in {"pending", "diagnostic_pass_with_warnings"} for dec in decisions)
    if anchor_gate:
        overall = "anchor_relative_branch_gate_pass_found"
    elif pending_or_diag and control_gate:
        overall = "control_pass_anchor_pending_or_fail"
    elif pending_or_diag:
        overall = "diagnostic_or_pending"
    elif control_gate:
        overall = "control_pass_anchor_fail"
    else:
        overall = "all_fail"
    payload = {
        "bootstrap_indexes": [str(path) for path in indexes],
        "overall_status": overall,
        "decisions": decisions,
        "rows": all_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(decisions, indexes), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "status": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
