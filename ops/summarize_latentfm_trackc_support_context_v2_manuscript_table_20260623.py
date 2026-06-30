#!/usr/bin/env python3
"""Build manuscript-table inputs for the support-context v2 diagnostic package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"

SMOKE_JSON = ROOT / f"reports/latentfm_trackc_routed_distill_smoke_decision_{RUN_NAME}.json"
UNCAPPED_JSON = ROOT / f"reports/latentfm_trackc_support_context_v2_uncapped_noharm_{RUN_NAME}_20260623_decision.json"
QUERY_JSON = ROOT / f"reports/latentfm_trackc_support_context_v2_query_once_decision_{RUN_NAME}_20260623.json"
FAILURE_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_query_failure_cases_20260623.json"
AUDIT_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_final_package_audit_20260623.json"

OUT_CSV = ROOT / "reports/latentfm_trackc_support_context_v2_manuscript_table_20260623.csv"
OUT_CAVEAT_CSV = ROOT / "reports/latentfm_trackc_support_context_v2_caveat_table_20260623.csv"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_MANUSCRIPT_TABLE_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, dict):
        return str(decision.get("status"))
    return str(payload.get("status"))


def t(payload: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    return ((payload.get("tables") or {}).get(section) or {}).get(key) or {}


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def metric_row(stage: str, group: str, metric: str, role: str, row: dict[str, Any], note: str = "") -> dict[str, Any]:
    return {
        "stage": stage,
        "group": group,
        "metric": metric,
        "role": role,
        "n_conditions": row.get("n_matched_conditions", ""),
        "n_datasets": row.get("n_matched_datasets", ""),
        "delta": row.get("delta_mean", ""),
        "ci95_low": row.get("ci95_low", ""),
        "ci95_high": row.get("ci95_high", ""),
        "p_improve": row.get("p_improvement", ""),
        "p_harm": row.get("p_harm", ""),
        "status": row.get("status", ""),
        "note": note,
    }


def build_rows(smoke: dict[str, Any], uncapped: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        metric_row(
            "support_val_capped",
            "test_multi",
            "pearson_pert",
            "primary_support_gain",
            t(smoke, "support_split", "test_multi:pearson_pert"),
            "support-val only; no held-out query",
        ),
        metric_row(
            "support_val_capped",
            "test_multi",
            "test_mmd_clamped",
            "support_mmd",
            t(smoke, "support_split", "test_multi:test_mmd_clamped"),
            "support-val only; no held-out query",
        ),
        metric_row(
            "canonical_uncapped_noharm",
            "test_single",
            "pearson_pert",
            "canonical_single_noharm",
            t(uncapped, "split", "test_single:pearson_pert"),
            "canonical support absent; Track A no-harm",
        ),
        metric_row(
            "canonical_uncapped_noharm",
            "test_single",
            "test_mmd_clamped",
            "canonical_single_mmd_noharm",
            t(uncapped, "split", "test_single:test_mmd_clamped"),
            "canonical support absent; Track A no-harm",
        ),
        metric_row(
            "canonical_uncapped_noharm",
            "family_gene",
            "pearson_pert",
            "canonical_family_noharm",
            t(uncapped, "family", "family_gene:pearson_pert"),
            "canonical support absent; Track A no-harm",
        ),
        metric_row(
            "canonical_uncapped_noharm",
            "family_gene",
            "test_mmd_clamped",
            "canonical_family_mmd_noharm",
            t(uncapped, "family", "family_gene:test_mmd_clamped"),
            "canonical support absent; Track A no-harm",
        ),
        metric_row(
            "heldout_query_once",
            "query_multi",
            "pearson_pert",
            "primary_query_gain",
            t(query, "split", "query_multi:pearson_pert"),
            "final one-shot held-out query; no tuning",
        ),
        metric_row(
            "heldout_query_once",
            "query_multi",
            "test_mmd_clamped",
            "primary_query_mmd",
            t(query, "split", "query_multi:test_mmd_clamped"),
            "final one-shot held-out query; no tuning",
        ),
        metric_row(
            "heldout_query_once",
            "query_multi_seen",
            "pearson_pert",
            "query_seen",
            t(query, "split", "query_multi_seen:pearson_pert"),
            "stratum diagnostic",
        ),
        metric_row(
            "heldout_query_once",
            "query_multi_unseen1",
            "pearson_pert",
            "query_unseen1",
            t(query, "split", "query_multi_unseen1:pearson_pert"),
            "stratum diagnostic",
        ),
        metric_row(
            "heldout_query_once",
            "query_multi_unseen2",
            "pearson_pert",
            "query_unseen2_pp_caveat",
            t(query, "split", "query_multi_unseen2:pearson_pert"),
            "weak Pearson caveat",
        ),
        metric_row(
            "heldout_query_once",
            "query_multi_unseen2",
            "test_mmd_clamped",
            "query_unseen2_mmd",
            t(query, "split", "query_multi_unseen2:test_mmd_clamped"),
            "MMD improves despite weak Pearson",
        ),
    ]
    return rows


def caveat_rows(failure: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in (failure.get("worst_pp_rows") or [])[:12]:
        out.append(
            {
                "type": "worst_pp",
                "stratum": row.get("stratum"),
                "dataset": row.get("dataset"),
                "condition": row.get("condition"),
                "genes": "+".join(row.get("genes") or []),
                "pp_delta": row.get("pp_delta"),
                "mmd_delta": row.get("mmd_delta"),
            }
        )
    for row in (failure.get("recurrent_gene_signals") or [])[:12]:
        out.append(
            {
                "type": "recurrent_gene",
                "stratum": "",
                "dataset": "",
                "condition": row.get("gene"),
                "genes": row.get("gene"),
                "pp_delta": row.get("pp_delta_mean"),
                "mmd_delta": row.get("mmd_delta_mean"),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_md(rows: list[dict[str, Any]], caveats: list[dict[str, Any]], statuses: dict[str, str], audit_status: str) -> str:
    lines = [
        "# Track C Support-Context V2 Manuscript Table",
        "",
        f"Status: `trackc_support_context_v2_manuscript_table_ready`",
        "",
        "## Boundary",
        "",
        "Read-only table built from the frozen final diagnostic package.  Do not use these tables for selection, tuning, or extra query access.",
        "",
        "## Source Statuses",
        "",
    ]
    for key, value in statuses.items():
        lines.append(f"- {key}: `{value}`")
    lines.append(f"- final_package_audit: `{audit_status}`")
    lines += [
        "",
        "## Main Metrics",
        "",
        "| stage | group | metric | role | n | delta | 95% CI | p_improve | p_harm | note |",
        "|---|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        ci = f"[{fmt(row['ci95_low'])},{fmt(row['ci95_high'])}]" if row.get("ci95_low") != "" else ""
        lines.append(
            f"| {row['stage']} | {row['group']} | {row['metric']} | {row['role']} | "
            f"{row['n_conditions']} | {fmt(row['delta'])} | {ci} | {fmt(row['p_improve'])} | "
            f"{fmt(row['p_harm'])} | {row['note']} |"
        )
    lines += [
        "",
        "## Caveat Rows",
        "",
        "| type | stratum | dataset | condition/gene | pp delta | MMD delta |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in caveats[:18]:
        lines.append(
            f"| {row['type']} | {row['stratum']} | {row['dataset']} | `{row['condition']}` | "
            f"{fmt(row['pp_delta'])} | {fmt(row['mmd_delta'])} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- Caveat CSV: `{OUT_CAVEAT_CSV}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    smoke = load_json(SMOKE_JSON)
    uncapped = load_json(UNCAPPED_JSON)
    query = load_json(QUERY_JSON)
    failure = load_json(FAILURE_JSON)
    audit = load_json(AUDIT_JSON)
    statuses = {
        "smoke": decision_status(smoke),
        "uncapped": decision_status(uncapped),
        "query": decision_status(query),
        "failure_cases": decision_status(failure),
    }
    rows = build_rows(smoke, uncapped, query)
    caveats = caveat_rows(failure)
    write_csv(OUT_CSV, rows)
    write_csv(OUT_CAVEAT_CSV, caveats)
    OUT_MD.write_text(render_md(rows, caveats, statuses, str(audit.get("status"))), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "trackc_support_context_v2_manuscript_table_ready",
                "rows": len(rows),
                "caveat_rows": len(caveats),
                "out_md": str(OUT_MD),
                "out_csv": str(OUT_CSV),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
