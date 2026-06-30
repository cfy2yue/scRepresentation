#!/usr/bin/env python3
"""Summarize row-level failure cases for the Track C hybrid composition gate.

This is a read-only report over the already produced query-free hybrid gate
JSON. It does not read held-out query, canonical test, canonical multi, active
logs, or GPU artifacts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_trackc_composition_hybrid_prior_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_hybrid_failure_cases_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_HYBRID_FAILURE_CASES_20260623.md"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def row_delta(row: dict[str, Any]) -> dict[str, Any]:
    pp_delta = float(row["candidate"]) - float(row["support_selected_route"])
    mmd_delta = float(row.get("candidate__test_mmd_clamped") or 0.0) - float(row.get("support_selected_route__test_mmd_clamped") or 0.0)
    return {
        "dataset": row["dataset"],
        "condition": row["condition"],
        "genes": row.get("genes") or [],
        "pp_delta": pp_delta,
        "mmd_clamped_delta": mmd_delta,
        "candidate_pp": row["candidate"],
        "route_pp": row["support_selected_route"],
        "raw_gene_covered": row.get("raw_gene_covered"),
        "fallback_genes": row.get("fallback_genes"),
        "total_genes": row.get("total_genes"),
        "raw_coverage_fraction": row.get("raw_coverage_fraction"),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row_delta(row) for row in payload["support_val_summary"]["rows"]]
    by_dataset = {}
    for ds in sorted({row["dataset"] for row in rows}):
        vals = [row["pp_delta"] for row in rows if row["dataset"] == ds]
        by_dataset[ds] = {
            "n": len(vals),
            "mean_pp_delta": float(np.mean(vals)),
            "min_pp_delta": float(np.min(vals)),
            "max_pp_delta": float(np.max(vals)),
            "n_negative": int(sum(v < 0 for v in vals)),
        }
    by_raw_cov: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"raw{int(row['raw_gene_covered'])}_fallback{int(row['fallback_genes'])}"
        by_raw_cov[key].append(row["pp_delta"])
    coverage_strata = {
        key: {
            "n": len(vals),
            "mean_pp_delta": float(np.mean(vals)),
            "min_pp_delta": float(np.min(vals)),
            "max_pp_delta": float(np.max(vals)),
            "n_negative": int(sum(v < 0 for v in vals)),
        }
        for key, vals in sorted(by_raw_cov.items())
    }
    return {
        "status": "composition_hybrid_failure_cases_ready_no_gpu",
        "source_status": payload["status"],
        "selected_spec": payload["selected_train_summary"]["spec"],
        "boundary": {
            **payload["boundary"],
            "source_json": str(IN_JSON),
            "read_only_existing_report": True,
        },
        "decision_context": {
            "hybrid_gate_reasons": payload["reasons"],
            "support_val_paired_pp_delta": payload["support_val_summary"]["paired_pp_delta"],
            "support_val_paired_mmd_delta": payload["support_val_summary"]["paired_mmd_delta"],
        },
        "dataset_summary": by_dataset,
        "coverage_strata": coverage_strata,
        "worst_pp_rows": sorted(rows, key=lambda row: row["pp_delta"])[:8],
        "best_pp_rows": sorted(rows, key=lambda row: row["pp_delta"], reverse=True)[:8],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Composition Hybrid Failure Cases",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Read-only summary of the existing hybrid composition gate JSON.",
        "- No held-out query, canonical test, canonical multi, active logs, or GPU artifacts are read.",
        f"- source JSON: `{payload['boundary']['source_json']}`",
        "",
        "## Decision Context",
        "",
        f"- selected spec: `{payload['selected_spec']}`",
        f"- source status: `{payload['source_status']}`",
        f"- source reasons: `{', '.join(payload['decision_context']['hybrid_gate_reasons'])}`",
        f"- support_val paired pp delta: `{fmt(payload['decision_context']['support_val_paired_pp_delta']['delta_mean'])}`",
        f"- support_val pp p_harm: `{fmt(payload['decision_context']['support_val_paired_pp_delta']['p_harm'])}`",
        f"- support_val paired MMD delta: `{fmt(payload['decision_context']['support_val_paired_mmd_delta']['delta_mean'])}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | n | mean pp delta | min | max | n negative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds, row in payload["dataset_summary"].items():
        lines.append(
            f"| {ds} | {row['n']} | {fmt(row['mean_pp_delta'])} | {fmt(row['min_pp_delta'])} | "
            f"{fmt(row['max_pp_delta'])} | {row['n_negative']} |"
        )
    lines.extend(["", "## Coverage Strata", "", "| raw/fallback genes | n | mean pp delta | min | max | n negative |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in payload["coverage_strata"].items():
        lines.append(
            f"| `{key}` | {row['n']} | {fmt(row['mean_pp_delta'])} | {fmt(row['min_pp_delta'])} | "
            f"{fmt(row['max_pp_delta'])} | {row['n_negative']} |"
        )
    for title, key in (("Worst PP Rows", "worst_pp_rows"), ("Best PP Rows", "best_pp_rows")):
        lines.extend(["", f"## {title}", "", "| dataset | condition | genes | raw/fallback/total | pp delta | MMD delta |", "|---|---|---|---:|---:|---:|"])
        for row in payload[key]:
            genes = "+".join(row["genes"])
            lines.append(
                f"| {row['dataset']} | `{row['condition']}` | `{genes}` | "
                f"{row['raw_gene_covered']}/{row['fallback_genes']}/{row['total_genes']} | "
                f"{fmt(row['pp_delta'])} | {fmt(row['mmd_clamped_delta'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The hybrid prior solves support coverage but the no-harm failure is driven by a few large negative rows, not by uniform small degradation. The next legal gate should predeclare a train-only outlier/no-harm safeguard rather than reselecting parameters on support_val.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    source = json.loads(IN_JSON.read_text(encoding="utf-8"))
    payload = summarize(source)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
