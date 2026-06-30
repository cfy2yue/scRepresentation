#!/usr/bin/env python3
"""Combine Wessels global-prior diagnostics into a final next-action report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    value = fnum(value)
    return "NA" if value is None else f"{value:.6f}"


def decide(latest: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    rows = latest.get("rows", []) or []
    best = max(
        rows,
        key=lambda row: (
            fnum(row.get("unseen2_pp_delta")) if fnum(row.get("unseen2_pp_delta")) is not None else -999.0,
            -(fnum(row.get("test_mmd_ratio")) if fnum(row.get("test_mmd_ratio")) is not None else 999.0),
        ),
        default={},
    )
    latest_pass = latest.get("decision", {}).get("status") == "pass"
    context_pass = context.get("gate", {}).get("status") == "pass"
    if latest_pass:
        return {
            "status": "global_prior_latest_pass",
            "next_action": "promote_latest_global_prior_to_all_split_diagnostic_after_checkpoint_selection_fix",
            "best_run": best.get("run"),
            "reason": "latest checkpoint strict gate passed after correcting the inactive-EMA interpretation",
        }
    if context_pass:
        return {
            "status": "context_prior_cpu_pass",
            "next_action": "launch_context_adapted_prior_wessels_gpu_diagnostic",
            "best_run": context.get("gate", {}).get("selected_adapter"),
            "reason": "global-prior latest failed but CPU context-adapted prior passed its pre-GPU gate",
        }
    return {
        "status": "global_and_context_prior_fail",
        "next_action": "run_combo_interaction_or_residual_feasibility_diagnostic_cpu_first",
        "best_run": best.get("run"),
        "reason": (
            "best/latest global train-only gene-mean priors fail Wessels unseen2, "
            "and Wessels-context additive adaptation is too weak for GPU launch"
        ),
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels After Global-Prior Final Decision",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        f"Next action: `{payload['decision']['next_action']}`",
        "",
        f"Reason: {payload['decision']['reason']}",
        "",
        "## Latest-Checkpoint Global-Prior Gate",
        "",
        "| run | status | test pp delta | unseen1 pp delta | unseen2 pp delta | test MMD ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["latest_rows"]:
        lines.append(
            f"| `{row.get('run')}` | {row.get('status')} | {fmt(row.get('test_pp_delta'))} | "
            f"{fmt(row.get('unseen1_pp_delta'))} | {fmt(row.get('unseen2_pp_delta'))} | "
            f"{fmt(row.get('test_mmd_ratio'))} |"
        )
    gate = payload["context_gate"]
    lines.extend(
        [
            "",
            "## Context-Adapted Additive Prior CPU Gate",
            "",
            f"Selected adapter: `{gate.get('selected_adapter')}`",
            "",
            f"Gate status: `{gate.get('status')}`; unseen2 Pearson delta vs raw global prior "
            f"{fmt(gate.get('unseen2_pearson_delta_vs_raw'))} vs >= {fmt(gate.get('min_delta'))}; "
            f"norm ratio {fmt(gate.get('selected_unseen2_norm_ratio'))} vs <= {fmt(gate.get('max_norm_ratio'))}.",
            "",
            "## Decision Notes",
            "",
            "- The inactive-EMA checkpoint-selection bug has been fixed in CoupledFM commit `ca82f37`; old `best.pt` posthoc should not be used as biological evidence.",
            "- Latest-checkpoint fair posthoc still fails, so the global train-only `gene_mean` prior branch should not be promoted.",
            "- Because the context-adapted additive prior CPU gate also fails, do not launch context-prior GPU arms yet.",
            "- Next work should be CPU-first: inspect whether any train-only multi/interaction residual signal exists, or whether Wessels failure is better addressed by residual preprocessing/normalization diagnostics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest-summary",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_global_prior_latest_summary_20260620.json"),
    )
    parser.add_argument(
        "--context-summary",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_context_prior_adapter_20260620.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_after_global_prior_decision_20260620.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_WESSELS_AFTER_GLOBAL_PRIOR_DECISION_20260620.md"),
    )
    args = parser.parse_args()

    latest = load(args.latest_summary)
    context = load(args.context_summary)
    payload = {
        "latest_summary": str(args.latest_summary),
        "context_summary": str(args.context_summary),
        "latest_rows": latest.get("rows", []) or [],
        "context_gate": context.get("gate", {}) or {},
        "decision": decide(latest, context),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "decision": payload["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
