#!/usr/bin/env python3
"""Decide the next Wessels action after global-prior diagnostics finish."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    value = fnum(value)
    return "NA" if value is None else f"{value:.6f}"


def load_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_rows(original: dict[str, Any] | None, sweep: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if original:
        gate = original.get("gate") or {}
        rows.append(
            {
                "run": original.get("run", "scf_globalprior010_add005_wessels_4k"),
                "source": "original",
                "status": gate.get("status"),
                "unseen2_pp_delta": fnum(gate.get("pp_delta")),
                "test_mmd_ratio": fnum(gate.get("test_mmd_ratio")),
            }
        )
    if sweep:
        for row in sweep.get("gate_rows", []) or []:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "run": row.get("run"),
                    "source": "sweep",
                    "status": row.get("status"),
                    "unseen2_pp_delta": fnum(row.get("unseen2_pp_delta")),
                    "test_mmd_ratio": fnum(row.get("test_mmd_ratio")),
                }
            )
    return rows


def candidate_rows_from_gate_audit(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in audit.get("runs", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "run": row.get("run"),
                "source": "strict_gate_audit",
                "status": row.get("status"),
                "unseen2_pp_delta": fnum(row.get("unseen2_pp_delta")),
                "test_mmd_ratio": fnum(row.get("test_mmd_ratio")),
            }
        )
    return rows


def decide(rows: list[dict[str, Any]], *, required_sources: int) -> dict[str, Any]:
    if len({row["source"] for row in rows}) < required_sources:
        return {
            "status": "pending",
            "next_action": "wait_for_global_prior_posthoc",
            "reason": "one or more global-prior summary files are missing",
        }
    passed = [row for row in rows if row.get("status") == "pass"]
    invalid = [row for row in rows if row.get("status") == "invalid_selection_mismatch"]
    if invalid:
        return {
            "status": "invalid_selection_mismatch",
            "next_action": "rerun_or_reaudit_global_prior_selection_mismatch",
            "reason": "one or more strict gate-audit rows have baseline/candidate selected-condition mismatch",
        }
    ranked = sorted(
        rows,
        key=lambda row: (
            fnum(row.get("unseen2_pp_delta")) if fnum(row.get("unseen2_pp_delta")) is not None else -999.0,
            -(fnum(row.get("test_mmd_ratio")) if fnum(row.get("test_mmd_ratio")) is not None else 999.0),
        ),
        reverse=True,
    )
    if passed:
        best = sorted(
            passed,
            key=lambda row: (
                fnum(row.get("unseen2_pp_delta")) or -999.0,
                -(fnum(row.get("test_mmd_ratio")) or 999.0),
            ),
            reverse=True,
        )[0]
        return {
            "status": "pass",
            "next_action": "promote_best_global_prior_to_all_split_diagnostic",
            "best_run": best.get("run"),
            "reason": "at least one Wessels global-prior diagnostic passed the stable-caps gate",
        }
    best = ranked[0] if ranked else {}
    return {
        "status": "fail",
        "next_action": "design_context_conditioned_prior_or_interaction_residual",
        "best_run": best.get("run"),
        "reason": (
            "global train-only prior coverage is full, but no injection-strength variant "
            "passed the Wessels unseen2/MMD gate"
        ),
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels Global Prior Next-Action Decision",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        f"Next action: `{payload['decision']['next_action']}`",
        "",
        f"Reason: {payload['decision']['reason']}",
        "",
        "## Candidate Gates",
        "",
        "| run | source | status | unseen2 pp delta | test MMD ratio |",
        "|---|---|---|---:|---:|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| `{row.get('run')}` | {row.get('source')} | {row.get('status')} | "
            f"{fmt(row.get('unseen2_pp_delta'))} | {fmt(row.get('test_mmd_ratio'))} |"
        )
    lines.extend(
        [
            "",
            "Rules:",
            "",
            "- pass: Wessels `test_multi_unseen2` pp delta >= +0.05 and test MMD ratio <= 1.15.",
            "- if all fail despite full prior-bank coverage, move to context-conditioned prior or interaction residual modeling.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-summary",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_global_prior_summary_20260620.json"),
    )
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_global_prior_sweep_summary_20260620.json"),
    )
    parser.add_argument(
        "--gate-audit",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_global_prior_gate_audit_20260620.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_global_prior_next_action_20260620.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_NEXT_ACTION_20260620.md"),
    )
    args = parser.parse_args()

    gate_audit = load_if_present(args.gate_audit)
    original = load_if_present(args.original_summary)
    sweep = load_if_present(args.sweep_summary)
    if gate_audit:
        rows = candidate_rows_from_gate_audit(gate_audit)
        required_sources = 1
    else:
        rows = candidate_rows(original, sweep)
        required_sources = 2
    decision = decide(rows, required_sources=required_sources)
    payload = {
        "original_summary": str(args.original_summary),
        "sweep_summary": str(args.sweep_summary),
        "gate_audit": str(args.gate_audit),
        "gate_audit_used": bool(gate_audit),
        "candidates": rows,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
