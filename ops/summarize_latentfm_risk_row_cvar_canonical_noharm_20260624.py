#!/usr/bin/env python3
"""Summarize frozen canonical no-harm for the risk-row CVaR candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_risk_row_cvar_allrisk_w020_2k_seed42"
RUN_DIR = ROOT / "runs/latentfm_risk_row_cvar_canonical_noharm_20260624" / RUN_NAME
GATE_JSON = RUN_DIR / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
OUT_JSON = ROOT / "reports/latentfm_risk_row_cvar_canonical_noharm_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_ROW_CVAR_CANONICAL_NOHARM_DECISION_20260624.md"


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: list[dict[str, Any]], stratum: str, metric: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def metric(rows: list[dict[str, Any]], stratum: str, metric_name: str) -> dict[str, Any]:
    row = find_row(rows, stratum, metric_name) or {}
    return {
        "delta_mean": row.get("delta_mean"),
        "p_harm": row.get("p_harm"),
        "p_improve": row.get("p_improve"),
        "ci95": row.get("ci95"),
        "status": row.get("status"),
    }


def main() -> int:
    gate = load_json(GATE_JSON)
    exit_code = read_text(RUN_DIR / "POSTHOC_EXIT_CODE")
    if gate is None:
        status = "risk_row_cvar_canonical_noharm_pending"
        gate_status = None
        reasons = []
        rows = []
    else:
        gate_status = (gate.get("gate") or {}).get("status")
        reasons = (gate.get("gate") or {}).get("reasons") or []
        rows = gate.get("paired_deltas") or []
        if exit_code != "0":
            status = "risk_row_cvar_canonical_noharm_posthoc_failed"
        elif gate_status == "candidate_gate_pass":
            status = "risk_row_cvar_canonical_noharm_pass_no_promotion"
        else:
            status = "risk_row_cvar_canonical_noharm_fail_close_recipe"

    metrics = {
        "cross_background_seen_gene:pearson_pert": metric(rows, "cross_background_seen_gene", "pearson_pert"),
        "all_test_single:pearson_pert": metric(rows, "all_test_single", "pearson_pert"),
        "all_test_single:test_mmd_clamped": metric(rows, "all_test_single", "test_mmd_clamped"),
        "family_gene:pearson_pert": metric(rows, "family_gene", "pearson_pert"),
        "family_gene:test_mmd_clamped": metric(rows, "family_gene", "test_mmd_clamped"),
    }
    payload = {
        "status": status,
        "run_name": RUN_NAME,
        "posthoc_exit_code": exit_code,
        "gate_status": gate_status,
        "gate_reasons": reasons,
        "metrics": metrics,
        "boundary": {
            "canonical_split_post_freeze": True,
            "canonical_multi_evaluated": False,
            "trackc_query_read": False,
            "promotion_authorized": False,
        },
        "decision": {
            "promotion_authorized": False,
            "trackc_query_authorized": False,
            "next_if_pass": "Seed/robustness confirmation only; no promotion claim yet.",
            "next_if_fail": "Close this exact risk-row CVaR recipe for promotion.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Row CVaR Canonical No-Harm Decision",
        "",
        f"Status: `{status}`",
        f"Gate status: `{gate_status}`",
        "",
        "## Boundary",
        "",
        "- Frozen canonical no-harm posthoc only.",
        "- Canonical multi is not evaluated in this gate.",
        "- Track C query is not read.",
        "- No promotion is authorized by this report.",
        "",
        "## Metrics",
        "",
        "| metric | delta | p_harm | p_improve | CI | status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, row in metrics.items():
        lines.append(
            f"| `{name}` | `{row.get('delta_mean')}` | `{row.get('p_harm')}` | "
            f"`{row.get('p_improve')}` | `{row.get('ci95')}` | `{row.get('status')}` |"
        )
    if reasons:
        lines.extend(["", "## Reasons", ""])
        for reason in reasons:
            lines.append(f"- `{reason}`")
    lines.extend(["", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
