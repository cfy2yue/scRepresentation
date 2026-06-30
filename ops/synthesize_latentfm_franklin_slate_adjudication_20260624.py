#!/usr/bin/env python3
"""Adjudicate Franklin's distinct-hypothesis slate against current evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_franklin_slate_adjudication_20260624.json"
OUT_MD = REPORTS / "LATENTFM_FRANKLIN_SLATE_ADJUDICATION_20260624.md"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> int:
    risk_code = load_json("latentfm_risk_row_cvar_loss_code_gate_20260624.json")
    scaling_matrix = load_json("latentfm_scaling_protocol_matrix_decision_20260624.json")
    scaling_canonical = load_json("latentfm_scaling_protocol_canonical_noharm_decision_20260624.json")
    ot_quality = load_json("latentfm_ot_pairing_quality_reliability_gate_20260624.json")

    rows = [
        {
            "hypothesis": "risk_row_cvar_topk_mmd_loss",
            "adjudication": (
                "code_unit_gate_pass_no_gpu"
                if risk_code.get("status") == "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu"
                else "blocked_no_gpu_code_gate_fail"
            ),
            "evidence": str(REPORTS / "LATENTFM_RISK_ROW_CVAR_LOSS_CODE_GATE_20260624.md"),
            "reason": (
                "Default-off tail-state API/unit tests now exist, but this gate "
                "alone does not authorize GPU or canonical no-harm."
                if risk_code.get("status") == "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu"
                else (
                    "Current train loop has condition-level MMD but lacks default-off "
                    "CVaR/top-k config and cross-condition tail state/history."
                )
            ),
            "next_if_reopened": (
                "External/code review plus separate launcher/provenance gate for exactly one capped train-only smoke."
                if risk_code.get("status") == "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu"
                else "Design and unit-test a tail-state API before any GPU launcher."
            ),
        },
        {
            "hypothesis": "metainfo_matched_composition_scaling",
            "adjudication": "covered_and_closed_by_existing_protocol",
            "evidence": str(REPORTS / "LATENTFM_SCALING_PROTOCOL_MATRIX_DECISION_20260624.md"),
            "reason": (
                f"Scaling matrix status={scaling_matrix.get('status')!r}; only cap60 passed "
                "internal, matched-budget breadth arms failed; cap60 canonical no-harm "
                f"status={scaling_canonical.get('status')!r}."
            ),
            "next_if_reopened": "Requires a materially new metainfo feature/control, not another cap/breadth split.",
        },
        {
            "hypothesis": "ot_pair_quality_gated_minibatch_loss",
            "adjudication": "covered_and_closed_by_existing_ot_quality_gate",
            "evidence": str(REPORTS / "LATENTFM_OT_PAIRING_QUALITY_RELIABILITY_GATE_20260624.md"),
            "reason": (
                f"OT quality reliability status={ot_quality.get('status')!r}; condition overlap is "
                f"{ot_quality.get('condition_overlap', 'NA')} and current synthesis reports "
                "random/Hungarian OT smokes failed Track A gates."
            ),
            "next_if_reopened": "Requires a new condition-level pairing-quality signal with controls, not a pair-mode/cost sweep.",
        },
    ]

    risk_ready = risk_code.get("status") == "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu"
    status = (
        "franklin_slate_risk_row_external_review_next_no_gpu"
        if risk_ready
        else "franklin_slate_no_gpu_candidate_remaining"
    )
    payload = {
        "status": status,
        "boundary": {
            "read_completed_reports_only": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "gpu_used": False,
        },
        "subagent": {
            "name": "Franklin",
            "agent_id": "019ef86f-bebb-78d2-a3df-b5b9e841fbfb",
        },
        "rows": rows,
        "decision": {
            "gpu_authorized": False,
            "next_action": (
                "No Franklin-slate GPU launch is authorized. Risk-row CVaR/top-k "
                "is now code/unit-gated and can proceed only to external review "
                "or a separate launcher/provenance gate for exactly one capped "
                "train-only smoke."
                if risk_ready
                else (
                    "No Franklin-slate GPU launch is authorized. Continue with "
                    "report/consolidation or find a genuinely exogenous train-only signal; "
                    "risk-row CVaR can reopen only after tail-state API/unit tests."
                )
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Franklin Slate Adjudication",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Reads completed reports/code-gate outputs only.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "- Uses no GPU.",
        "",
        "## Slate",
        "",
        "| hypothesis | adjudication | reason | evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['hypothesis']}` | `{row['adjudication']}` | "
            f"{row['reason']} | `{row['evidence']}` |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No Franklin-slate GPU launch is authorized.",
            "- Risk-row CVaR/top-k MMD has a code/unit gate pass and may proceed only to external review or a separate launcher/provenance gate.",
            "- Metainfo-matched scaling/composition and OT pair-quality are already covered by completed negative gates.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
