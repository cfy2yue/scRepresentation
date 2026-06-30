#!/usr/bin/env python3
"""Write a Track C endpoint-routed failure analysis after a failed smoke.

This script is intentionally pass/fail gated:

- missing decision -> exit 2;
- passing smoke -> exit 3, because uncapped no-harm is the next action;
- failing smoke -> write a no-query failure report from the smoke decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_trackc_endpoint_route_w05_replay1_2k_seed42"
DEFAULT_DECISION_JSON = (
    ROOT / "reports" / f"latentfm_trackc_routed_distill_smoke_decision_{RUN_NAME}.json"
)
DEFAULT_OUT_JSON = ROOT / "reports" / "latentfm_trackc_endpoint_routed_failure_analysis_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports" / "LATENTFM_TRACKC_ENDPOINT_ROUTED_FAILURE_ANALYSIS_20260622.md"

PASS_STATUS = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
FAIL_STATUSES = {
    "trackc_smoke_fail_support_gate_close_branch",
    "trackc_smoke_fail_canonical_harm_close_branch",
    "trackc_smoke_missing_required_metrics_close_branch",
}

KEY_ROWS = [
    ("support pp", "support_split", "test_multi:pearson_pert", "test:pearson_pert"),
    ("support mmd", "support_split", "test_multi:test_mmd_clamped", "test:test_mmd_clamped"),
    ("canonical single pp", "canonical_split", "test_single:pearson_pert", None),
    ("canonical single mmd", "canonical_split", "test_single:test_mmd_clamped", None),
    ("canonical family pp", "canonical_family", "family_gene:pearson_pert", None),
    ("canonical family mmd", "canonical_family", "family_gene:test_mmd_clamped", None),
    ("canonical multi diagnostic pp", "canonical_split", "test_multi:pearson_pert", None),
    ("canonical unseen2 diagnostic pp", "canonical_split", "test_multi_unseen2:pearson_pert", None),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def select_row(tables: dict[str, Any], table_name: str, primary: str, fallback: str | None) -> dict[str, Any] | None:
    table = tables.get(table_name) or {}
    row = table.get(primary)
    if row and row.get("status") == "ok" and int(row.get("n_matched_conditions") or 0) > 0:
        return row
    if fallback:
        fb = table.get(fallback)
        if fb and fb.get("status") == "ok" and int(fb.get("n_matched_conditions") or 0) > 0:
            return fb
    return row or (table.get(fallback) if fallback else None)


def row_summary(role: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"role": role, "status": "missing"}
    return {
        "role": role,
        "group": row.get("group"),
        "metric": row.get("metric"),
        "n_matched_conditions": row.get("n_matched_conditions"),
        "n_matched_datasets": row.get("n_matched_datasets"),
        "delta_mean": row.get("delta_mean"),
        "ci95_low": row.get("ci95_low"),
        "ci95_high": row.get("ci95_high"),
        "p_improvement": row.get("p_improvement"),
        "p_harm": row.get("p_harm"),
        "status": row.get("status"),
    }


def classify(reasons: list[str]) -> str:
    if any(r.startswith("missing_or_bad_") for r in reasons):
        return "metric_coverage_failure"
    if any(r.startswith("canonical_") for r in reasons):
        return "canonical_noharm_failure"
    return "support_material_gain_failure"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Track C Endpoint-Routed Failure Analysis",
        "",
        f"Status: `{payload['status']}`",
        f"Failure mode: `{payload['failure_mode']}`",
        f"Decision status: `{payload['decision_status']}`",
        f"Recommended action: `{payload['recommended_action']}`",
        "",
        "## Provenance",
        "",
        f"- decision_json: `{payload['decision_json']}`",
        f"- run_root: `{payload.get('run_root', 'NA')}`",
        "- held-out query used: `False`",
        "",
        "## Gate Reasons",
        "",
    ]
    reasons = payload.get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines += [
        "",
        "## Key Rows",
        "",
        "| role | group | metric | n cond | n ds | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {role} | {group} | {metric} | {n_cond} | {n_ds} | {delta} | [{lo}, {hi}] | {p_imp} | {p_harm} | {status} |".format(
                role=row.get("role", "NA"),
                group=row.get("group", "NA"),
                metric=row.get("metric", "NA"),
                n_cond=row.get("n_matched_conditions", 0),
                n_ds=row.get("n_matched_datasets", 0),
                delta=fmt(row.get("delta_mean")),
                lo=fmt(row.get("ci95_low")),
                hi=fmt(row.get("ci95_high")),
                p_imp=fmt(row.get("p_improvement")),
                p_harm=fmt(row.get("p_harm")),
                status=row.get("status", "NA"),
            )
        )
    lines += [
        "",
        "## Close Rule",
        "",
        "- Close endpoint-routed branch.",
        "- Do not run endpoint uncapped canonical no-harm.",
        "- Do not run held-out query.",
        "- Before another Track C GPU branch, run CPU/static failure analysis and focused external/subagent review.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", type=Path, default=DEFAULT_DECISION_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    if not args.decision_json.is_file():
        raise SystemExit(f"missing smoke decision JSON: {args.decision_json}")
    decision_payload = load_json(args.decision_json)
    decision = decision_payload.get("decision") or {}
    status = str(decision.get("status") or "")
    if status == PASS_STATUS:
        raise SystemExit("smoke passed; run uncapped canonical no-harm guard instead")
    if status not in FAIL_STATUSES:
        raise SystemExit(f"unrecognized smoke decision status: {status}")

    tables = decision_payload.get("tables") or {}
    rows = [
        row_summary(role, select_row(tables, table, primary, fallback))
        for role, table, primary, fallback in KEY_ROWS
    ]
    reasons = [str(r) for r in (decision.get("reasons") or [])]
    payload = {
        "status": "endpoint_failure_analysis_ready_close_branch",
        "decision_status": status,
        "recommended_action": "close_endpoint_branch_no_uncapped_no_query",
        "failure_mode": classify(reasons),
        "decision_json": str(args.decision_json),
        "run_root": decision_payload.get("run_root"),
        "reasons": reasons,
        "rows": rows,
        "query_used": False,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    if args.out_json.exists() or args.out_md.exists():
        raise FileExistsError("failure analysis outputs already exist; refusing to overwrite")
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
