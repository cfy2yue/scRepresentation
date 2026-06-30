#!/usr/bin/env python3
"""Build a single-entry reporting package for Track C support-context v2.

This reads only already-frozen reports/JSON artifacts. It does not inspect
held-out query for tuning, run models, or authorize any GPU branch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

DEFAULT_TABLE_CSV = REPORTS / "latentfm_trackc_support_context_v2_manuscript_table_20260623.csv"
DEFAULT_CAVEAT_CSV = REPORTS / "latentfm_trackc_support_context_v2_caveat_table_20260623.csv"
DEFAULT_CLAIM_JSON = REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json"
DEFAULT_AUDIT_JSON = REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json"
DEFAULT_FAILURE_JSON = REPORTS / "latentfm_trackc_support_context_v2_query_failure_cases_20260623.json"
DEFAULT_PORTFOLIO_JSON = REPORTS / "latentfm_post_support_context_v2_portfolio_decision_20260623.json"
DEFAULT_OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_REPORTING_PACKAGE_20260623.md"
DEFAULT_OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_reporting_package_20260623.json"

SOURCE_REPORTS = {
    "claim_readiness": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_CLAIM_READINESS_AUDIT_20260623.md",
    "final_package_audit": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_PACKAGE_AUDIT_20260623.md",
    "manuscript_table": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_MANUSCRIPT_TABLE_20260623.md",
    "final_synthesis": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_DIAGNOSTIC_SYNTHESIS_20260623.md",
    "query_decision": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_ONCE_DECISION_xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42_20260623.md",
    "query_freeze": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_FREEZE_xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42_20260623.md",
    "uncapped_noharm": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42_DECISION_20260623.md",
    "residual_uncapped_noharm": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42_DECISION_20260623.md",
    "failure_cases": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_FAILURE_CASES_20260623.md",
    "portfolio_decision": REPORTS / "LATENTFM_POST_SUPPORT_CONTEXT_V2_PORTFOLIO_DECISION_20260623.md",
    "next_candidate_review": REPORTS / "LATENTFM_HIGH_THROUGHPUT_NEXT_CANDIDATE_REVIEW_20260623_1045.md",
    "archetype_multilatent_negative": REPORTS / "LATENTFM_SOFT_ARCHETYPE_MULTILATENT_STATE_CPU_GATE_20260623.md",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def find_row(rows: list[dict[str, str]], *, role: str) -> dict[str, str]:
    for row in rows:
        if row.get("role") == role:
            return row
    raise KeyError(f"missing role: {role}")


def report_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines()[:24]:
        if line.startswith("Status:"):
            if "`" in line:
                return line.split("`")[1]
            return line.replace("Status:", "").strip()
    return "present"


def render_metric(row: dict[str, str]) -> str:
    return (
        f"| {row['stage']} | {row['group']} | {row['metric']} | {row['role']} | "
        f"{row['n_conditions']} | {row['n_datasets']} | {fmt(row['delta'])} | "
        f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {row['p_harm']} |"
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    table_rows = load_csv(args.table_csv)
    caveat_rows = load_csv(args.caveat_csv)
    claim = load_json(args.claim_json)
    audit = load_json(args.audit_json)
    failure = load_json(args.failure_json)
    portfolio = load_json(args.portfolio_json)

    primary_rows = {
        "support_pp": find_row(table_rows, role="primary_support_gain"),
        "support_mmd": find_row(table_rows, role="support_mmd"),
        "canonical_single_pp": find_row(table_rows, role="canonical_single_noharm"),
        "canonical_family_pp": find_row(table_rows, role="canonical_family_noharm"),
        "query_pp": find_row(table_rows, role="primary_query_gain"),
        "query_mmd": find_row(table_rows, role="primary_query_mmd"),
        "query_seen": find_row(table_rows, role="query_seen"),
        "query_unseen1": find_row(table_rows, role="query_unseen1"),
        "query_unseen2_pp": find_row(table_rows, role="query_unseen2_pp_caveat"),
        "query_unseen2_mmd": find_row(table_rows, role="query_unseen2_mmd"),
    }
    worst_pp = (failure.get("worst_pp_rows") or [{}])[0]
    sources = {
        name: {"path": str(path), "exists": path.exists(), "status": report_status(path)}
        for name, path in SOURCE_REPORTS.items()
    }

    failed = []
    if claim.get("failed_checks"):
        failed.append("claim_readiness_failed_checks_nonempty")
    if audit.get("failed_checks"):
        failed.append("final_package_audit_failed_checks_nonempty")
    if claim.get("status") != "claim_ready_as_frozen_support_context_v2_diagnostic_not_formal_multi_solution":
        failed.append("claim_readiness_status_unexpected")
    if audit.get("status") != "trackc_support_context_v2_final_package_audit_pass":
        failed.append("final_package_audit_status_unexpected")
    if portfolio.get("status") != "post_v2_portfolio_ready_reporting_plus_query_free_gates":
        failed.append("portfolio_status_unexpected")
    for name, info in sources.items():
        if not info["exists"]:
            failed.append(f"missing_source:{name}")

    status = "support_context_v2_reporting_package_ready" if not failed else "support_context_v2_reporting_package_needs_review"

    return {
        "status": status,
        "failed_checks": failed,
        "boundary": {
            "frozen_only": True,
            "query_tuning_forbidden": True,
            "gpu_authorization": "none",
            "claim_scope": "frozen_support_context_v2_diagnostic_not_formal_multi_solution",
            "new_gpu_requirement": "materially_new_query_free_cpu_gate",
        },
        "primary_object": {
            "run_name": claim.get("run_name"),
            "interpretation": "current strongest frozen Track C support-context diagnostic package",
        },
        "key_metrics": primary_rows,
        "caveat_rows": caveat_rows[:18],
        "failure_focus": {
            "worst_pp_row": worst_pp,
            "recurrent_gene_signals": failure.get("recurrent_gene_signals", [])[:10],
            "stratum_counts": failure.get("stratum_counts", {}),
        },
        "allowed_claims": claim.get("allowed_claims", []),
        "disallowed_claims": claim.get("disallowed_claims", []),
        "portfolio_decisions": portfolio.get("decisions", []),
        "source_reports": sources,
        "audit_statuses": {
            "claim_readiness": claim.get("status"),
            "final_package_audit": audit.get("status"),
            "portfolio": portfolio.get("status"),
        },
    }


def render_md(payload: dict[str, Any]) -> str:
    metrics = payload["key_metrics"]
    worst = payload["failure_focus"]["worst_pp_row"]
    sources = payload["source_reports"]

    lines: list[str] = [
        "# LatentFM Track C Support-Context V2 Reporting Package",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Scope Boundary",
        "",
        "- This package is a frozen reporting and handoff entry, not a selection artifact.",
        "- It does not authorize new GPU work or another held-out query.",
        "- Held-out query evidence is already consumed once and must not tune route, checkpoint, threshold, feature choice, alpha, or future branches.",
        "- Claim scope: frozen Track C support-context v2 diagnostic route, not a blanket formal multi-perturbation solution.",
        "",
        "## Frozen Object",
        "",
        f"- run: `{payload['primary_object']['run_name']}`",
        f"- interpretation: {payload['primary_object']['interpretation']}",
        "",
        "## Evidence Snapshot",
        "",
        "| stage | group | metric | role | rows | datasets | delta | 95% CI | p_harm |",
        "|---|---|---|---|---:|---:|---:|---|---:|",
        render_metric(metrics["support_pp"]),
        render_metric(metrics["support_mmd"]),
        render_metric(metrics["canonical_single_pp"]),
        render_metric(metrics["canonical_family_pp"]),
        render_metric(metrics["query_pp"]),
        render_metric(metrics["query_mmd"]),
        render_metric(metrics["query_seen"]),
        render_metric(metrics["query_unseen1"]),
        render_metric(metrics["query_unseen2_pp"]),
        render_metric(metrics["query_unseen2_mmd"]),
        "",
        "## Allowed Claim",
        "",
    ]
    for claim in payload["allowed_claims"]:
        lines.append(f"- {claim}")

    lines += ["", "## Disallowed Claims", ""]
    for claim in payload["disallowed_claims"]:
        lines.append(f"- {claim}")

    lines += [
        "",
        "## Required Limitations",
        "",
        "- Unseen2 Pearson evidence is weak; report it as a caveat even though unseen2 MMD improves.",
        "- Condition-level failures remain and must be shown alongside aggregate gains.",
        "- Residual v2 uncapped no-harm is robustness evidence only, not a second query candidate.",
        "- Canonical multi remains diagnostic/failure-analysis only and is not Track A checkpoint-selection evidence.",
        "",
        "## Failure Focus",
        "",
        f"- Worst pp row: `{worst.get('dataset')}` / `{worst.get('condition')}`, stratum `{worst.get('stratum')}`, pp delta `{fmt(worst.get('pp_delta'))}`, MMD delta `{fmt(worst.get('mmd_delta'))}`.",
        "- Recurrent weak genes include "
        + ", ".join(f"`{row.get('gene')}`" for row in payload["failure_focus"]["recurrent_gene_signals"][:8])
        + ".",
        "",
        "## Caveat Table Preview",
        "",
        "| type | stratum | dataset | condition/gene | pp delta | MMD delta |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in payload["caveat_rows"][:12]:
        label = row.get("condition") or row.get("genes") or row.get("condition_or_gene", "")
        lines.append(
            f"| {row.get('type', '')} | {row.get('stratum', '')} | {row.get('dataset', '')} | "
            f"`{label}` | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} |"
        )

    lines += ["", "## Closed Or Demoted Branches", ""]
    for branch in payload["portfolio_decisions"]:
        lines.append(
            f"- `{branch.get('branch')}`: `{branch.get('decision')}`; GPU `{branch.get('gpu_authorization')}`. "
            f"Next: {branch.get('next_action')}"
        )

    lines += [
        "",
        "## Figure And Table Plan",
        "",
        "- Main table: support-val capped gain, uncapped canonical no-harm, held-out query all/seen/unseen1/unseen2.",
        "- Provenance panel: safe trainselect split, full v2 query split, canonical split hashes, checkpoint hash, query-free freeze gate, one-shot query decision.",
        "- Failure panel: worst condition rows, recurrent weak genes, and unseen2 Pearson-vs-MMD contrast.",
        "- Boundary panel: allowed/disallowed claim wording and query-not-for-tuning rule.",
        "",
        "## Source Artifacts",
        "",
    ]
    for name, info in sources.items():
        lines.append(f"- `{name}`: `{info['status']}`; `{info['path']}`")

    lines += [
        "",
        "## Next Action",
        "",
        "Use this as the single reporting entry for the support-context v2 package. New experimental work needs a materially new query-free CPU gate with hypothesis, promotion gate, stop rule, launcher/RUN_STATUS, and fresh resource audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--caveat-csv", type=Path, default=DEFAULT_CAVEAT_CSV)
    parser.add_argument("--claim-json", type=Path, default=DEFAULT_CLAIM_JSON)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--failure-json", type=Path, default=DEFAULT_FAILURE_JSON)
    parser.add_argument("--portfolio-json", type=Path, default=DEFAULT_PORTFOLIO_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    payload = build_payload(args)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 1 if payload["failed_checks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
