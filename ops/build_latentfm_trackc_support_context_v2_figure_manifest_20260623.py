#!/usr/bin/env python3
"""Build a figure/provenance manifest for Track C support-context v2 reporting.

This is a read-only reporting helper. It does not generate figures, run models,
read new held-out query artifacts, or authorize GPU work.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_figure_manifest_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FIGURE_MANIFEST_20260623.md"

INPUTS = {
    "reporting_package_json": REPORTS / "latentfm_trackc_support_context_v2_reporting_package_20260623.json",
    "reporting_package_md": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_REPORTING_PACKAGE_20260623.md",
    "manuscript_table_csv": REPORTS / "latentfm_trackc_support_context_v2_manuscript_table_20260623.csv",
    "caveat_table_csv": REPORTS / "latentfm_trackc_support_context_v2_caveat_table_20260623.csv",
    "claim_readiness_json": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "final_package_audit_json": REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json",
    "failure_cases_json": REPORTS / "latentfm_trackc_support_context_v2_query_failure_cases_20260623.json",
    "portfolio_json": REPORTS / "latentfm_post_support_context_v2_portfolio_decision_20260623.json",
}

EXPECTED_STATUSES = {
    "reporting_package_json": "support_context_v2_reporting_package_ready",
    "claim_readiness_json": "claim_ready_as_frozen_support_context_v2_diagnostic_not_formal_multi_solution",
    "final_package_audit_json": "trackc_support_context_v2_final_package_audit_pass",
    "failure_cases_json": "trackc_support_context_v2_query_failure_cases_ready",
    "portfolio_json": "post_v2_portfolio_ready_reporting_plus_query_free_gates",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    rec: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        rec["size_bytes"] = path.stat().st_size
        rec["sha256"] = sha256_file(path)
    return rec


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def row_by_role(rows: list[dict[str, str]], role: str) -> dict[str, str]:
    for row in rows:
        if row.get("role") == role:
            return row
    raise KeyError(role)


def metric_label(row: dict[str, str]) -> str:
    return (
        f"{row['group']} {row['metric']} delta {fmt(row['delta'])}, "
        f"CI [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}], p_harm {row['p_harm']}"
    )


def panel(panel_id: str, title: str, purpose: str, sources: list[str], metrics: list[str], cautions: list[str]) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "title": title,
        "purpose": purpose,
        "source_artifacts": sources,
        "metrics_or_content": metrics,
        "cautions": cautions,
    }


def main() -> int:
    reporting = load_json(INPUTS["reporting_package_json"])
    table_rows = load_csv(INPUTS["manuscript_table_csv"])
    caveat_rows = load_csv(INPUTS["caveat_table_csv"])
    claim = load_json(INPUTS["claim_readiness_json"])
    final_audit = load_json(INPUTS["final_package_audit_json"])
    failure = load_json(INPUTS["failure_cases_json"])
    portfolio = load_json(INPUTS["portfolio_json"])

    artifacts = {name: artifact_record(path) for name, path in INPUTS.items()}
    checks: list[dict[str, Any]] = []
    for name, rec in artifacts.items():
        checks.append({"name": f"exists:{name}", "passed": rec["exists"], "evidence": rec["path"]})

    status_payloads = {
        "reporting_package_json": reporting,
        "claim_readiness_json": claim,
        "final_package_audit_json": final_audit,
        "failure_cases_json": failure,
        "portfolio_json": portfolio,
    }
    for name, expected in EXPECTED_STATUSES.items():
        observed = status_payloads[name].get("status")
        checks.append(
            {
                "name": f"status:{name}",
                "passed": observed == expected,
                "evidence": {"expected": expected, "observed": observed},
            }
        )

    support_pp = row_by_role(table_rows, "primary_support_gain")
    support_mmd = row_by_role(table_rows, "support_mmd")
    canonical_single = row_by_role(table_rows, "canonical_single_noharm")
    canonical_family = row_by_role(table_rows, "canonical_family_noharm")
    query_pp = row_by_role(table_rows, "primary_query_gain")
    query_mmd = row_by_role(table_rows, "primary_query_mmd")
    query_seen = row_by_role(table_rows, "query_seen")
    query_unseen1 = row_by_role(table_rows, "query_unseen1")
    query_unseen2_pp = row_by_role(table_rows, "query_unseen2_pp_caveat")
    query_unseen2_mmd = row_by_role(table_rows, "query_unseen2_mmd")

    worst_rows = caveat_rows[:12]
    recurrent_gene_rows = [row for row in caveat_rows if row.get("type") == "recurrent_gene"][:8]
    worst = failure["worst_pp_rows"][0]

    panels = [
        panel(
            "fig_trackc_v2_gate_chain",
            "Gate Chain And Provenance",
            "Show the order of support-val capped pass, uncapped canonical no-harm, query-free freeze, one-shot query, and reporting boundary.",
            ["final_package_audit_json", "claim_readiness_json", "reporting_package_json"],
            [
                f"split hashes: {final_audit.get('split_hashes')}",
                f"checkpoint hashes from freeze: {final_audit.get('freeze_hashes')}",
                f"CoupledFM commit: {final_audit.get('git', {}).get('coupledfm_commit')}",
            ],
            ["This panel must show query was final-only and not used for selection."],
        ),
        panel(
            "fig_trackc_v2_main_metrics",
            "Main Metric Table",
            "Compact table of support, canonical no-harm, and final query evidence.",
            ["manuscript_table_csv", "reporting_package_json"],
            [
                metric_label(support_pp),
                metric_label(support_mmd),
                metric_label(canonical_single),
                metric_label(canonical_family),
                metric_label(query_pp),
                metric_label(query_mmd),
            ],
            ["Do not merge support-val evidence and held-out query evidence into one selection metric."],
        ),
        panel(
            "fig_trackc_v2_query_strata",
            "Held-Out Query Strata",
            "Separate seen, unseen1, and unseen2 behavior.",
            ["manuscript_table_csv", "claim_readiness_json"],
            [metric_label(query_seen), metric_label(query_unseen1), metric_label(query_unseen2_pp), metric_label(query_unseen2_mmd)],
            ["Unseen2 Pearson is a caveat despite MMD improvement."],
        ),
        panel(
            "fig_trackc_v2_failure_cases",
            "Failure Cases",
            "Show worst condition-level rows and recurrent weak genes.",
            ["caveat_table_csv", "failure_cases_json"],
            [
                f"worst row: {worst['dataset']}/{worst['condition']} pp {fmt(worst['pp_delta'])}, MMD {fmt(worst['mmd_delta'])}",
                "top recurrent genes: " + ", ".join(row.get("gene") or row.get("condition") or row.get("genes", "") for row in recurrent_gene_rows),
            ],
            ["Do not hide condition-level failures behind aggregate query gains."],
        ),
        panel(
            "fig_trackc_v2_claim_boundary",
            "Claim Boundary",
            "State allowed/disallowed claims and next-gate rule.",
            ["claim_readiness_json", "portfolio_json", "reporting_package_json"],
            [
                "allowed: " + "; ".join(claim.get("allowed_claims", [])[:3]),
                "disallowed: " + "; ".join(claim.get("disallowed_claims", [])[:3]),
                "new GPU requires materially new query-free CPU gate",
            ],
            ["This panel should explicitly say not a blanket formal multi solution."],
        ),
    ]

    failed = [row for row in checks if not row["passed"]]
    status = "support_context_v2_figure_manifest_ready" if not failed else "support_context_v2_figure_manifest_needs_review"

    manifest = {
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "boundary": {
            "read_only_reporting": True,
            "generates_figures": False,
            "gpu_authorization": "none",
            "heldout_query_reuse_forbidden": True,
            "claim_scope": "frozen_support_context_v2_diagnostic_not_formal_multi_solution",
        },
        "checks": checks,
        "failed_checks": failed,
        "artifacts": artifacts,
        "panels": panels,
        "main_metric_rows": {
            "support_pp": support_pp,
            "support_mmd": support_mmd,
            "canonical_single": canonical_single,
            "canonical_family": canonical_family,
            "query_pp": query_pp,
            "query_mmd": query_mmd,
            "query_seen": query_seen,
            "query_unseen1": query_unseen1,
            "query_unseen2_pp": query_unseen2_pp,
            "query_unseen2_mmd": query_unseen2_mmd,
        },
        "caveat_preview": worst_rows,
        "portfolio_decisions": portfolio.get("decisions", []),
        "next_action": "Use this manifest to build figure panels only; new experiments require a materially new query-free CPU gate.",
    }
    OUT_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Support-Context V2 Figure Manifest",
        "",
        f"Timestamp: `{manifest['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Read-only reporting manifest; no models or evaluations were run.",
        "- Does not generate final figures yet; it defines figure-ready panels and provenance inputs.",
        "- Does not authorize GPU work or another held-out query.",
        "- Claim scope remains frozen support-context v2 diagnostic route, not formal multi solved.",
        "",
        "## Panel Plan",
        "",
        "| panel | title | purpose | key cautions |",
        "|---|---|---|---|",
    ]
    for p in panels:
        lines.append(
            f"| `{p['panel_id']}` | {p['title']} | {p['purpose']} | "
            + "; ".join(p["cautions"])
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evidence Snapshot",
            "",
            "| evidence | value |",
            "|---|---|",
            f"| support Pearson | {metric_label(support_pp)} |",
            f"| support MMD | {metric_label(support_mmd)} |",
            f"| canonical single no-harm | {metric_label(canonical_single)} |",
            f"| canonical family no-harm | {metric_label(canonical_family)} |",
            f"| held-out query Pearson | {metric_label(query_pp)} |",
            f"| held-out query MMD | {metric_label(query_mmd)} |",
            f"| query seen Pearson | {metric_label(query_seen)} |",
            f"| query unseen1 Pearson | {metric_label(query_unseen1)} |",
            f"| query unseen2 Pearson caveat | {metric_label(query_unseen2_pp)} |",
            f"| query unseen2 MMD | {metric_label(query_unseen2_mmd)} |",
            "",
            "## Failure Preview",
            "",
            "| type | stratum | dataset | condition/gene | pp delta | MMD delta |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in worst_rows[:12]:
        label = row.get("condition") or row.get("genes", "")
        lines.append(
            f"| {row.get('type', '')} | {row.get('stratum', '')} | {row.get('dataset', '')} | "
            f"`{label}` | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## Provenance Inputs",
            "",
            "| artifact | exists | size bytes | SHA256 |",
            "|---|---:|---:|---|",
        ]
    )
    for name, rec in artifacts.items():
        lines.append(
            f"| `{name}` | `{rec['exists']}` | `{rec.get('size_bytes', 'NA')}` | `{rec.get('sha256', 'NA')}` |"
        )
    lines.extend(
        [
            "",
            "## Reused Audited Hashes",
            "",
            f"- split hashes: `{json.dumps(final_audit.get('split_hashes', {}), sort_keys=True)}`",
            f"- checkpoint hashes from freeze gate: `{json.dumps(final_audit.get('freeze_hashes', {}), sort_keys=True)}`",
            f"- CoupledFM commit: `{final_audit.get('git', {}).get('coupledfm_commit')}`",
            "",
            "## Checks",
            "",
            "| check | passed | evidence |",
            "|---|---:|---|",
        ]
    )
    for row in checks:
        evidence = json.dumps(row["evidence"], sort_keys=True) if isinstance(row["evidence"], dict) else str(row["evidence"])
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {evidence[:220]} |")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "Use this manifest to build manuscript-style figure panels from the frozen package. Do not use it for route/checkpoint/threshold selection, residual query, or GPU launch authorization.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
