#!/usr/bin/env python3
"""Build a single-entry reporting package for the frozen LatentFM diagnostic.

This reads only already-frozen reports/JSON artifacts. It does not run models,
inspect held-out query for tuning, or authorize any GPU branch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

DEFAULT_TABLE_CSV = REPORTS / "latentfm_trackc_anchor_gated_blend_manuscript_table_20260623.csv"
DEFAULT_CI_JSON = REPORTS / "latentfm_trackc_anchor_gated_blend_reporting_ci_20260623.json"
DEFAULT_FAILURE_JSON = REPORTS / "latentfm_trackc_anchor_gated_blend_failure_cases_20260623.json"
DEFAULT_AUDIT_JSON = REPORTS / "latentfm_trackc_anchor_gated_blend_frozen_package_audit_20260623.json"
DEFAULT_PORTFOLIO_JSON = REPORTS / "latentfm_post_learned_gate_portfolio_decision_20260623.json"
DEFAULT_OUT_MD = REPORTS / "LATENTFM_FROZEN_DIAGNOSTIC_REPORTING_PACKAGE_20260623.md"
DEFAULT_OUT_JSON = REPORTS / "latentfm_frozen_diagnostic_reporting_package_20260623.json"

SOURCE_REPORTS = {
    "frozen_package_audit": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FROZEN_PACKAGE_AUDIT_20260623.md",
    "manuscript_table": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_MANUSCRIPT_TABLE_20260623.md",
    "reporting_ci": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_REPORTING_CI_20260623.md",
    "failure_cases": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FAILURE_CASES_20260623.md",
    "claim_readiness": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_CLAIM_READINESS_AUDIT_20260623.md",
    "synthesis": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FINAL_DIAGNOSTIC_SYNTHESIS_20260623.md",
    "provenance": REPORTS / "LATENTFM_TRACKC_ANCHOR_GATED_BLEND_ARTIFACT_PROVENANCE_20260623.md",
    "portfolio_decision": REPORTS / "LATENTFM_POST_LEARNED_GATE_PORTFOLIO_DECISION_20260623.md",
    "learned_gate_negative": REPORTS / "LATENTFM_TRACKC_LEARNED_ANCHOR_GATE_CPU_GATE_20260623.md",
    "archetype_negative": REPORTS / "LATENTFM_SOFT_ARCHETYPE_ORTHOGONAL_ROUTER_CPU_GATE_20260623.md",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def metric_row(rows: list[dict[str, str]], block: str) -> dict[str, str]:
    for row in rows:
        if row.get("block") == block:
            return row
    raise KeyError(f"missing table block: {block}")


def report_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if line.startswith("Status:"):
            return line.split("`")[1] if "`" in line else line.replace("Status:", "").strip()
    return "present"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    table_rows = load_table(args.table_csv)
    ci_payload = load_json(args.ci_json)
    failure_payload = load_json(args.failure_json)
    audit_payload = load_json(args.audit_json)
    portfolio_payload = load_json(args.portfolio_json)

    sources = {
        name: {"path": str(path), "exists": path.exists(), "status": report_status(path)}
        for name, path in SOURCE_REPORTS.items()
    }
    worst_pp = (failure_payload.get("worst_pp_rows") or [{}])[0]
    query_unseen2 = metric_row(table_rows, "held-out query unseen2")
    query_all = metric_row(table_rows, "held-out query all")
    support = metric_row(table_rows, "support selection")
    family = metric_row(table_rows, "canonical no-harm family")

    return {
        "status": "frozen_diagnostic_reporting_package_ready",
        "boundary": {
            "frozen_only": True,
            "query_tuning_forbidden": True,
            "gpu_authorization": "none",
            "claim_scope": "diagnostic_calibrator_not_formal_multi_solution",
        },
        "primary_object": {
            "formula": "pred = anchor_pred + gate * 0.75 * (support_teacher_pred - anchor_pred)",
            "interpretation": "current best frozen Track C diagnostic/calibrator",
        },
        "key_metrics": {
            "support_selection": support,
            "canonical_noharm_family": family,
            "heldout_query_all": query_all,
            "heldout_query_unseen2": query_unseen2,
        },
        "failure_focus": {
            "worst_pp_row": worst_pp,
            "stratum_counts": failure_payload.get("stratum_counts", {}),
            "recurrent_gene_failures": failure_payload.get("recurrent_gene_failures", [])[:8],
        },
        "portfolio_status": {
            "status": portfolio_payload.get("status"),
            "decision": portfolio_payload.get("decision"),
            "branches": portfolio_payload.get("branches", []),
        },
        "audit": {
            "frozen_package_status": audit_payload.get("status"),
            "audit_reasons": audit_payload.get("reasons", []),
            "ci_status": ci_payload.get("status"),
        },
        "source_reports": sources,
    }


def render_metric(row: dict[str, str]) -> str:
    return (
        f"| {row['block']} | {row['rows']} | {row['datasets']} | {fmt(row['pp_delta'])} | "
        f"[{fmt(row['pp_ci_low'])}, {fmt(row['pp_ci_high'])}] | {fmt(row['mmd_delta'])} | "
        f"[{fmt(row['mmd_ci_low'])}, {fmt(row['mmd_ci_high'])}] |"
    )


def render_md(payload: dict[str, Any]) -> str:
    metrics = payload["key_metrics"]
    worst = payload["failure_focus"]["worst_pp_row"]
    branches = payload["portfolio_status"]["branches"]
    sources = payload["source_reports"]

    lines = [
        "# LatentFM Frozen Diagnostic Reporting Package",
        "",
        "Status: `frozen_diagnostic_reporting_package_ready`",
        "",
        "## Scope Boundary",
        "",
        "- This package is a frozen reporting and handoff entry, not a selection artifact.",
        "- It does not authorize new GPU work.",
        "- Held-out query evidence is already consumed once and must not tune alpha, gate, threshold, checkpoint, route, or future branches.",
        "- Claim scope: frozen Track C diagnostic/calibrator, not a deployable formal multi solution.",
        "",
        "## Frozen Object",
        "",
        f"`{payload['primary_object']['formula']}`",
        "",
        "Current interpretation: current best frozen Track C diagnostic/calibrator.",
        "",
        "## Evidence Snapshot",
        "",
        "| block | rows | datasets | pp delta | pp 95% CI | MMD delta | MMD 95% CI |",
        "|---|---:|---:|---:|---|---:|---|",
        render_metric(metrics["support_selection"]),
        render_metric(metrics["canonical_noharm_family"]),
        render_metric(metrics["heldout_query_all"]),
        render_metric(metrics["heldout_query_unseen2"]),
        "",
        "## Allowed Claim",
        "",
        "The frozen anchor-gated support-teacher blend is supported as a diagnostic/calibrator: support-val, held-out aggregate, seen, and unseen1 evidence are positive, and canonical Track A no-harm is preserved.",
        "",
        "## Required Limitations",
        "",
        "- Unseen2 pearson_pert evidence is weak because its confidence interval crosses zero, although MMD hard-harm is not observed.",
        "- The support-teacher residual route is not a deployable train/support-derived gate; canonical no-harm relies on the frozen route boundary.",
        "- Formal multi capability remains a Track C research problem, not a solved claim.",
        "",
        "## Failure Focus",
        "",
        f"- Worst pp row: `{worst.get('dataset')}` / `{worst.get('condition')}`, stratum `{worst.get('stratum')}`, pp delta `{fmt(worst.get('pp_delta'))}`, MMD delta `{fmt(worst.get('mmd_delta'))}`.",
        "- Wessels unseen2 remains a fragile stratum: pp negative fraction `0.631579`, MMD harm fraction `0`.",
        "- Failure analysis should emphasize MAPK1, EP300/Mediator-related, and UBASH3B-like recurring genes rather than hiding condition-level losses behind aggregate gains.",
        "",
        "## Closed Or Demoted Branches",
        "",
    ]
    for branch in branches:
        lines.append(
            f"- `{branch.get('branch')}`: `{branch.get('decision')}`; GPU `{branch.get('gpu')}`. Next: {branch.get('next')}"
        )

    lines += [
        "",
        "## Figure And Table Plan",
        "",
        "- Main table: support selection, canonical no-harm single/family, held-out query all/seen/unseen1/unseen2.",
        "- Provenance panel: split hashes, anchor checkpoint, support-teacher checkpoint, posthoc decision, one-shot query artifact.",
        "- Failure panel: worst condition rows and Wessels unseen2 pp-vs-MMD contrast.",
        "- Boundary panel: allowed/disallowed claim wording and query-not-for-tuning rule.",
        "",
        "## Source Artifacts",
        "",
    ]
    for name, info in sources.items():
        status = info["status"]
        lines.append(f"- `{name}`: `{status}`; `{info['path']}`")

    lines += [
        "",
        "## Next Action",
        "",
        "Use this as the single reporting entry for the current best diagnostic package. New experimental work needs a materially new query-free CPU gate; do not relaunch consumed threshold, coverage, lowcount, cross-latent, Jiang, or archetype-threshold variants as GPU jobs.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--ci-json", type=Path, default=DEFAULT_CI_JSON)
    parser.add_argument("--failure-json", type=Path, default=DEFAULT_FAILURE_JSON)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--portfolio-json", type=Path, default=DEFAULT_PORTFOLIO_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    payload = build_payload(args)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
