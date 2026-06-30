#!/usr/bin/env python3
"""Build the post-support-set final handoff for the frozen v2 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_final_handoff_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_HANDOFF_20260623.md"

INPUTS = {
    "reporting_package": REPORTS / "latentfm_trackc_support_context_v2_reporting_package_20260623.json",
    "claim_readiness": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "figure_manifest": REPORTS / "latentfm_trackc_support_context_v2_figure_manifest_20260623.json",
    "manuscript_table_csv": REPORTS / "latentfm_trackc_support_context_v2_manuscript_table_20260623.csv",
    "caveat_table_csv": REPORTS / "latentfm_trackc_support_context_v2_caveat_table_20260623.csv",
    "failure_cases": REPORTS / "latentfm_trackc_support_context_v2_query_failure_cases_20260623.json",
    "final_package_audit": REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json",
    "post_summary_portfolio": REPORTS / "latentfm_post_support_set_summary_portfolio_decision_20260623.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def status_of(obj: dict[str, Any]) -> str | None:
    if obj.get("status") is not None:
        return str(obj["status"])
    decision = obj.get("decision")
    if isinstance(decision, dict) and decision.get("status") is not None:
        return str(decision["status"])
    return None


def compact_metric(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    ci95 = row.get("ci95")
    if ci95 is None and ("ci95_low" in row or "ci95_high" in row):
        ci95 = [row.get("ci95_low"), row.get("ci95_high")]
    return {
        "group": row.get("group"),
        "metric": row.get("metric"),
        "delta": row.get("delta_mean") if row.get("delta_mean") is not None else row.get("delta"),
        "ci95": ci95,
        "p_harm": row.get("p_harm"),
        "rows": row.get("n_matched_conditions") or row.get("n_conditions") or row.get("rows"),
        "datasets": row.get("n_matched_datasets") or row.get("n_datasets") or row.get("datasets"),
    }


def build_payload() -> dict[str, Any]:
    reporting = load_json(INPUTS["reporting_package"])
    claim = load_json(INPUTS["claim_readiness"])
    figure = load_json(INPUTS["figure_manifest"])
    failure = load_json(INPUTS["failure_cases"])
    final_audit = load_json(INPUTS["final_package_audit"])
    portfolio = load_json(INPUTS["post_summary_portfolio"])

    key = reporting.get("key_metrics") or {}
    return {
        "status": "support_context_v2_final_handoff_ready_post_support_set_closure",
        "timestamp": "2026-06-23 12:36 CST",
        "primary_object": reporting.get("primary_object"),
        "current_best": "frozen Track C support-context v2 diagnostic/reporting candidate",
        "claim_scope": (reporting.get("boundary") or {}).get("claim_scope"),
        "gpu_authorization": "none",
        "heldout_query_reuse_forbidden": True,
        "input_statuses": {
            "reporting_package": reporting.get("status"),
            "claim_readiness": claim.get("status"),
            "figure_manifest": figure.get("status"),
            "final_package_audit": final_audit.get("status"),
            "failure_cases": failure.get("status"),
            "post_summary_portfolio": portfolio.get("status"),
        },
        "key_metrics": {
            "support_pp": compact_metric(key.get("support_pp")),
            "support_mmd": compact_metric(key.get("support_mmd")),
            "canonical_single_pp": compact_metric(key.get("canonical_single_pp")),
            "canonical_family_pp": compact_metric(key.get("canonical_family_pp")),
            "query_pp": compact_metric(key.get("query_pp")),
            "query_mmd": compact_metric(key.get("query_mmd")),
            "query_seen": compact_metric(key.get("query_seen")),
            "query_unseen1": compact_metric(key.get("query_unseen1")),
            "query_unseen2_pp": compact_metric(key.get("query_unseen2_pp")),
            "query_unseen2_mmd": compact_metric(key.get("query_unseen2_mmd")),
        },
        "allowed_claims": reporting.get("allowed_claims") or [],
        "disallowed_claims": reporting.get("disallowed_claims") or [],
        "limitations": (claim.get("limitations") or []) if isinstance(claim.get("limitations"), list) else [
            "unseen2 Pearson is weak",
            "condition-level failures remain",
            "query artifact is consumed and cannot guide tuning",
        ],
        "failure_focus": reporting.get("failure_focus") or {},
        "caveat_preview": (reporting.get("caveat_rows") or [])[:12],
        "closed_after_post_summary": portfolio.get("closed_no_relaunch") or [],
        "branch_decisions": portfolio.get("decisions") or [],
        "figure_plan": {
            "panels": (figure.get("figure_panels") or figure.get("panels") or []),
            "artifacts": {
                name: artifact(path)
                for name, path in INPUTS.items()
                if name not in {"reporting_package", "claim_readiness", "figure_manifest", "failure_cases", "final_package_audit", "post_summary_portfolio"}
            },
        },
        "artifacts": {name: artifact(path) for name, path in INPUTS.items()},
        "next_action": (
            "Use this handoff for manuscript/reporting tables, figure panels, caveats, and failure analysis. "
            "New modeling requires a materially new query-free CPU gate; no GPU/query is authorized here."
        ),
    }


def fmt_num(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Final Handoff",
        "",
        f"Status: `{payload['status']}`",
        f"Current best: {payload['current_best']}",
        f"Claim scope: `{payload['claim_scope']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Held-out query reuse forbidden: `{payload['heldout_query_reuse_forbidden']}`",
        "",
        "## Key Metrics",
        "",
        "| item | group | metric | rows | datasets | delta | 95% CI | p_harm |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for name, row in payload["key_metrics"].items():
        ci = row.get("ci95") or [None, None]
        ci_s = f"[{fmt_num(ci[0])}, {fmt_num(ci[1])}]"
        lines.append(
            f"| `{name}` | {row.get('group')} | {row.get('metric')} | "
            f"{row.get('rows') or 'NA'} | {row.get('datasets') or 'NA'} | "
            f"{fmt_num(row.get('delta'))} | {ci_s} | {fmt_num(row.get('p_harm'))} |"
        )
    lines.extend(["", "## Allowed Claims", ""])
    lines.extend(f"- {item}" for item in payload["allowed_claims"])
    lines.extend(["", "## Disallowed Claims", ""])
    lines.extend(f"- {item}" for item in payload["disallowed_claims"])
    lines.extend(["", "## Failure/Caveat Preview", ""])
    worst = (payload.get("failure_focus") or {}).get("worst_pp_row") or {}
    if worst:
        lines.append(
            f"- Worst pp row: `{worst.get('dataset')}` / `{worst.get('condition')}`, "
            f"pp delta `{fmt_num(worst.get('pp_delta'))}`, MMD delta `{fmt_num(worst.get('mmd_delta'))}`."
        )
    recurrent = (payload.get("failure_focus") or {}).get("recurrent_gene_signals") or []
    if recurrent:
        genes = ", ".join(f"`{row.get('gene')}`" for row in recurrent[:8])
        lines.append(f"- Recurrent weak genes: {genes}.")
    lines.extend(["", "| type | stratum | dataset | condition/gene | pp delta | MMD delta |", "|---|---|---|---|---:|---:|"])
    for row in payload["caveat_preview"]:
        lines.append(
            f"| {row.get('type')} | {row.get('stratum')} | {row.get('dataset')} | "
            f"`{row.get('condition') or row.get('genes')}` | {fmt_num(row.get('pp_delta'))} | {fmt_num(row.get('mmd_delta'))} |"
        )
    lines.extend(["", "## Closed / Do Not Relaunch", ""])
    lines.extend(f"- {item}" for item in payload["closed_after_post_summary"])
    lines.extend(["", "## Artifact Entrypoints", "", "| artifact | exists | size | sha256 | path |", "|---|---:|---:|---|---|"])
    for name, meta in sorted(payload["artifacts"].items()):
        sha = meta.get("sha256")
        sha_s = "NA" if not sha else str(sha)[:16]
        lines.append(f"| `{name}` | `{meta.get('exists')}` | {meta.get('size_bytes') or 'NA'} | `{sha_s}` | `{meta.get('path')}` |")
    lines.extend(["", "## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
