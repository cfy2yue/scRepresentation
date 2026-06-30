#!/usr/bin/env python3
"""Build a compact final reporting bundle for frozen Track C v2.

The bundle is a single reporting entrypoint over already-frozen artifacts. It
does not train, evaluate, inspect active logs, tune, select, or authorize
held-out query reuse/GPU work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_compact_reporting_bundle_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_COMPACT_REPORTING_BUNDLE_20260623.md"

INPUTS = {
    "final_handoff": REPORTS / "latentfm_trackc_support_context_v2_final_handoff_20260623.json",
    "manuscript_narrative": REPORTS / "latentfm_trackc_support_context_v2_manuscript_narrative_20260623.json",
    "figure_panels": REPORTS / "latentfm_trackc_support_context_v2_figure_panels_20260623.json",
    "manuscript_table_csv": REPORTS / "latentfm_trackc_support_context_v2_manuscript_table_20260623.csv",
    "caveat_table_csv": REPORTS / "latentfm_trackc_support_context_v2_caveat_table_20260623.csv",
    "claim_readiness": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "final_package_audit": REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json",
}

EXPECTED_STATUSES = {
    "final_handoff": "support_context_v2_final_handoff_ready_post_support_set_closure",
    "manuscript_narrative": "support_context_v2_manuscript_narrative_ready",
    "figure_panels": "support_context_v2_figure_panels_ready",
    "claim_readiness": "claim_ready_as_frozen_support_context_v2_diagnostic_not_formal_multi_solution",
    "final_package_audit": "trackc_support_context_v2_final_package_audit_pass",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def metric_line(row: dict[str, Any]) -> str:
    ci = row.get("ci95") or [None, None]
    return (
        f"{row.get('group')} {row.get('metric')} delta {fmt(row.get('delta'))}, "
        f"95% CI [{fmt(ci[0])}, {fmt(ci[1])}], p_harm {fmt(row.get('p_harm'))}, "
        f"rows {row.get('rows')}"
    )


def short_hash(meta: dict[str, Any]) -> str:
    value = meta.get("sha256")
    return "NA" if not value else str(value)[:16]


def build_payload() -> dict[str, Any]:
    handoff = load_json(INPUTS["final_handoff"])
    narrative = load_json(INPUTS["manuscript_narrative"])
    figures = load_json(INPUTS["figure_panels"])
    claim = load_json(INPUTS["claim_readiness"])
    audit = load_json(INPUTS["final_package_audit"])

    checks: list[dict[str, Any]] = []
    for name, path in INPUTS.items():
        checks.append({"name": f"exists:{name}", "passed": path.is_file(), "evidence": str(path)})
    status_objects = {
        "final_handoff": handoff,
        "manuscript_narrative": narrative,
        "figure_panels": figures,
        "claim_readiness": claim,
        "final_package_audit": audit,
    }
    for name, expected in EXPECTED_STATUSES.items():
        observed = status_objects[name].get("status")
        checks.append({"name": f"status:{name}", "passed": observed == expected, "evidence": {"expected": expected, "observed": observed}})
    checks.append({"name": "figure_panels:failed_checks_zero", "passed": not figures.get("failed_checks"), "evidence": len(figures.get("failed_checks", []))})
    checks.append({"name": "narrative:failed_checks_zero", "passed": not (narrative.get("provenance") or {}).get("failed_checks"), "evidence": len((narrative.get("provenance") or {}).get("failed_checks", []))})
    checks.append({"name": "audit:failed_checks_zero", "passed": not audit.get("failed_checks"), "evidence": len(audit.get("failed_checks", []))})

    figure_files = {}
    for panel in figures.get("panels", []):
        panel_id = panel["panel_id"]
        figure_files[f"{panel_id}_png"] = Path(panel["png"])
        figure_files[f"{panel_id}_svg"] = Path(panel["svg"])
        checks.append({"name": f"figure:{panel_id}:nonblank", "passed": bool((panel.get("pixel_check") or {}).get("nonblank")), "evidence": panel.get("pixel_check")})

    figure_artifacts = {name: artifact(path) for name, path in figure_files.items()}
    for name, meta in figure_artifacts.items():
        checks.append({"name": f"exists:{name}", "passed": meta["exists"] and (meta["size_bytes"] or 0) > 1000, "evidence": meta})

    failed = [row for row in checks if not row["passed"]]
    metrics = handoff.get("key_metrics", {})
    payload = {
        "status": "support_context_v2_compact_reporting_bundle_ready" if not failed else "support_context_v2_compact_reporting_bundle_needs_review",
        "timestamp": "2026-06-23 12:50 CST",
        "boundary": {
            "reporting_only": True,
            "gpu_authorization": "none",
            "heldout_query_reuse_forbidden": True,
            "selection_or_tuning": False,
            "active_log_reads": False,
            "claim_scope": handoff.get("claim_scope"),
        },
        "executive_summary": narrative.get("result_paragraph"),
        "key_metrics": {
            "support_pp": metrics.get("support_pp"),
            "support_mmd": metrics.get("support_mmd"),
            "canonical_single_pp": metrics.get("canonical_single_pp"),
            "canonical_family_pp": metrics.get("canonical_family_pp"),
            "query_pp": metrics.get("query_pp"),
            "query_mmd": metrics.get("query_mmd"),
            "query_seen": metrics.get("query_seen"),
            "query_unseen1": metrics.get("query_unseen1"),
            "query_unseen2_pp": metrics.get("query_unseen2_pp"),
            "query_unseen2_mmd": metrics.get("query_unseen2_mmd"),
        },
        "primary_artifacts": {name: artifact(path) for name, path in INPUTS.items()},
        "figure_artifacts": figure_artifacts,
        "allowed_claims": handoff.get("allowed_claims", []),
        "disallowed_claims": handoff.get("disallowed_claims", []),
        "closed_no_relaunch": handoff.get("closed_after_post_summary", []),
        "caveat_paragraph": narrative.get("caveat_paragraph"),
        "checks": checks,
        "failed_checks": failed,
        "next_action": (
            "Use this compact bundle as the final reporting entry for the frozen diagnostic. "
            "Any further modeling requires a materially new query-free CPU gate."
        ),
    }
    return payload


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Compact Reporting Bundle",
        "",
        f"Status: `{payload['status']}`",
        f"Claim scope: `{payload['boundary']['claim_scope']}`",
        "GPU authorization: `none`",
        "Held-out query reuse forbidden: `True`",
        "",
        "## Executive Summary",
        "",
        payload["executive_summary"],
        "",
        "## Key Metrics",
        "",
    ]
    for label in [
        "support_pp",
        "support_mmd",
        "canonical_single_pp",
        "canonical_family_pp",
        "query_pp",
        "query_mmd",
        "query_seen",
        "query_unseen1",
        "query_unseen2_pp",
        "query_unseen2_mmd",
    ]:
        row = payload["key_metrics"].get(label) or {}
        lines.append(f"- `{label}`: {metric_line(row)}")

    lines.extend(["", "## Figures", "", "| panel artifact | path | sha256 |", "|---|---|---|"])
    for name, meta in sorted(payload["figure_artifacts"].items()):
        lines.append(f"| `{name}` | `{meta['path']}` | `{short_hash(meta)}` |")

    lines.extend(["", "## Tables And Text", "", "| artifact | path | sha256 |", "|---|---|---|"])
    for name, meta in sorted(payload["primary_artifacts"].items()):
        lines.append(f"| `{name}` | `{meta['path']}` | `{short_hash(meta)}` |")

    lines.extend(["", "## Caveat Paragraph", "", payload["caveat_paragraph"], ""])
    lines.extend(["## Allowed Claims", ""])
    lines.extend(f"- {item}" for item in payload["allowed_claims"])
    lines.extend(["", "## Disallowed Claims", ""])
    lines.extend(f"- {item}" for item in payload["disallowed_claims"])
    lines.extend(["", "## Closed / Do Not Relaunch", ""])
    lines.extend(f"- {item}" for item in payload["closed_no_relaunch"])
    lines.extend(["", "## Checks", "", "| check | passed | evidence |", "|---|---:|---|"])
    for row in payload["checks"]:
        evidence = row["evidence"]
        if isinstance(evidence, dict):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(f"| `{row['name']}` | `{row['passed']}` | `{evidence}` |")
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
