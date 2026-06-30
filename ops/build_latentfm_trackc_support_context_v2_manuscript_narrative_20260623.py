#!/usr/bin/env python3
"""Build manuscript-style narrative text for frozen Track C v2 reporting.

This is a reporting-only helper. It reads frozen handoff/table/figure/caveat
artifacts and writes a narrative/caption package. It does not train, evaluate,
select routes, read active logs, or authorize GPU/query reuse.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_manuscript_narrative_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_MANUSCRIPT_NARRATIVE_20260623.md"

INPUTS = {
    "final_handoff_json": REPORTS / "latentfm_trackc_support_context_v2_final_handoff_20260623.json",
    "final_handoff_md": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_HANDOFF_20260623.md",
    "figure_manifest_json": REPORTS / "latentfm_trackc_support_context_v2_figure_manifest_20260623.json",
    "manuscript_table_csv": REPORTS / "latentfm_trackc_support_context_v2_manuscript_table_20260623.csv",
    "caveat_table_csv": REPORTS / "latentfm_trackc_support_context_v2_caveat_table_20260623.csv",
    "claim_readiness_json": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "final_package_audit_json": REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json",
}

EXPECTED_STATUSES = {
    "final_handoff_json": "support_context_v2_final_handoff_ready_post_support_set_closure",
    "figure_manifest_json": "support_context_v2_figure_manifest_ready",
    "claim_readiness_json": "claim_ready_as_frozen_support_context_v2_diagnostic_not_formal_multi_solution",
    "final_package_audit_json": "trackc_support_context_v2_final_package_audit_pass",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def fmt(value: Any, signed: bool = True) -> str:
    if value in (None, ""):
        return "NA"
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:+.6f}" if signed else f"{number:.6f}"


def ci_text(row: dict[str, Any]) -> str:
    ci = row.get("ci95")
    if ci is None:
        ci = [row.get("ci95_low"), row.get("ci95_high")]
    if not isinstance(ci, list) or len(ci) != 2:
        return "[NA, NA]"
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def row_by_role(rows: list[dict[str, str]], role: str) -> dict[str, str]:
    for row in rows:
        if row.get("role") == role:
            return row
    raise KeyError(f"missing role {role}")


def metric_sentence(label: str, row: dict[str, Any]) -> str:
    n = row.get("n_conditions") or row.get("rows") or row.get("n")
    return (
        f"{label}: delta {fmt(row.get('delta'))}, 95% CI {ci_text(row)}, "
        f"p_harm {fmt(row.get('p_harm'), signed=False)}, n={n}."
    )


def short_sha(value: str | None) -> str:
    return "NA" if not value else value[:16]


def build_payload() -> dict[str, Any]:
    handoff = load_json(INPUTS["final_handoff_json"])
    figure = load_json(INPUTS["figure_manifest_json"])
    claim = load_json(INPUTS["claim_readiness_json"])
    final_audit = load_json(INPUTS["final_package_audit_json"])
    table_rows = load_csv(INPUTS["manuscript_table_csv"])
    caveat_rows = load_csv(INPUTS["caveat_table_csv"])

    artifacts = {name: artifact(path) for name, path in INPUTS.items()}
    checks: list[dict[str, Any]] = []
    for name, meta in artifacts.items():
        checks.append({"name": f"exists:{name}", "passed": meta["exists"], "evidence": meta["path"]})
    status_objects = {
        "final_handoff_json": handoff,
        "figure_manifest_json": figure,
        "claim_readiness_json": claim,
        "final_package_audit_json": final_audit,
    }
    for name, expected in EXPECTED_STATUSES.items():
        observed = status_objects[name].get("status")
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

    worst_rows = [row for row in caveat_rows if row.get("type") == "worst_pp"][:12]
    recurrent = [row for row in caveat_rows if row.get("type") == "recurrent_gene"][:8]

    status = (
        "support_context_v2_manuscript_narrative_ready"
        if not [row for row in checks if not row["passed"]]
        else "support_context_v2_manuscript_narrative_needs_review"
    )

    result_paragraph = (
        "A frozen Track C support-context v2 diagnostic route improved the "
        "safe support-val multi setting while preserving exact no-op behavior "
        "on canonical support-absent Track A evaluations. In the final one-shot "
        "held-out query diagnostic, aggregate Pearson and clamped MMD improved, "
        "with clear gains in seen and unseen1 strata. The result should be "
        "reported as a frozen diagnostic and not as evidence that general formal "
        "multi-perturbation capability is solved, because unseen2 Pearson remains "
        "weak and condition-level failures persist."
    )

    figure_captions = [
        {
            "figure": "Fig. 1",
            "title": "Frozen support-context v2 gate chain and aggregate evidence",
            "caption": (
                "The route was advanced through support-val capped evidence, "
                "uncapped canonical no-harm, query-free freeze, and a final "
                "one-shot held-out query diagnostic. "
                + metric_sentence("Support Pearson", support_pp)
                + " "
                + metric_sentence("Support MMD", support_mmd)
                + " "
                + metric_sentence("Held-out query Pearson", query_pp)
                + " "
                + metric_sentence("Held-out query MMD", query_mmd)
                + " Query rows were not used to select route, checkpoint, thresholds, or features."
            ),
        },
        {
            "figure": "Fig. 2",
            "title": "Held-out query strata and failure modes",
            "caption": (
                "Seen and unseen1 query strata showed positive Pearson deltas, "
                "whereas unseen2 Pearson was weak despite improved clamped MMD. "
                + metric_sentence("Seen Pearson", query_seen)
                + " "
                + metric_sentence("Unseen1 Pearson", query_unseen1)
                + " "
                + metric_sentence("Unseen2 Pearson", query_unseen2_pp)
                + " "
                + metric_sentence("Unseen2 MMD", query_unseen2_mmd)
                + " Worst-condition and recurrent-gene panels should remain visible next to aggregate gains."
            ),
        },
        {
            "figure": "Extended Data Fig. 1",
            "title": "Claim boundary and exact no-harm controls",
            "caption": (
                "Canonical support-absent Track A evaluations remained exact no-ops: "
                + metric_sentence("test_single Pearson", canonical_single)
                + " "
                + metric_sentence("family_gene Pearson", canonical_family)
                + " This supports no-harm for the frozen diagnostic route but does not authorize further query access or new GPU work."
            ),
        },
    ]

    table_caption = (
        "Table 1. Frozen Track C support-context v2 diagnostic metrics. Rows "
        "separate support-val evidence, canonical no-harm controls, and the "
        "final one-shot held-out query diagnostic; support-val and query rows "
        "must not be merged into a selection metric."
    )

    caveat_paragraph = (
        "The main caveats are weak unseen2 Pearson, visible condition-level harm, "
        "and repeated weakness in MAPK1, EP300, and mediator-complex genes. The "
        "worst Pearson row is "
        f"{worst_rows[0].get('dataset')}/{worst_rows[0].get('condition')} "
        f"with delta {fmt(worst_rows[0].get('pp_delta'))} and MMD delta {fmt(worst_rows[0].get('mmd_delta'))}. "
        "These failures argue for mechanism-specific follow-up rather than retuning this frozen package."
    )

    return {
        "status": status,
        "timestamp": "2026-06-23 12:48 CST",
        "boundary": {
            "reporting_only": True,
            "gpu_authorization": "none",
            "heldout_query_reuse_forbidden": True,
            "selection_or_tuning": False,
            "active_log_reads": False,
        },
        "claim_scope": handoff.get("claim_scope"),
        "result_paragraph": result_paragraph,
        "figure_captions": figure_captions,
        "table_caption": table_caption,
        "caveat_paragraph": caveat_paragraph,
        "allowed_claims": handoff.get("allowed_claims", []),
        "disallowed_claims": handoff.get("disallowed_claims", []),
        "closed_no_relaunch": handoff.get("closed_after_post_summary", []),
        "failure_preview": {"worst_pp_rows": worst_rows, "recurrent_genes": recurrent},
        "provenance": {
            "artifacts": artifacts,
            "checks": checks,
            "failed_checks": [row for row in checks if not row["passed"]],
            "split_hashes": final_audit.get("split_hashes"),
            "freeze_hashes": final_audit.get("freeze_hashes"),
            "coupledfm_commit": (final_audit.get("git") or {}).get("coupledfm_commit"),
            "figure_manifest_panels": [row.get("panel_id") for row in figure.get("panels", [])],
        },
        "next_action": (
            "Use this narrative with the frozen table and caveat CSV to draft result text and figure panels. "
            "Any new modeling still needs a materially new query-free CPU gate."
        ),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Manuscript Narrative",
        "",
        f"Status: `{payload['status']}`",
        f"Claim scope: `{payload['claim_scope']}`",
        "GPU authorization: `none`",
        "Held-out query reuse forbidden: `True`",
        "",
        "## Result Statement",
        "",
        payload["result_paragraph"],
        "",
        "## Figure Captions",
        "",
    ]
    for item in payload["figure_captions"]:
        lines.extend(
            [
                f"### {item['figure']}: {item['title']}",
                "",
                item["caption"],
                "",
            ]
        )
    lines.extend(["## Table Caption", "", payload["table_caption"], "", "## Caveat Paragraph", "", payload["caveat_paragraph"], ""])

    lines.extend(["## Allowed Claims", ""])
    lines.extend(f"- {item}" for item in payload["allowed_claims"])
    lines.extend(["", "## Disallowed Claims", ""])
    lines.extend(f"- {item}" for item in payload["disallowed_claims"])
    lines.extend(["", "## Closed / Do Not Relaunch", ""])
    lines.extend(f"- {item}" for item in payload["closed_no_relaunch"])

    lines.extend(["", "## Provenance Checks", "", "| check | passed | evidence |", "|---|---:|---|"])
    for row in payload["provenance"]["checks"]:
        evidence = row["evidence"]
        if isinstance(evidence, dict):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(f"| `{row['name']}` | `{row['passed']}` | `{evidence}` |")

    lines.extend(["", "## Artifact Hashes", "", "| artifact | exists | sha256 | path |", "|---|---:|---|---|"])
    for name, meta in sorted(payload["provenance"]["artifacts"].items()):
        lines.append(
            f"| `{name}` | `{meta['exists']}` | `{short_sha(meta.get('sha256'))}` | `{meta['path']}` |"
        )
    lines.extend(["", "## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(
        json.dumps(
            {"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
