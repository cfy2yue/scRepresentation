#!/usr/bin/env python3
"""Build manuscript-style summary tables for the frozen Track C blend.

The script reads already-frozen reporting JSONs only.  It does not run models,
read query for tuning, or authorize any selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CI_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_blend_reporting_ci_20260623.json"
DEFAULT_FAILURE_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_blend_failure_cases_20260623.json"
DEFAULT_AUDIT_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_blend_frozen_package_audit_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_MANUSCRIPT_TABLE_20260623.md"
DEFAULT_OUT_CSV = ROOT / "reports/latentfm_trackc_anchor_gated_blend_manuscript_table_20260623.csv"

ROWS = (
    ("support selection", "support_val_multi", "selection support; safe trainselect only"),
    ("canonical no-harm single", "canonical_test_single", "canonical Track A no-harm; gate=0 no-op"),
    ("canonical no-harm family", "canonical_family_gene", "canonical Track A no-harm; gate=0 no-op"),
    ("held-out query all", "query_all", "final diagnostic; not selection"),
    ("held-out query seen", "query_seen", "final diagnostic stratum; not selection"),
    ("held-out query unseen1", "query_unseen1", "final diagnostic stratum; not selection"),
    ("held-out query unseen2", "query_unseen2", "weak pp evidence; no MMD hard-harm"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def ci_text(metric: dict[str, Any]) -> str:
    return f"[{fmt(metric.get('ci_low'))}, {fmt(metric.get('ci_high'))}]"


def csv_rows(ci_payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for block, key, interpretation in ROWS:
        group = ci_payload["groups"][key]
        pp = group["pearson_pert_delta"]
        mmd = group["mmd_clamped_delta"]
        out.append(
            {
                "block": block,
                "group_key": key,
                "rows": group["n_rows"],
                "datasets": group["n_datasets"],
                "pp_delta": pp.get("observed"),
                "pp_ci_low": pp.get("ci_low"),
                "pp_ci_high": pp.get("ci_high"),
                "pp_p_positive": pp.get("p_positive"),
                "pp_p_harm_lt_minus_0p02": pp.get("p_harm_pp"),
                "mmd_delta": mmd.get("observed"),
                "mmd_ci_low": mmd.get("ci_low"),
                "mmd_ci_high": mmd.get("ci_high"),
                "mmd_p_harm_gt_0p005": mmd.get("p_harm_mmd"),
                "interpretation": interpretation,
            }
        )
    return out


def worst_row(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("worst_pp_rows") or []
    return rows[0] if rows else {}


def render_md(ci_payload: dict[str, Any], failure_payload: dict[str, Any], audit_payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    worst = worst_row(failure_payload)
    status = audit_payload.get("status")
    reasons = audit_payload.get("reasons") or []
    lines = [
        "# Track C Anchor-Gated Blend Manuscript Summary Table",
        "",
        "Status: `manuscript_table_ready_for_frozen_diagnostic`",
        "",
        "## Boundary",
        "",
        "This table summarizes frozen evidence only.  It must not be used to tune alpha, gate, threshold, checkpoint, route, or future branches.",
        "",
        "## Table",
        "",
        "| evidence block | rows | datasets | pp delta | pp 95% CI | pp p_positive | pp p_harm<-0.02 | MMD delta | MMD 95% CI | MMD p_harm>0.005 | interpretation |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['block']} | {row['rows']} | {row['datasets']} | {fmt(row['pp_delta'])} | "
            f"[{fmt(row['pp_ci_low'])}, {fmt(row['pp_ci_high'])}] | {fmt(row['pp_p_positive'])} | "
            f"{fmt(row['pp_p_harm_lt_minus_0p02'])} | {fmt(row['mmd_delta'])} | "
            f"[{fmt(row['mmd_ci_low'])}, {fmt(row['mmd_ci_high'])}] | "
            f"{fmt(row['mmd_p_harm_gt_0p005'])} | {row['interpretation']} |"
        )
    lines += [
        "",
        "## Claim Text",
        "",
        "Allowed wording: the frozen anchor-gated support-teacher blend is supported as a diagnostic/calibrator: support-val, held-out aggregate, seen, and unseen1 evidence are positive, and canonical Track A no-harm is preserved.",
        "",
        "Required limitation: unseen2 pearson_pert evidence is weak because its CI crosses zero, although MMD hard-harm is not observed.",
        "",
        "Disallowed wording: formal multi capability is fully solved, unseen2 pp generalization is strong, or the support-teacher checkpoint alone passed no-harm.",
        "",
        "## Failure Case To Report",
        "",
        f"* worst pp row: `{worst.get('dataset')}` / `{worst.get('condition')}`; stratum `{worst.get('stratum')}`; pp delta `{fmt(worst.get('pp_delta'))}`; MMD delta `{fmt(worst.get('mmd_delta'))}`; train-single seen genes `{worst.get('seen_gene_count_in_train_single')}/{worst.get('n_genes')}`.",
        "* Wessels unseen2 pp negative fraction: `0.631579`; MMD harm fraction: `0`.",
        "",
        "## Package Audit",
        "",
        f"* frozen package audit status: `{status}`",
        f"* audit reasons: `{reasons if reasons else 'none'}`",
        "",
        "## Outputs",
        "",
        f"* CSV: `{DEFAULT_OUT_CSV}`",
        f"* CI JSON: `{DEFAULT_CI_JSON}`",
        f"* failure JSON: `{DEFAULT_FAILURE_JSON}`",
        f"* frozen package audit JSON: `{DEFAULT_AUDIT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-json", type=Path, default=DEFAULT_CI_JSON)
    parser.add_argument("--failure-json", type=Path, default=DEFAULT_FAILURE_JSON)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    ci_payload = load_json(args.ci_json)
    failure_payload = load_json(args.failure_json)
    audit_payload = load_json(args.audit_json)
    rows = csv_rows(ci_payload)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    args.out_md.write_text(render_md(ci_payload, failure_payload, audit_payload, rows), encoding="utf-8")
    print(json.dumps({"status": "manuscript_table_ready_for_frozen_diagnostic", "out_md": str(args.out_md), "out_csv": str(args.out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
