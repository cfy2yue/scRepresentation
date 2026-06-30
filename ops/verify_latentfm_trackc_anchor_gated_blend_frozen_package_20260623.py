#!/usr/bin/env python3
"""Verify the frozen Track C anchor-gated blend evidence package.

This is a read-only robustness audit over frozen artifacts.  It does not run
models, read query for tuning, or authorize any new experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_blend_frozen_package_audit_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FROZEN_PACKAGE_AUDIT_20260623.md"

EXPECTED_HASHES = {
    "safe_trainselect_split": (
        ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json",
        "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20",
    ),
    "canonical_split": (
        ROOT / "dataset/biFlow_data/split_seed42.json",
        "bb82961387bc24d29f5821e52d26b958e101977652e811d87f0c411d00679054",
    ),
    "anchor_checkpoint": (
        ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620"
        / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt",
        "dc7ed20a9535256709955f7994f7894d886011813f23caf3fca630fd2c4f1c8b",
    ),
    "support_teacher_checkpoint": (
        ROOT
        / "CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623"
        / "xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt",
        "7bd9fceca75cbf8009dc64bedbe9b7d625c29003b49274b37955fa1d5c023657",
    ),
    "posthoc_decision_json": (
        ROOT / "reports/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json",
        "ebd71daca5f3fa53b1cf651d9fdbc4992b2bdb6de2a54ccfd63ffc6585bdec9b",
    ),
    "query_raw_json": (
        ROOT
        / "runs/latentfm_trackc_anchor_gated_blend_query_once_20260623_retry1"
        / "eval/anchor_gated_blend_query_once_ode20.json",
        "0c927457b022663fe00f2e4449e7d54e994751402f9f22ca0393bcbee793ffff",
    ),
    "query_decision_json": (
        ROOT / "reports/latentfm_trackc_anchor_gated_blend_query_once_decision_20260623.json",
        "d6b3d04f4e1c5cae6b01cfcb53ce43b9ec786d53311789243a40f5f2867df689",
    ),
    "blend_evaluator_script": (
        ROOT / "ops/evaluate_latentfm_trackc_anchor_gated_support_teacher_blend_20260623.py",
        "8c9cec63050b9fe4e60aaccb7c2fae800c752b52f9bbb763329e2e30a05dc2d9",
    ),
    "query_summarizer_script": (
        ROOT / "ops/summarize_latentfm_trackc_anchor_gated_blend_query_once_20260623.py",
        "da8636a66f459f0c887a2df795a4c8e48c41347978ff69c7395440d233d1e4b1",
    ),
}

REPORTS = {
    "route_freeze": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_ROUTE_FREEZE_20260623.md",
    "posthoc_gate": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_BLEND_POSTHOC_GATE_20260623.md",
    "query_decision": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_QUERY_ONCE_DECISION_20260623.md",
    "reporting_ci": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_REPORTING_CI_20260623.md",
    "failure_cases": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FAILURE_CASES_20260623.md",
    "claim_readiness": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_CLAIM_READINESS_AUDIT_20260623.md",
    "synthesis": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FINAL_DIAGNOSTIC_SYNTHESIS_20260623.md",
    "provenance": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_ARTIFACT_PROVENANCE_20260623.md",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_hashes() -> list[dict[str, Any]]:
    rows = []
    for name, (path, expected) in EXPECTED_HASHES.items():
        exists = path.exists()
        observed = sha256(path) if exists else None
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": exists,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "pass" if exists and observed == expected else "fail",
            }
        )
    return rows


def check_reports() -> list[dict[str, Any]]:
    rows = []
    for name, path in REPORTS.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append({"report": name, "path": str(path), "exists": exists, "size": size, "status": "pass" if size > 0 else "fail"})
    return rows


def check_payload_consistency() -> list[str]:
    reasons: list[str] = []
    posthoc = load_json(EXPECTED_HASHES["posthoc_decision_json"][0])
    query_raw = load_json(EXPECTED_HASHES["query_raw_json"][0])
    query_decision = load_json(EXPECTED_HASHES["query_decision_json"][0])
    reporting_ci = load_json(ROOT / "reports/latentfm_trackc_anchor_gated_blend_reporting_ci_20260623.json")
    failure = load_json(ROOT / "reports/latentfm_trackc_anchor_gated_blend_failure_cases_20260623.json")

    if posthoc.get("status") != "trackc_anchor_gated_support_teacher_blend_posthoc_gate_pass":
        reasons.append(f"posthoc_status_unexpected:{posthoc.get('status')}")
    support = posthoc.get("support") or {}
    if int(support.get("n_rows") or 0) != 24:
        reasons.append("support_n_rows_not_24")
    canonical = posthoc.get("canonical_noop") or {}
    if int(canonical.get("test_single_n_rows") or 0) != 540:
        reasons.append("canonical_test_single_n_rows_not_540")
    if int(canonical.get("family_gene_n_rows") or 0) != 697:
        reasons.append("canonical_family_gene_n_rows_not_697")
    for key, obj in canonical.items():
        if isinstance(obj, dict):
            for metric, value in obj.items():
                if float(value) != 0.0:
                    reasons.append(f"canonical_noop_nonzero:{key}:{metric}:{value}")

    if query_raw.get("scope") != "heldout_query_once":
        reasons.append(f"query_raw_scope_unexpected:{query_raw.get('scope')}")
    if abs(float(query_raw.get("alpha") or -999.0) - 0.75) > 1e-12:
        reasons.append(f"query_alpha_unexpected:{query_raw.get('alpha')}")
    safety = query_raw.get("safety") or {}
    if safety.get("heldout_query_read") is not True:
        reasons.append("query_raw_does_not_mark_heldout_query_read")
    if safety.get("query_result_may_select_or_tune") is not False:
        reasons.append("query_raw_allows_selection_or_tuning")
    if safety.get("canonical_multi_selection") is not False:
        reasons.append("query_raw_marks_canonical_multi_selection")

    if query_decision.get("status") != "trackc_anchor_gated_blend_query_diagnostic_candidate_supported":
        reasons.append(f"query_decision_status_unexpected:{query_decision.get('status')}")
    if query_decision.get("query_json") != str(EXPECTED_HASHES["query_raw_json"][0]):
        reasons.append("query_decision_points_to_unexpected_raw_json")
    if query_decision.get("posthoc_gate_json") != str(EXPECTED_HASHES["posthoc_decision_json"][0]):
        reasons.append("query_decision_points_to_unexpected_posthoc_json")

    expected_counts = {
        "heldout_query_multi_final_only": 174,
        "heldout_query_multi_seen_final_only": 57,
        "heldout_query_multi_unseen1_final_only": 68,
        "heldout_query_multi_unseen2_final_only": 49,
    }
    for group, n in expected_counts.items():
        rows = ((query_raw.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
        if len(rows) != n:
            reasons.append(f"query_raw_count_mismatch:{group}:{len(rows)}")

    if reporting_ci.get("boundary_reasons"):
        reasons.append(f"reporting_ci_boundary_reasons:{reporting_ci.get('boundary_reasons')}")
    if failure.get("n_rows") != 174:
        reasons.append(f"failure_case_n_rows_unexpected:{failure.get('n_rows')}")
    if failure.get("stratum_counts") != {"seen": 57, "unseen1": 68, "unseen2": 49}:
        reasons.append(f"failure_case_stratum_counts_unexpected:{failure.get('stratum_counts')}")
    return reasons


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Anchor-Gated Blend Frozen Package Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "Read-only consistency audit over frozen artifacts.  This does not authorize another query read, alpha sweep, checkpoint selection, or GPU run.",
        "",
        "## Hash Checks",
        "",
        "| artifact | status | expected SHA256 | observed SHA256 | path |",
        "|---|---|---|---|---|",
    ]
    for row in payload["hash_checks"]:
        lines.append(
            f"| {row['artifact']} | {row['status']} | `{row['expected_sha256']}` | "
            f"`{row['observed_sha256']}` | `{row['path']}` |"
        )
    lines += [
        "",
        "## Report Checks",
        "",
        "| report | status | size | path |",
        "|---|---|---:|---|",
    ]
    for row in payload["report_checks"]:
        lines.append(f"| {row['report']} | {row['status']} | {row['size']} | `{row['path']}` |")
    lines += [
        "",
        "## Consistency Reasons",
        "",
    ]
    if payload["reasons"]:
        lines.extend(f"* `{reason}`" for reason in payload["reasons"])
    else:
        lines.append("* none")
    lines += [
        "",
        "## Manuscript-Style Evidence Blocks",
        "",
        "| block | evidence | interpretation |",
        "|---|---|---|",
        "| support selection | support-val rows `24`; pp `+0.086939`; CI `[+0.007909,+0.177589]`; MMD `-0.010173` | selection support passed on safe trainselect only |",
        "| canonical no-harm | `test_single` `540` rows and `family_gene` `697` rows exact no-op deltas | Track A canonical no-harm preserved by gate=0 |",
        "| held-out diagnostic | query_all pp `+0.052359`; CI `[+0.030885,+0.073106]`; MMD hard-harm `0` | aggregate final diagnostic supported |",
        "| limitation | query_unseen2 pp `+0.003353`; CI `[-0.018534,+0.026311]`; Wessels unseen2 pp negative fraction `0.631579` | unseen2 pp must be worded conservatively |",
        "| failure cases | worst row `NormanWeissman2019_filtered/CNN1+MAPK1`, pp `-1.035826`, MMD `+0.058909` | condition-level failures remain and must be reported |",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    hash_checks = check_hashes()
    report_checks = check_reports()
    reasons = check_payload_consistency()
    reasons.extend(f"hash_check_failed:{row['artifact']}" for row in hash_checks if row["status"] != "pass")
    reasons.extend(f"report_check_failed:{row['report']}" for row in report_checks if row["status"] != "pass")
    payload = {
        "status": "trackc_anchor_gated_blend_frozen_package_audit_pass" if not reasons else "trackc_anchor_gated_blend_frozen_package_audit_fail",
        "reasons": reasons,
        "hash_checks": hash_checks,
        "report_checks": report_checks,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
