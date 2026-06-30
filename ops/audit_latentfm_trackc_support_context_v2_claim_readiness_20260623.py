#!/usr/bin/env python3
"""Build a claim-readiness audit for the frozen Track C support-context v2 package."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

RUN_NAME = "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"

FINAL_AUDIT_JSON = REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json"
QUERY_JSON = (
    REPORTS
    / "latentfm_trackc_support_context_v2_query_once_decision_"
    "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42_20260623.json"
)
UNCAPPED_JSON = (
    REPORTS
    / "latentfm_trackc_support_context_v2_uncapped_noharm_"
    "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42_20260623_decision.json"
)
RESIDUAL_UNCAPPED_JSON = (
    REPORTS
    / "latentfm_trackc_support_context_v2_uncapped_noharm_"
    "xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42_20260623_decision.json"
)
FAILURE_JSON = REPORTS / "latentfm_trackc_support_context_v2_query_failure_cases_20260623.json"
SYNTHESIS_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_DIAGNOSTIC_SYNTHESIS_20260623.md"
DECISIONS_MD = ROOT / "docs" / "DECISIONS.md"

OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_CLAIM_READINESS_AUDIT_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def ci(row: dict[str, Any]) -> str:
    return f"[{fmt(row['ci95_low'])},{fmt(row['ci95_high'])}]"


def metric(decision: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    key = f"{group}:{name}"
    return decision["tables"]["split"][key]


def noharm_metric(decision: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    key = f"{group}:{name}"
    for table_name in ("split", "family"):
        table = decision["tables"].get(table_name, {})
        if key in table:
            return table[key]
    raise KeyError(key)


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main() -> None:
    final_audit = load_json(FINAL_AUDIT_JSON)
    query = load_json(QUERY_JSON)
    uncapped = load_json(UNCAPPED_JSON)
    residual_uncapped = load_json(RESIDUAL_UNCAPPED_JSON)
    failure = load_json(FAILURE_JSON)
    synthesis_text = SYNTHESIS_MD.read_text()
    decisions_text = DECISIONS_MD.read_text()

    q_pp = metric(query, "query_multi", "pearson_pert")
    q_mmd = metric(query, "query_multi", "test_mmd_clamped")
    seen_pp = metric(query, "query_multi_seen", "pearson_pert")
    unseen1_pp = metric(query, "query_multi_unseen1", "pearson_pert")
    unseen2_pp = metric(query, "query_multi_unseen2", "pearson_pert")
    unseen2_mmd = metric(query, "query_multi_unseen2", "test_mmd_clamped")

    single_pp = noharm_metric(uncapped, "test_single", "pearson_pert")
    family_pp = noharm_metric(uncapped, "family_gene", "pearson_pert")
    single_mmd = noharm_metric(uncapped, "test_single", "test_mmd_clamped")
    family_mmd = noharm_metric(uncapped, "family_gene", "test_mmd_clamped")

    residual_single_pp = noharm_metric(residual_uncapped, "test_single", "pearson_pert")
    residual_family_pp = noharm_metric(residual_uncapped, "family_gene", "pearson_pert")

    worst = failure["worst_pp_rows"][0]
    recurrent_genes = failure["recurrent_gene_signals"][:8]

    checks = [
        check(
            "final_package_audit_passed",
            final_audit["status"] == "trackc_support_context_v2_final_package_audit_pass"
            and len(final_audit.get("failed_checks", [])) == 0,
            {"status": final_audit["status"], "failed_checks": final_audit.get("failed_checks", [])},
        ),
        check(
            "one_shot_query_supported_after_freeze",
            query["decision"]["status"] == "trackc_query_diagnostic_candidate_supported",
            query["decision"],
        ),
        check(
            "aggregate_query_pp_positive_ci",
            q_pp["delta_mean"] > 0 and q_pp["ci95_low"] > 0 and q_pp["p_harm"] <= 0.05,
            q_pp,
        ),
        check(
            "aggregate_query_mmd_improves",
            q_mmd["delta_mean"] < 0 and q_mmd["ci95_high"] < 0 and q_mmd["p_harm"] <= 0.05,
            q_mmd,
        ),
        check(
            "seen_and_unseen1_pp_supported",
            seen_pp["ci95_low"] > 0 and unseen1_pp["ci95_low"] > 0,
            {"seen": seen_pp, "unseen1": unseen1_pp},
        ),
        check(
            "unseen2_pp_recorded_as_caveat",
            unseen2_pp["ci95_low"] < 0 and unseen2_pp["p_harm"] > 0.20,
            unseen2_pp,
        ),
        check(
            "unseen2_mmd_no_hard_harm",
            unseen2_mmd["delta_mean"] < 0 and unseen2_mmd["ci95_high"] < 0 and unseen2_mmd["p_harm"] <= 0.05,
            unseen2_mmd,
        ),
        check(
            "canonical_uncapped_noharm_exact_noop",
            all(
                row["delta_mean"] == 0.0 and row["p_harm"] == 0.0
                for row in [single_pp, family_pp, single_mmd, family_mmd]
            ),
            {"test_single_pp": single_pp, "family_pp": family_pp, "test_single_mmd": single_mmd, "family_mmd": family_mmd},
        ),
        check(
            "residual_uncapped_used_only_as_robustness",
            residual_single_pp["delta_mean"] == 0.0
            and residual_family_pp["delta_mean"] == 0.0
            and "Do Not Use Residual V2 Uncapped Result To Trigger A Second Query" in decisions_text,
            {"residual_test_single_pp": residual_single_pp, "residual_family_pp": residual_family_pp},
        ),
        check(
            "failure_cases_reported",
            failure["status"] == "trackc_support_context_v2_query_failure_cases_ready" and failure["n_rows"] == 174,
            {"status": failure["status"], "n_rows": failure["n_rows"], "worst_pp_row": worst},
        ),
        check(
            "claim_boundaries_present",
            "Do not claim a blanket formal multi-perturbation solution" in synthesis_text
            and "Do not tune or launch residual query from this result" in synthesis_text
            and "Do not use canonical multi as Track A selection evidence" in synthesis_text,
            {"synthesis_md": str(SYNTHESIS_MD)},
        ),
    ]

    failed = [row for row in checks if not row["passed"]]
    status = (
        "claim_ready_as_frozen_support_context_v2_diagnostic_not_formal_multi_solution"
        if not failed
        else "claim_not_ready_requires_manual_review"
    )

    allowed_claims = [
        "Frozen Track C support-context v2 diagnostic route is supported after support-val, uncapped canonical no-harm, and query-free freeze gates.",
        "Held-out query aggregate Pearson and clamped MMD improve for the frozen route.",
        "Seen and unseen1 query Pearson strata improve; unseen2 is MMD-safe but Pearson-weak.",
        "Canonical Track A support-absent evaluations are exact no-ops for test_single and family_gene.",
        "Residual v2 uncapped no-harm is robustness evidence only, not a second query candidate.",
    ]
    disallowed_claims = [
        "Do not claim formal multi-perturbation capability is solved in general.",
        "Do not claim strong unseen2 Pearson generalization.",
        "Do not imply query results selected the route, checkpoint, thresholds, or features.",
        "Do not launch or justify a residual query from this package.",
        "Do not use canonical multi diagnostics as Track A selection evidence.",
    ]

    audit = {
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "run_name": RUN_NAME,
        "inputs": {
            "final_audit_json": str(FINAL_AUDIT_JSON),
            "query_json": str(QUERY_JSON),
            "uncapped_json": str(UNCAPPED_JSON),
            "residual_uncapped_json": str(RESIDUAL_UNCAPPED_JSON),
            "failure_json": str(FAILURE_JSON),
            "synthesis_md": str(SYNTHESIS_MD),
            "decisions_md": str(DECISIONS_MD),
        },
        "checks": checks,
        "failed_checks": failed,
        "metrics": {
            "query_multi_pearson_delta": q_pp["delta_mean"],
            "query_multi_pearson_ci": [q_pp["ci95_low"], q_pp["ci95_high"]],
            "query_multi_mmd_delta": q_mmd["delta_mean"],
            "seen_pearson_delta": seen_pp["delta_mean"],
            "unseen1_pearson_delta": unseen1_pp["delta_mean"],
            "unseen2_pearson_delta": unseen2_pp["delta_mean"],
            "unseen2_pearson_p_harm": unseen2_pp["p_harm"],
            "unseen2_mmd_delta": unseen2_mmd["delta_mean"],
            "worst_pp_condition": worst["condition"],
            "worst_pp_dataset": worst["dataset"],
            "worst_pp_delta": worst["pp_delta"],
            "worst_mmd_delta": worst["mmd_delta"],
        },
        "allowed_claims": allowed_claims,
        "disallowed_claims": disallowed_claims,
        "recommended_next_gate": "reporting/provenance only; new GPU requires a fresh query-free CPU gate, not this held-out query result",
    }

    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lines: list[str] = [
        "# Track C Support-Context V2 Claim Readiness Audit",
        "",
        f"Timestamp: `{audit['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "This audit reviews the frozen Track C support-context v2 package for manuscript/reporting claims.",
        "It is not a new model-selection step and does not authorize route, checkpoint, threshold, feature, alpha, or residual-query changes.",
        "",
        "## Evidence Passed",
        "",
        "| requirement | evidence | status |",
        "|---|---|---|",
    ]
    evidence_rows = [
        (
            "package audit",
            f"{final_audit['status']}; failed checks `{len(final_audit.get('failed_checks', []))}`",
        ),
        (
            "one-shot query after freeze",
            f"{query['decision']['status']}; action `{query['decision']['action']}`",
        ),
        (
            "aggregate query Pearson",
            f"delta `{fmt(q_pp['delta_mean'])}`, CI `{ci(q_pp)}`, p_harm `{q_pp['p_harm']}`",
        ),
        (
            "aggregate query MMD",
            f"delta `{fmt(q_mmd['delta_mean'])}`, CI `{ci(q_mmd)}`, p_harm `{q_mmd['p_harm']}`",
        ),
        (
            "seen/unseen1 Pearson",
            f"seen `{fmt(seen_pp['delta_mean'])}`; unseen1 `{fmt(unseen1_pp['delta_mean'])}`",
        ),
        (
            "canonical Track A no-harm",
            "uncapped test_single/family_gene Pearson and MMD deltas exact `0`",
        ),
        (
            "residual robustness boundary",
            "residual uncapped no-harm exact no-op; second residual query forbidden by DECISIONS.md",
        ),
        (
            "failure cases reported",
            f"n `{failure['n_rows']}`; worst `{worst['dataset']}/{worst['condition']}` pp delta `{fmt(worst['pp_delta'])}`",
        ),
    ]
    for requirement, evidence in evidence_rows:
        lines.append(f"| {requirement} | {evidence} | pass |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "| limitation | evidence | consequence |",
            "|---|---|---|",
            (
                f"| unseen2 Pearson is weak | delta `{fmt(unseen2_pp['delta_mean'])}`, "
                f"CI `{ci(unseen2_pp)}`, p_harm `{unseen2_pp['p_harm']}` | "
                "cannot claim strong unseen2 Pearson generalization |"
            ),
            (
                f"| worst condition-level failure remains | `{worst['dataset']}/{worst['condition']}` "
                f"pp delta `{fmt(worst['pp_delta'])}`, MMD delta `{fmt(worst['mmd_delta'])}` | "
                "must report failure cases, not just aggregate gains |"
            ),
            (
                "| recurrent weak genes remain | "
                + ", ".join(f"`{row['gene']}`" for row in recurrent_genes[:6])
                + " | motivates future biology/error analysis only |"
            ),
            "| residual variant is not a query candidate | residual uncapped no-harm passed after resfilm query path was fixed | no second v2-family query without a new query-blind protocol |",
            "| query artifact is consumed | package audit rules forbid future selection from query | no query-tuned rescue or threshold sweep |",
            "",
            "## Allowed Claim",
            "",
        ]
    )
    for claim in allowed_claims:
        lines.append(f"- {claim}")
    lines.extend(["", "## Disallowed Claims", ""])
    for claim in disallowed_claims:
        lines.append(f"- {claim}")
    lines.extend(
        [
            "",
            "## Recommended Next Gate",
            "",
            "Do not run another held-out query or GPU job from this evidence. Continue reporting/provenance/figure preparation. New modeling GPU work requires a materially new query-free CPU gate with a hypothesis, promotion gate, stop rule, launcher, RUN_STATUS, and fresh resource audit.",
            "",
            "## Machine Checks",
            "",
            "| check | passed | evidence |",
            "|---|---:|---|",
        ]
    )
    for row in checks:
        evidence = row["evidence"]
        if isinstance(evidence, dict):
            evidence_text = json.dumps(evidence, sort_keys=True)[:240]
        else:
            evidence_text = str(evidence)[:240]
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {evidence_text} |")

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print(f"status {status}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
