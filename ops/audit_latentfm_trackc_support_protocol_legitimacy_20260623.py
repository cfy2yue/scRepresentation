#!/usr/bin/env python3
"""Audit whether the frozen Track C blend can be framed as a support protocol.

This is a reporting/protocol audit only. It does not run models, read held-out
query for tuning, or authorize GPU. The goal is to separate two claims:

1. condition-only deployable residual gates are closed by collision evidence;
2. an explicitly support-context-conditional task definition can be reported as
   a frozen diagnostic/calibrator if the support context is part of the task
   interface and canonical Track A evaluation is defined as support-absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

FROZEN_PACKAGE = REPORTS / "latentfm_frozen_diagnostic_reporting_package_20260623.json"
POSTHOC_GATE = REPORTS / "latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json"
LEARNED_GATE = REPORTS / "latentfm_trackc_learned_anchor_gate_cpu_gate_20260623.json"
BIO_GATE = REPORTS / "latentfm_trackc_biological_prior_separability_20260623.json"
PROTOCOL = REPORTS / "latentfm_trackc_anchor_gated_support_teacher_protocol_20260623.json"
OUT_JSON = REPORTS / "latentfm_trackc_support_protocol_legitimacy_audit_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_PROTOCOL_LEGITIMACY_AUDIT_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status(path: Path, payload: dict[str, Any]) -> str:
    if "status" in payload:
        return str(payload["status"])
    decision = payload.get("decision")
    if isinstance(decision, dict) and "status" in decision:
        return str(decision["status"])
    return "unknown"


def posthoc_support_metrics(posthoc: dict[str, Any]) -> dict[str, Any]:
    support = posthoc.get("support", {})
    pp = support.get("pearson_pert_delta", {})
    mmd = support.get("test_mmd_delta", {})
    return {
        "n_rows": support.get("n_rows"),
        "pearson_pert_delta": pp.get("observed"),
        "pearson_pert_ci": [pp.get("ci_low"), pp.get("ci_high")],
        "pearson_pert_p_harm": pp.get("p_harm_pp"),
        "mmd_delta": mmd.get("observed"),
        "mmd_ci": [mmd.get("ci_low"), mmd.get("ci_high")],
        "mmd_p_harm": mmd.get("p_harm_mmd"),
    }


def canonical_noop_metrics(posthoc: dict[str, Any]) -> dict[str, Any]:
    return posthoc.get("canonical_noop", {})


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []

    frozen = payload["inputs"]["frozen_package"]
    posthoc = payload["inputs"]["posthoc_gate"]
    learned = payload["inputs"]["learned_gate"]
    bio = payload["inputs"]["biological_prior_gate"]

    if frozen.get("boundary", {}).get("query_tuning_forbidden") is not True:
        reasons.append("frozen_package_does_not_forbid_query_tuning")
    if frozen.get("boundary", {}).get("claim_scope") != "diagnostic_calibrator_not_formal_multi_solution":
        reasons.append("frozen_package_claim_scope_not_diagnostic")
    if posthoc.get("status") != "trackc_anchor_gated_support_teacher_blend_posthoc_gate_pass":
        reasons.append("posthoc_gate_not_passed")
    if (learned.get("decision") or {}).get("status") != "trackc_learned_anchor_gate_cpu_gate_fail_no_gpu":
        warnings.append("condition_metadata_gate_not_recorded_as_closed")
    if (bio.get("decision") or {}).get("status") != "trackc_biological_prior_separability_fail_no_gpu":
        warnings.append("biological_condition_gate_not_recorded_as_closed")

    collision = bio.get("collision", {})
    if collision.get("max_support_rows_under_exact_family_noop") != 0:
        warnings.append("condition_only_collision_not_structural")

    protocol_legitimate = not reasons
    return {
        "status": "trackc_support_protocol_legitimacy_ready_for_reporting_no_gpu"
        if protocol_legitimate
        else "trackc_support_protocol_legitimacy_fail",
        "gpu_authorization": "none",
        "next_authorization": "reporting_or_protocol_design_only" if protocol_legitimate else "none",
        "reasons": reasons,
        "warnings": warnings,
    }


def render(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    support = payload["evidence"]["support_metrics"]
    canon = payload["evidence"]["canonical_noop"]
    collision = payload["evidence"]["condition_only_collision"]
    lines = [
        "# Track C Support-Protocol Legitimacy Audit",
        "",
        f"Status: `{d['status']}`",
        f"GPU authorization: `{d['gpu_authorization']}`",
        "",
        "## Question",
        "",
        "Can the frozen anchor-gated support-teacher blend be reported as a support-context-conditional Track C diagnostic, rather than as a deployable condition-only residual gate?",
        "",
        "## Decision",
        "",
        "Yes, but only under a strict protocol definition: support context must be an explicit input to the task interface, and canonical Track A evaluation must be defined as support-context absent. This is a reporting/protocol boundary, not a GPU or formal-success authorization.",
        "",
        "## Evidence",
        "",
        f"- frozen package status: `{payload['source_status']['frozen_package']}`",
        f"- posthoc gate status: `{payload['source_status']['posthoc_gate']}`",
        f"- condition/train metadata gate status: `{payload['source_status']['learned_gate']}`",
        f"- biological condition-feature gate status: `{payload['source_status']['biological_prior_gate']}`",
        f"- support rows: `{support.get('n_rows')}`; pp delta `{fmt(support.get('pearson_pert_delta'))}`; pp p_harm `{fmt(support.get('pearson_pert_p_harm'))}`; MMD delta `{fmt(support.get('mmd_delta'))}`; MMD p_harm `{fmt(support.get('mmd_p_harm'))}`",
        f"- canonical no-op groups: `{sorted(canon.keys()) if isinstance(canon, dict) else canon}`",
        f"- condition-only collision: support rows `{collision.get('support_rows')}`, exact family matches `{collision.get('exact_family_matches')}`, max support rows under exact family no-op `{collision.get('max_support_rows_under_exact_family_noop')}`",
        "",
        "## Allowed Wording",
        "",
        "- The frozen blend is a support-context-conditional diagnostic/calibrator.",
        "- In support-present Track C evaluation, the frozen residual improves aggregate support/query diagnostic metrics.",
        "- In support-absent canonical Track A evaluation, the frozen protocol is exact no-op relative to the anchor.",
        "",
        "## Disallowed Wording",
        "",
        "- Do not call this a deployable condition-only gate.",
        "- Do not claim formal multi capability is solved.",
        "- Do not claim unseen2 pearson_pert generalization is strong.",
        "- Do not use the consumed held-out query to tune alpha, gate, checkpoint, threshold, or branch choice.",
        "- Do not relaunch GO/scGPT/CellNavi/condition-metadata threshold GPU smokes under this protocol label.",
        "",
        "## Minimal Next Work",
        "",
        "- For reporting: use the frozen diagnostic package and this audit as claim-boundary evidence.",
        "- For new experiments: design a model whose interface explicitly consumes support context and whose route is frozen before any query; selection must remain support-val only.",
        "- Any future GPU branch still needs its own CPU gate, launcher/RUN_STATUS, resource audit, and fail-close rules.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- `{r}`" for r in d["reasons"]] if d["reasons"] else ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- `{w}`" for w in d["warnings"]] if d["warnings"] else ["- none"])
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "frozen_package": load_json(FROZEN_PACKAGE),
        "posthoc_gate": load_json(POSTHOC_GATE),
        "learned_gate": load_json(LEARNED_GATE),
        "biological_prior_gate": load_json(BIO_GATE),
        "support_teacher_protocol": load_json(PROTOCOL),
    }
    payload = {
        "inputs": inputs,
        "source_paths": {
            "frozen_package": str(FROZEN_PACKAGE),
            "posthoc_gate": str(POSTHOC_GATE),
            "learned_gate": str(LEARNED_GATE),
            "biological_prior_gate": str(BIO_GATE),
            "support_teacher_protocol": str(PROTOCOL),
        },
        "source_status": {name: status(Path(path), inputs[name]) for name, path in {
            "frozen_package": FROZEN_PACKAGE,
            "posthoc_gate": POSTHOC_GATE,
            "learned_gate": LEARNED_GATE,
            "biological_prior_gate": BIO_GATE,
            "support_teacher_protocol": PROTOCOL,
        }.items()},
        "evidence": {
            "support_metrics": posthoc_support_metrics(inputs["posthoc_gate"]),
            "canonical_noop": canonical_noop_metrics(inputs["posthoc_gate"]),
            "condition_only_collision": inputs["biological_prior_gate"].get("collision", {}),
        },
    }
    payload["decision"] = decide(payload)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
