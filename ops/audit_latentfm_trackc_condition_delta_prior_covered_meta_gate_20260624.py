#!/usr/bin/env python3
"""Track C condition-delta / prior-covered multi meta gate.

This is a CPU-only evidence aggregator.  It does not read held-out query,
canonical test, canonical multi, active logs, or GPU artifacts.  The goal is to
decide whether the existing condition-prior / prior-covered multi evidence is
strong enough to authorize a new Track C GPU smoke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_condition_delta_prior_covered_meta_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_CONDITION_DELTA_PRIOR_COVERED_META_GATE_20260624.md"


INPUTS = {
    "additive_prior": REPORTS / "latentfm_trackc_composition_additive_prior_gate_20260623.json",
    "hybrid_prior": REPORTS / "latentfm_trackc_composition_hybrid_prior_gate_20260623.json",
    "noharm_calibrated": REPORTS / "latentfm_trackc_composition_noharm_calibrated_gate_20260623.json",
    "gene_risk": REPORTS / "latentfm_trackc_composition_gene_risk_gate_20260623.json",
    "module_coherence": REPORTS / "latentfm_trackc_composition_module_coherence_gate_20260623.json",
    "dataset_conditioned": REPORTS / "latentfm_trackc_dataset_conditioned_noharm_gate_20260624.json",
    "corum_complex": REPORTS / "latentfm_trackc_corum_complex_module_gate_20260624.json",
    "support_set_summary": REPORTS / "latentfm_trackc_support_set_task_summary_gate_20260623.json",
    "support_set_after_summary": REPORTS / "latentfm_trackc_support_set_task_after_summary_decision_20260623.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    num = fnum(value)
    if num is None:
        return "NA"
    return f"{num:+.6f}"


def support_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def row_for(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") == "missing" and "path" in payload:
        decision = {}
    else:
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    status = payload.get("status") or decision.get("status") or "missing"
    reasons = (
        payload.get("reasons")
        or payload.get("decision_reasons")
        or decision.get("reasons")
        or decision.get("decision_reasons")
        or []
    )
    if isinstance(reasons, str):
        reasons = [reasons]
    support = payload.get("support_val_summary") or {}
    if not support and "support_summary" in payload:
        support = payload["support_summary"]
    norman = support_dataset(support, "NormanWeissman2019_filtered")
    wessels = support_dataset(support, "Wessels")

    if name == "dataset_conditioned":
        support = {
            "paired_pp_delta": payload.get("support_val_gate", {}).get("paired_pp_delta")
            or payload.get("support_val_summary", {}).get("paired_pp_delta")
        }

    pp_delta = fnum(
        get_path(support, "paired_pp_delta", "delta_mean")
        if isinstance(support.get("paired_pp_delta"), dict)
        else support.get("paired_pp_delta")
    )
    pp_harm = fnum(
        get_path(support, "paired_pp_delta", "p_harm")
        if isinstance(support.get("paired_pp_delta"), dict)
        else support.get("paired_pp_p_harm")
    )
    mmd_delta = fnum(
        get_path(support, "paired_mmd_delta", "delta_mean")
        if isinstance(support.get("paired_mmd_delta"), dict)
        else support.get("paired_mmd_delta")
    )
    mmd_harm = fnum(
        get_path(support, "paired_mmd_delta", "p_harm")
        if isinstance(support.get("paired_mmd_delta"), dict)
        else support.get("paired_mmd_p_harm")
    )

    return {
        "name": name,
        "status": status,
        "gpu_authorization": payload.get("gpu_authorization"),
        "next_authorization": payload.get("next_authorization"),
        "support_pp_delta": pp_delta,
        "support_pp_p_harm": pp_harm,
        "support_mmd_delta": mmd_delta,
        "support_mmd_p_harm": mmd_harm,
        "norman_pp_delta": fnum(norman.get("delta_pp")),
        "wessels_pp_delta": fnum(wessels.get("delta_pp")),
        "wessels_closure": fnum(wessels.get("route_gap_closed_fraction")),
        "reasons": reasons,
    }


def main() -> None:
    payloads = {name: load_json(path) for name, path in INPUTS.items()}
    rows = [row_for(name, payload) for name, payload in payloads.items()]

    passed_rows = [
        row for row in rows
        if str(row.get("status", "")).endswith("_pass")
        or row.get("gpu_authorization") not in (None, "none")
    ]
    unsafe_rows = [
        row for row in rows
        if row["name"] in {"hybrid_prior", "noharm_calibrated", "gene_risk", "module_coherence", "corum_complex"}
        and (
            (row.get("norman_pp_delta") is not None and row["norman_pp_delta"] < -0.01)
            or (row.get("support_pp_p_harm") is not None and row["support_pp_p_harm"] > 0.20)
        )
    ]
    narrow_rows = [
        row for row in rows
        if row["name"] == "additive_prior"
        and any("coverage" in str(reason) for reason in row.get("reasons") or [])
    ]
    control_fail_rows = [
        row for row in rows
        if row["name"] in {"dataset_conditioned", "module_coherence", "corum_complex"}
        and any("control" in str(reason) or "shuffled" in str(reason) for reason in row.get("reasons") or [])
    ]
    support_set_closed = (
        payloads["support_set_after_summary"].get("status")
        == "support_set_task_after_summary_close_summary_rule_no_gpu"
    )

    reasons: list[str] = []
    if not passed_rows:
        reasons.append("no_existing_safe_trainselect_gate_grants_gpu_authorization")
    if narrow_rows:
        reasons.append("raw_additive_prior_signal_is_low_coverage")
    if unsafe_rows:
        reasons.append("coverage_expansion_or_noharm_rules_show_norman_or_bootstrap_harm")
    if control_fail_rows:
        reasons.append("external_or_dataset_controls_do_not_separate_from_signal")
    if support_set_closed:
        reasons.append("support_set_summary_rule_already_closed_after_train_multi_loo_failure")

    status = "trackc_condition_delta_prior_covered_meta_gate_no_direct_gpu_cpu_preflight_required" if reasons else "trackc_condition_delta_prior_covered_meta_gate_pass_protocol_review"
    action = (
        "do_not_launch_gpu; run_exact_prior_covered_trainselect_cpu_proxy_gate_first"
        if reasons
        else "protocol_review_before_any_gpu_smoke"
    )
    result = {
        "status": status,
        "gpu_authorization": "none",
        "action": action,
        "boundary": {
            "cpu_only_existing_reports": True,
            "safe_trainselect_only": True,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
        },
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "rows": rows,
        "decision_reasons": reasons,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Condition-Delta Prior-Covered Meta Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        f"Action: `{action}`",
        "",
        "## Boundary",
        "",
        "- CPU-only aggregation of existing query-free safe-trainselect Track C gates.",
        "- Does not read held-out query, canonical test, canonical multi, active logs, or new GPU artifacts.",
        "- This report answers whether `condition_delta_head_use_in_model=True` with `prior_covered_gene_multi` has enough existing evidence for a new GPU smoke.",
        "",
        "## Evidence Rows",
        "",
        "| gate | status | pp delta | pp p_harm | Norman pp | Wessels pp | Wessels closure | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reasons_s = ", ".join(str(x) for x in row.get("reasons") or []) or "none"
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | {fmt(row.get('support_pp_delta'))} | "
            f"{fmt(row.get('support_pp_p_harm'))} | {fmt(row.get('norman_pp_delta'))} | "
            f"{fmt(row.get('wessels_pp_delta'))} | {fmt(row.get('wessels_closure'))} | {reasons_s} |"
        )
    lines.extend([
        "",
        "## Decision Reasons",
        "",
    ])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- `none`"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The prior-covered condition-delta injection is code-available, but direct GPU launch is not justified by the existing portfolio. The raw additive prior helps Wessels under low coverage, while coverage-expansion and calibrated variants fail on Norman, bootstrap harm, or controls. The exact `condition_delta_head_use_in_model=True` + `prior_covered_gene_multi` hook should therefore be treated as a narrow CPU-gated branch: no GPU until a train/support-only proxy gate shows prior-covered full-coverage rows separate from shuffled/inverted controls without Norman harm.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
