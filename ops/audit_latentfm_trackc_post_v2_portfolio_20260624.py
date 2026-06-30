#!/usr/bin/env python3
"""Track C post-v2 portfolio audit.

This is a query-free next-action audit. It may read already finalized public
Track C reporting summaries to state the current best frozen route, but it
does not read raw held-out query artifacts, active logs, canonical multi
outputs, or launch training.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_post_v2_portfolio_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_POST_V2_PORTFOLIO_20260624.md"


INPUTS = {
    "final_handoff": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_HANDOFF_20260623.md",
    "claim_readiness": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "condition_delta_meta": REPORTS / "latentfm_trackc_condition_delta_prior_covered_meta_gate_20260624.json",
    "prior_covered_proxy": REPORTS / "latentfm_trackc_prior_covered_condition_delta_proxy_gate_20260624.json",
    "dataset_conditioned": REPORTS / "latentfm_trackc_dataset_conditioned_noharm_gate_20260624.json",
    "corum_complex": REPORTS / "latentfm_trackc_corum_complex_module_gate_20260624.json",
    "omnipath_preflight_md": REPORTS / "LATENTFM_TRACKC_OMNIPATH_PAIR_PRIOR_PREFLIGHT_20260623.md",
    "module_coherence": REPORTS / "latentfm_trackc_composition_module_coherence_gate_20260623.json",
    "support_set_summary": REPORTS / "latentfm_trackc_support_set_task_summary_gate_20260623.json",
    "support_set_after_summary": REPORTS / "latentfm_trackc_support_set_task_after_summary_decision_20260623.json",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def get_nested(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def summarize_gate(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"name": name, "path": str(path), "exists": False}
    d = read_json(path)
    status = d.get("status") or get_nested(d, "decision", "status")
    reasons = (
        d.get("decision_reasons")
        or d.get("reasons")
        or get_nested(d, "decision", "reasons", default=[])
    )
    if reasons is None:
        reasons = []
    support = (
        d.get("support_val_summary")
        or d.get("selected_support_summary")
        or d.get("support_proxy")
        or {}
    )
    selected = d.get("selected_spec") or get_nested(d, "selected_train_proxy", "spec")
    return {
        "name": name,
        "path": str(path),
        "exists": True,
        "status": status,
        "selected": selected,
        "reasons": reasons,
        "support_summary": support,
    }


def main() -> None:
    claim = read_json(INPUTS["claim_readiness"])

    current_best = {
        "route": "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42",
        "scope": "frozen_support_context_v2_diagnostic_not_formal_multi_solution",
        "claim_status": claim.get("status"),
        "query_reuse_forbidden": True,
        "aggregate_query_pp_delta": get_nested(claim, "metrics", "query_multi_pearson_delta"),
        "aggregate_query_mmd_delta": get_nested(claim, "metrics", "query_multi_mmd_delta"),
        "unseen2_pp_delta": get_nested(claim, "metrics", "unseen2_pearson_delta"),
        "canonical_uncapped_noharm_exact_noop": bool(
            get_nested(claim, "checks", "canonical_uncapped_noharm_exact_noop", "passed", default=False)
        ),
    }

    gates = [
        summarize_gate("condition_delta_prior_covered_meta", INPUTS["condition_delta_meta"]),
        summarize_gate("prior_covered_condition_delta_proxy", INPUTS["prior_covered_proxy"]),
        summarize_gate("dataset_conditioned_noharm", INPUTS["dataset_conditioned"]),
        summarize_gate("corum_complex_module", INPUTS["corum_complex"]),
        summarize_gate("composition_module_coherence", INPUTS["module_coherence"]),
        summarize_gate("support_set_summary", INPUTS["support_set_summary"]),
        summarize_gate("support_set_after_summary", INPUTS["support_set_after_summary"]),
    ]

    gate_failures = [g for g in gates if g.get("exists") and "fail" in str(g.get("status", ""))]
    gpu_authorized = False
    if any("pass" in str(g.get("status", "")) and "gpu" in str(g).lower() for g in gates):
        gpu_authorized = False

    next_query_free_gates = [
        {
            "name": "support_present_ablation_reproducibility_gate",
            "hypothesis": (
                "The v2 support-context gain should be reproducible as a support-present "
                "effect across train/support rows without relying on held-out query; zero, "
                "shuffled, and support-absent controls should collapse."
            ),
            "artifacts": [
                "safe trainselect split",
                "frozen v2 support-context posthoc summaries",
                "support-present vs support-absent eval artifacts if already available, otherwise CPU-only launcher gate first",
            ],
            "pass_criteria": [
                "support-val multi pp delta >= +0.04 with bootstrap p_harm <= 0.20",
                "Norman support pp delta >= -0.01 and Wessels closure >= +0.05",
                "zero/shuffled/support-absent controls each at least 0.02 below candidate",
                "no canonical/query/canonical-multi reads",
            ],
            "fail_close": "If controls do not collapse or Norman is harmed, do not run another v2-family GPU smoke.",
        },
        {
            "name": "trainselect_pair_type_stratified_support_gate",
            "hypothesis": (
                "The main remaining weakness is not generic prior coverage but heterogeneous "
                "pair classes; a route may be useful only for support-visible pair types "
                "that are stable in train_multi LOO and support_val_multi."
            ),
            "artifacts": [
                "condition metadata",
                "safe trainselect train_multi/support_val_multi rows",
                "existing composition/noharm calibrated row-level summaries",
            ],
            "pass_criteria": [
                "predeclared pair-type strata selected only on train_multi LOO",
                "support-val pp delta >= +0.03 and MMD delta <= 0",
                "every enabled dataset delta >= -0.01",
                "stratum-shuffle and inverted-dataset controls separated by >= 0.02",
            ],
            "fail_close": "If improvement tracks dataset identity or controls, close as another unsafe routing proxy.",
        },
        {
            "name": "single_gene_support_coverage_floor_gate",
            "hypothesis": (
                "Prior-covered failures may be due to asymmetric missing single-gene support; "
                "only gene-multi rows with both genes present in train single/support banks "
                "should receive condition-delta intervention."
            ),
            "artifacts": [
                "safe trainselect split",
                "condition_metadata.json",
                "existing additive/noharm calibrated gate row summaries",
            ],
            "pass_criteria": [
                "train_multi LOO finds a coverage floor that improves pp >= +0.02 with p_harm <= 0.20",
                "support-val has >= 3 enabled rows per key dataset or no GPU",
                "zero and shuffled-gene controls collapse",
            ],
            "fail_close": "If support coverage remains too small or Wessels-only, do not spend GPU.",
        },
    ]

    payload = {
        "status": "trackc_post_v2_portfolio_no_gpu_query_free_next_gate_required",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": {
            "reads_raw_heldout_query": False,
            "reads_canonical_multi_for_selection": False,
            "reads_active_logs": False,
            "launches_gpu": False,
            "uses_final_query_summary_for_context_only": True,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "current_best": current_best,
        "gpu_authorized": gpu_authorized,
        "closed_gate_count": len(gate_failures),
        "closed_gates": gates,
        "next_query_free_gates": next_query_free_gates,
        "recommended_next": "support_present_ablation_reproducibility_gate",
        "decision_reasons": [
            "frozen_v2_resfilm_is_supported_only_as_diagnostic_not_general_solution",
            "heldout_query_reuse_for_selection_forbidden",
            "latest_prior_covered_dataset_conditioned_corum_and_support_set_gates_do_not_authorize_gpu",
            "new_modeling_requires_materially_new_query_free_cpu_gate",
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# Track C Post-V2 Portfolio Audit")
    lines.append("")
    lines.append(f"Status: `{payload['status']}`")
    lines.append("GPU authorization: `none`")
    lines.append("Held-out query reuse for selection: `forbidden`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- Reads finalized reporting summaries and query-free gate reports only.")
    lines.append("- Does not read raw held-out query artifacts, canonical multi for selection, active logs, or launch GPU jobs.")
    lines.append("- Final query evidence is used only to state the frozen-route context; it does not select the next gate.")
    lines.append("")
    lines.append("## Current Best")
    lines.append("")
    lines.append(f"- route: `{current_best['route']}`")
    lines.append(f"- claim scope: `{current_best['scope']}`")
    lines.append(f"- claim status: `{current_best['claim_status']}`")
    lines.append(f"- aggregate query pp delta: `{current_best['aggregate_query_pp_delta']}`")
    lines.append(f"- aggregate query MMD delta: `{current_best['aggregate_query_mmd_delta']}`")
    lines.append(f"- unseen2 pp caveat delta: `{current_best['unseen2_pp_delta']}`")
    lines.append("- interpretation: supported frozen diagnostic route, not a formal general multi solution.")
    lines.append("")
    lines.append("## Closed / Demoted Query-Free Gates")
    lines.append("")
    lines.append("| gate | status | selected | reasons |")
    lines.append("|---|---|---|---|")
    for g in gates:
        reasons = "; ".join(map(str, g.get("reasons") or []))
        lines.append(
            f"| `{g['name']}` | `{g.get('status')}` | `{g.get('selected')}` | {reasons} |"
        )
    lines.append("")
    lines.append("## Next Query-Free Gates")
    lines.append("")
    for item in next_query_free_gates:
        lines.append(f"### `{item['name']}`")
        lines.append("")
        lines.append(f"- Hypothesis: {item['hypothesis']}")
        lines.append(f"- Required artifacts: {', '.join(item['artifacts'])}.")
        lines.append(f"- Pass criteria: {'; '.join(item['pass_criteria'])}.")
        lines.append(f"- Fail-close rule: {item['fail_close']}")
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(
        "Do not launch Track C GPU from the current prior-covered/CORUM/dataset-conditioned/support-set evidence. "
        "The next permitted modeling step is a query-free CPU gate, with "
        "`support_present_ablation_reproducibility_gate` as the current first choice pending external subagent review."
    )
    lines.append("")
    lines.append("## JSON")
    lines.append("")
    lines.append(f"`{OUT_JSON}`")
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(OUT_MD)
    print(OUT_JSON)
    print(payload["status"])


if __name__ == "__main__":
    main()
