#!/usr/bin/env python3
"""Synthesize the post-Locke LatentFM portfolio decision.

This is a read-only/reporting script. It consumes completed decision JSONs and
does not inspect canonical/query raw artifacts, active logs, or GPU outputs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(name: str) -> dict[str, Any]:
    path = REPORTS / name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data


def get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def first_result_summary(data: dict[str, Any], control: str, group: str) -> dict[str, Any]:
    for row in data.get("results", []):
        if row.get("control") == control and row.get("group") == group:
            return row.get("summary", row)
    return {}


def paired_delta(data: dict[str, Any], group: str, baseline: str) -> dict[str, Any]:
    for row in data.get("results", []):
        if row.get("control") == "main" and row.get("group") == group:
            for delta in row.get("paired_deltas", []):
                if delta.get("baseline") == baseline:
                    return delta
    return {}


def main() -> None:
    tracka_stop = {
        "name": "Track A model-search synthesis",
        "path": str(REPORTS / "LATENTFM_TRACKA_STOP_MODEL_SEARCH_SYNTHESIS_20260624.md"),
        "status": "tracka_stop_model_search_current_default_xverse_anchor",
        "decision": "current deployable/default remains xverse_8k_anchor; no Track A GPU search without genuinely new CPU gate",
    }
    ceiling = load_json("latentfm_tracka_identifiability_ceiling_20260624.json")
    trackc = load_json("latentfm_trackc_v2_family_closure_synthesis_20260624.json")
    ot = {
        "name": "OT minibatch pairing synthesis",
        "path": str(REPORTS / "LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md"),
        "status": "ot_wired_but_no_model_gain_close_current_ot_redesign",
        "decision": "OT is wired and not no-op, but random/Hungarian/current variants fail Track A gates; keep closed",
    }
    reliability = load_json("latentfm_trainonly_reliability_condition_gate_20260624.json")
    distributional = load_json("latentfm_distributional_mmd_harm_gate_20260624.json")
    forensic = load_json("latentfm_tracka_xverse_forensic_distillation_gate_20260624.json")
    jiang = load_json("latentfm_jiang_celltype_program_gate_20260624.json")
    prototype = load_json("latentfm_perturbation_equivariant_prototype_gate_20260624.json")
    factorized = load_json("latentfm_factorized_gene_context_gate_20260624.json")

    ceiling_decisions = get(ceiling, "decision.group_decisions", [])
    cross_ceiling = ceiling_decisions[0] if ceiling_decisions else {}
    family_ceiling = ceiling_decisions[1] if len(ceiling_decisions) > 1 else {}

    dist_cross = first_result_summary(
        distributional, "main", "internal_val_cross_background_seen_gene_proxy"
    )
    dist_family = first_result_summary(distributional, "main", "internal_val_family_gene_proxy")
    jiang_cross = first_result_summary(jiang, "main", "internal_val_cross_background_seen_gene_proxy")
    jiang_family = first_result_summary(jiang, "main", "internal_val_family_gene_proxy")
    forensic_cross = paired_delta(
        forensic, "internal_val_cross_background_seen_gene_proxy", "gene_raw_mean"
    )
    forensic_family = paired_delta(forensic, "internal_val_family_gene_proxy", "gene_raw_mean")

    trackc_best = trackc.get("current_best", {})
    trackc_decision = trackc.get("decision", {})
    reliability_runs = {
        row.get("run"): row for row in get(reliability, "decision.run_decisions", [])
    }
    cap60_reliability = reliability_runs.get("cap60_protocol", {})
    cap120_reliability = reliability_runs.get("cap120", {})
    trackc_expansion_metrics = {
        row.get("name"): row.get("key_metrics", {}) for row in trackc.get("expansion_gates", [])
    }

    branches = [
        tracka_stop,
        {
            "name": "Track A identifiability ceiling",
            "path": ceiling["_path"],
            "status": get(ceiling, "decision.status"),
            "gpu_authorized": get(ceiling, "decision.gpu_authorized"),
            "key_metrics": {
                "cross_oracle_pp": cross_ceiling.get("oracle_mean_pp_delta"),
                "cross_best_gate_pp": cross_ceiling.get("best_gate_mean_pp_delta"),
                "cross_safe_gates": cross_ceiling.get("n_safe_gates"),
                "family_oracle_pp": family_ceiling.get("oracle_mean_pp_delta"),
                "family_best_gate_pp": family_ceiling.get("best_gate_mean_pp_delta"),
                "family_safe_gates": family_ceiling.get("n_safe_gates"),
            },
        },
        ot,
        {
            "name": "Reliability-weighted condition gate",
            "path": reliability["_path"],
            "status": get(reliability, "decision.status"),
            "gpu_authorized": get(reliability, "decision.gpu_authorized"),
            "key_metrics": {
                "cap60_cross_pp": cap60_reliability.get("cross_mean_pp_delta"),
                "cap60_family_pp": cap60_reliability.get("family_mean_pp_delta"),
                "cap60_reasons": cap60_reliability.get("reasons"),
                "cap120_cross_pp": cap120_reliability.get("cross_mean_pp_delta"),
                "cap120_family_pp": cap120_reliability.get("family_mean_pp_delta"),
            },
        },
        {
            "name": "Track C support-context v2 family",
            "path": trackc["_path"],
            "status": trackc.get("status"),
            "gpu_authorized": trackc_decision.get("gpu_authorized"),
            "heldout_query_authorized": trackc_decision.get("heldout_query_authorized"),
            "current_best": trackc_best,
            "key_metrics": {
                "query_multi_pp": trackc_best.get("query_multi_pearson_delta"),
                "query_multi_mmd": trackc_best.get("query_multi_mmd_delta"),
                "unseen2_pp": trackc_best.get("unseen2_pearson_delta"),
                "pseudo_zero_overlap_pp": trackc_expansion_metrics.get("pseudo_episode", {}).get("zero_overlap_pp"),
                "jackknife_negative_rows": trackc_expansion_metrics.get("support_jackknife", {}).get("enabled_negative_rows"),
                "nonadditivity_support_pp": trackc_expansion_metrics.get("response_nonadditivity", {}).get("support_pp"),
            },
        },
        {
            "name": "Distributional MMD-harm safety gate",
            "path": distributional["_path"],
            "status": get(distributional, "decision.status"),
            "gpu_authorized": get(distributional, "decision.gpu_authorized"),
            "key_metrics": {
                "cross_pp": dist_cross.get("mean_pp_delta"),
                "family_pp": dist_family.get("mean_pp_delta"),
                "family_mmd": dist_family.get("mean_mmd_delta"),
            },
        },
        {
            "name": "Deployable forensic-risk distillation",
            "path": forensic["_path"],
            "status": get(forensic, "decision.status"),
            "gpu_authorization": get(forensic, "decision.gpu_authorization"),
            "key_metrics": {
                "cross_delta_vs_gene": forensic_cross.get("delta_mean"),
                "cross_p_harm": forensic_cross.get("p_harm"),
                "family_delta_vs_gene": forensic_family.get("delta_mean"),
            },
        },
        {
            "name": "Jiang cell-type response program",
            "path": jiang["_path"],
            "status": get(jiang, "decision.status"),
            "gpu_authorized": get(jiang, "decision.gpu_authorized"),
            "key_metrics": {
                "cross_delta_vs_gene": jiang_cross.get("delta_vs_gene"),
                "family_delta_vs_gene": jiang_family.get("delta_vs_gene"),
                "dataset_min": min(
                    x
                    for x in [
                        jiang_cross.get("dataset_min"),
                        jiang_family.get("dataset_min"),
                    ]
                    if x is not None
                ),
                "harm_frac": max(
                    x
                    for x in [
                        jiang_cross.get("harm_frac"),
                        jiang_family.get("harm_frac"),
                    ]
                    if x is not None
                ),
            },
        },
        {
            "name": "Perturbation-equivariant prototype",
            "path": prototype["_path"],
            "status": get(prototype, "decision.status"),
            "gpu_authorized": get(prototype, "decision.gpu_authorized"),
            "key_metrics": {
                "cross_pp": get(prototype, "decision.cross_mean_pp_delta"),
                "family_pp": get(prototype, "decision.family_mean_pp_delta"),
                "dataset_min": get(prototype, "decision.cross_dataset_min"),
            },
        },
        {
            "name": "Factorized gene x context",
            "path": factorized["_path"],
            "status": get(factorized, "decision.status"),
            "gpu_authorized": get(factorized, "decision.gpu_authorized"),
            "key_metrics": {
                "cross_pp": get(factorized, "decision.cross_mean_pp_delta"),
                "family_pp": get(factorized, "decision.family_mean_pp_delta"),
                "dataset_min": get(factorized, "decision.cross_dataset_min"),
            },
        },
    ]

    decision = {
        "status": "post_locke_portfolio_no_gpu_candidate_reporting_or_new_mechanism_gate_required",
        "gpu_authorized": False,
        "current_tracka_default": "xverse_8k_anchor",
        "current_trackc_best": trackc_best,
        "resource_note": (
            "GPU resources may be idle, but AGENTS requires a clear hypothesis, "
            "gate, failure rule, and authorization before GPU launch."
        ),
        "next_allowed_actions": [
            "paper-grade failure map / provenance / bootstrap evidence package for current best and closed branches",
            "new materially different CPU-first gate only if it is not another OT, reliability, scaling, archetype, TrackC-v2, forensic-risk, or Jiang-program variant",
            "external read-only review integration before any new high-compute branch",
        ],
        "blocked_actions": [
            "launching GPU from closed OT pair-mode or cost sweeps",
            "launching GPU from Track A scaling/reliability/proxy-routing branches",
            "using canonical multi or Track C held-out query for selection",
            "claiming formal multi success from frozen Track C v2 diagnostic",
        ],
    }

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": {
            "read_completed_decision_reports_only": True,
            "read_active_logs": False,
            "read_canonical_or_query_raw_artifacts": False,
            "launched_jobs": False,
            "used_gpu": False,
        },
        "decision": decision,
        "branches": branches,
    }

    json_path = REPORTS / "latentfm_post_locke_portfolio_decision_20260624.json"
    md_path = REPORTS / "LATENTFM_POST_LOCKE_PORTFOLIO_DECISION_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Post-Locke Portfolio Decision",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Boundary",
        "",
        "- Reads completed decision reports only.",
        "- Does not read active logs, canonical/raw held-out query artifacts, or canonical multi for selection.",
        "- Does not launch training, inference, embedding extraction, or GPU work.",
        "",
        "## Current Best",
        "",
        "- Track A deployable/default: `xverse_8k_anchor`.",
        "- Track C diagnostic: `xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42`.",
        f"- Track C frozen diagnostic query_multi Pearson/MMD deltas: `{trackc_best.get('query_multi_pearson_delta'):+.6f}` / `{trackc_best.get('query_multi_mmd_delta'):+.6f}`.",
        f"- Track C unseen2 Pearson delta remains weak: `{trackc_best.get('unseen2_pearson_delta'):+.6f}`.",
        "",
        "## Decision",
        "",
        "- GPU authorized: `False`.",
        "- Current action: do not launch more GPU jobs from the closed portfolio.",
        "- Next valid work is either paper-grade consolidation/failure-map/provenance, or a materially new CPU-first gate with negative controls and a hard stop rule.",
        "",
        "## Closed / Non-Promoted Branches",
        "",
        "| Branch | Status | Key evidence |",
        "|---|---|---|",
    ]

    for branch in branches:
        metrics = branch.get("key_metrics") or branch.get("decision_reasons") or branch.get("decision")
        if isinstance(metrics, dict):
            key = ", ".join(f"{k}={v}" for k, v in metrics.items())
        elif isinstance(metrics, list):
            key = "; ".join(str(x) for x in metrics[:4])
        else:
            key = str(metrics)
        lines.append(
            f"| {branch['name']} | `{branch.get('status')}` | {key}; path `{branch.get('path')}` |"
        )

    lines.extend(
        [
            "",
            "## Why Idle GPU Is Currently Correct",
            "",
            "The recent portfolio repeatedly finds average internal signal without deployable safety: worst-dataset harm, failed negative controls, canonical no-harm failures, or forbidden-oracle dependence. Launching GPU now would repeat closed branches rather than test a new hypothesis.",
            "",
            "## Next Allowed Gate Standard",
            "",
            "A new gate must be train-only/query-blind, materially different from closed families, and must specify: hypothesis, exact inputs, nested validation, bootstrap/CI, dataset-min/no-harm thresholds, shuffled/sign-inverted/equal-cost or equivalent negative controls, and a predeclared GPU smoke plus fail-close rule if it passes.",
            "",
            "## JSON",
            "",
            f"`{json_path}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
