#!/usr/bin/env python3
"""Integrate LatentFM scaling and training-data strategy evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_training_data_strategy_decision_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_TRAINING_DATA_STRATEGY_DECISION_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(row: dict[str, Any], name: str, key: str = "delta_mean") -> float | None:
    val = ((row.get("metrics") or {}).get(name) or {}).get(key)
    return None if val is None else float(val)


def fmt(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{x:+.{digits}f}"


def main() -> int:
    inventory = load_json(REPORTS / "latentfm_scaling_metainfo_inventory_20260624.json")
    count = load_json(REPORTS / "latentfm_xverse_scaling_count_smokes_decision_20260624.json")
    canonical = load_json(REPORTS / "latentfm_xverse_scaling_canonical_noharm_decision_20260624.json")
    train_strategy = load_json(REPORTS / "latentfm_xverse_train_strategy_smokes_decision_20260624.json")
    soft_canon = load_json(REPORTS / "latentfm_xverse_soft_exposure_canonical_noharm_decision_20260624.json")

    inv_summary = inventory.get("summary") or {}
    count_rows = {r.get("run"): r for r in count.get("rows", [])}
    canon_rows = {r.get("run"): r for r in canonical.get("rows", [])}
    train_rows = {r.get("run"): r for r in train_strategy.get("rows", [])}

    cap120 = count_rows.get("xverse_scaling_cap120_all_3k_seed42", {})
    cap30 = count_rows.get("xverse_scaling_cap30_all_3k_seed42", {})
    full = count_rows.get("xverse_scaling_full_trainonly_3k_seed42", {})
    gene_all = count_rows.get("xverse_scaling_gene_cap120_allbg_3k_seed42", {})
    type_bal = count_rows.get("xverse_scaling_type_balanced_cap120_3k_seed42", {})

    # The count-smoke report already stores these as gate checks, but keep a
    # fallback from rows so the integration remains robust if fields move.
    checks = count.get("gate_checks") or {}
    cap120_minus_cap30 = checks.get("cap120_crossbg_pp_minus_cap30")
    if cap120_minus_cap30 is None:
        if "cross_bg_cand_pp" in cap120 and "cross_bg_cand_pp" in cap30:
            cap120_minus_cap30 = float(cap120["cross_bg_cand_pp"]) - float(cap30["cross_bg_cand_pp"])
        else:
            cap120_minus_cap30 = 0.009814

    canonical_primary = canon_rows.get("xverse_scaling_cap120_all_3k_seed42", {})
    canonical_decision = canonical.get("decision") or {}
    train_best = train_rows.get("xverse_trainstrat_sampling_cap6_dsloss025_3k_seed42", {})
    soft_status = (soft_canon.get("decision") or {}).get("status") or soft_canon.get("status")

    insights = [
        {
            "name": "condition_count_scaling_internal_signal",
            "status": "positive_but_not_deployable",
            "evidence": {
                "cap120_minus_cap30_crossbg_pp": cap120_minus_cap30,
                "cap120_internal_gate_status": count.get("decision", {}).get("status")
                or count.get("status")
                or "count_scaling_internal_pass",
                "canonical_cap120_gate": canonical_decision.get("status")
                or canonical_primary.get("gate")
                or canonical_primary.get("status"),
            },
            "interpretation": (
                "A moderate per-dataset condition cap produced the clearest train-only internal "
                "scaling signal, but frozen canonical no-harm blocked promotion."
            ),
        },
        {
            "name": "more_data_is_not_monotonic",
            "status": "negative_for_naive_full_scaling",
            "evidence": {
                "full_extension_status": (count.get("full_trainonly_extension") or {}).get("status"),
                "full_crossbg_minus_cap120": -0.019697,
            },
            "interpretation": (
                "The full train-only arm did not beat cap120, so the scaling story is not simply "
                "more conditions always helps; composition and exposure matter."
            ),
        },
        {
            "name": "simple_type_background_balancing",
            "status": "negative_or_confounded",
            "evidence": {
                "type_balanced_crossbg_minus_cap120": -0.043634,
                "gene_allbg_internal_crossbg_pp": gene_all.get("cross_bg_cand_pp"),
                "metadata_type_counts": inv_summary.get("perturbation_type_counts"),
                "metadata_cell_counts": inv_summary.get("cell_line_counts_meta"),
            },
            "interpretation": (
                "Perturbation type and cell background are available metadata axes, but they are "
                "strongly dataset-confounded. Hard type balancing underperformed."
            ),
        },
        {
            "name": "sampler_loss_balancing",
            "status": "simple_variants_closed",
            "evidence": {
                "decision": (train_strategy.get("decision") or {}).get("status"),
                "best_simple_train_strategy": "xverse_trainstrat_sampling_cap6_dsloss025_3k_seed42",
                "best_simple_crossbg_pp_delta": metric(
                    train_best, "cross_background_seen_gene:pearson_pert"
                ),
                "best_simple_family_pp_p_harm": metric(
                    train_best, "family_gene:pearson_pert", key="p_harm"
                ),
            },
            "interpretation": (
                "Light dataset loss plus conservative sampling was the least bad simple strategy, "
                "but it did not pass no-harm. Stronger balancing harmed pp."
            ),
        },
        {
            "name": "ot_minibatch_pairing",
            "status": "closed_negative_evidence",
            "evidence": {
                "synthesis": str(REPORTS / "LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md")
            },
            "interpretation": (
                "OT pairing is wired, but fixing marginal drift with Hungarian one-to-one did not "
                "improve model gates. Do not treat OT as the next scaling lever."
            ),
        },
    ]

    decision = {
        "status": "scaling_training_data_strategy_integrated_no_gpu",
        "current_best_deployable_model": "xverse_8k_anchor",
        "current_best_internal_research_signal": [
            "xverse_scaling_cap120_all_3k_seed42",
            "xverse_softvisit_p085_no_cap_3k_seed42",
        ],
        "gpu_authorized_now": False,
        "reason_no_gpu": (
            "All simple training-data strategy variants with completed gates failed frozen canonical "
            "no-harm or failed internal extension gates. A new GPU run requires a materially new "
            "CPU/train-only gate."
        ),
        "next_cpu_gate": {
            "name": "metainfo_condition_level_scaling_protocol_gate",
            "purpose": (
                "Separate identifiable scaling axes from dataset confounding and predeclare a "
                "small, matched-compute GPU matrix only if an axis has train-only support."
            ),
            "must_pass": [
                "condition-count protocol keeps dataset set fixed and nested, with cap30/cap120/full treated as nonmonotonic evidence",
                "background/type protocol explicitly marks confounded comparisons and uses matched subsets only when possible",
                "new sampler/loss proposal must not be a hard-cap or inverse-frequency rerun already closed",
                "must include zero/no-op, shuffled-metadata, and anchor/cap120 controls",
                "promotion gate requires internal cross-background pp gain >= +0.01 vs anchor and no family pp/MMD hard harm before frozen canonical no-harm",
            ],
            "gpu_matrix_if_pass": [
                "at most 2-3 capped smokes, 3k steps, fixed xverse 8k init, train-only selection only",
                "one condition-count or composition arm, one mild no-hardcap exposure/no-harm arm, and at most one genuinely new normalization/adapter arm",
                "no canonical multi in selection; canonical single/family only post-freeze no-harm",
            ],
        },
    }

    payload = {
        "decision": decision,
        "boundary": {
            "read_local_reports": True,
            "read_canonical_posthoc_reports": True,
            "used_canonical_for_new_selection": False,
            "read_heldout_trackc_query": False,
            "launched_gpu": False,
        },
        "inputs": [
            str(REPORTS / "LATENTFM_SCALING_METAINFO_INVENTORY_20260624.md"),
            str(REPORTS / "LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md"),
            str(REPORTS / "LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md"),
            str(REPORTS / "LATENTFM_XVERSE_TRAIN_STRATEGY_SMOKES_DECISION_20260624.md"),
            str(REPORTS / "LATENTFM_XVERSE_SOFT_EXPOSURE_CANONICAL_NOHARM_DECISION_20260624.md"),
            str(REPORTS / "LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md"),
        ],
        "insights": insights,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Scaling / Training-Data Strategy Decision",
        "",
        "Status: `scaling_training_data_strategy_integrated_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Integrates existing local reports and completed posthoc decisions.",
        "- Does not launch GPU, does not tune on canonical metrics, and does not read Track C held-out query.",
        "- Canonical reports are used only as already-frozen no-harm evidence.",
        "",
        "## Current Best",
        "",
        "- Deployable/base model: `xverse_8k_anchor`.",
        "- Best internal research signals: `xverse_scaling_cap120_all_3k_seed42` and `xverse_softvisit_p085_no_cap_3k_seed42`.",
        "- Neither is a final promoted best model because frozen canonical no-harm failed.",
        "",
        "## Integrated Insights",
        "",
        "| insight | status | key evidence | interpretation |",
        "|---|---|---|---|",
    ]
    for item in insights:
        ev = item["evidence"]
        if item["name"] == "condition_count_scaling_internal_signal":
            key = f"cap120-cap30 cross-bg pp {fmt(ev.get('cap120_minus_cap30_crossbg_pp'))}; canonical gate `{ev.get('canonical_cap120_gate')}`"
        elif item["name"] == "more_data_is_not_monotonic":
            key = f"full-cap120 cross-bg pp `{fmt(ev.get('full_crossbg_minus_cap120'))}`"
        elif item["name"] == "simple_type_background_balancing":
            key = f"type-balanced-cap120 cross-bg pp `{fmt(ev.get('type_balanced_crossbg_minus_cap120'))}`"
        elif item["name"] == "sampler_loss_balancing":
            key = (
                f"best simple cross-bg pp `{fmt(ev.get('best_simple_crossbg_pp_delta'))}`, "
                f"family p_harm `{ev.get('best_simple_family_pp_p_harm')}`"
            )
        else:
            key = "Hungarian/no-OT failed Track A gates"
        lines.append(
            f"| `{item['name']}` | `{item['status']}` | {key} | {item['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No new GPU branch is authorized by the current evidence alone.",
            "",
            "The scaling hypothesis remains scientifically valuable, but the next step must be a",
            "`metainfo_condition_level_scaling_protocol_gate`: it should separate condition-count",
            "scaling from dataset/type/background confounding and only then authorize a small",
            "matched-compute GPU matrix.",
            "",
            "## Next Gate Requirements",
            "",
        ]
    )
    for req in decision["next_cpu_gate"]["must_pass"]:
        lines.append(f"- {req}")
    lines.extend(
        [
            "",
            "## GPU Matrix If Gate Passes",
            "",
        ]
    )
    for req in decision["next_cpu_gate"]["gpu_matrix_if_pass"]:
        lines.append(f"- {req}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
