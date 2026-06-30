#!/usr/bin/env python3
"""Build a current LatentFM next-action slate after branch gates.

CPU-only synthesis. It does not train, infer, read checkpoints, read canonical
multi, read Track C held-out query, or use GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
INVENTORY_JSON = REPORTS / "latentfm_current_gpu_candidate_inventory_20260625.json"
AXIS_JSON = REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.json"
OUT_JSON = REPORTS / "latentfm_next_action_slate_20260626.json"
OUT_MD = REPORTS / "LATENTFM_NEXT_ACTION_SLATE_20260626.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_row(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("branch") == branch), {})


def axis_row(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("axis") == axis), {})


def main() -> int:
    inventory = load_json(INVENTORY_JSON)
    axis = load_json(AXIS_JSON)
    inv_rows = inventory.get("rows", [])
    axis_rows = axis.get("rows", [])
    reagent = inventory_row(inv_rows, "reagent_read_support_source_artifacts")
    truecell = inventory_row(inv_rows, "true_cell_budget128_6k")
    truecell_meta = load_json(REPORTS / "latentfm_truecell_nonnoop_tail_protection_meta_gate_20260626.json")
    truecell_meta_failed = str(truecell_meta.get("status", "")).endswith("fail_no_gpu")
    chemical = inventory_row(inv_rows, "chemical_unseen_scaffold_v2_fixedstep_controls")
    trackc = inventory_row(inv_rows, "trackc_support_and_routed_distill")
    qc = inventory_row(inv_rows, "new_trainonly_artifact_overlap")
    reliability_v2 = inventory_row(inv_rows, "external_reliability_artifact_v2")
    norman_growth = inventory_row(inv_rows, "norman_program_growth_artifact")
    source_obs = inventory_row(inv_rows, "external_source_h5ad_obs_routes")
    source_scout = inventory_row(inv_rows, "condition_level_reliability_source_scout")
    acquisition_slate = inventory_row(inv_rows, "external_condition_artifact_acquisition_slate")
    gwt = inventory_row(inv_rows, "gwt_condition_reliability_artifact")
    target_axis = axis_row(axis_rows, "target_observability")
    target_v3 = load_json(REPORTS / "latentfm_target_observability_residual_v3_gate_20260626.json")
    target_v3_failed = str(target_v3.get("status", "")).endswith("fail_no_gpu")

    actions = [
        {
            "name": "external_reliability_artifact_acquisition_v2",
            "priority": 1,
            "type": "cpu_source_acquisition_or_preflight",
            "hypothesis": "A genuinely condition-level external reliability or dose/time artifact, distinct from read/UMI/QC, static gene priors, and source-block confounds, may explain reliable train conditions.",
            "why_not_duplicate": "Current reagent/read-support branch is closed by source-block shuffle p=0.0999, existing reliability-v2 fields failed preflight, Norman-only program/growth artifacts failed single-dataset/tail/MMD preflight, downloaded Frangieh/Dixit h5ad obs sources contain no unconsumed time/viability/growth/dose/replicate/program columns, the condition-level source scout found zero local-ready candidates, and GWT gene-level reliability artifacts failed strict tail/MMD preflight; this requires a truly condition-level source such as SciPlex dose/time or dataset-matched pseudobulk concordance.",
            "boundary": "train-only/internal outcome rows only; no checkpoint reads, canonical multi, Track C query, or held-out selection.",
            "next_gate": acquisition_slate.get("next_gate")
            or source_scout.get("next_gate")
            or reliability_v2.get("next_gate")
            or "source manifest plus strict preflight for replicate concordance, dose/time/viability/growth, or background-specific context requiring >=3 datasets, >=50 overlap rows, >=3 varying datasets, bootstrap lower >0, dataset min >= -0.020, MMD max <= +0.001, within-dataset shuffle p <= 0.01, and LODO/source-block positive.",
            "resources": "CPU/source metadata only; <=4 cores for parsing; no GPU.",
            "promotion_gate": "external review before any bounded reliability-aware sampler/loss/staged-training smoke.",
            "fail_close": "close if signal is carried by one dataset/source block, MMD unsafe rows, or within-dataset shuffle/null.",
            "evidence": reagent.get("evidence", [])
            + reliability_v2.get("evidence", [])
            + norman_growth.get("evidence", [])
            + source_obs.get("evidence", [])
            + source_scout.get("evidence", [])
            + acquisition_slate.get("evidence", [])
            + gwt.get("evidence", [])
            + [qc.get("evidence", [])],
            "gpu_authorized_now": False,
        },
        {
            "name": "true_cell_nonnoop_tail_protection_meta_closed" if truecell_meta_failed else "true_cell_nonnoop_tail_protection_cpu_gate",
            "priority": 2,
            "type": "closed_cpu_gate" if truecell_meta_failed else "cpu_gate_design",
            "hypothesis": "True-cell support has the strongest mechanism signal but needs a non-noop tail-protection route with canonical footprint before any GPU.",
            "why_not_duplicate": "Plain budget128 6k, exact-condition/S0/uncertainty fallback, and anchor replay are closed; this gate must prove nonzero frozen canonical footprint and tail repair before training.",
            "boundary": "train-only/internal condition metrics for design; frozen canonical single/family only as no-harm veto after route freeze; canonical multi and Track C query excluded.",
            "next_gate": (
                "closed by meta-gate: existing true-cell count, stratum, uncertainty fallback, and risk-row routes do not produce a non-noop safe canonical footprint; reopen only with materially new non-noop tail-protection mechanism or external reliability artifact"
                if truecell_meta_failed
                else "materialize a CPU-only candidate policy from existing true-cell multi-seed metrics that passes dataset-tail >= -0.02, bootstrap lower > 0, MMD no-harm, and nonzero canonical footprint."
            ),
            "resources": "CPU-only synthesis; no GPU until the gate passes and external review agrees.",
            "promotion_gate": "bounded GPU smoke only if internal cross/family >= +0.02, no MMD/tail veto, nonzero canonical footprint, and frozen no-harm route is predeclared.",
            "fail_close": "closed; do not launch true-cell sampler/loss/staged training from existing artifacts." if truecell_meta_failed else "close if it reproduces exact-condition/S0 no-op behavior or has zero canonical footprint.",
            "evidence": truecell.get("evidence", []) + ["reports/LATENTFM_TRUECELL_NONNOOP_TAIL_PROTECTION_META_GATE_20260626.md"],
            "gpu_authorized_now": False,
        },
        {
            "name": "target_observability_residual_v3_closed" if target_v3_failed else "target_observability_failure_localization_v3",
            "priority": 3,
            "type": "closed_diagnostic" if target_v3_failed else "cpu_diagnostic",
            "hypothesis": "Target observability has a weak positive clue but unsafe tails; a tighter residual/tail localization may identify why it fails, without GPU.",
            "why_not_duplicate": "Current target-observability v2 is tail-unsafe and not a training route; this is failure analysis only unless a strict residual gate passes.",
            "boundary": "train-only target/activity strata and completed internal rows; no training, canonical multi, Track C query, or checkpoint selection.",
            "next_gate": (
                "closed by residual v3: no policy passed residual tail/MMD and within-dataset shuffle p<=0.01; reopen only with genuinely new external target/reliability artifact"
                if target_v3_failed
                else "condition-level residual analysis controlling dataset/source/count with predeclared target strata; require dataset min >= -0.02, hard-harm veto, permutation p <= 0.01."
            ),
            "resources": "CPU-only; no GPU.",
            "promotion_gate": "only if it produces a new source/count/tail-safe artifact and external review agrees.",
            "fail_close": "closed; do not mutate target-observability policies without a new external artifact." if target_v3_failed else "close if high-observability gains remain concentrated in unsafe tails.",
            "evidence": [
                target_axis.get("source_reports", ""),
                "reports/LATENTFM_TARGET_OBSERVABILITY_RESIDUAL_V3_GATE_20260626.md",
            ],
            "gpu_authorized_now": False,
        },
        {
            "name": "chemical_v2_fixedstep_ack_route",
            "priority": 4,
            "type": "ack_gated_gpu_route",
            "hypothesis": "Independent chemical V2 fixed-step controls may still be the only prepared GPU route, but it is explicitly ACK-gated.",
            "why_not_duplicate": "Old same-split chemical pass was consumed by seed controls; V2 uses safe independent scaffold split and fixed-latest checkpoint protocol.",
            "boundary": "V2 protocol only; no canonical multi or Track C query; no launch without explicit ACK.",
            "next_gate": chemical.get("next_gate"),
            "resources": "If ACKed: current temporary cap max 2 GPUs, max 2 training jobs/GPU, <=24 cores; otherwise no GPU.",
            "promotion_gate": "real descriptor seeds must replicate against shuffled/random controls with no family-drug hard harm.",
            "fail_close": "close if seed43/44 real does not beat descriptor controls or family-drug hard harm appears.",
            "evidence": chemical.get("evidence", []),
            "gpu_authorized_now": False,
        },
        {
            "name": "trackc_support_only_new_mechanism",
            "priority": 5,
            "type": "future_cpu_design",
            "hypothesis": "Formal multi capability needs a materially new support-only mechanism on safe trainselect, not routed-distill/support-context repeats.",
            "why_not_duplicate": "Routed-distill and row-reliability V2 are closed before query; held-out query remains untouched.",
            "boundary": "safe trainselect split only; selection on support-val only; held-out query exactly once after freeze.",
            "next_gate": trackc.get("next_gate"),
            "resources": "CPU design/audit first; no GPU from existing evidence.",
            "promotion_gate": "support-val material gain with canonical no-harm before frozen query evaluation.",
            "fail_close": "close if support mechanism repeats routed-distill/row-reliability or canonical no-harm fails.",
            "evidence": trackc.get("evidence", []),
            "gpu_authorized_now": False,
        },
    ]
    payload = {
        "status": "latentfm_next_action_slate_no_immediate_gpu",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "current_default_model": axis.get("current_default_model", "xverse_8k_anchor"),
        "immediate_gpu_candidate_count": inventory.get("immediate_gpu_candidate_count", 0),
        "gpu_authorized": False,
        "actions": actions,
        "decision": "No non-ACK GPU launch is legal from current evidence; prioritize CPU gates/external artifacts unless explicit chemical V2 ACK is provided.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Next Action Slate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Default/deployable model: `{payload['current_default_model']}`",
        "",
        f"Immediate GPU candidate count: `{payload['immediate_gpu_candidate_count']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of current inventory and scaling axis matrix.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Slate",
        "",
        "| priority | action | type | hypothesis | next gate | GPU now |",
        "|---:|---|---|---|---|---|",
    ]
    for row in actions:
        lines.append(
            "| {priority} | `{name}` | `{type}` | {hypothesis} | {next_gate} | `{gpu}` |".format(
                priority=row["priority"],
                name=row["name"],
                type=row["type"],
                hypothesis=row["hypothesis"],
                next_gate=row["next_gate"],
                gpu=row["gpu_authorized_now"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- {payload['decision']}",
            "- Do not relaunch closed read/UMI/QC/OT/allmod/visit-cap/true-cell no-op branches.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
