#!/usr/bin/env python3
"""CPU-only gate for chemical gene/drug conflict isolation.

This audit asks whether a frozen-gene/chemical-only adapter is justified from
existing all-modality dose-aware evidence. It reads completed train-only/internal
reports only; it does not train, infer, use GPU, read canonical multi, or read
Track C query.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_chemical_gene_drug_conflict_isolation_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_CHEMICAL_GENE_DRUG_CONFLICT_ISOLATION_GATE_20260625.md"

INPUTS = {
    "upper_bound": REPORTS / "latentfm_allmodality_modality_router_upper_bound_gate_20260625.json",
    "router_control": REPORTS / "latentfm_allmodality_modality_router_control_gate_20260625.json",
    "family_stratified": REPORTS / "latentfm_allmodality_family_stratified_protocol_gate_20260625.json",
    "family_tradeoff": REPORTS / "latentfm_allmodality_family_tradeoff_gate_20260625.json",
    "current_inventory": REPORTS / "latentfm_current_gpu_candidate_inventory_20260625.json",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))

    upper = read_json(INPUTS["upper_bound"])
    control = read_json(INPUTS["router_control"])
    strat = read_json(INPUTS["family_stratified"])
    tradeoff = read_json(INPUTS["family_tradeoff"])
    inventory = read_json(INPUTS["current_inventory"])

    route = upper.get("best_route", {})
    actual = control.get("actual", {})
    controls = control.get("controls", {})
    family_trade_rows = tradeoff.get("run_summary", []) or tradeoff.get("rows", [])
    if not family_trade_rows:
        # The public JSON stores top-level summaries under report-specific keys
        # in some versions. Keep the gate robust by relying on the control gate.
        family_trade_rows = []

    drug_pp = float(actual.get("family_drug_pp_mean", route.get("family_drug", {}).get("pp_mean", 0.0)))
    gene_pp = float(actual.get("family_gene_pp_mean", route.get("family_gene", {}).get("pp_mean", 0.0)))
    all_pp = float(actual.get("all_pp_mean", route.get("all", {}).get("pp_mean", 0.0)))
    drug_harm = float(actual.get("family_drug_hard_harm_frac", route.get("family_drug", {}).get("pp_hard_harm_frac", 1.0)))
    all_harm = float(actual.get("all_hard_harm_frac", route.get("all", {}).get("pp_hard_harm_frac", 1.0)))
    drug_shuffle_p95 = float(controls.get("drug_pp_mean_p95", 0.0))
    all_shuffle_p95 = float(controls.get("all_pp_mean_p95", 0.0))
    drug_delta_vs_shuffle = float(control.get("deltas_vs_control_mean", {}).get("drug_pp", 0.0))
    all_delta_vs_shuffle = float(control.get("deltas_vs_control_mean", {}).get("all_pp", 0.0))
    passing_policies = int(strat.get("passing_count", 0))

    reasons: list[str] = []
    if route.get("gene_choice") != "anchor":
        reasons.append("upper_bound_route_does_not_freeze_gene_to_anchor")
    if gene_pp < -0.002:
        reasons.append("gene_anchor_route_not_gene_safe")
    if drug_pp < 0.015:
        reasons.append("drug_pp_below_plus_0p015_gate")
    if all_pp < 0.005:
        reasons.append("all_pp_below_plus_0p005_gate")
    if drug_harm > 0.25:
        reasons.append("drug_hard_harm_frac_gt_0p25")
    if all_harm > 0.25:
        reasons.append("all_hard_harm_frac_gt_0p25")
    if drug_pp <= drug_shuffle_p95:
        reasons.append("drug_pp_not_above_count_matched_shuffle_p95")
    if all_pp <= all_shuffle_p95:
        reasons.append("all_pp_not_above_count_matched_shuffle_p95")
    if drug_delta_vs_shuffle < 0.005:
        reasons.append("drug_delta_vs_shuffle_mean_lt_0p005")
    if all_delta_vs_shuffle < 0.005:
        reasons.append("all_delta_vs_shuffle_mean_lt_0p005")
    if passing_policies <= 0:
        reasons.append("family_stratified_passing_policies_zero")
    if inventory.get("branches", {}).get("allmodality_doseaware", {}).get("immediate_gpu") is True:
        reasons.append("inventory_inconsistent_allmodality_gpu_true")

    passed = len(reasons) == 0
    status = (
        "chemical_gene_drug_conflict_isolation_pass_gpu_candidate"
        if passed
        else "chemical_gene_drug_conflict_isolation_fail_no_gpu"
    )

    payload = {
        "status": status,
        "gpu_authorized": passed,
        "boundary": {
            "cpu_only": True,
            "reads_completed_trainonly_internal_reports": True,
            "reads_model_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "hypothesis": "a frozen-gene-anchor chemical adapter may isolate drug gains without gene harm",
        "optimistic_upper_bound_route": {
            "gene_choice": route.get("gene_choice"),
            "drug_choice": route.get("drug_choice"),
            "all_pp_mean": all_pp,
            "gene_pp_mean": gene_pp,
            "drug_pp_mean": drug_pp,
            "all_hard_harm_frac": all_harm,
            "drug_hard_harm_frac": drug_harm,
            "all_shuffle_p95": all_shuffle_p95,
            "drug_shuffle_p95": drug_shuffle_p95,
            "all_delta_vs_shuffle_mean": all_delta_vs_shuffle,
            "drug_delta_vs_shuffle_mean": drug_delta_vs_shuffle,
            "family_stratified_passing_policies": passing_policies,
        },
        "thresholds": {
            "gene_pp_min": -0.002,
            "drug_pp_min": 0.015,
            "all_pp_min": 0.005,
            "hard_harm_frac_max": 0.25,
            "must_exceed_count_matched_shuffle_p95": True,
            "delta_vs_shuffle_mean_min": 0.005,
            "family_stratified_passing_policies_min": 1,
        },
        "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in INPUTS.items()},
        "decision": {
            "status": status,
            "gpu_authorized": passed,
            "reasons": reasons,
            "next_action": (
                "prepare one bounded frozen-gene chemical adapter smoke"
                if passed
                else "do not launch frozen-gene chemical adapter GPU from current allmodality evidence; require a new CPU mechanism/control that beats shuffle and hard-harm"
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Chemical Gene/Drug Conflict Isolation Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{passed}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed all-modality dose-aware train-only/internal reports.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C held-out query.",
        "",
        "## Optimistic Frozen-Gene Upper Bound",
        "",
        "| gene route | drug route | all pp | gene pp | drug pp | all hard-harm | drug hard-harm | all shuffle p95 | drug shuffle p95 | stratified passes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| `{}` | `{}` | {:+.6f} | {:+.6f} | {:+.6f} | {:.3f} | {:.3f} | {:+.6f} | {:+.6f} | {} |".format(
            route.get("gene_choice"),
            route.get("drug_choice"),
            all_pp,
            gene_pp,
            drug_pp,
            all_harm,
            drug_harm,
            all_shuffle_p95,
            drug_shuffle_p95,
            passing_policies,
        ),
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['decision']['next_action']}`",
        "",
        "The existing `gene=anchor, drug=allmod` route is an optimistic inference-time upper bound for a frozen-gene chemical adapter. Because it is close to count-matched shuffle, has high drug hard-harm, and has zero family-stratified passing policies, it does not authorize a GPU smoke.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
