#!/usr/bin/env python3
"""CPU-only upper-bound gate for modality-routed allmod checkpoint selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625"
DECISION_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_smoke_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_allmodality_modality_router_upper_bound_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_ALLMODALITY_MODALITY_ROUTER_UPPER_BOUND_GATE_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def family_records(run_name: str, family: str) -> list[dict[str, Any]]:
    posthoc = RUN_ROOT / run_name / "posthoc_eval_internal"
    anchor = load_json(posthoc / "condition_family_eval_anchor_internal_ode20.json")
    candidate = load_json(posthoc / "condition_family_eval_candidate_internal_ode20.json")
    a_group = (anchor.get("groups") or {}).get(family) or {}
    c_group = (candidate.get("groups") or {}).get(family) or {}
    a_map = {condition_key(row): row for row in a_group.get("condition_metrics", [])}
    c_map = {condition_key(row): row for row in c_group.get("condition_metrics", [])}
    records = []
    for key in sorted(set(a_map) & set(c_map)):
        a = a_map[key]
        c = c_map[key]
        records.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "family": family,
                "anchor_pp": float(a["pearson_pert"]),
                "candidate_pp": float(c["pearson_pert"]),
                "anchor_mmd": float(a["test_mmd"]),
                "candidate_mmd": float(c["test_mmd"]),
                "pp_delta": float(c["pearson_pert"]) - float(a["pearson_pert"]),
                "mmd_delta": float(c["test_mmd"]) - float(a["test_mmd"]),
            }
        )
    return records


def summarize_delta(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    pp = [float(r["pp_delta"]) for r in records]
    mmd = [float(r["mmd_delta"]) for r in records]
    dataset_means = []
    for ds in sorted({r["dataset"] for r in records}):
        vals = [float(r["pp_delta"]) for r in records if r["dataset"] == ds]
        dataset_means.append({"dataset": ds, "n": len(vals), "pp_mean": float(mean(vals))})
    return {
        "n": len(records),
        "pp_mean": float(mean(pp)),
        "pp_min": float(min(pp)),
        "pp_hard_harm_frac": float(sum(x < -0.005 for x in pp) / len(pp)),
        "mmd_mean": float(mean(mmd)),
        "mmd_max": float(max(mmd)),
        "dataset_min_pp_mean": float(min(row["pp_mean"] for row in dataset_means)),
        "dataset_rows": dataset_means,
    }


def make_route(gene_choice: str, drug_choice: str, run_records: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    route_records: list[dict[str, Any]] = []
    if gene_choice != "anchor":
        route_records.extend(run_records[gene_choice]["family_gene"])
    else:
        for base_run in next(iter(run_records.values()))["family_gene"]:
            row = dict(base_run)
            row["candidate_pp"] = row["anchor_pp"]
            row["candidate_mmd"] = row["anchor_mmd"]
            row["pp_delta"] = 0.0
            row["mmd_delta"] = 0.0
            route_records.append(row)
    if drug_choice != "anchor":
        route_records.extend(run_records[drug_choice]["family_drug"])
    else:
        for base_run in next(iter(run_records.values()))["family_drug"]:
            row = dict(base_run)
            row["candidate_pp"] = row["anchor_pp"]
            row["candidate_mmd"] = row["anchor_mmd"]
            row["pp_delta"] = 0.0
            row["mmd_delta"] = 0.0
            route_records.append(row)
    gene = [r for r in route_records if r["family"] == "family_gene"]
    drug = [r for r in route_records if r["family"] == "family_drug"]
    all_summary = summarize_delta(route_records)
    gene_summary = summarize_delta(gene)
    drug_summary = summarize_delta(drug)
    gate = {
        "test_all_pp_delta_min": 0.005,
        "family_gene_pp_delta_min": -0.005,
        "family_drug_pp_delta_min": 0.005,
        "mmd_delta_max": 0.002,
        "dataset_min_pp_mean_min": -0.020,
    }
    pass_gate = (
        all_summary["pp_mean"] >= gate["test_all_pp_delta_min"]
        and gene_summary["pp_mean"] >= gate["family_gene_pp_delta_min"]
        and drug_summary["pp_mean"] >= gate["family_drug_pp_delta_min"]
        and max(all_summary["mmd_max"], gene_summary["mmd_max"], drug_summary["mmd_max"]) <= gate["mmd_delta_max"]
        and min(gene_summary["dataset_min_pp_mean"], drug_summary["dataset_min_pp_mean"]) >= gate["dataset_min_pp_mean_min"]
    )
    return {
        "gene_choice": gene_choice,
        "drug_choice": drug_choice,
        "all": all_summary,
        "family_gene": gene_summary,
        "family_drug": drug_summary,
        "pass_upper_bound_gate": bool(pass_gate),
        "gate": gate,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Modality Router Upper-Bound Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only upper-bound from existing train-only/internal condition metrics.",
        "- Tests a modality router that can choose anchor for gene conditions and one allmod checkpoint for drug conditions.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C held-out query.",
        "- A pass here is not a deployable claim; it only authorizes route implementation/controls as the next CPU step.",
        "",
        "## Top Routes",
        "",
        "| gene choice | drug choice | pass | all pp | gene pp | drug pp | drug dataset min | drug hard-harm frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["routes"][:8]:
        lines.append(
            f"| `{row['gene_choice']}` | `{row['drug_choice']}` | `{row['pass_upper_bound_gate']}` | "
            f"{row['all']['pp_mean']:+.6f} | {row['family_gene']['pp_mean']:+.6f} | "
            f"{row['family_drug']['pp_mean']:+.6f} | {row['family_drug']['dataset_min_pp_mean']:+.6f} | "
            f"{row['family_drug']['pp_hard_harm_frac']:.3f} |"
        )
    best = payload["best_route"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Upper-bound route candidate: gene `{best['gene_choice']}`, drug `{best['drug_choice']}`",
            f"- Upper-bound gate pass: `{best['pass_upper_bound_gate']}`",
            f"- GPU authorized now: `{payload['gpu_authorized']}`",
            f"- reasons: `{payload['reasons']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    decision = load_json(DECISION_JSON)
    run_names = [str(row["run_name"]) for row in decision.get("rows", [])]
    run_records = {
        run_name: {
            "family_gene": family_records(run_name, "family_gene"),
            "family_drug": family_records(run_name, "family_drug"),
        }
        for run_name in run_names
    }
    choices = ["anchor"] + run_names
    routes = [make_route(gene, drug, run_records) for gene in choices for drug in choices]
    routes = sorted(
        routes,
        key=lambda r: (
            bool(r["pass_upper_bound_gate"]),
            float(r["all"]["pp_mean"]),
            float(r["family_drug"]["pp_mean"]),
            -float(r["family_gene"]["pp_hard_harm_frac"]),
        ),
        reverse=True,
    )
    best = routes[0]
    status = (
        "allmodality_modality_router_upper_bound_pass_controls_needed"
        if best["pass_upper_bound_gate"]
        else "allmodality_modality_router_upper_bound_fail"
    )
    reasons = [
        "upper_bound_uses_existing_internal_metrics_only",
        "route_implementation_and_controls_not_yet_done",
        "drug_condition_hard_harm_fraction_remains_high",
    ]
    payload = {
        "status": status,
        "gpu_authorized": False,
        "best_route": best,
        "routes": routes,
        "reasons": reasons,
        "next_action": (
            "implement CPU route-control gate next: verify modality routing logic, add type/count-only and dose/scaffold controls, "
            "and only then consider eval-only routed posthoc; no training GPU authorized from this upper bound alone"
        ),
        "boundary": {
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_used": False,
            "source": "existing_posthoc_condition_metrics",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
