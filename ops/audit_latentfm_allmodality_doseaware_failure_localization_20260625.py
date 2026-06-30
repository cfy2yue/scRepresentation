#!/usr/bin/env python3
"""CPU-only failure localization for all-modality dose-aware smokes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_smoke_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_allmodality_doseaware_failure_localization_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_ALLMODALITY_DOSEAWARE_FAILURE_LOCALIZATION_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_budget_seed(run_name: str) -> tuple[int | None, int | None]:
    budget = None
    seed = None
    m_budget = re.search(r"budget(\d+)", run_name)
    m_seed = re.search(r"seed(\d+)", run_name)
    if m_budget:
        budget = int(m_budget.group(1))
    if m_seed:
        seed = int(m_seed.group(1))
    return budget, seed


def group_delta(row: dict[str, Any], group: str, metric: str) -> float:
    groups = row.get("groups") or {}
    return float((groups.get(group) or {}).get(metric, 0.0))


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    budget, seed = parse_budget_seed(str(row["run_name"]))
    all_pp = group_delta(row, "family:test_all", "delta_pearson_pert")
    gene_pp = group_delta(row, "family:family_gene", "delta_pearson_pert")
    drug_pp = group_delta(row, "family:family_drug", "delta_pearson_pert")
    all_mmd = group_delta(row, "family:test_all", "delta_mmd")
    gene_mmd = group_delta(row, "family:family_gene", "delta_mmd")
    drug_mmd = group_delta(row, "family:family_drug", "delta_mmd")
    signs = {
        "all_positive": all_pp >= 0.005,
        "gene_safe": gene_pp >= -0.005,
        "drug_positive": drug_pp >= 0.005,
        "mmd_safe": max(all_mmd, gene_mmd, drug_mmd) <= 0.002,
    }
    if signs["all_positive"] and signs["gene_safe"] and not signs["drug_positive"]:
        mechanism = "gene_all_positive_drug_negative"
    elif signs["drug_positive"] and not signs["gene_safe"]:
        mechanism = "drug_positive_gene_harm"
    elif all_pp < 0 and gene_pp < 0 and drug_pp < 0:
        mechanism = "global_pp_harm"
    else:
        mechanism = "mixed_or_low_signal"
    return {
        "run_name": row["run_name"],
        "budget": budget,
        "seed": seed,
        "status": row.get("status"),
        "all_pp_delta": all_pp,
        "gene_pp_delta": gene_pp,
        "drug_pp_delta": drug_pp,
        "all_mmd_delta": all_mmd,
        "gene_mmd_delta": gene_mmd,
        "drug_mmd_delta": drug_mmd,
        "mmd_max_delta": max(all_mmd, gene_mmd, drug_mmd),
        "signs": signs,
        "mechanism": mechanism,
        "reasons": row.get("reasons", []),
    }


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_mmd_safe = sum(1 for row in rows if row["signs"]["mmd_safe"])
    n_all_pos = sum(1 for row in rows if row["signs"]["all_positive"])
    n_gene_safe = sum(1 for row in rows if row["signs"]["gene_safe"])
    n_drug_pos = sum(1 for row in rows if row["signs"]["drug_positive"])
    n_full_pass_like = sum(
        1
        for row in rows
        if row["signs"]["all_positive"] and row["signs"]["gene_safe"] and row["signs"]["drug_positive"] and row["signs"]["mmd_safe"]
    )
    mechanisms: dict[str, int] = {}
    for row in rows:
        mechanisms[row["mechanism"]] = mechanisms.get(row["mechanism"], 0) + 1

    reasons: list[str] = []
    if n_full_pass_like == 0:
        reasons.append("no_arm_passes_all_gene_drug_gate")
    if n_mmd_safe == n:
        reasons.append("mmd_not_primary_failure")
    if mechanisms.get("gene_all_positive_drug_negative", 0) > 0 and mechanisms.get("drug_positive_gene_harm", 0) > 0:
        reasons.append("budget_dependent_gene_drug_tradeoff")
    if n_drug_pos == 1 and n_gene_safe < n:
        reasons.append("drug_signal_not_seed_or_budget_stable")
    if n_all_pos < 2:
        reasons.append("overall_signal_not_stable")

    gpu_authorized = False
    status = "allmodality_doseaware_fail_close_non_nested_no_gpu"
    next_cpu_gate = (
        "build family-stratified chemical/gene tradeoff gate before any mutation; "
        "candidate mutations must show predicted gene no-harm and drug improvement without using canonical multi or Track C query"
    )

    return {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "summary_counts": {
            "n_runs": n,
            "n_mmd_safe": n_mmd_safe,
            "n_all_pp_positive": n_all_pos,
            "n_gene_safe": n_gene_safe,
            "n_drug_pp_positive": n_drug_pos,
            "n_full_pass_like": n_full_pass_like,
            "mechanisms": mechanisms,
        },
        "reasons": reasons,
        "next_cpu_gate": next_cpu_gate,
        "mutation_boundary": {
            "direct_nested_allmod_gpu_authorized": False,
            "same_smoke_relaunch_authorized": False,
            "allowed_future_mutation_requires_cpu_gate": True,
            "candidate_families_to_audit": [
                "family-stratified replay or sampler that protects gene family while testing drug signal",
                "chemical-only protocol with explicit gene no-harm sentinel",
                "dose/scaffold shuffle controls before descriptor/dose claims",
            ],
        },
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Failure Localization",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of the manifest-fixed all-modality dose-aware smoke decision.",
        "- Uses train-only/internal family and split metrics only.",
        "- Does not read canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Localization",
        "",
        "| run | budget | seed | all pp | gene pp | drug pp | max MMD | mechanism |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{run_name}` | {budget} | {seed} | {all_pp_delta:+.6f} | {gene_pp_delta:+.6f} | {drug_pp_delta:+.6f} | {mmd_max_delta:+.6f} | `{mechanism}` |".format(
                **row
            )
        )
    counts = payload["summary_counts"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- full pass-like arms: `{counts['n_full_pass_like']}/{counts['n_runs']}`",
            f"- MMD-safe arms: `{counts['n_mmd_safe']}/{counts['n_runs']}`",
            f"- all-pp positive arms: `{counts['n_all_pp_positive']}/{counts['n_runs']}`",
            f"- drug-pp positive arms: `{counts['n_drug_pp_positive']}/{counts['n_runs']}`",
            f"- mechanisms: `{counts['mechanisms']}`",
            f"- reasons: `{payload['reasons']}`",
            "",
            "## Interpretation",
            "",
            "The valid posthoc result closes the non-nested all-modality smoke as a direct GPU path. "
            "MMD is largely safe, so the failure is not distributional collapse; it is a family-specific pp tradeoff. "
            "Budget64 seed42 improves all/gene but harms drug, while budget32 seed42 improves drug but harms gene/all. "
            "This supports a modality-tradeoff hypothesis, not naive all-modality scaling.",
            "",
            "## Next Gate",
            "",
            f"`{payload['next_cpu_gate']}`",
            "",
            "Future GPU mutations are blocked until a CPU gate predicts both gene no-harm and drug-family improvement with explicit controls.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    decision = load_json(IN_JSON)
    rows = [summarize_row(row) for row in decision.get("rows", [])]
    payload = {
        **decide(rows),
        "input_decision": str(IN_JSON),
        "rows": rows,
        "boundary": {
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_used": False,
            "eval_boundary": "train_only_internal_allmodality_doseaware",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": payload["gpu_authorized"], "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
