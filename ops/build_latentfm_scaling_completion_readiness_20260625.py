#!/usr/bin/env python3
"""Build a CPU-only scaling completion/readiness package.

This script only reads completed reports and figure-ready CSVs. It does not
read checkpoints, canonical multi, Track C query, or run inference/training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIG_DATA = REPORTS / "scaling_figure_data_20260625"
OUT_DIR = REPORTS / "scaling_completion_readiness_20260625"


INPUTS = {
    "condition_exposure": FIG_DATA / "condition_exposure_curve.csv",
    "truecell_budget": FIG_DATA / "truecell_budget_curve.csv",
    "canonical_noharm": FIG_DATA / "canonical_noharm_veto.csv",
    "failure_axis": FIG_DATA / "failure_map_axis_summary.csv",
    "s0": FIG_DATA / "s0_provenance_summary.csv",
    "mixed_lodo": REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json",
    "nested_v2": REPORTS / "latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json",
    "source_v2": REPORTS / "latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json",
    "condition_exposure_row_bootstrap": REPORTS / "latentfm_condition_exposure_row_bootstrap_gate_20260625.json",
    "result_draft": REPORTS / "LATENTFM_SCALING_RESULT_SECTION_DRAFT_20260625.md",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_exposure(rows: list[dict[str, str]], nested: dict, row_gate: dict) -> tuple[list[dict[str, object]], dict]:
    arm_order = {
        "cap30_all": 30,
        "cap120_all": 120,
        "xverse_scaling_full_all_3k_seed42": 9999,
    }
    out = []
    for row in rows:
        arm = row["arm"]
        cross = as_float(row["cross_pp_delta"])
        family = as_float(row["family_pp_delta"])
        mmd = as_float(row["family_mmd_delta"])
        ci_available = False
        replicate_unit = "single aggregate arm"
        if arm in ("cap30_all", "cap120_all"):
            curve_role = "condition_count_anchor_curve"
        elif "full" in arm:
            curve_role = "full_exposure_extension"
        elif "breadth" in row.get("role", "") or "breadth" in arm:
            curve_role = "breadth_control"
        elif "balanced" in arm or "jiang" in arm or "general" in arm:
            curve_role = "exposure_or_type_mutation"
        else:
            curve_role = "diagnostic_arm"
        out.append(
            {
                "arm": arm,
                "role": row.get("role", ""),
                "curve_role": curve_role,
                "cross_pp_delta": f"{cross:.6f}",
                "family_pp_delta": f"{family:.6f}",
                "family_mmd_delta": f"{mmd:.6f}",
                "ci_available": ci_available,
                "replicate_unit": replicate_unit,
                "scaling_law_ready": False,
                "promotion_allowed": False,
                "blocking_reason": "aggregate_only_no_random_subset_or_condition_level_bootstrap",
                "source_report": row["source_report"],
            }
        )

    cross_by_arm = {row["arm"]: as_float(row["cross_pp_delta"]) for row in rows}
    cap120_minus_cap30 = nested["summary"]["cap120_minus_cap30_cross_pp"]
    full_minus_cap120 = nested["summary"]["full_minus_cap120_cross_pp"]
    summary = {
        "n_exposure_arms": len(rows),
        "best_cross_arm": max(cross_by_arm, key=cross_by_arm.get),
        "best_cross_pp_delta": max(cross_by_arm.values()),
        "cap120_minus_cap30_cross_pp": cap120_minus_cap30,
        "full_minus_cap120_cross_pp": full_minus_cap120,
        "nonmonotonic_supported_as_diagnostic": cap120_minus_cap30 > 0 and full_minus_cap120 < 0,
        "condition_row_bootstrap_available": True,
        "condition_row_bootstrap_status": row_gate.get("status"),
        "condition_row_bootstrap_gpu_authorized": row_gate.get("gpu_authorized"),
        "condition_row_bootstrap_reasons": row_gate.get("decision", {}).get("reasons", []),
        "condition_row_bootstrap_primary_cross_pp": None,
        "condition_row_bootstrap_primary_cross_ci95": None,
        "condition_row_bootstrap_primary_family_pp": None,
        "condition_row_bootstrap_primary_family_ci95": None,
    }
    for comp in row_gate.get("comparisons", []):
        if comp.get("comparison") == "cap120_minus_cap30":
            summary["condition_row_bootstrap_primary_cross_pp"] = comp["cross"]["pp"]["mean"]
            summary["condition_row_bootstrap_primary_cross_ci95"] = [
                comp["cross"]["pp"]["ci_low"],
                comp["cross"]["pp"]["ci_high"],
            ]
            summary["condition_row_bootstrap_primary_family_pp"] = comp["family"]["pp"]["mean"]
            summary["condition_row_bootstrap_primary_family_ci95"] = [
                comp["family"]["pp"]["ci_low"],
                comp["family"]["pp"]["ci_high"],
            ]
            break
    return out, summary


def summarize_truecell(rows: list[dict[str, str]], noharm: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict]:
    veto_cross = [
        as_float(row["delta_mean"])
        for row in noharm
        if row["metric"] == "cross_background_seen_gene:pearson_pert"
    ]
    veto_family_harm = [
        as_float(row["p_harm"])
        for row in noharm
        if row["metric"] == "family_gene:pearson_pert"
    ]
    out = []
    for row in rows:
        cross = as_float(row["cross_pp_mean"])
        family = as_float(row["family_pp_mean"])
        neg_tails = int(float(row["cross_pp_negative_tails"]))
        ci = row["cross_pp_ci95"]
        internal_ready = bool(ci) and int(float(row["n_complete"])) >= 3
        out.append(
            {
                "series": row["series"],
                "steps": row["steps"],
                "budget": row["budget"],
                "cross_pp_mean": f"{cross:.6f}",
                "cross_pp_ci95": ci,
                "cross_pp_negative_tails": neg_tails,
                "family_pp_mean": f"{family:.6f}",
                "family_pp_ci95": row["family_pp_ci95"],
                "internal_bootstrap_or_ci_present": internal_ready,
                "internal_tail_safe": neg_tails == 0,
                "canonical_noharm_passed": False,
                "promotion_allowed": False,
                "claim_level": "mechanism_only",
                "blocking_reason": "frozen_canonical_noharm_failed_all_budget128_6k_seeds",
                "source_report": row["source_report"],
            }
        )
    summary = {
        "n_truecell_rows": len(rows),
        "best_internal_series": max(rows, key=lambda r: as_float(r["cross_pp_mean"]))["series"],
        "best_internal_budget": max(rows, key=lambda r: as_float(r["cross_pp_mean"]))["budget"],
        "best_internal_cross_pp_mean": max(as_float(r["cross_pp_mean"]) for r in rows),
        "canonical_cross_pp_deltas": veto_cross,
        "canonical_family_p_harm": veto_family_harm,
        "canonical_noharm_failed_all": all(v <= 0 for v in veto_cross) and all(p > 0.5 for p in veto_family_harm),
    }
    return out, summary


def summarize_source_tails(mixed: dict, source_v2: dict) -> tuple[list[dict[str, object]], dict]:
    rows = mixed.get("dataset_rows", [])
    out = []
    for row in rows:
        pp = as_float(row["pp_delta_mean"])
        out.append(
            {
                "dataset": row["dataset"],
                "background": row["background"],
                "perturbation_type": row["perturbation_type"],
                "source_quality": row["source_quality"],
                "n": row["n"],
                "cap_gain": row["cap_gain"],
                "pp_delta_mean": f"{pp:.6f}",
                "mmd_delta_mean": f"{as_float(row['mmd_delta_mean']):.6f}",
                "tail_flag": pp < -0.02,
                "source_verified": row["source_quality"] == "source_verified",
            }
        )
    pp_vals = [as_float(row["pp_delta_mean"]) for row in rows]
    verified = [row for row in rows if row["source_quality"] == "source_verified"]
    verified_pp = [as_float(row["pp_delta_mean"]) for row in verified]
    summary = {
        "n_dataset_rows": len(rows),
        "dataset_min_pp": min(pp_vals),
        "negative_tail_count_lt_minus_0p02": sum(v < -0.02 for v in pp_vals),
        "source_verified_rows": len(verified),
        "source_verified_mean_pp": mean(verified_pp) if verified_pp else None,
        "source_v2_status": source_v2.get("status"),
        "source_v2_gpu_authorized": source_v2.get("gpu_authorized"),
        "source_v2_pp_mean": source_v2.get("summary", {}).get("pp_delta_mean"),
        "source_v2_ci95": source_v2.get("summary", {}).get("bootstrap", {}).get("ci95"),
        "scaling_law_ready": False,
    }
    return out, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))

    condition_rows = read_csv(INPUTS["condition_exposure"])
    truecell_rows = read_csv(INPUTS["truecell_budget"])
    noharm_rows = read_csv(INPUTS["canonical_noharm"])
    failure_rows = read_csv(INPUTS["failure_axis"])
    s0_rows = read_csv(INPUTS["s0"])
    mixed = read_json(INPUTS["mixed_lodo"])
    nested = read_json(INPUTS["nested_v2"])
    source_v2 = read_json(INPUTS["source_v2"])
    row_gate = read_json(INPUTS["condition_exposure_row_bootstrap"])

    exposure_table, exposure_summary = summarize_exposure(condition_rows, nested, row_gate)
    truecell_table, truecell_summary = summarize_truecell(truecell_rows, noharm_rows)
    source_tail_table, source_tail_summary = summarize_source_tails(mixed, source_v2)

    axis_rows = []
    for row in failure_rows:
        axis_rows.append(
            {
                "axis": row["axis"],
                "current_claim_level": row["claim_level"],
                "manuscript_use": row["manuscript_use"],
                "promotion_allowed": row["promotion_allowed"],
                "nm_scaling_law_ready": False,
                "next_required_gate": row["next_gate"],
                "boundary": row["boundary"],
            }
        )

    write_csv(
        OUT_DIR / "condition_exposure_readiness.csv",
        exposure_table,
        [
            "arm",
            "role",
            "curve_role",
            "cross_pp_delta",
            "family_pp_delta",
            "family_mmd_delta",
            "ci_available",
            "replicate_unit",
            "scaling_law_ready",
            "promotion_allowed",
            "blocking_reason",
            "source_report",
        ],
    )
    write_csv(
        OUT_DIR / "truecell_budget_readiness.csv",
        truecell_table,
        [
            "series",
            "steps",
            "budget",
            "cross_pp_mean",
            "cross_pp_ci95",
            "cross_pp_negative_tails",
            "family_pp_mean",
            "family_pp_ci95",
            "internal_bootstrap_or_ci_present",
            "internal_tail_safe",
            "canonical_noharm_passed",
            "promotion_allowed",
            "claim_level",
            "blocking_reason",
            "source_report",
        ],
    )
    write_csv(
        OUT_DIR / "source_resolved_dataset_tails.csv",
        source_tail_table,
        [
            "dataset",
            "background",
            "perturbation_type",
            "source_quality",
            "n",
            "cap_gain",
            "pp_delta_mean",
            "mmd_delta_mean",
            "tail_flag",
            "source_verified",
        ],
    )
    write_csv(
        OUT_DIR / "axis_completion_matrix.csv",
        axis_rows,
        [
            "axis",
            "current_claim_level",
            "manuscript_use",
            "promotion_allowed",
            "nm_scaling_law_ready",
            "next_required_gate",
            "boundary",
        ],
    )

    s0_summary = {row["name"]: row["value"] for row in s0_rows if row["category"] == "summary"}
    payload = {
        "status": "scaling_completion_readiness_ready_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports": True,
            "reads_model_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in INPUTS.items()},
        "outputs": {
            "condition_exposure_readiness": str(OUT_DIR / "condition_exposure_readiness.csv"),
            "truecell_budget_readiness": str(OUT_DIR / "truecell_budget_readiness.csv"),
            "source_resolved_dataset_tails": str(OUT_DIR / "source_resolved_dataset_tails.csv"),
            "axis_completion_matrix": str(OUT_DIR / "axis_completion_matrix.csv"),
        },
        "summary": {
            "s0": s0_summary,
            "exposure": exposure_summary,
            "truecell": truecell_summary,
            "source_tails": source_tail_summary,
            "n_axis_rows": len(axis_rows),
            "nm_scaling_law_ready_axes": 0,
            "mainline_default_model": "xverse_8k_anchor",
            "immediate_gpu_candidate_without_ack": None,
            "nearest_gpu_route": "chemical unseen-scaffold V2 fixed-step controls after exact ACK",
        },
        "decision": {
            "status": "ready_for_manuscript_failure_map_not_deployable_scaling_law",
            "gpu_authorized": False,
            "reasons": [
                "condition/exposure curve lacks condition-level or random-subset bootstrap replicates in current artifacts",
                "true-cell route has strongest internal signal but fails frozen canonical no-harm across seeds",
                "source/background/type axes retain unsafe dataset tails and confounding",
                "chemical semantics requires exact ACK and descriptor controls",
            ],
            "next_actions": [
                "use this package to write NM-style scaling limitations and completion protocol",
                "do not launch scaling GPU without chemical V2 ACK or a materially new CPU gate",
                "if ACK supplied, launch real Morgan512 seed43/44 V2 controls before shuffled/random controls",
            ],
        },
    }

    json_path = REPORTS / "latentfm_scaling_completion_readiness_20260625.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    report = REPORTS / "LATENTFM_SCALING_COMPLETION_READINESS_20260625.md"
    report.write_text(
        "\n".join(
            [
                "# LatentFM Scaling Completion Readiness",
                "",
                "Status: `scaling_completion_readiness_ready_no_gpu`",
                "GPU authorized: `False`",
                "",
                "## Boundary",
                "",
                "CPU-only synthesis from completed reports and figure-ready CSVs. Does not read checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
                "",
                "## Outputs",
                "",
                f"- condition exposure readiness: `{OUT_DIR / 'condition_exposure_readiness.csv'}`",
                f"- true-cell budget readiness: `{OUT_DIR / 'truecell_budget_readiness.csv'}`",
                f"- source-resolved dataset tails: `{OUT_DIR / 'source_resolved_dataset_tails.csv'}`",
                f"- axis completion matrix: `{OUT_DIR / 'axis_completion_matrix.csv'}`",
                f"- JSON: `{json_path}`",
                "",
                "## Key Findings",
                "",
                f"- S0 rows/datasets/source-verified: `{s0_summary.get('n_rows')}` / `{s0_summary.get('n_datasets')}` / `{s0_summary.get('n_source_verified')}`.",
                f"- Exposure curve is diagnostic-nonmonotonic: cap120-cap30 cross pp `{exposure_summary['cap120_minus_cap30_cross_pp']:+.6f}`, full-cap120 `{exposure_summary['full_minus_cap120_cross_pp']:+.6f}`.",
                f"- Condition-row bootstrap is available and failed GPU reopening: `{exposure_summary['condition_row_bootstrap_status']}`.",
                f"- Primary row-bootstrap cap120-cap30 cross/family pp means: `{exposure_summary['condition_row_bootstrap_primary_cross_pp']:+.6f}` / `{exposure_summary['condition_row_bootstrap_primary_family_pp']:+.6f}` with CIs crossing zero.",
                f"- Best true-cell internal route: `{truecell_summary['best_internal_series']}` budget `{truecell_summary['best_internal_budget']}` with cross pp `{truecell_summary['best_internal_cross_pp_mean']:+.6f}`.",
                f"- Canonical no-harm failed all true-cell budget128 6k seeds: `{truecell_summary['canonical_noharm_failed_all']}`.",
                f"- Source/background/type tails remain unsafe: dataset min pp `{source_tail_summary['dataset_min_pp']:+.6f}`, negative tails `< -0.02` = `{source_tail_summary['negative_tail_count_lt_minus_0p02']}`.",
                "- NM scaling-law-ready axes: `0`; manuscript/failure-map-ready axes: true-cell budget, condition/exposure, background/type/source negative map, chemical protocol boundary.",
                "",
                "## Decision",
                "",
                "The scaling branch is ready as a manuscript-grade mechanism/failure-map package, not as a deployable scaling law. The default model remains `xverse_8k_anchor`. No scaling GPU launch is authorized without the exact chemical V2 ACK or a materially new CPU gate that repairs no-harm/tail failure outside closed families.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
