#!/usr/bin/env python3
"""Build figure-ready data tables for the LatentFM scaling/failure-map package.

CPU-only. Reads completed reports and extracts frozen summary values into flat
CSV tables. It does not read model checkpoints, canonical multi, held-out Track
C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "scaling_figure_data_20260625"
OUT_JSON = REPORTS / "latentfm_scaling_figure_data_package_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_FIGURE_DATA_PACKAGE_20260625.md"

INPUTS = {
    "s0": REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.json",
    "truecell_nested_3k": REPORTS / "latentfm_true_cell_count_nested_matrix_decision_20260624.json",
    "truecell_budget64_6k": REPORTS / "latentfm_true_cell_count_budget64_tail_stability_6k_decision_20260625.json",
    "truecell_budget128_6k": REPORTS / "latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json",
    "truecell_budget128_noharm": REPORTS / "latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json",
    "count_smokes": REPORTS / "latentfm_xverse_scaling_count_smokes_decision_20260624.json",
    "protocol_matrix": REPORTS / "latentfm_scaling_protocol_matrix_decision_20260624.json",
    "axis_claim_matrix": REPORTS / "latentfm_scaling_axis_claim_matrix_20260625.csv",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True}
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def ci_str(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    ci = obj.get("ci95")
    return "" if ci is None else json.dumps(ci)


def tail_value(row: dict[str, Any], key: str) -> Any:
    tail = row.get(key) or {}
    return tail.get("negative_tail_lt_minus_0p020")


def truecell_rows() -> list[dict[str, Any]]:
    out = []
    sources = [
        ("3k_nested", 3000, INPUTS["truecell_nested_3k"]),
        ("6k_budget64", 6000, INPUTS["truecell_budget64_6k"]),
        ("6k_budget128", 6000, INPUTS["truecell_budget128_6k"]),
    ]
    for label, steps, path in sources:
        data = load_json(path)
        for row in (data.get("matrix_summary") or {}).get("budget_rows", []):
            out.append(
                {
                    "series": label,
                    "steps": steps,
                    "budget": row.get("budget"),
                    "seed_passes": row.get("seed_passes"),
                    "n_complete": row.get("n_complete"),
                    "cross_pp_mean": row.get("cross_background_pp_delta_mean"),
                    "cross_pp_ci95": ci_str(row.get("cross_background_pp_condition_bootstrap")),
                    "cross_pp_negative_tails": tail_value(row, "cross_background_pp_dataset_tail"),
                    "family_pp_mean": row.get("family_gene_pp_delta_mean"),
                    "family_pp_ci95": ci_str(row.get("family_gene_pp_condition_bootstrap")),
                    "family_pp_negative_tails": tail_value(row, "family_gene_pp_dataset_tail"),
                    "family_mmd_mean": row.get("family_gene_mmd_delta_mean"),
                    "status": data.get("status"),
                    "source_report": str(path),
                }
            )
    return out


def exposure_rows() -> list[dict[str, Any]]:
    out = []
    count = load_json(INPUTS["count_smokes"])
    for row in count.get("rows", []):
        groups = row.get("groups") or {}
        cross = groups.get("internal_val_cross_background_seen_gene_proxy") or {}
        family = groups.get("internal_val_family_gene_proxy") or {}
        out.append(
            {
                "source_family": "count_smokes",
                "arm": row.get("arm"),
                "run": row.get("name"),
                "role": "condition_or_exposure",
                "cross_pp_delta": cross.get("delta_pearson_pert"),
                "cross_mmd_delta": cross.get("delta_mmd"),
                "family_pp_delta": family.get("delta_pearson_pert"),
                "family_mmd_delta": family.get("delta_mmd"),
                "status": row.get("status"),
                "source_report": str(INPUTS["count_smokes"]),
            }
        )
    protocol = load_json(INPUTS["protocol_matrix"])
    for row in protocol.get("rows", []):
        m = row.get("metrics") or {}
        out.append(
            {
                "source_family": "protocol_matrix",
                "arm": row.get("arm"),
                "run": row.get("name"),
                "role": row.get("role"),
                "cross_pp_delta": m.get("cross_pp_delta_vs_anchor"),
                "cross_mmd_delta": "",
                "family_pp_delta": m.get("family_gene_pp_delta_vs_anchor"),
                "family_mmd_delta": m.get("family_gene_mmd_delta_vs_anchor"),
                "status": row.get("status"),
                "source_report": str(INPUTS["protocol_matrix"]),
            }
        )
    return out


def noharm_rows() -> list[dict[str, Any]]:
    data = load_json(INPUTS["truecell_budget128_noharm"])
    out = []
    metric_keys = [
        "cross_background_seen_gene:pearson_pert",
        "all_test_single:pearson_pert",
        "family_gene:pearson_pert",
        "family_gene:test_mmd_clamped",
    ]
    for row in data.get("rows", []):
        metrics = row.get("metrics") or {}
        for key in metric_keys:
            m = metrics.get(key) or {}
            out.append(
                {
                    "seed": row.get("seed"),
                    "run": row.get("run"),
                    "metric": key,
                    "delta_mean": m.get("delta_mean"),
                    "ci95": json.dumps(m.get("ci95")) if m.get("ci95") is not None else "",
                    "p_harm": m.get("p_harm"),
                    "p_improve": m.get("p_improve"),
                    "gate_status": row.get("gate_status"),
                    "gate_reasons": ";".join(row.get("gate_reasons") or []),
                    "source_report": str(INPUTS["truecell_budget128_noharm"]),
                }
            )
    return out


def s0_rows() -> list[dict[str, Any]]:
    data = load_json(INPUTS["s0"])
    summary = data.get("summary") or {}
    out = []
    for key in ["n_rows", "n_datasets", "n_source_verified", "n_s0_resolved"]:
        out.append({"category": "summary", "name": key, "value": summary.get(key)})
    for category_key in ["modality_counts", "perturbation_type_counts", "missing_or_unresolved_reasons"]:
        for name, value in sorted((summary.get(category_key) or {}).items()):
            out.append({"category": category_key, "name": name, "value": value})
    return out


def failure_map_rows() -> list[dict[str, Any]]:
    rows = []
    if not INPUTS["axis_claim_matrix"].is_file():
        return rows
    with INPUTS["axis_claim_matrix"].open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "s0_provenance_summary_csv": OUT_DIR / "s0_provenance_summary.csv",
        "truecell_budget_curve_csv": OUT_DIR / "truecell_budget_curve.csv",
        "condition_exposure_curve_csv": OUT_DIR / "condition_exposure_curve.csv",
        "canonical_noharm_veto_csv": OUT_DIR / "canonical_noharm_veto.csv",
        "failure_map_axis_summary_csv": OUT_DIR / "failure_map_axis_summary.csv",
    }

    s0 = s0_rows()
    truecell = truecell_rows()
    exposure = exposure_rows()
    noharm = noharm_rows()
    failure = failure_map_rows()

    write_csv(outputs["s0_provenance_summary_csv"], s0, ["category", "name", "value"])
    write_csv(
        outputs["truecell_budget_curve_csv"],
        truecell,
        [
            "series",
            "steps",
            "budget",
            "seed_passes",
            "n_complete",
            "cross_pp_mean",
            "cross_pp_ci95",
            "cross_pp_negative_tails",
            "family_pp_mean",
            "family_pp_ci95",
            "family_pp_negative_tails",
            "family_mmd_mean",
            "status",
            "source_report",
        ],
    )
    write_csv(
        outputs["condition_exposure_curve_csv"],
        exposure,
        [
            "source_family",
            "arm",
            "run",
            "role",
            "cross_pp_delta",
            "cross_mmd_delta",
            "family_pp_delta",
            "family_mmd_delta",
            "status",
            "source_report",
        ],
    )
    write_csv(
        outputs["canonical_noharm_veto_csv"],
        noharm,
        ["seed", "run", "metric", "delta_mean", "ci95", "p_harm", "p_improve", "gate_status", "gate_reasons", "source_report"],
    )
    write_csv(
        outputs["failure_map_axis_summary_csv"],
        failure,
        ["axis", "claim_level", "support", "boundary", "next_gate", "manuscript_use", "promotion_allowed"],
    )

    payload = {
        "status": "scaling_figure_data_package_ready_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "counts": {
            "s0_rows": len(s0),
            "truecell_rows": len(truecell),
            "condition_exposure_rows": len(exposure),
            "canonical_noharm_rows": len(noharm),
            "failure_map_axis_rows": len(failure),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Figure Data Package",
        "",
        "Status: `scaling_figure_data_package_ready_no_gpu`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only extraction of figure-ready data from completed reports.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            f"- JSON: `{OUT_JSON}`",
            "",
            "## Counts",
            "",
        ]
    )
    for key, value in payload["counts"].items():
        lines.append(f"- {key}: `{value}`")
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
