#!/usr/bin/env python3
"""CPU-only provenance and estimand matrix gate for LatentFM scaling.

This gate joins existing dataset/source inventory, split/metainfo inventory,
and h5ad obs-column audit outputs. It does not train, infer, read active logs,
read canonical multi, or inspect held-out Track C query artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DATASET_INV = REPORTS / "latentfm_dataset_scaling_inventory_20260624.json"
META_INV = REPORTS / "latentfm_scaling_metainfo_inventory_20260624.json"
OBS_AUDIT = REPORTS / "latentfm_condition_level_metainfo_scaling_audit_20260624.json"
EVIDENCE_MATRIX = REPORTS / "latentfm_scaling_law_evidence_matrix_gate_20260624.json"

OUT_JSON = REPORTS / "latentfm_scaling_provenance_estimand_matrix_gate_20260624.json"
OUT_CSV = REPORTS / "latentfm_scaling_provenance_estimand_matrix_20260624.csv"
OUT_MD = REPORTS / "LATENTFM_SCALING_PROVENANCE_ESTIMAND_MATRIX_GATE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def axis_columns(obs_row: dict[str, Any], axis: str) -> list[str]:
    cols = []
    for item in (obs_row.get("candidate_columns") or {}).get(axis, []):
        col = str(item.get("column") or "")
        if col:
            cols.append(col)
    return sorted(set(cols))


def source_quality(status: str) -> str:
    if status.startswith("primary_"):
        return "source_verified"
    if "doi_identified" in status:
        return "doi_needs_spotcheck"
    if "candidate" in status or "needs" in status:
        return "needs_precise_mapping"
    return "unknown_or_unverified"


def main() -> int:
    dataset_inv = load_json(DATASET_INV)
    meta_inv = load_json(META_INV)
    obs_audit = load_json(OBS_AUDIT)
    evidence = load_json(EVIDENCE_MATRIX)

    source_by_dataset: dict[str, dict[str, Any]] = {}
    buckets_by_dataset: dict[str, set[str]] = defaultdict(set)
    multi_condition_counts: Counter[str] = Counter()
    for row in dataset_inv.get("rows", []):
        ds = str(row.get("dataset") or "")
        if not ds:
            continue
        buckets_by_dataset[ds].add(str(row.get("bucket") or "unknown"))
        if str(row.get("bucket") or "") == "multiple":
            multi_condition_counts[ds] += int(row.get("n_conditions_selected_bucket") or 0)
        existing = source_by_dataset.get(ds)
        if existing is None or source_quality(str(row.get("source_check_status") or "")) == "source_verified":
            source_by_dataset[ds] = row

    obs_by_dataset = {str(row.get("dataset")): row for row in obs_audit.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for meta in meta_inv.get("rows", []):
        ds = str(meta.get("dataset") or "")
        source = source_by_dataset.get(ds, {})
        obs = obs_by_dataset.get(ds, {})
        bg = str(meta.get("cell_line_meta") or source.get("cell_line") or "unknown")
        ptype = str(meta.get("perturbation_type") or source.get("perturbation_type") or "unknown")
        row = {
            "dataset": ds,
            "perturbation_type": ptype,
            "cell_background_source": bg,
            "modality": str(source.get("modality") or ("chemical" if ptype == "drug" else "gene")),
            "buckets": "+".join(sorted(buckets_by_dataset.get(ds, set()))) or "unknown",
            "n_cells_selected_total": int(source.get("n_cells_selected_total") or meta.get("n_cells_gt_stack") or 0),
            "n_conditions_selected_source": int(source.get("n_conditions_selected_bucket") or 0),
            "n_conditions_trainonly": int(meta.get("trainonly_crossbg_v2_train") or 0),
            "n_conditions_cap30": int(meta.get("cap30_all_v2_train") or 0),
            "n_conditions_cap120": int(meta.get("cap120_all_v2_train") or 0),
            "n_internal_cross": int(meta.get("trainonly_crossbg_v2_internal_cross") or 0),
            "n_internal_family": int(meta.get("trainonly_crossbg_v2_internal_family") or 0),
            "n_multi_conditions_selected": int(multi_condition_counts.get(ds, 0)),
            "obs_condition_columns": ";".join(axis_columns(obs, "condition")),
            "obs_cell_background_columns": ";".join(axis_columns(obs, "cell_background")),
            "obs_perturbation_type_columns": ";".join(axis_columns(obs, "perturbation_type")),
            "obs_dose_columns": ";".join(axis_columns(obs, "dose")),
            "obs_cell_type_n_unique": int(meta.get("obs_cell_type_n_unique") or 0),
            "source_label": str(source.get("source_label") or ""),
            "source_url": str(source.get("source_url") or ""),
            "source_check_status": str(source.get("source_check_status") or "missing_source_inventory"),
            "source_quality": source_quality(str(source.get("source_check_status") or "")),
        }
        rows.append(row)

    type_counts = Counter(r["perturbation_type"] for r in rows)
    bg_counts = Counter(r["cell_background_source"] for r in rows)
    quality_counts = Counter(r["source_quality"] for r in rows)
    modality_counts = Counter(r["modality"] for r in rows)
    datasets_with_obs_bg = [r["dataset"] for r in rows if r["obs_cell_background_columns"]]
    datasets_with_dose = [r["dataset"] for r in rows if r["obs_dose_columns"]]
    datasets_with_multi = [r["dataset"] for r in rows if r["n_multi_conditions_selected"] > 0]
    cap120_gt_cap30 = [r for r in rows if r["n_conditions_cap120"] > r["n_conditions_cap30"]]
    source_unverified = [r["dataset"] for r in rows if r["source_quality"] != "source_verified"]

    evidence_axes = {row.get("axis"): row for row in evidence.get("axes", [])}
    estimands = [
        {
            "axis": "condition_count",
            "status": "cpu_design_ready_not_gpu",
            "available_evidence": (
                f"{len(cap120_gt_cap30)}/{len(rows)} datasets have cap120 > cap30; "
                f"prior cap120-cap30 internal pp +0.009814 but canonical no-harm failed"
            ),
            "missing_for_nm": "matched steps/seeds/bootstrap plus no-harm transfer calibration",
            "next_gate": "mixed_effect_lodo_condition_count_gate",
            "gpu_trigger": ">=3 seed protocol with no sign flip, dataset-min pp >= -0.02, and no-harm surrogate not high-risk",
            "fail_close": "seed sign flip, tail harm, or surrogate high-risk keeps condition-count diagnostic only",
            "gpu_authorized": False,
        },
        {
            "axis": "dataset_background_breadth",
            "status": "confounded_cpu_only",
            "available_evidence": (
                f"{len(bg_counts)} source backgrounds, {len(datasets_with_obs_bg)} obs-background datasets; "
                f"evidence blocker: {evidence_axes.get('dataset_background_breadth', {}).get('blocker', 'unknown')}"
            ),
            "missing_for_nm": "source-verified matched backgrounds, leave-background/dataset-out estimand, shuffled dataset-ID control",
            "next_gate": "source_matched_lodo_background_gate",
            "gpu_trigger": "real matched background signal beats shuffled controls with safe tails",
            "fail_close": "controls do not collapse or background/type tails remain negative",
            "gpu_authorized": False,
        },
        {
            "axis": "perturbation_type_breadth",
            "status": "confounded_cpu_only",
            "available_evidence": (
                f"{dict(type_counts)}; prior type-balanced arm failed and source strata tails are negative"
            ),
            "missing_for_nm": "type-heldout comparisons within shared backgrounds plus shuffled type labels",
            "next_gate": "matched_type_background_negative_control_gate",
            "gpu_trigger": "type breadth remains positive within matched backgrounds and negative controls collapse",
            "fail_close": "CRISPRa/CRISPRko or drug tails remain negative",
            "gpu_authorized": False,
        },
        {
            "axis": "target_gene_coverage",
            "status": "diagnostic_cpu_only",
            "available_evidence": "prior coverage gate failed: rho -0.057855, permutation p 0.437281, min dataset pp -0.231049",
            "missing_for_nm": "nested target-coverage subsets with matched condition/background/type and per-dataset tail bootstrap",
            "next_gate": "target_coverage_causal_protocol_gate_v2",
            "gpu_trigger": "coverage coefficient CI > 0, permutation support, dataset-min pp >= -0.02",
            "fail_close": "rho near zero, non-significant permutation, or dataset-tail harm",
            "gpu_authorized": False,
        },
        {
            "axis": "perturbation_multiplicity",
            "status": "separate_trackc_required",
            "available_evidence": f"{len(datasets_with_multi)} datasets with selected multi buckets; selected multi conditions total {sum(multi_condition_counts.values())}",
            "missing_for_nm": "safe multi-aware train/support/query split audit and support-val-only route selection",
            "next_gate": "trackc_multi_split_provenance_audit",
            "gpu_trigger": "support-trainselect split passes leakage/count audit and canonical no-harm route is predeclared",
            "fail_close": "insufficient support coverage or any canonical no-harm fail before query",
            "gpu_authorized": False,
        },
        {
            "axis": "deployable_noharm_transfer",
            "status": "negative_cpu_surrogate_needed",
            "available_evidence": "10/10 internal-pass-like scaling candidates failed frozen canonical no-harm; 8/10 pp hard-harm",
            "missing_for_nm": "train-only high-risk surrogate with leave-family-out validation and negative controls",
            "next_gate": "noharm_surrogate_transfer_v2",
            "gpu_trigger": "surrogate identifies a materially new tail-safe candidate distinct from closed branches",
            "fail_close": "surrogate cannot separate high-risk candidates or has no positive safe examples",
            "gpu_authorized": False,
        },
    ]

    reasons = []
    if source_unverified:
        reasons.append("some_sources_need_precise_mapping_or_full_text_spotcheck")
    if len(datasets_with_obs_bg) < len(rows):
        reasons.append("obs_level_cell_background_incomplete")
    if not any(item["gpu_authorized"] for item in estimands):
        reasons.append("estimand_matrix_requires_cpu_gates_before_gpu")
    reasons.extend(
        [
            "canonical_noharm_transfer_currently_negative",
            "background_type_and_dataset_identity_confounded",
        ]
    )

    status = "scaling_provenance_estimand_matrix_ready_cpu_next_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_existing_metadata_reports_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "dataset_inventory": str(DATASET_INV),
            "metainfo_inventory": str(META_INV),
            "obs_audit": str(OBS_AUDIT),
            "evidence_matrix": str(EVIDENCE_MATRIX),
        },
        "summary": {
            "n_datasets": len(rows),
            "perturbation_type_counts": dict(type_counts),
            "source_background_counts": dict(bg_counts),
            "modality_counts": dict(modality_counts),
            "source_quality_counts": dict(quality_counts),
            "datasets_with_obs_cell_background": datasets_with_obs_bg,
            "datasets_with_dose": datasets_with_dose,
            "datasets_with_selected_multi": datasets_with_multi,
            "selected_multi_conditions_total": int(sum(multi_condition_counts.values())),
            "datasets_cap120_gt_cap30": len(cap120_gt_cap30),
            "source_unverified_or_needs_mapping": source_unverified,
        },
        "dataset_rows": rows,
        "estimands": estimands,
        "reasons": reasons,
        "next_action": "run mixed_effect_lodo_condition_count_gate and noharm_surrogate_transfer_v2 before any scaling GPU",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "modality",
            "perturbation_type",
            "cell_background_source",
            "buckets",
            "n_cells_selected_total",
            "n_conditions_selected_source",
            "n_conditions_trainonly",
            "n_conditions_cap30",
            "n_conditions_cap120",
            "n_multi_conditions_selected",
            "obs_condition_columns",
            "obs_cell_background_columns",
            "obs_perturbation_type_columns",
            "obs_dose_columns",
            "source_quality",
            "source_check_status",
            "source_label",
            "source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# LatentFM Scaling Provenance / Estimand Matrix Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of existing metadata/provenance reports.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- datasets: `{len(rows)}`",
        f"- perturbation types: `{dict(type_counts)}`",
        f"- source backgrounds: `{dict(bg_counts)}`",
        f"- source quality: `{dict(quality_counts)}`",
        f"- datasets with obs-level cell-background columns: `{len(datasets_with_obs_bg)}/{len(rows)}`",
        f"- datasets with dose columns: `{len(datasets_with_dose)}/{len(rows)}`",
        f"- datasets with selected multi bucket: `{datasets_with_multi}`",
        f"- selected multi conditions total: `{sum(multi_condition_counts.values())}`",
        f"- datasets with cap120 > cap30 train conditions: `{len(cap120_gt_cap30)}/{len(rows)}`",
        "",
        "## Estimand Matrix",
        "",
        "| axis | status | available evidence | next gate | GPU trigger | GPU |",
        "|---|---|---|---|---|---|",
    ]
    for item in estimands:
        lines.append(
            f"| `{item['axis']}` | `{item['status']}` | {item['available_evidence']} | "
            f"`{item['next_gate']}` | {item['gpu_trigger']} | `{item['gpu_authorized']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- next action: `mixed_effect_lodo_condition_count_gate` plus `noharm_surrogate_transfer_v2` before any scaling GPU.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- CSV: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
