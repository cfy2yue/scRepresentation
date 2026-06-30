#!/usr/bin/env python3
"""CPU feasibility audit for a Track A endpoint-shape auxiliary route.

The route is only materially new if it can use train-only condition-level
endpoint-shape information without duplicating closed same-gene transport,
response-normalizer, dataset-scale, or nuisance-residual routes.

This audit is report-only: no train loop, inference, checkpoint selection,
canonical multi selection, Track C query, held-out exact-row selection, or GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
FORENSICS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
SAMEGENE_REPORT = ROOT / "reports/latentfm_tracka_samegene_transport_cpu_gate_20260627.json"
NORMALIZATION_CLOSURE = ROOT / "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_CLOSURE_20260624.md"
NUISANCE_REPORT = ROOT / "reports/latentfm_xverse_nuisance_residual_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_tracka_endpoint_shape_aux_feasibility_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_ENDPOINT_SHAPE_AUX_FEASIBILITY_20260627.md"


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") != "internal_val_cross_background_seen_gene_proxy":
                continue
            count = fnum(row.get("gene_train_count"))
            if count is None:
                continue
            rows.append(
                {
                    "dataset": str(row.get("dataset")),
                    "condition": str(row.get("condition")),
                    "gene": str(row.get("gene", "")).strip().upper(),
                    "gene_train_count": int(count),
                    "anchor_pearson_pert": fnum(row.get("anchor_pearson_pert")),
                    "gene_raw_mean": fnum(row.get("gene_raw_mean")),
                    "dataset_mean": fnum(row.get("dataset_mean")),
                    "global_mean": fnum(row.get("global_mean")),
                    "shrink_k8": fnum(row.get("shrink_k8")),
                }
            )
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = load_rows()
    counts = np.asarray([row["gene_train_count"] for row in rows], dtype=int)
    datasets = sorted({row["dataset"] for row in rows})
    zero_frac = float(np.mean(counts == 0)) if counts.size else 1.0
    ge1_frac = float(np.mean(counts >= 1)) if counts.size else 0.0
    ge2_frac = float(np.mean(counts >= 2)) if counts.size else 0.0
    ge3_frac = float(np.mean(counts >= 3)) if counts.size else 0.0
    max_count = int(counts.max()) if counts.size else 0
    median_count = float(np.median(counts)) if counts.size else 0.0
    per_dataset = {}
    for ds in datasets:
        part = np.asarray([row["gene_train_count"] for row in rows if row["dataset"] == ds], dtype=int)
        per_dataset[ds] = {
            "n": int(part.size),
            "frac_ge1": float(np.mean(part >= 1)) if part.size else 0.0,
            "frac_ge2": float(np.mean(part >= 2)) if part.size else 0.0,
            "median": float(np.median(part)) if part.size else 0.0,
            "max": int(part.max()) if part.size else 0,
        }

    samegene = load_json(SAMEGENE_REPORT)
    nuisance = load_json(NUISANCE_REPORT)
    normalization_text = NORMALIZATION_CLOSURE.read_text(encoding="utf-8") if NORMALIZATION_CLOSURE.is_file() else ""
    samegene_status = samegene.get("status", "missing")
    samegene_best = samegene.get("best_candidate", samegene.get("best", {}))
    if not isinstance(samegene_best, dict):
        samegene_best = {}
    nuisance_status = nuisance.get("status", "missing")

    reasons: list[str] = []
    if ge2_frac < 0.50:
        reasons.append("same_gene_train_endpoint_coverage_ge2_lt_50pct")
    if ge3_frac < 0.20:
        reasons.append("same_gene_train_endpoint_coverage_ge3_lt_20pct")
    if max_count < 5:
        reasons.append("same_gene_train_endpoint_max_count_lt_5")
    reasons.append("existing_samegene_transport_gate_failed")
    if "response normalization / dataset-scale PCA" in normalization_text:
        reasons.append("dataset_scale_endpoint_shape_duplicates_closed_response_normalizer_family")
    if nuisance_status == "nuisance_residual_gate_fail_no_gpu":
        reasons.append("nuisance_residual_shape_alignment_gate_failed")
    reasons.append("real_trainonly_mmd_noharm_not_run_no_gpu")

    status = "tracka_endpoint_shape_aux_feasibility_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training_loop": False,
            "dataset_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "heldout_exact_rows_used_for_selection": False,
        },
        "inputs": {
            "residual_forensics": str(FORENSICS),
            "samegene_transport_gate": str(SAMEGENE_REPORT),
            "normalization_closure": str(NORMALIZATION_CLOSURE),
            "nuisance_residual_gate": str(NUISANCE_REPORT),
        },
        "coverage": {
            "n_rows": int(len(rows)),
            "n_datasets": int(len(datasets)),
            "zero_train_count_fraction": zero_frac,
            "ge1_fraction": ge1_frac,
            "ge2_fraction": ge2_frac,
            "ge3_fraction": ge3_frac,
            "median_train_count": median_count,
            "max_train_count": max_count,
            "per_dataset": per_dataset,
        },
        "prior_gate_status": {
            "samegene_status": samegene_status,
            "samegene_best_candidate": samegene_best,
            "nuisance_status": nuisance_status,
            "normalization_closure_present": NORMALIZATION_CLOSURE.is_file(),
        },
        "decision_reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Endpoint-Shape Auxiliary Feasibility Audit",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/report-only feasibility audit. No training loop, dataset inference, checkpoint selection, canonical multi selection, Track C query, held-out exact-row selection, or GPU.",
        "",
        "## Coverage",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| `n_rows` | {len(rows)} |",
        f"| `n_datasets` | {len(datasets)} |",
        f"| `zero_train_count_fraction` | {zero_frac:.6f} |",
        f"| `ge1_fraction` | {ge1_frac:.6f} |",
        f"| `ge2_fraction` | {ge2_frac:.6f} |",
        f"| `ge3_fraction` | {ge3_frac:.6f} |",
        f"| `median_train_count` | {median_count:.6f} |",
        f"| `max_train_count` | {max_count} |",
        "",
        "## Prior Gate Context",
        "",
        f"- Same-gene transport status: `{samegene_status}`",
        f"- Nuisance residual status: `{nuisance_status}`",
        f"- Normalization/training-data closure present: `{NORMALIZATION_CLOSURE.is_file()}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not launch an endpoint-shape auxiliary GPU smoke from the current evidence. A legal reopen would need a new train-only condition-level endpoint source with stronger same-gene/background coverage and a metric/no-harm gate that is not a response-normalizer, dataset-scale, nuisance-residual, or same-gene transport duplicate.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
