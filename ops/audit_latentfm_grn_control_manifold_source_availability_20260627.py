#!/usr/bin/env python3
"""Audit source availability for GRN context and control-manifold support."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/grn_control_manifold_source_availability_20260627"
OUT_CSV = OUT_DIR / "source_availability_matrix.csv"
OUT_JSON = ROOT / "reports/latentfm_grn_control_manifold_source_availability_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_GRN_CONTROL_MANIFOLD_SOURCE_AVAILABILITY_20260627.md"


def status(path: Path) -> str:
    return "present" if path.exists() else "missing"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        rows.append(kwargs)

    add(
        route="background_specific_grn_context",
        source="OmniPath/CollecTRI/DoRothEA local TF-target files",
        source_paths=[
            str(ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_edges.tsv"),
            str(ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_gene_features.tsv"),
        ],
        source_presence=";".join(
            [
                status(ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_edges.tsv"),
                status(ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_gene_features.tsv"),
            ]
        ),
        background_specific=False,
        condition_level=False,
        prior_gate_status="background_specific_grn_context_source_audit_no_gpu; generic_omnipath_prior_gates_failed",
        evidence="Existing files are gene-level / TF-target pair-level. Prior report says no local small file maps (background, TF, target) to context-specific confidence.",
        gpu_authorized=False,
        next_action="No GPU. Reopen only with a genuinely background-specific GRN source and degree/TF-label/source controls.",
    )

    add(
        route="external_control_manifold_density",
        source="cellgene census processed tissue h5ad centroids",
        source_paths=[
            str(ROOT / "dataset/cellgene_census/processed/celltype_metainfo.csv"),
            str(ROOT / "dataset/cellgene_census/processed/blood/blood_top6000var.h5ad"),
            str(ROOT / "dataset/cellgene_census/processed/lung/lung_top6000var.h5ad"),
        ],
        source_presence=";".join(
            [
                status(ROOT / "dataset/cellgene_census/processed/celltype_metainfo.csv"),
                status(ROOT / "dataset/cellgene_census/processed/blood/blood_top6000var.h5ad"),
                status(ROOT / "dataset/cellgene_census/processed/lung/lung_top6000var.h5ad"),
            ]
        ),
        background_specific=True,
        condition_level=False,
        prior_gate_status="no_external_atlas_gate_yet; train_only_control_state_support_gate_fail_no_gpu",
        evidence="External atlas files are tissue/celltype controls, not perturbation-condition artifacts. Current local control-state support gate already failed tail/control criteria.",
        gpu_authorized=False,
        next_action="No immediate GPU. A new CPU gate would first need a frozen background mapping and residual controls proving signal is not tissue/dataset/source proxy.",
    )

    add(
        route="external_control_manifold_density",
        source="Tabula Sapiens atlas_TS h5ad files",
        source_paths=[
            str(ROOT / "dataset/raw/atlas_TS/TS_Blood_filtered.h5ad"),
            str(ROOT / "dataset/raw/atlas_TS/TS_Lung_filtered.h5ad"),
        ],
        source_presence=";".join(
            [
                status(ROOT / "dataset/raw/atlas_TS/TS_Blood_filtered.h5ad"),
                status(ROOT / "dataset/raw/atlas_TS/TS_Lung_filtered.h5ad"),
            ]
        ),
        background_specific=True,
        condition_level=False,
        prior_gate_status="source_available_but_no_short_condition_gate",
        evidence="Large control atlas h5ads exist. They do not directly encode local perturbation targets or condition-level support; using them requires a separate preprocessing branch and source/background-proxy controls.",
        gpu_authorized=False,
        next_action="No immediate GPU. Do not launch from atlas density until a cheap source-map/preflight shows nonconstant within-background variation and tail/MMD no-harm.",
    )

    add(
        route="train_control_state_support",
        source="local train-only control/GT embeddings",
        source_paths=[
            str(ROOT / "reports/LATENTFM_CONTROL_STATE_SUPPORT_GATE_20260624.md"),
            str(ROOT / "reports/latentfm_control_state_support_gate_20260624.json"),
        ],
        source_presence=";".join(
            [
                status(ROOT / "reports/LATENTFM_CONTROL_STATE_SUPPORT_GATE_20260624.md"),
                status(ROOT / "reports/latentfm_control_state_support_gate_20260624.json"),
            ]
        ),
        background_specific=True,
        condition_level=True,
        prior_gate_status="control_state_support_gate_fail_no_gpu",
        evidence="Train-only gate had mean pp gains for cap120 but failed dataset-min and inverted-control criteria; cap60 protocol also failed signal/control criteria.",
        gpu_authorized=False,
        next_action="No GPU. Keep as mechanism evidence unless a materially new external control-manifold gate beats source/background/count controls.",
    )

    fields = [
        "route",
        "source",
        "source_paths",
        "source_presence",
        "background_specific",
        "condition_level",
        "prior_gate_status",
        "evidence",
        "gpu_authorized",
        "next_action",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], sort_keys=True) if isinstance(row[k], list) else row[k] for k in fields})

    payload = {
        "status": "grn_control_manifold_source_availability_no_immediate_gpu",
        "gpu_authorized": False,
        "immediate_gpu_candidates": [],
        "rows": rows,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM GRN / Control-Manifold Source Availability 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only availability audit.",
        "- No training, inference, expression-matrix computation, canonical multi selection, Track C query, or GPU.",
        "- Goal: close or route the remaining Helmholtz slate items after guide-sequence and true-effect source availability failed.",
        "",
        "## Bottom Line",
        "",
        "- Existing OmniPath/CollecTRI/DoRothEA files are not background-specific and have already failed generic prior gates.",
        "- Local train-only control-state support already failed its strict nested LODO gate.",
        "- External atlas/cellgene census control sources exist, but they are tissue/celltype controls, not condition-level perturbation artifacts; using them would require a new source-map and anti-source-proxy CPU gate before any GPU.",
        "- No immediate GPU is authorized.",
        "",
        "## Matrix",
        "",
        "| route | source | background-specific | condition-level | prior status | decision |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['route']}` | {row['source']} | `{row['background_specific']}` | `{row['condition_level']}` | `{row['prior_gate_status']}` | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Reopen Standard",
            "",
            "A GRN/context route requires a genuinely background-specific source, not global TF degree or target membership. It must pass degree-preserving, TF-label, source/background, tail, and MMD controls.",
            "",
            "An external control-manifold route requires a frozen mapping from local cell backgrounds to atlas/census controls, proof of nonconstant within-background variation, residualization against dataset/source/background labels, and a strict train/internal no-harm gate before GPU.",
            "",
            "## Outputs",
            "",
            f"- csv: `{OUT_CSV}`",
            f"- json: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
