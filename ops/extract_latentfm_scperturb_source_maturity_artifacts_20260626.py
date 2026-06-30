#!/usr/bin/env python3
"""Materialize scPerturb catalog source-maturity artifacts for LatentFM.

CPU/source-only extractor. It maps small scPerturb catalog metadata to local
datasets and writes condition-level `dataset,condition,artifact_value` CSVs for
completed train-only/internal outcome rows.

It does not train, infer, read checkpoints, read canonical multi, read Track C
query, read expression matrices, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CATALOG = ROOT / "reports/external_artifact_sources_20260626/scperturb_data_table_20260626.csv"
REPORT_DIR = ROOT / "reports/scperturb_source_maturity_artifacts_20260626"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
S0 = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUTCOME_FILES = [
    ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
    ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]

OUT_MANIFEST = ROOT / "configs/latentfm_scperturb_source_maturity_artifact_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_scperturb_source_maturity_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACTS_20260626.md"

TOKEN_MAP = {
    "Adamson": ["adamson"],
    "DixitRegev2016_K562_TFs_High_MOI": ["dixit"],
    "Frangieh": ["frangieh"],
    "GasperiniShendure2019_lowMOI": ["gasperini"],
    "NormanWeissman2019_filtered": ["norman"],
    "Papalexi": ["papalexi"],
    "ReplogleWeissman2022_K562_gwps": ["replogle"],
    "Replogle_RPE1essential": ["replogle"],
}

ARTIFACTS = {
    "scperturb_reported_cells_log10": {
        "path": REPORT_DIR / "scperturb_reported_cells_log10.csv",
        "description": "log10 reported total cells in the scPerturb catalog row.",
        "value_key": "reported_cells_log10",
    },
    "scperturb_perturbation_count_log10": {
        "path": REPORT_DIR / "scperturb_perturbation_count_log10.csv",
        "description": "log10 maximum perturbation count parsed from the scPerturb catalog row.",
        "value_key": "perturbation_count_log10",
    },
    "scperturb_cells_per_perturbation_log10": {
        "path": REPORT_DIR / "scperturb_cells_per_perturbation_log10.csv",
        "description": "log10 reported cells per parsed perturbation count.",
        "value_key": "cells_per_perturbation_log10",
    },
    "scperturb_timepoint_count": {
        "path": REPORT_DIR / "scperturb_timepoint_count.csv",
        "description": "Maximum parsed timepoint count/range endpoint from the scPerturb catalog row.",
        "value_key": "timepoint_count",
    },
    "scperturb_dose_count": {
        "path": REPORT_DIR / "scperturb_dose_count.csv",
        "description": "Maximum parsed dose count/range endpoint from the scPerturb catalog row.",
        "value_key": "dose_count",
    },
    "scperturb_h5ad_available": {
        "path": REPORT_DIR / "scperturb_h5ad_available.csv",
        "description": "Binary indicator that the scPerturb catalog row lists h5ad availability.",
        "value_key": "h5ad_available",
    },
}


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def parse_numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", norm(text))]


def parsed_max(text: str) -> float | None:
    nums = parse_numbers(text)
    return max(nums) if nums else None


def log10_or_none(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return math.log10(value)


def row_text(row: dict[str, str]) -> str:
    return " ".join(norm(row.get(k)) for k in ("Shorthand", "Title", "Treatment", "Cell source", "Data location"))


def load_catalog() -> list[dict[str, str]]:
    with CATALOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_outcome_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in OUTCOME_FILES:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not {"dataset", "condition"}.issubset(reader.fieldnames or []):
                continue
            for row in reader:
                ds = norm(row.get("dataset"))
                cond = norm(row.get("condition"))
                if ds and cond:
                    keys.add((ds, cond))
    return keys


def read_s0_rows(outcome_keys: set[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with S0.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            if (ds, cond) not in outcome_keys:
                continue
            rows.append(
                {
                    "dataset": ds,
                    "condition": cond,
                    "modality": norm(row.get("modality")),
                    "perturbation_type": norm(row.get("perturbation_type")),
                    "cell_background": norm(row.get("cell_background_source")),
                }
            )
    return rows


def select_catalog_rows(catalog: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for ds, tokens in TOKEN_MAP.items():
        matches = [row for row in catalog if any(tok in row_text(row).lower() for tok in tokens)]
        if not matches:
            continue
        # Prefer rows with direct h5ad links, then largest reported cell count.
        matches.sort(
            key=lambda row: (
                bool(norm(row.get(".h5ad availability"))),
                parsed_max(norm(row.get("Reported cells total"))) or -1,
            ),
            reverse=True,
        )
        row = matches[0]
        reported_cells = parsed_max(norm(row.get("Reported cells total")))
        perturbations = parsed_max(norm(row.get("# perturbations")))
        timepoints = parsed_max(norm(row.get("# timepoints")))
        doses = parsed_max(norm(row.get("# doses")))
        rec = {
            "local_dataset": ds,
            "scperturb_shorthand": norm(row.get("Shorthand")),
            "title": norm(row.get("Title")),
            "treatment": norm(row.get("Treatment")),
            "technique": norm(row.get("Technique")),
            "data_location": norm(row.get("Data location")),
            "reported_cells": reported_cells,
            "perturbation_count": perturbations,
            "timepoint_count": timepoints,
            "dose_count": doses,
            "h5ad_available": 1.0 if norm(row.get(".h5ad availability")) else 0.0,
        }
        rec["reported_cells_log10"] = log10_or_none(reported_cells)
        rec["perturbation_count_log10"] = log10_or_none(perturbations)
        rec["cells_per_perturbation_log10"] = log10_or_none(
            (reported_cells / perturbations) if reported_cells and perturbations else None
        )
        selected[ds] = rec
    return selected


def write_artifact(name: str, artifact_rows: list[dict[str, Any]]) -> int:
    spec = ARTIFACTS[name]
    value_key = spec["value_key"]
    path = spec["path"]
    fields = [
        "dataset",
        "condition",
        "artifact_value",
        "modality",
        "perturbation_type",
        "cell_background",
        "source",
        "source_dataset_row",
        "source_file",
    ]
    kept = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in artifact_rows:
            value = row.get(value_key)
            if value is None:
                continue
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact_value": value,
                    "modality": row["modality"],
                    "perturbation_type": row["perturbation_type"],
                    "cell_background": row["cell_background"],
                    "source": "scPerturb_catalog_source_maturity",
                    "source_dataset_row": row["scperturb_shorthand"],
                    "source_file": str(CATALOG),
                }
            )
            kept += 1
    return kept


def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    outcome_keys = read_outcome_keys()
    s0_rows = read_s0_rows(outcome_keys)
    selected = select_catalog_rows(load_catalog())
    artifact_rows = []
    for row in s0_rows:
        cat = selected.get(row["dataset"])
        if not cat:
            continue
        artifact_rows.append({**row, **cat})

    row_counts = {name: write_artifact(name, artifact_rows) for name in ARTIFACTS}
    by_dataset = Counter(row["dataset"] for row in artifact_rows)
    by_modality = Counter(row["modality"] for row in artifact_rows)

    manifest = {
        "version": "20260626_scperturb_source_maturity",
        "boundary": {
            "source": "scPerturb catalog source-level metadata",
            "uses_training": False,
            "uses_gpu": False,
            "uses_expression_matrices": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_train_only_internal_rows": True,
        },
        "artifacts": [],
    }
    for priority, (name, spec) in enumerate(ARTIFACTS.items(), start=1):
        manifest["artifacts"].append(
            {
                "artifact": name,
                "description": spec["description"],
                "priority": priority,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": [
                    "modality",
                    "perturbation_type",
                    "cell_background",
                    "source",
                    "source_dataset_row",
                    "source_file",
                ],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Dataset-level source-maturity artifacts must pass within-dataset variation/source controls before any GPU; expected to be audit-only if constant per dataset.",
                "source_files": [str(spec["path"].relative_to(ROOT))],
            }
        )
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    payload = {
        "timestamp": timestamp,
        "status": "scperturb_source_maturity_artifacts_materialized_cpu_preflight_next",
        "boundary": manifest["boundary"],
        "outcome_keys": len(outcome_keys),
        "s0_overlap_rows": len(s0_rows),
        "matched_catalog_datasets": sorted(selected),
        "mapped_artifact_rows": len(artifact_rows),
        "artifact_row_counts": row_counts,
        "datasets": dict(sorted(by_dataset.items())),
        "modalities": dict(sorted(by_modality.items())),
        "manifest": str(OUT_MANIFEST),
        "outputs": [str(spec["path"]) for spec in ARTIFACTS.values()],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM scPerturb Source-Maturity Artifacts",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "Status: `scperturb_source_maturity_artifacts_materialized_cpu_preflight_next`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only extraction from the small scPerturb catalog CSV.",
        "- Maps source-level metadata to completed train-only/internal outcome-row keys only.",
        "- Does not train, infer, read checkpoints, canonical multi, Track C query, expression matrices, or use GPU.",
        "",
        "## Summary",
        "",
        f"- outcome keys: `{payload['outcome_keys']}`",
        f"- S0 overlap rows: `{payload['s0_overlap_rows']}`",
        f"- matched catalog datasets: `{payload['matched_catalog_datasets']}`",
        f"- mapped artifact rows: `{payload['mapped_artifact_rows']}`",
        f"- artifact row counts: `{row_counts}`",
        f"- datasets: `{dict(sorted(by_dataset.items()))}`",
        f"- modalities: `{dict(sorted(by_modality.items()))}`",
        "",
        "## Outputs",
        "",
    ]
    for name, spec in ARTIFACTS.items():
        lines.append(f"- `{name}`: `{spec['path']}`")
    lines.extend(
        [
            f"- manifest: `{OUT_MANIFEST}`",
            f"- JSON: `{OUT_JSON}`",
            "",
            "## Decision",
            "",
            "These artifacts test whether source maturity/catalog scale is a usable scaling axis. They do not authorize GPU until strict preflight plus source/dataset controls pass.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
