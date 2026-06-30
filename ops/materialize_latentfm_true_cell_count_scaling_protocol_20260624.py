#!/usr/bin/env python3
"""Materialize fixed-condition protocol manifests for true cell-count scaling."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
TABLE = ROOT / "reports/latentfm_scaling_law_condition_table_20260624.tsv"
OUT_DIR = ROOT / "reports/latentfm_true_cell_count_scaling_protocol_20260624"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_scaling_protocol_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_SCALING_PROTOCOL_20260624.md"

PROTOCOLS = [
    {
        "name": "all_modality_fixed64_budget16_32_64",
        "min_cells": 64,
        "budgets": [16, 32, 64],
        "modalities": ["gene", "chemical"],
        "purpose": "Full-modality shallow true cell-count ladder preserving both gene and chemical conditions.",
    },
    {
        "name": "gene_only_fixed256_budget64_128_256",
        "min_cells": 256,
        "budgets": [64, 128, 256],
        "modalities": ["gene"],
        "purpose": "Gene-only deeper true cell-count ladder where high per-condition cell counts are available.",
    },
]

SUBSAMPLE_SEEDS = [42, 43, 44]


def parse_int(value: object) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(str(value)))
    except Exception:
        return 0


def read_rows() -> list[dict[str, str]]:
    with TABLE.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def select_rows(rows: list[dict[str, str]], protocol: dict[str, object]) -> list[dict[str, str]]:
    modalities = set(protocol["modalities"])
    min_cells = int(protocol["min_cells"])
    selected = [
        r for r in rows
        if r.get("modality") in modalities and parse_int(r.get("n_cells")) >= min_cells
    ]
    selected.sort(key=lambda r: (r.get("dataset") or "", r.get("condition") or "", r.get("modality") or ""))
    return selected


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    datasets = sorted({r["dataset"] for r in rows})
    modalities: dict[str, int] = {}
    ptypes: dict[str, int] = {}
    sources: dict[str, int] = {}
    for r in rows:
        modalities[r.get("modality") or "unknown"] = modalities.get(r.get("modality") or "unknown", 0) + 1
        ptypes[r.get("perturbation_type") or "unknown"] = ptypes.get(r.get("perturbation_type") or "unknown", 0) + 1
        sources[r.get("source_quality") or "unknown"] = sources.get(r.get("source_quality") or "unknown", 0) + 1
    return {
        "n_conditions": len(rows),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "modalities": modalities,
        "perturbation_types": ptypes,
        "source_quality_rows": sources,
    }


def write_protocol_tsv(path: Path, rows: list[dict[str, str]], protocol: dict[str, object]) -> None:
    fields = [
        "protocol",
        "budget_cells_per_condition",
        "subsample_seed",
        "dataset",
        "condition",
        "modality",
        "perturbation",
        "gene",
        "perturbation_type",
        "backgrounds",
        "n_cells_available",
        "source_h5ad",
        "source_quality",
        "source_label",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for budget in protocol["budgets"]:
            for seed in SUBSAMPLE_SEEDS:
                for row in rows:
                    writer.writerow(
                        {
                            "protocol": protocol["name"],
                            "budget_cells_per_condition": budget,
                            "subsample_seed": seed,
                            "dataset": row.get("dataset") or "",
                            "condition": row.get("condition") or "",
                            "modality": row.get("modality") or "",
                            "perturbation": row.get("perturbation") or "",
                            "gene": row.get("gene") or "",
                            "perturbation_type": row.get("perturbation_type") or "",
                            "backgrounds": row.get("backgrounds") or "",
                            "n_cells_available": row.get("n_cells") or "",
                            "source_h5ad": row.get("source_h5ad") or "",
                            "source_quality": row.get("source_quality") or "",
                            "source_label": row.get("source_label") or "",
                        }
                    )


def render_md(payload: dict[str, object]) -> str:
    lines = [
        "# LatentFM True Cell-Count Scaling Protocol",
        "",
        "Status: `true_cell_count_scaling_protocol_materialized_no_gpu`",
        "",
        "## Boundary",
        "",
        "- CPU-only protocol manifest materialization.",
        "- Reads only the frozen scaling-law condition table and feasibility gate.",
        "- Does not read expression matrices, model outputs, canonical metrics, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Protocols",
        "",
        "| protocol | purpose | budgets | conditions | datasets | modalities | manifest |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for p in payload["protocols"]:
        lines.append(
            "| {name} | {purpose} | `{budgets}` | {n_conditions} | {n_datasets} | `{modalities}` | `{manifest}` |".format(
                name=p["name"],
                purpose=p["purpose"],
                budgets=p["budgets"],
                n_conditions=p["summary"]["n_conditions"],
                n_datasets=p["summary"]["n_datasets"],
                modalities=p["summary"]["modalities"],
                manifest=p["manifest_tsv"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- GPU authorized: `False`",
            "- These manifests define fixed condition identities and deterministic subsampling budgets only.",
            "- They do not contain sampled cell indices or train-only pert-mean artifacts yet.",
            "",
            "## Next Gate",
            "",
            "A materializer/provenance gate must read the source H5ADs, produce deterministic per-budget/per-seed train-only pert-mean artifacts, and verify no canonical multi or Track C query usage. Only after that artifact gate and a tail/no-harm stop rule may a bounded GPU smoke be considered.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    protocols = []
    for protocol in PROTOCOLS:
        selected = select_rows(rows, protocol)
        manifest = OUT_DIR / f"{protocol['name']}.tsv"
        write_protocol_tsv(manifest, selected, protocol)
        protocols.append(
            {
                **protocol,
                "subsample_seeds": SUBSAMPLE_SEEDS,
                "manifest_tsv": str(manifest),
                "summary": summarize(selected),
            }
        )
    payload = {
        "status": "true_cell_count_scaling_protocol_materialized_no_gpu",
        "boundary": {
            "cpu_only": True,
            "reads_expression_matrices": False,
            "reads_model_outputs": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "input_table": str(TABLE),
        "protocols": protocols,
        "gpu_authorized": False,
        "next_action": "write_and_run_artifact_materializer_before_any_gpu",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_md(payload))
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
