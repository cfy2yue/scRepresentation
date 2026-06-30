#!/usr/bin/env python3
"""Build an auditable dataset inventory for LatentFM scaling studies.

This is a short CPU task. It reads only local metadata/benchmark summaries and
does not inspect canonical held-out outcomes or model artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_dataset_scaling_inventory_20260624.json"
OUT_TSV = ROOT / "reports/latentfm_dataset_scaling_inventory_20260624.tsv"
OUT_MD = ROOT / "reports/LATENTFM_DATASET_SCALING_INVENTORY_20260624.md"


SOURCE_MAP = {
    "Adamson": {
        "source_label": "Adamson et al. Cell 2016 UPR Perturb-seq",
        "url": "https://doi.org/10.1016/j.cell.2016.11.048",
        "check_status": "doi_identified_needs_full_text_spotcheck",
    },
    "DixitRegev2016_K562_TFs_High_MOI": {
        "source_label": "Dixit et al. Cell 2016 Perturb-seq",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5181115/",
        "check_status": "primary_open_text_checked",
    },
    "Frangieh": {
        "source_label": "Frangieh et al. Nat Genet 2021 Perturb-CITE-seq",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33649592/",
        "check_status": "primary_pubmed_checked",
    },
    "GasperiniShendure2019_lowMOI": {
        "source_label": "Gasperini et al. CRISPRi enhancer Perturb-seq",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6690346/",
        "check_status": "primary_open_text_checked",
    },
    "Jiang_IFNB": {
        "source_label": "Jiang cytokine-state Perturb-seq collection",
        "url": "https://www.biorxiv.org/content/10.1101/2024.01.29.576933.full",
        "check_status": "candidate_source_needs_precise_dataset_mapping",
    },
    "Jiang_IFNG": {
        "source_label": "Jiang cytokine-state Perturb-seq collection",
        "url": "https://www.biorxiv.org/content/10.1101/2024.01.29.576933.full",
        "check_status": "candidate_source_needs_precise_dataset_mapping",
    },
    "Jiang_INS": {
        "source_label": "Jiang cytokine-state Perturb-seq collection",
        "url": "https://www.biorxiv.org/content/10.1101/2024.01.29.576933.full",
        "check_status": "candidate_source_needs_precise_dataset_mapping",
    },
    "Jiang_TGFB": {
        "source_label": "Jiang cytokine-state Perturb-seq collection",
        "url": "https://www.biorxiv.org/content/10.1101/2024.01.29.576933.full",
        "check_status": "candidate_source_needs_precise_dataset_mapping",
    },
    "Jiang_TNFA": {
        "source_label": "Jiang cytokine-state Perturb-seq collection",
        "url": "https://www.biorxiv.org/content/10.1101/2024.01.29.576933.full",
        "check_status": "candidate_source_needs_precise_dataset_mapping",
    },
    "Nadig_hepg2": {
        "source_label": "Nadig HepG2 common-essential Perturb-seq",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667",
        "check_status": "geo_checked_primary_paper_needs_spotcheck",
    },
    "Nadig_jurket": {
        "source_label": "Nadig Jurkat common-essential Perturb-seq",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667",
        "check_status": "geo_checked_primary_paper_needs_spotcheck",
    },
    "NormanWeissman2019_filtered": {
        "source_label": "Norman et al. Science 2019 combinatorial Perturb-seq",
        "url": "https://www.science.org/doi/10.1126/science.aax4438",
        "check_status": "primary_landing_checked",
    },
    "Papalexi": {
        "source_label": "Papalexi A375 CRISPRko Perturb-seq",
        "url": "",
        "check_status": "needs_precise_primary_source_mapping",
    },
    "Replogle_RPE1essential": {
        "source_label": "Replogle et al. Cell 2022 genome-scale CRISPRi",
        "url": "https://pubmed.ncbi.nlm.nih.gov/35688146/",
        "check_status": "primary_pubmed_checked",
    },
    "ReplogleWeissman2022_K562_gwps": {
        "source_label": "Replogle et al. Cell 2022 genome-scale CRISPRi",
        "url": "https://pubmed.ncbi.nlm.nih.gov/35688146/",
        "check_status": "primary_pubmed_checked",
    },
    "Schmidt": {
        "source_label": "Schmidt et al. Science 2022 primary T-cell CRISPRa/i",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9307090/",
        "check_status": "primary_open_text_checked",
    },
    "TianActivation": {
        "source_label": "Tian et al. human-neuron CRISPRa/i screens",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8254803/",
        "check_status": "primary_open_text_checked_for_screen_context",
    },
    "TianInhibition": {
        "source_label": "Tian et al. human-neuron CRISPRa/i screens",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8254803/",
        "check_status": "primary_open_text_checked_for_screen_context",
    },
    "Wessels": {
        "source_label": "Wessels et al. Nat Methods 2023 Cas13 RNA Perturb-seq",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10030154/",
        "check_status": "primary_open_text_checked",
    },
    "sciplex3_A549": {
        "source_label": "Srivatsan et al. Science 2020 sci-Plex3",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/",
        "check_status": "primary_open_text_checked",
    },
    "sciplex3_K562": {
        "source_label": "Srivatsan et al. Science 2020 sci-Plex3",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/",
        "check_status": "primary_open_text_checked",
    },
    "sciplex3_MCF7": {
        "source_label": "Srivatsan et al. Science 2020 sci-Plex3",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/",
        "check_status": "primary_open_text_checked",
    },
}


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_int(value: str | int | None) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def split_backgrounds(cell_line: str) -> list[str]:
    return [x.strip() for x in cell_line.split("/") if x.strip()]


def load_rows() -> list[dict[str, object]]:
    meta_files = [
        ("gene", ROOT / "dataset/raw/genepert_bench/metainfo.json"),
        ("chemical", ROOT / "dataset/raw/chemicalpert_DE5000/metainfo.json"),
    ]
    brief_files = {
        "gene": ROOT / "dataset/raw/genepert_bench/brief_benchmark_meta.csv",
        "chemical": ROOT / "dataset/raw/chemicalpert_bench/brief_benchmark_meta.csv",
    }

    metadata: dict[str, dict[str, object]] = {}
    for modality, path in meta_files:
        for row in read_json(path):
            item = dict(row)
            item["modality"] = modality
            metadata[item["dataset"]] = item

    rows: list[dict[str, object]] = []
    for modality, path in brief_files.items():
        for row in read_csv(path):
            dataset = row["dataset"]
            meta = metadata.get(dataset, {})
            cell_line = str(meta.get("cell_line", ""))
            source = SOURCE_MAP.get(dataset, {})
            out = {
                "dataset": dataset,
                "modality": modality,
                "bucket": row["bucket"],
                "output_file": row["output_file"],
                "perturbation_type": meta.get("perturbation_type", ""),
                "cell_line": cell_line,
                "cell_backgrounds": split_backgrounds(cell_line),
                "n_cell_backgrounds": len(split_backgrounds(cell_line)),
                "is_mixed_source_dataset": row["is_mixed_source_dataset"],
                "n_cells_total_source": to_int(row["n_cells_total_source"]),
                "n_control_total_source": to_int(row["n_control_total_source"]),
                "n_control_selected": to_int(row["n_control_selected"]),
                "n_conditions_total_bucket": to_int(row["n_conditions_total_bucket"]),
                "n_conditions_selected_bucket": to_int(row["n_conditions_selected_bucket"]),
                "n_cells_selected_total": to_int(row["n_cells_selected_total"]),
                "source_label": source.get("source_label", ""),
                "source_url": source.get("url", ""),
                "source_check_status": source.get("check_status", "missing_source_mapping"),
                "metadata_fields": sorted(meta.keys()),
            }
            rows.append(out)
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    unique = {}
    buckets_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ds = str(row["dataset"])
        unique.setdefault(ds, row)
        buckets_by_dataset[ds].add(str(row["bucket"]))

    type_counter = Counter(str(row["perturbation_type"]) for row in unique.values())
    modality_counter = Counter(str(row["modality"]) for row in unique.values())
    background_counter: Counter[str] = Counter()
    for row in unique.values():
        for bg in row["cell_backgrounds"]:
            background_counter[str(bg)] += 1

    bucket_counter = Counter(str(row["bucket"]) for row in rows)
    selected_conditions_by_modality_bucket: dict[str, int] = defaultdict(int)
    total_conditions_by_modality_bucket: dict[str, int] = defaultdict(int)
    selected_cells_by_modality_bucket: dict[str, int] = defaultdict(int)
    for row in rows:
        key = f"{row['modality']}:{row['bucket']}"
        selected_conditions_by_modality_bucket[key] += int(row["n_conditions_selected_bucket"])
        total_conditions_by_modality_bucket[key] += int(row["n_conditions_total_bucket"])
        selected_cells_by_modality_bucket[key] += int(row["n_cells_selected_total"])

    source_status_counter = Counter(str(row["source_check_status"]) for row in unique.values())
    multi_datasets = sorted(ds for ds, buckets in buckets_by_dataset.items() if "multiple" in buckets)
    mixed_background_datasets = sorted(
        str(row["dataset"]) for row in unique.values() if int(row["n_cell_backgrounds"]) > 1
    )

    return {
        "n_inventory_rows": len(rows),
        "n_unique_datasets": len(unique),
        "datasets_by_modality": dict(sorted(modality_counter.items())),
        "datasets_by_perturbation_type": dict(sorted(type_counter.items())),
        "datasets_by_cell_background": dict(sorted(background_counter.items())),
        "bucket_rows": dict(sorted(bucket_counter.items())),
        "selected_conditions_by_modality_bucket": dict(
            sorted(selected_conditions_by_modality_bucket.items())
        ),
        "total_conditions_by_modality_bucket": dict(
            sorted(total_conditions_by_modality_bucket.items())
        ),
        "selected_cells_by_modality_bucket": dict(
            sorted(selected_cells_by_modality_bucket.items())
        ),
        "source_check_status_by_dataset": dict(sorted(source_status_counter.items())),
        "datasets_with_multiple_bucket": multi_datasets,
        "datasets_with_multiple_cell_backgrounds": mixed_background_datasets,
        "caveats": [
            "cell_line strings with slash-separated backgrounds are dataset-level metadata; "
            "condition-level assignment needs h5ad obs inspection in a later CPU audit.",
            "Jiang, Papalexi, Adamson, and Nadig mappings need more precise primary-source "
            "spot checks before a final paper-grade dataset table.",
            "This inventory does not read canonical held-out outcomes, query splits, or model logs.",
        ],
    }


def write_tsv(rows: list[dict[str, object]]) -> None:
    columns = [
        "dataset",
        "modality",
        "bucket",
        "perturbation_type",
        "cell_line",
        "n_cell_backgrounds",
        "is_mixed_source_dataset",
        "n_cells_total_source",
        "n_control_total_source",
        "n_control_selected",
        "n_conditions_total_bucket",
        "n_conditions_selected_bucket",
        "n_cells_selected_total",
        "output_file",
        "source_label",
        "source_url",
        "source_check_status",
    ]
    with OUT_TSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            writer.writerow({col: item.get(col, "") for col in columns})


def md_table(rows: list[dict[str, object]], limit: int = 40) -> str:
    cols = [
        "dataset",
        "modality",
        "bucket",
        "perturbation_type",
        "cell_line",
        "n_conditions_selected_bucket",
        "n_cells_selected_total",
        "source_check_status",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | {len(rows) - limit} more rows |")
    return "\n".join(lines)


def write_md(rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    checked = sum(
        1
        for row in {str(r["dataset"]): r for r in rows}.values()
        if "checked" in str(row["source_check_status"])
    )
    text = f"""# LatentFM Dataset Scaling Inventory

Status: `dataset_scaling_inventory_built`

## Boundary

- Short CPU-only inventory.
- Reads local metainfo and brief benchmark metadata only.
- Does not read canonical held-out outcomes, Track C query data, active logs, or GPU artifacts.

## Outputs

- JSON: `{OUT_JSON}`
- TSV: `{OUT_TSV}`

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```

## Inventory Table

{md_table(rows)}

## Immediate Interpretation

- The scaling-effect branch is feasible because dataset-level metadata already
  exposes perturbation type, cell background, selected condition counts, source
  cell counts, control counts, and single/multiple buckets.
- Current local benchmark inventory has `{summary['n_unique_datasets']}` unique
  datasets and `{summary['n_inventory_rows']}` row-level dataset-bucket entries.
- At least `{checked}` dataset mappings have an initial checked external source
  status; remaining entries are explicitly marked for follow-up rather than
  silently treated as verified.
- Multi-background dataset strings such as `K562/A549/RPE1` and
  `HEK293FT/THP-1` need condition-level h5ad obs inspection before background
  balancing can be finalized.

## Recommended Next Gate

Run a short CPU obs-schema audit on the h5ad files to identify condition-level
columns for cell line/background, perturbation name, dose/pathway, control flag,
and single-vs-multiple perturbation. This should produce an obs-column inventory
only; no model training or held-out outcome reading.
"""
    OUT_MD.write_text(text)


def main() -> None:
    rows = load_rows()
    rows = sorted(rows, key=lambda r: (str(r["modality"]), str(r["dataset"]), str(r["bucket"])))
    summary = summarize(rows)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True))
    write_tsv(rows)
    write_md(rows, summary)
    print(
        json.dumps(
            {
                "status": "dataset_scaling_inventory_built",
                "out_md": str(OUT_MD),
                "out_json": str(OUT_JSON),
                "out_tsv": str(OUT_TSV),
                "n_rows": len(rows),
                "n_unique_datasets": summary["n_unique_datasets"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
