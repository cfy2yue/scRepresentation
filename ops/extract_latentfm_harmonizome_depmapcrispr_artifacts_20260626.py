#!/usr/bin/env python3
"""Extract Harmonizome DepMap CRISPR dependency artifacts for LatentFM.

CPU/source-only extractor. It reads Harmonizome DepMap CRISPR gene-cell-line
associations plus the frozen S0 provenance table and writes condition-level
`dataset,condition,artifact_value` CSVs for existing train-only outcome rows.

It does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SRC = ROOT / "reports/external_artifact_sources_20260626/harmonizome_depmapcrispr"
REPORT_DIR = ROOT / "reports/harmonizome_depmapcrispr_artifacts_20260626"
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

EDGE_GZ = SRC / "gene_attribute_edges.txt.gz"
ATTR_GZ = SRC / "attribute_list_entries.txt.gz"
GENE_GZ = SRC / "gene_list_terms.txt.gz"

OUT_Z = REPORT_DIR / "harmonizome_depmapcrispr_matched_cellline_zscore.csv"
OUT_IND = REPORT_DIR / "harmonizome_depmapcrispr_matched_cellline_indicator.csv"
OUT_GLOBAL_MAX = REPORT_DIR / "harmonizome_depmapcrispr_global_max_zscore.csv"
OUT_GLOBAL_FRAC = REPORT_DIR / "harmonizome_depmapcrispr_global_hit_fraction.csv"
OUT_MANIFEST = ROOT / "configs/latentfm_harmonizome_depmapcrispr_artifact_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_harmonizome_depmapcrispr_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_HARMONIZOME_DEPMAPCRISPR_ARTIFACTS_20260626.md"


BACKGROUND_TO_CELL_LINE = {
    "A549": "A549",
    "K562": "K562",
    "MCF7": "MCF7",
    "MCF-7": "MCF7",
    "HEPG2": "HEPG2",
    "JURKAT": "JURKAT",
    "RPE1": "RPE1SS6",
}


def norm(text: str | None) -> str:
    return "" if text is None else str(text).strip()


def upper(text: str | None) -> str:
    return norm(text).upper().replace("-", "")


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


def read_s0_map(outcome_keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with S0.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            key = (ds, cond)
            if key not in outcome_keys:
                continue
            if norm(row.get("modality")) != "gene":
                continue
            gene = norm(row.get("gene")) or cond
            background = norm(row.get("cell_background_source"))
            cell_line = BACKGROUND_TO_CELL_LINE.get(background.upper(), BACKGROUND_TO_CELL_LINE.get(background))
            out[key] = {
                "dataset": ds,
                "condition": cond,
                "gene": gene,
                "background": background,
                "cell_line": cell_line or "",
            }
    return out


def read_terms(path: Path, key_col: str) -> set[str]:
    values: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            val = norm(row.get(key_col))
            if val:
                values.add(upper(val))
    return values


def read_edges(
    needed_genes: set[str], needed_cell_lines: set[str]
) -> tuple[dict[tuple[str, str], float], dict[str, list[float]]]:
    edges: dict[tuple[str, str], float] = {}
    by_gene: dict[str, list[float]] = defaultdict(list)
    with gzip.open(EDGE_GZ, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = upper(row.get("Gene"))
            if gene not in needed_genes:
                continue
            try:
                z = float(row.get("Z-score", ""))
            except ValueError:
                continue
            by_gene[gene].append(z)
            cell = upper(row.get("Cell Line"))
            if cell in needed_cell_lines:
                edges[(gene, cell)] = z
    return edges, by_gene


def write_artifact(path: Path, rows: list[dict[str, str | float]], value_key: str) -> None:
    fields = ["dataset", "condition", "artifact_value", "target", "cell_background", "source_cell_line", "source", "source_file"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact_value": row[value_key],
                    "target": row["gene"],
                    "cell_background": row["background"],
                    "source_cell_line": row["cell_line"],
                    "source": "Harmonizome_DepMap_CRISPR_Gene_Dependency",
                    "source_file": str(EDGE_GZ),
                }
            )


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    outcome_keys = read_outcome_keys()
    s0 = read_s0_map(outcome_keys)
    depmap_genes = read_terms(GENE_GZ, "Gene")
    depmap_cell_lines = read_terms(ATTR_GZ, "Cell Line")

    global_mapped = [
        row
        for row in s0.values()
        if upper(row["gene"]) in depmap_genes
    ]
    matched_mapped = [
        row
        for row in global_mapped
        if upper(row["cell_line"]) in depmap_cell_lines
    ]
    needed_genes = {upper(row["gene"]) for row in global_mapped}
    needed_cells = {upper(row["cell_line"]) for row in matched_mapped}
    edges, global_edges = read_edges(needed_genes, needed_cells)
    n_depmap_cell_lines = len(depmap_cell_lines)

    matched_artifact_rows: list[dict[str, str | float]] = []
    for row in matched_mapped:
        key = (upper(row["gene"]), upper(row["cell_line"]))
        z = edges.get(key, 0.0)
        matched_artifact_rows.append(
            {
                **row,
                "zscore": z,
                "indicator": 1.0 if key in edges else 0.0,
            }
        )

    global_artifact_rows: list[dict[str, str | float]] = []
    for row in global_mapped:
        gene_values = global_edges.get(upper(row["gene"]), [])
        global_artifact_rows.append(
            {
                **row,
                "global_max_zscore": max(gene_values) if gene_values else 0.0,
                "global_hit_fraction": (len(gene_values) / n_depmap_cell_lines) if n_depmap_cell_lines else 0.0,
            }
        )

    write_artifact(OUT_Z, matched_artifact_rows, "zscore")
    write_artifact(OUT_IND, matched_artifact_rows, "indicator")
    write_artifact(OUT_GLOBAL_MAX, global_artifact_rows, "global_max_zscore")
    write_artifact(OUT_GLOBAL_FRAC, global_artifact_rows, "global_hit_fraction")

    by_dataset = Counter(row["dataset"] for row in global_artifact_rows)
    by_background = Counter(row["background"] for row in global_artifact_rows)
    matched_by_dataset = Counter(row["dataset"] for row in matched_artifact_rows)
    nonzero_by_dataset = Counter(row["dataset"] for row in matched_artifact_rows if float(row["indicator"]) > 0)
    manifest = {
        "version": "20260626_harmonizome_depmapcrispr",
        "boundary": {
            "source": "Harmonizome DepMap CRISPR Gene Dependency gene-cell-line Z-score associations",
            "uses_training": False,
            "uses_gpu": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_train_only_internal_rows": True,
        },
        "artifacts": [
            {
                "artifact": "harmonizome_depmapcrispr_matched_cellline_zscore",
                "description": "Target gene dependency Z-score in matched DepMap/Harmonizome cell line; absent association encoded as 0.",
                "priority": 1,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "source_cell_line", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Must pass shuffle/LODO/source controls before GPU; this preflight is not sufficient alone.",
                "source_files": [str(OUT_Z.relative_to(ROOT))],
            },
            {
                "artifact": "harmonizome_depmapcrispr_matched_cellline_indicator",
                "description": "Binary target gene dependency indicator in matched DepMap/Harmonizome cell line.",
                "priority": 2,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "source_cell_line", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Must pass shuffle/LODO/source controls before GPU; this preflight is not sufficient alone.",
                "source_files": [str(OUT_IND.relative_to(ROOT))],
            },
            {
                "artifact": "harmonizome_depmapcrispr_global_max_zscore",
                "description": "Target-level maximum DepMap/Harmonizome CRISPR dependency Z-score across all cell lines; not cell-line matched.",
                "priority": 3,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "source_cell_line", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Higher confounding risk than matched-cell-line dependency; must beat target/actionability and gene-label shuffle controls before GPU.",
                "source_files": [str(OUT_GLOBAL_MAX.relative_to(ROOT))],
            },
            {
                "artifact": "harmonizome_depmapcrispr_global_hit_fraction",
                "description": "Target-level fraction of DepMap/Harmonizome cell lines with a CRISPR dependency association; not cell-line matched.",
                "priority": 4,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "source_cell_line", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Higher confounding risk than matched-cell-line dependency; must beat target/actionability and gene-label shuffle controls before GPU.",
                "source_files": [str(OUT_GLOBAL_FRAC.relative_to(ROOT))],
            },
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "status": "harmonizome_depmapcrispr_artifacts_materialized_cpu_preflight_next",
        "boundary": manifest["boundary"],
        "source_files": {
            "edges": str(EDGE_GZ),
            "attributes": str(ATTR_GZ),
            "genes": str(GENE_GZ),
        },
        "outcome_keys": len(outcome_keys),
        "s0_gene_background_mapped_keys": len(s0),
        "matched_cellline_artifact_rows": len(matched_artifact_rows),
        "global_target_artifact_rows": len(global_artifact_rows),
        "datasets": dict(sorted(by_dataset.items())),
        "matched_cellline_datasets": dict(sorted(matched_by_dataset.items())),
        "backgrounds": dict(sorted(by_background.items())),
        "nonzero_dependency_rows": int(sum(1 for row in matched_artifact_rows if float(row["indicator"]) > 0)),
        "nonzero_global_dependency_rows": int(
            sum(1 for row in global_artifact_rows if float(row["global_hit_fraction"]) > 0)
        ),
        "nonzero_dependency_by_dataset": dict(sorted(nonzero_by_dataset.items())),
        "manifest": str(OUT_MANIFEST),
        "outputs": [str(OUT_Z), str(OUT_IND), str(OUT_GLOBAL_MAX), str(OUT_GLOBAL_FRAC)],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Harmonizome DepMap CRISPR Artifacts",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "Status: `harmonizome_depmapcrispr_artifacts_materialized_cpu_preflight_next`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only extraction from Harmonizome DepMap CRISPR Gene Dependency files.",
        "- Uses frozen S0 provenance only for gene/background mapping and completed outcome-row keys only for overlap targeting.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- outcome keys: `{summary['outcome_keys']}`",
        f"- S0 gene/background mapped keys: `{summary['s0_gene_background_mapped_keys']}`",
        f"- matched-cellline artifact rows: `{summary['matched_cellline_artifact_rows']}`",
        f"- global target artifact rows: `{summary['global_target_artifact_rows']}`",
        f"- nonzero dependency rows: `{summary['nonzero_dependency_rows']}`",
        f"- nonzero global dependency rows: `{summary['nonzero_global_dependency_rows']}`",
        f"- datasets: `{dict(sorted(by_dataset.items()))}`",
        f"- matched-cellline datasets: `{dict(sorted(matched_by_dataset.items()))}`",
        f"- backgrounds: `{dict(sorted(by_background.items()))}`",
        "",
        "## Outputs",
        "",
        f"- Z-score artifact: `{OUT_Z}`",
        f"- indicator artifact: `{OUT_IND}`",
        f"- global max Z-score artifact: `{OUT_GLOBAL_MAX}`",
        f"- global hit-fraction artifact: `{OUT_GLOBAL_FRAC}`",
        f"- manifest: `{OUT_MANIFEST}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "These files only materialize a plausible external dependency artifact. They do not authorize GPU until the strict external-artifact preflight, shuffle/LODO controls, tail/MMD checks, and an external-review checkpoint pass.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
