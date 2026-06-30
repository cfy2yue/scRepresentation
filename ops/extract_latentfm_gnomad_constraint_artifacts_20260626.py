#!/usr/bin/env python3
"""Extract gnomAD gene-constraint artifacts for LatentFM scaling gates.

CPU/source-only extractor. It maps public gnomAD v2.1.1 gene constraint
metrics onto frozen S0 gene-perturbation rows that already have train-only
internal outcome proxies.

It does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SRC = ROOT / "reports/external_artifact_sources_20260626/gnomad_constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt"
REPORT_DIR = ROOT / "reports/gnomad_constraint_artifacts_20260626"
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

OUT_LOEUF_SCORE = REPORT_DIR / "gnomad_lof_constraint_score_neglog10_loeuf.csv"
OUT_PLI = REPORT_DIR / "gnomad_pli.csv"
OUT_MIS_Z = REPORT_DIR / "gnomad_mis_z.csv"
OUT_OE_LOF_UPPER = REPORT_DIR / "gnomad_oe_lof_upper.csv"
OUT_MANIFEST = ROOT / "configs/latentfm_gnomad_constraint_artifact_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_gnomad_constraint_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_GNOMAD_CONSTRAINT_ARTIFACTS_20260626.md"


def norm(text: str | None) -> str:
    value = "" if text is None else str(text).strip()
    if value.lower() in {"", "nan", "na", "none", "<na>"}:
        return ""
    return value


def upper(text: str | None) -> str:
    return norm(text).upper()


def to_float(text: str | None) -> float | None:
    value = norm(text)
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


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


def read_s0_gene_rows(outcome_keys: set[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with S0.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            if (ds, cond) not in outcome_keys:
                continue
            if norm(row.get("modality")) != "gene":
                continue
            gene = norm(row.get("gene")) or cond
            if not gene:
                continue
            rows.append(
                {
                    "dataset": ds,
                    "condition": cond,
                    "gene": gene,
                    "cell_background": norm(row.get("cell_background_source")),
                    "perturbation_type": norm(row.get("perturbation_type")),
                }
            )
    return rows


def read_gnomad() -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    with SRC.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = upper(row.get("gene"))
            if not gene:
                continue
            oe_upper = to_float(row.get("oe_lof_upper"))
            pli = to_float(row.get("pLI"))
            mis_z = to_float(row.get("mis_z"))
            if oe_upper is None and pli is None and mis_z is None:
                continue
            rec: dict[str, float] = {}
            if oe_upper is not None:
                rec["oe_lof_upper"] = oe_upper
                rec["lof_constraint_score"] = -math.log10(max(oe_upper, 1e-12))
            if pli is not None:
                rec["pLI"] = pli
            if mis_z is not None:
                rec["mis_z"] = mis_z
            metrics.setdefault(gene, rec)
    return metrics


def write_artifact(path: Path, rows: list[dict[str, str | float]], value_key: str) -> int:
    fields = [
        "dataset",
        "condition",
        "artifact_value",
        "target",
        "cell_background",
        "perturbation_type",
        "source",
        "source_file",
    ]
    kept = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if value_key not in row:
                continue
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact_value": row[value_key],
                    "target": row["gene"],
                    "cell_background": row["cell_background"],
                    "perturbation_type": row["perturbation_type"],
                    "source": "gnomAD_v2.1.1_gene_constraint",
                    "source_file": str(SRC),
                }
            )
            kept += 1
    return kept


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    outcome_keys = read_outcome_keys()
    s0_rows = read_s0_gene_rows(outcome_keys)
    gnomad = read_gnomad()

    artifact_rows: list[dict[str, str | float]] = []
    for row in s0_rows:
        metrics = gnomad.get(upper(row["gene"]))
        if not metrics:
            continue
        artifact_rows.append({**row, **metrics})

    counts = {
        "lof_constraint_score_neglog10_loeuf": write_artifact(OUT_LOEUF_SCORE, artifact_rows, "lof_constraint_score"),
        "pli": write_artifact(OUT_PLI, artifact_rows, "pLI"),
        "mis_z": write_artifact(OUT_MIS_Z, artifact_rows, "mis_z"),
        "oe_lof_upper": write_artifact(OUT_OE_LOF_UPPER, artifact_rows, "oe_lof_upper"),
    }
    by_dataset = Counter(str(row["dataset"]) for row in artifact_rows)
    by_background = Counter(str(row["cell_background"]) for row in artifact_rows)

    manifest = {
        "version": "20260626_gnomad_constraint",
        "boundary": {
            "source": "gnomAD v2.1.1 LoF metrics by gene",
            "uses_training": False,
            "uses_gpu": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_train_only_internal_rows": True,
        },
        "artifacts": [
            {
                "artifact": "gnomad_lof_constraint_score_neglog10_loeuf",
                "description": "Gene-level constraint score, -log10(oe_lof_upper). Higher means more LoF constrained.",
                "priority": 1,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "perturbation_type", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Tail-risk artifact only; must pass gene-label shuffle, dataset/source controls, LODO, and MMD/tail checks before GPU.",
                "source_files": [str(OUT_LOEUF_SCORE.relative_to(ROOT))],
            },
            {
                "artifact": "gnomad_pli",
                "description": "Gene-level pLI constraint probability. Higher means more LoF constrained.",
                "priority": 2,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "perturbation_type", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Secondary constraint artifact; same controls required before GPU.",
                "source_files": [str(OUT_PLI.relative_to(ROOT))],
            },
            {
                "artifact": "gnomad_mis_z",
                "description": "Gene-level missense Z constraint score.",
                "priority": 3,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "perturbation_type", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Secondary constraint artifact; same controls required before GPU.",
                "source_files": [str(OUT_MIS_Z.relative_to(ROOT))],
            },
            {
                "artifact": "gnomad_oe_lof_upper",
                "description": "Raw LOEUF-like upper bound; lower means more constrained, included as sign-control.",
                "priority": 4,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target", "cell_background", "perturbation_type", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 3,
                "minimum_overlap_rows": 20,
                "promotion_note": "Sign-control only; lower values indicate stronger constraint.",
                "source_files": [str(OUT_OE_LOF_UPPER.relative_to(ROOT))],
            },
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "status": "gnomad_constraint_artifacts_materialized_cpu_preflight_next",
        "boundary": manifest["boundary"],
        "source_file": str(SRC),
        "outcome_keys": len(outcome_keys),
        "s0_gene_rows": len(s0_rows),
        "gnomad_gene_count": len(gnomad),
        "mapped_artifact_rows": len(artifact_rows),
        "artifact_row_counts": counts,
        "datasets": dict(sorted(by_dataset.items())),
        "backgrounds": dict(sorted(by_background.items())),
        "manifest": str(OUT_MANIFEST),
        "outputs": [str(OUT_LOEUF_SCORE), str(OUT_PLI), str(OUT_MIS_Z), str(OUT_OE_LOF_UPPER)],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM gnomAD Constraint Artifacts",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "Status: `gnomad_constraint_artifacts_materialized_cpu_preflight_next`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only extraction from gnomAD v2.1.1 gene-level constraint metrics.",
        "- Uses frozen S0 provenance only for gene target mapping and completed outcome-row keys only for overlap targeting.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- outcome keys: `{summary['outcome_keys']}`",
        f"- S0 gene rows: `{summary['s0_gene_rows']}`",
        f"- gnomAD genes: `{summary['gnomad_gene_count']}`",
        f"- mapped artifact rows: `{summary['mapped_artifact_rows']}`",
        f"- artifact row counts: `{counts}`",
        f"- datasets: `{dict(sorted(by_dataset.items()))}`",
        f"- backgrounds: `{dict(sorted(by_background.items()))}`",
        "",
        "## Outputs",
        "",
        f"- LOEUF constraint score artifact: `{OUT_LOEUF_SCORE}`",
        f"- pLI artifact: `{OUT_PLI}`",
        f"- missense Z artifact: `{OUT_MIS_Z}`",
        f"- raw oe_lof_upper artifact: `{OUT_OE_LOF_UPPER}`",
        f"- manifest: `{OUT_MANIFEST}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "These files only materialize plausible target-level external artifacts. They do not authorize GPU until strict preflight plus tail-risk, shuffle, and LODO controls pass.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
