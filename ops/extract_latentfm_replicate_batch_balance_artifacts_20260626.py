#!/usr/bin/env python3
"""Extract condition-level replicate/batch-balance artifacts.

CPU/metadata-only extractor. It uses available source metadata fields that look
like replicate or processing batches:

* Norman GEO cell identities: `gemgroup`
* Dixit processed scPerturb h5ad obs: `batch`
* Frangieh processed scPerturb h5ad obs: constant library protocol/MOI control

It does not read expression matrices, checkpoints, canonical multi, Track C
query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
S0 = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUTCOME_FILES = [
    ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
    ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]

NORMAN_GEO = ROOT / "reports/external_artifact_sources_20260626/norman_geo/GSE133344_filtered_cell_identities.csv.gz"
DIXIT_H5AD = ROOT / "reports/external_artifact_sources_20260626/dixit_figshare/Dixit_2016.h5ad"
FRANGIEH_H5AD = ROOT / "reports/external_artifact_sources_20260626/frangieh_figshare/Frangieh_2021.h5ad"

REPORT_DIR = ROOT / "reports/replicate_batch_balance_artifacts_20260626"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ENTROPY = REPORT_DIR / "replicate_batch_entropy_norm.csv"
OUT_COUNT = REPORT_DIR / "replicate_batch_count.csv"
OUT_MIN_FRAC = REPORT_DIR / "replicate_min_batch_fraction.csv"
OUT_SINGLETON = REPORT_DIR / "replicate_single_batch_indicator.csv"
OUT_MANIFEST = ROOT / "configs/latentfm_replicate_batch_balance_artifact_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_replicate_batch_balance_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACTS_20260626.md"


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


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


def read_s0_meta(outcome_keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with S0.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            key = (ds, cond)
            if key not in outcome_keys:
                continue
            out[key] = {
                "dataset": ds,
                "condition": cond,
                "modality": norm(row.get("modality")),
                "perturbation_type": norm(row.get("perturbation_type")),
                "cell_background": norm(row.get("cell_background_source")),
            }
    return out


def clean_norman_guide(value: str) -> str:
    left = value.split("__", 1)[0]
    parts = left.split("_")
    if len(parts) < 2:
        return left
    parts = ["ctrl" if "NegCtrl" in part else part for part in parts[:2]]
    if parts[0] == "ctrl" and parts[1] == "ctrl":
        return "ctrl"
    if parts[0] == "ctrl":
        return parts[1]
    if parts[1] == "ctrl":
        return parts[0]
    return f"{parts[0]}+{parts[1]}"


def canonicalize(condition: str, valid: set[str]) -> str | None:
    if condition in valid:
        return condition
    if "+" in condition:
        a, b = condition.split("+", 1)
        rev = f"{b}+{a}"
        if rev in valid:
            return rev
    return None


def entropy_norm(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p)
    return ent / math.log(len(counts))


def rows_from_group_counts(
    dataset: str,
    grouped: dict[str, Counter[str]],
    s0_meta: dict[tuple[str, str], dict[str, str]],
    source: str,
    source_file: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for condition, counts in grouped.items():
        key = (dataset, condition)
        if key not in s0_meta:
            continue
        total = sum(counts.values())
        n_batches = len(counts)
        min_frac = min(counts.values()) / total if total else 0.0
        out.append(
            {
                **s0_meta[key],
                "batch_entropy_norm": entropy_norm(counts),
                "batch_count": float(n_batches),
                "min_batch_fraction": min_frac,
                "single_batch_indicator": 1.0 if n_batches <= 1 else 0.0,
                "n_cells": total,
                "source": source,
                "source_file": str(source_file),
            }
        )
    return out


def load_norman(s0_meta: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    dataset = "NormanWeissman2019_filtered"
    valid = {cond for ds, cond in s0_meta if ds == dataset}
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    with gzip.open(NORMAN_GEO, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = canonicalize(clean_norman_guide(norm(row.get("guide_identity"))), valid)
            gemgroup = norm(row.get("gemgroup"))
            if condition and gemgroup:
                grouped[condition][gemgroup] += 1
    return rows_from_group_counts(dataset, grouped, s0_meta, "Norman_GEO_gemgroup", NORMAN_GEO)


def load_h5ad_dataset(
    dataset: str,
    path: Path,
    condition_col: str,
    batch_col: str,
    s0_meta: dict[tuple[str, str], dict[str, str]],
    source: str,
) -> list[dict[str, Any]]:
    import anndata as ad

    valid = {cond for ds, cond in s0_meta if ds == dataset}
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs[[condition_col, batch_col]].copy()
    finally:
        adata.file.close()
    for _, row in obs.iterrows():
        condition = canonicalize(norm(row.get(condition_col)), valid)
        batch = norm(row.get(batch_col))
        if condition and batch:
            grouped[condition][batch] += 1
    return rows_from_group_counts(dataset, grouped, s0_meta, source, path)


def write_artifact(path: Path, rows: list[dict[str, Any]], value_key: str) -> int:
    fields = [
        "dataset",
        "condition",
        "artifact_value",
        "modality",
        "perturbation_type",
        "cell_background",
        "n_cells",
        "source",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact_value": row[value_key],
                    "modality": row["modality"],
                    "perturbation_type": row["perturbation_type"],
                    "cell_background": row["cell_background"],
                    "n_cells": row["n_cells"],
                    "source": row["source"],
                    "source_file": row["source_file"],
                }
            )
    return len(rows)


def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    outcome_keys = read_outcome_keys()
    s0_meta = read_s0_meta(outcome_keys)
    rows = []
    rows.extend(load_norman(s0_meta))
    rows.extend(
        load_h5ad_dataset(
            "DixitRegev2016_K562_TFs_High_MOI",
            DIXIT_H5AD,
            "perturbation_name",
            "batch",
            s0_meta,
            "Dixit_scPerturb_processed_batch",
        )
    )
    rows.extend(
        load_h5ad_dataset(
            "Frangieh",
            FRANGIEH_H5AD,
            "perturbation_name",
            "library_preparation_protocol",
            s0_meta,
            "Frangieh_scPerturb_processed_library_protocol",
        )
    )
    rows.sort(key=lambda r: (r["dataset"], r["condition"]))

    counts = {
        "replicate_batch_entropy_norm": write_artifact(OUT_ENTROPY, rows, "batch_entropy_norm"),
        "replicate_batch_count": write_artifact(OUT_COUNT, rows, "batch_count"),
        "replicate_min_batch_fraction": write_artifact(OUT_MIN_FRAC, rows, "min_batch_fraction"),
        "replicate_single_batch_indicator": write_artifact(OUT_SINGLETON, rows, "single_batch_indicator"),
    }
    by_dataset = Counter(row["dataset"] for row in rows)
    varying_by_artifact = {}
    for key in ("batch_entropy_norm", "batch_count", "min_batch_fraction", "single_batch_indicator"):
        varying_by_artifact[key] = sorted(
            ds for ds in by_dataset if len({round(float(row[key]), 8) for row in rows if row["dataset"] == ds}) >= 2
        )

    manifest = {
        "version": "20260626_replicate_batch_balance",
        "boundary": {
            "source": "Norman GEO gemgroup and scPerturb processed obs batch/library metadata",
            "uses_training": False,
            "uses_gpu": False,
            "uses_expression_matrices": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_train_only_internal_rows": True,
        },
        "artifacts": [
            {
                "artifact": "replicate_batch_entropy_norm",
                "description": "Normalized entropy of condition cells across replicate/batch/gemgroup labels.",
                "priority": 1,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["modality", "perturbation_type", "cell_background", "n_cells", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 2,
                "minimum_overlap_rows": 20,
                "promotion_note": "Requires source-block and within-dataset shuffle controls before GPU; preflight alone is not sufficient.",
                "source_files": [str(OUT_ENTROPY.relative_to(ROOT))],
            },
            {
                "artifact": "replicate_batch_count",
                "description": "Number of replicate/batch/gemgroup labels represented by a condition.",
                "priority": 2,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["modality", "perturbation_type", "cell_background", "n_cells", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 2,
                "minimum_overlap_rows": 20,
                "promotion_note": "Requires source-block and within-dataset shuffle controls before GPU; preflight alone is not sufficient.",
                "source_files": [str(OUT_COUNT.relative_to(ROOT))],
            },
            {
                "artifact": "replicate_min_batch_fraction",
                "description": "Smallest represented batch fraction; higher can indicate more balanced replicate support.",
                "priority": 3,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["modality", "perturbation_type", "cell_background", "n_cells", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 2,
                "minimum_overlap_rows": 20,
                "promotion_note": "Requires source-block and within-dataset shuffle controls before GPU; preflight alone is not sufficient.",
                "source_files": [str(OUT_MIN_FRAC.relative_to(ROOT))],
            },
            {
                "artifact": "replicate_single_batch_indicator",
                "description": "Indicator that a condition is represented by one batch/gemgroup only; negative control for replicate support.",
                "priority": 4,
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["modality", "perturbation_type", "cell_background", "n_cells", "source", "source_file"],
                "minimum_datasets": 3,
                "minimum_varying_datasets": 2,
                "minimum_overlap_rows": 20,
                "promotion_note": "Negative-control artifact; should not authorize GPU unless independently validated.",
                "source_files": [str(OUT_SINGLETON.relative_to(ROOT))],
            },
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    payload = {
        "timestamp": timestamp,
        "status": "replicate_batch_balance_artifacts_materialized_cpu_preflight_next",
        "boundary": manifest["boundary"],
        "outcome_keys": len(outcome_keys),
        "mapped_rows": len(rows),
        "datasets": dict(sorted(by_dataset.items())),
        "artifact_row_counts": counts,
        "varying_by_artifact": varying_by_artifact,
        "manifest": str(OUT_MANIFEST),
        "outputs": [str(OUT_ENTROPY), str(OUT_COUNT), str(OUT_MIN_FRAC), str(OUT_SINGLETON)],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Replicate/Batch Balance Artifacts",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "Status: `replicate_batch_balance_artifacts_materialized_cpu_preflight_next`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/metadata-only extraction from Norman GEO cell identities and scPerturb processed obs batch/library fields.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "- Uses completed train-only/internal outcome-row keys only for overlap targeting.",
        "",
        "## Summary",
        "",
        f"- outcome keys: `{len(outcome_keys)}`",
        f"- mapped rows: `{len(rows)}`",
        f"- datasets: `{dict(sorted(by_dataset.items()))}`",
        f"- artifact row counts: `{counts}`",
        f"- varying by artifact: `{varying_by_artifact}`",
        "",
        "## Outputs",
        "",
        f"- entropy artifact: `{OUT_ENTROPY}`",
        f"- count artifact: `{OUT_COUNT}`",
        f"- min-fraction artifact: `{OUT_MIN_FRAC}`",
        f"- single-batch artifact: `{OUT_SINGLETON}`",
        f"- manifest: `{OUT_MANIFEST}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "These files materialize a possible replicate/batch-balance source artifact. They do not authorize GPU until strict preflight plus source/dataset/within-dataset shuffle controls pass.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
