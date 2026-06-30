#!/usr/bin/env python3
"""CPU preflight for JUMP-CP / Cell Painting morphology activity sources.

This is intentionally report-only:

* no GPU, training, inference, checkpoints, canonical multi Track A selection,
  or Track C held-out query;
* no large Cell Painting profile matrix download;
* no Chemical V2 authorization because exact ACK is absent;
* no reuse of DepMap, Replogle, gnomAD, LINCS, QC, or guide-support routes.

The route can only authorize a future GPU smoke if small local morphology
metadata/reproducibility/activity artifacts exist and support train-only
condition overlap, source/batch controls, shuffle controls, and MMD/tail vetoes.
The expected outcome for this preflight is source_preflight/no-gpu or blocker.
"""

from __future__ import annotations

import csv
import json
import math
import os
import ssl
import urllib.request
from collections import Counter
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

OUT_MD = ROOT / "reports/LATENTFM_JUMP_CP_MORPHOLOGY_SOURCE_GATE_20260627.md"
OUT_JSON = ROOT / "reports/latentfm_jump_cp_morphology_source_gate_20260627.json"
MANIFEST_DIR = ROOT / "reports/jump_cp_morphology_source_gate_20260627"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
OUT_SOURCE_MANIFEST = MANIFEST_DIR / "source_url_probe_manifest.tsv"
OUT_LOCAL_CANDIDATES = MANIFEST_DIR / "local_candidate_files.tsv"
OUT_LOCAL_UNIVERSE = MANIFEST_DIR / "local_trainonly_condition_universe.tsv"
OUT_SCHEMA_PREVIEW = MANIFEST_DIR / "local_schema_preview.tsv"
GENERATED_OUTPUT_PATHS = {OUT_MD, OUT_JSON, OUT_SOURCE_MANIFEST, OUT_LOCAL_CANDIDATES, OUT_LOCAL_UNIVERSE, OUT_SCHEMA_PREVIEW}

SOURCE_URLS = [
    {
        "source": "JUMP-CP Cell Painting Gallery",
        "kind": "s3_metadata_prefix_listing",
        "url": "https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&prefix=cpg0016-jump/metadata/&max-keys=50",
        "download_policy": "Small S3 prefix listing only; do not download profiles.",
        "notes": "Public metadata location probe for cpg0016-jump.",
    },
    {
        "source": "JUMP-CP Cell Painting Gallery",
        "kind": "s3_workspace_metadata_prefix_listing",
        "url": "https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&prefix=cpg0016-jump/source_4/workspace/metadata/&max-keys=50",
        "download_policy": "Small S3 prefix listing only; do not download profiles.",
        "notes": "Source-specific metadata listing probe.",
    },
    {
        "source": "JUMP-CP Cell Painting Gallery",
        "kind": "s3_profile_prefix_listing_not_download",
        "url": "https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&prefix=cpg0016-jump/source_4/workspace/profiles/&max-keys=10",
        "download_policy": "Listing only. Profile parquet/CSV matrices are large and are not downloaded.",
        "notes": "Confirms profile namespace existence without fetching profile matrices.",
    },
    {
        "source": "jump-cellpainting/datasets",
        "kind": "github_repository",
        "url": "https://github.com/jump-cellpainting/datasets",
        "download_policy": "HTML/HEAD probe only.",
        "notes": "Public JUMP dataset repository entry point.",
    },
    {
        "source": "jump-cellpainting/datasets metadata",
        "kind": "github_metadata_tree",
        "url": "https://github.com/jump-cellpainting/datasets/tree/main/metadata",
        "download_policy": "HTML/HEAD probe only.",
        "notes": "Metadata tree entry point; GitHub API may be rate-limited.",
    },
]

FILENAME_TERMS = (
    "jump",
    "cellpainting",
    "cell_painting",
    "cell-painting",
    "cpg0016",
    "morpholog",
    "morphology",
    "profile_repro",
    "profile-repro",
    "replicate_repro",
    "replicate-repro",
    "phenotypic_activity",
    "phenotypic-activity",
    "activity",
)
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "software",
    "scFM_third_party",
    "external_review",
    "node_modules",
    ".cache",
    "logs",
    "runs",
}
SCAN_ROOTS = [
    ROOT / "reports",
    ROOT / "configs",
    ROOT / "ops",
    ROOT / "dataset",
    ROOT / "CellClip",
    ROOT / "scFMBench",
    ROOT / "CoupledFM",
]
TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz", ".json", ".jsonl")


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
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
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not {"dataset", "condition"}.issubset(fields):
                continue
            for row in reader:
                dataset = norm(row.get("dataset"))
                condition = norm(row.get("condition"))
                if dataset and condition:
                    keys.add((dataset, condition))
    return keys


def read_s0_universe(outcome_keys: set[tuple[str, str]]) -> list[dict[str, str]]:
    if not S0.is_file():
        return []
    rows: list[dict[str, str]] = []
    with S0.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            dataset = norm(row.get("dataset"))
            condition = norm(row.get("condition"))
            if (dataset, condition) not in outcome_keys:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "modality": norm(row.get("modality")),
                    "perturbation_type": norm(row.get("perturbation_type")),
                    "perturbation": norm(row.get("perturbation")),
                    "gene": norm(row.get("gene")),
                    "dose": norm(row.get("dose")),
                    "pathway": norm(row.get("pathway")),
                    "cell_background": norm(row.get("cell_background_source")),
                    "n_cells": norm(row.get("n_cells")),
                    "source_label": norm(row.get("source_label")),
                    "source_url": norm(row.get("source_url")),
                }
            )
    return rows


def summarize_universe(rows: list[dict[str, str]]) -> dict[str, Any]:
    datasets = Counter(row["dataset"] for row in rows)
    modalities = Counter(row["modality"] for row in rows)
    ptypes = Counter(row["perturbation_type"] for row in rows)
    backgrounds = Counter(row["cell_background"] for row in rows if row["cell_background"])
    chemical_modality_terms = {"chemical", "drug", "compound", "small_molecule", "small molecule"}
    gene_modality_terms = {"gene", "genetic", "genetic perturbation"}
    gene_pert_terms = {"crispri", "crisprko", "crispr", "crispra", "cas13", "orf", "overexpression"}
    chemicals = []
    genes = []
    for row in rows:
        modality = row["modality"].lower()
        ptype = row["perturbation_type"].lower()
        if modality in chemical_modality_terms or ptype in chemical_modality_terms:
            chemicals.append(row)
        if modality in gene_modality_terms or ptype in gene_pert_terms:
            genes.append(row)
    sciplex = [
        row
        for row in rows
        if row["dataset"].startswith("sciplex3_") or "sci-Plex3" in row["source_label"]
    ]
    n_cells = [to_float(row["n_cells"]) for row in rows]
    n_cells = [x for x in n_cells if x is not None]
    return {
        "trainonly_outcome_conditions": len(rows),
        "datasets": len(datasets),
        "dataset_counts_top20": datasets.most_common(20),
        "modality_counts": modalities.most_common(),
        "perturbation_type_counts": ptypes.most_common(),
        "cell_background_counts_top20": backgrounds.most_common(20),
        "chemical_or_drug_rows": len(chemicals),
        "gene_or_genetic_rows": len(genes),
        "sciplex3_rows": len(sciplex),
        "sciplex3_backgrounds": sorted({row["cell_background"] for row in sciplex if row["cell_background"]}),
        "sciplex3_conditions_with_dose": sum(1 for row in sciplex if row["dose"]),
        "n_cells_min": min(n_cells) if n_cells else None,
        "n_cells_median_approx": sorted(n_cells)[len(n_cells) // 2] if n_cells else None,
        "n_cells_max": max(n_cells) if n_cells else None,
        "condition_overlap_interpretation": (
            "The local train-only outcome rows used by this preflight are gene perturbation rows "
            "across multiple backgrounds/datasets. JUMP-CP gene/ORF/CRISPR target overlap is "
            "plausible in principle, but cannot be measured without a local JUMP/Cell Painting "
            "metadata table. Chemical overlap is not supported by this local outcome subset and "
            "would need a separate train-only chemical universe plus exact non-ACK boundary."
        ),
    }


def write_universe(rows: list[dict[str, str]]) -> None:
    fields = [
        "dataset",
        "condition",
        "modality",
        "perturbation_type",
        "perturbation",
        "gene",
        "dose",
        "pathway",
        "cell_background",
        "n_cells",
        "source_label",
        "source_url",
    ]
    with OUT_LOCAL_UNIVERSE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def is_candidate_name(path: Path) -> bool:
    name = path.name.lower()
    return any(term in name for term in FILENAME_TERMS)


def kind_guess(name: str) -> str:
    low = name.lower()
    if "cpg0016" in low or "jump" in low:
        if any(term in low for term in ("metadata", "manifest", "plate", "well", "compound", "crispr", "orf")):
            return "jump_cp_metadata_candidate"
        if any(term in low for term in ("profile", "repro", "activity", "phenotypic")):
            return "jump_cp_profile_activity_candidate"
        return "name_mentions_jump_cp"
    if "cellpainting" in low or "cell_painting" in low or "cell-painting" in low:
        return "cell_painting_candidate"
    if "morpholog" in low or "morphology" in low:
        return "morphology_candidate"
    if "profile" in low and ("repro" in low or "activity" in low):
        return "profile_reproducibility_or_activity_candidate"
    return "weak_name_match"


def usable_local_source(name: str, size_bytes: int) -> bool:
    low = name.lower()
    likely_table = low.endswith(TABLE_SUFFIXES)
    has_schema_hint = any(
        term in low
        for term in (
            "jump",
            "cpg0016",
            "cellpainting",
            "cell_painting",
            "metadata",
            "manifest",
            "profile_repro",
            "replicate_repro",
            "phenotypic_activity",
        )
    )
    return likely_table and has_schema_hint and 0 < size_bytes <= 50_000_000


def scan_local_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            path_dir = Path(dirpath)
            if any(part in SKIP_DIR_NAMES for part in path_dir.parts):
                continue
            for filename in filenames:
                path = path_dir / filename
                if path in GENERATED_OUTPUT_PATHS or MANIFEST_DIR in path.parents:
                    continue
                if not is_candidate_name(path):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rows.append(
                    {
                        "path": str(path),
                        "size_bytes": stat.st_size,
                        "kind_guess": kind_guess(path.name),
                        "local_source_usable": usable_local_source(path.name, stat.st_size),
                    }
                )
    rows.sort(key=lambda row: (not row["local_source_usable"], row["path"]))
    return rows


def write_local_candidates(rows: list[dict[str, Any]]) -> None:
    fields = ["path", "size_bytes", "kind_guess", "local_source_usable"]
    with OUT_LOCAL_CANDIDATES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sniff_table_schema(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "field_count": 0,
        "fields_preview": "",
        "row_preview_count": 0,
        "error": "",
    }
    if path.suffix == ".gz" or path.stat().st_size > 5_000_000:
        out["error"] = "schema_preview_skipped_for_gzip_or_large_file"
        return out
    try:
        sample = path.read_text(encoding="utf-8", errors="replace")[:100_000]
        first = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = "\t" if first.count("\t") >= first.count(",") else ","
        fields = [field.strip() for field in first.split(delimiter) if field.strip()]
        out.update(
            {
                "readable": True,
                "field_count": len(fields),
                "fields_preview": "|".join(fields[:40]),
                "row_preview_count": max(0, min(5, len(sample.splitlines()) - 1)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - schema preview is optional audit evidence.
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def write_schema_preview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previews = []
    for row in rows:
        if not row["local_source_usable"]:
            continue
        previews.append(sniff_table_schema(Path(row["path"])))
    fields = ["path", "readable", "field_count", "fields_preview", "row_preview_count", "error"]
    with OUT_SCHEMA_PREVIEW.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for preview in previews:
            writer.writerow({field: preview.get(field, "") for field in fields})
    return previews


def url_probe(url: str, timeout: int = 15) -> dict[str, Any]:
    headers = {"User-Agent": "latentfm-cpu-preflight/1.0", "Range": "bytes=0-65535"}
    req = urllib.request.Request(url, headers=headers)
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            body = response.read(65_536)
            return {
                "ok": True,
                "status": int(response.status),
                "content_length": int(response.headers.get("Content-Length", "0") or 0),
                "content_type": response.headers.get("Content-Type", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "bytes_read": len(body),
                "text_probe": body[:500].decode("utf-8", errors="replace").replace("\n", " ")[:500],
                "error": "",
            }
    except Exception as exc:  # noqa: BLE001 - audit records failure, does not crash.
        return {
            "ok": False,
            "status": None,
            "content_length": None,
            "content_type": "",
            "last_modified": "",
            "bytes_read": 0,
            "text_probe": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_urls() -> list[dict[str, Any]]:
    rows = []
    for spec in SOURCE_URLS:
        rows.append({**spec, **url_probe(spec["url"])})
    return rows


def write_source_manifest(rows: list[dict[str, Any]]) -> None:
    fields = [
        "source",
        "kind",
        "url",
        "ok",
        "status",
        "content_length",
        "content_type",
        "last_modified",
        "bytes_read",
        "download_policy",
        "notes",
        "text_probe",
        "error",
    ]
    with OUT_SOURCE_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_field_assessment() -> list[dict[str, Any]]:
    fields = [
        (
            "phenotypic_activity",
            "condition-level or profile-level activity flag/score derived from Cell Painting morphology signal",
        ),
        (
            "profile_reproducibility",
            "replicate reproducibility / percent-replicating / replicate correlation-like metric",
        ),
        (
            "profile_norm",
            "profile norm or distance-from-control magnitude, if computed after documented plate/batch normalization",
        ),
        (
            "perturbation_mapping",
            "compound/InChIKey/PubChem/JUMP ID or ORF/CRISPR target-gene mapping",
        ),
        (
            "source_batch_plate_well",
            "source, plate, batch, well, cell line, and treatment metadata needed for confound controls",
        ),
    ]
    rows = []
    for field, meaning in fields:
        rows.append(
            {
                "field": field,
                "meaning": meaning,
                "local_schema_confirmed": False,
                "condition_level_usable_if_present": True,
                "required_aggregation": (
                    "aggregate from JUMP/Cell Painting replicate or profile rows to train-only "
                    "condition keys without using canonical multi or Track C held-out query"
                ),
                "current_gate": "blocked_until_local_jump_cp_metadata_and_activity_schema_exist",
            }
        )
    return rows


def gate_decision(
    local_candidates: list[dict[str, Any]],
    source_probes: list[dict[str, Any]],
    schema_previews: list[dict[str, Any]],
    universe_summary: dict[str, Any],
) -> dict[str, Any]:
    usable_sources = [row for row in local_candidates if row["local_source_usable"]]
    readable_schemas = [row for row in schema_previews if row.get("readable")]
    reachable_metadata = [
        row
        for row in source_probes
        if row.get("ok") and row.get("kind") != "s3_profile_prefix_listing_not_download"
    ]
    blockers = []
    if not usable_sources:
        blockers.append("no_local_jump_cp_cell_painting_metadata_or_activity_table")
    if not readable_schemas:
        blockers.append("local_jump_cp_schema_unconfirmed")
    blockers.extend(
        [
            "condition_level_join_not_run",
            "local_condition_overlap_unmeasured",
            "compound_or_gene_perturbation_mapping_unverified",
            "cell_line_background_source_batch_plate_confound_controls_unverified",
            "activity_profile_norm_replicate_reproducibility_fields_unverified",
            "within_dataset_shuffle_control_not_definable_without_local_join",
            "source_block_and_plate_batch_controls_not_definable_without_local_join",
            "mmd_tail_veto_not_definable_without_local_join",
            "chemical_v2_exact_ack_absent",
        ]
    )
    controls_definable = bool(usable_sources and readable_schemas)
    evidence_pass = False
    gpu_authorized = bool(controls_definable and evidence_pass)
    status = "source_preflight_no_gpu" if reachable_metadata else "manual_download_blocker_no_gpu"
    return {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "local_source_state": "absent" if not usable_sources else "present_schema_unvalidated",
        "public_source_state": "reachable_metadata_index" if reachable_metadata else "unverified_or_unreachable",
        "controls_definable": controls_definable,
        "evidence_pass": evidence_pass,
        "blockers": blockers,
        "positive_evidence": [
            "Public JUMP-CP/Cell Painting Gallery metadata or repository endpoints were reachable by small Range/HTML/S3-list probes."
            if reachable_metadata
            else "No public metadata endpoint was confirmed reachable in this run.",
            universe_summary["condition_overlap_interpretation"],
        ],
        "tail_risk_assessment": [
            "Morphology source is external to the expression outcome and could be non-ACK, but only after local metadata proves no held-out query/canonical selection leakage.",
            "Cell line/background mismatch risk is high: JUMP-CP morphology profiles and LatentFM expression conditions may not share backgrounds, batches, doses, or perturbation definitions.",
            "Source/plate/batch confounding risk is high until source-block, plate/batch, and cell-line controls are explicitly measurable.",
            "Profile norm or activity can be an artifact if it mostly captures imaging QC, plating, source, or replicate count rather than perturbation biology.",
        ],
        "required_to_unlock_cpu_signal_gate": [
            "Acquire only small JUMP-CP metadata/activity/reproducibility manifests; do not download profile matrices.",
            "Confirm columns for perturbation IDs, compound names/InChIKey/PubChem or target genes, perturbation type, cell line/background, dose/time, source, plate, well, and batch.",
            "Confirm activity/reproducibility/profile-norm columns and whether they are precomputed, normalized, and replicate-aware.",
            "Join to train-only condition keys and report overlap by dataset, modality, perturbation type, background, and dose.",
            "Require variation across at least three train-only datasets/background/source blocks before treating the field as a source artifact.",
            "Pass within-dataset shuffle, source/block, plate/batch, perturbation-type, background/count, MMD no-harm, and dataset-tail no-harm controls.",
            "Keep Chemical V2 disabled unless the exact ACK is supplied separately.",
        ],
    }


def write_markdown(payload: dict[str, Any]) -> None:
    source_lines = []
    for row in payload["source_probes"]:
        size = row.get("content_length")
        size_text = "NA" if not size else f"{int(size) / 1024:.1f} KiB"
        source_lines.append(
            f"| `{row['source']}` | `{row['kind']}` | `{row.get('status')}` | `{size_text}` | `{row.get('bytes_read')}` | {row['url']} |"
        )

    local_candidates = payload["local_candidates"]
    usable = [row for row in local_candidates if row["local_source_usable"]]
    candidate_lines = [
        f"| `{Path(row['path']).name}` | `{row['kind_guess']}` | `{row['size_bytes']}` | `{row['local_source_usable']}` |"
        for row in local_candidates[:20]
    ]
    if not candidate_lines:
        candidate_lines = ["| none | NA | NA | False |"]

    schema_lines = [
        f"| `{Path(row['path']).name}` | `{row['readable']}` | `{row['field_count']}` | `{row['fields_preview']}` | `{row['error']}` |"
        for row in payload["schema_previews"]
    ]
    if not schema_lines:
        schema_lines = ["| none | False | 0 | NA | no usable local source |"]

    summary = payload["local_universe_summary"]
    gate = payload["gate"]
    field_lines = [
        f"| `{row['field']}` | {row['meaning']} | `{row['local_schema_confirmed']}` | `{row['current_gate']}` |"
        for row in payload["field_assessment"]
    ]
    blockers = "\n".join(f"- `{item}`" for item in gate["blockers"])
    risk = "\n".join(f"- {item}" for item in gate["tail_risk_assessment"])
    required = "\n".join(f"- {item}" for item in gate["required_to_unlock_cpu_signal_gate"])

    md = f"""# JUMP-CP / Cell Painting Morphology Source Gate

Timestamp: `{payload['timestamp']}`

Status: `{gate['status']}`

GPU authorized: `{gate['gpu_authorized']}`

## Boundary

- CPU/report-only source preflight.
- No GPU, training, inference, checkpoints, canonical multi Track A selection,
  Track C held-out query, or large Cell Painting profile matrix download.
- Chemical V2 exact ACK is absent, so this report cannot authorize Chemical V2.
- This route does not repeat DepMap, Replogle, gnomAD, LINCS, QC, or
  guide-support.

## Local Source Scan

Result: `{gate['local_source_state']}`

Usable local JUMP/Cell Painting metadata or activity tables found: `{len(usable)}`

| candidate | kind | size_bytes | usable_local_source |
|---|---:|---:|---:|
{chr(10).join(candidate_lines)}

The scan found no local JUMP-CP/Cell Painting morphology metadata,
activity/profile-norm, or replicate-reproducibility table that can be joined
now. Filename-only weak matches are not accepted as source evidence.

## Local Schema Preview

| candidate | readable | field_count | fields_preview | error |
|---|---:|---:|---|---|
{chr(10).join(schema_lines)}

## Public Metadata Probe

These probes used small S3 listings or first-byte HTML/Range requests only. No
profile matrix was downloaded.

| source | kind | http_status | content_length | bytes_read | url |
|---|---|---:|---:|---:|---|
{chr(10).join(source_lines)}

Interpretation: public metadata entry points are plausibly reachable, but the
route is still blocked until small metadata/activity manifests are present
locally and their schemas are confirmed.

## Local Condition Universe

- train-only outcome conditions: `{summary['trainonly_outcome_conditions']}`
- datasets: `{summary['datasets']}`
- chemical/drug rows: `{summary['chemical_or_drug_rows']}`
- gene/genetic rows: `{summary['gene_or_genetic_rows']}`
- SciPlex3 rows: `{summary['sciplex3_rows']}`
- SciPlex3 backgrounds: `{', '.join(summary['sciplex3_backgrounds']) or 'NA'}`
- SciPlex3 rows with dose: `{summary['sciplex3_conditions_with_dose']}`
- n_cells min/median/max: `{summary['n_cells_min']}` / `{summary['n_cells_median_approx']}` / `{summary['n_cells_max']}`

Gene perturbation overlap is plausible in principle, but not measured. Chemical
overlap is not supported by this local outcome subset. A valid route must show
exact train-only overlap by compound/gene perturbation, background, dose/time,
and source block without using canonical multi or held-out Track C query
information.

## Field Assessment

| field | interpretation | local schema confirmed | current gate |
|---|---|---:|---|
{chr(10).join(field_lines)}

Activity, profile norm, and replicate reproducibility are only acceptable as
external artifacts after documented aggregation to train-only condition keys.
They are not accepted as evidence in this run because no local source schema
or join exists.

## Confound And Tail-Risk Assessment

{risk}

## Controls And Leakage Boundary

- Train-only/internal rows only for any future selection.
- No canonical multi for Track A selection.
- No Track C held-out query.
- No profile matrix download in this preflight.
- Required negative controls before any promotion: within-dataset shuffle,
  source/block, plate/batch, perturbation-type, background/count, MMD no-harm,
  dataset-tail no-harm, and external review.

## Blockers

{blockers}

## Required Next Action

{required}

## Output Files

- JSON: `{OUT_JSON}`
- source manifest: `{OUT_SOURCE_MANIFEST}`
- local candidates: `{OUT_LOCAL_CANDIDATES}`
- local universe: `{OUT_LOCAL_UNIVERSE}`
- schema preview: `{OUT_SCHEMA_PREVIEW}`
"""
    OUT_MD.write_text(md, encoding="utf-8")


def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    outcome_keys = read_outcome_keys()
    universe_rows = read_s0_universe(outcome_keys)
    write_universe(universe_rows)
    universe_summary = summarize_universe(universe_rows)

    local_candidates = scan_local_candidates()
    write_local_candidates(local_candidates)
    schema_previews = write_schema_preview(local_candidates)
    source_probes = probe_urls()
    write_source_manifest(source_probes)

    payload = {
        "timestamp": timestamp,
        "runtime_classification": "Short CPU/file audit and small URL metadata probe.",
        "boundary": {
            "gpu": False,
            "training": False,
            "inference": False,
            "canonical_multi_tracka_selection": False,
            "trackc_heldout_query": False,
            "large_profile_matrix_download": False,
            "chemical_v2_exact_ack": False,
            "repeated_routes_excluded": ["DepMap", "Replogle", "gnomAD", "LINCS", "QC", "guide-support"],
        },
        "local_candidates": local_candidates,
        "schema_previews": schema_previews,
        "source_probes": source_probes,
        "local_universe_summary": universe_summary,
        "field_assessment": build_field_assessment(),
        "gate": gate_decision(local_candidates, source_probes, schema_previews, universe_summary),
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "source_manifest": str(OUT_SOURCE_MANIFEST),
            "local_candidates": str(OUT_LOCAL_CANDIDATES),
            "local_universe": str(OUT_LOCAL_UNIVERSE),
            "schema_preview": str(OUT_SCHEMA_PREVIEW),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
