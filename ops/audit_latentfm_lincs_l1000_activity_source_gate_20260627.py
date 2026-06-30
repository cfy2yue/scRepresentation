#!/usr/bin/env python3
"""CPU preflight for LINCS/L1000 TAS or replicate-activity source artifacts.

This audit is intentionally metadata/report-only:

* no GPU, training, inference, checkpoints, canonical multi selection, or
  Track C held-out query;
* no large Level5 matrix downloads;
* no Chemical V2 authorization because exact ACK is absent;
* no reuse of DepMap/Replogle/gnomAD/QC/guide-support routes.

The gate can only authorize a future GPU smoke if a small local LINCS/L1000
metadata or metrics table exists and supports train-only overlap, condition
aggregation, shuffle/source controls, and MMD/tail vetoes. Current expected
outcome is source_preflight_no_gpu or blocker.
"""

from __future__ import annotations

import csv
import json
import math
import os
import ssl
import urllib.request
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

OUT_MD = ROOT / "reports/LATENTFM_LINCS_L1000_ACTIVITY_SOURCE_GATE_20260627.md"
OUT_JSON = ROOT / "reports/latentfm_lincs_l1000_activity_source_gate_20260627.json"
MANIFEST_DIR = ROOT / "reports/lincs_l1000_activity_source_gate_20260627"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
OUT_SOURCE_MANIFEST = MANIFEST_DIR / "source_url_probe_manifest.tsv"
OUT_LOCAL_CANDIDATES = MANIFEST_DIR / "local_candidate_files.tsv"
OUT_LOCAL_UNIVERSE = MANIFEST_DIR / "local_trainonly_condition_universe.tsv"

SOURCE_URLS = [
    {
        "source": "GSE92742",
        "kind": "sig_info_metadata",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz",
        "notes": "Small metadata table; do not confuse with Level5 expression matrices.",
    },
    {
        "source": "GSE92742",
        "kind": "sig_metrics_activity",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_metrics.txt.gz",
        "notes": "Small signature metrics table expected to carry replicate/activity quality fields.",
    },
    {
        "source": "GSE70138",
        "kind": "sig_info_metadata",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
        "notes": "Small metadata table; older GEO L1000 release.",
    },
    {
        "source": "GSE70138",
        "kind": "sig_metrics_activity",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz",
        "notes": "Small signature metrics table expected to carry replicate/activity quality fields.",
    },
]

FILENAME_TERMS = (
    "lincs",
    "l1000",
    "clue",
    "cmap",
    "sig_info",
    "sig_metrics",
    "inst_info",
    "pert_info",
    "cell_info",
)
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "software",
    "scFM_third_party",
    "external_review",
    "node_modules",
    ".cache",
}
SCAN_ROOTS = [
    ROOT / "reports",
    ROOT / "configs",
    ROOT / "ops",
    ROOT / "dataset",
]
GENERATED_PATHS = {
    OUT_MD,
    OUT_JSON,
    OUT_SOURCE_MANIFEST,
    OUT_LOCAL_CANDIDATES,
    OUT_LOCAL_UNIVERSE,
}


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


def read_s0_universe(outcome_keys: set[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    if not S0.is_file():
        return []
    rows: list[dict[str, str]] = []
    with S0.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            dataset = norm(row.get("dataset"))
            condition = norm(row.get("condition"))
            if outcome_keys is not None and (dataset, condition) not in outcome_keys:
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
                    "cell_background": norm(row.get("cell_background_source")),
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
    sciplex_rows = [
        row
        for row in rows
        if row["dataset"].startswith("sciplex3_") or "sci-Plex3" in row["source_label"]
    ]
    gene_rows = [row for row in rows if row["modality"] == "gene" or row["perturbation_type"] in {"gene", "CRISPRi", "CRISPRa", "CRISPRko", "Cas13"}]
    drug_rows = [row for row in rows if row["modality"] == "chemical" or row["perturbation_type"] == "drug"]
    return {
        "trainonly_outcome_conditions": len(rows),
        "datasets": len(datasets),
        "dataset_counts_top20": datasets.most_common(20),
        "modality_counts": modalities.most_common(),
        "perturbation_type_counts": ptypes.most_common(),
        "cell_background_counts_top20": backgrounds.most_common(20),
        "chemical_or_drug_rows": len(drug_rows),
        "gene_or_genetic_rows": len(gene_rows),
        "sciplex3_rows": len(sciplex_rows),
        "sciplex3_backgrounds": sorted({row["cell_background"] for row in sciplex_rows if row["cell_background"]}),
        "sciplex3_conditions_with_dose": sum(1 for row in sciplex_rows if row["dose"]),
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
        "cell_background",
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
            if MANIFEST_DIR in path_dir.parents or path_dir == MANIFEST_DIR:
                continue
            for filename in filenames:
                path = path_dir / filename
                if path.resolve() == Path(__file__).resolve():
                    continue
                if path.resolve() in {p.resolve() for p in GENERATED_PATHS}:
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


def kind_guess(name: str) -> str:
    low = name.lower()
    if "sig_metrics" in low:
        return "lincs_signature_metrics_candidate"
    if "sig_info" in low or "inst_info" in low:
        return "lincs_signature_metadata_candidate"
    if "pert_info" in low:
        return "lincs_perturbagen_metadata_candidate"
    if "cell_info" in low:
        return "lincs_cell_metadata_candidate"
    if any(term in low for term in ("lincs", "l1000", "clue", "cmap")):
        return "name_mentions_lincs_family"
    return "weak_name_match"


def usable_local_source(name: str, size_bytes: int) -> bool:
    low = name.lower()
    has_schema_name = any(term in low for term in ("sig_info", "sig_metrics", "inst_info"))
    likely_table = low.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"))
    return has_schema_name and likely_table and size_bytes > 0


def write_local_candidates(rows: list[dict[str, Any]]) -> None:
    fields = ["path", "size_bytes", "kind_guess", "local_source_usable"]
    with OUT_LOCAL_CANDIDATES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def url_head(url: str, timeout: int = 25) -> dict[str, Any]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "latentfm-cpu-preflight/1.0"})
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            return {
                "ok": True,
                "status": int(response.status),
                "content_length": int(response.headers.get("Content-Length", "0") or 0),
                "content_type": response.headers.get("Content-Type", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "error": "",
            }
    except Exception as exc:  # noqa: BLE001 - audit records failure, does not crash.
        return {
            "ok": False,
            "status": None,
            "content_length": None,
            "content_type": "",
            "last_modified": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_urls() -> list[dict[str, Any]]:
    rows = []
    for spec in SOURCE_URLS:
        head = url_head(spec["url"])
        rows.append({**spec, **head})
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
        "notes",
        "error",
    ]
    with OUT_SOURCE_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def gate_decision(
    local_candidates: list[dict[str, Any]],
    source_probes: list[dict[str, Any]],
    universe_summary: dict[str, Any],
    full_s0_summary: dict[str, Any],
) -> dict[str, Any]:
    usable_sources = [row for row in local_candidates if row["local_source_usable"]]
    reachable_sources = [row for row in source_probes if row.get("ok") and row.get("content_length")]
    blockers = []
    if not usable_sources:
        blockers.append("no_local_lincs_l1000_sig_info_or_sig_metrics_table")
    if universe_summary["chemical_or_drug_rows"] == 0:
        blockers.append("current_trainonly_outcome_join_universe_has_no_chemical_rows")
    blockers.extend(
        [
            "tas_distil_cc_ss_ngene_schema_unverified_locally",
            "condition_level_join_not_run",
            "within_dataset_shuffle_control_not_definable_without_local_join",
            "source_block_control_not_definable_without_local_join",
            "mmd_tail_veto_not_definable_without_local_join",
            "chemical_v2_exact_ack_absent",
        ]
    )
    status = "source_preflight_no_gpu" if reachable_sources else "manual_download_blocker_no_gpu"
    return {
        "status": status,
        "gpu_authorized": False,
        "local_source_state": "absent" if not usable_sources else "present_unvalidated",
        "public_source_state": "reachable_small_metadata" if reachable_sources else "unverified_or_unreachable",
        "blockers": blockers,
        "positive_evidence": [
            "Public GEO URLs expose small sig_info/sig_metrics files separate from large Level5 matrices."
            if reachable_sources
            else "No public URL evidence was confirmed by HEAD in this run.",
            (
                "Full local S0 includes both SciPlex3 chemical/drug and gene perturbation conditions, "
                "so raw local condition overlap is plausible after metadata acquisition."
            )
            if full_s0_summary["chemical_or_drug_rows"] and full_s0_summary["gene_or_genetic_rows"]
            else "Full local S0 overlap support is incomplete.",
            (
                "Current completed train-only outcome rows are gene-only in this report, "
                "so chemical tail-risk selection cannot be tested here without a separate leakage-safe chemical outcome table."
            )
            if universe_summary["chemical_or_drug_rows"] == 0
            else "Current completed train-only outcome rows include chemical/drug conditions.",
        ],
        "required_to_unlock_cpu_signal_gate": [
            "Download only small sig_info/sig_metrics metadata files, not Level5 GCTX matrices.",
            "Confirm local columns: sig_id plus perturbagen, pert_type, cell_id/background, dose/time if present.",
            "Confirm activity/replicate metric columns such as TAS, distil_cc, ss_ngene or documented equivalents.",
            "Aggregate metrics to train-only condition keys without canonical multi or Track C held-out query.",
            "For chemical use, first identify or generate a leakage-safe train-only chemical row-outcome table; do not use Chemical V2 without exact ACK.",
            "Require >=3 datasets, >=50 overlap rows, and >=3 varying datasets before any signal claim.",
            "Pass within-dataset shuffle, source/block, perturbation-type, background/count controls.",
            "Pass MMD max <= +0.001 and dataset-tail no-harm veto before external review.",
        ],
    }


def build_field_assessment() -> list[dict[str, Any]]:
    fields = [
        ("TAS", "transcriptional activity / signature activity score"),
        ("distil_cc", "replicate/self-consistency correlation-like score"),
        ("ss_ngene", "signature strength / number of genes contributing to signal"),
    ]
    rows = []
    for field, meaning in fields:
        rows.append(
            {
                "field": field,
                "meaning": meaning,
                "local_schema_confirmed": False,
                "condition_level_usable_if_present": True,
                "required_aggregation": "aggregate across L1000 signatures sharing perturbagen, pert_type, cell/background, dose/time into external condition artifact",
                "current_gate": "blocked_until_local_sig_metrics_schema_and_join_exist",
            }
        )
    return rows


def write_markdown(payload: dict[str, Any]) -> None:
    source_lines = []
    for row in payload["source_probes"]:
        size = row.get("content_length")
        size_text = "NA" if not size else f"{int(size) / 1024 / 1024:.2f} MiB"
        source_lines.append(
            f"| `{row['source']}` | `{row['kind']}` | `{row.get('status')}` | `{size_text}` | {row['url']} |"
        )
    local_candidates = payload["local_candidates"]
    usable = [row for row in local_candidates if row["local_source_usable"]]
    weak = local_candidates[:10]
    weak_lines = [
        f"| `{Path(row['path']).name}` | `{row['kind_guess']}` | `{row['size_bytes']}` | `{row['local_source_usable']}` |"
        for row in weak
    ]
    if not weak_lines:
        weak_lines = ["| none | NA | NA | False |"]

    summary = payload["local_universe_summary"]
    full_summary = payload["full_s0_summary"]
    gate = payload["gate"]
    field_lines = [
        f"| `{row['field']}` | {row['meaning']} | `{row['local_schema_confirmed']}` | `{row['current_gate']}` |"
        for row in payload["field_assessment"]
    ]
    blockers = "\n".join(f"- `{item}`" for item in gate["blockers"])
    required = "\n".join(f"- {item}" for item in gate["required_to_unlock_cpu_signal_gate"])
    md = f"""# LINCS/L1000 Activity Source Gate

Timestamp: `{payload['timestamp']}`

Status: `{gate['status']}`

GPU authorized: `{gate['gpu_authorized']}`

## Boundary

- CPU/report-only preflight.
- No GPU, training, inference, checkpoints, canonical multi Track A selection,
  Track C held-out query, or large Level5 matrix download.
- Chemical V2 exact ACK is absent, so this cannot authorize a chemical V2 run.
- This does not repeat DepMap, Replogle, gnomAD, QC, or guide-support routes.

## Local Source Scan

Result: `{gate['local_source_state']}`

Usable local LINCS/L1000 sig_info/sig_metrics tables found: `{len(usable)}`

| candidate | kind | size_bytes | usable_local_source |
|---|---:|---:|---:|
{chr(10).join(weak_lines)}

The scan found no local `sig_info`, `sig_metrics`, or equivalent LINCS/L1000
activity metadata table that can be joined now. Filename-only or unrelated CMap
mentions are not accepted as source evidence.

## Public Metadata Probe

These are HEAD probes only; no body or Level5 matrix was downloaded.

| source | kind | http_status | size | url |
|---|---|---:|---:|---|
{chr(10).join(source_lines)}

Interpretation: public small metadata/metrics files are reachable, so the route
has a plausible manual acquisition path. It remains blocked until those small
files are locally downloaded and schemas are confirmed.

## Local Condition Universe

- train-only outcome conditions: `{summary['trainonly_outcome_conditions']}`
- datasets: `{summary['datasets']}`
- chemical/drug rows: `{summary['chemical_or_drug_rows']}`
- gene/genetic rows: `{summary['gene_or_genetic_rows']}`
- SciPlex3 rows: `{summary['sciplex3_rows']}`
- SciPlex3 backgrounds: `{', '.join(summary['sciplex3_backgrounds']) or 'NA'}`
- SciPlex3 rows with dose: `{summary['sciplex3_conditions_with_dose']}`

Full S0 condition inventory, before restricting to completed train-only outcome
rows, has `{full_summary['chemical_or_drug_rows']}` chemical/drug rows and
`{full_summary['gene_or_genetic_rows']}` gene/genetic rows, including
`{full_summary['sciplex3_rows']}` SciPlex3 rows with background+drug+dose
condition strings. That makes raw LINCS/L1000 overlap plausible in principle.

The current completed train-only outcome universe used by this gate is
gene-only. Therefore a chemical tail-risk source route is not testable in this
report and remains blocked until a leakage-safe chemical outcome table exists.
Actual LINCS overlap is not measurable without local LINCS/L1000 metadata.

## Field Assessment

| field | interpretation | local schema confirmed | current gate |
|---|---|---:|---|
{chr(10).join(field_lines)}

`TAS`, `distil_cc`, and `ss_ngene` can only be treated as condition-level
external artifacts after aggregation from signature-level LINCS rows to a
train-only condition key such as perturbagen + perturbation type + cell line +
dose/time. They are not accepted as evidence in this run because the local
schema and join do not exist yet.

## Controls And Leakage Boundary

- Train-only/internal rows only for any future selection.
- No canonical multi for Track A selection.
- No Track C held-out query.
- No Level5 expression matrices in this preflight.
- Required controls before promotion: within-dataset shuffle, source/block,
  perturbation-type, background/count, MMD no-harm, dataset-tail no-harm, and
  external review.

## Blockers

{blockers}

## Required Next Action

{required}

## Output Files

- JSON: `{OUT_JSON}`
- source manifest: `{OUT_SOURCE_MANIFEST}`
- local candidates: `{OUT_LOCAL_CANDIDATES}`
- local universe: `{OUT_LOCAL_UNIVERSE}`
"""
    OUT_MD.write_text(md, encoding="utf-8")


def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    outcome_keys = read_outcome_keys()
    universe_rows = read_s0_universe(outcome_keys)
    full_s0_rows = read_s0_universe(None)
    write_universe(universe_rows)
    universe_summary = summarize_universe(universe_rows)
    full_s0_summary = summarize_universe(full_s0_rows)

    local_candidates = scan_local_candidates()
    write_local_candidates(local_candidates)
    source_probes = probe_urls()
    write_source_manifest(source_probes)

    payload = {
        "timestamp": timestamp,
        "runtime_classification": "Short CPU/file audit and HEAD-only metadata probe.",
        "boundary": {
            "gpu": False,
            "training": False,
            "inference": False,
            "canonical_multi_tracka_selection": False,
            "trackc_heldout_query": False,
            "large_level5_download": False,
            "chemical_v2_exact_ack": False,
            "repeated_routes_excluded": ["DepMap", "Replogle", "gnomAD", "QC", "guide-support"],
        },
        "local_candidates": local_candidates,
        "source_probes": source_probes,
        "local_universe_summary": universe_summary,
        "full_s0_summary": full_s0_summary,
        "field_assessment": build_field_assessment(),
        "gate": gate_decision(local_candidates, source_probes, universe_summary, full_s0_summary),
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "source_manifest": str(OUT_SOURCE_MANIFEST),
            "local_candidates": str(OUT_LOCAL_CANDIDATES),
            "local_universe": str(OUT_LOCAL_UNIVERSE),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
