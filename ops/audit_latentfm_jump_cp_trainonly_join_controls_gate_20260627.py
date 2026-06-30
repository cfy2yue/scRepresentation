#!/usr/bin/env python3
"""Train-only join/control gate for JUMP-CP small gene metadata.

This CPU-only gate consumes the previously materialized JUMP-CP small metadata
tables. It checks whether CRISPR/ORF metadata can be joined to LatentFM
train-only gene conditions and whether the available fields are sufficient for
a source artifact. It deliberately avoids Cell Painting profile matrices.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path("/data/cyx/1030/scLatent")
JUMP_DIR = ROOT / "reports/jump_cp_small_metadata_schema_20260627"
S0 = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUTCOME_FILES = [
    ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
    ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]

CRISPR = JUMP_DIR / "metadata__crispr.csv.gz"
ORF = JUMP_DIR / "metadata__orf.csv.gz"
WELL = JUMP_DIR / "metadata__well.csv.gz"
PLATE = JUMP_DIR / "metadata__plate.csv.gz"
SITE_COUNT = JUMP_DIR / "stats__cpg0016_site_count.csv"
WELL_COUNT = JUMP_DIR / "stats__cpg0016_well_count.csv"

OUT_DIR = ROOT / "reports/jump_cp_trainonly_join_controls_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_jump_cp_trainonly_join_controls_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JUMP_CP_TRAINONLY_JOIN_CONTROLS_GATE_20260627.md"
OUT_GENE = OUT_DIR / "jump_cp_gene_source_features.csv"
OUT_JOIN = OUT_DIR / "jump_cp_s0_gene_join_rows.csv"


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def gene_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value).lower())


def to_float(value: object) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


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
                dataset = norm_text(row.get("dataset"))
                condition = norm_text(row.get("condition"))
                if dataset and condition:
                    keys.add((dataset, condition))
    return keys


def read_s0_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with S0.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "dataset": norm_text(row.get("dataset")),
                    "condition": norm_text(row.get("condition")),
                    "membership": norm_text(row.get("canonical_seed42_membership")),
                    "modality": norm_text(row.get("modality")),
                    "perturbation_type": norm_text(row.get("perturbation_type")),
                    "bucket": norm_text(row.get("bucket")),
                    "nperts": norm_text(row.get("nperts")),
                    "perturbation": norm_text(row.get("perturbation")),
                    "gene": norm_text(row.get("gene")),
                    "cell_background": norm_text(row.get("cell_background_source")),
                    "n_cells": norm_text(row.get("n_cells")),
                }
            )
    return rows


def load_gene_metadata() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
    by_gene: dict[str, list[dict[str, str]]] = defaultdict(list)
    jcp_to_gene: dict[str, dict[str, str]] = {}
    specs = [
        ("crispr", CRISPR),
        ("orf", ORF),
    ]
    for source_modality, path in specs:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = norm_text(row.get("Metadata_Symbol"))
                jcp = norm_text(row.get("Metadata_JCP2022"))
                key = gene_key(symbol)
                if not key or not jcp:
                    continue
                rec = {
                    "jump_modality": source_modality,
                    "jcp_id": jcp,
                    "symbol": symbol,
                    "ncbi_gene_id": norm_text(row.get("Metadata_NCBI_Gene_ID")),
                    "pert_type": norm_text(row.get("Metadata_pert_type")),
                }
                by_gene[key].append(rec)
                jcp_to_gene[jcp] = rec
    return by_gene, jcp_to_gene


def load_plate_map() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with gzip.open(PLATE, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            plate = norm_text(row.get("Metadata_Plate"))
            if not plate:
                continue
            out[plate] = {
                "source": norm_text(row.get("Metadata_Source")),
                "batch": norm_text(row.get("Metadata_Batch")),
                "plate_type": norm_text(row.get("Metadata_PlateType")),
            }
    return out


def load_count_map(path: Path, field: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = norm_text(row.get("Metadata_Source"))
            plate = norm_text(row.get("Metadata_Plate"))
            val = to_float(row.get(field))
            if source and plate and val is not None:
                out[(source, plate)] = val
    return out


def build_gene_features(jcp_to_gene: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    plate_map = load_plate_map()
    site_counts = load_count_map(SITE_COUNT, "n_sites")
    well_counts = load_count_map(WELL_COUNT, "n_wells")
    acc: dict[tuple[str, str], dict[str, object]] = {}
    with gzip.open(WELL, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            jcp = norm_text(row.get("Metadata_JCP2022"))
            meta = jcp_to_gene.get(jcp)
            if not meta:
                continue
            source = norm_text(row.get("Metadata_Source"))
            plate = norm_text(row.get("Metadata_Plate"))
            well = norm_text(row.get("Metadata_Well"))
            pmeta = plate_map.get(plate, {})
            batch = pmeta.get("batch", "")
            plate_type = pmeta.get("plate_type", "")
            key = (gene_key(meta["symbol"]), meta["jump_modality"])
            rec = acc.setdefault(
                key,
                {
                    "gene_key": key[0],
                    "symbol": meta["symbol"],
                    "jump_modality": meta["jump_modality"],
                    "jcp_ids": set(),
                    "sources": set(),
                    "plates": set(),
                    "batches": set(),
                    "plate_types": set(),
                    "wells": set(),
                    "site_values": [],
                    "well_count_values": [],
                },
            )
            rec["jcp_ids"].add(jcp)
            if source:
                rec["sources"].add(source)
            if plate:
                rec["plates"].add(plate)
            if batch:
                rec["batches"].add(batch)
            if plate_type:
                rec["plate_types"].add(plate_type)
            if source and plate and well:
                rec["wells"].add((source, plate, well))
            sc = site_counts.get((source, plate))
            wc = well_counts.get((source, plate))
            if sc is not None:
                rec["site_values"].append(sc)
            if wc is not None:
                rec["well_count_values"].append(wc)

    rows: list[dict[str, object]] = []
    for rec in acc.values():
        site_vals = rec["site_values"]
        well_vals = rec["well_count_values"]
        rows.append(
            {
                "gene_key": rec["gene_key"],
                "symbol": rec["symbol"],
                "jump_modality": rec["jump_modality"],
                "jcp_id_count": len(rec["jcp_ids"]),
                "source_count": len(rec["sources"]),
                "plate_count": len(rec["plates"]),
                "batch_count": len(rec["batches"]),
                "plate_type_count": len(rec["plate_types"]),
                "well_position_count": len(rec["wells"]),
                "site_count_sum": sum(site_vals),
                "site_count_median": median(site_vals) if site_vals else "",
                "well_count_sum": sum(well_vals),
                "well_count_median": median(well_vals) if well_vals else "",
                "sources": ";".join(sorted(rec["sources"])),
                "plate_types": ";".join(sorted(rec["plate_types"])),
            }
        )
    rows.sort(key=lambda r: (str(r["gene_key"]), str(r["jump_modality"])))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def row_gene_keys(row: dict[str, str]) -> set[str]:
    keys = set()
    for field in ("gene", "perturbation", "condition"):
        text = norm_text(row.get(field))
        if not text:
            continue
        for part in re.split(r"[+;,|/]", text):
            key = gene_key(part)
            if key:
                keys.add(key)
    return keys


def join_s0(gene_features: list[dict[str, object]], s0_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for feat in gene_features:
        by_gene[str(feat["gene_key"])].append(feat)

    out: list[dict[str, object]] = []
    for row in s0_rows:
        if row["modality"] != "gene" and row["perturbation_type"] not in {"CRISPRi", "CRISPRa", "CRISPRko", "Cas13", "gene"}:
            continue
        for key in row_gene_keys(row):
            for feat in by_gene.get(key, []):
                out.append(
                    {
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                        "membership": row["membership"],
                        "modality": row["modality"],
                        "perturbation_type": row["perturbation_type"],
                        "bucket": row["bucket"],
                        "nperts": row["nperts"],
                        "s0_gene_key": key,
                        "s0_perturbation": row["perturbation"],
                        "s0_cell_background": row["cell_background"],
                        "jump_symbol": feat["symbol"],
                        "jump_modality": feat["jump_modality"],
                        "jcp_id_count": feat["jcp_id_count"],
                        "source_count": feat["source_count"],
                        "plate_count": feat["plate_count"],
                        "batch_count": feat["batch_count"],
                        "plate_type_count": feat["plate_type_count"],
                        "well_position_count": feat["well_position_count"],
                        "site_count_sum": feat["site_count_sum"],
                        "well_count_sum": feat["well_count_sum"],
                        "sources": feat["sources"],
                        "plate_types": feat["plate_types"],
                    }
                )
    out.sort(key=lambda r: (str(r["dataset"]), str(r["condition"]), str(r["jump_modality"])))
    return out


def summarize_join(rows: list[dict[str, object]]) -> dict[str, object]:
    datasets = Counter(str(r["dataset"]) for r in rows)
    modalities = Counter(str(r["jump_modality"]) for r in rows)
    memberships = Counter(str(r["membership"]) for r in rows)
    source_counts = [int(r["source_count"]) for r in rows if str(r.get("source_count", "")).isdigit()]
    plate_counts = [int(r["plate_count"]) for r in rows if str(r.get("plate_count", "")).isdigit()]
    return {
        "join_rows": len(rows),
        "unique_s0_conditions": len({(r["dataset"], r["condition"]) for r in rows}),
        "unique_s0_genes": len({r["s0_gene_key"] for r in rows}),
        "datasets": len(datasets),
        "dataset_counts_top20": datasets.most_common(20),
        "jump_modality_counts": modalities.most_common(),
        "membership_counts": memberships.most_common(),
        "source_count_min_median_max": [
            min(source_counts) if source_counts else 0,
            median(source_counts) if source_counts else 0,
            max(source_counts) if source_counts else 0,
        ],
        "plate_count_min_median_max": [
            min(plate_counts) if plate_counts else 0,
            median(plate_counts) if plate_counts else 0,
            max(plate_counts) if plate_counts else 0,
        ],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "profile_matrix_downloaded": False,
        "canonical_multi_tracka_selection_used": False,
        "trackc_heldout_query_used": False,
        "chemical_v2_ack": False,
        "source_tables": [
            str(CRISPR),
            str(ORF),
            str(WELL),
            str(PLATE),
            str(SITE_COUNT),
            str(WELL_COUNT),
        ],
    }
    missing = [str(p) for p in (CRISPR, ORF, WELL, PLATE, SITE_COUNT, WELL_COUNT) if not p.is_file()]
    if missing:
        out = {
            "status": "jump_cp_trainonly_join_controls_missing_source_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# JUMP-CP Train-only Join Controls Gate\n\nMissing source files; no GPU authorized.\n", encoding="utf-8")
        print(json.dumps({"status": out["status"], "gpu_authorized": False}, indent=2))
        return 0

    by_gene, jcp_to_gene = load_gene_metadata()
    gene_features = build_gene_features(jcp_to_gene)
    gene_fields = [
        "gene_key",
        "symbol",
        "jump_modality",
        "jcp_id_count",
        "source_count",
        "plate_count",
        "batch_count",
        "plate_type_count",
        "well_position_count",
        "site_count_sum",
        "site_count_median",
        "well_count_sum",
        "well_count_median",
        "sources",
        "plate_types",
    ]
    write_csv(OUT_GENE, gene_features, gene_fields)

    s0_rows = read_s0_rows()
    outcome_keys = read_outcome_keys()
    canonical_train_gene_rows = [
        r
        for r in s0_rows
        if r["membership"] == "train"
        and (r["modality"] == "gene" or r["perturbation_type"] in {"CRISPRi", "CRISPRa", "CRISPRko", "Cas13", "gene"})
    ]
    outcome_gene_rows = [
        r
        for r in s0_rows
        if (r["dataset"], r["condition"]) in outcome_keys
        and (r["modality"] == "gene" or r["perturbation_type"] in {"CRISPRi", "CRISPRa", "CRISPRko", "Cas13", "gene"})
    ]
    train_join = join_s0(gene_features, canonical_train_gene_rows)
    outcome_join = join_s0(gene_features, outcome_gene_rows)
    all_gene_join = join_s0(gene_features, [r for r in s0_rows if r["modality"] == "gene"])
    join_fields = [
        "dataset",
        "condition",
        "membership",
        "modality",
        "perturbation_type",
        "bucket",
        "nperts",
        "s0_gene_key",
        "s0_perturbation",
        "s0_cell_background",
        "jump_symbol",
        "jump_modality",
        "jcp_id_count",
        "source_count",
        "plate_count",
        "batch_count",
        "plate_type_count",
        "well_position_count",
        "site_count_sum",
        "well_count_sum",
        "sources",
        "plate_types",
    ]
    write_csv(OUT_JOIN, train_join, join_fields)

    train_summary = summarize_join(train_join)
    outcome_summary = summarize_join(outcome_join)
    all_gene_summary = summarize_join(all_gene_join)
    global_summary = {
        "jump_gene_symbols": len(by_gene),
        "jump_gene_feature_rows": len(gene_features),
        "jump_jcp_gene_ids": len(jcp_to_gene),
        "canonical_train_gene_rows": len(canonical_train_gene_rows),
        "current_outcome_gene_rows": len(outcome_gene_rows),
        "all_s0_gene_rows": len([r for r in s0_rows if r["modality"] == "gene"]),
    }
    reasons = []
    if train_summary["unique_s0_conditions"] >= 50 and train_summary["datasets"] >= 3:
        reasons.append("trainonly_gene_overlap_passes_size_screen")
    else:
        reasons.append("trainonly_gene_overlap_size_screen_failed")
    reasons.extend(
        [
            "only_source_plate_batch_count_fields_available",
            "activity_reproducibility_profile_norm_missing",
            "cell_background_dose_time_missing",
            "source_plate_batch_are_confound_controls_not_deployable_activity_signal",
            "shuffle_source_mmd_tail_gates_not_applicable_without_candidate_signal",
            "chemical_v2_exact_ack_absent",
            "no_gpu_from_join_controls_only",
        ]
    )
    status = "jump_cp_trainonly_join_controls_fail_no_gpu"
    out = {
        "status": status,
        "gpu_authorized": False,
        "boundary": boundary,
        "global_summary": global_summary,
        "train_join_summary": train_summary,
        "current_outcome_join_summary": outcome_summary,
        "all_s0_gene_join_summary": all_gene_summary,
        "reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "gene_source_features": str(OUT_GENE),
            "train_join_rows": str(OUT_JOIN),
        },
        "next_action": (
            "Close JUMP-CP small metadata as a direct GPU route. Reopen only if "
            "a bounded, reviewed profile-derived activity/reproducibility source "
            "is acquired without violating profile-matrix/download and leakage "
            "constraints, or use the current join as diagnostic coverage evidence."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# JUMP-CP Train-only Join Controls Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only join/control gate over small JUMP-CP metadata.",
        "- No profile matrices, training, inference, canonical multi Track A selection, Track C held-out query, or GPU.",
        "",
        "## Source Coverage",
        "",
        f"- JUMP gene symbols: `{global_summary['jump_gene_symbols']}`",
        f"- JUMP gene feature rows: `{global_summary['jump_gene_feature_rows']}`",
        f"- JUMP gene JCP ids: `{global_summary['jump_jcp_gene_ids']}`",
        f"- canonical train gene rows in S0: `{global_summary['canonical_train_gene_rows']}`",
        "",
        "## Train-only Join",
        "",
        f"- join rows: `{train_summary['join_rows']}`",
        f"- unique S0 conditions: `{train_summary['unique_s0_conditions']}`",
        f"- unique S0 genes: `{train_summary['unique_s0_genes']}`",
        f"- datasets: `{train_summary['datasets']}`",
        f"- modality counts: `{train_summary['jump_modality_counts']}`",
        f"- source count min/median/max: `{train_summary['source_count_min_median_max']}`",
        f"- plate count min/median/max: `{train_summary['plate_count_min_median_max']}`",
        "",
        "## Current Outcome Join",
        "",
        f"- current outcome gene rows: `{global_summary['current_outcome_gene_rows']}`",
        f"- join rows: `{outcome_summary['join_rows']}`",
        f"- unique S0 conditions: `{outcome_summary['unique_s0_conditions']}`",
        f"- datasets: `{outcome_summary['datasets']}`",
        f"- membership counts: `{outcome_summary['membership_counts']}`",
        "",
        "## Decision",
        "",
        "No GPU is authorized. JUMP-CP small metadata has broad train-only gene overlap, but the available fields are perturbation/source/plate/batch coverage fields. Activity, reproducibility, profile norm, cell/background, dose, and time are absent from the small metadata, so the route cannot define a deployable no-harm training signal from this gate alone.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- gene source features: `{OUT_GENE}`",
        f"- train join rows: `{OUT_JOIN}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
