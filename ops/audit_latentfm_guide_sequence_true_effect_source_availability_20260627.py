#!/usr/bin/env python3
"""Audit current source availability for guide-sequence and true-effect routes.

CPU/report-only. This does not train, infer, read canonical multi for
selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/guide_sequence_true_effect_source_availability_20260627"
OUT_CSV = OUT_DIR / "source_availability_matrix.csv"
OUT_JSON = ROOT / "reports/latentfm_guide_sequence_true_effect_source_availability_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_GUIDE_SEQUENCE_TRUE_EFFECT_SOURCE_AVAILABILITY_20260627.md"


def read_header(path: Path, delimiter: str = ",") -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace", newline="") as handle:
        return next(csv.reader(handle, delimiter=delimiter))


def split_counts(dataset: str) -> dict[str, int]:
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    data = payload.get(dataset, {})
    return {k: len(v) for k, v in data.items() if isinstance(v, list)}


def exists(paths: list[str]) -> bool:
    return all((ROOT / p).exists() if not p.startswith("/") else Path(p).exists() for p in paths)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        candidates.append(kwargs)

    norman = ROOT / "reports/external_artifact_sources_20260626/norman_geo/GSE133344_filtered_cell_identities.csv.gz"
    add(
        route="guide_sequence_efficiency",
        dataset="NormanWeissman2019_filtered",
        source_status="local_small_metadata_available",
        local_split_counts=split_counts("NormanWeissman2019_filtered"),
        source_files=[str(norman)],
        observed_columns=read_header(norman) if norman.exists() else [],
        has_guide_sequence=False,
        has_true_effect_size=False,
        has_only_support_or_counts=True,
        current_gate_status="closed_duplicate_reagent_support_no_gpu",
        next_action="No GPU. Only reopen if an external guide-library table with spacer/protospacer sequences is found and frozen before outcome joins.",
    )

    dixit = ROOT / "reports/external_artifact_sources_20260626/dixit_geo/extracted/GSM2396860_k562_tfs_highmoi_cbc_gbc_dict_strict.csv.gz"
    add(
        route="guide_sequence_efficiency",
        dataset="DixitRegev2016_K562_TFs_High_MOI",
        source_status="local_small_metadata_available",
        local_split_counts=split_counts("DixitRegev2016_K562_TFs_High_MOI"),
        source_files=[str(dixit), str(ROOT / "reports/external_artifact_sources_20260626/dixit_geo/filelist.txt")],
        observed_columns=read_header(dixit) if dixit.exists() else [],
        has_guide_sequence=False,
        has_true_effect_size=False,
        has_only_support_or_counts=True,
        current_gate_status="closed_duplicate_reagent_support_no_gpu",
        next_action="No GPU. GEO cbc/gbc dictionaries expose guide IDs and cell barcodes, not guide spacer sequences or condition effect sizes.",
    )

    adamson = ROOT / "reports/external_artifact_sources_20260627/adamson_gasperini_crispri_scout/adamson_gse90546/GSM2406675_10X001_cell_identities.csv.gz"
    add(
        route="adamson_true_response_or_guide_sequence",
        dataset="Adamson",
        source_status="sample_level_cell_identity_small_tables_available_raw_tar_not_downloaded",
        local_split_counts=split_counts("Adamson"),
        source_files=[
            str(adamson),
            str(ROOT / "reports/external_artifact_sources_20260627/adamson_gasperini_crispri_scout/adamson_gse90546/filelist.txt"),
        ],
        observed_columns=read_header(adamson) if adamson.exists() else [],
        has_guide_sequence=False,
        has_true_effect_size=False,
        has_only_support_or_counts=True,
        current_gate_status="adamson_guide_support_preview_diagnostic_no_gpu",
        next_action="No GPU. Existing small tables are read/UMI/coverage/cell-count support. RAW filelist shows matrices/cell identities, not compact DE/effect-size tables.",
    )

    papalexi = ROOT / "reports/external_artifact_sources_20260627/papalexi_gse153056_scout/GSE153056_ECCITE_metadata.tsv.gz"
    add(
        route="guide_sequence_efficiency_or_true_effect",
        dataset="Papalexi",
        source_status="local_small_metadata_available",
        local_split_counts=split_counts("Papalexi"),
        source_files=[str(papalexi)],
        observed_columns=read_header(papalexi, delimiter="\t") if papalexi.exists() else [],
        has_guide_sequence=False,
        has_true_effect_size=False,
        has_only_support_or_counts=True,
        current_gate_status="papalexi_author_metadata_preview_diagnostic_no_gpu",
        next_action="No GPU. Single dataset, n=10 test_single per seed, support/count/phase metadata, MMD-confounded.",
    )

    gwt = ROOT / "reports/gwt_condition_reliability_artifacts_20260626/source_tables/guide_kd_efficiency.csv"
    add(
        route="guide_knockdown_reliability_not_sequence_design",
        dataset="external_GWT_CD4_T_cells",
        source_status="local_small_processed_tables_available",
        local_split_counts={"mapped_local_conditions": "see GWT artifact preflight"},
        source_files=[str(gwt)],
        observed_columns=read_header(gwt) if gwt.exists() else [],
        has_guide_sequence=False,
        has_true_effect_size=True,
        has_only_support_or_counts=False,
        current_gate_status="gwt_condition_reliability_preflight_fail_no_gpu",
        next_action="No GPU. This is gene-level guide knockdown/reliability and has already failed strict preflight; no spacer/protospacer sequence field is present.",
    )

    gasperini = ROOT / "reports/external_artifact_sources_20260627/adamson_gasperini_crispri_scout/gasperini_gse120861/GSE120861_all_deg_results.at_scale.txt.gz"
    add(
        route="true_effect_size",
        dataset="GasperiniShendure2019_lowMOI",
        source_status="local_processed_effect_table_available",
        local_split_counts=split_counts("GasperiniShendure2019_lowMOI"),
        source_files=[str(gasperini)],
        observed_columns=read_header(gasperini, delimiter="\t") if gasperini.exists() else [],
        has_guide_sequence=False,
        has_true_effect_size=True,
        has_only_support_or_counts=False,
        current_gate_status="gasperini_author_self_effect_preview_fail_no_gpu",
        next_action="No GPU. Well-aligned self-effect/knockdown table exists but single-source preview is weak and shuffle-failing.",
    )

    replogle_manifest = ROOT / "configs/latentfm_replogle_bulk_artifact_manifest_20260627.json"
    add(
        route="true_effect_size_or_difficulty",
        dataset="ReplogleWeissman2022_K562_gwps/Replogle_RPE1essential",
        source_status="local_processed_bulk_h5ad_artifacts_available",
        local_split_counts={
            "K562_gwps": split_counts("ReplogleWeissman2022_K562_gwps"),
            "RPE1essential": split_counts("Replogle_RPE1essential"),
        },
        source_files=[str(replogle_manifest)],
        observed_columns=["cnv_score_z", "TE_ratio", "std_leverage_score", "QC controls"],
        has_guide_sequence=False,
        has_true_effect_size=True,
        has_only_support_or_counts=False,
        current_gate_status="replogle_trainonly_internal_feasibility_fail_no_gpu_mmd_confounded",
        next_action="No GPU. Strong difficulty signal exists but is MMD-confounded in strict and train-only gates.",
    )

    frangieh_manifest = ROOT / "configs/latentfm_frangieh_orcs_response_artifact_manifest_20260627.json"
    add(
        route="true_response_or_fitness",
        dataset="Frangieh",
        source_status="local_processed_orcs_response_table_available",
        local_split_counts=split_counts("Frangieh"),
        source_files=[str(frangieh_manifest)],
        observed_columns=["MAGeCK neg/pos lfc/fdr response and fitness columns"],
        has_guide_sequence=False,
        has_true_effect_size=True,
        has_only_support_or_counts=False,
        current_gate_status="frangieh_orcs_response_preview_fail_no_gpu",
        next_action="No GPU. Single-source response/fitness signal below threshold and shuffle/MMD controls fail.",
    )

    jiang_manifest = ROOT / "configs/latentfm_jiang_author_de_artifact_manifest_20260627.json"
    add(
        route="background_response_effect_size",
        dataset="Jiang_*",
        source_status="local_author_de_tables_available",
        local_split_counts={k: split_counts(k) for k in ["Jiang_IFNB", "Jiang_IFNG", "Jiang_INS", "Jiang_TGFB", "Jiang_TNFA"]},
        source_files=[str(jiang_manifest)],
        observed_columns=["background-resolved beta/log2FC/p value variants"],
        has_guide_sequence=False,
        has_true_effect_size=True,
        has_only_support_or_counts=False,
        current_gate_status="jiang_author_de_signal_gate_fail_no_gpu_lodo_fragile",
        next_action="No GPU. Weak signal is dataset/stimulus fragile and flips in LODO.",
    )

    nadig_filelist = ROOT / "reports/external_artifact_sources_20260627/nadig_gse264667_scout/filelist.txt"
    add(
        route="nadig_response_or_guide_sequence",
        dataset="Nadig_hepg2/Nadig_jurket",
        source_status="series_filelist_available_only_large_raw_matrices",
        local_split_counts={"hepg2": split_counts("Nadig_hepg2"), "jurket": split_counts("Nadig_jurket")},
        source_files=[str(nadig_filelist)],
        observed_columns=["10x barcodes/features/matrix filelist only"],
        has_guide_sequence=False,
        has_true_effect_size=False,
        has_only_support_or_counts=False,
        current_gate_status="nadig_gse264667_small_table_scout_negative_no_gpu",
        next_action="No GPU. No compact condition/effect/concordance table found; large h5ad/raw branch would need a separate justified preprocessing plan.",
    )

    depmap = ROOT / "reports/external_artifact_sources_20260627/depmap_24q4_figshare/CRISPRGeneEffect.csv"
    add(
        route="cellline_dependency_effect",
        dataset="DepMap_matched_backgrounds",
        source_status="local_dependency_matrix_available",
        local_split_counts={"matched_local_gene_background_rows": "see DepMap artifact report"},
        source_files=[str(depmap)],
        observed_columns=["CRISPRGeneEffect matrix"],
        has_guide_sequence=False,
        has_true_effect_size=True,
        has_only_support_or_counts=False,
        current_gate_status="depmap_dependency_residual_mmd_gate_fail_no_gpu",
        next_action="No GPU. Real dependency signal exists but fails residual MMD no-harm/confound gate.",
    )

    fields = [
        "route",
        "dataset",
        "source_status",
        "has_guide_sequence",
        "has_true_effect_size",
        "has_only_support_or_counts",
        "current_gate_status",
        "next_action",
        "source_files",
        "observed_columns",
        "local_split_counts",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow({k: json.dumps(row[k], sort_keys=True) if isinstance(row[k], (list, dict)) else row[k] for k in fields})

    immediate_gpu = []
    current_sequence_sources = [r for r in candidates if r["has_guide_sequence"]]
    unresolved_short_sources = []
    payload = {
        "status": "guide_sequence_true_effect_source_availability_no_immediate_gpu",
        "gpu_authorized": False,
        "immediate_gpu_candidates": immediate_gpu,
        "current_guide_sequence_sources": current_sequence_sources,
        "unresolved_short_sources": unresolved_short_sources,
        "candidates": candidates,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Guide-Sequence / True-Effect Source Availability 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only availability audit.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "- Goal: decide whether Helmholtz's guide-sequence / true-response slate has a current short source that can enter a strict CPU gate.",
        "",
        "## Bottom Line",
        "",
        "- Current local/downloaded sources contain `0` usable sgRNA spacer/protospacer sequence sources.",
        "- Several true-effect or response sources exist, but all have already failed their strict or diagnostic gates.",
        "- Existing guide-related small tables are guide identity/read/UMI/coverage/count/reliability metadata, not frozen sequence-design efficiency.",
        "- Therefore no immediate GPU is authorized from this slate.",
        "",
        "## Matrix",
        "",
        "| route | dataset | guide sequence? | true effect? | current status | decision |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['route']}` | `{row['dataset']}` | `{row['has_guide_sequence']}` | `{row['has_true_effect_size']}` | `{row['current_gate_status']}` | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Next Valid Gate",
            "",
            "A guide-sequence route can reopen only if an external guide-library table exposes actual spacer/protospacer sequences for at least 3 datasets and >=50 local conditions, and a frozen external design-efficiency score is computed before outcome joins. The gate must beat read/UMI/count/source controls and pass dataset-tail/MMD no-harm.",
            "",
            "A true-effect source can reopen only if it is materially new relative to the closed GWT/SciPlex/Jiang/DepMap/Replogle/Frangieh/Gasperini/Adamson/Papalexi/Nadig branches, has train/internal discovery-confirm support, and passes shuffle/source/QC/count/tail/MMD controls.",
            "",
            "## Outputs",
            "",
            f"- csv: `{OUT_CSV}`",
            f"- json: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "guide_sequence_sources": len(current_sequence_sources), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
