#!/usr/bin/env python3
"""Build a concrete external condition-artifact acquisition slate.

CPU/report-only. This turns the scaling source-scout gap into prioritized
source leads, local alignment keys, and strict pre-GPU gates. It does not
download data beyond recording already range-probed small-table headers, train,
infer, read checkpoints, read canonical multi, read Track C query outputs, or
use GPU.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "external_condition_artifact_acquisition_slate_20260626"
OUT_MD = REPORTS / "LATENTFM_EXTERNAL_CONDITION_ARTIFACT_ACQUISITION_SLATE_20260626.md"
OUT_JSON = REPORTS / "latentfm_external_condition_artifact_acquisition_slate_20260626.json"
OUT_SOURCES = OUT_DIR / "source_candidate_matrix.csv"
OUT_ALIGNMENT = OUT_DIR / "local_alignment_matrix.csv"
OUT_URLS = OUT_DIR / "url_probe_manifest.tsv"
OUT_GATE = OUT_DIR / "artifact_gate_protocol.csv"


SOURCE_CANDIDATES = [
    {
        "source_id": "gwt_cd4_tcell_perturbseq_2025",
        "priority": "P0",
        "artifact_family": "replicate_concordance_or_pseudobulk_quality",
        "datasets_potentially_informed": "Jiang cytokine/background; Replogle K562/RPE1; generic CRISPRi target-reliability benchmark",
        "verified_small_tables": "guide_kd_efficiency, DE_stats, K562_comparison, sample_metadata",
        "candidate_columns": "guide_mean_expr,ntc_mean_expr,signif_knockdown,crossdonor_correlation_mean,crossguide_correlation,donor_correlation_mean,n_degs_MASH_*",
        "alignment_key": "target gene symbol or Ensembl ID plus culture_condition/background-like context",
        "why_promising": "provides direct guide knockdown, DE, cross-donor, and cross-guide quality fields rather than read-depth/QC proxies",
        "main_risk": "not one of the canonical training datasets; must be used as an external reliability/context prior and cannot substitute for dataset-matched condition-level evidence without overlap controls",
        "next_action": "materialize a small normalized artifact table only for genes overlapping current train-only conditions, then run strict CPU gate",
        "gpu_now": "false",
    },
    {
        "source_id": "sciplex3_drug_dose_time",
        "priority": "P0",
        "artifact_family": "dose_time_viability_growth_or_chemical_context",
        "datasets_potentially_informed": "sciplex3_A549, sciplex3_K562, sciplex3_MCF7",
        "verified_small_tables": "Sci-Plex repo/GEO/Harmonizome condition-level drug-dose-time signature metadata",
        "candidate_columns": "cell_line,drug,dose,timepoint,signature strength,DE gene counts; possible hash-count/cell-yield proxies as negative controls only",
        "alignment_key": "cell_background + drug name + dose; local condition strings include background_drug_dose",
        "why_promising": "directly matches local chemical condition structure and can test dose/time-aware scaling rather than generic drug-level priors",
        "main_risk": "viability/growth may be unavailable as independent side assay; DE signature strength can be outcome-derived and must be kept train-only/internal",
        "next_action": "first build a no-download manifest of exact local condition normalization and source URLs; only then acquire smallest metadata/signature tables",
        "gpu_now": "false",
    },
    {
        "source_id": "norman_replogle_author_pseudobulk",
        "priority": "P1",
        "artifact_family": "replicate_concordance_or_pseudobulk_quality",
        "datasets_potentially_informed": "NormanWeissman2019_filtered, ReplogleWeissman2022_K562_gwps, Replogle_RPE1essential",
        "verified_small_tables": "author supplement or public processed pseudobulk/DE/guide-quality tables to be located",
        "candidate_columns": "replicate correlation, guide-level concordance, pseudobulk DE reproducibility, knockdown effect, usable-condition flag",
        "alignment_key": "dataset + gene symbol; for Norman multi conditions exact gene1+gene2 plus single-gene components",
        "why_promising": "large overlap with current gene perturbation conditions and direct link to perturbation reliability",
        "main_risk": "gene-level static priors or single-dataset Norman-only artifacts cannot authorize GPU; multi-gene names require careful exact/element split handling",
        "next_action": "source scout supplement/figshare/GEO small tables, then materialize only if condition-level or condition+background variation exists",
        "gpu_now": "false",
    },
    {
        "source_id": "jiang_stimulus_background_context",
        "priority": "P1",
        "artifact_family": "background_specific_context_or_pseudobulk_quality",
        "datasets_potentially_informed": "Jiang_IFNB, Jiang_IFNG, Jiang_INS, Jiang_TGFB, Jiang_TNFA",
        "verified_small_tables": "local reports show Jiang background/stimulus structure; external per-background pseudobulk/context table still needed",
        "candidate_columns": "cell_background,stimulus,target_gene,response_program_score,replicate/pseudobulk quality,TF/context activity",
        "alignment_key": "stimulus dataset + target gene + cell_background",
        "why_promising": "tests the user's central cross-background scaling question directly",
        "main_risk": "current local condition labels are gene-only; aggregation across backgrounds can create stimulus/source confound if not split correctly",
        "next_action": "verify whether author supplement has background-resolved pseudobulk/response program tables; otherwise keep as lead",
        "gpu_now": "false",
    },
    {
        "source_id": "frangieh_melanoma_response_quality",
        "priority": "P1",
        "artifact_family": "replicate_concordance_or_viability_growth_burden",
        "datasets_potentially_informed": "Frangieh",
        "verified_small_tables": "processed h5ad obs already exhausted; need author supplement/figshare condition-level quality or melanoma response side table",
        "candidate_columns": "gene perturbation, replicate concordance, pseudobulk DE quality, viability/growth/fitness burden if independently measured",
        "alignment_key": "dataset + gene symbol + A375 background",
        "why_promising": "condition overlap is sizable and background is well-defined",
        "main_risk": "source-specific single-dataset artifact and gene-level fitness priors are confounded; current read/guide-support route already failed",
        "next_action": "look only for true condition-level supplement tables not duplicate read/guide-support obs",
        "gpu_now": "false",
    },
]


LOCAL_ALIGNMENT = [
    {
        "dataset_or_study": "sciplex3_A549/K562/MCF7",
        "local_condition_example": "A549_2Methoxyestradiol_0.001",
        "best_external_artifact": "dose_time_viability_growth_or_chemical_context",
        "local_alignment_key": "dataset + condition; preferably cell_background + drug + dose",
        "risk": "dose strings and units need normalization; viability/growth may be drug-level rather than condition-level",
        "priority": "P0",
    },
    {
        "dataset_or_study": "NormanWeissman2019_filtered",
        "local_condition_example": "KLF1; CEBPE+RUNX1T1; AHR+KLF1",
        "best_external_artifact": "replicate_concordance_or_author_pseudobulk_quality",
        "local_alignment_key": "dataset + exact condition; split gene1+gene2 for component checks",
        "risk": "Norman-only cannot authorize GPU alone; multi-gene guide-level alignment is complex",
        "priority": "P0_acquisition_lead",
    },
    {
        "dataset_or_study": "Jiang_IFNB/IFNG/INS/TGFB/TNFA",
        "local_condition_example": "ADAR; AKT1; BATF2",
        "best_external_artifact": "background_specific_context_or_stimulus_pseudobulk_quality",
        "local_alignment_key": "stimulus dataset + gene + cell_background",
        "risk": "local condition aggregation across backgrounds can introduce stimulus/source confound",
        "priority": "P0/P1",
    },
    {
        "dataset_or_study": "DixitRegev2016_K562_TFs_High_MOI",
        "local_condition_example": "ELF1; CREB1; EGR1",
        "best_external_artifact": "replicate_concordance_or_tf_activity_context",
        "local_alignment_key": "dataset + gene; optional background=K562",
        "risk": "small condition count and TF-only source bias",
        "priority": "P1",
    },
    {
        "dataset_or_study": "Frangieh",
        "local_condition_example": "IFNGR2; JAK2; CD274",
        "best_external_artifact": "pseudobulk_quality_or_viability_growth_burden",
        "local_alignment_key": "dataset + gene + background=A375",
        "risk": "single-source confound; prior read/guide support route already failed",
        "priority": "P1",
    },
    {
        "dataset_or_study": "ReplogleWeissman2022_K562_gwps / Replogle_RPE1essential",
        "local_condition_example": "RPL3; PINK1; TFAM; SLC1A5",
        "best_external_artifact": "replicate_concordance_or_background_specific_dependency_context",
        "local_alignment_key": "dataset + gene + background(K562/RPE1)",
        "risk": "external artifacts are often static gene priors; require condition/background variation and controls",
        "priority": "P1",
    },
    {
        "dataset_or_study": "Nadig_hepg2 / Nadig_jurket",
        "local_condition_example": "TFAM; GFM1; SLC1A5",
        "best_external_artifact": "background_specific_context_or_pseudobulk_quality",
        "local_alignment_key": "dataset + gene; infer background from study name if externally verified",
        "risk": "background often missing in local obs; gene-only route is high risk",
        "priority": "P1",
    },
    {
        "dataset_or_study": "GasperiniShendure2019_lowMOI / Wessels",
        "local_condition_example": "ALDH1A2+CAPZA2; DOT1L+GFI1",
        "best_external_artifact": "combinatorial_replicate_concordance_or_context_interaction",
        "local_alignment_key": "dataset + exact multi condition plus component genes",
        "risk": "multi naming mismatch and interaction confound",
        "priority": "P2",
    },
]


URL_PROBES = [
    {
        "source_id": "gwt_cd4_tcell_perturbseq_2025",
        "url": "https://github.com/emdann/GWT_perturbseq_analysis_2025",
        "probe_status": "web_search_confirmed_repository",
        "header_or_note": "Repository describes additional supplementary tables and metadata for genome-wide perturb-seq in primary human CD4+ T cells.",
    },
    {
        "source_id": "gwt_guide_kd_efficiency",
        "url": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/guide_kd_efficiency.suppl_table.csv",
        "probe_status": "range_header_confirmed_206",
        "header_or_note": "guide_mean_expr,guide_std_expr,guide_n,ntc_mean_expr,ntc_std_expr,ntc_n,t_statistic,p_value,adj_p_value,signif_knockdown,perturbed_gene_id,rank,high_confidence_no_effect_guides,culture_condition",
    },
    {
        "source_id": "gwt_de_stats",
        "url": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/DE_stats.suppl_table.csv",
        "probe_status": "range_header_confirmed_206",
        "header_or_note": "target_contrast_gene_name,culture_condition,n_cells_target,n_total_de_genes,ontarget_effect_size,ontarget_significant,crossdonor_correlation_mean,crossdonor_correlation_min,crossguide_correlation",
    },
    {
        "source_id": "gwt_k562_comparison",
        "url": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/K562_comparison.suppl_table.csv",
        "probe_status": "range_header_confirmed_206",
        "header_or_note": "target_contrast_gene_name,logfc_pearson_r,comparison,condition,donor_correlation_mean,n_degs_MASH_K562,n_degs_MASH_Rest,n_degs_MASH_Stim48hr,n_degs_MASH_Stim8hr",
    },
    {
        "source_id": "gwt_sample_metadata",
        "url": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/sample_metadata.suppl_table.csv",
        "probe_status": "range_header_confirmed_206",
        "header_or_note": "cell_sample_id,10xrun_id,donor_id,culture_condition,library_id,library_prep_kit,probe_hyb_loading,GEM_loading,sequencing_platform",
    },
    {
        "source_id": "sciplex_repo",
        "url": "https://github.com/cole-trapnell-lab/sci-plex",
        "probe_status": "web_search_confirmed_repository",
        "header_or_note": "Repository documents Sci-Plex processing pipelines and primary/secondary analyses, including sciPlex3 large screen.",
    },
    {
        "source_id": "sciplex_harmonizome",
        "url": "https://maayanlab.cloud/Harmonizome/dataset/Sci-Plex%2BDrug%2BPerturbation%2BSignatures",
        "probe_status": "web_search_confirmed_metadata",
        "header_or_note": "Harmonizome describes Sci-Plex drug signatures for A549, K562, and MCF7 cells treated with 188 compounds at 4 doses.",
    },
    {
        "source_id": "sciplex_pmc",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/",
        "probe_status": "web_search_confirmed_paper",
        "header_or_note": "Paper introduces Sci-Plex nuclear hashing for thousands of independent perturbations at single-cell resolution.",
    },
]


GATE_PROTOCOL = [
    {
        "stage": "source_acceptance",
        "criterion": "small_table_first",
        "threshold": "metadata/supplement table only; no large h5ad/expression download in this stage",
        "fail_close": "do not proceed if only large raw data or no documented schema is available",
    },
    {
        "stage": "schema",
        "criterion": "condition_level_keys",
        "threshold": "must map to dataset, condition or dataset, condition, background/dose/time",
        "fail_close": "close if artifact is only dataset-level constant or static target-gene prior",
    },
    {
        "stage": "coverage",
        "criterion": "minimum_overlap",
        "threshold": ">=3 datasets, >=50 overlap rows, >=3 varying datasets",
        "fail_close": "source lead only if coverage or within-dataset variation fails",
    },
    {
        "stage": "statistics",
        "criterion": "effect_and_tail",
        "threshold": "bootstrap lower >0; dataset min >= -0.020; MMD max <= +0.001; within-dataset shuffle p<=0.01",
        "fail_close": "no GPU if CI/tail/MMD/shuffle fails",
    },
    {
        "stage": "confounds",
        "criterion": "source_block_and_controls",
        "threshold": "LODO/source/background/type/family sensitivity positive; partial-control source/count/QC/read/UMI/batch",
        "fail_close": "no weighting/sampler/curriculum if source or QC proxy carries signal",
    },
    {
        "stage": "promotion",
        "criterion": "external_review_then_bounded_gpu",
        "threshold": "strict CPU gate pass plus external audit before any bounded GPU smoke",
        "fail_close": "do not launch GPU directly from acquisition or from single-source pass",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    payload = {
        "timestamp": timestamp,
        "status": "external_condition_artifact_acquisition_slate_ready_no_gpu",
        "gpu_authorized": False,
        "immediate_gpu_candidate_count": 0,
        "source_candidate_count": len(SOURCE_CANDIDATES),
        "p0_candidate_count": sum(1 for row in SOURCE_CANDIDATES if row["priority"] == "P0"),
        "local_alignment_count": len(LOCAL_ALIGNMENT),
        "url_probe_count": len(URL_PROBES),
        "decision": "Start with P0 small-table acquisition for GWT CD4 T cell reliability fields and SciPlex dose/time context; do not launch GPU until strict CPU gate passes.",
        "outputs": {
            "source_candidate_matrix": str(OUT_SOURCES),
            "local_alignment_matrix": str(OUT_ALIGNMENT),
            "url_probe_manifest": str(OUT_URLS),
            "artifact_gate_protocol": str(OUT_GATE),
        },
        "boundary": {
            "cpu_only": True,
            "downloads_large_data": False,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
    }

    write_csv(
        OUT_SOURCES,
        SOURCE_CANDIDATES,
        [
            "source_id",
            "priority",
            "artifact_family",
            "datasets_potentially_informed",
            "verified_small_tables",
            "candidate_columns",
            "alignment_key",
            "why_promising",
            "main_risk",
            "next_action",
            "gpu_now",
        ],
    )
    write_csv(
        OUT_ALIGNMENT,
        LOCAL_ALIGNMENT,
        [
            "dataset_or_study",
            "local_condition_example",
            "best_external_artifact",
            "local_alignment_key",
            "risk",
            "priority",
        ],
    )
    write_csv(OUT_URLS, URL_PROBES, ["source_id", "url", "probe_status", "header_or_note"], delimiter="\t")
    write_csv(OUT_GATE, GATE_PROTOCOL, ["stage", "criterion", "threshold", "fail_close"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM External Condition Artifact Acquisition Slate",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only acquisition slate.",
        "- Does not download large data, train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- URL probes record source schemas/headers only; they are not accepted training artifacts.",
        "",
        "## Decision",
        "",
        f"- {payload['decision']}",
        "- P0 sources: GWT CD4 T-cell Perturb-seq reliability fields and SciPlex3 dose/time chemical context.",
        "- P1 sources: Norman/Replogle/Jiang/Frangieh author pseudobulk or background-context tables.",
        "- GPU remains blocked until strict CPU gate and external audit pass.",
        "",
        "## Source Candidate Matrix",
        "",
        "| priority | source | artifact family | alignment key | next action |",
        "|---|---|---|---|---|",
    ]
    for row in SOURCE_CANDIDATES:
        lines.append(
            f"| `{row['priority']}` | `{row['source_id']}` | {row['artifact_family']} | {row['alignment_key']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Local Alignment Matrix",
            "",
            "| priority | dataset/study | condition example | alignment key | risk |",
            "|---|---|---|---|---|",
        ]
    )
    for row in LOCAL_ALIGNMENT:
        lines.append(
            f"| `{row['priority']}` | `{row['dataset_or_study']}` | `{row['local_condition_example']}` | {row['local_alignment_key']} | {row['risk']} |"
        )
    lines.extend(
        [
            "",
            "## Gate Protocol",
            "",
            "| stage | criterion | threshold | fail-close rule |",
            "|---|---|---|---|",
        ]
    )
    for row in GATE_PROTOCOL:
        lines.append(
            f"| `{row['stage']}` | `{row['criterion']}` | {row['threshold']} | {row['fail_close']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- Source candidate matrix: `{OUT_SOURCES}`",
            f"- Local alignment matrix: `{OUT_ALIGNMENT}`",
            f"- URL probe manifest: `{OUT_URLS}`",
            f"- Artifact gate protocol: `{OUT_GATE}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
