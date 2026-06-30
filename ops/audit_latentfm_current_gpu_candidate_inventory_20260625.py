#!/usr/bin/env python3
"""Current LatentFM GPU-candidate inventory with stale-pass consumption.

This CPU-only audit is intentionally conservative: earlier pass/candidate gates
are marked consumed when a later seed/no-harm/control/non-noop gate closes the
same branch.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_current_gpu_candidate_inventory_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CURRENT_GPU_CANDIDATE_INVENTORY_20260625.md"


def load(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {"_missing": True, "_path": str(p)}
    return json.loads(p.read_text(encoding="utf-8"))


def status(path: str) -> str | None:
    data = load(path)
    return data.get("status") or (data.get("decision") or {}).get("status")


def main() -> int:
    rows = [
        {
            "branch": "chemical_unseen_seed42_drug_scaffold",
            "latest_status": status("reports/latentfm_chemical_unseen_drug_scaffold_smoke_decision_20260625.json"),
            "immediate_gpu": False,
            "state": "consumed_by_seed_controls",
            "evidence": [
                "reports/LATENTFM_CHEMICAL_UNSEEN_DRUG_SCAFFOLD_SMOKE_DECISION_20260625.md",
                "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_SEED_CONTROL_DECISION_20260625.md",
            ],
            "reason": "seed42 scaffold pass was not replicated; seed controls are 1/3 pass with median family-drug pp +0.001327",
            "next_gate": "none for same-split branch; use V2 protocol only",
        },
        {
            "branch": "chemical_unseen_scaffold_v2_fixedstep_controls",
            "latest_status": status("reports/latentfm_chemical_v2_fixedstep_launcher_protocol_audit_20260625.json"),
            "immediate_gpu": False,
            "state": "protocol_safe_ack_required",
            "evidence": [
                "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_CPU_UNLOCK_20260625.md",
                "reports/LATENTFM_CHEMICAL_V2_FIXEDSTEP_LAUNCHER_PROTOCOL_AUDIT_20260625.md",
                "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_EXTERNAL_AUDIT_LORENTZ_20260625.md",
                "reports/LATENTFM_CHEMICAL_V2_LAUNCH_ACK_EXTERNAL_AUDIT_HERSCHEL_20260625.md",
            ],
            "reason": "launcher is now fixed-latest and train-eval disabled, but external audits still require explicit protocol ACK before GPU",
            "next_gate": "explicit protocol ACK, then launch real_morgan512 seed43/44 before shuffled/random controls",
        },
        {
            "branch": "uncertainty_gated_anchor_fallback",
            "latest_status": status("reports/latentfm_uncertainty_gated_anchor_fallback_nonnoop_gate_20260625.json"),
            "immediate_gpu": False,
            "state": "consumed_by_exact_noop_nonnoop_fail",
            "evidence": [
                "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_GATE_20260625.md",
                "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_CANONICAL_NOHARM_20260625.md",
                "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_NONNOOP_GATE_20260625.md",
            ],
            "reason": "internal pass has zero canonical single/family enabled footprint; mechanism only",
            "next_gate": "new train-only route that maps to nontrivial canonical footprint and passes no-harm",
        },
        {
            "branch": "true_cell_budget128_6k",
            "latest_status": status("reports/latentfm_truecell_nonnoop_tail_protection_meta_gate_20260626.json")
            or status("reports/latentfm_truecell_scaling_count_tail_completion_gate_20260625.json"),
            "immediate_gpu": False,
            "state": "closed_by_nonnoop_tail_protection_meta_gate",
            "evidence": [
                "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_DECISION_20260625.md",
                "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
                "reports/LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
                "reports/LATENTFM_TRUECELL_STRATUM_TAIL_PROTECTION_GATE_20260625.md",
                "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_NONNOOP_GATE_20260625.md",
                "reports/LATENTFM_TRUECELL_RISKROW_COMPLEMENTARITY_GATE_20260625.md",
                "reports/LATENTFM_TRUECELL_NONNOOP_TAIL_PROTECTION_META_GATE_20260626.md",
            ],
            "reason": "strong internal mechanism but existing non-noop/tail-protection routes are consumed: stratum and uncertainty fallback have zero canonical footprint, risk-row fails to protect true-cell canonical tails, and 3/3 budget128 6k seeds fail frozen canonical no-harm",
            "next_gate": "reopen only with a materially new non-noop tail-protection mechanism or external reliability artifact that maps to nonzero canonical footprint and passes tail/MMD/bootstrap/no-harm controls",
        },
        {
            "branch": "allmodality_doseaware",
            "latest_status": status("reports/latentfm_chemical_gene_drug_conflict_isolation_gate_20260625.json"),
            "immediate_gpu": False,
            "state": "closed_by_family_tradeoff_stratified_control_and_gene_drug_isolation",
            "evidence": [
                "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_SMOKE_DECISION_20260625.md",
                "reports/LATENTFM_ALLMODALITY_FAMILY_TRADEOFF_GATE_20260625.md",
                "reports/LATENTFM_ALLMODALITY_MODALITY_ROUTER_CONTROL_GATE_20260625.md",
                "reports/LATENTFM_ALLMODALITY_FAMILY_STRATIFIED_PROTOCOL_GATE_20260625.md",
                "reports/LATENTFM_CHEMICAL_GENE_DRUG_CONFLICT_ISOLATION_GATE_20260625.md",
            ],
            "reason": "0/4 arms pass all/gene/drug gate; router upper bound collapses under count-matched control; 0/56 policies pass; frozen-gene optimistic upper bound has drug hard-harm 0.380 and does not beat shuffle p95",
            "next_gate": "materially new allmod/chemical mechanism with CPU evidence beating shuffle and hard-harm; not simple family-stratified replay or frozen-gene adapter",
        },
        {
            "branch": "visit_cap_curriculum",
            "latest_status": status("reports/latentfm_visit_cap_curriculum_smoke_decision_p07_cap4_20260625.json"),
            "immediate_gpu": False,
            "state": "closed_by_two_internal_failures",
            "evidence": [
                "reports/LATENTFM_VISIT_CAP_CURRICULUM_SMOKE_DECISION_P05_CAP3_20260625.md",
                "reports/LATENTFM_VISIT_CAP_CURRICULUM_SMOKE_DECISION_P07_CAP4_20260625.md",
            ],
            "reason": "p05/cap3 and p07/cap4 both harm cross/family_gene internally",
            "next_gate": "do not mutate visit-cap without materially new CPU mechanism",
        },
        {
            "branch": "condition_count_background_target_ot",
            "latest_status": status("reports/latentfm_source_background_type_hierarchical_matched_gate_20260626.json")
            or status("reports/latentfm_condition_exposure_hierarchical_bootstrap_lodo_gate_20260626.json")
            or status("reports/latentfm_target_observability_residual_v3_gate_20260626.json")
            or status("reports/latentfm_condition_exposure_row_bootstrap_gate_20260625.json")
            or "multiple_fail_no_gpu",
            "immediate_gpu": False,
            "state": "closed_or_diagnostic_row_bootstrap_failed",
            "evidence": [
                "reports/LATENTFM_CONDITION_COUNT_TAIL_SAFE_SUBSET_GATE_20260625.md",
                "reports/LATENTFM_SOURCE_VERIFIED_BACKGROUND_TYPE_V2_GATE_20260625.md",
                "reports/LATENTFM_TARGET_OBSERVABILITY_V2_GATE_20260625.md",
                "reports/LATENTFM_TARGET_OBSERVABILITY_RESIDUAL_V3_GATE_20260626.md",
                "reports/LATENTFM_OT_PAIR_QUALITY_FAILURE_CORRELATION_GATE_20260625.md",
                "reports/LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md",
                "reports/LATENTFM_CONDITION_EXPOSURE_HIERARCHICAL_BOOTSTRAP_LODO_GATE_20260626.md",
                "reports/LATENTFM_SOURCE_BACKGROUND_TYPE_HIERARCHICAL_MATCHED_GATE_20260626.md",
            ],
            "reason": "all recent CPU gates fail tail/control requirements; condition-exposure hierarchical bootstrap/LODO fails with cross CI low -0.017525, dataset min -0.231049, 6 negative dataset tails, signflip p 0.1934, leave-type min -0.022744, seed sign flip, and frozen no-harm failure; source/background/type hierarchical matched gate also fails with pp mean -0.005700, CI [-0.028558,+0.011138], dataset min -0.092902, 3 negative tails, min background/type -0.051013/-0.027994, confound gate failed, and no-harm calibration failed; target residual v3 still fails tail/MMD/within-dataset shuffle controls",
            "next_gate": "fresh non-duplicate CPU mechanism only",
        },
        {
            "branch": "richer_prior_tail_noharm_mechanisms",
            "latest_status": "multiple_fail_no_gpu",
            "immediate_gpu": False,
            "state": "closed_or_artifact_insufficient",
            "evidence": [
                "reports/LATENTFM_MULTIPRIOR_TAILRISK_MASK_GATE_20260625.md",
                "reports/LATENTFM_BACKGROUND_TARGET_ACTIONABILITY_GATE_20260625.md",
                "reports/LATENTFM_RESPONSE_PROGRAM_TRUST_REGION_ARTIFACT_SUFFICIENCY_20260625.md",
                "reports/LATENTFM_RESPONSE_PROGRAM_PROJECTION_GATE_20260625.md",
            ],
            "reason": "multi-prior tail-risk mask fails control/tail gates; background-target actionability has unsafe tails and weak shuffle separation; response-program projection artifact exists but its gate fails CI/tail/hard-harm/MMD criteria",
            "next_gate": "new train-only artifact or mechanism only; do not launch richer-prior GPU from existing artifacts",
        },
        {
            "branch": "scaling_nm_completion",
            "latest_status": status("reports/latentfm_jiang_guide_cytokine_context_gate_20260625.json")
            or status("reports/latentfm_qc_support_reliability_gate_20260625.json")
            or status("reports/latentfm_source_background_type_hierarchical_matched_gate_20260626.json")
            or status("reports/latentfm_condition_exposure_hierarchical_bootstrap_lodo_gate_20260626.json")
            or status("reports/latentfm_condition_exposure_row_bootstrap_gate_20260625.json")
            or "mechanism_map_not_deployable",
            "immediate_gpu": False,
            "state": "scientific_branch_active_no_model_promotion",
            "evidence": [
                "reports/LATENTFM_SCALING_NM_COMPLETION_EXTERNAL_AUDIT_PTOLEMY_20260625.md",
                "reports/LATENTFM_SCALING_COMPLETION_PLAN_20260625.md",
                "reports/LATENTFM_SCALING_NM_CLAIM_MATRIX_V2_20260625.md",
                "reports/LATENTFM_SCALING_COMPLETION_READINESS_20260625.md",
                "reports/LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
                "reports/LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md",
                "reports/LATENTFM_CONDITION_EXPOSURE_HIERARCHICAL_BOOTSTRAP_LODO_GATE_20260626.md",
                "reports/LATENTFM_SOURCE_BACKGROUND_TYPE_HIERARCHICAL_MATCHED_GATE_20260626.md",
                "reports/LATENTFM_NEW_ARTIFACT_SOURCE_FEASIBILITY_20260625.md",
                "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
                "reports/LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
            ],
            "reason": "strongest true-cell/cell-budget internal signal is mechanism-only after count/tail completion and frozen canonical no-harm veto; condition/exposure row-bootstrap plus hierarchical bootstrap/LODO fails; source/background/type hierarchical matched gate fails; broad QC/support metadata fails CI/shuffle/tail gate; Jiang guide/cytokine signal has only 8 overlaps and fails shuffle/tail; other scaling axes are diagnostic, hint-only, or negative",
            "next_gate": "pre-registered nested CPU/statistical package, materially new train-only artifact, or explicit V2 ACK; no direct GPU model promotion",
        },
        {
            "branch": "training_set_metadata_artifacts",
            "latest_status": status("reports/latentfm_jiang_guide_cytokine_context_gate_20260625.json")
            or status("reports/latentfm_qc_support_reliability_gate_20260625.json")
            or status("reports/latentfm_new_artifact_source_feasibility_20260625.json"),
            "immediate_gpu": False,
            "state": "closed_or_supplement_only",
            "evidence": [
                "reports/LATENTFM_NEW_ARTIFACT_SOURCE_FEASIBILITY_20260625.md",
                "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
                "reports/LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
            ],
            "reason": "QC/support is broad but fails bootstrap/shuffle/tail; Jiang guide/cytokine has an intriguing mixscale correlation but only 8 overlaps and unsafe dataset tails; SciPlex technical/chemical columns have 0 current outcome overlap and remain V2-ACK-gated",
            "next_gate": "new outcome-overlap artifact or preregistered external metadata; do not launch generic QC filtering, weighted loss, hard balancing, or Jiang-specialized GPU",
        },
        {
            "branch": "reagent_read_support_source_artifacts",
            "latest_status": status("reports/latentfm_reagent_read_support_source_block_lodo_gate_20260626.json")
            or status("reports/latentfm_reagent_read_support_mmd_safe_residual_gate_20260626.json")
            or status("reports/latentfm_reagent_read_support_combined_signal_gate_20260626.json")
            or status("reports/latentfm_reagent_read_support_combined_preflight_20260626.json"),
            "immediate_gpu": False,
            "state": "source_complete_positive_signal_but_confound_gate_failed",
            "evidence": [
                "reports/LATENTFM_NORMAN_GEO_REAGENT_SIGNAL_GATE_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_MANIFEST_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_PREFLIGHT_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_SIGNAL_GATE_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_MMD_BLOCKER_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_MMD_SAFE_RESIDUAL_GATE_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_SOURCE_BLOCK_LODO_GATE_20260626.md",
                "runs/latentfm_external_artifact_frangieh_processed_download_20260626/RUN_STATUS.md",
                "runs/latentfm_external_artifact_dixit_rawtar_download_retry1_20260626/RUN_STATUS.md",
                "runs/latentfm_external_artifact_dixit_figshare_processed_download_20260626/RUN_STATUS.md",
            ],
            "reason": "Norman/Frangieh/Dixit source acquisition and extraction completed, and read/guide-support has a real MMD-safe residual signal, but the final source-block/LODO confound gate fails because within-dataset shuffle p=0.0999; no weighting/sampler/staged-training GPU smoke is authorized",
            "next_gate": "keep as mechanism/failure-map evidence unless a genuinely new external reliability artifact or stricter source-block signal passes; do not mutate this into GPU training from current evidence",
        },
        {
            "branch": "new_trainonly_artifact_overlap",
            "latest_status": status("reports/latentfm_new_trainonly_artifact_overlap_gate_20260626.json"),
            "immediate_gpu": False,
            "state": "no_unconsumed_obs_artifact_columns",
            "evidence": [
                "reports/LATENTFM_NEW_TRAINONLY_ARTIFACT_OVERLAP_GATE_20260626.md",
                "reports/LATENTFM_NEW_ARTIFACT_SOURCE_FEASIBILITY_20260625.md",
                "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
                "reports/LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
            ],
            "reason": "preflight found zero unclassified non-consumed obs columns with train-only outcome overlap; all existing obs columns are consumed, forbidden label/split columns, chemical ACK-gated, protocol-only, or failed QC/Jiang/source/target families",
            "next_gate": "requires a genuinely new external train-only artifact/metadata source, then bootstrap/shuffle/source/count/tail controls before GPU",
        },
        {
            "branch": "external_reliability_artifact_v2",
            "latest_status": status("reports/latentfm_external_reliability_v2_preflight_20260626.json"),
            "immediate_gpu": False,
            "state": "failed_cpu_preflight_no_gpu",
            "evidence": [
                "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_MANIFEST_20260626.md",
                "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_SOURCE_BLOCK_LODO_GATE_20260626.md",
            ],
            "reason": "independent reliability-v2 artifacts distinct from raw read/UMI depth were tested from existing Norman/Frangieh/Dixit metadata, but assignment fraction, source cell support, and guide multiplicity all fail preflight with negative dataset tails and/or MMD veto; this does not reopen GPU",
            "next_gate": "only a genuinely new external reliability family, such as replicate concordance, dose/time/viability, or source-maturity metadata, should be considered; do not mutate existing read/coverage artifacts into training",
        },
        {
            "branch": "norman_program_growth_artifact",
            "latest_status": status("reports/latentfm_norman_program_growth_artifact_preflight_20260626.json"),
            "immediate_gpu": False,
            "state": "single_dataset_growth_program_fail_no_gpu",
            "evidence": [
                "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACTS_20260626.md",
                "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACT_PREFLIGHT_20260626.md",
                "configs/latentfm_norman_program_growth_artifact_manifest_20260626.json",
            ],
            "reason": "Norman curation gene-program artifacts are non-QC and biologically meaningful, but strict preflight fails: only 1 dataset/1 varying dataset, pp proxy mean -0.103852, dataset min -0.103852, MMD max +0.032024; no training route is authorized",
            "next_gate": "reopen only if comparable growth/viability/program artifacts are materialized for multiple independent datasets and pass source/LODO/tail/MMD controls",
        },
        {
            "branch": "external_source_h5ad_obs_routes",
            "latest_status": status("reports/latentfm_external_source_h5ad_obs_routes_20260626.json"),
            "immediate_gpu": False,
            "state": "no_new_unconsumed_source_obs_artifact",
            "evidence": [
                "reports/LATENTFM_EXTERNAL_SOURCE_H5AD_OBS_ROUTES_20260626.md",
                "reports/LATENTFM_REAGENT_READ_SUPPORT_SOURCE_BLOCK_LODO_GATE_20260626.md",
                "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md",
                "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACT_PREFLIGHT_20260626.md",
            ],
            "reason": "Downloaded Frangieh/Dixit processed h5ad obs fields were scanned in backed mode; Frangieh fields are label/QC/read-UMI-guide/source-protocol only, Dixit fields are label/QC/sg-intergenic guide-support/source-cluster only; no time, viability, growth, dose, replicate-concordance, or program candidate columns remain",
            "next_gate": "requires external acquisition beyond current h5ad obs sources, such as multi-dataset replicate concordance, independent viability/growth, dose/time, or source-maturity artifacts",
        },
        {
            "branch": "harmonizome_depmapcrispr_dependency_artifact",
            "latest_status": status("reports/latentfm_harmonizome_depmapcrispr_artifact_preflight_20260626.json"),
            "immediate_gpu": False,
            "state": "external_dependency_prior_fail_no_gpu",
            "evidence": [
                "reports/LATENTFM_DEPMAP_DEPENDENCY_ARTIFACT_API_PROBE_20260626.md",
                "reports/LATENTFM_HARMONIZOME_DEPMAPCRISPR_ARTIFACTS_20260626.md",
                "reports/LATENTFM_HARMONIZOME_DEPMAPCRISPR_ARTIFACT_PREFLIGHT_20260626.md",
                "reports/LATENTFM_HARMONIZOME_DEPMAPCRISPR_FAILURE_LOCALIZATION_20260626.md",
                "configs/latentfm_harmonizome_depmapcrispr_artifact_manifest_20260626.json",
            ],
            "reason": "DepMap official API was verification-blocked, so Harmonizome DepMap CRISPR was acquired as a legal alternate source; matched-cell-line artifacts have 0 varying datasets/nonzero hits, and broader global target-level artifacts cover 17 datasets/127 rows but fail strict preflight because pp mean is -0.000656, dataset min pp is -0.135215, and MMD max is +0.016631; failure localization shows MMD-safe global rows still fail high-low/shuffle/tail criteria",
            "next_gate": "do not launch dependency-prior GPU from current artifacts; reopen only with richer cell-line-matched dependency/viability source that passes variation, tail, MMD, shuffle, and LODO controls",
        },
        {
            "branch": "gnomad_constraint_tailrisk_artifact",
            "latest_status": status("reports/latentfm_gnomad_constraint_tailrisk_20260626.json"),
            "immediate_gpu": False,
            "state": "external_target_constraint_tailrisk_fail_no_gpu",
            "evidence": [
                "reports/LATENTFM_GNOMAD_CONSTRAINT_ARTIFACTS_20260626.md",
                "reports/LATENTFM_GNOMAD_CONSTRAINT_ARTIFACT_PREFLIGHT_20260626.md",
                "reports/LATENTFM_GNOMAD_CONSTRAINT_TAILRISK_20260626.md",
                "configs/latentfm_gnomad_constraint_artifact_manifest_20260626.json",
            ],
            "reason": "gnomAD v2.1.1 gene-constraint metrics were materialized as target-level artifacts with broad coverage (17 datasets/159 overlap rows), but strict preflight fails due to dataset tail -0.110268 and MMD max +0.016631; tail-risk localization finds missense-z all-rows high-low -0.066798 with shuffle p 0.049975, but this disappears after MMD-safe filtering (p 0.166917), and LOEUF/pLI controls show no stable within-dataset signal",
            "next_gate": "do not launch constraint-weighted sampler/loss GPU from current gnomAD artifacts; reopen only if a target/difficulty artifact remains significant after MMD-safe filtering plus within-dataset shuffle/LODO controls",
        },
        {
            "branch": "scperturb_source_maturity_artifact",
            "latest_status": status("reports/latentfm_scperturb_source_maturity_artifact_preflight_20260626.json"),
            "immediate_gpu": False,
            "state": "source_maturity_dataset_level_fail_no_gpu",
            "evidence": [
                "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACTS_20260626.md",
                "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACT_PREFLIGHT_20260626.md",
                "configs/latentfm_scperturb_source_maturity_artifact_manifest_20260626.json",
            ],
            "reason": "scPerturb catalog source-maturity metadata was materialized across 6 local gene datasets/77 rows, including reported cells, perturbation count, timepoint/dose count, cells per perturbation, and h5ad availability; all artifacts fail strict preflight because they are dataset-level constants with 0 varying datasets and unsafe tails/MMD (for example perturbation-count pp mean -0.030225, dataset min -0.066643, MMD max +0.026384)",
            "next_gate": "do not use source-maturity/catalog scale as a training weight; reopen only with condition-level dose/time/viability or replicate metadata that varies within datasets and passes tail/MMD/shuffle/LODO controls",
        },
        {
            "branch": "replicate_batch_balance_artifact",
            "latest_status": status("reports/latentfm_replicate_batch_balance_artifact_preflight_20260626.json"),
            "immediate_gpu": False,
            "state": "replicate_batch_balance_fail_no_gpu",
            "evidence": [
                "reports/LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACTS_20260626.md",
                "reports/LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACT_PREFLIGHT_20260626.md",
                "configs/latentfm_replicate_batch_balance_artifact_manifest_20260626.json",
            ],
            "reason": "Norman GEO gemgroup plus Dixit/Frangieh scPerturb obs batch/library metadata were materialized into condition-level replicate/batch balance artifacts covering 3 datasets/43 rows; entropy and min-batch-fraction have 2 varying datasets but fail strict preflight with pp mean -0.057773, dataset min -0.066643, and MMD max +0.026384, while batch-count/single-batch controls lack variation",
            "next_gate": "do not launch reliability-aware sampler/staged training from current replicate/batch artifacts; reopen only with true multi-dataset replicate concordance or author pseudobulk concordance that is not batch/source/count confounded and passes strict controls",
        },
        {
            "branch": "background_specific_grn_context_artifact",
            "latest_status": status("reports/latentfm_background_specific_grn_context_source_audit_20260626.json"),
            "immediate_gpu": False,
            "state": "no_background_specific_grn_source_no_gpu",
            "evidence": [
                "reports/LATENTFM_BACKGROUND_SPECIFIC_GRN_CONTEXT_SOURCE_AUDIT_20260626.md",
                "reports/LATENTFM_TRACKA_OMNIPATH_TF_PRIOR_METRIC_GATE_20260623.md",
                "reports/LATENTFM_TRACKA_OMNIPATH_RESPONSE_MODULE_GATE_20260623.md",
                "reports/LATENTFM_TRACKC_OMNIPATH_PAIR_PRIOR_PREFLIGHT_20260623.md",
            ],
            "reason": "existing OmniPath/CollecTRI/DoRothEA files are gene-level or TF-target pair-level, not background-specific; static TF prior, response-neighbor module, and Track C pair-prior coverage gates are already closed, so reusing these files would duplicate target/actionability/OmniPath failures rather than create a new scaling axis",
            "next_gate": "reopen only with a genuinely background-specific GRN source that varies by dataset/background and beats target/dependency/constraint/generic-OmniPath controls under MMD/tail/shuffle/LODO gates",
        },
        {
            "branch": "condition_level_reliability_source_scout",
            "latest_status": status("reports/latentfm_condition_level_reliability_source_scout_20260626.json"),
            "immediate_gpu": False,
            "state": "no_local_ready_candidate_external_acquisition_required",
            "evidence": [
                "reports/LATENTFM_CONDITION_LEVEL_RELIABILITY_SOURCE_SCOUT_20260626.md",
                "reports/condition_level_reliability_source_scout_20260626/candidate_source_matrix.csv",
                "reports/condition_level_reliability_source_scout_20260626/scaling_law_completion_experiment_matrix.csv",
                "reports/condition_level_reliability_source_scout_20260626/mainline_translation.csv",
            ],
            "reason": "subagent and local source-scout audit found zero immediate local condition-level reliability artifacts; existing obs/source-derived candidates are already closed or source leads only. Remaining non-duplicate scaling unlocks require external small tables such as replicate concordance, dose/time/viability/growth, or true background-specific context, followed by strict CPU gates",
            "next_gate": "acquire a small external condition-level artifact table, then require >=3 datasets, >=50 overlap rows, >=3 varying datasets, bootstrap lower >0, dataset min >= -0.020, MMD max <= +0.001, within-dataset shuffle p<=0.01, LODO/source-block pass, and confound controls before any GPU",
        },
        {
            "branch": "external_condition_artifact_acquisition_slate",
            "latest_status": status("reports/latentfm_external_condition_artifact_acquisition_slate_20260626.json"),
            "immediate_gpu": False,
            "state": "source_leads_ready_materialization_not_done_no_gpu",
            "evidence": [
                "reports/LATENTFM_EXTERNAL_CONDITION_ARTIFACT_ACQUISITION_SLATE_20260626.md",
                "reports/external_condition_artifact_acquisition_slate_20260626/source_candidate_matrix.csv",
                "reports/external_condition_artifact_acquisition_slate_20260626/local_alignment_matrix.csv",
                "reports/external_condition_artifact_acquisition_slate_20260626/url_probe_manifest.tsv",
                "reports/external_condition_artifact_acquisition_slate_20260626/artifact_gate_protocol.csv",
            ],
            "reason": "P0/P1 source leads and local alignment keys are defined, including GWT CD4 T-cell reliability fields and SciPlex3 dose/time chemical context, but no artifact has been materialized or passed the strict CPU gate; this is an acquisition plan, not training evidence",
            "next_gate": "GWT P0 is now materialized and failed preflight; next external-artifact route is SciPlex dose/time small-table materialization or another truly condition-level P1 source, followed by strict CPU gate before any GPU",
        },
        {
            "branch": "gwt_condition_reliability_artifact",
            "latest_status": status("reports/latentfm_gwt_condition_reliability_artifact_preflight_20260627.json")
            or status("reports/latentfm_gwt_condition_reliability_artifacts_20260626.json"),
            "immediate_gpu": False,
            "state": "materialized_but_strict_preflight_failed_no_gpu",
            "evidence": [
                "reports/LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACTS_20260626.md",
                "configs/latentfm_gwt_condition_reliability_artifact_manifest_20260626.json",
                "reports/LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACT_PREFLIGHT_20260627.md",
                "reports/latentfm_gwt_condition_reliability_artifact_preflight_20260627.json",
                "reports/latentfm_gwt_condition_reliability_artifact_preflight_20260627_rows.csv",
            ],
            "reason": "GWT guide knockdown/crossguide/crossdonor/K562 comparison artifacts cover many datasets and varying rows, but all four fail strict preflight by dataset-tail and MMD veto; k562_logfc_pearson has pp mean +0.023407 but dataset min -0.181065 and MMD max +0.346044, so it is unsafe/source-confounded as a training signal",
            "next_gate": "do not mutate GWT gene-level reliability into weighting/sampler GPU; use as negative evidence unless a future source provides dataset-matched condition-level replicate concordance that passes tail/MMD/shuffle/LODO controls",
        },
        {
            "branch": "trackc_support_and_routed_distill",
            "latest_status": "multiple_trackc_fail_no_gpu",
            "immediate_gpu": False,
            "state": "closed_before_query",
            "evidence": [
                "reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_route_condprior_w05_replay1_2k_seed42.md",
                "reports/LATENTFM_TRACKC_ACTIVE_STATUS_CLOSURE_20260624.md",
                "reports/LATENTFM_TRACKC_TRAINMULTI_ROW_RELIABILITY_ARTIFACT_20260624.md",
                "reports/LATENTFM_TRACKC_ROW_RELIABILITY_V2_GATE_20260624.md",
            ],
            "reason": "routed-distill support/canonical gate failed; trainmulti row-level reliability artifact exists but V2 gate has 0/72 pass specs and unsafe negative enabled rows; no held-out query eval",
            "next_gate": "materially new support-only mechanism on safe trainselect split, not row-reliability V2 or routed-distill",
        },
    ]
    immediate = [r for r in rows if r["immediate_gpu"]]
    payload = {
        "status": "latentfm_current_gpu_candidate_inventory_no_immediate_gpu",
        "gpu_authorized": False,
        "immediate_gpu_candidate_count": len(immediate),
        "rows": rows,
        "decision": {
            "resource_blocker": False,
            "evidence_blocker": True,
            "next_action": "wait for next-slate audit or explicit V2 protocol ACK; otherwise run CPU-first non-duplicate gates",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Current GPU Candidate Inventory",
        "",
        "Status: `latentfm_current_gpu_candidate_inventory_no_immediate_gpu`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of recent gate/decision reports.",
        "- Does not train, infer, read canonical multi, read Track C query, or use GPU.",
        "- Marks stale pass reports as consumed when later gates closed the branch.",
        "",
        "## Inventory",
        "",
        "| branch | immediate GPU | state | latest status | reason | next gate |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['branch']}` | `{row['immediate_gpu']}` | `{row['state']}` | "
            f"`{row['latest_status']}` | {row['reason']} | {row['next_gate']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- Current blocker is evidence/protocol, not GPU availability.",
        "- Do not reuse earlier pass/candidate reports without checking the later consuming veto listed here.",
        "- V2 fixed-step controls are technically safe after `TRAIN_EVAL_ENABLED=0`, but still require explicit protocol ACK before GPU.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
