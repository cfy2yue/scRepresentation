#!/usr/bin/env python3
"""Build a condition-level reliability/scaling source scout.

CPU/report-only. This script inventories local closed source-artifact routes,
defines the smallest external condition-level artifact acquisitions that could
reopen scaling, and translates the current scaling evidence into mainline
training-set guidance. It does not download data, train, infer, read
checkpoints, read canonical multi, read Track C query outputs, or use GPU.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "condition_level_reliability_source_scout_20260626"
OUT_MD = REPORTS / "LATENTFM_CONDITION_LEVEL_RELIABILITY_SOURCE_SCOUT_20260626.md"
OUT_JSON = REPORTS / "latentfm_condition_level_reliability_source_scout_20260626.json"
OUT_CLOSED = OUT_DIR / "closed_local_sources.csv"
OUT_CANDIDATES = OUT_DIR / "candidate_source_matrix.csv"
OUT_COMPLETION = OUT_DIR / "scaling_law_completion_experiment_matrix.csv"
OUT_TRANSLATION = OUT_DIR / "mainline_translation.csv"
OUT_INPUTS = OUT_DIR / "input_manifest.tsv"


CLOSED_LOCAL_SOURCES = [
    {
        "route": "read_umi_qc_assignment_sourcecell_guide_multiplicity",
        "latest_status": "external_reliability_v2_preflight_fail_no_gpu",
        "why_closed": "existing reliability-v2 fields are read/UMI/QC or source/protocol proxies; negative tails and/or MMD/source-block controls fail",
        "evidence": "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md",
        "reusable_as": "negative/control evidence only",
    },
    {
        "route": "local_h5ad_obs_columns",
        "latest_status": "no_new_unconsumed_condition_artifact",
        "why_closed": "rich obs scan found label/QC/protocol/source columns but no unconsumed time, dose, viability, growth, replicate-concordance, or program-quality columns",
        "evidence": "reports/LATENTFM_LOCAL_H5AD_OBS_ARTIFACT_PREFLIGHT_20260626.md; reports/LATENTFM_RICH_H5AD_OBS_METADATA_ROUTES_20260626.md",
        "reusable_as": "source inventory",
    },
    {
        "route": "external_frangieh_dixit_h5ad_obs",
        "latest_status": "external_source_h5ad_obs_no_new_artifact_candidates_no_gpu",
        "why_closed": "downloaded processed obs fields are consumed label/QC/read-UMI/guide/source-cluster/protocol fields",
        "evidence": "reports/LATENTFM_EXTERNAL_SOURCE_H5AD_OBS_ROUTES_20260626.md",
        "reusable_as": "source inventory",
    },
    {
        "route": "scperturb_source_maturity",
        "latest_status": "source_maturity_dataset_level_fail_no_gpu",
        "why_closed": "catalog maturity fields are dataset-level constants with zero varying datasets; cannot support condition-level weighting",
        "evidence": "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACT_PREFLIGHT_20260626.md",
        "reusable_as": "dataset/source confound negative evidence",
    },
    {
        "route": "replicate_batch_balance",
        "latest_status": "replicate_batch_balance_fail_no_gpu",
        "why_closed": "batch/library balance is not replicate concordance and fails pp/tail/MMD controls",
        "evidence": "reports/LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACT_PREFLIGHT_20260626.md",
        "reusable_as": "batch-confound negative evidence",
    },
    {
        "route": "norman_program_growth",
        "latest_status": "single_dataset_growth_program_fail_no_gpu",
        "why_closed": "biologically meaningful Norman-only program/growth labels cover one dataset and fail tail/MMD/preflight criteria",
        "evidence": "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACT_PREFLIGHT_20260626.md",
        "reusable_as": "example source schema and single-dataset caution",
    },
    {
        "route": "generic_omnipath_collectri_dorothea_grn",
        "latest_status": "no_background_specific_grn_source_no_gpu",
        "why_closed": "existing GRN priors are static gene/TF-target priors, not background-specific context artifacts; reusing them duplicates closed prior routes",
        "evidence": "reports/LATENTFM_BACKGROUND_SPECIFIC_GRN_CONTEXT_SOURCE_AUDIT_20260626.md",
        "reusable_as": "negative boundary for background-context claims",
    },
    {
        "route": "condition_exposure_source_background_type_scaling",
        "latest_status": "hierarchical_matched_gates_fail_no_gpu",
        "why_closed": "condition exposure has positive local means but unstable CI/tails/no-harm; source/background/type matched gate has negative mean and confound/no-harm failures",
        "evidence": "reports/LATENTFM_CONDITION_EXPOSURE_HIERARCHICAL_BOOTSTRAP_LODO_GATE_20260626.md; reports/LATENTFM_SOURCE_BACKGROUND_TYPE_HIERARCHICAL_MATCHED_GATE_20260626.md",
        "reusable_as": "failure-map and reviewer control evidence",
    },
]


CANDIDATE_SOURCES = [
    {
        "candidate": "replicate_concordance_or_author_pseudobulk_quality",
        "priority": "P0",
        "local_ready": "false",
        "artifact_schema": "dataset,condition,artifact_value,n_replicates,metric_name,source,evidence_url",
        "minimal_source": "author supplement, GEO/figshare tables, replicate-level pseudobulk DE, replicate correlation, guide/replicate concordance",
        "why_new": "measures reproducibility of the perturbation effect rather than read depth, batch balance, or source maturity",
        "cpu_gate": ">=3 datasets; >=50 overlap rows; >=3 varying datasets; within-dataset variation; bootstrap lower > 0; dataset min >= -0.020; MMD max <= +0.001; within-dataset shuffle p <= 0.01; LODO/source-block pass; partial-control source/count/QC/read/UMI/batch",
        "gpu_unlock": "only after strict CPU gate plus external audit; then bounded reliability-aware sampler/loss/staged-training smoke",
        "stop_rule": "close if metric is dataset-level constant, source-block carried, overlaps <50, varying datasets <3, or any tail/MMD/shuffle/LODO gate fails",
        "mainline_use_if_pass": "candidate training-set weighting or curriculum signal",
    },
    {
        "candidate": "dose_time_viability_growth_side_assay",
        "priority": "P0",
        "local_ready": "false",
        "artifact_schema": "dataset,condition,artifact_value,assay,time_unit,dose_unit,target_gene,source,evidence_url",
        "minimal_source": "independent condition-level time, dose, viability, toxicity, fitness, growth, or side-assay tables",
        "why_new": "captures biological perturbation strength or cellular tolerance, not cell-count/QC proxies",
        "cpu_gate": "same strict condition-level gate; additionally require independent assay provenance and no use of held-out labels",
        "gpu_unlock": "only if signal remains after source/count/QC controls and improves tails rather than selecting easy/source-specific rows",
        "stop_rule": "close if it is Norman-only, dataset-only, or just a cell-count/read-depth proxy",
        "mainline_use_if_pass": "dose/time-aware sampling, stage-specific curriculum, or no-harm filter",
    },
    {
        "candidate": "background_specific_context_or_grn",
        "priority": "P1",
        "local_ready": "false",
        "artifact_schema": "dataset,cell_background,condition,artifact_value,source,evidence_url or dataset,cell_background,tf,target,edge_confidence",
        "minimal_source": "context-specific TF activity, regulon, enhancer-target, or background-specific GRN/source table",
        "why_new": "tests whether cross-background scaling works when the target/context relation is explicitly background-aware",
        "cpu_gate": "condition-level aggregation plus matched background/type/source controls, LODO by background/type/source, and target-prior negative controls",
        "gpu_unlock": "only if it beats generic target/actionability/OmniPath/dependency controls and protects tails",
        "stop_rule": "close if source is static gene-level or pair-level prior without background variation",
        "mainline_use_if_pass": "background-conditioned adapter or sampler stratum",
    },
    {
        "candidate": "condition_level_external_reliability_meta_table",
        "priority": "P1",
        "local_ready": "false",
        "artifact_schema": "dataset,condition,artifact_value,artifact_family,source,evidence_url",
        "minimal_source": "curated small tables from publications that directly rate perturbation quality, response consistency, or author-reported usable-condition status",
        "why_new": "would provide a condition-level reliability prior independent of current obs/QC/read/source artifacts",
        "cpu_gate": "same strict condition-level gate, plus family-block sensitivity so one artifact family cannot carry the effect",
        "gpu_unlock": "only if at least one artifact family and leave-family analyses pass",
        "stop_rule": "close if author quality flags are source-wide constants or duplicate existing QC/read-support measures",
        "mainline_use_if_pass": "condition reliability weighting with family/dropout controls",
    },
]


SCALING_COMPLETION_MATRIX = [
    {
        "axis": "true_cell_or_cell_budget",
        "current_result": "strongest positive mechanism signal: budget128 6k internal cross/family/MMD +0.059142/+0.062067/-0.001395",
        "why_not_law_ready": "frozen canonical no-harm failed all seeds; existing non-noop/tail-protection routes have zero or unsafe canonical footprint",
        "nm_level_missing": "non-noop tail-protection mechanism with frozen canonical no-harm and multi-seed stability",
        "gpu_status": "no_gpu_from_existing_artifacts",
    },
    {
        "axis": "condition_exposure_or_condition_count",
        "current_result": "moderate exposure has positive means but hierarchical CI crosses zero and tails are unsafe",
        "why_not_law_ready": "dataset min -0.231049, six negative dataset tails, signflip p about 0.19/0.21, seed sign flip, and no-harm failure",
        "nm_level_missing": "pre-registered exposure grid with nested dataset/condition bootstrap, LODO, full-vs-moderate, and tail/no-harm controls",
        "gpu_status": "closed_current_route",
    },
    {
        "axis": "background_type_source_breadth",
        "current_result": "source/background/type matched gate fails with pp mean -0.005700 and CI [-0.028558,+0.011138]",
        "why_not_law_ready": "negative tails, background/type minima below zero, confound and no-harm calibration fail",
        "nm_level_missing": "balanced design or external context artifact that varies within datasets/backgrounds and passes LODO",
        "gpu_status": "closed_current_route",
    },
    {
        "axis": "target_or_prior_observability",
        "current_result": "dependency, constraint, actionability, response-program, and target-observability routes are diagnostic or failed",
        "why_not_law_ready": "signals collapse under MMD-safe filtering, within-dataset shuffle, or tail controls",
        "nm_level_missing": "condition-level or background-matched prior that beats target-level/source confounds",
        "gpu_status": "no_gpu",
    },
    {
        "axis": "perturbation_type_or_chemical_semantics",
        "current_result": "chemical V2 is the only prepared GPU route but exact ACK-gated; allmodality/dose-aware routes fail hard-harm and controls",
        "why_not_law_ready": "current chemical evidence is protocol-gated and not integrated into general scaling law",
        "nm_level_missing": "ACKed V2 fixed-step controls and independent descriptor/control replication",
        "gpu_status": "ack_required",
    },
    {
        "axis": "ot_minibatch_pair_quality",
        "current_result": "OT is not authorized as a training improvement from current evidence",
        "why_not_law_ready": "pair-quality/causal effect and tail/no-harm benefit have not passed a strict non-duplicate CPU gate",
        "nm_level_missing": "pair-quality audit with null/shuffle, tail, MMD, and downstream no-harm mapping before any OT weighting/sampling",
        "gpu_status": "default_off",
    },
]


MAINLINE_TRANSLATION = [
    {
        "insight": "do not assume monotonic more-data scaling",
        "mainline_action": "avoid generic full-exposure, hard broadening, or source-balanced GPU runs unless a CPU gate shows tail-safe benefit",
        "evidence": "condition exposure and source/background/type hierarchical gates fail",
    },
    {
        "insight": "cell support is useful but not sufficient",
        "mainline_action": "keep true-cell/cell-budget support as a cautious design prior, but require non-noop tail protection and frozen canonical no-harm before training",
        "evidence": "true-cell budget128 6k positive internal signal but no-harm veto",
    },
    {
        "insight": "dataset/source/background/type are confound strata",
        "mainline_action": "use these as audit strata and LODO blockers, not as simple weights or claims of scaling",
        "evidence": "source/background/type matched gate and source-maturity/replicate-batch failures",
    },
    {
        "insight": "condition-level reliability may be the next useful training-set axis",
        "mainline_action": "acquire small external condition-level tables first, then run strict CPU gates before any sampler/loss/curriculum smoke",
        "evidence": "local obs/source artifacts exhausted; subagent audit found no immediate materializable candidate",
    },
    {
        "insight": "OT pair construction remains an audit point",
        "mainline_action": "keep OT default-off until pair quality predicts tail-safe gains under null/shuffle controls",
        "evidence": "OT routes already in closed/diagnostic inventory without current GPU authorization",
    },
]


INPUTS = [
    "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md",
    "reports/LATENTFM_LOCAL_H5AD_OBS_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_RICH_H5AD_OBS_METADATA_ROUTES_20260626.md",
    "reports/LATENTFM_EXTERNAL_SOURCE_H5AD_OBS_ROUTES_20260626.md",
    "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_BACKGROUND_SPECIFIC_GRN_CONTEXT_SOURCE_AUDIT_20260626.md",
    "reports/LATENTFM_CONDITION_EXPOSURE_HIERARCHICAL_BOOTSTRAP_LODO_GATE_20260626.md",
    "reports/LATENTFM_SOURCE_BACKGROUND_TYPE_HIERARCHICAL_MATCHED_GATE_20260626.md",
    "reports/LATENTFM_SCALING_LAW_READY_EVIDENCE_TABLE_20260626.md",
    "reports/LATENTFM_CURRENT_GPU_CANDIDATE_INVENTORY_20260625.md",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    input_rows = []
    for rel in INPUTS:
        path = ROOT / rel
        input_rows.append(
            {
                "path": str(path),
                "exists": str(path.exists()).lower(),
                "size": path.stat().st_size if path.exists() else "",
            }
        )

    local_ready = [row for row in CANDIDATE_SOURCES if row["local_ready"] == "true"]
    payload = {
        "timestamp": timestamp,
        "status": "condition_level_reliability_source_scout_no_local_ready_candidate",
        "gpu_authorized": False,
        "immediate_gpu_candidate_count": 0,
        "local_ready_candidate_count": len(local_ready),
        "closed_local_source_count": len(CLOSED_LOCAL_SOURCES),
        "candidate_source_count": len(CANDIDATE_SOURCES),
        "scaling_axis_count": len(SCALING_COMPLETION_MATRIX),
        "decision": "No local condition-level reliability/scaling artifact is ready for GPU. Complete scaling law requires external small-table acquisition plus strict CPU gates.",
        "outputs": {
            "closed_local_sources": str(OUT_CLOSED),
            "candidate_source_matrix": str(OUT_CANDIDATES),
            "scaling_law_completion_experiment_matrix": str(OUT_COMPLETION),
            "mainline_translation": str(OUT_TRANSLATION),
            "input_manifest": str(OUT_INPUTS),
        },
        "boundary": {
            "cpu_only": True,
            "downloads": False,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
    }

    write_csv(OUT_CLOSED, CLOSED_LOCAL_SOURCES, ["route", "latest_status", "why_closed", "evidence", "reusable_as"])
    write_csv(
        OUT_CANDIDATES,
        CANDIDATE_SOURCES,
        [
            "candidate",
            "priority",
            "local_ready",
            "artifact_schema",
            "minimal_source",
            "why_new",
            "cpu_gate",
            "gpu_unlock",
            "stop_rule",
            "mainline_use_if_pass",
        ],
    )
    write_csv(
        OUT_COMPLETION,
        SCALING_COMPLETION_MATRIX,
        ["axis", "current_result", "why_not_law_ready", "nm_level_missing", "gpu_status"],
    )
    write_csv(OUT_TRANSLATION, MAINLINE_TRANSLATION, ["insight", "mainline_action", "evidence"])
    write_tsv(OUT_INPUTS, input_rows, ["path", "exists", "size"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Condition-Level Reliability Source Scout",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only source scout.",
        "- Does not download data, train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- This report converts the current scaling state into an acquisition/gate matrix; it is not a model-promotion claim.",
        "",
        "## Bottom Line",
        "",
        "- Local immediate condition-level reliability artifact candidates: `0`.",
        f"- Closed local/source-derived routes: `{len(CLOSED_LOCAL_SOURCES)}`.",
        f"- External small-table candidate families: `{len(CANDIDATE_SOURCES)}`.",
        "- Current scaling is valuable as mechanism/failure-map evidence, but not yet a systematic deployable scaling law.",
        "- The next non-ACK scaling unlock is external condition-level evidence, not another GPU replay of closed weighting/sampling routes.",
        "",
        "## What Scaling Has Shown So Far",
        "",
        "| axis | current result | why not law-ready | GPU status |",
        "|---|---|---|---|",
    ]
    for row in SCALING_COMPLETION_MATRIX:
        lines.append(
            f"| `{row['axis']}` | {row['current_result']} | {row['why_not_law_ready']} | `{row['gpu_status']}` |"
        )

    lines.extend(
        [
            "",
            "## External Acquisition Matrix",
            "",
            "| priority | candidate | local ready | why new | GPU unlock |",
            "|---|---|---|---|---|",
        ]
    )
    for row in CANDIDATE_SOURCES:
        lines.append(
            f"| `{row['priority']}` | `{row['candidate']}` | `{row['local_ready']}` | {row['why_new']} | {row['gpu_unlock']} |"
        )

    lines.extend(
        [
            "",
            "## Mainline Translation",
            "",
            "| insight | mainline action |",
            "|---|---|",
        ]
    )
    for row in MAINLINE_TRANSLATION:
        lines.append(f"| {row['insight']} | {row['mainline_action']} |")

    lines.extend(
        [
            "",
            "## Strict CPU Gate For Any New Artifact",
            "",
            "- At least `3` datasets, `>=50` overlap rows, and `>=3` varying datasets.",
            "- Real within-dataset artifact variation; no dataset-level constants.",
            "- Bootstrap lower bound `> 0` and dataset minimum pp `>= -0.020`.",
            "- MMD max `<= +0.001`.",
            "- Within-dataset shuffle `p <= 0.01`.",
            "- Leave-dataset/source/background/type/family sensitivity must remain positive enough to avoid one-source carrying.",
            "- Partial-control source/count/QC/read/UMI/batch confounds before any GPU launch.",
            "",
            "## Decision",
            "",
            "- No local scaling artifact currently authorizes a non-ACK GPU experiment.",
            "- Use external small-table acquisition for replicate concordance, dose/time/viability/growth, or background-specific context if scaling is pursued further.",
            "- Keep `xverse_8k_anchor` as the deployable/default model until a strict gate authorizes a new training route.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- Closed local sources: `{OUT_CLOSED}`",
            f"- Candidate source matrix: `{OUT_CANDIDATES}`",
            f"- Scaling law completion matrix: `{OUT_COMPLETION}`",
            f"- Mainline translation: `{OUT_TRANSLATION}`",
            f"- Input manifest: `{OUT_INPUTS}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
