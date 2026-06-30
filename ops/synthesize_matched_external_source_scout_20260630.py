#!/usr/bin/env python3
"""Synthesize the post-audit matched external source scout.

This is a CPU/report-only gate. It does not train, infer, inspect held-out
Track C query, or use GPU. The goal is to make the post-audit source queue
explicit enough that future launches cannot quietly revive consumed routes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "matched_external_source_scout_20260630"
REPORT = OUT_DIR / "LATENTFM_MATCHED_EXTERNAL_SOURCE_SCOUT_20260630.md"
JSON_OUT = OUT_DIR / "matched_external_source_scout_20260630.json"
CLOSED_CSV = OUT_DIR / "matched_external_source_closed_routes_20260630.csv"
OPEN_CSV = OUT_DIR / "matched_external_source_open_leads_20260630.csv"
LOCAL_CSV = OUT_DIR / "matched_external_source_local_admission_20260630.csv"


@dataclass(frozen=True)
class ClosedRoute:
    family: str
    latest_status: str
    direct_fail_reason: str
    evidence: str
    residual_use: str


@dataclass(frozen=True)
class OpenLead:
    lead: str
    why_not_duplicate: str
    minimum_external_table: str
    immediate_action: str
    cpu_gate: str
    fail_close: str


@dataclass(frozen=True)
class LocalCandidate:
    candidate: str
    local_artifact_ready: bool
    short_cpu_gate_now: bool
    reason: str
    next_requirement: str


def read_text(path: str) -> str:
    full = ROOT / path
    if not full.exists():
        return ""
    return full.read_text(encoding="utf-8", errors="replace")


def status_from_report(path: str, default: str = "missing") -> str:
    text = read_text(path)
    for line in text.splitlines():
        if line.lower().startswith("status:"):
            return line.split(":", 1)[1].strip().strip("`")
    return default


def require_inputs() -> dict[str, str]:
    inputs = {
        "post_audit_queue": "reports/post_audit_source_scaling_admission_queue_20260630/LATENTFM_POST_AUDIT_SOURCE_SCALING_ADMISSION_QUEUE_20260630.md",
        "benchmark_control": "reports/tracka_benchmark_control_consolidation_20260630/LATENTFM_TRACKA_BENCHMARK_CONTROL_CONSOLIDATION_20260630.md",
        "source_slate": "reports/LATENTFM_EXTERNAL_MATCHED_ARTIFACT_NEXT_SOURCE_SLATE_20260627.md",
        "multisource": "reports/LATENTFM_EXTERNAL_RESPONSE_EFFECT_MULTISOURCE_RESIDUAL_GATE_20260627.md",
    }
    missing = [name for name, rel in inputs.items() if not (ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required evidence files: {missing}")
    return inputs


def build_closed_routes() -> list[ClosedRoute]:
    return [
        ClosedRoute(
            family="Jiang author DE / background context",
            latest_status=status_from_report(
                "reports/LATENTFM_JIANG_BACKGROUND_RESOLVED_GATE_20260627.md"
            ),
            direct_fail_reason=(
                "condition-level aggregate and background-resolved artifacts fail "
                "shuffle/LODO/LOBO/MMD gates; frozen eval outcomes lack a real "
                "background key, so direct background joins would pseudoreplicate"
            ),
            evidence=(
                "reports/LATENTFM_JIANG_AUTHOR_DE_ARTIFACT_GATE_20260627.md; "
                "reports/LATENTFM_JIANG_BACKGROUND_RESOLVED_GATE_20260627.md; "
                "reports/LATENTFM_JIANG_BACKGROUND_SCHEMA_SPECIFIC_GATE_20260627.md"
            ),
            residual_use="failure anatomy and possible future background-resolved evaluator only",
        ),
        ClosedRoute(
            family="DepMap / Project SCORE dependency",
            latest_status=status_from_report(
                "reports/LATENTFM_DEPMAP_MMD_MATCHED_DEPENDENCY_NOHARM_GATE_20260627.md"
            ),
            direct_fail_reason=(
                "initial dependency signal exists, but MMD-matched no-harm fails; "
                "tail-risk veto rows remain below seed threshold and MMD gap is unstable"
            ),
            evidence=(
                "reports/LATENTFM_DEPMAP_24Q4_DEPENDENCY_GATE_20260627.md; "
                "reports/LATENTFM_DEPMAP_MMD_MATCHED_DEPENDENCY_NOHARM_GATE_20260627.md; "
                "reports/depmap_tailrisk_veto_gate_20260630/LATENTFM_DEPMAP_TAILRISK_VETO_GATE_20260630.md"
            ),
            residual_use="tail-risk/failure-analysis covariate, not a GPU admission source",
        ),
        ClosedRoute(
            family="Replogle / Norman replicate or bulk artifacts",
            latest_status=status_from_report(
                "reports/LATENTFM_NORMAN_REPLOGLE_REPLICATE_CONCORDANCE_GATE_20260627.md"
            ),
            direct_fail_reason=(
                "no true replicate-concordance source found; bulk difficulty/QC "
                "signals are test-selected or strongly MMD/QC-confounded"
            ),
            evidence=(
                "reports/LATENTFM_NORMAN_REPLOGLE_REPLICATE_CONCORDANCE_GATE_20260627.md; "
                "reports/LATENTFM_REPLOGLE_BULK_ARTIFACT_STRICT_V2_20260627.md; "
                "reports/LATENTFM_REPLOGLE_BULK_RESIDUALIZED_SOURCE_LODO_V2_GATE_20260627.md"
            ),
            residual_use="diagnostic mechanism evidence only",
        ),
        ClosedRoute(
            family="GWT condition reliability",
            latest_status=status_from_report(
                "reports/LATENTFM_GWT_RESIDUALIZED_MMDMATCHED_GATE_20260628.md"
            ),
            direct_fail_reason=(
                "best knockdown-fraction artifact has a positive pp high-low but "
                "fails residual-rho and dataset-tail requirements; other artifacts "
                "fail signal, shuffle, MMD, or tail controls"
            ),
            evidence=(
                "reports/LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACT_PREFLIGHT_20260627.md; "
                "reports/LATENTFM_GWT_RESIDUALIZED_MMDMATCHED_GATE_20260628.md"
            ),
            residual_use="external reliability negative/control evidence",
        ),
        ClosedRoute(
            family="LINCS/L1000 small metadata",
            latest_status=status_from_report(
                "reports/LATENTFM_LINCS_GSE92742_OUTCOME_PANEL_DECISION_20260627.md"
            ),
            direct_fail_reason=(
                "small metadata join works, but strict train-outcome signal is "
                "underpowered and outcome panel has zero review-only passes"
            ),
            evidence=(
                "reports/LATENTFM_LINCS_L1000_GSE92742_CONDITION_JOIN_GATE_20260627.md; "
                "reports/LATENTFM_LINCS_GSE92742_SIGNAL_CONTROL_GATE_20260627.md; "
                "reports/LATENTFM_LINCS_GSE92742_OUTCOME_PANEL_DECISION_20260627.md"
            ),
            residual_use="source/chemical overlap context and mechanism diagnostics",
        ),
        ClosedRoute(
            family="JUMP-CP small metadata / morphology",
            latest_status=status_from_report(
                "reports/LATENTFM_JUMP_CP_TRAINONLY_JOIN_CONTROLS_GATE_20260627.md"
            ),
            direct_fail_reason=(
                "train-only gene overlap exists, but available small fields are "
                "source/plate/batch coverage controls; activity/reproducibility/profile "
                "norm and background/dose/time are absent"
            ),
            evidence=(
                "reports/LATENTFM_JUMP_CP_MORPHOLOGY_SOURCE_GATE_20260627.md; "
                "reports/LATENTFM_JUMP_CP_TRAINONLY_JOIN_CONTROLS_GATE_20260627.md"
            ),
            residual_use="control-source map and future acquisition lead only",
        ),
        ClosedRoute(
            family="SciPlex dose/background/time",
            latest_status=status_from_report(
                "reports/LATENTFM_SCIPLEX_DOSE_BACKGROUND_INTERACTION_GATE_20260628.md"
            ),
            direct_fail_reason=(
                "dose-specific outcome gate fails pp CI, dataset tail, and MMD; "
                "background interaction has no passing backgrounds; local SciPlex "
                "files are single-timepoint and condition metadata omits dose/time"
            ),
            evidence=(
                "reports/LATENTFM_SCIPLEX_DOSE_SPECIFIC_OUTCOME_GATE_20260627.md; "
                "reports/LATENTFM_SCIPLEX_DOSE_BACKGROUND_INTERACTION_GATE_20260628.md; "
                "reports/sciplex_explicit_time_vector_preflight_20260629/LATENTFM_SCIPLEX_EXPLICIT_TIME_VECTOR_PREFLIGHT_20260629.md"
            ),
            residual_use="dose/background control and biology descriptor only",
        ),
        ClosedRoute(
            family="Frangieh/Dixit/local h5ad obs and scPerturb maturity",
            latest_status="external_obs_or_catalog_artifacts_consumed_no_gpu",
            direct_fail_reason=(
                "processed obs/catalog surfaces expose label/QC/read/guide/source or "
                "dataset-level maturity fields, not independent condition-level "
                "response/reproducibility artifacts"
            ),
            evidence=(
                "reports/LATENTFM_EXTERNAL_SOURCE_H5AD_OBS_ROUTES_20260626.md; "
                "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACT_PREFLIGHT_20260626.md"
            ),
            residual_use="source inventory and negative boundary",
        ),
        ClosedRoute(
            family="Nadig GSE264667 short small-table scout",
            latest_status=status_from_report(
                "reports/LATENTFM_NADIG_GSE264667_SOURCE_SCOUT_20260627.md"
            ),
            direct_fail_reason=(
                "series supplement exposes large expression/raw artifacts but no "
                "small processed condition/effect/reproducibility table"
            ),
            evidence="reports/LATENTFM_NADIG_GSE264667_SOURCE_SCOUT_20260627.md",
            residual_use="future large preprocessing branch only if explicitly justified",
        ),
    ]


def build_local_candidates() -> list[LocalCandidate]:
    return [
        LocalCandidate(
            candidate="matched_external_artifact_source_gate",
            local_artifact_ready=False,
            short_cpu_gate_now=False,
            reason=(
                "all currently materialized source families are closed, diagnostic-only, "
                "underpowered, MMD/QC-confounded, missing key fields, or ACK-gated"
            ),
            next_requirement=(
                "new verified small table keyed by dataset/background/condition with "
                "condition-level effect, response quality, reproducibility, viability, "
                "or non-QC biological activity"
            ),
        ),
        LocalCandidate(
            candidate="nonstatic_observable_information_redesign",
            local_artifact_ready=True,
            short_cpu_gate_now=True,
            reason=(
                "not GPU-eligible, but can be designed CPU-only from current scaling "
                "artifacts if it explicitly residualizes abundance/detection/source "
                "and tests nonstatic condition-specific information"
            ),
            next_requirement=(
                "build a balanced residualized design matrix; require matched-pair "
                "feasibility, source/dataset controls, MMD no-harm, and dual-baseline "
                "dominance before any GPU"
            ),
        ),
    ]


def build_open_leads() -> list[OpenLead]:
    return [
        OpenLead(
            lead="author replicate-concordance or pseudobulk-quality tables across Norman/Replogle/Jiang-like screens",
            why_not_duplicate=(
                "not a Replogle bulk-difficulty/QC repeat: the required field is a "
                "condition-level reproducibility or pseudobulk agreement score, "
                "selected before LatentFM evaluation"
            ),
            minimum_external_table=(
                "dataset, condition, target_gene, cell_background, artifact_value, "
                "metric_name, n_replicates, source_url"
            ),
            immediate_action=(
                "source scout only author supplements/figshare/GEO small tables; "
                "reject read/UMI/cell-count/guide-support-only columns"
            ),
            cpu_gate=(
                ">=50 overlap rows, >=3 datasets/backgrounds, within-dataset variation, "
                "shuffle collapse, LODO/source-block pass, MMD harm <= +0.001"
            ),
            fail_close=(
                "close if one dataset/source carries the effect, source is static "
                "gene-level, or signal vanishes against source/control baseline"
            ),
        ),
        OpenLead(
            lead="background-specific response-program or context tables for Jiang-like cytokine screens",
            why_not_duplicate=(
                "not current Jiang background artifact repeat: it requires a real "
                "background-resolved outcome/evaluator key or independent response "
                "program table, avoiding pseudoreplicated condition-level outcomes"
            ),
            minimum_external_table=(
                "stimulus_dataset, target_gene, cell_background, response_program_score, "
                "quality_or_reproducibility_metric, source_url"
            ),
            immediate_action=(
                "manual schema review of author DE/program archives; only proceed if "
                "background rows can be joined to background-resolved LatentFM outcomes"
            ),
            cpu_gate=(
                "within-background and leave-background controls, leave-stimulus/source "
                "controls, dataset-tail and MMD no-harm, dual-baseline dominance"
            ),
            fail_close=(
                "close if frozen outcome remains condition-only or if stimulus/source "
                "confounding explains the artifact"
            ),
        ),
        OpenLead(
            lead="independent viability/growth/fitness or response-burden side assays",
            why_not_duplicate=(
                "not DepMap/Frangieh static viability repeat: it needs condition and "
                "background specificity plus no-harm under matched rows"
            ),
            minimum_external_table=(
                "dataset, condition, target_or_drug, cell_background, assay, "
                "artifact_value, time_or_dose_if_applicable, source_url"
            ),
            immediate_action=(
                "search only for small processed author side-assay tables; keep large "
                "expression-only downloads out until a schema exists"
            ),
            cpu_gate=(
                "matched high-low or residual association against train/internal rows; "
                "source/count/QC controls; MMD and source/control dual-baseline veto"
            ),
            fail_close=(
                "close if it is single-source, dataset-level, or simply selects high-MMD "
                "hard rows without a no-harm route"
            ),
        ),
        OpenLead(
            lead="JUMP-CP/LINCS-like activity/reproducibility fields only if true activity metrics are acquired",
            why_not_duplicate=(
                "not current small-metadata overlap repeat: current files lack activity "
                "or profile-reproducibility fields"
            ),
            minimum_external_table=(
                "condition key, perturbation id, cell line, perturbation type, "
                "activity/reproducibility/profile-norm metric, source/batch block"
            ),
            immediate_action=(
                "acquire a small normalized profile-reproducibility/activity manifest, "
                "not full Level5/profile matrices"
            ),
            cpu_gate=(
                "condition-level join, source/plate/batch controls, perturbation-type "
                "controls, exact-background sensitivity, no-harm and dual-baseline gate"
            ),
            fail_close=(
                "close if only source/plate/batch/count fields are available or chemical "
                "rows remain ACK-gated"
            ),
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    require_inputs()
    if REPORT.exists() or JSON_OUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing scout outputs under {OUT_DIR}"
        )

    closed = build_closed_routes()
    local = build_local_candidates()
    open_leads = build_open_leads()

    closed_rows = [asdict(row) for row in closed]
    local_rows = [asdict(row) for row in local]
    open_rows = [asdict(row) for row in open_leads]

    status = "matched_external_source_scout_no_local_gpu_gate"
    gpu_authorized = False
    local_short_cpu_ready = [
        row.candidate for row in local if row.short_cpu_gate_now and not row.local_artifact_ready
    ]

    payload = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": status,
        "gpu_authorized": gpu_authorized,
        "closed_route_count": len(closed_rows),
        "local_admission_candidates": local_rows,
        "open_leads": open_rows,
        "closed_routes": closed_rows,
        "decision": {
            "local_matched_external_source_cpu_gate_ready": False,
            "next_cpu_direction": "nonstatic_observable_information_redesign",
            "next_source_action": "external_small_table_acquisition_or_schema_scout",
            "default_model": "xverse_8k_anchor",
        },
    }

    write_csv(CLOSED_CSV, closed_rows)
    write_csv(LOCAL_CSV, local_rows)
    write_csv(OPEN_CSV, open_rows)
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# LatentFM Matched External Source Scout 20260630

Created: `{payload["created"]}`

Status: `{status}`

GPU authorized: `{gpu_authorized}`

## Boundary

- CPU/report-only post-audit source scout.
- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.
- Future model candidates must beat `max(anchor, source/control)`, not anchor alone.

## Bottom Line

- Current local artifacts provide `0` matched external source CPU gates that can
  directly authorize GPU.
- The only immediately runnable local next step is CPU-only design for
  `nonstatic_observable_information_redesign`; this still cannot authorize GPU
  without a later strict admission gate.
- A new matched external source route requires a genuinely new small table with
  condition/background keys and an artifact that is not read depth, UMI, source,
  plate, cell count, static gene prior, or test-metric-selected outcome.

## Local Admission Queue

{markdown_table(local_rows, ["candidate", "local_artifact_ready", "short_cpu_gate_now", "reason", "next_requirement"])}

## Consumed / Closed Source Families

{markdown_table(closed_rows, ["family", "latest_status", "direct_fail_reason", "evidence", "residual_use"])}

## Open Acquisition Leads

{markdown_table(open_rows, ["lead", "why_not_duplicate", "minimum_external_table", "immediate_action", "cpu_gate", "fail_close"])}

## Decision

- Keep default/deployable model as `xverse_8k_anchor`.
- Do not launch a LatentFM GPU run from current matched-source evidence.
- If pursuing source routes, first acquire or verify one of the open small-table
  leads and rerun a strict CPU gate.
- For immediate local progress, move to the CPU-only nonstatic observable
  information redesign matrix; GPU remains blocked until it passes a
  dual-baseline/no-harm admission gate.

## Outputs

- JSON: `{JSON_OUT}`
- closed routes: `{CLOSED_CSV}`
- local admission queue: `{LOCAL_CSV}`
- open leads: `{OPEN_CSV}`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
