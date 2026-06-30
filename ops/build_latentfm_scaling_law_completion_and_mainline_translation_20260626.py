#!/usr/bin/env python3
"""Build a compact scaling-law completion and mainline-translation report.

This is intentionally CPU/report-only. It summarizes completed evidence and
does not inspect checkpoints, canonical multi, Track C query, or launch jobs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "scaling_law_completion_and_mainline_translation_20260626"
MD = REPORTS / "LATENTFM_SCALING_LAW_COMPLETION_AND_MAINLINE_TRANSLATION_20260626.md"
JSON_OUT = REPORTS / "latentfm_scaling_law_completion_and_mainline_translation_20260626.json"


INPUTS = [
    "reports/LATENTFM_SCALING_LOCKDOWN_AND_MAINLINE_USE_20260626.md",
    "reports/LATENTFM_SCALING_FINAL_PACKAGE_INDEX_20260626.md",
    "reports/LATENTFM_SCALING_LAW_READY_EVIDENCE_TABLE_20260626.md",
    "reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md",
    "reports/LATENTFM_SCALING_NM_FINAL_COMPLETION_BLUEPRINT_20260625.md",
    "reports/LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
    "reports/LATENTFM_TRUECELL_STRATUM_TAIL_PROTECTION_GATE_20260625.md",
    "reports/LATENTFM_TRUECELL_RISKROW_COMPLEMENTARITY_GATE_20260625.md",
    "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_NONNOOP_GATE_20260625.md",
    "reports/LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_BACKGROUND_SPECIFIC_GRN_CONTEXT_SOURCE_AUDIT_20260626.md",
    "reports/LATENTFM_NEXT_ACTION_SLATE_20260626.md",
]


AXES = [
    {
        "axis": "true_cell_per_condition_support",
        "current_conclusion": "strongest positive mechanism; 6k budget128 internal cross/family/MMD +0.059142/+0.062067/-0.001395",
        "claim_scope": "main-text mechanism with no-harm veto",
        "blocking_evidence": "3k curve peaked, 6k budget64 tails unsafe, frozen canonical no-harm failed seeds 42/43/44",
        "mainline_use": "training-set design prior: prefer moderate per-condition support only behind tail/no-harm gates",
        "gpu_now": False,
    },
    {
        "axis": "condition_exposure_count",
        "current_conclusion": "local small positive average but nonmonotonic and tail-unsafe",
        "claim_scope": "failure map or supplement",
        "blocking_evidence": "CI/sign controls fail; tails include TianInhibition/HEXA -0.765316 and Replogle K562/CDK2 -0.330753",
        "mainline_use": "do not assume full exposure helps; use matched controls before any exposure curriculum",
        "gpu_now": False,
    },
    {
        "axis": "background_source_breadth",
        "current_conclusion": "confounded failure axis rather than clean scaling law",
        "claim_scope": "source/background failure map",
        "blocking_evidence": "source-resolved and matched gates leave negative tails and source/background/type confounding",
        "mainline_use": "audit strata for tail localization; no naive cross-background broadening",
        "gpu_now": False,
    },
    {
        "axis": "perturbation_type_breadth",
        "current_conclusion": "negative/tradeoff evidence; allmodality/type policies do not pass hard-harm/tail/shuffle gates",
        "claim_scope": "failure map",
        "blocking_evidence": "family/dose/pathway/allmodality policies fail promotion criteria",
        "mainline_use": "do not expand allmod/type route without new matched CPU gate",
        "gpu_now": False,
    },
    {
        "axis": "target_observability_actionability",
        "current_conclusion": "hint only; residual/actionability positives do not survive tail/permutation controls",
        "claim_scope": "localization covariate or supplement",
        "blocking_evidence": "residual v3 tail/MMD and within-dataset shuffle fail",
        "mainline_use": "failure localization covariate only; no target-weighted training",
        "gpu_now": False,
    },
    {
        "axis": "metadata_qc_reagent_source_maturity_replicate_batch_grn_jiang_ot",
        "current_conclusion": "negative or supplemental evidence only",
        "claim_scope": "guardrail/failure map",
        "blocking_evidence": "QC/support CI/shuffle/tails fail; source-maturity and replicate/batch artifacts fail strict preflight; existing GRN sources are not background-specific; Jiang overlap too small and shuffle p 0.4859; OT lacks pair-quality gate",
        "mainline_use": "do not use generic weighting, hard balancing, source-maturity weighting, replicate/batch weighting, background-GRN conditioning, read-support weighting, Jiang-specialized training, or OT sweeps",
        "gpu_now": False,
    },
    {
        "axis": "chemical_semantics",
        "current_conclusion": "separate ACK-gated protocol route, not current scaling success",
        "claim_scope": "protocol branch",
        "blocking_evidence": "requires exact chemical V2 ACK and fixed-step controls before launch",
        "mainline_use": "launch only with explicit ACK and fresh resource audit",
        "gpu_now": False,
    },
]


GAPS = [
    ("pre_registered_estimands", "CPU/report", "Define denominator, matched contrast, axis, selection boundary, and promotion threshold for every scaling axis."),
    ("matched_controls_per_axis", "CPU/report first; GPU later if training needed", "Count/source/background/type matched controls plus shuffle/permutation/LODO; axes without controls remain exploratory."),
    ("hierarchical_uncertainty", "CPU/report", "Condition bootstrap, dataset bootstrap/random effects, seed stability, and dataset-tail bounds in one table."),
    ("noharm_transfer", "CPU/report for completed runs; GPU only after new route", "Internal gains must pass frozen canonical cross_background_seen_gene, all_test_single, and family_gene; canonical multi remains diagnostic only."),
    ("multi_budget_multi_seed_factorial_curves", "GPU-heavy", "Required before calling a true scaling law: multiple budgets/seeds with held-out law-fit validation and matched controls."),
    ("external_condition_level_artifact", "CPU acquisition/preflight; GPU only after pass", "Replicate concordance, dose/time/viability, or background-specific context artifact distinct from read/UMI/QC/source labels; current source-maturity and replicate/batch proxies already failed."),
    ("law_fit_and_holdout", "CPU if metrics exist; GPU to generate missing factorial data", "Fit law only after predeclared axes and held-out validation; current package should not claim this."),
]


MAINLINE = [
    ("default_model", "keep xverse_8k_anchor", "No scaling-derived checkpoint passes no-harm promotion."),
    ("training_data_prior", "prefer moderate per-condition true-cell support", "Strongest internal mechanism, but only behind non-noop tail/no-harm gates."),
    ("audit_strata", "track source/background/type/reagent/QC/target strata", "Useful for tail localization and failure analysis, not direct loss weights."),
    ("avoid", "no generic QC filtering, hard balancing, source-maturity weighting, replicate/batch weighting, background-GRN conditioning from current OmniPath files, read-support weighting, naive broadening, OT sweeps", "All are closed or supplemental-only under current evidence."),
    ("next_cpu_gate", "external reliability artifact acquisition v2/v3", "Only genuinely new condition-level artifacts can reopen sampler/loss routes."),
    ("next_cpu_gate", "true-cell non-noop tail-protection", "Must map to nonzero canonical footprint and pass tail/MMD/no-harm controls before GPU."),
    ("next_gpu_gate", "chemical V2 fixed-step route", "ACK required before fresh resource audit and launch."),
]


PORTFOLIO = [
    ("now", "CPU/report", "scaling final package polish and reviewer-ready axis/gap/mainline tables", "running via this report", False),
    ("now", "CPU gate", "true-cell non-noop tail-protection meta-gate from existing evidence", "run next/alongside", False),
    ("next", "CPU source/preflight", "new external condition-level reliability artifact", "only if materially distinct from read/UMI/QC/source", False),
    ("conditional", "GPU training", "true-cell or external-artifact sampler/loss/staged-training route", "only after CPU gate, external review, and fresh audit", True),
    ("ACK-only", "GPU training", "chemical V2 real Morgan512 seed43/44 fixed-step controls", "only after exact user ACK", True),
    ("future", "GPU matrix", "factorial scaling-law curves across budget/source/background/type/seed", "requires pre-registration and resource allocation; not current no-ACK route", True),
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict] | list[tuple], headers: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row[h] for h in headers])
            else:
                writer.writerow(row)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    inputs = []
    for rel in INPUTS:
        p = ROOT / rel
        inputs.append({"path": str(p), "exists": p.exists(), "sha256": sha256(p)})

    axis_csv = OUT_DIR / "axis_current_conclusions.csv"
    gaps_csv = OUT_DIR / "scaling_law_gap_matrix.csv"
    mainline_csv = OUT_DIR / "mainline_translation.csv"
    portfolio_csv = OUT_DIR / "experiment_portfolio.csv"
    inputs_csv = OUT_DIR / "input_manifest.csv"

    write_csv(axis_csv, AXES, ["axis", "current_conclusion", "claim_scope", "blocking_evidence", "mainline_use", "gpu_now"])
    write_csv(gaps_csv, GAPS, ["gap", "resource_class", "required_work"])
    write_csv(mainline_csv, MAINLINE, ["decision", "action", "reason"])
    write_csv(portfolio_csv, PORTFOLIO, ["phase", "resource_class", "experiment_or_work", "gate_or_status", "may_need_gpu"])
    write_csv(inputs_csv, inputs, ["path", "exists", "sha256"])

    status = "scaling_law_completion_translation_ready_no_immediate_gpu"
    payload = {
        "timestamp": timestamp,
        "status": status,
        "default_model": "xverse_8k_anchor",
        "gpu_authorized_now": False,
        "axis_count": len(AXES),
        "gap_count": len(GAPS),
        "mainline_decision_count": len(MAINLINE),
        "portfolio_count": len(PORTFOLIO),
        "outputs": {
            "axis_csv": str(axis_csv),
            "gaps_csv": str(gaps_csv),
            "mainline_csv": str(mainline_csv),
            "portfolio_csv": str(portfolio_csv),
            "input_manifest": str(inputs_csv),
        },
        "inputs": inputs,
        "subagent_external_audit": {
            "agent": "Nash",
            "conclusion": "Scaling is report-grade conservative axis audit/mechanism-failure-map, not a systematic deployable scaling law.",
            "priority": "CPU/report completion plus true-cell non-noop and external artifact gates before any non-ACK GPU.",
        },
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    axis_rows = [[a["axis"], a["current_conclusion"], a["claim_scope"], a["blocking_evidence"], a["mainline_use"], f'`{str(a["gpu_now"]).lower()}`'] for a in AXES]
    gap_rows = [[g, r, w] for g, r, w in GAPS]
    mainline_rows = [[d, a, r] for d, a, r in MAINLINE]
    portfolio_rows = [[p, r, e, g, f"`{str(m).lower()}`"] for p, r, e, g, m in PORTFOLIO]

    md = f"""# LatentFM Scaling Law Completion And Mainline Translation

Timestamp: `{timestamp}`

Status: `{status}`

Default/deployable model: `xverse_8k_anchor`

Immediate non-ACK GPU authorized: `False`

## Boundary

- CPU/report-only synthesis of completed scaling reports and an independent Nash subagent audit.
- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.
- Canonical single/family evidence is used only as frozen no-harm veto context from completed reports.

## Bottom Line

- Current scaling evidence is scientifically valuable and report-ready as a leakage-safe multi-axis mechanism/failure-map.
- It is not yet a systematic deployable scaling law and does not promote a checkpoint over `xverse_8k_anchor`.
- The strongest mainline-useful insight is moderate true-cell/per-condition support, but every training change still needs a non-noop tail/no-harm CPU gate before GPU.
- Naive broadening by condition count, background, source, perturbation type, QC/reagent support, or OT is explicitly blocked by existing negative controls and tails.

## Axis Conclusions

{md_table(["axis", "current conclusion", "claim scope", "blocking evidence", "mainline use", "GPU now"], axis_rows)}

## What A Systematic NM-Level Scaling Law Still Needs

{md_table(["gap", "resource class", "required work"], gap_rows)}

## Mainline Translation

{md_table(["decision", "action", "reason"], mainline_rows)}

## Experiment Portfolio To Finish Scaling Properly

{md_table(["phase", "resource class", "experiment/work", "gate or status", "may need GPU"], portfolio_rows)}

## Independent Audit Integration

- Nash subagent conclusion: scaling is currently a conservative axis audit / mechanism-failure-map, not a systematic deployable scaling law.
- Nash agreed the next useful gates are `external_reliability_artifact_acquisition_v2/v3` and `true_cell_nonnoop_tail_protection_cpu_gate`; GPU is conditional on those passing or on exact chemical V2 ACK.

## Decision

- Treat scaling as a Nature Methods-style axis-specific audit with failure maps and no-harm vetoes.
- Use it to guide future train-set design, especially moderate true-cell support, but do not launch generic scaling GPU from current evidence.
- Next immediate work is CPU: close or refine the true-cell non-noop tail-protection gate and continue searching for genuinely new external condition-level reliability artifacts.

## Outputs

- JSON: `{JSON_OUT}`
- Axis conclusions: `{axis_csv}`
- Scaling-law gaps: `{gaps_csv}`
- Mainline translation: `{mainline_csv}`
- Experiment portfolio: `{portfolio_csv}`
- Input manifest: `{inputs_csv}`
"""
    MD.write_text(md)


if __name__ == "__main__":
    main()
