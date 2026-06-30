#!/usr/bin/env python3
"""Post-audit CPU admission queue for source/artifact and scaling redesign.

This is a short CPU/report-only synthesis. It does not train, infer, inspect
canonical multi for selection, read Track C query, select checkpoints, or use
GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "post_audit_source_scaling_admission_queue_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def main() -> int:
    inputs = {
        "benchmark_control": REPORTS
        / "tracka_benchmark_control_consolidation_20260630"
        / "tracka_benchmark_control_consolidation_20260630.json",
        "new_artifact_source_feasibility": REPORTS / "latentfm_new_artifact_source_feasibility_20260625.json",
        "external_artifact_preflight": REPORTS / "latentfm_external_artifact_preflight_20260626.json",
        "gwt_preflight": REPORTS / "latentfm_gwt_condition_reliability_artifact_preflight_20260627.json",
        "jiang_author_de": REPORTS / "latentfm_jiang_author_de_artifact_gate_20260627.json",
        "replogle_strict_v2": REPORTS / "latentfm_replogle_bulk_artifact_strict_v2_20260627.json",
        "depmap_tailrisk": REPORTS
        / "depmap_tailrisk_veto_gate_20260630"
        / "depmap_tailrisk_veto_gate_20260630.json",
        "external_multisource": REPORTS / "latentfm_external_response_effect_multisource_residual_gate_20260627.json",
        "observable_gene_budget": REPORTS
        / "observable_gene_budget_scaling_law_gate_20260630"
        / "latentfm_observable_gene_budget_scaling_law_gate_20260630.json",
        "downstream_scaling_x": REPORTS
        / "downstream_information_scaling_x_gate_20260628"
        / "downstream_information_scaling_x_gate_20260628.json",
        "scaling_v2_matched": REPORTS
        / "scaling_v2_matched_information_gate_20260628"
        / "latentfm_scaling_v2_matched_information_gate_20260628.json",
    }
    payloads = {key: read_json(path) for key, path in inputs.items()}

    external_fail_reasons = [
        "existing_external_source_gates_fail_or_diagnostic_only",
        "GWT preflight has broad overlap but dataset-min and MMD fail",
        "Jiang author-DE fails cross-seed/shuffle/LODO gate",
        "Replogle strict V2 is diagnostic/test-metric-selected and MMD/QC-confounded",
        "DepMap has tail-risk signal but row count/MMD no-harm are unstable",
        "multisource residual gate lacks independent train/internal source families and is MMD-confounded",
    ]

    observable = payloads["observable_gene_budget"]
    observable_metrics = observable.get("decision_metrics", {})
    scaling_v2 = payloads["scaling_v2_matched"]
    scaling_status = scaling_v2.get("status", "unknown")
    max_pairs = None
    try:
        summary_path = REPORTS / "scaling_v2_matched_information_gate_20260628" / "scaling_v2_matched_information_summary.csv"
        summary = pd.read_csv(summary_path)
        max_pairs = int(pd.to_numeric(summary["n_pairs"], errors="coerce").max())
    except Exception:
        max_pairs = None

    candidates: list[dict[str, Any]] = [
        {
            "candidate": "matched_external_artifact_source_gate",
            "status": "source_scout_needed_cpu_only",
            "gpu_authorized_now": False,
            "cpu_authorized_now": True,
            "hypothesis": (
                "A genuinely condition/background matched external artifact may add exogenous "
                "train-only information that can beat both anchor and source/control."
            ),
            "current_evidence": "; ".join(external_fail_reasons),
            "next_runnable_action": (
                "source-scout or materialize only a new verified small table with condition/background keys; "
                "then run strict CPU admission against max(anchor, source/control)"
            ),
            "promotion_gate": (
                ">=100 primary rows, >=5 datasets or >=3 independent source/background families, "
                "within-source shuffle collapse, LODO same-sign, pp CI-low >0 versus max(anchor, source/control), "
                "dataset-min >= -0.02, MMD harm <= +0.001"
            ),
            "fail_close_rule": (
                "close if source is static gene-only, QC/read-depth/cell-count, overlap <50, "
                "source families <3, MMD-confounded, or does not beat source/control"
            ),
            "inputs": "LATENTFM_EXTERNAL_MATCHED_ARTIFACT_NEXT_SOURCE_SLATE_20260627.md; external source gate JSONs",
        },
        {
            "candidate": "nonstatic_observable_information_redesign",
            "status": "descriptor_positive_intervention_blocked_cpu_design_only",
            "gpu_authorized_now": False,
            "cpu_authorized_now": True,
            "hypothesis": (
                "Observable response budget is real as a descriptor, but only a nonstatic residualized "
                "information axis beyond abundance/detection/mean controls can justify training changes."
            ),
            "current_evidence": (
                f"observable descriptor pass={fmt_bool(observable.get('descriptor_pass'))}; "
                f"HVG-specific gate={observable.get('hvg_specific_intervention_gate', 'unknown')}; "
                f"top1000 HVG share={observable_metrics.get('all_top1000_hvg_share', 'NA')}; "
                f"max matched split pairs={max_pairs}; scaling_v2_status={scaling_status}"
            ),
            "next_runnable_action": (
                "build a new CPU design matrix only if it introduces nonstatic residualized axes "
                "not collinear with abundance/detection/source; otherwise keep as manuscript descriptor"
            ),
            "promotion_gate": (
                "axis survives abundance/mean/detection controls, source/dataset LODO, permutation/placebo, "
                "and yields >=300 matched pairs across >=15 datasets or a separately justified RawFM gene-budget route"
            ),
            "fail_close_rule": (
                "close for GPU if HVG advantage remains explained by abundance/detection, matched pairs collapse, "
                "LODO flips sign, or MMD/tail risk lacks precontrol"
            ),
            "inputs": "observable_gene_budget_scaling_law_gate_20260630; downstream_information_scaling_x; scaling_v2_matched_information",
        },
        {
            "candidate": "zscape_dynamic_translation",
            "status": "descriptor_only_fail_closed_for_model_route",
            "gpu_authorized_now": False,
            "cpu_authorized_now": False,
            "hypothesis": (
                "ZSCAPE dynamic response may inspire biological scaling variables but current rows do not pass specificity "
                "or train-set translation gates."
            ),
            "current_evidence": (
                "prospective strict controls partial; focused prospective specificity 0/3; "
                "train-set translation/neighborhood balance and null gates fail"
            ),
            "next_runnable_action": (
                "wait for independent zebrafish audit; no LatentFM CPU/GPU admission from current ZSCAPE rows"
            ),
            "promotion_gate": (
                "future route would need heldout/wrong-time/wrong-lineage/wrong-target specificity and a train-only "
                "translation/no-harm gate before model use"
            ),
            "fail_close_rule": "current branch already fail-closed for model use",
            "inputs": "ZSCAPE prospective strict/specificity decisions and true-timecourse readiness reports",
        },
    ]

    gpu_candidates = [row for row in candidates if row["gpu_authorized_now"]]
    cpu_candidates = [row for row in candidates if row["cpu_authorized_now"]]
    status = "post_audit_source_scaling_queue_cpu_only_no_gpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "post_audit_source_scaling_admission_queue_rows_20260630.csv"
    json_path = OUT_DIR / "post_audit_source_scaling_admission_queue_20260630.json"
    md_path = OUT_DIR / "LATENTFM_POST_AUDIT_SOURCE_SCALING_ADMISSION_QUEUE_20260630.md"

    pd.DataFrame(candidates).to_csv(rows_path, index=False)
    out = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "gpu_candidate_count": len(gpu_candidates),
        "cpu_candidate_count": len(cpu_candidates),
        "default_model": "xverse_8k_anchor",
        "benchmark_rule": "future candidates must beat max(anchor, source/control) with MMD/tail no-harm",
        "inputs": {key: str(path) for key, path in inputs.items()},
        "outputs": {"rows": str(rows_path), "markdown": str(md_path)},
        "boundary": "cpu_report_only_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
        "candidates": candidates,
    }
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Post-Audit Source/Scaling Admission Queue 20260630",
        "",
        f"Created: {out['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis after Godel and Lorentz external audits.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- Future candidates must beat `max(anchor, source/control)`, not anchor alone.",
        "",
        "## Queue",
        "",
        "| candidate | status | GPU now | CPU now | next action |",
        "|---|---|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['candidate']}` | `{row['status']}` | "
            f"{fmt_bool(row['gpu_authorized_now'])} | {fmt_bool(row['cpu_authorized_now'])} | "
            f"{row['next_runnable_action']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- There are `0` GPU candidates from the post-audit slate.",
        "- Carry forward two CPU-only directions: matched external source scouting and nonstatic observable-information redesign.",
        "- ZSCAPE remains descriptor/failure-analysis only until an independent audit identifies a new specificity-safe route.",
        "",
        "## Fail-Close Summary",
        "",
        "- Do not relaunch current count/HVG/ZSCAPE/RawFM/tail/support/exact/analog artifacts as GPU smokes.",
        "- Close any external source that is static gene-only, QC/read-depth/cell-count, too narrow, or MMD-confounded.",
        "- Close observable/gene-budget intervention routes unless a nonstatic residualized axis survives controls and matched-pair feasibility.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{json_path}`",
        f"- rows: `{rows_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
