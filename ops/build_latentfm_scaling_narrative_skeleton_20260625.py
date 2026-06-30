#!/usr/bin/env python3
"""Build a manuscript narrative skeleton for the LatentFM scaling package.

Short CPU/report task. Reads completed claim/failure, provenance, and figure QA
artifacts only. Does not read checkpoints, canonical multi, Track C query,
train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
CLAIM_JSON = REPORTS / "latentfm_scaling_nm_claim_failure_package_20260625.json"
PROV_JSON = REPORTS / "latentfm_scaling_nm_provenance_manifest_20260625.json"
FIG_JSON = REPORTS / "latentfm_scaling_figure_readiness_20260625.json"
OUT_DIR = REPORTS / "scaling_narrative_skeleton_20260625"
OUT_MD = REPORTS / "LATENTFM_SCALING_NARRATIVE_SKELETON_20260625.md"
OUT_JSON = REPORTS / "latentfm_scaling_narrative_skeleton_20260625.json"
OUT_SECTIONS = OUT_DIR / "result_sections.tsv"
OUT_CHECKLIST = OUT_DIR / "reviewer_checklist.tsv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def section_rows() -> list[dict[str, str]]:
    return [
        {
            "section": "Scaling audit design",
            "message": "We treat scaling as a set of predeclared data axes rather than a single monotonic more-data claim.",
            "primary_artifacts": "LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md; LATENTFM_SCALING_NM_PROVENANCE_MANIFEST_20260625.md",
            "figures_tables": "axis_claim_boundary.csv; artifact_manifest.tsv; claim_to_artifact_map.tsv",
            "allowed_claim": "leakage-safe cross-dataset scaling-axis audit with explicit provenance and claim boundaries",
            "forbidden_claim": "absolute first perturbation-prediction scaling law; deployable monotonic scaling law",
        },
        {
            "section": "True-cell support is the strongest positive mechanism",
            "message": "The strongest internal signal is moderate per-condition true-cell support, especially 6k budget128, but canonical no-harm veto prevents checkpoint promotion.",
            "primary_artifacts": "LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
            "figures_tables": "Fig_scaling_truecell_budget; Fig_scaling_noharm_veto; truecell_budget_readiness.csv",
            "allowed_claim": "true-cell support is a training-data mechanism and design guidance",
            "forbidden_claim": "true-cell route is deployable or replaces xverse_8k_anchor",
        },
        {
            "section": "Condition exposure is nonmonotonic and tail-unsafe",
            "message": "Moderate exposure has local positive signal, but row-bootstrap uncertainty, sign controls, and large negative tails rule out a monotonic exposure law.",
            "primary_artifacts": "LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md",
            "figures_tables": "Fig_scaling_exposure_nonmonotonic; top_failure_cases.csv; condition_exposure_readiness.csv",
            "allowed_claim": "condition exposure can help locally but needs tail/no-harm controls",
            "forbidden_claim": "more conditions or full exposure is uniformly better",
        },
        {
            "section": "Background/type/source broadening is a failure-map axis",
            "message": "Source-resolved and background/type V2 gates show negative source/background/type tails, so broadening data is not automatically beneficial.",
            "primary_artifacts": "LATENTFM_SCALING_SOURCE_RESOLVED_ESTIMAND_V2_GATE_20260625.md; LATENTFM_SOURCE_VERIFIED_BACKGROUND_TYPE_V2_GATE_20260625.md",
            "figures_tables": "FigS_scaling_failure_map; source_resolved_dataset_tails.csv",
            "allowed_claim": "background/type/source axes identify failure modes and confounding",
            "forbidden_claim": "cross-background/type breadth alone proves scaling",
        },
        {
            "section": "Metadata/QC and Jiang context are supplemental only",
            "message": "Broad QC/support and Jiang guide/cytokine/mixscale signals fail bootstrap, shuffle, overlap, or tail gates and cannot drive training changes.",
            "primary_artifacts": "LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md; LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
            "figures_tables": "mainline_training_guidance.csv; top_failure_cases.csv",
            "allowed_claim": "metadata helps failure analysis and future artifact design",
            "forbidden_claim": "QC filtering, weighted loss, hard balancing, or Jiang-specific GPU training is justified",
        },
        {
            "section": "Literature and benchmark boundary",
            "message": "Existing X-Cell/X-Atlas-Pisces scaling-law language and Nature Methods baseline findings require conservative novelty and benchmark framing.",
            "primary_artifacts": "literature_claim_boundary.csv",
            "figures_tables": "literature_claim_boundary.csv",
            "allowed_claim": "our contribution is a leakage-safe, no-harm-vetoed, multi-axis audit",
            "forbidden_claim": "nobody has proposed perturbation-prediction scaling before",
        },
    ]


def checklist_rows(claim_data: dict[str, Any], prov_data: dict[str, Any], fig_data: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "item": "claim_boundaries_defined",
            "status": "pass" if len(claim_data.get("axis_claim_rows", [])) >= 8 else "fail",
            "evidence": "axis_claim_boundary.csv and LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md",
        },
        {
            "item": "failure_cases_available",
            "status": "pass" if len(claim_data.get("failure_cases", [])) >= 20 else "fail",
            "evidence": "top_failure_cases.csv",
        },
        {
            "item": "literature_boundary_available",
            "status": "pass" if len(claim_data.get("literature_rows", [])) >= 2 else "fail",
            "evidence": "literature_claim_boundary.csv",
        },
        {
            "item": "artifact_hashes_complete",
            "status": "pass" if (prov_data.get("summary") or {}).get("n_missing") == 0 else "fail",
            "evidence": "artifact_manifest.tsv; claim_to_artifact_map.tsv",
        },
        {
            "item": "figures_qc_pass",
            "status": "pass" if (fig_data.get("summary") or {}).get("n_fail") == 0 else "fail",
            "evidence": "LATENTFM_SCALING_FIGURE_READINESS_20260625.md",
        },
        {
            "item": "model_promotion_blocked",
            "status": "pass",
            "evidence": "default remains xverse_8k_anchor; GPU authorized false in package reports",
        },
    ]


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Scaling Narrative Skeleton",
        "",
        "Timestamp: `2026-06-25 23:50 CST`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report narrative scaffold from completed claim/failure, provenance, and figure-QA artifacts.",
        "- Does not read checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "- Default model remains `xverse_8k_anchor`.",
        "",
        "## One-Sentence Positioning",
        "",
        "LatentFM scaling is best presented as a leakage-safe, cross-dataset, axis-specific perturbation-prediction audit with explicit no-harm vetoes and failure maps, not as a deployable monotonic scaling law.",
        "",
        "## Result Section Skeleton",
        "",
        "| section | message | key figures/tables | forbidden claim |",
        "|---|---|---|---|",
    ]
    for row in payload["sections"]:
        lines.append(
            f"| {row['section']} | {row['message']} | `{row['figures_tables']}` | {row['forbidden_claim']} |"
        )
    lines += [
        "",
        "## Reviewer Checklist",
        "",
        "| item | status | evidence |",
        "|---|---|---|",
    ]
    for row in payload["checklist"]:
        lines.append(f"| `{row['item']}` | `{row['status']}` | {row['evidence']} |")
    lines += [
        "",
        "## Manuscript-Safe Wording",
        "",
        "- Use: `scaling-axis audit`, `mechanism/failure-map`, `no-harm veto`, `source/background/type confounding`, `training-data guidance`.",
        "- Avoid: `first scaling law in perturbation prediction`, `more data always helps`, `deployed scaling model`, `checkpoint improvement`, `chemical scaling success`.",
        "",
        "## Outputs",
        "",
        f"- result sections: `{OUT_SECTIONS}`",
        f"- reviewer checklist: `{OUT_CHECKLIST}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "Narrative package is manuscript support only. It does not authorize GPU training. The nearest GPU route remains chemical V2 after exact ACK and fresh resource audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    claim_data = read_json(CLAIM_JSON)
    prov_data = read_json(PROV_JSON)
    fig_data = read_json(FIG_JSON)
    sections = section_rows()
    checklist = checklist_rows(claim_data, prov_data, fig_data)
    payload = {
        "status": "scaling_narrative_skeleton_ready_no_gpu",
        "gpu_authorized": False,
        "sections": sections,
        "checklist": checklist,
        "summary": {
            "n_sections": len(sections),
            "n_checklist": len(checklist),
            "checklist_pass": sum(1 for row in checklist if row["status"] == "pass"),
            "checklist_fail": sum(1 for row in checklist if row["status"] != "pass"),
        },
        "outputs": {"sections": str(OUT_SECTIONS), "checklist": str(OUT_CHECKLIST), "json": str(OUT_JSON), "md": str(OUT_MD)},
    }
    write_tsv(OUT_SECTIONS, sections, ["section", "message", "primary_artifacts", "figures_tables", "allowed_claim", "forbidden_claim"])
    write_tsv(OUT_CHECKLIST, checklist, ["item", "status", "evidence"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
