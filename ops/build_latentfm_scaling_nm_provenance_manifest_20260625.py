#!/usr/bin/env python3
"""Build a provenance manifest for the LatentFM scaling NM package.

Short CPU/report task. Hashes completed reports, tables, JSON payloads, and
figures. Does not read checkpoints, canonical multi, Track C query, train,
infer, or use GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "scaling_nm_provenance_manifest_20260625"
OUT_MD = REPORTS / "LATENTFM_SCALING_NM_PROVENANCE_MANIFEST_20260625.md"
OUT_JSON = REPORTS / "latentfm_scaling_nm_provenance_manifest_20260625.json"
OUT_TSV = OUT_DIR / "artifact_manifest.tsv"
OUT_MAP = OUT_DIR / "claim_to_artifact_map.tsv"

ARTIFACTS = [
    "reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md",
    "reports/LATENTFM_SCALING_NM_FINAL_COMPLETION_BLUEPRINT_20260625.md",
    "reports/LATENTFM_SCALING_NM_COMPLETION_PORTFOLIO_20260625.md",
    "reports/LATENTFM_SCALING_COMPLETION_READINESS_20260625.md",
    "reports/LATENTFM_SCALING_RESULT_SECTION_DRAFT_20260625.md",
    "reports/LATENTFM_SCALING_FIGURES_20260625.md",
    "reports/LATENTFM_SCALING_FIGURE_DATA_PACKAGE_20260625.md",
    "reports/LATENTFM_SCALING_MANUSCRIPT_ASSETS_20260625.md",
    "reports/LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
    "reports/LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md",
    "reports/LATENTFM_SCALING_SOURCE_RESOLVED_ESTIMAND_V2_GATE_20260625.md",
    "reports/LATENTFM_SOURCE_VERIFIED_BACKGROUND_TYPE_V2_GATE_20260625.md",
    "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
    "reports/LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
    "reports/LATENTFM_SCHRODINGER_EXTERNAL_AUDIT_INTEGRATION_20260625.md",
    "reports/latentfm_scaling_nm_claim_failure_package_20260625.json",
    "reports/scaling_nm_claim_failure_package_20260625/axis_claim_boundary.csv",
    "reports/scaling_nm_claim_failure_package_20260625/top_failure_cases.csv",
    "reports/scaling_nm_claim_failure_package_20260625/literature_claim_boundary.csv",
    "reports/scaling_nm_claim_failure_package_20260625/mainline_training_guidance.csv",
    "reports/scaling_completion_readiness_20260625/axis_completion_matrix.csv",
    "reports/scaling_completion_readiness_20260625/condition_exposure_readiness.csv",
    "reports/scaling_completion_readiness_20260625/source_resolved_dataset_tails.csv",
    "reports/scaling_completion_readiness_20260625/truecell_budget_readiness.csv",
    "reports/scaling_figure_data_20260625/canonical_noharm_veto.csv",
    "reports/scaling_figure_data_20260625/condition_exposure_curve.csv",
    "reports/scaling_figure_data_20260625/failure_map_axis_summary.csv",
    "reports/scaling_figure_data_20260625/s0_provenance_summary.csv",
    "reports/scaling_figure_data_20260625/truecell_budget_curve.csv",
    "reports/scaling_figures_20260625/Fig_scaling_truecell_budget.png",
    "reports/scaling_figures_20260625/Fig_scaling_truecell_budget.svg",
    "reports/scaling_figures_20260625/Fig_scaling_exposure_nonmonotonic.png",
    "reports/scaling_figures_20260625/Fig_scaling_exposure_nonmonotonic.svg",
    "reports/scaling_figures_20260625/Fig_scaling_noharm_veto.png",
    "reports/scaling_figures_20260625/Fig_scaling_noharm_veto.svg",
    "reports/scaling_figures_20260625/FigS_scaling_failure_map.png",
    "reports/scaling_figures_20260625/FigS_scaling_failure_map.svg",
    "reports/scaling_figures_20260625/FigS_scaling_S0_provenance.png",
    "reports/scaling_figures_20260625/FigS_scaling_S0_provenance.svg",
]

CLAIMS = [
    {
        "claim": "true_cell_support_is_strongest_mechanism_not_promotion",
        "allowed_scope": "main_text_mechanism_only",
        "artifacts": [
            "reports/LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
            "reports/scaling_completion_readiness_20260625/truecell_budget_readiness.csv",
            "reports/scaling_figures_20260625/Fig_scaling_truecell_budget.png",
            "reports/scaling_figures_20260625/Fig_scaling_noharm_veto.png",
        ],
    },
    {
        "claim": "condition_exposure_is_nonmonotonic_and_tail_unsafe",
        "allowed_scope": "main_text_mechanism_failure_map",
        "artifacts": [
            "reports/LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md",
            "reports/scaling_completion_readiness_20260625/condition_exposure_readiness.csv",
            "reports/scaling_figures_20260625/Fig_scaling_exposure_nonmonotonic.png",
            "reports/scaling_nm_claim_failure_package_20260625/top_failure_cases.csv",
        ],
    },
    {
        "claim": "background_type_source_are_confounded_failure_axes",
        "allowed_scope": "supplement_or_failure_map",
        "artifacts": [
            "reports/LATENTFM_SCALING_SOURCE_RESOLVED_ESTIMAND_V2_GATE_20260625.md",
            "reports/LATENTFM_SOURCE_VERIFIED_BACKGROUND_TYPE_V2_GATE_20260625.md",
            "reports/scaling_completion_readiness_20260625/source_resolved_dataset_tails.csv",
            "reports/scaling_figures_20260625/FigS_scaling_failure_map.png",
        ],
    },
    {
        "claim": "metadata_qc_and_jiang_context_do_not_authorize_training",
        "allowed_scope": "supplement_or_failure_map",
        "artifacts": [
            "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
            "reports/LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
            "reports/scaling_nm_claim_failure_package_20260625/mainline_training_guidance.csv",
        ],
    },
    {
        "claim": "novelty_boundary_is_scaling_axis_audit_not_absolute_first_law",
        "allowed_scope": "claim_boundary",
        "artifacts": [
            "reports/scaling_nm_claim_failure_package_20260625/literature_claim_boundary.csv",
            "reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md",
        ],
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_row(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    exists = path.is_file()
    row = {
        "path": rel,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": sha256(path) if exists else "",
        "kind": path.suffix.lstrip(".").lower() if exists else "",
    }
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Scaling NM Provenance Manifest",
        "",
        "Timestamp: `2026-06-25 23:34 CST`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact provenance manifest.",
        "- Hashes completed reports, tables, JSON payloads, and figures.",
        "- Does not read checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- artifacts listed: `{payload['summary']['n_artifacts']}`",
        f"- artifacts present: `{payload['summary']['n_present']}`",
        f"- missing artifacts: `{payload['summary']['n_missing']}`",
        f"- claim mappings: `{payload['summary']['n_claims']}`",
        "",
        "## Claim To Artifact Map",
        "",
        "| claim | allowed scope | n artifacts | all present |",
        "|---|---|---:|---|",
    ]
    for claim in payload["claim_rows"]:
        lines.append(
            f"| `{claim['claim']}` | `{claim['allowed_scope']}` | {claim['n_artifacts']} | `{claim['all_present']}` |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- artifact manifest: `{OUT_TSV}`",
        f"- claim map: `{OUT_MAP}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "This manifest is provenance support only. It does not authorize GPU training. The nearest GPU route remains chemical V2 after exact ACK and fresh resource audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    artifact_rows = [artifact_row(rel) for rel in ARTIFACTS]
    existing = {row["path"]: row for row in artifact_rows if row["exists"]}
    claim_rows = []
    map_rows = []
    for claim in CLAIMS:
        all_present = all(rel in existing for rel in claim["artifacts"])
        claim_rows.append(
            {
                "claim": claim["claim"],
                "allowed_scope": claim["allowed_scope"],
                "n_artifacts": len(claim["artifacts"]),
                "all_present": all_present,
            }
        )
        for rel in claim["artifacts"]:
            row = existing.get(rel) or artifact_row(rel)
            map_rows.append(
                {
                    "claim": claim["claim"],
                    "allowed_scope": claim["allowed_scope"],
                    "artifact": rel,
                    "exists": row["exists"],
                    "sha256": row["sha256"],
                }
            )
    payload = {
        "status": "scaling_nm_provenance_manifest_ready_no_gpu",
        "gpu_authorized": False,
        "summary": {
            "n_artifacts": len(artifact_rows),
            "n_present": sum(1 for r in artifact_rows if r["exists"]),
            "n_missing": sum(1 for r in artifact_rows if not r["exists"]),
            "n_claims": len(CLAIMS),
        },
        "artifacts": artifact_rows,
        "claim_rows": claim_rows,
        "claim_map": map_rows,
        "outputs": {"artifact_manifest": str(OUT_TSV), "claim_map": str(OUT_MAP), "json": str(OUT_JSON), "md": str(OUT_MD)},
    }
    write_tsv(OUT_TSV, artifact_rows, ["path", "exists", "size_bytes", "sha256", "kind"])
    write_tsv(OUT_MAP, map_rows, ["claim", "allowed_scope", "artifact", "exists", "sha256"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
