#!/usr/bin/env python3
"""Build reproduction command/script provenance for LatentFM scaling package.

Short CPU/report task. Records commands, script hashes, and expected outputs for
the completed scaling/NM report package. It does not run experiments, read
checkpoints, read canonical multi, read Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "scaling_reproduction_manifest_20260625"
OUT_MD = REPORTS / "LATENTFM_SCALING_REPRODUCTION_MANIFEST_20260625.md"
OUT_JSON = REPORTS / "latentfm_scaling_reproduction_manifest_20260625.json"
OUT_COMMANDS = OUT_DIR / "reproduction_commands.tsv"
OUT_SCRIPTS = OUT_DIR / "script_hashes.tsv"

PY = "conda run -n scdfm python"

STEPS = [
    {
        "step": "s0_provenance_freeze",
        "command": f"{PY} ops/build_latentfm_scaling_s0_provenance_freeze_20260625.py",
        "script": "ops/build_latentfm_scaling_s0_provenance_freeze_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_S0_PROVENANCE_FREEZE_20260625.md",
            "reports/latentfm_scaling_s0_provenance_freeze_20260625.json",
            "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv",
        ],
        "purpose": "Freeze S0/source provenance for scaling axes.",
    },
    {
        "step": "truecell_count_tail_completion",
        "command": f"{PY} ops/audit_latentfm_truecell_scaling_count_tail_completion_gate_20260625.py",
        "script": "ops/audit_latentfm_truecell_scaling_count_tail_completion_gate_20260625.py",
        "outputs": [
            "reports/LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md",
            "reports/latentfm_truecell_scaling_count_tail_completion_gate_20260625.json",
            "reports/latentfm_truecell_scaling_count_tail_completion_rows_20260625.csv",
        ],
        "purpose": "Gate true-cell support/count scaling with tails and canonical no-harm veto.",
    },
    {
        "step": "condition_exposure_row_bootstrap",
        "command": f"{PY} ops/audit_latentfm_condition_exposure_row_bootstrap_gate_20260625.py",
        "script": "ops/audit_latentfm_condition_exposure_row_bootstrap_gate_20260625.py",
        "outputs": [
            "reports/LATENTFM_CONDITION_EXPOSURE_ROW_BOOTSTRAP_GATE_20260625.md",
            "reports/latentfm_condition_exposure_row_bootstrap_gate_20260625.json",
            "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
        ],
        "purpose": "Bootstrap condition-level exposure differences and tail controls.",
    },
    {
        "step": "source_resolved_estimand_v2",
        "command": f"{PY} ops/audit_latentfm_scaling_source_resolved_estimand_v2_gate_20260625.py",
        "script": "ops/audit_latentfm_scaling_source_resolved_estimand_v2_gate_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_SOURCE_RESOLVED_ESTIMAND_V2_GATE_20260625.md",
            "reports/latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json",
        ],
        "purpose": "Gate source-resolved background/type confounding.",
    },
    {
        "step": "completion_readiness",
        "command": f"{PY} ops/build_latentfm_scaling_completion_readiness_20260625.py",
        "script": "ops/build_latentfm_scaling_completion_readiness_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_COMPLETION_READINESS_20260625.md",
            "reports/latentfm_scaling_completion_readiness_20260625.json",
            "reports/scaling_completion_readiness_20260625/axis_completion_matrix.csv",
        ],
        "purpose": "Summarize axis readiness and claim scope.",
    },
    {
        "step": "figure_data",
        "command": f"{PY} ops/build_latentfm_scaling_figure_data_20260625.py",
        "script": "ops/build_latentfm_scaling_figure_data_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_FIGURE_DATA_PACKAGE_20260625.md",
            "reports/latentfm_scaling_figure_data_package_20260625.json",
            "reports/scaling_figure_data_20260625/truecell_budget_curve.csv",
        ],
        "purpose": "Build figure-ready data tables.",
    },
    {
        "step": "plot_figures",
        "command": f"{PY} ops/plot_latentfm_scaling_figures_20260625.py",
        "script": "ops/plot_latentfm_scaling_figures_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_FIGURES_20260625.md",
            "reports/latentfm_scaling_figures_20260625.json",
            "reports/scaling_figures_20260625/Fig_scaling_truecell_budget.png",
        ],
        "purpose": "Render scaling PNG/SVG figures from figure-data CSVs.",
    },
    {
        "step": "metadata_artifact_feasibility",
        "command": f"{PY} ops/audit_latentfm_new_artifact_source_feasibility_20260625.py",
        "script": "ops/audit_latentfm_new_artifact_source_feasibility_20260625.py",
        "outputs": [
            "reports/LATENTFM_NEW_ARTIFACT_SOURCE_FEASIBILITY_20260625.md",
            "reports/latentfm_new_artifact_source_feasibility_20260625.json",
            "reports/latentfm_new_artifact_source_feasibility_rows_20260625.csv",
        ],
        "purpose": "Check raw obs metadata as possible new train-only artifacts.",
    },
    {
        "step": "qc_support_gate",
        "command": f"{PY} ops/audit_latentfm_qc_support_reliability_gate_20260625.py",
        "script": "ops/audit_latentfm_qc_support_reliability_gate_20260625.py",
        "outputs": [
            "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
            "reports/latentfm_qc_support_reliability_gate_20260625.json",
            "reports/latentfm_qc_support_reliability_rows_20260625.csv",
        ],
        "purpose": "Gate QC/support metadata as a training-set signal.",
    },
    {
        "step": "jiang_context_gate",
        "command": f"{PY} ops/audit_latentfm_jiang_guide_cytokine_context_gate_20260625.py",
        "script": "ops/audit_latentfm_jiang_guide_cytokine_context_gate_20260625.py",
        "outputs": [
            "reports/LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md",
            "reports/latentfm_jiang_guide_cytokine_context_gate_20260625.json",
            "reports/latentfm_jiang_guide_cytokine_context_rows_20260625.csv",
        ],
        "purpose": "Gate Jiang guide/cytokine/mixscale context.",
    },
    {
        "step": "claim_failure_package",
        "command": f"{PY} ops/build_latentfm_scaling_nm_claim_failure_package_20260625.py",
        "script": "ops/build_latentfm_scaling_nm_claim_failure_package_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md",
            "reports/latentfm_scaling_nm_claim_failure_package_20260625.json",
            "reports/scaling_nm_claim_failure_package_20260625/top_failure_cases.csv",
        ],
        "purpose": "Build claim boundary, failure cases, literature boundary, and guidance.",
    },
    {
        "step": "provenance_manifest",
        "command": f"{PY} ops/build_latentfm_scaling_nm_provenance_manifest_20260625.py",
        "script": "ops/build_latentfm_scaling_nm_provenance_manifest_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_NM_PROVENANCE_MANIFEST_20260625.md",
            "reports/latentfm_scaling_nm_provenance_manifest_20260625.json",
            "reports/scaling_nm_provenance_manifest_20260625/artifact_manifest.tsv",
        ],
        "purpose": "Hash reports, tables, figures, and claim-to-artifact mappings.",
    },
    {
        "step": "figure_readiness",
        "command": f"{PY} ops/audit_latentfm_scaling_figure_readiness_20260625.py",
        "script": "ops/audit_latentfm_scaling_figure_readiness_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_FIGURE_READINESS_20260625.md",
            "reports/latentfm_scaling_figure_readiness_20260625.json",
            "reports/scaling_figure_readiness_20260625/figure_readiness.csv",
        ],
        "purpose": "QA PNG/SVG dimensions, nonblank signal, and manifest hash consistency.",
    },
    {
        "step": "narrative_skeleton",
        "command": f"{PY} ops/build_latentfm_scaling_narrative_skeleton_20260625.py",
        "script": "ops/build_latentfm_scaling_narrative_skeleton_20260625.py",
        "outputs": [
            "reports/LATENTFM_SCALING_NARRATIVE_SKELETON_20260625.md",
            "reports/latentfm_scaling_narrative_skeleton_20260625.json",
            "reports/scaling_narrative_skeleton_20260625/reviewer_checklist.tsv",
        ],
        "purpose": "Build manuscript narrative skeleton and reviewer checklist.",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    command_rows = []
    script_rows = []
    seen_scripts = set()
    for step in STEPS:
        script_path = ROOT / step["script"]
        outputs = [ROOT / rel for rel in step["outputs"]]
        command_rows.append(
            {
                "step": step["step"],
                "purpose": step["purpose"],
                "command": step["command"],
                "script": step["script"],
                "script_exists": script_path.is_file(),
                "outputs_present": all(p.exists() for p in outputs),
                "outputs": ";".join(step["outputs"]),
            }
        )
        if step["script"] not in seen_scripts:
            seen_scripts.add(step["script"])
            script_rows.append(
                {
                    "script": step["script"],
                    "exists": script_path.is_file(),
                    "size_bytes": script_path.stat().st_size if script_path.is_file() else 0,
                    "sha256": sha256(script_path) if script_path.is_file() else "",
                }
            )
    return command_rows, script_rows


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Scaling Reproduction Manifest",
        "",
        "Timestamp: `2026-06-25 23:58 CST`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report reproduction manifest for completed scaling artifacts.",
        "- Records commands, script hashes, and expected outputs.",
        "- Does not launch experiments, read checkpoints, read canonical multi, read Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- command steps: `{payload['summary']['n_steps']}`",
        f"- scripts hashed: `{payload['summary']['n_scripts']}`",
        f"- missing scripts: `{payload['summary']['missing_scripts']}`",
        f"- steps with missing outputs: `{payload['summary']['steps_missing_outputs']}`",
        "",
        "## Commands",
        "",
        "| step | script exists | outputs present | command |",
        "|---|---|---|---|",
    ]
    for row in payload["command_rows"]:
        lines.append(
            f"| `{row['step']}` | `{row['script_exists']}` | `{row['outputs_present']}` | `{row['command']}` |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- commands: `{OUT_COMMANDS}`",
        f"- script hashes: `{OUT_SCRIPTS}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "This is reproduction/provenance support only. It does not authorize GPU training.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    command_rows, script_rows = build_rows()
    summary = {
        "n_steps": len(command_rows),
        "n_scripts": len(script_rows),
        "missing_scripts": sum(1 for r in script_rows if not r["exists"]),
        "steps_missing_outputs": sum(1 for r in command_rows if not r["outputs_present"]),
    }
    payload = {
        "status": "scaling_reproduction_manifest_ready_no_gpu"
        if summary["missing_scripts"] == 0 and summary["steps_missing_outputs"] == 0
        else "scaling_reproduction_manifest_incomplete_no_gpu",
        "gpu_authorized": False,
        "summary": summary,
        "command_rows": command_rows,
        "script_rows": script_rows,
        "outputs": {"commands": str(OUT_COMMANDS), "script_hashes": str(OUT_SCRIPTS), "json": str(OUT_JSON), "md": str(OUT_MD)},
    }
    write_tsv(OUT_COMMANDS, command_rows, ["step", "purpose", "command", "script", "script_exists", "outputs_present", "outputs"])
    write_tsv(OUT_SCRIPTS, script_rows, ["script", "exists", "size_bytes", "sha256"])
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
