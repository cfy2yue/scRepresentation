#!/usr/bin/env python3
"""Audit all-modality true-cell-count artifact readiness.

CPU-only. This formalizes whether existing all-modality partial artifacts may be
used for GPU training or claims.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ARTIFACT_ROOT = ROOT / "runs/latentfm_true_cell_count_scaling_capped_h5_20260624/artifacts"
OUT_JSON = REPORTS / "latentfm_true_cell_count_allmodality_readiness_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_READINESS_GATE_20260625.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text())


def main() -> None:
    protocol = load_json(REPORTS / "latentfm_true_cell_count_scaling_protocol_20260624.json")
    materializer = load_json(REPORTS / "latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json")
    invalid_readme = ARTIFACT_ROOT / "INVALID_PARTIAL_ARTIFACTS_README.md"
    all_dirs = sorted(ARTIFACT_ROOT.glob("all_modality_fixed64_budget16_32_64_*"))
    rows = []
    for d in all_dirs:
        files = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        rows.append(
            {
                "dir": str(d),
                "manifest_exists": (d / "manifest.json").exists(),
                "pert_means_exists": (d / "pert_means.npz").exists(),
                "sampled_indices_exists": (d / "sampled_indices.npz").exists(),
                "tmp_files": [f for f in files if f.endswith(".tmp")],
                "n_files": len(files),
            }
        )

    materialized_run_ids = [r.get("run_id", "") for r in materializer.get("materialized_rows", [])]
    plan_run_ids = [r.get("run_id", "") for r in materializer.get("plan_rows", [])]
    protocol_rows = protocol.get("protocols", [])
    if not protocol_rows and "protocols" not in protocol:
        protocol_rows = protocol.get("rows", [])

    allmod_in_materialized = [x for x in materialized_run_ids if "all_modality" in x]
    allmod_in_plan = [x for x in plan_run_ids if "all_modality" in x]
    invalid_text = invalid_readme.read_text() if invalid_readme.exists() else ""

    reasons = []
    if invalid_readme.exists():
        reasons.append("invalid_partial_artifacts_readme_present")
    if "not launcher-ready" in invalid_text or "drug-level" in invalid_text:
        reasons.append("sciplex_dose_level_vs_xverse_drug_level_label_mismatch")
    if not allmod_in_materialized:
        reasons.append("no_all_modality_rows_in_materializer_gate")
    if any(row["tmp_files"] for row in rows):
        reasons.append("partial_tmp_files_present")
    if not any(row["sampled_indices_exists"] for row in rows):
        reasons.append("sampled_indices_missing_for_all_modality")

    status = "allmodality_truecell_readiness_fail_no_gpu" if reasons else "allmodality_truecell_readiness_pass_cpu_next"
    payload = {
        "boundary": {
            "cpu_only": True,
            "gpu": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "reads_artifact_manifests_only": True,
        },
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "decision": {
            "use_existing_allmodality_artifacts": False,
            "launch_gpu": False,
            "next_action": "fix_or_redesign_label_compatibility_before_materialization" if reasons else "run_schema_dryload_design_gates",
        },
        "inputs": {
            "protocol_json": str(REPORTS / "latentfm_true_cell_count_scaling_protocol_20260624.json"),
            "materializer_json": str(REPORTS / "latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json"),
            "invalid_readme": str(invalid_readme),
        },
        "all_modality_artifact_dirs": rows,
        "allmod_in_materialized": allmod_in_materialized,
        "allmod_in_plan": allmod_in_plan,
        "protocol_rows_count": len(protocol_rows),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# LatentFM True-Cell All-Modality Readiness Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact/readiness audit.",
        "- Does not read canonical multi, held-out Track C query, train, infer, or use GPU.",
        "- Existing partial all-modality artifacts are audited for launcher readiness only.",
        "",
        "## Findings",
        "",
        f"- invalid partial README present: `{invalid_readme.exists()}`",
        f"- all-modality materialized rows in official materializer gate: `{len(allmod_in_materialized)}`",
        f"- all-modality plan rows in official materializer gate: `{len(allmod_in_plan)}`",
        "",
        "| artifact dir | manifest | pert means | sampled indices | tmp files |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dir']}` | `{row['manifest_exists']}` | `{row['pert_means_exists']}` | `{row['sampled_indices_exists']}` | `{row['tmp_files']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- GPU authorized: `False`.",
            "- Do not use existing all-modality partial artifacts for training or claims.",
            "- Reason: current xverse split/H5 labels are drug-level while the SciPlex all-modality protocol rows are dose-level, so the train/eval intersection collapses to gene-only.",
            "- Next action: only reopen after a corrected label-compatibility/materializer gate regenerates artifacts and passes schema/dry-load/design controls.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
