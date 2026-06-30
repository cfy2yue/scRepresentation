#!/usr/bin/env python3
"""Synthesize true time-course / zebrafish readiness for LatentFM translation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_OUT_DIR = ROOT / "reports/zscape_true_timecourse_readiness_gate_20260630"

JSON_INPUTS = {
    "dynamic_information": ROOT / "reports/zscape_dynamic_information_modeling_gate_20260628/zscape_dynamic_information_modeling_gate_20260628.json",
    "state_preserved_time_vector": ROOT / "reports/zscape_state_preserved_time_vector_gate_20260629/zscape_state_preserved_time_vector_gate_20260629.json",
    "embryo_vector_consistency": ROOT / "reports/zscape_embryo_vector_consistency_gate_20260628/zscape_embryo_vector_consistency_gate_20260628.json",
    "heldout_dynamic_specificity": ROOT / "reports/zscape_embryo_heldout_dynamic_specificity_gate_20260628/zscape_embryo_heldout_dynamic_specificity_gate_20260628.json",
    "periderm_substate_qc": ROOT / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628/zscape_periderm_substate_time_qc_ot_module_gate_20260628.json",
    "latent_readiness": ROOT / "reports/zscape_scfm_latent_readiness_20260628/zscape_scfm_latent_readiness_20260628.json",
}
CSV_INPUTS = {
    "dynamic_information_rows": ROOT / "reports/zscape_dynamic_information_modeling_gate_20260628/zscape_dynamic_information_row_synthesis.csv",
}

LOCAL_ARTIFACTS = {
    "zscape_perturb_metadata": ROOT / "dataset/external/zscape_20260628/GSE202639_zperturb_full_cell_metadata.csv.gz",
    "zscape_reference_metadata": ROOT / "dataset/external/zscape_20260628/GSE202639_reference_cell_metadata.csv.gz",
    "zscape_raw_counts": ROOT / "dataset/external/zscape_20260628/GSE202639_zperturb_full_raw_counts.RDS.gz",
    "selected_counts_npz": ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_counts_csc.npz",
    "squidiff_repo": ROOT / "external_review/Squidiff",
    "squidiff_repro_repo": ROOT / "external_review/Squidiff_reproducibility",
    "zscape_paper": ROOT / "scFMBench/ref/s41586-023-06720-2.pdf",
    "squidiff_paper": ROOT / "scFMBench/ref/squidiff.pdf",
}


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return json.load(fh)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def read_dynamic_rows(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        return [], []
    geometry: list[str] = []
    specificity_failed: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row_id = str(row.get("row_id", ""))
            if str(row.get("geometry_gate", "")).lower() == "true":
                geometry.append(row_id)
            if str(row.get("modeling_class", "")) == "geometry_positive_module_specificity_failed":
                specificity_failed.append(row_id)
    return geometry, specificity_failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    data = {name: load_json(path) for name, path in JSON_INPUTS.items()}
    state = data["state_preserved_time_vector"]
    embryo = data["embryo_vector_consistency"]
    heldout = data["heldout_dynamic_specificity"]
    substate = data["periderm_substate_qc"]
    latent = data["latent_readiness"]

    positive_vector_rows = as_list(state.get("positive_vector_rows"))
    model_ready_rows = as_list(state.get("model_constraint_ready_rows"))
    wrong_controls = as_list(state.get("wrong_control_rows_failing_vector_gate"))
    geometry_positive, geometry_specificity_failed = read_dynamic_rows(CSV_INPUTS["dynamic_information_rows"])
    heldout_row_summary = as_list(heldout.get("row_summary"))
    heldout_all_pass = [
        row.get("row_id")
        for row in heldout_row_summary
        if bool(row.get("all_query_gates"))
    ]
    substate_row_summary = as_list(substate.get("row_summary"))
    substate_all_pass = [
        row.get("row_id")
        for row in substate_row_summary
        if bool(row.get("all_query_gates"))
    ]
    artifact_rows = [
        {
            "artifact": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
        for name, path in LOCAL_ARTIFACTS.items()
    ]

    expression_timecourse_ready = (
        bool(positive_vector_rows)
        and len(positive_vector_rows) >= 2
        and all(row in geometry_positive for row in positive_vector_rows)
    )
    replicate_consistency_positive = bool(embryo.get("positive_rows")) and not bool(embryo.get("positive_gate_is_discriminative"))
    specificity_blocked = not model_ready_rows and not heldout_all_pass and not substate_all_pass
    latent_route_ready = bool(latent.get("gpu_authorized_next")) or bool(latent.get("latent_route_ready"))
    local_data_ready = all(row["exists"] for row in artifact_rows if row["artifact"] in {
        "zscape_perturb_metadata",
        "zscape_reference_metadata",
        "zscape_raw_counts",
        "selected_counts_npz",
    })
    squidiff_context_ready = all(row["exists"] for row in artifact_rows if row["artifact"] in {"squidiff_repo", "squidiff_repro_repo", "squidiff_paper"})

    if expression_timecourse_ready and specificity_blocked:
        status = "zscape_true_timecourse_readiness_biology_positive_model_route_blocked_no_gpu"
    elif expression_timecourse_ready:
        status = "zscape_true_timecourse_readiness_review_required_no_gpu"
    else:
        status = "zscape_true_timecourse_readiness_insufficient_no_gpu"

    decision = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "expression_timecourse_ready": expression_timecourse_ready,
        "local_data_ready": local_data_ready,
        "squidiff_context_ready": squidiff_context_ready,
        "replicate_consistency_positive_but_nonspecific": replicate_consistency_positive,
        "specificity_blocked": specificity_blocked,
        "latent_route_ready": latent_route_ready,
        "positive_vector_rows": positive_vector_rows,
        "wrong_control_rows_failing_vector_gate": wrong_controls,
        "geometry_positive_rows": geometry_positive,
        "geometry_positive_specificity_failed_rows": geometry_specificity_failed,
        "heldout_all_pass_rows": heldout_all_pass,
        "substate_all_pass_rows": substate_all_pass,
        "interpretation": (
            "ZSCAPE provides expression-space, embryo-replicated, state-preserved dynamic geometry "
            "for periderm noto/smo, but current module/pathway specificity and species-safe latent "
            "route gates block LatentFM/RawFM loss or architecture changes."
        ),
        "inputs": {name: str(path) for name, path in {**JSON_INPUTS, **CSV_INPUTS}.items()},
        "artifacts": artifact_rows,
        "boundary": "cpu_report_only_synthesis_no_training_no_inference_no_gpu_no_new_ot_no_canonical_multi_no_trackc_query",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "zscape_true_timecourse_readiness_gate_20260630.json"
    json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts_csv = args.out_dir / "zscape_true_timecourse_artifact_readiness.csv"
    with artifacts_csv.open("w", encoding="utf-8") as fh:
        fh.write("artifact,path,exists,size_bytes\n")
        for row in artifact_rows:
            fh.write(f"{row['artifact']},{row['path']},{row['exists']},{row['size_bytes'] or ''}\n")

    md_path = args.out_dir / "LATENTFM_ZSCAPE_TRUE_TIMECOURSE_READINESS_GATE_20260630.md"
    lines = [
        "# LatentFM ZSCAPE True Time-Course Readiness Gate",
        "",
        f"Created: {decision['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only synthesis of frozen ZSCAPE dynamic, embryo, heldout-specificity, substate/QC, latent-readiness, and Squidiff-context materials.",
        "* No new OT pairing, training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* The goal is to separate biological insight from model-ready constraints.",
        "",
        "## Data Readiness",
        "",
        f"* Local ZSCAPE expression data ready: `{local_data_ready}`.",
        f"* Squidiff context/repro assets ready: `{squidiff_context_ready}`.",
        f"* Expression time-course geometry ready: `{expression_timecourse_ready}`.",
        f"* Species-safe true latent route ready: `{latent_route_ready}`.",
        "",
        "## Biological Findings",
        "",
        f"* State-preserved vector-positive rows: `{', '.join(positive_vector_rows) if positive_vector_rows else 'none'}`.",
        f"* Wrong-control rows failing vector gate: `{', '.join(wrong_controls) if wrong_controls else 'none'}`.",
        f"* Geometry-positive rows: `{', '.join(geometry_positive) if geometry_positive else 'none'}`.",
        f"* Geometry-positive but specificity-failed rows: `{', '.join(geometry_specificity_failed) if geometry_specificity_failed else 'none'}`.",
        f"* Embryo vector consistency is broad but nonspecific: `{replicate_consistency_positive}`.",
        f"* Heldout module-specific rows passing all gates: `{', '.join(map(str, heldout_all_pass)) if heldout_all_pass else 'none'}`.",
        f"* Substate/QC module-specific rows passing all gates: `{', '.join(map(str, substate_all_pass)) if substate_all_pass else 'none'}`.",
        "",
        "## Modeling Decision",
        "",
        "* Keep ZSCAPE as a biological dynamic-insight branch: periderm `noto/smo` support state-preserved response geometry aligned with the normal 24h-to-36h periderm vector.",
        "* Do not convert current ZSCAPE modules, time vectors, or pseudo-OT pairs into LatentFM/RawFM losses or architecture changes.",
        "* Squidiff is a useful conceptual template for trajectory-conditioned generation, but the local ZSCAPE evidence is snapshot/pseudo-pair expression geometry, not true lineage-paired model supervision.",
        "* A model route requires a species-safe latent representation or raw-expression model design plus specificity gates that beat wrong-time, wrong-lineage, and wrong-target controls.",
        "",
        "## Next Action",
        "",
        "* Use ZSCAPE periderm `noto/smo` as biology/failure-analysis examples and as motivation for future state-preserving dynamic constraints.",
        "* For executable LatentFM work, continue with CPU-first gates or engineering throughput work until a new leakage-safe model route is authorized.",
        "",
        "## Outputs",
        "",
        f"* JSON: `{json_path}`",
        f"* Artifact readiness: `{artifacts_csv}`",
        "",
        "## Sources",
        "",
    ]
    for name, path in {**JSON_INPUTS, **CSV_INPUTS}.items():
        lines.append(f"* {name}: `{path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
