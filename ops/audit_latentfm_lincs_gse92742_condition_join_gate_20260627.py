#!/usr/bin/env python3
"""Condition-level join gate for the complete GSE92742 LINCS small metadata.

This reuses the already reviewed GSE70138 small-metadata join functions, but
points them at GSE92742 `sig_info` and `sig_metrics`. It does not download
Level5 matrices, train, infer, select checkpoints, or use GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE_DIR = ROOT / "reports/external_artifact_sources_20260627/lincs_l1000_geo_small"
BASE_SCRIPT = ROOT / "ops/audit_latentfm_lincs_gse70138_condition_join_gate_20260627.py"

SIG_INFO = SOURCE_DIR / "GSE92742_Broad_LINCS_sig_info.txt.gz"
SIG_METRICS = SOURCE_DIR / "GSE92742_Broad_LINCS_sig_metrics.txt.gz"

OUT_DIR = ROOT / "reports/lincs_l1000_gse92742_condition_join_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_lincs_l1000_gse92742_condition_join_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_LINCS_L1000_GSE92742_CONDITION_JOIN_GATE_20260627.md"
OUT_AGG = OUT_DIR / "gse92742_condition_level_activity.csv"
OUT_OVERLAP = OUT_DIR / "gse92742_s0_overlap_rows.csv"


def load_base_module():
    spec = importlib.util.spec_from_file_location("lincs_gse70138_gate_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base gate script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_base_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "large_level5_download": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "chemical_v2_ack": False,
        "source_release": "GSE92742_small_metadata_only",
        "base_join_logic": str(BASE_SCRIPT),
    }
    missing = [str(p) for p in (SIG_INFO, SIG_METRICS) if not p.is_file()]
    if missing:
        out = {
            "status": "lincs_gse92742_condition_join_missing_source_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# LINCS GSE92742 Condition Join Gate\n\nMissing source files; no GPU authorized.\n", encoding="utf-8")
        print(json.dumps({"status": out["status"], "gpu_authorized": False}, indent=2))
        return 0

    base.SIG_INFO = SIG_INFO
    base.SIG_METRICS = SIG_METRICS

    lincs_rows, lincs_summary = base.aggregate_lincs()
    agg_fields = [
        "pert_iname",
        "pert_key",
        "pert_type",
        "cell_id",
        "cell_key",
        "pert_idose",
        "pert_itime",
        "sig_count",
        "tas_mean",
        "tas_n",
        "distil_cc_q75_mean",
        "distil_cc_q75_n",
        "distil_ss_mean",
        "distil_ss_n",
        "distil_nsample_mean",
        "distil_nsample_n",
    ]
    base.write_csv(OUT_AGG, lincs_rows, agg_fields)

    outcome_keys = base.read_outcome_keys()
    trainonly_rows = base.read_s0_rows(outcome_keys)
    full_s0_rows = base.read_s0_rows(None)
    trainonly_overlap = base.overlap_rows(lincs_rows, trainonly_rows)
    full_overlap = base.overlap_rows(lincs_rows, full_s0_rows)
    base.write_csv(
        OUT_OVERLAP,
        full_overlap,
        [
            "dataset",
            "condition",
            "membership",
            "modality",
            "perturbation_type",
            "s0_perturbation",
            "s0_cell_background",
            "s0_dose",
            "lincs_pert_iname",
            "lincs_pert_type",
            "lincs_cell_id",
            "lincs_pert_idose",
            "lincs_pert_itime",
            "lincs_sig_count",
            "tas_mean",
            "distil_cc_q75_mean",
        ],
    )

    trainonly_summary = base.summarize_overlap(trainonly_overlap)
    full_summary = base.summarize_overlap(full_overlap)
    reasons: list[str] = []
    if trainonly_summary["overlap_rows"] < 50:
        reasons.append("trainonly_overlap_below_50")
    if trainonly_summary["unique_s0_conditions"] < 50:
        reasons.append("trainonly_unique_condition_overlap_below_50")
    if trainonly_summary["exact_cell_background_match_rows"] == 0:
        reasons.append("trainonly_exact_background_match_zero")
    if full_summary["overlap_rows"] > 0:
        reasons.append("full_s0_overlap_is_diagnostic_or_ack_gated")
    reasons.extend(
        [
            "chemical_v2_exact_ack_absent",
            "shuffle_source_mmd_tail_gates_not_run",
            "no_gpu_from_schema_or_overlap_only",
        ]
    )

    status = "lincs_gse92742_condition_join_fail_no_gpu"
    gpu_authorized = False
    out = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": boundary,
        "lincs_summary": lincs_summary,
        "trainonly_outcome_universe_rows": len(trainonly_rows),
        "full_s0_rows": len(full_s0_rows),
        "trainonly_overlap_summary": trainonly_summary,
        "full_s0_overlap_summary": full_summary,
        "reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "condition_level_activity": str(OUT_AGG),
            "s0_overlap_rows": str(OUT_OVERLAP),
        },
        "next_action": (
            "Do not launch GPU from GSE92742 schema/overlap alone. If overlap "
            "is materially stronger than GSE70138, the next step is a strict "
            "train-only LINCS signal/control gate with within-dataset shuffle, "
            "source/type/background controls, and MMD/tail veto; broad chemical "
            "overlap remains Chemical-V2-ACK gated."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LINCS/L1000 GSE92742 Condition Join Gate",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- Uses only complete GSE92742 `sig_info` and `sig_metrics` small metadata.",
        "- Reuses the GSE70138 small-metadata join logic for comparable evidence.",
        "- No Level5 matrices, training, inference, canonical multi selection, Track C held-out query, or GPU.",
        "- Chemical V2 exact ACK is absent; chemical overlap is diagnostic only.",
        "",
        "## LINCS Materialization",
        "",
        f"- sig_info rows: `{lincs_summary['sig_info_rows']}`",
        f"- sig_metrics rows: `{lincs_summary['metrics_rows']}`",
        f"- joined signature rows: `{lincs_summary['joined_signature_rows']}`",
        f"- condition-level rows: `{lincs_summary['condition_level_rows']}`",
        f"- perturbation types: `{lincs_summary['pert_type_counts']}`",
        "",
        "## Overlap",
        "",
        f"- current train-only outcome universe rows: `{len(trainonly_rows)}`",
        f"- train-only overlap rows: `{trainonly_summary['overlap_rows']}`",
        f"- train-only unique S0 conditions: `{trainonly_summary['unique_s0_conditions']}`",
        f"- train-only exact cell/background match rows: `{trainonly_summary['exact_cell_background_match_rows']}`",
        f"- train-only prefix cell/background match rows: `{trainonly_summary['prefix_cell_background_match_rows']}`",
        f"- full S0 overlap rows: `{full_summary['overlap_rows']}`",
        f"- full S0 unique S0 conditions: `{full_summary['unique_s0_conditions']}`",
        f"- full S0 membership counts: `{full_summary['membership_counts']}`",
        f"- full S0 modality counts: `{full_summary['modality_counts']}`",
        f"- exact cell/background match rows: `{full_summary['exact_cell_background_match_rows']}`",
        f"- prefix cell/background match rows: `{full_summary['prefix_cell_background_match_rows']}`",
        "",
        "## Decision",
        "",
        "No GPU is authorized. This gate only establishes whether the small metadata can be joined to the frozen S0/outcome universe. Any model use would require a separate train-only signal/control gate and no-harm veto; chemical overlap remains ACK-gated.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- condition-level activity CSV: `{OUT_AGG}`",
        f"- S0 overlap CSV: `{OUT_OVERLAP}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
