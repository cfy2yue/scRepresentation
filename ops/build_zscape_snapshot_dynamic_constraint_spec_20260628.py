#!/usr/bin/env python3
"""Build a ZSCAPE snapshot dynamic constraint specification.

This CPU/report-only synthesis freezes which ZSCAPE expression-space rows are
eligible as future dynamic-constraint candidates and records why latent/model
training remains blocked.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
STRICT_ROWS = ROOT / "runs/zscape_expression_ot_strict_controls_gate_20260628/zscape_expression_ot_strict_controls_gate_20260628_082748/outputs/zscape_expression_ot_strict_primary_rows.csv"
STRICT_DIAG = ROOT / "runs/zscape_expression_ot_strict_controls_gate_20260628/zscape_expression_ot_strict_controls_gate_20260628_082748/outputs/zscape_expression_ot_strict_diagnostics.csv"
TRAJ_ROWS = ROOT / "runs/zscape_expression_trajectory_time_gate_20260628/zscape_expression_trajectory_time_gate_20260628_084025/outputs/zscape_expression_trajectory_time_perturb_alignment.csv"
TEMPORAL_CONTROLS = ROOT / "runs/zscape_expression_trajectory_time_gate_20260628/zscape_expression_trajectory_time_gate_20260628_084025/outputs/zscape_expression_trajectory_time_temporal_controls.csv"
FIXED_ROWS = ROOT / "runs/zscape_bioinformation_fixedcell_robustness_gate_20260628/zscape_bioinformation_fixedcell_robustness_gate_20260628_112326/outputs/zscape_bioinformation_fixedcell_row_results.csv"
PLACEBO_ROWS = ROOT / "runs/zscape_periderm_placebo_control_20260628/zscape_periderm_placebo_control_20260628_133618/outputs/zscape_periderm_placebo_rows.csv"
PLACEBO_JSON = ROOT / "runs/zscape_periderm_placebo_control_20260628/zscape_periderm_placebo_control_20260628_133618/outputs/zscape_periderm_placebo_control_20260628.json"
EMBRYO_MODULE_ROWS = ROOT / "reports/zscape_embryo_pseudobulk_module_gate_20260628/zscape_embryo_pseudobulk_module_rows.csv"
LATENT_READY_JSON = ROOT / "reports/zscape_scfm_latent_readiness_20260628/zscape_scfm_latent_readiness_20260628.json"
LATENT_PROXY_JSON = ROOT / "reports/zscape_latent_proxy_reconciliation_20260628/zscape_latent_proxy_reconciliation_20260628.json"
OUT_DIR = ROOT / "reports/zscape_snapshot_dynamic_constraint_spec_20260628"
OUT_ROWS = OUT_DIR / "zscape_snapshot_dynamic_constraint_rows.csv"
OUT_TEMPORAL = OUT_DIR / "zscape_snapshot_dynamic_temporal_controls.csv"
OUT_JSON = OUT_DIR / "zscape_snapshot_dynamic_constraint_spec_20260628.json"
OUT_MD = OUT_DIR / "LATENTFM_ZSCAPE_SNAPSHOT_DYNAMIC_CONSTRAINT_SPEC_20260628.md"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def diag_pivot() -> dict[str, dict[str, float]]:
    if not STRICT_DIAG.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in pd.read_csv(STRICT_DIAG).itertuples(index=False):
        out.setdefault(str(row.row_id), {})[str(row.diagnostic)] = float(row.ot)
    return out


def embryo_summary() -> dict[str, dict[str, Any]]:
    if not EMBRYO_MODULE_ROWS.exists():
        return {}
    frame = pd.read_csv(EMBRYO_MODULE_ROWS)
    frame = frame[frame["qc_filtered"] == False]  # noqa: E712
    out: dict[str, dict[str, Any]] = {}
    for row_id, part in frame.groupby("row_id", sort=True):
        gates = part["embryo_module_gate"].astype(bool)
        best = part.sort_values("directed_diff_ci95_low", ascending=False).iloc[0]
        out[str(row_id)] = {
            "embryo_module_gates": int(gates.sum()),
            "embryo_module_total": int(part.shape[0]),
            "embryo_module_min_ci_low": float(part["directed_diff_ci95_low"].min()),
            "best_embryo_module_query": str(best["query_name"]),
            "best_embryo_module_directed_diff": float(best["directed_embryo_mean_diff"]),
            "best_embryo_module_p": float(best["welch_p_value"]),
        }
    return out


def placebo_summary() -> dict[str, dict[str, Any]]:
    if not PLACEBO_ROWS.exists():
        return {}
    frame = pd.read_csv(PLACEBO_ROWS)
    out: dict[str, dict[str, Any]] = {}
    for row_id, part in frame.groupby("row_id", sort=True):
        ok = part[part["status"] == "ok"].copy()
        if ok.empty:
            out[str(row_id)] = {"placebo_ok_modes": 0}
            continue
        out[str(row_id)] = {
            "placebo_ok_modes": int(ok.shape[0]),
            "placebo_max_real_beats_fraction": float(ok["real_beats_placebo_fraction"].max()),
            "placebo_min_real_vs_placebo_ratio": float(ok["real_vs_placebo_ratio_median"].min()),
            "placebo_modes": ";".join(ok["placebo_mode"].astype(str).tolist()),
        }
    return out


def build_rows() -> list[dict[str, Any]]:
    strict = pd.read_csv(STRICT_ROWS)
    traj = pd.read_csv(TRAJ_ROWS).set_index("row_id").to_dict("index")
    fixed = pd.read_csv(FIXED_ROWS).set_index("row_id").to_dict("index")
    diag = diag_pivot()
    embryo = embryo_summary()
    placebo = placebo_summary()
    branch_placebo = read_json(PLACEBO_JSON) if PLACEBO_JSON.exists() else {}
    branch_placebo_pass = str(branch_placebo.get("status", "")).endswith("pass_no_gpu")
    rows: list[dict[str, Any]] = []
    for row in strict.itertuples(index=False):
        row_id = str(row.row_id)
        trow = traj.get(row_id, {})
        frow = fixed.get(row_id, {})
        drow = diag.get(row_id, {})
        erow = embryo.get(row_id, {})
        prow = placebo.get(row_id, {})
        strict_gate = truthy(row.strict_row_gate)
        trajectory_gate = truthy(trow.get("alignment_gate", False))
        fixed_gate = truthy(frow.get("strict_row_gate", False))
        embryo_gate = int(erow.get("embryo_module_gates", 0)) >= 1
        state_preserved = (
            float(row.matched_subtype_jsd) <= 0.02
            and abs(float(row.expression_library_smd)) <= 0.05
        )
        expression_candidate = (
            str(row.cell_type_broad) == "periderm"
            and strict_gate
            and trajectory_gate
            and fixed_gate
            and embryo_gate
            and branch_placebo_pass
        )
        if expression_candidate:
            recommended_use = "expression_snapshot_dynamic_constraint_candidate"
        elif str(row.cell_type_broad) == "mature fast muscle":
            recommended_use = "negative_confounded_comparator"
        elif str(row.cell_type_broad) == "periderm" and strict_gate and fixed_gate:
            recommended_use = "periderm_state_preservation_diagnostic"
        else:
            recommended_use = "diagnostic_only"
        rows.append(
            {
                "row_id": row_id,
                "lineage": row.cell_type_broad,
                "target": row.gene_target,
                "timepoint": row.timepoint,
                "strict_row_gate": strict_gate,
                "strict_effect_ratio": float(row.effect_ratio_vs_max_null_p95),
                "matched_subtype_jsd": float(row.matched_subtype_jsd),
                "expression_library_smd": float(row.expression_library_smd),
                "state_preserved_by_threshold": state_preserved,
                "wrong_time_control_ot": drow.get("wrong_time_control", np.nan),
                "wrong_lineage_control_ot": drow.get("wrong_lineage_control", np.nan),
                "trajectory_alignment_gate": trajectory_gate,
                "trajectory_cosine": trow.get("cosine_to_lineage_time_vector", np.nan),
                "trajectory_wrong_lineage_max": trow.get("max_cosine_to_wrong_lineage_time_vector", np.nan),
                "trajectory_margin": trow.get("cosine_margin_vs_wrong_lineage", np.nan),
                "fixed_cell_gate": fixed_gate,
                "fixed_cell_effect_ratio": frow.get("effect_ratio_vs_max_null_p95", np.nan),
                "branch_placebo_pass": branch_placebo_pass,
                "placebo_ok_modes": prow.get("placebo_ok_modes", 0),
                "placebo_min_real_vs_placebo_ratio": prow.get("placebo_min_real_vs_placebo_ratio", np.nan),
                "placebo_max_real_beats_fraction": prow.get("placebo_max_real_beats_fraction", np.nan),
                "embryo_module_gates": erow.get("embryo_module_gates", 0),
                "embryo_module_total": erow.get("embryo_module_total", 0),
                "embryo_module_min_ci_low": erow.get("embryo_module_min_ci_low", np.nan),
                "best_embryo_module_query": erow.get("best_embryo_module_query", ""),
                "best_embryo_module_directed_diff": erow.get("best_embryo_module_directed_diff", np.nan),
                "best_embryo_module_p": erow.get("best_embryo_module_p", np.nan),
                "expression_constraint_candidate": expression_candidate,
                "recommended_use": recommended_use,
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    fields = list(rows[0].keys()) if rows else []
    write_csv(OUT_ROWS, rows, fields)
    temporal = pd.read_csv(TEMPORAL_CONTROLS)
    temporal.to_csv(OUT_TEMPORAL, index=False)
    latent_ready = read_json(LATENT_READY_JSON)
    latent_proxy = read_json(LATENT_PROXY_JSON)
    direct_assets = latent_ready.get("summary", {}).get("direct_compatible_assets", [])
    latent_blocked = not direct_assets
    proxy_failed = latent_proxy.get("status") == "zscape_latent_proxy_reconciliation_fail_no_gpu"
    candidates = [r for r in rows if r["expression_constraint_candidate"]]
    state_preserved_candidates = [r for r in candidates if r["state_preserved_by_threshold"]]
    status = "zscape_snapshot_dynamic_constraint_spec_expression_ready_latent_blocked_no_gpu"
    if len(candidates) < 2:
        status = "zscape_snapshot_dynamic_constraint_spec_partial_no_gpu"
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "expression_constraint_candidates": len(candidates),
        "state_preserved_expression_candidates": len(state_preserved_candidates),
        "candidate_rows": [r["row_id"] for r in candidates],
        "latent_blocked": latent_blocked,
        "latent_proxy_failed": proxy_failed,
        "direct_compatible_assets": direct_assets,
        "rows_csv": str(OUT_ROWS),
        "temporal_controls_csv": str(OUT_TEMPORAL),
        "next_action": (
            "keep as CPU expression-space constraint spec; only move to latent/model code gate "
            "after species-safe encoder or frozen orthology-loss audit"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Snapshot Dynamic Constraint Spec",
        "",
        f"Timestamp: `{payload['created_at']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis over frozen ZSCAPE expression-space gates.",
        "- ZSCAPE is treated as discrete-time snapshot perturbation evidence, not single-cell lineage pairing.",
        "- No model training, inference, true scFM embedding extraction, canonical multi, Track C query, or checkpoint selection.",
        "",
        "## Candidate Summary",
        "",
        f"- expression constraint candidates: `{len(candidates)}`.",
        f"- candidates meeting strict state-preservation threshold: `{len(state_preserved_candidates)}`.",
        f"- candidate rows: `{', '.join(payload['candidate_rows']) if candidates else 'none'}`.",
        f"- direct Danio/metazoa-compatible latent assets: `{direct_assets}`.",
        f"- latent proxy reconciliation failed: `{proxy_failed}`.",
        "",
        "## Candidate Rows",
        "",
        "| row | strict | trajectory margin | fixed-cell | embryo modules | state preserved | recommended use |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["recommended_use"] == "diagnostic_only":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(row["strict_row_gate"]),
                    fmt_float(row["trajectory_margin"]),
                    str(row["fixed_cell_gate"]),
                    f"{row['embryo_module_gates']}/{row['embryo_module_total']}",
                    str(row["state_preserved_by_threshold"]),
                    str(row["recommended_use"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Constraint Template",
            "",
            "If a future species-safe latent route exists, the expression candidates define only a bounded design template:",
            "",
            "- temporal tangent: align predicted perturbation displacement with the periderm `24h->36h` control snapshot tangent for supported rows;",
            "- wrong-lineage/time contrast: penalize alignment to wrong-lineage or wrong-time tangents;",
            "- state preservation: retain subtype/library balance constraints as no-harm diagnostics, not as training labels;",
            "- embryo uncertainty: report embryo-replicate module support and do not treat cell-level bootstrap as independent biological replication;",
            "- negative controls: keep mature-fast-muscle rows as strong-but-confounded comparators because strict controls failed.",
            "",
            "## Decision",
            "",
            "This spec is expression-ready but latent-blocked. It can guide a future code/design gate, but cannot launch ZSCAPE-derived LatentFM training now.",
            "",
            "## Outputs",
            "",
            f"- rows: `{OUT_ROWS}`",
            f"- temporal controls: `{OUT_TEMPORAL}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
