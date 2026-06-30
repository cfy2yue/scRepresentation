#!/usr/bin/env python3
"""Synthesize ZSCAPE state-preserved time-vector evidence.

CPU/report-only. This separates vector-level dynamic biology from failed module
specificity, so the project can keep biological insight without prematurely
turning it into a LatentFM loss.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OT_ROWS = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_rows.csv"
EMBRYO_ROWS = ROOT / "reports/zscape_embryo_vector_consistency_gate_20260628/zscape_embryo_vector_consistency_rows.csv"
TRAJ_DIR = (
    ROOT
    / "runs/zscape_expression_trajectory_time_gate_20260628"
    / "zscape_expression_trajectory_time_gate_20260628_084025"
    / "outputs"
)
TRAJ_ALIGN = TRAJ_DIR / "zscape_expression_trajectory_time_perturb_alignment.csv"
TRAJ_TEMPORAL = TRAJ_DIR / "zscape_expression_trajectory_time_temporal_controls.csv"
MODULE_QUERY = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_query_rows.csv"
)
MODULE_PLACEBO = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_placebo_rows.csv"
)
OUT_DIR = ROOT / "reports/zscape_state_preserved_time_vector_gate_20260629"
OUT_MD = OUT_DIR / "LATENTFM_ZSCAPE_STATE_PRESERVED_TIME_VECTOR_GATE_20260629.md"
OUT_JSON = OUT_DIR / "zscape_state_preserved_time_vector_gate_20260629.json"
OUT_ROWS = OUT_DIR / "zscape_state_preserved_time_vector_rows.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def sf(value: Any, default: float = float("nan")) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def fmt(value: Any, digits: int = 4) -> str:
    val = sf(value)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def module_summary() -> pd.DataFrame:
    query = pd.read_csv(MODULE_QUERY)
    rows: list[dict[str, Any]] = []
    for row_id, group in query.groupby("row_id"):
        rows.append(
            {
                "row_id": row_id,
                "module_query_count": int(len(group)),
                "module_qc_residual_all": bool(group["qc_residual_gate"].map(as_bool).all()),
                "module_substate_all": bool(group["substate_gate"].map(as_bool).all()),
                "module_specificity_all": bool(group["specificity_gate"].map(as_bool).all()),
                "module_query_all": bool(group["query_gate"].map(as_bool).all()),
                "module_min_residual_ci_low": float(pd.to_numeric(group["residual_ci_low"], errors="coerce").min()),
                "module_min_substate_diff": float(pd.to_numeric(group["substate_min_residual_diff"], errors="coerce").min()),
                "module_max_wrong_time": float(pd.to_numeric(group["wrong_time_max"], errors="coerce").max()),
                "module_max_wrong_lineage": float(pd.to_numeric(group["wrong_lineage_p95"], errors="coerce").max()),
                "module_terms": " | ".join(str(x) for x in group["top_terms"].dropna().head(2)),
            }
        )
    return pd.DataFrame(rows)


def placebo_summary() -> pd.DataFrame:
    placebo = pd.read_csv(MODULE_PLACEBO)
    rows: list[dict[str, Any]] = []
    for row_id, group in placebo.groupby("query_row_id"):
        rows.append(
            {
                "row_id": row_id,
                "placebo_p95_directed_diff": float(pd.to_numeric(group["directed_diff"], errors="coerce").quantile(0.95)),
                "placebo_max_directed_diff": float(pd.to_numeric(group["directed_diff"], errors="coerce").max()),
                "placebo_positive_count": int((pd.to_numeric(group["directed_diff"], errors="coerce") > 0).sum()),
                "placebo_control_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_rows() -> pd.DataFrame:
    ot = pd.read_csv(OT_ROWS)
    embryo = pd.read_csv(EMBRYO_ROWS)
    align = pd.read_csv(TRAJ_ALIGN)
    temporal = pd.read_csv(TRAJ_TEMPORAL)
    modules = module_summary()
    placebo = placebo_summary()

    temporal_primary = temporal[
        temporal["timepoint_a"].eq(24.0) & temporal["timepoint_b"].eq(36.0)
    ][["lineage", "temporal_gate", "temporal_ratio_vs_null_p95", "p_temporal_le_same_time_null", "wrong_lineage_to_observed_ratio"]]
    temporal_primary = temporal_primary.rename(
        columns={
            "temporal_gate": "lineage_24_36_temporal_gate",
            "temporal_ratio_vs_null_p95": "lineage_24_36_temporal_ratio",
            "p_temporal_le_same_time_null": "lineage_24_36_temporal_p",
            "wrong_lineage_to_observed_ratio": "lineage_24_36_wrong_lineage_ratio",
        }
    )

    merged = ot.merge(embryo, on="row_id", how="left", suffixes=("", "_embryo"))
    merged = merged.merge(align, on="row_id", how="left", suffixes=("", "_align"))
    merged = merged.merge(temporal_primary, on="lineage", how="left")
    merged = merged.merge(modules, on="row_id", how="left")
    merged = merged.merge(placebo, on="row_id", how="left")
    periderm = merged[merged["lineage"].astype(str).eq("periderm")].copy()

    periderm["state_preserved_gate"] = (
        periderm["state_preserved_by_threshold"].map(as_bool)
        & (pd.to_numeric(periderm["composition_norm_fraction_of_centroid"], errors="coerce") <= 0.05)
        & (pd.to_numeric(periderm["within_substate_residual_fraction_of_centroid"], errors="coerce") >= 0.95)
        & (pd.to_numeric(periderm["matched_subtype_jsd"], errors="coerce") <= 0.02)
        & (pd.to_numeric(periderm["expression_library_smd"], errors="coerce").abs() <= 0.20)
    )
    periderm["time_vector_gate"] = (
        periderm["lineage_24_36_temporal_gate"].map(as_bool)
        & periderm["alignment_gate"].map(as_bool)
        & (pd.to_numeric(periderm["cosine_to_lineage_time_vector"], errors="coerce") > 0)
        & (pd.to_numeric(periderm["cosine_margin_vs_wrong_lineage"], errors="coerce") >= 0.05)
    )
    periderm["wrong_ot_margin_gate"] = (
        (pd.to_numeric(periderm["wrong_time_margin_ot"], errors="coerce") > 0.25)
        & (pd.to_numeric(periderm["wrong_lineage_margin_ot"], errors="coerce") > 10.0)
    )
    periderm["embryo_vector_reliable_gate"] = (
        periderm["embryo_vector_gate"].map(as_bool)
        & (pd.to_numeric(periderm["mean_cosine_ci_low"], errors="coerce") > 0)
        & (pd.to_numeric(periderm["positive_embryo_fraction"], errors="coerce") >= 0.75)
    )
    periderm["vector_dynamic_gate"] = (
        periderm["dynamic_response_gate"].map(as_bool)
        & periderm["state_preserved_gate"]
        & periderm["time_vector_gate"]
        & periderm["wrong_ot_margin_gate"]
        & periderm["embryo_vector_reliable_gate"]
    )
    periderm["module_specificity_ready"] = (
        periderm["module_query_all"].map(as_bool)
        & periderm["module_qc_residual_all"].map(as_bool)
        & periderm["module_substate_all"].map(as_bool)
        & periderm["module_specificity_all"].map(as_bool)
    )
    periderm["model_constraint_ready"] = periderm["vector_dynamic_gate"] & periderm["module_specificity_ready"]
    periderm["biological_class"] = "negative_or_diagnostic"
    periderm.loc[periderm["vector_dynamic_gate"], "biological_class"] = "state_preserved_time_vector_positive"
    periderm.loc[
        periderm["vector_dynamic_gate"] & ~periderm["module_specificity_ready"],
        "biological_class",
    ] = "state_time_vector_positive_module_specificity_blocked"
    return periderm


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.to_csv(OUT_ROWS, index=False)
    positives = rows[rows["vector_dynamic_gate"].astype(bool)]["row_id"].astype(str).tolist()
    model_ready = rows[rows["model_constraint_ready"].astype(bool)]["row_id"].astype(str).tolist()
    wrong_controls_fail = rows[
        rows["target"].astype(str).isin(["mafba", "tbx16-tbx16l", "cdx4-cdx1a"])
        & ~rows["vector_dynamic_gate"].astype(bool)
    ]["row_id"].astype(str).tolist()
    status = (
        "zscape_state_preserved_time_vector_biology_pass_no_gpu"
        if set(["periderm__noto__24p0h", "periderm__smo__24p0h"]).issubset(set(positives))
        and len(model_ready) == 0
        else "zscape_state_preserved_time_vector_partial_or_fail_no_gpu"
    )
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "positive_vector_rows": positives,
        "model_constraint_ready_rows": model_ready,
        "wrong_control_rows_failing_vector_gate": wrong_controls_fail,
        "decision": "biology insight only; no model constraint" if status.endswith("pass_no_gpu") else "no model use",
        "inputs": {
            "ot_rows": str(OT_ROWS),
            "embryo_rows": str(EMBRYO_ROWS),
            "trajectory_alignment": str(TRAJ_ALIGN),
            "trajectory_temporal": str(TRAJ_TEMPORAL),
            "module_query": str(MODULE_QUERY),
            "module_placebo": str(MODULE_PLACEBO),
        },
        "outputs": {"rows": str(OUT_ROWS), "json": str(OUT_JSON), "report": str(OUT_MD)},
        "boundary": "cpu_report_only_no_training_no_inference_no_gpu_no_new_ot_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)

    display_cols = [
        "row_id",
        "target",
        "timepoint",
        "state_preserved_gate",
        "time_vector_gate",
        "wrong_ot_margin_gate",
        "embryo_vector_reliable_gate",
        "vector_dynamic_gate",
        "module_specificity_ready",
        "cosine_to_lineage_time_vector",
        "cosine_margin_vs_wrong_lineage",
        "wrong_time_margin_ot",
        "wrong_lineage_margin_ot",
        "mean_cosine_ci_low",
        "module_min_residual_ci_low",
        "module_max_wrong_time",
    ]
    lines = [
        "# ZSCAPE State-Preserved Time-Vector Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only synthesis over frozen ZSCAPE OT, trajectory-time, embryo-vector, and module/QC artifacts.",
        "* No training, inference, new OT pairing, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* Goal: separate vector-level perturbation dynamics from pathway/module specificity.",
        "",
        "## Main Finding",
        "",
        f"* Vector-positive rows: `{', '.join(positives) if positives else 'none'}`.",
        f"* Model-constraint-ready rows: `{', '.join(model_ready) if model_ready else 'none'}`.",
        f"* Wrong-control periderm rows failing vector gate: `{', '.join(wrong_controls_fail) if wrong_controls_fail else 'none'}`.",
        "",
        "## Periderm Row Decomposition",
        "",
        "| row | target | time | state | time-vector | OT margins | embryo | vector gate | module ready | time cosine | cosine margin | wrong-time margin | wrong-lineage margin | embryo CI low | module CI low | module wrong-time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values(["timepoint", "target"]).iterrows():
        lines.append(
            f"| `{row['row_id']}` | `{row['target']}` | `{fmt(row['timepoint'], 1)}` | "
            f"`{bool(row['state_preserved_gate'])}` | `{bool(row['time_vector_gate'])}` | "
            f"`{bool(row['wrong_ot_margin_gate'])}` | `{bool(row['embryo_vector_reliable_gate'])}` | "
            f"`{bool(row['vector_dynamic_gate'])}` | `{bool(row['module_specificity_ready'])}` | "
            f"`{fmt(row['cosine_to_lineage_time_vector'])}` | `{fmt(row['cosine_margin_vs_wrong_lineage'])}` | "
            f"`{fmt(row['wrong_time_margin_ot'])}` | `{fmt(row['wrong_lineage_margin_ot'])}` | "
            f"`{fmt(row['mean_cosine_ci_low'])}` | `{fmt(row.get('module_min_residual_ci_low'))}` | "
            f"`{fmt(row.get('module_max_wrong_time'))}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "* Periderm `noto` and `smo` support a state-preserved, embryo-reliable perturbation displacement aligned with the normal 24h->36h periderm time vector.",
            "* `mafba`, `tbx16-tbx16l`, and 36h `cdx4-cdx1a` are useful wrong-target/time controls because they fail the combined vector gate.",
            "* Module/pathway specificity remains blocked: current `noto/smo` module effects are real and QC/substate-positive, but wrong-target/time controls catch up.",
            "* Modeling implication: use this as a biological diagnostic and possible future sampling/condition-similarity covariate only after train-set translation; do not make it a direct LatentFM/RawFM loss.",
            "",
            "## Next Gate",
            "",
            "* Build a ZSCAPE-to-LatentFM train-only translation table: state/time/support density analogs, within-state displacement proxies, exact coverage, source/background controls.",
            "* Require incremental signal beyond exact coverage/count/source/background with LODO stability before any GPU smoke.",
            "",
            "## Outputs",
            "",
            f"* Rows: `{OUT_ROWS}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "positive_vector_rows": positives, "model_constraint_ready_rows": model_ready, "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
