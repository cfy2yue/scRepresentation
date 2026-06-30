#!/usr/bin/env python3
"""State/context support readiness gate for downstream information scaling."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
GT_STACK = ROOT / "dataset/biFlow_data/gt_stack"
CONTROL_GEOMETRY_JSON = ROOT / "reports/latentfm_control_state_support_geometry_v2_gate_20260627.json"
CONTROL_SUPPORT_JSON = ROOT / "reports/latentfm_control_state_support_gate_20260624.json"
SOURCE_BG_JSON = ROOT / "reports/latentfm_source_background_type_hierarchical_matched_gate_20260626.json"
DOWNSTREAM_X = ROOT / "reports/downstream_information_scaling_x_gate_20260628/downstream_information_scaling_x_readiness_rows.csv"
OUT_DIR = ROOT / "reports/state_context_support_scaling_gate_20260628"


STATE_KEYWORDS = (
    "cell_type",
    "celltype",
    "cell_line",
    "cellline",
    "lineage",
    "tissue",
    "organ",
    "subtype",
)
CONTEXT_KEYWORDS = (
    "batch",
    "sample",
    "donor",
    "patient",
    "condition",
    "background",
    "cytokine",
    "treatment",
    "perturbation_type",
)


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def obs_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as h:
        if "obs" not in h:
            return []
        return sorted(list(h["obs"].keys()))


def key_groups(keys: list[str]) -> tuple[list[str], list[str]]:
    low = {k: k.lower() for k in keys}
    state = [k for k, v in low.items() if any(tok in v for tok in STATE_KEYWORDS)]
    context = [k for k, v in low.items() if any(tok in v for tok in CONTEXT_KEYWORDS)]
    return state, context


def inventory_gt_stack() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(GT_STACK.glob("*.h5ad")):
        keys = obs_keys(path)
        state, context = key_groups(keys)
        rows.append(
            {
                "dataset": path.stem,
                "n_obs_keys": len(keys),
                "state_keys": ";".join(state),
                "context_keys": ";".join(context),
                "has_state_key": bool(state),
                "has_context_key": bool(context),
                "all_obs_keys": ";".join(keys),
            }
        )
    return pd.DataFrame(rows)


def build_readiness(
    inventory: pd.DataFrame,
    geom: dict[str, Any],
    support: dict[str, Any],
    source_bg: dict[str, Any],
    downstream: pd.DataFrame,
) -> pd.DataFrame:
    state_row = {}
    if not downstream.empty:
        sub = downstream[downstream["x_family"] == "state_context_support"]
        if not sub.empty:
            state_row = sub.iloc[0].to_dict()

    support_decision = support.get("decision", {})
    run_decisions = support_decision.get("run_decisions", [])
    support_summaries = []
    for rd in run_decisions:
        support_summaries.append(
            f"{rd.get('run')}:cross={fmt(rd.get('cross_mean_pp_delta'))},family={fmt(rd.get('family_mean_pp_delta'))},passed={rd.get('passed')}"
        )

    rows = [
        {
            "component": "metadata_materializability",
            "status": "pass_preflight" if int(inventory["has_state_key"].sum()) > 0 else "fail_no_state_keys",
            "evidence": f"gt_stack datasets={len(inventory)}, state-key datasets={int(inventory['has_state_key'].sum())}, context-key datasets={int(inventory['has_context_key'].sum())}",
            "gpu_authorized": False,
            "next_gate": "materialize condition-level train-safe state/context support from obs fields",
        },
        {
            "component": "control_state_geometry_v2",
            "status": geom.get("status", "missing"),
            "evidence": (
                f"best={geom.get('best_feature', {}).get('feature')}; "
                f"rho_bad_pp={fmt(geom.get('best_feature', {}).get('rho_bad_pp'))}; "
                f"shuffle_p={fmt(geom.get('best_feature', {}).get('shuffle_p_abs'))}; "
                f"joined_datasets={geom.get('n_joined_datasets')}"
            ),
            "gpu_authorized": bool(geom.get("gpu_authorized", False)),
            "next_gate": "use as dataset-level covariate only; direct control geometry axis failed shuffle/LODO",
        },
        {
            "component": "nested_control_state_support",
            "status": "control_state_support_gate_fail_no_gpu",
            "evidence": "; ".join(support_summaries)[:900],
            "gpu_authorized": bool(support_decision.get("gpu_authorized", False)),
            "next_gate": "avoid replaying old support-only GPU route; extract interpretable state/context covariates",
        },
        {
            "component": "source_background_type_hierarchical",
            "status": source_bg.get("status", "missing"),
            "evidence": (
                f"pp_delta_mean={fmt(source_bg.get('summary', {}).get('pp_delta_mean'))}; "
                f"pp_ci={source_bg.get('summary', {}).get('pp_bootstrap_ci95')}; "
                f"tails={source_bg.get('summary', {}).get('negative_tails_lt_minus_0p02')}"
            ),
            "gpu_authorized": bool(source_bg.get("gpu_authorized", False)),
            "next_gate": "background/type are failure-localization covariates, not launch axes",
        },
        {
            "component": "downstream_information_x_family",
            "status": state_row.get("evidence_status", "missing"),
            "evidence": state_row.get("current_support", "state/context support not yet materialized in downstream x matrix"),
            "gpu_authorized": False,
            "next_gate": state_row.get(
                "next_cpu_gate",
                "condition-level state entropy/effective cluster count matrix with source-family LODO",
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_report(out_dir: Path, inventory: pd.DataFrame, readiness: pd.DataFrame) -> None:
    gpu = bool(readiness["gpu_authorized"].any()) if not readiness.empty else False
    lines: list[str] = []
    lines.append("# LatentFM State/Context Support Scaling Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append("Status: `state_context_support_scaling_gate_covariate_only_no_gpu`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/report-only synthesis plus h5ad obs-key metadata inventory.")
    lines.append("- Does not load expression matrices, train, infer, select checkpoints, use canonical multi, or use Track C query.")
    lines.append("")
    lines.append("## Metadata Inventory")
    lines.append("")
    lines.append(f"- gt_stack datasets: `{len(inventory)}`.")
    lines.append(f"- datasets with state-like obs keys: `{int(inventory['has_state_key'].sum())}`.")
    lines.append(f"- datasets with context-like obs keys: `{int(inventory['has_context_key'].sum())}`.")
    lines.append("")
    lines.append("| dataset | state keys | context keys |")
    lines.append("|---|---|---|")
    for _, row in inventory.iterrows():
        lines.append(f"| {row['dataset']} | {row['state_keys']} | {row['context_keys']} |")
    lines.append("")
    lines.append("## Readiness")
    lines.append("")
    lines.append("| component | status | evidence | next gate |")
    lines.append("|---|---|---|---|")
    for _, row in readiness.iterrows():
        lines.append(
            f"| {row['component']} | {row['status']} | {str(row['evidence']).replace('|', '/')} | {row['next_gate']} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- State/context support remains biologically plausible and materializable from existing metadata.")
    lines.append("- It is not GPU-launch-ready because prior support/geometry/background gates failed robustness or tail criteria.")
    lines.append("- The next useful action is a train-safe condition-level matrix with state entropy/effective-count, background/context fields, source-family LODO, and dual-baseline no-harm.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- metadata inventory: `{out_dir / 'state_context_metadata_inventory.csv'}`")
    lines.append(f"- readiness rows: `{out_dir / 'state_context_support_readiness_rows.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'state_context_support_scaling_gate_20260628.json'}`")
    (out_dir / "LATENTFM_STATE_CONTEXT_SUPPORT_SCALING_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = inventory_gt_stack()
    geom = read_json(CONTROL_GEOMETRY_JSON)
    support = read_json(CONTROL_SUPPORT_JSON)
    source_bg = read_json(SOURCE_BG_JSON)
    downstream = pd.read_csv(DOWNSTREAM_X) if DOWNSTREAM_X.exists() else pd.DataFrame()
    readiness = build_readiness(inventory, geom, support, source_bg, downstream)

    inv_path = OUT_DIR / "state_context_metadata_inventory.csv"
    rows_path = OUT_DIR / "state_context_support_readiness_rows.csv"
    inventory.to_csv(inv_path, index=False)
    readiness.to_csv(rows_path, index=False)
    obj = {
        "timestamp": now_cst(),
        "status": "state_context_support_scaling_gate_covariate_only_no_gpu",
        "gpu_authorized_next": bool(readiness["gpu_authorized"].any()) if not readiness.empty else False,
        "metadata": {
            "n_datasets": int(len(inventory)),
            "state_key_datasets": int(inventory["has_state_key"].sum()) if not inventory.empty else 0,
            "context_key_datasets": int(inventory["has_context_key"].sum()) if not inventory.empty else 0,
        },
        "outputs": {
            "inventory": str(inv_path),
            "readiness": str(rows_path),
            "report": str(OUT_DIR / "LATENTFM_STATE_CONTEXT_SUPPORT_SCALING_GATE_20260628.md"),
        },
    }
    write_json(OUT_DIR / "state_context_support_scaling_gate_20260628.json", obj)
    write_report(OUT_DIR, inventory, readiness)


if __name__ == "__main__":
    main()
