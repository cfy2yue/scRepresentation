#!/usr/bin/env python3
"""CPU-only gate for condition-level background-balanced xverse splits.

This reads h5ad `.obs` metadata only.  It does not read expression matrices,
model outputs, canonical outcomes, Track C query, active logs, or GPU artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
OBS_SCHEMA_JSON = ROOT / "reports/latentfm_xverse_obs_schema_background_gate_20260624.json"
GT_STACK = ROOT / "dataset/biFlow_data/gt_stack"
OUT_JSON = ROOT / "reports/latentfm_xverse_condition_background_balance_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONDITION_BACKGROUND_BALANCE_GATE_20260624.md"

CONDITION_COLS = ("perturbation", "condition", "cov", "drug", "drug_dose_name")
BACKGROUND_COLS = ("cell_type", "cell_line", "celltype", "tissue_type", "cov")
MIXED_DATASETS = ("Jiang_IFNB", "Jiang_IFNG", "Jiang_INS", "Jiang_TGFB", "Jiang_TNFA", "sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in cols:
            return col
    return None


def inspect_dataset(ds: str, train_conditions: list[str]) -> dict[str, Any]:
    path = GT_STACK / f"{ds}.h5ad"
    if not path.is_file():
        return {"dataset": ds, "status": "missing_h5ad", "path": str(path)}
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs
        cols = [str(c) for c in obs.columns]
        cond_col = choose_col(cols, CONDITION_COLS)
        bg_col = choose_col(cols, BACKGROUND_COLS)
        if cond_col is None or bg_col is None:
            return {
                "dataset": ds,
                "status": "missing_condition_or_background_col",
                "condition_col": cond_col,
                "background_col": bg_col,
                "obs_columns": cols,
            }
        cond_series = obs[cond_col].astype(str)
        bg_series = obs[bg_col].astype(str)
        signatures: dict[str, list[str]] = {}
        missing = []
        for cond in train_conditions:
            mask = cond_series == str(cond)
            if int(mask.sum()) == 0:
                missing.append(str(cond))
                continue
            backgrounds = sorted(set(bg_series.loc[mask].tolist()))
            signatures[str(cond)] = backgrounds
        sig_counts = Counter(tuple(v) for v in signatures.values())
        singleton_background_conditions = sum(1 for vals in signatures.values() if len(vals) == 1)
        return {
            "dataset": ds,
            "status": "ok",
            "path": str(path),
            "condition_col": cond_col,
            "background_col": bg_col,
            "train_conditions": len(train_conditions),
            "mapped_train_conditions": len(signatures),
            "missing_train_conditions": len(missing),
            "unique_background_signatures": len(sig_counts),
            "singleton_background_conditions": singleton_background_conditions,
            "signature_counts": [
                {"background_signature": list(sig), "n_conditions": int(n)}
                for sig, n in sig_counts.most_common()
            ],
            "missing_examples": missing[:10],
        }
    finally:
        a.file.close()


def main() -> None:
    base = load_json(BASE_SPLIT)
    obs_schema = load_json(OBS_SCHEMA_JSON) if OBS_SCHEMA_JSON.is_file() else {}
    rows = []
    for ds in MIXED_DATASETS:
        train = [str(x) for x in (base.get(ds) or {}).get("train") or []]
        rows.append(inspect_dataset(ds, train))

    jiang_rows = [r for r in rows if str(r["dataset"]).startswith("Jiang_") and r.get("status") == "ok"]
    sciplex_rows = [r for r in rows if str(r["dataset"]).startswith("sciplex3_") and r.get("status") == "ok"]
    jiang_no_condition_level_degrees = all(
        r.get("unique_background_signatures") == 1
        and (r.get("signature_counts") or [{}])[0].get("background_signature")
        == ["A549", "BXPC3", "HAP1", "HT29", "K562", "MCF7"]
        for r in jiang_rows
    )
    sciplex_dataset_fixed = all(r.get("unique_background_signatures") == 1 for r in sciplex_rows)
    reasons = []
    if jiang_no_condition_level_degrees:
        reasons.append("jiang_conditions_share_identical_six_cell_type_signature")
    if sciplex_dataset_fixed:
        reasons.append("sciplex_background_is_dataset_level_not_condition_level")
    if not rows or any(r.get("status") != "ok" for r in rows):
        reasons.append("some_mixed_background_obs_mapping_failed")
    if jiang_no_condition_level_degrees and sciplex_dataset_fixed:
        reasons.append("no_condition_level_background_balancing_degrees_of_freedom")

    status = (
        "condition_background_balance_gate_fail_no_gpu"
        if reasons
        else "condition_background_balance_gate_pass_split_builder_next_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "none" if reasons else "cpu_split_builder_only",
        "decision_reasons": reasons,
        "boundary": {
            "obs_only": True,
            "expression_matrix_read": False,
            "model_outputs_read": False,
            "canonical_outcomes_read": False,
            "trackc_query_read": False,
            "active_logs_read": False,
            "gpu_artifacts_read": False,
        },
        "inputs": {
            "base_split": str(BASE_SPLIT),
            "obs_schema": str(OBS_SCHEMA_JSON),
            "gt_stack": str(GT_STACK),
        },
        "obs_schema_status": obs_schema.get("status"),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM xverse Condition Background Balance Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only `.obs` metadata audit.",
        "- Does not read expression matrices, model outputs, canonical outcomes, Track C query, active logs, or GPU artifacts.",
        "",
        "## Results",
        "",
        "| dataset | condition col | background col | train conditions | mapped | unique bg signatures | top signature |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        top = (row.get("signature_counts") or [{}])[0]
        sig = ",".join(top.get("background_signature") or [])
        if top.get("n_conditions") is not None:
            sig = f"{sig} ({top.get('n_conditions')})"
        lines.append(
            f"| `{row['dataset']}` | `{row.get('condition_col')}` | `{row.get('background_col')}` | "
            f"{row.get('train_conditions', 0)} | {row.get('mapped_train_conditions', 0)} | "
            f"{row.get('unique_background_signatures', 0)} | `{sig}` |"
        )
    lines.extend([
        "",
        "## Decision Reasons",
        "",
    ])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- `none`"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The proposed condition-level background-balanced split has no useful degrees of freedom in the current xverse condition representation. Jiang train conditions all carry the same six-cell-type signature, while sciplex backgrounds are already separated by dataset. Any new split would reduce to dataset-level/background-family balancing, which is already covered by failed gene/background/type/exposure branches. No GPU smoke is authorized from this route.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
