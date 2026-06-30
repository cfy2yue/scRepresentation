#!/usr/bin/env python3
"""Materialize train/test condition-level state/context support metadata.

CPU-only metadata pass. Reads h5ad ``obs`` columns and canonical split JSON,
but never loads expression matrices or changes splits.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
GT_STACK = ROOT / "dataset/biFlow_data/gt_stack"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
INVENTORY = ROOT / "reports/state_context_support_scaling_gate_20260628/state_context_metadata_inventory.csv"
OUT_DIR = ROOT / "reports/state_context_condition_matrix_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


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


def decode_array(arr: Any) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind in {"S", "O"}:
        return np.asarray([x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in out])
    return out.astype(str)


def read_obs_column(handle: h5py.File, key: str) -> np.ndarray | None:
    if "obs" not in handle or key not in handle["obs"]:
        return None
    obs = handle["obs"]
    node = obs[key]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"][:], dtype=int)
        cats = decode_array(node["categories"][:])
        valid = (codes >= 0) & (codes < len(cats))
        out = np.full(codes.shape, "", dtype=object)
        out[valid] = cats[codes[valid]]
        return out.astype(str)
    if "__categories" in obs and key in obs["__categories"]:
        codes = np.asarray(node[:], dtype=int)
        cats = decode_array(obs["__categories"][key][:])
        valid = (codes >= 0) & (codes < len(cats))
        out = np.full(codes.shape, "", dtype=object)
        out[valid] = cats[codes[valid]]
        return out.astype(str)
    if isinstance(node, h5py.Dataset):
        try:
            return node.asstr()[:].astype(str)
        except TypeError:
            return decode_array(node[:])
    return None


def entropy(values: pd.Series) -> float:
    vc = values.dropna().astype(str)
    vc = vc[vc != ""]
    if vc.empty:
        return 0.0
    counts = np.asarray(list(Counter(vc).values()), dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def split_field(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [x for x in str(value).split(";") if x]


def condition_matrix_for_dataset(
    dataset: str,
    split: dict[str, Any],
    state_keys: list[str],
    context_keys: list[str],
) -> list[dict[str, Any]]:
    path = GT_STACK / f"{dataset}.h5ad"
    if not path.exists():
        return []
    roles = {
        "train": split.get("train", []),
        "test_single": split.get("test_single", []),
        "test_multi": split.get("test_multi", []),
    }
    needed_conds = {str(c) for conds in roles.values() for c in conds}
    with h5py.File(path, "r") as h:
        pert = read_obs_column(h, "perturbation")
        if pert is None:
            return []
        data: dict[str, np.ndarray] = {"perturbation": pert}
        for key in sorted(set(state_keys + context_keys)):
            col = read_obs_column(h, key)
            if col is not None and len(col) == len(pert):
                data[key] = col
    obs = pd.DataFrame(data)
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs = obs[obs["perturbation"].isin(needed_conds)].copy()
    stats: dict[str, dict[str, Any]] = {}
    for cond, sub in obs.groupby("perturbation", sort=False):
        state_uniques: list[int] = []
        state_entropies: list[float] = []
        context_uniques: list[int] = []
        for key in state_keys:
            if key in sub.columns:
                vals = sub[key].dropna().astype(str)
                vals = vals[vals != ""]
                state_uniques.append(int(vals.nunique()))
                state_entropies.append(entropy(vals))
        for key in context_keys:
            if key in sub.columns:
                vals = sub[key].dropna().astype(str)
                vals = vals[vals != ""]
                context_uniques.append(int(vals.nunique()))
        stats[str(cond)] = {
            "n_gt_cells": int(len(sub)),
            "max_state_unique": int(max(state_uniques) if state_uniques else 0),
            "max_state_entropy": float(max(state_entropies) if state_entropies else 0.0),
            "max_context_unique": int(max(context_uniques) if context_uniques else 0),
            "has_state_context_signal": bool(state_uniques or context_uniques),
        }
    rows: list[dict[str, Any]] = []
    for role, conds in roles.items():
        for cond in conds:
            st = stats.get(
                str(cond),
                {
                    "n_gt_cells": 0,
                    "max_state_unique": 0,
                    "max_state_entropy": 0.0,
                    "max_context_unique": 0,
                    "has_state_context_signal": False,
                },
            )
            rows.append(
                {
                    "dataset": dataset,
                    "condition": cond,
                    "split_role": role,
                    "n_gt_cells": st["n_gt_cells"],
                    "n_state_keys": int(len(state_keys)),
                    "n_context_keys": int(len(context_keys)),
                    "max_state_unique": st["max_state_unique"],
                    "max_state_entropy": st["max_state_entropy"],
                    "max_context_unique": st["max_context_unique"],
                    "has_state_context_signal": st["has_state_context_signal"],
                }
            )
    return rows


def write_report(out_dir: Path, matrix: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    train = matrix[matrix["split_role"] == "train"] if not matrix.empty else pd.DataFrame()
    train_state = int((train["max_state_unique"] > 0).sum()) if not train.empty else 0
    train_context = int((train["max_context_unique"] > 0).sum()) if not train.empty else 0
    gpu = False
    status = "state_context_condition_matrix_materialized_no_gpu"
    lines: list[str] = []
    lines.append("# LatentFM State/Context Condition Matrix")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append(f"Status: `{status}`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU-only h5ad obs metadata pass over canonical `split_seed42.json`.")
    lines.append("- Does not read expression `X`, train, infer, select checkpoints, use canonical multi for selection, or use Track C query.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Condition rows: `{len(matrix)}`.")
    lines.append(f"- Train condition rows: `{len(train)}`.")
    lines.append(f"- Train rows with state-like signal: `{train_state}`.")
    lines.append(f"- Train rows with context-like signal: `{train_context}`.")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("| dataset | rows | train rows | state train rows | context train rows | max state entropy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['rows'])} | {int(row['train_rows'])} | "
            f"{int(row['state_train_rows'])} | {int(row['context_train_rows'])} | {fmt(row['max_state_entropy'])} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- The condition-level state/context matrix is now materialized and can be joined to downstream scaling outcomes.")
    lines.append("- State-like metadata is sparse and concentrated in Jiang/sciplex datasets; context-like perturbation-type metadata is broad.")
    lines.append("- This does not authorize GPU by itself; next gate is association/LODO against frozen outcomes with source/background controls.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- matrix: `{out_dir / 'state_context_condition_matrix.csv'}`")
    lines.append(f"- dataset summary: `{out_dir / 'state_context_dataset_summary.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'state_context_condition_matrix_20260628.json'}`")
    (out_dir / "LATENTFM_STATE_CONTEXT_CONDITION_MATRIX_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    inventory = pd.read_csv(INVENTORY)
    inv_map = inventory.set_index("dataset")
    rows: list[dict[str, Any]] = []
    for dataset, ds_split in split.items():
        if dataset not in inv_map.index:
            continue
        inv = inv_map.loc[dataset]
        state_keys = split_field(inv.get("state_keys"))
        context_keys = split_field(inv.get("context_keys"))
        rows.extend(condition_matrix_for_dataset(dataset, ds_split, state_keys, context_keys))
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        dataset_summary = pd.DataFrame()
    else:
        summaries: list[dict[str, Any]] = []
        for dataset, grp in matrix.groupby("dataset", dropna=False):
            train = grp[grp["split_role"] == "train"]
            summaries.append(
                {
                    "dataset": dataset,
                    "rows": int(len(grp)),
                    "train_rows": int(len(train)),
                    "state_train_rows": int((train["max_state_unique"] > 0).sum()),
                    "context_train_rows": int((train["max_context_unique"] > 0).sum()),
                    "max_state_entropy": float(train["max_state_entropy"].max()) if not train.empty else 0.0,
                    "median_train_cells": float(train["n_gt_cells"].median()) if not train.empty else 0.0,
                }
            )
        dataset_summary = pd.DataFrame(summaries)

    matrix_path = OUT_DIR / "state_context_condition_matrix.csv"
    summary_path = OUT_DIR / "state_context_dataset_summary.csv"
    matrix.to_csv(matrix_path, index=False)
    dataset_summary.to_csv(summary_path, index=False)
    obj = {
        "timestamp": now_cst(),
        "status": "state_context_condition_matrix_materialized_no_gpu",
        "gpu_authorized_next": False,
        "n_condition_rows": int(len(matrix)),
        "n_train_rows": int((matrix["split_role"] == "train").sum()) if not matrix.empty else 0,
        "train_rows_with_state_signal": int(((matrix["split_role"] == "train") & (matrix["max_state_unique"] > 0)).sum()) if not matrix.empty else 0,
        "train_rows_with_context_signal": int(((matrix["split_role"] == "train") & (matrix["max_context_unique"] > 0)).sum()) if not matrix.empty else 0,
        "outputs": {
            "matrix": str(matrix_path),
            "dataset_summary": str(summary_path),
            "report": str(OUT_DIR / "LATENTFM_STATE_CONTEXT_CONDITION_MATRIX_20260628.md"),
        },
    }
    write_json(OUT_DIR / "state_context_condition_matrix_20260628.json", obj)
    write_report(OUT_DIR, matrix, dataset_summary)


if __name__ == "__main__":
    main()
