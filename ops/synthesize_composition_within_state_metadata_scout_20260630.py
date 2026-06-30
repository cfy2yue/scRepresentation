#!/usr/bin/env python3
"""Metadata scout for composition-vs-within-state LatentFM axes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
RAW_PATHS = [
    ROOT / "dataset/raw/genepert_DE5000/TianActivation.h5ad",
    ROOT / "dataset/raw/genepert_DE5000/GasperiniShendure2019_lowMOI.h5ad",
    ROOT / "dataset/raw/genepert_DE5000/Papalexi.h5ad",
    ROOT / "dataset/raw/genepert_DE5000/TianInhibition.h5ad",
    ROOT / "dataset/raw/genepert_DE5000/Frangieh.h5ad",
    ROOT / "dataset/raw/chemicalpert_bench/sciplex3_A549.h5ad",
    ROOT / "dataset/raw/chemicalpert_bench/sciplex3_K562.h5ad",
    ROOT / "dataset/raw/chemicalpert_bench/sciplex3_MCF7.h5ad",
]
OUT_DIR = ROOT / "reports/composition_within_state_metadata_scout_20260630"
OUT_JSON = OUT_DIR / "composition_within_state_metadata_scout_20260630.json"
OUT_CSV = OUT_DIR / "composition_within_state_metadata_scout_20260630.csv"
OUT_MD = OUT_DIR / "LATENTFM_COMPOSITION_WITHIN_STATE_METADATA_SCOUT_20260630.md"
KEYWORDS = ("cell", "type", "cluster", "state", "time", "dose", "lineage", "pert", "condition", "batch")


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def scan(path: Path) -> dict[str, Any]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs_cols = [str(c) for c in adata.obs.columns]
        interesting = [c for c in obs_cols if any(tok in c.lower() for tok in KEYWORDS)]
        col_unique: dict[str, int] = {}
        top_values: dict[str, dict[str, int]] = {}
        for col in interesting:
            vals = adata.obs[col].astype(str)
            col_unique[col] = int(vals.nunique(dropna=True))
            top_values[col] = {clean(k): int(v) for k, v in vals.value_counts().head(5).items()}
        state_like = [
            c
            for c in interesting
            if any(tok in c.lower() for tok in ("cell_type", "celltype", "cluster", "state", "lineage"))
            and col_unique.get(c, 0) > 1
        ]
        composition_ready = bool(state_like)
        return {
            "path": str(path),
            "dataset": path.stem,
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "obs_columns": obs_cols,
            "interesting_columns": interesting,
            "interesting_unique_counts": col_unique,
            "top_values": top_values,
            "state_like_variable_columns": state_like,
            "composition_within_state_ready": composition_ready,
        }
    finally:
        adata.file.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [scan(path) for path in RAW_PATHS if path.is_file()]
    df = pd.DataFrame(
        {
            "dataset": row["dataset"],
            "path": row["path"],
            "n_obs": row["shape"][0],
            "n_vars": row["shape"][1],
            "interesting_columns": ";".join(row["interesting_columns"]),
            "state_like_variable_columns": ";".join(row["state_like_variable_columns"]),
            "composition_within_state_ready": row["composition_within_state_ready"],
        }
        for row in rows
    )
    ready_count = int(df["composition_within_state_ready"].sum()) if len(df) else 0
    reasons: list[str] = []
    if ready_count < 3:
        reasons.append("variable_state_like_columns_in_lt_3_raw_datasets")
    if ready_count == 0:
        reasons.append("no_raw_dataset_has_within_dataset_celltype_cluster_state_variation")
    status = "composition_within_state_metadata_scout_fail_no_gpu" if reasons else "composition_within_state_metadata_scout_ready_for_cpu_materializer"
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "datasets_scanned": len(rows),
        "composition_ready_datasets": ready_count,
        "reasons": reasons,
        "rows": rows,
        "outputs": {"csv": str(OUT_CSV), "report": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    df.to_csv(OUT_CSV, index=False)

    lines = [
        "# LatentFM Composition-vs-Within-State Metadata Scout 20260630",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only metadata scout.",
        "- Checks whether local raw h5ad files expose variable within-dataset cell type/cluster/state labels needed for composition-vs-within-state x.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | cells | genes | variable state-like columns | ready |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | {row['shape'][0]} | {row['shape'][1]} | "
            f"`{';'.join(row['state_like_variable_columns'])}` | `{row['composition_within_state_ready']}` |"
        )
    lines.extend(["", "## Decision", ""])
    if reasons:
        lines.append("Current raw metadata does not support a composition-vs-within-state materializer.")
        lines.extend(f"- reason: `{reason}`" for reason in reasons)
        lines.append("- SciPlex files have dose/cell-line fields, but cell type and time are constant within each file; genepert files expose perturbation/QC fields only.")
    else:
        lines.append("Metadata readiness passed; next step would be CPU materialization.")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- CSV: `{OUT_CSV}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
