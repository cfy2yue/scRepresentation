#!/usr/bin/env python3
"""Audit downloaded external source h5ad obs fields for new artifact routes.

CPU/source-only. Reads `.obs` metadata in backed mode from externally downloaded
source h5ad files. It does not read expression matrices, checkpoints, canonical
multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_external_source_h5ad_obs_routes_20260626.json"
OUT_MD = REPORTS / "LATENTFM_EXTERNAL_SOURCE_H5AD_OBS_ROUTES_20260626.md"
OUT_CSV = REPORTS / "latentfm_external_source_h5ad_obs_routes_20260626.csv"

SOURCES = [
    {
        "dataset": "Frangieh",
        "source": "Frangieh_2021_processed_figshare",
        "path": REPORTS / "external_artifact_sources_20260626/frangieh_figshare/Frangieh_2021.h5ad",
    },
    {
        "dataset": "DixitRegev2016_K562_TFs_High_MOI",
        "source": "Dixit_2016_processed_figshare",
        "path": REPORTS / "external_artifact_sources_20260626/dixit_figshare/Dixit_2016.h5ad",
    },
]

CONSUMED_TOKENS = {
    "read_umi_guide_support_consumed": [
        "umi",
        "read",
        "guide",
        "sgrna",
        "grna",
        "sg",
        "intergenic",
        "moi",
        "coverage",
        "assignment",
        "barcode",
    ],
    "qc_support_consumed": [
        "n_genes",
        "total_counts",
        "pct_counts",
        "mt",
        "ribo",
        "counts",
    ],
    "target_actionability_consumed": ["target"],
    "source_background_type_consumed": ["cell_type", "celltype", "cell_line", "organism", "tissue", "batch", "cluster", "library"],
    "identifier_or_label_forbidden": ["condition", "perturbation", "gene", "name", "id", "index"],
}

OPEN_ROUTE_TOKENS = {
    "time_maturity_candidate": ["time", "timepoint", "hour", "day", "duration", "maturity", "recovery"],
    "viability_growth_candidate": ["viability", "fitness", "growth", "survival", "toxicity", "depletion"],
    "replicate_concordance_candidate": ["replicate", "well", "plate", "donor"],
    "dose_candidate": ["dose", "concentration"],
    "program_candidate": ["program", "state"],
}

PRIOR_CLOSED_ARTIFACT_REPORTS = [
    "reports/LATENTFM_REAGENT_READ_SUPPORT_SOURCE_BLOCK_LODO_GATE_20260626.md",
    "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md",
    "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACT_PREFLIGHT_20260626.md",
    "reports/LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md",
]


def classify_column(col: str) -> tuple[str, str]:
    low = col.lower()
    for family, tokens in CONSUMED_TOKENS.items():
        if any(tok in low for tok in tokens):
            return family, "closed_or_forbidden"
    for family, tokens in OPEN_ROUTE_TOKENS.items():
        if any(tok in low for tok in tokens):
            return family, "candidate_needs_materialization"
    return "unclassified_metadata", "not_actionable_without_semantics"


def scalar_examples(series: Any, limit: int = 5) -> list[str]:
    vals = []
    for value in series.dropna().astype(str).unique()[:limit]:
        text = str(value)
        if len(text) > 80:
            text = text[:77] + "..."
        vals.append(text)
    return vals


def audit_source(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(spec["path"])
    if not path.is_file():
        return [], {
            "dataset": spec["dataset"],
            "source": spec["source"],
            "path": str(path),
            "status": "missing_source",
            "obs_rows": 0,
            "obs_cols": 0,
            "candidate_columns": [],
        }
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    try:
        adata.file.close()
    except Exception:
        pass
    rows = []
    candidate_cols = []
    class_counts: Counter[str] = Counter()
    for col in obs.columns:
        family, decision = classify_column(str(col))
        nonnull = int(obs[col].notna().sum())
        unique = int(obs[col].dropna().astype(str).nunique())
        examples = scalar_examples(obs[col])
        class_counts[family] += 1
        row = {
            "dataset": spec["dataset"],
            "source": spec["source"],
            "path": str(path),
            "column": str(col),
            "family": family,
            "decision": decision,
            "nonnull_rows": nonnull,
            "unique_values": unique,
            "examples": ";".join(examples),
        }
        rows.append(row)
        if decision == "candidate_needs_materialization":
            candidate_cols.append(str(col))
    summary = {
        "dataset": spec["dataset"],
        "source": spec["source"],
        "path": str(path),
        "status": "scanned",
        "obs_rows": int(obs.shape[0]),
        "obs_cols": int(obs.shape[1]),
        "class_counts": dict(sorted(class_counts.items())),
        "candidate_columns": candidate_cols,
    }
    return rows, summary


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def main() -> int:
    all_rows: list[dict[str, Any]] = []
    summaries = []
    for spec in SOURCES:
        rows, summary = audit_source(spec)
        all_rows.extend(rows)
        summaries.append(summary)

    candidate_cols = [
        (s["dataset"], s["source"], ",".join(s.get("candidate_columns", [])))
        for s in summaries
        if s.get("candidate_columns")
    ]
    status = (
        "external_source_h5ad_obs_candidate_columns_found_cpu_materialize_next"
        if candidate_cols
        else "external_source_h5ad_obs_no_new_artifact_candidates_no_gpu"
    )
    # Candidate columns in these source h5ads are still not automatically GPU
    # routes. Most are expected to be protocol-only or closed by prior gates.
    gpu_authorized = False

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "reads_obs_only": True,
            "reads_expression": False,
            "reads_checkpoints": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_training": False,
            "uses_gpu": False,
        },
        "summaries": summaries,
        "candidate_columns": candidate_cols,
        "prior_closed_reports": PRIOR_CLOSED_ARTIFACT_REPORTS,
        "rows_csv": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = ["dataset", "source", "path", "column", "family", "decision", "nonnull_rows", "unique_values", "examples"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    summary_rows = [
        [
            s["dataset"],
            s["source"],
            s["status"],
            s["obs_rows"],
            s["obs_cols"],
            s.get("class_counts", {}),
            ",".join(s.get("candidate_columns", [])),
        ]
        for s in summaries
    ]
    candidate_preview = [
        [r["dataset"], r["source"], r["column"], r["family"], r["unique_values"], r["examples"]]
        for r in all_rows
        if r["decision"] == "candidate_needs_materialization"
    ][:30]

    md = f"""# LatentFM External Source h5ad Obs Route Audit

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- Reads only `.obs` metadata from externally downloaded processed h5ad files in backed mode.
- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.
- This is a source-route audit; candidate columns still require condition-level materialization and strict preflight.

## Source Summary

{md_table(["dataset", "source", "status", "obs rows", "obs cols", "class counts", "candidate columns"], summary_rows)}

## Candidate Column Preview

{md_table(["dataset", "source", "column", "family", "unique values", "examples"], candidate_preview) if candidate_preview else "No candidate columns found."}

## Decision

- Read/UMI/guide/cell-count/QC/source/background/target-like fields remain closed by existing gates.
- Any time/viability/growth/replicate/dose/program candidate seen here is not a GPU route until materialized as `dataset,condition,artifact_value` and passed strict multi-dataset controls.
- If no candidate columns appear, the external h5ad sources add no new non-duplicate artifact route beyond already closed read/guide-support evidence.

## Prior Closure Reports

{md_table(["report"], [[p] for p in PRIOR_CLOSED_ARTIFACT_REPORTS])}

## Outputs

- JSON: `{OUT_JSON}`
- rows CSV: `{OUT_CSV}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
