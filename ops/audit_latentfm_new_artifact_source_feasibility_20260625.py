#!/usr/bin/env python3
"""Audit unused obs-metadata artifact sources for LatentFM scaling/mainline.

Short CPU task. Reads AnnData ``.obs`` metadata and completed train-only row
metrics only. It does not read expression matrices, checkpoints, canonical
multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OBS_SCHEMA_JSON = REPORTS / "latentfm_dataset_obs_schema_audit_20260624.json"
CONDITION_INV_JSON = REPORTS / "latentfm_condition_level_inventory_20260624.json"
OUT_JSON = REPORTS / "latentfm_new_artifact_source_feasibility_20260625.json"
OUT_CSV = REPORTS / "latentfm_new_artifact_source_feasibility_rows_20260625.csv"
OUT_MD = REPORTS / "LATENTFM_NEW_ARTIFACT_SOURCE_FEASIBILITY_20260625.md"

ROW_METRIC_FILES = [
    REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    REPORTS / "latentfm_background_target_actionability_rows_20260625.csv",
    REPORTS / "latentfm_lodo_domain_conflict_rows_20260625.csv",
    REPORTS / "latentfm_multiprior_tailrisk_mask_rows_20260625.csv",
    REPORTS / "latentfm_response_program_projection_rows_20260625.csv",
    REPORTS / "latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]

ARTIFACT_CLASSES = {
    "technical_replicate_batch": [
        "plate",
        "well",
        "replicate",
        "time",
        "batch",
        "sample",
        "donor",
    ],
    "guide_cytokine_context": [
        "guide",
        "sgrna",
        "cytokine_treatment",
        "mixscale_score",
    ],
    "cell_quality_qc": [
        "ncounts",
        "ngenes",
        "n_genes",
        "n_genes_by_counts",
        "total_counts",
        "total_counts_mt",
        "pct_counts_mt",
        "percent_mito",
        "percent_ribo",
    ],
    "chemical_semantics_protocol": [
        "dose",
        "dose_value",
        "drug_dose_name",
        "cov_drug_dose_name",
        "pathway",
        "pathway_level_1",
        "pathway_level_2",
        "target",
        "chembl-id",
    ],
}

CONSUMED_OR_GATED = {
    "chemical_semantics_protocol": (
        "covered by chemical/scaffold protocol boundary and exact-ACK-gated V2 "
        "controls; not a new non-ACK GPU route"
    ),
    "technical_replicate_batch": (
        "mostly SciPlex chemical technical metadata; potentially useful as a "
        "chemical protocol/control covariate, but no current non-ACK LatentFM "
        "GPU route"
    ),
    "guide_cytokine_context": (
        "Jiang-specific biological/guide context; useful for a narrow CPU "
        "stratification audit, but not broad enough for a scaling-law claim"
    ),
    "cell_quality_qc": (
        "broadly covered and not identical to background/type/source scaling; "
        "requires a CPU outcome-overlap gate before any training change"
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def pick_first(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def infer_condition_col(columns: list[str], modality: str) -> str | None:
    if modality == "chemical":
        return pick_first(
            columns,
            ["cov_drug_dose_name", "drug_dose_name", "cov_drug", "condition", "cov", "perturbation"],
        )
    return pick_first(columns, ["perturbation", "condition", "gene", "target"])


def is_control_condition(text: str) -> bool:
    return text.lower() in {
        "control",
        "ctrl",
        "ntc",
        "non-targeting",
        "non_targeting",
        "dmso",
        "vehicle",
    }


def columns_for_class(columns: list[str], class_name: str) -> list[str]:
    tokens = ARTIFACT_CLASSES[class_name]
    hits = []
    for col in columns:
        low = col.lower()
        if any(tok in low for tok in tokens):
            hits.append(col)
    return sorted(set(hits))


def load_outcome_keys() -> tuple[set[tuple[str, str]], dict[str, int]]:
    keys: set[tuple[str, str]] = set()
    by_file: dict[str, int] = {}
    for path in ROW_METRIC_FILES:
        if not path.is_file():
            continue
        local = set()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "dataset" not in (reader.fieldnames or []) or "condition" not in (reader.fieldnames or []):
                continue
            for row in reader:
                ds = norm(row.get("dataset"))
                cond = norm(row.get("condition"))
                if ds and cond:
                    local.add((ds, cond))
        keys.update(local)
        by_file[path.name] = len(local)
    return keys, by_file


def inspect_dataset(file_row: dict[str, Any], outcome_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    path = Path(str(file_row["path"]))
    modality = str(file_row.get("modality", ""))
    dataset = str(file_row.get("dataset", ""))
    bucket = str(file_row.get("bucket", ""))
    if not path.is_file():
        return []

    obj = ad.read_h5ad(path, backed="r")
    try:
        obs = obj.obs.copy()
    finally:
        if getattr(obj, "file", None) is not None:
            obj.file.close()

    columns = [str(c) for c in obs.columns]
    condition_col = infer_condition_col(columns, modality)
    if condition_col is None or condition_col not in obs.columns:
        return []

    condition_series = obs[condition_col].map(norm)
    non_control = ~condition_series.map(is_control_condition)
    obs = obs.loc[non_control].copy()
    condition_series = obs[condition_col].map(norm)
    condition_values = sorted(c for c in condition_series.unique().tolist() if c)
    condition_set = {(dataset, c) for c in condition_values}
    overlap_conditions = sorted(c for _, c in (condition_set & outcome_keys))

    rows = []
    for class_name in ARTIFACT_CLASSES:
        class_cols = columns_for_class(columns, class_name)
        if not class_cols:
            continue
        nonnull_conditions = set()
        within_condition_variation = set()
        column_unique_counts: dict[str, int] = {}
        numeric_columns = []
        for col in class_cols:
            if col not in obs.columns:
                continue
            s = obs[col]
            nonnull_mask = s.notna() & (s.map(norm) != "")
            if nonnull_mask.any():
                nonnull_conditions.update(condition_series.loc[nonnull_mask].unique().tolist())
            try:
                column_unique_counts[col] = int(s.dropna().map(norm).nunique())
            except Exception:
                column_unique_counts[col] = -1
            if pd.api.types.is_numeric_dtype(s):
                numeric_columns.append(col)
            if col == condition_col:
                continue
            grouped = obs.loc[nonnull_mask, [condition_col, col]].groupby(condition_col, observed=True)
            for cond, frame in grouped:
                if frame[col].dropna().map(norm).nunique() > 1:
                    within_condition_variation.add(norm(cond))

        rows.append(
            {
                "artifact_class": class_name,
                "dataset": dataset,
                "bucket": bucket,
                "modality": modality,
                "source_h5ad": str(path),
                "columns": class_cols,
                "numeric_columns": sorted(numeric_columns),
                "n_cells": int(len(obs)),
                "n_conditions": int(len(condition_values)),
                "n_nonnull_conditions": int(len(nonnull_conditions)),
                "n_within_condition_variation": int(len(within_condition_variation)),
                "n_outcome_overlap_conditions": int(len(overlap_conditions)),
                "example_overlap_conditions": overlap_conditions[:8],
                "column_unique_counts": column_unique_counts,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], outcome_by_file: dict[str, int]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    for class_name in ARTIFACT_CLASSES:
        local = [r for r in rows if r["artifact_class"] == class_name]
        datasets = sorted({r["dataset"] for r in local})
        modalities = Counter(r["modality"] for r in local)
        columns = sorted({c for r in local for c in r["columns"]})
        overlap = sum(int(r["n_outcome_overlap_conditions"]) for r in local)
        conditions = sum(int(r["n_conditions"]) for r in local)
        nonnull_conditions = sum(int(r["n_nonnull_conditions"]) for r in local)
        within_var = sum(int(r["n_within_condition_variation"]) for r in local)
        if class_name == "cell_quality_qc" and len(datasets) >= 10 and overlap >= 100:
            decision = "cpu_gate_candidate"
            next_gate = (
                "condition-level QC/cell-support reliability gate over completed train-only row metrics; "
                "test whether QC/support strata predict tail harm and whether a non-noop training-set "
                "filter/weighting rule beats count/source controls"
            )
        elif class_name == "guide_cytokine_context" and len(datasets) >= 3 and overlap > 0:
            decision = "narrow_cpu_stratification_candidate"
            next_gate = (
                "Jiang-only guide/cytokine/mixscale stratification audit; supplement or mechanism route only"
            )
        elif class_name == "technical_replicate_batch" and overlap > 0:
            decision = "chemical_protocol_covariate_candidate_ack_gated"
            next_gate = (
                "use as SciPlex chemical V2 protocol covariate/control after exact ACK; no non-ACK GPU"
            )
        else:
            decision = "diagnostic_only_no_gpu"
            next_gate = "no immediate LatentFM GPU route"
        by_class[class_name] = {
            "datasets": len(datasets),
            "dataset_names": datasets,
            "modalities": dict(sorted(modalities.items())),
            "columns": columns,
            "conditions": int(conditions),
            "nonnull_conditions": int(nonnull_conditions),
            "within_condition_variation": int(within_var),
            "outcome_overlap_conditions": int(overlap),
            "decision": decision,
            "next_gate": next_gate,
            "caveat": CONSUMED_OR_GATED[class_name],
        }
    immediate_gpu = []
    cpu_candidates = [
        cls
        for cls, info in by_class.items()
        if info["decision"] in {"cpu_gate_candidate", "narrow_cpu_stratification_candidate"}
    ]
    return {
        "status": (
            "new_artifact_source_feasibility_cpu_gate_candidates_no_gpu"
            if cpu_candidates
            else "new_artifact_source_feasibility_no_gpu"
        ),
        "gpu_authorized": False,
        "immediate_gpu_candidates": immediate_gpu,
        "cpu_gate_candidates": cpu_candidates,
        "outcome_metric_files": outcome_by_file,
        "classes": by_class,
        "decision": {
            "default_model": "xverse_8k_anchor",
            "no_gpu_reason": (
                "new obs-metadata artifacts need CPU outcome gates and no-harm/tail controls before "
                "training; chemical technical artifacts remain exact-ACK-gated"
            ),
            "recommended_next_action": (
                "run a short CPU condition-level QC/support reliability gate first; if it passes, "
                "design a bounded leakage-safe filter/weighting smoke"
            ),
        },
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "artifact_class",
        "dataset",
        "bucket",
        "modality",
        "n_cells",
        "n_conditions",
        "n_nonnull_conditions",
        "n_within_condition_variation",
        "n_outcome_overlap_conditions",
        "columns",
        "numeric_columns",
        "example_overlap_conditions",
        "source_h5ad",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("columns", "numeric_columns", "example_overlap_conditions"):
                out[key] = ";".join(str(x) for x in out[key])
            writer.writerow({k: out.get(k, "") for k in fields})


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM New Artifact Source Feasibility",
        "",
        "Timestamp: `2026-06-25`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only obs-metadata and completed train-only row-metric audit.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "- Canonical no-harm remains a downstream veto, not a selection signal here.",
        "",
        "## Summary",
        "",
        "| artifact class | datasets | conditions | outcome-overlap conditions | decision | next gate |",
        "|---|---:|---:|---:|---|---|",
    ]
    for cls, info in payload["classes"].items():
        lines.append(
            f"| `{cls}` | {info['datasets']} | {info['conditions']} | "
            f"{info['outcome_overlap_conditions']} | `{info['decision']}` | {info['next_gate']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `cell_quality_qc` is the only broad, non-chemical obs-metadata class with enough coverage to justify a new CPU gate.",
        "- `guide_cytokine_context` is real but Jiang-specific; it can support mechanism/failure analysis, not a general scaling-law claim.",
        "- `technical_replicate_batch` and `chemical_semantics_protocol` are most useful inside the chemical V2 protocol after exact ACK; they do not unlock non-ACK GPU training.",
        "- No immediate GPU launch is authorized by this audit. A GPU smoke would require the QC/support CPU gate to produce a non-noop filter/weighting rule that passes tail/no-harm controls, or chemical V2 exact ACK.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    obs_payload = read_json(OBS_SCHEMA_JSON)
    _ = read_json(CONDITION_INV_JSON)
    outcome_keys, outcome_by_file = load_outcome_keys()
    rows: list[dict[str, Any]] = []
    for file_row in obs_payload["rows"]:
        if file_row.get("status") != "ok":
            continue
        rows.extend(inspect_dataset(file_row, outcome_keys))
    payload = summarize(rows, outcome_by_file)
    payload["rows"] = rows
    payload["outputs"] = {"json": str(OUT_JSON), "csv": str(OUT_CSV), "md": str(OUT_MD)}
    write_csv(rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
