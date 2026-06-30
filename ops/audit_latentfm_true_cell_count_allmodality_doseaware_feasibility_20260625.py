#!/usr/bin/env python3
"""Feasibility gate for dose-aware all-modality true-cell artifacts.

This checks whether existing xverse per-cell embeddings can support a
query-blind dose-level SciPlex internal split, avoiding the current drug-level
latent H5 collapse.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
PROTOCOL_TSV = ROOT / "reports/latentfm_true_cell_count_scaling_protocol_20260624/all_modality_fixed64_budget16_32_64.tsv"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
XVERSE_EMB_ROOT = ROOT / "scFM_output/embeddings/xverse"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_feasibility_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_FEASIBILITY_GATE_20260625.md"

SCIPLEX_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
EXCLUDED_SPLIT_KEYS = {"canonical_test_reference"}
MINIMA = {"train_gene": 50, "eval_gene": 20, "train_chemical": 50, "eval_chemical": 20}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_protocol_rows() -> list[dict[str, str]]:
    with PROTOCOL_TSV.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def split_roles(groups: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    train = {str(x) for x in groups.get("train") or []}
    canonical = {str(x) for x in groups.get("canonical_test_reference") or []}
    eval_set: set[str] = set()
    for key, values in groups.items():
        if key == "train" or key in EXCLUDED_SPLIT_KEYS or not isinstance(values, list):
            continue
        eval_set.update(str(x) for x in values)
    return train, eval_set, canonical


def background_from_dataset(dataset: str) -> str:
    return dataset.replace("sciplex3_", "", 1)


def drug_from_cov_drug(dataset: str, cov_drug: str) -> str:
    bg = background_from_dataset(dataset)
    val = str(cov_drug)
    if val.startswith(bg + "_"):
        return val[len(bg) + 1 :]
    return val


def numeric_dose(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def gene_counts(protocol_rows: list[dict[str, str]], split: dict[str, Any]) -> dict[str, Any]:
    train_n = 0
    eval_n = 0
    datasets = defaultdict(lambda: {"train": 0, "eval": 0})
    for row in protocol_rows:
        if row.get("modality") != "gene":
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        train, eval_set, _canonical = split_roles(split.get(ds) or {})
        if cond in train:
            train_n += 1
            datasets[ds]["train"] += 1
        if cond in eval_set:
            eval_n += 1
            datasets[ds]["eval"] += 1
    return {"train_conditions": train_n, "eval_conditions": eval_n, "datasets": dict(sorted(datasets.items()))}


def summarize_sciplex_dataset(dataset: str, split: dict[str, Any]) -> dict[str, Any]:
    emb_dir = XVERSE_EMB_ROOT / dataset / "raw"
    latent_path = emb_dir / "latent.npy"
    obs_path = emb_dir / "obs.parquet"
    meta_path = emb_dir / "meta.json"
    row: dict[str, Any] = {
        "dataset": dataset,
        "embedding_dir": str(emb_dir),
        "latent_exists": latent_path.exists(),
        "obs_exists": obs_path.exists(),
        "meta_exists": meta_path.exists(),
    }
    if not (latent_path.exists() and obs_path.exists() and meta_path.exists()):
        row["status"] = "missing_embedding_artifact"
        return row

    latent = np.load(latent_path, mmap_mode="r")
    obs = pd.read_parquet(obs_path)
    meta = load_json(meta_path)
    required_cols = {"cov_drug_dose_name", "cov_drug", "condition", "control", "dose"}
    missing_cols = sorted(required_cols - set(obs.columns))
    row.update(
        {
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype),
            "latent_dim": int(latent.shape[1]) if len(latent.shape) == 2 else None,
            "obs_rows": int(len(obs)),
            "meta_latent_dim": meta.get("latent_dim"),
            "missing_obs_columns": missing_cols,
            "source_adata": meta.get("source_adata"),
        }
    )
    if int(latent.shape[0]) != int(len(obs)):
        row["status"] = "latent_obs_row_mismatch"
        return row
    if missing_cols:
        row["status"] = "missing_obs_columns"
        return row
    if int(latent.shape[1]) != 384:
        row["status"] = "latent_dim_not_384"
        return row

    train_drugs, _eval_drugs, canonical_drugs = split_roles(split.get(dataset) or {})
    pert = obs[obs["control"].astype(str).isin({"0", "False", "false"})].copy()
    pert["drug"] = pert["cov_drug"].map(lambda x: drug_from_cov_drug(dataset, str(x)))
    pert["dose_condition"] = pert["cov_drug_dose_name"].astype(str)
    pert = pert[pert["drug"].isin(train_drugs)].copy()
    grouped = (
        pert.groupby(["drug", "dose_condition"], observed=True)
        .agg(n_cells=("dose_condition", "size"), dose=("dose", "first"))
        .reset_index()
    )
    eligible = grouped[grouped["n_cells"] >= 64].copy()

    train_conditions: set[str] = set()
    eval_conditions: set[str] = set()
    single_dose_train_only = 0
    for drug, g in eligible.groupby("drug", sort=True):
        g = g.copy()
        g["_dose_num"] = g["dose"].map(numeric_dose)
        g = g.sort_values(["_dose_num", "dose_condition"])
        conds = list(g["dose_condition"])
        if len(conds) >= 2:
            eval_conditions.add(conds[-1])
            train_conditions.update(conds[:-1])
        elif conds:
            train_conditions.add(conds[0])
            single_dose_train_only += 1

    canonical_overlap = set(eligible["drug"]) & canonical_drugs
    row.update(
        {
            "status": "ok",
            "base_train_drugs": len(train_drugs),
            "base_canonical_drugs": len(canonical_drugs),
            "eligible_train_drugs": int(eligible["drug"].nunique()),
            "eligible_dose_conditions": int(eligible["dose_condition"].nunique()),
            "proposed_train_dose_conditions": len(train_conditions),
            "proposed_eval_dose_conditions": len(eval_conditions),
            "single_dose_train_only_drugs": single_dose_train_only,
            "canonical_drug_overlap_after_train_filter": len(canonical_overlap),
            "example_train_dose_conditions": sorted(train_conditions)[:6],
            "example_eval_dose_conditions": sorted(eval_conditions)[:6],
        }
    )
    return row


def decide(gene: dict[str, Any], sciplex_rows: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    train_chem = sum(int(r.get("proposed_train_dose_conditions", 0)) for r in sciplex_rows)
    eval_chem = sum(int(r.get("proposed_eval_dose_conditions", 0)) for r in sciplex_rows)
    bad_status = [r for r in sciplex_rows if r.get("status") != "ok"]
    canonical_overlap = sum(int(r.get("canonical_drug_overlap_after_train_filter", 0)) for r in sciplex_rows)

    if gene["train_conditions"] < MINIMA["train_gene"]:
        reasons.append("gene_train_conditions_below_minimum")
    if gene["eval_conditions"] < MINIMA["eval_gene"]:
        reasons.append("gene_eval_conditions_below_minimum")
    if bad_status:
        reasons.append("sciplex_xverse_embedding_artifacts_not_ready")
    if train_chem < MINIMA["train_chemical"]:
        reasons.append("doseaware_chemical_train_conditions_below_minimum")
    if eval_chem < MINIMA["eval_chemical"]:
        reasons.append("doseaware_chemical_eval_conditions_below_minimum")
    if canonical_overlap:
        reasons.append("canonical_drugs_leaked_into_train_filtered_dose_set")

    summary = {
        "gene_train_conditions": gene["train_conditions"],
        "gene_eval_conditions": gene["eval_conditions"],
        "doseaware_chemical_train_conditions": train_chem,
        "doseaware_chemical_eval_conditions": eval_chem,
        "bad_sciplex_artifact_rows": len(bad_status),
        "canonical_overlap_after_train_filter": canonical_overlap,
    }
    status = "allmodality_doseaware_feasibility_pass_cpu_materializer_next" if not reasons else "allmodality_doseaware_feasibility_fail_no_gpu"
    return status, reasons, summary


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Feasibility Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only feasibility audit for dose-aware all-modality true-cell artifacts.",
        "- Reads protocol TSV, xverse train-only split, and existing xverse per-cell embedding metadata/obs.",
        "- Does not materialize H5 artifacts, train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "- Canonical reference drugs are excluded from the proposed train/internal chemical split.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## SciPlex Rows",
            "",
            "| dataset | status | latent shape | eligible drugs | eligible dose conds | proposed train dose conds | proposed eval dose conds | source |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["sciplex_rows"]:
        lines.append(
            "| {dataset} | `{status}` | `{shape}` | {drugs} | {eligible} | {train} | {eval} | `{source}` |".format(
                dataset=row["dataset"],
                status=row.get("status"),
                shape=row.get("latent_shape"),
                drugs=row.get("eligible_train_drugs", 0),
                eligible=row.get("eligible_dose_conditions", 0),
                train=row.get("proposed_train_dose_conditions", 0),
                eval=row.get("proposed_eval_dose_conditions", 0),
                source=row.get("source_adata", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- CPU materializer authorized next: `{payload['cpu_materializer_authorized_next']}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            "- Reasons: " + (", ".join(f"`{r}`" for r in payload["reasons"]) if payload["reasons"] else "`none`"),
            "",
            "## Next Action",
            "",
            "If this gate passes, write a dose-aware CPU materializer that builds capped latent H5 artifacts from `scFM_output/embeddings/xverse/sciplex3_*/raw/latent.npy` grouped by `cov_drug_dose_name`, with deterministic train/eval dose-condition assignment and sampled-row provenance. GPU remains blocked until materializer, schema, dryload, and design controls pass.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    protocol_rows = read_protocol_rows()
    split = load_json(BASE_SPLIT)
    gene = gene_counts(protocol_rows, split)
    sciplex_rows = [summarize_sciplex_dataset(ds, split) for ds in SCIPLEX_DATASETS]
    status, reasons, summary = decide(gene, sciplex_rows)
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "materializes_artifacts": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
            "excluded_split_keys": sorted(EXCLUDED_SPLIT_KEYS),
        },
        "inputs": {
            "protocol_tsv": str(PROTOCOL_TSV),
            "base_split": str(BASE_SPLIT),
            "xverse_embedding_root": str(XVERSE_EMB_ROOT),
        },
        "gene_rows": gene,
        "sciplex_rows": sciplex_rows,
        "summary": summary,
        "reasons": reasons,
        "cpu_materializer_authorized_next": not reasons,
        "gpu_authorized": False,
        "next_action": "write_doseaware_cpu_materializer_schema_dryload_design_gates" if not reasons else "fix_feasibility_blockers_before_materializer",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
