#!/usr/bin/env python3
"""Backfill condition_metadata.json for dose-aware all-modality artifacts.

The CPU materializer intentionally builds H5/sampled-index artifacts first.
This script adds the sidecar needed by LatentFM perturbation conditioning,
especially for SciPlex dose-level conditions whose H5 condition key should not
be used as the chemical descriptor key.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
DRUG_INDEX_JSON = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625/drug_index.json"
BASE_METADATA_JSON = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_condition_metadata_backfill_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_CONDITION_METADATA_BACKFILL_20260625.md"
SCIPLEX_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}


def load_json(path: Path) -> Any:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def clean_drug_key(dataset: str, cov_drug: str) -> str:
    value = str(cov_drug or "").strip()
    bg = dataset.replace("sciplex3_", "", 1)
    prefix = f"{bg}_"
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def gene_entry(dataset: str, cond: str, base_metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    ds_meta = base_metadata.get(dataset) or {}
    base = ds_meta.get(cond)
    if base is None:
        return None, f"missing_base_gene_metadata:{dataset}:{cond}"
    entry = dict(base)
    genes = entry.get("genes")
    if not genes:
        genes = [g.strip().upper() for g in str(cond).split("+") if g.strip()]
    entry.update(
        {
            "genes": [str(g).strip().upper() for g in genes if str(g).strip()],
            "combo_id": 1 if "+" in str(cond) else 0,
            "nperts_obs": len([g for g in str(cond).split("+") if g.strip()]),
            "chem_source": None,
            "chem_obs_value": None,
            "metadata_source": str(BASE_METADATA_JSON),
        }
    )
    if not str(entry.get("perturbation_type_raw", "")).strip():
        return None, f"empty_base_gene_perturbation_type:{dataset}:{cond}"
    return entry, None


def chem_entry(dataset: str, cond: str, meta: dict[str, Any], drug_index: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    drug = clean_drug_key(dataset, str(meta.get("cov_drug", "")))
    reason = None
    if not drug:
        reason = "empty_drug_key"
    elif drug not in drug_index:
        reason = f"drug_key_missing_from_cache:{drug}"
    entry = {
        "genes": [],
        "perturbation_type_raw": "drug",
        "combo_id": 0,
        "nperts_obs": 1,
        "chem_source": f"drug={drug}",
        "chem_obs_value": drug,
        "dose_condition": str(cond),
        "dose": str(meta.get("dose", "")),
        "pathway": str(meta.get("pathway", "")),
        "target": str(meta.get("target", "")),
        "cov_drug": str(meta.get("cov_drug", "")),
    }
    return entry, reason


def backfill_row(row: dict[str, Any], drug_index: dict[str, Any], base_metadata: dict[str, Any], *, write: bool) -> dict[str, Any]:
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    reasons: list[str] = []
    if not data_dir.exists():
        reasons.append("missing_data_dir")
    if not split_file.exists():
        reasons.append("missing_split_file")
    summary_path = data_dir / "sampled_indices_summary.json.gz"
    if not summary_path.exists():
        reasons.append("missing_sampled_indices_summary")
    if reasons:
        return {"run_id": row["run_id"], "status": "fail", "reasons": reasons}
    split = load_json(split_file)
    with gzip.open(summary_path, "rt", encoding="utf-8") as handle:
        sampled_summary = json.load(handle)
    metadata: dict[str, dict[str, Any]] = {}
    missing_drug_keys: list[str] = []
    missing_gene_metadata: list[str] = []
    counts = {"gene": 0, "chemical": 0}
    for dataset, groups in sorted(split.items()):
        conds = sorted(set(groups.get("train") or []) | set(groups.get("internal_val_allmodality_doseaware") or []))
        ds_meta: dict[str, Any] = {}
        if dataset in SCIPLEX_DATASETS:
            sampled_ds = sampled_summary.get(dataset) or {}
            for cond in conds:
                entry, reason = chem_entry(dataset, cond, sampled_ds.get(cond) or {}, drug_index)
                ds_meta[str(cond)] = entry
                counts["chemical"] += 1
                if reason:
                    missing_drug_keys.append(f"{dataset}:{cond}:{reason}")
        else:
            for cond in conds:
                entry, reason = gene_entry(str(dataset), str(cond), base_metadata)
                if entry is not None:
                    ds_meta[str(cond)] = entry
                    counts["gene"] += 1
                if reason:
                    missing_gene_metadata.append(reason)
        metadata[str(dataset)] = ds_meta
    if missing_drug_keys:
        reasons.append(f"drug_cache_misses:{len(missing_drug_keys)}")
    if missing_gene_metadata:
        reasons.append(f"gene_metadata_misses:{len(missing_gene_metadata)}")
    if write and not reasons:
        (data_dir / "condition_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "run_id": row["run_id"],
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "counts": counts,
        "condition_metadata_path": str(data_dir / "condition_metadata.json"),
        "missing_drug_keys_preview": missing_drug_keys[:20],
        "missing_gene_metadata_preview": missing_gene_metadata[:20],
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Condition Metadata Backfill",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only metadata sidecar generation.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "- SciPlex dose-level H5 condition keys are mapped to drug-level descriptor keys through `chem_obs_value` / `chem_source`.",
        "",
        f"- write mode: `{payload['write']}`",
        f"- drug index: `{DRUG_INDEX_JSON}`",
        f"- base gene metadata: `{BASE_METADATA_JSON}`",
        "",
        "## Rows",
        "",
        "| run id | status | counts | reasons |",
        "|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | `{row.get('counts', {})}` | {', '.join(row.get('reasons') or []) or 'none'} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    materializer = load_json(MATERIALIZER_JSON)
    rows = materializer.get("materialized_rows") or []
    reasons: list[str] = []
    if materializer.get("status") != "allmodality_doseaware_materialized_no_gpu":
        reasons.append(f"materializer_not_complete:{materializer.get('status')}")
    if not rows:
        reasons.append("no_materialized_rows")
    drug_index = load_json(DRUG_INDEX_JSON)
    if not isinstance(drug_index, dict) or not drug_index:
        reasons.append("drug_index_missing_or_empty")
        drug_index = {}
    base_metadata = load_json(BASE_METADATA_JSON)
    if not isinstance(base_metadata, dict) or not base_metadata:
        reasons.append("base_metadata_missing_or_empty")
        base_metadata = {}
    audit_rows = [] if reasons else [
        backfill_row(row, drug_index, base_metadata, write=bool(args.write)) for row in rows
    ]
    if reasons:
        status = "allmodality_doseaware_condition_metadata_not_ready_no_gpu"
    elif all(row.get("status") == "ok" for row in audit_rows):
        status = "allmodality_doseaware_condition_metadata_written_no_gpu" if args.write else "allmodality_doseaware_condition_metadata_dryrun_pass_no_gpu"
    else:
        status = "allmodality_doseaware_condition_metadata_fail_no_gpu"
    payload = {
        "status": status,
        "reasons": reasons,
        "write": bool(args.write),
        "materializer_json": str(MATERIALIZER_JSON),
        "drug_index_json": str(DRUG_INDEX_JSON),
        "base_metadata_json": str(BASE_METADATA_JSON),
        "rows": audit_rows,
        "gpu_authorized": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
