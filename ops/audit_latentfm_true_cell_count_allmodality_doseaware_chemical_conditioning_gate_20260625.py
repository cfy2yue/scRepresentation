#!/usr/bin/env python3
"""Chemical-conditioning dry-load gate for dose-aware all-modality artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.condition_emb.chempert.chem_resolver import chem_keys_for_metadata  # noqa: E402
from model.condition_emb.genepert.perturbation import PERT_TYPE_DRUG  # noqa: E402
from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402


MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
METADATA_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_condition_metadata_backfill_20260625.json"
DRYLOAD_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json"
BIFLOW_DIR = ROOT / "dataset/biFlow_data"
GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DRUG_CACHE = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625"
CHEM_DIM = 512
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_chemical_conditioning_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_CHEMICAL_CONDITIONING_GATE_20260625.md"

SCIPLEX_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}


def load_json(path: Path) -> Any:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_split_for_loader(split: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds, groups in split.items():
        out[str(ds)] = {
            "train": [str(c) for c in groups.get("train") or []],
            "test": [str(c) for c in groups.get("internal_val_allmodality_doseaware") or groups.get("test") or []],
        }
    return out


def audit_dataset_mode(row: dict[str, Any], *, mode: str) -> dict[str, Any]:
    reasons: list[str] = []
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    metadata_path = data_dir / "condition_metadata.json"
    for path in [metadata_path, split_file, data_dir / "manifest.json"]:
        if not path.exists():
            reasons.append(f"missing:{path.name}")
    if reasons:
        return {"mode": mode, "status": "fail", "reasons": reasons}
    raw_split = load_json(split_file)
    split = normalize_split_for_loader(raw_split)
    ds = CrossDatasetFMDataset(
        str(data_dir),
        split,
        batch_size=8,
        seed=int(row.get("seed", 42)),
        mode=mode,
        min_cells=8,
        ds_alpha=1.0,
        use_pert_condition=True,
        gene_embedding_cache_dir=str(GENE_CACHE),
        biflow_dir=str(BIFLOW_DIR),
        latent_backbone="xverse",
        pert_chem_enabled=True,
        drug_emb_cache_dir=str(DRUG_CACHE),
        chem_fallback_embed_dim=CHEM_DIM,
        max_chem_keys=4,
        perturbation_family_filter="all",
        silent=True,
    )
    chem_checked = 0
    gene_checked = 0
    examples: list[dict[str, Any]] = []
    for ds_name in ds.ds_names:
        for cond in ds.ds_conds[ds_name][:10]:
            meta = ds.metadata_for_condition(ds_name, cond)
            meta_e = ds.enrich_metadata_with_chem(meta)
            if ds_name in SCIPLEX_DATASETS:
                chem_checked += 1
                keys = chem_keys_for_metadata(meta)
                if not keys:
                    reasons.append(f"{mode}:{ds_name}:{cond}:no_chem_key")
                    continue
                if not meta_e.chem_emb_list:
                    reasons.append(f"{mode}:{ds_name}:{cond}:no_chem_embedding")
                    continue
                dims = [int(v.shape[0]) for v in meta_e.chem_emb_list]
                if any(dim != CHEM_DIM for dim in dims):
                    reasons.append(f"{mode}:{ds_name}:{cond}:chem_dim_{dims}")
                pb = ds._perturbation_batch_for_condition(ds_name, cond)  # noqa: SLF001 - explicit gate.
                pert_type = int(pb[2][0].item())
                chem_emb = pb[5]
                chem_mask = pb[6]
                if pert_type != PERT_TYPE_DRUG:
                    reasons.append(f"{mode}:{ds_name}:{cond}:type_id_{pert_type}_not_drug")
                if chem_emb is None or tuple(chem_emb.shape[-2:])[-1] != CHEM_DIM:
                    reasons.append(f"{mode}:{ds_name}:{cond}:batch_chem_missing_or_wrong_dim")
                if chem_mask is None or float(chem_mask[0].sum().item()) <= 0:
                    reasons.append(f"{mode}:{ds_name}:{cond}:batch_chem_mask_empty")
                examples.append({"dataset": ds_name, "condition": cond, "keys": keys[:4], "chem_dims": dims[:4]})
            else:
                gene_checked += 1
            if chem_checked >= 6 and gene_checked >= 6:
                break
        if chem_checked >= 6 and gene_checked >= 6:
            break
    if chem_checked <= 0:
        reasons.append(f"{mode}:no_chemical_conditions_checked")
    if gene_checked <= 0:
        reasons.append(f"{mode}:no_gene_conditions_checked")
    return {
        "mode": mode,
        "status": "ok" if not reasons else "fail",
        "reasons": reasons[:20],
        "chem_checked": chem_checked,
        "gene_checked": gene_checked,
        "examples": examples[:6],
    }


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    train = audit_dataset_mode(row, mode="train")
    test = audit_dataset_mode(row, mode="test")
    reasons = []
    for part in [train, test]:
        if part.get("status") != "ok":
            reasons.extend([f"{part['mode']}:{r}" for r in part.get("reasons", [])])
    return {
        "run_id": row["run_id"],
        "status": "ok" if not reasons else "fail",
        "reasons": reasons[:20],
        "train": train,
        "test": test,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Chemical Conditioning Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only perturbation-conditioning dry-load gate.",
        f"- Checks dose-aware SciPlex conditions resolve to drug-level Morgan cache keys and {CHEM_DIM}-d chemical tensors.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "",
        "## Rows",
        "",
        "| run id | status | train chem/gene | test chem/gene | reasons |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        tr = row.get("train") or {}
        te = row.get("test") or {}
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | {tr.get('chem_checked', 0)}/{tr.get('gene_checked', 0)} | {te.get('chem_checked', 0)}/{te.get('gene_checked', 0)} | {', '.join(row.get('reasons') or []) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized by this gate alone: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    materializer = load_json(MATERIALIZER_JSON)
    metadata = load_json(METADATA_JSON)
    dryload = load_json(DRYLOAD_JSON)
    rows = materializer.get("materialized_rows") or []
    reasons: list[str] = []
    if materializer.get("status") != "allmodality_doseaware_materialized_no_gpu":
        reasons.append(f"materializer_not_complete:{materializer.get('status')}")
    if metadata.get("status") != "allmodality_doseaware_condition_metadata_written_no_gpu":
        reasons.append(f"metadata_not_written:{metadata.get('status')}")
    if dryload.get("status") != "allmodality_doseaware_dryload_pass_no_gpu":
        reasons.append(f"structural_dryload_not_pass:{dryload.get('status')}")
    if not rows:
        reasons.append("no_materialized_rows")
    audit_rows = [] if reasons else [audit_row(row) for row in rows]
    if reasons:
        status = "allmodality_doseaware_chemical_conditioning_not_ready_no_gpu"
        next_action = "run after materialization, metadata backfill, and structural dryload pass"
    elif all(row.get("status") == "ok" for row in audit_rows):
        status = "allmodality_doseaware_chemical_conditioning_pass_no_gpu"
        next_action = "run design controls and resource audit before bounded GPU smoke"
    else:
        status = "allmodality_doseaware_chemical_conditioning_fail_no_gpu"
        next_action = "fix condition_metadata or drug-cache wiring"
    payload = {
        "status": status,
        "reasons": reasons,
        "inputs": {
            "materializer_json": str(MATERIALIZER_JSON),
            "metadata_json": str(METADATA_JSON),
            "dryload_json": str(DRYLOAD_JSON),
            "drug_cache": str(DRUG_CACHE),
        },
        "rows": audit_rows,
        "gpu_authorized": False,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
