#!/usr/bin/env python3
"""CPU provenance gate for true drug molecule/dose scaling semantics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path("/data/cyx/1030/scLatent")
RAW_DIR = ROOT / "dataset/raw/chemicalpert_bench"
DRUG_MANIFEST = ROOT / "dataset/drug_cache/sciplex_label_identity_561/manifest.json"
CAP120_INTERNAL = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_scaling_drug_semantics_provenance_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_DRUG_SEMANTICS_PROVENANCE_GATE_20260624.md"


def read_cat(obj: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(obj, h5py.Group):
        codes = obj["codes"][:]
        cats_obj = obj["categories"]
        cats = cats_obj["values"][:] if isinstance(cats_obj, h5py.Group) else cats_obj[:]
        cats_s = [x.decode() if isinstance(x, bytes) else str(x) for x in cats]
        return [cats_s[int(c)] if int(c) >= 0 else "" for c in codes]
    arr = obj[:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf) if scaf is not None else ""


def internal_sciplex_rows() -> int:
    payload = load_json(CAP120_INTERNAL)
    n = 0
    for group in (payload.get("groups") or {}).values():
        for row in (group or {}).get("condition_metrics") or []:
            if str(row.get("dataset") or "").startswith("sciplex3_"):
                n += 1
    return n


def main() -> int:
    manifest = load_json(DRUG_MANIFEST)
    dataset_rows: dict[str, dict[str, Any]] = {}
    all_drugs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in sorted(RAW_DIR.glob("sciplex3_*.h5ad")):
        dataset = path.stem
        with h5py.File(path, "r") as handle:
            obs = handle["obs"]
            conds = read_cat(obs["condition"])
            smiles = read_cat(obs["SMILES"])
            doses = read_cat(obs["dose_value"])
            pathways = read_cat(obs["pathway"]) if "pathway" in obs else [""] * len(conds)
            targets = read_cat(obs["target"]) if "target" in obs else [""] * len(conds)
        by_drug: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for cond, smi, dose, pathway, target in zip(conds, smiles, doses, pathways, targets):
            if not cond or cond.lower() == "control":
                continue
            by_drug[cond]["smiles"].add(smi)
            by_drug[cond]["doses"].add(dose)
            by_drug[cond]["pathways"].add(pathway)
            by_drug[cond]["targets"].add(target)
            all_drugs[cond]["smiles"].add(smi)
            all_drugs[cond]["datasets"].add(dataset)
        valid = 0
        scaffolds = set()
        for vals in by_drug.values():
            smi = sorted(s for s in vals["smiles"] if s)
            scaf = scaffold(smi[0]) if smi else None
            if scaf is not None:
                valid += 1
                scaffolds.add(scaf)
        dataset_rows[dataset] = {
            "n_drugs": len(by_drug),
            "n_valid_smiles": valid,
            "n_unique_scaffolds": len(scaffolds),
            "n_dose_values": len({d for vals in by_drug.values() for d in vals["doses"] if d}),
            "n_pathways": len({p for vals in by_drug.values() for p in vals["pathways"] if p}),
            "n_targets": len({t for vals in by_drug.values() for t in vals["targets"] if t}),
        }
    shared_all3 = sum(1 for vals in all_drugs.values() if len(vals["datasets"]) == 3)
    valid_global = 0
    scaffolds_global = set()
    for vals in all_drugs.values():
        smi = sorted(s for s in vals["smiles"] if s)
        scaf = scaffold(smi[0]) if smi else None
        if scaf is not None:
            valid_global += 1
            scaffolds_global.add(scaf)
    sciplex_internal_rows = internal_sciplex_rows()
    reasons = []
    if manifest.get("embed_kind") != "drug_label_identity":
        reasons.append("unexpected_existing_drug_cache_kind")
    if valid_global < 100:
        reasons.append("low_valid_smiles_coverage")
    if sciplex_internal_rows <= 0:
        reasons.append("no_sciplex_rows_in_current_internal_scaling_eval")
    reasons.append("descriptor_cache_not_yet_built_for_latentfm")
    status = "drug_semantics_provenance_ready_but_metric_blocked_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "provenance_only": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "current_drug_cache": manifest,
        "summary": {
            "datasets": dataset_rows,
            "global_unique_drugs": len(all_drugs),
            "global_valid_smiles_drugs": valid_global,
            "global_unique_scaffolds": len(scaffolds_global),
            "drugs_shared_by_all_3_backgrounds": shared_all3,
            "current_internal_scaling_sciplex_rows": sciplex_internal_rows,
        },
        "reasons": reasons,
        "next_action": "build descriptor cache and chemical train-only internal eval before any drug-semantic GPU smoke",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Scaling Drug Semantics Provenance Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only provenance gate for SciPlex SMILES/dose/pathway metadata.",
        "- Does not read canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- current drug cache kind: `{manifest.get('embed_kind')}`",
        f"- global unique drugs: `{len(all_drugs)}`",
        f"- valid SMILES drugs: `{valid_global}`",
        f"- unique Murcko scaffolds: `{len(scaffolds_global)}`",
        f"- drugs shared by all 3 SciPlex backgrounds: `{shared_all3}`",
        f"- current internal scaling SciPlex rows: `{sciplex_internal_rows}`",
        "",
        "| dataset | drugs | valid SMILES | scaffolds | dose values | pathways | targets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, row in sorted(dataset_rows.items()):
        lines.append(
            f"| `{ds}` | {row['n_drugs']} | {row['n_valid_smiles']} | {row['n_unique_scaffolds']} | "
            f"{row['n_dose_values']} | {row['n_pathways']} | {row['n_targets']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- Drug/dose semantics are available and distinct from the existing label-identity cache, but current scaling internal eval has no SciPlex rows; build descriptor cache and chemical train-only eval first.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": payload["summary"], "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
