#!/usr/bin/env python3
"""Build a SciPlex SMILES Morgan descriptor cache for LatentFM drug semantics."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path("/data/cyx/1030/scLatent")
RAW_DIR = ROOT / "dataset/raw/chemicalpert_bench"
OUT_DIR = ROOT / "dataset/drug_cache/sciplex_smiles_morgan2048_20260624"
N_BITS = 2048
RADIUS = 2


def read_cat(obj: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(obj, h5py.Group):
        codes = obj["codes"][:]
        cats_obj = obj["categories"]
        cats = cats_obj["values"][:] if isinstance(cats_obj, h5py.Group) else cats_obj[:]
        cats_s = [x.decode() if isinstance(x, bytes) else str(x) for x in cats]
        return [cats_s[int(c)] if int(c) >= 0 else "" for c in codes]
    arr = obj[:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf) if scaf is not None else ""


def main() -> int:
    drug_to_smiles: dict[str, set[str]] = {}
    drug_to_datasets: dict[str, set[str]] = {}
    drug_to_doses: dict[str, set[str]] = {}
    drug_to_pathways: dict[str, set[str]] = {}
    drug_to_targets: dict[str, set[str]] = {}
    for path in sorted(RAW_DIR.glob("sciplex3_*.h5ad")):
        dataset = path.stem
        with h5py.File(path, "r") as handle:
            obs = handle["obs"]
            conds = read_cat(obs["condition"])
            smiles = read_cat(obs["SMILES"])
            doses = read_cat(obs["dose_value"])
            pathways = read_cat(obs["pathway"])
            targets = read_cat(obs["target"])
        for cond, smi, dose, pathway, target in zip(conds, smiles, doses, pathways, targets):
            if not cond or cond.lower() == "control":
                continue
            drug_to_smiles.setdefault(cond, set()).add(smi)
            drug_to_datasets.setdefault(cond, set()).add(dataset)
            drug_to_doses.setdefault(cond, set()).add(dose)
            drug_to_pathways.setdefault(cond, set()).add(pathway)
            drug_to_targets.setdefault(cond, set()).add(target)

    drugs = sorted(drug_to_smiles)
    # Reserve 0 for pad and 1 for unknown to match existing LatentFM drug cache convention.
    embeddings = np.zeros((len(drugs) + 2, N_BITS), dtype=np.float32)
    index = {"<pad>": 0, "<unk>": 1}
    rows = []
    invalid = []
    for offset, drug in enumerate(drugs, start=2):
        smiles_values = sorted(s for s in drug_to_smiles[drug] if s)
        smi = smiles_values[0] if smiles_values else ""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(drug)
            vec = np.zeros((N_BITS,), dtype=np.float32)
            scaf = ""
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
            vec = np.fromiter((int(bit) for bit in fp.ToBitString()), dtype=np.float32, count=N_BITS)
            scaf = scaffold_smiles(mol)
        embeddings[offset] = vec
        index[drug] = offset
        rows.append(
            {
                "drug": drug,
                "index": offset,
                "smiles": smi,
                "scaffold": scaf,
                "datasets": sorted(drug_to_datasets.get(drug, set())),
                "dose_values": sorted(d for d in drug_to_doses.get(drug, set()) if d),
                "pathways": sorted(p for p in drug_to_pathways.get(drug, set()) if p),
                "targets": sorted(t for t in drug_to_targets.get(drug, set()) if t),
                "bit_count": int(vec.sum()),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "drug_embeddings.npy", embeddings)
    (OUT_DIR / "drug_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["key\tindex", *[f"{drug}\t{idx}" for drug, idx in sorted(index.items()) if drug not in {"<pad>", "<unk>"}]]
    (OUT_DIR / "drug_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT_DIR / "drug_metadata.tsv").open("w", encoding="utf-8") as handle:
        handle.write("index\tdrug\tsmiles\tscaffold\tbit_count\tdatasets\tdose_values\tpathways\ttargets\n")
        for row in rows:
            handle.write(
                f"{row['index']}\t{row['drug']}\t{row['smiles']}\t{row['scaffold']}\t{row['bit_count']}\t"
                f"{','.join(row['datasets'])}\t{','.join(row['dose_values'])}\t"
                f"{','.join(row['pathways'])}\t{','.join(row['targets'])}\n"
            )
    manifest = {
        "embed_kind": "smiles_morgan_fingerprint",
        "source": "dataset/raw/chemicalpert_bench sciplex3_* obs.SMILES",
        "embed_dim": N_BITS,
        "radius": RADIUS,
        "num_rows": int(embeddings.shape[0]),
        "num_keys": len(drugs),
        "pad_index": 0,
        "unk_index": 1,
        "invalid_drugs": invalid,
        "all_drugs_shared_by_backgrounds": sum(1 for d in drugs if len(drug_to_datasets.get(d, set())) == 3),
        "artifact_files": {
            "embeddings": str(OUT_DIR / "drug_embeddings.npy"),
            "index_json": str(OUT_DIR / "drug_index.json"),
            "index_tsv": str(OUT_DIR / "drug_index.tsv"),
            "metadata_tsv": str(OUT_DIR / "drug_metadata.tsv"),
        },
        "notes": [
            "This cache is not wired into LatentFM training by this script.",
            "Dose is recorded in the TSV metadata but not encoded in the drug-level fingerprint vector.",
            "Any GPU smoke using this cache still requires a train-only chemical gate and RUN_STATUS.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
