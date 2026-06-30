#!/usr/bin/env python3
"""Build CPU-only loader splits for SciPlex unseen-drug/scaffold scaling probes."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DRUG_META = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625/drug_metadata.tsv"
OUT_DIR = ROOT / "dataset/biFlow_data/xverse_chemical_unseen_drug_scaffold_splits_20260625"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_drug_scaffold_loader_splits_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_DRUG_SCAFFOLD_LOADER_SPLITS_20260625.md"
SCIPLEX = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def read_drug_meta() -> dict[str, dict[str, str]]:
    with DRUG_META.open(newline="") as handle:
        return {r["drug"]: r for r in csv.DictReader(handle, delimiter="\t")}


def mode_eval_drugs(meta: dict[str, dict[str, str]], mode: str) -> set[str]:
    if mode == "unseen_drug":
        return {d for d in meta if stable_fraction(f"drug:{d}") < 0.20}
    if mode == "unseen_scaffold":
        eval_scaffolds = {r["scaffold"] for r in meta.values() if stable_fraction(f"scaffold:{r['scaffold']}") < 0.20}
        return {d for d, r in meta.items() if r["scaffold"] in eval_scaffolds}
    raise ValueError(mode)


def summarize(split: dict[str, Any], meta: dict[str, dict[str, str]], mode: str) -> dict[str, Any]:
    train_drugs: set[str] = set()
    test_drugs: set[str] = set()
    for ds in SCIPLEX:
        train_drugs.update(split[ds]["train"])
        test_drugs.update(split[ds]["test"])
    train_scaffolds = {meta[d]["scaffold"] for d in train_drugs if d in meta}
    test_scaffolds = {meta[d]["scaffold"] for d in test_drugs if d in meta}
    by_pathway_test = Counter(meta[d]["pathways"] for d in test_drugs if d in meta)
    return {
        "mode": mode,
        "train_drugs": len(train_drugs),
        "test_drugs": len(test_drugs),
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "drug_overlap": sorted(train_drugs & test_drugs),
        "scaffold_overlap": sorted(train_scaffolds & test_scaffolds),
        "test_pathways": dict(by_pathway_test.most_common()),
        "dataset_counts": {
            ds: {"train": len(split[ds]["train"]), "test": len(split[ds]["test"])}
            for ds in SCIPLEX
        },
    }


def main() -> int:
    base = load_json(BASE_SPLIT)
    meta = read_drug_meta()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("unseen_drug", "unseen_scaffold"):
        eval_drugs = mode_eval_drugs(meta, mode)
        train_drugs = set(meta) - eval_drugs
        split = json.loads(json.dumps(base))
        for ds in SCIPLEX:
            split[ds] = {
                "train": sorted(train_drugs),
                "test": sorted(eval_drugs),
            }
        out_file = OUT_DIR / f"split_seed42_xverse_chemical_{mode}_v1.json"
        out_file.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        row = summarize(split, meta, mode)
        leakage_ok = not row["drug_overlap"] and (mode == "unseen_drug" or not row["scaffold_overlap"])
        row.update(
            {
                "status": "ok" if leakage_ok else "fail",
                "split_file": str(out_file),
                "leakage_ok": leakage_ok,
                "notes": "Gene datasets retain base train-only cross-background split; SciPlex datasets are replaced with drug-level train/test holdout.",
            }
        )
        rows.append(row)
    status = "chemical_unseen_drug_scaffold_loader_splits_ready_no_gpu" if all(r["status"] == "ok" for r in rows) else "chemical_unseen_drug_scaffold_loader_splits_fail_no_gpu"
    payload = {
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_authorized": False,
        "boundary": {
            "task": "CPU-only loader split materializer",
            "uses_training": False,
            "uses_model_outputs": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "base_gene_split": str(BASE_SPLIT),
        },
        "rows": rows,
        "next_action": "dry-load loader splits and build train-only pert means before any GPU smoke" if status.endswith("ready_no_gpu") else "fix split leakage",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Chemical Unseen-Drug/Scaffold Loader Splits",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split materializer.",
        "- Gene datasets retain the train-only cross-background split.",
        "- SciPlex datasets use deterministic drug-level train/test holdouts.",
        "- No training, model outputs, canonical multi, or Track C query.",
        "",
        "| mode | status | train drugs | test drugs | train scaffolds | test scaffolds | drug overlap | scaffold overlap | split |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | `{row['status']}` | {row['train_drugs']} | {row['test_drugs']} | "
            f"{row['train_scaffolds']} | {row['test_scaffolds']} | {len(row['drug_overlap'])} | "
            f"{len(row['scaffold_overlap'])} | `{row['split_file']}` |"
        )
    lines += ["", "## JSON", "", f"`{OUT_JSON}`", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
