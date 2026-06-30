#!/usr/bin/env python3
"""CPU unlock gate for independent chemical unseen-scaffold V2 controls."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402


BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BIFLOW_DIR = ROOT / "dataset/biFlow_data"
DRUG_CACHE = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625"
DRUG_META = DRUG_CACHE / "drug_metadata.tsv"
HELPER = ROOT / "ops/build_latentfm_xverse_scaling_splits_20260624.py"
OUT_SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_chemical_unseen_scaffold_v2_splits_20260625"
OUT_ARTIFACT_DIR = ROOT / "runs/latentfm_chemical_unseen_scaffold_v2_cpu_unlock_20260625/artifacts"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_v2_cpu_unlock_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_CPU_UNLOCK_20260625.md"
SCIPLEX = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
SPLIT_SEEDS = (43, 44)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper():
    spec = importlib.util.spec_from_file_location("xverse_scaling_split_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stable_fraction(seed: int, scaffold: str) -> float:
    digest = hashlib.sha256(f"scaffold_v2\t{seed}\t{scaffold}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def read_drug_meta() -> dict[str, dict[str, str]]:
    with DRUG_META.open(newline="") as handle:
        return {r["drug"]: r for r in csv.DictReader(handle, delimiter="\t")}


def dryload(split: dict[str, Any], *, seed: int, max_batches: int = 2) -> dict[str, Any]:
    reasons: list[str] = []
    results: dict[str, Any] = {}
    for mode in ("train", "test"):
        ds = CrossDatasetFMDataset(
            str(DATA_DIR),
            split,
            batch_size=8,
            seed=seed,
            mode=mode,
            min_cells=8,
            ds_alpha=1.0,
            use_pert_condition=False,
            biflow_dir=str(BIFLOW_DIR),
            latent_backbone="xverse",
            perturbation_family_filter="all",
            silent=True,
        )
        checked = 0
        iterator = iter(ds)
        for _ in range(max_batches):
            batch = next(iterator)
            src, gt, ds_name, cond_name = batch[:4]
            if tuple(src.shape) != tuple(gt.shape):
                reasons.append(f"{mode}_src_gt_shape_mismatch")
            if src.ndim != 2 or gt.ndim != 2 or src.shape[1] != 384 or gt.shape[1] != 384:
                reasons.append(f"{mode}_bad_embedding_shape")
            if not ds_name or not cond_name:
                reasons.append(f"{mode}_empty_name")
            checked += 1
        results[mode] = {
            "total_conditions": int(ds.total_conditions),
            "epoch_steps": int(ds.epoch_steps),
            "batches_checked": checked,
        }
        if ds.total_conditions <= 0 or ds.epoch_steps <= 0:
            reasons.append(f"{mode}_empty_dataset")
    return {"status": "ok" if not reasons else "fail", "reasons": reasons[:20], "results": results}


def build_split(seed: int, meta: dict[str, dict[str, str]], base: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scaffolds = sorted({r["scaffold"] for r in meta.values()})
    eval_scaffolds = {s for s in scaffolds if stable_fraction(seed, s) < 0.20}
    eval_drugs = {d for d, r in meta.items() if r["scaffold"] in eval_scaffolds}
    train_drugs = set(meta) - eval_drugs
    split = json.loads(json.dumps(base))
    for ds in SCIPLEX:
        split[ds] = {"train": sorted(train_drugs), "test": sorted(eval_drugs)}
    train_scaffolds = {meta[d]["scaffold"] for d in train_drugs}
    test_scaffolds = {meta[d]["scaffold"] for d in eval_drugs}
    test_pathways = Counter(meta[d]["pathways"] or "unknown" for d in eval_drugs)
    summary = {
        "split_seed": seed,
        "train_drugs": len(train_drugs),
        "test_drugs": len(eval_drugs),
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "drug_overlap": len(train_drugs & eval_drugs),
        "scaffold_overlap": len(train_scaffolds & test_scaffolds),
        "test_pathways": dict(test_pathways.most_common()),
        "sciplex_dataset_counts": {ds: {"train": len(train_drugs), "test": len(eval_drugs)} for ds in SCIPLEX},
    }
    return split, summary


def copy_text_artifacts(dst: Path, manifest_note: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("drug_index.tsv", "drug_index.json", "drug_metadata.tsv"):
        shutil.copy2(DRUG_CACHE / name, dst / name)
    manifest = load_json(DRUG_CACHE / "manifest.json")
    manifest["source"] = dst.name
    manifest["control_note"] = manifest_note
    manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["artifact_files"] = {
        "embeddings": str(dst / "drug_embeddings.npy"),
        "index_json": str(dst / "drug_index.json"),
        "index_tsv": str(dst / "drug_index.tsv"),
        "metadata_tsv": str(dst / "drug_metadata.tsv"),
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_control_caches() -> list[dict[str, Any]]:
    src = np.load(DRUG_CACHE / "drug_embeddings.npy")
    controls = []
    drug_rows = np.arange(2, src.shape[0], dtype=np.int64)

    shuffled_dir = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625_shuffled_control"
    copy_text_artifacts(shuffled_dir, "Drug rows are permuted while index labels stay fixed; pad/unk rows unchanged.")
    rng = np.random.default_rng(20260625)
    shuffled = np.asarray(src, dtype=np.float32).copy()
    perm = drug_rows.copy()
    rng.shuffle(perm)
    shuffled[drug_rows] = shuffled[perm]
    np.save(shuffled_dir / "drug_embeddings.npy", shuffled.astype(np.float32))
    controls.append({"name": "shuffled_morgan512", "cache_dir": str(shuffled_dir), "shape": list(shuffled.shape)})

    random_dir = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625_random_control"
    copy_text_artifacts(random_dir, "Drug rows are deterministic random Gaussian controls; pad/unk rows unchanged.")
    rng = np.random.default_rng(20260626)
    random_emb = np.asarray(src, dtype=np.float32).copy()
    random_emb[drug_rows] = rng.standard_normal((len(drug_rows), src.shape[1])).astype(np.float32)
    np.save(random_dir / "drug_embeddings.npy", random_emb.astype(np.float32))
    controls.append({"name": "random_morgan512", "cache_dir": str(random_dir), "shape": list(random_emb.shape)})
    return controls


def main() -> int:
    base = load_json(BASE_SPLIT)
    meta = read_drug_meta()
    helper = load_helper()
    OUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SPLIT_SEEDS:
        split, summary = build_split(seed, meta, base)
        split_file = OUT_SPLIT_DIR / f"split_seed{seed}_xverse_chemical_unseen_scaffold_v2.json"
        split_file.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dry = dryload(split, seed=seed)
        means, audit = helper.compute_train_pert_means(DATA_DIR, split)
        pert_file = OUT_ARTIFACT_DIR / f"unseen_scaffold_v2_seed{seed}_trainonly_pert_means.npz"
        np.savez_compressed(pert_file, **means)
        pass_minima = {
            "zero_drug_overlap": summary["drug_overlap"] == 0,
            "zero_scaffold_overlap": summary["scaffold_overlap"] == 0,
            "test_drugs_ge_20": summary["test_drugs"] >= 20,
            "test_scaffolds_ge_20": summary["test_scaffolds"] >= 20,
            "train_drugs_ge_100": summary["train_drugs"] >= 100,
            "dryload_ok": dry["status"] == "ok",
            "pert_means_all_ok": all(r.get("status") in {"ok", "empty_train_dataset"} for r in audit),
        }
        rows.append(
            {
                **summary,
                "status": "ok" if all(pass_minima.values()) else "fail",
                "split_file": str(split_file),
                "dryload": dry,
                "pert_means_file": str(pert_file),
                "n_datasets_with_means": len(means),
                "pass_minima": pass_minima,
            }
        )
    controls = build_control_caches()
    control_ok = all(Path(c["cache_dir"], "drug_embeddings.npy").is_file() for c in controls)
    status = "chemical_unseen_scaffold_v2_cpu_unlock_ready_protocol_next_no_gpu" if all(r["status"] == "ok" for r in rows) and control_ok else "chemical_unseen_scaffold_v2_cpu_unlock_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "task": "CPU-only independent scaffold split and negative-control artifact gate",
            "uses_training": False,
            "uses_model_outputs": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
        },
        "rows": rows,
        "control_caches": controls,
        "next_action": (
            "external review and fixed-step launcher/negative-control protocol before any GPU"
            if status.endswith("protocol_next_no_gpu")
            else "fix split/control artifact failures; do not launch GPU"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Chemical Unseen-Scaffold V2 CPU Unlock",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only independent scaffold split, dry-load, train-only pert-means, and control-cache gate.",
        "- No training, model outputs, canonical multi, or Track C query.",
        "- This report does not authorize GPU by itself.",
        "",
        "## Splits",
        "",
        "| seed | status | train drugs | test drugs | train scaffolds | test scaffolds | scaffold overlap | train conds | test conds |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        train = row["dryload"]["results"]["train"]
        test = row["dryload"]["results"]["test"]
        lines.append(
            f"| {row['split_seed']} | `{row['status']}` | {row['train_drugs']} | {row['test_drugs']} | "
            f"{row['train_scaffolds']} | {row['test_scaffolds']} | {row['scaffold_overlap']} | "
            f"{train['total_conditions']} | {test['total_conditions']} |"
        )
    lines += [
        "",
        "## Control Caches",
        "",
        "| control | cache | shape |",
        "|---|---|---|",
    ]
    for c in controls:
        lines.append(f"| `{c['name']}` | `{c['cache_dir']}` | `{c['shape']}` |")
    lines += [
        "",
        "## Decision",
        "",
        "- GPU authorized: `False`",
        f"- next action: {payload['next_action']}",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
