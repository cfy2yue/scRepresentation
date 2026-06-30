#!/usr/bin/env python3
"""Build train-only xverse scaling splits for staged LatentFM probes.

The generated splits keep the existing train-only validation groups fixed and
only subsample canonical-train-derived `train` conditions. They never read
canonical held-out outcomes or model artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_DIR = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624"
DEFAULT_ARTIFACT_DIR = ROOT / "runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_scaling_splits_v2_20260624.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SCALING_SPLITS_V2_20260624.md"

K562_BACKGROUND_DATASETS = {
    "Adamson",
    "DixitRegev2016_K562_TFs_High_MOI",
    "GasperiniShendure2019_lowMOI",
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_gwps",
}

TYPE_BALANCED_TARGETS = {
    "CRISPRi": 550,
    "drug": 250,
    "CRISPRa": 10_000,
    "CRISPRko": 10_000,
    "Cas13": 10_000,
    "unknown": 0,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_score(seed: int, ds: str, cond: str) -> str:
    raw = f"{seed}\t{ds}\t{cond}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def entry_for(metadata: dict[str, Any], ds: str, cond: str) -> dict[str, Any]:
    obj = (metadata.get(ds) or {}).get(cond) or {}
    return obj if isinstance(obj, dict) else {}


def perturbation_type(metadata: dict[str, Any], ds: str, cond: str) -> str:
    entry = entry_for(metadata, ds, cond)
    raw = str(entry.get("perturbation_type_raw", entry.get("perturbation_type", ""))).strip()
    if raw:
        return raw
    if "sciplex" in ds.lower():
        return "drug"
    return "unknown"


def is_drug(metadata: dict[str, Any], ds: str, cond: str) -> bool:
    ptype = perturbation_type(metadata, ds, cond).lower()
    return ptype in {"drug", "chemical", "compound", "small molecule", "small-molecule"}


def rank_conditions(
    base_train: list[str],
    *,
    ds: str,
    seed: int,
) -> list[str]:
    return sorted([str(c) for c in base_train], key=lambda c: stable_score(seed, ds, c))


def take_cap(conds: list[str], cap: int | None) -> list[str]:
    if cap is not None:
        conds = conds[: min(int(cap), len(conds))]
    return sorted(conds)


def condition_index_map(h5: h5py.File) -> dict[str, int]:
    out = {}
    for idx, cond in enumerate(h5["conditions"][:]):
        key = cond.decode("utf-8") if isinstance(cond, bytes) else str(cond)
        out[key] = int(idx)
    return out


def compute_train_pert_means(data_dir: Path, split: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    means: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for ds, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        if not train:
            audit.append(
                {
                    "dataset": ds,
                    "train_conditions_used": 0,
                    "train_cells_used": 0,
                    "n_missing_conditions": 0,
                    "status": "empty_train_dataset",
                }
            )
            continue
        h5_path = data_dir / f"{ds}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"missing dataset H5: {h5_path}")
        with h5py.File(h5_path, "r") as h5:
            cmap = condition_index_map(h5)
            offsets = h5["gt/offsets"][:]
            emb = h5["gt/emb"]
            total = None
            n_cells = 0
            used = 0
            missing = []
            for cond in train:
                idx = cmap.get(cond)
                if idx is None:
                    missing.append(cond)
                    continue
                lo = int(offsets[idx])
                hi = int(offsets[idx + 1])
                if hi <= lo:
                    continue
                arr = np.asarray(emb[lo:hi], dtype=np.float64)
                summed = arr.sum(axis=0, dtype=np.float64)
                total = summed if total is None else total + summed
                n_cells += int(arr.shape[0])
                used += 1
            if total is None or n_cells <= 0:
                raise ValueError(f"no train GT cells found for {ds}")
            means[ds] = (total / float(n_cells)).astype(np.float32)
            audit.append(
                {
                    "dataset": ds,
                    "train_conditions_used": used,
                    "train_cells_used": n_cells,
                    "n_missing_conditions": len(missing),
                    "missing_conditions": missing[:10],
                    "status": "ok",
                }
            )
    return means, audit


def summarize_split(
    split: dict[str, Any],
    metadata: dict[str, Any],
    *,
    arm: str,
    base_split: dict[str, Any],
) -> dict[str, Any]:
    ptype_counts: Counter[str] = Counter()
    dataset_counts: dict[str, int] = {}
    val_total = 0
    train_total = 0
    datasets_with_train = 0
    ood_val_datasets = 0
    train_subset_violations = []
    train_eval_overlap_violations = []
    for ds, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        test = [str(c) for c in groups.get("test") or []]
        base_groups = base_split.get(ds) or {}
        base_train = {str(c) for c in base_groups.get("train") or []}
        eval_groups = []
        for key, val in base_groups.items():
            if key == "train" or not isinstance(val, list):
                continue
            eval_groups.extend(str(c) for c in val)
        eval_set = set(eval_groups)
        train_set = set(train)
        if not train_set.issubset(base_train):
            train_subset_violations.append(ds)
        if train_set & eval_set:
            train_eval_overlap_violations.append(ds)
        train_total += len(train)
        val_total += len(test)
        dataset_counts[ds] = len(train)
        if train:
            datasets_with_train += 1
        elif test:
            ood_val_datasets += 1
        for cond in train:
            ptype_counts[perturbation_type(metadata, ds, cond)] += 1

    fixed_val_mismatches = []
    for ds, groups in sorted(base_split.items()):
        base_test = [str(c) for c in groups.get("test") or []]
        new_test = [str(c) for c in (split.get(ds) or {}).get("test") or []]
        if base_test != new_test:
            fixed_val_mismatches.append(ds)

    reasons = []
    if train_total <= 0:
        reasons.append("no_train_conditions")
    if train_subset_violations:
        reasons.append("train_not_subset_of_base_train")
    if train_eval_overlap_violations:
        reasons.append("train_overlaps_base_eval_groups")
    if fixed_val_mismatches:
        reasons.append("validation_groups_not_fixed")
    if arm == "gene_cap120_allbg" and datasets_with_train < 19:
        reasons.append("gene_allbg_datasets_with_train_lt_19")
    elif arm not in {"gene_cap120_allbg", "gene_cap120_k562bg"} and datasets_with_train < 20:
        reasons.append("non_background_arm_datasets_with_train_lt_20")
    if arm == "gene_cap120_k562bg" and ood_val_datasets <= 0:
        reasons.append("background_arm_has_no_ood_validation_datasets")
    if arm == "type_balanced_cap120":
        dominant_share = 0.0 if train_total <= 0 else max(ptype_counts.values() or [0]) / float(train_total)
        if train_total < 1000:
            reasons.append("type_balanced_train_conditions_lt_1000")
        if dominant_share > 0.55:
            reasons.append("type_balanced_dominant_type_share_gt_0p55")
        if ptype_counts.get("drug", 0) < 200:
            reasons.append("type_balanced_drug_conditions_lt_200")
    if max(dataset_counts.values() or [0]) > 120 and arm != "reference_full_trainonly":
        reasons.append("non_reference_dataset_count_exceeds_120")

    return {
        "arm": arm,
        "train_conditions": int(train_total),
        "validation_conditions": int(val_total),
        "datasets_with_train": int(datasets_with_train),
        "ood_val_datasets": int(ood_val_datasets),
        "perturbation_type_counts": dict(sorted(ptype_counts.items())),
        "dataset_train_counts": dataset_counts,
        "fixed_validation_group_mismatches": fixed_val_mismatches,
        "train_subset_violations": train_subset_violations,
        "train_eval_overlap_violations": train_eval_overlap_violations,
        "gate_status": "pass_cpu_split_gate" if not reasons else "fail_cpu_split_gate",
        "gate_reasons": reasons,
    }


def copy_groups_with_new_train(groups: dict[str, Any], train: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in groups.items():
        if isinstance(val, list):
            out[key] = [str(x) for x in val]
        else:
            out[key] = val
    out["train"] = train
    return out


def build_arms(base_split: dict[str, Any], metadata: dict[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {
        "cap30_all": {},
        "cap120_all": {},
        "type_balanced_cap120": {},
        "gene_cap120_allbg": {},
        "gene_cap120_k562bg": {},
    }
    cap120_by_dataset: dict[str, list[str]] = {}
    for ds, groups in sorted(base_split.items()):
        ds_s = str(ds)
        ranked_all = rank_conditions([str(c) for c in groups.get("train") or []], ds=ds_s, seed=seed)
        cap120 = take_cap(ranked_all, 120)
        cap120_by_dataset[ds_s] = cap120
        cap30 = take_cap(ranked_all, 30)
        gene_cap120 = [c for c in cap120 if not is_drug(metadata, ds_s, c)]
        k562_gene_cap120 = gene_cap120 if ds_s in K562_BACKGROUND_DATASETS else []
        arms["cap30_all"][ds_s] = copy_groups_with_new_train(groups, cap30)
        arms["cap120_all"][ds_s] = copy_groups_with_new_train(groups, sorted(gene_cap120 + [c for c in cap120 if c not in gene_cap120]))
        arms["gene_cap120_allbg"][ds_s] = copy_groups_with_new_train(groups, sorted(gene_cap120))
        arms["gene_cap120_k562bg"][ds_s] = copy_groups_with_new_train(groups, sorted(k562_gene_cap120))
    type_balanced = build_type_balanced_cap120(cap120_by_dataset, metadata, seed)
    for ds, groups in sorted(base_split.items()):
        arms["type_balanced_cap120"][str(ds)] = copy_groups_with_new_train(
            groups, sorted(type_balanced.get(str(ds), []))
        )
    return arms


def build_type_balanced_cap120(
    cap120_by_dataset: dict[str, list[str]],
    metadata: dict[str, Any],
    seed: int,
) -> dict[str, list[str]]:
    """Downsample the cap120 arm to reduce perturbation-type dominance.

    The arm stays a subset of cap120_all, so it changes training composition
    without expanding the condition universe or touching validation groups.
    """
    buckets: dict[str, list[tuple[str, str, str]]] = {}
    for ds, conds in sorted(cap120_by_dataset.items()):
        for cond in conds:
            ptype = perturbation_type(metadata, ds, cond)
            key = "drug" if ptype.lower() in {"drug", "chemical", "compound", "small molecule", "small-molecule"} else ptype
            buckets.setdefault(key, []).append((stable_score(seed + 17, ds, cond), ds, cond))
    selected: dict[str, set[str]] = {ds: set() for ds in cap120_by_dataset}
    for ptype, rows in sorted(buckets.items()):
        rows = sorted(rows)
        target = TYPE_BALANCED_TARGETS.get(ptype, len(rows))
        for _, ds, cond in rows[: min(len(rows), target)]:
            selected.setdefault(ds, set()).add(cond)

    # Keep broad dataset coverage if deterministic type downsampling drops a
    # small dataset entirely. This repair still selects only from cap120_all.
    for ds, conds in sorted(cap120_by_dataset.items()):
        if selected.get(ds):
            continue
        ranked = sorted((stable_score(seed + 29, ds, cond), cond) for cond in conds)
        if ranked:
            selected.setdefault(ds, set()).add(ranked[0][1])
    return {ds: sorted(conds) for ds, conds in selected.items()}


def train_set(split: dict[str, Any]) -> set[tuple[str, str]]:
    out = set()
    for ds, groups in split.items():
        for cond in groups.get("train") or []:
            out.add((str(ds), str(cond)))
    return out


def nested_checks(arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("cap30_all", "cap120_all"),
        ("type_balanced_cap120", "cap120_all"),
        ("gene_cap120_allbg", "cap120_all"),
        ("gene_cap120_k562bg", "gene_cap120_allbg"),
    ]
    rows = []
    for child, parent in pairs:
        child_set = train_set(arms[child])
        parent_set = train_set(arms[parent])
        missing = sorted(child_set - parent_set)[:10]
        union = child_set | parent_set
        rows.append(
            {
                "child": child,
                "parent": parent,
                "child_train_conditions": len(child_set),
                "parent_train_conditions": len(parent_set),
                "is_subset": not missing,
                "missing_examples": [f"{ds}:{cond}" for ds, cond in missing],
                "jaccard": float(len(child_set & parent_set) / max(1, len(union))),
            }
        )
    return rows


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Scaling Split Audit",
        "",
        f"Status: `{payload['overall_status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split construction.",
        "- Base validation groups are copied from the train-only cross-background split.",
        "- Train conditions are subsampled only from train-only `train` groups.",
        "- v2 uses a single arm-independent per-dataset ranking so smaller arms are nested in larger arms where scientifically intended.",
        "- Canonical held-out outcomes and Track C query data are not read.",
        "",
        "## Provenance",
        "",
        f"- base split: `{payload['base_split']}`",
        f"- data dir: `{payload['data_dir']}`",
        f"- condition metadata: `{payload['condition_metadata']}`",
        f"- output dir: `{payload['out_dir']}`",
        f"- seed: `{payload['seed']}`",
        f"- pert means computed: `{payload['compute_pert_means']}`",
        "",
        "## Arm Summary",
        "",
        "| arm | gate | train conds | val conds | datasets train | OOD val datasets | perturbation types | split | pert means |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["arms"]:
        ptypes = ", ".join(f"{k}:{v}" for k, v in row["summary"]["perturbation_type_counts"].items())
        means = row.get("pert_means_file") or ""
        lines.append(
            f"| `{row['arm']}` | `{row['summary']['gate_status']}` | "
            f"{row['summary']['train_conditions']} | {row['summary']['validation_conditions']} | "
            f"{row['summary']['datasets_with_train']} | {row['summary']['ood_val_datasets']} | "
            f"{ptypes} | `{row['split_file']}` | `{means}` |"
        )
    lines += [
        "",
        "## Nestedness Checks",
        "",
        "| child | parent | subset | child train | parent train | Jaccard |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["nested_checks"]:
        lines.append(
            f"| `{row['child']}` | `{row['parent']}` | `{row['is_subset']}` | "
            f"{row['child_train_conditions']} | {row['parent_train_conditions']} | "
            f"{row['jaccard']:.3f} |"
        )
    lines += [
        "",
        "## Intended Comparisons",
        "",
        "- Count scaling: `cap30_all` vs `cap120_all` vs the existing full train-only split.",
        "- Perturbation-type balancing: `type_balanced_cap120` vs `cap120_all`.",
        "- Perturbation-type scaling: `gene_cap120_allbg` vs `cap120_all`.",
        "- Background scaling: `gene_cap120_k562bg` vs `gene_cap120_allbg`.",
        "",
        "Gate interpretation:",
        "- `pass_cpu_split_gate` means the split artifact is leakage-safe and the intended nestedness checks pass.",
        "- GPU training still requires a fresh resource audit, detached launcher, RUN_STATUS, and frozen-checkpoint posthoc gate.",
        "",
    ]
    failed = [row for row in payload["arms"] if row["summary"]["gate_reasons"]]
    if failed:
        lines += ["## Gate Reasons", ""]
        for row in failed:
            reasons = ", ".join(f"`{r}`" for r in row["summary"]["gate_reasons"])
            lines.append(f"- `{row['arm']}`: {reasons}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, default=DEFAULT_BASE_SPLIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-pert-means", action="store_true")
    args = parser.parse_args()

    metadata_path = args.data_dir / "condition_metadata.json"
    base_split = load_json(args.base_split)
    metadata = load_json(metadata_path)
    arms = build_arms(base_split, metadata, int(args.seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    arm_payloads = []
    overall_reasons = []
    for arm, split in sorted(arms.items()):
        split_file = args.out_dir / f"split_seed42_xverse_trainonly_scaling_{arm}_v2.json"
        split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = summarize_split(split, metadata, arm=arm, base_split=base_split)
        pert_means_file = ""
        pert_mean_audit = []
        if args.compute_pert_means:
            means, pert_mean_audit = compute_train_pert_means(args.data_dir, split)
            pert_means_file = str(args.artifact_dir / f"xverse_trainonly_scaling_{arm}_v2_pert_means.npz")
            np.savez_compressed(pert_means_file, **means)
        if summary["gate_status"] != "pass_cpu_split_gate":
            overall_reasons.extend([f"{arm}:{r}" for r in summary["gate_reasons"]])
        arm_payloads.append(
            {
                "arm": arm,
                "split_file": str(split_file),
                "pert_means_file": pert_means_file,
                "summary": summary,
                "pert_mean_audit": pert_mean_audit,
            }
        )

    nesting = nested_checks(arms)
    for row in nesting:
        if not row["is_subset"]:
            overall_reasons.append(f"nestedness_failed:{row['child']}_not_subset_{row['parent']}")

    payload = {
        "base_split": str(args.base_split),
        "data_dir": str(args.data_dir),
        "condition_metadata": str(metadata_path),
        "out_dir": str(args.out_dir),
        "artifact_dir": str(args.artifact_dir),
        "seed": int(args.seed),
        "compute_pert_means": bool(args.compute_pert_means),
        "overall_status": "pass_cpu_split_gate" if not overall_reasons else "fail_cpu_split_gate",
        "overall_reasons": overall_reasons,
        "nested_checks": nesting,
        "arms": arm_payloads,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": payload["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
