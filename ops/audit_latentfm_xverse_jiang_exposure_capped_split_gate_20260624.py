#!/usr/bin/env python3
"""Build/audit a Jiang exposure-capped xverse split as a CPU-only gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BIFLOW = ROOT / "dataset/biFlow_data"
SPLIT_DIR = BIFLOW / "xverse_scaling_splits_v2_20260624"
BASE_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json"
OUT_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_jiang_exposure_capped_v2.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_jiang_exposure_capped_split_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_JIANG_EXPOSURE_CAPPED_SPLIT_GATE_20260624.md"

JIANG_DATASETS = {"Jiang_IFNB", "Jiang_IFNG", "Jiang_INS", "Jiang_TGFB", "Jiang_TNFA"}
PRESERVE_DATASETS = {"NormanWeissman2019_filtered", "Wessels"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_key(seed: int, ds: str, cond: str) -> str:
    return hashlib.sha256(f"{seed}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def condition_sizes(ds: str) -> dict[str, int]:
    with h5py.File(DATA_DIR / f"{ds}.h5", "r") as h5:
        conds = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in h5["conditions"][:]]
        offsets = h5["gt/offsets"][:]
        return {cond: int(offsets[i + 1] - offsets[i]) for i, cond in enumerate(conds)}


def n_eff(n: int, ds_alpha: float = 0.7) -> int:
    if n <= 0:
        return 0
    return max(1, min(n, int(math.ceil(n**ds_alpha))))


def split_exposure(split: dict[str, Any]) -> dict[str, Any]:
    rows = []
    total = 0.0
    for ds, groups in sorted(split.items()):
        sizes = condition_sizes(str(ds))
        train = [str(c) for c in groups.get("train") or [] if str(c) in sizes]
        selected = n_eff(len(train))
        visits = [max(1, math.ceil(sizes[c] / 64)) for c in train]
        avg_visit = sum(visits) / max(1, len(visits))
        steps = float(selected) * float(avg_visit)
        total += steps
        rows.append(
            {
                "dataset": str(ds),
                "train_conditions": len(train),
                "selected_conditions_per_epoch": selected,
                "epoch_steps_est": steps,
            }
        )
    for row in rows:
        row["epoch_step_share"] = row["epoch_steps_est"] / max(1.0, total)
    jiang_share = sum(row["epoch_step_share"] for row in rows if row["dataset"] in JIANG_DATASETS)
    max_share = max((row["epoch_step_share"] for row in rows), default=0.0)
    entropy = 0.0
    for row in rows:
        p = row["epoch_step_share"]
        if p > 0:
            entropy -= p * math.log(p)
    return {
        "rows": rows,
        "epoch_steps_est": total,
        "jiang_epoch_step_share": jiang_share,
        "max_dataset_epoch_step_share": max_share,
        "normalized_dataset_step_entropy": entropy / math.log(max(2, len(rows))),
    }


def perturbation_type(metadata: dict[str, Any], ds: str, cond: str) -> str:
    entry = ((metadata.get(ds) or {}).get(cond) or {})
    raw = str(entry.get("perturbation_type_raw", entry.get("perturbation_type", ""))).strip()
    if raw:
        return raw
    if "sciplex" in ds.lower():
        return "drug"
    return "unknown"


def ptype_counts(split: dict[str, Any], metadata: dict[str, Any]) -> dict[str, int]:
    out = Counter()
    for ds, groups in split.items():
        for cond in groups.get("train") or []:
            out[perturbation_type(metadata, str(ds), str(cond))] += 1
    return dict(sorted(out.items()))


def bg_counts_for_conditions(ds: str, conditions: list[str]) -> dict[str, int]:
    path = BIFLOW / "gt_stack" / f"{ds}.h5ad"
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs[["perturbation", "cell_type"]].copy()
        obs["perturbation"] = obs["perturbation"].astype(str)
        obs["cell_type"] = obs["cell_type"].astype(str)
        mask = obs["perturbation"].isin(set(conditions))
        counts = obs.loc[mask, "cell_type"].value_counts().to_dict()
        return {str(k): int(v) for k, v in sorted(counts.items())}
    finally:
        a.file.close()


def js_divergence(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    pa = np.array([a.get(k, 0) for k in keys], dtype=np.float64)
    pb = np.array([b.get(k, 0) for k in keys], dtype=np.float64)
    if pa.sum() <= 0 or pb.sum() <= 0:
        return 1.0
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    def kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def jiang_bg_audit(split: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ds in sorted(JIANG_DATASETS):
        groups = split.get(ds) or {}
        train = [str(c) for c in groups.get("train") or []]
        eval_conds = sorted(
            {
                str(c)
                for key in (
                    "test",
                    "test_single",
                    "internal_val_cross_background_seen_gene_proxy",
                    "internal_val_family_gene_proxy",
                )
                for c in (groups.get(key) or [])
            }
        )
        train_counts = bg_counts_for_conditions(ds, train)
        eval_counts = bg_counts_for_conditions(ds, eval_conds)
        total = sum(train_counts.values())
        rows.append(
            {
                "dataset": ds,
                "train_conditions": len(train),
                "train_background_counts": train_counts,
                "eval_background_counts": eval_counts,
                "train_max_background_share": 0.0 if total <= 0 else max(train_counts.values() or [0]) / total,
                "train_eval_js_divergence": js_divergence(train_counts, eval_counts),
            }
        )
    return rows


def clone_with_removed(split: dict[str, Any], removed: set[tuple[str, str]]) -> dict[str, Any]:
    out = {}
    for ds, groups in sorted(split.items()):
        new = {}
        for key, val in groups.items():
            if isinstance(val, list):
                values = [str(x) for x in val]
                if key == "train":
                    values = [c for c in values if (str(ds), c) not in removed]
                new[key] = values
            else:
                new[key] = val
        out[str(ds)] = new
    return out


def build_candidate(base: dict[str, Any], *, seed: int) -> dict[str, Any]:
    sizes = {ds: condition_sizes(ds) for ds in JIANG_DATASETS}
    removed: set[tuple[str, str]] = set()
    current = clone_with_removed(base, removed)
    for _ in range(500):
        exp = split_exposure(current)
        max_row = max(exp["rows"], key=lambda r: r["epoch_step_share"])
        if exp["jiang_epoch_step_share"] <= 0.45 and max_row["epoch_step_share"] <= 0.12:
            return current
        candidate_datasets = [r for r in exp["rows"] if r["dataset"] in JIANG_DATASETS and r["train_conditions"] > 3]
        if not candidate_datasets:
            return current
        # Prefer removing from the highest-share Jiang dataset. Within it,
        # remove the largest-cell train condition first to reduce microstep share.
        target_ds = max(candidate_datasets, key=lambda r: r["epoch_step_share"])["dataset"]
        train = [str(c) for c in (current.get(target_ds) or {}).get("train") or []]
        if len(train) <= 3:
            return current
        removable = sorted(
            train,
            key=lambda c: (-sizes[target_ds].get(c, 0), stable_key(seed, target_ds, c)),
        )
        removed.add((target_ds, removable[0]))
        current = clone_with_removed(base, removed)
    return current


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-split", type=Path, default=OUT_SPLIT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    base = read_json(BASE_SPLIT)
    metadata = read_json(DATA_DIR / "condition_metadata.json")
    candidate = build_candidate(base, seed=int(args.seed))
    base_exp = split_exposure(base)
    cand_exp = split_exposure(candidate)
    bg_rows = jiang_bg_audit(candidate)
    train_conditions = sum(len(groups.get("train") or []) for groups in candidate.values())
    datasets_with_train = sum(1 for groups in candidate.values() if groups.get("train"))
    preserve_missing = []
    for ds in PRESERVE_DATASETS:
        base_train = set(str(c) for c in (base.get(ds) or {}).get("train") or [])
        cand_train = set(str(c) for c in (candidate.get(ds) or {}).get("train") or [])
        preserve_missing.extend(f"{ds}:{c}" for c in sorted(base_train - cand_train))

    reasons = []
    if train_conditions < 900:
        reasons.append("train_conditions_lt_900")
    if datasets_with_train < 20:
        reasons.append("datasets_with_train_lt_20")
    if preserve_missing:
        reasons.append("norman_wessels_not_preserved")
    if cand_exp["jiang_epoch_step_share"] > 0.45:
        reasons.append("jiang_epoch_step_share_gt_0p45")
    if cand_exp["max_dataset_epoch_step_share"] > 0.12:
        reasons.append("max_dataset_epoch_step_share_gt_0p12")
    for row in bg_rows:
        if row["train_eval_js_divergence"] > 0.10:
            reasons.append(f"{row['dataset']}:train_eval_js_gt_0p10")
        if row["train_max_background_share"] > 0.30:
            reasons.append(f"{row['dataset']}:train_max_bg_share_gt_0p30")

    status = "jiang_exposure_capped_split_gate_pass" if not reasons else "jiang_exposure_capped_split_gate_fail"
    if status == "jiang_exposure_capped_split_gate_pass":
        args.out_split.parent.mkdir(parents=True, exist_ok=True)
        args.out_split.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "status": status,
        "reasons": reasons,
        "boundary": {
            "uses": ["type_balanced train-only split", "condition_metadata perturbation types", "h5 GT offsets", "Jiang h5ad obs perturbation/cell_type"],
            "forbidden": ["expression matrices", "canonical outcomes", "Track C query", "model outputs", "active logs", "posthoc predictions"],
        },
        "files": {
            "base_split": str(BASE_SPLIT),
            "candidate_split": str(args.out_split) if status == "jiang_exposure_capped_split_gate_pass" else "",
        },
        "base": {
            "train_conditions": sum(len(g.get("train") or []) for g in base.values()),
            "perturbation_type_counts": ptype_counts(base, metadata),
            "exposure": {k: v for k, v in base_exp.items() if k != "rows"},
        },
        "candidate": {
            "train_conditions": train_conditions,
            "datasets_with_train": datasets_with_train,
            "perturbation_type_counts": ptype_counts(candidate, metadata),
            "exposure": {k: v for k, v in cand_exp.items() if k != "rows"},
            "jiang_background_rows": bg_rows,
        },
        "removed_conditions": {
            ds: sorted(set((base.get(ds) or {}).get("train") or []) - set((candidate.get(ds) or {}).get("train") or []))
            for ds in sorted(JIANG_DATASETS)
        },
        "preserve_missing": preserve_missing[:30],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md)}, indent=2))
    return 0


def render_md(payload: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)
    lines = [
        "# LatentFM xverse Jiang Exposure-Capped Split Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split-composition gate over the type-balanced train-only split.",
        "- Uses h5 GT offsets for exposure and Jiang `.obs` perturbation/cell_type for background distribution checks.",
        "- Does not read expression matrices, canonical outcomes, Track C query, model outputs, active logs, or posthoc predictions.",
        "",
        "## Summary",
        "",
        "| arm | train conds | Jiang step share | max dataset step share | entropy | ptypes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in ("base", "candidate"):
        row = payload[name]
        ptypes = ", ".join(f"{k}:{v}" for k, v in row["perturbation_type_counts"].items())
        lines.append(
            f"| `{name}` | {row['train_conditions']} | {f(row['exposure']['jiang_epoch_step_share'])} | "
            f"{f(row['exposure']['max_dataset_epoch_step_share'])} | {f(row['exposure']['normalized_dataset_step_entropy'])} | {ptypes} |"
        )
    lines += [
        "",
        "## Jiang Background Checks",
        "",
        "| dataset | train conds | train/eval JS | train max background share |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["candidate"]["jiang_background_rows"]:
        lines.append(
            f"| `{row['dataset']}` | {row['train_conditions']} | "
            f"{f(row['train_eval_js_divergence'])} | {f(row['train_max_background_share'])} |"
        )
    if payload["reasons"]:
        lines += ["", "Gate reasons:"]
        lines.extend(f"- `{reason}`" for reason in payload["reasons"])
    lines += [
        "",
        "## Candidate Split",
        "",
        f"`{payload['files']['candidate_split'] or 'not written because gate failed'}`",
        "",
        "## JSON",
        "",
        f"`{payload['out_json']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
