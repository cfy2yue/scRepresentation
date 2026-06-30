#!/usr/bin/env python3
"""Build CPU-only split artifacts for the LatentFM scaling protocol gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BIOFLOW = ROOT / "dataset/biFlow_data"
BASE_SPLIT = BIOFLOW / "split_seed42_xverse_trainonly_crossbg_val_v2.json"
OUT_DIR = BIOFLOW / "xverse_scaling_protocol_splits_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_protocol_splits_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_PROTOCOL_SPLITS_20260624.md"
META_FILES = [
    ROOT / "dataset/raw/genepert_DE5000/metainfo.json",
    ROOT / "dataset/raw/chemicalpert_DE5000/metainfo.json",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in META_FILES:
        for row in load_json(path):
            out[str(row["dataset"])] = dict(row)
    return out


def stable_score(seed: int, ds: str, cond: str) -> str:
    return hashlib.sha256(f"{seed}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def rank_train(base: dict[str, Any], ds: str, seed: int) -> list[str]:
    return sorted(
        [str(c) for c in (base.get(ds) or {}).get("train", [])],
        key=lambda c: stable_score(seed, ds, c),
    )


def copy_with_train(groups: dict[str, Any], train: list[str]) -> dict[str, Any]:
    out = {k: ([str(x) for x in v] if isinstance(v, list) else v) for k, v in groups.items()}
    out["train"] = sorted(str(c) for c in train)
    return out


def is_primary_tracka(base: dict[str, Any], meta: dict[str, dict[str, Any]], ds: str) -> bool:
    if meta.get(ds, {}).get("perturbation_type") == "drug":
        return False
    item = base.get(ds) or {}
    return (
        len(item.get("internal_val_cross_background_seen_gene_proxy", [])) > 0
        and len(item.get("internal_val_family_gene_proxy", [])) > 0
    )


def capped_split(base: dict[str, Any], datasets: list[str], cap_by_dataset: dict[str, int], seed: int) -> dict[str, Any]:
    out = {}
    for ds, groups in sorted(base.items()):
        if ds not in datasets:
            out[ds] = copy_with_train(groups, [])
            continue
        ranked = rank_train(base, ds, seed)
        cap = max(0, int(cap_by_dataset.get(ds, 0)))
        out[ds] = copy_with_train(groups, ranked[: min(cap, len(ranked))])
    return out


def greedy_budget_counts(base: dict[str, Any], datasets: list[str], cap: int, budget: int, seed: int) -> dict[str, int]:
    counts = {ds: 0 for ds in datasets}
    # Round-robin across ranked datasets keeps matched-budget arms from becoming
    # a disguised single-dataset arm.
    active = [ds for ds in sorted(datasets, key=lambda d: stable_score(seed + 101, d, "__dataset__")) if rank_train(base, ds, seed)]
    remaining = int(budget)
    while remaining > 0 and active:
        progressed = False
        for ds in active:
            if remaining <= 0:
                break
            limit = min(cap, len(rank_train(base, ds, seed)))
            if counts[ds] < limit:
                counts[ds] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return counts


def summarize(split: dict[str, Any], meta: dict[str, dict[str, Any]], primary: list[str]) -> dict[str, Any]:
    counts = {ds: len((split.get(ds) or {}).get("train", [])) for ds in sorted(split)}
    active = [ds for ds, n in counts.items() if n > 0]
    total = sum(counts.values())
    ptypes: Counter[str] = Counter()
    bgs: Counter[str] = Counter()
    for ds in active:
        n = counts[ds]
        ptypes[str(meta.get(ds, {}).get("perturbation_type", "unknown"))] += n
        bgs[str(meta.get(ds, {}).get("cell_line", "unknown"))] += n
    violations = []
    for ds, groups in split.items():
        train = set(str(c) for c in groups.get("train", []))
        base_train = set(str(c) for c in (BASE_OBJ.get(ds) or {}).get("train", []))
        eval_set = set()
        for key, val in (BASE_OBJ.get(ds) or {}).items():
            if key != "train" and isinstance(val, list):
                eval_set.update(str(c) for c in val)
        if not train.issubset(base_train):
            violations.append(f"{ds}:train_not_subset_base")
        if train & eval_set:
            violations.append(f"{ds}:train_eval_overlap")
    return {
        "train_conditions": total,
        "active_datasets": len(active),
        "primary_datasets_with_train": sum(1 for ds in primary if counts.get(ds, 0) > 0),
        "min_train_per_active_dataset": min((counts[ds] for ds in active), default=0),
        "max_train_per_active_dataset": max((counts[ds] for ds in active), default=0),
        "max_dataset_share": max((n / total for n in counts.values()), default=0.0) if total else 0.0,
        "perturbation_type_counts": dict(sorted(ptypes.items())),
        "background_counts": dict(sorted(bgs.items())),
        "violations": violations,
        "gate_status": "pass_cpu_split_artifact_gate" if total > 0 and not violations else "fail_cpu_split_artifact_gate",
    }


BASE_OBJ: dict[str, Any] = {}


def main() -> int:
    global BASE_OBJ
    seed = 42
    BASE_OBJ = load_json(BASE_SPLIT)
    meta = load_meta()
    primary = [ds for ds in sorted(BASE_OBJ) if is_primary_tracka(BASE_OBJ, meta, ds)]

    top_by_size = sorted(primary, key=lambda ds: (-len(rank_train(BASE_OBJ, ds, seed)), ds))
    few = top_by_size[:4]
    mid = top_by_size[:8]
    many = primary

    arms: dict[str, dict[str, Any]] = {
        "cap60_primary19": capped_split(BASE_OBJ, primary, {ds: 60 for ds in primary}, seed),
        "breadth_few_deep_4ds_cap120_budget480": capped_split(
            BASE_OBJ, few, greedy_budget_counts(BASE_OBJ, few, 120, 480, seed), seed
        ),
        "breadth_mid_8ds_cap60_budget480": capped_split(
            BASE_OBJ, mid, greedy_budget_counts(BASE_OBJ, mid, 60, 480, seed), seed
        ),
        "breadth_many_shallow_19ds_cap30_budget480": capped_split(
            BASE_OBJ, many, greedy_budget_counts(BASE_OBJ, many, 30, 480, seed), seed
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, split in arms.items():
        path = OUT_DIR / f"split_seed42_xverse_scaling_protocol_{name}.json"
        path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({"arm": name, "split_file": str(path), "summary": summarize(split, meta, primary)})

    overall = "pass_cpu_split_artifact_gate"
    reasons = []
    totals = [r["summary"]["train_conditions"] for r in rows if r["arm"].startswith("breadth_")]
    if max(totals) - min(totals) > 24:
        overall = "fail_cpu_split_artifact_gate"
        reasons.append("breadth_budget_mismatch_gt_5pct")
    for row in rows:
        if row["summary"]["gate_status"] != "pass_cpu_split_artifact_gate":
            overall = "fail_cpu_split_artifact_gate"
            reasons.append(f"{row['arm']}:{row['summary']['gate_status']}")

    payload = {
        "status": overall,
        "boundary": {
            "read_split_json": True,
            "read_metainfo_json": True,
            "wrote_split_json": True,
            "computed_pert_means": False,
            "read_expression": False,
            "read_canonical_metrics": False,
            "read_trackc_query": False,
            "launched_gpu": False,
        },
        "base_split": str(BASE_SPLIT),
        "out_dir": str(OUT_DIR),
        "seed": seed,
        "primary_tracka_datasets": primary,
        "arms": rows,
        "reasons": reasons,
        "next_action": "compute pert-means in a detached CPU job only if these split artifacts are accepted for a GPU matrix",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Scaling Protocol Splits",
        "",
        f"Status: `{overall}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split artifact construction.",
        "- Reads train-only split JSON and local metainfo only.",
        "- Writes new protocol-specific split JSONs; does not compute pert means, read expression, read canonical metrics, read Track C query, or use GPU.",
        "",
        "## Arms",
        "",
        "| arm | gate | train conds | active datasets | primary datasets | min/max per active dataset | max ds share | split |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        s = row["summary"]
        lines.append(
            f"| `{row['arm']}` | `{s['gate_status']}` | {s['train_conditions']} | "
            f"{s['active_datasets']} | {s['primary_datasets_with_train']} | "
            f"{s['min_train_per_active_dataset']}/{s['max_train_per_active_dataset']} | "
            f"{s['max_dataset_share']:.3f} | `{row['split_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "These split artifacts pass the CPU split artifact gate, but they do not authorize GPU by themselves.",
            "Before any GPU matrix, compute train-only pert-mean artifacts and write a launcher/RUN_STATUS with exact resource audit and predeclared gates.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": overall, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
