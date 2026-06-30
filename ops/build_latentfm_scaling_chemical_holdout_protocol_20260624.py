#!/usr/bin/env python3
"""Build a train-only SciPlex chemical holdout split for scaling diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BIOFLOW = ROOT / "dataset/biFlow_data"
BASE_SPLIT = BIOFLOW / "split_seed42_xverse_trainonly_crossbg_val_v2.json"
CAP120_SPLIT = BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_DIR = BIOFLOW / "xverse_scaling_chemical_holdout_splits_20260624"
OUT_SPLIT = OUT_DIR / "split_seed42_xverse_scaling_cap120_chemical_holdout_v1.json"
OUT_JSON = ROOT / "reports/latentfm_scaling_chemical_holdout_protocol_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_CHEMICAL_HOLDOUT_PROTOCOL_20260624.md"

CHEMICAL_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_groups(groups: dict[str, Any]) -> dict[str, Any]:
    return {k: ([str(x) for x in v] if isinstance(v, list) else v) for k, v in groups.items()}


def main() -> int:
    base = load_json(BASE_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    split: dict[str, dict[str, Any]] = {}
    rows = []
    reasons = []
    for ds, groups in sorted(cap120.items()):
        out = copy_groups(groups)
        out["test"] = []
        out["test_single"] = []
        out["family_drug_trainonly_holdout"] = []
        if ds in CHEMICAL_DATASETS:
            base_train = set(str(x) for x in (base.get(ds) or {}).get("train") or [])
            cap_train = set(str(x) for x in groups.get("train") or [])
            canonical_ref = set(str(x) for x in (base.get(ds) or {}).get("canonical_test_reference") or [])
            holdout_set = base_train - cap_train
            holdout = sorted(holdout_set)
            if holdout_set & cap_train:
                reasons.append(f"{ds}:holdout_train_overlap")
            if holdout_set & canonical_ref:
                reasons.append(f"{ds}:holdout_canonical_reference_overlap")
            out["test"] = holdout
            out["test_single"] = holdout
            out["family_drug_trainonly_holdout"] = holdout
            rows.append(
                {
                    "dataset": ds,
                    "base_train": len(base_train),
                    "cap120_train": len(cap_train),
                    "holdout": len(holdout),
                    "canonical_reference_overlap": len(holdout_set & canonical_ref),
                    "examples": holdout[:10],
                }
            )
        split[str(ds)] = out

    train_eval_overlap = []
    for ds, groups in split.items():
        train = set(str(x) for x in groups.get("train") or [])
        for key in ("test", "test_single", "family_drug_trainonly_holdout"):
            overlap = train & set(str(x) for x in groups.get(key) or [])
            if overlap:
                train_eval_overlap.append(f"{ds}:{key}:{len(overlap)}")
    if train_eval_overlap:
        reasons.append("train_eval_overlap")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "chemical_holdout_protocol_ready_no_gpu" if not reasons else "chemical_holdout_protocol_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "train_only_parent": True,
            "canonical_reference_excluded": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "base_split": str(BASE_SPLIT),
            "cap120_split": str(CAP120_SPLIT),
        },
        "split_file": str(OUT_SPLIT),
        "rows": rows,
        "reasons": reasons,
        "total_holdout": sum(row["holdout"] for row in rows),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Chemical Holdout Protocol",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split/protocol artifact.",
        "- Chemical holdout is `base train-only parent train` minus `cap120 train`; canonical reference drugs are excluded.",
        "- Does not train, infer, launch GPU, read canonical multi, or read Track C query.",
        "",
        "## Rows",
        "",
        "| dataset | base train | cap120 train | chemical holdout | canonical reference overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | {row['base_train']} | {row['cap120_train']} | "
            f"{row['holdout']} | {row['canonical_reference_overlap']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- total chemical holdout conditions: `{payload['total_holdout']}`",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False` by this split builder alone.",
            "- Next allowed action: bounded train-only chemical eval of frozen anchor/candidate checkpoints with RUN_STATUS.",
            "",
            "## Output",
            "",
            f"- split: `{OUT_SPLIT}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "split": str(OUT_SPLIT), "total_holdout": payload["total_holdout"]}, indent=2))
    return 0 if status.endswith("ready_no_gpu") else 4


if __name__ == "__main__":
    raise SystemExit(main())
