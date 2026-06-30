#!/usr/bin/env python3
"""Materialize a chemical-count matched random control for pathway sampling.

The pathway-quota candidate changes chemical composition within sciplex while
leaving gene/non-chemical cap120 coverage untouched. This control keeps exactly
the same number of chemical train conditions per sciplex dataset as the
pathway-quota candidate, but samples those drugs from the cap120 parent without
using pathway labels. It is a train-only negative-control split; it does not
read model outputs, canonical metrics, or Track C query data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BIOFLOW = ROOT / "dataset/biFlow_data"
REPORTS = ROOT / "reports"
CAP120_SPLIT = BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
CANDIDATE_SPLIT = (
    BIOFLOW
    / "xverse_modality_pathway_sampling_splits_20260624"
    / "split_seed42_xverse_modality_pathway_quota12_cap120_parent.json"
)
OUT_DIR = BIOFLOW / "xverse_modality_pathway_sampling_splits_20260624"
OUT_SPLIT = OUT_DIR / "split_seed42_xverse_modality_pathway_randomcount_cap120_parent.json"
OUT_ARTIFACT_DIR = ROOT / "runs/latentfm_modality_pathway_sampling_artifacts_20260624/artifacts"
OUT_PERT_MEANS = OUT_ARTIFACT_DIR / "pathway_randomcount_cap120_parent_trainonly_pert_means.npz"
OUT_JSON = REPORTS / "latentfm_modality_pathway_randomcount_control_artifacts_20260624.json"
OUT_MD = REPORTS / "LATENTFM_MODALITY_PATHWAY_RANDOMCOUNT_CONTROL_ARTIFACTS_20260624.md"
HELPER = ROOT / "ops/build_latentfm_xverse_scaling_splits_20260624.py"

CHEMICAL_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}
SEED = 20260624 + 17


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper():
    spec = importlib.util.spec_from_file_location("xverse_scaling_split_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stable_score(ds: str, cond: str) -> str:
    return hashlib.sha256(f"{SEED}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def copy_groups(groups: dict[str, Any], train: list[str]) -> dict[str, Any]:
    out = {k: ([str(x) for x in v] if isinstance(v, list) else v) for k, v in groups.items()}
    out["train"] = sorted(str(x) for x in train)
    return out


def jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    return float(len(a & b) / max(1, len(a | b)))


def train_set(split: dict[str, Any], chemical_only: bool = False) -> set[tuple[str, str]]:
    out = set()
    for ds, groups in split.items():
        if chemical_only and str(ds) not in CHEMICAL_DATASETS:
            continue
        for cond in groups.get("train") or []:
            out.add((str(ds), str(cond)))
    return out


def build_randomcount(cap120: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds, groups in sorted(cap120.items()):
        ds_s = str(ds)
        cap_train = [str(x) for x in groups.get("train") or []]
        if ds_s not in CHEMICAL_DATASETS:
            out[ds_s] = copy_groups(groups, cap_train)
            continue
        n = len((candidate.get(ds_s) or {}).get("train") or [])
        ranked = sorted(cap_train, key=lambda cond: stable_score(ds_s, cond))
        out[ds_s] = copy_groups(groups, ranked[:n])
    return out


def safety(split: dict[str, Any], cap120: dict[str, Any]) -> list[str]:
    reasons = []
    for ds, groups in split.items():
        train = {str(x) for x in groups.get("train") or []}
        cap_groups = cap120.get(ds) or {}
        cap_train = {str(x) for x in cap_groups.get("train") or []}
        eval_set = set()
        for key, val in cap_groups.items():
            if key != "train" and isinstance(val, list):
                eval_set.update(str(x) for x in val)
        if not train.issubset(cap_train):
            reasons.append(f"{ds}:train_not_subset_cap120")
        if train & eval_set:
            reasons.append(f"{ds}:train_eval_overlap")
        for key, val in cap_groups.items():
            if key == "train" or not isinstance(val, list):
                continue
            if [str(x) for x in groups.get(key, [])] != [str(x) for x in val]:
                reasons.append(f"{ds}:{key}_validation_changed")
                break
    return reasons


def main() -> int:
    cap120 = load_json(CAP120_SPLIT)
    candidate = load_json(CANDIDATE_SPLIT)
    split = build_randomcount(cap120, candidate)
    reasons = safety(split, cap120)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")

    helper = load_helper()
    means, audit = helper.compute_train_pert_means(DATA_DIR, split)
    np.savez_compressed(OUT_PERT_MEANS, **means)
    bad = [row for row in audit if row.get("status") not in {"ok", "empty_train_dataset"}]
    if bad:
        reasons.append("pert_mean_audit_bad_rows")
    status = "pass_randomcount_control_ready_for_bounded_gpu_smoke" if not reasons else "fail_randomcount_control_no_gpu"
    candidate_all = train_set(candidate)
    random_all = train_set(split)
    payload = {
        "status": status,
        "gpu_authorized_by_this_script": False,
        "boundary": {
            "read_cap120_split": str(CAP120_SPLIT),
            "read_candidate_split": str(CANDIDATE_SPLIT),
            "read_train_h5_gt_embeddings": True,
            "read_model_outputs": False,
            "read_canonical_metrics": False,
            "read_trackc_query": False,
            "launched_gpu": False,
        },
        "split_file": str(OUT_SPLIT),
        "pert_means_file": str(OUT_PERT_MEANS),
        "chemical_counts": {
            ds: len((split.get(ds) or {}).get("train") or []) for ds in sorted(CHEMICAL_DATASETS)
        },
        "jaccard_vs_pathway_candidate_all": jaccard(random_all, candidate_all),
        "jaccard_vs_pathway_candidate_chemical": jaccard(
            train_set(split, chemical_only=True), train_set(candidate, chemical_only=True)
        ),
        "n_datasets_with_means": len(means),
        "reasons": reasons,
        "audit": audit,
        "next_action": (
            "launch one bounded train-only randomcount control smoke"
            if status == "pass_randomcount_control_ready_for_bounded_gpu_smoke"
            else "do not launch GPU; inspect reasons"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Modality/Pathway Random-Count Control Artifacts",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only negative-control artifact generation.",
        "- Keeps gene/non-chemical cap120 train coverage and matches pathway candidate chemical counts.",
        "- Does not read model outputs, canonical metrics, Track C query, or launch GPU.",
        "",
        "## Outputs",
        "",
        f"- split: `{OUT_SPLIT}`",
        f"- train-only pert means: `{OUT_PERT_MEANS}`",
        f"- chemical counts: `{payload['chemical_counts']}`",
        f"- chemical Jaccard vs pathway candidate: `{payload['jaccard_vs_pathway_candidate_chemical']:.3f}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
