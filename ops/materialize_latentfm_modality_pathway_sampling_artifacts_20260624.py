#!/usr/bin/env python3
"""Materialize accepted modality/pathway sampling split artifacts.

This is CPU-only artifact preparation after the metadata gate produced a
candidate split. It writes the accepted split under dataset/biFlow_data and
computes train-only pert means for that exact split. It does not read canonical
metrics, model outputs, Track C query, or launch GPU jobs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BIOFLOW = ROOT / "dataset/biFlow_data"
REPORTS = ROOT / "reports"
CANDIDATE_SPLIT = REPORTS / "latentfm_modality_pathway_sampling_candidate_split_20260624.json"
GATE_JSON = REPORTS / "latentfm_modality_pathway_sampling_gate_20260624.json"
OUT_DIR = BIOFLOW / "xverse_modality_pathway_sampling_splits_20260624"
OUT_SPLIT = OUT_DIR / "split_seed42_xverse_modality_pathway_quota12_cap120_parent.json"
OUT_ARTIFACT_DIR = ROOT / "runs/latentfm_modality_pathway_sampling_artifacts_20260624/artifacts"
OUT_PERT_MEANS = OUT_ARTIFACT_DIR / "pathway_quota12_cap120_parent_trainonly_pert_means.npz"
OUT_JSON = REPORTS / "latentfm_modality_pathway_sampling_artifacts_20260624.json"
OUT_MD = REPORTS / "LATENTFM_MODALITY_PATHWAY_SAMPLING_ARTIFACTS_20260624.md"
HELPER = ROOT / "ops/build_latentfm_xverse_scaling_splits_20260624.py"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper():
    spec = importlib.util.spec_from_file_location("xverse_scaling_split_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    gate = load_json(GATE_JSON)
    if gate.get("status") != "modality_pathway_sampling_gate_pass_candidate_design_no_immediate_gpu":
        raise SystemExit(f"pathway gate not in candidate state: {gate.get('status')!r}")
    split = load_json(CANDIDATE_SPLIT)
    if not isinstance(split, dict) or not split:
        raise SystemExit(f"candidate split is not a nonempty split dict: {CANDIDATE_SPLIT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")

    helper = load_helper()
    means, audit = helper.compute_train_pert_means(DATA_DIR, split)
    np.savez_compressed(OUT_PERT_MEANS, **means)
    bad = [row for row in audit if row.get("status") not in {"ok", "empty_train_dataset"}]
    status = "pass_artifacts_ready_for_bounded_gpu_smoke" if not bad else "fail_artifact_audit_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized_by_this_script": False,
        "boundary": {
            "read_candidate_split": str(CANDIDATE_SPLIT),
            "read_gate_json": str(GATE_JSON),
            "read_train_h5_gt_embeddings": True,
            "read_canonical_metrics": False,
            "read_trackc_query": False,
            "launched_gpu": False,
        },
        "split_file": str(OUT_SPLIT),
        "pert_means_file": str(OUT_PERT_MEANS),
        "n_datasets_with_means": len(means),
        "audit": audit,
        "bad_audit_rows": bad,
        "next_action": (
            "prepare one bounded train-only GPU smoke launcher/RUN_STATUS"
            if status == "pass_artifacts_ready_for_bounded_gpu_smoke"
            else "do not launch GPU; inspect bad audit rows"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Modality/Pathway Sampling Artifacts",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact materialization.",
        "- Reads candidate split and train H5 GT embeddings only.",
        "- Does not read canonical metrics, Track C query, or launch GPU.",
        "",
        "## Outputs",
        "",
        f"- split: `{OUT_SPLIT}`",
        f"- train-only pert means: `{OUT_PERT_MEANS}`",
        f"- datasets with means: `{len(means)}`",
        "",
        "## Decision",
        "",
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
