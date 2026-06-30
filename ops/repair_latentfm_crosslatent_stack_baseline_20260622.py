#!/usr/bin/env python3
"""Repair the failed stack row in the cross-latent train-only baseline build.

The first build produced all train-only pert means and successful
scfoundation/scldm baseline gates, but stack failed before the gate because its
manifest omitted ``condition_metadata_file``. After fixing the gate script's
metadata fallback, this repair reruns only stack and updates the combined
baseline summary.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BUILD_SCRIPT = ROOT / "ops/build_latentfm_crosslatent_trainonly_baselines_20260622.py"
GENE_GATE_SCRIPT = ROOT / "ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DATA_DIR = ROOT / "dataset/latentfm_full/stack"
BASELINE_RUN = ROOT / "runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622"
PERT_MEANS = BASELINE_RUN / "artifacts/stack_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
COMBINED_JSON = ROOT / "reports/latentfm_crosslatent_tracka_trainonly_baselines_20260622.json"
COMBINED_MD = ROOT / "reports/LATENTFM_CROSSLATENT_TRACKA_TRAINONLY_BASELINES_20260622.md"
STACK_JSON = ROOT / "reports/latentfm_crosslatent_stack_gene_reliability_router_gate_20260622.json"
STACK_MD = ROOT / "reports/LATENTFM_CROSSLATENT_STACK_GENE_RELIABILITY_ROUTER_GATE_20260622.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_build_module() -> Any:
    spec = importlib.util.spec_from_file_location("crosslatent_baseline_build", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_stack_gate() -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(GENE_GATE_SCRIPT),
        "--data-dir",
        str(DATA_DIR),
        "--split-file",
        str(SPLIT),
        "--pert-means-file",
        str(PERT_MEANS),
        "--out-json",
        str(STACK_JSON),
        "--out-md",
        str(STACK_MD),
        "--max-train-per-dataset",
        "768",
        "--max-cells-per-condition",
        "256",
        "--n-boot",
        "2000",
        "--seed",
        "42",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = load_json(STACK_JSON) if STACK_JSON.is_file() else None
    return {
        "latent": "stack",
        "cmd": cmd,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "out_json": str(STACK_JSON),
        "out_md": str(STACK_MD),
        "decision": None if payload is None else payload.get("decision"),
        "selected_model": None if payload is None else payload.get("selected_model"),
        "n_train_rows": None if payload is None else payload.get("n_train_rows"),
        "n_val_rows": None if payload is None else payload.get("n_val_rows"),
    }


def main() -> int:
    for required in (GENE_GATE_SCRIPT, BUILD_SCRIPT, SPLIT, DATA_DIR / "manifest.json", PERT_MEANS, COMBINED_JSON):
        if not required.exists():
            raise FileNotFoundError(required)

    combined = load_json(COMBINED_JSON)
    gate = run_stack_gate()
    found = False
    for row in combined.get("latents", []):
        if row.get("latent") == "stack":
            row["baseline_gate"] = gate
            found = True
            break
    if not found:
        raise RuntimeError("combined baseline JSON has no stack row")

    failures = []
    for row in combined.get("latents", []):
        if row.get("pert_mean_status") != "ok":
            failures.append(f"{row.get('latent')}: pert means failed")
            continue
        gate_row = row.get("baseline_gate") or {}
        if gate_row.get("returncode") != 0:
            failures.append(f"{row.get('latent')}: baseline gate failed")

    if failures:
        combined["status"] = "crosslatent_baseline_build_failed"
        combined["recommended_action"] = "inspect_logs_before_gpu_comparator"
        combined["reason"] = "; ".join(failures)
    else:
        combined["status"] = "crosslatent_trainonly_baselines_ready_for_protocol_review"
        combined["recommended_action"] = "review_baseline_reports_then_decide_detached_gpu_anchor_comparator"
        combined["reason"] = "all comparator train-only pert means and baseline gates were generated"
    combined["repair_note"] = (
        "stack baseline gate repaired after adding condition_metadata.json "
        "fallback for manifests without condition_metadata_file"
    )

    build = load_build_module()
    COMBINED_JSON.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    COMBINED_MD.write_text(build.render(combined), encoding="utf-8")
    print(COMBINED_MD)
    print(COMBINED_JSON)
    print(combined["status"])
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
