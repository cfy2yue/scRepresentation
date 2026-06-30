#!/usr/bin/env python3
"""Build train-only pert means and baseline gates for cross-latent Track A.

This long CPU artifact job prepares the prerequisites for a later GPU
cross-latent anchor internal-val comparator. It reads only the train-only
split's train/internal-val rows and does not use canonical test, canonical
multi, or Track C query outcomes for selection.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
CONSTRUCT_SCRIPT = ROOT / "ops/construct_latentfm_xverse_trainonly_crossbg_val_split_20260622.py"
GENE_GATE_SCRIPT = ROOT / "ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py"
RUN_ROOT = ROOT / "runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622"
ARTIFACT_DIR = RUN_ROOT / "artifacts"
REPORT_JSON = ROOT / "reports/latentfm_crosslatent_tracka_trainonly_baselines_20260622.json"
REPORT_MD = ROOT / "reports/LATENTFM_CROSSLATENT_TRACKA_TRAINONLY_BASELINES_20260622.md"

LATENTS = {
    "stack": ROOT / "dataset/latentfm_full/stack",
    "scfoundation": ROOT / "dataset/latentfm_full/scfoundation",
    "scldm": ROOT / "dataset/latentfm_full/scldm",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_construct_module() -> Any:
    spec = importlib.util.spec_from_file_location("construct_trainonly_crossbg", CONSTRUCT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {CONSTRUCT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_gene_gate(latent: str, data_dir: Path, pert_means: Path) -> dict[str, Any]:
    out_json = ROOT / f"reports/latentfm_crosslatent_{latent}_gene_reliability_router_gate_20260622.json"
    out_md = ROOT / f"reports/LATENTFM_CROSSLATENT_{latent.upper()}_GENE_RELIABILITY_ROUTER_GATE_20260622.md"
    cmd = [
        sys.executable,
        str(GENE_GATE_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--split-file",
        str(SPLIT),
        "--pert-means-file",
        str(pert_means),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
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
    status = "ok" if proc.returncode == 0 else "failed"
    payload = None
    if out_json.is_file():
        payload = load_json(out_json)
    return {
        "latent": latent,
        "cmd": cmd,
        "returncode": proc.returncode,
        "status": status,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "decision": None if payload is None else payload.get("decision"),
        "selected_model": None if payload is None else payload.get("selected_model"),
        "n_train_rows": None if payload is None else payload.get("n_train_rows"),
        "n_val_rows": None if payload is None else payload.get("n_val_rows"),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Cross-Latent Track A Train-Only Baseline Build",
        "",
        f"Status: `{payload['status']}`",
        f"Recommended action: `{payload['recommended_action']}`",
        "",
        "## Scope",
        "",
        "- Builds latent-specific train-only pert means for the cross-background v2 split.",
        "- Runs the train-only gene-reliability baseline gate separately for each comparator latent.",
        "- Does not run model inference and does not read canonical test/multi/query outcomes for selection.",
        "",
        "## Outputs",
        "",
        "| latent | pert means | baseline report | gate status | selected model | train rows | val rows |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in payload["latents"]:
        decision = row.get("baseline_gate", {}).get("decision") or {}
        lines.append(
            f"| `{row['latent']}` | `{row['pert_means_file']}` | "
            f"`{row.get('baseline_gate', {}).get('out_md')}` | "
            f"`{decision.get('status', row.get('status'))}` | "
            f"`{row.get('baseline_gate', {}).get('selected_model')}` | "
            f"{row.get('baseline_gate', {}).get('n_train_rows')} | "
            f"{row.get('baseline_gate', {}).get('n_val_rows')} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- `{payload['status']}`: {payload['reason']}",
        f"- next action: `{payload['recommended_action']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    split = load_json(SPLIT)
    construct = load_construct_module()
    rows = []
    failures = []
    for latent, data_dir in LATENTS.items():
        pert_means_file = ARTIFACT_DIR / f"{latent}_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
        pert_audit_file = ARTIFACT_DIR / f"{latent}_trainonly_pert_means_audit_20260622.json"
        row: dict[str, Any] = {
            "latent": latent,
            "data_dir": str(data_dir),
            "pert_means_file": str(pert_means_file),
            "pert_mean_audit": str(pert_audit_file),
        }
        try:
            means, audit = construct.compute_train_pert_means(data_dir, split)
            np.savez_compressed(pert_means_file, **means)
            pert_audit_file.write_text(json.dumps(audit, indent=2), encoding="utf-8")
            row["pert_mean_status"] = "ok"
            row["pert_mean_datasets"] = len(means)
            row["pert_mean_cells"] = int(sum(int(x.get("train_cells_used", 0)) for x in audit))
        except Exception as exc:  # noqa: BLE001
            row["pert_mean_status"] = "failed"
            row["error"] = repr(exc)
            failures.append(f"{latent}: pert means failed")
            rows.append(row)
            continue
        gate = run_gene_gate(latent, data_dir, pert_means_file)
        row["baseline_gate"] = gate
        if gate["returncode"] != 0:
            failures.append(f"{latent}: baseline gate failed")
        rows.append(row)

    if failures:
        status = "crosslatent_baseline_build_failed"
        action = "inspect_logs_before_gpu_comparator"
        reason = "; ".join(failures)
    else:
        status = "crosslatent_trainonly_baselines_ready_for_protocol_review"
        action = "review_baseline_reports_then_decide_detached_gpu_anchor_comparator"
        reason = "all comparator train-only pert means and baseline gates were generated"
    payload = {
        "split_file": str(SPLIT),
        "artifact_dir": str(ARTIFACT_DIR),
        "latents": rows,
        "status": status,
        "recommended_action": action,
        "reason": reason,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render(payload), encoding="utf-8")
    print(REPORT_MD)
    print(REPORT_JSON)
    print(status)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
