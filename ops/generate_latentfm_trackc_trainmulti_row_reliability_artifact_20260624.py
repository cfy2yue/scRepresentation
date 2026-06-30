#!/usr/bin/env python3
"""Generate Track C train_multi row-level reliability artifact.

This CPU-only artifact fills the gap identified by the row-reliability V2
artifact gate: existing jackknife reports saved CV summaries but not the
train_multi leave-one-condition row tables. It reuses the same safe trainselect
split, hash guard, specs, and scoring functions as the original support
jackknife gate.

It does not read held-out Track C query, canonical metrics, canonical multi,
active logs, train models, infer, or use GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
JACKKNIFE_MODULE = ROOT / "ops/audit_latentfm_trackc_support_jackknife_reliability_gate_20260624.py"
OUT_JSON = ROOT / "reports/latentfm_trackc_trainmulti_row_reliability_artifact_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_TRAINMULTI_ROW_RELIABILITY_ARTIFACT_20260624.md"


def load_jackknife_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_jackknife_reliability_gate", JACKKNIFE_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {JACKKNIFE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def spec_row_gate(summary: dict[str, Any]) -> dict[str, Any]:
    pp = summary.get("paired_pp_delta") or {}
    by_ds = summary.get("by_dataset") or {}
    norman = by_ds.get("NormanWeissman2019_filtered") or {}
    wessels = by_ds.get("Wessels") or {}
    reasons: list[str] = []
    if float(pp.get("delta_mean") if pp.get("delta_mean") is not None else -999.0) < 0.02:
        reasons.append("train_cv_pp_delta_below_0p02")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("train_cv_p_harm_above_0p20")
    if int(summary.get("enabled_rows") or 0) < 6:
        reasons.append("enabled_rows_below_6")
    if int(summary.get("enabled_negative_rows") or 0) > 2:
        reasons.append("enabled_negative_rows_gt_2")
    if float(summary.get("enabled_min_pp_delta") if summary.get("enabled_min_pp_delta") is not None else -999.0) < -0.02:
        reasons.append("enabled_min_pp_below_minus_0p02")
    if float(norman.get("pp_delta") if norman.get("pp_delta") is not None else -999.0) < -0.01:
        reasons.append("norman_pp_below_minus_0p01")
    if float(wessels.get("pp_delta") if wessels.get("pp_delta") is not None else -999.0) < 0.02:
        reasons.append("wessels_pp_below_0p02")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "pp_delta": pp.get("delta_mean"),
        "p_harm": pp.get("p_harm"),
        "enabled_rows": summary.get("enabled_rows"),
        "enabled_negative_rows": summary.get("enabled_negative_rows"),
        "enabled_min_pp_delta": summary.get("enabled_min_pp_delta"),
        "norman_pp": norman.get("pp_delta"),
        "wessels_pp": wessels.get("pp_delta"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/latentfm_full/xverse")
    parser.add_argument("--split-file", type=Path, default=ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json")
    parser.add_argument(
        "--pert-means-file",
        type=Path,
        default=ROOT / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
    )
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    jack = load_jackknife_module()
    residual_mod = jack.load_residual_module()
    support_mod = residual_mod.load_support_module()
    split = support_mod.load_json(args.split_file)
    guard = residual_mod.split_guard(args.split_file, split)
    if guard["sha256"] != jack.EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")
    for ds in jack.FOCUS_DATASETS:
        obj = split.get(ds) or {}
        if set(obj.get("support_val_multi") or []) & set(obj.get("heldout_query_multi_final_only") or []):
            raise RuntimeError(f"{ds}: support_val_multi overlaps heldout query")

    manifest = support_mod.load_json(args.data_dir / "manifest.json")
    metadata = support_mod.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {key: value.astype(np.float32) for key, value in np.load(args.pert_means_file).items()}
    train_rows = support_mod.collect_role_rows(
        args.data_dir,
        split,
        metadata,
        "train_multi",
        max_cells=args.max_cells_per_condition,
    )
    support_val = support_mod.collect_role_rows(
        args.data_dir,
        split,
        metadata,
        "support_val_multi",
        max_cells=args.max_cells_per_condition,
    )
    single = support_mod.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells_per_condition)

    spec_tables: list[dict[str, Any]] = []
    pass_specs = []
    for spec in jack.specs():
        rows = jack.cv_rows(train_rows, spec, residual_mod, support_mod, single, pert_means)
        summary = jack.summarize(residual_mod, rows, n_boot=args.n_boot, seed=args.seed, include_mmd=False)
        gate = spec_row_gate(summary)
        record = {
            "spec": spec.name,
            "spec_config": spec.__dict__,
            "summary": summary,
            "gate": gate,
            "rows": rows,
        }
        spec_tables.append(record)
        if gate["pass"]:
            pass_specs.append(record)

    spec_tables = sorted(
        spec_tables,
        key=lambda item: (
            bool(item["gate"]["pass"]),
            float(item["gate"].get("pp_delta") if item["gate"].get("pp_delta") is not None else -999.0),
            float(item["gate"].get("wessels_pp") if item["gate"].get("wessels_pp") is not None else -999.0),
            int(item["gate"].get("enabled_rows") or 0),
        ),
        reverse=True,
    )
    best = spec_tables[0] if spec_tables else {}
    status = "trackc_trainmulti_row_reliability_artifact_ready_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_safe_trainselect_split": True,
            "selection_role": "train_multi_leave_one_condition_only",
            "support_val_scoring": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "split_guard": guard,
        "inputs": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "pert_means_file": str(args.pert_means_file),
            "jackknife_module": str(JACKKNIFE_MODULE),
        },
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows_metadata_only": len(support_val),
        "n_specs": len(spec_tables),
        "n_pass_specs_train_only": len(pass_specs),
        "best_spec": {
            "spec": best.get("spec"),
            "gate": best.get("gate"),
            "summary": best.get("summary"),
        },
        "spec_tables": spec_tables,
        "next_action": "run row-reliability V2 CPU gate using this artifact; no GPU authorized by artifact generation",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track C Train-Multi Row Reliability Artifact",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact generation.",
        "- Uses safe trainselect `train_multi` leave-one-condition rows only.",
        "- Does not score support_val for selection, read canonical metrics, canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- train_multi rows: `{len(train_rows)}`",
        f"- support_val rows counted for metadata only: `{len(support_val)}`",
        f"- specs with row tables: `{len(spec_tables)}`",
        f"- train-only pass specs under V2 row gate: `{len(pass_specs)}`",
        f"- best spec: `{best.get('spec')}`",
        "",
        "## Top Specs",
        "",
        "| spec | pass | pp delta | p_harm | enabled | neg enabled | min enabled pp | Norman | Wessels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in spec_tables[:20]:
        gate = item["gate"]
        lines.append(
            f"| `{item['spec']}` | `{gate['pass']}` | {fmt(gate.get('pp_delta'))} | "
            f"{fmt(gate.get('p_harm'))} | {gate.get('enabled_rows')} | {gate.get('enabled_negative_rows')} | "
            f"{fmt(gate.get('enabled_min_pp_delta'))} | {fmt(gate.get('norman_pp'))} | {fmt(gate.get('wessels_pp'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- GPU authorized: `False`",
            "- This artifact only supplies the missing train_multi row-level tables for a subsequent V2 CPU gate.",
            "",
            "## JSON",
            "",
            f"`{args.out_json}`",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "out_md": str(args.out_md),
                "n_specs": len(spec_tables),
                "n_pass_specs_train_only": len(pass_specs),
                "gpu_authorized": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
