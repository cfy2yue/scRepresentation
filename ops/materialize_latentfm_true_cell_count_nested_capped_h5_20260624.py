#!/usr/bin/env python3
"""Nested-v2 capped-H5 materializer for true cell-count scaling.

This script intentionally writes to separate nested-v2 paths so the exploratory
non-nested artifacts/runs remain reproducible. It reuses the audited capped-H5
materializer implementation but overrides the sampling rule: for the same
dataset/condition/seed, lower budgets are strict subsets of higher budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import materialize_latentfm_true_cell_count_capped_h5_20260624 as base


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DATA_ROOT = ROOT / "runs/latentfm_true_cell_count_scaling_nested_capped_h5_20260624/artifacts"
OUT_SPLIT_ROOT = ROOT / "dataset/biFlow_data/xverse_true_cell_count_scaling_nested_splits_20260624"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_nested_capped_h5_materializer_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_NESTED_CAPPED_H5_MATERIALIZER_GATE_20260624.md"

NESTED_MAX_BUDGET_BY_PROTOCOL = {
    "gene_only_fixed256_budget64_128_256": 256,
}


def stable_seed(*parts: object) -> int:
    raw = "\t".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def nested_sample_indices(n: int, k: int, *, key: str) -> np.ndarray:
    if k <= 0 or n <= k:
        return np.arange(n, dtype=np.int64)
    parts = key.split("|")
    if len(parts) != 5:
        return base.sample_indices(n, k, key=key)
    group, dataset, condition, seed, _cap = parts
    rng = np.random.default_rng(stable_seed("nested_v2", group, dataset, condition, seed))
    order = rng.permutation(np.arange(n, dtype=np.int64))
    return np.sort(order[: int(k)])


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM True Cell-Count Nested Capped-H5 Materializer Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested-v2 capped latent-H5 materializer gate.",
        "- Writes to independent nested-v2 artifact and split directories.",
        "- Train-condition sampled rows are nested within seed: budget64 subset budget128 subset budget256.",
        "- Does not read canonical metrics, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Mode",
        "",
        f"- materialized: `{payload['materialized']}`",
        "",
        "## Plan Rows",
        "",
        "| run id | budget | seed | train modalities | eval modalities | estimated rows | missing | low train | launcher ready | reasons | data dir |",
        "|---|---:|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in payload["plan_rows"]:
        lines.append(
            f"| `{row['run_id']}` | {row['budget']} | {row['seed']} | `{row.get('train_modality_counts', {})}` | `{row.get('eval_modality_counts', {})}` | {row['estimated_total_rows_ctrl_gt_each']} | {sum(m['n'] for m in row['missing'])} | {sum(m['n'] for m in row['low_train'])} | `{row.get('launcher_ready')}` | {', '.join(row.get('launcher_readiness_reasons') or []) or 'none'} | `{row['data_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def rewrite_row_paths(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["data_dir"] = str(OUT_DATA_ROOT / row["run_id"])
    row["split_file"] = str(OUT_SPLIT_ROOT / f"split_{row['run_id']}.json")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialize", action="store_true", help="write nested capped H5 data dirs")
    ap.add_argument("--only-run-id", default="")
    ap.add_argument("--only-launcher-ready", action="store_true")
    args = ap.parse_args()

    # Patch only this process. The original non-nested script and artifacts stay unchanged.
    base.OUT_DATA_ROOT = OUT_DATA_ROOT
    base.OUT_SPLIT_ROOT = OUT_SPLIT_ROOT
    base.sample_indices = nested_sample_indices

    base_split = base.load_json(base.BASE_SPLIT)
    metadata = base.load_json(base.BASE_DATA_DIR / "condition_metadata.json")
    protocol_payload = base.load_json(base.PROTOCOL_JSON)
    plan_rows: list[dict[str, Any]] = []
    for protocol in protocol_payload["protocols"]:
        if protocol["name"] not in NESTED_MAX_BUDGET_BY_PROTOCOL:
            continue
        plan_rows.extend(base.build_plan(protocol, base_split, metadata))
    plan_rows = [rewrite_row_paths(row) for row in plan_rows]
    if args.only_run_id:
        plan_rows = [row for row in plan_rows if row["run_id"] == args.only_run_id]
        if not plan_rows:
            raise SystemExit(f"unknown run id: {args.only_run_id}")
    if args.only_launcher_ready:
        plan_rows = [row for row in plan_rows if row.get("launcher_ready")]
        if not plan_rows:
            raise SystemExit("no launcher-ready rows")

    bad = [row for row in plan_rows if row["missing"] or row["low_train"] or not row.get("launcher_ready")]
    materialized_rows = []
    if args.materialize:
        if bad:
            raise SystemExit("refusing to materialize plan rows with missing, low-train, or launcher-readiness failures")
        for row in plan_rows:
            materialized_rows.append(base.materialize_plan_row(row))

    status = "nested_capped_h5_materializer_dryrun_pass_no_gpu" if not bad else "nested_capped_h5_materializer_dryrun_fail_no_gpu"
    if args.materialize:
        status = "nested_capped_h5_materialized_no_gpu" if all(r.get("status") == "ok" for r in materialized_rows) else "nested_capped_h5_materialized_check_no_gpu"
    public_rows = [{k: v for k, v in row.items() if k != "split"} for row in plan_rows]
    payload = {
        "status": status,
        "materialized": bool(args.materialize),
        "nested_sampling": {
            "enabled": True,
            "method": "same_seed_permutation_prefix_sorted",
            "expected_budget_ladder": [64, 128, 256],
        },
        "boundary": {
            "cpu_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
            "excluded_split_keys": sorted(base.EXCLUDED_SPLIT_KEYS),
        },
        "base_data_dir": str(base.BASE_DATA_DIR),
        "base_split": str(base.BASE_SPLIT),
        "protocol_json": str(base.PROTOCOL_JSON),
        "plan_rows": public_rows,
        "materialized_rows": materialized_rows,
        "bad_rows": [{k: v for k, v in row.items() if k != "split"} for row in bad],
        "only_launcher_ready": bool(args.only_launcher_ready),
        "gpu_authorized": False,
        "next_action": "review dry-run; materialize nested-v2 only if exploratory seed42 smoke shows a real signal"
        if (not args.materialize and not bad)
        else ("run nested-v2 post-materialization gates before any nested-v2 GPU" if args.materialize else "fix or select only launcher-ready rows before nested artifact generation"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "materialized": bool(args.materialize), "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
