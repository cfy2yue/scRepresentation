#!/usr/bin/env python3
"""CPU gate for source-verified scaling strata.

This audit reads existing train-only scaling posthoc artifacts plus local
metadata. It asks whether the cap120>cap30 scaling signal is broad across
source-verified cell/background and perturbation-type strata, rather than
being a dataset-count proxy or a few-tail artifact.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_source_verified_background_type_strata_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_SOURCE_VERIFIED_BACKGROUND_TYPE_STRATA_GATE_20260624.md"

RUNS = {
    "cap30_all": RUN_ROOT / "xverse_scaling_cap30_all_3k_seed42",
    "cap120_all": RUN_ROOT / "xverse_scaling_cap120_all_3k_seed42",
}
GROUP = "internal_val_cross_background_seen_gene_proxy"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def condition_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = ((payload.get("groups") or {}).get(GROUP) or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def load_run(run_dir: Path) -> dict[str, Any]:
    eval_dir = run_dir / "posthoc_eval_internal"
    return {
        "candidate": load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json"),
        "anchor": load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json"),
    }


def dataset_meta() -> dict[str, dict[str, Any]]:
    inventory = load_json(ROOT / "reports/latentfm_scaling_metainfo_inventory_20260624.json")
    out = {}
    for row in inventory.get("rows") or []:
        ds = str(row.get("dataset") or "")
        if not ds:
            continue
        out[ds] = {
            "background": str(row.get("cell_line_meta") or "unknown"),
            "perturbation_type": str(row.get("perturbation_type") or "unknown"),
            "obs_cell_type_n_unique": int(row.get("obs_cell_type_n_unique") or 0),
            "obs_pathway_n_unique": int(row.get("obs_pathway_n_unique") or 0),
            "obs_cov_drug_n_unique": int(row.get("obs_cov_drug_n_unique") or 0),
            "trainonly_train": int(row.get("trainonly_crossbg_v2_train") or 0),
            "cap120_train": int(row.get("cap120_all_v2_train") or 0),
            "cap30_train": int(row.get("cap30_all_v2_train") or 0),
        }
    return out


def paired_condition_deltas(meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    runs = {name: load_run(path) for name, path in RUNS.items()}
    c30_c = condition_rows(runs["cap30_all"]["candidate"])
    c30_a = condition_rows(runs["cap30_all"]["anchor"])
    c120_c = condition_rows(runs["cap120_all"]["candidate"])
    c120_a = condition_rows(runs["cap120_all"]["anchor"])
    rows = []
    for key in sorted(set(c30_c) & set(c30_a) & set(c120_c) & set(c120_a)):
        ds, cond = key
        vals = {
            "c30_c_pp": c30_c[key].get("pearson_pert"),
            "c30_a_pp": c30_a[key].get("pearson_pert"),
            "c120_c_pp": c120_c[key].get("pearson_pert"),
            "c120_a_pp": c120_a[key].get("pearson_pert"),
            "c30_c_mmd": c30_c[key].get("test_mmd_clamped"),
            "c30_a_mmd": c30_a[key].get("test_mmd_clamped"),
            "c120_c_mmd": c120_c[key].get("test_mmd_clamped"),
            "c120_a_mmd": c120_a[key].get("test_mmd_clamped"),
        }
        if any(v is None for v in vals.values()):
            continue
        md = meta.get(ds, {})
        pp30 = float(vals["c30_c_pp"]) - float(vals["c30_a_pp"])
        pp120 = float(vals["c120_c_pp"]) - float(vals["c120_a_pp"])
        mmd30 = float(vals["c30_c_mmd"]) - float(vals["c30_a_mmd"])
        mmd120 = float(vals["c120_c_mmd"]) - float(vals["c120_a_mmd"])
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "background": md.get("background", "unknown"),
                "perturbation_type": md.get("perturbation_type", "unknown"),
                "pp_delta_cap120_minus_cap30": pp120 - pp30,
                "mmd_delta_cap120_minus_cap30": mmd120 - mmd30,
            }
        )
    return rows


def summarize_by(rows: list[dict[str, Any]], key: str, *, min_n: int = 5) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    out = {}
    for name, vals in sorted(buckets.items()):
        if len(vals) < min_n:
            continue
        out[name] = {
            "n": len(vals),
            "datasets": sorted({v["dataset"] for v in vals}),
            "dataset_count": len({v["dataset"] for v in vals}),
            "pp_delta_mean": mean([v["pp_delta_cap120_minus_cap30"] for v in vals]),
            "mmd_delta_mean": mean([v["mmd_delta_cap120_minus_cap30"] for v in vals]),
        }
    return out


def crossing(meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bg_to_types: dict[str, set[str]] = defaultdict(set)
    type_to_bgs: dict[str, set[str]] = defaultdict(set)
    bg_to_ds: dict[str, set[str]] = defaultdict(set)
    type_to_ds: dict[str, set[str]] = defaultdict(set)
    for ds, md in meta.items():
        bg = str(md.get("background") or "unknown")
        typ = str(md.get("perturbation_type") or "unknown")
        bg_to_types[bg].add(typ)
        type_to_bgs[typ].add(bg)
        bg_to_ds[bg].add(ds)
        type_to_ds[typ].add(ds)
    return {
        "background_count": len(bg_to_types),
        "type_count": len(type_to_bgs),
        "backgrounds_with_ge2_types": {k: sorted(v) for k, v in bg_to_types.items() if len(v) >= 2},
        "types_with_ge2_backgrounds": {k: sorted(v) for k, v in type_to_bgs.items() if len(v) >= 2},
        "background_dataset_counts": {k: len(v) for k, v in sorted(bg_to_ds.items())},
        "type_dataset_counts": {k: len(v) for k, v in sorted(type_to_ds.items())},
    }


def decide(backgrounds: dict[str, Any], types: dict[str, Any], cross: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    positive_bg = [k for k, v in backgrounds.items() if float(v.get("pp_delta_mean") or -999.0) > 0]
    positive_type = [k for k, v in types.items() if float(v.get("pp_delta_mean") or -999.0) > 0]
    min_bg = min((float(v["pp_delta_mean"]) for v in backgrounds.values()), default=-999.0)
    min_type = min((float(v["pp_delta_mean"]) for v in types.values()), default=-999.0)
    if len(backgrounds) < 4:
        reasons.append("too_few_background_strata_with_condition_support")
    if len(positive_bg) < max(3, math.ceil(0.6 * max(1, len(backgrounds)))):
        reasons.append("background_positive_coverage_insufficient")
    if min_bg < -0.02:
        reasons.append("background_tail_harm_below_minus_0p02")
    if len(types) < 3:
        reasons.append("too_few_perturbation_type_strata_with_condition_support")
    if len(positive_type) < max(2, math.ceil(0.6 * max(1, len(types)))):
        reasons.append("type_positive_coverage_insufficient")
    if min_type < -0.02:
        reasons.append("type_tail_harm_below_minus_0p02")
    if len(cross.get("backgrounds_with_ge2_types") or {}) < 2:
        reasons.append("source_metadata_background_type_crossing_too_sparse")
    if len(cross.get("types_with_ge2_backgrounds") or {}) < 2:
        reasons.append("source_metadata_type_background_crossing_too_sparse")
    status = "source_verified_strata_gate_pass_split_builder_next" if not reasons else "source_verified_strata_gate_fail_no_gpu"
    return status, reasons


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    meta = dataset_meta()
    rows = paired_condition_deltas(meta)
    background_summary = summarize_by(rows, "background")
    type_summary = summarize_by(rows, "perturbation_type")
    cross = crossing(meta)
    status, reasons = decide(background_summary, type_summary, cross)
    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "reasons": reasons,
        "boundary": {
            "reads_train_only_posthoc": True,
            "reads_local_metainfo": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "launches_gpu": False,
        },
        "n_paired_conditions": len(rows),
        "background_summary": background_summary,
        "perturbation_type_summary": type_summary,
        "metadata_crossing": cross,
        "next_action": (
            "build source-verified background/type split builder with shuffled-source controls"
            if status.endswith("_next")
            else "treat background/type scaling as confounded diagnostics; do not launch GPU from this gate"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Scaling Source-Verified Background/Type Strata Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only audit of existing train-only cap30/cap120 posthoc and source metadata.",
        "- Does not train, launch GPU, read canonical multi, or read Track C query.",
        "",
        "## Background Strata",
        "",
        "| background | n | datasets | mean pp cap120-cap30 | mean MMD cap120-cap30 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in background_summary.items():
        lines.append(
            f"| `{name}` | {row['n']} | {row['dataset_count']} | {fmt(row['pp_delta_mean'])} | {fmt(row['mmd_delta_mean'])} |"
        )
    lines.extend(["", "## Perturbation-Type Strata", "", "| type | n | datasets | mean pp cap120-cap30 | mean MMD cap120-cap30 |", "|---|---:|---:|---:|---:|"])
    for name, row in type_summary.items():
        lines.append(
            f"| `{name}` | {row['n']} | {row['dataset_count']} | {fmt(row['pp_delta_mean'])} | {fmt(row['mmd_delta_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Source Crossing",
            "",
            f"- backgrounds with >=2 perturbation types: `{cross['backgrounds_with_ge2_types']}`",
            f"- perturbation types with >=2 backgrounds: `{cross['types_with_ge2_backgrounds']}`",
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": payload["gpu_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
