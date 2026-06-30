#!/usr/bin/env python3
"""CPU-only LODO/mixed-effect proxy gate for condition-count scaling.

The gate asks whether the cap120-vs-cap30 train-only signal is robust enough
after dataset/background/type leave-out checks to justify any new GPU matrix.
It uses completed train-only summaries only and does not read canonical multi
or held-out Track C query artifacts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
TARGET_GATE = REPORTS / "latentfm_scaling_target_gene_coverage_protocol_gate_20260624.json"
SOURCE_GATE = REPORTS / "latentfm_scaling_source_verified_background_type_strata_gate_20260624.json"
PROVENANCE_GATE = REPORTS / "latentfm_scaling_provenance_estimand_matrix_gate_20260624.json"

OUT_JSON = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_MIXED_EFFECT_LODO_CONDITION_COUNT_GATE_20260624.md"

SEED = 20260624


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def weighted_mean(rows: list[dict[str, Any]], key: str = "pp_delta_mean") -> float | None:
    total_n = sum(int(r.get("n") or 0) for r in rows)
    if total_n <= 0:
        return None
    return sum(float(r.get(key) or 0.0) * int(r.get("n") or 0) for r in rows) / total_n


def ci(values: list[float], lo: float = 0.025, hi: float = 0.975) -> list[float | None]:
    if not values:
        return [None, None]
    vals = sorted(values)
    n = len(vals)
    return [vals[min(n - 1, max(0, int(lo * n)))], vals[min(n - 1, max(0, int(hi * n)))]]


def bootstrap_dataset_means(ds_rows: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    rng = random.Random(SEED)
    if not ds_rows:
        return {"n_boot": 0, "mean_ci": [None, None], "p_le_zero": None}
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(ds_rows) for _ in ds_rows]
        vals.append(float(weighted_mean(sample) or 0.0))
    return {
        "n_boot": n_boot,
        "mean_ci": ci(vals),
        "p_le_zero": sum(1 for v in vals if v <= 0.0) / len(vals),
        "median": median(vals),
    }


def leave_one(rows: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    labels = sorted({str(r.get(label_key) or "") for r in rows})
    out = []
    for label in labels:
        kept = [r for r in rows if str(r.get(label_key) or "") != label]
        out.append(
            {
                "left_out": label,
                "n_kept": sum(int(r.get("n") or 0) for r in kept),
                "pp_delta_mean": weighted_mean(kept),
                "mmd_delta_mean": weighted_mean(kept, key="mmd_delta_mean"),
            }
        )
    return out


def main() -> int:
    target = load_json(TARGET_GATE)
    source = load_json(SOURCE_GATE)
    provenance = load_json(PROVENANCE_GATE)

    cap = (target.get("comparisons") or {}).get("cap120_minus_cap30") or {}
    fam = (target.get("comparisons") or {}).get("cap120_minus_cap30_family_proxy") or {}
    ds_meta = {row["dataset"]: row for row in provenance.get("dataset_rows", [])}

    ds_rows = []
    for ds, item in sorted((cap.get("dataset_means") or {}).items()):
        meta = ds_meta.get(ds, {})
        ds_rows.append(
            {
                "dataset": ds,
                "n": int(item.get("n") or 0),
                "pp_delta_mean": float(item.get("pp_delta_mean") or 0.0),
                "mmd_delta_mean": float(item.get("mmd_delta_mean") or 0.0),
                "background": str(meta.get("cell_background_source") or "unknown"),
                "perturbation_type": str(meta.get("perturbation_type") or "unknown"),
                "cap_gain": int(meta.get("n_conditions_cap120") or 0) - int(meta.get("n_conditions_cap30") or 0),
                "source_quality": str(meta.get("source_quality") or "unknown"),
            }
        )

    lodo_dataset = leave_one(ds_rows, "dataset")
    lodo_background = leave_one(ds_rows, "background")
    lodo_type = leave_one(ds_rows, "perturbation_type")
    boot = bootstrap_dataset_means(ds_rows)

    negative_datasets = [r for r in ds_rows if r["pp_delta_mean"] < -0.02]
    severe_negative_datasets = [r for r in ds_rows if r["pp_delta_mean"] < -0.05]
    positive_datasets = [r for r in ds_rows if r["pp_delta_mean"] > 0.0]
    min_lodo_dataset = min((float(r["pp_delta_mean"]) for r in lodo_dataset if r["pp_delta_mean"] is not None), default=None)
    min_lodo_background = min((float(r["pp_delta_mean"]) for r in lodo_background if r["pp_delta_mean"] is not None), default=None)
    min_lodo_type = min((float(r["pp_delta_mean"]) for r in lodo_type if r["pp_delta_mean"] is not None), default=None)

    bg_summary = source.get("background_summary") or {}
    type_summary = source.get("perturbation_type_summary") or {}
    negative_backgrounds = [
        {"background": k, **v} for k, v in bg_summary.items() if float(v.get("pp_delta_mean") or 0.0) < -0.02
    ]
    negative_types = [
        {"perturbation_type": k, **v}
        for k, v in type_summary.items()
        if float(v.get("pp_delta_mean") or 0.0) < -0.02
    ]

    reasons = []
    if float(cap.get("pp_delta_mean") or -999.0) < 0.010:
        reasons.append("condition_weighted_mean_pp_below_0p010")
    if float(fam.get("pp_delta_mean") or -999.0) < 0.010:
        reasons.append("family_proxy_mean_pp_below_0p010")
    if float(cap.get("mmd_delta_mean") or 999.0) > 0.001:
        reasons.append("condition_weighted_mmd_above_0p001")
    if float(cap.get("min_dataset_pp_delta") or 0.0) < -0.02:
        reasons.append("dataset_tail_pp_below_minus_0p02")
    if len(negative_datasets) > 2:
        reasons.append("too_many_negative_dataset_tails")
    if severe_negative_datasets:
        reasons.append("severe_negative_dataset_tail")
    if boot.get("mean_ci", [None, None])[0] is None or float(boot["mean_ci"][0]) <= 0.0:
        reasons.append("dataset_bootstrap_ci_touches_zero")
    if min_lodo_dataset is not None and min_lodo_dataset < 0.005:
        reasons.append("leave_one_dataset_min_below_0p005")
    if min_lodo_background is not None and min_lodo_background < 0.005:
        reasons.append("leave_one_background_min_below_0p005")
    if min_lodo_type is not None and min_lodo_type < 0.005:
        reasons.append("leave_one_type_min_below_0p005")
    if negative_backgrounds:
        reasons.append("source_background_negative_tails")
    if negative_types:
        reasons.append("source_type_negative_tails")

    status = "scaling_mixed_effect_lodo_condition_count_fail_no_gpu"
    if not reasons:
        status = "scaling_mixed_effect_lodo_condition_count_pass_gpu_protocol_next"

    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "boundary": {
            "cpu_only": True,
            "reads_train_only_completed_reports": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_datasets": len(ds_rows),
            "n_conditions": cap.get("n_conditions"),
            "condition_weighted_pp_delta": cap.get("pp_delta_mean"),
            "condition_weighted_mmd_delta": cap.get("mmd_delta_mean"),
            "family_proxy_pp_delta": fam.get("pp_delta_mean"),
            "family_proxy_mmd_delta": fam.get("mmd_delta_mean"),
            "dataset_unweighted_pp_mean": mean([r["pp_delta_mean"] for r in ds_rows]) if ds_rows else None,
            "dataset_median_pp_delta": median([r["pp_delta_mean"] for r in ds_rows]) if ds_rows else None,
            "dataset_min_pp_delta": cap.get("min_dataset_pp_delta"),
            "positive_dataset_count": len(positive_datasets),
            "negative_dataset_tail_count_lt_minus_0p02": len(negative_datasets),
            "severe_negative_dataset_tail_count_lt_minus_0p05": len(severe_negative_datasets),
            "bootstrap_dataset_mean": boot,
            "min_leave_one_dataset_pp_delta": min_lodo_dataset,
            "min_leave_one_background_pp_delta": min_lodo_background,
            "min_leave_one_type_pp_delta": min_lodo_type,
        },
        "dataset_rows": ds_rows,
        "leave_one_dataset": lodo_dataset,
        "leave_one_background": lodo_background,
        "leave_one_perturbation_type": lodo_type,
        "negative_backgrounds": negative_backgrounds,
        "negative_perturbation_types": negative_types,
        "reasons": reasons,
        "next_action": (
            "build predeclared GPU condition-count micro-matrix"
            if status.endswith("_next")
            else "do not launch condition-count scaling GPU; use no-harm surrogate v2 or new tail-safe mechanism first"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Mixed-Effect / LODO Condition-Count Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed train-only scaling reports.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- paired conditions: `{cap.get('n_conditions')}`",
        f"- datasets: `{len(ds_rows)}`",
        f"- condition-weighted pp delta: `{fmt(cap.get('pp_delta_mean'))}`",
        f"- condition-weighted MMD delta: `{fmt(cap.get('mmd_delta_mean'))}`",
        f"- family proxy pp/MMD delta: `{fmt(fam.get('pp_delta_mean'))}` / `{fmt(fam.get('mmd_delta_mean'))}`",
        f"- dataset unweighted pp mean/median: `{fmt(payload['summary']['dataset_unweighted_pp_mean'])}` / `{fmt(payload['summary']['dataset_median_pp_delta'])}`",
        f"- dataset min pp delta: `{fmt(cap.get('min_dataset_pp_delta'))}`",
        f"- negative dataset tails `< -0.02`: `{len(negative_datasets)}`",
        f"- bootstrap dataset-mean pp CI: `{[fmt(x) for x in boot.get('mean_ci', [])]}`; p<=0 `{boot.get('p_le_zero')}`",
        f"- min leave-one dataset/background/type pp: `{fmt(min_lodo_dataset)}` / `{fmt(min_lodo_background)}` / `{fmt(min_lodo_type)}`",
        "",
        "## Dataset Tails",
        "",
        "| dataset | n | background | type | cap gain | pp delta | MMD delta |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in sorted(ds_rows, key=lambda r: r["pp_delta_mean"]):
        lines.append(
            f"| `{row['dataset']}` | {row['n']} | `{row['background']}` | `{row['perturbation_type']}` | "
            f"{row['cap_gain']} | {fmt(row['pp_delta_mean'])} | {fmt(row['mmd_delta_mean'])} |"
        )
    lines.extend(
        [
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
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": payload["gpu_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
