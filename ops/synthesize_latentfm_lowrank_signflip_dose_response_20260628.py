#!/usr/bin/env python3
"""Synthesize low-rank signflip dose-response diagnostics.

CPU/report-only mechanism synthesis over existing internal proxy posthoc rows.
No training, inference, GPU, canonical multi selection, or Track C query.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_lowrank_signflip_dose_response_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_LOWRANK_SIGNFLIP_DOSE_RESPONSE_20260628.md"
OUT_ROWS = ROOT / "reports/lowrank_signflip_dose_response_20260628/alpha_group_rows.csv"
OUT_SLOPES = ROOT / "reports/lowrank_signflip_dose_response_20260628/source_group_slopes.csv"

SOURCES = [
    {
        "source": "lowrank_5accepted_source",
        "accepted": 5,
        "alphas": {
            1.0: ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_5accepted_20260628_0000/posthoc/internal_eval_vs_anchor_summary.csv",
            -1.0: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from5step_alphas_m1_m0p5_m0p25_p0p25_20260628_0007/alpha_m1/posthoc/internal_eval_vs_anchor_summary.csv",
            -0.5: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from5step_alphas_m1_m0p5_m0p25_p0p25_20260628_0007/alpha_m0p5/posthoc/internal_eval_vs_anchor_summary.csv",
            -0.25: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from5step_alphas_m1_m0p5_m0p25_p0p25_20260628_0007/alpha_m0p25/posthoc/internal_eval_vs_anchor_summary.csv",
            0.25: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from5step_alphas_m1_m0p5_m0p25_p0p25_20260628_0007/alpha_0p25/posthoc/internal_eval_vs_anchor_summary.csv",
        },
    },
    {
        "source": "lowrank_20accepted_source",
        "accepted": 20,
        "alphas": {
            1.0: ROOT / "runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_20accepted_20260627_2356/posthoc/internal_eval_vs_anchor_summary.csv",
            -1.0: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from20step_alphas_m1_m3_m10_20260628_0029/alpha_m1/posthoc/internal_eval_vs_anchor_summary.csv",
            -3.0: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from20step_alphas_m1_m3_m10_20260628_0029/alpha_m3/posthoc/internal_eval_vs_anchor_summary.csv",
            -10.0: ROOT / "runs/latentfm_lowrank_signflip_diagnostic_20260628/xverse_lowrank_signflip_from20step_alphas_m1_m3_m10_20260628_0029/alpha_m10/posthoc/internal_eval_vs_anchor_summary.csv",
        },
    },
]


def f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in SOURCES:
        for alpha, path in source["alphas"].items():
            for row in read_summary(path):
                mean_pp = f(row.get("mean_delta_pearson_pert"))
                dataset_min = f(row.get("dataset_min_delta_pearson_pert"))
                ci_low = f(row.get("dataset_bootstrap_ci_low"))
                mean_mmd = f(row.get("mean_delta_mmd_clamped"))
                max_mmd = f(row.get("max_dataset_delta_mmd_clamped"))
                tail_safe = dataset_min >= -0.02 and ci_low >= -0.005 and max_mmd <= 0.005
                out.append(
                    {
                        "source": source["source"],
                        "accepted": source["accepted"],
                        "alpha": alpha,
                        "group": row["group"],
                        "n_joined": int(float(row.get("n_joined", 0) or 0)),
                        "datasets": int(float(row.get("datasets", 0) or 0)),
                        "mean_delta_pearson_pert": mean_pp,
                        "dataset_min_delta_pearson_pert": dataset_min,
                        "dataset_bootstrap_ci_low": ci_low,
                        "mean_delta_mmd_clamped": mean_mmd,
                        "max_dataset_delta_mmd_clamped": max_mmd,
                        "tail_safe_for_internal_screen": tail_safe,
                        "path": str(path),
                    }
                )
    return out


def summarize_slopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({row["source"] for row in rows}):
        for group in sorted({row["group"] for row in rows if row["source"] == source}):
            subset = sorted([row for row in rows if row["source"] == source and row["group"] == group], key=lambda r: float(r["alpha"]))
            xs = np.asarray([float(row["alpha"]) for row in subset], dtype=np.float64)
            ys = np.asarray([float(row["mean_delta_pearson_pert"]) for row in subset], dtype=np.float64)
            slope, intercept = np.polyfit(xs, ys, deg=1) if len(xs) >= 2 else (float("nan"), float("nan"))
            best_tail_safe = max(
                (row for row in subset if row["tail_safe_for_internal_screen"]),
                key=lambda r: float(r["mean_delta_pearson_pert"]),
                default=None,
            )
            out.append(
                {
                    "source": source,
                    "group": group,
                    "n_alpha": len(subset),
                    "linear_slope_mean_pp_vs_alpha": float(slope),
                    "linear_intercept": float(intercept),
                    "anti_aligned": bool(slope < 0.0),
                    "best_tail_safe_alpha": best_tail_safe["alpha"] if best_tail_safe else "",
                    "best_tail_safe_mean_pp": best_tail_safe["mean_delta_pearson_pert"] if best_tail_safe else "",
                    "best_tail_safe_dataset_min": best_tail_safe["dataset_min_delta_pearson_pert"] if best_tail_safe else "",
                    "best_tail_safe_ci_low": best_tail_safe["dataset_bootstrap_ci_low"] if best_tail_safe else "",
                }
            )
    return out


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Low-Rank Signflip Dose-Response",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of existing safe internal proxy posthoc summaries.",
        "- Includes original positive low-rank checkpoints as `alpha=+1` and frozen signflip checkpoints.",
        "- No training, inference, GPU, canonical multi selection, Track A canonical checkpoint selection, or Track C query.",
        "",
        "## Decision",
        "",
        f"- reasons: `{payload['reasons']}`",
        f"- recommended frozen canonical candidate: `{payload['recommended_canonical_candidate']}`",
        "",
        "## Source-Group Slopes",
        "",
        "| source | group | n alpha | slope pp vs alpha | anti-aligned | best tail-safe alpha | best tail-safe pp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["slopes"]:
        lines.append(
            f"| `{row['source']}` | `{row['group']}` | {row['n_alpha']} | "
            f"{row['linear_slope_mean_pp_vs_alpha']:.6g} | `{row['anti_aligned']}` | "
            f"{row['best_tail_safe_alpha']} | {row['best_tail_safe_mean_pp']} |"
        )
    lines.extend(
        [
            "",
            "## Alpha Rows",
            "",
            "| source | alpha | group | mean pp | dataset min | CI low | mean MMD | tail safe |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(payload["rows"], key=lambda r: (r["source"], r["group"], float(r["alpha"]))):
        lines.append(
            f"| `{row['source']}` | {row['alpha']} | `{row['group']}` | "
            f"{row['mean_delta_pearson_pert']:.6g} | {row['dataset_min_delta_pearson_pert']:.6g} | "
            f"{row['dataset_bootstrap_ci_low']:.6g} | {row['mean_delta_mmd_clamped']:.6g} | "
            f"`{row['tail_safe_for_internal_screen']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- alpha rows: `{OUT_ROWS}`",
            f"- slopes: `{OUT_SLOPES}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = load_rows()
    slopes = summarize_slopes(rows)
    write_csv(OUT_ROWS, rows)
    write_csv(OUT_SLOPES, slopes)

    reasons: list[str] = []
    if not all(row["anti_aligned"] for row in slopes):
        reasons.append("not_all_source_group_slopes_anti_aligned")
    # High negative scale should not be treated as safe if it has tail harm.
    high_mag_rows = [row for row in rows if float(row["alpha"]) <= -10.0]
    if any(row["tail_safe_for_internal_screen"] for row in high_mag_rows):
        reasons.append("highest_negative_alpha_tail_safe_unexpected_check_thresholds")
    if not any(row["source"] == "lowrank_20accepted_source" and float(row["alpha"]) == -3.0 and row["tail_safe_for_internal_screen"] for row in rows):
        reasons.append("alpha_m3_not_tail_safe")
    status = (
        "lowrank_signflip_dose_response_supports_reverse_objective_wait_canonical"
        if not reasons
        else "lowrank_signflip_dose_response_inconclusive_no_gpu"
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "recommended_canonical_candidate": "lowrank_20accepted_source alpha=-3; already running frozen canonical no-harm",
        "boundary": {
            "cpu_only": True,
            "uses_gpu": False,
            "trains_model": False,
            "runs_inference": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
        },
        "rows": rows,
        "slopes": slopes,
        "outputs": {"json": str(OUT_JSON), "md": str(OUT_MD), "rows": str(OUT_ROWS), "slopes": str(OUT_SLOPES)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD), "reasons": reasons}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
