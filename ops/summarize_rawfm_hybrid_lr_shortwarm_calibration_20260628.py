#!/usr/bin/env python3
"""Summarize RawFM hybrid short-warmup calibration smokes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/rawfm_hybrid_lr_shortwarm_calibration_20260628"

REFERENCE_ROWS = [
    {
        "label": "default_nommd_candidate_ref",
        "gamma": "none",
        "role": "candidate_ref",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_residual128_abundance128_hybrid_k256_smoke_20260628_2018",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_residual128_abundance128_hybrid_k256_seed42/ot",
    },
    {
        "label": "default_nommd_control_ref",
        "gamma": "none",
        "role": "control_ref",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_confound128_abundance128_hybrid_control_k256_smoke_20260628_2018",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_confound128_abundance128_hybrid_control_k256_seed42/ot",
    },
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    val = finite(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def latest_manifest() -> Path | None:
    manifests = sorted(OUT_DIR.glob("launch_manifest_*.tsv"))
    return manifests[-1] if manifests else None


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "label": row["label"],
                    "gamma": row["gamma"],
                    "role": row["role"],
                    "run_dir": Path(row["run_dir"]),
                    "output": Path(row["output"]),
                }
            )
    return rows


def read_exit(run_dir: Path) -> str:
    path = run_dir / "EXIT_CODE"
    if not path.exists():
        return "running"
    return path.read_text(encoding="utf-8").strip() or "unknown"


def read_log_rows(output: Path) -> list[dict[str, Any]]:
    log = output / "train_log.jsonl"
    if not log.exists():
        return []
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_final(output: Path) -> dict[str, Any]:
    final = None
    for obj in read_log_rows(output):
        if obj.get("eval_type") == "final_test":
            final = obj
    return final or {}


def read_mmd_genes(run_dir: Path) -> list[int]:
    log = run_dir / "logs/run.log"
    if not log.exists():
        return []
    text = log.read_text(encoding="utf-8", errors="replace")
    return [int(x) for x in re.findall(r"\bmmd_genes=(\d+)\b", text)]


def collect_rows(manifest: Path | None) -> list[dict[str, Any]]:
    specs = list(REFERENCE_ROWS)
    if manifest is not None:
        specs.extend(read_manifest(manifest))
    out = []
    for spec in specs:
        final = read_final(spec["output"])
        counts = read_mmd_genes(spec["run_dir"])
        out.append(
            {
                "label": spec["label"],
                "gamma": spec["gamma"],
                "role": spec["role"],
                "exit_code": read_exit(spec["run_dir"]),
                "has_final": bool(final),
                "best_pt_exists": (spec["output"] / "best.pt").exists(),
                "direct_pearson": finite(final.get("eval_direct_pearson")),
                "pearson_delta_ctrl": finite(final.get("eval_pearson_delta_ctrl")),
                "corr_ctrl_mean": finite(final.get("eval_corr_ctrl_mean")),
                "corr_pert_mean": finite(final.get("eval_corr_pert_mean")),
                "mmd": finite(final.get("eval_mmd")),
                "train_mmd_steps": len(counts),
                "train_mmd_visible_genes_min": min(counts) if counts else math.nan,
                "train_mmd_visible_genes_max": max(counts) if counts else math.nan,
                "run_dir": str(spec["run_dir"]),
                "output": str(spec["output"]),
            }
        )
    return out


def v(df: pd.DataFrame, label: str, metric: str) -> float:
    s = df.loc[df["label"] == label, metric]
    return finite(s.iloc[0]) if not s.empty else float("nan")


def decide(df: pd.DataFrame, manifest: Path | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "status": "rawfm_hybrid_lr_shortwarm_not_launched",
            "gpu_authorized_next": False,
            "decision": "no_manifest_found",
        }
    active = [
        "lrfast_nommd_candidate",
        "lrfast_nommd_control",
        "lrfast_g010_candidate",
        "lrfast_g010_control",
    ]
    pending = df[df["label"].isin(active) & (~df["has_final"] | (df["exit_code"] == "running"))]
    failed = df[df["label"].isin(active) & (~df["exit_code"].isin(["0", "running"]))]
    if not pending.empty:
        return {
            "status": "rawfm_hybrid_lr_shortwarm_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "decision": "wait_for_natural_completion",
            "pending_labels": pending["label"].tolist(),
        }
    if not failed.empty:
        return {
            "status": "rawfm_hybrid_lr_shortwarm_fail",
            "gpu_authorized_next": False,
            "decision": "read_failed_logs_before_relaunch",
            "failed_labels": failed["label"].tolist(),
        }
    pp_delta_nommd = v(df, "lrfast_nommd_candidate", "corr_pert_mean") - v(df, "lrfast_nommd_control", "corr_pert_mean")
    pp_delta_mmd = v(df, "lrfast_g010_candidate", "corr_pert_mean") - v(df, "lrfast_g010_control", "corr_pert_mean")
    mmd_delta_control = v(df, "lrfast_g010_control", "mmd") - v(df, "lrfast_g010_candidate", "mmd")
    mmd_improve_vs_lrfast_nommd = v(df, "lrfast_nommd_candidate", "mmd") - v(df, "lrfast_g010_candidate", "mmd")
    pp_retention_vs_lrfast_nommd = v(df, "lrfast_g010_candidate", "corr_pert_mean") - v(df, "lrfast_nommd_candidate", "corr_pert_mean")
    direct_drop_vs_default_nommd = v(df, "default_nommd_candidate_ref", "direct_pearson") - v(df, "lrfast_nommd_candidate", "direct_pearson")
    steps = int(v(df, "lrfast_g010_candidate", "train_mmd_steps")) if math.isfinite(v(df, "lrfast_g010_candidate", "train_mmd_steps")) else 0
    min_genes = int(v(df, "lrfast_g010_candidate", "train_mmd_visible_genes_min")) if steps else 0
    max_genes = int(v(df, "lrfast_g010_candidate", "train_mmd_visible_genes_max")) if steps else 0
    mask_ok = steps > 0 and min_genes == 256 and max_genes == 256
    best_exists = bool(df.loc[df["label"] == "lrfast_g010_candidate", "best_pt_exists"].iloc[0])
    pass_gate = bool(
        pp_delta_mmd >= 0.01
        and mmd_delta_control >= -0.005
        and mmd_improve_vs_lrfast_nommd >= 0.01
        and direct_drop_vs_default_nommd <= 0.02
        and mask_ok
        and not best_exists
    )
    partial_gate = bool(
        not pass_gate
        and pp_delta_mmd >= 0.005
        and mmd_improve_vs_lrfast_nommd >= 0.005
        and direct_drop_vs_default_nommd <= 0.03
        and mask_ok
        and not best_exists
    )
    if pass_gate:
        status = "rawfm_hybrid_lr_shortwarm_pass_promote_noharm_validation"
        decision = "promote_shortwarm_mmd_to_second_dataset_or_longer_noharm"
    elif partial_gate:
        status = "rawfm_hybrid_lr_shortwarm_partial"
        decision = "mutate_lr_schedule_or_gamma_with_more_steps"
    else:
        status = "rawfm_hybrid_lr_shortwarm_fail"
        decision = "close_shortwarm_mmd_mutation_and_move_to_new_information_axis"
    return {
        "status": status,
        "gpu_authorized_next": pass_gate,
        "decision": decision,
        "pp_delta_nommd_vs_control": pp_delta_nommd,
        "pp_delta_mmd_vs_control": pp_delta_mmd,
        "mmd_delta_vs_mmd_control": mmd_delta_control,
        "mmd_improve_vs_lrfast_nommd": mmd_improve_vs_lrfast_nommd,
        "pp_retention_vs_lrfast_nommd": pp_retention_vs_lrfast_nommd,
        "direct_drop_vs_default_nommd": direct_drop_vs_default_nommd,
        "mask_ok": mask_ok,
        "pass_gate": pass_gate,
        "partial_gate": partial_gate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="", help="launch_manifest TSV; default uses latest")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest) if args.manifest else latest_manifest()
    df = pd.DataFrame(collect_rows(manifest))
    decision = decide(df, manifest)
    suffix = manifest.stem.replace("launch_manifest_", "") if manifest else "not_launched"
    csv_path = OUT_DIR / f"rawfm_hybrid_lr_shortwarm_rows_{suffix}.csv"
    json_path = OUT_DIR / f"rawfm_hybrid_lr_shortwarm_{suffix}.json"
    report_path = OUT_DIR / f"LATENTFM_RAWFM_HYBRID_LR_SHORTWARM_CALIBRATION_{suffix}.md"
    df.to_csv(csv_path, index=False)
    payload = {
        "timestamp": now_cst(),
        "manifest": str(manifest) if manifest else "",
        **decision,
        "rows_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RawFM Hybrid Short-Warmup Calibration",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Manifest: `{payload['manifest']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized next: `{payload['gpu_authorized_next']}`",
        "",
        "## Boundary",
        "",
        "- Report-only posthoc over Wessels hybrid short-warmup smokes.",
        "- No checkpoint selection; final-only fixed-step evaluation.",
        "- Canonical multi and Track C query are not used.",
        "",
        "## Results",
        "",
        "| label | gamma | role | exit | final | direct | pc | pp | MMD | mmd steps | mmd genes min/max | best.pt |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['gamma']} | {row['role']} | {row['exit_code']} | {row['has_final']} | "
            f"{fmt(row['direct_pearson'])} | {fmt(row['corr_ctrl_mean'])} | "
            f"{fmt(row['corr_pert_mean'])} | {fmt(row['mmd'], 6)} | "
            f"{int(finite(row['train_mmd_steps'])) if math.isfinite(finite(row['train_mmd_steps'])) else 0} | "
            f"{fmt(row['train_mmd_visible_genes_min'], 0)}/{fmt(row['train_mmd_visible_genes_max'], 0)} | {row['best_pt_exists']} |"
        )
    lines.extend(["", "## Gate", "", f"- decision: `{payload['decision']}`"])
    for key, value in payload.items():
        if key in {
            "pp_delta_nommd_vs_control",
            "pp_delta_mmd_vs_control",
            "mmd_delta_vs_mmd_control",
            "mmd_improve_vs_lrfast_nommd",
            "pp_retention_vs_lrfast_nommd",
            "direct_drop_vs_default_nommd",
            "mask_ok",
            "pass_gate",
            "partial_gate",
        }:
            lines.append(f"- {key}: `{fmt(value) if isinstance(value, float) else value}`")
    if "pending_labels" in payload:
        lines.append(f"- pending labels: `{', '.join(payload['pending_labels'])}`")
    if "failed_labels" in payload:
        lines.append(f"- failed labels: `{', '.join(payload['failed_labels'])}`")
    lines.extend(["", "## Outputs", "", f"- rows: `{csv_path}`", f"- JSON: `{json_path}`", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
