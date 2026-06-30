#!/usr/bin/env python3
"""Summarize RawFM hybrid MMD gamma calibration smokes."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/rawfm_hybrid_mmd_gamma_calibration_20260628"

RUNS = {
    "nommd_candidate_ref": {
        "gamma": "none",
        "role": "candidate_ref",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_residual128_abundance128_hybrid_k256_smoke_20260628_2018",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_residual128_abundance128_hybrid_k256_seed42/ot",
    },
    "nommd_control_ref": {
        "gamma": "none",
        "role": "control_ref",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_confound128_abundance128_hybrid_control_k256_smoke_20260628_2018",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_confound128_abundance128_hybrid_control_k256_seed42/ot",
    },
    "g0001_candidate_ref": {
        "gamma": "0.001",
        "role": "candidate_ref",
        "run_dir": ROOT / "runs/rawfm_wessels_mmd_hybrid_residual128_abundance128_k256_smoke_20260628_2127",
        "output": ROOT / "CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628/wessels_residual128_abundance128_hybrid_mmd_k256_seed42/ot",
    },
    "g0001_control_ref": {
        "gamma": "0.001",
        "role": "control_ref",
        "run_dir": ROOT / "runs/rawfm_wessels_mmd_hybrid_confound128_abundance128_control_k256_smoke_20260628_2127",
        "output": ROOT / "CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628/wessels_confound128_abundance128_hybrid_control_mmd_k256_seed42/ot",
    },
    "g002_candidate": {
        "gamma": "0.02",
        "role": "candidate",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_mmd_g002_residual128_abundance128_hybrid_k256_smoke_20260628_2144",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_mmd_gamma_calibration_20260628/wessels_g002_residual128_abundance128_hybrid_k256_seed42/ot",
    },
    "g002_control": {
        "gamma": "0.02",
        "role": "control",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_mmd_g002_confound128_abundance128_hybrid_control_k256_smoke_20260628_2144",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_mmd_gamma_calibration_20260628/wessels_g002_confound128_abundance128_hybrid_control_k256_seed42/ot",
    },
    "g010_candidate": {
        "gamma": "0.10",
        "role": "candidate",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_mmd_g010_residual128_abundance128_hybrid_k256_smoke_20260628_2144",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_mmd_gamma_calibration_20260628/wessels_g010_residual128_abundance128_hybrid_k256_seed42/ot",
    },
    "g010_control": {
        "gamma": "0.10",
        "role": "control",
        "run_dir": ROOT / "runs/rawfm_wessels_hybrid_mmd_g010_confound128_abundance128_hybrid_control_k256_smoke_20260628_2144",
        "output": ROOT / "CoupledFM/output/rawfm_hybrid_mmd_gamma_calibration_20260628/wessels_g010_confound128_abundance128_hybrid_control_k256_seed42/ot",
    },
}


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
    return [int(x) for x in re.findall(r"\bmmd_genes=(\d+)\b", log.read_text(encoding="utf-8", errors="replace"))]


def rows() -> list[dict[str, Any]]:
    out = []
    for label, cfg in RUNS.items():
        final = read_final(cfg["output"])
        counts = read_mmd_genes(cfg["run_dir"])
        out.append(
            {
                "label": label,
                "gamma": cfg["gamma"],
                "role": cfg["role"],
                "exit_code": read_exit(cfg["run_dir"]),
                "has_final": bool(final),
                "best_pt_exists": (cfg["output"] / "best.pt").exists(),
                "direct_pearson": finite(final.get("eval_direct_pearson")),
                "pearson_delta_ctrl": finite(final.get("eval_pearson_delta_ctrl")),
                "corr_ctrl_mean": finite(final.get("eval_corr_ctrl_mean")),
                "corr_pert_mean": finite(final.get("eval_corr_pert_mean")),
                "mmd": finite(final.get("eval_mmd")),
                "train_mmd_steps": len(counts),
                "train_mmd_visible_genes_min": min(counts) if counts else math.nan,
                "train_mmd_visible_genes_max": max(counts) if counts else math.nan,
                "run_dir": str(cfg["run_dir"]),
                "output": str(cfg["output"]),
            }
        )
    return out


def v(df: pd.DataFrame, label: str, metric: str) -> float:
    s = df.loc[df["label"] == label, metric]
    return finite(s.iloc[0]) if not s.empty else float("nan")


def gamma_gate(df: pd.DataFrame, gamma_label: str, cand: str, ctrl: str) -> dict[str, Any]:
    pp_delta = v(df, cand, "corr_pert_mean") - v(df, ctrl, "corr_pert_mean")
    mmd_delta_control = v(df, ctrl, "mmd") - v(df, cand, "mmd")
    mmd_improve_nommd = v(df, "nommd_candidate_ref", "mmd") - v(df, cand, "mmd")
    pp_retention_nommd = v(df, cand, "corr_pert_mean") - v(df, "nommd_candidate_ref", "corr_pert_mean")
    steps = int(v(df, cand, "train_mmd_steps")) if math.isfinite(v(df, cand, "train_mmd_steps")) else 0
    min_genes = int(v(df, cand, "train_mmd_visible_genes_min")) if steps else 0
    max_genes = int(v(df, cand, "train_mmd_visible_genes_max")) if steps else 0
    mask_ok = steps > 0 and min_genes == 256 and max_genes == 256
    best_exists = bool(df.loc[df["label"] == cand, "best_pt_exists"].iloc[0])
    pass_gate = bool(
        pp_delta >= 0.01
        and mmd_delta_control >= -0.005
        and mmd_improve_nommd >= 0.01
        and mask_ok
        and not best_exists
    )
    partial_gate = bool(
        not pass_gate
        and mmd_improve_nommd >= 0.01
        and pp_delta >= 0.005
        and mask_ok
        and not best_exists
    )
    return {
        f"{gamma_label}_pass": pass_gate,
        f"{gamma_label}_partial": partial_gate,
        f"{gamma_label}_pp_delta_vs_control": pp_delta,
        f"{gamma_label}_mmd_delta_vs_control": mmd_delta_control,
        f"{gamma_label}_mmd_improvement_vs_nommd": mmd_improve_nommd,
        f"{gamma_label}_pp_retention_vs_nommd": pp_retention_nommd,
        f"{gamma_label}_mask_ok": mask_ok,
    }


def decide(df: pd.DataFrame) -> dict[str, Any]:
    active = ["g002_candidate", "g002_control", "g010_candidate", "g010_control"]
    pending = df[df["label"].isin(active) & (~df["has_final"] | (df["exit_code"] == "running"))]
    failed = df[df["label"].isin(active) & (~df["exit_code"].isin(["0", "running"]))]
    if not pending.empty:
        return {
            "status": "rawfm_hybrid_mmd_gamma_calibration_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "decision": "wait_for_natural_completion",
            "pending_labels": pending["label"].tolist(),
        }
    if not failed.empty:
        return {
            "status": "rawfm_hybrid_mmd_gamma_calibration_fail",
            "gpu_authorized_next": False,
            "decision": "read_failed_logs_before_relaunch",
            "failed_labels": failed["label"].tolist(),
        }
    g002 = gamma_gate(df, "g002", "g002_candidate", "g002_control")
    g010 = gamma_gate(df, "g010", "g010_candidate", "g010_control")
    any_pass = g002["g002_pass"] or g010["g010_pass"]
    any_partial = g002["g002_partial"] or g010["g010_partial"]
    if any_pass:
        status = "rawfm_hybrid_mmd_gamma_calibration_pass_promote_noharm_validation"
        decision = "promote_passing_gamma_to_second_dataset_or_uncapped_noharm"
    elif any_partial:
        status = "rawfm_hybrid_mmd_gamma_calibration_partial"
        decision = "mutate_near_partial_gamma_or_support_balance"
    else:
        status = "rawfm_hybrid_mmd_gamma_calibration_fail_close_mmd_family"
        decision = "close_hybrid_mmd_gamma_family_and_move_to_new_x_or_zscape_cpu_repair"
    return {
        "status": status,
        "gpu_authorized_next": bool(any_pass),
        "decision": decision,
        **g002,
        **g010,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows())
    decision = decide(df)
    csv_path = OUT_DIR / "rawfm_hybrid_mmd_gamma_calibration_rows.csv"
    json_path = OUT_DIR / "rawfm_hybrid_mmd_gamma_calibration_20260628.json"
    df.to_csv(csv_path, index=False)
    payload = {"timestamp": now_cst(), **decision, "rows_csv": str(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Hybrid MMD Gamma Calibration",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized next: `{payload['gpu_authorized_next']}`",
        "",
        "## Boundary",
        "",
        "- Report-only posthoc over Wessels hybrid MMD gamma calibration smokes.",
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
        if key.startswith(("g002_", "g010_")):
            lines.append(f"- {key}: `{fmt(value) if isinstance(value, float) else value}`")
    if "pending_labels" in payload:
        lines.append(f"- pending labels: `{', '.join(payload['pending_labels'])}`")
    if "failed_labels" in payload:
        lines.append(f"- failed labels: `{', '.join(payload['failed_labels'])}`")
    lines.extend(["", "## Outputs", "", f"- rows: `{csv_path}`", f"- JSON: `{json_path}`", ""])
    (OUT_DIR / "LATENTFM_RAWFM_HYBRID_MMD_GAMMA_CALIBRATION_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
