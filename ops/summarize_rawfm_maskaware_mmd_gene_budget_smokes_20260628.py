#!/usr/bin/env python3
"""Summarize RawFM mask-aware MMD gene-budget smokes."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/rawfm_maskaware_mmd_gene_budget_smoke_comparison_20260628"

RUNS = {
    "residual_topk_nommd_ref": {
        "family": "nommd_reference_candidate",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_response_abundance_residual_topk_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_response_abundance_residual_topk_k256_seed42/ot",
    },
    "residual_confound_nommd_ref": {
        "family": "nommd_reference_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_residual_confound_matched_random_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_residual_confound_matched_random_k256_seed42/ot",
    },
    "hybrid_nommd_ref": {
        "family": "nommd_reference_candidate",
        "run_dir": ROOT
        / "runs/rawfm_wessels_hybrid_residual128_abundance128_hybrid_k256_smoke_20260628_2018",
        "output": ROOT
        / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_residual128_abundance128_hybrid_k256_seed42/ot",
    },
    "hybrid_control_nommd_ref": {
        "family": "nommd_reference_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_hybrid_confound128_abundance128_hybrid_control_k256_smoke_20260628_2018",
        "output": ROOT
        / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_confound128_abundance128_hybrid_control_k256_seed42/ot",
    },
    "residual_topk_mmd": {
        "family": "maskaware_mmd_candidate",
        "run_dir": ROOT / "runs/rawfm_wessels_mmd_residual_topk_k256_smoke_20260628_2127",
        "output": ROOT
        / "CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628/wessels_response_abundance_residual_topk_mmd_k256_seed42/ot",
    },
    "residual_confound_mmd_control": {
        "family": "maskaware_mmd_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_mmd_residual_confound_control_k256_smoke_20260628_2127",
        "output": ROOT
        / "CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628/wessels_residual_confound_matched_random_mmd_k256_seed42/ot",
    },
    "hybrid_mmd": {
        "family": "maskaware_mmd_candidate",
        "run_dir": ROOT
        / "runs/rawfm_wessels_mmd_hybrid_residual128_abundance128_k256_smoke_20260628_2127",
        "output": ROOT
        / "CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628/wessels_residual128_abundance128_hybrid_mmd_k256_seed42/ot",
    },
    "hybrid_mmd_control": {
        "family": "maskaware_mmd_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_mmd_hybrid_confound128_abundance128_control_k256_smoke_20260628_2127",
        "output": ROOT
        / "CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628/wessels_confound128_abundance128_hybrid_control_mmd_k256_seed42/ot",
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


def read_final_metrics(output: Path) -> dict[str, Any]:
    final = None
    for obj in read_log_rows(output):
        if obj.get("eval_type") == "final_test":
            final = obj
    return final or {}


def read_mmd_gene_counts(output: Path) -> list[int]:
    counts = []
    for obj in read_log_rows(output):
        if "train_mmd_visible_genes" in obj:
            counts.append(int(obj["train_mmd_visible_genes"]))
    return counts


def read_runlog_mmd_gene_counts(run_dir: Path) -> list[int]:
    log = run_dir / "logs/run.log"
    if not log.exists():
        return []
    counts = []
    for match in re.finditer(r"\bmmd_genes=(\d+)\b", log.read_text(encoding="utf-8", errors="replace")):
        counts.append(int(match.group(1)))
    return counts


def make_rows() -> list[dict[str, Any]]:
    rows = []
    for label, cfg in RUNS.items():
        metrics = read_final_metrics(cfg["output"])
        mmd_counts = read_mmd_gene_counts(cfg["output"])
        if not mmd_counts:
            mmd_counts = read_runlog_mmd_gene_counts(cfg["run_dir"])
        rows.append(
            {
                "label": label,
                "family": cfg["family"],
                "exit_code": read_exit(cfg["run_dir"]),
                "has_final": bool(metrics),
                "best_pt_exists": (cfg["output"] / "best.pt").exists(),
                "direct_pearson": finite(metrics.get("eval_direct_pearson")),
                "pearson_delta_ctrl": finite(metrics.get("eval_pearson_delta_ctrl")),
                "corr_ctrl_mean": finite(metrics.get("eval_corr_ctrl_mean")),
                "corr_pert_mean": finite(metrics.get("eval_corr_pert_mean")),
                "mmd": finite(metrics.get("eval_mmd")),
                "single_corr_pert": finite((metrics.get("eval_single") or {}).get("corr_pert_mean")),
                "multi_corr_pert": finite((metrics.get("eval_multi") or {}).get("corr_pert_mean")),
                "train_mmd_steps": len(mmd_counts),
                "train_mmd_visible_genes_min": min(mmd_counts) if mmd_counts else math.nan,
                "train_mmd_visible_genes_max": max(mmd_counts) if mmd_counts else math.nan,
                "run_dir": str(cfg["run_dir"]),
                "output": str(cfg["output"]),
            }
        )
    return rows


def val(df: pd.DataFrame, label: str, metric: str) -> float:
    s = df.loc[df["label"] == label, metric]
    if s.empty:
        return float("nan")
    return finite(s.iloc[0])


def branch_decision(df: pd.DataFrame, name: str, cand: str, ctrl: str, nommd: str) -> dict[str, Any]:
    pp_delta = val(df, cand, "corr_pert_mean") - val(df, ctrl, "corr_pert_mean")
    mmd_delta_vs_control = val(df, ctrl, "mmd") - val(df, cand, "mmd")
    mmd_improvement_vs_nommd = val(df, nommd, "mmd") - val(df, cand, "mmd")
    pp_retention_vs_nommd = val(df, cand, "corr_pert_mean") - val(df, nommd, "corr_pert_mean")
    best_exists = bool(df.loc[df["label"] == cand, "best_pt_exists"].iloc[0])
    mmd_steps = int(val(df, cand, "train_mmd_steps"))
    min_genes = int(val(df, cand, "train_mmd_visible_genes_min")) if mmd_steps else 0
    max_genes = int(val(df, cand, "train_mmd_visible_genes_max")) if mmd_steps else 0
    mask_ok = mmd_steps > 0 and min_genes == 256 and max_genes == 256
    pass_gate = bool(
        pp_delta >= 0.01
        and mmd_delta_vs_control >= -0.005
        and mmd_improvement_vs_nommd > 0.0
        and mask_ok
        and not best_exists
    )
    partial_gate = bool(
        not pass_gate
        and pp_delta >= 0.005
        and mmd_improvement_vs_nommd >= 0.01
        and mask_ok
        and not best_exists
    )
    if pass_gate:
        status = f"{name}_pass_promote_second_dataset_noharm"
    elif partial_gate:
        status = f"{name}_partial_mutate_gamma_or_support"
    else:
        status = f"{name}_fail_close_exact_mmd"
    return {
        f"{name}_status": status,
        f"{name}_pass": pass_gate,
        f"{name}_partial": partial_gate,
        f"{name}_pp_delta_vs_control": pp_delta,
        f"{name}_mmd_delta_vs_control": mmd_delta_vs_control,
        f"{name}_mmd_improvement_vs_nommd": mmd_improvement_vs_nommd,
        f"{name}_pp_retention_vs_nommd": pp_retention_vs_nommd,
        f"{name}_mask_ok": mask_ok,
    }


def decide(df: pd.DataFrame) -> dict[str, Any]:
    mmd_labels = ["residual_topk_mmd", "residual_confound_mmd_control", "hybrid_mmd", "hybrid_mmd_control"]
    pending = df[df["label"].isin(mmd_labels) & (~df["has_final"] | (df["exit_code"] == "running"))]
    failed = df[df["label"].isin(mmd_labels) & (~df["exit_code"].isin(["0", "running"]))]
    if not pending.empty:
        return {
            "status": "rawfm_maskaware_mmd_smoke_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "decision": "wait_for_natural_completion",
            "pending_labels": pending["label"].tolist(),
        }
    if not failed.empty:
        return {
            "status": "rawfm_maskaware_mmd_smoke_fail",
            "gpu_authorized_next": False,
            "decision": "read_failed_logs_before_relaunch",
            "failed_labels": failed["label"].tolist(),
        }

    residual = branch_decision(
        df,
        "residual",
        "residual_topk_mmd",
        "residual_confound_mmd_control",
        "residual_topk_nommd_ref",
    )
    hybrid = branch_decision(
        df,
        "hybrid",
        "hybrid_mmd",
        "hybrid_mmd_control",
        "hybrid_nommd_ref",
    )
    any_pass = residual["residual_pass"] or hybrid["hybrid_pass"]
    any_partial = residual["residual_partial"] or hybrid["hybrid_partial"]
    if any_pass:
        status = "rawfm_maskaware_mmd_smoke_pass_promote_noharm_validation"
        decision = "launch_second_dataset_or_uncapped_noharm_for_passing_branch_only"
    elif any_partial:
        status = "rawfm_maskaware_mmd_smoke_partial_mutate"
        decision = "mutate_gamma_support_or_budget_before_more_gpu"
    else:
        status = "rawfm_maskaware_mmd_smoke_fail_close_exact_stabilizer"
        decision = "close_exact_mmd_stabilizer_and_return_to_zscape_cpu_or_new_x"
    return {
        "status": status,
        "gpu_authorized_next": bool(any_pass),
        "decision": decision,
        **residual,
        **hybrid,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(make_rows())
    decision = decide(df)
    csv_path = OUT_DIR / "rawfm_maskaware_mmd_gene_budget_smoke_rows.csv"
    json_path = OUT_DIR / "rawfm_maskaware_mmd_gene_budget_smoke_comparison_20260628.json"
    df.to_csv(csv_path, index=False)
    payload = {"timestamp": now_cst(), **decision, "rows_csv": str(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Mask-Aware MMD Gene-Budget Smoke Comparison",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized next: `{payload['gpu_authorized_next']}`",
        "",
        "## Boundary",
        "",
        "- Report-only posthoc over fixed-step/no-selection Wessels k=256 smokes.",
        "- MMD smokes use deterministic gene-budget masks, `--gene-mask-prob 0`, and mask-aware train MMD.",
        "- No canonical multi or Track C query enters selection.",
        "",
        "## Results",
        "",
        "| label | family | exit | final | direct | pdelta | pc | pp | MMD | mmd steps | mmd genes min/max | best.pt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['family']} | {row['exit_code']} | {row['has_final']} | "
            f"{fmt(row['direct_pearson'])} | {fmt(row['pearson_delta_ctrl'])} | "
            f"{fmt(row['corr_ctrl_mean'])} | {fmt(row['corr_pert_mean'])} | "
            f"{fmt(row['mmd'], 6)} | {int(finite(row['train_mmd_steps'])) if math.isfinite(finite(row['train_mmd_steps'])) else 0} | "
            f"{fmt(row['train_mmd_visible_genes_min'], 0)}/{fmt(row['train_mmd_visible_genes_max'], 0)} | "
            f"{row['best_pt_exists']} |"
        )
    lines.extend(["", "## Gate", "", f"- decision: `{payload['decision']}`"])
    for key, value in payload.items():
        if key.startswith(("residual_", "hybrid_")):
            lines.append(f"- {key}: `{fmt(value) if isinstance(value, float) else value}`")
    if "pending_labels" in payload:
        lines.append(f"- pending labels: `{', '.join(payload['pending_labels'])}`")
    if "failed_labels" in payload:
        lines.append(f"- failed labels: `{', '.join(payload['failed_labels'])}`")
    lines.extend(["", "## Outputs", "", f"- rows: `{csv_path}`", f"- JSON: `{json_path}`", ""])
    (OUT_DIR / "LATENTFM_RAWFM_MASKAWARE_MMD_GENE_BUDGET_SMOKE_COMPARISON_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
