#!/usr/bin/env python3
"""Summarize RawFM hybrid residual+abundance Wessels k=256 smokes."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/rawfm_hybrid_gene_budget_smoke_comparison_20260628"

RUNS = {
    "full_residual_topk_reference": {
        "family": "prior_structural_candidate_reference",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_response_abundance_residual_topk_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_response_abundance_residual_topk_k256_seed42/ot",
    },
    "residual_confound_matched_reference": {
        "family": "prior_structural_control_reference",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_residual_confound_matched_random_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_residual_confound_matched_random_k256_seed42/ot",
    },
    "residual_abundance_matched_reference": {
        "family": "prior_structural_control_reference",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_residual_abundance_matched_random_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_residual_abundance_matched_random_k256_seed42/ot",
    },
    "hybrid_residual128_abundance128": {
        "family": "hybrid_candidate",
        "run_dir": ROOT
        / "runs/rawfm_wessels_hybrid_residual128_abundance128_hybrid_k256_smoke_20260628_2018",
        "output": ROOT
        / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_residual128_abundance128_hybrid_k256_seed42/ot",
    },
    "hybrid_confound128_abundance128_control": {
        "family": "hybrid_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_hybrid_confound128_abundance128_hybrid_control_k256_smoke_20260628_2018",
        "output": ROOT
        / "CoupledFM/output/rawfm_hybrid_gene_budget_smoke_20260628/wessels_confound128_abundance128_hybrid_control_k256_seed42/ot",
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


def read_final_metrics(output: Path) -> dict[str, Any]:
    log = output / "train_log.jsonl"
    if not log.exists():
        return {}
    final = None
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("eval_type") == "final_test":
            final = obj
    return final or {}


def make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, cfg in RUNS.items():
        metrics = read_final_metrics(cfg["output"])
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
                "run_dir": str(cfg["run_dir"]),
                "output": str(cfg["output"]),
            }
        )
    return rows


def value(df: pd.DataFrame, label: str, metric: str) -> float:
    vals = df.loc[df["label"] == label, metric]
    if vals.empty:
        return float("nan")
    return finite(vals.iloc[0])


def decide(df: pd.DataFrame) -> dict[str, Any]:
    pending = df[df["label"].str.startswith("hybrid_") & (~df["has_final"] | (df["exit_code"] == "running"))]
    failed = df[df["label"].str.startswith("hybrid_") & (~df["exit_code"].isin(["0", "running"]))]
    if not pending.empty:
        return {
            "status": "rawfm_hybrid_gene_budget_smoke_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "decision": "wait_for_natural_completion",
            "pending_labels": pending["label"].tolist(),
        }
    if not failed.empty:
        return {
            "status": "rawfm_hybrid_gene_budget_smoke_fail",
            "gpu_authorized_next": False,
            "decision": "read_failed_logs_before_relaunch",
            "failed_labels": failed["label"].tolist(),
        }

    cand = "hybrid_residual128_abundance128"
    ctrl = "hybrid_confound128_abundance128_control"
    full = "full_residual_topk_reference"
    matched_refs = ["residual_confound_matched_reference", "residual_abundance_matched_reference"]

    cand_pp = value(df, cand, "corr_pert_mean")
    ctrl_pp = value(df, ctrl, "corr_pert_mean")
    cand_mmd = value(df, cand, "mmd")
    ctrl_mmd = value(df, ctrl, "mmd")
    full_mmd = value(df, full, "mmd")
    best_ref_pp = max(value(df, r, "corr_pert_mean") for r in matched_refs)
    best_ref_mmd = min(value(df, r, "mmd") for r in matched_refs)

    pp_delta_vs_hybrid_control = cand_pp - ctrl_pp
    pp_delta_vs_prior_matched_controls = cand_pp - best_ref_pp
    mmd_delta_vs_hybrid_control = ctrl_mmd - cand_mmd
    mmd_delta_vs_full_residual = full_mmd - cand_mmd
    mmd_delta_vs_prior_best_control = best_ref_mmd - cand_mmd
    best_exists = bool(df.loc[df["label"] == cand, "best_pt_exists"].iloc[0])

    pass_gate = bool(
        math.isfinite(pp_delta_vs_hybrid_control)
        and pp_delta_vs_hybrid_control >= 0.01
        and math.isfinite(mmd_delta_vs_hybrid_control)
        and mmd_delta_vs_hybrid_control >= -0.005
        and math.isfinite(mmd_delta_vs_full_residual)
        and mmd_delta_vs_full_residual > 0.0
        and not best_exists
    )
    mutate_gate = bool(
        not pass_gate
        and math.isfinite(pp_delta_vs_hybrid_control)
        and pp_delta_vs_hybrid_control >= 0.005
        and math.isfinite(mmd_delta_vs_full_residual)
        and mmd_delta_vs_full_residual >= 0.025
        and not best_exists
    )

    if pass_gate:
        status = "rawfm_hybrid_gene_budget_smoke_pass_promote_to_second_dataset_noharm"
        decision = "promote_only_to_second_dataset_or_uncapped_noharm_gate"
        gpu_next = True
    elif mutate_gate:
        status = "rawfm_hybrid_gene_budget_smoke_partial_mutate_no_k_sweep"
        decision = "mutate_hybrid_or_add_mask_aware_noharm_before_more_gpu"
        gpu_next = False
    else:
        status = "rawfm_hybrid_gene_budget_smoke_fail_close_exact_hybrid"
        decision = "close_exact_hybrid_and_move_to_mask_aware_mmd_or_zscape_cpu_repair"
        gpu_next = False

    return {
        "status": status,
        "gpu_authorized_next": gpu_next,
        "decision": decision,
        "pass_gate": pass_gate,
        "mutate_gate": mutate_gate,
        "pp_delta_vs_hybrid_control": pp_delta_vs_hybrid_control,
        "pp_delta_vs_prior_matched_controls": pp_delta_vs_prior_matched_controls,
        "mmd_delta_vs_hybrid_control": mmd_delta_vs_hybrid_control,
        "mmd_delta_vs_full_residual": mmd_delta_vs_full_residual,
        "mmd_delta_vs_prior_best_control": mmd_delta_vs_prior_best_control,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(make_rows())
    decision = decide(df)
    csv_path = OUT_DIR / "rawfm_hybrid_gene_budget_smoke_rows.csv"
    json_path = OUT_DIR / "rawfm_hybrid_gene_budget_smoke_comparison_20260628.json"
    df.to_csv(csv_path, index=False)
    payload = {
        "timestamp": now_cst(),
        **decision,
        "rows_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Hybrid Gene-Budget Smoke Comparison",
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
        "- Hybrid candidate uses 128 residualized-response genes plus 128 abundance-anchor genes.",
        "- Hybrid control uses 128 residual-confound genes plus an abundance-anchor half.",
        "- No training, no inference, no checkpoint selection, no canonical multi, no Track C query.",
        "",
        "## Results",
        "",
        "| label | family | exit | final | direct | pdelta | pc | pp | MMD | single pp | multi pp | best.pt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['family']} | {row['exit_code']} | {row['has_final']} | "
            f"{fmt(row['direct_pearson'])} | {fmt(row['pearson_delta_ctrl'])} | "
            f"{fmt(row['corr_ctrl_mean'])} | {fmt(row['corr_pert_mean'])} | "
            f"{fmt(row['mmd'], 6)} | {fmt(row['single_corr_pert'])} | "
            f"{fmt(row['multi_corr_pert'])} | {row['best_pt_exists']} |"
        )

    lines.extend(["", "## Gate", "", f"- decision: `{payload['decision']}`"])
    for key in [
        "pass_gate",
        "mutate_gate",
        "pp_delta_vs_hybrid_control",
        "pp_delta_vs_prior_matched_controls",
        "mmd_delta_vs_hybrid_control",
        "mmd_delta_vs_full_residual",
        "mmd_delta_vs_prior_best_control",
    ]:
        if key in payload:
            val = payload[key]
            lines.append(f"- {key}: `{fmt(val) if isinstance(val, float) else val}`")
    if "pending_labels" in payload:
        lines.append(f"- pending labels: `{', '.join(payload['pending_labels'])}`")
    if "failed_labels" in payload:
        lines.append(f"- failed labels: `{', '.join(payload['failed_labels'])}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- rows: `{csv_path}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    (OUT_DIR / "LATENTFM_RAWFM_HYBRID_GENE_BUDGET_SMOKE_COMPARISON_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
