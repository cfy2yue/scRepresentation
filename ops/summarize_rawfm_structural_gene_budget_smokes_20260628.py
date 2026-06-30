#!/usr/bin/env python3
"""Summarize RawFM structural Wessels k=256 gene-budget smokes."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/rawfm_structural_gene_budget_smoke_comparison_20260628"

RUNS = {
    "response_topk": {
        "family": "old_naive_candidate",
        "run_dir": ROOT / "runs/rawfm_wessels_gene_budget_response_topk_k256_smoke_20260628_1907",
        "output": ROOT / "CoupledFM/output/rawfm_gene_budget_smoke_20260628/wessels_response_topk_k256_seed42/ot",
    },
    "abundance_topk": {
        "family": "old_control",
        "run_dir": ROOT / "runs/rawfm_wessels_gene_budget_abundance_topk_k256_smoke_20260628_1917",
        "output": ROOT / "CoupledFM/output/rawfm_gene_budget_smoke_20260628/wessels_abundance_topk_k256_seed42/ot",
    },
    "abundance_matched_random": {
        "family": "old_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_gene_budget_abundance_matched_random_k256_smoke_20260628_1917",
        "output": ROOT
        / "CoupledFM/output/rawfm_gene_budget_smoke_20260628/wessels_abundance_matched_random_k256_seed42/ot",
    },
    "random_gene_set": {
        "family": "old_control",
        "run_dir": ROOT / "runs/rawfm_wessels_gene_budget_random_gene_set_k256_smoke_20260628_1926",
        "output": ROOT / "CoupledFM/output/rawfm_gene_budget_smoke_20260628/wessels_random_gene_set_k256_seed42/ot",
    },
    "response_abundance_residual_topk": {
        "family": "structural_candidate",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_response_abundance_residual_topk_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_response_abundance_residual_topk_k256_seed42/ot",
    },
    "residual_confound_matched_random": {
        "family": "structural_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_residual_confound_matched_random_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_residual_confound_matched_random_k256_seed42/ot",
    },
    "condition_diversity_topk": {
        "family": "structural_candidate_exploratory",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_condition_diversity_topk_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_condition_diversity_topk_k256_seed42/ot",
    },
    "residual_abundance_matched_random": {
        "family": "structural_control",
        "run_dir": ROOT
        / "runs/rawfm_wessels_structural_residual_abundance_matched_random_k256_smoke_20260628_1956",
        "output": ROOT
        / "CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628/wessels_residual_abundance_matched_random_k256_seed42/ot",
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
    rows = []
    for label, cfg in RUNS.items():
        exit_code = read_exit(cfg["run_dir"])
        metrics = read_final_metrics(cfg["output"])
        best_exists = (cfg["output"] / "best.pt").exists()
        rows.append(
            {
                "label": label,
                "family": cfg["family"],
                "exit_code": exit_code,
                "has_final": bool(metrics),
                "best_pt_exists": best_exists,
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


def delta(df: pd.DataFrame, label: str, controls: list[str], metric: str, higher: bool = True) -> float:
    cand = finite(df.loc[df["label"] == label, metric].iloc[0])
    vals = [finite(df.loc[df["label"] == c, metric].iloc[0]) for c in controls]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals or not math.isfinite(cand):
        return float("nan")
    base = max(vals) if higher else min(vals)
    return cand - base if higher else base - cand


def decide(df: pd.DataFrame) -> dict[str, Any]:
    pending = df[~df["has_final"] | (df["exit_code"].astype(str) == "running")]
    failed = df[(df["exit_code"].astype(str) != "0") & (df["exit_code"].astype(str) != "running")]
    if not pending.empty:
        return {
            "status": "rawfm_structural_gene_budget_smoke_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "pending_labels": pending["label"].tolist(),
            "decision": "wait_for_natural_completion",
        }
    if not failed.empty:
        return {
            "status": "rawfm_structural_gene_budget_smoke_fail",
            "gpu_authorized_next": False,
            "failed_labels": failed["label"].tolist(),
            "decision": "read_failed_logs_before_relaunch",
        }

    residual_controls = ["residual_confound_matched_random", "residual_abundance_matched_random"]
    broad_controls = [
        "abundance_topk",
        "abundance_matched_random",
        "random_gene_set",
        "residual_confound_matched_random",
        "residual_abundance_matched_random",
    ]
    residual_pp_delta = delta(df, "response_abundance_residual_topk", residual_controls, "corr_pert_mean")
    residual_mmd_delta = delta(df, "response_abundance_residual_topk", residual_controls, "mmd", higher=False)
    diversity_pp_delta = delta(df, "condition_diversity_topk", broad_controls, "corr_pert_mean")
    diversity_mmd_delta = delta(df, "condition_diversity_topk", broad_controls, "mmd", higher=False)

    residual_pass = bool(
        math.isfinite(residual_pp_delta)
        and residual_pp_delta >= 0.01
        and math.isfinite(residual_mmd_delta)
        and residual_mmd_delta >= -0.005
        and not bool(df.loc[df["label"] == "response_abundance_residual_topk", "best_pt_exists"].iloc[0])
    )
    diversity_pass = bool(
        math.isfinite(diversity_pp_delta)
        and diversity_pp_delta >= 0.01
        and math.isfinite(diversity_mmd_delta)
        and diversity_mmd_delta >= -0.005
        and not bool(df.loc[df["label"] == "condition_diversity_topk", "best_pt_exists"].iloc[0])
    )

    if residual_pass or diversity_pass:
        status = "rawfm_structural_gene_budget_smoke_candidate_pass_for_next_noharm_gate"
        decision = "promote_only_to_second_dataset_or_uncapped_noharm_gate"
    else:
        status = "rawfm_structural_gene_budget_smoke_controls_win_close_or_mutate"
        decision = "close_losing_x_or_mutate_before_any_k_sweep"
    return {
        "status": status,
        "gpu_authorized_next": False,
        "residual_pass": residual_pass,
        "diversity_pass": diversity_pass,
        "residual_pp_delta_vs_matched_controls": residual_pp_delta,
        "residual_mmd_delta_vs_matched_controls": residual_mmd_delta,
        "diversity_pp_delta_vs_broad_controls": diversity_pp_delta,
        "diversity_mmd_delta_vs_broad_controls": diversity_mmd_delta,
        "decision": decision,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    df = pd.DataFrame(rows)
    decision = decide(df)
    csv_path = OUT_DIR / "rawfm_structural_gene_budget_smoke_rows.csv"
    df.to_csv(csv_path, index=False)
    json_path = OUT_DIR / "rawfm_structural_gene_budget_smoke_comparison_20260628.json"
    payload = {
        "timestamp": now_cst(),
        **decision,
        "rows_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Structural Gene-Budget Smoke Comparison",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- Report-only posthoc over fixed-step/no-selection Wessels k=256 smokes.",
        "- No training, no inference, no checkpoint selection, no canonical multi, no Track C query.",
        "- Passing here only authorizes a later no-harm/second-dataset gate, not a final model claim.",
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
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- decision: `{payload['decision']}`",
        ]
    )
    for key in [
        "residual_pass",
        "diversity_pass",
        "residual_pp_delta_vs_matched_controls",
        "residual_mmd_delta_vs_matched_controls",
        "diversity_pp_delta_vs_broad_controls",
        "diversity_mmd_delta_vs_broad_controls",
    ]:
        if key in payload:
            lines.append(f"- {key}: `{fmt(payload[key]) if isinstance(payload[key], float) else payload[key]}`")
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
    (OUT_DIR / "LATENTFM_RAWFM_STRUCTURAL_GENE_BUDGET_SMOKE_COMPARISON_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
