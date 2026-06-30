#!/usr/bin/env python3
"""Summarize RawFM low-dose residual short-warmup smokes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/rawfm_lowdose_residual_lrfast_smoke_comparison_20260628"


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


def collect_rows(manifest: Path | None) -> list[dict[str, Any]]:
    specs = read_manifest(manifest) if manifest is not None else []
    out = []
    for spec in specs:
        final = read_final(spec["output"])
        out.append(
            {
                "label": spec["label"],
                "role": spec["role"],
                "exit_code": read_exit(spec["run_dir"]),
                "has_final": bool(final),
                "best_pt_exists": (spec["output"] / "best.pt").exists(),
                "direct_pearson": finite(final.get("eval_direct_pearson")),
                "pearson_delta_ctrl": finite(final.get("eval_pearson_delta_ctrl")),
                "corr_ctrl_mean": finite(final.get("eval_corr_ctrl_mean")),
                "corr_pert_mean": finite(final.get("eval_corr_pert_mean")),
                "mmd": finite(final.get("eval_mmd")),
                "run_dir": str(spec["run_dir"]),
                "output": str(spec["output"]),
            }
        )
    return out


def v(df: pd.DataFrame, label: str, metric: str) -> float:
    s = df.loc[df["label"] == label, metric]
    return finite(s.iloc[0]) if not s.empty else float("nan")


def dose_gate(df: pd.DataFrame, dose: str, cand: str, ctrl: str) -> dict[str, Any]:
    pp_delta = v(df, cand, "corr_pert_mean") - v(df, ctrl, "corr_pert_mean")
    mmd_delta = v(df, ctrl, "mmd") - v(df, cand, "mmd")
    direct_delta = v(df, cand, "direct_pearson") - v(df, ctrl, "direct_pearson")
    best_exists = bool(df.loc[df["label"] == cand, "best_pt_exists"].iloc[0])
    pass_gate = bool(
        pp_delta >= 0.01
        and mmd_delta >= -0.005
        and v(df, cand, "mmd") <= 0.030
        and direct_delta >= -0.02
        and not best_exists
    )
    partial_gate = bool(
        not pass_gate
        and pp_delta >= 0.005
        and mmd_delta >= -0.010
        and v(df, cand, "mmd") <= 0.040
        and direct_delta >= -0.03
        and not best_exists
    )
    return {
        f"{dose}_pass": pass_gate,
        f"{dose}_partial": partial_gate,
        f"{dose}_pp_delta_vs_control": pp_delta,
        f"{dose}_mmd_delta_vs_control": mmd_delta,
        f"{dose}_direct_delta_vs_control": direct_delta,
        f"{dose}_candidate_mmd": v(df, cand, "mmd"),
    }


def decide(df: pd.DataFrame, manifest: Path | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "status": "rawfm_lowdose_residual_lrfast_not_launched",
            "gpu_authorized_next": False,
            "decision": "no_manifest_found",
        }
    pending = df[(~df["has_final"]) | (df["exit_code"] == "running")]
    failed = df[~df["exit_code"].isin(["0", "running"])]
    if not pending.empty:
        return {
            "status": "rawfm_lowdose_residual_lrfast_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "decision": "wait_for_natural_completion",
            "pending_labels": pending["label"].tolist(),
        }
    if not failed.empty:
        return {
            "status": "rawfm_lowdose_residual_lrfast_fail",
            "gpu_authorized_next": False,
            "decision": "read_failed_logs_before_relaunch",
            "failed_labels": failed["label"].tolist(),
        }
    g32 = dose_gate(
        df,
        "residual32",
        "residual32_abundance96_random128",
        "confound32_abundance96_random128_control",
    )
    g64 = dose_gate(
        df,
        "residual64",
        "residual64_abundance96_random96",
        "confound64_abundance96_random96_control",
    )
    any_pass = g32["residual32_pass"] or g64["residual64_pass"]
    any_partial = g32["residual32_partial"] or g64["residual64_partial"]
    if any_pass:
        status = "rawfm_lowdose_residual_lrfast_pass_promote_noharm_validation"
        decision = "promote_passing_dose_to_second_dataset_or_longer_noharm"
    elif any_partial:
        status = "rawfm_lowdose_residual_lrfast_partial"
        decision = "mutate_near_partial_dose_or_add_ot_pool_ablation"
    else:
        status = "rawfm_lowdose_residual_lrfast_fail_close_lowdose_family"
        decision = "close_lowdose_residual_family_or_switch_to_ot_pairing_axis"
    return {
        "status": status,
        "gpu_authorized_next": bool(any_pass),
        "decision": decision,
        **g32,
        **g64,
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
    csv_path = OUT_DIR / f"rawfm_lowdose_residual_lrfast_rows_{suffix}.csv"
    json_path = OUT_DIR / f"rawfm_lowdose_residual_lrfast_{suffix}.json"
    report_path = OUT_DIR / f"LATENTFM_RAWFM_LOWDOSE_RESIDUAL_LRFAST_SMOKE_COMPARISON_{suffix}.md"
    df.to_csv(csv_path, index=False)
    payload = {
        "timestamp": now_cst(),
        "manifest": str(manifest) if manifest else "",
        **decision,
        "rows_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RawFM Low-Dose Residual Short-Warmup Smoke Comparison",
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
        "- Report-only posthoc over Wessels low-dose residual smokes.",
        "- No checkpoint selection; final-only fixed-step evaluation.",
        "- Canonical multi and Track C query are not used.",
        "",
        "## Results",
        "",
        "| label | role | exit | final | direct | pc | pp | MMD | best.pt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['role']} | {row['exit_code']} | {row['has_final']} | "
            f"{fmt(row['direct_pearson'])} | {fmt(row['corr_ctrl_mean'])} | "
            f"{fmt(row['corr_pert_mean'])} | {fmt(row['mmd'], 6)} | {row['best_pt_exists']} |"
        )
    lines.extend(["", "## Gate", "", f"- decision: `{payload['decision']}`"])
    for key, value in payload.items():
        if key.startswith(("residual32_", "residual64_")):
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
