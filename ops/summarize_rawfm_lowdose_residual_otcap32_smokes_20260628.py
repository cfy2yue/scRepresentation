#!/usr/bin/env python3
"""Summarize RawFM low-dose residual OT cap32 smokes."""

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
OUT_DIR = ROOT / "reports/rawfm_lowdose_residual_otcap32_smoke_comparison_20260628"


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
                    "mode": row["mode"],
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


def read_final(output: Path) -> dict[str, Any]:
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


def collect_rows(manifest: Path | None) -> list[dict[str, Any]]:
    specs = read_manifest(manifest) if manifest is not None else []
    out = []
    for spec in specs:
        final = read_final(spec["output"])
        out.append(
            {
                "label": spec["label"],
                "mode": spec["mode"],
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


def row_value(df: pd.DataFrame, mode: str, role: str, metric: str) -> float:
    s = df.loc[(df["mode"] == mode) & (df["role"] == role), metric]
    return finite(s.iloc[0]) if not s.empty else float("nan")


def mode_gate(df: pd.DataFrame, mode: str) -> dict[str, Any]:
    pp_delta = row_value(df, mode, "candidate", "corr_pert_mean") - row_value(df, mode, "control", "corr_pert_mean")
    mmd_delta = row_value(df, mode, "control", "mmd") - row_value(df, mode, "candidate", "mmd")
    direct_delta = row_value(df, mode, "candidate", "direct_pearson") - row_value(df, mode, "control", "direct_pearson")
    pass_gate = bool(
        pp_delta >= 0.01
        and mmd_delta >= -0.005
        and row_value(df, mode, "candidate", "mmd") <= 0.030
        and direct_delta >= -0.02
    )
    return {
        f"{mode}_pass": pass_gate,
        f"{mode}_pp_delta_vs_control": pp_delta,
        f"{mode}_mmd_delta_vs_control": mmd_delta,
        f"{mode}_direct_delta_vs_control": direct_delta,
        f"{mode}_candidate_mmd": row_value(df, mode, "candidate", "mmd"),
    }


def decide(df: pd.DataFrame, manifest: Path | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "status": "rawfm_lowdose_residual_otcap32_not_launched",
            "gpu_authorized_next": False,
            "decision": "no_manifest_found",
        }
    pending = df[(~df["has_final"]) | (df["exit_code"] == "running")]
    failed = df[~df["exit_code"].isin(["0", "running"])]
    if not pending.empty:
        return {
            "status": "rawfm_lowdose_residual_otcap32_pending_wait_no_polling",
            "gpu_authorized_next": False,
            "decision": "wait_for_natural_completion",
            "pending_labels": pending["mode"].astype(str).str.cat(pending["role"].astype(str), sep="/").tolist(),
        }
    if not failed.empty:
        return {
            "status": "rawfm_lowdose_residual_otcap32_fail",
            "gpu_authorized_next": False,
            "decision": "read_failed_logs_before_relaunch",
            "failed_labels": failed["mode"].astype(str).str.cat(failed["role"].astype(str), sep="/").tolist(),
        }
    assignment = mode_gate(df, "assignment")
    multinomial = mode_gate(df, "multinomial")
    any_pass = assignment["assignment_pass"] or multinomial["multinomial_pass"]
    return {
        "status": "rawfm_lowdose_residual_otcap32_pass" if any_pass else "rawfm_lowdose_residual_otcap32_fail",
        "gpu_authorized_next": bool(any_pass),
        "decision": "compare_passing_mode_to_default_lowdose_baseline" if any_pass else "close_otcap32_modes_or_revisit_pairing_later",
        **assignment,
        **multinomial,
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
    csv_path = OUT_DIR / f"rawfm_lowdose_residual_otcap32_rows_{suffix}.csv"
    json_path = OUT_DIR / f"rawfm_lowdose_residual_otcap32_{suffix}.json"
    report_path = OUT_DIR / f"LATENTFM_RAWFM_LOWDOSE_RESIDUAL_OTCAP32_SMOKE_COMPARISON_{suffix}.md"
    df.to_csv(csv_path, index=False)
    payload = {
        "timestamp": now_cst(),
        "manifest": str(manifest) if manifest else "",
        **decision,
        "rows_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RawFM Low-Dose Residual OT Cap32 Smoke Comparison",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Manifest: `{payload['manifest']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized next: `{payload['gpu_authorized_next']}`",
        "",
        "## Results",
        "",
        "| mode | role | exit | final | direct | pc | pp | MMD | best.pt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['mode']} | {row['role']} | {row['exit_code']} | {row['has_final']} | "
            f"{fmt(row['direct_pearson'])} | {fmt(row['corr_ctrl_mean'])} | "
            f"{fmt(row['corr_pert_mean'])} | {fmt(row['mmd'], 6)} | {row['best_pt_exists']} |"
        )
    lines.extend(["", "## Gate", "", f"- decision: `{payload['decision']}`"])
    for key, value in payload.items():
        if key.startswith(("assignment_", "multinomial_")):
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
