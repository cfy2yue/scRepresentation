#!/usr/bin/env python3
"""Summarize the Track A cross-latent gene-reliability adapter block.

This is a read-only status/decision aggregator. It reads the launch manifest,
run marker files, and per-run Track A candidate gate JSONs if present. It does
not read training logs, canonical multi outputs, or held-out Track C query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_MANIFEST = ROOT / "reports/latentfm_crosslatent_tracka_gene_reliability_adapter_manifest_20260623.jsonl"
DEFAULT_RUN_ROOT = ROOT / "runs/latentfm_crosslatent_tracka_gene_reliability_adapter_20260623"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_crosslatent_tracka_gene_reliability_adapter_summary_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_CROSSLATENT_TRACKA_GENE_RELIABILITY_ADAPTER_SUMMARY_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:
            return None
        return val
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    val = fnum(value)
    if val is None:
        return "NA"
    return f"{val:+.6f}"


def gate_row(gate: dict[str, Any] | None, stratum: str, metric: str) -> dict[str, Any] | None:
    if not gate:
        return None
    for row in gate.get("paired_deltas") or []:
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def summarize_run(row: dict[str, Any], run_root: Path) -> dict[str, Any]:
    run_name = str(row["run_name"])
    run_dir = run_root / run_name
    train_exit = read_text(run_dir / "EXIT_CODE")
    posthoc_exit = read_text(run_dir / "POSTHOC_EXIT_CODE")
    gate_path = Path(str(row.get("gate_json") or ""))
    decision_path = Path(str(row.get("decision_md") or ""))
    gate = load_json(gate_path) if gate_path.is_file() else None
    gate_decision = (gate or {}).get("gate") or {}
    status = "running_or_pending"
    reasons: list[str] = []
    if train_exit is not None and train_exit != "0":
        status = "training_failed"
        reasons.append(f"training_exit_{train_exit}")
    elif posthoc_exit is not None and posthoc_exit != "0":
        status = "posthoc_failed"
        reasons.append(f"posthoc_exit_{posthoc_exit}")
    elif gate is not None:
        status = str(gate_decision.get("status") or "gate_missing_status")
        reasons.extend(str(r) for r in gate_decision.get("reasons") or [])
    elif train_exit == "0" and posthoc_exit == "0":
        status = "posthoc_complete_missing_gate"
        reasons.append("posthoc_exit_0_but_gate_json_missing")
    elif train_exit == "0" and posthoc_exit is None:
        status = "posthoc_pending"
    elif train_exit is None:
        status = "training_running"

    key_rows = {
        "cross_background_seen_gene_pp": gate_row(gate, "cross_background_seen_gene", "pearson_pert"),
        "all_test_single_pp": gate_row(gate, "all_test_single", "pearson_pert"),
        "all_test_single_mmd": gate_row(gate, "all_test_single", "test_mmd_clamped"),
        "family_gene_pp": gate_row(gate, "family_gene", "pearson_pert"),
        "family_gene_mmd": gate_row(gate, "family_gene", "test_mmd_clamped"),
    }
    return {
        **row,
        "run_dir": str(run_dir),
        "train_exit": train_exit,
        "posthoc_exit": posthoc_exit,
        "gate_json_exists": gate_path.is_file(),
        "decision_md_exists": decision_path.is_file(),
        "status": status,
        "reasons": reasons,
        "key_metrics": {
            key: None if value is None else {
                "delta_mean": value.get("delta_mean"),
                "ci95": value.get("ci95"),
                "p_improve": value.get("p_improve"),
                "p_harm": value.get("p_harm"),
                "n_matched_conditions": value.get("n_matched_conditions"),
                "n_matched_datasets": value.get("n_matched_datasets"),
            }
            for key, value in key_rows.items()
        },
    }


def overall_status(runs: list[dict[str, Any]]) -> str:
    if any(r["status"] in {"training_failed", "posthoc_failed"} for r in runs):
        return "tracka_gene_reliability_adapter_block_has_failed_jobs"
    if any(r["status"] == "posthoc_complete_missing_gate" for r in runs):
        return "tracka_gene_reliability_adapter_block_missing_gate_artifacts"
    if any(r["status"] in {"training_running", "posthoc_pending", "running_or_pending"} for r in runs):
        return "tracka_gene_reliability_adapter_block_pending"
    if any(r["status"] == "candidate_gate_pass" for r in runs):
        return "tracka_gene_reliability_adapter_has_pass_candidate_needs_seed_robustness"
    if all(r["gate_json_exists"] for r in runs):
        return "tracka_gene_reliability_adapter_all_failed_close_or_nearmiss"
    return "tracka_gene_reliability_adapter_unknown"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Cross-Latent Gene-Reliability Adapter Summary",
        "",
        f"Status: `{payload['overall_status']}`",
        "",
        "## Boundary",
        "",
        "- Reads launch manifest, marker files, and per-run Track A candidate gates only.",
        "- Does not read canonical multi outputs or Track C held-out query.",
        "- Canonical multi selection weight remains 0.",
        "",
        "## Runs",
        "",
        "| run | latent | aggregation | train exit | posthoc exit | status | crossbg pp delta | all-single pp p_harm | family pp p_harm |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["runs"]:
        km = row["key_metrics"]
        cross = (km.get("cross_background_seen_gene_pp") or {})
        all_pp = (km.get("all_test_single_pp") or {})
        fam_pp = (km.get("family_gene_pp") or {})
        lines.append(
            "| `{run}` | `{latent}` | `{agg}` | `{te}` | `{pe}` | `{status}` | {cross} | {allph} | {famph} |".format(
                run=row["run_name"],
                latent=row.get("latent"),
                agg=row.get("aggregation"),
                te=row.get("train_exit") if row.get("train_exit") is not None else "NA",
                pe=row.get("posthoc_exit") if row.get("posthoc_exit") is not None else "NA",
                status=row["status"],
                cross=fmt(cross.get("delta_mean")),
                allph=fmt(all_pp.get("p_harm")),
                famph=fmt(fam_pp.get("p_harm")),
            )
        )
    lines += ["", "## Gate Reasons", ""]
    any_reason = False
    for row in payload["runs"]:
        if row["reasons"]:
            any_reason = True
            lines.append(f"### `{row['run_name']}`")
            for reason in row["reasons"]:
                lines.append(f"- `{reason}`")
    if not any_reason:
        lines.append("- none or pending")
    lines += [
        "",
        "## Next Action",
        "",
    ]
    if payload["overall_status"].endswith("pending"):
        lines.append("Wait for the next allowed long-job check window; do not poll repeatedly.")
    elif payload["overall_status"] == "tracka_gene_reliability_adapter_has_pass_candidate_needs_seed_robustness":
        lines.append("Freeze the passing candidate route/checkpoint and design a seed/anchor robustness check before any strong claim.")
    elif payload["overall_status"] == "tracka_gene_reliability_adapter_all_failed_close_or_nearmiss":
        lines.append("Close this adapter block unless a predeclared near-miss rule justifies one targeted follow-up.")
    else:
        lines.append("Inspect failed marker files and fix only if the cause is operational rather than a gate failure.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest)
    runs = [summarize_run(row, args.run_root) for row in manifest_rows]
    payload = {
        "manifest": str(args.manifest),
        "run_root": str(args.run_root),
        "query_read": False,
        "canonical_multi_selection_weight": 0,
        "runs": runs,
    }
    payload["overall_status"] = overall_status(runs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "status": payload["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
