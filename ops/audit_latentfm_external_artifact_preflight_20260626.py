#!/usr/bin/env python3
"""Preflight external train-only artifact candidates for LatentFM.

Short CPU task. Reads a manifest and completed train-only/internal row metrics.
It does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CONFIG = Path(
    os.environ.get(
        "LATENTFM_EXTERNAL_ARTIFACT_CONFIG",
        str(ROOT / "configs/latentfm_external_artifact_manifest_20260626.json"),
    )
)
REPORTS = ROOT / "reports"
OUT_PREFIX = os.environ.get("LATENTFM_EXTERNAL_ARTIFACT_OUT_PREFIX", "latentfm_external_artifact_preflight_20260626")
OUT_TITLE = os.environ.get("LATENTFM_EXTERNAL_ARTIFACT_OUT_TITLE", "LATENTFM_EXTERNAL_ARTIFACT_PREFLIGHT_20260626")
OUT_JSON = REPORTS / f"{OUT_PREFIX}.json"
OUT_MD = REPORTS / f"{OUT_TITLE}.md"
OUT_CSV = REPORTS / f"{OUT_PREFIX}_rows.csv"

ROW_METRIC_FILES = [
    REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    REPORTS / "latentfm_qc_support_reliability_rows_20260625.csv",
    REPORTS / "latentfm_response_program_projection_rows_20260625.csv",
    REPORTS / "latentfm_lodo_domain_conflict_rows_20260625.csv",
    REPORTS / "latentfm_background_target_actionability_rows_20260625.csv",
    REPORTS / "latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_outcome_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in ROW_METRIC_FILES:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "dataset" not in fields or "condition" not in fields:
                continue
            for row in reader:
                ds = norm(row.get("dataset"))
                cond = norm(row.get("condition"))
                if not ds or not cond:
                    continue
                key = (ds, cond)
                rec = rows.setdefault(key, {"dataset": ds, "condition": cond})
                for pp_key in ("cross_pp_diff", "pp_delta", "pp_mean", "truecell_pp_delta_mean"):
                    val = to_float(row.get(pp_key))
                    if val is not None:
                        rec.setdefault("pp_values", []).append(val)
                for mmd_key in ("cross_mmd_diff", "mmd_delta", "mmd_mean", "truecell_mmd_delta_mean"):
                    val = to_float(row.get(mmd_key))
                    if val is not None:
                        rec.setdefault("mmd_values", []).append(val)
    for rec in rows.values():
        pp_values = rec.get("pp_values") or []
        mmd_values = rec.get("mmd_values") or []
        rec["pp_proxy_mean"] = sum(pp_values) / len(pp_values) if pp_values else None
        rec["mmd_proxy_max"] = max(mmd_values) if mmd_values else None
    return rows


def read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def summarize_artifact(spec: dict[str, Any], outcome_rows: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_files = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in spec.get("source_files", [])]
    reasons: list[str] = []
    row_out: list[dict[str, Any]] = []
    if not source_files:
        reasons.append("no_source_files_declared")

    all_rows: list[dict[str, Any]] = []
    for path in source_files:
        if not path.is_file():
            reasons.append(f"missing_source_file:{path}")
            continue
        table = read_table(path)
        for row in table:
            row["_source_file"] = str(path)
        all_rows.extend(table)

    required = list(spec.get("required_columns", []))
    missing_schema = []
    for path in source_files:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = set(reader.fieldnames or [])
        missing = [col for col in required if col not in fields]
        if missing:
            missing_schema.append(f"{path}:{','.join(missing)}")
    if missing_schema:
        reasons.extend(f"missing_required_columns:{item}" for item in missing_schema)

    dataset_values: dict[str, list[float]] = defaultdict(list)
    overlap_rows = 0
    pp_values: list[float] = []
    mmd_values: list[float] = []
    for row in all_rows:
        ds = norm(row.get("dataset"))
        cond = norm(row.get("condition"))
        value = to_float(row.get("artifact_value"))
        if not ds or not cond or value is None:
            continue
        dataset_values[ds].append(value)
        outcome = outcome_rows.get((ds, cond))
        overlap = outcome is not None
        if overlap:
            overlap_rows += 1
            if outcome.get("pp_proxy_mean") is not None:
                pp_values.append(float(outcome["pp_proxy_mean"]))
            if outcome.get("mmd_proxy_max") is not None:
                mmd_values.append(float(outcome["mmd_proxy_max"]))
        row_out.append(
            {
                "artifact": spec["artifact"],
                "dataset": ds,
                "condition": cond,
                "artifact_value": value,
                "outcome_overlap": overlap,
                "pp_proxy_mean": None if outcome is None else outcome.get("pp_proxy_mean"),
                "mmd_proxy_max": None if outcome is None else outcome.get("mmd_proxy_max"),
                "source_file": row.get("_source_file", ""),
            }
        )

    datasets = sorted(dataset_values)
    varying_datasets = sorted(ds for ds, values in dataset_values.items() if len({round(v, 8) for v in values}) >= 2)
    min_datasets = int(spec.get("minimum_datasets", 3))
    min_overlap = int(spec.get("minimum_overlap_rows", 50))
    min_varying = int(spec.get("minimum_varying_datasets", 3))
    if len(datasets) < min_datasets:
        reasons.append(f"dataset_count_below_{min_datasets}")
    if overlap_rows < min_overlap:
        reasons.append(f"overlap_rows_below_{min_overlap}")
    if len(varying_datasets) < min_varying:
        reasons.append(f"varying_dataset_count_below_{min_varying}")

    pp_mean = sum(pp_values) / len(pp_values) if pp_values else None
    dataset_pp_means = {}
    for ds in datasets:
        vals = [
            float(row["pp_proxy_mean"])
            for row in row_out
            if row["dataset"] == ds and row["pp_proxy_mean"] is not None
        ]
        if vals:
            dataset_pp_means[ds] = sum(vals) / len(vals)
    dataset_min_pp = min(dataset_pp_means.values()) if dataset_pp_means else None
    mmd_max = max(mmd_values) if mmd_values else None
    if pp_mean is None:
        reasons.append("no_overlapped_pp_proxy_values")
    if dataset_min_pp is None:
        reasons.append("no_dataset_tail_pp_proxy")
    elif dataset_min_pp < -0.020:
        reasons.append("dataset_min_pp_below_minus_0p020")
    if mmd_max is not None and mmd_max > 0.001:
        reasons.append("mmd_proxy_max_above_0p001")

    status = "pass_needs_control_gate_no_gpu" if not reasons else "fail_no_gpu"
    summary = {
        "artifact": spec["artifact"],
        "priority": spec.get("priority"),
        "status": status,
        "gpu_authorized": False,
        "source_files": [str(p) for p in source_files],
        "datasets": len(datasets),
        "dataset_names": datasets,
        "varying_datasets": len(varying_datasets),
        "overlap_rows": overlap_rows,
        "pp_proxy_mean": pp_mean,
        "dataset_min_pp_proxy": dataset_min_pp,
        "mmd_proxy_max": mmd_max,
        "reasons": reasons,
        "promotion_note": spec.get("promotion_note", ""),
    }
    return summary, row_out


def write_rows(rows: list[dict[str, Any]]) -> None:
    fields = [
        "artifact",
        "dataset",
        "condition",
        "artifact_value",
        "outcome_overlap",
        "pp_proxy_mean",
        "mmd_proxy_max",
        "source_file",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM External Artifact Preflight",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only preflight for externally acquired train-only artifacts.",
        "- Reads completed train-only/internal row metrics as outcome proxies.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Artifact Summary",
        "",
        "| artifact | status | datasets | varying datasets | overlap rows | pp proxy mean | dataset min pp | MMD proxy max | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["artifacts"]:
        lines.append(
            f"| `{row['artifact']}` | `{row['status']}` | {row['datasets']} | "
            f"{row['varying_datasets']} | {row['overlap_rows']} | {fmt(row['pp_proxy_mean'])} | "
            f"{fmt(row['dataset_min_pp_proxy'])} | {fmt(row['mmd_proxy_max'])} | "
            f"{', '.join(row['reasons']) or 'none'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- pass candidates: `{payload['pass_candidates']}`",
        f"- action: `{payload['decision']['action']}`",
        "",
        "## Required Source Schema",
        "",
        "Each source file should be CSV/TSV with at least:",
        "",
        "```text",
        "dataset,condition,artifact_value",
        "```",
        "",
        "Optional columns include target/guide/time/unit/source/evidence URL, depending on the artifact.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    manifest = load_json(CONFIG)
    outcome_rows = load_outcome_rows()
    summaries = []
    all_rows: list[dict[str, Any]] = []
    for spec in manifest.get("artifacts", []):
        summary, rows = summarize_artifact(spec, outcome_rows)
        summaries.append(summary)
        all_rows.extend(rows)
    pass_candidates = [row["artifact"] for row in summaries if row["status"].startswith("pass")]
    payload = {
        "status": "external_artifact_preflight_pass_candidates_no_gpu" if pass_candidates else "external_artifact_preflight_fail_no_gpu",
        "gpu_authorized": False,
        "manifest": str(CONFIG),
        "outcome_row_count": len(outcome_rows),
        "artifacts": summaries,
        "pass_candidates": pass_candidates,
        "decision": {
            "action": (
                "run artifact-specific shuffle/LODO/source/count/tail controls before GPU"
                if pass_candidates
                else "no external artifact source files pass preflight; acquire real source files or use chemical V2 ACK/reporting route"
            )
        },
    }
    write_rows(all_rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
