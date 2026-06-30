#!/usr/bin/env python3
"""Materialize Replogle author bulk h5ad obs artifacts for LatentFM.

CPU/report-only. Reads author normalized bulk h5ad metadata, parses target genes
from obs index strings, and maps rows to local Replogle single-gene conditions.
No training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/replogle_figshare_bulk"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/replogle_bulk_artifacts_20260627"
OUT_CSV = OUT_DIR / "replogle_bulk_condition_artifacts.csv"
MANIFEST_JSON = ROOT / "configs/latentfm_replogle_bulk_artifact_manifest_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_replogle_bulk_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_ARTIFACTS_20260627.md"

SOURCES = {
    "K562_essential": {
        "path": SRC_DIR / "K562_essential_normalized_bulk_01.h5ad",
        "expected_size": 79766954,
        "local_dataset": "ReplogleWeissman2022_K562_gwps",
    },
    "K562_gwps": {
        "path": SRC_DIR / "K562_gwps_normalized_bulk_01.h5ad",
        "expected_size": 374587922,
        "local_dataset": "ReplogleWeissman2022_K562_gwps",
    },
    "RPE1": {
        "path": SRC_DIR / "rpe1_normalized_bulk_01.h5ad",
        "expected_size": 95350546,
        "local_dataset": "Replogle_RPE1essential",
    },
}

ARTIFACT_COLUMNS = {
    "control_expr": "response_candidate",
    "fold_expr": "response_candidate",
    "pct_expr": "response_candidate",
    "core_control": "response_candidate",
    "mean_leverage_score": "response_candidate",
    "std_leverage_score": "response_candidate",
    "energy_test_p_value": "response_candidate",
    "anderson_darling_counts": "response_candidate",
    "mann_whitney_counts": "response_candidate",
    "TE_ratio": "response_candidate",
    "cnv_score_z": "response_candidate",
    "UMI_count_unfiltered": "qc_control",
    "num_cells_unfiltered": "qc_control",
    "num_cells_filtered": "qc_control",
    "z_gemgroup_UMI": "qc_control",
    "mitopercent": "qc_control",
}


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "<na>"}:
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
    return out if math.isfinite(out) else None


def parse_replogle_obs_index_target(value: Any) -> str:
    text = norm(value)
    if not text:
        return ""
    parts = text.split("_")
    if len(parts) >= 4 and parts[0].isdigit() and parts[-1].startswith("ENSG"):
        return parts[1].strip()
    return ""


def load_local_conditions() -> dict[str, dict[str, str]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for source in SOURCES.values():
        dataset = source["local_dataset"]
        ds_parts = split.get(dataset, {})
        for split_name, conditions in ds_parts.items():
            if not isinstance(conditions, list):
                continue
            for condition in conditions:
                out.setdefault(dataset, {})[str(condition).upper()] = split_name
    return out


def validate_sources() -> list[str]:
    errors = []
    for label, spec in SOURCES.items():
        path = spec["path"]
        if not path.is_file():
            errors.append(f"{label}:missing")
        elif path.stat().st_size != spec["expected_size"]:
            errors.append(f"{label}:size_{path.stat().st_size}_expected_{spec['expected_size']}")
    return errors


def materialize_one(label: str, spec: dict[str, Any], local_conditions: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    dataset = spec["local_dataset"]
    local = local_conditions.get(dataset, {})
    rows: list[dict[str, Any]] = []
    a = ad.read_h5ad(spec["path"], backed="r")
    try:
        obs = a.obs
        available = [col for col in ARTIFACT_COLUMNS if col in obs.columns]
        for obs_name, obs_row in obs.iterrows():
            target = parse_replogle_obs_index_target(obs_name)
            if not target:
                continue
            split_name = local.get(target.upper())
            if split_name is None:
                continue
            for col in available:
                value = to_float(obs_row[col])
                if value is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": target,
                        "split": split_name,
                        "cell_background": "K562" if "K562" in label else "RPE1",
                        "source_label": label,
                        "obs_index": str(obs_name),
                        "artifact": f"replogle_bulk_{label}_{col}",
                        "artifact_value": value,
                        "artifact_role": ARTIFACT_COLUMNS[col],
                        "raw_column": col,
                    }
                )
    finally:
        a.file.close()
    return rows


def write_failure(status: str, errors: list[str]) -> int:
    payload = {
        "status": status,
        "gpu_authorized": False,
        "errors": errors,
        "outputs": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# LatentFM Replogle Bulk Artifacts 2026-06-27\n\n"
        f"Status: `{status}`\n\nGPU authorized: `False`\n\n"
        f"Errors: `{errors}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "errors": errors}, indent=2))
    return 2


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = validate_sources()
    if errors:
        return write_failure("replogle_bulk_artifacts_missing_or_incomplete_sources_no_gpu", errors)

    local_conditions = load_local_conditions()
    rows: list[dict[str, Any]] = []
    for label, spec in SOURCES.items():
        rows.extend(materialize_one(label, spec, local_conditions))

    fields = [
        "dataset",
        "condition",
        "split",
        "cell_background",
        "source_label",
        "obs_index",
        "artifact",
        "artifact_value",
        "artifact_role",
        "raw_column",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_dataset: dict[str, int] = defaultdict(int)
    by_artifact: dict[str, int] = defaultdict(int)
    by_role: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)
    unique_conditions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_dataset[row["dataset"]] += 1
        by_artifact[row["artifact"]] += 1
        by_role[row["artifact_role"]] += 1
        by_split[row["split"]] += 1
        unique_conditions[row["dataset"]].add(row["condition"])

    manifest = {
        "status": "replogle_bulk_artifacts_materialized_no_gpu",
        "artifacts": [
            {
                "artifact": artifact,
                "source_files": [str(OUT_CSV)],
                "required_columns": ["dataset", "condition", "artifact_value", "artifact_role"],
                "minimum_datasets": 2,
                "minimum_overlap_rows": 50,
                "minimum_varying_datasets": 2,
                "promotion_note": "CPU gate only. response_candidate artifacts must beat qc_control artifacts and pass no-harm gates before GPU.",
            }
            for artifact in sorted(by_artifact)
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "status": "replogle_bulk_artifacts_materialized_no_gpu",
        "gpu_authorized": False,
        "rows": len(rows),
        "dataset_row_counts": dict(sorted(by_dataset.items())),
        "dataset_unique_conditions": {k: len(v) for k, v in sorted(unique_conditions.items())},
        "artifact_counts": dict(sorted(by_artifact.items())),
        "role_counts": dict(sorted(by_role.items())),
        "split_counts": dict(sorted(by_split.items())),
        "outputs": {"csv": str(OUT_CSV), "manifest": str(MANIFEST_JSON), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Replogle Bulk Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only materialization of Replogle author normalized bulk h5ad obs artifacts.",
        "- Target genes are parsed from obs index strings such as `0_A1BG_P1_ENSG...`.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- rows: `{len(rows)}`",
        f"- unique conditions by dataset: `{payload['dataset_unique_conditions']}`",
        f"- role counts: `{payload['role_counts']}`",
        "",
        "| artifact | role | rows |",
        "|---|---|---:|",
    ]
    for artifact, count in sorted(by_artifact.items()):
        role = artifact.rsplit("_", 1)[-1]
        # Use the row table as ground truth for role because column names contain underscores.
        role_values = {row["artifact_role"] for row in rows if row["artifact"] == artifact}
        role_text = ",".join(sorted(role_values))
        lines.append(f"| `{artifact}` | `{role_text}` | {count} |")
    lines += [
        "",
        "## Decision",
        "",
        "This is a source artifact only. Run strict association/shuffle/LODO/residual/no-harm gates before any GPU smoke.",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST_JSON}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
