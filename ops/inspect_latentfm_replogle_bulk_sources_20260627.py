#!/usr/bin/env python3
"""Inspect Replogle et al. 2022 author bulk h5ad sources.

CPU/report-only. Opens h5ad files in backed mode and summarizes obs/var schema
and local split overlap. It does not train, infer, read checkpoints, read
canonical multi for selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/replogle_figshare_bulk"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_JSON = ROOT / "reports/latentfm_replogle_bulk_source_inspection_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_SOURCE_INSPECTION_20260627.md"

FILES = {
    "K562_essential": {
        "path": SRC_DIR / "K562_essential_normalized_bulk_01.h5ad",
        "expected_md5": "30496767641cd2e660ee6ecb5baee132",
        "local_datasets": ["ReplogleWeissman2022_K562_gwps"],
    },
    "K562_gwps": {
        "path": SRC_DIR / "K562_gwps_normalized_bulk_01.h5ad",
        "expected_md5": "a3dfaa94ea8724217f5ecb1e14a5f0c8",
        "local_datasets": ["ReplogleWeissman2022_K562_gwps"],
    },
    "RPE1": {
        "path": SRC_DIR / "rpe1_normalized_bulk_01.h5ad",
        "expected_md5": "6f1e7d6a09e2f869759e3c4526b7f171",
        "local_datasets": ["Replogle_RPE1essential"],
    },
}


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "<na>"}:
        return ""
    return text


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def sample_values(series: Any, n: int = 8) -> list[str]:
    try:
        vals = series.astype(str).dropna().unique().tolist()
    except Exception:
        vals = []
    return [str(v) for v in vals[:n]]


def parse_replogle_obs_index_target(value: Any) -> str:
    """Parse target gene from author bulk obs index like 0_A1BG_P1_ENSG..."""
    text = norm(value)
    if not text:
        return ""
    parts = text.split("_")
    if len(parts) >= 4 and parts[0].isdigit() and parts[-1].startswith("ENSG"):
        return parts[1].strip()
    return ""


def inspect_one(label: str, spec: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    path = spec["path"]
    row: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "expected_md5": spec["expected_md5"],
        "local_datasets": spec["local_datasets"],
        "status": "missing",
    }
    if not path.is_file():
        return row
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs
        var = a.var
        obs_columns = [str(c) for c in obs.columns]
        var_columns = [str(c) for c in var.columns]
        obs_index = [str(x) for x in obs.index[:8]]
        var_index = [str(x) for x in var.index[:8]]
        likely_condition_cols = [
            c
            for c in obs_columns
            if any(token in c.lower() for token in ("gene", "target", "perturb", "guide", "condition", "feature", "grna"))
        ]
        likely_quality_cols = [
            c
            for c in obs_columns
            if any(token in c.lower() for token in ("replicate", "batch", "umi", "read", "cell", "count", "score", "qc"))
        ]
        obs_samples = {c: sample_values(obs[c]) for c in likely_condition_cols[:12]}
        local_conditions = set()
        for ds in spec["local_datasets"]:
            parts = split.get(ds, {})
            for split_name in ("train", "test", "test_single"):
                local_conditions.update(str(x).upper() for x in parts.get(split_name, []))
        index_targets = [parse_replogle_obs_index_target(x) for x in obs.index]
        index_target_values = {x.upper() for x in index_targets if x}
        index_overlap = sorted(index_target_values & local_conditions)
        overlap_by_col: dict[str, Any] = {}
        for col in likely_condition_cols:
            vals = {str(v).strip().upper() for v in sample_values(obs[col], n=100000) if norm(v)}
            overlap = sorted(vals & local_conditions)
            overlap_by_col[col] = {
                "unique_sampled": len(vals),
                "local_overlap_count": len(overlap),
                "local_overlap_examples": overlap[:12],
            }
        if index_target_values:
            overlap_by_col["obs_index_parsed_target_gene"] = {
                "unique_sampled": len(index_target_values),
                "local_overlap_count": len(index_overlap),
                "local_overlap_examples": index_overlap[:12],
            }
        row.update(
            {
                "status": "readable",
                "shape": [int(a.n_obs), int(a.n_vars)],
                "obs_columns": obs_columns,
                "var_columns": var_columns,
                "obs_index_examples": obs_index,
                "var_index_examples": var_index,
                "likely_condition_cols": likely_condition_cols,
                "likely_quality_cols": likely_quality_cols,
                "condition_value_examples": obs_samples,
                "obs_index_parsed_target_gene_examples": [x for x in index_targets if x][:12],
                "local_condition_count": len(local_conditions),
                "overlap_by_condition_col": overlap_by_col,
                "layers": list(a.layers.keys()),
                "obsm": list(a.obsm.keys()),
                "varm": list(a.varm.keys()),
                "uns_keys": list(a.uns.keys()),
            }
        )
    finally:
        a.file.close()
    return row


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Replogle Bulk Source Inspection 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only backed-mode h5ad schema inspection.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Summary",
        "",
        "| source | status | shape | likely condition cols | best local overlap |",
        "|---|---|---:|---|---:|",
    ]
    for label, row in payload["sources"].items():
        overlaps = []
        for col, item in (row.get("overlap_by_condition_col") or {}).items():
            overlaps.append((int(item.get("local_overlap_count", 0)), col))
        best = max(overlaps)[0] if overlaps else 0
        lines.append(
            f"| `{label}` | `{row.get('status')}` | `{row.get('shape')}` | "
            f"`{', '.join(row.get('likely_condition_cols', [])[:6])}` | {best} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    split = load_split()
    sources = {label: inspect_one(label, spec, split) for label, spec in FILES.items()}
    missing = [label for label, row in sources.items() if row["status"] == "missing"]
    readable = [label for label, row in sources.items() if row["status"] == "readable"]
    best_overlap = 0
    for row in sources.values():
        for item in (row.get("overlap_by_condition_col") or {}).values():
            best_overlap = max(best_overlap, int(item.get("local_overlap_count", 0)))
    if missing:
        status = "replogle_bulk_sources_missing_or_incomplete_no_gpu"
        decision = f"Missing sources: {missing}. Wait for download completion before materialization."
        rc = 2
    elif not readable:
        status = "replogle_bulk_sources_unreadable_no_gpu"
        decision = "No readable h5ad sources; close or fix source acquisition."
        rc = 1
    elif best_overlap <= 0:
        status = "replogle_bulk_sources_readable_but_no_local_condition_overlap_no_gpu"
        decision = "Bulk files are readable but no local condition overlap was detected in likely condition columns; manual schema review required before materialization."
        rc = 0
    else:
        status = "replogle_bulk_sources_schema_ready_for_cpu_materializer_no_gpu"
        decision = "Bulk files are readable and likely condition columns overlap local Replogle targets. Next step is CPU materialization and strict gates; no GPU from source inspection alone."
        rc = 0
    payload = {
        "status": status,
        "gpu_authorized": False,
        "sources": sources,
        "decision": decision,
        "outputs": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    write_outputs(payload)
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
