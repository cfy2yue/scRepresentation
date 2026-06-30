#!/usr/bin/env python3
"""Inspect downloaded DepMap 24Q4 source files for LatentFM materialization.

CPU/report-only. Does not train, infer, read checkpoints, read canonical multi
for selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/depmap_24q4_figshare"
MODEL = SRC_DIR / "Model.csv"
SCREEN = SRC_DIR / "CRISPRScreenMap.csv"
GENE_EFFECT = SRC_DIR / "CRISPRGeneEffect.csv"
SCAFFOLD = ROOT / "reports/depmap_24q4_artifact_scaffold_20260627/depmap_24q4_gene_condition_scaffold.csv"
OUT_JSON = ROOT / "reports/latentfm_depmap_24q4_source_inspection_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_DEPMAP_24Q4_SOURCE_INSPECTION_20260627.md"

TARGET_MODELS = {
    "ACH-000551": "K562",
    "ACH-002464": "RPE1",
    "ACH-000219": "A375",
    "ACH-000681": "A549",
    "ACH-000019": "MCF7",
    "ACH-000739": "HepG2",
    "ACH-000995": "Jurkat",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gene_symbol(col: str) -> str:
    return col.split(" (", 1)[0].strip()


def main() -> int:
    payload: dict[str, Any] = {
        "status": "depmap_24q4_sources_missing_or_incomplete",
        "gpu_authorized": False,
        "files": {p.name: {"path": str(p), "exists": p.is_file(), "size": p.stat().st_size if p.is_file() else None} for p in [MODEL, SCREEN, GENE_EFFECT]},
        "target_models": TARGET_MODELS,
        "model_hits": {},
        "screen_hits": {},
        "gene_effect_model_hits": {},
        "scaffold_rows": None,
        "gene_columns": None,
        "decision": "",
    }
    missing = [name for name, row in payload["files"].items() if not row["exists"]]
    if missing:
        payload["decision"] = f"Missing source files: {missing}. Wait for download or fix source acquisition."
        write_outputs(payload)
        return 2

    model_rows = read_csv_rows(MODEL)
    model_id_cols = [c for c in (model_rows[0].keys() if model_rows else []) if c.lower() in {"modelid", "model_id", "depmap_id", "depmapid"}]
    model_id_col = model_id_cols[0] if model_id_cols else "ModelID"
    for row in model_rows:
        mid = row.get(model_id_col) or row.get("ModelID") or row.get("DepMap_ID") or row.get("DepMapID")
        if mid in TARGET_MODELS:
            payload["model_hits"][mid] = {k: row.get(k, "") for k in list(row)[:12]}

    screen_rows = read_csv_rows(SCREEN)
    for row in screen_rows:
        mid = row.get("ModelID") or row.get("DepMap_ID") or row.get("DepMapID") or row.get("model_id")
        if mid in TARGET_MODELS:
            payload["screen_hits"].setdefault(mid, 0)
            payload["screen_hits"][mid] += 1

    with GENE_EFFECT.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        payload["gene_columns"] = {
            "total_columns": len(header),
            "id_column": header[0] if header else None,
            "example_gene_columns": header[1:11],
            "example_gene_symbols": [gene_symbol(c) for c in header[1:11]],
        }
        for row in reader:
            if not row:
                continue
            mid = row[0]
            if mid in TARGET_MODELS:
                payload["gene_effect_model_hits"][mid] = {
                    "cell_line": TARGET_MODELS[mid],
                    "nonempty_first_100": sum(1 for x in row[1:101] if x not in {"", "NA", "nan"}),
                }
            if len(payload["gene_effect_model_hits"]) == len(TARGET_MODELS):
                break

    if SCAFFOLD.is_file():
        with SCAFFOLD.open(newline="", encoding="utf-8") as handle:
            payload["scaffold_rows"] = sum(1 for _ in handle) - 1

    missing_models = sorted(set(TARGET_MODELS) - set(payload["gene_effect_model_hits"]))
    if missing_models:
        payload["status"] = "depmap_24q4_source_inspection_partial_model_coverage_no_gpu"
        payload["decision"] = f"DepMap files are readable, but gene-effect rows are missing target ModelIDs: {missing_models}. Materializer must exclude missing models."
    else:
        payload["status"] = "depmap_24q4_source_inspection_ready_for_cpu_materializer_no_gpu"
        payload["decision"] = (
            "DepMap source files are readable and target ModelIDs are present in CRISPRGeneEffect. "
            "Next step is a CPU materializer over the scaffold followed by strict artifact preflight controls."
        )

    write_outputs(payload)
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM DepMap 24Q4 Source Inspection 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only source inspection.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## File Status",
        "",
        "| file | exists | size |",
        "|---|---:|---:|",
    ]
    for name, row in payload["files"].items():
        lines.append(f"| `{name}` | `{row['exists']}` | `{row['size']}` |")
    lines += [
        "",
        "## Coverage",
        "",
        f"- scaffold rows: `{payload.get('scaffold_rows')}`",
        f"- Model.csv target hits: `{sorted(payload.get('model_hits', {}))}`",
        f"- CRISPRScreenMap target hits: `{payload.get('screen_hits', {})}`",
        f"- CRISPRGeneEffect target hits: `{sorted(payload.get('gene_effect_model_hits', {}))}`",
        f"- gene columns: `{payload.get('gene_columns')}`",
        "",
        "## Decision",
        "",
        payload.get("decision", ""),
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
