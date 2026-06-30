#!/usr/bin/env python3
"""Inspect the Jiang Zenodo Mixscale DE archive for LatentFM artifact use.

CPU/report-only. This script does not train, infer, read canonical multi for
selection, read Track C query, or use GPU. It only checks whether the downloaded
author archive has condition/background-specific tabular DE outputs that can be
materialized into a later strict CPU gate.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/jiang_zenodo_v2_1"
ZIP_PATH = SRC_DIR / "DE_results_all_pathway.zip"
OUT_JSON = ROOT / "reports/latentfm_jiang_zenodo_de_archive_inspection_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_ZENODO_DE_ARCHIVE_INSPECTION_20260627.md"


def decode_sample(data: bytes) -> str:
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


def sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:10])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except Exception:
        if "\t" in sample:
            return "\t"
        return ","


def inspect_member(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    with zf.open(name) as handle:
        raw = handle.read(65536)
    text = decode_sample(raw)
    lines = [x for x in text.splitlines() if x.strip()]
    delimiter = sniff_delimiter(text)
    header: list[str] = []
    first_rows: list[dict[str, str]] = []
    if lines:
        reader = csv.DictReader(io.StringIO("\n".join(lines[:8])), delimiter=delimiter)
        header = list(reader.fieldnames or [])
        for row in reader:
            first_rows.append({k: row.get(k, "") for k in header[:12]})
            if len(first_rows) >= 3:
                break
    lower_cols = {c.lower(): c for c in header}
    likely_keys = {
        "regulator": any(x in lower_cols for x in ["regulator", "perturbation", "gene", "target"]),
        "cell_background": any(x in lower_cols for x in ["cell_line", "celltype", "cell_type", "cell", "background"]),
        "pathway": any(x in lower_cols for x in ["pathway", "cytokine", "stimulation"]),
        "effect": any(
            x in lower_cols
            for x in ["logfc", "avg_log2fc", "p_val_adj", "fdr", "statistic", "score", "mixscale_score"]
        ),
    }
    return {
        "name": name,
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "header": header,
        "first_rows": first_rows,
        "likely_keys": likely_keys,
        "artifact_candidate": all(likely_keys.values()),
    }


def main() -> int:
    payload: dict[str, Any] = {
        "status": "jiang_zenodo_de_archive_missing_or_invalid",
        "zip_path": str(ZIP_PATH),
        "gpu_authorized": False,
        "members": [],
        "sampled_tables": [],
        "artifact_candidate_tables": [],
        "decision": "",
    }
    if not ZIP_PATH.is_file():
        payload["decision"] = "Archive is not downloaded yet; wait for source acquisition to finish."
        write_outputs(payload)
        return 2
    if not zipfile.is_zipfile(ZIP_PATH):
        payload["decision"] = "Archive exists but is not a valid zip yet; likely partial download."
        write_outputs(payload)
        return 3

    with zipfile.ZipFile(ZIP_PATH) as zf:
        infos = [x for x in zf.infolist() if not x.is_dir()]
        payload["members"] = [
            {
                "name": x.filename,
                "file_size": x.file_size,
                "compress_size": x.compress_size,
            }
            for x in infos
        ]
        table_like = [
            x.filename
            for x in infos
            if x.filename.lower().endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"))
        ]
        for name in table_like[:40]:
            try:
                payload["sampled_tables"].append(inspect_member(zf, name))
            except Exception as exc:
                payload["sampled_tables"].append({"name": name, "error": repr(exc)})

    payload["artifact_candidate_tables"] = [
        x["name"] for x in payload["sampled_tables"] if x.get("artifact_candidate")
    ]
    if payload["artifact_candidate_tables"]:
        payload["status"] = "jiang_zenodo_de_archive_has_candidate_tables_cpu_materialization_next"
        payload["decision"] = (
            "Archive has table-like files with regulator/background/pathway/effect columns. "
            "Next step is a CPU materializer to aggregate author DE response strength into "
            "dataset,condition,cell_background,artifact_value rows, then run strict shuffle/LODO/tail/MMD gates."
        )
    elif payload["sampled_tables"]:
        payload["status"] = "jiang_zenodo_de_archive_tables_need_manual_schema_review_no_gpu"
        payload["decision"] = (
            "Archive is valid and table-like files exist, but automatic header sniffing did not find "
            "the full regulator/background/pathway/effect schema. Do a schema-specific CPU review before any gate."
        )
    else:
        payload["status"] = "jiang_zenodo_de_archive_no_table_like_files_no_gpu"
        payload["decision"] = "Archive is valid but no CSV/TSV/TXT tables were found; inspect member formats manually."

    write_outputs(payload)
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Jiang Zenodo DE Archive Inspection 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only archive schema inspection.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Source",
        "",
        f"- zip: `{payload['zip_path']}`",
        f"- member count: `{len(payload.get('members', []))}`",
        f"- sampled table count: `{len(payload.get('sampled_tables', []))}`",
        f"- automatic candidate tables: `{len(payload.get('artifact_candidate_tables', []))}`",
        "",
        "## Candidate Tables",
        "",
    ]
    if payload.get("artifact_candidate_tables"):
        for name in payload["artifact_candidate_tables"][:20]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none detected automatically")
    lines += [
        "",
        "## Sampled Headers",
        "",
    ]
    for table in payload.get("sampled_tables", [])[:20]:
        lines.append(f"### `{table.get('name')}`")
        if "error" in table:
            lines.append(f"- error: `{table['error']}`")
        else:
            lines.append(f"- delimiter: `{table.get('delimiter')}`")
            lines.append(f"- likely keys: `{table.get('likely_keys')}`")
            lines.append(f"- header: `{table.get('header')}`")
        lines.append("")
    lines += [
        "## Decision",
        "",
        payload.get("decision", ""),
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
