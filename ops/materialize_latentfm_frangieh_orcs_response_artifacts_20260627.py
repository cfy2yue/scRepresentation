#!/usr/bin/env python3
"""Materialize Frangieh ORCS/BioGRID response-burden artifacts.

CPU/source-only. Parses the author Combined Supplementary Tables xlsx with the
Python standard library, extracts gene-level MAGeCK sheets, and maps local
Frangieh target genes. No training, inference, checkpoint selection, canonical
multi selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_XLSX = ROOT / "reports/external_artifact_sources_20260627/frangieh_orcs/Combined_Supplementary_Tables.xlsx"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/frangieh_orcs_response_artifacts_20260627"
OUT_CSV = OUT_DIR / "frangieh_orcs_response_artifacts.csv"
MANIFEST_JSON = ROOT / "configs/latentfm_frangieh_orcs_response_artifact_manifest_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_frangieh_orcs_response_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_FRANGIEH_ORCS_RESPONSE_ARTIFACTS_20260627.md"
SOURCE_URL = "https://orcs.thebiogrid.org/uploads/processed/636d4a0d95f53/NIHMS1699873-supplement-Combined_Supplementary_Tables.xlsx"

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

GENE_LEVEL_CONTEXTS = {
    "Supplementary Table 2a": "day14_vs_day7_fitness",
    "Supplementary Table 2c": "ifng_vs_control",
    "Supplementary Table 2e": "til_1to1_vs_control",
    "Supplementary Table 2g": "til_1to1_vs_ifng",
    "Supplementary Table 2i": "til_2to1_vs_control",
    "Supplementary Table 2k": "til_2to1_vs_ifng",
    "Supplementary Table 2m": "til_4to1_vs_control",
    "Supplementary Table 2o": "til_4to1_vs_ifng",
}

VALUE_COLUMNS = {
    "neg|score": "response_candidate",
    "neg|p-value": "significance_control",
    "neg|fdr": "significance_control",
    "neg|goodsgrna": "guide_support_control",
    "neg|lfc": "response_candidate",
    "pos|score": "response_candidate",
    "pos|p-value": "significance_control",
    "pos|fdr": "significance_control",
    "pos|goodsgrna": "guide_support_control",
    "pos|lfc": "response_candidate",
}


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "<na>"}:
        return ""
    return text


def fnum(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def colnum(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        return 0
    out = 0
    for ch in match.group(1):
        out = out * 26 + ord(ch) - 64
    return out - 1


class XlsxReader:
    def __init__(self, path: Path):
        self.path = path
        self.z = zipfile.ZipFile(path)
        self.shared_strings = self._read_shared_strings()
        self.sheets = self._read_sheets()

    def close(self) -> None:
        self.z.close()

    def _read_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.z.namelist():
            return []
        root = ET.fromstring(self.z.read("xl/sharedStrings.xml"))
        out = []
        for si in root.findall("a:si", NS):
            texts = [t.text or "" for t in si.findall(".//a:t", NS)]
            out.append("".join(texts))
        return out

    def _read_sheets(self) -> dict[str, str]:
        workbook = ET.fromstring(self.z.read("xl/workbook.xml"))
        rels = ET.fromstring(self.z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        out = {}
        for sheet in workbook.find("a:sheets", NS):
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            out[sheet.attrib["name"]] = "xl/" + rel_map[rid]
        return out

    def rows(self, sheet_name: str) -> list[list[str]]:
        root = ET.fromstring(self.z.read(self.sheets[sheet_name]))
        out: list[list[str]] = []
        for row in root.findall(".//a:sheetData/a:row", NS):
            vals: list[str] = []
            for cell in row.findall("a:c", NS):
                idx = colnum(cell.attrib.get("r", "A1"))
                while len(vals) <= idx:
                    vals.append("")
                value = cell.find("a:v", NS)
                text = ""
                if value is not None and value.text is not None:
                    text = self.shared_strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
                vals[idx] = text
            if any(vals):
                out.append(vals)
        return out


def load_local_split() -> dict[str, str]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = {}
    for split_name, conditions in split.get("Frangieh", {}).items():
        if isinstance(conditions, list):
            for condition in conditions:
                out[str(condition).upper()] = split_name
    return out


def find_header(rows: list[list[str]]) -> tuple[int, list[str]]:
    for idx, row in enumerate(rows):
        low = [str(x).strip().lower() for x in row]
        if "id" in low and "neg|lfc" in low and "pos|lfc" in low:
            return idx, [str(x).strip() for x in row]
    raise ValueError("gene-level header not found")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_XLSX.is_file():
        payload = {"status": "frangieh_orcs_source_missing_no_gpu", "gpu_authorized": False, "source": str(SRC_XLSX)}
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2

    local = load_local_split()
    reader = XlsxReader(SRC_XLSX)
    rows_out: list[dict[str, Any]] = []
    sheet_counts: dict[str, Any] = {}
    try:
        for sheet, context in GENE_LEVEL_CONTEXTS.items():
            rows = reader.rows(sheet)
            header_idx, header = find_header(rows)
            hmap = {name: i for i, name in enumerate(header)}
            parsed = 0
            matched_genes = set()
            for row in rows[header_idx + 1 :]:
                gene = norm(row[hmap["id"]] if hmap["id"] < len(row) else "")
                if not gene:
                    continue
                parsed += 1
                split_name = local.get(gene.upper())
                if split_name is None:
                    continue
                matched_genes.add(gene)
                for col, role in VALUE_COLUMNS.items():
                    idx = hmap.get(col)
                    if idx is None or idx >= len(row):
                        continue
                    value = fnum(row[idx])
                    if value is None:
                        continue
                    rows_out.append(
                        {
                            "dataset": "Frangieh",
                            "condition": gene,
                            "target_gene": gene,
                            "split": split_name,
                            "cell_background": "A375_unverified",
                            "response_context": context,
                            "artifact": f"frangieh_orcs_{context}_{col.replace('|', '_').replace('-', 'm')}",
                            "artifact_value": value,
                            "artifact_role": role,
                            "raw_column": col,
                            "source_sheet": sheet,
                            "source_url": SOURCE_URL,
                        }
                    )
            sheet_counts[sheet] = {"context": context, "parsed_genes": parsed, "matched_local_genes": len(matched_genes)}
    finally:
        reader.close()

    fields = [
        "dataset",
        "condition",
        "target_gene",
        "split",
        "cell_background",
        "response_context",
        "artifact",
        "artifact_value",
        "artifact_role",
        "raw_column",
        "source_sheet",
        "source_url",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    artifacts = sorted({r["artifact"] for r in rows_out})
    unique_genes = sorted({r["condition"] for r in rows_out})
    manifest = {
        "status": "frangieh_orcs_response_artifacts_materialized_no_gpu",
        "artifacts": [
            {
                "artifact": artifact,
                "source_files": [str(OUT_CSV)],
                "required_columns": ["dataset", "condition", "response_context", "artifact_value"],
                "single_source_note": "Frangieh/A375 only; preview diagnostic until second source or strict source-control exists.",
            }
            for artifact in artifacts
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "frangieh_orcs_response_artifacts_materialized_no_gpu",
        "gpu_authorized": False,
        "source_xlsx": str(SRC_XLSX),
        "source_url": SOURCE_URL,
        "rows": len(rows_out),
        "unique_local_genes": len(unique_genes),
        "sheet_counts": sheet_counts,
        "artifacts": len(artifacts),
        "outputs": {"csv": str(OUT_CSV), "manifest": str(MANIFEST_JSON), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Frangieh ORCS Response Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only parser for author ORCS/BioGRID Combined Supplementary Tables.",
        "- Extracts gene-level Supplementary Table 2 sheets only; no h5ad expression, training, inference, canonical multi selection, or Track C query.",
        "- Single-source Frangieh/A375 branch: diagnostic preview only until source-control or second source exists.",
        "",
        "## Summary",
        "",
        f"- rows: `{len(rows_out)}`",
        f"- unique local genes: `{len(unique_genes)}`",
        f"- artifacts: `{len(artifacts)}`",
        "",
        "| sheet | context | parsed genes | matched local genes |",
        "|---|---|---:|---:|",
    ]
    for sheet, item in sheet_counts.items():
        lines.append(f"| `{sheet}` | `{item['context']}` | {item['parsed_genes']} | {item['matched_local_genes']} |")
    lines += [
        "",
        "## Decision",
        "",
        "Run a single-source preview gate. No GPU is authorized from materialization.",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST_JSON}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows_out), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
