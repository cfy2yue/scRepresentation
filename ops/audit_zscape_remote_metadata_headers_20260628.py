#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import io
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OUT_DIR = Path("/data/cyx/1030/scLatent/reports/zscape_remote_metadata_header_audit_20260628")
MD_PATH = Path("/data/cyx/1030/scLatent/reports/LATENTFM_ZSCAPE_REMOTE_METADATA_HEADER_AUDIT_20260628.md")
JSON_PATH = Path("/data/cyx/1030/scLatent/reports/latentfm_zscape_remote_metadata_header_audit_20260628.json")

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202639/suppl"
FILES = {
    "reference_cell_metadata": f"{BASE}/GSE202639_reference_cell_metadata.csv.gz",
    "reference_gene_metadata": f"{BASE}/GSE202639_reference_gene_metadata.csv.gz",
    "zperturb_full_cell_metadata": f"{BASE}/GSE202639_zperturb_full_cell_metadata.csv.gz",
    "zperturb_full_gene_metadata": f"{BASE}/GSE202639_zperturb_full_gene_metadata.csv.gz",
    "zperturb_full_run1_hashTable": f"{BASE}/GSE202639_zperturb_full_run1_hashTable.txt.gz",
    "zperturb_full_run2_hashTable": f"{BASE}/GSE202639_zperturb_full_run2_hashTable.txt.gz",
}

CATEGORY_PATTERNS = {
    "time_or_stage": [
        r"time",
        r"hpf",
        r"stage",
        r"development",
        r"age",
    ],
    "cell_type_or_lineage": [
        r"cell.?type",
        r"annotation",
        r"lineage",
        r"tissue",
        r"organ",
        r"cluster",
        r"subcluster",
    ],
    "sample_or_embryo": [
        r"embryo",
        r"sample",
        r"hash",
        r"well",
        r"batch",
        r"experiment",
        r"run",
        r"library",
        r"plate",
    ],
    "perturbation_or_condition": [
        r"perturb",
        r"target",
        r"guide",
        r"grna",
        r"gene",
        r"morpholino",
        r"genotype",
        r"condition",
        r"treatment",
        r"control",
        r"injected",
    ],
    "qc_or_abundance": [
        r"umi",
        r"count",
        r"n.?gene",
        r"mito",
        r"doublet",
        r"qc",
    ],
}


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def head_size(url: str) -> str:
    req = Request(url, method="HEAD", headers={"User-Agent": "LatentFM-ZSCAPE-metadata-audit"})
    try:
        with urlopen(req, timeout=45) as resp:
            return resp.headers.get("Content-Length", "")
    except Exception:
        return ""


def read_remote_rows(url: str, delimiter: str, max_rows: int = 4) -> list[list[str]]:
    req = Request(url, headers={"User-Agent": "LatentFM-ZSCAPE-metadata-audit"})
    with urlopen(req, timeout=90) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            text = io.TextIOWrapper(gz)
            reader = csv.reader(text, delimiter=delimiter)
            rows: list[list[str]] = []
            for row in reader:
                rows.append(row)
                if len(rows) >= max_rows:
                    break
            return rows


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for category, patterns in CATEGORY_PATTERNS.items():
        matched: list[str] = []
        for col in columns:
            lowered = col.lower()
            if any(re.search(pattern, lowered) for pattern in patterns):
                matched.append(col)
        hits[category] = matched
    return hits


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []

    for name, url in FILES.items():
        delimiter = "\t" if name.endswith("hashTable") else ","
        size = head_size(url)
        try:
            rows = read_remote_rows(url, delimiter=delimiter)
            columns = rows[0] if rows else []
            category_hits = classify_columns(columns)
            records.append(
                {
                    "name": name,
                    "url": url,
                    "content_length": size,
                    "delimiter": "\\t" if delimiter == "\t" else ",",
                    "n_columns": len(columns),
                    "columns": columns,
                    "category_hits": category_hits,
                    "sample_rows": rows[1:],
                    "status": "ok",
                }
            )
        except (HTTPError, URLError, OSError, EOFError, UnicodeDecodeError) as exc:
            errors.append({"name": name, "url": url, "error": repr(exc)})
            records.append(
                {
                    "name": name,
                    "url": url,
                    "content_length": size,
                    "delimiter": "\\t" if delimiter == "\t" else ",",
                    "n_columns": 0,
                    "columns": [],
                    "category_hits": {},
                    "sample_rows": [],
                    "status": "error",
                    "error": repr(exc),
                }
            )

    inventory_path = OUT_DIR / "remote_metadata_column_inventory.csv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "status", "n_columns", "content_length", "category", "matched_columns"])
        for record in records:
            if record["category_hits"]:
                for category, cols in record["category_hits"].items():
                    writer.writerow(
                        [
                            record["name"],
                            record["status"],
                            record["n_columns"],
                            record["content_length"],
                            category,
                            "|".join(cols),
                        ]
                    )
            else:
                writer.writerow(
                    [
                        record["name"],
                        record["status"],
                        record["n_columns"],
                        record["content_length"],
                        "",
                        "",
                    ]
                )

    full_cell = next((r for r in records if r["name"] == "zperturb_full_cell_metadata"), None)
    required = ["time_or_stage", "cell_type_or_lineage", "sample_or_embryo", "perturbation_or_condition"]
    missing_required = []
    if full_cell is None or full_cell["status"] != "ok":
        missing_required = required
    else:
        hits = full_cell["category_hits"]
        missing_required = [category for category in required if not hits.get(category)]

    status = "zscape_remote_metadata_header_gate_pass" if not missing_required and not errors else "zscape_remote_metadata_header_gate_partial"
    if missing_required:
        status = "zscape_remote_metadata_header_gate_fail"

    payload = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "inventory_csv": str(inventory_path),
        "missing_required_zperturb_full_cell_categories": missing_required,
        "records": records,
        "errors": errors,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Remote Metadata Header Audit",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Remote metadata header/sample-row audit only.",
        "- Streams compressed metadata headers and a few rows from GEO supplementary files.",
        "- Does not download large RDS/CDS/raw-count matrices.",
        "- No training, inference, embedding, canonical multi, or Track C query use.",
        "",
        "## Required ZPERTURB Full Cell Metadata Categories",
        "",
        f"- missing required categories: `{missing_required}`",
        "",
        "## File Inventory",
        "",
        "| File | Status | Columns | Size header | Key detected categories |",
        "|---|---:|---:|---:|---|",
    ]
    for record in records:
        category_summary = []
        for category, cols in record.get("category_hits", {}).items():
            if cols:
                category_summary.append(f"{category}: {len(cols)}")
        lines.append(
            "| "
            + " | ".join(
                [
                    record["name"],
                    record["status"],
                    str(record["n_columns"]),
                    str(record["content_length"] or "NA"),
                    ", ".join(category_summary) or "none",
                ]
            )
            + " |"
        )

    if full_cell and full_cell["status"] == "ok":
        lines.extend(
            [
                "",
                "## ZPERTURB Full Cell Metadata Candidate Columns",
                "",
            ]
        )
        for category in required:
            lines.append(f"- `{category}`: `{full_cell['category_hits'].get(category, [])}`")

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if status == "zscape_remote_metadata_header_gate_pass":
        lines.extend(
            [
                "Proceed to a metadata coverage audit that downloads/streams the cell metadata",
                "into a provenance directory and computes",
                "`perturbation x timepoint x embryo/sample x cell_type` coverage.",
                "This still does not authorize GPU training.",
            ]
        )
    elif status == "zscape_remote_metadata_header_gate_partial":
        lines.extend(
            [
                "Some remote files were readable, but not all files completed cleanly.",
                "Proceed only after confirming the missing/error files are not required for",
                "the coverage table or after retrying them in a detached download.",
            ]
        )
    else:
        lines.extend(
            [
                "Do not proceed to coverage or model work from these remote metadata headers.",
                "Find an alternate ZSCAPE metadata source or another annotated perturbation atlas.",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- inventory CSV: `{inventory_path}`",
            f"- JSON: `{JSON_PATH}`",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(MD_PATH)
    print(JSON_PATH)
    print(inventory_path)
    print(status)
    return 0 if status != "zscape_remote_metadata_header_gate_fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
