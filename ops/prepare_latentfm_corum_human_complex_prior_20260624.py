#!/usr/bin/env python3
"""Normalize CORUM human protein-complex prior for LatentFM gates."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
PRIOR_DIR = ROOT / "dataset/external_priors/corum_complexes_20260624"
RAW_TXT = PRIOR_DIR / "humanComplexes.txt"
OPENAPI = PRIOR_DIR / "openapi.json"
FILE_INFO = PRIOR_DIR / "file_info.json"
CURRENT_RELEASE = PRIOR_DIR / "current_release.json"
OUT_COMPLEXES = PRIOR_DIR / "corum_human_complexes_normalized.tsv"
OUT_GENE_COMPLEXES = PRIOR_DIR / "corum_human_gene_complexes.tsv"
OUT_SUMMARY = PRIOR_DIR / "corum_human_complex_prior_summary.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_semicolon(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(";") if x.strip()]


def load_release() -> dict[str, Any]:
    if CURRENT_RELEASE.exists():
        return json.loads(CURRENT_RELEASE.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    complexes = []
    gene_to_complexes: dict[str, list[dict[str, str]]] = defaultdict(list)
    with RAW_TXT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            genes = sorted({g.upper() for g in split_semicolon(row.get("subunits_gene_name", "")) if g})
            if len(genes) < 2:
                continue
            complex_id = str(row.get("complex_id") or "").strip()
            complex_name = str(row.get("complex_name") or "").strip()
            item = {
                "complex_id": complex_id,
                "complex_name": complex_name,
                "organism": str(row.get("organism") or "").strip(),
                "cell_line": str(row.get("cell_line") or "").strip(),
                "pmid": str(row.get("pmid") or "").strip(),
                "n_genes": len(genes),
                "genes": genes,
                "functions_go_id": split_semicolon(row.get("functions_go_id", "")),
                "functions_go_name": split_semicolon(row.get("functions_go_name", "")),
                "fcgs_id": split_semicolon(row.get("fcgs_id", "")),
                "fcgs_name": split_semicolon(row.get("fcgs_name", "")),
                "fcgs_category_name": split_semicolon(row.get("fcgs_category_name", "")),
            }
            complexes.append(item)
            for gene in genes:
                gene_to_complexes[gene].append({"complex_id": complex_id, "complex_name": complex_name})

    OUT_COMPLEXES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_COMPLEXES.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "complex_id",
            "complex_name",
            "organism",
            "cell_line",
            "pmid",
            "n_genes",
            "genes",
            "functions_go_id",
            "functions_go_name",
            "fcgs_id",
            "fcgs_name",
            "fcgs_category_name",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for item in sorted(complexes, key=lambda x: int(x["complex_id"]) if str(x["complex_id"]).isdigit() else 10**12):
            writer.writerow({k: ";".join(v) if isinstance(v, list) else v for k, v in item.items()})

    with OUT_GENE_COMPLEXES.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["gene", "n_complexes", "complex_ids", "complex_names"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for gene in sorted(gene_to_complexes):
            rows = gene_to_complexes[gene]
            writer.writerow(
                {
                    "gene": gene,
                    "n_complexes": len(rows),
                    "complex_ids": ";".join(row["complex_id"] for row in rows),
                    "complex_names": ";".join(row["complex_name"] for row in rows),
                }
            )

    gene_counts = [len(v) for v in gene_to_complexes.values()]
    summary = {
        "status": "ok",
        "timestamp": "2026-06-24 00:25 CST",
        "source": {
            "resource": "CORUM human protein complexes",
            "base_url": "https://mips.helmholtz-muenchen.de/fastapi-corum",
            "download_endpoint": "/public/file/download_current_file?file_id=human&file_format=txt",
            "release": load_release(),
            "tls_note": "curl used -k because local certificate validation failed for the official CORUM host",
        },
        "raw_files": {
            str(RAW_TXT): sha256(RAW_TXT),
            str(OPENAPI): sha256(OPENAPI) if OPENAPI.exists() else None,
            str(FILE_INFO): sha256(FILE_INFO) if FILE_INFO.exists() else None,
            str(CURRENT_RELEASE): sha256(CURRENT_RELEASE) if CURRENT_RELEASE.exists() else None,
        },
        "outputs": {
            str(OUT_COMPLEXES): sha256(OUT_COMPLEXES),
            str(OUT_GENE_COMPLEXES): sha256(OUT_GENE_COMPLEXES),
        },
        "n_complexes_min2": len(complexes),
        "n_unique_genes": len(gene_to_complexes),
        "gene_complex_count": {
            "min": min(gene_counts) if gene_counts else 0,
            "median": float(sorted(gene_counts)[len(gene_counts) // 2]) if gene_counts else 0.0,
            "max": max(gene_counts) if gene_counts else 0,
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "n_complexes": len(complexes), "n_genes": len(gene_to_complexes), "summary": str(OUT_SUMMARY)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
