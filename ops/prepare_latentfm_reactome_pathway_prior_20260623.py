#!/usr/bin/env python3
"""Acquire and normalize Reactome pathway gene sets as an external prior."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "dataset" / "external_priors" / "reactome_pathways_current_20260623"
REPORTS = ROOT / "reports"

SOURCE_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
SOURCE_PAGE = "https://reactome.org/download/current/"
RELEASE_LABEL = "Reactome current download ReactomePathways.gmt.zip"

RAW_ZIP = OUT_DIR / "ReactomePathways.gmt.zip"
RAW_GMT = OUT_DIR / "ReactomePathways.gmt"
GENE_PATHWAYS_TSV = OUT_DIR / "reactome_gene_pathways.tsv"
SUMMARY_JSON = OUT_DIR / "reactome_pathway_prior_summary.json"

OUT_JSON = REPORTS / "latentfm_reactome_pathway_prior_acquisition_20260623.json"
OUT_MD = REPORTS / "LATENTFM_REACTOME_PATHWAY_PRIOR_ACQUISITION_20260623.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_needed() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_ZIP.is_file() and RAW_ZIP.stat().st_size > 0:
        return {"downloaded": False, "reason": "existing_file_reused"}
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "LatentFM-reactome-prior-acquisition/20260623"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        headers = dict(response.headers.items())
        tmp = RAW_ZIP.with_suffix(".gmt.zip.tmp")
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(RAW_ZIP)
    return {"downloaded": True, "reason": "downloaded_from_source", "source_url": SOURCE_URL, "headers": headers}


def extract_and_parse() -> dict[str, Any]:
    with zipfile.ZipFile(RAW_ZIP) as zf:
        names = zf.namelist()
        gmt_names = [name for name in names if name.endswith(".gmt")]
        if len(gmt_names) != 1:
            raise RuntimeError(f"expected one GMT in zip, found {gmt_names}")
        with zf.open(gmt_names[0]) as src, RAW_GMT.open("wb") as dst:
            dst.write(src.read())

    pathway_genes: dict[str, list[str]] = {}
    gene_pathways: dict[str, set[str]] = defaultdict(set)
    with RAW_GMT.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathway = parts[0]
            desc = parts[1]
            genes = sorted({gene.upper() for gene in parts[2:] if gene})
            pathway_genes[pathway] = genes
            for gene in genes:
                gene_pathways[gene].add(pathway)

    with GENE_PATHWAYS_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene", "n_reactome_pathways", "reactome_pathways"])
        for gene in sorted(gene_pathways):
            pathways = sorted(gene_pathways[gene])
            writer.writerow([gene, len(pathways), ";".join(pathways)])

    sizes = [len(genes) for genes in pathway_genes.values()]
    summary = {
        "status": "reactome_pathway_prior_acquired_normalized_no_gpu",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "release_label": RELEASE_LABEL,
        "raw_zip": str(RAW_ZIP),
        "raw_gmt": str(RAW_GMT),
        "gene_pathways_tsv": str(GENE_PATHWAYS_TSV),
        "n_pathways": len(pathway_genes),
        "n_genes": len(gene_pathways),
        "pathway_size_min": min(sizes) if sizes else None,
        "pathway_size_median": float(sorted(sizes)[len(sizes) // 2]) if sizes else None,
        "pathway_size_max": max(sizes) if sizes else None,
        "hashes": {
            "raw_zip": sha256_file(RAW_ZIP),
            "raw_gmt": sha256_file(RAW_GMT),
            "gene_pathways_tsv": sha256_file(GENE_PATHWAYS_TSV),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Reactome Pathway Prior Acquisition",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- External prior acquisition/normalization only.",
        "- No model training, canonical test, canonical multi, held-out query, active logs, or route/checkpoint selection.",
        "- This creates a hashed independent-prior input for a later query-free CPU gate only.",
        "",
        "## Source",
        "",
        f"- source page: `{payload['source_page']}`",
        f"- source URL: `{payload['source_url']}`",
        "",
        "## Summary",
        "",
        f"- pathways: `{payload['n_pathways']}`",
        f"- genes: `{payload['n_genes']}`",
        f"- pathway size min/median/max: `{payload['pathway_size_min']}` / `{payload['pathway_size_median']}` / `{payload['pathway_size_max']}`",
        "",
        "## Artifacts",
        "",
        "| artifact | path | size | sha256 |",
        "|---|---|---:|---|",
    ]
    for name, meta in payload["artifacts"].items():
        sha = meta.get("sha256")
        lines.append(f"| `{name}` | `{meta['path']}` | {meta['size_bytes']} | `{str(sha)[:16] if sha else 'NA'}` |")
    lines.extend(["", "## Next Gate", ""])
    lines.append("Use this prior only in a later train-only/internal proxy CPU gate with shuffled-prior controls. No GPU is authorized by acquisition alone.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    download_info = download_if_needed()
    summary = extract_and_parse()
    payload = {
        **summary,
        "timestamp": "2026-06-23 13:07 CST",
        "download_info": download_info,
        "boundary": {
            "external_prior_acquisition_only": True,
            "gpu_authorization": "none",
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "selection_or_tuning": False,
        },
        "artifacts": {
            "raw_zip": artifact(RAW_ZIP),
            "raw_gmt": artifact(RAW_GMT),
            "gene_pathways_tsv": artifact(GENE_PATHWAYS_TSV),
            "summary_json": artifact(SUMMARY_JSON),
        },
        "next_authorization": "query_free_cpu_gate_protocol_only",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
