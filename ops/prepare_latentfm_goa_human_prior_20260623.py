#!/usr/bin/env python3
"""Acquire and normalize GOA human annotations as an independent prior.

This short CPU/network task downloads the public Gene Ontology human GAF,
records provenance hashes, and emits simple gene-to-GO artifacts. It does not
train, evaluate, select routes/checkpoints, read canonical/query outputs, or
authorize GPU work.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "dataset" / "external_priors" / "goa_human_20260519"
REPORTS = ROOT / "reports"

SOURCE_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
SOURCE_PAGE = "https://current.geneontology.org/products/pages/downloads.html"
RELEASE_LABEL = "Gene Ontology annotations 2026-05-19 release"

RAW_GZ = OUT_DIR / "goa_human.gaf.gz"
GENE_TERMS_TSV = OUT_DIR / "goa_human_gene_terms.tsv"
TERM_GENES_GMT = OUT_DIR / "goa_human_term_genes.gmt"
SUMMARY_JSON = OUT_DIR / "goa_human_prior_summary.json"

OUT_JSON = REPORTS / "latentfm_goa_human_prior_acquisition_20260623.json"
OUT_MD = REPORTS / "LATENTFM_GOA_HUMAN_PRIOR_ACQUISITION_20260623.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_needed() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_GZ.is_file() and RAW_GZ.stat().st_size > 0:
        return {"downloaded": False, "reason": "existing_file_reused"}
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "LatentFM-prior-acquisition/20260623 (research provenance audit)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        headers = dict(response.headers.items())
        if status is not None and int(status) >= 400:
            raise RuntimeError(f"download failed with status {status}")
        tmp = RAW_GZ.with_suffix(".gaf.gz.tmp")
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(RAW_GZ)
    return {"downloaded": True, "reason": "downloaded_from_source", "source_url": SOURCE_URL, "headers": headers}


def parse_gaf() -> dict[str, Any]:
    gene_terms: dict[str, set[str]] = defaultdict(set)
    term_genes: dict[str, set[str]] = defaultdict(set)
    aspect_counts: dict[str, int] = defaultdict(int)
    evidence_counts: dict[str, int] = defaultdict(int)
    object_type_counts: dict[str, int] = defaultdict(int)
    skipped_not = 0
    skipped_short = 0
    n_rows = 0
    with gzip.open(RAW_GZ, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                skipped_short += 1
                continue
            n_rows += 1
            qualifier = parts[3]
            if "NOT" in qualifier.split("|"):
                skipped_not += 1
                continue
            symbol = parts[2].strip()
            go_id = parts[4].strip()
            evidence = parts[6].strip()
            aspect = parts[8].strip()
            object_type = parts[11].strip()
            if not symbol or not go_id:
                continue
            gene = symbol.upper()
            gene_terms[gene].add(go_id)
            term_genes[go_id].add(gene)
            aspect_counts[aspect] += 1
            evidence_counts[evidence] += 1
            object_type_counts[object_type] += 1

    with GENE_TERMS_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene", "n_go_terms", "go_terms"])
        for gene in sorted(gene_terms):
            terms = sorted(gene_terms[gene])
            writer.writerow([gene, len(terms), ";".join(terms)])

    with TERM_GENES_GMT.open("w", encoding="utf-8") as handle:
        for term in sorted(term_genes):
            genes = sorted(term_genes[term])
            if len(genes) >= 2:
                handle.write("\t".join([term, f"{RELEASE_LABEL} {term}", *genes]) + "\n")

    summary = {
        "status": "goa_human_prior_acquired_normalized_no_gpu",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "release_label": RELEASE_LABEL,
        "raw_gaf_gz": str(RAW_GZ),
        "gene_terms_tsv": str(GENE_TERMS_TSV),
        "term_genes_gmt": str(TERM_GENES_GMT),
        "n_gaf_rows": n_rows,
        "n_genes": len(gene_terms),
        "n_go_terms": len(term_genes),
        "n_gmt_terms_min2_genes": sum(1 for genes in term_genes.values() if len(genes) >= 2),
        "skipped_not_qualifier_rows": skipped_not,
        "skipped_short_rows": skipped_short,
        "aspect_counts": dict(sorted(aspect_counts.items())),
        "top_evidence_counts": dict(sorted(evidence_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
        "object_type_counts": dict(sorted(object_type_counts.items())),
        "hashes": {
            "raw_gaf_gz": sha256_file(RAW_GZ),
            "gene_terms_tsv": sha256_file(GENE_TERMS_TSV),
            "term_genes_gmt": sha256_file(TERM_GENES_GMT),
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
        "# GOA Human Prior Acquisition",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- External prior acquisition/normalization only.",
        "- No model training, no canonical test, no canonical multi, no held-out query, no active logs, no route/checkpoint selection.",
        "- This does not authorize a GPU run; it only creates a hashed independent-prior input for a later query-free CPU gate.",
        "",
        "## Source",
        "",
        f"- release: {payload['release_label']}",
        f"- source page: `{payload['source_page']}`",
        f"- source URL: `{payload['source_url']}`",
        "",
        "## Summary",
        "",
        f"- GAF rows parsed: `{payload['n_gaf_rows']}`",
        f"- genes with GO terms: `{payload['n_genes']}`",
        f"- GO terms: `{payload['n_go_terms']}`",
        f"- GMT terms with at least 2 genes: `{payload['n_gmt_terms_min2_genes']}`",
        f"- NOT-qualified rows skipped: `{payload['skipped_not_qualifier_rows']}`",
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
    lines.append(
        "Use this prior only in a later train-only/internal proxy CPU gate with disjoint confirmation, "
        "shuffled-prior controls, and comparisons against closed Track A baselines. No GPU is authorized by acquisition alone."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    download_info = download_if_needed()
    summary = parse_gaf()
    payload = {
        **summary,
        "timestamp": "2026-06-23 13:00 CST",
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
            "raw_gaf_gz": artifact(RAW_GZ),
            "gene_terms_tsv": artifact(GENE_TERMS_TSV),
            "term_genes_gmt": artifact(TERM_GENES_GMT),
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
