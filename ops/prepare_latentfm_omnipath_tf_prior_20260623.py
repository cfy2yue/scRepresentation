#!/usr/bin/env python3
"""Acquire and normalize OmniPath TF-target priors for LatentFM.

This short CPU/network task downloads signed human TF-target interactions from
the OmniPath interactions endpoint. It creates hashed provenance artifacts only:
no model training, no canonical/query outputs, no route selection, and no GPU
authorization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "dataset" / "external_priors" / "omnipath_tf_20260623"
REPORTS = ROOT / "reports"

SOURCE_PAGE = "https://r.omnipathdb.org/reference/collectri.html"
INTERACTIONS_DOC = "https://r.omnipathdb.org/reference/import_omnipath_interactions.html"
DATASETS = ("collectri", "dorothea", "tf_target")
SOURCE_URL = (
    "https://omnipathdb.org/interactions?"
    "datasets=collectri,dorothea,tf_target&organisms=9606&genesymbols=yes&format=tsv"
)

RAW_TSV = OUT_DIR / "omnipath_collectri_dorothea_tftarget_raw.tsv"
EDGES_TSV = OUT_DIR / "omnipath_tf_target_edges.tsv"
GENE_FEATURES_TSV = OUT_DIR / "omnipath_tf_target_gene_features.tsv"
SUMMARY_JSON = OUT_DIR / "omnipath_tf_prior_summary.json"
OUT_JSON = REPORTS / "latentfm_omnipath_tf_prior_acquisition_20260623.json"
OUT_MD = REPORTS / "LATENTFM_OMNIPATH_TF_PRIOR_ACQUISITION_20260623.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_if_needed() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_TSV.is_file() and RAW_TSV.stat().st_size > 0:
        return {"downloaded": False, "reason": "existing_file_reused", "source_url": SOURCE_URL}
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "LatentFM-OmniPath-prior-acquisition/20260623"},
    )
    tmp = RAW_TSV.with_suffix(".tsv.tmp")
    with urllib.request.urlopen(req, timeout=180) as response:
        headers = dict(response.headers.items())
        status = getattr(response, "status", None)
        if status is not None and int(status) >= 400:
            raise RuntimeError(f"download failed with status {status}")
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    tmp.replace(RAW_TSV)
    return {"downloaded": True, "reason": "downloaded_from_source", "source_url": SOURCE_URL, "headers": headers}


def split_symbols(value: str) -> list[str]:
    out = []
    for token in str(value or "").replace(",", "_").split("_"):
        token = token.strip().upper()
        if token and not token.startswith("COMPLEX:"):
            out.append(token)
    return sorted(set(out))


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_edges() -> dict[str, Any]:
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    rows = 0
    skipped_complex_source = 0
    skipped_missing_symbol = 0
    with RAW_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows += 1
            sources = split_symbols(row.get("source_genesymbol") or "")
            targets = split_symbols(row.get("target_genesymbol") or "")
            if "_" in str(row.get("source_genesymbol") or ""):
                skipped_complex_source += 1
            if not sources or not targets:
                skipped_missing_symbol += 1
                continue
            stim = truthy(row.get("consensus_stimulation") or row.get("is_stimulation") or "")
            inhib = truthy(row.get("consensus_inhibition") or row.get("is_inhibition") or "")
            directed = truthy(row.get("consensus_direction") or row.get("is_directed") or "")
            for source in sources:
                for target in targets:
                    key = (source, target)
                    item = edge_map.setdefault(
                        key,
                        {
                            "tf": source,
                            "target": target,
                            "n_raw_rows": 0,
                            "directed_votes": 0,
                            "stimulation_votes": 0,
                            "inhibition_votes": 0,
                        },
                    )
                    item["n_raw_rows"] += 1
                    item["directed_votes"] += int(directed)
                    item["stimulation_votes"] += int(stim)
                    item["inhibition_votes"] += int(inhib)

    edges = []
    for item in edge_map.values():
        sign = 0
        if item["stimulation_votes"] > item["inhibition_votes"]:
            sign = 1
        elif item["inhibition_votes"] > item["stimulation_votes"]:
            sign = -1
        item["sign"] = sign
        edges.append(item)
    edges.sort(key=lambda x: (x["tf"], x["target"]))

    with EDGES_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "tf",
                "target",
                "sign",
                "n_raw_rows",
                "directed_votes",
                "stimulation_votes",
                "inhibition_votes",
            ],
        )
        writer.writeheader()
        writer.writerows(edges)

    by_gene: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in edges:
        by_gene[edge["tf"]]["tf_out_degree"] += 1
        by_gene[edge["target"]]["target_in_degree"] += 1
        if edge["sign"] > 0:
            by_gene[edge["tf"]]["tf_activation_out_degree"] += 1
            by_gene[edge["target"]]["target_activation_in_degree"] += 1
        elif edge["sign"] < 0:
            by_gene[edge["tf"]]["tf_inhibition_out_degree"] += 1
            by_gene[edge["target"]]["target_inhibition_in_degree"] += 1

    feature_fields = [
        "gene",
        "tf_out_degree",
        "target_in_degree",
        "tf_activation_out_degree",
        "tf_inhibition_out_degree",
        "target_activation_in_degree",
        "target_inhibition_in_degree",
    ]
    with GENE_FEATURES_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=feature_fields)
        writer.writeheader()
        for gene in sorted(by_gene):
            row = {"gene": gene}
            row.update({field: int(by_gene[gene][field]) for field in feature_fields if field != "gene"})
            writer.writerow(row)

    summary = {
        "status": "omnipath_tf_prior_acquired_normalized_no_gpu",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "interactions_doc": INTERACTIONS_DOC,
        "datasets": list(DATASETS),
        "raw_rows": rows,
        "deduplicated_edges": len(edges),
        "genes_with_features": len(by_gene),
        "tf_genes": len({edge["tf"] for edge in edges}),
        "target_genes": len({edge["target"] for edge in edges}),
        "signed_edges": sum(1 for edge in edges if edge["sign"] != 0),
        "activation_edges": sum(1 for edge in edges if edge["sign"] > 0),
        "inhibition_edges": sum(1 for edge in edges if edge["sign"] < 0),
        "skipped_complex_source_rows": skipped_complex_source,
        "skipped_missing_symbol_rows": skipped_missing_symbol,
        "hashes": {
            "raw_tsv": sha256_file(RAW_TSV),
            "edges_tsv": sha256_file(EDGES_TSV),
            "gene_features_tsv": sha256_file(GENE_FEATURES_TSV),
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
        "# OmniPath TF Prior Acquisition",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- External prior acquisition/normalization only.",
        "- No model training, no canonical test, no canonical multi, no held-out query, no active logs, no route/checkpoint selection.",
        "- This does not authorize a GPU run; it only creates hashed independent-prior inputs for later query-free CPU gates.",
        "",
        "## Source",
        "",
        f"- datasets: `{','.join(payload['datasets'])}`",
        f"- source page: `{payload['source_page']}`",
        f"- interactions doc: `{payload['interactions_doc']}`",
        f"- source URL: `{payload['source_url']}`",
        "",
        "## Summary",
        "",
        f"- raw rows: `{payload['raw_rows']}`",
        f"- deduplicated directed edges: `{payload['deduplicated_edges']}`",
        f"- genes with features: `{payload['genes_with_features']}`",
        f"- TF genes: `{payload['tf_genes']}`",
        f"- target genes: `{payload['target_genes']}`",
        f"- signed edges: `{payload['signed_edges']}`",
        f"- activation/inhibition edges: `{payload['activation_edges']}` / `{payload['inhibition_edges']}`",
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
        "Use this prior only in a later train-only/internal proxy CPU gate with shuffled-prior controls. "
        "Acquisition alone is not model evidence and gives no GPU authorization."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    download = download_if_needed()
    summary = parse_edges()
    payload = {
        **summary,
        "timestamp": "2026-06-23 13:35 CST",
        "download_info": download,
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
            "raw_tsv": artifact(RAW_TSV),
            "edges_tsv": artifact(EDGES_TSV),
            "gene_features_tsv": artifact(GENE_FEATURES_TSV),
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
