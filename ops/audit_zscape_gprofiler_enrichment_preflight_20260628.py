#!/usr/bin/env python3
"""ZSCAPE Danio rerio pathway-enrichment preflight.

This script consumes the frozen expression/log1p DE preflight outputs and runs
a CPU-only g:Profiler enrichment query. It uses Ensembl gene IDs and the genes
tested in the selected expression matrix as a custom background.

It is a biological interpretation preflight, not a LatentFM model gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_EXPR_DIR = ROOT / "reports/zscape_expression_latent_biology_preflight_20260628"
DEFAULT_FLOW_ROWS = ROOT / "reports/zscape_flow_constraint_feasibility_20260628/zscape_flow_constraint_feasibility_rows.csv"
DEFAULT_BACKGROUND = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_gene_names.txt"
DEFAULT_OUT = ROOT / "reports/zscape_gprofiler_enrichment_preflight_20260628"
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_background(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            gene = line.strip()
            if gene and gene not in seen:
                seen.add(gene)
                out.append(gene)
    return out


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    if abs(val) < 1e-3 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def safe_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def build_queries(
    top_genes: list[dict[str, str]],
    flow_rows: dict[str, dict[str, str]],
    top_n: int,
    min_query_genes: int,
    primary_only: bool,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    by_row: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_genes:
        by_row[row["row_id"]].append(row)

    queries: dict[str, list[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row_id, rows in sorted(by_row.items()):
        flow = flow_rows.get(row_id, {})
        if primary_only and flow.get("audit_role") != "primary_mechanism_test":
            continue

        for direction, sign in (("up", 1), ("down", -1)):
            if sign > 0:
                selected = [r for r in rows if safe_float(r.get("welch_z_proxy", "")) > 0]
                selected.sort(key=lambda r: -safe_float(r.get("welch_z_proxy", "")))
            else:
                selected = [r for r in rows if safe_float(r.get("welch_z_proxy", "")) < 0]
                selected.sort(key=lambda r: safe_float(r.get("welch_z_proxy", "")))

            seen: set[str] = set()
            genes: list[str] = []
            symbols: list[str] = []
            for item in selected:
                gene_id = item.get("gene_id", "").strip()
                if not gene_id or gene_id in seen:
                    continue
                seen.add(gene_id)
                genes.append(gene_id)
                symbols.append(item.get("gene_symbol", "").strip())
                if len(genes) >= top_n:
                    break
            if len(genes) < min_query_genes:
                continue

            query_name = f"{row_id}.{direction}"
            queries[query_name] = genes
            metadata[query_name] = {
                "query_name": query_name,
                "row_id": row_id,
                "direction": direction,
                "lineage": flow.get("lineage", ""),
                "target": flow.get("target", ""),
                "timepoint": flow.get("timepoint", ""),
                "audit_role": flow.get("audit_role", ""),
                "constraint_feasibility_class": flow.get("constraint_feasibility_class", ""),
                "recommended_constraint_use": flow.get("recommended_constraint_use", ""),
                "strict_row_gate": flow.get("strict_row_gate", ""),
                "trajectory_alignment_gate": flow.get("trajectory_alignment_gate", ""),
                "hvg2000_response_energy_share": flow.get("hvg2000_response_energy_share", ""),
                "input_gene_count": len(genes),
                "top_symbols": ";".join([s for s in symbols if s][:12]),
            }
    return queries, metadata


def post_gprofiler(queries: dict[str, list[str]], background: list[str], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "organism": args.organism,
        "query": queries,
        "sources": args.sources,
        "user_threshold": args.user_threshold,
        "significance_threshold_method": args.threshold_method,
        "domain_scope": "custom_annotated" if args.custom_background else "annotated",
        "ordered": False,
        "no_evidences": False,
        "measure_underrepresentation": False,
    }
    if args.custom_background:
        payload["background"] = background
    resp = requests.post(GPROFILER_URL, json=payload, timeout=args.timeout_seconds)
    out = {
        "request_url": GPROFILER_URL,
        "http_status": resp.status_code,
        "payload_summary": {
            "organism": args.organism,
            "sources": args.sources,
            "user_threshold": args.user_threshold,
            "threshold_method": args.threshold_method,
            "domain_scope": payload["domain_scope"],
            "custom_background": bool(args.custom_background),
            "background_genes": len(background) if args.custom_background else 0,
            "query_count": len(queries),
        },
    }
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        out.update({"ok": False, "error": f"non_json_response: {exc}", "text_prefix": resp.text[:2000]})
        return out
    out["response"] = body
    out["ok"] = resp.ok and isinstance(body, dict)
    if not resp.ok:
        out["error"] = body
    return out


def summarize(
    api: dict[str, Any],
    query_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    response = api.get("response") or {}
    terms_raw = response.get("result") or []
    meta = response.get("meta") or {}
    genes_meta = meta.get("genes_metadata") or {}
    query_gene_meta = genes_meta.get("query") or {}

    terms: list[dict[str, Any]] = []
    term_counts: dict[str, int] = defaultdict(int)
    top_terms: dict[str, list[str]] = defaultdict(list)

    for item in terms_raw:
        qname = item.get("query", "")
        qmeta = query_meta.get(qname, {})
        row = {
            **qmeta,
            "source": item.get("source", ""),
            "native": item.get("native", ""),
            "name": item.get("name", ""),
            "p_value": item.get("p_value", ""),
            "significant": item.get("significant", ""),
            "term_size": item.get("term_size", ""),
            "query_size": item.get("query_size", ""),
            "intersection_size": item.get("intersection_size", ""),
            "effective_domain_size": item.get("effective_domain_size", ""),
            "precision": item.get("precision", ""),
            "recall": item.get("recall", ""),
            "parents": ";".join(item.get("parents") or []),
            "description": item.get("description", ""),
        }
        terms.append(row)
        if str(item.get("significant", "")).lower() == "true":
            term_counts[qname] += 1
            if len(top_terms[qname]) < 5:
                top_terms[qname].append(f"{item.get('source')}:{item.get('name')} p={fmt(item.get('p_value'), 2)}")

    summaries: list[dict[str, Any]] = []
    for qname, qmeta in sorted(query_meta.items()):
        mapping = ((query_gene_meta.get(qname) or {}).get("mapping") or {})
        mapped_inputs = len(mapping)
        input_count = int(qmeta.get("input_gene_count", 0))
        mapping_rate = mapped_inputs / max(input_count, 1)
        summaries.append(
            {
                **qmeta,
                "mapped_input_genes": mapped_inputs,
                "mapping_rate": mapping_rate,
                "significant_term_count": term_counts.get(qname, 0),
                "top_terms": " | ".join(top_terms.get(qname, [])),
            }
        )

    meta_summary = {
        "gprofiler_timestamp": meta.get("timestamp", ""),
        "gprofiler_version": meta.get("version", ""),
        "failed_gene_count_global": len(genes_meta.get("failed") or []),
        "ambiguous_gene_count_global": len(genes_meta.get("ambiguous") or {}),
        "duplicate_gene_count_global": len(genes_meta.get("duplicates") or []),
        "raw_term_count": len(terms_raw),
    }
    return summaries, terms, meta_summary


def write_report(
    out_dir: Path,
    api: dict[str, Any],
    summaries: list[dict[str, Any]],
    terms: list[dict[str, Any]],
    meta_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    report = out_dir / "LATENTFM_ZSCAPE_GPROFILER_ENRICHMENT_PREFLIGHT_20260628.md"
    primary = [r for r in summaries if r.get("audit_role") == "primary_mechanism_test"]
    best = [
        r
        for r in summaries
        if r.get("constraint_feasibility_class") == "best_candidate_pending_fixedcell_placebo"
    ]
    lines: list[str] = []
    lines.append("# LatentFM ZSCAPE g:Profiler Enrichment Preflight")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append("Status: `zscape_gprofiler_enrichment_preflight_no_gpu`" if api.get("ok") else "Status: `zscape_gprofiler_enrichment_preflight_blocked`")
    lines.append("")
    lines.append("GPU authorized: `False`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/network-only pathway enrichment preflight from frozen expression-space DE outputs.")
    lines.append("- Input DE used size-factor normalization followed by one `log1p`; this script does not redo normalization or QC filtering.")
    lines.append("- Queries use Ensembl gene IDs; background is the selected ZSCAPE expression gene universe when custom background is enabled.")
    lines.append("- This is not a LatentFM checkpoint-selection artifact and cannot authorize ZSCAPE GPU training.")
    lines.append("")
    lines.append("## g:Profiler Request")
    lines.append("")
    payload = api.get("payload_summary") or {}
    lines.append(f"- organism: `{payload.get('organism', args.organism)}`")
    lines.append(f"- sources: `{','.join(payload.get('sources', args.sources))}`")
    lines.append(f"- threshold: `{payload.get('user_threshold', args.user_threshold)}` with `{payload.get('threshold_method', args.threshold_method)}`")
    lines.append(f"- domain scope: `{payload.get('domain_scope', '')}`")
    lines.append(f"- custom background genes: `{payload.get('background_genes', 0)}`")
    lines.append(f"- query count: `{payload.get('query_count', len(summaries))}`")
    lines.append(f"- HTTP status: `{api.get('http_status', '')}`")
    lines.append(f"- g:Profiler timestamp/version: `{meta_summary.get('gprofiler_timestamp', '')}` / `{meta_summary.get('gprofiler_version', '')}`")
    lines.append("")
    if not api.get("ok"):
        lines.append("## Blocker")
        lines.append("")
        lines.append("The API request did not return a valid successful g:Profiler response. Keep this branch as blocked until rerun.")
        lines.append("")
        lines.append("```text")
        lines.append(str(api.get("error") or api.get("text_prefix") or "unknown error")[:2000])
        lines.append("```")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Mapping/Term Summary")
    lines.append("")
    lines.append(f"- raw returned term rows: `{meta_summary.get('raw_term_count', 0)}`")
    lines.append(f"- global failed genes reported by g:Profiler: `{meta_summary.get('failed_gene_count_global', 0)}`")
    lines.append(f"- global ambiguous genes reported by g:Profiler: `{meta_summary.get('ambiguous_gene_count_global', 0)}`")
    lines.append(f"- global duplicate genes reported by g:Profiler: `{meta_summary.get('duplicate_gene_count_global', 0)}`")
    lines.append("")
    lines.append("## Primary Rows")
    lines.append("")
    lines.append("| lineage | target | time | direction | class | input | mapping | sig terms | top terms |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---|")
    for row in primary:
        lines.append(
            "| {lineage} | {target} | {timepoint} | {direction} | `{klass}` | {input_gene_count} | {mapping} | {sig} | {terms} |".format(
                lineage=row.get("lineage", ""),
                target=row.get("target", ""),
                timepoint=row.get("timepoint", ""),
                direction=row.get("direction", ""),
                klass=row.get("constraint_feasibility_class", ""),
                input_gene_count=row.get("input_gene_count", ""),
                mapping=fmt(row.get("mapping_rate", 0), 3),
                sig=row.get("significant_term_count", 0),
                terms=str(row.get("top_terms", "")).replace("|", "/")[:180],
            )
        )
    lines.append("")
    lines.append("## Best-Candidate Periderm Interpretation")
    lines.append("")
    if not best:
        lines.append("- No best-candidate periderm query was present in this run.")
    else:
        for row in best:
            lines.append(
                "- `{row_id}` `{direction}`: mapping `{mapping}`, significant terms `{sig}`; top terms: {terms}".format(
                    row_id=row.get("row_id", ""),
                    direction=row.get("direction", ""),
                    mapping=fmt(row.get("mapping_rate", 0), 3),
                    sig=row.get("significant_term_count", 0),
                    terms=row.get("top_terms", "") or "none",
                )
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Enrichment can support biological interpretation of expression-space responses, but only after fixed-cell/placebo gates confirm the response is not a cell-composition artifact.")
    lines.append("- Strong mitochondrial/electron-transport terms should be treated cautiously because they may reflect stress, capture, or QC/library effects rather than target-specific biology.")
    lines.append("- Latent-space conclusions still require a separate frozen scFM/ZSCAPE embedding extraction protocol; the current SVD/UMAP diagnostics are only proxies.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- summary: `{out_dir / 'zscape_gprofiler_enrichment_summary.csv'}`")
    lines.append(f"- terms: `{out_dir / 'zscape_gprofiler_enrichment_terms.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'zscape_gprofiler_enrichment_preflight_20260628.json'}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr-dir", type=Path, default=DEFAULT_EXPR_DIR)
    parser.add_argument("--flow-rows", type=Path, default=DEFAULT_FLOW_ROWS)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--organism", default="drerio")
    parser.add_argument("--sources", nargs="+", default=["GO:BP", "GO:MF", "GO:CC", "REAC", "WP", "KEGG"])
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-query-genes", type=int, default=10)
    parser.add_argument("--user-threshold", type=float, default=0.05)
    parser.add_argument("--threshold-method", default="g_SCS")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--custom-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--primary-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_genes = read_csv(args.expr_dir / "zscape_expression_de_top_genes.csv")
    flow_rows = {row["row_id"]: row for row in read_csv(args.flow_rows)}
    background = read_background(args.background)
    queries, query_meta = build_queries(top_genes, flow_rows, args.top_n, args.min_query_genes, args.primary_only)

    api = post_gprofiler(queries, background, args) if queries else {"ok": False, "error": "no_queries_built"}
    summaries, terms, meta_summary = summarize(api, query_meta) if api.get("ok") else ([], [], {})

    summary_fields = [
        "query_name",
        "row_id",
        "direction",
        "lineage",
        "target",
        "timepoint",
        "audit_role",
        "constraint_feasibility_class",
        "recommended_constraint_use",
        "strict_row_gate",
        "trajectory_alignment_gate",
        "hvg2000_response_energy_share",
        "input_gene_count",
        "mapped_input_genes",
        "mapping_rate",
        "significant_term_count",
        "top_symbols",
        "top_terms",
    ]
    term_fields = [
        "query_name",
        "row_id",
        "direction",
        "lineage",
        "target",
        "timepoint",
        "constraint_feasibility_class",
        "source",
        "native",
        "name",
        "p_value",
        "significant",
        "term_size",
        "query_size",
        "intersection_size",
        "effective_domain_size",
        "precision",
        "recall",
        "parents",
        "description",
    ]
    write_csv(args.out_dir / "zscape_gprofiler_enrichment_summary.csv", summaries, summary_fields)
    write_csv(args.out_dir / "zscape_gprofiler_enrichment_terms.csv", terms, term_fields)
    payload = {
        "timestamp": now_cst(),
        "status": "zscape_gprofiler_enrichment_preflight_no_gpu" if api.get("ok") else "zscape_gprofiler_enrichment_preflight_blocked",
        "inputs": {
            "expr_dir": str(args.expr_dir),
            "flow_rows": str(args.flow_rows),
            "background": str(args.background),
        },
        "query_count": len(queries),
        "summary_rows": len(summaries),
        "term_rows": len(terms),
        "meta_summary": meta_summary,
        "api": api,
    }
    (args.out_dir / "zscape_gprofiler_enrichment_preflight_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir, api, summaries, terms, meta_summary, args)


if __name__ == "__main__":
    main()
