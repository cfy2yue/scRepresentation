#!/usr/bin/env python3
"""Audit xVERSE gene-id resolution readiness for the LatentFM DE5000 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/data/cyx/1030/scLatent/scFM_output/embedding_runs/manifest_latentfm_full_de5000.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/xverse_gene_id_readiness_20260620.json"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/XVERSE_GENE_ID_READINESS_20260620.md"))
    args = parser.parse_args()

    fm_root = Path("/data/cyx/1030/scLatent/scFMBench/fm")
    if str(fm_root) not in sys.path:
        sys.path.insert(0, str(fm_root))
    from adapters.xverse.encoder import _default_gene_ids_path, _load_gene_list, _resolve_ensembl_series

    model_ensg = _load_gene_list(_default_gene_ids_path())
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = Path(rec["path"])
        adata = ad.read_h5ad(path, backed="r")
        try:
            _vals, report = _resolve_ensembl_series(adata, gene_col=None, model_ensg=model_ensg)
            status = "pass" if int(report.get("aligned_to_xverse") or 0) > 0 else "fail"
            error = ""
        except Exception as exc:
            report = {}
            status = "fail"
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "dataset_id": rec.get("dataset_id"),
            "path": str(path),
            "status": status,
            "n_vars": int(adata.n_vars),
            "var_columns": list(map(str, adata.var.columns)),
            "var_names_sample": [str(x) for x in list(adata.var_names[:5])],
            "report": report,
            "error": error,
        })
        adata.file.close()

    summary = {
        "manifest": str(args.manifest),
        "n_datasets": len(rows),
        "n_pass": sum(1 for r in rows if r["status"] == "pass"),
        "n_fail": sum(1 for r in rows if r["status"] != "pass"),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# xVERSE Gene-ID Readiness",
        "",
        f"- manifest: `{args.manifest}`",
        f"- datasets: {summary['n_datasets']}",
        f"- pass: {summary['n_pass']}",
        f"- fail: {summary['n_fail']}",
        "",
        "| dataset | status | source | n_vars | mapped ENSG | aligned xVERSE | unique aligned | notes |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report = row.get("report") or {}
        notes = row.get("error") or ""
        if report.get("source") == "var_names_symbol_map":
            notes = f"symbol_hits={report.get('symbol_hits')}; map_size={report.get('symbol_map_size')}"
        lines.append(
            "| {dataset} | {status} | {source} | {n_vars} | {mapped} | {aligned} | {unique} | {notes} |".format(
                dataset=row["dataset_id"],
                status=row["status"],
                source=report.get("source", ""),
                n_vars=row["n_vars"],
                mapped=report.get("mapped_ensg", ""),
                aligned=report.get("aligned_to_xverse", ""),
                unique=report.get("unique_aligned_to_xverse", ""),
                notes=str(notes).replace("|", "/"),
            )
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
