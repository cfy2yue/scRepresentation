#!/usr/bin/env python3
"""Audit whether installed scFM encoders can legally embed ZSCAPE zebrafish cells.

The current ZSCAPE matrix uses Danio rerio Ensembl IDs (ENSDARG...). Most local
foundation-model assets are human-oriented. This CPU-only audit checks direct
gene/vocabulary compatibility before any GPU embedding extraction is launched.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_GENES = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_gene_names.txt"
DEFAULT_GENE_META = ROOT / "dataset/external/zscape_20260628/GSE202639_zperturb_full_gene_metadata.csv.gz"
DEFAULT_OUT = ROOT / "reports/zscape_scfm_latent_readiness_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def read_gene_symbols(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, str] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gid = str(row.get("id", "")).strip()
            sym = str(row.get("gene_short_name", "")).strip()
            if gid:
                out[gid] = sym
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_json_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(obj, dict):
        return {str(k) for k in obj}
    if isinstance(obj, list):
        return {str(x) for x in obj}
    return set()


def load_tf_h5_gene_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        import h5py
    except Exception:
        return set()
    try:
        with h5py.File(path, "r") as h5:
            if "arrays" in h5:
                return {str(k) for k in h5["arrays"].keys()}
    except Exception:
        return set()
    return set()


def load_h5ad_var_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        import anndata as ad
    except Exception:
        return set()
    try:
        obj = ad.read_h5ad(path, backed="r")
        return {str(x) for x in obj.var_names}
    except Exception:
        return set()


def load_stack_genes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        obj = pickle.load(open(path, "rb"))
    except Exception:
        return set()
    if isinstance(obj, dict):
        vals = obj.keys()
    else:
        vals = obj
    try:
        return {str(x) for x in vals}
    except TypeError:
        return set()


def overlap_report(name: str, vocab: set[str], z_ids: set[str], z_symbols: set[str], note: str) -> dict[str, Any]:
    vocab_upper = {x.upper() for x in vocab}
    z_symbol_upper = {x.upper() for x in z_symbols if x}
    id_overlap = len(vocab & z_ids)
    symbol_overlap = len(vocab & z_symbols)
    symbol_upper_overlap = len(vocab_upper & z_symbol_upper)
    # Symbol overlap across species is not a legal direct-embedding criterion:
    # many gene symbols are conserved or collide across species, while the model
    # embedding table and pretrained distribution remain human-oriented. Treat it
    # as a useful orthology-mapping hint only.
    direct_ok = id_overlap >= 1000
    return {
        "model_or_asset": name,
        "vocab_size": len(vocab),
        "zscape_id_overlap": id_overlap,
        "zscape_symbol_overlap_exact": symbol_overlap,
        "zscape_symbol_overlap_upper": symbol_upper_overlap,
        "direct_species_compatible": direct_ok,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    parser.add_argument("--gene-meta", type=Path, default=DEFAULT_GENE_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    z_gene_ids = read_lines(args.genes)
    gene_symbols = read_gene_symbols(args.gene_meta)
    z_ids = set(z_gene_ids)
    z_symbols = {gene_symbols.get(g, "") for g in z_gene_ids}
    z_symbols = {s for s in z_symbols if s}
    z_prefix_counts: dict[str, int] = {}
    for gid in z_gene_ids:
        prefix = "ENSDARG" if gid.startswith("ENSDARG") else ("ENSG" if gid.startswith("ENSG") else gid.split("0", 1)[0][:12])
        z_prefix_counts[prefix] = z_prefix_counts.get(prefix, 0) + 1

    assets = {
        "xverse_human_ensg": read_lines(ROOT / "scFM_third_party/xVERSE_code/main/ensg_keys_high_quality.txt"),
        "transcriptformer_tf_sapiens_gene_h5": sorted(
            load_tf_h5_gene_keys(ROOT / "scFM_pretrained/transcriptformer/tf_sapiens/vocabs/homo_sapiens_gene.h5")
        ),
        "nicheformer_model_mean_var": sorted(
            load_h5ad_var_names(ROOT / "scFM_pretrained/nicheformer/theislab_Nicheformer/model.h5ad")
        ),
        "nicheformer_vocab": sorted(load_json_keys(ROOT / "scFM_pretrained/nicheformer/theislab_Nicheformer/vocab.json")),
        "scgpt_vocab": sorted(load_json_keys(ROOT / "scFM_pretrained/scgpt/vocab.json")),
        "stack_genelist": sorted(load_stack_genes(ROOT / "scFM_pretrained/stack/basecount_1000per_15000max.pkl")),
    }

    rows = [
        overlap_report(
            "xVERSE local checkpoint vocab",
            set(assets["xverse_human_ensg"]),
            z_ids,
            z_symbols,
            "Human ENSG gene list; direct Danio ENSDARG overlap should be zero.",
        ),
        overlap_report(
            "TranscriptFormer tf_sapiens gene vocab",
            set(assets["transcriptformer_tf_sapiens_gene_h5"]),
            z_ids,
            z_symbols,
            "Only tf_sapiens checkpoint is installed locally; tf_metazoa is absent.",
        ),
        overlap_report(
            "NicheFormer model mean var_names",
            set(assets["nicheformer_model_mean_var"]),
            z_ids,
            z_symbols,
            "Installed model mean is human ENSG-like.",
        ),
        overlap_report(
            "NicheFormer tokenizer vocab",
            set(assets["nicheformer_vocab"]),
            z_ids,
            z_symbols,
            "Tokenizer has species tokens, but local vocab inspection does not establish Danio support.",
        ),
        overlap_report(
            "scGPT vocab",
            set(assets["scgpt_vocab"]),
            z_ids,
            z_symbols,
            "Symbol overlap can exist but does not prove species-safe Danio encoding.",
        ),
        overlap_report(
            "Stack genelist",
            set(assets["stack_genelist"]),
            z_ids,
            z_symbols,
            "Stack local checkpoint/genelist must be treated as human-oriented unless orthology is frozen.",
        ),
    ]

    tf_metazoa_dir = ROOT / "scFM_pretrained/transcriptformer/tf_metazoa"
    niche_vocab = set(assets["nicheformer_vocab"])
    summary = {
        "timestamp": now_cst(),
        "status": "zscape_scfm_latent_readiness_blocked_no_gpu",
        "zscape_gene_count": len(z_gene_ids),
        "zscape_unique_gene_count": len(z_ids),
        "zscape_symbol_count": len(z_symbols),
        "zscape_gene_prefix_counts": z_prefix_counts,
        "tf_metazoa_installed": bool((tf_metazoa_dir / "config.json").is_file() and (tf_metazoa_dir / "model_weights.pt").is_file()),
        "nicheformer_species_tokens": sorted([x for x in niche_vocab if x.startswith("[SPECIES_")]),
        "direct_compatible_assets": [r["model_or_asset"] for r in rows if r["direct_species_compatible"]],
        "decision": "no_true_scfm_zscape_embedding_launch_without_species_gate",
    }

    write_csv(
        args.out_dir / "zscape_scfm_latent_readiness_rows.csv",
        rows,
        [
            "model_or_asset",
            "vocab_size",
            "zscape_id_overlap",
            "zscape_symbol_overlap_exact",
            "zscape_symbol_overlap_upper",
            "direct_species_compatible",
            "note",
        ],
    )
    (args.out_dir / "zscape_scfm_latent_readiness_20260628.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    report = args.out_dir / "LATENTFM_ZSCAPE_SCFM_LATENT_READINESS_20260628.md"
    lines = [
        "# LatentFM ZSCAPE scFM Latent Readiness",
        "",
        f"Timestamp: `{summary['timestamp']}`",
        "",
        "Status: `zscape_scfm_latent_readiness_blocked_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only species/vocabulary compatibility audit before any ZSCAPE scFM embedding extraction.",
        "- Does not train, infer, use canonical multi, or read Track C query.",
        "- The goal is to prevent human-vocabulary embeddings from being misread as zebrafish biological latent structure.",
        "",
        "## ZSCAPE Gene Space",
        "",
        f"- genes in selected expression matrix: `{summary['zscape_gene_count']}`",
        f"- unique gene ids: `{summary['zscape_unique_gene_count']}`",
        f"- gene-id prefixes: `{summary['zscape_gene_prefix_counts']}`",
        f"- unique gene symbols: `{summary['zscape_symbol_count']}`",
        "",
        "## Local Encoder Compatibility",
        "",
        "| asset | vocab | ENSDARG/ID overlap | symbol overlap upper | direct compatible | note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {asset} | {vocab} | {idov} | {symov} | `{ok}` | {note} |".format(
                asset=row["model_or_asset"],
                vocab=row["vocab_size"],
                idov=row["zscape_id_overlap"],
                symov=row["zscape_symbol_overlap_upper"],
                ok=row["direct_species_compatible"],
                note=row["note"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Do not launch true scFM ZSCAPE embedding extraction from the installed human-oriented assets yet.",
            "- Current latent-space ZSCAPE conclusions remain limited to log1p-HVG SVD and metadata UMAP proxy diagnostics.",
            "- A true scFM latent route needs one of: validated `tf_metazoa`/Danio-compatible checkpoint, explicit Danio-to-human orthology mapping with loss audit, or a zebrafish-trained representation.",
            "- If a compatible checkpoint is added, first create a small 128-cell smoke h5ad with `.X=log1p` and `layers['counts']`, then run one detached embedding smoke with RUN_STATUS.",
            "",
            "## Outputs",
            "",
            f"- rows: `{args.out_dir / 'zscape_scfm_latent_readiness_rows.csv'}`",
            f"- JSON: `{args.out_dir / 'zscape_scfm_latent_readiness_20260628.json'}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
