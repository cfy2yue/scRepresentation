#!/usr/bin/env python3
"""Summarize LatentFM condition metadata and condition-cache coverage.

This is intentionally lightweight: it reads only ``condition_metadata.json``
and, when provided, GeneEmbeddingCache / DrugEmbeddingCache indices. It never
opens large HDF5 embedding matrices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from model.condition_emb.chempert.chem_resolver import _keys_from_chem_source
from model.condition_emb.chempert.drug_cache import DrugEmbeddingCache
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


def _chem_keys(entry: Mapping[str, Any]) -> list[str]:
    obs = _clean_str(entry.get("chem_obs_value"))
    if obs:
        return [obs]
    src = _clean_str(entry.get("chem_source"))
    if src:
        return _keys_from_chem_source(src)
    return []


def _type(entry: Mapping[str, Any]) -> str:
    return _clean_str(entry.get("perturbation_type_raw", entry.get("perturbation_type"))) or "NA"


def _genes(entry: Mapping[str, Any]) -> Sequence[Any]:
    g = entry.get("genes")
    return g if isinstance(g, list) else []


def _load_drug_cache(cache_dir: str) -> DrugEmbeddingCache | None:
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"drug cache directory not found: {root}")
    return DrugEmbeddingCache(root)


def _load_gene_cache(cache_dir: str) -> GeneEmbeddingCache | None:
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"gene cache directory not found: {root}")
    return GeneEmbeddingCache(root)


def _gene_cache_hit(cache: GeneEmbeddingCache, symbol: str) -> bool:
    idx = cache.lookup(str(symbol))
    return int(idx) not in {int(cache.pad_index), int(cache.unk_index)}


def summarize(
    data_dir: Path,
    *,
    gene_cache_dir: str = "",
    drug_cache_dir: str = "",
) -> Dict[str, Any]:
    meta_path = data_dir / "condition_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing condition metadata: {meta_path}")
    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{meta_path} must contain a JSON object")

    gene_cache = _load_gene_cache(gene_cache_dir)
    drug_cache = _load_drug_cache(drug_cache_dir)
    datasets: Dict[str, Any] = {}
    total = {
        "conditions": 0,
        "gene_conditions": 0,
        "gene_cache_hits": 0,
        "gene_cache_lookups": 0,
        "chem_conditions": 0,
        "chem_cache_hits": 0,
        "chem_cache_lookups": 0,
        "gene_empty": 0,
    }

    for ds_name, ds_obj in sorted(obj.items()):
        if not isinstance(ds_obj, dict):
            continue
        type_counts: Dict[str, int] = {}
        chem_conditions = 0
        chem_keys = 0
        chem_hits = 0
        gene_empty = 0
        gene_conditions = 0
        gene_keys = 0
        gene_hits = 0
        multi_gene = 0

        for entry in ds_obj.values():
            if not isinstance(entry, dict):
                continue
            typ = _type(entry)
            type_counts[typ] = type_counts.get(typ, 0) + 1
            genes = list(_genes(entry))
            if not genes:
                gene_empty += 1
            else:
                gene_conditions += 1
                gene_keys += len(genes)
                if len(genes) > 1:
                    multi_gene += 1
                if gene_cache is not None:
                    for gene in genes:
                        gene_hits += int(_gene_cache_hit(gene_cache, str(gene)))
            keys = _chem_keys(entry)
            if keys:
                chem_conditions += 1
                chem_keys += len(keys)
                if drug_cache is not None:
                    for key in keys:
                        _, hit = drug_cache.lookup(str(key))
                        chem_hits += int(bool(hit))

        n = len(ds_obj)
        total["conditions"] += n
        total["gene_conditions"] += gene_conditions
        total["gene_cache_hits"] += gene_hits
        total["gene_cache_lookups"] += gene_keys if gene_cache is not None else 0
        total["chem_conditions"] += chem_conditions
        total["chem_cache_hits"] += chem_hits
        total["chem_cache_lookups"] += chem_keys if drug_cache is not None else 0
        total["gene_empty"] += gene_empty
        datasets[ds_name] = {
            "conditions": n,
            "type_counts": dict(sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "gene_empty": gene_empty,
            "multi_gene": multi_gene,
            "gene_conditions": gene_conditions,
            "gene_key_lookups": gene_keys,
            "gene_cache_hits": gene_hits if gene_cache is not None else None,
            "gene_cache_hit_rate": (
                None if gene_cache is None or gene_keys == 0 else float(gene_hits / gene_keys)
            ),
            "chem_conditions": chem_conditions,
            "chem_key_lookups": chem_keys,
            "chem_cache_hits": chem_hits if drug_cache is not None else None,
            "chem_cache_hit_rate": (
                None if drug_cache is None or chem_keys == 0 else float(chem_hits / chem_keys)
            ),
        }

    return {
        "data_dir": str(data_dir),
        "gene_cache_dir": gene_cache_dir or None,
        "gene_cache_checked": gene_cache is not None,
        "drug_cache_dir": drug_cache_dir or None,
        "drug_cache_checked": drug_cache is not None,
        "total": total,
        "datasets": datasets,
    }


def _print_table(summary: Mapping[str, Any]) -> None:
    print(json.dumps(summary["total"], indent=2, ensure_ascii=False))
    print(
        "\ndataset\tconditions\ttypes\tgene_empty\tmulti_gene\t"
        "gene_conditions\tgene_hit_rate\tchem_conditions\tchem_hit_rate"
    )
    for ds, row in summary["datasets"].items():
        types = ",".join(f"{k}:{v}" for k, v in row["type_counts"].items())
        gene_hit_rate = row["gene_cache_hit_rate"]
        chem_hit_rate = row["chem_cache_hit_rate"]
        gene_hit_s = "NA" if gene_hit_rate is None else f"{gene_hit_rate:.3f}"
        chem_hit_s = "NA" if chem_hit_rate is None else f"{chem_hit_rate:.3f}"
        print(
            f"{ds}\t{row['conditions']}\t{types}\t{row['gene_empty']}\t"
            f"{row['multi_gene']}\t{row['gene_conditions']}\t{gene_hit_s}\t"
            f"{row['chem_conditions']}\t{chem_hit_s}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--gene-cache-dir", default="")
    ap.add_argument("--drug-cache-dir", default="")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    summary = summarize(
        args.data_dir.expanduser().resolve(),
        gene_cache_dir=args.gene_cache_dir,
        drug_cache_dir=args.drug_cache_dir,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
