#!/usr/bin/env python3
"""Export drug labels from LatentFM condition metadata as a DrugEmbeddingCache.

This is a pragmatic in-distribution chemical condition source when no molecular
or State tx perturbation-vector cache is available. It creates one row per
chemical metadata key, with PAD/UNK rows reserved at indices 0/1. By default the
embedding dimension equals the number of keys, so rows form a label identity
matrix. This is not a molecular encoder and must not be used for unseen-drug
zero-shot claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from model.condition_emb.chempert.chem_resolver import _keys_from_chem_source


def _clean(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


def _chem_keys(entry: Mapping[str, Any]) -> list[str]:
    obs = _clean(entry.get("chem_obs_value"))
    if obs:
        return [obs]
    src = _clean(entry.get("chem_source"))
    if src:
        return _keys_from_chem_source(src)
    return []


def collect_keys(condition_metadata: Path) -> list[str]:
    obj = json.loads(condition_metadata.read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise TypeError(f"{condition_metadata} must contain a JSON object")
    keys: set[str] = set()
    for ds_obj in obj.values():
        if not isinstance(ds_obj, Mapping):
            continue
        for entry in ds_obj.values():
            if not isinstance(entry, Mapping):
                continue
            for key in _chem_keys(entry):
                kk = _clean(key)
                if kk:
                    keys.add(kk)
    return sorted(keys)


def export_label_identity_cache(keys: Iterable[str], out_dir: Path, *, embed_dim: int = 0) -> dict[str, Any]:
    labels = sorted({str(k).strip() for k in keys if str(k).strip()})
    if not labels:
        raise ValueError("no chemical labels found")
    dim = int(embed_dim) if int(embed_dim) > 0 else len(labels)
    if dim < len(labels):
        raise ValueError(f"embed_dim={dim} is smaller than number of labels={len(labels)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mat = np.zeros((len(labels) + 2, dim), dtype=np.float32)
    for i in range(len(labels)):
        mat[i + 2, i] = 1.0

    index = {label: i for i, label in enumerate(labels, start=2)}
    np.save(out_dir / "drug_embeddings.npy", mat)
    np.save(out_dir / "embeddings.npy", mat)

    lines = ["key\tindex", *[f"{label}\t{idx}" for label, idx in index.items()]]
    (out_dir / "drug_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "drug_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "embed_kind": "drug_label_identity",
        "source": "LatentFM condition_metadata chem keys",
        "embed_dim": dim,
        "num_rows": int(mat.shape[0]),
        "num_keys": len(labels),
        "pad_index": 0,
        "unk_index": 1,
        "warning": (
            "Label identity vectors are useful for in-distribution chemical conditions only. "
            "They are not molecular encodings and do not support unseen-drug zero-shot claims."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition-metadata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Embedding dimension. Default 0 uses exact label-identity dimension.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    keys = collect_keys(args.condition_metadata.expanduser().resolve())
    manifest = export_label_identity_cache(keys, args.out_dir.expanduser().resolve(), embed_dim=args.embed_dim)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
