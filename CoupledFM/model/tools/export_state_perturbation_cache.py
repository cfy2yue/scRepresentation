#!/usr/bin/env python3
"""Export a State tx ``pert_onehot_map.pt`` as a CoupledFM drug embedding cache.

State transition runs store perturbation labels in ``pert_onehot_map.pt``.  For
chemical perturbation datasets this map can be used as a pragmatic short-term
drug condition embedding source for LatentFM.  It is label-based, not a SMILES
foundation encoder, so it should not be used to claim zero-shot generalization to
unseen compounds.

Output layout matches ``DrugEmbeddingCache``:

    drug_embeddings.npy
    drug_index.tsv
    drug_index.json
    manifest.json

For legacy callers, ``embeddings.npy`` and ``index.tsv`` are written too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _as_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty perturbation vector")
    return arr


def _load_map(path: Path) -> Mapping[Any, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, Mapping):
        raise TypeError(f"expected mapping in {path}, got {type(obj)}")
    return obj


def export_state_perturbation_cache(
    pert_onehot_map: Mapping[Any, Any],
    out_dir: Path,
    *,
    drop_control: str = "",
) -> dict:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    drop = str(drop_control or "").strip()
    for key in sorted(pert_onehot_map.keys(), key=lambda x: str(x)):
        label = str(key).strip()
        if not label:
            continue
        if drop and label == drop:
            continue
        vec = _as_vector(pert_onehot_map[key])
        rows.append(vec)
        labels.append(label)
    if not rows:
        raise ValueError("no perturbation vectors to export")
    dim = int(rows[0].size)
    for label, vec in zip(labels, rows):
        if int(vec.size) != dim:
            raise ValueError(f"{label!r}: dim {vec.size} != {dim}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Row 0 is PAD and row 1 is UNK for DrugEmbeddingCache. Misses still use a
    # deterministic Gaussian fallback, so UNK is just a reserved placeholder.
    mat = np.zeros((len(rows) + 2, dim), dtype=np.float32)
    for i, vec in enumerate(rows, start=2):
        mat[i] = vec

    index = {label: i for i, label in enumerate(labels, start=2)}
    np.save(out_dir / "drug_embeddings.npy", mat)
    np.save(out_dir / "embeddings.npy", mat)
    lines = ["key\tindex", *[f"{label}\t{idx}" for label, idx in index.items()]]
    (out_dir / "drug_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "drug_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "embed_kind": "state_tx_perturbation_label",
        "source": "State tx pert_onehot_map.pt",
        "embed_dim": dim,
        "num_rows": int(mat.shape[0]),
        "num_keys": len(labels),
        "pad_index": 0,
        "unk_index": 1,
        "drop_control": drop or None,
        "warning": (
            "Label-based State perturbation vectors are useful for in-distribution chemical "
            "conditions but are not molecular encodings and do not support unseen-drug zero-shot claims."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pert-onehot-map", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--drop-control",
        default="",
        help="Optional exact control label to exclude from the exported cache.",
    )
    args = ap.parse_args()
    mp = _load_map(args.pert_onehot_map.expanduser().resolve())
    manifest = export_state_perturbation_cache(mp, args.out_dir.expanduser().resolve(), drop_control=args.drop_control)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
