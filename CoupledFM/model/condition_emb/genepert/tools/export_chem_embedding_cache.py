#!/usr/bin/env python3
"""Export ``embeddings.npy`` + ``index.tsv`` + ``manifest.json`` for :class:`~condition_emb.genepert.chem_embedding_hook.ChemEmbeddingCache`.

Training / dataset code resolves vectors via :func:`~condition_emb.genepert.chem_embedding_hook.resolve_chem_embedding`
when ``chem_emb_source_dir`` points at the output directory.

Formats:

* ``passthrough_dict`` — load ``dict[str, Sequence[float]]`` (JSON or pickle) and materialize the cache.
  Use this for UniMol / MolFormer / ad-hoc numpy outputs once you have vectors in memory or in a dict file.

* ``unimol`` — **not implemented**; add a ``--checkpoint`` / batch inference path here when UniMol is integrated.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Union

import numpy as np


def _ensure_repo_root_on_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_mapping(path: Path) -> Dict[str, Any]:
    suf = path.suffix.lower()
    if suf in (".json", ".jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    elif suf in (".pkl", ".pickle"):
        with path.open("rb") as fh:
            obj = pickle.load(fh)
    else:
        try:
            with path.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except json.JSONDecodeError:
            with path.open("rb") as fh:
                obj = pickle.load(fh)
    if not isinstance(obj, MutableMapping):
        raise TypeError(f"expected dict-like mapping, got {type(obj)}")
    return dict(obj)


def export_passthrough_dict(
    data: Mapping[str, Union[Sequence[float], Any]],
    out_dir: Path,
    *,
    pad_index: int = 0,
) -> None:
    """Write cache: row ``pad_index`` is zero-pad; keys map to subsequent rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data:
        raise ValueError("empty dict")
    keys = sorted(data.keys(), key=lambda k: str(k))
    first = np.asarray(data[keys[0]], dtype=np.float64).reshape(-1)
    dim = int(first.size)
    if dim == 0:
        raise ValueError("embedding dimension must be positive")
    rows: list[np.ndarray] = [np.zeros((dim,), dtype=np.float32)]
    lines = ["key\tindex"]
    for i, k in enumerate(keys, start=1):
        arr = np.asarray(data[k], dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"key {k!r}: length {arr.size} != {dim}")
        rows.append(arr.copy())
        sk = str(k).strip()
        lines.append(f"{sk}\t{i}")
    mat = np.stack(rows, axis=0)
    np.save(out_dir / "embeddings.npy", mat)
    (out_dir / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "embed_kind": "molecule",
        "format": "passthrough_dict",
        "embed_dim": dim,
        "num_rows": int(mat.shape[0]),
        "pad_index": int(pad_index),
        "num_keys": len(keys),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")


def export_unimol(_args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "UniMol export is not implemented yet. Insert inference here: load UniMol checkpoint, "
        "encode SMILES strings, then call export_passthrough_dict(smiles_to_vec, out_dir). "
        "Alternatively, dump smiles→vector in Python and use --format passthrough_dict."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=("passthrough_dict", "unimol"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSON or pickle dict for passthrough_dict (key → list of floats).",
    )
    args = p.parse_args()
    _ensure_repo_root_on_path()
    if args.format == "unimol":
        export_unimol(args)
        return
    if args.input is None:
        raise SystemExit("--input is required for passthrough_dict")
    inp = Path(args.input).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")
    data = _load_mapping(inp)
    export_passthrough_dict(data, Path(args.out_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
