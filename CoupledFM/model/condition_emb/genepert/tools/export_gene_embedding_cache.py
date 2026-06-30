#!/usr/bin/env python3
"""Export ``gene_embeddings.npy`` + ``gene_index.tsv`` + ``manifest.json`` for :class:`~condition_emb.genepert.gene_cache.GeneEmbeddingCache`.

Primary input: UCE / State style ``gene_symbol_to_embedding_ESM2.pt`` — a ``dict`` mapping
gene symbol → 1-D float tensor / ndarray.

Training with PerturbationConditionEncoder (``latent``, ``coupled``, ``raw_independent``):

    Set YAML / ``DataConfig.pert_gene_emb_cache_dir`` (or latent ``Config``) to the
    directory containing ``gene_embeddings.npy``, ``gene_index.tsv``, and ``manifest.json``
    emitted by this tool. When ``use_pert_condition`` is True and the encoder uses
    ``pretrained_*`` modes, loaders expect :class:`~condition_emb.genepert.gene_cache.GeneEmbeddingCache`
    at that path — no runtime ``torch.load`` of the megabyte-sized ``.pt``.

Example (full genome ESM2 table)::

    cd /path/to/CoupledFM && PYTHONPATH=. python condition_emb/genepert/tools/export_gene_embedding_cache.py \\
      --format esm2_pt \\
      --out-dir ./pretrainckpt/genepert_cache/full_human_esm2

Example (restricted to CellNavi ``gene_name.txt``)::

    cd /path/to/CoupledFM && PYTHONPATH=. python condition_emb/genepert/tools/export_gene_embedding_cache.py \\
      --format cellnavi_ckpt \\
      --ckpt-path pretrainckpt/cellnavi/data/pretrain/pretrain_weights.pth \\
      --gene-name-path pretrainckpt/cellnavi/data/gene_name.txt \\
      --restrict-genes ./pretrainckpt/cellnavi/data/gene_name.txt \\
      --out-dir ./pretrainckpt/genepert_cache/cellnavi_embed_gene

``--input`` is optional if ``UCE_ESM2_PT`` points to a Homo sapiens (or custom) checkpoint, else
defaults to ``pretrainckpt/uce/model_files/protein_embeddings/Homo_sapiens.GRCh38.*.pt``.

``--format scgpt_ckpt`` exports rows of the scGPT :class:`GeneEncoder` embedding table
(``encoder.embedding.weight``) keyed by vocabulary token → HGNC symbol.

``--format cellnavi_ckpt`` exports CellNavi ``embed_gene`` rows (see :func:`export_from_cellnavi_ckpt`).

``--format uce_all_tokens`` is reserved — use ``uce_all_tokens`` when implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _repo_root_from_here() -> Path:
    """CoupledFM repository root from this file (under ``model/condition_emb/genepert/tools``)."""
    return Path(__file__).resolve().parents[4]


def _ensure_repo_root_on_path() -> Path:
    root = _repo_root_from_here()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _default_human_esm2_pt(repo_root: Path) -> Path:
    return (
        repo_root
        / "pretrainckpt/uce/model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
    )


def _default_cellnavi_gene_name(repo_root: Path) -> Path:
    return repo_root / "pretrainckpt/cellnavi/data/gene_name.txt"


def _default_scgpt_vocab(repo_root: Path) -> Path:
    return repo_root / "pretrainckpt/scgpt/vocab.json"


def resolve_esm2_pt_path(cli_input: Optional[Path]) -> Tuple[Path, str]:
    """Return (existing path, label describing resolution). Raises SystemExit if missing."""
    repo = _repo_root_from_here()
    env = os.environ.get("UCE_ESM2_PT", "").strip()

    if cli_input is not None and str(cli_input).strip():
        p = cli_input.expanduser().resolve()
        label = "--input"
    elif env:
        p = Path(env).expanduser().resolve()
        label = "UCE_ESM2_PT"
    else:
        p = _default_human_esm2_pt(repo).resolve()
        label = "default_human_GRCh38"

    if not p.is_file():
        msg = (
            f"Gene embedding checkpoint not found: {p}\n"
            "  Pass --input PATH, set UCE_ESM2_PT, or place Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt under\n"
            f"  {repo}/pretrainckpt/uce/model_files/protein_embeddings/\n"
        )
        raise SystemExit(msg)

    return p, label


def load_restrict_ordered_unique_symbols(path: Path) -> Tuple[List[str], int]:
    """Load gene symbols from a newline list or JSON; dedupe preserving order.

    Symbols are normalized with :func:`~condition_emb.genepert.perturbation.normalize_gene_symbol`.
    Returns (canonical symbols, raw non-empty token count before dedupe).
    """
    _ensure_repo_root_on_path()
    from model.condition_emb.genepert.perturbation import normalize_gene_symbol

    raw_path = path.expanduser().resolve()
    txt = raw_path.read_text(encoding="utf-8")
    tokens_in: List[str] = []

    if raw_path.suffix.lower() == ".json":
        try:
            data = json.loads(txt)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON restrict file {raw_path}: {e}") from e
        if isinstance(data, list):
            iterable: Iterable[Any] = data
        elif isinstance(data, dict):
            inner = None
            for key in ("genes", "symbols", "gene_symbols"):
                if key in data:
                    inner = data[key]
                    break
            if inner is None:
                inner = next((v for v in data.values() if isinstance(v, list)), [])
            iterable = inner if isinstance(inner, list) else []
        else:
            iterable = []
        for obj in iterable:
            s = str(obj).strip() if obj is not None else ""
            if s:
                tokens_in.append(normalize_gene_symbol(s))
    else:
        for ln in txt.splitlines():
            ln_st = ln.split("#", 1)[0].strip()
            if not ln_st:
                continue
            tokens_in.append(normalize_gene_symbol(ln_st))

    raw_nonempty = len(tokens_in)
    seen: Dict[str, None] = {}
    ordered_unique: List[str] = []
    for t in tokens_in:
        if not t or t in {"NAN", "NONE"}:
            continue
        if t not in seen:
            seen[t] = None
            ordered_unique.append(t)

    return ordered_unique, raw_nonempty


def _load_pt_dictionary(src: Path) -> Tuple[Dict[str, np.ndarray], int, int, int]:
    _ensure_repo_root_on_path()
    import torch

    from model.condition_emb.genepert.perturbation import normalize_gene_symbol

    obj = torch.load(str(src), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict in {src}, got {type(obj)}")

    merged: Dict[str, np.ndarray] = {}
    dup = 0
    bad_shape = 0
    embed_dim: int | None = None

    for k, v in obj.items():
        sym = normalize_gene_symbol(str(k))
        if not sym or sym in {"NAN", "NONE"}:
            continue
        if isinstance(v, torch.Tensor):
            vec = v.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            vec = np.asarray(v, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            bad_shape += 1
            continue
        if embed_dim is None:
            embed_dim = int(vec.shape[0])
        elif vec.shape[0] != embed_dim:
            bad_shape += 1
            continue
        if sym in merged:
            dup += 1
            continue
        merged[sym] = vec

    if embed_dim is None or not merged:
        raise ValueError(f"no usable embeddings found in {src} (bad_shape={bad_shape})")

    return merged, int(embed_dim), dup, bad_shape


def export_from_esm2_pt(
    src: Path,
    out_dir: Path,
    *,
    unk_strategy: str = "zeros",
    restrict_genes_path: Optional[Path] = None,
) -> None:
    merged, embed_dim, dup, bad_shape = _load_pt_dictionary(src)

    restrict_source_resolved: Optional[str] = None
    num_vocab_genes: int = len(merged)
    symbols_sorted: List[str]

    missing_symbols: List[str] = []
    if restrict_genes_path is not None:
        restrict_tokens, _ = load_restrict_ordered_unique_symbols(restrict_genes_path)
        restrict_source_resolved = str(Path(restrict_genes_path).expanduser().resolve())
        num_vocab_genes = len(restrict_tokens)
        symbols_sorted = [s for s in restrict_tokens if s in merged]
        num_found = len(symbols_sorted)
        num_missing = num_vocab_genes - num_found
        missing_symbols = [s for s in restrict_tokens if s not in merged]
    else:
        num_found = len(merged)
        num_missing = 0
        symbols_sorted = sorted(merged.keys())

    if not symbols_sorted:
        raise ValueError("after restriction, no gene embeddings remain to export")

    manifest_extra: Dict[str, Any] = {
        "source": "gene_symbol_to_embedding_ESM2_pt",
        "source_model": "UCE protein_embeddings dict (ESM2-derived)",
        "source_semantics": "Precomputed dict gene_symbol -> 1D embedding vector; not online ESM2 inference",
        "source_path": str(src.resolve()),
        "restrict_source": restrict_source_resolved,
        "num_vocab_genes": int(num_vocab_genes),
        "num_found": int(num_found),
        "num_missing": int(num_missing),
        "missing_symbols_count": int(len(missing_symbols)),
        "missing_symbols": missing_symbols[:2048],
        "stats": {"duplicate_symbols_skipped": dup, "bad_or_mismatched_vectors": bad_shape},
    }
    sub_merged = {s: merged[s] for s in symbols_sorted}
    _emit_gene_cache_matrix(
        symbols_sorted, sub_merged, embed_dim, out_dir, unk_strategy, manifest_extra
    )


def export_from_uce_all_tokens(_src: Path, _out_dir: Path, **_kw: Any) -> None:
    raise NotImplementedError(
        "--format uce_all_tokens requires loading all_tokens.torch plus protein embeddings; "
        "not wired in CoupledFM yet. Use --format esm2_pt."
    )


def export_scgpt_reserved(_src: Path, _out_dir: Path) -> None:
    raise NotImplementedError(
        "Use --format scgpt_ckpt with --ckpt-path and --vocab-path "
        "(or set SCGPT_CKPT / SCGPT_VOCAB_JSON)."
    )


def export_from_scgpt_ckpt(
    ckpt_path: Path,
    vocab_path: Path,
    out_dir: Path,
    *,
    unk_strategy: str = "zeros",
    restrict_genes_path: Optional[Path] = None,
) -> None:
    """Export scGPT ``encoder.embedding.weight`` rows keyed by gene symbol from ``vocab.json``."""
    _ensure_repo_root_on_path()
    import torch

    from model.condition_emb.genepert.perturbation import normalize_gene_symbol

    ckpt = torch.load(str(ckpt_path.expanduser().resolve()), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if not isinstance(state, dict):
        raise ValueError("checkpoint must expose a dict state_dict (or be a flat state dict)")

    w_key = None
    for preferred in ("encoder.embedding.weight", "module.encoder.embedding.weight"):
        if preferred in state:
            w_key = preferred
            break
    if w_key is None:
        cands = [k for k in state if k.endswith("encoder.embedding.weight")]
        if cands:
            w_key = sorted(cands)[0]
    if w_key is None:
        raise ValueError(
            "no encoder.embedding.weight in checkpoint; "
            f"sample keys: {list(state.keys())[:30]}"
        )

    W = state[w_key].detach().cpu().float().numpy()
    if W.ndim != 2:
        raise ValueError(f"expected 2D embedding weights, got {W.shape}")
    embed_dim = int(W.shape[1])
    n_vocab_ckpt = int(W.shape[0])

    raw_vocab = json.loads(Path(vocab_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(raw_vocab, dict):
        raise ValueError(f"vocab JSON must be an object token->id, got {type(raw_vocab)}")

    merged: Dict[str, np.ndarray] = {}
    for token, idx in raw_vocab.items():
        t = str(token)
        if t.startswith("<"):
            continue
        sym = normalize_gene_symbol(t)
        if not sym or sym in {"NAN", "NONE"}:
            continue
        ii = int(idx)
        if ii < 0 or ii >= n_vocab_ckpt:
            continue
        if sym not in merged:
            merged[sym] = W[ii].astype(np.float32)

    restrict_source_resolved: Optional[str] = None
    num_vocab_genes: int = len(merged)
    missing_symbols: List[str] = []

    if restrict_genes_path is not None:
        restrict_tokens, _ = load_restrict_ordered_unique_symbols(restrict_genes_path)
        restrict_source_resolved = str(Path(restrict_genes_path).expanduser().resolve())
        num_vocab_genes = len(restrict_tokens)
        symbols_sorted = [s for s in restrict_tokens if s in merged]
        num_found = len(symbols_sorted)
        num_missing = num_vocab_genes - num_found
        missing_symbols = [s for s in restrict_tokens if s not in merged]
    else:
        num_found = len(merged)
        num_missing = 0
        symbols_sorted = sorted(merged.keys())

    if not symbols_sorted:
        raise ValueError("after restriction, no gene embeddings remain to export")

    manifest_extra: Dict[str, Any] = {
        "source": "scgpt_encoder_embedding",
        "source_model": "scGPT GeneEncoder embedding table",
        "source_semantics": "Vocabulary token rows from encoder.embedding.weight; token id from vocab.json",
        "source_path": str(Path(ckpt_path).resolve()),
        "vocab_path": str(Path(vocab_path).resolve()),
        "state_dict_key": w_key,
        "restrict_source": restrict_source_resolved,
        "num_vocab_genes": int(num_vocab_genes),
        "num_found": int(num_found),
        "num_missing": int(num_missing),
        "vocab_tokens_used": int(len(merged)),
        "missing_symbols_count": int(len(missing_symbols)),
        "missing_symbols": missing_symbols[:2048],
    }
    sub_merged = {s: merged[s] for s in symbols_sorted}
    _emit_gene_cache_matrix(
        symbols_sorted, sub_merged, embed_dim, out_dir, unk_strategy, manifest_extra
    )


def _emit_gene_cache_matrix(
    symbols_sorted: List[str],
    merged: Dict[str, np.ndarray],
    embed_dim: int,
    out_dir: Path,
    unk_strategy: str,
    manifest_extra: Dict[str, Any],
) -> None:
    """Write gene_embeddings.npy, gene_index.tsv, manifest.json and validate."""
    if not symbols_sorted:
        raise ValueError("after restriction, no gene embeddings remain to export")
    ngenes = len(symbols_sorted)
    pad_index = 0
    unk_index = 1
    n_rows = 2 + ngenes
    mat = np.zeros((n_rows, embed_dim), dtype=np.float32)
    if unk_strategy == "mean":
        stacked = np.stack([merged[s] for s in symbols_sorted], axis=0)
        mat[unk_index] = stacked.mean(axis=0).astype(np.float32)
    elif unk_strategy != "zeros":
        raise ValueError("unk_strategy must be 'zeros' or 'mean'")

    lines = ["#\tgene_symbol\tindex", f"PAD\t{pad_index}", f"UNK\t{unk_index}"]
    for i, sym in enumerate(symbols_sorted):
        row_ix = i + 2
        mat[row_ix] = merged[sym]
        lines.append(f"{sym}\t{row_ix}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "gene_embeddings.npy"), mat)
    (out_dir / "gene_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    base_manifest: Dict[str, Any] = {
        "format_version": 1,
        "embed_dim": embed_dim,
        "num_symbols": int(ngenes),
        "num_rows": n_rows,
        "pad_index": pad_index,
        "unk_index": unk_index,
        "unk_strategy": unk_strategy,
    }
    base_manifest.update(manifest_extra)
    (out_dir / "manifest.json").write_text(json.dumps(base_manifest, indent=2), encoding="ascii")

    from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache

    cache = GeneEmbeddingCache(out_dir)
    cache.validate_index_bounds()


def export_from_cellnavi_ckpt(
    ckpt_path: Path,
    gene_name_path: Path,
    out_dir: Path,
    *,
    unk_strategy: str = "zeros",
    restrict_genes_path: Optional[Path] = None,
) -> None:
    """Export rows of CellNavi ``embed_gene.0`` (``nn.Embedding(40002, d_model)``) by gene symbol."""
    _ensure_repo_root_on_path()
    import torch

    from model.condition_emb.genepert.perturbation import normalize_gene_symbol

    ckpt = torch.load(str(ckpt_path.expanduser().resolve()), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must expose a dict state_dict (or be a flat state dict)")

    w_key = None
    candidates = [k for k in state if "embed_gene" in k and "weight" in k]
    for preferred in ("embed_gene.0.weight", "module.embed_gene.0.weight"):
        if preferred in state:
            w_key = preferred
            break
    if w_key is None and candidates:
        w_key = sorted(candidates)[0]
    if w_key is None:
        raise ValueError(
            "no embed_gene.*.weight in checkpoint; "
            f"sample keys: {list(state.keys())[:25]}"
        )

    W = state[w_key].detach().cpu().float().numpy()
    if W.ndim != 2:
        raise ValueError(f"expected 2D embedding weights, got {W.shape}")
    embed_dim = int(W.shape[1])
    max_tid = int(W.shape[0])

    raw_lines = gene_name_path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    symbols_ordered: List[str] = []
    for s in raw_lines:
        st = s.split("#", 1)[0].strip()
        if not st:
            continue
        symbols_ordered.append(normalize_gene_symbol(st))

    merged: Dict[str, np.ndarray] = {}
    for i, sym in enumerate(symbols_ordered):
        if not sym or sym in {"NAN", "NONE"}:
            continue
        if i >= max_tid:
            break
        if sym not in merged:
            merged[sym] = W[i].astype(np.float32)

    restrict_source_resolved: Optional[str] = None
    num_vocab_genes: int = len(merged)

    missing_symbols: List[str] = []
    if restrict_genes_path is not None:
        restrict_tokens, _ = load_restrict_ordered_unique_symbols(restrict_genes_path)
        restrict_source_resolved = str(Path(restrict_genes_path).expanduser().resolve())
        num_vocab_genes = len(restrict_tokens)
        symbols_sorted = [s for s in restrict_tokens if s in merged]
        num_found = len(symbols_sorted)
        num_missing = num_vocab_genes - num_found
        missing_symbols = [s for s in restrict_tokens if s not in merged]
    else:
        num_found = len(merged)
        num_missing = 0
        symbols_sorted = sorted(merged.keys())

    if not symbols_sorted:
        raise ValueError("after restriction, no gene embeddings remain to export")

    sub_merged_cn = {s: merged[s] for s in symbols_sorted}
    _emit_gene_cache_matrix(
        symbols_sorted,
        sub_merged_cn,
        embed_dim,
        out_dir,
        unk_strategy,
        {
            "source": "cellnavi_embed_gene",
            "source_model": "SparseCellNaviEncoder / CellNavi pretrain",
            "source_semantics": "Gene token rows from embed_gene.nn.Embedding; gene_name.txt line index = token id",
            "source_path": str(Path(ckpt_path).resolve()),
            "gene_name_source": str(Path(gene_name_path).resolve()),
            "state_dict_key": w_key,
            "restrict_source": restrict_source_resolved,
            "num_vocab_genes": int(num_vocab_genes),
            "num_found": int(num_found),
            "num_missing": int(num_missing),
            "missing_symbols_count": int(len(missing_symbols)),
            "missing_symbols": missing_symbols[:2048],
            "max_token_rows_in_ckpt": max_tid,
        },
    )


def main(argv: Tuple[str, ...] | None = None) -> None:
    repo = _repo_root_from_here()
    ap = argparse.ArgumentParser(description="Export GeneEmbeddingCache bundle from pretrained tensors.")
    ap.add_argument(
        "--format",
        choices=("esm2_pt", "scgpt_ckpt", "cellnavi_ckpt", "scgpt", "uce_all_tokens"),
        default="esm2_pt",
        help="scgpt is deprecated alias that errors with migration hint",
    )
    ap.add_argument("--input", type=Path, default=None, help="esm2_pt: optional path to .pt dict")
    ap.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="cellnavi_ckpt / scgpt_ckpt: checkpoint .pt/.pth",
    )
    ap.add_argument(
        "--gene-name-path",
        type=Path,
        default=None,
        help="cellnavi_ckpt: gene_name.txt (default: pretrainckpt/cellnavi/data/gene_name.txt)",
    )
    ap.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        help="scgpt_ckpt: vocab.json (default: pretrainckpt/scgpt/vocab.json)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--unk-strategy", choices=("zeros", "mean"), default="zeros")
    ap.add_argument(
        "--restrict-genes",
        type=Path,
        default=None,
        help="newline list or JSON list of HGNC symbols; export only intersections with the checkpoint",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.format == "scgpt":
        export_scgpt_reserved(Path("."), args.out_dir)
    elif args.format == "uce_all_tokens":
        src, _ = resolve_esm2_pt_path(args.input)
        export_from_uce_all_tokens(src, args.out_dir)
    elif args.format == "cellnavi_ckpt":
        ckpt = args.ckpt_path
        if ckpt is None or not str(ckpt).strip():
            env_ck = os.environ.get("CELLNAVI_CKPT", "").strip()
            ckpt = Path(env_ck) if env_ck else None
        if ckpt is None or not ckpt.expanduser().is_file():
            raise SystemExit(
                "--format cellnavi_ckpt requires --ckpt-path or CELLNAVI_CKPT pointing to a CellNavi .pth"
            )
        gnp = args.gene_name_path
        if gnp is None or not str(gnp).strip():
            gnp = _default_cellnavi_gene_name(repo)
        if not gnp.expanduser().is_file():
            raise SystemExit(f"cellnavi gene list not found: {gnp}")
        export_from_cellnavi_ckpt(
            ckpt.expanduser().resolve(),
            gnp.expanduser().resolve(),
            args.out_dir,
            unk_strategy=args.unk_strategy,
            restrict_genes_path=args.restrict_genes,
        )
    elif args.format == "scgpt_ckpt":
        ckpt = args.ckpt_path
        if ckpt is None or not str(ckpt).strip():
            env_ck = os.environ.get("SCGPT_CKPT", "").strip()
            ckpt = Path(env_ck) if env_ck else None
        if ckpt is None or not ckpt.expanduser().is_file():
            raise SystemExit(
                "--format scgpt_ckpt requires --ckpt-path or SCGPT_CKPT pointing to an scGPT checkpoint"
            )
        voc = args.vocab_path
        if voc is None or not str(voc).strip():
            env_v = os.environ.get("SCGPT_VOCAB_JSON", "").strip()
            voc = Path(env_v) if env_v else _default_scgpt_vocab(repo)
        if not Path(voc).expanduser().is_file():
            raise SystemExit(f"scGPT vocab JSON not found: {voc}")
        export_from_scgpt_ckpt(
            ckpt.expanduser().resolve(),
            Path(voc).expanduser().resolve(),
            args.out_dir,
            unk_strategy=args.unk_strategy,
            restrict_genes_path=args.restrict_genes,
        )
    else:
        src_p, resolved_via = resolve_esm2_pt_path(args.input)
        export_from_esm2_pt(
            src_p,
            args.out_dir,
            unk_strategy=args.unk_strategy,
            restrict_genes_path=args.restrict_genes,
        )
        mpath = Path(args.out_dir) / "manifest.json"
        m = json.loads(mpath.read_text(encoding="ascii"))
        nv = int(m["num_vocab_genes"])
        nf = int(m["num_found"])
        nm = int(m["num_missing"])
        pct = (100.0 * nf / nv) if nv else 0.0
        oov_pct = (100.0 * nm / nv) if nv else 0.0
        print(
            json.dumps(
                {
                    "source_resolved_via": resolved_via,
                    "esm2_pt": str(src_p),
                    "out_dir": str(Path(args.out_dir).resolve()),
                    "num_vocab_genes": nv,
                    "num_found_esm2": nf,
                    "num_oov": nm,
                    "hit_rate": round(pct, 4),
                    "oov_rate": round(oov_pct, 4),
                    "restrict_source": m.get("restrict_source"),
                    "embed_dim": m.get("embed_dim"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
