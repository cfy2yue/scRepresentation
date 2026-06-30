"""
xVERSE cell embedding adapter.

Benchmark protocol: the encoder input is only ``adata.X`` (aligned to the model's
17999-gene ENSG list). No perturbation metadata is fed as a separate conditional
stream.

``CellEmbeddingbyGene`` pools over all ``num_genes`` with a learned per-gene
attention softmax. Unmeasured genes are ``-1``; any other value (including 0) is
treated as observed.

If ``obsm['pert_var_idx']`` is present and ``force_pert=True``, we only enforce
**observability / coverage** in the aligned value matrix: indices listed there
must map to finite, non-sentinel slots (filling 0.0 when needed so a gene is not
left as ``-1`` unmeasured). This is not a perturbation condition—only a
consistency check on ``X`` after alignment.

``tissue`` selects a fixed tissue embedding id required by the pretrained
``bio_encoder``; it is not per-cell perturbation metadata.

We do **not** mutate ``third_party/xVERSE_code``.
"""

from __future__ import annotations

import importlib.util
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import torch
import paths

from .._common import histogram_pert_kept




def _xverse_code_root() -> Path:
    return paths.third_party_root() / "xVERSE_code"


def _load_utils_model():
    """Import ``main.utils_model`` from ``third_party/xVERSE_code``.

    Also registers a ``main`` package shim so ``from main.utils_model import …``
    inside the third-party code continues to work without sys.path pollution.
    """
    root = _xverse_code_root()
    if not (root / "main" / "utils_model.py").is_file():
        raise FileNotFoundError(f"xVERSE source not found under {root}")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Isolate import to a private name to avoid clashing with other "main" pkgs.
    utils_path = root / "main" / "utils_model.py"
    spec = importlib.util.spec_from_file_location(
        "latent_bench_xverse_utils_model", utils_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Pre-register so any intra-module relative-ish imports resolve.
    sys.modules.setdefault("latent_bench_xverse_utils_model", mod)
    spec.loader.exec_module(mod)
    return mod


def _default_gene_ids_path() -> Path:
    return _xverse_code_root() / "main" / "ensg_keys_high_quality.txt"


def _load_gene_list(path: Path) -> List[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _tissue_name_to_id(name: str) -> int:
    csv = _xverse_code_root() / "main" / "tissue_name_to_id_map.csv"
    import pandas as pd

    df = pd.read_csv(csv)
    m = dict(
        zip(
            df["tissue_name"].astype(str).str.strip().str.lower(),
            df["tissue_id"].astype(int),
        )
    )
    key = str(name).strip().lower()
    if key not in m:
        raise KeyError(f"Tissue '{name}' not in xVERSE tissue map ({csv}).")
    return int(m[key])


def _clean_ensg(value: object) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    text = text.split(".")[0]
    return text if text.startswith("ENSG") else ""


def _load_symbol_to_ensembl_map() -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Build a conservative split-independent HGNC approved-symbol -> ENSG map."""
    mapping: Dict[str, str] = {}
    source_counts: Dict[str, int] = {}

    def add(symbol: object, ensg: object, source: str) -> None:
        key = str(symbol).strip().upper()
        val = _clean_ensg(ensg)
        if not key or not val:
            return
        if key not in mapping:
            mapping[key] = val
            source_counts[source] = source_counts.get(source, 0) + 1

    hgnc = paths.third_party_root() / "scFoundation" / "SCAD" / "data" / "processing" / "HGNC_symbol_all_genes.tsv"
    if hgnc.is_file():
        try:
            with open(hgnc, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    status = str(row.get("Status", "")).strip().lower()
                    if status and status != "approved":
                        continue
                    add(row.get("Approved symbol", ""), row.get("Ensembl gene ID", ""), "hgnc_approved_symbol")
        except Exception:
            pass

    return mapping, {
        "symbol_map_size": len(mapping),
        "symbol_map_sources": source_counts,
        "symbol_map_policy": "HGNC approved symbols only; aliases and previous symbols are not used",
    }


def _series_report(
    source: str,
    vals: Sequence[object],
    *,
    model_ensg: Optional[set[str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray([_clean_ensg(v) for v in vals], dtype=object)
    mapped = int(sum(1 for x in arr if x))
    aligned = int(sum(1 for x in arr if model_ensg is not None and x in model_ensg))
    report: dict[str, Any] = {
        "source": source,
        "n_vars": int(len(arr)),
        "mapped_ensg": mapped,
        "aligned_to_xverse": aligned if model_ensg is not None else None,
        "unique_aligned_to_xverse": (
            len({str(x) for x in arr if model_ensg is not None and x in model_ensg})
            if model_ensg is not None
            else None
        ),
    }
    if extra:
        report.update(extra)
    return arr, report


def _resolve_ensembl_series(
    adata: ad.AnnData,
    gene_col: Optional[str],
    model_ensg: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Return ENSG ids plus provenance; fall back from bad symbol-like columns."""
    model_set = (
        set(str(x).strip().upper().split(".")[0] for x in model_ensg)
        if model_ensg is not None
        else None
    )
    if gene_col is not None:
        if gene_col not in adata.var.columns:
            raise KeyError(f"gene_col '{gene_col}' not in adata.var")
        arr, report = _series_report(
            f"explicit_var_column:{gene_col}",
            adata.var[gene_col].astype(str).to_numpy(),
            model_ensg=model_set,
        )
        if report["mapped_ensg"] == 0:
            raise KeyError(f"gene_col '{gene_col}' contains no ENSG-like ids")
        return arr, report

    candidates = ["Ensembl_ID", "ENSEMBL", "ensemblid", "ensembl_id", "gene_ids", "gene_id", "ensembl"]
    attempted: list[dict[str, Any]] = []
    best_report: Optional[dict[str, Any]] = None
    for c in candidates:
        if c not in adata.var.columns:
            continue
        arr, report = _series_report(
            f"var_column:{c}",
            adata.var[c].astype(str).to_numpy(),
            model_ensg=model_set,
        )
        attempted.append(report)
        if report["mapped_ensg"] > 0 and (model_set is None or int(report.get("aligned_to_xverse") or 0) > 0):
            return arr, {**report, "attempted_sources": attempted}
        if best_report is None or int(report.get("aligned_to_xverse") or 0) > int(best_report.get("aligned_to_xverse") or 0):
            best_report = report

    idx = adata.var_names.astype(str).str.strip()
    var_arr, var_report = _series_report("var_names_ensg", idx.to_numpy(dtype=object), model_ensg=model_set)
    attempted.append(var_report)
    if var_report["mapped_ensg"] > 0 and (model_set is None or int(var_report.get("aligned_to_xverse") or 0) > 0):
        return var_arr, {**var_report, "attempted_sources": attempted}

    symbol_map, map_report = _load_symbol_to_ensembl_map()
    mapped_vals = []
    symbol_hits = 0
    for symbol in idx.to_numpy(dtype=object):
        ensg = symbol_map.get(str(symbol).strip().upper(), "")
        if ensg:
            symbol_hits += 1
        mapped_vals.append(ensg)
    sym_arr, sym_report = _series_report(
        "var_names_symbol_map",
        mapped_vals,
        model_ensg=model_set,
        extra={
            **map_report,
            "symbol_hits": int(symbol_hits),
            "attempted_sources": attempted,
        },
    )
    if sym_report["mapped_ensg"] > 0 and (model_set is None or int(sym_report.get("aligned_to_xverse") or 0) > 0):
        return sym_arr, sym_report

    detail = {
        "attempted_sources": attempted,
        "best_source": best_report,
        "symbol_report": sym_report,
    }
    raise KeyError(
        "Could not resolve usable Ensembl ids from adata.var columns, ENSG var_names, "
        f"or local symbol maps. Detail: {detail}"
    )


def _build_alignment(
    model_ensg: Sequence[str], adata_ensg: np.ndarray
) -> np.ndarray:
    """Return ``model_to_adata[i] = adata column index or -1``, length = len(model_ensg)."""
    adata_map: Dict[str, int] = {}
    for j, g in enumerate(adata_ensg):
        if g and g not in adata_map:  # first-seen wins, skips dup ENSGs
            adata_map[g] = j
    out = np.full(len(model_ensg), -1, dtype=np.int64)
    for i, g in enumerate(model_ensg):
        j = adata_map.get(g, -1)
        if j >= 0:
            out[i] = j
    return out


def _to_dense_row(mat, i: int) -> np.ndarray:
    if hasattr(mat, "getrow"):
        return np.asarray(mat.getrow(i).todense(), dtype=np.float32).ravel()
    return np.asarray(mat[i], dtype=np.float32).ravel()


def _strip_state_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _load_checkpoint(
    checkpoint: str | os.PathLike, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.nn.Module]]:
    try:
        obj = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(checkpoint, map_location=device)

    if isinstance(obj, torch.nn.Module):
        return {}, obj

    if isinstance(obj, dict):
        sd = obj.get("model_state_dict", obj)
        if not isinstance(sd, dict):
            raise RuntimeError(
                f"Unexpected checkpoint payload: type(model_state_dict)={type(sd)}"
            )
        return _strip_state_prefix(sd), None

    raise RuntimeError(f"Unsupported checkpoint object: {type(obj)}")


def encode(
    adata: ad.AnnData,
    *,
    checkpoint: Optional[str] = None,
    force_pert: bool = True,
    batch_size: int = 16,
    hidden_dim: int = 384,
    tissue: str | int = "blood",
    num_tissues: int = 64,
    gene_col: Optional[str] = None,
    gene_ids_path: Optional[str] = None,
    device: Optional[str] = None,
    input_is_log1p: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Compute xVERSE bio-encoder embeddings for every cell in ``adata``.

    Args:
        adata: Input AnnData. ``adata.var`` should contain an ENSG column
            (``Ensembl_ID`` / ``gene_ids`` / …) or ``var_names`` should be ENSG.
        checkpoint: Path to ``xVERSE_384.pth``. Falls back to
            ``LATENT_BENCH_XVERSE_CKPT`` / ``pretrained/xVerse/xVERSE_384.pth``.
        force_pert: If True and ``obsm['pert_var_idx']`` exists, run observability
            checks (finite, non-``-1`` slots for listed var columns in model space)
            and emit ``pert_kept_histogram``. Does not add a perturbation condition
            to the encoder—only coverage-style validation of aligned expression.
        batch_size: Cells per forward.
        hidden_dim: Checkpoint's hidden width (384 for ``xVERSE_384.pth``).
        tissue: Tissue name (looked up in ``tissue_name_to_id_map.csv``) or an
            integer id in ``[0, num_tissues)``.
        num_tissues: Tissue embedding table size (matches checkpoint).
        gene_col: Optional explicit ``adata.var`` column with ENSG ids.
        gene_ids_path: Override for the 17999 ENSG list shipped with xVERSE.
        device: ``cuda`` / ``cpu``. Auto-selects when ``None``.
        input_is_log1p: Benchmark convention is that ``adata.X`` is already
            ``log1p(normalize_total)``-transformed; xVERSE applies ``log1p``
            internally (``main/utils_model.py``), so we ``expm1`` the input
            first when this flag is True (default). Set False only if
            ``adata.X`` really holds raw counts.

    Returns:
        ``(embeddings (n_cells, hidden_dim) float32, meta dict)``. ``meta`` includes
        alignment fields and, when ``obsm['pert_var_idx']`` is present, histogram
        metadata as described below.
    """
    utils_model = _load_utils_model()
    XVerseModel = utils_model.XVerseModel

    ckpt_path = checkpoint or os.environ.get(
        "LATENT_BENCH_XVERSE_CKPT",
        str(paths.pretrained_root() / "xVerse" / "xVERSE_384.pth"),
    )
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"xVERSE checkpoint missing: {ckpt_path}")

    gene_ids_file = Path(gene_ids_path) if gene_ids_path else _default_gene_ids_path()
    model_ensg = _load_gene_list(gene_ids_file)
    total_gene = len(model_ensg)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = XVerseModel(
        num_samples=None,
        hidden_dim=hidden_dim,
        total_gene=total_gene,
        num_tissues=num_tissues,
        masks=None,
    ).to(dev)

    sd, loaded_module = _load_checkpoint(ckpt_path, dev)
    if loaded_module is not None:
        src = loaded_module.state_dict()
        sd = _strip_state_prefix(src)
    # Checkpoint has extra keys (sample_emb, film_gamma/beta, sample_classifier_bio)
    # because the pretrain model used ``num_samples`` > 0. We construct the
    # inference-time model without sample head, so strict=False is required.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    wanted = {
        "gene_embedding.weight",
        "tissue_gene_bias.weight",
        "bio_encoder.gene_emb",
        "bio_encoder.tissue_emb.weight",
    }
    missing_critical = [k for k in missing if k in wanted]
    if missing_critical:
        raise RuntimeError(
            f"xVERSE state_dict missing critical keys: {missing_critical}"
        )
    model.eval()

    adata_ensg, gene_id_report = _resolve_ensembl_series(adata, gene_col, model_ensg=model_ensg)
    m2a = _build_alignment(model_ensg, adata_ensg)  # len total_gene
    n_aligned = int((m2a >= 0).sum())
    if n_aligned == 0:
        raise RuntimeError(
            "0 genes aligned between adata (ENSG) and xVERSE model genes "
            "(ensg_keys_high_quality.txt). Pass gene_col=... or upstream-map."
        )

    X = adata.X
    X_dense_row = _to_dense_row  # alias

    # Optional: observability for columns listed in obsm['pert_var_idx'] (adata var indices).
    pert_var_idx = adata.obsm.get("pert_var_idx", None)
    pert_present = pert_var_idx is not None
    per_cell_pert_kept: List[int] = []
    pert_bad: List[Tuple[int, int]] = []
    n_slot_repairs: int = 0  # aligned slots set from unmeasured/invalid to observed 0

    if pert_var_idx is not None:
        pert_var_idx = np.asarray(pert_var_idx, dtype=np.int64)
        if pert_var_idx.ndim == 1:
            pert_var_idx = pert_var_idx.reshape(-1, 1)
        n_obs = int(adata.n_obs)
        if pert_var_idx.shape[0] < n_obs:
            n_pad = n_obs - int(pert_var_idx.shape[0])
            pad = np.full((n_pad, pert_var_idx.shape[1]), -1, dtype=np.int64)
            pert_var_idx = np.vstack([pert_var_idx, pad])
    # Reverse alignment: adata column -> model gene index (-1 if dropped).
    a2m = np.full(adata.n_vars, -1, dtype=np.int64)
    for i, j in enumerate(m2a):
        if j >= 0:
            a2m[int(j)] = i

    # Forward in batches.
    tissue_id_int = tissue if isinstance(tissue, int) else _tissue_name_to_id(tissue)
    if not (0 <= tissue_id_int < num_tissues):
        raise ValueError(
            f"tissue_id {tissue_id_int} out of range [0, {num_tissues})"
        )

    n_cells = adata.n_obs
    out = np.zeros((n_cells, hidden_dim), dtype=np.float32)

    use_amp = dev.type == "cuda"
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            stop = min(start + batch_size, n_cells)
            # Build values matrix [B, total_gene], -1 for unmeasured.
            values = np.full((stop - start, total_gene), -1.0, dtype=np.float32)
            for bi, ci in enumerate(range(start, stop)):
                row = X_dense_row(X, ci)
                # assign observed genes at their model positions
                mask = m2a >= 0
                cols = m2a[mask]
                vals = row[cols]
                if input_is_log1p:
                    vals = np.expm1(np.clip(vals, a_min=0.0, a_max=None))
                else:
                    vals = np.clip(vals, a_min=0.0, a_max=None)
                values[bi, mask] = vals

                if force_pert and pert_var_idx is not None:
                    kept = 0
                    for p_col in pert_var_idx[ci]:
                        p_col = int(p_col)
                        if p_col < 0:
                            continue
                        if p_col >= adata.n_vars:
                            pert_bad.append((ci, p_col))
                            continue
                        model_pos = int(a2m[p_col])
                        if model_pos < 0:
                            # Gene absent from xVERSE list; skip (cannot mark observed).
                            continue
                        v = values[bi, model_pos]
                        if not np.isfinite(v) or v < 0:
                            # Ensure an observed slot (0 counts), not unmeasured (-1).
                            values[bi, model_pos] = 0.0
                            v = 0.0
                            n_slot_repairs += 1
                        kept += 1
                    per_cell_pert_kept.append(kept)

            vtensor = torch.from_numpy(values).to(dev)
            ttensor = torch.full(
                (vtensor.shape[0],), int(tissue_id_int), dtype=torch.long, device=dev
            )

            if use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    z_bio, _ = model.bio_encoder(vtensor, ttensor)
            else:
                z_bio, _ = model.bio_encoder(vtensor, ttensor)

            out[start:stop] = z_bio.float().detach().cpu().numpy()

    if pert_bad:
        raise AssertionError(
            f"pert_var_idx references {len(pert_bad)} out-of-range columns, "
            f"e.g. {pert_bad[:3]}"
        )

    meta: dict = {
        "encoder_role": "ExpressionOnlyEncoder",
        "n_model_genes": total_gene,
        "n_aligned_genes": n_aligned,
        "gene_id_resolution": gene_id_report,
        "tissue_id": tissue_id_int,
        "hidden_dim": hidden_dim,
        "input_is_log1p": input_is_log1p,
        "force_pert": bool(force_pert),
        "pert_var_idx_present": bool(pert_present),
        # Per protocol (per_cell_pert_design.md): full-gene models without token sampling use false —
        # perturbation is already in X; no truncation/sampling coverage path exists.
        "force_pert_effective": False,
        # When force_pert + pert_var_idx, the adapter may set aligned slots from unmeasured (-1) to
        # observed 0 for observability; this is not a separate condition stream.
        "pert_var_idx_slot_repair": bool(force_pert and pert_present),
        "pert_var_idx_slot_repair_count": int(n_slot_repairs),
        "pert_source": "obsm_pert_var_idx" if pert_present else None,
    }
    if force_pert and pert_var_idx is not None:
        meta["pert_kept_histogram"] = histogram_pert_kept(per_cell_pert_kept)
    elif pert_var_idx is not None:
        meta["pert_kept_histogram"] = histogram_pert_kept(
            [int((pv >= 0).sum()) for pv in pert_var_idx]
        )
    return out, meta
