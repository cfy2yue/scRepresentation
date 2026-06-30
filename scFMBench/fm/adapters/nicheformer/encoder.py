"""NicheFormer expression-only adapter.

NicheFormer has existed in two checkpoint layouts: the original Lightning
``.ckpt`` from the GitHub/Mendeley workflow and the newer HuggingFace
``model.safetensors`` export.  This adapter keeps both paths and uses the same
direct h5ad tokenization policy for each.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

import paths

GENE_ID_CANDIDATES = ("ensembl_id", "ensemblid", "Ensembl_ID", "ENSEMBL", "gene_id", "feature_id", "gene_ids")
COUNT_LAYER_CANDIDATES = ("counts", "raw_counts", "count")
DEFAULT_CONTEXT_TOKENS = {
    "specie": 5,  # human
    "assay": 11,  # 10x 3' v3
    "modality": 3,  # dissociated
}


try:
    import numba

    @numba.jit(nopython=True, nogil=True)
    def _sub_tokenize_data(x: np.ndarray, max_seq_len: int = -1, aux_tokens: int = 30) -> np.ndarray:
        scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
        for i, cell in enumerate(x):
            nonzero_mask = np.nonzero(cell)[0]
            sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
            sorted_indices = sorted_indices + aux_tokens
            if max_seq_len:
                scores = np.zeros(max_seq_len, dtype=np.int32)
            else:
                scores = np.zeros_like(cell, dtype=np.int32)
            scores[: len(sorted_indices)] = sorted_indices.astype(np.int32)
            scores_final[i, :] = scores
        return scores_final

except Exception:

    def _sub_tokenize_data(x: np.ndarray, max_seq_len: int = -1, aux_tokens: int = 30) -> np.ndarray:
        out_len = max_seq_len if max_seq_len > 0 else x.shape[1]
        scores_final = np.empty((x.shape[0], out_len), dtype=np.int32)
        for i, cell in enumerate(x):
            nonzero_mask = np.nonzero(cell)[0]
            sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len] + aux_tokens
            scores = np.zeros(out_len, dtype=np.int32)
            scores[: len(sorted_indices)] = sorted_indices.astype(np.int32)
            scores_final[i, :] = scores
        return scores_final


def _sf_normalize(x: np.ndarray) -> np.ndarray:
    counts = np.asarray(x.sum(axis=1), dtype=np.float32)
    counts += counts == 0.0
    scale = (10000.0 / counts).reshape((-1, 1))
    out = x.copy()
    np.multiply(out, scale, out=out)
    return out


def _third_party_src() -> Path:
    return paths.third_party_root() / "nicheformer" / "src"


def _checkpoint_path() -> Path:
    value = os.environ.get("LATENT_BENCH_NICHEFORMER_CKPT", "").strip()
    return Path(value).expanduser().resolve() if value else paths.pretrained_root() / "nicheformer" / "nicheformer.ckpt"


def _hf_dir_path() -> Path:
    value = os.environ.get("LATENT_BENCH_NICHEFORMER_HF_DIR", "").strip()
    default = paths.pretrained_root() / "nicheformer" / "theislab_Nicheformer"
    return Path(value).expanduser().resolve() if value else default


def _resolve_weights() -> tuple[str, Path]:
    ckpt = _checkpoint_path()
    if ckpt.is_file():
        return "lightning_ckpt", ckpt
    hf_dir = _hf_dir_path()
    if (hf_dir / "config.json").is_file() and (hf_dir / "model.safetensors").is_file():
        return "huggingface_safetensors", hf_dir
    raise FileNotFoundError(
        "NicheFormer weights missing. Expected either "
        f"{ckpt} or a HuggingFace snapshot with config.json + model.safetensors at {hf_dir}. "
        "Set LATENT_BENCH_NICHEFORMER_CKPT or LATENT_BENCH_NICHEFORMER_HF_DIR to override."
    )


def _mean_h5ad_path() -> Path:
    value = os.environ.get("LATENT_BENCH_NICHEFORMER_MEAN_H5AD", "").strip()
    default = paths.third_party_root() / "nicheformer" / "data" / "model_means" / "model.h5ad"
    return Path(value).expanduser().resolve() if value else default


def _counts_layer_name(adata: ad.AnnData) -> str | None:
    configured = os.environ.get("LATENT_BENCH_NICHEFORMER_COUNTS_LAYER", "").strip()
    candidates = (configured,) if configured else COUNT_LAYER_CANDIDATES
    for candidate in candidates:
        if candidate and candidate in adata.layers:
            return candidate
    return None


def _select_count_input(adata: ad.AnnData, input_is_log1p: bool) -> tuple[ad.AnnData, str]:
    if not input_is_log1p:
        return adata, "X"
    layer = _counts_layer_name(adata)
    if layer is not None:
        out = adata.copy()
        out.X = adata.layers[layer].copy()
        return out, f"layers[{layer!r}]"
    raise ValueError(
        "NicheFormer direct tokenization expects count-like X, while benchmark X is marked log1p and no "
        "raw-count layer was found. This adapter will not apply a second log1p and will not silently "
        "expm1(log1p X) into pseudo-counts. Provide a raw-count layer such as layers['counts']; only pass "
        "--no-input-is-log1p when X is genuinely count-like."
    )


def _use_ensembl_var_names_if_needed(adata: ad.AnnData, target_genes: set[str]) -> ad.AnnData:
    overlap = len(target_genes.intersection(adata.var_names.astype(str)))
    if overlap >= 1000:
        return adata
    for candidate in GENE_ID_CANDIDATES:
        if candidate in adata.var.columns:
            values = adata.var[candidate].astype(str)
            if len(target_genes.intersection(values)) >= 1000:
                out = adata.copy()
                out.var_names = values
                return out
    return adata


def _aligned_counts_and_means(
    adata: ad.AnnData, mean_h5ad: Path, input_is_log1p: bool
) -> tuple[ad.AnnData, np.ndarray, str]:
    from scipy.sparse import issparse

    if not mean_h5ad.is_file():
        raise FileNotFoundError(f"NicheFormer model mean h5ad missing: {mean_h5ad}")
    mean_adata = ad.read_h5ad(mean_h5ad)
    target_genes = mean_adata.var_names.astype(str)
    if mean_adata.X.shape[0] != 1:
        raise ValueError(f"Expected one-row NicheFormer mean h5ad, got {mean_adata.shape}")
    count_adata, counts_source = _select_count_input(adata, input_is_log1p)
    count_adata = _use_ensembl_var_names_if_needed(count_adata, set(target_genes))
    missing = [g for g in target_genes[:100] if g not in count_adata.var_names]
    if len(missing) == 100:
        raise ValueError(
            "NicheFormer mean genes do not match input var_names. Use Ensembl IDs as var_names "
            "or provide an Ensembl var column such as 'ensemblid', 'Ensembl_ID', or 'ENSEMBL'."
        )
    aligned = count_adata[:, target_genes.intersection(count_adata.var_names)].copy()
    if aligned.n_vars < 1000:
        raise ValueError(f"Only {aligned.n_vars} NicheFormer genes overlap input; refusing to encode.")
    mean_x = mean_adata[:, aligned.var_names].X
    means = (mean_x.toarray() if issparse(mean_x) else np.asarray(mean_x)).reshape(-1).astype(np.float32)
    return aligned, means, counts_source


def _model_attr(model: Any, name: str, default: Any = None) -> Any:
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, name):
        return getattr(cfg, name)
    hparams = getattr(model, "hparams", None)
    if hparams is not None and hasattr(hparams, name):
        return getattr(hparams, name)
    return default


def _context_token(field: str) -> int:
    env = os.environ.get(f"LATENT_BENCH_NICHEFORMER_{field.upper()}_TOKEN", "").strip()
    if env:
        return int(env)
    return DEFAULT_CONTEXT_TOKENS[field]


def _load_model(weights_kind: str, weights_path: Path, device: str) -> Any:
    import torch

    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if weights_kind == "huggingface_safetensors":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(weights_path),
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        src = _third_party_src()
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from nicheformer.models._nicheformer import Nicheformer

        model = Nicheformer.load_from_checkpoint(str(weights_path), map_location="cpu")
    model.eval()
    model.to(dev)
    return model


def _encode_batch(model: Any, weights_kind: str, tokens: "torch.Tensor", dev: "torch.device") -> "torch.Tensor":
    import torch

    layer = int(os.environ.get("LATENT_BENCH_NICHEFORMER_LAYER", "-1"))
    if weights_kind == "huggingface_safetensors":
        input_ids = tokens.to(dev)
        attention_mask = input_ids.ne(0)
        return model.get_embeddings(input_ids=input_ids, attention_mask=attention_mask, layer=layer)

    batch: dict[str, torch.Tensor] = {"X": tokens.to(dev)}
    for field in ("specie", "assay", "modality"):
        if bool(_model_attr(model, field, False)):
            batch[field] = torch.full((tokens.shape[0],), _context_token(field), dtype=torch.int64, device=dev)
    return model.get_embeddings(batch, layer=layer)


def encode(
    adata: ad.AnnData,
    *,
    device: str = "cuda",
    batch_size: int = 4,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mean-pooled NicheFormer cell embeddings from the last layer."""
    del force_pert, show_progress
    weights_kind, weights_path = _resolve_weights()
    src = _third_party_src()
    if not src.is_dir():
        raise FileNotFoundError(f"NicheFormer source checkout missing: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    from scipy.sparse import issparse
    from torch.utils.data import DataLoader, TensorDataset

    aligned, means, counts_source = _aligned_counts_and_means(adata, _mean_h5ad_path(), input_is_log1p)
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = _load_model(weights_kind, weights_path, device)

    aux_fields = ("specie", "assay", "modality")
    aux_count = sum(bool(_model_attr(model, name, False)) for name in aux_fields)
    context_length = int(_model_attr(model, "context_length"))
    max_seq_len = context_length - aux_count
    if max_seq_len < 128:
        raise ValueError(f"Invalid NicheFormer context length after aux tokens: {max_seq_len}")

    tokens_chunks: list[np.ndarray] = []
    chunk_size = int(os.environ.get("LATENT_BENCH_NICHEFORMER_TOKEN_CHUNK", "512"))
    for start in range(0, aligned.n_obs, chunk_size):
        chunk = aligned.X[start : start + chunk_size]
        x = chunk.toarray() if issparse(chunk) else np.asarray(chunk)
        x = _sf_normalize(np.nan_to_num(x).astype(np.float32, copy=False))
        x = x / np.where(means == 0, 1.0, means).reshape(1, -1)
        tokens_chunks.append(_sub_tokenize_data(x, max_seq_len, 30).astype(np.int64))
    tokens = np.concatenate(tokens_chunks, axis=0)
    if weights_kind == "huggingface_safetensors" and aux_count:
        context_cols = [
            np.full((tokens.shape[0], 1), _context_token(field), dtype=np.int64)
            for field in ("specie", "assay", "modality")
            if bool(_model_attr(model, field, False))
        ]
        tokens = np.concatenate([*context_cols, tokens], axis=1)
    ds = TensorDataset(torch.from_numpy(tokens))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            emb = _encode_batch(model, weights_kind, x_batch, dev)
            outputs.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
    z = np.concatenate(outputs, axis=0)
    meta: dict[str, Any] = {
        "encoder_role": "ExpressionOnlyEncoder",
        "model_family": "NicheFormer",
        "official_repo": "https://github.com/theislab/nicheformer",
        "weights_kind": weights_kind,
        "weights_path": str(weights_path),
        "mean_h5ad": str(_mean_h5ad_path()),
        "counts_source": counts_source,
        "pooling": "official get_embeddings mean pooling",
        "layer": int(os.environ.get("LATENT_BENCH_NICHEFORMER_LAYER", "-1")),
        "n_overlap_genes": int(aligned.n_vars),
        "max_seq_len": int(max_seq_len),
        "context_length": int(context_length),
        "context_tokens": {
            field: _context_token(field) for field in aux_fields if bool(_model_attr(model, field, False))
        },
        "batch_size": int(batch_size),
        "input_is_log1p": bool(input_is_log1p),
        "third_party_src": str(src),
        "force_pert_effective": False,
        "pert_source": None,
    }
    return z, meta
