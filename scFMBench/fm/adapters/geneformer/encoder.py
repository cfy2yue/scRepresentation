"""
Geneformer V2 cell embedding from expression-only token sequences.

latent_bench protocol: model input reflects **only** ranked expression tokens
(plus ``<cls>``). ``obsm['pert_var_idx']`` (when enabled with ``force_pert``) is
used solely so sampling/truncation at ``max_len`` never drops those gene tokens;
it is **not** a separate condition stream and protected tokens must never be
inserted as a block immediately after ``<cls>``.

Follows the tokenization math in
``third_party/Geneformer/geneformer/tokenizer.py`` (``tokenize_cell`` /
``rank_genes``) and the V2 cell-pooling convention in
``third_party/Geneformer/geneformer/emb_extractor.py`` (mean of non-CLS
non-pad hidden states from the last layer).
"""

from __future__ import annotations

import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from transformers import BertForMaskedLM
import paths

from .._common import histogram_pert_kept

_GF_DIR = paths.third_party_root() / "Geneformer" / "geneformer"
_TOKEN_DICT_FILE = _GF_DIR / "token_dictionary_gc104M.pkl"
_GENE_MEDIAN_FILE = _GF_DIR / "gene_median_dictionary_gc104M.pkl"
_ENSEMBL_MAPPING_FILE = _GF_DIR / "ensembl_mapping_dict_gc104M.pkl"


def _load_pickle(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _rank_tokens_for_cell(
    x_row: np.ndarray,
    coding_loc: np.ndarray,
    coding_tokens: np.ndarray,
    norm_factor: np.ndarray,
    target_sum: float = 10_000.0,
) -> np.ndarray:
    """Return token ids sorted by descending Geneformer-normalized expression.

    ``x_row`` is the full gene-expression vector in dataset-var order. Only
    columns in ``coding_loc`` (present in the Geneformer vocab) are used.
    """
    sub = x_row[coding_loc]
    total = float(sub.sum())
    if total <= 0.0:
        return np.asarray([], dtype=np.int64)
    norm = sub / total * target_sum / norm_factor
    nz = np.nonzero(norm)[0]
    if nz.size == 0:
        return np.asarray([], dtype=np.int64)
    order = np.argsort(-norm[nz])
    return coding_tokens[nz[order]].astype(np.int64)


def _merge_rank_tokens_with_protected(
    rank_tokens: np.ndarray,
    protected_tokens: List[int],
    *,
    cls_token_id: int,
    max_len: int,
) -> List[int]:
    """Preserve expression rank as the primary trajectory; inject missing protected only at the tail.

    Unlike a perturbation-prefix layout (``<cls>`` + all pert tokens + rest),
    we keep the longest initial segment of the expression-ranked (deduplicated)
    sequence that fits in ``max_len - 1`` slots, evicting only **non-protected**
    tail tokens to make room so every protected vocabulary id appears somewhere
    after ``<cls>`` but never as a dedicated prefix block.
    """
    protected_unique: List[int] = []
    seen_protected: set[int] = set()
    for tid in protected_tokens:
        tid = int(tid)
        if tid in seen_protected:
            continue
        seen_protected.add(tid)
        protected_unique.append(tid)

    if len(protected_unique) > max_len - 1:
        raise ValueError(
            f"Geneformer protected gene count ({len(protected_unique)}) exceeds max_len-1 ({max_len - 1})."
        )

    budget = max_len - 1

    rank_list: List[int] = []
    seen_rank: set[int] = set()
    for tid in rank_tokens.tolist():
        tid = int(tid)
        if tid in seen_rank:
            continue
        seen_rank.add(tid)
        rank_list.append(tid)
    rank_pos = {tid: i for i, tid in enumerate(rank_list)}
    protected_set = set(protected_unique)

    selected: List[int] = []
    for tid in rank_list:
        if len(selected) >= budget:
            break
        selected.append(tid)

    selected_set = set(selected)
    missing = [tid for tid in protected_unique if tid not in selected_set]

    while missing and len(selected) + len(missing) > budget:
        j = len(selected) - 1
        while j >= 0 and selected[j] in protected_set:
            j -= 1
        if j < 0:
            raise ValueError(
                "Geneformer protected coverage could not be satisfied: all selected tail positions are protected."
            )
        removed = selected.pop(j)
        selected_set.discard(removed)

    missing = [tid for tid in protected_unique if tid not in selected_set]
    if missing:
        # Append missing in expression-rank order (first occurrence in full rank list).
        for tid in sorted(missing, key=lambda t: rank_pos.get(t, 10**9)):
            if len(selected) >= budget:
                break
            if tid not in selected_set:
                selected.append(tid)
                selected_set.add(tid)

    return [cls_token_id, *selected]


def encode(
    adata: ad.AnnData,
    *,
    model_dir: Optional[str] = None,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    batch_size: int = 8,
    model_input_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Returns:
        (n_cells, hidden_size) float32 embeddings, metadata dict.

    Notes:
        * When ``force_pert=True`` and ``obsm['pert_var_idx']`` is present, those
          column indices map to vocab tokens used **only** as a protected set for
          truncation: the main sequence remains expression-ranked (prefix of the
          rank list), and any protected token not in that prefix is appended after
          evicting lowest-priority non-protected tail slots — never as a ``<cls>``
          prefix block.
        * Geneformer's tokenizer normalizes by row-sum then divides by a per-gene
          median factor derived from **raw counts**; rank is not invariant to
          ``log1p`` because the per-gene divisor then combines with a concave
          transform of the counts. Benchmark convention is ``adata.X`` is already
          ``log1p``-transformed, so we ``expm1`` the dense counts first when
          ``input_is_log1p=True`` (default).
    """
    model_dir_p = Path(model_dir or os.environ.get(
        "LATENT_BENCH_GENEFORMER_MODEL_DIR",
        str(paths.pretrained_root() / "geneformer" / "Geneformer-V2-316M"),
    ))
    if not model_dir_p.is_dir():
        raise FileNotFoundError(f"Geneformer model dir missing: {model_dir_p}")

    gene_token_dict: Dict[str, int] = _load_pickle(_TOKEN_DICT_FILE)
    gene_median_dict: Dict[str, float] = _load_pickle(_GENE_MEDIAN_FILE)
    ensembl_mapping_dict: Dict[str, str] = _load_pickle(_ENSEMBL_MAPPING_FILE)

    cls_id = int(gene_token_dict["<cls>"])
    pad_id = int(gene_token_dict["<pad>"])

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BertForMaskedLM.from_pretrained(
        str(model_dir_p),
        output_hidden_states=True,
        local_files_only=True,
    )
    model.to(dev)
    model.eval()

    max_len = int(model_input_size or model.config.max_position_embeddings)
    hidden_size = int(model.config.hidden_size)

    ensembl_col = None
    for cand in ("Ensembl_ID", "ENSEMBL", "ensemblid", "ensembl_id", "gene_ids", "gene_id", "ensembl"):
        if cand in adata.var.columns:
            ensembl_col = cand
            break
    if ensembl_col is None:
        raise KeyError(
            "An Ensembl column is required for the Geneformer adapter; tried "
            "['Ensembl_ID', 'ensemblid', 'ensembl_id', 'gene_ids', 'gene_id', 'ensembl']."
        )

    var_ensembl_raw = adata.var[ensembl_col].astype(str).tolist()
    gene_keys_set = set(gene_token_dict.keys())
    collapsed: List[Optional[str]] = []
    for e in var_ensembl_raw:
        eid_u = e.upper().split(".")[0]
        mapped = ensembl_mapping_dict.get(eid_u, eid_u)
        if mapped in gene_keys_set and mapped in gene_median_dict:
            collapsed.append(mapped)
        else:
            collapsed.append(None)

    coding_loc = np.asarray(
        [i for i, v in enumerate(collapsed) if v is not None], dtype=np.int64
    )
    if coding_loc.size == 0:
        raise RuntimeError(
            "No genes from adata map into Geneformer token dictionary. "
            "Check adata.var['Ensembl_ID']."
        )
    coding_ensembl = [collapsed[i] for i in coding_loc]  # type: ignore[index]
    coding_tokens = np.asarray(
        [gene_token_dict[e] for e in coding_ensembl], dtype=np.int64
    )
    norm_factor = np.asarray(
        [float(gene_median_dict[e]) for e in coding_ensembl], dtype=np.float64
    )

    dataset_col_to_token: Dict[int, int] = {
        int(i): int(gene_token_dict[collapsed[i]])  # type: ignore[index]
        for i in coding_loc.tolist()
    }

    X = adata.X
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)
    X_dense = X_dense.astype(np.float32, copy=False)
    if input_is_log1p:
        # Benchmark convention: adata.X = log1p(normalize_total). Geneformer's
        # median-scaling divides by raw-count statistics so rank is not preserved
        # under log1p; undo it before tokenization.
        X_dense = np.expm1(np.clip(X_dense, 0.0, None)).astype(np.float32, copy=False)

    pert_var_idx = adata.obsm.get("pert_var_idx")
    if pert_var_idx is not None:
        pert_var_idx = np.asarray(pert_var_idx, dtype=np.int64)

    n_cells = adata.n_obs
    per_cell_token_seqs: List[List[int]] = []
    per_cell_kept: List[int] = []

    for i in range(n_cells):
        rank_tokens = _rank_tokens_for_cell(
            X_dense[i], coding_loc, coding_tokens, norm_factor
        )

        pert_tokens: List[int] = []
        if force_pert and pert_var_idx is not None:
            row = pert_var_idx[i] if i < pert_var_idx.shape[0] else np.asarray([])
            seen_tok: set = set()
            for j in np.asarray(row).ravel().tolist():
                j = int(j)
                if j < 0:
                    continue
                tid = dataset_col_to_token.get(j)
                if tid is None or tid in seen_tok:
                    continue
                seen_tok.add(tid)
                pert_tokens.append(tid)
        per_cell_kept.append(len(pert_tokens))

        tokens = _merge_rank_tokens_with_protected(
            rank_tokens,
            pert_tokens,
            cls_token_id=cls_id,
            max_len=max_len,
        )
        per_cell_token_seqs.append(tokens)

    out = np.zeros((n_cells, hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch = per_cell_token_seqs[start:end]
            lens = [len(seq) for seq in batch]
            if max(lens) == 0:
                raise RuntimeError(
                    "Geneformer adapter: empty token sequences after build; "
                    "this should not happen (check vocab / <cls> vs <pad> ids)."
                )
            batch_max = min(max(lens), max_len)
            input_ids = np.full((len(batch), batch_max), pad_id, dtype=np.int64)
            attn = np.zeros((len(batch), batch_max), dtype=np.int64)
            for bi, seq in enumerate(batch):
                L = min(len(seq), batch_max)
                if L > 0:
                    input_ids[bi, :L] = seq[:L]
                    attn[bi, :L] = 1
            input_ids_t = torch.from_numpy(input_ids).to(dev)
            attn_t = torch.from_numpy(attn).to(dev)
            outputs = model(input_ids=input_ids_t, attention_mask=attn_t)
            hidden = outputs.hidden_states[-1]  # (B, L, H)
            # Geneformer V2 emb_mode="cell": exclude <cls>, mean over non-pad.
            non_cls = hidden[:, 1:, :]
            lens_t = torch.tensor(
                [max(L - 1, 1) for L in lens], device=dev, dtype=torch.float32
            )
            pos = torch.arange(non_cls.size(1), device=dev).unsqueeze(0)
            mask = (pos < (lens_t - 0).long().unsqueeze(1)).unsqueeze(-1)
            pooled = (non_cls * mask).sum(dim=1) / lens_t.unsqueeze(1)
            out[start:end] = pooled.float().cpu().numpy()

    meta = {
        "encoder_role": "ExpressionOnlyEncoder",
        "hidden_dim": int(hidden_size),
        "pert_kept_histogram": histogram_pert_kept(per_cell_kept),
        "max_len": max_len,
        "hidden_size": hidden_size,
        "n_genes_mapped_to_vocab": int(coding_loc.size),
        "n_genes_total": int(adata.n_vars),
        "input_is_log1p": bool(input_is_log1p),
        "force_pert": bool(force_pert),
        "pert_var_idx_present": pert_var_idx is not None,
        "pert_source": "obsm_pert_var_idx" if pert_var_idx is not None else None,
        "force_pert_effective": bool(force_pert and pert_var_idx is not None),
    }
    return out, meta
