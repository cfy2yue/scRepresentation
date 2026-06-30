"""
scGPT cell embedding (latent_bench expression-only protocol).

Gene ids and expression values come from ``adata.X`` (non-zero genes, ranked).
If ``obsm['pert_var_idx']`` exists and ``force_pert=True``, those columns are a
**protected-gene mask** for truncation/sampling only. There is no
``obs['perturbation']`` parsing path and no separate condition / prefix token
stream beyond ``<cls>`` + sampled expression tokens.
"""

from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import anndata as ad
import numpy as np
import torch
import paths
from torch.utils.data import DataLoader, SequentialSampler

from .._common import histogram_pert_kept

# Process-local flash-attn execution state (updated by patched encoder forward).
_FLASH_STATE: Dict[str, Any] = {}


def _install_optional_dependency_shims() -> None:
    """Provide tiny shims for notebook-only imports used by upstream scGPT."""
    if "IPython" not in sys.modules:
        mod = types.ModuleType("IPython")
        mod.get_ipython = lambda: None  # type: ignore[attr-defined]
        sys.modules["IPython"] = mod
    if "torchtext.vocab" not in sys.modules:
        torchtext_mod = types.ModuleType("torchtext")
        vocab_mod = types.ModuleType("torchtext.vocab")

        class _SimpleVocab:
            def __init__(self, vocab=None):
                if isinstance(vocab, _SimpleVocab):
                    self._itos = list(vocab._itos)
                    self._stoi = dict(vocab._stoi)
                elif isinstance(vocab, dict):
                    self._itos = [None] * len(vocab)
                    for tok, idx in vocab.items():
                        self._itos[int(idx)] = tok
                    self._stoi = {tok: i for i, tok in enumerate(self._itos)}
                else:
                    self._itos = []
                    self._stoi = {}
                self.vocab = self
                self._default_index: Optional[int] = None

            def __contains__(self, token):
                return token in self._stoi

            def __len__(self):
                return len(self._itos)

            def __getitem__(self, token):
                if token in self._stoi:
                    return self._stoi[token]
                if self._default_index is not None:
                    return self._default_index
                raise KeyError(token)

            def __call__(self, tokens):
                if isinstance(tokens, str):
                    return self[tokens]
                return [self[token] for token in tokens]

            def insert_token(self, token, index):
                index = int(index)
                if token in self._stoi:
                    return
                while len(self._itos) < index:
                    self._itos.append(f"<unused_{len(self._itos)}>")
                if index == len(self._itos):
                    self._itos.append(token)
                else:
                    self._itos.insert(index, token)
                self._stoi = {tok: i for i, tok in enumerate(self._itos)}

            def append_token(self, token):
                self.insert_token(token, len(self._itos))

            def set_default_index(self, index):
                self._default_index = int(index)

            def get_stoi(self):
                return dict(self._stoi)

            def get_itos(self):
                return list(self._itos)

        def _vocab(ordered_dict, min_freq=1):
            v = _SimpleVocab()
            for tok, freq in ordered_dict.items():
                if int(freq) >= int(min_freq):
                    v.append_token(tok)
            return v

        vocab_mod.Vocab = _SimpleVocab
        vocab_mod.vocab = _vocab
        torchtext_mod.vocab = vocab_mod
        sys.modules["torchtext"] = torchtext_mod
        sys.modules["torchtext.vocab"] = vocab_mod
    if "datasets" not in sys.modules:
        datasets_mod = types.ModuleType("datasets")

        class _Dataset:  # pragma: no cover - only satisfies upstream import side effects.
            pass

        def _load_dataset(*_args, **_kwargs):
            raise ImportError("datasets is not installed; scBank utilities are unavailable in this env")

        datasets_mod.Dataset = _Dataset  # type: ignore[attr-defined]
        datasets_mod.load_dataset = _load_dataset  # type: ignore[attr-defined]
        sys.modules["datasets"] = datasets_mod


def _flash_state_reset(device: torch.device) -> None:
    """Initialize flash-attn status fields before a single encode() run."""
    global _FLASH_STATE
    disable = os.environ.get("LATENT_BENCH_SCGPT_DISABLE_FLASH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    fa_avail = False
    fa_ver = "?"
    try:
        import flash_attn  # noqa: F401

        fa_avail = True
        import flash_attn as _fa_mod

        fa_ver = getattr(_fa_mod, "__version__", "?")
    except Exception:
        fa_avail = False
    if disable:
        fa_avail = False
    _FLASH_STATE = {
        "requested": True,
        "fa_available": bool(fa_avail),
        "fa_version": str(fa_ver),
        "torch_version": str(torch.__version__),
        "used_flash": False,
        "used_fallback": False,
        "fallback_reason": None,
        "device_is_cuda": device.type == "cuda",
    }


def _flash_attn_meta() -> Dict[str, Any]:
    """Metadata fields for meta.json (call after encode() forward loop)."""
    global _FLASH_STATE
    dev_cuda = bool(_FLASH_STATE.get("device_is_cuda", False))
    fa_ok = bool(_FLASH_STATE.get("fa_available", False))
    used_flash = bool(_FLASH_STATE.get("used_flash", False))
    used_fb = bool(_FLASH_STATE.get("used_fallback", False))
    effective = bool(dev_cuda and fa_ok and used_flash and not used_fb)
    out: Dict[str, Any] = {
        "flash_attn_requested": True,
        "flash_attn_effective": effective,
        "flash_attn_version": _FLASH_STATE.get("fa_version", "?"),
        "torch_version": _FLASH_STATE.get("torch_version", str(torch.__version__)),
    }
    if not effective and _FLASH_STATE.get("fallback_reason"):
        out["flash_attn_fallback_reason"] = _FLASH_STATE["fallback_reason"]
    return out


def _unwrap_attn_output(attn_raw: Any) -> torch.Tensor:
    """scGPT uses ``[0]`` for legacy tuple APIs; MHA may return a plain tensor."""
    if isinstance(attn_raw, tuple):
        return cast(torch.Tensor, attn_raw[0])
    return cast(torch.Tensor, attn_raw)


def _run_fallback_self_attn(
    layer: torch.nn.Module,
    x: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    PyTorch (eager) attention via a temporary SelfAttention inner module.
    Needed when MHA was built with use_flash_attn=True (FlashSelfAttention cannot take key_padding_mask).
    """
    from flash_attn.modules.mha import SelfAttention

    mha = layer.self_attn
    saved_inner = mha.inner_attn
    saved_uf = bool(mha.use_flash_attn)
    try:
        if saved_uf and not isinstance(saved_inner, SelfAttention):
            inner = saved_inner
            mha.inner_attn = SelfAttention(
                causal=inner.causal,
                softmax_scale=inner.softmax_scale,
                attention_dropout=inner.drop.p,
            ).to(device=x.device)
        mha.use_flash_attn = False
        out = mha(x, key_padding_mask=key_padding_mask)
        return _unwrap_attn_output(out)
    finally:
        mha.inner_attn = saved_inner
        mha.use_flash_attn = saved_uf


def _run_packed_flash_self_attn(
    layer: torch.nn.Module,
    x: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Run FlashMHA in packed or dense flash path. ``pad_mask`` is True at PAD tokens
    (same convention as scGPT ``src_key_padding_mask``).

    flash_attn kernels require fp16/bf16; LayerNorm outputs in an outer autocast block
    may still be fp32, so we cast for the MHA call and cast outputs back to ``x.dtype``.
    """
    mha = layer.self_attn
    mha.use_flash_attn = True
    orig_dtype = x.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        working_dtype = orig_dtype
        x_work = x
    else:
        working_dtype = torch.float16
        x_work = x.to(dtype=working_dtype)
    if pad_mask is None or not pad_mask.any().item():
        out = mha(x_work)
        return _unwrap_attn_output(out).to(orig_dtype)
    keep = (~pad_mask).to(dtype=torch.int32)
    from flash_attn.bert_padding import pad_input, unpad_input

    packed, indices, cu_seqlens, max_seqlen, *_ = unpad_input(x_work, keep)
    out_packed = mha(packed, cu_seqlens=cu_seqlens, max_seqlen=int(max_seqlen))
    out_packed = _unwrap_attn_output(out_packed)
    out = pad_input(out_packed, indices, x.shape[0], x.shape[1])
    return out.to(orig_dtype)


def _patched_flash_transformer_encoder_forward(
    self: torch.nn.Module,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    src_key_padding_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """FlashTransformerEncoderLayer.forward with flash-attn 2.x packed padding support."""
    global _FLASH_STATE
    if src_mask is not None:
        raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

    pad_mask = src_key_padding_mask
    if pad_mask is not None and pad_mask.dtype != torch.bool:
        pad_mask = pad_mask.bool()

    if pad_mask is None or not pad_mask.any().item():
        key_pad_mha: Optional[torch.Tensor] = None
    else:
        key_pad_mha = ~pad_mask

    use_flash = bool(_FLASH_STATE.get("fa_available", False)) and src.is_cuda

    def _attention_branch(x_in: torch.Tensor) -> torch.Tensor:
        nonlocal use_flash
        if use_flash:
            try:
                out = _run_packed_flash_self_attn(self, x_in, pad_mask)
                _FLASH_STATE["used_flash"] = True
                return out
            except Exception as e:
                use_flash = False
                _FLASH_STATE["used_fallback"] = True
                if _FLASH_STATE.get("fallback_reason") is None:
                    _FLASH_STATE["fallback_reason"] = f"exception:{type(e).__name__}:{e}"
                return _run_fallback_self_attn(self, x_in, key_pad_mha)
        _FLASH_STATE["used_fallback"] = True
        if _FLASH_STATE.get("fallback_reason") is None:
            if not src.is_cuda:
                _FLASH_STATE["fallback_reason"] = "cpu"
            elif not _FLASH_STATE.get("fa_available", False):
                _FLASH_STATE["fallback_reason"] = "flash_attn_unavailable"
        return _run_fallback_self_attn(self, x_in, key_pad_mha)

    if self.norm_scheme == "pre":
        src = self.norm1(src)
        src2 = _attention_branch(src)
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
    else:
        src2 = _attention_branch(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

    return src


def _patch_encoder_layer_forward() -> None:
    _ensure_scgpt_path()
    from scgpt.model.model import FlashTransformerEncoderLayer

    if getattr(_patched_flash_transformer_encoder_forward, "_lb_flashpatched", False):
        return
    FlashTransformerEncoderLayer.forward = _patched_flash_transformer_encoder_forward
    _patched_flash_transformer_encoder_forward._lb_flashpatched = True  # type: ignore[attr-defined]


def _scgpt_dir() -> Path:
    return paths.third_party_root() / "scGPT-main"


def _ensure_scgpt_path() -> None:
    _install_optional_dependency_shims()
    p = _scgpt_dir()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    root = paths.delivery_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _patch_flash_mha() -> None:
    """flash-attn>=2.x: map legacy kwargs and enable Flash kernels (see encoder forward patch)."""
    try:
        try:
            import flash_attn.flash_attention as _fa
        except ModuleNotFoundError:
            import types
            from flash_attn.modules.mha import MHA

            _fa = types.ModuleType("flash_attn.flash_attention")
            _fa.FlashMHA = MHA  # type: ignore[attr-defined]
            sys.modules["flash_attn.flash_attention"] = _fa

        orig = _fa.FlashMHA
        if getattr(orig, "_lb_patched", False):
            return

        class _Compat(orig):  # type: ignore[misc]
            _lb_patched = True

            def __init__(self, *args, **kwargs):
                kwargs.pop("batch_first", None)
                if "attention_dropout" in kwargs and "dropout" not in kwargs:
                    kwargs["dropout"] = kwargs.pop("attention_dropout")
                kwargs["use_flash_attn"] = True
                super().__init__(*args, **kwargs)
                self.use_flash_attn = True
                self.batch_first = True

        _fa.FlashMHA = _Compat
    except Exception:
        pass


@dataclass
class ProtectedGeneDataCollator:
    """Expression-only collator that keeps protected genes during truncation."""

    base: object  # DataCollator instance

    def __call__(self, examples: List[dict]) -> dict:
        from scgpt.preprocess import binning

        base = self.base
        device = examples[0]["genes"].device
        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = base.max_length if max_ori_len >= base.max_length else max_ori_len

        padded_genes = []
        padded_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"].clone()
            protected_mask = examples[i]["protected_mask"].to(dtype=torch.bool, device=genes.device)
            if base.do_binning and 1 < len(expressions):
                b = binning(row=expressions[1:], n_bins=51)
                expressions[1:] = b.to(expressions.dtype)
            genes, expressions = self._sample_or_truncate_plus_pad_keep_protected(
                genes, expressions, protected_mask, _max_length
            )
            padded_genes.append(genes)
            padded_expressions.append(expressions)

        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)
        data_dict = {"gene": padded_genes, "expr": padded_expressions}
        if base.do_mlm:
            masked_expressions = base._mask(padded_expressions)
        else:
            masked_expressions = padded_expressions
        data_dict["masked_expr"] = masked_expressions
        return data_dict

    def _sample_or_truncate_plus_pad_keep_protected(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        protected_mask: torch.BoolTensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions) == len(protected_mask)
        if len(genes) <= max_length:
            return self.base._pad(genes, expressions, max_length) if len(genes) < max_length else (genes, expressions)

        device = genes.device
        keep = protected_mask.clone()
        keep[0] = True  # always keep <cls>
        protected_tail = torch.nonzero(keep[1:], as_tuple=False).flatten() + 1
        if len(protected_tail) > max_length - 1:
            raise ValueError(
                f"scGPT protected gene count ({len(protected_tail)}) exceeds max_length-1 ({max_length - 1})."
            )

        selected_tail = protected_tail
        remaining_slots = max_length - 1 - len(selected_tail)
        if remaining_slots > 0:
            candidate_tail = torch.nonzero(~keep[1:], as_tuple=False).flatten() + 1
            if len(candidate_tail) > remaining_slots:
                sampled = candidate_tail[torch.randperm(len(candidate_tail), device=device)[:remaining_slots]]
            else:
                sampled = candidate_tail
            selected_tail = torch.cat([selected_tail, sampled], dim=0)

        selected = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), torch.sort(selected_tail).values],
            dim=0,
        )
        genes = genes[selected]
        expressions = expressions[selected]
        if len(genes) < max_length:
            genes, expressions = self.base._pad(genes, expressions, max_length)
        return genes, expressions


def _map_pert_var_idx_to_kept_columns(
    pert_var_idx: Optional[np.ndarray],
    keep_mask: np.ndarray,
    n_cells: int,
) -> List[List[int]]:
    rows: List[List[int]] = [[] for _ in range(n_cells)]
    if pert_var_idx is None:
        return rows
    pert_var_idx = np.asarray(pert_var_idx, dtype=np.int64)
    kept_cols = np.where(keep_mask)[0]
    old_to_new = np.full(len(keep_mask), -1, dtype=np.int64)
    old_to_new[kept_cols] = np.arange(len(kept_cols), dtype=np.int64)
    for i in range(min(n_cells, pert_var_idx.shape[0])):
        mapped: List[int] = []
        seen: set[int] = set()
        for x in np.asarray(pert_var_idx[i]).ravel():
            j = int(x)
            if j < 0 or j >= len(old_to_new):
                continue
            jj = int(old_to_new[j])
            if jj < 0 or jj in seen:
                continue
            seen.add(jj)
            mapped.append(jj)
        rows[i] = mapped
    return rows


def encode(
    adata: ad.AnnData,
    *,
    model_dir: Optional[str] = None,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    max_length: int = 1200,
    batch_size: int = 64,
    gene_col: str = "feature_name",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Returns:
        (n_cells, embsize) float32, metadata dict

    Notes:
        * scGPT's ``DataCollator`` discretizes expression via ``do_binning=True``
          (rank-based, 51 bins via ``np.quantile`` cutoffs — see
          ``third_party/scGPT-main/scgpt/preprocess.py`` lines 293–302), so the
          encoder is **invariant to any strictly monotonic transform** (e.g.
          log1p) of the expression values at *binned* positions.
        * The ``<cls>`` position (index 0) is excluded from binning and carries
          the constant ``pad_value`` from the model config.
        * ``force_pert`` toggles whether ``obsm['pert_var_idx']`` (when present)
          participates in truncation/sampling as a **protected-gene coverage**
          mask only. Values still come from ``adata.X``; there is no obs-string
          perturbation path and no separate condition token stream.
    """
    _ensure_scgpt_path()
    _patch_flash_mha()
    _patch_encoder_layer_forward()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _flash_state_reset(dev)

    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    model_dir = Path(
        model_dir
        or os.environ.get("LATENT_BENCH_SCGPT_MODEL_DIR", str(paths.pretrained_root() / "scgpt"))
    )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "scGPT model dir missing. Set LATENT_BENCH_SCGPT_MODEL_DIR or pass model_dir= "
            "(directory containing vocab.json, args.json, best_model.pt)."
        )

    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    if gene_col != "index" and gene_col not in adata.var.columns:
        # Some datasets (e.g. sciplex chemical perturbation) keep gene symbols
        # directly in var_names and do not carry feature_name. Fall back to
        # index rather than failing before vocab intersection.
        gene_col = "index"

    if gene_col == "index":
        gene_names = list(adata.var.index)
    else:
        gene_names = adata.var[gene_col].tolist()
    pert_var_idx_src = adata.obsm.get("pert_var_idx", None)
    id_in_vocab = np.array([vocab[g] if g in vocab else -1 for g in gene_names], dtype=int)
    keep_mask = id_in_vocab >= 0
    mapped_protected_rows = _map_pert_var_idx_to_kept_columns(pert_var_idx_src, keep_mask, adata.n_obs)
    adata = adata[:, keep_mask].copy()
    genes = [g for g, k in zip(gene_names, keep_mask.tolist()) if k]
    gene_ids = np.array(vocab(genes), dtype=int)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    vocab.set_default_index(vocab["<pad>"])

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    try:
        sd = torch.load(model_file, map_location=dev, weights_only=False)
    except TypeError:
        sd = torch.load(model_file, map_location=dev)
    load_pretrained(model, sd, verbose=False)
    model.to(dev)
    model.eval()

    count_matrix = adata.X
    count_matrix = count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids
            self.protected_rows = mapped_protected_rows

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            cls_id = vocab["<cls>"]
            pad_val = float(model_configs["pad_value"])
            genes_list: List[int] = []
            values_list: List[float] = []
            protected_mask: List[bool] = []
            genes_list.append(cls_id)
            values_list.append(pad_val)
            protected_mask.append(True)

            protected_idx = self.protected_rows[idx] if force_pert else []
            protected_set = {int(j) for j in protected_idx if 0 <= int(j) < len(row)}
            candidate_idx = protected_set.union(int(j) for j in np.nonzero(row)[0].tolist())
            ordered_idx = sorted(candidate_idx, key=lambda j: (-float(row[j]), int(j)))
            for j in ordered_idx:
                genes_list.append(int(self.gene_ids[j]))
                values_list.append(float(row[j]))
                protected_mask.append(j in protected_set)

            g = torch.tensor(genes_list, dtype=torch.long)
            v = torch.tensor(values_list, dtype=torch.float32)
            return {
                "id": idx,
                "genes": g,
                "expressions": v,
                "protected_mask": torch.tensor(protected_mask, dtype=torch.bool),
            }

    base_collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    collator = ProtectedGeneDataCollator(base=base_collator)
    dataset = _Dataset()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size),
        pin_memory=True,
    )

    embsize = model_configs["embsize"]
    cell_embeddings = np.zeros((len(dataset), embsize), dtype=np.float32)
    count = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
        for data_dict in loader:
            input_gene_ids = data_dict["gene"].to(dev)
            src_key_padding_mask = input_gene_ids.eq(vocab[model_configs["pad_token"]])
            out = model._encode(
                input_gene_ids,
                data_dict["expr"].to(dev),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )
            emb = out[:, 0, :].cpu().numpy()
            cell_embeddings[count : count + len(emb)] = emb
            count += len(emb)

    cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)

    pert_var_idx_present = pert_var_idx_src is not None
    meta = {
        "encoder_role": "ExpressionOnlyEncoder",
        "hidden_dim": int(embsize),
        "input_is_log1p": bool(input_is_log1p),
        "force_pert": bool(force_pert),
        "pert_var_idx_present": bool(pert_var_idx_present),
        "pert_source": "obsm_pert_var_idx" if pert_var_idx_present else None,
        "force_pert_effective": bool(force_pert and pert_var_idx_present),
        **_flash_attn_meta(),
    }
    if force_pert and pert_var_idx_present:
        ks = [len(x) for x in mapped_protected_rows]
        meta["pert_kept_histogram"] = histogram_pert_kept(ks)

    return cell_embeddings, meta
