"""
State (SE) cell embedding adapter (latent_bench protocol).

Forward data is ``adata.X`` only. ``obsm['pert_var_idx']`` may be used **only**
when ``force_pert=True`` as a per-cell protected-gene mask during sentence
packing so those columns are not sampled away. The adapter strips
``obs['perturbation'|'condition'|'gene']`` before calling third-party inference so
State never falls back to string-parsed condition genes; with ``force_pert=False``
those columns are still stripped and ``obsm['pert_var_idx']`` is removed from the
on-disk copy so the collator follows the **expression-only** sentence path (no
batch-union condition genes).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import paths

from .._common import histogram_pert_kept




def _ensure_state_path() -> Path:
    src = paths.third_party_root() / "state" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def encode(
    adata: ad.AnnData,
    *,
    checkpoint: Optional[str] = None,
    protein_embeds_path: Optional[str] = None,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    batch_size: Optional[int] = None,
    emb_key: str = "X_emb",
    prefetch_batches: int = 2,
    n_collate_workers: int = 1,
    dataloader_num_workers: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """Encode ``adata.X`` with State (SE-600M).

    Parameters
    ----------
    input_is_log1p
        Declares the magnitude convention of ``adata.X``. latent_bench's
        canonical preprocessing is ``log1p(normalize_total)`` (so
        ``input_is_log1p=True``). This value is enforced deterministically:
        we monkey-patch ``VCIDatasetSentenceCollator.is_raw_integer_counts``
        on the State third_party loader so that
        ``sample_cell_sentences`` takes the matching branch regardless of
        the built-in magnitude heuristic (loader.py ~lines 579-601,
        ``RAW_COUNT_HEURISTIC_THRESHOLD=35``,
        ``EXPONENTIATED_UMIS_LIMIT=5_000_000``). Pass ``False`` only when
        feeding raw UMI counts.

    Returns:
        embeddings (n_cells, D) float32
        meta dict with ``pert_kept_histogram`` when applicable and
        ``input_is_log1p`` always set.
    """
    _ensure_state_path()
    from .per_cell_collator_patch import (
        install as install_per_cell_collator,
        install_input_mode,
    )

    has_pert_matrix = "pert_var_idx" in adata.obsm
    # Always install: replaces batch-union condition sampling with per-cell
    # ``pert_var_matrix`` / ``sampling_pert_per_cell`` routing (pure expression when both absent).
    install_per_cell_collator()

    install_input_mode(is_log1p=bool(input_is_log1p))

    ckpt = checkpoint or os.environ.get(
        "LATENT_BENCH_STATE_CKPT",
        "",
    )
    if not ckpt or not Path(ckpt).is_file():
        raise FileNotFoundError(
            "State checkpoint not found. Pass checkpoint=... or set "
            "LATENT_BENCH_STATE_CKPT to a path of SE-600M .ckpt (or equivalent)."
        )

    from state.emb.inference import Inference
    from state.emb.utils import get_precision_config
    import torch

    pe_path = protein_embeds_path or os.environ.get("LATENT_BENCH_STATE_PE", "")
    if not pe_path:
        cand = Path(ckpt).parent / "protein_embeddings.pt"
        if cand.is_file():
            pe_path = str(cand)
    pe = None
    if pe_path and Path(pe_path).is_file():
        pe = torch.load(pe_path, map_location="cpu", weights_only=False)
    inf = Inference(protein_embeds=pe)
    inf.load_model(ckpt)

    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        adata_in = adata.copy()
        # Expression-only: strip columns State's inference would use for obs-based
        # condition-gene sampling (see third_party state/emb/inference.py).
        for col in ("perturbation", "condition", "gene"):
            if col in adata_in.obs.columns:
                del adata_in.obs[col]
        if not (force_pert and has_pert_matrix) and "pert_var_idx" in adata_in.obsm:
            del adata_in.obsm["pert_var_idx"]

        adata_in.write_h5ad(tmp_path)
        pert_hist: dict = {
            "input_is_log1p": bool(input_is_log1p),
            "force_pert": bool(force_pert),
            "force_pert_effective": bool(force_pert and has_pert_matrix),
            "per_cell_collator_patch": True,
        }
        if has_pert_matrix:
            m = np.asarray(adata.obsm["pert_var_idx"], dtype=np.int32)
            counts = [int(np.sum(r >= 0)) for r in m]
            pert_hist["pert_kept_histogram"] = histogram_pert_kept(counts)

        emb = inf.encode_adata(
            tmp_path,
            output_adata_path=None,
            emb_key=emb_key,
            dataset_name="latent_bench",
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
            n_collate_workers=n_collate_workers,
            dataloader_num_workers=dataloader_num_workers,
        )
        emb = np.asarray(emb, dtype=np.float32)
        pert_hist["encoder_role"] = "ExpressionOnlyEncoder"
        pert_hist["hidden_dim"] = int(emb.shape[1])
        pert_hist["pert_var_idx_present"] = bool(has_pert_matrix)
        pert_hist["pert_source"] = "obsm_pert_var_idx" if has_pert_matrix else None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return emb, pert_hist
