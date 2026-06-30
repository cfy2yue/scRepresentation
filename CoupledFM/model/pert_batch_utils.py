"""Helpers for raw/latent perturbation_batch plumbing (train / infer / eval)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from model.condition_emb.genepert.perturbation import (
    ConditionMetadata,
    PerturbationBatch,
    perturbation_tuple_slice_rows,
    perturbation_tuple_to_device,
)
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache


def unpack_training_batch(
    batch: Tuple[Any, ...],
) -> Tuple[
    Tuple[Any, ...],
    Optional[Tuple[torch.Tensor, ...]],
    Any,
]:
    """Split CoupledFMDataset batch into core fields, optional perturbation tuple, latent slot.

    * **12** items: ``(..., dx_prior_t, latent_data)`` — legacy, no perturbation tensors.
    * **13** items: ``(..., dx_prior_t, perturbation_batch, latent_data)``.
    """
    n = len(batch)
    if n == 12:
        return batch[:-1], None, batch[-1]
    if n == 13:
        return batch[:-2], batch[-2], batch[-1]
    raise ValueError(
        f"CoupledFMDataset batch must have length 12 or 13, got {n}"
    )


def perturbation_batch_to_device(
    pb: Optional[Tuple[torch.Tensor, ...]],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, ...]]:
    if pb is None:
        return None
    return perturbation_tuple_to_device(pb, device)


def slice_perturbation_batch(
    pb: Optional[Tuple[torch.Tensor, ...]],
    start: int,
    end: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Row-slice a perturbation tuple onto ``device`` (``[start:end)``)."""
    if pb is None:
        return None
    sl = perturbation_tuple_slice_rows(pb, start, end)
    return perturbation_tuple_to_device(sl, device)


def null_perturbation_batch(
    batch_size: int,
    max_genes: int,
    *,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Unconditional / CFG-drop rows (chem slots ``None``)."""
    b = int(batch_size)
    k = max(1, int(max_genes))
    z = torch.zeros(b, k, dtype=torch.long, device=device)
    m = torch.zeros(b, k, dtype=torch.bool, device=device)
    tid = torch.zeros(b, dtype=torch.long, device=device)
    npt = torch.zeros(b, dtype=torch.long, device=device)
    combo = torch.zeros(b, dtype=torch.long, device=device)
    return z, m, tid, npt, combo, None, None


def latent_fm_wants_perturbation(latent_fm: Any) -> bool:
    if latent_fm is None:
        return False
    cfg = getattr(latent_fm, "_config", None) or {}
    return bool(cfg.get("use_pert_condition", False))


def build_perturbation_batch_from_cond(
    cond: str,
    batch_size: int,
    *,
    cache: Optional[GeneEmbeddingCache],
    max_genes: int,
    device: torch.device,
    dataset: Optional[Any] = None,
    ds_name: Optional[str] = None,
    max_chem_slots: int = 4,
    chem_backend: Optional[Any] = None,
    chem_metainfo: Optional[Mapping[str, Any]] = None,
    chem_max_keys: Optional[int] = None,
    chem_legacy_dirs: Optional[Sequence[str]] = None,
    pert_chem_enabled: bool = False,
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]
]:
    """Build **seven-slot** perturbation tuple (``as_tuple_full``); ``None`` if unusable."""
    del chem_metainfo  # reserved for obs-string metainfo augmentation (dataset=None path).
    if cache is None or batch_size <= 0:
        return None
    slots = (
        int(getattr(dataset, "max_chem_keys", max_chem_slots))
        if dataset is not None
        else int(max_chem_slots)
    )
    if dataset is not None and ds_name is not None and hasattr(dataset, "metadata_for_condition"):
        meta = dataset.metadata_for_condition(str(ds_name), str(cond))
        if hasattr(dataset, "enrich_metadata_with_chem"):
            meta = dataset.enrich_metadata_with_chem(meta)
    elif dataset is not None and hasattr(dataset, "enrich_metadata_with_chem"):
        meta = ConditionMetadata.from_obs_fields(cond, perturbation_field=None)
        meta = dataset.enrich_metadata_with_chem(meta)
    else:
        meta = ConditionMetadata.from_obs_fields(cond, perturbation_field=None)
        if pert_chem_enabled and chem_backend is not None:
            from dataclasses import replace

            from model.condition_emb.chempert.chem_resolver import (
                resolve_chemical_embeddings_for_metadata,
            )

            mock_cfg = type("Cfg", (), {"chem_obs_column": ""})()
            mx = int(chem_max_keys) if chem_max_keys is not None else int(max_chem_slots)
            vecs = resolve_chemical_embeddings_for_metadata(
                meta,
                mock_cfg,
                backend=chem_backend,
                legacy_chem_dirs=list(chem_legacy_dirs) if chem_legacy_dirs else None,
                max_keys=mx,
            )
            meta = replace(meta, chem_emb_list=list(vecs))
    rows = [meta] * int(batch_size)
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=int(max_genes),
        max_chem_slots=int(slots),
        device=device,
    )
    if pb.combo_ids is None:
        return None
    return pb.as_tuple_full()


def try_load_gene_cache_for_inference(
    mc: Any,
    dc: Any,
) -> Optional[GeneEmbeddingCache]:
    """Load ``GeneEmbeddingCache`` when raw model needs pretrained pert embeddings."""
    if not getattr(mc, "use_pert_condition", False):
        return None
    pm = str(getattr(mc, "pert_embed_mode", "random_learned")).lower().strip()
    dcache = str(getattr(dc, "pert_gene_emb_cache_dir", "") or "").strip()
    if pm.startswith("pretrained"):
        if not dcache:
            return None
        try:
            return GeneEmbeddingCache(Path(dcache).expanduser())
        except FileNotFoundError as e:
            hint = (
                f"\n  Fix: run `python -m model.condition_emb.genepert.tools.export_gene_embedding_cache` "
                f"(see condition_emb/genepert/README.md) or sync gene_embeddings.npy into {dcache!r}."
            )
            raise FileNotFoundError(str(e) + hint) from e
    if dcache:
        try:
            return GeneEmbeddingCache(Path(dcache).expanduser())
        except FileNotFoundError as e:
            hint = (
                f"\n  Fix: place gene_embeddings.npy + gene_index under {dcache!r} "
                f"(see genepert README)."
            )
            raise FileNotFoundError(str(e) + hint) from e
    return None
