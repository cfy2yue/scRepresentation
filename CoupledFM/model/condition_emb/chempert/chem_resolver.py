"""Resolve chemical perturbation vectors into ``ConditionMetadata.chem_emb_list``."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from model.condition_emb.genepert.chem_embedding_hook import ChemEmbeddingCache, parse_chem_source_fields

from .drug_cache import DrugBackend, DrugEmbeddingCache, deterministic_standard_normal_vec

if TYPE_CHECKING:
    from model.condition_emb.genepert.perturbation import ConditionMetadata

_RESOLVE_HIT_LOGGED = False


def _split_cocktail_obs_values(raw: str) -> List[str]:
    s = " ".join(str(raw).strip().split())
    if not s:
        return []
    for sep in ("|", ";"):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def chem_keys_for_metadata(meta: "ConditionMetadata") -> List[str]:
    obs_val = getattr(meta, "chem_obs_value", None)
    if isinstance(obs_val, str) and obs_val.strip():
        return _split_cocktail_obs_values(obs_val)
    cs = meta.chem_source
    if isinstance(cs, str) and cs.strip():
        return _keys_from_chem_source(cs)
    return []


def _keys_from_chem_source(chem_source: str) -> List[str]:
    parts = [p.strip() for p in str(chem_source).split("|") if p.strip()]
    keys: List[str] = []
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k_l = k.strip().lower()
        v = v.strip()
        if not v:
            continue
        if k_l in ("drug", "compound", "chemical"):
            keys.append(v)
        elif k_l == "smiles":
            keys.append(v)
        elif k_l in ("chembl_id", "chembl-id", "chembl"):
            keys.append(v)
        elif k_l in ("inchikey", "inchi_key"):
            keys.append(v)
    if not keys:
        sm, ch, dr = parse_chem_source_fields("|".join(parts) if parts else chem_source)
        if sm:
            keys.append(sm.strip())
        if ch:
            keys.append(ch.strip())
        if dr:
            keys.append(dr.strip())
    out: List[str] = []
    seen = set()
    for kk in keys:
        s = kk.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_chemical_embed_backend(cfg: Any, *, fallback_dim: int) -> DrugBackend:
    ddir = str(getattr(cfg, "drug_emb_cache_dir", "") or "").strip()
    return DrugEmbeddingCache.from_dir_or_random(ddir if ddir else None, dim=int(fallback_dim))


def _maybe_warn_hit() -> None:
    global _RESOLVE_HIT_LOGGED  # noqa: PLW0603
    if not _RESOLVE_HIT_LOGGED:
        warnings.warn(
            "Chemical embedding cache matched at least one key (suppressing further hit logs per process).",
            UserWarning,
            stacklevel=3,
        )
        _RESOLVE_HIT_LOGGED = True


def _legacy_single_vector(meta: "ConditionMetadata", root: Path) -> Optional[np.ndarray]:
    cache = ChemEmbeddingCache(root)
    sm, ch, dr = parse_chem_source_fields(meta.chem_source or "")
    tries = (
        (sm.strip() if sm else ""),
        (ch.strip() if ch else ""),
        (dr.strip() if dr else ""),
    )
    for raw in tries:
        if not raw:
            continue
        vec = cache.lookup_embedding(raw)
        if vec is not None:
            return vec.astype(np.float32, copy=False)
    return None


def resolve_chemical_embeddings_for_metadata(
    meta: "ConditionMetadata",
    cfg: Any,
    *,
    backend: DrugBackend,
    legacy_chem_dirs: Optional[Sequence[str]] = None,
    max_keys: int = 4,
) -> List[np.ndarray]:
    keys = chem_keys_for_metadata(meta)
    if not keys:
        return []

    kmax = max(1, int(max_keys))
    orig_len = len(keys)
    keys = keys[:kmax]
    if orig_len > kmax:
        warnings.warn(
            f"truncating cocktail keys len={orig_len} to max_chem_keys={kmax}",
            UserWarning,
            stacklevel=2,
        )

    hits: List[bool] = []
    out: List[np.ndarray] = []
    dim = int(getattr(backend, "embed_dim"))
    any_hit = False

    for key in keys:
        vec, hit = backend.lookup(key)
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != dim:
            v = deterministic_standard_normal_vec(str(key), dim)
        hits.append(bool(hit))
        if hit:
            any_hit = True
        out.append(v)

    if legacy_chem_dirs and keys and not any(hits):
        lv: Optional[np.ndarray] = None
        for d in legacy_chem_dirs:
            rp = Path(str(d)).expanduser()
            if not rp.is_dir():
                continue
            lv = _legacy_single_vector(meta, rp)
            if lv is not None:
                break
        if lv is not None and len(out) > 0:
            lv = lv.reshape(-1).astype(np.float32, copy=False)
            ld = int(lv.size)
            if ld == dim:
                out[0] = lv
                any_hit = True

    if any_hit:
        _maybe_warn_hit()

    return out




def resolve_first_chemical_embedding(meta: "ConditionMetadata", cfg: Any) -> Optional[np.ndarray]:
    """Backward-compatible hook: return the first molecule vector slot (legacy tests)."""
    if not getattr(cfg, "pert_chem_enabled", False):
        return None
    backend = getattr(cfg, "_chem_embed_backend", None)
    dim_fb = int(getattr(cfg, "chem_fallback_embed_dim", 512) or 512)
    if backend is None:
        backend = load_chemical_embed_backend(cfg, fallback_dim=max(8, dim_fb))
    leg = []
    lc = str(getattr(cfg, "chem_emb_source_dir", "") or "").strip()
    if lc:
        leg.append(lc)
    vals = resolve_chemical_embeddings_for_metadata(
        meta, cfg, backend=backend, legacy_chem_dirs=(leg if leg else None),
    )
    return vals[0] if vals else None
