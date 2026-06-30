"""Perturbation string parsing, type normalization, batch tensor helpers."""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .gene_cache import GeneEmbeddingCache

# ---------------------------------------------------------------------------
# Small discrete table (UnifiedConditionEncoder `type_scale` aligned to these ids).
#
# 0=null, 1=ko, 2=crispri (includes knockdown), 3=cas13, 4=crispra (includes overexpression),
# 5=drug.  Unknown textual types are recorded via :func:`seen_unknown_perturbation_types`
# then mapped to PERT_TYPE_NULL for stable checkpoints.
PERT_TYPE_NULL = 0
PERT_TYPE_KNOCKOUT = 1
PERT_TYPE_CRISPRI = 2
PERT_TYPE_CAS13 = 3
PERT_TYPE_CRISPRA = 4
PERT_TYPE_DRUG = 5

_NUM_PERT_TYPES = 6

_UNKNOWN_PERT_RAW_M: set[str] = set()
_UNKNOWN_LOGGED_ONCE_M: set[str] = set()


def reset_unknown_perturbation_types() -> None:
    """Clear module-global unknown buffers (unittest only)."""
    _UNKNOWN_PERT_RAW_M.clear()
    _UNKNOWN_LOGGED_ONCE_M.clear()


def seen_unknown_perturbation_types() -> FrozenSet[str]:
    """Process-local immutable view of unrecognized ``perturbation_type`` strings."""
    return frozenset(_UNKNOWN_PERT_RAW_M)


def _record_unknown_perturbation_type(raw: Optional[str]) -> None:
    s = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
    s = " ".join(s.strip().split())
    if not s:
        return
    if s not in _UNKNOWN_LOGGED_ONCE_M:
        warnings.warn(
            f"Unrecognized perturbation_type `{s}` -> PERT_TYPE_NULL; "
            f"tracked in perturbation.seen_unknown_perturbation_types().",
            UserWarning,
            stacklevel=3,
        )
        _UNKNOWN_LOGGED_ONCE_M.add(s)
    _UNKNOWN_PERT_RAW_M.add(s)

_CONTROL_TOKENS = frozenset(
    {
        "",
        "nan",
        "none",
        "null",
        "control",
        "ctrl",
        "ntc",
        "non-targeting",
        "nontargeting",
        "pbs",
        "unperturbed",
        "wildtype",
        "wt",
    }
)

# Aliases -> canonical token used in _CANON_TO_ID
_SYN_TO_CANON: dict[str, str] = {
    "ko": "ko",
    "knock-out": "ko",
    "knock_out": "ko",
    "knockout": "ko",
    "crisprko": "ko",
    "crispriko": "ko",
    "crispr-ko": "ko",
    "cas9 knockout": "ko",
    "cas9_knockout": "ko",
    "cas9ko": "ko",
    "gene knockout": "ko",
    "knockdown": "crispri",
    "kd": "crispri",
    "rna knockdown": "crispri",
    "shrna": "crispri",
    "sirna": "crispri",
    "crispri": "crispri",
    "crispr-i": "crispri",
    "crispr_i": "crispri",
    "activation": "crispra",
    "crispra": "crispra",
    "crispr-a": "crispra",
    "crispr_a": "crispra",
    "gain of function": "crispra",
    "overexpress": "crispra",
    "overexpression": "crispra",
    "overexpr": "crispra",
    "oe": "crispra",
    "cas13": "cas13",
    "cas-13": "cas13",
    "chemical": "drug",
    "compound": "drug",
    "drug": "drug",
    "small molecule": "drug",
    "smi": "drug",
    "smiles": "drug",
    "smile": "drug",
    "smiles_string": "drug",
    "molecule": "drug",
    "small-molecule": "drug",
}

_CANON_TO_ID = {
    "": PERT_TYPE_NULL,
    "ko": PERT_TYPE_KNOCKOUT,
    "crispri": PERT_TYPE_CRISPRI,
    "cas13": PERT_TYPE_CAS13,
    "crispra": PERT_TYPE_CRISPRA,
    "drug": PERT_TYPE_DRUG,
}

# Back-compat names (old callers / tests)
PERT_TYPE_KNOCKDOWN = PERT_TYPE_CRISPRI
PERT_TYPE_OVEREXPR = PERT_TYPE_CRISPRA


def normalize_gene_symbol(symbol: str) -> str:
    """Uppercase stripped HGNC-style symbol for lookups (matches ``GeneEmbeddingCache``)."""
    return symbol.strip().upper()


def canonical_sorted_gene_tuple(*parts: str) -> Tuple[str, ...]:
    """Dedupe by normalized symbol, sort lexicographically (uppercase keys), return uppercase tokens."""
    seen: dict[str, None] = {}
    for p in parts:
        k = normalize_gene_symbol(p) if p else ""
        if k and k not in seen:
            seen[k] = None
    return tuple(sorted(seen.keys()))


def canonicalize_gene_list_from_raw(
    gene_field: object,
    perturbation_field: Optional[object] = None,
) -> Tuple[str, ...]:
    """Parse ``gene`` / ``perturbation`` columns and return sorted unique gene symbols."""
    g1 = parse_perturbation_gene_strings(gene_field)
    g2 = parse_perturbation_gene_strings(perturbation_field) if perturbation_field is not None else ()
    genes = g1 if g1 else g2
    return canonical_sorted_gene_tuple(*genes)


def infer_nperts_from_obs(genes: Tuple[str, ...], raw_nperts: Optional[object]) -> int:
    """Reconcile ``obs['nperts']`` with parsed gene tokens (biFlow / legacy DE5000 fallbacks).

    * Missing or non-finite ``nperts`` → ``len(genes)``.
    * ``nperts == 0`` and no genes → control.
    * ``nperts == 0`` but genes present → trust gene list (column inconsistency).
    * Genes empty but ``nperts > 0`` → trust ``nperts`` (gene column missing formatting).
    * Both positive but disagree → prefer ``len(genes)`` when ``genes`` non-empty else ``nperts``.
    """
    gn = len(genes)
    if raw_nperts is None:
        return gn
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(raw_nperts, (np.floating, float)) and math.isnan(float(raw_nperts)):
            return gn
    except Exception:
        pass
    try:
        n = int(raw_nperts)
    except (TypeError, ValueError):
        return gn
    if n < 0:
        return gn
    if n == 0 and gn == 0:
        return 0
    if n == 0 and gn > 0:
        return gn
    if n > 0 and gn == 0:
        return n
    if n != gn:
        return gn if gn > 0 else n
    return n


def num_perturbation_types() -> int:
    return _NUM_PERT_TYPES


_WHITESPACE_RE = re.compile(r"\s+")


def _raw_to_clean_str(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(raw, (np.floating, np.integer)):
            x = raw.item()
            if isinstance(x, float) and math.isnan(x):
                return None
            raw = x
    except Exception:
        pass
    s = str(raw).strip()
    if not s or s.lower() in _CONTROL_TOKENS:
        return None
    ls = s.lower()
    if ls in ("nan", "none", "<na>"):
        return None
    return s


def parse_perturbation_gene_strings(raw: object) -> Tuple[str, ...]:
    """Parse perturbation genes from ``obs['gene']`` / ``obs['perturbation']`` style fields.

    Accepts formats like ``A+B``, comma lists, whitespace separated tokens.
    Control / NaN / empty -> empty tuple.

    Bare ``+/-`` separators are stripped; malformed empty tokens skipped.
    """
    s = _raw_to_clean_str(raw)
    if s is None:
        return ()

    normalized = _WHITESPACE_RE.sub(" ", s)

    sep_tokens = (
        normalized.replace(",", "+").replace("|", "+").replace(";", "+").replace("/", "+").replace("&", "+")
    )
    # split on "+" with optional surrounding space
    parts = []
    for chunk in sep_tokens.split("+"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sub = _WHITESPACE_RE.split(chunk)
        for p in sub:
            p = p.strip()
            if not p or p.lower() in _CONTROL_TOKENS:
                continue
            parts.append(p)
    return tuple(parts)


def canonicalize_perturbation_type(raw: Optional[object]) -> str:
    """Return canonical type key: ``ko|crispri|cas13|crispra|drug`` or ``""`` (null)."""
    s = _raw_to_clean_str(raw)
    if s is None:
        return ""
    key = s.strip().lower()
    if key in ("ko", "knockout"):
        return "ko"
    if key in ("crispri", "crispra", "cas13", "drug"):
        return key
    if key in _SYN_TO_CANON:
        return _SYN_TO_CANON[key]
    alnum = re.sub(r"[^a-z0-9]+", "", key)
    token_map = {
        "crispri": "crispri",
        "crispra": "crispra",
        "crisprko": "ko",
        "crispy": "ko",
        "sirna": "crispri",
        "shrna": "crispri",
        "cas13": "cas13",
    }
    for tok, canon in token_map.items():
        if tok in alnum:
            if tok == "crispy" and "crispri" in alnum:
                continue
            return canon
    if "cas13" in alnum or "cas13" in key.replace(" ", ""):
        return "cas13"
    if "cas9" in key.replace(" ", "") and "knockout" in key:
        return "ko"
    if "knockout" in key and "knockdown" not in key:
        return "ko"
    if "knockdown" in key:
        return "crispri"
    if "overexpression" in key or "overexpress" in key:
        return "crispra"
    if "sirna" in key or "shrna" in key:
        return "crispri"
    return ""


def perturbation_type_to_id(raw: Optional[object]) -> int:
    """Map user text to a small int table; 0 = null / unknown / unconditioned."""
    raw_clean = _raw_to_clean_str(raw)
    c = canonicalize_perturbation_type(raw)
    if c != "":
        return int(_CANON_TO_ID[c])
    if raw_clean is None:
        return PERT_TYPE_NULL
    _record_unknown_perturbation_type(raw_clean)
    return PERT_TYPE_NULL


@dataclass
class ConditionMetadata:
    """One cell / sample worth of perturbation metadata."""

    genes: Tuple[str, ...]
    perturbation_type_raw: Optional[str] = None
    combo_id: int = 0
    nperts_obs: Optional[int] = None
    # Chemical: single vector (legacy), or list for cocktail (same dim per vector).
    chem_emb: Optional[np.ndarray] = None
    chem_emb_list: Optional[List[np.ndarray]] = None
    chem_source: Optional[str] = None
    # When datasets wire ``chem_obs_column``, stash the trimmed obs cell text as cache key(s)
    chem_obs_value: Optional[str] = None

    def resolved_nperts(self) -> int:
        return infer_nperts_from_obs(self.genes, self.nperts_obs)

    @classmethod
    def from_obs_fields(
        cls,
        gene_field: object,
        *,
        perturbation_field: Optional[object] = None,
        perturbation_type_field: Optional[object] = None,
        nperts_field: Optional[object] = None,
        combo_id: int = 0,
        sort_genes: bool = True,
    ) -> "ConditionMetadata":
        """Build metadata from biFlow-style ``obs`` columns (see ``docs/data_contract.md``).

        * **Genes**: prefer ``gene`` parsing; if empty, parse ``perturbation`` (e.g. ``A + B``, ``control``).
        * **Order**: when ``sort_genes`` (default), genes are deduped and sorted for stable combo keys /
          embedding lookups; set ``False`` to keep parse order (single-gene rows unchanged).
        * **Type**: raw string preserved for ``perturbation_type_to_id`` (0 = null / unknown).
        * **nperts_obs**: stored for ``infer_nperts_from_obs`` reconciliation in ``PerturbationBatch``.
        """
        if sort_genes:
            genes = canonicalize_gene_list_from_raw(gene_field, perturbation_field)
        else:
            g1 = parse_perturbation_gene_strings(gene_field)
            g2 = parse_perturbation_gene_strings(perturbation_field) if perturbation_field is not None else ()
            genes = g1 if g1 else g2
        ptype = None
        if perturbation_type_field is not None:
            ps = _raw_to_clean_str(perturbation_type_field)
            ptype = ps
        n_obs: Optional[int] = None
        if nperts_field is not None:
            try:
                import pandas as pd  # noqa: WPS433

                if pd.isna(nperts_field):
                    n_obs = None
                else:
                    import numpy as np  # noqa: WPS433

                    if isinstance(nperts_field, (np.integer,)):
                        n_obs = int(nperts_field)
                    elif isinstance(nperts_field, (float, np.floating)) and not math.isnan(float(nperts_field)):
                        n_obs = int(nperts_field)
                    elif isinstance(nperts_field, int) and not isinstance(nperts_field, bool):
                        n_obs = int(nperts_field)
                    elif isinstance(nperts_field, str) and nperts_field.strip():
                        n_obs = int(float(nperts_field.strip()))
            except Exception:
                n_obs = None
        return cls(
            genes=genes,
            perturbation_type_raw=ptype,
            combo_id=int(combo_id),
            nperts_obs=n_obs,
            chem_emb=None,
            chem_emb_list=None,
            chem_source=None,
            chem_obs_value=None,
        )


@dataclass
class PerturbationBatch:
    """Batched perturbation tensors for encoders."""

    pert_gene_ids: torch.Tensor  # (B, K) int64
    pert_mask: torch.Tensor  # (B, K) bool
    pert_type_id: torch.Tensor  # (B,) int64
    nperts: torch.Tensor  # (B,) int64
    combo_ids: Optional[torch.Tensor] = None  # (B,) int64 optional
    # Chemical embeddings: ``(B, K_chem, D)`` preferred; legacy ``(B, D)`` rows-only.
    chem_emb: Optional[torch.Tensor] = None  # float32 optional
    # Mask: ``(B, K_chem)`` bool/float preferred; legacy ``(B,)`` interpolation weight.
    chem_mask: Optional[torch.Tensor] = None

    def __iter__(self):
        """Legacy iteration: yield **five** tensors (omit chem slots).

        Older call sites unpacked ``gid, mk, tid, npt, cid = perturbation_batch``;
        unpacking seven values requires :meth:`as_tuple_full`.
        """
        return iter(self.as_tuple_legacy())

    def as_tuple_legacy(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """``(pert_gene_ids, pert_mask, pert_type_id, nperts, combo_ids)`` for legacy dataloaders."""
        return (
            self.pert_gene_ids,
            self.pert_mask,
            self.pert_type_id,
            self.nperts,
            self.combo_ids,
        )

    def as_tuple_full(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Seven slots including optional ``chem_emb`` / ``chem_mask`` (either may be ``None``).

        ``chem_emb`` may be ``(B, K_chem, D)`` or legacy ``(B, D)``; ``chem_mask`` matched shape.
        """
        return (
            self.pert_gene_ids,
            self.pert_mask,
            self.pert_type_id,
            self.nperts,
            self.combo_ids,
            self.chem_emb,
            self.chem_mask,
        )

    def to_training_tuple(self) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward-compatible alias for :meth:`as_tuple_full`."""
        return self.as_tuple_full()

    @classmethod
    def from_metadata_list(
        cls,
        rows: Sequence[ConditionMetadata],
        cache: GeneEmbeddingCache,
        *,
        max_genes: int = 8,
        max_chem_slots: int = 4,
        device: Optional[torch.device] = None,
        combo_ids_explicit: Optional[Sequence[int]] = None,
    ) -> "PerturbationBatch":
        """Vectorize ConditionMetadata rows using ``GeneEmbeddingCache`` lookups."""
        b = len(rows)
        device = device or torch.device("cpu")
        k = max(1, int(max_genes))

        gid = torch.zeros(b, k, dtype=torch.long, device=device)
        mask = torch.zeros(b, k, dtype=torch.bool, device=device)
        tid = torch.zeros(b, dtype=torch.long, device=device)
        npt = torch.zeros(b, dtype=torch.long, device=device)

        combos: Optional[List[int]] = None
        if combo_ids_explicit is not None:
            if len(combo_ids_explicit) != b:
                raise ValueError("combo_ids_explicit length must match batch")
            combos = [int(x) for x in combo_ids_explicit]

        warn_gene_trunc = {"ok": False}

        for i, row in enumerate(rows):
            tid[i] = perturbation_type_to_id(row.perturbation_type_raw)
            npt[i] = row.resolved_nperts()
            if len(row.genes) > k:
                if not warn_gene_trunc["ok"]:
                    warnings.warn(
                        f"truncating gene list len={len(row.genes)} to max_genes={k}; "
                        f"row.nperts ({row.resolved_nperts()}) may exceed mask sum",
                        UserWarning,
                        stacklevel=2,
                    )
                    warn_gene_trunc["ok"] = True
            genes_take = list(row.genes)[:k]
            for j, sym in enumerate(genes_take):
                gid[i, j] = cache.lookup(sym)
                mask[i, j] = True

        combo_t: Optional[torch.Tensor] = None
        if combos is None:
            combo_t = torch.tensor([r.combo_id for r in rows], dtype=torch.long, device=device)
        else:
            combo_t = torch.tensor(combos, dtype=torch.long, device=device)

        kc = max(1, int(max_chem_slots))
        chem_emb_t: Optional[torch.Tensor] = None
        chem_mask_t: Optional[torch.Tensor] = None
        d_chem: Optional[int] = None
        stacked_rows: List[List[np.ndarray]] = []

        def _accum_vec(cols_out: List[np.ndarray], arr_in: np.ndarray) -> None:
            nonlocal d_chem
            arr = np.asarray(arr_in, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return
            di = int(arr.shape[0])
            if d_chem is None:
                d_chem = di
            elif di != d_chem:
                raise ValueError(
                    "Inconsistent chem_emb dimension across batch "
                    f"(expected {d_chem}, got {di})"
                )
            cols_out.append(arr)

        warn_trunc = {"ok": False}

        for row in rows:
            cols: List[np.ndarray] = []
            if row.chem_emb_list:
                for ce in row.chem_emb_list:
                    _accum_vec(cols, np.asarray(ce, dtype=np.float32))
                    if len(cols) >= kc:
                        if not warn_trunc["ok"]:
                            warnings.warn(
                                f"truncating cocktail to max_chem_slots={kc}",
                                UserWarning,
                                stacklevel=2,
                            )
                            warn_trunc["ok"] = True
                        break
            elif row.chem_emb is not None:
                _accum_vec(cols, np.asarray(row.chem_emb, dtype=np.float32))
            stacked_rows.append(cols)

        if d_chem is not None:
            chem_emb_t = torch.zeros(b, kc, d_chem, dtype=torch.float32, device=device)
            chem_mask_t = torch.zeros(b, kc, dtype=torch.float32, device=device)
            for i, cols in enumerate(stacked_rows):
                for j, arr in enumerate(cols[:kc]):
                    chem_emb_t[i, j].copy_(torch.from_numpy(arr).to(device=device))
                    chem_mask_t[i, j] = 1.0

        return cls(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=tid,
            nperts=npt,
            combo_ids=combo_t,
            chem_emb=chem_emb_t,
            chem_mask=chem_mask_t,
        )


def perturbation_batch_to_device(pb: PerturbationBatch, device: torch.device) -> PerturbationBatch:
    return PerturbationBatch(
        pert_gene_ids=pb.pert_gene_ids.to(device),
        pert_mask=pb.pert_mask.to(device),
        pert_type_id=pb.pert_type_id.to(device),
        nperts=pb.nperts.to(device),
        combo_ids=None if pb.combo_ids is None else pb.combo_ids.to(device),
        chem_emb=None if pb.chem_emb is None else pb.chem_emb.to(device),
        chem_mask=None if pb.chem_mask is None else pb.chem_mask.to(device),
    )


def unpack_perturbation_tuple(
    pb: Sequence[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Split a dataloader perturbation tuple into core fields + optional chem slots."""
    n = len(pb)
    if n == 5:
        gid, mk, tid, npt, cid = pb
        return gid, mk, tid, npt, cid, None, None
    if n == 7:
        gid, mk, tid, npt, cid, cem, cmask = pb
        return gid, mk, tid, npt, cid, cem, cmask
    raise ValueError(f"perturbation tuple must have length 5 or 7, got {n}")


def perturbation_tuple_to_device(
    pb: Tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """``Tensor.to(device)`` for each tensor in a 5- or 7-tuple."""
    gid, mk, tid, npt, cid, cem, cmask = unpack_perturbation_tuple(pb)
    cid_dev = None if cid is None else cid.to(device, non_blocking=non_blocking)
    out5: Tuple[torch.Tensor, ...] = (
        gid.to(device, non_blocking=non_blocking),
        mk.to(device, non_blocking=non_blocking),
        tid.to(device, non_blocking=non_blocking),
        npt.to(device, non_blocking=non_blocking),
        cid_dev,  # type: ignore[assignment]
    )
    if cem is None:
        return out5 + (None, None)
    return out5 + (
        cem.to(device, non_blocking=non_blocking),
        None
        if cmask is None
        else cmask.to(device, non_blocking=non_blocking),
    )


def perturbation_tuple_slice_rows(
    pb: Tuple[torch.Tensor, ...],
    start: int,
    end: int,
) -> Tuple[torch.Tensor, ...]:
    """Row-slice ``[start:end)`` for each tensor in a perturbation tuple."""
    gid, mk, tid, npt, cid, cem, cmask = unpack_perturbation_tuple(pb)
    base = (
        gid[start:end].contiguous(),
        mk[start:end].contiguous(),
        tid[start:end].contiguous(),
        npt[start:end].contiguous(),
        None if cid is None else cid[start:end].contiguous(),
    )
    if cem is None:
        return base + (None, None)
    return base + (
        cem[start:end].contiguous(),
        None if cmask is None else cmask[start:end].contiguous(),
    )
