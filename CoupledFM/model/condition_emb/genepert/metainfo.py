"""Per-dataset perturbation_type fallback from ``metainfo.json`` (genepert layouts)."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from .perturbation import ConditionMetadata

_MIXED_DATASET_FALLBACK_WARNED: set[str] = set()


def load_dataset_metainfo(
    path: Union[str, Path],
    *,
    allow_missing: bool = True,
) -> Dict[str, str]:
    """Load ``{dataset_name: perturbation_type_raw}`` from a genepert-style JSON file.

    Expects a list of objects with ``dataset`` and ``perturbation_type`` keys.
    Mixed strings like ``CRISPRa+CRISPRi`` are stored verbatim for callers.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"pert metainfo not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            ds = item.get("dataset")
            pt = item.get("perturbation_type")
            if ds is None or pt is None:
                continue
            key = str(ds).strip()
            if not key:
                continue
            out[key] = str(pt).strip()
    return out


def resolve_perturbation_type(
    metainfo: Mapping[str, str],
    ds_name: str,
    row_type: Optional[str] = None,
    *,
    allow_override_by_row: bool = True,
) -> Optional[str]:
    """Prefer per-cell ``row_type`` when present; else dataset-level metainfo.

    Dataset-level entries containing ``+`` (e.g. mixed CRISPRa+CRISPRi) are **not**
    applied as a global condition type: returns ``None`` and emits a one-time warning
    per dataset so training relies on per-cell ``obs['perturbation_type']``.
    """
    ds_key = str(ds_name).strip()
    if allow_override_by_row and row_type is not None:
        rt = str(row_type).strip()
        if rt:
            return rt

    raw = metainfo.get(ds_key)
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    if "+" in s:
        if ds_key not in _MIXED_DATASET_FALLBACK_WARNED:
            _MIXED_DATASET_FALLBACK_WARNED.add(ds_key)
            warnings.warn(
                f"Dataset {ds_key!r} has mixed perturbation_type {s!r} in metainfo; "
                "prefer per-cell obs['perturbation_type']. Dataset-level fallback is omitted (null).",
                UserWarning,
                stacklevel=2,
            )
        return None
    return s


def apply_pert_metainfo_fallback(
    meta: ConditionMetadata,
    ds_name: str,
    metainfo: Optional[Mapping[str, str]],
    *,
    use_pert_condition: bool,
) -> ConditionMetadata:
    """If ``use_pert_condition`` and obs type is missing, fill from metainfo when allowed."""
    if not use_pert_condition or not metainfo:
        return meta
    if meta.perturbation_type_raw is not None:
        return meta
    resolved = resolve_perturbation_type(
        metainfo,
        ds_name,
        row_type=meta.perturbation_type_raw,
        allow_override_by_row=True,
    )
    if not resolved:
        return meta
    return replace(meta, perturbation_type_raw=resolved)
