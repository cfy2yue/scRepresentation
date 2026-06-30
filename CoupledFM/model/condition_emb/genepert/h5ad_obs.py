"""BiFlow ``gt/*.h5ad`` observation columns: names, dtypes, and robust reads.

Contract reference: ``docs/data_contract.md``. Training dataloaders historically
only streamed ``obs['perturbation']`` via h5py for speed; ``gene`` /
``perturbation_type`` / ``nperts`` follow legacy DE5000 / ``standardize_sp_datasets``.

Field semantics (code-level):

* ``obs['perturbation']``: categorical or string; ``control`` / equivalents mark
  controls; multi-target strings may look like ``GENE1 + GENE2`` or comma lists.
* ``obs['gene']``: optional parallel column (often single upper-case symbol or
  ``CTRL`` when control); preferred source when present because it avoids parsing
  human-readable ``perturbation`` labels.
* ``obs['perturbation_type']``: categorical/string (e.g. ``CRISPRi``); missing /
  unknown maps to id ``0`` via :func:`~condition_emb.genepert.perturbation.perturbation_type_to_id`.
* ``obs['nperts']``: int counts; inconsistent with gene lists is reconciled by
  :func:`~condition_emb.genepert.perturbation.infer_nperts_from_obs`.

Alternate column names below are **optional** fallbacks when primary columns are
absent (new datasets); primary biFlow columns stay authoritative when present.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

from .perturbation import (
    ConditionMetadata,
    _raw_to_clean_str,
    canonicalize_perturbation_type,
)

# Primary biFlow columns (see data_contract.md).
OBS_GENE = "gene"
OBS_PERTURBATION = "perturbation"
OBS_PERTURBATION_TYPE = "perturbation_type"
OBS_NPERTS = "nperts"

# Fallback column names if primary keys are missing.
FALLBACK_GENE_COLUMNS: Tuple[str, ...] = ("target_gene", "perturbation_gene", "pert_gene")
FALLBACK_PERT_COLUMNS: Tuple[str, ...] = ("pert", "condition", "pert_name")
FALLBACK_TYPE_COLUMNS: Tuple[str, ...] = ("pert_type", "perturbation_kind", "crispr_type")
FALLBACK_NPERTS_COLUMNS: Tuple[str, ...] = ("n_perturbations", "num_perts")

OBS_DRUG = "drug"
OBS_COMPOUND = "compound"
OBS_SMILES = "smiles"
OBS_INCHI = "inchi"
OBS_INCHIKEY = "inchikey"
OBS_DOSE = "dose"
OBS_CHEMBL_ID = "chembl_id"

FALLBACK_DRUG_COLUMNS: Tuple[str, ...] = (
    "drug_name",
    "drug_id",
    "compound_id",
    "chemical",
    "pert_drug",
    "cmap_name",
)
FALLBACK_COMPOUND_COLUMNS: Tuple[str, ...] = (
    "cmpd",
    "compound_name",
    "molecule",
    "compounds",
    "chemical_compound",
)
FALLBACK_SMILES_COLUMNS: Tuple[str, ...] = (
    "canonical_smiles",
    "smi",
    "SMILES",
    "compound_smiles",
)
FALLBACK_INCHI_COLUMNS: Tuple[str, ...] = (
    "inchi",
    "InChI",
    "inchi_standard",
)
FALLBACK_INCHIKEY_COLUMNS: Tuple[str, ...] = (
    "inchikey",
    "InChIKey",
    "INCHIKEY",
    "inchi_key",
)
FALLBACK_DOSE_COLUMNS: Tuple[str, ...] = (
    "dose_value",
    "concentration",
    "dosage",
    "conc",
    "dose_um",
    "dose_uM",
)
FALLBACK_CHEMBL_ID_COLUMNS: Tuple[str, ...] = (
    "chembl-ID",
    "chembl_id",
    "CHEMBL_ID",
    "chembl",
)
FALLBACK_DOSE_UNIT_COLUMNS: Tuple[str, ...] = (
    "dose_unit",
    "dose_units",
    "conc_unit",
    "concentration_unit",
    "unit",
)


def _first_existing_column(obs: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in obs.columns:
            return c
    return None


def pick_obs_columns(obs: pd.DataFrame) -> Mapping[str, Optional[str]]:
    """Map logical roles to actual ``obs`` column names (or None)."""
    gene_col = OBS_GENE if OBS_GENE in obs.columns else _first_existing_column(obs, FALLBACK_GENE_COLUMNS)
    pert_col = (
        OBS_PERTURBATION
        if OBS_PERTURBATION in obs.columns
        else _first_existing_column(obs, FALLBACK_PERT_COLUMNS)
    )
    ptype_col = (
        OBS_PERTURBATION_TYPE
        if OBS_PERTURBATION_TYPE in obs.columns
        else _first_existing_column(obs, FALLBACK_TYPE_COLUMNS)
    )
    npert_col = OBS_NPERTS if OBS_NPERTS in obs.columns else _first_existing_column(obs, FALLBACK_NPERTS_COLUMNS)
    drug_col = OBS_DRUG if OBS_DRUG in obs.columns else _first_existing_column(obs, FALLBACK_DRUG_COLUMNS)
    compound_col = (
        OBS_COMPOUND if OBS_COMPOUND in obs.columns else _first_existing_column(obs, FALLBACK_COMPOUND_COLUMNS)
    )
    smiles_col = OBS_SMILES if OBS_SMILES in obs.columns else _first_existing_column(obs, FALLBACK_SMILES_COLUMNS)
    inchi_col = OBS_INCHI if OBS_INCHI in obs.columns else _first_existing_column(obs, FALLBACK_INCHI_COLUMNS)
    inchikey_col = (
        OBS_INCHIKEY if OBS_INCHIKEY in obs.columns else _first_existing_column(obs, FALLBACK_INCHIKEY_COLUMNS)
    )
    dose_col = OBS_DOSE if OBS_DOSE in obs.columns else _first_existing_column(obs, FALLBACK_DOSE_COLUMNS)
    dose_unit_col = (
        "dose_unit"
        if "dose_unit" in obs.columns
        else _first_existing_column(obs, FALLBACK_DOSE_UNIT_COLUMNS)
    )
    chembl_col = (
        OBS_CHEMBL_ID if OBS_CHEMBL_ID in obs.columns else _first_existing_column(obs, FALLBACK_CHEMBL_ID_COLUMNS)
    )
    return {
        "gene": gene_col,
        "perturbation": pert_col,
        "perturbation_type": ptype_col,
        "nperts": npert_col,
        "drug": drug_col,
        "compound": compound_col,
        "smiles": smiles_col,
        "inchi": inchi_col,
        "inchikey": inchikey_col,
        "chembl_id": chembl_col,
        "dose": dose_col,
        "dose_unit": dose_unit_col,
    }


def condition_metadata_from_obs_row(
    obs: pd.DataFrame,
    idx: int,
    *,
    columns: Optional[Mapping[str, Optional[str]]] = None,
    sort_genes: bool = True,
) -> ConditionMetadata:
    """One-row :class:`~condition_emb.genepert.perturbation.ConditionMetadata` from ``obs``.

    Uses :meth:`~condition_emb.genepert.perturbation.ConditionMetadata.from_obs_fields`
    with column resolution via :func:`pick_obs_columns`.

    Rows whose ``perturbation_type`` canonicalizes to ``drug`` (e.g. sciplex3) have
    ``genes`` cleared to ``()`` — there is no CRISPR target gene — and
    ``chem_source`` starts with ``drug=`` from ``obs['perturbation']`` when
    available. Keys ``smiles=``, ``chembl_id=``, ``dose=``, ``dose_unit=`` align
    with :func:`~condition_emb.genepert.chem_embedding_hook.resolve_chem_embedding`
    lookup priority (SMILES, ChEMBL id, drug name).
    """
    cols = columns or pick_obs_columns(obs)
    gene_col = cols.get("gene")
    pert_col = cols.get("perturbation")
    ptype_col = cols.get("perturbation_type")
    npert_col = cols.get("nperts")

    row = obs.iloc[int(idx)]
    gf = row[gene_col] if gene_col is not None else None
    pf = row[pert_col] if pert_col is not None else None
    tf = row[ptype_col] if ptype_col is not None else None
    nf = row[npert_col] if npert_col is not None else None
    meta = ConditionMetadata.from_obs_fields(
        gf,
        perturbation_field=pf,
        perturbation_type_field=tf,
        nperts_field=nf,
        sort_genes=sort_genes,
    )
    ptype_canon = canonicalize_perturbation_type(tf) if ptype_col is not None else ""
    if ptype_canon == "drug":
        meta = dataclasses.replace(meta, genes=())

    parts: list[str] = []
    drug_from_pert = False
    if ptype_canon == "drug" and pert_col is not None:
        dr = _raw_to_clean_str(pf)
        if dr is not None:
            parts.append(f"drug={dr}")
            drug_from_pert = True

    for key in ("drug", "compound", "smiles", "inchi", "inchikey", "chembl_id", "dose", "dose_unit"):
        if key == "drug" and drug_from_pert:
            continue
        col = cols.get(key)
        if col is None:
            continue
        try:
            v = row[col]
        except Exception:
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "<na>"):
            continue
        parts.append(f"{key}={s}")
    chem_source = "|".join(parts) if parts else None
    if chem_source is not None:
        meta = dataclasses.replace(meta, chem_source=chem_source)
    return meta
