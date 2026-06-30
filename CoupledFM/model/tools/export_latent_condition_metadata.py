#!/usr/bin/env python3
"""Export LatentFM ``condition_metadata.json`` from biFlow GT h5ad files.

This avoids scanning large AnnData ``obs`` tables at every LatentFM training
startup.  The output is the same sidecar consumed by
``model.latent.dataset.CrossDatasetFMDataset``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode_bytes(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _read_obs_column(obs: h5py.Group, name: str) -> list[str] | np.ndarray | None:
    if name not in obs:
        return None
    obj = obs[name]
    if isinstance(obj, h5py.Group) and "codes" in obj and "categories" in obj:
        codes = np.asarray(obj["codes"][:], dtype=np.int64)
        cats = np.asarray([_decode_bytes(x) for x in obj["categories"][:]], dtype=object)
        out = np.empty(len(codes), dtype=object)
        valid = (codes >= 0) & (codes < len(cats))
        out[:] = ""
        out[valid] = cats[codes[valid]]
        return out.tolist()
    if isinstance(obj, h5py.Dataset):
        arr = obj[:]
        if arr.dtype.kind in {"S", "O", "U"}:
            return [_decode_bytes(x) for x in arr]
        return arr
    return None


def _clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _first_value(columns: dict[str, Any], row_idx: int, names: tuple[str, ...]) -> str:
    for name in names:
        col = columns.get(name)
        if col is None:
            continue
        try:
            val = col[int(row_idx)]
        except Exception:
            continue
        s = _clean(val)
        if s:
            return s
    return ""


def _split_genes(text: str) -> list[str]:
    text = _clean(text)
    if not text:
        return []
    out: list[str] = []
    for part in text.replace(",", "+").replace("|", "+").split("+"):
        for token in part.split():
            token = token.strip().upper()
            if token and token not in {"CONTROL", "CTRL", "DMSO"}:
                out.append(token)
    return out


def _is_drug_dataset(ds_name: str) -> bool:
    d = ds_name.lower()
    return any(tok in d for tok in ("sciplex", "chempert", "chemical", "drug"))


def _metadata_for_condition(
    *,
    ds_name: str,
    cond: str,
    row_idx: int,
    columns: dict[str, Any],
) -> dict[str, Any]:
    ptype = _first_value(
        columns,
        row_idx,
        ("perturbation_type", "pert_type", "perturbation_kind", "modality"),
    )
    if not ptype and _is_drug_dataset(ds_name):
        ptype = "drug"

    ptype_l = ptype.strip().lower()
    is_drug = ptype_l in {"drug", "chemical", "compound", "small molecule"} or _is_drug_dataset(ds_name)

    chem_obs_value = ""
    if is_drug:
        chem_obs_value = _first_value(
            columns,
            row_idx,
            ("cov_drug", "condition", "perturbation", "drug", "drug_dose_name", "cov_drug_dose_name"),
        ) or str(cond)
        genes: list[str] = []
    else:
        gene_val = _first_value(columns, row_idx, ("gene", "target", "perturbation")) or str(cond)
        genes = _split_genes(gene_val)

    meta: dict[str, Any] = {
        "perturbation_type_raw": ptype or None,
        "genes": genes,
        "condition_col": "condition" if _is_drug_dataset(ds_name) else "perturbation",
    }
    if chem_obs_value:
        meta["chem_obs_value"] = chem_obs_value
        meta["chem_source"] = f"drug={chem_obs_value}"
    return {k: v for k, v in meta.items() if v is not None}


def _first_indices_by_condition(h5ad_path: Path, condition_names: list[str]) -> dict[str, int]:
    wanted = set(map(str, condition_names))
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        cond_col = "condition" if "condition" in obs else "perturbation"
        values = _read_obs_column(obs, cond_col)
    if values is None:
        return {}
    first: dict[str, int] = {}
    for i, v in enumerate(values):
        s = _clean(v)
        if s in wanted and s not in first:
            first[s] = int(i)
            if len(first) == len(wanted):
                break
    return first


def _load_columns_for_rows(h5ad_path: Path) -> dict[str, Any]:
    names = (
        "condition",
        "perturbation",
        "perturbation_type",
        "pert_type",
        "gene",
        "target",
        "cov_drug",
        "drug",
        "drug_dose_name",
        "cov_drug_dose_name",
    )
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        return {name: _read_obs_column(obs, name) for name in names if name in obs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-dir", type=Path, required=True)
    ap.add_argument("--biflow-gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest_path = args.latent_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    out_path = args.out or (args.latent_dir / "condition_metadata.json")
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, dict[str, Any]]] = {}
    missing: dict[str, list[str]] = {}

    for ds_name, ds_meta in sorted(manifest.get("datasets", {}).items()):
        conds = [str(c) for c in ds_meta.get("conditions", [])]
        h5ad_path = args.biflow_gt_dir / f"{ds_name}.h5ad"
        if not h5ad_path.is_file():
            missing[ds_name] = conds
            continue
        first = _first_indices_by_condition(h5ad_path, conds)
        columns = _load_columns_for_rows(h5ad_path)
        ds_out: dict[str, dict[str, Any]] = {}
        ds_missing: list[str] = []
        for cond in conds:
            if cond not in first:
                ds_missing.append(cond)
                continue
            ds_out[cond] = _metadata_for_condition(
                ds_name=ds_name,
                cond=cond,
                row_idx=first[cond],
                columns=columns,
            )
        result[ds_name] = ds_out
        if ds_missing:
            missing[ds_name] = ds_missing
        print(f"[{ds_name}] metadata={len(ds_out)}/{len(conds)} missing={len(ds_missing)}", flush=True)

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    miss_path = out_path.with_name(out_path.stem + ".missing.json")
    miss_path.write_text(json.dumps(missing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {miss_path}")
    total = sum(len(v) for v in result.values())
    expected = sum(len(v.get("conditions", [])) for v in manifest.get("datasets", {}).values())
    print(f"total metadata={total}/{expected}")
    return 0 if total == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
