"""Smoke test: per-dataset perturbation_type fallback from metainfo.json."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.utils.conditioning.metainfo import (
    apply_pert_metainfo_fallback,
    load_dataset_metainfo,
    resolve_perturbation_type,
)
from model.utils.conditioning.perturbation import ConditionMetadata, perturbation_type_to_id


META_PATH = str(_ROOT / "data/raw/genepert_DE5000/metainfo.json")


def main() -> None:
    print("[smoke_metainfo_fallback] start")
    mi = load_dataset_metainfo(META_PATH)
    assert mi, "metainfo.json empty"
    print(f"  loaded {len(mi)} dataset entries; sample={list(mi.items())[:3]}")

    # 1. Adamson -> CRISPRi (dataset level)
    assert resolve_perturbation_type(mi, "Adamson", row_type=None) == "CRISPRi"

    # 2. Per-cell override takes precedence.
    assert resolve_perturbation_type(mi, "Adamson", row_type="CRISPRko") == "CRISPRko"

    # 3. Schmidt is CRISPRa+CRISPRi in metainfo -> dataset-level fallback must warn + return None.
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        v = resolve_perturbation_type(mi, "Schmidt", row_type=None)
        assert v is None, v
        assert any("mixed perturbation_type" in str(w.message) for w in wl)
    # 3b. When per-cell obs provides a concrete type, use it.
    assert resolve_perturbation_type(mi, "Schmidt", row_type="CRISPRa") == "CRISPRa"

    # 4. apply_pert_metainfo_fallback: null -> filled for Adamson.
    meta = ConditionMetadata(genes=("TP53",), perturbation_type_raw=None, combo_id=1)
    out = apply_pert_metainfo_fallback(meta, "Adamson", mi, use_pert_condition=True)
    assert out.perturbation_type_raw == "CRISPRi"
    assert perturbation_type_to_id(out.perturbation_type_raw) > 0

    # 5. use_pert_condition=False leaves meta untouched.
    out_off = apply_pert_metainfo_fallback(meta, "Adamson", mi, use_pert_condition=False)
    assert out_off.perturbation_type_raw is None

    # 6. Schmidt with null row stays null (no bad global fallback).
    meta_s = ConditionMetadata(genes=("IL2",), perturbation_type_raw=None, combo_id=2)
    out_s = apply_pert_metainfo_fallback(meta_s, "Schmidt", mi, use_pert_condition=True)
    assert out_s.perturbation_type_raw is None

    # 7. Explicit row type wins even on Schmidt.
    meta_s2 = ConditionMetadata(genes=("IL2",), perturbation_type_raw="CRISPRa", combo_id=3)
    out_s2 = apply_pert_metainfo_fallback(meta_s2, "Schmidt", mi, use_pert_condition=True)
    assert out_s2.perturbation_type_raw == "CRISPRa"

    print("  Adamson null -> CRISPRi; Schmidt null stays None; explicit row_type preserved.")
    print("[smoke_metainfo_fallback] OK")


if __name__ == "__main__":
    main()
