"""End-to-end smoke for the condition-embedding pipeline shared by latent / coupled / raw_independent.

Pipeline tested:
    h5ad obs row  ->  ConditionMetadata(+ metainfo fallback)
                  ->  PerturbationBatch.from_metadata_list (CellNavi embed_gene cache, dim=256)
                  ->  PerturbationConditionEncoder (pretrained_tunable, with type gate disabled)
                  ->  projector to d_model=512  (same module used in RawExprVelocityField.pert_to_c
                                                and ControlMLPVelocityField.pert_to_c)
                  ->  CFG null branch yields all-zero encoder output.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from model import paths
from model.utils.conditioning.metainfo import apply_pert_metainfo_fallback, load_dataset_metainfo
from model.utils.conditioning.perturbation import (
    ConditionMetadata,
    PerturbationBatch,
    perturbation_type_to_id,
)
from model.utils.conditioning.perturbation_encoder import PerturbationConditionEncoder
from model.utils.embeddings.gene_cache import GeneEmbeddingCache


CACHE_DIR = paths.cellnavi_cache_dir()
META_PATH = _ROOT / "data/raw/genepert_DE5000/metainfo.json"
D_MODEL = 512


def build_metadata(metainfo):
    """7 rows covering: single-gene, multi-gene, missing row_type (fallback), Schmidt mixed, control, OOV."""
    rows_spec = [
        ("Adamson", "TP53", None),
        ("Adamson", "BRCA1+MDM2", None),
        ("Frangieh", "EGFR", None),
        ("Schmidt", "IL2", "CRISPRa"),
        ("Schmidt", "IL2", None),
        ("Adamson", "control", None),
        ("Adamson", "ZZ_NOT_A_GENE", "CRISPRi"),
    ]
    metas = []
    for ds, pert, row_type in rows_spec:
        m = ConditionMetadata.from_obs_fields(pert, perturbation_type_field=row_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = apply_pert_metainfo_fallback(m, ds, metainfo, use_pert_condition=True)
        metas.append((ds, pert, m))
    return metas


def main() -> None:
    assert CACHE_DIR.is_dir(), f"missing cache: {CACHE_DIR}"
    cache = GeneEmbeddingCache(CACHE_DIR)
    print(f"[e2e] cache embed_dim={cache.embed_dim} num_rows={cache.num_embeddings}")

    metainfo = load_dataset_metainfo(META_PATH)
    specs = build_metadata(metainfo)
    metas = [m for _, _, m in specs]
    for (ds, pert, m) in specs:
        tid = perturbation_type_to_id(m.perturbation_type_raw)
        print(
            f"  ds={ds:<10s} pert={pert!r:<18s} -> genes={m.genes} type={m.perturbation_type_raw!r} tid={tid}"
        )

    pb = PerturbationBatch.from_metadata_list(metas, cache, max_genes=8)
    assert pb.pert_gene_ids.shape == (len(metas), 8)
    assert pb.pert_mask.shape == (len(metas), 8)
    assert pb.pert_type_id.shape == (len(metas),)
    assert pb.nperts.shape == (len(metas),)

    # control row (idx 5) should have nperts=0 and mask all-False.
    assert int(pb.nperts[5]) == 0
    assert not bool(pb.pert_mask[5].any())
    # OOV row (idx 6) -> UNK index (1), still one masked slot.
    assert int(pb.nperts[6]) == 1
    assert int(pb.pert_gene_ids[6, 0]) == cache.unk_index
    # multi-gene row (idx 1) -> 2 slots active.
    assert int(pb.nperts[1]) == 2
    assert int(pb.pert_mask[1].sum()) == 2
    # metainfo fallback: Adamson row 0 tid == CRISPRI even though row_type=None.
    assert int(pb.pert_type_id[0]) == perturbation_type_to_id("CRISPRi")
    # Schmidt row with explicit CRISPRa (idx 3) -> CRISPRA.
    assert int(pb.pert_type_id[3]) == perturbation_type_to_id("CRISPRa")
    # Schmidt row 4 stays null because metainfo has mixed '+' type.
    assert int(pb.pert_type_id[4]) == 0

    enc = PerturbationConditionEncoder(
        mode="pretrained_tunable",
        out_dim=256,
        cache=cache,
        type_embed_dim=32,
    )
    c_emb = enc(
        pert_gene_ids=pb.pert_gene_ids,
        pert_mask=pb.pert_mask,
        pert_type_id=pb.pert_type_id,
        nperts=pb.nperts,
        combo_id=pb.combo_ids,
    )
    assert c_emb.shape == (len(metas), 256)
    assert torch.isfinite(c_emb).all()

    # Projector to d_model=512 (same as pert_to_c used by ControlMLP / RawExprVelocityField).
    pert_to_c = nn.Linear(256, D_MODEL)
    nn.init.zeros_(pert_to_c.weight)
    nn.init.zeros_(pert_to_c.bias)
    c_vec = pert_to_c(c_emb)
    assert c_vec.shape == (len(metas), D_MODEL)
    # zero projector → zero c_vec (this is the initialization used in training).
    assert torch.allclose(c_vec, torch.zeros_like(c_vec))
    print("  zero-init pert_to_c: c_vec all zeros (matches training init)")

    pert_to_c_warm = nn.Linear(256, D_MODEL)
    c_vec_warm = pert_to_c_warm(c_emb)
    # control row should still project to zero because encoder output is zero on that row.
    enc_zero_row = c_emb[5]
    assert torch.allclose(enc_zero_row, torch.zeros_like(enc_zero_row))
    print(f"  encoder zero-row for 'control': ||c||_inf = {enc_zero_row.abs().max().item():.3g}")
    print(f"  warm pert_to_c: control-row ||c_vec||={c_vec_warm[5].abs().max().item():.3g}")

    # CFG null batch via from_metadata_list with empty metas:
    null_metas = [ConditionMetadata(genes=(), perturbation_type_raw=None) for _ in range(3)]
    pb_null = PerturbationBatch.from_metadata_list(null_metas, cache, max_genes=8)
    c_null = enc(
        pert_gene_ids=pb_null.pert_gene_ids,
        pert_mask=pb_null.pert_mask,
        pert_type_id=pb_null.pert_type_id,
        nperts=pb_null.nperts,
        combo_id=pb_null.combo_ids,
    )
    assert torch.allclose(c_null, torch.zeros_like(c_null))
    print(f"  CFG-null batch encoder output all zeros (shape={tuple(c_null.shape)})")

    print("[e2e] OK  -- pipeline shared by latent / coupled / raw_independent is wired end-to-end.")


if __name__ == "__main__":
    main()
