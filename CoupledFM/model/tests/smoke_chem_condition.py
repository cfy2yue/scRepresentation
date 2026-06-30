"""Smoke test: chemical perturbation branch of PerturbationConditionEncoder and ControlMLPVelocityField."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from model.utils.conditioning.perturbation import (
    ConditionMetadata,
    PerturbationBatch,
    perturbation_batch_to_device,
)
from model.utils.conditioning.perturbation_encoder import PerturbationConditionEncoder
from model.utils.embeddings.gene_cache import GeneEmbeddingCache


def _fake_cache(embed_dim: int = 32) -> GeneEmbeddingCache:
    """Build a tiny on-the-fly GeneEmbeddingCache with 8 rows (PAD/UNK + 6 genes)."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="gene_cache_smoke_"))
    genes = ["PAD", "UNK", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"]
    with (tmp / "gene_index.tsv").open("w") as fh:
        fh.write("symbol\tindex\n")
        for i, g in enumerate(genes):
            fh.write(f"{g}\t{i}\n")
    arr = np.random.default_rng(0).standard_normal((len(genes), embed_dim)).astype(np.float32)
    arr[0] = 0.0
    np.save(tmp / "gene_embeddings.npy", arr)
    (tmp / "manifest.json").write_text("{}")
    return GeneEmbeddingCache(tmp)


def _run_encoder(batch_size: int, chem_dim: int, use_chem: bool) -> None:
    cache = _fake_cache(embed_dim=32)
    enc = PerturbationConditionEncoder(
        mode="pretrained_tunable",
        out_dim=48,
        cache=cache,
        type_embed_dim=8,
        chem_emb_dim=chem_dim if use_chem else 0,
    )

    rows = []
    rng = np.random.default_rng(1)
    for i in range(batch_size):
        meta = ConditionMetadata(
            genes=("GENE1",) if i % 2 == 0 else ("GENE2", "GENE3"),
            perturbation_type_raw="CRISPRi" if i % 2 == 0 else None,
            combo_id=i + 1,
            nperts_obs=None,
            chem_emb=(rng.standard_normal(chem_dim).astype(np.float32) if (use_chem and i % 2 == 1) else None),
            chem_source="drug=foo" if (use_chem and i % 2 == 1) else None,
        )
        rows.append(meta)

    pb = PerturbationBatch.from_metadata_list(rows, cache, max_genes=4)
    out = enc(
        pert_gene_ids=pb.pert_gene_ids,
        pert_mask=pb.pert_mask,
        pert_type_id=pb.pert_type_id,
        nperts=pb.nperts,
        combo_id=pb.combo_ids,
        chem_emb=pb.chem_emb,
        chem_mask=pb.chem_mask,
    )
    assert out.shape == (batch_size, 48), out.shape
    assert torch.isfinite(out).all(), "encoder output has NaN/Inf"
    if use_chem:
        assert pb.chem_emb is not None and pb.chem_mask is not None
        chem_rows = pb.chem_mask > 0
        assert chem_rows.any(), "expected at least one chem row"
    print(
        f"  encoder batch={batch_size} chem={use_chem} chem_rows={int(pb.chem_mask.sum()) if pb.chem_mask is not None else 0} out={tuple(out.shape)}"
    )


def _run_velocity_field() -> None:
    from model.latent.models.mlp import ControlMLPVelocityField

    cache = _fake_cache(embed_dim=32)
    model = ControlMLPVelocityField(
        emb_dim=16,
        d_model=64,
        n_layers=2,
        use_pert_condition=True,
        pert_embed_mode="pretrained_tunable",
        pert_type_emb_dim=8,
        gene_embedding_cache=cache,
        pert_chem_emb_dim=12,
    )

    bsz = 3
    z_t = torch.randn(bsz, 16)
    t = torch.rand(bsz)
    z_src = torch.randn(bsz, 16)

    rows = [
        ConditionMetadata(genes=("GENE1",), perturbation_type_raw="CRISPRi", combo_id=1),
        ConditionMetadata(
            genes=(),
            perturbation_type_raw=None,
            combo_id=2,
            chem_emb=np.random.default_rng(2).standard_normal(12).astype(np.float32),
            chem_source="drug=bar",
        ),
        ConditionMetadata(genes=("GENE2", "GENE3"), perturbation_type_raw="CRISPRa", combo_id=3),
    ]
    pb = PerturbationBatch.from_metadata_list(rows, cache, max_genes=4)

    v = model(
        z_t,
        t,
        z_src,
        pert_gene_ids=pb.pert_gene_ids,
        pert_mask=pb.pert_mask,
        pert_type_id=pb.pert_type_id,
        nperts=pb.nperts,
        combo_id=pb.combo_ids,
        chem_emb=pb.chem_emb,
        chem_mask=pb.chem_mask,
    )
    assert v.shape == (bsz, 16), v.shape
    assert torch.isfinite(v).all()
    print(f"  velocity_field chem_rows={int(pb.chem_mask.sum()) if pb.chem_mask is not None else 0} v={tuple(v.shape)}")


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    print("[smoke_chem_condition] start")
    _run_encoder(batch_size=4, chem_dim=0, use_chem=False)
    _run_encoder(batch_size=4, chem_dim=16, use_chem=True)
    _run_velocity_field()
    print("[smoke_chem_condition] OK")


if __name__ == "__main__":
    main()
