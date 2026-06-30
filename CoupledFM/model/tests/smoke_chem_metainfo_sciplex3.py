"""Smoke: sciplex3 obs → ConditionMetadata, chem cache resolve, encoder with chem_emb_dim.

Override data path with env ``SCIPLEX3_H5AD`` (absolute path to ``*.h5ad``); default under
``<repo>/data/raw/chemicalpert_DE5000/`` if present.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.condition_emb.genepert.chem_embedding_hook import (  # noqa: E402
    parse_chem_source_fields,
    resolve_chem_embedding,
)
from model.condition_emb.genepert.h5ad_obs import condition_metadata_from_obs_row  # noqa: E402
from model.condition_emb.genepert.perturbation import ConditionMetadata, PerturbationBatch  # noqa: E402
from model.condition_emb.genepert.perturbation_encoder import PerturbationConditionEncoder  # noqa: E402
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
H5AD_PATH = Path(os.environ.get(
    "SCIPLEX3_H5AD",
    str(_REPO / "data/raw/chemicalpert_DE5000/sciplex3_A549.h5ad"),
)).expanduser()


def _fake_gene_cache(embed_dim: int = 32) -> GeneEmbeddingCache:
    tmp = Path(tempfile.mkdtemp(prefix="gene_cache_sciplex3_"))
    genes = ["PAD", "UNK", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"]
    with (tmp / "gene_index.tsv").open("w") as fh:
        fh.write("symbol\tindex\n")
        for i, g in enumerate(genes):
            fh.write(f"{g}\t{i}\n")
    arr = np.random.default_rng(0).standard_normal((len(genes), embed_dim)).astype(np.float32)
    arr[0] = 0.0
    np.save(tmp / "gene_embeddings.npy", arr)
    (tmp / "manifest.json").write_text("{}", encoding="ascii")
    return GeneEmbeddingCache(tmp)


def main() -> None:
    print("[smoke_chem_metainfo_sciplex3] start")
    if not H5AD_PATH.is_file():
        raise FileNotFoundError(f"missing fixture: {H5AD_PATH}")

    import model.condition_emb.genepert.chem_embedding_hook as ceh

    ceh._CHEM_CACHE_BY_DIR.clear()
    ceh._RESOLVE_HIT_LOGGED = False

    backed = ad.read_h5ad(str(H5AD_PATH), backed="r")
    obs = backed.obs
    ctrl = obs["control"].astype(int).to_numpy()
    idx_all = np.nonzero(ctrl == 0)[0]
    rng = np.random.default_rng(42)
    idxs = list(rng.choice(idx_all, size=3, replace=False).astype(int))

    metas: list[ConditionMetadata] = []
    for ix in idxs:
        m = condition_metadata_from_obs_row(obs, ix)
        metas.append(m)
        assert m.genes == (), m.genes
        assert m.chem_emb is None
        assert m.chem_source is not None
        assert "smiles=" in m.chem_source
        assert "drug=" in m.chem_source

    cfg_off = SimpleNamespace(chem_emb_source_dir="")
    assert resolve_chem_embedding(metas[0], cfg_off) is None

    smiles_keys: list[str] = []
    for m in metas:
        sm, _, _ = parse_chem_source_fields(m.chem_source)
        assert sm is not None
        smiles_keys.append(sm)

    tool = _ROOT / "tools/export_chem_embedding_cache.py"
    tmp_root = Path(tempfile.mkdtemp(prefix="chem_cache_sciplex3_"))
    payload_path = tmp_root / "vecs.json"
    emb_dict = {s: np.random.default_rng(7).standard_normal(8).astype(float).tolist() for s in smiles_keys}
    payload_path.write_text(json.dumps(emb_dict), encoding="utf-8")
    cache_dir = tmp_root / "cache"
    subprocess.check_call(
        [
            sys.executable,
            str(tool),
            "--format",
            "passthrough_dict",
            "--input",
            str(payload_path),
            "--out-dir",
            str(cache_dir),
        ],
        cwd=str(_ROOT),
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
    )

    cfg_on = SimpleNamespace(chem_emb_source_dir=str(cache_dir))
    resolved: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for m in metas:
            v = resolve_chem_embedding(m, cfg_on)
            assert v is not None and v.shape == (8,), v
            assert v.dtype == np.float32
            assert np.isfinite(v).all()
            resolved.append(v)

    rows: list[ConditionMetadata] = [
        dataclasses.replace(m, chem_emb=resolved[i]) for i, m in enumerate(metas)
    ]

    gcache = _fake_gene_cache(32)
    pb = PerturbationBatch.from_metadata_list(rows, gcache, max_genes=4)
    enc = PerturbationConditionEncoder(
        mode="pretrained_tunable",
        out_dim=48,
        cache=gcache,
        type_embed_dim=8,
        chem_emb_dim=8,
    )
    out = enc(
        pert_gene_ids=pb.pert_gene_ids,
        pert_mask=pb.pert_mask,
        pert_type_id=pb.pert_type_id,
        nperts=pb.nperts,
        combo_id=pb.combo_ids,
        chem_emb=pb.chem_emb,
        chem_mask=pb.chem_mask,
    )
    assert out.shape == (3, 48), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf in encoder output"
    print("[smoke_chem_metainfo_sciplex3] OK rows=", idxs, "out=", tuple(out.shape))


if __name__ == "__main__":
    main()
