from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model.raw_pretrain.data_source import H5adTissueShard, TissueShardSource, discover_h5ad_shards
from model.raw_pretrain.dataset import PairwisePretrainDataset


class _Vocab:
    def __init__(self, symbols: list[str]):
        self.gene2token = {s: i + 10 for i, s in enumerate(symbols)}


def _write_h5ad(path: Path, tissue: str, genes: list[str], x: np.ndarray) -> None:
    import anndata as ad

    path.parent.mkdir(parents=True, exist_ok=True)
    obs = pd.DataFrame(
        {
            "tissue": [tissue] * x.shape[0],
            "cell_type": ["ct"] * x.shape[0],
            "cell_type_ontology_term_id": ["CL:0"] * x.shape[0],
            "cluster_id": [f"{tissue}::{i}" for i in range(x.shape[0])],
            "cluster_size": [1] * x.shape[0],
            "cluster_frac": [1.0 / max(x.shape[0], 1)] * x.shape[0],
        },
        index=[f"{tissue}_{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame(
        {"feature_name": genes},
        index=[str(i) for i in range(len(genes))],
    )
    ad.AnnData(X=x.astype(np.float32), obs=obs, var=var).write_h5ad(path)


def test_discover_prefers_metainfo_and_feature_name(tmp_path: Path):
    root = tmp_path / "processed"
    h5 = root / "blood" / "blood_top6000var.h5ad"
    _write_h5ad(h5, "blood", ["TP53", "ACTB", "MALAT1"], np.array([[0, 1, 2], [2, 1, 0]]))
    (root / "tissue_metainfo.csv").write_text(
        "tissue,path,n_cells,n_genes,n_celltypes\n"
        f"blood,{h5.resolve()},2,3,1\n",
        encoding="utf-8",
    )

    shards = discover_h5ad_shards(
        root,
        _Vocab(["TP53", "ACTB", "MALAT1"]),
        gene_symbol_column="feature_name",
        min_gene_hit_rate=0.8,
    )

    assert len(shards) == 1
    summary = shards[0].schema_summary()
    assert summary["path"] == "blood/blood_top6000var.h5ad"
    assert summary["gene_symbol_source"] == "var['feature_name']"
    assert summary["gene_hit_count"] == 3
    assert shards[0].gene_ids_cellnavi().tolist() == [10, 11, 12]


def test_h5ad_shard_rejects_low_gene_hit_rate(tmp_path: Path):
    h5 = tmp_path / "blood" / "blood_top6000var.h5ad"
    _write_h5ad(h5, "blood", ["TP53", "NOPE1", "NOPE2"], np.array([[0, 1, 2], [2, 1, 0]]))

    with pytest.raises(ValueError, match="gene hit rate"):
        H5adTissueShard(
            h5,
            _Vocab(["TP53"]),
            gene_symbol_column="feature_name",
            min_gene_hit_rate=0.8,
            root_dir=tmp_path,
        )


class _Source(TissueShardSource):
    def __init__(self, name: str, x: np.ndarray, gene_ids: list[int]):
        self._name = name
        self._x = x.astype(np.float32)
        self._gene_ids = np.asarray(gene_ids, dtype=np.int64)

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_units(self) -> int:
        return int(self._x.shape[0])

    def gene_ids_cellnavi(self) -> np.ndarray:
        return self._gene_ids.copy()

    def get_expr(self, row_idx: int) -> np.ndarray:
        return self._x[int(row_idx)].copy()

    def schema_summary(self) -> dict:
        return {"name": self._name, "n_obs": self.n_units, "n_vars": int(self._x.shape[1])}


def test_batches_are_tissue_homogeneous_with_different_gene_axes():
    sources = [
        _Source("a", np.array([[0, 0], [1, 0], [0, 1]]), [10, 11]),
        _Source("b", np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]]), [20, 21, 22]),
    ]
    ds = PairwisePretrainDataset(
        sources,
        rank=0,
        world_size=1,
        batch_size=2,
        max_pert_genes=2,
        seed=1,
    )

    batches = list(iter(ds))
    assert batches
    for b in batches:
        gid = b["gene_ids"]
        assert gid.dim() == 1
        assert gid.tolist() in ([10, 11], [20, 21, 22])
        assert b["x_ctrl"].shape[1] == gid.numel()


def test_pseudo_target_gene_sampling_is_seeded_and_from_delta_candidates():
    src = _Source(
        "a",
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        [10, 11, 12],
    )
    ds = PairwisePretrainDataset(
        [src],
        rank=0,
        world_size=1,
        batch_size=1,
        max_pert_genes=3,
        seed=7,
    )
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = ds._one_sample(0, 0, rng1)
    b = ds._one_sample(0, 0, rng2)

    assert torch.equal(a["pert_gene_ids"], b["pert_gene_ids"])
    valid = a["pert_mask"].bool()
    assert set(a["pert_gene_ids"][valid].tolist()).issubset({10, 12})
    assert set(a["pert_signs"][valid].tolist()).issubset({1.0})
    assert torch.all((a["pert_mags"][valid] > 0) & (a["pert_mags"][valid] <= 1))
    assert torch.all(a["pert_gene_ids"][~valid] == -1)


def test_zero_delta_pair_is_skipped():
    src = _Source("a", np.array([[0.0, 0.0], [0.0, 0.0]]), [10, 11])
    ds = PairwisePretrainDataset([src], rank=0, world_size=1, batch_size=1, seed=7)
    with pytest.raises(RuntimeError, match="delta"):
        ds._one_sample(0, 0, np.random.default_rng(1))
