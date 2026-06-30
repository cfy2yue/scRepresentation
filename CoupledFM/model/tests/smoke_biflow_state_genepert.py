"""Smoke biFlow genepert layouts (control_state/gt_state) + metainfo perturbation types."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import tempfile

import numpy as np
import torch

from model import paths

BIFLOW = paths.biflow_dir()
META_PATH = _REPO / "data/raw/genepert_DE5000/metainfo.json"
GENE_NAME = paths.gene_name_path()
NICHENET_NODE2IDX = paths.nichenet_node2idx_path()
GENE_CACHE = paths.cellnavi_cache_dir()
FM_DATA = _ROOT / "latent/fm_data"
ADAMSON_H5 = FM_DATA / "Adamson.h5"


def _minimal_embedding_cache(dest: Path) -> None:
    """Tiny cache so PerturbationBatch can run lookups without pretrained assets."""
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"embed_dim": 8, "source": "smoke_dummy"}
    (dest / "manifest.json").write_text(
        __import__("json").dumps(manifest),
        encoding="ascii",
    )
    emb = np.zeros((32, 8), dtype=np.float32)
    np.save(dest / "gene_embeddings.npy", emb)
    lines = ["<pad>\t0", "<unk>\t1", "TP53\t2", "BRCA1\t3", "MDM2\t4"]
    (dest / "gene_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_coupled_smoke(cache_dir: Path) -> None:
    from model.data.dataset import CoupledFMDataset
    from model.utils.data.split import build_canonical_split, canonical_split_path, load_split_json
    from model.utils.data.vocab import GeneVocab
    from model.condition_emb.genepert.perturbation import perturbation_type_to_id

    split_path = canonical_split_path(BIFLOW, seed=42)

    vocab = GeneVocab(str(GENE_NAME), str(NICHENET_NODE2IDX))

    if split_path.is_file():
        split_full = load_split_json(split_path)
    else:
        split_full = build_canonical_split(
            BIFLOW,
            vocab,
            seed=42,
            min_cells=16,
            coupling_mode="baseline",
            dataset_names=["Adamson"],
            verbose=True,
            latent_backbone="state",
        )
        if not split_full:
            print(
                "  [coupled] skip: build_canonical_split returned empty "
                "(need Adamson control/gt layouts under biFlow)"
            )
            return

    if "Adamson" not in split_full:
        print("  [coupled] skip: Adamson absent from loaded split manifest")
        return

    split = {k: v for k, v in split_full.items() if k == "Adamson"}

    ds = CoupledFMDataset(
        str(BIFLOW),
        vocab,
        split,
        mode="train",
        coupling_mode="baseline",
        batch_size=8,
        min_cells=16,
        ds_alpha=1.0,
        use_raw_pert_condition=True,
        max_pert_genes=8,
        pert_gene_emb_cache_dir=str(cache_dir),
        use_h5ad_pert_metadata=False,
        pert_metainfo_path=str(META_PATH),
        dataset_names=["Adamson"],
        latent_backbone="state",
    )
    assert ds.ds_names, "CoupledFMDataset: no datasets loaded"
    b = next(iter(ds))
    assert b[8] == "Adamson"
    cond = b[9]
    print(f"  [coupled] cond={cond!r} Adamson baseline batch ok")
    assert b[-2] is not None
    gid, pert_mask, pert_type_id, nperts, combo_id, chem_emb, chem_mask = b[-2]
    assert chem_emb is None and chem_mask is None
    x_t = b[0]
    x_ctrl_ref = b[1]
    t_bt = b[2]
    dx_t = b[4]
    tc = t_bt.view(-1, 1) if t_bt.dim() == 1 else t_bt
    x_gt_rec = x_t + (1.0 - tc) * dx_t
    print(
        "  [coupled] pert_type_id (first rows):",
        pert_type_id[:4].tolist(),
        "unique:",
        torch.unique(pert_type_id).tolist(),
    )
    print("  [coupled] nperts (first rows):", nperts[:4].tolist())
    print("  [coupled] pert_mask row sum:", int(pert_mask[0].sum()))
    print("  [coupled] x_t shape:", tuple(x_t.shape))
    print("  [coupled] x_ctrl_ref shape:", tuple(x_ctrl_ref.shape))
    print("  [coupled] x_gt reconstructed shape:", tuple(x_gt_rec.shape))
    exp_tid = perturbation_type_to_id("CRISPRi")
    if str(cond).lower() != "control":
        assert int(pert_type_id[0]) == exp_tid, (
            f"expected CRISPRi -> {exp_tid}, got {int(pert_type_id[0])}"
        )
    ds.close()


def run_latent_smoke(cache_dir: Path) -> None:
    from model.latent.dataset import CrossDatasetFMDataset, load_or_create_split

    if not ADAMSON_H5.is_file():
        print(f"  [latent] skip: missing {ADAMSON_H5}")
        return

    manifest = {"datasets": {"Adamson": {"conditions": []}}}
    split = load_or_create_split(
        str(FM_DATA),
        manifest,
        test_ratio=0.1,
        seed=42,
        biflow_dir=str(BIFLOW),
    )
    assert "Adamson" in split

    ds = CrossDatasetFMDataset(
        str(FM_DATA),
        split,
        batch_size=16,
        seed=42,
        mode="train",
        min_cells=8,
        ds_alpha=1.0,
        use_pert_condition=True,
        max_pert_genes=8,
        gene_embedding_cache_dir=str(cache_dir),
        biflow_dir=str(BIFLOW),
        latent_backbone="state",
        use_h5ad_pert_metadata=False,
        pert_metainfo_path=str(META_PATH),
    )
    it = iter(ds)
    src, gt, ds_name, cond, pert = next(it)
    assert ds_name == "Adamson"
    assert pert is not None
    pg, pm, pt, np_, cid, chem_e, chem_m = pert
    assert chem_e is None and chem_m is None
    print(f"  [latent] cond={cond!r} src={tuple(src.shape)} gt={tuple(gt.shape)}")
    print(
        "  [latent] perturbation_batch: pert_type_id sample",
        pt[:4].tolist(),
        "nperts",
        np_[:4].tolist(),
        "combo_id",
        cid[:4].tolist(),
    )
    assert pm.shape[1] == 8
    ds.close()


def main() -> None:
    print("[smoke_biflow_state_genepert] start")
    if not BIFLOW.is_dir():
        print(f"  skip: missing {BIFLOW} (local biFlow checkout not present)")
        print("[smoke_biflow_state_genepert] SKIP")
        return
    assert META_PATH.is_file(), META_PATH
    assert GENE_NAME.is_file(), GENE_NAME
    assert NICHENET_NODE2IDX.is_file(), NICHENET_NODE2IDX

    if GENE_CACHE.is_dir():
        cache_dir = GENE_CACHE
        print(f"  using gene cache: {cache_dir}")
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="smoke_gene_cache_"))
        _minimal_embedding_cache(cache_dir)
        print(f"  using temp gene cache: {cache_dir}")

    run_coupled_smoke(cache_dir)
    run_latent_smoke(cache_dir)

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        from model.condition_emb.genepert.perturbation_encoder import PerturbationConditionEncoder

        enc = PerturbationConditionEncoder(
            mode="random_learned",
            out_dim=64,
            num_embeddings_random=256,
            embed_dim_random=32,
            type_embed_dim=16,
            cache=None,
        ).to(dev)
        B = 4
        K = 8
        g = torch.zeros(B, K, dtype=torch.long, device=dev)
        m = torch.zeros(B, K, dtype=torch.bool, device=dev)
        m[:, 0] = True
        tid = torch.full((B,), 2, dtype=torch.long, device=dev)
        npt = torch.ones(B, dtype=torch.long, device=dev)
        cid = torch.zeros(B, dtype=torch.long, device=dev)
        y = enc(pert_gene_ids=g, pert_mask=m, pert_type_id=tid, nperts=npt, combo_id=cid)
        assert y.shape == (B, 64)
        print("  [optional] encoder cuda:0 forward ok ->", tuple(y.shape))
    else:
        print("  [optional] no CUDA; skip encoder forward")

    print("[smoke_biflow_state_genepert] OK")


if __name__ == "__main__":
    main()
