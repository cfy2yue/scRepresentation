"""Guards for unified condition embedding plan (gene + chem + direction scale)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model.condition_emb.chempert.drug_cache import (
    DrugEmbeddingCache,
    RandomDrugEmbeddingFallback,
    deterministic_standard_normal_vec,
)
from model.condition_emb.genepert import perturbation as P
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache
from model.condition_emb.genepert.perturbation import ConditionMetadata, PerturbationBatch
from model.condition_emb.genepert.perturbation_encoder import UnifiedConditionEncoder


def _minimal_gene_cache(root: Path, dim: int = 16) -> None:
    rng = np.random.default_rng(0)
    pad = np.zeros((1, dim), dtype=np.float32)
    unk = rng.standard_normal((1, dim)).astype(np.float32)
    tp53 = rng.standard_normal((1, dim)).astype(np.float32)
    arr = np.vstack([pad, unk, tp53])
    np.save(root / "gene_embeddings.npy", arr)
    (root / "gene_index.tsv").write_text("symbol\tindex\nTP53\t2\n", encoding="ascii")
    (root / "manifest.json").write_text(json.dumps({"source": "test"}), encoding="ascii")


def _minimal_drug_cache(root: Path, dim: int = 8) -> None:
    rng = np.random.default_rng(1)
    pad = np.zeros((1, dim), dtype=np.float32)
    unk = rng.standard_normal((1, dim)).astype(np.float32)
    aspirin = rng.standard_normal((1, dim)).astype(np.float32)
    arr = np.vstack([pad, unk, aspirin])
    np.save(root / "drug_embeddings.npy", arr)
    (root / "drug_index.tsv").write_text("key\tindex\naspirin\t2\n", encoding="ascii")
    (root / "manifest.json").write_text(json.dumps({"source": "test-drug"}), encoding="ascii")


class TestUnifiedPlanGuards(unittest.TestCase):
    def test_num_pert_types_six(self) -> None:
        self.assertEqual(P.num_perturbation_types(), 6)

    def test_type_ids_alias(self) -> None:
        self.assertEqual(P.perturbation_type_to_id("knockdown"), P.PERT_TYPE_CRISPRI)
        self.assertEqual(P.perturbation_type_to_id("overexpression"), P.PERT_TYPE_CRISPRA)
        self.assertEqual(P.perturbation_type_to_id("cas13 assay"), P.PERT_TYPE_CAS13)

    def test_unknown_tracked_then_null(self) -> None:
        P.reset_unknown_perturbation_types()
        tid = P.perturbation_type_to_id("totally_unknown_type_xx")
        self.assertEqual(tid, P.PERT_TYPE_NULL)
        self.assertTrue("totally_unknown_type_xx" in P.seen_unknown_perturbation_types())
        P.reset_unknown_perturbation_types()

    def test_pretrained_with_type_gate_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _minimal_gene_cache(root)
            gc = GeneEmbeddingCache(root)
            with self.assertRaises(ValueError):
                UnifiedConditionEncoder("pretrained_with_type_gate", 8, cache=gc)

    def test_type_scale_default(self) -> None:
        enc = UnifiedConditionEncoder(
            "random_learned",
            24,
            num_embeddings_random=32,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=(0.0, -1.0, -1.0, -1.0, 1.0, 1.0),
        )
        ts = enc.type_scale.detach().cpu().numpy().tolist()
        self.assertEqual(ts, [0.0, -1.0, -1.0, -1.0, 1.0, 1.0])
        self.assertIsNotNone(enc.type_scale)
        self.assertIsNone(enc.type_vector_scale)

    def test_vector_scale_adapter_params(self) -> None:
        enc = UnifiedConditionEncoder(
            "random_learned",
            24,
            num_embeddings_random=32,
            embed_dim_random=8,
            chem_emb_dim=None,
            type_adapter_mode="vector_scale",
            pert_type_scale_init=(0.0, -1.0, -1.0, -1.0, 1.0, 1.0),
        )
        self.assertIsNone(getattr(enc, "type_scale", None))
        self.assertEqual(tuple(enc.type_vector_scale.shape), (6, 24))

    def test_unified_encoder_shapes(self) -> None:
        B, K, G, C = 3, 4, 16, 8
        ge = torch.randn(B, K, G)
        m = torch.zeros(B, K, dtype=torch.bool)
        m[:, 0] = True
        npt = torch.ones(B, dtype=torch.long)

        enc = UnifiedConditionEncoder(
            "random_learned",
            32,
            num_embeddings_random=128,
            embed_dim_random=G,
            gene_projector_hidden=0,
            chem_emb_dim=C,
            chem_projector_hidden=16,
            pert_type_scale_init=(0.0, -1.0, -1.0, -1.0, 1.0, 1.0),
        )
        gid = torch.zeros(B, K, dtype=torch.long)
        tid = torch.zeros(B, dtype=torch.long)

        chem2 = torch.randn(B, C)
        out_a = enc(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt, chem_emb=chem2)
        self.assertEqual(tuple(out_a.shape), (B, 32))

        chem3 = torch.randn(B, 2, C)
        cm = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        tid2 = torch.full((B,), P.PERT_TYPE_DRUG)
        out_b = enc(
            pert_gene_ids=gid, pert_mask=m, pert_type_id=tid2,
            nperts=torch.zeros(B, dtype=torch.long), chem_emb=chem3, chem_mask=cm,
        )
        self.assertEqual(tuple(out_b.shape), (B, 32))

    def test_drug_fallback_deterministic(self) -> None:
        v1 = deterministic_standard_normal_vec("qq", 12)
        v2 = deterministic_standard_normal_vec("qq", 12)
        self.assertFalse(np.allclose(v1, deterministic_standard_normal_vec("zz", 12)))
        np.testing.assert_allclose(v1, v2)

    def test_drug_embedding_cache_hit_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _minimal_drug_cache(root)
            cache = DrugEmbeddingCache(root)
            v_hit, ok = cache.lookup("aspirin")
            self.assertTrue(ok)
            self.assertEqual(tuple(v_hit.shape), (8,))
            v_miss, ok2 = cache.lookup("zzz_nonexistent_row")
            self.assertFalse(ok2)
            self.assertEqual(tuple(v_miss.shape), (8,))

    def test_random_drug_fallback(self) -> None:
        fb = RandomDrugEmbeddingFallback(dim=5)
        a, ha = fb.lookup("a")
        self.assertFalse(ha)
        self.assertEqual(a.shape, (5,))

    def test_perturbation_batch_chem_three_d(self) -> None:
        with tempfile.TemporaryDirectory() as gd:
            gdir = Path(gd)
            _minimal_gene_cache(gdir)
            gc = GeneEmbeddingCache(gdir)
            B = 2
            metas = [
                ConditionMetadata(
                    genes=("TP53",),
                    perturbation_type_raw="KO",
                    chem_emb_list=[np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32) * 2],
                    chem_source=None,
                ),
                ConditionMetadata(genes=("TP53",), perturbation_type_raw="control"),
            ]
            pb = PerturbationBatch.from_metadata_list(
                metas, gc, max_genes=8, max_chem_slots=4, device=torch.device("cpu"),
            )
            self.assertEqual(tuple(pb.chem_emb.shape), (B, 4, 4))
            self.assertGreater(int(pb.chem_mask[0].sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
