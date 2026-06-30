"""Tests for multi-pool aggregation in UnifiedConditionEncoder."""
from __future__ import annotations

import unittest

import torch

from model.condition_emb.genepert.perturbation_encoder import UnifiedConditionEncoder

_PTSI = (0.0, -1.0, -1.0, -1.0, 1.0, 1.0)


class TestMultiPoolAggregation(unittest.TestCase):
    def _make(self, ops=None, scales=None, *, seed=0):
        torch.manual_seed(seed)
        kw = dict(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
        )
        if ops is not None:
            kw["pool_aggregations"] = tuple(ops)
        if scales is not None:
            kw["pool_scale_init"] = tuple(float(x) for x in scales)
        return UnifiedConditionEncoder(**kw)

    def test_pairwise_default_off_has_no_state_keys(self):
        enc = self._make(("mean", "max"), (1.0, 0.5), seed=40)
        sd = enc.state_dict()
        self.assertEqual(enc.pairwise_mode, "off")
        self.assertIsNone(enc.pair_to_out)
        self.assertFalse(any(k.startswith("pair_to_out.") for k in sd))

    def test_pairwise_zero_init_matches_off_for_multi_and_single(self):
        off = self._make(("mean", "max", "min"), (1.0, 1.0, 1.0), seed=41)
        torch.manual_seed(41)
        on = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pool_aggregations=("mean", "max", "min"),
            pool_scale_init=(1.0, 1.0, 1.0),
            pairwise_mode="hadamard_mean",
        )
        gid = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]], dtype=torch.long)
        m = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        npt = torch.tensor([3, 1], dtype=torch.long)
        tid = torch.zeros(2, dtype=torch.long)
        a = off(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt)
        b = on(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt)
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))

    def test_pairwise_order_invariant_and_combo_id_invariant(self):
        enc = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pool_aggregations=("mean", "max", "min"),
            pool_scale_init=(1.0, 0.5, 0.5),
            pairwise_mode="hadamard_mean",
        )
        gid = torch.tensor([[5, 7, 0], [7, 5, 0]], dtype=torch.long)
        m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        npt = torch.tensor([2, 2], dtype=torch.long)
        tid = torch.zeros(2, dtype=torch.long)
        cid = torch.tensor([101, 202], dtype=torch.long)
        out = enc(
            pert_gene_ids=gid,
            pert_mask=m,
            pert_type_id=tid,
            nperts=npt,
            combo_id=cid,
        )
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-6, rtol=1e-6))

    def test_pairwise_zero_for_single_and_control_rows(self):
        enc = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pairwise_mode="hadamard_mean",
        )
        ge = torch.randn(3, 4, 8)
        m = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
        proj = enc.gene_to_out(ge.to(dtype=torch.float32))
        pair = enc._pairwise_content(proj, m)
        self.assertTrue(torch.allclose(pair[0], torch.zeros_like(pair[0]), atol=1e-7))
        self.assertTrue(torch.allclose(pair[1], torch.zeros_like(pair[1]), atol=1e-7))

    def test_pairwise_concat_linear_keeps_pool_fuse_shape(self):
        enc = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pool_aggregations=("mean", "max"),
            pool_scale_init=(1.0, 0.5),
            pool_fusion_mode="concat_linear",
            pairwise_mode="hadamard_mean",
        )
        self.assertEqual(tuple(enc.pool_fuse.weight.shape), (16, 32))
        gid = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
        m = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        out = enc(
            pert_gene_ids=gid,
            pert_mask=m,
            pert_type_id=torch.zeros(2, dtype=torch.long),
            nperts=torch.tensor([2, 3], dtype=torch.long),
        )
        self.assertEqual(tuple(out.shape), (2, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_pairwise_gets_gradient_on_multi(self):
        enc = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pairwise_mode="hadamard_mean",
        )
        gid = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        m = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        out = enc(
            pert_gene_ids=gid,
            pert_mask=m,
            pert_type_id=torch.zeros(2, dtype=torch.long),
            nperts=torch.tensor([3, 2], dtype=torch.long),
        )
        (out.square().sum()).backward()
        self.assertIsNotNone(enc.pair_to_out.weight.grad)
        self.assertGreater(float(enc.pair_to_out.weight.grad.abs().sum()), 0.0)

    def test_default_eq_explicit_mean(self):
        enc0 = self._make()
        enc1 = self._make(("mean",), (1.0,))
        B, K = 2, 3
        gid = torch.randint(0, 64, (B, K))
        m = torch.zeros(B, K, dtype=torch.float32)
        m[:, :2] = 1.0
        npt = torch.tensor([2, 1], dtype=torch.long)
        tid = torch.zeros(B, dtype=torch.long)
        a = enc0(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt)
        b = enc1(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt)
        self.assertTrue(torch.allclose(a, b, atol=1e-7))

    def test_single_pert_mean_eq_sum_eq_max_eq_min(self):
        enc = self._make(("mean", "sum", "max", "min"), (1.0, 1.0, 1.0, 1.0))
        B, K = 2, 4
        ge = torch.randn(B, K, 8)
        m = torch.zeros(B, K, dtype=torch.float32)
        m[:, 0:1] = 1.0
        proj = enc.gene_to_out(ge.to(dtype=torch.float32))
        gm = enc._pool_one("mean", proj, m)
        gs = enc._pool_one("sum", proj, m)
        gx = enc._pool_one("max", proj, m)
        gn = enc._pool_one("min", proj, m)
        self.assertTrue(torch.allclose(gm, gs, atol=1e-6))
        self.assertTrue(torch.allclose(gm, gx, atol=1e-6))
        self.assertTrue(torch.allclose(gm, gn, atol=1e-6))

    def test_single_pert_three_paths_sum_eq_single_pool(self):
        e3 = self._make(("mean", "max", "min"), (2.0, 0.5, 0.5), seed=1)
        e1 = self._make(("mean",), (3.0,), seed=1)
        sd1 = e1.state_dict()
        sd3 = e3.state_dict()
        for k in sd1:
            if k == "pool_scale":
                continue
            if k in sd3 and sd3[k].shape == sd1[k].shape:
                sd3[k].copy_(sd1[k])
        B, K = 2, 3
        ge = torch.randn(B, K, 8)
        m = torch.zeros(B, K, dtype=torch.float32)
        m[:, :1] = 1.0
        npt = torch.ones(B, dtype=torch.long)
        tid = torch.zeros(B, dtype=torch.long)
        o3 = e3(pert_gene_emb=ge, pert_mask=m, pert_type_id=tid, nperts=npt)
        o1 = e1(pert_gene_emb=ge, pert_mask=m, pert_type_id=tid, nperts=npt)
        self.assertTrue(torch.allclose(o3, o1, atol=1e-5))

    def test_multi_pert_three_paths_differ(self):
        enc = self._make(("mean", "max", "min"), (1.0, 1.0, 1.0), seed=2)
        B, K = 1, 4
        ge = torch.randn(B, K, 8)
        m = torch.ones(B, K, dtype=torch.float32)
        proj = enc.gene_to_out(ge.to(dtype=torch.float32))
        pm = enc._pool_one("mean", proj, m)
        px = enc._pool_one("max", proj, m)
        pn = enc._pool_one("min", proj, m)
        self.assertFalse(torch.allclose(pm, px, atol=1e-3))
        self.assertFalse(torch.allclose(pm, pn, atol=1e-3))
        self.assertFalse(torch.allclose(px, pn, atol=1e-3))

    def test_multi_pert_sum_preserves_additive_content(self):
        enc = self._make(("mean", "sum"), (1.0, 1.0), seed=22)
        B, K = 1, 4
        ge = torch.randn(B, K, 8)
        m = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        proj = enc.gene_to_out(ge.to(dtype=torch.float32))
        pm = enc._pool_one("mean", proj, m)
        ps = enc._pool_one("sum", proj, m)
        self.assertTrue(torch.allclose(ps, pm * 3.0, atol=1e-5))

    def test_all_zero_mask_finite_and_zero(self):
        enc = self._make(("mean", "max", "min"), (1.0, 0.5, 0.5), seed=3)
        B, K = 3, 4
        gid = torch.randint(0, 64, (B, K))
        m = torch.zeros(B, K, dtype=torch.float32)
        npt = torch.zeros(B, dtype=torch.long)
        tid = torch.zeros(B, dtype=torch.long)
        out = enc(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_state_dict_has_pool_scale(self):
        enc = self._make(("mean", "sum", "max", "min"), (1.0, 0.5, 0.5, 0.5))
        sd = enc.state_dict()
        self.assertIn("pool_scale", sd)
        self.assertEqual(tuple(sd["pool_scale"].shape), (4,))

    def test_old_ckpt_strict_false_load(self):
        old = self._make(("mean",), (1.0,), seed=4)
        sd_old = {k: v for k, v in old.state_dict().items() if k != "pool_scale"}
        new = self._make(("mean", "max", "min"), (1.0, 0.5, 0.5), seed=99)
        new.load_state_dict(sd_old, strict=False)
        self.assertEqual(new.pool_scale.numel(), 3)
        self.assertAlmostEqual(float(new.pool_scale[0].detach()), 1.0, places=5)

    def test_value_error_len_mismatch(self):
        with self.assertRaises(ValueError):
            self._make(("mean", "max"), (1.0,))

    def test_value_error_unknown_op(self):
        with self.assertRaises(ValueError):
            self._make(("median",), (1.0,))

    def test_value_error_empty_ops(self):
        with self.assertRaises(ValueError):
            UnifiedConditionEncoder(
                "random_learned",
                16,
                num_embeddings_random=64,
                embed_dim_random=8,
                chem_emb_dim=None,
                pert_type_scale_init=_PTSI,
                pool_aggregations=(),
                pool_scale_init=(),
            )


    def test_concat_linear_vector_gate_forward(self) -> None:
        enc = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=16,
            num_embeddings_random=64,
            embed_dim_random=8,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pool_aggregations=("mean", "max"),
            pool_scale_init=(1.0, 0.5),
            pool_fusion_mode="concat_linear",
            type_adapter_mode="vector_scale_gate",
        )
        B, K = 2, 3
        gid = torch.randint(0, 64, (B, K))
        m = torch.zeros(B, K, dtype=torch.float32)
        m[:, :2] = 1.0
        npt = torch.tensor([2, 1], dtype=torch.long)
        tid = torch.zeros(B, dtype=torch.long)
        out = enc(pert_gene_ids=gid, pert_mask=m, pert_type_id=tid, nperts=npt)
        self.assertEqual(tuple(out.shape), (B, 16))

    def test_concat_linear_gradients(self) -> None:
        enc = UnifiedConditionEncoder(
            mode="random_learned",
            out_dim=12,
            num_embeddings_random=48,
            embed_dim_random=6,
            chem_emb_dim=None,
            pert_type_scale_init=_PTSI,
            pool_aggregations=("mean",),
            pool_scale_init=(1.0,),
            pool_fusion_mode="concat_linear",
            type_adapter_mode="vector_scale_gate",
        )
        gid = torch.randint(1, 40, (2, 4))
        m = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        out = enc(
            pert_gene_ids=gid,
            pert_mask=m,
            pert_type_id=torch.zeros(2, dtype=torch.long),
            nperts=torch.tensor([2, 0], dtype=torch.long),
        )
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(enc.pool_fuse.weight.grad)
        self.assertIsNotNone(enc.type_vector_scale.grad)
        self.assertIsNotNone(enc.type_gate_delta.grad)


if __name__ == "__main__":
    unittest.main()
