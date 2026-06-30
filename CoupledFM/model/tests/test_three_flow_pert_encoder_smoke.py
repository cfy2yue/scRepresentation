"""Smoke: perturbation branch builds and runs a single forward on three stacks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.models.velocity_field import RawExprVelocityField  # noqa: E402
from model.latent.models.mlp import ControlMLPVelocityField  # noqa: E402


class TestThreeFlowPertEncoderSmoke(unittest.TestCase):
    def test_control_mlp_concat_gate(self) -> None:
        B, K, emb = 2, 4, 2058
        m = ControlMLPVelocityField(
            emb_dim=emb,
            d_model=64,
            n_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            use_pert_condition=True,
            pert_embed_mode="random_learned",
            pert_cond_dim=32,
            pert_gene_emb_dim=8,
            pert_encoder_num_embeddings=128,
            pool_aggregations=("mean", "max"),
            pool_scale_init=(1.0, 0.5),
            pool_fusion_mode="concat_linear",
            type_adapter_mode="vector_scale_gate",
            use_pert_in_fusion=True,
        )
        gid = torch.randint(1, 120, (B, K), dtype=torch.long)
        pmask = torch.zeros(B, K, dtype=torch.float32)
        pmask[:, :2] = 1.0
        tid = torch.zeros(B, dtype=torch.long)
        npt = torch.full((B,), 2, dtype=torch.long)
        xt = torch.randn(B, emb)
        x0 = torch.randn(B, emb)
        t = torch.rand(B)
        v = m(
            xt, t, x0,
            pert_gene_ids=gid,
            pert_mask=pmask,
            pert_type_id=tid,
            nperts=npt,
        )
        self.assertEqual(v.shape, (B, emb))
        self.assertTrue(torch.isfinite(v).all())

    def test_raw_expr_velocity_concat_gate(self) -> None:
        G, B, K, d_model = 11, 2, 3, 64
        vf = RawExprVelocityField(
            d_model=d_model,
            n_layer=1,
            n_head=4,
            d_ff=128,
            dropout=0.0,
            d_latent=32,
            coupling_mode="ot",
            use_latent_resampler=False,
            use_pert_token=False,
            use_pert_condition=True,
            pert_embed_mode="random_learned",
            pert_cond_dim=24,
            pert_gene_emb_dim=8,
            pert_encoder_num_embeddings=96,
            pool_aggregations=("mean", "max"),
            pool_scale_init=(1.0, 1.0),
            pool_fusion_mode="concat_linear",
            type_adapter_mode="vector_scale_gate",
        )
        x_t = torch.randn(B, G)
        x_c = torch.randn(B, G)
        t = torch.rand(B)
        gene_ids = torch.randint(0, 40000, (G,), dtype=torch.long)
        gid = torch.randint(1, 90, (B, K), dtype=torch.long)
        pmask = torch.zeros(B, K, dtype=torch.float32)
        pmask[:, 0] = 1.0
        tid = torch.zeros(B, dtype=torch.long)
        npt = torch.ones(B, dtype=torch.long)
        v = vf(
            x_t,
            x_c,
            t,
            gene_ids,
            aux_emb=torch.randn(B, 32),
            pert_gene_ids=gid,
            pert_mask=pmask,
            pert_type_id=tid,
            nperts=npt,
        )
        self.assertEqual(v.shape, (B, G))
        self.assertTrue(torch.isfinite(v).all())

    def test_coupled_independent_raw_forward(self) -> None:
        G, B, K, d_model = 9, 2, 3, 56
        vf = RawExprVelocityField(
            d_model=d_model,
            n_layer=1,
            n_head=4,
            d_ff=112,
            dropout=0.0,
            d_latent=24,
            coupling_mode="ot",
            use_latent_resampler=False,
            use_pert_token=False,
            use_pert_condition=True,
            pert_embed_mode="random_learned",
            pert_cond_dim=20,
            pert_gene_emb_dim=8,
            pert_encoder_num_embeddings=80,
            pool_aggregations=("mean",),
            pool_scale_init=(1.0,),
            pool_fusion_mode="sum",
            type_adapter_mode="scalar",
        )
        x_t = torch.randn(B, G)
        x_c = torch.randn(B, G)
        t = torch.rand(B)
        gene_ids = torch.randint(0, 40000, (G,), dtype=torch.long)
        gid = torch.randint(1, 70, (B, K), dtype=torch.long)
        pmask = torch.zeros(B, K, dtype=torch.float32)
        pmask[:, 0] = 1.0
        tid = torch.zeros(B, dtype=torch.long)
        npt = torch.ones(B, dtype=torch.long)
        v = vf(
            x_t,
            x_c,
            t,
            gene_ids,
            aux_emb=torch.randn(B, 24),
            pert_gene_ids=gid,
            pert_mask=pmask,
            pert_type_id=tid,
            nperts=npt,
        )
        self.assertEqual(v.shape, (B, G))


if __name__ == "__main__":
    unittest.main()
