"""Tests for condition-delta additive/interaction decomposition surfaces."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.models.mlp import ControlMLPVelocityField  # noqa: E402
from model.latent.eval_condition_delta_decomposition import _summarize  # noqa: E402


def _tiny_condition_delta_model() -> ControlMLPVelocityField:
    return ControlMLPVelocityField(
        emb_dim=6,
        d_model=16,
        n_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_cond_dim=8,
        pert_gene_emb_dim=4,
        pert_encoder_num_embeddings=32,
        condition_delta_head_hidden=16,
        use_condition_delta_head=True,
    )


def test_interaction_condition_delta_is_combo_minus_additive() -> None:
    torch.manual_seed(7)
    model = _tiny_condition_delta_model()
    gid = torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    tid = torch.zeros(2, dtype=torch.long)
    npt = mask.sum(dim=1).to(dtype=torch.long)

    combo = model.predict_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=tid,
        nperts=npt,
    )
    additive = model.predict_additive_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=tid,
        nperts=npt,
    )
    interaction = model.predict_interaction_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=tid,
        nperts=npt,
    )

    assert combo.shape == additive.shape == interaction.shape == (2, 6)
    assert torch.isfinite(interaction).all()
    assert torch.allclose(interaction, combo - additive)


def test_interaction_condition_delta_requires_enabled_head() -> None:
    model = ControlMLPVelocityField(
        emb_dim=4,
        d_model=8,
        n_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_cond_dim=8,
        pert_gene_emb_dim=4,
        pert_encoder_num_embeddings=32,
        use_condition_delta_head=False,
    )
    gid = torch.tensor([[2, 3]], dtype=torch.long)
    mask = torch.tensor([[1, 1]], dtype=torch.float32)

    try:
        model.predict_interaction_condition_delta(pert_gene_ids=gid, pert_mask=mask)
    except RuntimeError as exc:
        assert "condition_delta_head is disabled" in str(exc)
    else:
        raise AssertionError("disabled condition_delta_head should reject interaction prediction")


def test_decomposition_summary_reports_geometry_ratios() -> None:
    rows = [
        {
            "dataset": "D",
            "groups": "test_multi_seen",
            "combo_endpoint_cosine": 0.1,
            "additive_endpoint_cosine": 0.2,
            "interaction_endpoint_cosine": -0.1,
            "combo_pert_residual_cosine": 0.3,
            "additive_pert_residual_cosine": 0.4,
            "interaction_pert_residual_cosine": -0.2,
            "combo_additive_cosine": 0.99,
            "additive_norm_ratio": 2.0,
            "interaction_norm_ratio": 1.0,
        },
        {
            "dataset": "D",
            "groups": "test_multi_seen",
            "combo_endpoint_cosine": 0.3,
            "additive_endpoint_cosine": 0.4,
            "interaction_endpoint_cosine": -0.3,
            "combo_pert_residual_cosine": 0.5,
            "additive_pert_residual_cosine": 0.6,
            "interaction_pert_residual_cosine": -0.4,
            "combo_additive_cosine": 1.0,
            "additive_norm_ratio": 1.8,
            "interaction_norm_ratio": 0.9,
        },
    ]

    summary = _summarize(rows)
    assert len(summary) == 1
    assert summary[0]["n"] == 2
    assert summary[0]["mean_combo_endpoint_cosine"] == 0.2
    assert summary[0]["mean_combo_additive_cosine"] == 0.995
    assert summary[0]["mean_additive_norm_ratio"] == 1.9
    assert summary[0]["mean_interaction_norm_ratio"] == 0.95
