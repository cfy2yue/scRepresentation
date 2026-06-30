"""Smoke tests for LatentFM gene-condition embedding source compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.config import Config
from model.latent.train import (
    apply_finetune_freeze,
    build_trackc_support_set_task_bank,
    build_model,
    fill_condition_embedding_source,
    load_model_weights_only,
    make_trackc_support_set_task_batch,
    validate_support_context_config,
)
from model.condition_emb.genepert import PERT_TYPE_CRISPRI, PERT_TYPE_DRUG, PERT_TYPE_NULL
from model.utils.conditioning.perturbation import ConditionMetadata, PerturbationBatch
from model.utils.embeddings.gene_cache import GeneEmbeddingCache


_CACHE_ROOT = Path("/data/cyx/1030/scLatent/pretrainckpt/genepert_cache")


@pytest.mark.parametrize("source,expected_dim", [("scgpt_embed_gene", 512), ("cellnavi_embed_gene", 256)])
def test_latent_condition_embedding_source_shape_smoke(source: str, expected_dim: int) -> None:
    cache_dir = _CACHE_ROOT / source
    if not (cache_dir / "manifest.json").is_file():
        pytest.skip(f"missing gene embedding cache: {cache_dir}")

    cache = GeneEmbeddingCache(cache_dir)
    assert cache.embed_dim == expected_dim
    rows = [
        ConditionMetadata(
            genes=("CEBPE", "SPI1"),
            perturbation_type_raw="CRISPRi",
            combo_id=7,
            nperts_obs=2,
        ),
        ConditionMetadata(
            genes=("TP53",),
            perturbation_type_raw="knockout",
            combo_id=8,
            nperts_obs=1,
        ),
        ConditionMetadata(genes=(), perturbation_type_raw=None, combo_id=0, nperts_obs=0),
    ]
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=16,
        device=torch.device("cpu"),
    ).as_tuple_full()

    cfg = Config(
        emb_dim=3072,
        mlp_d_model=512,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="pretrained_frozen",
        pert_gene_emb_cache_dir=str(cache_dir),
        pert_condition_embedding_source=source,
        pert_cond_dim=512,
        pert_gene_projector_hidden=1024,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pool_fusion_mode="sum",
        pert_type_adapter_mode="scalar",
        pert_to_c_init_mode="xavier_small",
        use_pert_in_fusion=True,
        condition_delta_head_use_in_model=True,
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    gid, mask, tid, nperts, cid, chem_emb, chem_mask = pb

    assert mask.sum(dim=1).tolist() == [2, 1, 0]
    assert nperts.tolist() == [2, 1, 0]
    with torch.no_grad():
        x = torch.randn(3, 3072)
        t = torch.linspace(0.0, 1.0, 3)
        pert_projection = model._pert_projection(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=tid,
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        velocity = model(
            x,
            t,
            x,
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=tid,
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        delta = model.predict_condition_delta(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=tid,
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )

    assert pert_projection.shape == (3, 512)
    assert velocity.shape == (3, 3072)
    assert delta.shape == (3, 3072)
    assert torch.isfinite(pert_projection).all()
    assert torch.isfinite(velocity).all()
    assert torch.isfinite(delta).all()


def test_condition_embedding_source_is_inferred_from_cache_dir() -> None:
    cfg = Config(
        use_pert_condition=True,
        pert_gene_emb_cache_dir=str(_CACHE_ROOT / "cellnavi_embed_gene"),
        pert_condition_embedding_source="",
    )

    fill_condition_embedding_source(cfg)

    assert cfg.pert_condition_embedding_source == "cellnavi_embed_gene"


def test_condition_embedding_source_mismatch_warns() -> None:
    cfg = Config(
        use_pert_condition=True,
        pert_gene_emb_cache_dir=str(_CACHE_ROOT / "cellnavi_embed_gene"),
        pert_condition_embedding_source="scgpt_embed_gene",
    )

    with pytest.warns(RuntimeWarning, match="does not match"):
        fill_condition_embedding_source(cfg)


def test_init_checkpoint_skips_shape_mismatch_for_condition_source_swap(tmp_path: Path) -> None:
    scgpt_cache = _CACHE_ROOT / "scgpt_embed_gene"
    cellnavi_cache = _CACHE_ROOT / "cellnavi_embed_gene"
    if not (scgpt_cache / "manifest.json").is_file() or not (cellnavi_cache / "manifest.json").is_file():
        pytest.skip("missing scGPT or CellNavi gene embedding cache")

    base_kwargs = dict(
        emb_dim=64,
        mlp_d_model=32,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="pretrained_frozen",
        pert_cond_dim=32,
        pert_gene_projector_hidden=16,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pool_fusion_mode="sum",
        pert_type_adapter_mode="scalar",
        pert_to_c_init_mode="xavier_small",
        use_pert_in_fusion=True,
    )
    scgpt_cfg = Config(
        **base_kwargs,
        pert_gene_emb_cache_dir=str(scgpt_cache),
        pert_condition_embedding_source="scgpt_embed_gene",
    )
    cellnavi_cfg = Config(
        **base_kwargs,
        pert_gene_emb_cache_dir=str(cellnavi_cache),
        pert_condition_embedding_source="cellnavi_embed_gene",
    )
    source_model = build_model(scgpt_cfg, torch.device("cpu"))
    target_model = build_model(cellnavi_cfg, torch.device("cpu"))
    ckpt = tmp_path / "scgpt_condition.pt"
    torch.save({"model": source_model.state_dict()}, ckpt)

    missing, unexpected, skipped = load_model_weights_only(
        ckpt,
        target_model,
        torch.device("cpu"),
        strict=False,
    )

    assert not unexpected
    assert skipped
    assert any("pert_encoder.gene_table" in key or "pert_encoder.gene_to_out" in key for key in skipped)
    assert missing


def test_null_perturbation_type_keeps_gene_condition_visible() -> None:
    cache_dir = _CACHE_ROOT / "scgpt_embed_gene"
    if not (cache_dir / "manifest.json").is_file():
        pytest.skip(f"missing gene embedding cache: {cache_dir}")

    cache = GeneEmbeddingCache(cache_dir)
    pb = PerturbationBatch.from_metadata_list(
        [
            ConditionMetadata(
                genes=("TP53",),
                perturbation_type_raw="CRISPRi",
                combo_id=8,
                nperts_obs=1,
            )
        ],
        cache,
        max_genes=16,
        device=torch.device("cpu"),
    ).as_tuple_full()
    gid, mask, tid, nperts, cid, chem_emb, chem_mask = pb

    cfg = Config(
        emb_dim=3072,
        mlp_d_model=512,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="pretrained_frozen",
        pert_gene_emb_cache_dir=str(cache_dir),
        pert_condition_embedding_source="scgpt_embed_gene",
        pert_cond_dim=512,
        pert_gene_projector_hidden=1024,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pool_fusion_mode="sum",
        pert_type_adapter_mode="scalar",
        pert_to_c_init_mode="xavier_small",
        use_pert_in_fusion=True,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        null_projection = model._pert_projection(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=torch.full_like(tid, PERT_TYPE_NULL),
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        crispri_projection = model._pert_projection(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=torch.full_like(tid, PERT_TYPE_CRISPRI),
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )

    assert torch.isfinite(null_projection).all()
    assert torch.isfinite(crispri_projection).all()
    assert null_projection.norm().item() > 1e-6
    assert crispri_projection.norm().item() > 1e-6


def test_condition_metadata_sorts_and_dedupes_gene_combos_by_default() -> None:
    ab = ConditionMetadata.from_obs_fields(
        "TP53 + CEBPE",
        perturbation_type_field="CRISPRi",
        combo_id=11,
    )
    ba = ConditionMetadata.from_obs_fields(
        "CEBPE + TP53",
        perturbation_type_field="CRISPRi",
        combo_id=12,
    )
    repeated = ConditionMetadata.from_obs_fields(
        "TP53 + TP53 + CEBPE",
        perturbation_type_field="CRISPRi",
        combo_id=13,
    )

    assert ab.genes == ba.genes == repeated.genes == ("CEBPE", "TP53")
    assert ab.resolved_nperts() == ba.resolved_nperts() == repeated.resolved_nperts() == 2


def test_unified_condition_projection_is_invariant_to_gene_order_and_combo_id() -> None:
    cache_dir = _CACHE_ROOT / "scgpt_embed_gene"
    if not (cache_dir / "manifest.json").is_file():
        pytest.skip(f"missing gene embedding cache: {cache_dir}")

    cache = GeneEmbeddingCache(cache_dir)
    rows = [
        ConditionMetadata.from_obs_fields(
            "TP53 + CEBPE",
            perturbation_type_field="CRISPRi",
            combo_id=101,
        ),
        ConditionMetadata.from_obs_fields(
            "CEBPE + TP53",
            perturbation_type_field="CRISPRi",
            combo_id=202,
        ),
    ]
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=16,
        device=torch.device("cpu"),
    ).as_tuple_full()
    gid, mask, tid, nperts, cid, chem_emb, chem_mask = pb
    assert cid.tolist() == [101, 202]

    cfg = Config(
        emb_dim=3072,
        mlp_d_model=512,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="pretrained_frozen",
        pert_gene_emb_cache_dir=str(cache_dir),
        pert_condition_embedding_source="scgpt_embed_gene",
        pert_cond_dim=512,
        pert_gene_projector_hidden=1024,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pool_fusion_mode="sum",
        pert_type_adapter_mode="scalar",
        use_pert_in_fusion=True,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        pert_projection = model._pert_projection(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=tid,
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )

    assert torch.allclose(pert_projection[0], pert_projection[1], atol=1e-6, rtol=1e-6)


def test_unified_condition_projection_ignores_combo_id_for_same_gene_set() -> None:
    cache_dir = _CACHE_ROOT / "scgpt_embed_gene"
    if not (cache_dir / "manifest.json").is_file():
        pytest.skip(f"missing gene embedding cache: {cache_dir}")

    cache = GeneEmbeddingCache(cache_dir)
    rows = [
        ConditionMetadata(
            genes=("TP53", "CEBPE"),
            perturbation_type_raw="CRISPRi",
            combo_id=1,
            nperts_obs=2,
        ),
        ConditionMetadata(
            genes=("TP53", "CEBPE"),
            perturbation_type_raw="CRISPRi",
            combo_id=4095,
            nperts_obs=2,
        ),
    ]
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=16,
        device=torch.device("cpu"),
    ).as_tuple_full()
    gid, mask, tid, nperts, cid, chem_emb, chem_mask = pb
    assert cid.tolist() == [1, 4095]

    cfg = Config(
        emb_dim=3072,
        mlp_d_model=512,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="pretrained_frozen",
        pert_gene_emb_cache_dir=str(cache_dir),
        pert_condition_embedding_source="scgpt_embed_gene",
        pert_cond_dim=512,
        pert_gene_projector_hidden=1024,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pool_fusion_mode="sum",
        pert_type_adapter_mode="scalar",
        use_pert_in_fusion=True,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        pert_projection = model._pert_projection(
            pert_gene_ids=gid,
            pert_mask=mask,
            pert_type_id=tid,
            nperts=nperts,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )

    assert torch.allclose(pert_projection[0], pert_projection[1], atol=1e-6, rtol=1e-6)


def test_latent_control_mlp_pairwise_condition_forward_smoke() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pool_fusion_mode="concat_linear",
        pert_type_adapter_mode="scalar",
        pert_pairwise_mode="hadamard_mean",
        pert_to_c_init_mode="xavier_small",
        use_pert_in_fusion=True,
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    x_t = torch.randn(2, 32)
    x_0 = torch.randn(2, 32)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    gid = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = model(
        x_t,
        t,
        x_0,
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=torch.full((2,), PERT_TYPE_CRISPRI, dtype=torch.long),
        nperts=torch.tensor([2, 3], dtype=torch.long),
        combo_id=torch.tensor([11, 12], dtype=torch.long),
    )
    assert tuple(out.shape) == (2, 32)
    assert torch.isfinite(out).all()


def test_trackc_support_context_default_off_has_no_state_and_rejects_context() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    assert not any("support_context_to_c" in key for key in model.state_dict())
    assert not any("support_context_to_v" in key for key in model.state_dict())
    assert not any("support_context_to_v_scale" in key for key in model.state_dict())

    x_t = torch.randn(2, 8)
    x_0 = torch.randn(2, 8)
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)
    with torch.no_grad():
        out = model(x_t, t, x_0)
    assert tuple(out.shape) == (2, 8)

    with pytest.raises(ValueError, match="support_context"):
        model(
            x_t,
            t,
            x_0,
            support_context=torch.randn(2, 4),
        )


def test_trackc_support_set_task_default_off_has_no_state_and_rejects_task() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    assert not any("support_set_task_to_c" in key for key in model.state_dict())

    x_t = torch.randn(2, 8)
    x_0 = torch.randn(2, 8)
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)
    with torch.no_grad():
        out = model(x_t, t, x_0)
    assert tuple(out.shape) == (2, 8)

    with pytest.raises(ValueError, match="support_set_task"):
        model(
            x_t,
            t,
            x_0,
            support_set_task=torch.randn(2, 4),
        )


def test_trackc_support_context_requires_control_mlp_and_positive_dim() -> None:
    with pytest.raises(ValueError, match="control_mlp"):
        build_model(
            Config(
                model_type="mlp",
                trackc_support_context_use_in_model=True,
                trackc_support_context_dim=4,
            ),
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="support_context_dim"):
        build_model(
            Config(
                emb_dim=8,
                mlp_d_model=4,
                mlp_n_layers=1,
                model_type="control_mlp",
                trackc_support_context_use_in_model=True,
                trackc_support_context_dim=0,
            ),
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="control_mlp"):
        build_model(
            Config(
                model_type="mlp",
                trackc_support_set_task_use_in_model=True,
                trackc_support_set_task_dim=4,
            ),
            torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="support_set_task_dim"):
        build_model(
            Config(
                emb_dim=8,
                mlp_d_model=4,
                mlp_n_layers=1,
                model_type="control_mlp",
                trackc_support_set_task_use_in_model=True,
                trackc_support_set_task_dim=0,
            ),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="support_context_dim"):
        build_model(
            Config(
                emb_dim=8,
                mlp_d_model=4,
                mlp_n_layers=1,
                model_type="control_mlp",
                trackc_support_residual_use_in_model=True,
                trackc_support_context_dim=0,
            ),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="support_context_dim"):
        build_model(
            Config(
                emb_dim=8,
                mlp_d_model=4,
                mlp_n_layers=1,
                model_type="control_mlp",
                trackc_support_film_use_in_model=True,
                trackc_support_context_dim=0,
            ),
            torch.device("cpu"),
        )


def test_trackc_support_context_fail_closed_validation() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_context_use_in_model=True,
        trackc_support_context_dim=3,
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    x_t = torch.randn(2, 8)
    x_0 = torch.randn(2, 8)
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires support_context"):
        model(x_t, t, x_0)
    with pytest.raises(ValueError, match="batch size"):
        model(x_t, t, x_0, support_context=torch.randn(1, 3))
    with pytest.raises(ValueError, match="feature dimension"):
        model(x_t, t, x_0, support_context=torch.randn(2, 4))
    bad = torch.randn(2, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model(x_t, t, x_0, support_context=bad)


def test_trackc_support_set_task_fail_closed_validation() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=3,
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    x_t = torch.randn(2, 8)
    x_0 = torch.randn(2, 8)
    t = torch.tensor([0.25, 0.75], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires support_set_task"):
        model(x_t, t, x_0)
    with pytest.raises(ValueError, match="batch size"):
        model(x_t, t, x_0, support_set_task=torch.randn(1, 3))
    with pytest.raises(ValueError, match="feature dimension"):
        model(x_t, t, x_0, support_set_task=torch.randn(2, 4))
    bad = torch.randn(2, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model(x_t, t, x_0, support_set_task=bad)


def test_trackc_support_context_forward_consumes_context_signal() -> None:
    torch.manual_seed(7)
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_context_use_in_model=True,
        trackc_support_context_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        model.support_context_to_c.weight.copy_(torch.eye(4))
        block = model.blocks[0]
        block.ada[-1].weight.zero_()
        block.ada[-1].bias.zero_()
        block.ada[-1].weight[:4, :4].copy_(torch.eye(4))
        block.ada[-1].bias[8:12].fill_(1.0)
        block.mlp[0].weight.zero_()
        block.mlp[0].bias.zero_()
        block.mlp[0].weight[:4, :4].copy_(torch.eye(4))
        block.mlp[3].weight.zero_()
        block.mlp[3].bias.zero_()
        block.mlp[3].weight[:, :4].copy_(torch.eye(4))
        model.output_proj.weight.copy_(torch.eye(4))
        model.output_proj.bias.zero_()

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    context_a = torch.zeros(2, 4)
    context_b = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])
    with torch.no_grad():
        out_a = model(x_t, t, x_0, support_context=context_a)
        out_b = model(x_t, t, x_0, support_context=context_b)

    assert tuple(out_a.shape) == (2, 4)
    assert tuple(out_b.shape) == (2, 4)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert not torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_trackc_support_set_task_forward_consumes_task_signal() -> None:
    torch.manual_seed(7)
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        model.support_set_task_to_c.weight.copy_(torch.eye(4))
        block = model.blocks[0]
        block.ada[-1].weight.zero_()
        block.ada[-1].bias.zero_()
        block.ada[-1].weight[:4, :4].copy_(torch.eye(4))
        block.ada[-1].bias[8:12].fill_(1.0)
        block.mlp[0].weight.zero_()
        block.mlp[0].bias.zero_()
        block.mlp[0].weight[:4, :4].copy_(torch.eye(4))
        block.mlp[3].weight.zero_()
        block.mlp[3].bias.zero_()
        block.mlp[3].weight[:, :4].copy_(torch.eye(4))
        model.output_proj.weight.copy_(torch.eye(4))
        model.output_proj.bias.zero_()

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    task_a = torch.zeros(2, 4)
    task_b = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])
    with torch.no_grad():
        out_a = model(x_t, t, x_0, support_set_task=task_a)
        out_b = model(x_t, t, x_0, support_set_task=task_b)

    assert tuple(out_a.shape) == (2, 4)
    assert tuple(out_b.shape) == (2, 4)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert not torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_trackc_support_residual_forward_consumes_context_signal() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_residual_use_in_model=True,
        trackc_support_context_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        model.support_context_to_v.weight.copy_(torch.eye(4))

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    context_a = torch.zeros(2, 4)
    context_b = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])
    with torch.no_grad():
        out_a = model(x_t, t, x_0, support_context=context_a)
        out_b = model(x_t, t, x_0, support_context=context_b)

    assert tuple(out_a.shape) == (2, 4)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert torch.allclose(out_b - out_a, context_b, atol=1e-6, rtol=1e-6)


def test_trackc_support_film_forward_consumes_context_signal() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_film_use_in_model=True,
        trackc_support_context_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    with torch.no_grad():
        model.output_proj.weight.copy_(torch.eye(4))
        model.output_proj.bias.fill_(1.0)
        model.support_context_to_v.weight.copy_(torch.eye(4))
        model.support_context_to_v_scale.weight.copy_(0.5 * torch.eye(4))

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    context_a = torch.zeros(2, 4)
    context_b = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])
    with torch.no_grad():
        out_a = model(x_t, t, x_0, support_context=context_a)
        out_b = model(x_t, t, x_0, support_context=context_b)

    assert tuple(out_a.shape) == (2, 4)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_b).all()
    assert not torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_trackc_support_residual_zero_context_exact_noop_after_nonzero_weights() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_residual_use_in_model=True,
        trackc_support_context_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    zero_context = torch.zeros(2, 4)
    nonzero_context = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])

    with torch.no_grad():
        before = model(x_t, t, x_0, support_context=zero_context)
        model.support_context_to_v.weight.copy_(torch.eye(4))
        zero_after = model(x_t, t, x_0, support_context=zero_context)
        masked_after = model(
            x_t,
            t,
            x_0,
            support_context=nonzero_context,
            support_context_present=torch.zeros(2),
        )
        present_after = model(
            x_t,
            t,
            x_0,
            support_context=nonzero_context,
            support_context_present=torch.ones(2),
        )

    assert torch.allclose(before, zero_after, atol=1e-6, rtol=1e-6)
    assert torch.allclose(before, masked_after, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(before, present_after, atol=1e-6, rtol=1e-6)


def test_trackc_support_set_task_zero_task_exact_noop_after_nonzero_weights() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    zero_task = torch.zeros(2, 4)
    nonzero_task = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])

    with torch.no_grad():
        block = model.blocks[0]
        block.ada[-1].weight.zero_()
        block.ada[-1].bias.zero_()
        block.ada[-1].weight[:4, :4].copy_(torch.eye(4))
        block.ada[-1].bias[8:12].fill_(1.0)
        block.mlp[0].weight.zero_()
        block.mlp[0].bias.zero_()
        block.mlp[0].weight[:4, :4].copy_(torch.eye(4))
        block.mlp[3].weight.zero_()
        block.mlp[3].bias.zero_()
        block.mlp[3].weight[:, :4].copy_(torch.eye(4))
        model.output_proj.weight.copy_(torch.eye(4))
        model.output_proj.bias.zero_()
        before = model(x_t, t, x_0, support_set_task=zero_task)
        model.support_set_task_to_c.weight.copy_(torch.eye(4))
        zero_after = model(x_t, t, x_0, support_set_task=zero_task)
        masked_after = model(
            x_t,
            t,
            x_0,
            support_set_task=nonzero_task,
            support_set_task_present=torch.zeros(2),
        )
        present_after = model(
            x_t,
            t,
            x_0,
            support_set_task=nonzero_task,
            support_set_task_present=torch.ones(2),
        )

    assert torch.allclose(before, zero_after, atol=1e-6, rtol=1e-6)
    assert torch.allclose(before, masked_after, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(before, present_after, atol=1e-6, rtol=1e-6)


def test_trackc_support_set_task_min_support_count_gates_token(tmp_path: Path) -> None:
    safe_split = tmp_path / "split_seed42_multi_support_v2_trainselect.json"
    safe_split.write_text("{}", encoding="utf-8")
    anchor_path = tmp_path / "anchor_means.json"
    candidate_path = tmp_path / "candidate_means.json"

    def row(condition: str, pred: list[float]) -> dict[str, object]:
        return {
            "dataset": "D",
            "condition": condition,
            "pred_mean": pred,
        }

    anchor = {
        "split_file": str(safe_split),
        "groups": {
            "train_multi": {
                "condition_metrics": [
                    row("A+B", [0.0, 0.0, 0.0]),
                    row("A+C", [0.0, 0.0, 0.0]),
                    row("B+D", [0.0, 0.0, 0.0]),
                ]
            }
        },
    }
    candidate = {
        "split_file": str(safe_split),
        "groups": {
            "train_multi": {
                "condition_metrics": [
                    row("A+B", [2.0, 0.0, 0.0]),
                    row("A+C", [1.0, 0.0, 0.0]),
                    row("B+D", [0.0, 1.0, 0.0]),
                ]
            }
        },
    }
    import json

    anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    cfg = Config(
        emb_dim=3,
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=3,
        trackc_support_set_task_source="shared_gene_condition_means",
        trackc_support_set_task_safe_split_file=str(safe_split),
        trackc_support_set_task_anchor_condition_means=str(anchor_path),
        trackc_support_set_task_candidate_condition_means=str(candidate_path),
        trackc_support_set_task_min_support_count=2,
    )
    bank = build_trackc_support_set_task_bank(cfg)
    task, present = make_trackc_support_set_task_batch(
        bank,
        "D",
        "A+B",
        2,
        cfg,
        torch.device("cpu"),
    )
    assert present is not None
    assert task is not None
    assert torch.allclose(present, torch.ones(2, 1))
    assert torch.allclose(task[0], torch.tensor([0.5, 0.5, 0.0]))

    task_single, present_single = make_trackc_support_set_task_batch(
        bank,
        "D",
        "A+C",
        2,
        cfg,
        torch.device("cpu"),
    )
    assert present_single is not None
    assert task_single is not None
    assert torch.allclose(present_single, torch.zeros(2, 1))
    assert torch.allclose(task_single, torch.zeros(2, 3))


def test_trackc_support_film_zero_context_exact_noop_after_nonzero_weights() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_film_use_in_model=True,
        trackc_support_context_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()

    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    zero_context = torch.zeros(2, 4)
    nonzero_context = torch.tensor([[1.0, -0.5, 0.25, 0.0], [0.0, 0.5, -1.0, 1.5]])

    with torch.no_grad():
        before = model(x_t, t, x_0, support_context=zero_context)
        model.support_context_to_v.weight.copy_(torch.eye(4))
        model.support_context_to_v_scale.weight.copy_(0.5 * torch.eye(4))
        zero_after = model(x_t, t, x_0, support_context=zero_context)
        masked_after = model(
            x_t,
            t,
            x_0,
            support_context=nonzero_context,
            support_context_present=torch.zeros(2, 1),
        )
        present_after = model(
            x_t,
            t,
            x_0,
            support_context=nonzero_context,
            support_context_present=torch.ones(2, 1),
        )

    assert torch.allclose(before, zero_after, atol=1e-6, rtol=1e-6)
    assert torch.allclose(before, masked_after, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(before, present_after, atol=1e-6, rtol=1e-6)


def test_trackc_support_present_mask_validation() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_residual_use_in_model=True,
        trackc_support_context_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    context = torch.randn(2, 4)

    with pytest.raises(ValueError, match="support_context_present"):
        model(x_t, t, x_0, support_context=context, support_context_present=torch.ones(2, 2))
    with pytest.raises(ValueError, match="0/1"):
        model(x_t, t, x_0, support_context=context, support_context_present=torch.tensor([0.5, 1.0]))


def test_trackc_support_set_task_present_mask_validation() -> None:
    cfg = Config(
        emb_dim=4,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=4,
    )
    model = build_model(cfg, torch.device("cpu")).eval()
    x_t = torch.randn(2, 4)
    x_0 = torch.randn(2, 4)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)
    task = torch.randn(2, 4)

    with pytest.raises(ValueError, match="support_set_task_present"):
        model(x_t, t, x_0, support_set_task=task, support_set_task_present=torch.ones(2, 2))
    with pytest.raises(ValueError, match="0/1"):
        model(x_t, t, x_0, support_set_task=task, support_set_task_present=torch.tensor([0.5, 1.0]))


def test_support_context_adapter_finetune_scope_only_trains_context_bridge() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_context_use_in_model=True,
        trackc_support_context_dim=8,
        finetune_trainable_scope="support_context_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("support_context_to_c.") for name in trainable)


def test_support_set_task_adapter_finetune_scope_only_trains_task_bridge() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=8,
        finetune_trainable_scope="support_set_task_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("support_set_task_to_c.") for name in trainable)


def test_support_residual_adapter_finetune_scope_only_trains_residual_operator() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_residual_use_in_model=True,
        trackc_support_context_dim=8,
        finetune_trainable_scope="support_residual_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("support_context_to_v.") for name in trainable)


def test_support_film_adapter_finetune_scope_only_trains_film_operator() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_film_use_in_model=True,
        trackc_support_context_dim=8,
        finetune_trainable_scope="support_film_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert any(name.startswith("support_context_to_v.") for name in trainable)
    assert any(name.startswith("support_context_to_v_scale.") for name in trainable)
    assert all(
        name.startswith("support_context_to_v.") or name.startswith("support_context_to_v_scale.")
        for name in trainable
    )


def test_support_context_adapter_scope_requires_enabled_context_bridge() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        finetune_trainable_scope="support_context_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="support_context"):
        apply_finetune_freeze(model, cfg)


def test_support_set_task_adapter_scope_requires_enabled_task_bridge() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        finetune_trainable_scope="support_set_task_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="support_set_task"):
        apply_finetune_freeze(model, cfg)


def test_support_residual_adapter_scope_requires_enabled_residual_operator() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        finetune_trainable_scope="support_residual_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="support_residual"):
        apply_finetune_freeze(model, cfg)


def test_support_film_adapter_scope_requires_enabled_film_operator() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        finetune_trainable_scope="support_film_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="support_film"):
        apply_finetune_freeze(model, cfg)


def test_support_residual_config_uses_support_context_validation() -> None:
    cfg = Config(
        emb_dim=8,
        mlp_d_model=4,
        mlp_n_layers=1,
        dropout=0.0,
        model_type="control_mlp",
        trackc_support_residual_use_in_model=True,
        trackc_support_context_dim=8,
        trackc_support_context_source="off",
    )
    with pytest.raises(ValueError, match="routed_distill_target"):
        validate_support_context_config(cfg)

    cfg.trackc_support_context_source = "routed_distill_target"
    with pytest.raises(ValueError, match="bank_split_file"):
        validate_support_context_config(cfg)

    cfg.trackc_support_residual_use_in_model = False
    cfg.trackc_support_film_use_in_model = True
    cfg.trackc_support_context_source = "off"
    with pytest.raises(ValueError, match="routed_distill_target"):
        validate_support_context_config(cfg)


def test_pairwise_adapter_finetune_scope_only_trains_pair_to_out() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pairwise_mode="hadamard_mean",
        finetune_trainable_scope="pairwise_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("pert_encoder.pair_to_out.") for name in trainable)


def test_pairwise_adapter_finetune_scope_requires_pairwise_branch() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        pert_pairwise_mode="off",
        finetune_trainable_scope="pairwise_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="pair_to_out"):
        apply_finetune_freeze(model, cfg)


def test_pairwise_condition_adapter_finetune_scope_trains_projection_bridge() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=8,
        pert_pool_aggregations=("mean", "max", "min"),
        pert_pool_scale_init=(1.0, 1.0, 1.0),
        pert_pairwise_mode="hadamard_mean",
        condition_delta_head_use_in_model=True,
        finetune_trainable_scope="pairwise_condition_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert any(name.startswith("pert_encoder.pair_to_out.") for name in trainable)
    assert any(name.startswith("pert_to_c.") for name in trainable)
    assert any(name.startswith("condition_delta_to_c.") for name in trainable)
    assert not any(name.startswith("condition_delta_head.") for name in trainable)
    assert all(
        name.startswith(("pert_encoder.pair_to_out.", "pert_to_c.", "condition_delta_to_c."))
        for name in trainable
    )


def test_pairwise_condition_adapter_finetune_scope_requires_pairwise_branch() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=8,
        pert_pairwise_mode="off",
        condition_delta_head_use_in_model=True,
        finetune_trainable_scope="pairwise_condition_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="pair_to_out"):
        apply_finetune_freeze(model, cfg)


def test_condition_prior_adapter_finetune_scope_only_trains_prior_head_and_bridge() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        condition_delta_head_use_in_model=True,
        condition_prior_delta_loss_weight=0.05,
        finetune_trainable_scope="condition_prior_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    apply_finetune_freeze(model, cfg)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert any(name.startswith("condition_delta_head.") for name in trainable)
    assert any(name.startswith("condition_delta_to_c.") for name in trainable)
    assert all(
        name.startswith(("condition_delta_head.", "condition_delta_to_c."))
        for name in trainable
    )


def test_condition_prior_adapter_finetune_scope_requires_model_bridge() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        condition_delta_head_loss_weight=1.0,
        condition_delta_head_use_in_model=False,
        finetune_trainable_scope="condition_prior_adapter",
    )
    model = build_model(cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="condition_delta_to_c"):
        apply_finetune_freeze(model, cfg)


def test_condition_prior_gene_shrink_aggregation_builds_dataset_specific_targets() -> None:
    from model.latent.train import _aggregate_condition_prior_records

    records = {
        "A": [
            ("G1", "gene", torch.tensor([1.0, 0.0])),
            ("G1", "gene", torch.tensor([3.0, 0.0])),
            ("G2", "gene", torch.tensor([0.0, 4.0])),
        ],
        "B": [
            ("G1", "gene", torch.tensor([5.0, 0.0])),
            ("G2", "gene", torch.tensor([0.0, 8.0])),
        ],
    }
    out = _aggregate_condition_prior_records(records, aggregation="gene_shrink_k2")
    assert set(out) == {"A", "B"}
    a = {gene: delta for gene, _ptype, delta in out["A"]}
    b = {gene: delta for gene, _ptype, delta in out["B"]}

    # G1 has 3 train observations globally, so alpha = 3 / (3 + 2).
    g1_global = torch.tensor([3.0, 0.0])
    ds_a = torch.tensor([4.0 / 3.0, 4.0 / 3.0])
    expected_a_g1 = 0.6 * g1_global + 0.4 * ds_a
    assert torch.allclose(a["G1"], expected_a_g1.float())

    # Dataset-specific shrink targets differ because the dataset mean differs.
    assert not torch.allclose(a["G1"], b["G1"])
    assert set(a) == {"G1", "G2"}


def test_condition_prior_gene_shrink_jiang_lowcount_mask_uses_dataset_mean() -> None:
    from model.latent.train import _aggregate_condition_prior_records

    records = {
        "Jiang_IFNG": [
            ("LOW", "gene", torch.tensor([10.0, 0.0])),
            ("HIGH", "gene", torch.tensor([0.0, 2.0])),
        ],
        "Other": [
            ("HIGH", "gene", torch.tensor([0.0, 6.0])),
            ("HIGH", "gene", torch.tensor([0.0, 10.0])),
        ],
    }
    out = _aggregate_condition_prior_records(
        records,
        aggregation="gene_shrink_k2_jiang_lowcount_mask",
    )
    jiang = {gene: delta for gene, _ptype, delta in out["Jiang_IFNG"]}
    other = {gene: delta for gene, _ptype, delta in out["Other"]}

    jiang_dataset_mean = torch.tensor([5.0, 1.0])
    assert torch.allclose(jiang["LOW"], jiang_dataset_mean)

    high_global = torch.tensor([0.0, 6.0])
    alpha = 3.0 / 5.0
    expected_high = alpha * high_global + (1.0 - alpha) * jiang_dataset_mean
    assert torch.allclose(jiang["HIGH"], expected_high.float())
    assert not torch.allclose(other["HIGH"], torch.tensor([0.0, 8.0]))


def test_condition_prior_gene_shrink_dataset_negative_mask_uses_dataset_mean_for_fallback_datasets() -> None:
    from model.latent.train import _aggregate_condition_prior_records

    records = {
        "NormanWeissman2019_filtered": [
            ("G1", "gene", torch.tensor([10.0, 0.0])),
            ("G2", "gene", torch.tensor([0.0, 2.0])),
        ],
        "Other": [
            ("G1", "gene", torch.tensor([2.0, 0.0])),
            ("G2", "gene", torch.tensor([0.0, 6.0])),
        ],
    }
    out = _aggregate_condition_prior_records(
        records,
        aggregation="gene_shrink_k2_dataset_negative_mask",
    )
    norman = {gene: delta for gene, _ptype, delta in out["NormanWeissman2019_filtered"]}
    other = {gene: delta for gene, _ptype, delta in out["Other"]}

    norman_dataset_mean = torch.tensor([5.0, 1.0])
    assert torch.allclose(norman["G1"], norman_dataset_mean)
    assert torch.allclose(norman["G2"], norman_dataset_mean)

    other_dataset_mean = torch.tensor([1.0, 3.0])
    g1_global = torch.tensor([6.0, 0.0])
    expected_other_g1 = 0.5 * g1_global + 0.5 * other_dataset_mean
    assert torch.allclose(other["G1"], expected_other_g1.float())

    out_k4 = _aggregate_condition_prior_records(
        records,
        aggregation="gene_shrink_k4_dataset_negative_mask",
    )
    norman_k4 = {gene: delta for gene, _ptype, delta in out_k4["NormanWeissman2019_filtered"]}
    other_k4 = {gene: delta for gene, _ptype, delta in out_k4["Other"]}
    assert torch.allclose(norman_k4["G1"], norman_dataset_mean)
    assert torch.allclose(norman_k4["G2"], norman_dataset_mean)

    alpha_k4 = 2.0 / 6.0
    expected_other_k4_g1 = alpha_k4 * g1_global + (1.0 - alpha_k4) * other_dataset_mean
    assert torch.allclose(other_k4["G1"], expected_other_k4_g1.float())


def test_condition_delta_in_model_gene_multi_gate() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        condition_delta_head_use_in_model=True,
        condition_delta_in_model_filter="gene_multi",
    )
    model = build_model(cfg, torch.device("cpu"))
    gid = torch.tensor([[2, 3, 0], [4, 0, 0], [5, 6, 0], [7, 8, 0]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.float32)
    tid = torch.tensor([PERT_TYPE_CRISPRI, PERT_TYPE_CRISPRI, PERT_TYPE_DRUG, PERT_TYPE_CRISPRI])
    nperts = torch.tensor([2, 1, 2, 2])
    chem_mask = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    gate = model._condition_delta_in_model_gate(
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=tid,
        nperts=nperts,
        chem_mask=chem_mask,
    )
    assert gate.tolist() == [True, False, False, False]


def test_condition_delta_in_model_gene_single_gate() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        condition_delta_head_use_in_model=True,
        condition_delta_in_model_filter="gene_single",
    )
    model = build_model(cfg, torch.device("cpu"))
    gid = torch.tensor([[2, 0, 0], [3, 4, 0], [5, 0, 0], [6, 0, 0]])
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    tid = torch.tensor([PERT_TYPE_CRISPRI, PERT_TYPE_CRISPRI, PERT_TYPE_DRUG, PERT_TYPE_CRISPRI])
    nperts = torch.tensor([1, 2, 1, 1])
    chem_mask = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    gate = model._condition_delta_in_model_gate(
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=tid,
        nperts=nperts,
        chem_mask=chem_mask,
    )
    assert gate.tolist() == [True, False, False, False]


def test_condition_delta_in_model_prior_covered_gene_multi_gate() -> None:
    cfg = Config(
        emb_dim=32,
        mlp_d_model=16,
        mlp_n_layers=1,
        dropout=0.0,
        use_pert_condition=True,
        pert_embed_mode="random_learned",
        pert_encoder_num_embeddings=64,
        pert_gene_emb_dim=8,
        pert_cond_dim=16,
        condition_delta_head_use_in_model=True,
        condition_delta_in_model_filter="prior_covered_gene_multi",
    )
    model = build_model(cfg, torch.device("cpu"))
    model.set_condition_delta_prior_gene_ids([2, 3, 5])
    gid = torch.tensor([[2, 3, 0], [2, 4, 0], [5, 0, 0], [2, 3, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.float32)
    tid = torch.full((4,), PERT_TYPE_CRISPRI, dtype=torch.long)
    nperts = torch.tensor([2, 2, 1, 3])
    gate = model._condition_delta_in_model_gate(
        pert_gene_ids=gid,
        pert_mask=mask,
        pert_type_id=tid,
        nperts=nperts,
        chem_mask=None,
    )
    assert gate.tolist() == [True, False, False, True]
