import json

import numpy as np
import pytest
import torch

from model.latent.config import Config
from model.latent.fm_ot import CondOTPath
from model.latent.response_normalizer import ResponseNormalizer, sha256_file
from model.latent.train import anchor_replay_filter_matches, response_geometry_filter_matches, train_step


class _ConstantVelocity(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value, dtype=torch.float32))

    def forward(self, x_t, t, src):
        del t, src
        return self.value.to(device=x_t.device, dtype=x_t.dtype).expand_as(x_t)


def _write_artifact(path, split_file, *, emb_dim=4):
    comps = np.eye(emb_dim, dtype=np.float32)[:2]
    metadata = {
        "artifact_type": "latentfm_response_normalizer",
        "fit_scope": "train_only",
        "emb_dim": emb_dim,
        "split_sha256": sha256_file(split_file),
        "global_median_norm": 2.0,
        "n_train_residuals": 3,
        "pca_components": 2,
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        dataset_scale_factors_json=np.asarray(json.dumps({"ds_a": 2.0, "ds_b": 0.5})),
        pca_mean=np.zeros(emb_dim, dtype=np.float32),
        pca_components=comps,
        pca_scales=np.asarray([2.0, 0.5], dtype=np.float32),
    )


def test_response_normalizer_roundtrip_dataset_scale_pca(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"ds_a": {"train": ["a"], "test": ["b"]}}')
    artifact = tmp_path / "rn.npz"
    _write_artifact(artifact, split)

    rn = ResponseNormalizer.from_npz(
        artifact,
        mode="dataset_scale_pca",
        strict_split_file=split,
        strict_emb_dim=4,
    )
    delta = torch.tensor([[4.0, 1.0, 3.0, -2.0], [1.0, -2.0, 0.5, 0.25]])
    transformed = rn.transform_delta("ds_a", delta)
    restored = rn.inverse_delta("ds_a", transformed)
    assert torch.allclose(restored, delta, atol=1e-5)


def test_response_normalizer_split_hash_mismatch_fails_closed(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"ds_a": {"train": ["a"], "test": ["b"]}}')
    artifact = tmp_path / "rn.npz"
    _write_artifact(artifact, split)
    split.write_text('{"ds_a": {"train": ["a", "leak"], "test": ["b"]}}')

    with pytest.raises(ValueError, match="split hash mismatch"):
        ResponseNormalizer.from_npz(
            artifact,
            mode="dataset_scale",
            strict_split_file=split,
            strict_emb_dim=4,
        )


def test_response_normalizer_missing_dataset_uses_identity_scale(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"ds_a": {"train": ["a"], "test": ["b"]}}')
    artifact = tmp_path / "rn.npz"
    _write_artifact(artifact, split)
    rn = ResponseNormalizer.from_npz(
        artifact,
        mode="dataset_scale",
        strict_split_file=split,
        strict_emb_dim=4,
    )
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(rn.transform_delta("unknown_ds", delta), delta)


def test_train_step_response_geometry_loss_is_default_off_and_active_when_enabled(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"ds_a": {"train": ["a"], "test": ["b"]}}')
    artifact = tmp_path / "rn.npz"
    _write_artifact(artifact, split)
    rn = ResponseNormalizer.from_npz(
        artifact,
        mode="dataset_scale",
        strict_split_file=split,
        strict_emb_dim=4,
    )
    cfg = Config(emb_dim=4, use_mmd=False)
    src = torch.zeros(3, 4)
    gt = torch.tensor([[2.0, 0.0, 0.0, 0.0]]).expand_as(src).clone()
    model = _ConstantVelocity([0.0, 0.0, 0.0, 0.0])
    path = CondOTPath()

    torch.manual_seed(7)
    out_off = train_step(src, gt, model, path, cfg, torch.device("cpu"))
    assert float(out_off["response_geometry"]) == 0.0

    torch.manual_seed(7)
    out_on = train_step(
        src,
        gt,
        model,
        path,
        cfg,
        torch.device("cpu"),
        ds_name="ds_a",
        response_geometry_weight_t=1.0,
        response_normalizer=rn,
    )
    assert float(out_on["response_geometry"]) > 0.0
    assert float(out_on["total"]) > float(out_off["total"])


def test_train_step_anchor_replay_loss_is_default_off_and_active():
    cfg = Config(emb_dim=4, use_mmd=False)
    src = torch.zeros(3, 4)
    gt = torch.ones(3, 4)
    model = _ConstantVelocity([0.0, 0.0, 0.0, 0.0])
    anchor = _ConstantVelocity([1.0, 0.0, 0.0, 0.0])
    path = CondOTPath()

    torch.manual_seed(11)
    out_off = train_step(src, gt, model, path, cfg, torch.device("cpu"))
    assert float(out_off["anchor_replay"]) == 0.0

    torch.manual_seed(11)
    out_on = train_step(
        src,
        gt,
        model,
        path,
        cfg,
        torch.device("cpu"),
        anchor_replay_weight_t=1.0,
        anchor_model=anchor,
    )
    assert float(out_on["anchor_replay"]) > 0.0
    assert float(out_on["total"]) > float(out_off["total"])


def _pert_tuple(*, nperts: int, tid: int = 1, has_gene: bool = True, has_chem: bool = False):
    gid = torch.tensor([[7, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 0] if has_gene else [0, 0]], dtype=torch.bool)
    type_id = torch.tensor([tid], dtype=torch.long)
    npt = torch.tensor([nperts], dtype=torch.long)
    combo_id = torch.tensor([0], dtype=torch.long)
    chem_emb = torch.zeros(1, 1, 3)
    chem_mask = torch.tensor([[1 if has_chem else 0]], dtype=torch.bool)
    return gid, mask, type_id, npt, combo_id, chem_emb, chem_mask


def test_response_geometry_filter_defaults_to_all():
    cfg = Config(response_geometry_condition_filter="all")
    assert response_geometry_filter_matches(cfg, None)
    assert response_geometry_filter_matches(cfg, _pert_tuple(nperts=1))


def test_response_geometry_filter_gene_multi_uses_only_metadata():
    cfg = Config(response_geometry_condition_filter="gene_multi")

    assert response_geometry_filter_matches(cfg, _pert_tuple(nperts=2, tid=1, has_gene=True))
    assert not response_geometry_filter_matches(cfg, _pert_tuple(nperts=1, tid=1, has_gene=True))
    assert not response_geometry_filter_matches(cfg, _pert_tuple(nperts=2, tid=5, has_gene=True))
    assert not response_geometry_filter_matches(cfg, _pert_tuple(nperts=2, tid=1, has_gene=False))
    assert not response_geometry_filter_matches(cfg, _pert_tuple(nperts=2, tid=1, has_gene=True, has_chem=True))


def test_response_geometry_filter_rejects_unknown_mode():
    cfg = Config(response_geometry_condition_filter="outcome_aware")
    with pytest.raises(ValueError, match="response_geometry_condition_filter"):
        response_geometry_filter_matches(cfg, _pert_tuple(nperts=2))


def test_anchor_replay_filter_non_gene_multi_uses_only_metadata():
    cfg = Config(anchor_replay_condition_filter="non_gene_multi")

    assert not anchor_replay_filter_matches(cfg, _pert_tuple(nperts=2, tid=1, has_gene=True))
    assert anchor_replay_filter_matches(cfg, _pert_tuple(nperts=1, tid=1, has_gene=True))
    assert anchor_replay_filter_matches(cfg, _pert_tuple(nperts=2, tid=5, has_gene=True))
    assert anchor_replay_filter_matches(cfg, _pert_tuple(nperts=2, tid=1, has_gene=False))
    assert anchor_replay_filter_matches(cfg, _pert_tuple(nperts=2, tid=1, has_gene=True, has_chem=True))


def test_anchor_replay_filter_rejects_unknown_mode():
    cfg = Config(anchor_replay_condition_filter="outcome_aware")
    with pytest.raises(ValueError, match="anchor_replay_condition_filter"):
        anchor_replay_filter_matches(cfg, _pert_tuple(nperts=1))
