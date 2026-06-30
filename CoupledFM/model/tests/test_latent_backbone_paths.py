import pytest

from model.latent.dataset import CrossDatasetFMDataset
from model.utils.data.biflow_paths import (
    iter_biflow_dataset_stems,
    normalize_latent_backbone,
    resolve_biflow_control_gt_h5ad,
)


def test_xverse_is_valid_latent_backbone(tmp_path):
    assert normalize_latent_backbone(" xVerse ") == "xverse"
    assert iter_biflow_dataset_stems(tmp_path, latent_backbone="xverse") == []
    assert resolve_biflow_control_gt_h5ad(tmp_path, "Adamson", latent_backbone="xverse") is None


def test_invalid_latent_backbone_still_fails():
    with pytest.raises(ValueError, match="latent_backbone"):
        normalize_latent_backbone("not_a_backbone")


def test_cross_dataset_close_tolerates_partial_initialization():
    ds = CrossDatasetFMDataset.__new__(CrossDatasetFMDataset)
    ds.close()
