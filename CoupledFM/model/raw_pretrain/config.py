"""Dataclass config for ``model.raw_pretrain``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from model import paths


def _repo_root() -> Path:
    return paths.delivery_root()


@dataclass
class RawPretrainConfig:
    processed_dir: Path = field(
        default_factory=paths.cellgene_processed_dir,
    )
    tissue_metainfo_path: Path | None = None
    gene_symbol_column: str = "feature_name"
    min_gene_hit_rate: float = 0.80
    output_dir: Path = field(default_factory=lambda: _repo_root() / "output/cellgene_pretrain")

    gene_name_path: Path = field(
        default_factory=paths.gene_name_path,
    )
    nichenet_node2idx_path: Path = field(
        default_factory=paths.nichenet_node2idx_path,
    )
    pretrained_ckpt: Path = field(
        default_factory=paths.cellnavi_pretrain_ckpt_path,
    )

    # Matches cellgene_census/run_preprocess.py and manifest_*.json (bin 0 + 1..50).
    num_bins: int = 51
    # Processed Census centroid shards have tissue-specific top-gene axes.
    strict_same_genes: bool = False

    batch_size: int = 64
    micro_batch: int = 8
    epochs: int = 200
    steps_per_epoch: int = 2000
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    min_lr_ratio: float = 0.1

    seed: int = 42
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # conditioning sampling (dataset)
    cond_tau: float = 1.0
    cond_alpha: float = 1.0
    pseudo_delta_min: float = 0.0
    max_pert_genes: int = 24

    # losses (v2: MSE velocity + optional endpoint; no direction loss)
    loss_type: str = "mse"  # mse | smooth_l1
    smooth_l1_beta: float = 1.0
    loss_velocity_weight: float = 1.0
    loss_endpoint_weight: float = 0.25
    loss_weighting: str = "none"
    min_snr_gamma: float = 5.0
    time_sampling: str = "uniform"
    xt_noise_sigma_max: float = 0.0

    # direction adapter → legacy_cond_vec on velocity field
    adapter_cond_dim: int = 128
    adapter_n_heads: int = 4

    ckpt_every_steps: int = 5000
    log_every_steps: int = 50

    # EMA (optional; same helper as main train)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after: int = 0
    ema_dynamic: bool = True

    # velocity backbone (subset of ModelConfig defaults)
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 16
    d_ff: int = 1024
    dropout: float = 0.1
    attn_backend: str = "sdpa"

    def __post_init__(self) -> None:
        if self.tissue_metainfo_path is None:
            self.tissue_metainfo_path = self.processed_dir / "tissue_metainfo.csv"


__all__ = ["RawPretrainConfig"]
