"""
Configuration for CoupledFM: raw expression FM with optional OT + latent guidance.

Three coupling modes:
  - baseline: random pairing, no latent (same as FM/raw/FM_nn)
  - ot:       OT pairing in latent space, no CLS injection
  - coupled:  OT pairing + latent z_t injected into CLS token

Paths default under ``RAW_INDEPENDENT_ROOT`` (repo root of raw_independent);
override with env vars (see DataConfig / TrainConfig fields).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from model import paths

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(os.environ.get("RAW_INDEPENDENT_ROOT", paths.delivery_root()))


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 16
    d_ff: int = 1024
    dropout: float = 0.1
    attn_mode: str = "diff"       # "diff" | "self_only"
    # attention backend:
    #   "sdpa"   — F.scaled_dot_product_attention (supports attn_bias)
    #   "flash"  — flash_attn (fastest; falls back to sdpa if attn_bias given)
    #   "linear" — ELU+1 kernel linear attention (no bias)
    #   "sparse" — CellNavi-style scatter on edge_index (O(E·H))
    attn_backend: str = "sdpa"
    d_latent: int = 2058
    use_pert_token: bool = True
    num_pert_ids: int = 10000
    use_graph: bool = True
    graph_bias_mode: str = field(default_factory=lambda: os.environ.get(
        "GRAPH_BIAS_MODE", os.environ.get("RAW_GRAPH_BIAS_MODE", "sdpa_bias"),
    ))  # none | sdpa_bias | sparse
    use_lora: bool = False
    lora_rank: int = 16
    lora_target: Tuple[str, ...] = (".attn.fc_", ".ffn.linear")
    use_latent_resampler: bool = field(default_factory=lambda: os.environ.get(
        "RAW_USE_LATENT_RESAMPLER", "0") == "1")
    latent_resampler_n_tokens: int = field(default_factory=lambda: int(
        os.environ.get("RAW_LATENT_RESAMPLER_N_TOKENS", "8")))
    latent_resampler_n_head: int = field(default_factory=lambda: int(
        os.environ.get("RAW_LATENT_RESAMPLER_N_HEAD", "4")))
    cross_attn_independent_kv: bool = field(default_factory=lambda: os.environ.get(
        "RAW_CROSS_ATTN_INDEPENDENT_KV", "0") == "1")
    value_encoder: str = field(default_factory=lambda: os.environ.get(
        "RAW_VALUE_ENCODER", "linear"))  # linear | fourier
    fourier_n_freqs: int = field(default_factory=lambda: int(
        os.environ.get("RAW_FOURIER_N_FREQS", "32")))

    # Perturbation conditioning for raw velocity field (safe OFF by default).
    use_pert_condition: bool = False
    pert_embed_mode: str = "pretrained_frozen"
    pert_encoder_num_embeddings: int = 8192
    pert_gene_emb_dim: int = 256
    pert_cond_dim: int = 512
    pert_type_emb_dim: int = 32
    pert_encoder_dropout: float = 0.0
    max_combo_id_exclusive: int = 4096
    # Max gene slots in PerturbationBatch / encoder (aligns with latent Config.max_pert_genes).
    max_pert_genes: int = 16
    # If > 0 and ``forward(..., cond_vec=...)`` is set, adds zero-init projection into c_vec.
    # Default 0: legacy ``cond_vec`` is ignored after a warning.
    legacy_cond_vec_dim: int = 0
    pert_chem_emb_dim: int = 512
    pert_chem_projector_hidden: int = 0
    pert_gene_projector_hidden: int = 0
    pert_type_scale_init: Tuple[float, ...] = (0.0, -1.0, -1.0, -1.0, 1.0, 1.0)
    pert_pool_aggregations: Tuple[str, ...] = ("mean",)
    pert_pool_scale_init: Tuple[float, ...] = (1.0,)
    pert_pool_fusion_mode: str = field(default_factory=lambda: os.environ.get(
        "PERT_POOL_FUSION_MODE", "sum",
    ).strip().lower() or "sum")
    pert_type_adapter_mode: str = field(default_factory=lambda: os.environ.get(
        "PERT_TYPE_ADAPTER_MODE", "scalar",
    ).strip().lower() or "scalar")
    pert_condition_embedding_source: str = field(default_factory=lambda: os.environ.get(
        "PERT_EMBED_SOURCE", "",
    ).strip())


@dataclass
class DataConfig:
    biflow_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_BIFLOW_DIR",
        str(paths.biflow_dir())))
    # "state" | "uce" | "stack" | "scldm" | "scfoundation"; see ``utils.data.biflow_paths``.
    latent_backbone: str = field(default_factory=lambda: os.environ.get(
        "RAW_LATENT_BACKBONE", "stack"))
    latent_data_dir: str = field(default_factory=lambda: os.environ.get("RAW_LATENT_DATA_DIR", ""))
    raw_data_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_DATA_DIR", str(_REPO_ROOT / "data/raw/DE5000")))
    # Legacy paths: ``fm_data/pert_means.npz`` / ``ctrl_means.npz`` store **latent**
    # embeddings (d_latent), not gene expression.  CoupledFM ``train.py`` does **not**
    # load these for ``corr_pert_mean``; it builds gene-space means from biFlow
    # ``_DatasetHandle`` (``ctrl_mean_gene`` / ``compute_gt_mean_gene``).
    # Deprecated: not loaded by current CoupledFM train.py (gene-space means are
    # rebuilt from biFlow handles). Default empty; opt-in via env var only.
    pert_means_path: str = field(default_factory=lambda: os.environ.get("RAW_PERT_MEANS", ""))
    ctrl_means_path: str = field(default_factory=lambda: os.environ.get("RAW_CTRL_MEANS", ""))

    gene_name_path: str = field(default_factory=lambda: os.environ.get(
        "RAW_GENE_NAME", str(paths.gene_name_path())))
    nichenet_node2idx_path: str = field(default_factory=lambda: os.environ.get(
        "RAW_NICHENET_NODE2IDX",
        str(paths.nichenet_node2idx_path())))
    nichenet_graph_pkl_path: str = field(default_factory=lambda: os.environ.get(
        "RAW_NICHENET_GRAPH_PKL",
        str(paths.nichenet_graph_pkl_path())))

    split_seed: int = 42
    # Optional explicit split JSON. When set, train.py loads this file read-only
    # instead of building or reusing the canonical biflow split.
    split_file: str = field(default_factory=lambda: os.environ.get("RAW_SPLIT_FILE", ""))
    # True: train.py will build/save canonical split under biflow_dir when missing.
    # (Legacy latent_data_dir JSON split is removed; keep True.)
    explicit_pert_split: bool = True
    min_cells_per_cond: int = 16  # control/GT cells per condition
    # subset of datasets to load; empty list = all datasets
    datasets: List[str] = field(default_factory=list)

    # Perturbation batch (when ``model.use_pert_condition``): GeneEmbeddingCache root.
    pert_gene_emb_cache_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_PERT_GENE_EMB_CACHE_DIR",
        str(paths.scgpt_cache_dir()),
    ))
    # When True (and gt h5ad has obs columns), build per-condition metadata once from obs.
    # This is the safe default for mixed gene / chemical / multi-perturbation
    # datasets; set RAW_USE_H5AD_PERT_METADATA=0 only for legacy string-only runs.
    use_h5ad_pert_metadata: bool = field(default_factory=lambda: os.environ.get(
        "RAW_USE_H5AD_PERT_METADATA", "1") == "1")
    pert_metainfo_path: str = field(default_factory=lambda: os.environ.get(
        "RAW_PERT_METAINFO_PATH",
        str(_REPO_ROOT / "data/raw/genepert_DE5000/metainfo.json"),
    ))
    # chemicalpert / sciplex3 metainfo（list[dict]）；供后续化学条件接入。RAW_CHEMICAL_METAINFO_PATH 覆盖；置空关闭。
    chemical_metainfo_path: str = field(default_factory=lambda: os.environ.get(
        "RAW_CHEMICAL_METAINFO_PATH",
        str(_REPO_ROOT / "data/raw/chemicalpert_DE5000/metainfo.json"),
    ))
    chem_emb_source_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_CHEM_EMB_SOURCE_DIR", ""))
    drug_emb_cache_dir: str = field(default_factory=lambda: (
        os.environ.get("RAW_DRUG_EMB_CACHE_DIR", "").strip()
    ))
    max_chem_keys: int = field(default_factory=lambda: int(os.environ.get(
        "RAW_MAX_CHEM_KEYS", "4")))
    chem_fallback_embed_dim: int = field(default_factory=lambda: int(os.environ.get(
        "RAW_CHEM_FALLBACK_EMB_DIM", "512")))
    chem_obs_column: str = field(default_factory=lambda: os.environ.get(
        "RAW_CHEM_OBS_COLUMN",
        os.environ.get("RAW_CHEM_EMB_OBS_COLUMN", ""),
    ))
    # Gate chemical embedding slot in PerturbationBatch / dataset (aligns with coupled).
    pert_chem_enabled: bool = field(default_factory=lambda: os.environ.get(
        "RAW_PERT_CHEM_ENABLED", "0") == "1")
    # When True alongside ``model.use_pert_condition=true``, dataloader yields perturbation slot
    # (see CoupledFMDataset docstring). Mutually gated in ``train.py`` (use_raw_cond alone is invalid).
    use_raw_cond: bool = field(default_factory=lambda: os.environ.get(
        "RAW_USE_RAW_COND", "0") == "1")


@dataclass
class TrainConfig:
    coupling_mode: str = "ot"  # "baseline" | "ot" | "coupled"
    # OT 对齐用的特征空间：latent（需 emb）| de（需 de_dir JSON）| raw（全 in_vocab）
    ot_feature: str = field(default_factory=lambda: os.environ.get(
        "RAW_OT_FEATURE", "de"))
    de_k: int = field(default_factory=lambda: int(os.environ.get("RAW_DE_K", "1024")))
    de_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_DE_DIR",
        str(paths.de_dir())))

    pretrained_ckpt: str = field(default_factory=lambda: os.environ.get(
        "RAW_PRETRAINED_CKPT",
        str(paths.cellnavi_pretrain_ckpt_path())))
    latent_fm_ckpt: str = ""

    # Per-rank OT pairing batch（未设 global_ot_batch 时：每张卡每步各跑这么多条 OT）
    batch_size: int = 64
    # DDP：若设置（如 60），则每步**全局** OT 条数 = global_ot_batch，自动 batch_size = global_ot_batch // world_size
    global_ot_batch: Optional[int] = None
    # GPU 上前向/反传的 micro-chunk；gwps 级长序列 (N≈7000) 时 flash/linear 在 _safe_train_mb 中 eff_mb 可能再被压低
    micro_batch: int = 32
    # 每多少个「数据 batch」再 optimizer.step 一次（梯度平均）；1=每 batch 更新（旧行为）
    grad_accum_steps: int = 1
    epochs: int = 200

    # ── learning rate ──────────────────────────────────────────────
    # Fine-tuning CellNavi pretrained weights: use a smaller peak LR than
    # training from scratch.  Pretrained Q/K/V/FFN are already well-
    # initialised; a large LR would destroy these features.
    # New modules (adaLN, GeneadaLN, latent projectors, output proj) are
    # zero-initialised so they tolerate higher LR but benefit from the
    # same warm-up.
    lr: float = 5e-5           # peak LR (reduced from 1e-4 for fine-tuning)
    weight_decay: float = 1e-2
    warmup_epochs: int = 2     # shorter warmup: pretrained params need less
    warmup_steps: int = 1000   # >0：用 step 粒度 warmup 覆盖 warmup_epochs（更稳）
    # Deprecated: LR floor is **only** ``lr * min_lr_ratio`` (HF-style). This field
    # is no longer applied in the optimizer schedule (see train.py).
    min_lr: float = 1e-6
    min_lr_ratio: float = 0.1  # cosine 不衰到 0，保底 lr*min_lr_ratio（HF 风格）
    grad_clip: float = 1.0

    # ── 分组 optimizer：backbone vs new modules ────────────────────
    # 默认关闭以保向后兼容；开启后 backbone（CellNavi 预训 attn/ffn/embed_gene）
    # 用更小 lr/WD，新模块（t_embed/adaln/out_proj/latent_proj 等零初始化层）用更大 lr。
    use_param_groups: bool = False
    lr_new_module_mult: float = 3.0     # new_lr = lr * mult
    weight_decay_backbone: float = 1e-3 # backbone WD（小，避免把预训权重拉偏）
    weight_decay_new: float = 1e-2      # new modules WD
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95            # FM/diffusion 常用 0.95 > 0.999
    # AdamW vs Adam：分组必须 AdamW（需独立 WD）；非分组时遵循旧行为（AdamW）
    use_adamw: bool = True

    # ── 两阶段微调：先冻 backbone，后解冻低 lr ─────────────────────
    # stage1 (epoch < two_stage_freeze_epochs): 冻结 backbone_keys 对应参数，仅训新模块
    # stage2 (epoch ≥ two_stage_freeze_epochs): 全解冻，backbone lr=lr*stage2_backbone_mult
    two_stage_ft: bool = False
    two_stage_freeze_epochs: int = 3
    stage2_backbone_mult: float = 0.1   # stage2 backbone lr = base lr * mult（典型 0.1 → 5e-6）
    stage2_backbone_warmup_steps: int = 500  # stage2 切换后 backbone lr 线性 warmup 步数

    # ── EMA shadow weights ─────────────────────────────────────────
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after: int = 0           # 0 表示直接更新；两阶段场景可设 warmup_steps+freeze end
    ema_update_every: int = 1
    ema_dynamic: bool = True            # True：decay = min(target, (1+step)/(10+step))

    ds_alpha: float = 0.7

    # latent z_t computation:
    #   "interp" = (1-t)*z_ctrl + t*z_gt   ← teacher forcing，训练快、但有 train-test mismatch
    #   "ode"    = 让 latent FM ODE 自演化到 t，与推理一致
    #   "curriculum" = 训练初期走 interp，后期线性过渡到 ode（推荐）
    latent_z_mode: str = "curriculum"
    latent_ode_steps: int = 20
    # curriculum schedule（仅 latent_z_mode="curriculum" 生效）
    curriculum_warmup_steps: int = 2000     # 这之前 p_ode=0（纯 interp）
    curriculum_anneal_steps: int = 20000    # warmup 结束后再跑这么多 step 线性 ramp 到 p_ode=max_prob
    curriculum_max_prob: float = 1.0        # 最终 ode 采样概率

    # OT settings (used when coupling_mode in ["ot", "coupled"])
    # 默认改为 GPU 上的 torch_sinkhorn：彻底消除原本 CPU bound 的 pot.emd；
    # "exact" / "sinkhorn" 仍走 CPU POT，仅用于对照或 debug。
    ot_method: str = field(default_factory=lambda: os.environ.get(
        "OT_METHOD", "torch_sinkhorn"))
    ot_threads: int = 4
    ot_sinkhorn_reg: float = field(default_factory=lambda: float(
        os.environ.get("OT_SINKHORN_REG", "0.05")))
    ot_sinkhorn_iter: int = field(default_factory=lambda: int(
        os.environ.get("OT_SINKHORN_ITER", "50")))
    # OT 候选池 cap；None = 使用 batch_size（minibatch OT）
    ot_emb_cap_src: Optional[int] = None
    ot_emb_cap_gt: Optional[int] = None
    ot_cost: str = field(default_factory=lambda: os.environ.get("RAW_OT_COST", "cosine"))
    ot_sample_mode: str = field(default_factory=lambda: os.environ.get(
        "RAW_OT_SAMPLE_MODE", "assignment"))  # assignment | multinomial

    # Flow time t 采样（dataset + latent FM 共用模块）
    time_sampling: str = field(default_factory=lambda: os.environ.get(
        "RAW_TIME_SAMPLING", "logit_normal"))

    # 主损失加权：none | min_snr
    loss_weighting: str = "min_snr"
    min_snr_gamma: float = 5.0

    # 数据增强（Phase 2）
    gene_mask_prob: float = 0.1
    gene_mask_all_prob: float = 0.0
    # Explicit deterministic raw-gene budget manifest. Entries are local keep
    # indices relative to each dataset handle's gene_ids_valid / in-vocab genes.
    gene_budget_manifest_path: str = field(default_factory=lambda: os.environ.get(
        "RAW_GENE_BUDGET_MANIFEST", ""))
    gene_budget_label: str = field(default_factory=lambda: os.environ.get(
        "RAW_GENE_BUDGET_LABEL", ""))
    xt_noise_sigma_max: float = 0.0
    pert_idx_mode: str = "zero"  # zero | random
    cfg_drop_prob: float = field(default_factory=lambda: float(
        os.environ.get("RAW_CFG_DROP_PROB", "0.1")))
    use_residual_flow: bool = field(default_factory=lambda: os.environ.get(
        "RAW_USE_RESIDUAL_FLOW", "0") == "1")

    # MMD regularization
    use_mmd: bool = True
    mmd_gamma_max: float = 0.005
    mmd_warmup_start: int = 10000
    mmd_warmup_end: int = 100000
    # 若两者均非 None：训练开始时按 total_opt_steps * frac 覆写上面的绝对 step
    mmd_warmup_start_frac: Optional[float] = 0.1
    mmd_warmup_end_frac: Optional[float] = 0.5
    mmd_every: int = 15
    mmd_epoch_start: int = 3
    # OOM-safe MMD: 当 batch × N × d 过大一次 forward 塞不下时，按 cell 维切 micro-chunk
    # 再 forward 算 x1_hat；0 表示关闭（默认用整 batch，行为与原来一致）。
    mmd_micro_chunk: int = 0

    # mixed precision (disable with CLI --no_amp or --fp64)
    use_amp: bool = True
    amp_dtype: str = "bfloat16"   # "float16" | "bfloat16"
    # full float64 on GPU (slow, huge memory; only for numerical debugging)
    fp64_training: bool = False
    # print v_pred / grad non-finite diagnostics (rank 0)
    debug_nan: bool = False
    # torch.autograd.set_detect_anomaly(True) — very slow
    detect_anomaly: bool = False

    seed: int = field(default_factory=lambda: int(os.environ.get("RAW_SEED", "42")))
    device: str = "cuda"

    # ── evaluation schedule ────────────────────────────────────────
    log_every: int = 1
    # 0 means use the full DataLoader epoch. This is mainly for fast smoke tests.
    max_train_steps_per_epoch: int = field(default_factory=lambda: int(os.environ.get("RAW_MAX_TRAIN_STEPS_PER_EPOCH", "0")))

    # Step-level: fast val monitoring on sampled conditions
    val_every_steps: int = 500
    # "auto": use explicit ``val`` split if present; otherwise preserve legacy
    # test-as-val only for canonical splits. Explicit split-file runs without a
    # val key must use fixed_steps_no_selection or disable training-time eval.
    val_split_key: str = field(default_factory=lambda: os.environ.get("RAW_VAL_SPLIT_KEY", "auto"))
    val_sample_ratio: float = 0.2   # per-dataset: ceil(n_test * ratio) then clamp
    val_min_per_ds: int = 5         # per-dataset val count lower target (tiny ds: use all)
    val_max_per_ds: int = 20        # per-dataset val count upper cap
    val_max_cells: int = 32         # max cells per condition in val (0=unlimited)
    val_ode_steps: int = 10         # ODE steps for fast val (was 5)

    # Epoch-level: full test evaluation every N epochs for early stopping
    selection_protocol: str = field(default_factory=lambda: os.environ.get(
        "RAW_SELECTION_PROTOCOL", "metric"))
    run_initial_val: bool = field(default_factory=lambda: os.environ.get(
        "RAW_RUN_INITIAL_VAL", "1") == "1")
    run_final_test: bool = field(default_factory=lambda: os.environ.get(
        "RAW_RUN_FINAL_TEST", "1") == "1")
    test_split_key: str = field(default_factory=lambda: os.environ.get("RAW_TEST_SPLIT_KEY", "test"))
    test_every_epoch: int = 1
    test_max_per_ds: int = 60       # cap per-dataset conditions in full test
    test_max_cells: int = 128       # cap cells per condition in test
    eval_ode_steps: int = 10        # epoch-end full TEST（与 val 对齐，Phase 5.1）
    # ODE integrator for val & test (see ``inference.integrate``; steps still set above)
    val_ode_method: str = "euler"  # euler | midpoint | rk4

    # Early stopping: stop after N consecutive full-test evaluations where the
    # selected test metric does not improve (patience × test_every_epoch epochs).
    early_stop_patience: int = 4    # e.g. 4 full tests with no improvement when test_every_epoch=1
    selection_metric: str = "corr_pert_mean"   # corr_pert_mean | corr_minus_mmd | pearson_delta_ctrl | mmd
    selection_mmd_lambda: float = 0.5
    loss_guard_epochs: int = 3                 # before this epoch, do not consume patience
    min_epochs_before_stop: int = 6            # hard floor for early stopping

    output_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_OUTPUT_DIR", str(_REPO_ROOT / "runs")))

    resume_from: Optional[str] = None


@dataclass
class InferenceConfig:
    ckpt_path: str = ""
    coupling_mode: str = "coupled"
    # Standalone inference supports the main non-residual flow target. Residual-flow
    # checkpoints need a condition-specific GT-control prior that is only available
    # inside train/eval, so predict_dataset fails fast if this is enabled.
    use_residual_flow: bool = False
    latent_fm_ckpt: str = ""
    method: str = "euler"          # "euler" | "midpoint" | "rk4"
    n_steps: int = 10
    cfg_w: float = field(default_factory=lambda: float(
        os.environ.get("RAW_INFERENCE_CFG_W", "1.0")))
    device: str = "cuda"
    output_dir: str = field(default_factory=lambda: os.environ.get(
        "RAW_INFERENCE_OUTPUT_DIR", str(_REPO_ROOT / "inference_out")))

    # Fallback when ckpt/config.json omits model fields (inference standalone).
    use_pert_condition: bool = False
    pert_embed_mode: str = "pretrained_frozen"
    pert_gene_emb_cache_dir: str = field(default_factory=lambda: str(
        paths.scgpt_cache_dir()))
    pert_encoder_num_embeddings: int = 8192
    pert_gene_emb_dim: int = 256
    pert_cond_dim: int = 512
    max_pert_genes: int = 16
    pert_type_emb_dim: int = 32
    pert_encoder_dropout: float = 0.0
    max_combo_id_exclusive: int = 4096
    legacy_cond_vec_dim: int = 0
    use_h5ad_pert_metadata: bool = True
    pert_chem_emb_dim: int = 512
    pert_chem_projector_hidden: int = 0
    pert_gene_projector_hidden: int = 0
    pert_type_scale_init: Tuple[float, ...] = (0.0, -1.0, -1.0, -1.0, 1.0, 1.0)
    pert_pool_aggregations: Tuple[str, ...] = ("mean",)
    pert_pool_scale_init: Tuple[float, ...] = (1.0,)
    pert_pool_fusion_mode: str = "sum"
    pert_type_adapter_mode: str = "scalar"
    pert_condition_embedding_source: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
