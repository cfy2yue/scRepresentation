"""
Latent FM training configuration.

All hyperparameters are defined here as a single dataclass,
manageable via CLI with tyro.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from model import paths

_LATENT_PKG = Path(__file__).resolve().parent
_MODEL_PKG = _LATENT_PKG.parent
_REPO_ROOT = _MODEL_PKG.parent


@dataclass
class Config:
    # ── data ──────────────────────────────────────────────────────
    data_dir: str = str(_LATENT_PKG / "fm_data")
    # canonical train/test split 所在 biFlow 根目录（latent/raw/coupled 共用单一真相）
    biflow_dir: str = str(paths.biflow_dir())
    # AnnData layout: ``control_{latent_backbone}/`` + ``gt_{latent_backbone}/``.
    # Canonical Stack/scLDM/scFoundation roots live under ``data/latent_data/{backbone}``;
    # legacy ``control_center/`` + ``gt/`` is used only when preferred paths are missing.
    latent_backbone: str = "stack"  # "state" | "uce" | "stack" | "scldm" | "scfoundation" | "xverse"
    manifest: str = "manifest.json"
    test_ratio: float = 0.1
    split_seed: int = 42
    split_file: str = ""  # if set, load this JSON directly instead of auto-generating
    # Optional override for dataset-level perturbation means used by train-time
    # pearson_pert selection and pert_residual losses. Formal final evaluation
    # can still use the canonical all-condition artifact, but any training-time
    # selection/loss must point this to a train-only artifact.
    pert_means_file: str = ""

    # ── model ─────────────────────────────────────────────────────
    model_type: str = "control_mlp"  # "control_mlp" | "mlp"
    emb_dim: int = 2058

    # MLP / ControlMLP velocity field
    mlp_d_model: int = 512
    mlp_n_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # ── training ──────────────────────────────────────────────────
    lr: float = 1e-4
    # Pretrained gene embedding table (pert_encoder.gene_table) vs FM/MLP body — same idea as coupled raw.
    use_param_groups: bool = False
    lr_new_module_mult: float = 3.0
    weight_decay_backbone: float = 0.001
    weight_decay_new: float = 0.01
    weight_decay: float = 0.0  # single-group AdamW when use_param_groups=False
    eta_min: float = 1e-7
    warmup_steps: int = 1000
    total_steps: int = 10_000_000
    lr_decay_steps: int = 150_000  # cosine cycle length (~15 epochs); LR clamps at eta_min after this
    batch_size: int = 256   # OT microbatch / pair size; smaller conditions are bootstrap-resampled
    grad_accum_steps: int = 1  # optimizer effective batch = batch_size * grad_accum_steps
    min_cells: int = 32    # skip conditions with source pool or GT < this (training); test uses 16
    scale_noise: float = 0.02  # multiplicative Gaussian noise: x *= (1 + scale_noise * N(0,1))
    ds_alpha: float = 0.7  # temperature for dataset balancing: each ds contributes ceil(N^alpha) conds/epoch
    min_selected_conditions_per_dataset: int = 0  # floor for selected train conditions per dataset; 0=off
    condition_visit_power: float = 1.0  # power on ceil(n_gt/batch_size) visits; 1.0=legacy
    condition_visit_cap: int = 0  # cap per-condition visits per epoch; 0=off
    perturbation_family_filter: str = "all"  # all | gene | drug; diagnostic filter, default keeps legacy behavior
    grad_clip: float = 1.0
    seed: int = 42

    # ── flow matching ─────────────────────────────────────────────
    use_mmd: bool = True
    gamma: float = 0.03             # final MMD weight
    gamma_warmup_start: int = 50000 # start MMD warmup after this step
    gamma_warmup_end: int = 100000  # reach gamma_max at this step
    mmd_ode_steps: int = 0          # 0=single-step (legacy), >0=differentiable N-step ODE for MMD
    # unbiased: legacy (can be negative → clamp in loss); biased: includes diagonal, ≥0 for RBF (usually).
    mmd_estimator: str = "unbiased"  # unbiased | biased
    mmd_every: int = 1              # apply MMD loss every N training steps (1=every step)
    # Optional comma/semicolon-separated dataset allow-list for MMD loss. Empty
    # preserves legacy behavior and applies MMD to all train batches.
    mmd_dataset_filter: str = ""
    # Optional risk-row tail-state MMD branch. This is default-off and distinct
    # from scalar MMD gamma: an online train-only history records condition-level
    # MMD values and adds extra MMD pressure only for previously high-tail risk
    # rows in the configured risk datasets.
    risk_row_cvar_loss_weight: float = 0.0
    risk_row_cvar_loss_warmup_start: int = 0
    risk_row_cvar_loss_warmup_end: int = 0
    risk_row_cvar_dataset_filter: str = ""
    risk_row_cvar_history_size: int = 256
    risk_row_cvar_min_history: int = 8
    risk_row_cvar_top_frac: float = 0.20
    risk_row_cvar_mmd_threshold: float = 0.005
    # 与 raw/coupled 共用 ``utils/train/time_sampling.py`` 的语义
    time_sampling: str = field(
        default_factory=lambda: os.environ.get("LATENT_TIME_SAMPLING", "logit_normal"),
    )  # uniform | logit_normal | lognormal

    # ── loss rebalancing ──────────────────────────────────────────
    ds_loss_alpha: float = 0.0      # per-dataset weight: 0=off, 0.5=sqrt inverse freq, 1.0=full
    ds_loss_warmup_start: int = 0   # step at which dataset loss weights start; independent of MMD warmup
    # Optional per-condition loss weights loaded from a CSV/TSV with
    # ``dataset``, ``condition``, and a weight column. Empty preserves legacy
    # behavior. When normalization is enabled, matched train-condition weights
    # are rescaled to mean 1 so LR semantics stay comparable.
    condition_loss_weight_file: str = ""
    condition_loss_weight_column: str = "weight"
    condition_loss_weight_normalize_mean: bool = True

    # ── condition-level direction regularization ──────────────────
    # Optional endpoint direction loss for perturbation learning:
    # align mean(predicted endpoint - source) with mean(GT - source) within
    # the current condition batch. Default 0 keeps legacy FM semantics.
    direction_loss_weight: float = 0.0
    direction_loss_warmup_start: int = 0
    direction_loss_warmup_end: int = 0
    # Optional endpoint mean-delta MSE. This directly matches condition-level
    # perturbation displacement magnitude and shape, complementing cosine-only
    # direction loss.
    endpoint_delta_loss_weight: float = 0.0
    endpoint_delta_loss_warmup_start: int = 0
    endpoint_delta_loss_warmup_end: int = 0
    # Optional train-only response-geometry auxiliary loss. The artifact must
    # be fitted from canonical train conditions only via
    # ``python -m model.latent.fit_response_normalizer``. This does not alter
    # raw FM velocity semantics or raw-space evaluation metrics; it only adds a
    # batch endpoint-delta loss in dataset-scale/PCA response coordinates.
    response_geometry_loss_weight: float = 0.0
    response_geometry_loss_warmup_start: int = 0
    response_geometry_loss_warmup_end: int = 0
    response_normalization_mode: str = "off"  # off | dataset_scale | pca_subspace | dataset_scale_pca
    response_normalization_artifact: str = ""
    response_normalization_strict_split: bool = True
    # Optional deployable metadata filter for response-geometry auxiliary loss.
    # ``gene_multi`` uses only perturbation metadata available at train/inference
    # time: at least two gene slots, non-drug type, and no held-out response data.
    response_geometry_condition_filter: str = "all"  # all | gene_multi
    # Optional dataset-centered perturbation residual direction loss. This
    # mirrors pearson_pert evaluation by aligning mean(pred - dataset_pert_mean)
    # with mean(GT - dataset_pert_mean), rather than source-centered deltas.
    pert_residual_direction_loss_weight: float = 0.0
    pert_residual_direction_loss_warmup_start: int = 0
    pert_residual_direction_loss_warmup_end: int = 0
    # Optional condition-contrastive residual loss. The current predicted
    # perturbation residual is pulled toward its matched GT residual and pushed
    # away from a small queue of previous condition residuals. This targets
    # condition-specific discriminability rather than only endpoint fit.
    pert_residual_contrastive_loss_weight: float = 0.0
    pert_residual_contrastive_loss_warmup_start: int = 0
    pert_residual_contrastive_loss_warmup_end: int = 0
    pert_residual_contrastive_temperature: float = 0.10
    pert_residual_contrastive_bank_size: int = 256
    pert_residual_contrastive_min_norm: float = 1e-6
    # Optional soft relational residual loss. Unlike the hard InfoNCE-style
    # contrastive objective above, this matches the target residual's similarity
    # distribution over a residual queue, making it less brittle when biological
    # perturbations have graded similarity rather than one-positive-many-negative
    # labels.
    pert_residual_relational_loss_weight: float = 0.0
    pert_residual_relational_loss_warmup_start: int = 0
    pert_residual_relational_loss_warmup_end: int = 0
    pert_residual_relational_temperature: float = 0.10
    pert_residual_relational_target_temperature: float = 0.10
    # Optional synthetic composition regularizer for zero-shot multi-perturbation
    # generalization. It uses two same-dataset single-gene residual means to
    # supervise the model under the combined gene condition at t=0. Default 0
    # keeps the original FM objective unchanged.
    composition_delta_loss_weight: float = 0.0
    composition_delta_loss_warmup_start: int = 0
    composition_delta_loss_warmup_end: int = 0
    composition_delta_loss_every: int = 1
    composition_delta_bank_size: int = 512
    composition_delta_min_norm: float = 1e-6
    # Optional auxiliary condition->delta head. This directly aligns the
    # perturbation-condition representation with observed latent response
    # deltas. Default 0 keeps checkpoint architecture unchanged.
    condition_delta_head_loss_weight: float = 0.0
    condition_delta_head_loss_warmup_start: int = 0
    condition_delta_head_loss_warmup_end: int = 0
    condition_delta_head_hidden: int = 1024
    # Target frame for the auxiliary condition->delta head:
    # - endpoint_delta: legacy/default mean(GT - source)
    # - pert_residual: metric-aligned mean(GT) - dataset_pert_mean, matching
    #   the residual frame used by pearson_pert evaluation.
    condition_delta_head_target: str = "endpoint_delta"
    # Optional second-stage use of the learned condition->delta prediction.
    # When True, the predicted latent delta is projected back to d_model and
    # added to the velocity field conditioning vector. Default False preserves
    # old architecture/evaluation behavior.
    condition_delta_head_use_in_model: bool = False
    # Optional low-rank condition-only output residual. This is default-off and
    # independent of the condition_delta_head path: it projects the audited
    # perturbation conditioning vector through a zero-initialized low-rank
    # velocity residual. It is intended for strict no-op/provenance-gated
    # adapter experiments.
    condition_lowrank_residual_use_in_model: bool = False
    condition_lowrank_residual_rank: int = 32
    # Deploy-time gate for condition_delta_head_use_in_model:
    # - all: legacy behavior;
    # - gene_multi: only non-drug/non-chem multi-gene perturbations;
    # - prior_covered_gene_multi: same as gene_multi, plus every active gene id
    #   must be present in the train-single condition-prior bank allowlist.
    # - allowlisted_gene_single: same as gene_single, plus the active gene id
    #   must be present in condition_delta_allowlist_gene_file.
    condition_delta_in_model_filter: str = "all"  # all | gene_single | gene_multi | prior_covered_gene_multi | allowlisted_gene_single
    condition_delta_allowlist_gene_file: str = ""  # optional CSV/TSV/text gene allowlist for gated delta injection
    # Optional additive composition supervision for the same condition-delta
    # head. For synthetic two-gene conditions, predict each single-gene atom
    # through the head and sum the atoms, then match the summed response delta.
    # Default 0 keeps existing objective/checkpoint behavior unchanged.
    additive_condition_delta_loss_weight: float = 0.0
    additive_condition_delta_loss_warmup_start: int = 0
    additive_condition_delta_loss_warmup_end: int = 0
    # Optional train-single condition-prior teacher. This builds a per-dataset
    # bank from canonical train single-gene response deltas only, samples
    # synthetic multi-gene conditions, and supervises the t=0 velocity mean to
    # match the summed train-single prior. It is distinct from
    # composition_delta_loss because the prior bank is deterministic and
    # split-auditable rather than an online queue of recently seen batches.
    condition_prior_delta_loss_weight: float = 0.0
    condition_prior_delta_loss_warmup_start: int = 0
    condition_prior_delta_loss_warmup_end: int = 0
    condition_prior_delta_loss_every: int = 1
    # Optional deterministic train-single prior supervision for the additive
    # condition-delta head. This trains atom-sum predictions directly, unlike
    # condition_prior_delta_loss which supervises the t=0 velocity path.
    condition_prior_additive_delta_loss_weight: float = 0.0
    condition_prior_additive_delta_loss_warmup_start: int = 0
    condition_prior_additive_delta_loss_warmup_end: int = 0
    condition_prior_bank_max_cells: int = 512
    condition_prior_bank_min_norm: float = 1e-6
    # Number of genes sampled from the deterministic prior bank. Default 2
    # preserves synthetic multi-composition behavior; explicit 1 is for
    # Track A single-gene reliability adapter smokes.
    condition_prior_num_genes: int = 2
    # Source for deterministic condition-prior teacher:
    # - same_dataset: legacy behavior; build a per-dataset bank from the active
    #   training dataset only.
    # - global: build one no-leakage bank from canonical train single-gene
    #   conditions across datasets and use it for every active dataset.
    condition_prior_bank_scope: str = "same_dataset"  # same_dataset | global
    condition_prior_bank_split_file: str = ""  # optional full canonical split for global scope
    # condition: keep one record per train single condition; gene_mean: average
    # train single deltas per gene before sampling synthetic combinations.
    # gene_shrink_k{value}: per-dataset empirical-Bayes gene prior,
    # alpha * global_gene_mean + (1-alpha) * dataset_mean, where
    # alpha = global_gene_count / (global_gene_count + k). This is default-off
    # and intended for train-only gene-reliability adapter smokes.
    # gene_shrink_k{value}_jiang_lowcount_mask: same shrink prior, but for the
    # predeclared Jiang_IFNG/Jiang_TNFA low-count gene safety valve,
    # global_gene_count <= 1 falls back to the dataset mean.
    # gene_shrink_k{value}_dataset_negative_mask: broader Track A CPU-gated
    # fallback list where train-only internal proxy showed shrink < dataset
    # mean; use only as an explicit experimental branch.
    condition_prior_bank_aggregation: str = "condition"  # condition | gene_mean | gene_shrink_k2 | gene_shrink_k4 | gene_shrink_k2_jiang_lowcount_mask | gene_shrink_k2_dataset_negative_mask | gene_shrink_k4_dataset_negative_mask
    # Optional anchor-replay/no-harm loss for small finetune modules. This keeps
    # train-present low-risk strata close to a frozen anchor checkpoint while
    # pairwise/synthetic branches learn on their auxiliary objectives.
    anchor_replay_loss_weight: float = 0.0
    anchor_replay_loss_warmup_start: int = 0
    anchor_replay_loss_warmup_end: int = 0
    # all | non_gene_multi. The latter uses only perturbation metadata and is
    # intended for pairwise branches where train-present single/drug behavior
    # must not drift while synthetic multi-gene priors train small adapters.
    anchor_replay_condition_filter: str = "all"
    # Optional comma/semicolon-separated dataset allow-list for anchor replay.
    # Empty preserves legacy behavior and applies replay to all datasets.
    anchor_replay_dataset_filter: str = ""
    # Defaults to init_checkpoint when empty.
    anchor_replay_checkpoint: str = ""
    # Default-off compatibility guard: when the no-harm anchor was selected and
    # reported with EMA, replay should optionally load that EMA shadow rather
    # than the raw live model weights.
    anchor_replay_checkpoint_use_ema: bool = False
    # Optional Track C routed support-teacher distillation. This is for
    # true-multi adapter smokes only. The route file and teacher bank must be
    # built from Track C train/support decisions, while final query conditions
    # remain held out. Default 0 preserves legacy behavior.
    trackc_routed_distill_loss_weight: float = 0.0
    trackc_routed_distill_loss_warmup_start: int = 0
    trackc_routed_distill_loss_warmup_end: int = 0
    trackc_routed_distill_route_file: str = ""
    # Endpoint-level variant of the same routed support teacher. This directly
    # supervises predicted endpoint cells toward source + routed teacher delta,
    # while the legacy routed distill loss supervises the condition-delta head.
    trackc_routed_endpoint_loss_weight: float = 0.0
    trackc_routed_endpoint_loss_warmup_start: int = 0
    trackc_routed_endpoint_loss_warmup_end: int = 0
    # Optional train-only split used only to build routed teacher banks. This
    # lets a Track C smoke train with a route-focused split while preserving
    # additive/global single-gene teachers from the full trainselect train set.
    trackc_routed_distill_bank_split_file: str = ""
    # Current implementation supervises the condition-delta head in endpoint
    # delta frame: mean(GT) - mean(control). Pearson_pert centering is still
    # applied only at evaluation.
    trackc_routed_distill_target_frame: str = "endpoint_delta"
    # Optional train-multi memory readout for Track C routed teacher targets.
    # Default "off" preserves the older additive/dataset routes. When enabled,
    # the memory bank is still built only from the configured train-only bank
    # split and must not include support-val or held-out query conditions.
    trackc_routed_distill_memory_mode: str = "off"  # off | jaccard | overlap
    trackc_routed_distill_memory_k: int = 3
    trackc_routed_distill_memory_min_score: float = 0.25
    trackc_routed_distill_memory_scope: str = "same_dataset"  # same_dataset | all_dataset

    # ── OT ────────────────────────────────────────────────────────
    # "torch_sinkhorn": GPU 上纯 torch log-space Sinkhorn（默认，消除 CPU bound）
    # "exact" / "sinkhorn": CPU POT（兼容旧流水线，需要 OTPrefetchIter 多线程）
    ot_method: str = "torch_sinkhorn"
    # How to convert the mini-batch transport plan into training pairs:
    # "multinomial" = current default, sample transport mass with replacement;
    # "assignment" = greedy one-to-one assignment from the same plan;
    # "hungarian" = true min-cost one-to-one assignment from the cost matrix;
    # "random" = no OT, keep independent same-condition random source/GT draws.
    ot_pair_mode: str = "multinomial"
    ot_threads: int = 4
    ot_sinkhorn_reg: float = 0.05
    ot_sinkhorn_iter: int = 50
    # 以下仅 CPU 后端（exact / sinkhorn）下生效
    prefetch: int = 8
    n_ot_workers: int = 6

    # ── EMA / AMP ─────────────────────────────────────────────────
    # FM/diffusion 模型对 EMA 极为敏感；默认开启，评估时自动 swap 到 EMA 权重。
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after: int = 1000   # 等 warmup 结束再开始 EMA（避免平均早期噪声）
    ema_update_every: int = 1
    # AMP：bf16 无需 GradScaler，对 A100/3090+ 几乎无损且 30-40% 加速。
    use_amp: bool = True
    amp_dtype: str = "bf16"        # "bf16" | "fp16" | "off"

    # ── logging / checkpointing / early stopping ────────────────
    print_every: int = 200
    # Mid-epoch eval was removed for full-data runs. Keep this as a lightweight
    # periodic latest.pt checkpoint interval so long epochs can resume after
    # preemption without waiting for epoch-end evaluation.
    eval_every: int = 2000
    # Evaluation caps for smoke tests / hyperparameter search. Defaults keep
    # formal training and final reporting on the full test split.
    eval_max_conditions: int = 0  # 0 = all test conditions
    eval_max_conditions_per_dataset: int = 0  # 0 = all test conditions per dataset
    eval_max_mse_cells: int = 0  # per condition; 0 = all paired cells
    eval_max_mmd_cells: int = 2048  # per condition
    eval_max_chunk: int = 256  # ODE/eval chunk size; lower values reduce final-eval peak VRAM
    eval_save_condition_means: bool = False  # default-off artifact for posthoc residual/blend audits
    train_eval_enabled: bool = True  # False skips epoch/final IID/OOD eval for train-only smokes
    save_dir: str = "checkpoints"
    log_file: str = "train.log"
    patience: int = 10
    selection_metric: str = "test_mse"  # test_mse | test_mmd | direct_pearson | pearson_ctrl | pearson_pert | pearson_pert_minus_mmd | pearson_ctrl_minus_mmd
    selection_mmd_lambda: float = 1.0

    # ── device ────────────────────────────────────────────────────
    gpu: int = 0

    # ── finetune / warm-start ───────────────────────────────────────
    # If set: load only ``model`` weights from this .pt (must contain key ``model``),
    # use a fresh optimizer, step=0, reset best_score; do NOT resume ``save_dir/latest.pt``.
    init_checkpoint: str = ""
    # Default-off compatibility guard for anchor-preserving finetunes. When the
    # anchor checkpoint was selected/evaluated with EMA, enabling this makes the
    # warm-start base match that selected anchor instead of the raw live weights.
    init_checkpoint_use_ema: bool = False
    # Finetune only: for ``control_mlp``, freeze ``shared_enc`` so IR/GT share the same frozen encoder.
    freeze_shared_enc: bool = False
    # Finetune-only trainable scope. ``all`` preserves legacy behavior.
    # ``pairwise_adapter`` freezes the warm-started model except
    # ``pert_encoder.pair_to_out.*`` for interaction-only smoke tests.
    # ``type_adapter`` freezes the warm-started model except perturbation-type
    # scaling/gating parameters inside ``pert_encoder.type_*``.
    # ``pairwise_condition_adapter`` additionally opens the small condition
    # projection bridge (``pert_to_c.*`` when present and
    # ``condition_delta_to_c.*`` when enabled) while keeping the base flow and
    # shared condition-delta head frozen.
    # ``condition_prior_adapter`` is for anchor-preserving train-single
    # synthetic-composition probes: only the newly initialized condition-delta
    # head and its zero-initialized bridge into the conditioning path train.
    # ``support_context_adapter`` trains only the Track C support-context bridge.
    # ``support_residual_adapter`` trains only the Track C support residual
    # output operator.
    # ``support_film_adapter`` trains only the Track C support output FiLM
    # shift/scale operator.
    # ``support_set_task_adapter`` trains only the distinct Track C
    # support-set task bridge.
    finetune_trainable_scope: str = "all"  # all | type_adapter | pairwise_adapter | pairwise_condition_adapter | condition_prior_adapter | support_context_adapter | support_residual_adapter | support_film_adapter | support_set_task_adapter

    # ── perturbation conditioning (ControlMLP only; safe OFF by default) ──
    # ``latent_fm_arch_version``: checkpoint / FrozenLatentFM compatibility tag (v1=legacy weights only).
    latent_fm_arch_version: str = "v1"
    use_pert_condition: bool = False
    pert_embed_mode: str = (
        "pretrained_frozen"  # PerturbationConditionEncoder mode
        # pretrained_frozen | pretrained_tunable | pretrained_with_type_gate | combo_id_baseline
    )
    pert_gene_emb_cache_dir: str = str(paths.scgpt_cache_dir())  # GeneEmbeddingCache root
    pert_encoder_num_embeddings: int = 8192  # random_learned table size (use ≥ vocab if ids from cache spill)
    pert_gene_emb_dim: int = 256  # random_learned learned gene slot dim only
    pert_cond_dim: int = 512  # encoder output (= adaLN additive cond width after projection); default aligns mlp_d_model
    max_pert_genes: int = 16
    pert_type_emb_dim: int = 32
    pert_encoder_dropout: float = 0.0
    max_combo_id_exclusive: int = 4096  # combo_id_baseline embedding upper bound (exclusive upper ID clamp)

    # When True and biFlow GT AnnData exists with obs columns: refine metadata per condition once.
    use_h5ad_pert_metadata: bool = False
    # Per-dataset perturbation_type fallback JSON; empty string disables. Used only when use_pert_condition and obs type is null.
    pert_metainfo_path: str = str(_REPO_ROOT / "data/raw/genepert_DE5000/metainfo.json")
    # chemicalpert / sciplex3 等药筛数据的 metainfo（与 genepert 相同：list[dict]，字段含 dataset、perturbation_type、cell_line）。
    # 供后续接 chemical dataloader 与 UniMol 向量时使用；空字符串关闭。
    chemical_metainfo_path: str = str(_REPO_ROOT / "data/raw/chemicalpert_DE5000/metainfo.json")
    drug_emb_cache_dir: str = field(
        default_factory=lambda: (
            os.environ.get("RAW_DRUG_EMB_CACHE_DIR", "")
            or os.environ.get("LATENT_DRUG_EMB_CACHE_DIR", "")
        ).strip(),
    )
    max_chem_keys: int = 4
    chem_fallback_embed_dim: int = 512
    # Optional hook: obs column for precomputed chem vectors (not wired yet).
    chem_obs_column: str = ""
    chem_emb_source_dir: str = ""
    # Encoder chemical branch (ControlMLP only; 0 = disabled, same as latent raw ``pert_chem_*``).
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
    # Optional gene-gene interaction token inside UnifiedConditionEncoder.
    # Default off keeps old checkpoints and state_dicts unchanged.
    pert_pairwise_mode: str = "off"  # off | hadamard_mean
    pert_condition_embedding_source: str = field(default_factory=lambda: os.environ.get(
        "PERT_EMBED_SOURCE", "",
    ).strip())
    # Resolve chemical perturbation embeddings in the dataloader (needs ``chem_emb_source_dir`` wiring).
    pert_chem_enabled: bool = False

    # ── ablation knobs (v3) ───────────────────────────────────────
    # zero (default, original ada-zero behaviour); xavier_small (gain=0.1) unlocks
    # pert signal from step 0 instead of waiting for pert_to_c to depart from 0.
    pert_to_c_init_mode: str = "zero"  # zero | xavier_small
    # When True, the pert projection (after pert_to_c) is also added to the
    # main hidden stream `h` right after fusion — not just into the adaLN cond.
    use_pert_in_fusion: bool = False
    # Track C experimental support-context path. This is intentionally
    # default-off and has no parameters unless enabled, preserving legacy
    # checkpoint/state_dict compatibility. Any runtime context source must be
    # split-audited separately before training.
    trackc_support_context_use_in_model: bool = False
    # Optional Track C support-conditioned residual operator. It consumes the
    # same audited support_context but adds a direct output/velocity residual
    # path instead of only modifying the adaLN conditioning vector. Default-off.
    trackc_support_residual_use_in_model: bool = False
    # Optional Track C support-conditioned FiLM-style output operator. It uses
    # the same support_context to add a direct residual plus a support-derived
    # scale on the absolute current output. Default-off.
    trackc_support_film_use_in_model: bool = False
    trackc_support_context_dim: int = 0
    trackc_support_context_source: str = "off"  # off | routed_distill_target
    # Eval-only negative control for Track C support-context diagnostics.
    # Training always uses ``actual``. ``zero`` and ``shuffle_condition`` are
    # used only by explicit posthoc eval CLIs to prove support-present effects.
    trackc_support_context_eval_control: str = "actual"  # actual | zero | shuffle_condition
    # Optional train/eval support-context mask. Default ``off`` preserves all
    # existing Track C support-context behavior. Experimental values are based
    # only on the safe trainselect split named by
    # ``trackc_routed_distill_bank_split_file``.
    trackc_support_context_pair_type_filter: str = "off"  # off | none_train_single | both_train_multi_gene | none_train_single_both_train_multi_gene
    # Distinct Track C support-set task adapter. This is a default-off code
    # boundary for future safe-trainselect CPU gates; it consumes an already
    # encoded support-set/task summary through the explicit ``support_set_task``
    # forward argument and does not reuse routed support_context sources.
    trackc_support_set_task_use_in_model: bool = False
    trackc_support_set_task_dim: int = 0
    trackc_support_set_task_source: str = "off"  # off | shared_gene_condition_means
    trackc_support_set_task_safe_split_file: str = ""
    trackc_support_set_task_anchor_condition_means: str = ""
    trackc_support_set_task_candidate_condition_means: str = ""
    trackc_support_set_task_scale: float = 1.0
    trackc_support_set_task_min_support_count: int = 1
    trackc_support_set_task_eval_control: str = "actual"  # actual | zero | shuffle_condition | absent

    def make_save_dir(self) -> Path:
        p = Path(self.save_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
