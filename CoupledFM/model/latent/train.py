#!/usr/bin/env python3
"""
Latent Flow Matching training script.

Current maintained models:
  - ``control_mlp``: default ControlMLP velocity field
  - ``mlp``: plain MLP velocity field

Training loop:
  1. Build a globally shuffled list of (dataset, condition) visits for the epoch
  2. Perform mini-batch OT pairing of source-pool / GT within each condition
  3. Sample the CondOT linear path: x_t = (1-t)*src + t*GT, dx = GT - src
  4. Optimize velocity MSE with optional MMD regularization
  5. Run full evaluation at the end of each epoch
"""

import json
import csv
import gc
import math
import os
import sys
import time
import queue
import threading
import warnings
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import tyro
from torch.nn.parallel import DistributedDataParallel as DDP

import dataclasses
import hashlib

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset, load_or_create_split
from model.latent.models import MLPVelocityField, ControlMLPVelocityField
from model.latent.fm_ot import CondOTPath, OTPlanSampler, median_sigmas, mmd2_biased, mmd2_unbiased
from model.latent.response_normalizer import ResponseNormalizer

from model.utils.train.ema import ModelEMA
from model.utils.train.schedulers import lr_warmup_cosine_to_eta_min
from model.utils.train.time_sampling import sample_t_torch
from model.utils.conditioning.perturbation import ConditionMetadata, PerturbationBatch
from model.utils.embeddings.gene_cache import GeneEmbeddingCache
from model.condition_emb.genepert import PERT_TYPE_DRUG


LAST_CONDITION_PRIOR_BANK_SUMMARY: dict[str, Any] = {}
LAST_TRACKC_ROUTED_DISTILL_SUMMARY: dict[str, Any] = {}


def _amp_autocast_ctx(cfg: Config, device: torch.device):
    """Return a ``torch.autocast`` context (or nullcontext) per cfg."""
    from contextlib import nullcontext
    if not getattr(cfg, "use_amp", False) or str(getattr(cfg, "amp_dtype", "off")).lower() == "off":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        str(cfg.amp_dtype).lower(), torch.bfloat16
    )
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_condition_loss_weight_table(
    path: str,
    *,
    weight_column: str = "weight",
) -> dict[tuple[str, str], float]:
    """Load optional per-condition loss weights from CSV/TSV."""
    path_s = str(path or "").strip()
    if not path_s:
        return {}
    p = Path(path_s).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"condition_loss_weight_file not found: {p}")
    delimiter = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    out: dict[tuple[str, str], float] = {}
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        required = {"dataset", "condition", str(weight_column)}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"condition_loss_weight_file missing columns {missing}: {p}")
        for row in reader:
            ds = str(row.get("dataset", "")).strip()
            cond = str(row.get("condition", "")).strip()
            if not ds or not cond:
                continue
            weight = float(row[str(weight_column)])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"bad condition loss weight for {(ds, cond)}: {weight!r}")
            out[(ds, cond)] = weight
    if not out:
        raise ValueError(f"condition_loss_weight_file loaded zero rows: {p}")
    return out

def build_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    support_context_enabled = bool(getattr(cfg, "trackc_support_context_use_in_model", False))
    support_residual_enabled = bool(getattr(cfg, "trackc_support_residual_use_in_model", False))
    support_film_enabled = bool(getattr(cfg, "trackc_support_film_use_in_model", False))
    support_set_task_enabled = bool(getattr(cfg, "trackc_support_set_task_use_in_model", False))
    if (
        support_context_enabled
        or support_residual_enabled
        or support_film_enabled
        or support_set_task_enabled
    ) and cfg.model_type != "control_mlp":
        raise ValueError("Track C support context/residual paths require model_type=control_mlp")
    kwargs = dict(
        emb_dim=cfg.emb_dim,
        d_model=cfg.mlp_d_model,
        n_layers=cfg.mlp_n_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )
    use_pert = bool(getattr(cfg, "use_pert_condition", False))
    if cfg.model_type == "control_mlp":
        kwargs.update(
            dict(
                trackc_support_context_use_in_model=support_context_enabled,
                trackc_support_residual_use_in_model=support_residual_enabled,
                trackc_support_film_use_in_model=support_film_enabled,
                trackc_support_context_dim=int(getattr(cfg, "trackc_support_context_dim", 0) or 0),
                trackc_support_set_task_use_in_model=support_set_task_enabled,
                trackc_support_set_task_dim=int(getattr(cfg, "trackc_support_set_task_dim", 0) or 0),
            )
        )
    if use_pert:
        if cfg.model_type != "control_mlp":
            raise ValueError("use_pert_condition requires model_type=control_mlp")
        mode = str(getattr(cfg, "pert_embed_mode", "random_learned")).lower().strip()
        enc_cache = None
        if mode.startswith("pretrained"):
            d = str(getattr(cfg, "pert_gene_emb_cache_dir", "") or "").strip()
            if not d:
                raise ValueError("pert_gene_emb_cache_dir is required for pretrained perturbation_encoder modes")
            enc_cache = GeneEmbeddingCache(Path(d).expanduser())
        kwargs.update(
            dict(
                use_pert_condition=True,
                pert_embed_mode=getattr(cfg, "pert_embed_mode", "random_learned"),
                pert_cond_dim=int(getattr(cfg, "pert_cond_dim", cfg.mlp_d_model)),
                pert_type_emb_dim=int(getattr(cfg, "pert_type_emb_dim", 32)),
                pert_encoder_num_embeddings=int(getattr(cfg, "pert_encoder_num_embeddings", 8192)),
                pert_gene_emb_dim=int(getattr(cfg, "pert_gene_emb_dim", 256)),
                pert_encoder_dropout=float(getattr(cfg, "pert_encoder_dropout", 0.0)),
                max_combo_id_exclusive=int(getattr(cfg, "max_combo_id_exclusive", 4096)),
                gene_embedding_cache=enc_cache,
                pert_chem_emb_dim=int(getattr(cfg, "pert_chem_emb_dim", 0) or 0),
                pert_chem_projector_hidden=int(getattr(cfg, "pert_chem_projector_hidden", 0) or 0),
                pert_gene_projector_hidden=int(getattr(cfg, "pert_gene_projector_hidden", 0) or 0),
                pert_type_scale_init=tuple(
                    getattr(cfg, "pert_type_scale_init", (0.0, -1.0, -1.0, -1.0, 1.0, 1.0)),
                ),
                pool_aggregations=tuple(getattr(cfg, "pert_pool_aggregations", ("mean",))),
                pool_scale_init=tuple(float(x) for x in getattr(cfg, "pert_pool_scale_init", (1.0,))),
                pool_fusion_mode=str(getattr(cfg, "pert_pool_fusion_mode", "sum")),
                type_adapter_mode=str(getattr(cfg, "pert_type_adapter_mode", "scalar")),
                pairwise_mode=str(getattr(cfg, "pert_pairwise_mode", "off")),
                condition_embedding_source=(
                    str(getattr(cfg, "pert_condition_embedding_source", "") or "").strip() or None
                ),
                pert_to_c_init_mode=str(getattr(cfg, "pert_to_c_init_mode", "zero")),
                use_pert_in_fusion=bool(getattr(cfg, "use_pert_in_fusion", False)),
                use_condition_delta_head=(
                    float(getattr(cfg, "condition_delta_head_loss_weight", 0.0) or 0.0) > 0
                    or bool(getattr(cfg, "condition_delta_head_use_in_model", False))
                    or float(getattr(cfg, "additive_condition_delta_loss_weight", 0.0) or 0.0) > 0
                    or float(getattr(cfg, "condition_prior_additive_delta_loss_weight", 0.0) or 0.0) > 0
                    or float(getattr(cfg, "trackc_routed_distill_loss_weight", 0.0) or 0.0) > 0
                    or float(getattr(cfg, "trackc_routed_endpoint_loss_weight", 0.0) or 0.0) > 0
                ),
                condition_delta_head_hidden=int(getattr(cfg, "condition_delta_head_hidden", 1024)),
                condition_delta_head_use_in_model=bool(
                    getattr(cfg, "condition_delta_head_use_in_model", False)
                ),
                condition_lowrank_residual_use_in_model=bool(
                    getattr(cfg, "condition_lowrank_residual_use_in_model", False)
                ),
                condition_lowrank_residual_rank=int(
                    getattr(cfg, "condition_lowrank_residual_rank", 32)
                ),
                condition_delta_in_model_filter=str(
                    getattr(cfg, "condition_delta_in_model_filter", "all") or "all"
                ),
            )
        )
        model = ControlMLPVelocityField(**kwargs)
    elif cfg.model_type == "control_mlp":
        model = ControlMLPVelocityField(**kwargs)
    elif cfg.model_type == "mlp":
        model = MLPVelocityField(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")
    return model.to(device)


def count_params(model: torch.nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.1f}K"


def fill_condition_embedding_source(cfg: Config) -> None:
    """Populate and sanity-check the condition embedding provenance label."""
    if not bool(getattr(cfg, "use_pert_condition", False)):
        return
    source = str(getattr(cfg, "pert_condition_embedding_source", "") or "").strip()
    cache_dir = str(getattr(cfg, "pert_gene_emb_cache_dir", "") or "").strip()
    if source:
        if cache_dir:
            cache_name = Path(cache_dir).expanduser().name
            if cache_name and source != cache_name:
                warnings.warn(
                    "pert_condition_embedding_source does not match "
                    f"pert_gene_emb_cache_dir basename: source={source!r}, "
                    f"cache={cache_name!r}. Set PERT_EMBED_SOURCE with GENE_CACHE "
                    "when switching condition caches.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return
    if not cache_dir:
        return
    cfg.pert_condition_embedding_source = Path(cache_dir).expanduser().name


def _pert_to_device(pb: Optional[tuple], device: torch.device) -> Optional[tuple]:
    if pb is None:
        return None
    if len(pb) == 5:
        gid, mk, tid, npt, cid = pb
        ce, cm = None, None
    else:
        gid, mk, tid, npt, cid, ce, cm = pb
    if cm is None:
        bsz = int(gid.shape[0])
        cm = torch.zeros((bsz, 1), dtype=torch.bool, device=gid.device)
    return (
        gid.to(device, non_blocking=True),
        mk.to(device, non_blocking=True),
        tid.to(device, non_blocking=True),
        npt.to(device, non_blocking=True),
        None if cid is None else cid.to(device, non_blocking=True),
        None if ce is None else ce.to(device, non_blocking=True),
        cm.to(device, non_blocking=True),
    )


def _stable_int_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _pert_for_eval_batch(
    dataset: CrossDatasetFMDataset,
    ds_name: str,
    cond: str,
    batch_size: int,
):
    cache = dataset.gene_embedding_cache
    if cache is None:
        raise RuntimeError("evaluation with perturb conditioning requires dataset.use_pert_condition=True")
    meta = dataset.metadata_for_condition(ds_name, cond)
    meta = dataset.enrich_metadata_with_chem(meta)
    rows = [meta] * int(batch_size)
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=int(dataset.max_pert_genes),
        max_chem_slots=int(getattr(dataset, "max_chem_keys", 4)),
        device=torch.device("cpu"),
    )
    return pb.as_tuple_full()


def _unpack_pert_up_to7(pb: tuple) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Unpack perturbation_batch from dataloader (5 legacy or 7 full)."""
    if len(pb) == 5:
        gid, mk, tid, npt, cid = pb
        return gid, mk, tid, npt, cid, None, None
    gid, mk, tid, npt, cid, ce, cm = pb
    return gid, mk, tid, npt, cid, ce, cm


def _pert_chunk(pb: tuple, start: int, end: int) -> tuple:
    gid, mk, tid, npt, cid, ce, cm = _unpack_pert_up_to7(pb)
    ce_chunk = (
        None
        if ce is None
        else ce[start:end].contiguous()
    )
    cm_chunk = (
        None
        if cm is None
        else cm[start:end].contiguous()
    )
    return (
        gid[start:end].contiguous(),
        mk[start:end].contiguous(),
        tid[start:end].contiguous(),
        npt[start:end].contiguous(),
        None if cid is None else cid[start:end].contiguous(),
        ce_chunk,
        cm_chunk,
    )


def _single_gene_composition_key(meta: ConditionMetadata) -> Optional[tuple[str, Optional[str]]]:
    """Return a stable key for single-gene perturbations usable in composition replay."""
    genes = tuple(str(g).strip().upper() for g in getattr(meta, "genes", ()) if str(g).strip())
    if len(genes) != 1:
        return None
    if int(meta.resolved_nperts()) != 1:
        return None
    if getattr(meta, "chem_emb", None) is not None or getattr(meta, "chem_emb_list", None):
        return None
    ptype = getattr(meta, "perturbation_type_raw", None)
    return genes[0], (None if ptype is None else str(ptype))


def _make_gene_combo_perturbation_batch(
    *,
    genes: tuple[str, ...],
    perturbation_type_raw: Optional[str],
    batch_size: int,
    cache: GeneEmbeddingCache,
    max_genes: int,
    max_chem_keys: int,
) -> tuple:
    """Build a CPU perturbation tuple for a synthetic gene-combo condition."""
    clean_genes = tuple(sorted({str(g).strip().upper() for g in genes if str(g).strip()}))
    meta = ConditionMetadata(
        genes=clean_genes,
        perturbation_type_raw=perturbation_type_raw,
        combo_id=0,
        nperts_obs=len(clean_genes),
    )
    rows = [meta] * int(batch_size)
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=int(max_genes),
        max_chem_slots=int(max_chem_keys),
        device=torch.device("cpu"),
    )
    return pb.as_tuple_full()


def _sample_rows_mean(
    arr: np.ndarray,
    *,
    max_cells: int,
    seed: int,
) -> torch.Tensor:
    """Return a deterministic capped mean without loading extra copies."""
    n = int(arr.shape[0])
    if n <= 0:
        raise ValueError("cannot average an empty condition slice")
    if max_cells > 0 and n > max_cells:
        rng = np.random.RandomState(int(seed))
        idx = np.sort(rng.choice(n, size=int(max_cells), replace=False))
        arr = arr[idx]
    return torch.from_numpy(np.asarray(arr, dtype=np.float32).mean(axis=0))


def _aggregate_condition_prior_records(
    records_by_dataset: dict[str, list[tuple[str, Optional[str], torch.Tensor]]],
    *,
    aggregation: str,
) -> dict[str, list[tuple[str, Optional[str], torch.Tensor]]]:
    """Optionally collapse condition-level prior records to per-gene means."""
    mode = str(aggregation or "condition").strip().lower()
    if mode == "condition":
        return records_by_dataset
    if mode == "gene_mean":
        out: dict[str, list[tuple[str, Optional[str], torch.Tensor]]] = {}
        for ds_name, records in records_by_dataset.items():
            by_gene: dict[str, list[tuple[Optional[str], torch.Tensor]]] = defaultdict(list)
            for gene, ptype, delta in records:
                by_gene[str(gene).strip().upper()].append((ptype, delta.float().cpu()))
            rows: list[tuple[str, Optional[str], torch.Tensor]] = []
            for gene, vals in sorted(by_gene.items()):
                ptypes = {ptype for ptype, _delta in vals if ptype is not None}
                ptype = next(iter(ptypes)) if len(ptypes) == 1 else None
                delta = torch.stack([delta for _ptype, delta in vals], dim=0).mean(dim=0).float()
                rows.append((gene, ptype, delta))
            if rows:
                out[ds_name] = rows
        return out
    if mode.startswith("gene_shrink_k"):
        jiang_lowcount_mask = mode.endswith("_jiang_lowcount_mask")
        dataset_negative_mask = mode.endswith("_dataset_negative_mask")
        guarded_dataset_negative = {
            "Adamson",
            "DixitRegev2016_K562_TFs_High_MOI",
            "GasperiniShendure2019_lowMOI",
            "Jiang_IFNG",
            "Jiang_TNFA",
            "NormanWeissman2019_filtered",
            "Schmidt",
        }
        k_s = mode.removeprefix("gene_shrink_k")
        if jiang_lowcount_mask:
            k_s = k_s.removesuffix("_jiang_lowcount_mask")
        if dataset_negative_mask:
            k_s = k_s.removesuffix("_dataset_negative_mask")
        try:
            k_value = float(k_s)
        except ValueError as exc:
            raise ValueError(
                "condition_prior_bank_aggregation gene shrink mode must look like "
                "'gene_shrink_k2', 'gene_shrink_k4', or "
                "'gene_shrink_k2_jiang_lowcount_mask', or "
                "'gene_shrink_k2_dataset_negative_mask'; got "
                f"{aggregation!r}"
            ) from exc
        if k_value <= 0:
            raise ValueError(f"gene shrink k must be positive; got {k_value}")
        all_deltas: list[torch.Tensor] = []
        by_gene: dict[str, list[tuple[Optional[str], torch.Tensor]]] = defaultdict(list)
        by_dataset: dict[str, list[torch.Tensor]] = defaultdict(list)
        for ds_name, records in records_by_dataset.items():
            for gene, ptype, delta in records:
                gene_key = str(gene).strip().upper()
                delta_cpu = delta.float().cpu()
                by_gene[gene_key].append((ptype, delta_cpu))
                by_dataset[str(ds_name)].append(delta_cpu)
                all_deltas.append(delta_cpu)
        if not all_deltas:
            return {}
        global_mean = torch.stack(all_deltas, dim=0).mean(dim=0).float()
        gene_stats: dict[str, tuple[Optional[str], torch.Tensor, int]] = {}
        for gene, vals in by_gene.items():
            ptypes = {ptype for ptype, _delta in vals if ptype is not None}
            ptype = next(iter(ptypes)) if len(ptypes) == 1 else None
            gene_mean = torch.stack([delta for _ptype, delta in vals], dim=0).mean(dim=0).float()
            gene_stats[gene] = (ptype, gene_mean, len(vals))
        out = {}
        for ds_name in sorted(records_by_dataset):
            ds_vals = by_dataset.get(str(ds_name)) or []
            ds_mean = torch.stack(ds_vals, dim=0).mean(dim=0).float() if ds_vals else global_mean
            rows = []
            for gene, (ptype, gene_mean, gene_count) in sorted(gene_stats.items()):
                alpha = float(gene_count) / (float(gene_count) + k_value)
                use_dataset_mean = (
                    jiang_lowcount_mask
                    and str(ds_name) in {"Jiang_IFNG", "Jiang_TNFA"}
                    and int(gene_count) <= 1
                ) or (
                    dataset_negative_mask
                    and str(ds_name) in guarded_dataset_negative
                )
                if use_dataset_mean:
                    delta = ds_mean.float()
                else:
                    delta = (alpha * gene_mean + (1.0 - alpha) * ds_mean).float()
                rows.append((gene, ptype, delta))
            if rows:
                out[str(ds_name)] = rows
        return out
    if mode != "gene_mean":
        raise ValueError(
            "condition_prior_bank_aggregation must be one of: condition, gene_mean, "
            "gene_shrink_k{positive_number}, or "
            "gene_shrink_k{positive_number}_jiang_lowcount_mask, or "
            "gene_shrink_k{positive_number}_dataset_negative_mask; "
            f"got {aggregation!r}"
        )
    raise AssertionError("unreachable condition-prior aggregation mode")


def _summarize_condition_prior_bank(
    *,
    raw_bank: dict[str, list[tuple[str, Optional[str], torch.Tensor]]],
    final_bank: dict[str, list[tuple[str, Optional[str], torch.Tensor]]],
    scope: str,
    aggregation: str,
    split_file: str,
    max_cells: int,
    min_norm: float,
    skipped: int,
    raw_records: int,
) -> dict[str, Any]:
    """Return JSON-safe provenance for the deterministic condition-prior bank."""
    per_gene: dict[str, dict[str, Any]] = {}
    for ds_name, records in raw_bank.items():
        for gene, ptype, _delta in records:
            key = str(gene).strip().upper()
            item = per_gene.setdefault(
                key,
                {
                    "raw_condition_count": 0,
                    "source_datasets": defaultdict(int),
                    "perturbation_types": defaultdict(int),
                },
            )
            item["raw_condition_count"] += 1
            item["source_datasets"][str(ds_name)] += 1
            if ptype is not None:
                item["perturbation_types"][str(ptype)] += 1

    final_per_gene: dict[str, dict[str, Any]] = {}
    for bank_name, records in final_bank.items():
        for gene, ptype, delta in records:
            key = str(gene).strip().upper()
            raw = per_gene.get(key, {})
            final_per_gene[key] = {
                "bank": str(bank_name),
                "raw_condition_count": int(raw.get("raw_condition_count", 0)),
                "source_datasets": dict(sorted((raw.get("source_datasets") or {}).items())),
                "perturbation_types": dict(sorted((raw.get("perturbation_types") or {}).items())),
                "final_perturbation_type": None if ptype is None else str(ptype),
                "delta_norm": float(delta.float().norm().item()),
                "delta_dim": int(delta.numel()),
            }

    summary = {
        "scope": scope,
        "aggregation": aggregation,
        "split_file": split_file,
        "max_cells": int(max_cells),
        "min_norm": float(min_norm),
        "skipped_records": int(skipped),
        "raw_records": int(raw_records),
        "final_records": int(sum(len(v) for v in final_bank.values())),
        "raw_records_by_dataset": {k: len(v) for k, v in sorted(raw_bank.items())},
        "final_records_by_bank": {k: len(v) for k, v in sorted(final_bank.items())},
        "genes": dict(sorted(final_per_gene.items())),
    }
    aggregation_mode = str(aggregation).strip().lower()
    if aggregation_mode.endswith("_jiang_lowcount_mask"):
        summary["guarded_fallback"] = {
            "mode": "jiang_lowcount_mask",
            "fallback_datasets": ["Jiang_IFNG", "Jiang_TNFA"],
            "gene_train_count_threshold": 1,
            "fallback_target": "dataset_mean",
        }
    if aggregation_mode.endswith("_dataset_negative_mask"):
        summary["guarded_fallback"] = {
            "mode": "dataset_negative_mask",
            "fallback_datasets": [
                "Adamson",
                "DixitRegev2016_K562_TFs_High_MOI",
                "GasperiniShendure2019_lowMOI",
                "Jiang_IFNG",
                "Jiang_TNFA",
                "NormanWeissman2019_filtered",
                "Schmidt",
            ],
            "fallback_target": "dataset_mean",
        }
    return summary


def _condition_prior_source_dataset(
    dataset: CrossDatasetFMDataset,
    cfg: Config,
    *,
    log=None,
) -> tuple[CrossDatasetFMDataset, str]:
    """Return the dataset used to build train-only prior records."""
    scope = str(getattr(cfg, "condition_prior_bank_scope", "same_dataset") or "same_dataset")
    scope = scope.strip().lower()
    if scope in {"same_dataset", "dataset", "local"}:
        return dataset, "same_dataset"
    if scope not in {"global", "cross_dataset", "all_datasets"}:
        raise ValueError(
            "condition_prior_bank_scope must be one of: same_dataset, global; "
            f"got {getattr(cfg, 'condition_prior_bank_scope', None)!r}"
        )

    split_path_s = str(getattr(cfg, "condition_prior_bank_split_file", "") or "").strip()
    if not split_path_s:
        split_path_s = str(getattr(cfg, "split_file", "") or "").strip()
    if not split_path_s:
        split_path_s = str(Path(getattr(cfg, "biflow_dir", "")) / f"split_seed{int(cfg.split_seed)}.json")
    split_path = Path(split_path_s).expanduser()
    if not split_path.is_file():
        raise FileNotFoundError(
            "condition_prior_bank_scope=global requires a readable canonical split; "
            f"missing {split_path}"
        )
    with split_path.open("r", encoding="utf-8") as handle:
        source_split = json.load(handle)
    if not isinstance(source_split, dict):
        raise ValueError(f"condition_prior_bank_split_file is not a split dict: {split_path}")
    if log is not None:
        log(f"Condition-prior global bank split: {split_path}")
    source_ds = CrossDatasetFMDataset(
        data_dir=str(dataset.data_dir),
        split=source_split,
        batch_size=int(dataset.batch_size),
        seed=int(dataset.seed),
        mode="train",
        min_cells=int(dataset.min_cells),
        ds_alpha=1.0,
        scale_noise=0.0,
        min_selected_conditions_per_dataset=0,
        condition_visit_power=1.0,
        condition_visit_cap=0,
        use_pert_condition=True,
        max_pert_genes=int(dataset.max_pert_genes),
        gene_embedding_cache_dir=str(getattr(cfg, "pert_gene_emb_cache_dir", "") or ""),
        biflow_dir=str(getattr(cfg, "biflow_dir", "") or ""),
        use_h5ad_pert_metadata=bool(getattr(cfg, "use_h5ad_pert_metadata", False)),
        pert_metainfo_path=str(getattr(cfg, "pert_metainfo_path", "") or ""),
        chem_emb_source_dir=str(getattr(cfg, "chem_emb_source_dir", "") or ""),
        chem_obs_column=str(getattr(cfg, "chem_obs_column", "") or ""),
        drug_emb_cache_dir=str(getattr(cfg, "drug_emb_cache_dir", "") or ""),
        max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
        chemical_metainfo_path=str(getattr(cfg, "chemical_metainfo_path", "") or ""),
        chem_fallback_embed_dim=int(getattr(cfg, "chem_fallback_embed_dim", 512)),
        latent_backbone=str(getattr(cfg, "latent_backbone", "state") or "state"),
        pert_chem_enabled=bool(getattr(cfg, "pert_chem_enabled", False)),
        perturbation_family_filter=str(getattr(cfg, "perturbation_family_filter", "all") or "all"),
        ddp_rank=0,
        ddp_world_size=1,
        ddp_sync_min_len=False,
        silent=True,
    )
    return source_ds, "global"


def build_condition_prior_delta_bank(
    dataset: CrossDatasetFMDataset,
    cfg: Config,
    *,
    log=None,
) -> dict[str, list[tuple[str, Optional[str], torch.Tensor]]]:
    """Build a split-auditable train-single response prior bank.

    The bank is constructed only from ``dataset`` in train mode, so held-out
    multi-perturbation GT is never read. Each record is
    ``(gene_symbol, perturbation_type_raw, mean(gt)-mean(src))`` on CPU.
    """
    velocity_weight = float(getattr(cfg, "condition_prior_delta_loss_weight", 0.0) or 0.0)
    additive_head_weight = float(getattr(cfg, "condition_prior_additive_delta_loss_weight", 0.0) or 0.0)
    if velocity_weight <= 0 and additive_head_weight <= 0:
        return {}
    if dataset.gene_embedding_cache is None:
        raise ValueError("condition-prior delta losses require use_pert_condition=True")

    global LAST_CONDITION_PRIOR_BANK_SUMMARY
    source_dataset, bank_scope = _condition_prior_source_dataset(dataset, cfg, log=log)
    max_cells = int(getattr(cfg, "condition_prior_bank_max_cells", 512) or 512)
    min_norm = float(getattr(cfg, "condition_prior_bank_min_norm", 1e-6) or 1e-6)
    bank_raw: dict[str, list[tuple[str, Optional[str], torch.Tensor]]] = {}
    total = 0
    skipped = 0
    for ds_name in source_dataset.ds_names:
        handle = source_dataset.handles[ds_name]
        rows: list[tuple[str, Optional[str], torch.Tensor]] = []
        for cond in source_dataset.ds_conds.get(ds_name, []):
            meta = source_dataset.metadata_for_condition(ds_name, cond)
            key = _single_gene_composition_key(meta)
            if key is None:
                skipped += 1
                continue
            src = handle.read_src(cond)
            gt = handle.read_gt(cond)
            seed_base = _stable_int_hash(f"condition_prior:{ds_name}:{cond}")
            src_mean = _sample_rows_mean(src, max_cells=max_cells, seed=seed_base)
            gt_mean = _sample_rows_mean(gt, max_cells=max_cells, seed=seed_base + 17)
            delta = (gt_mean - src_mean).float()
            if torch.isfinite(delta).all() and delta.norm().item() > min_norm:
                rows.append((key[0], key[1], delta.cpu()))
                total += 1
            else:
                skipped += 1
        if rows:
            bank_raw[ds_name] = rows
    provenance_bank_raw = bank_raw
    if bank_scope == "global" and not str(
        getattr(cfg, "condition_prior_bank_aggregation", "condition") or "condition"
    ).strip().lower().startswith("gene_shrink_k"):
        global_rows: list[tuple[str, Optional[str], torch.Tensor]] = []
        for ds_name in sorted(bank_raw):
            global_rows.extend(bank_raw[ds_name])
        bank_raw = {"__global__": global_rows} if global_rows else {}
    if source_dataset is not dataset:
        for handle in source_dataset.handles.values():
            handle.close()
    aggregation = str(getattr(cfg, "condition_prior_bank_aggregation", "condition") or "condition")
    bank = _aggregate_condition_prior_records(bank_raw, aggregation=aggregation)
    split_file = ""
    if bank_scope == "global":
        split_file = str(getattr(cfg, "condition_prior_bank_split_file", "") or getattr(cfg, "split_file", ""))
    LAST_CONDITION_PRIOR_BANK_SUMMARY = _summarize_condition_prior_bank(
        raw_bank=provenance_bank_raw,
        final_bank=bank,
        scope=bank_scope,
        aggregation=aggregation,
        split_file=split_file,
        max_cells=max_cells,
        min_norm=min_norm,
        skipped=skipped,
        raw_records=total,
    )
    if log is not None:
        log(
            "Condition-prior delta bank: "
            f"scope={bank_scope} aggregation={aggregation} datasets={len(bank)} "
            f"raw_records={total} records={sum(len(v) for v in bank.values())} "
            f"skipped={skipped} max_cells={max_cells}"
        )
        for ds_name in sorted(bank):
            log(f"  condition_prior_bank[{ds_name}] = {len(bank[ds_name])}")
    return bank


def trackc_routed_distill_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional Track C routed support-teacher distillation."""
    weight = float(getattr(cfg, "trackc_routed_distill_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "trackc_routed_distill_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "trackc_routed_distill_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def trackc_routed_endpoint_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional Track C routed endpoint supervision."""
    weight = float(getattr(cfg, "trackc_routed_endpoint_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "trackc_routed_endpoint_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "trackc_routed_endpoint_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def _load_trackc_routed_distill_routes(cfg: Config) -> dict[str, str]:
    route_file_s = str(getattr(cfg, "trackc_routed_distill_route_file", "") or "").strip()
    if not route_file_s:
        raise ValueError("trackc_routed_distill_loss_weight > 0 requires trackc_routed_distill_route_file")
    route_path = Path(route_file_s).expanduser()
    if not route_path.is_absolute():
        route_path = route_path.resolve()
    if not route_path.is_file():
        raise FileNotFoundError(f"trackc_routed_distill_route_file not found: {route_path}")
    with route_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    routes = payload.get("route") or payload.get("routes") or payload
    if not isinstance(routes, dict):
        raise ValueError(f"trackc routed route file must contain a route dict: {route_path}")
    out = {str(ds): str(route).strip() for ds, route in routes.items() if str(route).strip()}
    allowed = {"additive_single_sum", "dataset_multi_mean", "train_multi_memory"}
    bad = {ds: route for ds, route in out.items() if route not in allowed}
    if bad:
        raise ValueError(f"unsupported Track C routes in {route_path}: {bad}")
    if not out:
        raise ValueError(f"empty Track C route mapping: {route_path}")
    return out


def _trackc_routed_distill_source_dataset(
    dataset: CrossDatasetFMDataset,
    cfg: Config,
    *,
    log=None,
) -> tuple[CrossDatasetFMDataset, str, str]:
    """Return the dataset used to build Track C routed teacher banks."""
    split_path_s = str(getattr(cfg, "trackc_routed_distill_bank_split_file", "") or "").strip()
    if not split_path_s:
        return dataset, "training_split", ""
    split_path = Path(split_path_s).expanduser()
    if not split_path.is_absolute():
        split_path = split_path.resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"trackc_routed_distill_bank_split_file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as handle:
        source_split = json.load(handle)
    if not isinstance(source_split, dict):
        raise ValueError(f"trackc_routed_distill_bank_split_file is not a split dict: {split_path}")
    if log is not None:
        log(f"Track C routed distill bank split: {split_path}")
    source_ds = CrossDatasetFMDataset(
        data_dir=str(dataset.data_dir),
        split=source_split,
        batch_size=int(dataset.batch_size),
        seed=int(dataset.seed),
        mode="train",
        min_cells=int(dataset.min_cells),
        ds_alpha=1.0,
        scale_noise=0.0,
        min_selected_conditions_per_dataset=0,
        condition_visit_power=1.0,
        condition_visit_cap=0,
        use_pert_condition=True,
        max_pert_genes=int(dataset.max_pert_genes),
        gene_embedding_cache_dir=str(getattr(cfg, "pert_gene_emb_cache_dir", "") or ""),
        biflow_dir=str(getattr(cfg, "biflow_dir", "") or ""),
        use_h5ad_pert_metadata=bool(getattr(cfg, "use_h5ad_pert_metadata", False)),
        pert_metainfo_path=str(getattr(cfg, "pert_metainfo_path", "") or ""),
        chem_emb_source_dir=str(getattr(cfg, "chem_emb_source_dir", "") or ""),
        chem_obs_column=str(getattr(cfg, "chem_obs_column", "") or ""),
        drug_emb_cache_dir=str(getattr(cfg, "drug_emb_cache_dir", "") or ""),
        max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
        chemical_metainfo_path=str(getattr(cfg, "chemical_metainfo_path", "") or ""),
        chem_fallback_embed_dim=int(getattr(cfg, "chem_fallback_embed_dim", 512)),
        latent_backbone=str(getattr(cfg, "latent_backbone", "state") or "state"),
        pert_chem_enabled=bool(getattr(cfg, "pert_chem_enabled", False)),
        perturbation_family_filter=str(getattr(cfg, "perturbation_family_filter", "all") or "all"),
        silent=True,
    )
    return source_ds, "bank_split_file", str(split_path)


def _multi_gene_composition_key(meta: ConditionMetadata) -> Optional[tuple[str, ...]]:
    genes = tuple(sorted({str(g).strip().upper() for g in getattr(meta, "genes", ()) if str(g).strip()}))
    if len(genes) < 2:
        return None
    if int(meta.resolved_nperts()) < 2:
        return None
    if getattr(meta, "chem_emb", None) is not None or getattr(meta, "chem_emb_list", None):
        return None
    return genes


def _gene_match_score(a: tuple[str, ...], b: tuple[str, ...], mode: str) -> float:
    aset = {str(g).strip().upper() for g in a if str(g).strip()}
    bset = {str(g).strip().upper() for g in b if str(g).strip()}
    if not aset or not bset:
        return 0.0
    mode_l = str(mode).strip().lower()
    if mode_l == "jaccard":
        return float(len(aset & bset)) / float(len(aset | bset))
    if mode_l == "overlap":
        return float(len(aset & bset))
    raise ValueError(f"unsupported Track C memory mode: {mode!r}")


def build_trackc_routed_distill_bank(
    dataset: CrossDatasetFMDataset,
    cfg: Config,
    *,
    log=None,
) -> dict[str, Any]:
    """Build train-only support route teacher bank for Track C.

    The active ``dataset`` is the training split supplied to ``train.py``. This
    function deliberately reads only ``dataset.ds_conds`` in train mode, so
    final query conditions remain inaccessible when callers use the
    train-selection split.
    """
    if (
        trackc_routed_distill_loss_schedule(10**12, cfg) <= 0
        and trackc_routed_endpoint_loss_schedule(10**12, cfg) <= 0
        and not _support_context_source_active(cfg)
    ):
        return {}
    if dataset.gene_embedding_cache is None:
        raise ValueError("Track C routed distillation requires use_pert_condition=True")
    target_frame = str(getattr(cfg, "trackc_routed_distill_target_frame", "endpoint_delta") or "endpoint_delta")
    target_frame = target_frame.strip().lower()
    if target_frame != "endpoint_delta":
        raise ValueError("trackc_routed_distill_target_frame currently supports only endpoint_delta")
    routes = _load_trackc_routed_distill_routes(cfg)
    max_cells = int(getattr(cfg, "condition_prior_bank_max_cells", 512) or 512)
    min_norm = float(getattr(cfg, "condition_prior_bank_min_norm", 1e-6) or 1e-6)

    source_dataset, bank_source, bank_split_file = _trackc_routed_distill_source_dataset(
        dataset,
        cfg,
        log=log,
    )

    single_by_ds_gene: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    single_global_gene: dict[str, list[torch.Tensor]] = defaultdict(list)
    multi_by_ds: dict[str, list[torch.Tensor]] = defaultdict(list)
    multi_memory_by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for ds_name in source_dataset.ds_names:
        handle = source_dataset.handles[ds_name]
        for cond in source_dataset.ds_conds.get(ds_name, []):
            meta = source_dataset.metadata_for_condition(ds_name, cond)
            single_key = _single_gene_composition_key(meta)
            multi_key = _multi_gene_composition_key(meta)
            if single_key is None and multi_key is None:
                skipped += 1
                continue
            seed_base = _stable_int_hash(f"trackc_route:{ds_name}:{cond}")
            src_mean = _sample_rows_mean(handle.read_src(cond), max_cells=max_cells, seed=seed_base)
            gt_mean = _sample_rows_mean(handle.read_gt(cond), max_cells=max_cells, seed=seed_base + 17)
            delta = (gt_mean - src_mean).float()
            if not torch.isfinite(delta).all() or delta.norm().item() <= min_norm:
                skipped += 1
                continue
            if single_key is not None:
                gene = str(single_key[0]).strip().upper()
                single_by_ds_gene[str(ds_name)][gene].append(delta.cpu())
                single_global_gene[gene].append(delta.cpu())
            elif multi_key is not None:
                multi_by_ds[str(ds_name)].append(delta.cpu())
                multi_memory_by_ds[str(ds_name)].append(
                    {
                        "dataset": str(ds_name),
                        "condition": str(cond),
                        "genes": tuple(multi_key),
                        "delta": delta.cpu(),
                    }
                )

    gene_mean_by_ds: dict[str, dict[str, torch.Tensor]] = {}
    for ds_name, by_gene in single_by_ds_gene.items():
        gene_mean_by_ds[ds_name] = {
            gene: torch.stack(vals, dim=0).mean(dim=0).float().cpu()
            for gene, vals in by_gene.items()
            if vals
        }
    global_gene_mean = {
        gene: torch.stack(vals, dim=0).mean(dim=0).float().cpu()
        for gene, vals in single_global_gene.items()
        if vals
    }
    dataset_multi_mean = {
        ds_name: torch.stack(vals, dim=0).mean(dim=0).float().cpu()
        for ds_name, vals in multi_by_ds.items()
        if vals
    }
    bank = {
        "routes": routes,
        "gene_mean_by_dataset": gene_mean_by_ds,
        "global_gene_mean": global_gene_mean,
        "dataset_multi_mean": dataset_multi_mean,
        "multi_memory_by_dataset": dict(multi_memory_by_ds),
        "memory_mode": str(getattr(cfg, "trackc_routed_distill_memory_mode", "off") or "off").strip().lower(),
        "memory_k": int(getattr(cfg, "trackc_routed_distill_memory_k", 3) or 3),
        "memory_min_score": float(getattr(cfg, "trackc_routed_distill_memory_min_score", 0.25) or 0.0),
        "memory_scope": str(
            getattr(cfg, "trackc_routed_distill_memory_scope", "same_dataset") or "same_dataset"
        ).strip().lower(),
    }

    global LAST_TRACKC_ROUTED_DISTILL_SUMMARY
    LAST_TRACKC_ROUTED_DISTILL_SUMMARY = {
        "route_file": str(Path(getattr(cfg, "trackc_routed_distill_route_file", "")).expanduser()),
        "bank_source": bank_source,
        "bank_split_file": bank_split_file,
        "bank_source_train_conditions": int(source_dataset.total_conditions),
        "target_frame": target_frame,
        "max_cells": max_cells,
        "min_norm": min_norm,
        "routes": routes,
        "skipped_records": skipped,
        "single_genes_by_dataset": {ds: len(vals) for ds, vals in sorted(gene_mean_by_ds.items())},
        "global_single_genes": len(global_gene_mean),
        "train_multi_records_by_dataset": {ds: len(vals) for ds, vals in sorted(multi_by_ds.items())},
        "train_multi_memory_records_by_dataset": {
            ds: len(vals) for ds, vals in sorted(multi_memory_by_ds.items())
        },
        "dataset_multi_mean_available": sorted(dataset_multi_mean),
        "memory_mode": bank["memory_mode"],
        "memory_k": bank["memory_k"],
        "memory_min_score": bank["memory_min_score"],
        "memory_scope": bank["memory_scope"],
    }
    if log is not None:
        log(
            "Track C routed distill bank: "
            f"source={bank_source} "
            f"routes={routes} single_ds={LAST_TRACKC_ROUTED_DISTILL_SUMMARY['single_genes_by_dataset']} "
            f"global_single_genes={len(global_gene_mean)} multi_ds={sorted(dataset_multi_mean)} "
            f"skipped={skipped}"
        )
    return bank


def get_trackc_routed_distill_target(
    bank: dict[str, Any],
    ds_name: str,
    meta: ConditionMetadata,
) -> Optional[torch.Tensor]:
    """Return CPU endpoint-delta teacher target for one real multi condition."""
    if not bank:
        return None
    ds = str(ds_name)
    route = (bank.get("routes") or {}).get(ds)
    if not route:
        return None
    genes = _multi_gene_composition_key(meta)
    if genes is None:
        return None
    if route == "dataset_multi_mean":
        target = (bank.get("dataset_multi_mean") or {}).get(ds)
        return None if target is None else target.float().cpu()
    if route == "train_multi_memory":
        mode = str(bank.get("memory_mode") or "off").strip().lower()
        if mode == "off":
            raise ValueError("Track C train_multi_memory route requires memory_mode != off")
        scope = str(bank.get("memory_scope") or "same_dataset").strip().lower()
        if scope not in {"same_dataset", "all_dataset"}:
            raise ValueError(f"unsupported Track C memory scope: {scope!r}")
        k = max(int(bank.get("memory_k") or 1), 1)
        min_score = float(bank.get("memory_min_score") or 0.0)
        by_dataset = bank.get("multi_memory_by_dataset") or {}
        records = []
        if scope == "same_dataset":
            records.extend(by_dataset.get(ds) or [])
        else:
            for vals in by_dataset.values():
                records.extend(vals or [])
        candidates = []
        for row in records:
            row_genes = tuple(row.get("genes") or ())
            score = _gene_match_score(genes, row_genes, mode)
            if score >= min_score:
                candidates.append((score, row))
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (item[0], str(item[1].get("dataset", "")), str(item[1].get("condition", ""))),
            reverse=True,
        )
        selected = candidates[:k]
        weights = torch.tensor(
            [max(float(score), 1e-6) for score, _ in selected],
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        vals = torch.stack([row["delta"].float().cpu() for _, row in selected], dim=0)
        return (weights[:, None] * vals).sum(dim=0).float().cpu()
    if route == "additive_single_sum":
        by_ds = (bank.get("gene_mean_by_dataset") or {}).get(ds) or {}
        global_by_gene = bank.get("global_gene_mean") or {}
        vals = []
        for gene in genes:
            val = by_ds.get(gene)
            if val is None:
                val = global_by_gene.get(gene)
            if val is None:
                return None
            vals.append(val.float().cpu())
        return torch.stack(vals, dim=0).sum(dim=0).float().cpu()
    raise ValueError(f"unsupported Track C route: {route!r}")


def configure_condition_delta_prior_gate(
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    bank: dict[str, list[tuple[str, Optional[str], torch.Tensor]]],
    cfg: Config,
    *,
    log=None,
) -> None:
    """Install train-prior gene allowlist for gated condition-delta injection."""
    mode = str(getattr(cfg, "condition_delta_in_model_filter", "all") or "all").strip().lower()
    if mode not in {"prior_covered_gene_multi", "allowlisted_gene_single"}:
        return
    inner = _unwrap_model(model)
    setter = getattr(inner, "set_condition_delta_prior_gene_ids", None)
    if setter is None:
        raise RuntimeError("condition_delta_in_model_filter requires model allowlist setter")
    cache = dataset.gene_embedding_cache
    if cache is None:
        raise RuntimeError(f"{mode} requires dataset.gene_embedding_cache")
    gene_file = str(getattr(cfg, "condition_delta_allowlist_gene_file", "") or "").strip()
    genes: list[str]
    if gene_file:
        gene_path = Path(gene_file)
        if not gene_path.is_absolute():
            gene_path = Path(cfg.save_dir).resolve() / gene_file
        if not gene_path.exists():
            raise FileNotFoundError(f"condition_delta_allowlist_gene_file not found: {gene_path}")
        parsed: list[str] = []
        for lineno, line in enumerate(gene_path.read_text(encoding="utf-8").splitlines(), start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in raw.replace("\t", ",").split(",")]
            if lineno == 1 and parts and parts[0].lower() in {"gene", "genes", "target"}:
                continue
            gene = parts[0].strip().upper()
            if gene:
                parsed.append(gene)
        genes = sorted(set(parsed))
        if not genes:
            raise ValueError(f"condition_delta_allowlist_gene_file has no genes: {gene_path}")
    elif mode == "prior_covered_gene_multi":
        genes = sorted({str(gene).strip().upper() for rows in bank.values() for gene, _ptype, _delta in rows})
    else:
        raise ValueError("allowlisted_gene_single requires condition_delta_allowlist_gene_file")
    ids = [cache.lookup(gene) for gene in genes]
    ids = [idx for idx in ids if idx not in {int(cache.pad_index), int(cache.unk_index)}]
    setter(ids)
    if log is not None:
        log(
            "Condition-delta prior gate: "
            f"mode={mode} genes={len(genes)} usable_gene_ids={len(set(ids))} "
            f"gene_file={gene_file or 'condition_prior_bank'}"
        )


def sample_condition_prior_teacher(
    *,
    bank: dict[str, list[tuple[str, Optional[str], torch.Tensor]]],
    ds_name: str,
    step: int,
    cond: str,
    batch_size: int,
    cache: GeneEmbeddingCache,
    max_genes: int,
    max_chem_keys: int,
    num_genes: int,
) -> tuple[Optional[torch.Tensor], Optional[tuple]]:
    """Sample a deterministic synthetic multi-gene prior target for one step."""
    records = bank.get(ds_name) or bank.get("__global__") or []
    n_genes = max(1, int(num_genes))
    if len(records) < n_genes:
        return None, None
    base = _stable_int_hash(f"condition_prior:{step}:{ds_name}:{cond}")
    picked: list[tuple[str, Optional[str], torch.Tensor]] = []
    used: set[str] = set()
    for off in range(len(records) * 2):
        rec = records[(base + off * 7919) % len(records)]
        if rec[0] in used:
            continue
        picked.append(rec)
        used.add(rec[0])
        if len(picked) >= n_genes:
            break
    if len(picked) < n_genes:
        return None, None
    genes = tuple(rec[0] for rec in picked)
    ptypes = [rec[1] for rec in picked if rec[1] is not None]
    ptype = ptypes[0] if ptypes and all(p == ptypes[0] for p in ptypes) else None
    target = torch.stack([rec[2] for rec in picked], dim=0).sum(dim=0).float()
    if not torch.isfinite(target).all():
        return None, None
    pb = _make_gene_combo_perturbation_batch(
        genes=genes,
        perturbation_type_raw=ptype,
        batch_size=int(batch_size),
        cache=cache,
        max_genes=int(max_genes),
        max_chem_keys=int(max_chem_keys),
    )
    return target, pb


def _fm_ckpt_velocity_pert(m, x, t, s, gid, mk, tid, npt, cid, chem_emb, chem_mask):
    return m(
        x,
        t,
        s,
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=chem_emb,
        chem_mask=chem_mask,
    )


def _model_uses_support_context(model: torch.nn.Module) -> bool:
    """Read support-context flag from inner module when DDP-wrapped."""
    inner = _unwrap_model(model)
    return bool(
        getattr(inner, "trackc_support_context_use_in_model", False)
        or getattr(inner, "trackc_support_residual_use_in_model", False)
        or getattr(inner, "trackc_support_film_use_in_model", False)
    )


def _model_uses_support_set_task(model: torch.nn.Module) -> bool:
    """Read support-set task-adapter flag from inner module when DDP-wrapped."""
    inner = _unwrap_model(model)
    return bool(getattr(inner, "trackc_support_set_task_use_in_model", False))


def _model_support_context_dim(model: torch.nn.Module) -> int:
    return int(getattr(_unwrap_model(model), "trackc_support_context_dim", 0) or 0)


def _zero_support_context_for(
    model: torch.nn.Module,
    x_t: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not _model_uses_support_context(model):
        return None
    dim = _model_support_context_dim(model)
    if dim <= 0:
        raise RuntimeError("support-context model has non-positive context dimension")
    return x_t.new_zeros((int(x_t.shape[0]), dim))


def _support_context_source_active(cfg: Config) -> bool:
    source = str(getattr(cfg, "trackc_support_context_source", "off") or "off").strip().lower()
    uses_support = bool(
        getattr(cfg, "trackc_support_context_use_in_model", False)
        or getattr(cfg, "trackc_support_residual_use_in_model", False)
        or getattr(cfg, "trackc_support_film_use_in_model", False)
    )
    return uses_support and source not in {"", "off"}


def validate_support_context_config(cfg: Config) -> None:
    if not bool(
        getattr(cfg, "trackc_support_context_use_in_model", False)
        or getattr(cfg, "trackc_support_residual_use_in_model", False)
        or getattr(cfg, "trackc_support_film_use_in_model", False)
    ):
        return
    source = str(getattr(cfg, "trackc_support_context_source", "off") or "off").strip().lower()
    if source != "routed_distill_target":
        raise ValueError(
            "Track C support context/residual paths require "
            "trackc_support_context_source='routed_distill_target' for train/eval runs"
        )
    if int(getattr(cfg, "trackc_support_context_dim", 0) or 0) != int(getattr(cfg, "emb_dim", 0) or 0):
        raise ValueError(
            "Track C support context/residual paths with routed_distill_target require "
            "trackc_support_context_dim == emb_dim"
        )
    split_s = str(getattr(cfg, "trackc_routed_distill_bank_split_file", "") or "").strip()
    if not split_s:
        raise ValueError(
            "support-context routed source requires trackc_routed_distill_bank_split_file "
            "so eval/posthoc cannot build context from the active evaluation split"
        )
    pair_filter = str(getattr(cfg, "trackc_support_context_pair_type_filter", "off") or "off").strip().lower()
    if pair_filter not in {"", "off", "none_train_single", "both_train_multi_gene", "none_train_single_both_train_multi_gene"}:
        raise ValueError(f"unsupported trackc_support_context_pair_type_filter: {pair_filter!r}")


@lru_cache(maxsize=8)
def _load_trackc_support_pair_filter_split(split_file: str) -> dict[str, Any]:
    with open(split_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pair_genes_from_condition(cond: str) -> Optional[tuple[str, str]]:
    parts = [part.strip() for part in str(cond).split("+") if part.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _trackc_support_context_pair_type_allowed(cfg: Config, ds_name: str, cond: str) -> bool:
    pair_filter = str(getattr(cfg, "trackc_support_context_pair_type_filter", "off") or "off").strip().lower()
    if pair_filter in {"", "off"}:
        return True
    pair = _pair_genes_from_condition(cond)
    if pair is None:
        return False
    split_s = str(getattr(cfg, "trackc_routed_distill_bank_split_file", "") or "").strip()
    if not split_s:
        raise ValueError("pair-type support-context filter requires trackc_routed_distill_bank_split_file")
    split = _load_trackc_support_pair_filter_split(split_s)
    groups = split.get(str(ds_name)) or {}
    train_single = {str(x) for x in groups.get("train_single") or []}
    train_multi_genes: set[str] = set()
    for item in groups.get("train_multi") or []:
        item_pair = _pair_genes_from_condition(str(item))
        if item_pair is not None:
            train_multi_genes.update(item_pair)
    single_cov = sum(g in train_single for g in pair)
    multi_gene_cov = sum(g in train_multi_genes for g in pair)
    if pair_filter == "none_train_single":
        return single_cov == 0
    if pair_filter == "both_train_multi_gene":
        return multi_gene_cov == 2
    if pair_filter == "none_train_single_both_train_multi_gene":
        return single_cov == 0 and multi_gene_cov == 2
    raise ValueError(f"unsupported trackc_support_context_pair_type_filter: {pair_filter!r}")


def make_trackc_support_context_batch(
    bank: dict[str, Any],
    dataset: CrossDatasetFMDataset,
    ds_name: str,
    cond: str,
    batch_size: int,
    cfg: Config,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not _support_context_source_active(cfg):
        return None
    source = str(getattr(cfg, "trackc_support_context_source", "off") or "off").strip().lower()
    if source != "routed_distill_target":
        raise ValueError(f"unsupported trackc_support_context_source: {source!r}")
    if not bank:
        raise RuntimeError("support-context source requested but routed context bank is empty")
    dim = int(getattr(cfg, "trackc_support_context_dim", 0) or 0)
    if not _trackc_support_context_pair_type_allowed(cfg, ds_name, cond):
        return torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32)
    meta = dataset.metadata_for_condition(ds_name, cond)
    target = get_trackc_routed_distill_target(bank, ds_name, meta)
    if target is None:
        return torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    if target.ndim != 1 or int(target.numel()) != dim:
        raise RuntimeError(
            "support-context routed target has wrong shape: "
            f"{tuple(target.shape)} for dim={dim}"
        )
    if not torch.isfinite(target).all():
        raise RuntimeError("support-context routed target contains non-finite values")
    return target.unsqueeze(0).expand(int(batch_size), -1).contiguous()


def _support_context_eval_control_mode(cfg: Config) -> str:
    mode = str(getattr(cfg, "trackc_support_context_eval_control", "actual") or "actual").strip().lower()
    if mode not in {"actual", "zero", "shuffle_condition"}:
        raise ValueError(f"unsupported trackc_support_context_eval_control: {mode!r}")
    return mode


def _build_trackc_support_context_shuffle_targets(
    bank: dict[str, Any],
    dataset: CrossDatasetFMDataset,
    cond_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], torch.Tensor]:
    """Deterministic condition-level context shuffle for query-free controls."""
    rows: list[tuple[tuple[str, str], torch.Tensor]] = []
    for ds_name, cond in cond_pairs:
        target = get_trackc_routed_distill_target(bank, ds_name, dataset.metadata_for_condition(ds_name, cond))
        if target is not None:
            rows.append(((str(ds_name), str(cond)), target.float().cpu()))
    if len(rows) < 2:
        return {}
    shifted: dict[tuple[str, str], torch.Tensor] = {}
    for idx, (key, _target) in enumerate(rows):
        shifted[key] = rows[(idx + 1) % len(rows)][1]
    return shifted


def _apply_trackc_support_context_eval_control(
    support_context: Optional[torch.Tensor],
    *,
    cfg: Config,
    ds_name: str,
    cond: str,
    batch_size: int,
    device: torch.device,
    shuffle_targets: Optional[dict[tuple[str, str], torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    mode = _support_context_eval_control_mode(cfg)
    if mode == "actual":
        return support_context
    dim = int(getattr(cfg, "trackc_support_context_dim", 0) or 0)
    if mode == "zero":
        return torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32)
    if not _trackc_support_context_pair_type_allowed(cfg, ds_name, cond):
        return torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32)
    target = (shuffle_targets or {}).get((str(ds_name), str(cond)))
    if target is None:
        return torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    if target.ndim != 1 or int(target.numel()) != dim:
        raise RuntimeError(
            "shuffled support-context target has wrong shape: "
            f"{tuple(target.shape)} for dim={dim}"
        )
    return target.unsqueeze(0).expand(int(batch_size), -1).contiguous()


LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY: dict[str, Any] = {}


def _support_set_task_source_active(cfg: Config) -> bool:
    source = str(getattr(cfg, "trackc_support_set_task_source", "off") or "off").strip().lower()
    return bool(getattr(cfg, "trackc_support_set_task_use_in_model", False)) and source not in {"", "off"}


def _support_set_task_eval_control_mode(cfg: Config) -> str:
    mode = str(getattr(cfg, "trackc_support_set_task_eval_control", "actual") or "actual").strip().lower()
    if mode not in {"actual", "zero", "shuffle_condition", "absent"}:
        raise ValueError(f"unsupported trackc_support_set_task_eval_control: {mode!r}")
    return mode


def validate_support_set_task_config(cfg: Config) -> None:
    if not bool(getattr(cfg, "trackc_support_set_task_use_in_model", False)):
        return
    source = str(getattr(cfg, "trackc_support_set_task_source", "off") or "off").strip().lower()
    if source in {"", "off"}:
        return
    if source != "shared_gene_condition_means":
        raise ValueError(f"unsupported trackc_support_set_task_source: {source!r}")
    if int(getattr(cfg, "trackc_support_set_task_dim", 0) or 0) != int(getattr(cfg, "emb_dim", 0) or 0):
        raise ValueError(
            "Track C support-set task source requires trackc_support_set_task_dim == emb_dim"
        )
    split_s = str(getattr(cfg, "trackc_support_set_task_safe_split_file", "") or "").strip()
    if not split_s:
        raise ValueError("trackc_support_set_task_safe_split_file is required for support-set sources")
    split_path = Path(split_s).expanduser()
    if not split_path.is_file():
        raise FileNotFoundError(f"trackc_support_set_task_safe_split_file not found: {split_path}")
    if split_path.name == "split_seed42_multi_support_v2.json":
        raise ValueError("full Track C v2 query split is forbidden for support-set task selection/source")
    for attr in ("trackc_support_set_task_anchor_condition_means", "trackc_support_set_task_candidate_condition_means"):
        path_s = str(getattr(cfg, attr, "") or "").strip()
        if not path_s:
            raise ValueError(f"{attr} is required for shared_gene_condition_means source")
        path = Path(path_s).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{attr} not found: {path}")
    if int(getattr(cfg, "trackc_support_set_task_min_support_count", 1) or 1) < 1:
        raise ValueError("trackc_support_set_task_min_support_count must be >= 1")
    _support_set_task_eval_control_mode(cfg)


def _load_trackc_condition_mean_payload(path_s: str) -> dict[str, Any]:
    path = Path(path_s).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _condition_mean_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def _row_vector(row: dict[str, Any], key: str, *, row_id: str) -> torch.Tensor:
    value = row.get(key)
    if value is None:
        raise ValueError(f"condition-mean row missing {key}: {row_id}")
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 1 or not torch.isfinite(tensor).all():
        raise ValueError(f"condition-mean row has invalid {key}: {row_id}")
    return tensor


def build_trackc_support_set_task_bank(cfg: Config, *, log=None) -> dict[str, list[dict[str, Any]]]:
    """Build query-conditioned support-set tokens from safe trainselect artifacts.

    Records are train_multi residuals from candidate minus anchor condition means.
    Query-time token construction always excludes the query condition itself, so
    train_multi rows cannot see their own target residual.
    """
    if not _support_set_task_source_active(cfg):
        return {}
    validate_support_set_task_config(cfg)
    source = str(getattr(cfg, "trackc_support_set_task_source", "off") or "off").strip().lower()
    if source != "shared_gene_condition_means":
        raise ValueError(f"unsupported trackc_support_set_task_source: {source!r}")

    safe_split = str(Path(str(getattr(cfg, "trackc_support_set_task_safe_split_file"))).expanduser().resolve())
    anchor = _load_trackc_condition_mean_payload(str(getattr(cfg, "trackc_support_set_task_anchor_condition_means")))
    candidate = _load_trackc_condition_mean_payload(str(getattr(cfg, "trackc_support_set_task_candidate_condition_means")))
    for role, payload in (("anchor", anchor), ("candidate", candidate)):
        split_file = str(payload.get("split_file") or "")
        if not split_file:
            raise ValueError(f"{role} condition-mean artifact is missing split_file")
        if str(Path(split_file).expanduser().resolve()) != safe_split:
            raise ValueError(
                f"{role} condition-mean split mismatch: {split_file} != {safe_split}"
            )
        if Path(split_file).name == "split_seed42_multi_support_v2.json":
            raise ValueError("full Track C v2 query split is forbidden for support-set task bank")

    anchor_rows = {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in _condition_mean_rows(anchor, "train_multi")
    }
    candidate_rows = {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in _condition_mean_rows(candidate, "train_multi")
    }
    if set(anchor_rows) != set(candidate_rows):
        raise ValueError("anchor/candidate train_multi condition-mean rows do not match")

    dim = int(getattr(cfg, "trackc_support_set_task_dim", 0) or 0)
    bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for key in sorted(anchor_rows):
        ds_name, cond = key
        genes = _pair_genes_from_condition(cond)
        if genes is None:
            skipped += 1
            continue
        row_id = f"{ds_name}:{cond}"
        a_pred = _row_vector(anchor_rows[key], "pred_mean", row_id=row_id)
        c_pred = _row_vector(candidate_rows[key], "pred_mean", row_id=row_id)
        residual = c_pred - a_pred
        if int(residual.numel()) != dim:
            raise ValueError(f"support-set residual dim mismatch for {row_id}: {residual.numel()} != {dim}")
        if not torch.isfinite(residual).all():
            raise ValueError(f"support-set residual contains non-finite values: {row_id}")
        bank[ds_name].append(
            {
                "condition": cond,
                "genes": tuple(str(g).upper() for g in genes),
                "residual": residual.cpu(),
            }
        )

    global LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY
    LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY = {
        "source": source,
        "safe_split_file": safe_split,
        "anchor_condition_means": str(Path(str(getattr(cfg, "trackc_support_set_task_anchor_condition_means"))).expanduser()),
        "candidate_condition_means": str(Path(str(getattr(cfg, "trackc_support_set_task_candidate_condition_means"))).expanduser()),
        "datasets": {ds: len(rows) for ds, rows in sorted(bank.items())},
        "records": sum(len(rows) for rows in bank.values()),
        "skipped_non_pair": skipped,
        "dim": dim,
        "policy": "same_dataset_shared_gene_train_multi_residual_excluding_query_condition",
        "min_support_count": int(getattr(cfg, "trackc_support_set_task_min_support_count", 1) or 1),
    }
    if log is not None:
        log(
            "Track C support-set task bank: "
            f"source={source} records={LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY['records']} "
            f"datasets={len(bank)} dim={dim} skipped={skipped}"
        )
        for ds_name in sorted(bank):
            log(f"  support_set_task_bank[{ds_name}] = {len(bank[ds_name])}")
    return dict(bank)


def _support_set_task_token_for(
    bank: dict[str, list[dict[str, Any]]],
    ds_name: str,
    cond: str,
    cfg: Config,
) -> tuple[Optional[torch.Tensor], bool]:
    genes = _pair_genes_from_condition(cond)
    dim = int(getattr(cfg, "trackc_support_set_task_dim", 0) or 0)
    if genes is None or dim <= 0:
        return None, False
    qgenes = {str(g).upper() for g in genes}
    rows = []
    for row in bank.get(str(ds_name), []):
        if str(row.get("condition")) == str(cond):
            continue
        rgenes = {str(g).upper() for g in row.get("genes", ())}
        if qgenes & rgenes:
            rows.append(row["residual"].float())
    min_support = int(getattr(cfg, "trackc_support_set_task_min_support_count", 1) or 1)
    if len(rows) < min_support:
        return None, False
    token = torch.stack(rows, dim=0).mean(dim=0)
    token = token * float(getattr(cfg, "trackc_support_set_task_scale", 1.0) or 1.0)
    if int(token.numel()) != dim or not torch.isfinite(token).all():
        raise RuntimeError(f"invalid support-set task token for {ds_name}:{cond}")
    return token.cpu(), True


def make_trackc_support_set_task_batch(
    bank: dict[str, list[dict[str, Any]]],
    ds_name: str,
    cond: str,
    batch_size: int,
    cfg: Config,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not _support_set_task_source_active(cfg):
        return None, None
    if not bank:
        raise RuntimeError("support-set task source requested but bank is empty")
    dim = int(getattr(cfg, "trackc_support_set_task_dim", 0) or 0)
    token, present = _support_set_task_token_for(bank, ds_name, cond, cfg)
    if token is None:
        task = torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32)
        mask = torch.zeros((int(batch_size), 1), device=device, dtype=torch.float32)
        return task, mask
    task = token.to(device=device, dtype=torch.float32).unsqueeze(0).expand(int(batch_size), -1).contiguous()
    mask = torch.full((int(batch_size), 1), 1.0 if present else 0.0, device=device, dtype=torch.float32)
    return task, mask


def _build_trackc_support_set_task_shuffle_targets(
    bank: dict[str, list[dict[str, Any]]],
    cond_pairs: list[tuple[str, str]],
    cfg: Config,
) -> dict[tuple[str, str], torch.Tensor]:
    rows: list[tuple[tuple[str, str], torch.Tensor]] = []
    for ds_name, cond in cond_pairs:
        token, present = _support_set_task_token_for(bank, ds_name, cond, cfg)
        if present and token is not None:
            rows.append(((str(ds_name), str(cond)), token.float().cpu()))
    if len(rows) < 2:
        return {}
    shifted: dict[tuple[str, str], torch.Tensor] = {}
    for idx, (key, _target) in enumerate(rows):
        shifted[key] = rows[(idx + 1) % len(rows)][1]
    return shifted


def _apply_trackc_support_set_task_eval_control(
    support_set_task: Optional[torch.Tensor],
    support_set_task_present: Optional[torch.Tensor],
    *,
    cfg: Config,
    ds_name: str,
    cond: str,
    batch_size: int,
    device: torch.device,
    shuffle_targets: Optional[dict[tuple[str, str], torch.Tensor]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not _support_set_task_source_active(cfg):
        return support_set_task, support_set_task_present
    mode = _support_set_task_eval_control_mode(cfg)
    if mode == "actual":
        return support_set_task, support_set_task_present
    dim = int(getattr(cfg, "trackc_support_set_task_dim", 0) or 0)
    if mode == "zero":
        return (
            torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32),
            torch.ones((int(batch_size), 1), device=device, dtype=torch.float32),
        )
    if mode == "absent":
        return (
            torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32),
            torch.zeros((int(batch_size), 1), device=device, dtype=torch.float32),
        )
    target = (shuffle_targets or {}).get((str(ds_name), str(cond)))
    if target is None:
        return (
            torch.zeros((int(batch_size), dim), device=device, dtype=torch.float32),
            torch.zeros((int(batch_size), 1), device=device, dtype=torch.float32),
        )
    target = target.to(device=device, dtype=torch.float32)
    if target.ndim != 1 or int(target.numel()) != dim:
        raise RuntimeError(f"shuffled support-set task target has wrong shape: {tuple(target.shape)}")
    return (
        target.unsqueeze(0).expand(int(batch_size), -1).contiguous(),
        torch.ones((int(batch_size), 1), device=device, dtype=torch.float32),
    )


def _model_latent_velocity(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    x_0: torch.Tensor,
    perturbation_batch: Optional[tuple],
    support_context: Optional[torch.Tensor] = None,
    support_context_present: Optional[torch.Tensor] = None,
    support_set_task: Optional[torch.Tensor] = None,
    support_set_task_present: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _model_uses_support_context(model) and support_context is None:
        support_context = _zero_support_context_for(model, x_t)
    if not _model_uses_pert(model):
        return model(
            x_t,
            t,
            x_0,
            support_context=support_context,
            support_context_present=support_context_present,
            support_set_task=support_set_task,
            support_set_task_present=support_set_task_present,
        )
    if perturbation_batch is None:
        raise ValueError("use_pert_condition model requires perturbation_batch on device")
    gid, mk, tid, npt, cid, chem_emb, chem_mask = _unpack_pert_up_to7(perturbation_batch)
    return model(
        x_t,
        t,
        x_0,
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=chem_emb,
        chem_mask=chem_mask,
        support_context=support_context,
        support_context_present=support_context_present,
        support_set_task=support_set_task,
        support_set_task_present=support_set_task_present,
    )


def _model_condition_delta(
    model: torch.nn.Module,
    perturbation_batch: tuple,
) -> Optional[torch.Tensor]:
    inner = _unwrap_model(model)
    if getattr(inner, "condition_delta_head", None) is None:
        return None
    gid, mk, tid, npt, cid, chem_emb, chem_mask = _unpack_pert_up_to7(perturbation_batch)
    return inner.predict_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=chem_emb,
        chem_mask=chem_mask,
    )


def _model_additive_condition_delta(
    model: torch.nn.Module,
    perturbation_batch: tuple,
) -> Optional[torch.Tensor]:
    inner = _unwrap_model(model)
    if getattr(inner, "condition_delta_head", None) is None:
        return None
    gid, mk, tid, npt, cid, chem_emb, chem_mask = _unpack_pert_up_to7(perturbation_batch)
    return inner.predict_additive_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=chem_emb,
        chem_mask=chem_mask,
    )


def apply_finetune_freeze(model: torch.nn.Module, cfg: Config) -> None:
    """Optionally freeze subsets of parameters for finetuning (warm-start)."""
    scope = str(getattr(cfg, "finetune_trainable_scope", "all") or "all").strip().lower()
    allowed_scopes = {
        "all",
        "type_adapter",
        "pairwise_adapter",
        "pairwise_condition_adapter",
        "condition_prior_adapter",
        "condition_lowrank_residual_adapter",
        "support_context_adapter",
        "support_residual_adapter",
        "support_film_adapter",
        "support_set_task_adapter",
    }
    if scope not in allowed_scopes:
        raise ValueError(
            "finetune_trainable_scope must be one of "
            "'all', 'type_adapter', 'pairwise_adapter', 'pairwise_condition_adapter', "
            "'condition_prior_adapter', 'condition_lowrank_residual_adapter', "
            "'support_context_adapter', or "
            "'support_residual_adapter', 'support_film_adapter', or "
            "'support_set_task_adapter'"
        )
    if scope == "type_adapter":
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = name.startswith("pert_encoder.type_")
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                "finetune_trainable_scope='type_adapter' requires "
                "perturbation type adapter parameters; enable use_pert_condition."
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope == "support_set_task_adapter":
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = name.startswith("support_set_task_to_c.")
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                "finetune_trainable_scope='support_set_task_adapter' requires "
                "trackc_support_set_task_use_in_model=True"
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope == "support_film_adapter":
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = name.startswith("support_context_to_v.") or name.startswith("support_context_to_v_scale.")
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                "finetune_trainable_scope='support_film_adapter' requires "
                "trackc_support_film_use_in_model=True"
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope == "support_residual_adapter":
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = name.startswith("support_context_to_v.")
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                "finetune_trainable_scope='support_residual_adapter' requires "
                "trackc_support_residual_use_in_model=True"
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope == "support_context_adapter":
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = name.startswith("support_context_to_c.")
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                "finetune_trainable_scope='support_context_adapter' requires "
                "trackc_support_context_use_in_model=True"
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope == "condition_prior_adapter":
        allowed_prefixes = ("condition_delta_head.", "condition_delta_to_c.")
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = any(name.startswith(prefix) for prefix in allowed_prefixes)
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        has_head = any(name.startswith("condition_delta_head.") for name in trainable_names)
        has_bridge = any(name.startswith("condition_delta_to_c.") for name in trainable_names)
        if not has_head or not has_bridge:
            raise RuntimeError(
                "finetune_trainable_scope='condition_prior_adapter' requires "
                "condition_delta_head and condition_delta_to_c parameters; enable "
                "condition_delta_head_use_in_model with a condition-prior loss."
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope == "condition_lowrank_residual_adapter":
        allowed_prefixes = ("condition_lowrank_residual_down.", "condition_lowrank_residual_up.")
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = any(name.startswith(prefix) for prefix in allowed_prefixes)
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        has_down = any(name.startswith("condition_lowrank_residual_down.") for name in trainable_names)
        has_up = any(name.startswith("condition_lowrank_residual_up.") for name in trainable_names)
        if not has_down or not has_up:
            raise RuntimeError(
                "finetune_trainable_scope='condition_lowrank_residual_adapter' requires "
                "condition_lowrank_residual_use_in_model=True"
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return
    if scope in {"pairwise_adapter", "pairwise_condition_adapter"}:
        if scope == "pairwise_adapter":
            allowed_prefixes = ("pert_encoder.pair_to_out.",)
        else:
            allowed_prefixes = (
                "pert_encoder.pair_to_out.",
                "pert_to_c.",
                "condition_delta_to_c.",
            )
        trainable_names: list[str] = []
        for name, param in model.named_parameters():
            keep = any(name.startswith(prefix) for prefix in allowed_prefixes)
            param.requires_grad = keep
            if keep:
                trainable_names.append(name)
        has_pairwise = any(name.startswith("pert_encoder.pair_to_out.") for name in trainable_names)
        if not has_pairwise:
            raise RuntimeError(
                f"finetune_trainable_scope={scope!r} found no "
                "pert_encoder.pair_to_out.* parameters; enable pert_pairwise_mode=hadamard_mean"
            )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[train] Finetune scope: {scope}; trainable tensors="
            f"{trainable_names}; params={n_trainable:,}",
            flush=True,
        )
        return

    if not getattr(cfg, "freeze_shared_enc", False):
        return
    if cfg.model_type != "control_mlp":
        print(
            "[train] WARNING: freeze_shared_enc is only supported for control_mlp; ignoring",
            flush=True,
        )
        return
    enc = getattr(model, "shared_enc", None)
    if enc is None:
        print("[train] WARNING: model has no shared_enc; freeze_shared_enc ignored", flush=True)
        return
    for p in enc.parameters():
        p.requires_grad = False
    print("[train] Finetune freeze: shared_enc parameters frozen", flush=True)


# Pretrained GeneEmbeddingTable lives at ``pert_encoder...gene_table...`` (see condition_emb.genepert).
_BACKBONE_SUBSTR = ("gene_table", "_emb_table")


def _is_backbone_name(name: str) -> bool:
    return any(s in name for s in _BACKBONE_SUBSTR)


def _split_params(model: torch.nn.Module):
    raw = model.module if hasattr(model, "module") else model
    bb, nw, bb_names, nw_names = [], [], [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        if _is_backbone_name(n):
            bb.append(p)
            bb_names.append(n)
        else:
            nw.append(p)
            nw_names.append(n)
    return bb, nw, bb_names, nw_names


def _build_latent_optimizer(model: torch.nn.Module, cfg: Config, *, rank: int, verbose: bool) -> torch.optim.Optimizer:
    if not getattr(cfg, "use_param_groups", False):
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters for optimizer")
        return torch.optim.AdamW(
            trainable,
            lr=cfg.lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

    bb, nw, bb_n, nw_n = _split_params(model)
    is_rank0 = rank == 0
    if verbose and is_rank0:
        n_bb = sum(p.numel() for p in bb)
        n_nw = sum(p.numel() for p in nw)
        print(
            f"[param_groups] backbone={len(bb_n)} tensors ({n_bb:,} params)  "
            f"new_modules={len(nw_n)} tensors ({n_nw:,} params)",
            flush=True,
        )

    if not bb:
        if not nw:
            raise RuntimeError("No trainable parameters for optimizer")
        if verbose and is_rank0:
            print("[param_groups] no gene_table backbone; single AdamW group", flush=True)
        return torch.optim.AdamW(nw, lr=cfg.lr, weight_decay=cfg.weight_decay_new)
    if not nw:
        return torch.optim.AdamW(bb, lr=cfg.lr, weight_decay=cfg.weight_decay_backbone)

    return torch.optim.AdamW(
        [
            {
                "params": bb,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay_backbone,
                "name": "backbone",
            },
            {
                "params": nw,
                "lr": cfg.lr * cfg.lr_new_module_mult,
                "weight_decay": cfg.weight_decay_new,
                "name": "new_modules",
            },
        ]
    )


def set_lr(optimizer, lr_val: float, cfg: Optional[Config] = None):
    if cfg is None or not getattr(cfg, "use_param_groups", False):
        for pg in optimizer.param_groups:
            pg["lr"] = lr_val
        return
    mult = float(getattr(cfg, "lr_new_module_mult", 1.0))
    backbone_lr = lr_val
    new_lr = lr_val * mult
    for pg in optimizer.param_groups:
        name = pg.get("name", "")
        if name == "backbone":
            pg["lr"] = backbone_lr
        elif name == "new_modules":
            pg["lr"] = new_lr
        else:
            pg["lr"] = lr_val


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _model_uses_pert(model: torch.nn.Module) -> bool:
    """Read ``use_pert_condition`` from the inner module when ``model`` is DDP-wrapped."""
    return bool(getattr(_unwrap_model(model), "use_pert_condition", False))


def save_checkpoint(path, model, optimizer, step, best_score, ema=None, config_dict=None):
    payload = {
        "step": step,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if config_dict is not None:
        payload["config"] = config_dict
    torch.save(payload, path)


def checkpoint_ema_is_active(ckpt: dict, cfg: Config) -> bool:
    """Return whether a checkpoint's EMA shadow has actually been updated."""
    if "ema" not in ckpt or not bool(getattr(cfg, "use_ema", False)):
        return False
    try:
        step = int(ckpt.get("step", 0) or 0)
    except (TypeError, ValueError):
        step = 0
    update_after = int(getattr(cfg, "ema_update_after", 0) or 0)
    meta = ckpt.get("ema", {}).get("__meta__") if isinstance(ckpt.get("ema"), dict) else None
    if meta is not None:
        try:
            num_updates = int(float(meta.detach().cpu().flatten()[3].item()))
            return num_updates > 0
        except Exception:
            pass
    return step >= update_after


def load_checkpoint(path, model, optimizer, device, ema=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        try:
            ema.load_state_dict(ckpt["ema"], strict=False)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[train] WARNING: failed to load EMA state ({e}); starting fresh EMA")
    best_score = ckpt.get("best_score", ckpt.get("best_loss", float("inf")))
    return ckpt.get("step", 0), best_score


def recover_best_score_from_best_checkpoint(
    *,
    latest_score: float,
    best_path: Path,
    metric_name: str,
    device: torch.device,
) -> float:
    """Prefer a finite score from ``best.pt`` when ``latest.pt`` has stale metadata."""
    try:
        latest = float(latest_score)
    except (TypeError, ValueError):
        latest = float("nan")
    if math.isfinite(latest):
        return latest
    if not best_path.is_file():
        return latest
    try:
        ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
        best = float(ckpt.get("best_score", ckpt.get("best_loss", latest)))
    except Exception as e:  # pragma: no cover - defensive
        print(f"[train] WARNING: failed to recover best_score from {best_path} ({e})")
        return latest
    if not math.isfinite(best):
        return latest
    print(
        f"[train] Recovered best_{metric_name}={best:.6f} from {best_path} "
        f"because latest.pt stored {latest_score!r}"
    )
    return best


def load_model_weights_only(
    path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    strict: bool = True,
    prefer_ema: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Load checkpoint weights only (no optimizer / step).

    By default this loads ``ckpt['model']`` to preserve legacy behavior. When
    ``prefer_ema`` is true and the checkpoint has an active EMA shadow, load
    matching ``shadow.*`` tensors instead. This is useful for anchor-preserving
    finetunes whose no-harm baseline is evaluated with EMA weights.
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint must be a dict with 'model' key: {path}")
    state = ckpt["model"]
    if prefer_ema and checkpoint_ema_is_active(ckpt, Config()):
        ema = ckpt.get("ema")
        if isinstance(ema, dict):
            ema_state = {
                key[len("shadow."):]: value
                for key, value in ema.items()
                if isinstance(key, str) and key.startswith("shadow.")
            }
            if ema_state:
                state = ema_state
    skipped_shape_mismatch: list[str] = []
    if not strict:
        current = model.state_dict()
        filtered = {}
        for key, value in state.items():
            if key in current and tuple(current[key].shape) != tuple(value.shape):
                skipped_shape_mismatch.append(
                    f"{key}: checkpoint{tuple(value.shape)} != model{tuple(current[key].shape)}"
                )
                continue
            filtered[key] = value
        state = filtered
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys), skipped_shape_mismatch


def selection_metric_direction(metric_name: str) -> str:
    if metric_name in {"test_mse", "test_mae", "test_mmd"}:
        return "min"
    if metric_name in {
        "direct_pearson",
        "pearson_ctrl",
        "pearson_pert",
        "pearson_pert_minus_mmd",
        "pearson_ctrl_minus_mmd",
    }:
        return "max"
    raise ValueError(f"Unsupported selection_metric: {metric_name}")


def selection_metric_value(metric_name: str, metrics: dict, *, mmd_lambda: float = 1.0) -> float:
    """Return a scalar model-selection score from an evaluation result dict."""
    if metric_name == "pearson_pert_minus_mmd":
        return float(metrics["pearson_pert"]) - float(mmd_lambda) * float(metrics["test_mmd"])
    if metric_name == "pearson_ctrl_minus_mmd":
        return float(metrics["pearson_ctrl"]) - float(mmd_lambda) * float(metrics["test_mmd"])
    return float(metrics[metric_name])


def dataset_loss_weights_active(step: int, cfg: Config) -> bool:
    """Whether per-dataset loss weights should be applied at this step."""
    if float(getattr(cfg, "ds_loss_alpha", 0.0) or 0.0) <= 0.0:
        return False
    warmup_start = int(getattr(cfg, "ds_loss_warmup_start", 0) or 0)
    return int(step) >= warmup_start


def is_better_score(metric_name: str, candidate: float, best_so_far: float) -> bool:
    direction = selection_metric_direction(metric_name)
    if direction == "min":
        return candidate < best_so_far
    return candidate > best_so_far


# ---------------------------------------------------------------------------
# OT prefetch pipeline
# ---------------------------------------------------------------------------

class OTPrefetchIter:
    """In-loop OT wrapper.

    - GPU 路径（``ot_method == "torch_sinkhorn"``，默认）：单线程即可，把 batch 上传到 GPU，
      调用 ``sinkhorn_pair`` 做配对，**彻底消除原本 CPU bound 的 ``pot.emd`` 线程池**。
    - CPU 路径（``ot_method in {"exact","sinkhorn"}``）：保留原多线程预取逻辑，仅作对照 / debug。
    """

    def __init__(self, dataset, ot_sampler: OTPlanSampler,
                 prefetch_n: int = 4, n_ot_workers: int = 3,
                 device: Optional[torch.device] = None,
                 pair_mode: str = "multinomial"):
        self.dataset = dataset
        self.ot_sampler = ot_sampler
        self.prefetch_n = prefetch_n
        self.n_ot_workers = n_ot_workers
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._gpu = getattr(ot_sampler, "method", "exact") == "torch_sinkhorn"
        self.pair_mode = str(pair_mode or "multinomial").strip().lower()
        if self.pair_mode not in {"multinomial", "assignment", "hungarian", "random"}:
            raise ValueError(
                "ot_pair_mode must be 'multinomial', 'assignment', 'hungarian', or 'random', "
                f"got {pair_mode!r}"
            )

    def __iter__(self):
        if self._gpu:
            yield from self._iter_gpu()
        else:
            yield from self._iter_cpu_threaded()

    # -------- GPU path (default) --------

    def _iter_gpu(self):
        from model.utils.data.ot_pairer import hungarian_pair, sinkhorn_pair
        reg = getattr(self.ot_sampler, "reg", 0.05)
        n_iter = getattr(self.ot_sampler, "n_iter", 50)
        raw_q: queue.Queue = queue.Queue(maxsize=max(1, int(self.prefetch_n)))
        sentinel = object()
        error_holder = [None]

        def _reader():
            try:
                for item in self.dataset:
                    raw_q.put(item)
            except Exception as e:
                error_holder[0] = e
            finally:
                raw_q.put(sentinel)

        reader_t = threading.Thread(target=_reader, daemon=True)
        reader_t.start()

        while True:
            item = raw_q.get()
            if item is sentinel:
                if error_holder[0] is not None:
                    raise error_holder[0]
                return
            if len(item) == 4:
                src_np, gt_np, ds_name, cond = item
                perturbation_batch = None
            else:
                src_np, gt_np, ds_name, cond, perturbation_batch = item
            src = src_np if isinstance(src_np, torch.Tensor) else torch.as_tensor(np.asarray(src_np))
            gt = gt_np if isinstance(gt_np, torch.Tensor) else torch.as_tensor(np.asarray(gt_np))
            src = src.to(self.device, dtype=torch.float32, non_blocking=True)
            gt = gt.to(self.device, dtype=torch.float32, non_blocking=True)
            perturbation_batch_gpu = _pert_to_device(perturbation_batch, self.device)
            if self.pair_mode == "random":
                yield src, gt, ds_name, cond, perturbation_batch_gpu
                continue
            n_pair = src.shape[0]
            if self.pair_mode == "hungarian":
                i, j = hungarian_pair(src, gt, n_samples=n_pair)
                yield src[i], gt[j], ds_name, cond, perturbation_batch_gpu
                continue
            i, j = sinkhorn_pair(
                src,
                gt,
                n_samples=n_pair,
                reg=reg,
                n_iter=n_iter,
                use_assignment=self.pair_mode == "assignment",
            )
            yield src[i], gt[j], ds_name, cond, perturbation_batch_gpu

    # -------- CPU path (legacy, threaded POT) --------

    def _iter_cpu_threaded(self):
        raw_q: queue.Queue = queue.Queue(maxsize=self.prefetch_n * 2)
        ready_q: queue.Queue = queue.Queue(maxsize=self.prefetch_n)
        _sentinel = object()
        _error_holder = [None]

        def _reader():
            try:
                for item in self.dataset:
                    if len(item) == 4:
                        src_np, gt_np, ds_name, cond = item
                        perturbation_batch = None
                    else:
                        src_np, gt_np, ds_name, cond, perturbation_batch = item
                    src_arr = src_np.numpy() if isinstance(src_np, torch.Tensor) else np.asarray(src_np)
                    gt_arr = gt_np.numpy() if isinstance(gt_np, torch.Tensor) else np.asarray(gt_np)
                    raw_q.put((src_arr, gt_arr, ds_name, cond, perturbation_batch))
            except Exception as e:
                _error_holder[0] = e
            finally:
                for _ in range(self.n_ot_workers):
                    raw_q.put(_sentinel)

        def _ot_worker():
            while True:
                item = raw_q.get()
                if item is _sentinel:
                    ready_q.put(_sentinel)
                    return
                src_arr, gt_arr, ds_name, cond, perturbation_batch = item
                try:
                    if self.pair_mode == "random":
                        src_paired, gt_paired = src_arr, gt_arr
                    elif self.pair_mode == "hungarian":
                        from model.utils.data.ot_pairer import assign_from_cost_hungarian, compute_ot_cost

                        src_t = torch.as_tensor(src_arr, dtype=torch.float32)
                        gt_t = torch.as_tensor(gt_arr, dtype=torch.float32)
                        cost = compute_ot_cost(src_t, gt_t)
                        i, j = assign_from_cost_hungarian(cost, n_samples=src_t.shape[0])
                        src_paired, gt_paired = src_arr[i.cpu().numpy()], gt_arr[j.cpu().numpy()]
                    else:
                        src_paired, gt_paired = self.ot_sampler.sample_plan_np(
                            src_arr,
                            gt_arr,
                            use_assignment=self.pair_mode == "assignment",
                        )
                    ready_q.put((
                        torch.from_numpy(src_paired).float(),
                        torch.from_numpy(gt_paired).float(),
                        ds_name,
                        cond,
                        perturbation_batch,
                    ))
                except Exception as e:
                    _error_holder[0] = e
                    ready_q.put(_sentinel)
                    return

        reader_t = threading.Thread(target=_reader, daemon=True)
        reader_t.start()
        workers = []
        for _ in range(self.n_ot_workers):
            t = threading.Thread(target=_ot_worker, daemon=True)
            t.start()
            workers.append(t)

        sentinels_seen = 0
        device = self.device

        def _to_dev(pb):
            return _pert_to_device(pb, device)

        while sentinels_seen < self.n_ot_workers:
            item = ready_q.get()
            if item is _sentinel:
                sentinels_seen += 1
                continue
            src_t, gt_t, ds_name, cond, perturbation_batch = item
            yield src_t, gt_t, ds_name, cond, _to_dev(perturbation_batch)

        reader_t.join(timeout=5)
        for t in workers:
            t.join(timeout=5)
        if _error_holder[0] is not None:
            raise _error_holder[0]


# ---------------------------------------------------------------------------
# Training step (receives already OT-paired data)
# ---------------------------------------------------------------------------

def gamma_schedule(step: int, cfg: Config) -> float:
    """Two-phase MMD warmup: no MMD early, then exponential ramp."""
    if not cfg.use_mmd:
        return 0.0

    # Fallback defaults aligned with `Config` in latent/config.py.
    warmup_start = getattr(cfg, "gamma_warmup_start", 50000)
    warmup_end = getattr(cfg, "gamma_warmup_end", 100000)

    if step < warmup_start:
        # Phase 1: no MMD
        return 0.0

    if step >= warmup_end:
        # Phase 3: constant at gamma_max
        return cfg.gamma

    # Phase 2: exponential warmup
    progress = (step - warmup_start) / (warmup_end - warmup_start)
    # Exponential: 1 - exp(-5×progress)
    return cfg.gamma * (1.0 - math.exp(-5.0 * progress))


def direction_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional condition-level direction loss."""
    weight = float(getattr(cfg, "direction_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "direction_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "direction_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def endpoint_delta_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional endpoint mean-delta MSE loss."""
    weight = float(getattr(cfg, "endpoint_delta_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "endpoint_delta_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "endpoint_delta_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def response_geometry_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional train-only response-geometry auxiliary loss."""
    weight = float(getattr(cfg, "response_geometry_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "response_geometry_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "response_geometry_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def response_geometry_filter_matches(cfg: Config, perturbation_batch: Optional[tuple]) -> bool:
    """Return whether response-geometry loss should apply to this condition batch.

    The filter is intentionally limited to deployable perturbation metadata. It
    never reads evaluation strata, held-out outcomes, residuals, or posthoc
    metrics.
    """
    mode = str(getattr(cfg, "response_geometry_condition_filter", "all") or "all").strip().lower()
    if mode in {"", "all"}:
        return True
    if mode != "gene_multi":
        raise ValueError(
            "response_geometry_condition_filter must be 'all' or 'gene_multi', "
            f"got {getattr(cfg, 'response_geometry_condition_filter', None)!r}"
        )
    if perturbation_batch is None:
        return False
    gid, mk, tid, npt, _cid, _ce, cm = _unpack_pert_up_to7(perturbation_batch)
    has_gene = bool((mk > 0).any().item())
    if not has_gene:
        return False
    max_nperts = int(torch.max(npt).detach().cpu().item()) if npt.numel() else 0
    if max_nperts < 2:
        return False
    is_drug = bool((tid == PERT_TYPE_DRUG).any().item())
    has_chem = cm is not None and bool((cm > 0).any().item())
    return not is_drug and not has_chem


def anchor_replay_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional anchor-replay/no-harm loss."""
    weight = float(getattr(cfg, "anchor_replay_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "anchor_replay_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "anchor_replay_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def anchor_replay_filter_matches(cfg: Config, perturbation_batch: Optional[tuple]) -> bool:
    """Return whether anchor replay should apply to this train batch.

    This gate is deliberately metadata-only. It does not inspect dataset names,
    held-out labels, posthoc metrics, or response residuals.
    """
    mode = str(getattr(cfg, "anchor_replay_condition_filter", "all") or "all").strip().lower()
    if mode in {"", "all"}:
        return True
    if mode != "non_gene_multi":
        raise ValueError(
            "anchor_replay_condition_filter must be 'all' or 'non_gene_multi', "
            f"got {getattr(cfg, 'anchor_replay_condition_filter', None)!r}"
        )
    return not response_geometry_filter_matches(
        dataclasses.replace(cfg, response_geometry_condition_filter="gene_multi"),
        perturbation_batch,
    )


def dataset_filter_matches(filter_value: str, ds_name: str) -> bool:
    """Return whether a dataset allow-list permits the current train batch."""
    raw = str(filter_value or "").strip()
    if not raw:
        return True
    if not ds_name:
        raise RuntimeError("Dataset-filtered train loss requires a non-empty dataset name.")
    allowed = {
        item.strip()
        for chunk in raw.split(";")
        for item in chunk.split(",")
        if item.strip()
    }
    return str(ds_name) in allowed


def risk_row_cvar_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional online risk-row CVaR/top-tail MMD loss."""
    weight = float(getattr(cfg, "risk_row_cvar_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "risk_row_cvar_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "risk_row_cvar_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


class RiskRowCvarTailState:
    """Online train-only state for risk-row CVaR/top-tail MMD weighting.

    The training loader yields one condition per step, so a true simultaneous
    differentiable top-k over many conditions is not available. This helper is
    deliberately explicit about the approximation: it uses detached historical
    condition MMD values to decide whether the current condition was previously
    in a high-risk tail, then the current differentiable MMD receives the extra
    loss weight. No canonical/test/query data enter this state.
    """

    def __init__(
        self,
        *,
        history_size: int = 256,
        min_history: int = 8,
        top_frac: float = 0.20,
        threshold: float = 0.005,
    ) -> None:
        self.history_size = max(1, int(history_size))
        self.min_history = max(1, int(min_history))
        self.top_frac = min(max(float(top_frac), 1e-6), 1.0)
        self.threshold = float(threshold)
        self._history: dict[str, deque[tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._latest: dict[tuple[str, str], float] = {}

    def update(self, ds_name: str, cond: str, mmd_value: float) -> None:
        ds = str(ds_name)
        condition = str(cond)
        value = float(mmd_value)
        if not math.isfinite(value):
            return
        self._latest[(ds, condition)] = value
        self._history[ds].append((condition, value))

    def dataset_cvar(self, ds_name: str) -> Optional[float]:
        values = [float(v) for _, v in self._history.get(str(ds_name), [])]
        if len(values) < self.min_history:
            return None
        k = max(1, int(math.ceil(len(values) * self.top_frac)))
        top = sorted(values, reverse=True)[:k]
        return float(sum(top) / len(top))

    def should_apply(self, ds_name: str, cond: str) -> bool:
        ds = str(ds_name)
        cvar = self.dataset_cvar(ds)
        if cvar is None or cvar <= self.threshold:
            return False
        value = self._latest.get((ds, str(cond)))
        if value is None:
            return False
        values = [float(v) for _, v in self._history.get(ds, [])]
        if len(values) < self.min_history:
            return False
        k = max(1, int(math.ceil(len(values) * self.top_frac)))
        cutoff = sorted(values, reverse=True)[k - 1]
        return value >= cutoff and value > self.threshold


def risk_row_cvar_batch_control(
    step: int,
    cfg: Config,
    state: RiskRowCvarTailState,
    ds_name: str,
    cond: str,
) -> tuple[bool, float]:
    """Return ``(observe, extra_weight)`` for one train batch."""
    base_weight = risk_row_cvar_loss_schedule(step, cfg)
    observe = (
        base_weight > 0
        and dataset_filter_matches(getattr(cfg, "risk_row_cvar_dataset_filter", ""), ds_name)
    )
    if not observe:
        return False, 0.0
    if state.should_apply(ds_name, cond):
        return True, base_weight
    return True, 0.0


def pert_residual_direction_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional dataset-centered perturbation residual direction loss."""
    weight = float(getattr(cfg, "pert_residual_direction_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "pert_residual_direction_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "pert_residual_direction_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def pert_residual_contrastive_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional condition-contrastive residual loss."""
    weight = float(getattr(cfg, "pert_residual_contrastive_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "pert_residual_contrastive_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "pert_residual_contrastive_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def pert_residual_relational_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional soft relational perturbation residual loss."""
    weight = float(getattr(cfg, "pert_residual_relational_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "pert_residual_relational_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "pert_residual_relational_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def composition_delta_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional synthetic composition delta loss."""
    weight = float(getattr(cfg, "composition_delta_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    every = max(1, int(getattr(cfg, "composition_delta_loss_every", 1) or 1))
    if step % every != 0:
        return 0.0
    warmup_start = int(getattr(cfg, "composition_delta_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "composition_delta_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def condition_delta_head_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional auxiliary condition-to-latent-delta loss."""
    weight = float(getattr(cfg, "condition_delta_head_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "condition_delta_head_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "condition_delta_head_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def additive_condition_delta_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional additive condition-delta atom-composition loss."""
    weight = float(getattr(cfg, "additive_condition_delta_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    warmup_start = int(getattr(cfg, "additive_condition_delta_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "additive_condition_delta_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def condition_prior_delta_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional train-single condition-prior teacher loss."""
    weight = float(getattr(cfg, "condition_prior_delta_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    every = max(1, int(getattr(cfg, "condition_prior_delta_loss_every", 1) or 1))
    if step % every != 0:
        return 0.0
    warmup_start = int(getattr(cfg, "condition_prior_delta_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "condition_prior_delta_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def condition_prior_additive_delta_loss_schedule(step: int, cfg: Config) -> float:
    """Warm up optional train-single prior supervision for additive head atoms."""
    weight = float(getattr(cfg, "condition_prior_additive_delta_loss_weight", 0.0) or 0.0)
    if weight <= 0:
        return 0.0
    every = max(1, int(getattr(cfg, "condition_prior_delta_loss_every", 1) or 1))
    if step % every != 0:
        return 0.0
    warmup_start = int(getattr(cfg, "condition_prior_additive_delta_loss_warmup_start", 0) or 0)
    warmup_end = int(getattr(cfg, "condition_prior_additive_delta_loss_warmup_end", 0) or 0)
    if warmup_end <= warmup_start:
        return weight if step >= warmup_start else 0.0
    if step < warmup_start:
        return 0.0
    if step >= warmup_end:
        return weight
    progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
    return weight * (1.0 - math.exp(-5.0 * progress))


def _pert_residual_relational_loss(
    pred_resid: torch.Tensor,
    target_resid: torch.Tensor,
    residual_bank: torch.Tensor,
    *,
    temperature: float = 0.10,
    target_temperature: float = 0.10,
) -> torch.Tensor:
    """Match soft residual-neighborhood structure for one condition.

    The target residual defines a similarity distribution over the current
    condition plus recent condition residuals. The predicted residual is trained
    to recover that distribution. This is intentionally softer than treating
    every other residual as a hard negative.
    """
    if residual_bank is None or residual_bank.numel() == 0:
        return pred_resid.new_zeros(())

    temperature = max(float(temperature), 1e-4)
    target_temperature = max(float(target_temperature), 1e-4)
    device = pred_resid.device

    pred_unit = F.normalize(pred_resid.float().unsqueeze(0), dim=-1)
    target_unit = F.normalize(target_resid.detach().float().unsqueeze(0), dim=-1)
    bank = residual_bank.to(device=device, dtype=torch.float32)
    candidates = F.normalize(
        torch.cat([target_resid.detach().float().unsqueeze(0).to(device), bank], dim=0),
        dim=-1,
    )
    student_logp = F.log_softmax((pred_unit @ candidates.t()) / temperature, dim=-1)
    with torch.no_grad():
        teacher_p = F.softmax((target_unit @ candidates.t()) / target_temperature, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean")


def train_step(
    src_paired: torch.Tensor,
    gt_paired: torch.Tensor,
    model: torch.nn.Module,
    path: CondOTPath,
    cfg: Config,
    device: torch.device,
    ds_name: str = "",
    gamma_t: float = 0.0,
    direction_weight_t: float = 0.0,
    endpoint_delta_weight_t: float = 0.0,
    response_geometry_weight_t: float = 0.0,
    response_normalizer: Optional[ResponseNormalizer] = None,
    pert_residual_direction_weight_t: float = 0.0,
    pert_residual_contrastive_weight_t: float = 0.0,
    pert_residual_relational_weight_t: float = 0.0,
    pert_residual_contrastive_bank: Optional[torch.Tensor] = None,
    pert_mean_ref: Optional[torch.Tensor] = None,
    composition_delta_weight_t: float = 0.0,
    composition_delta_target: Optional[torch.Tensor] = None,
    composition_perturbation_batch: Optional[tuple] = None,
    condition_prior_delta_weight_t: float = 0.0,
    condition_prior_delta_target: Optional[torch.Tensor] = None,
    condition_prior_perturbation_batch: Optional[tuple] = None,
    condition_prior_additive_delta_weight_t: float = 0.0,
    trackc_routed_distill_weight_t: float = 0.0,
    trackc_routed_distill_target: Optional[torch.Tensor] = None,
    trackc_routed_endpoint_weight_t: float = 0.0,
    anchor_replay_weight_t: float = 0.0,
    anchor_model: Optional[torch.nn.Module] = None,
    condition_delta_head_weight_t: float = 0.0,
    additive_condition_delta_weight_t: float = 0.0,
    risk_row_cvar_weight_t: float = 0.0,
    risk_row_cvar_observe: bool = False,
    perturbation_batch: Optional[tuple] = None,
    support_context: Optional[torch.Tensor] = None,
    support_context_present: Optional[torch.Tensor] = None,
    support_set_task: Optional[torch.Tensor] = None,
    support_set_task_present: Optional[torch.Tensor] = None,
) -> dict:
    """src_paired / gt_paired are already OT-matched; just do FM + loss."""
    B = src_paired.size(0)

    src = src_paired.to(device)
    gt = gt_paired.to(device)
    perturbation_batch_eval = perturbation_batch
    if _model_uses_pert(model):
        if perturbation_batch_eval is None:
            raise RuntimeError("train_step: perturbation_batch required for use_pert_condition models")

    t = sample_t_torch(B, device, mode=cfg.time_sampling)
    ps = path.sample(x_0=src, x_1=gt, t=t)

    with _amp_autocast_ctx(cfg, device):
        v_pred = _model_latent_velocity(
            model,
            ps.x_t,
            ps.t,
            src,
            perturbation_batch_eval,
            support_context=support_context,
            support_context_present=support_context_present,
            support_set_task=support_set_task,
            support_set_task_present=support_set_task_present,
        )

        mse = F.mse_loss(v_pred, ps.dx_t)
        loss = mse
        mmd_val = torch.zeros((), device=device, dtype=torch.float32)
        direction_val = torch.zeros((), device=device, dtype=torch.float32)
        endpoint_delta_val = torch.zeros((), device=device, dtype=torch.float32)
        response_geometry_val = torch.zeros((), device=device, dtype=torch.float32)
        pert_residual_direction_val = torch.zeros((), device=device, dtype=torch.float32)
        pert_residual_contrastive_val = torch.zeros((), device=device, dtype=torch.float32)
        pert_residual_relational_val = torch.zeros((), device=device, dtype=torch.float32)
        composition_delta_val = torch.zeros((), device=device, dtype=torch.float32)
        condition_prior_delta_val = torch.zeros((), device=device, dtype=torch.float32)
        condition_prior_additive_delta_val = torch.zeros((), device=device, dtype=torch.float32)
        trackc_routed_distill_val = torch.zeros((), device=device, dtype=torch.float32)
        trackc_routed_endpoint_val = torch.zeros((), device=device, dtype=torch.float32)
        anchor_replay_val = torch.zeros((), device=device, dtype=torch.float32)
        condition_delta_head_val = torch.zeros((), device=device, dtype=torch.float32)
        additive_condition_delta_val = torch.zeros((), device=device, dtype=torch.float32)
        x1_hat = None
        pred_delta = None
        target_delta = None
        pred_resid = None
        target_resid = None

        use_standard_mmd = (
            cfg.use_mmd
            and gamma_t > 0
            and dataset_filter_matches(getattr(cfg, "mmd_dataset_filter", ""), ds_name)
        )
        use_risk_row_mmd = bool(risk_row_cvar_observe) or risk_row_cvar_weight_t > 0
        if use_standard_mmd or use_risk_row_mmd:
            if cfg.mmd_ode_steps > 0:
                x1_hat = ode_integrate_diff(
                    model, src, src, n_steps=cfg.mmd_ode_steps,
                    perturbation_batch=perturbation_batch_eval,
                    support_context=support_context,
                    support_context_present=support_context_present,
                    support_set_task=support_set_task,
                    support_set_task_present=support_set_task_present,
                )
            else:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            # median_sigmas / MMD 用 fp32 保数值稳定
            sigmas = median_sigmas(gt.float())
            est = str(getattr(cfg, "mmd_estimator", "unbiased")).lower().strip()
            if est == "biased":
                mmd_fn = mmd2_biased
            elif est == "unbiased":
                mmd_fn = mmd2_unbiased
            else:
                raise ValueError(f"Unknown mmd_estimator={cfg.mmd_estimator!r} (use unbiased|biased)")
            mmd_raw = mmd_fn(x1_hat.float(), gt.float(), sigmas)
            mmd_t = mmd_raw if est == "biased" else torch.clamp(mmd_raw, min=0.0)
            loss = mse + (float(gamma_t) + float(risk_row_cvar_weight_t)) * mmd_t
            mmd_val = mmd_raw.detach()

        if direction_weight_t > 0:
            if x1_hat is None:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            pred_delta = (x1_hat.float() - src.float()).mean(dim=0)
            target_delta = (gt.float() - src.float()).mean(dim=0)
            denom = pred_delta.norm() * target_delta.norm() + 1e-8
            direction_t = 1.0 - torch.sum(pred_delta * target_delta) / denom
            direction_t = torch.clamp(direction_t, min=0.0, max=2.0)
            loss = loss + float(direction_weight_t) * direction_t
            direction_val = direction_t.detach()

        if endpoint_delta_weight_t > 0:
            if x1_hat is None:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            if pred_delta is None or target_delta is None:
                pred_delta = (x1_hat.float() - src.float()).mean(dim=0)
                target_delta = (gt.float() - src.float()).mean(dim=0)
            endpoint_delta_t = F.mse_loss(pred_delta, target_delta)
            loss = loss + float(endpoint_delta_weight_t) * endpoint_delta_t
            endpoint_delta_val = endpoint_delta_t.detach()

        if response_geometry_weight_t > 0 and response_geometry_filter_matches(cfg, perturbation_batch_eval):
            if response_normalizer is None:
                raise RuntimeError("response_geometry_loss_weight > 0 requires response_normalization_artifact")
            if not ds_name:
                raise RuntimeError("response_geometry_loss requires ds_name")
            if x1_hat is None:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            if pred_delta is None or target_delta is None:
                pred_delta = (x1_hat.float() - src.float()).mean(dim=0)
                target_delta = (gt.float() - src.float()).mean(dim=0)
            pred_resp = response_normalizer.transform_delta(str(ds_name), pred_delta.float())
            target_resp = response_normalizer.transform_delta(str(ds_name), target_delta.detach().float())
            response_geometry_t = F.mse_loss(pred_resp, target_resp)
            loss = loss + float(response_geometry_weight_t) * response_geometry_t
            response_geometry_val = response_geometry_t.detach()

        need_pert_resid = (
            (
                pert_residual_direction_weight_t > 0
                or pert_residual_contrastive_weight_t > 0
                or pert_residual_relational_weight_t > 0
            )
            and pert_mean_ref is not None
        )
        if need_pert_resid:
            if x1_hat is None:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            pert_mean_ref = pert_mean_ref.to(device=device, dtype=torch.float32)
            pred_resid = x1_hat.float().mean(dim=0) - pert_mean_ref
            target_resid = gt.float().mean(dim=0) - pert_mean_ref

        if pert_residual_direction_weight_t > 0 and pred_resid is not None and target_resid is not None:
            denom = pred_resid.norm() * target_resid.norm() + 1e-8
            pert_residual_direction_t = 1.0 - torch.sum(pred_resid * target_resid) / denom
            pert_residual_direction_t = torch.clamp(pert_residual_direction_t, min=0.0, max=2.0)
            loss = loss + float(pert_residual_direction_weight_t) * pert_residual_direction_t
            pert_residual_direction_val = pert_residual_direction_t.detach()

        if (
            pert_residual_contrastive_weight_t > 0
            and pred_resid is not None
            and target_resid is not None
            and pert_residual_contrastive_bank is not None
            and pert_residual_contrastive_bank.numel() > 0
        ):
            temp = max(float(getattr(cfg, "pert_residual_contrastive_temperature", 0.10) or 0.10), 1e-4)
            pred_unit = F.normalize(pred_resid.float().unsqueeze(0), dim=-1)
            pos = target_resid.detach().float().unsqueeze(0)
            neg = pert_residual_contrastive_bank.to(device=device, dtype=torch.float32)
            candidates = F.normalize(torch.cat([pos, neg], dim=0), dim=-1)
            logits = pred_unit @ candidates.t()
            labels = torch.zeros(1, device=device, dtype=torch.long)
            contrastive_t = F.cross_entropy(logits / temp, labels)
            loss = loss + float(pert_residual_contrastive_weight_t) * contrastive_t
            pert_residual_contrastive_val = contrastive_t.detach()

        if (
            pert_residual_relational_weight_t > 0
            and pred_resid is not None
            and target_resid is not None
            and pert_residual_contrastive_bank is not None
            and pert_residual_contrastive_bank.numel() > 0
        ):
            relational_t = _pert_residual_relational_loss(
                pred_resid,
                target_resid,
                pert_residual_contrastive_bank,
                temperature=getattr(cfg, "pert_residual_relational_temperature", 0.10),
                target_temperature=getattr(cfg, "pert_residual_relational_target_temperature", 0.10),
            )
            loss = loss + float(pert_residual_relational_weight_t) * relational_t
            pert_residual_relational_val = relational_t.detach()

        if (
            composition_delta_weight_t > 0
            and composition_delta_target is not None
            and composition_perturbation_batch is not None
        ):
            comp_pb = _pert_to_device(composition_perturbation_batch, device)
            t0 = torch.zeros(B, device=device, dtype=torch.float32)
            comp_v = _model_latent_velocity(
                model,
                src,
                t0,
                src,
                comp_pb,
                support_context=support_context,
                support_context_present=support_context_present,
                support_set_task=support_set_task,
                support_set_task_present=support_set_task_present,
            )
            comp_pred_delta = comp_v.float().mean(dim=0)
            comp_target = composition_delta_target.to(device=device, dtype=torch.float32)
            composition_delta_t = F.mse_loss(comp_pred_delta, comp_target)
            loss = loss + float(composition_delta_weight_t) * composition_delta_t
            composition_delta_val = composition_delta_t.detach()

        if (
            condition_prior_delta_weight_t > 0
            and condition_prior_delta_target is not None
            and condition_prior_perturbation_batch is not None
        ):
            prior_pb = _pert_to_device(condition_prior_perturbation_batch, device)
            t0 = torch.zeros(B, device=device, dtype=torch.float32)
            prior_v = _model_latent_velocity(
                model,
                src,
                t0,
                src,
                prior_pb,
                support_context=support_context,
                support_context_present=support_context_present,
                support_set_task=support_set_task,
                support_set_task_present=support_set_task_present,
            )
            prior_pred_delta = prior_v.float().mean(dim=0)
            prior_target = condition_prior_delta_target.to(device=device, dtype=torch.float32)
            condition_prior_delta_t = F.mse_loss(prior_pred_delta, prior_target)
            loss = loss + float(condition_prior_delta_weight_t) * condition_prior_delta_t
            condition_prior_delta_val = condition_prior_delta_t.detach()

        if (
            condition_prior_additive_delta_weight_t > 0
            and condition_prior_delta_target is not None
            and condition_prior_perturbation_batch is not None
        ):
            prior_pb = _pert_to_device(condition_prior_perturbation_batch, device)
            prior_additive_pred = _model_additive_condition_delta(model, prior_pb)
            if prior_additive_pred is None:
                raise RuntimeError("condition_prior_additive_delta loss requested but model head is disabled")
            prior_target = condition_prior_delta_target.to(device=device, dtype=prior_additive_pred.dtype)
            prior_target = prior_target.unsqueeze(0).expand_as(prior_additive_pred)
            condition_prior_additive_delta_t = F.mse_loss(prior_additive_pred, prior_target)
            loss = loss + float(condition_prior_additive_delta_weight_t) * condition_prior_additive_delta_t
            condition_prior_additive_delta_val = condition_prior_additive_delta_t.detach()

        if trackc_routed_distill_weight_t > 0 and trackc_routed_distill_target is not None:
            if perturbation_batch_eval is None:
                raise RuntimeError("Track C routed distillation requires perturbation_batch")
            delta_pred = _model_condition_delta(model, perturbation_batch_eval)
            if delta_pred is None:
                raise RuntimeError("Track C routed distillation requested but condition_delta_head is disabled")
            route_target = trackc_routed_distill_target.to(device=device, dtype=delta_pred.dtype)
            route_target = route_target.unsqueeze(0).expand_as(delta_pred)
            trackc_routed_distill_t = F.mse_loss(delta_pred, route_target)
            loss = loss + float(trackc_routed_distill_weight_t) * trackc_routed_distill_t
            trackc_routed_distill_val = trackc_routed_distill_t.detach()

        if trackc_routed_endpoint_weight_t > 0 and trackc_routed_distill_target is not None:
            if x1_hat is None:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            route_target = trackc_routed_distill_target.to(device=device, dtype=x1_hat.dtype)
            endpoint_target = src.to(device=device, dtype=x1_hat.dtype) + route_target.unsqueeze(0)
            trackc_routed_endpoint_t = F.mse_loss(x1_hat, endpoint_target.expand_as(x1_hat))
            loss = loss + float(trackc_routed_endpoint_weight_t) * trackc_routed_endpoint_t
            trackc_routed_endpoint_val = trackc_routed_endpoint_t.detach()

        if (
            anchor_replay_weight_t > 0
            and anchor_replay_filter_matches(cfg, perturbation_batch_eval)
            and dataset_filter_matches(getattr(cfg, "anchor_replay_dataset_filter", ""), ds_name)
        ):
            if anchor_model is None:
                raise RuntimeError("anchor_replay_loss_weight > 0 requires a frozen anchor_model")
            if x1_hat is None:
                x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
            with torch.no_grad():
                anchor_v = _model_latent_velocity(anchor_model, ps.x_t, ps.t, src, perturbation_batch_eval)
                anchor_x1_hat = ps.x_t + anchor_v * (1.0 - t).unsqueeze(-1)
            anchor_replay_t = F.mse_loss(x1_hat.float(), anchor_x1_hat.detach().float())
            loss = loss + float(anchor_replay_weight_t) * anchor_replay_t
            anchor_replay_val = anchor_replay_t.detach()

        if condition_delta_head_weight_t > 0:
            if perturbation_batch_eval is None:
                raise RuntimeError("condition_delta_head loss requires perturbation_batch")
            delta_pred = _model_condition_delta(model, perturbation_batch_eval)
            if delta_pred is None:
                raise RuntimeError("condition_delta_head loss requested but model head is disabled")
            head_target = str(getattr(cfg, "condition_delta_head_target", "endpoint_delta") or "endpoint_delta")
            head_target = head_target.strip().lower()
            if head_target == "endpoint_delta":
                if target_delta is None:
                    target_delta = (gt.float() - src.float()).mean(dim=0)
                target_for_head = target_delta
            elif head_target == "pert_residual":
                if target_resid is None:
                    if pert_mean_ref is None:
                        raise RuntimeError(
                            "condition_delta_head_target='pert_residual' requires pert_mean_ref"
                        )
                    pert_mean_ref = pert_mean_ref.to(device=device, dtype=torch.float32)
                    target_resid = gt.float().mean(dim=0) - pert_mean_ref
                target_for_head = target_resid
            else:
                raise ValueError(
                    "condition_delta_head_target must be 'endpoint_delta' or 'pert_residual', "
                    f"got {getattr(cfg, 'condition_delta_head_target', None)!r}"
                )
            target_rows = target_for_head.detach().to(device=device, dtype=delta_pred.dtype).unsqueeze(0)
            target_rows = target_rows.expand_as(delta_pred)
            condition_delta_head_t = F.mse_loss(delta_pred, target_rows)
            loss = loss + float(condition_delta_head_weight_t) * condition_delta_head_t
            condition_delta_head_val = condition_delta_head_t.detach()

        if (
            additive_condition_delta_weight_t > 0
            and composition_delta_target is not None
            and composition_perturbation_batch is not None
        ):
            comp_pb = _pert_to_device(composition_perturbation_batch, device)
            additive_delta_pred = _model_additive_condition_delta(model, comp_pb)
            if additive_delta_pred is None:
                raise RuntimeError("additive_condition_delta loss requested but model head is disabled")
            additive_target = composition_delta_target.to(device=device, dtype=additive_delta_pred.dtype)
            additive_target = additive_target.unsqueeze(0).expand_as(additive_delta_pred)
            additive_condition_delta_t = F.mse_loss(additive_delta_pred, additive_target)
            loss = loss + float(additive_condition_delta_weight_t) * additive_condition_delta_t
            additive_condition_delta_val = additive_condition_delta_t.detach()

    return {
        "loss": loss,
        "mse": mse.detach(),
        "mmd": mmd_val,
        "direction": direction_val,
        "endpoint_delta": endpoint_delta_val,
        "response_geometry": response_geometry_val,
        "pert_residual_direction": pert_residual_direction_val,
        "pert_residual_contrastive": pert_residual_contrastive_val,
        "pert_residual_relational": pert_residual_relational_val,
        "composition_delta": composition_delta_val,
        "condition_prior_delta": condition_prior_delta_val,
        "condition_prior_additive_delta": condition_prior_additive_delta_val,
        "trackc_routed_distill": trackc_routed_distill_val,
        "trackc_routed_endpoint": trackc_routed_endpoint_val,
        "anchor_replay": anchor_replay_val,
        "condition_delta_head": condition_delta_head_val,
        "additive_condition_delta": additive_condition_delta_val,
        "risk_row_cvar_weight": torch.as_tensor(float(risk_row_cvar_weight_t), device=device),
        "target_residual": target_resid.detach() if target_resid is not None else None,
        "total": loss.detach(),
    }


# ---------------------------------------------------------------------------
# Differentiable ODE integration (for training MMD loss)
# ---------------------------------------------------------------------------

def ode_integrate_diff(
    model: torch.nn.Module,
    x0: torch.Tensor,
    src: torch.Tensor,
    n_steps: int = 10,
    perturbation_batch: Optional[tuple] = None,
    support_context: Optional[torch.Tensor] = None,
    support_context_present: Optional[torch.Tensor] = None,
    support_set_task: Optional[torch.Tensor] = None,
    support_set_task_present: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable Euler ODE integration from t=0 to t=1.

    Unlike ode_integrate (which uses @torch.no_grad), this version retains
    the computation graph so gradients can flow back through the ODE trajectory
    into the model parameters.  Used for ODE-based MMD training loss.
    """
    dt = 1.0 / n_steps
    x = x0
    use_pert = _model_uses_pert(model)
    use_support_context = _model_uses_support_context(model)
    use_support_set_task = _model_uses_support_set_task(model)
    if use_support_context and support_context is None:
        support_context = _zero_support_context_for(model, x0)
    for i in range(n_steps):
        t_val = i * dt
        t_vec = torch.full((x.size(0),), t_val, device=x.device, dtype=torch.float32)
        if use_pert:
            if perturbation_batch is None:
                raise RuntimeError("ode_integrate_diff requires perturbation_batch for pert-conditioned models")
            if use_support_context or use_support_set_task:
                v = _model_latent_velocity(
                    model,
                    x,
                    t_vec,
                    src,
                    perturbation_batch,
                    support_context=support_context,
                    support_context_present=support_context_present,
                    support_set_task=support_set_task,
                    support_set_task_present=support_set_task_present,
                )
                x = x + v * dt
                continue
            gid, mk, tid, npt, cid, chem_emb, chem_mask = _unpack_pert_up_to7(
                perturbation_batch
            )
            v = torch.utils.checkpoint.checkpoint(
                _fm_ckpt_velocity_pert,
                model,
                x,
                t_vec,
                src,
                gid,
                mk,
                tid,
                npt,
                cid,
                chem_emb,
                chem_mask,
                use_reentrant=False,
            )
        elif use_support_context or use_support_set_task:
            v = _model_latent_velocity(
                model,
                x,
                t_vec,
                src,
                None,
                support_context=support_context,
                support_context_present=support_context_present,
                support_set_task=support_set_task,
                support_set_task_present=support_set_task_present,
            )
        else:
            v = torch.utils.checkpoint.checkpoint(model, x, t_vec, src, use_reentrant=False)
        x = x + v * dt
    return x


# ---------------------------------------------------------------------------
# ODE integration (Euler)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ode_integrate(
    model: torch.nn.Module,
    x0: torch.Tensor,
    src: torch.Tensor,
    cfg: Optional[Config] = None,
    n_steps: int = 20,
    perturbation_batch: Optional[tuple] = None,
    support_context: Optional[torch.Tensor] = None,
    support_context_present: Optional[torch.Tensor] = None,
    support_set_task: Optional[torch.Tensor] = None,
    support_set_task_present: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler integration from t=0 to t=1. Returns predicted x_1."""
    del cfg  # retained for backwards-compatible call sites
    dt = 1.0 / n_steps
    x = x0.clone()
    use_pert = _model_uses_pert(model)
    for i in range(n_steps):
        t_val = i * dt
        t_vec = torch.full((x.size(0),), t_val, device=x.device, dtype=torch.float32)
        if use_pert:
            if perturbation_batch is None:
                raise ValueError(
                    "ode_integrate on a pert-conditioned model requires perturbation_batch "
                    "(or use latent_utils.null_perturbation_batch)."
                )
        v = _model_latent_velocity(
            model,
            x,
            t_vec,
            src,
            perturbation_batch if use_pert else None,
            support_context=support_context,
            support_context_present=support_context_present,
            support_set_task=support_set_task,
            support_set_task_present=support_set_task_present,
        )
        x = x + v * dt
    return x


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays."""
    a_m = a - a.mean()
    b_m = b - b.mean()
    numer = (a_m * b_m).sum()
    denom = math.sqrt((a_m ** 2).sum() * (b_m ** 2).sum() + 1e-12)
    return float(numer / denom)


def _eval_rank(seed: int, *parts: object) -> int:
    text = "eval_select:" + ":".join(str(p) for p in (seed, *parts))
    return _stable_int_hash(text)


def _select_eval_condition_pairs(
    dataset: CrossDatasetFMDataset,
    cfg: Config,
) -> tuple[list[tuple[str, str]], int]:
    """Return capped eval pairs using stable hashes instead of input list order."""
    seed = int(getattr(cfg, "seed", 0) or 0)
    max_per_ds = int(getattr(cfg, "eval_max_conditions_per_dataset", 0) or 0)
    pairs: list[tuple[str, str]] = []
    n_available = 0
    for ds_name in dataset.ds_names:
        conds = list(dict.fromkeys(str(c) for c in dataset.ds_conds[ds_name]))
        n_available += len(conds)
        if max_per_ds > 0 and len(conds) > max_per_ds:
            conds = sorted(
                conds,
                key=lambda c: (_eval_rank(seed, "per_dataset", ds_name, c), ds_name, c),
            )[:max_per_ds]
        conds = sorted(conds)
        pairs.extend((ds_name, cond) for cond in conds)

    max_conds = int(getattr(cfg, "eval_max_conditions", 0) or 0)
    if max_conds > 0 and len(pairs) > max_conds:
        pairs = sorted(
            pairs,
            key=lambda p: (_eval_rank(seed, "global", p[0], p[1]), p[0], p[1]),
        )[:max_conds]
    pairs = sorted(pairs)
    return pairs, n_available



@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    path: CondOTPath,
    cfg: Config,
    device: torch.device,
    ctrl_means: dict = None,
    pert_means: dict = None,
    ode_steps: int = 20,
    max_chunk: int = 512,
    max_mmd_cells: int = 2048,
    show_progress: bool = False,
    progress_desc: str = "eval",
    trackc_support_context_bank: Optional[dict[str, Any]] = None,
    trackc_support_set_task_bank: Optional[dict[str, Any]] = None,
) -> dict:
    """Epoch-end eval over ALL test conditions using full source-pool / GT per condition.

    Three-phase pipeline for efficient GPU utilisation:
      Phase 1  – per-condition MSE (velocity) + collect ODE inputs on CPU
      Phase 2  – batched ODE integration across ALL conditions (GPU-dense)
      Phase 3  – per-condition MMD & Pearson from batched predictions

    Metrics (all per-condition -> per-dataset mean -> overall mean):
      - test_mse / test_mae: pooled velocity error
      - test_mmd: MMD^2 between ODE-predicted and GT distributions
      - direct_pearson / pearson_ctrl / pearson_pert
    """
    model.eval()
    from collections import defaultdict

    _mse_acc = torch.zeros(1, device=device, dtype=torch.float64)
    _mae_acc = torch.zeros(1, device=device, dtype=torch.float64)
    all_mse_n = 0

    per_ds_mmd = defaultdict(list)
    per_ds_mmd_biased = defaultdict(list)
    per_ds_mmd_clamped = defaultdict(list)
    per_ds_direct = defaultdict(list)
    per_ds_p_ctrl = defaultdict(list)
    per_ds_p_pert = defaultdict(list)
    condition_metrics = []
    count = 0

    cond_pairs, n_available_conditions = _select_eval_condition_pairs(dataset, cfg)
    max_mse_cells = int(getattr(cfg, "eval_max_mse_cells", 0) or 0)
    max_mmd_cells = int(getattr(cfg, "eval_max_mmd_cells", max_mmd_cells) or max_mmd_cells)

    # ── Phase 1: per-condition MSE + collect ODE inputs ───────────
    ode_src_parts = []
    pert_ode_parts = []
    support_context_ode_parts = []
    support_set_task_ode_parts = []
    support_set_task_present_ode_parts = []
    cond_meta = []
    use_pe = _model_uses_pert(model)
    use_support_context = _model_uses_support_context(model)
    use_support_set_task = _model_uses_support_set_task(model)
    support_context_source_active = _support_context_source_active(cfg)
    support_set_task_source_active = _support_set_task_source_active(cfg)
    if use_support_context and support_context_source_active and trackc_support_context_bank is None:
        validate_support_context_config(cfg)
        trackc_support_context_bank = build_trackc_routed_distill_bank(dataset, cfg)
    if use_support_set_task and support_set_task_source_active and trackc_support_set_task_bank is None:
        validate_support_set_task_config(cfg)
        trackc_support_set_task_bank = build_trackc_support_set_task_bank(cfg)
    support_context_eval_control = _support_context_eval_control_mode(cfg)
    support_context_shuffle_targets = (
        _build_trackc_support_context_shuffle_targets(trackc_support_context_bank or {}, dataset, cond_pairs)
        if use_support_context
        and support_context_source_active
        and support_context_eval_control == "shuffle_condition"
        else {}
    )
    support_set_task_eval_control = _support_set_task_eval_control_mode(cfg)
    support_set_task_shuffle_targets = (
        _build_trackc_support_set_task_shuffle_targets(trackc_support_set_task_bank or {}, cond_pairs, cfg)
        if use_support_set_task
        and support_set_task_source_active
        and support_set_task_eval_control == "shuffle_condition"
        else {}
    )

    for ds_name, cond in cond_pairs:
        handle = dataset.handles[ds_name]
        ctrl_mean = ctrl_means.get(ds_name) if ctrl_means else None
        pert_mean = pert_means.get(ds_name) if pert_means else None

        src_np = handle.read_src(cond)
        gt_np = handle.read_gt(cond)
        src_full = torch.from_numpy(src_np).float()
        gt_full = torch.from_numpy(gt_np).float()

        seed = int(getattr(cfg, "seed", 0) or 0)
        rng_mse = np.random.RandomState(_eval_rank(seed, "mse_cells", ds_name, cond) % (2**32 - 1))
        rng_mmd = np.random.RandomState(_eval_rank(seed, "mmd_cells", ds_name, cond) % (2**32 - 1))
        n = min(src_full.size(0), gt_full.size(0))
        if max_mse_cells > 0:
            n = min(n, max_mse_cells)
        src_shuf = src_full[rng_mse.permutation(src_full.size(0))[:n]]
        gt_shuf = gt_full[rng_mse.permutation(gt_full.size(0))[:n]]

        for start in range(0, n, max_chunk):
            end = min(start + max_chunk, n)
            src_c = src_shuf[start:end].to(device)
            gt_c = gt_shuf[start:end].to(device)
            B = src_c.size(0)

            t = sample_t_torch(B, device, mode=cfg.time_sampling)
            ps = path.sample(x_0=src_c, x_1=gt_c, t=t)
            if use_pe:
                pb_cpu = _pert_for_eval_batch(dataset, ds_name, cond, B)
                pb_dev = _pert_to_device(pb_cpu, device)
            else:
                pb_dev = None
            support_context = None
            support_set_task = None
            support_set_task_present = None
            if use_support_context and support_context_source_active:
                support_context = make_trackc_support_context_batch(
                    trackc_support_context_bank or {},
                    dataset,
                    ds_name,
                    cond,
                    B,
                    cfg,
                    device,
                )
                support_context = _apply_trackc_support_context_eval_control(
                    support_context,
                    cfg=cfg,
                    ds_name=ds_name,
                    cond=cond,
                    batch_size=B,
                    device=device,
                    shuffle_targets=support_context_shuffle_targets,
                )
            if use_support_set_task and support_set_task_source_active:
                support_set_task, support_set_task_present = make_trackc_support_set_task_batch(
                    trackc_support_set_task_bank or {},
                    ds_name,
                    cond,
                    B,
                    cfg,
                    device,
                )
                support_set_task, support_set_task_present = _apply_trackc_support_set_task_eval_control(
                    support_set_task,
                    support_set_task_present,
                    cfg=cfg,
                    ds_name=ds_name,
                    cond=cond,
                    batch_size=B,
                    device=device,
                    shuffle_targets=support_set_task_shuffle_targets,
                )
            v_pred = _model_latent_velocity(
                model,
                ps.x_t,
                ps.t,
                src_c,
                pb_dev if use_pe else None,
                support_context=support_context,
                support_set_task=support_set_task,
                support_set_task_present=support_set_task_present,
            )
            _mse_acc += F.mse_loss(v_pred, ps.dx_t, reduction="sum").to(torch.float64)
            _mae_acc += F.l1_loss(v_pred, ps.dx_t, reduction="sum").to(torch.float64)
            all_mse_n += B * cfg.emb_dim

        n_src = src_full.size(0)
        n_gt = gt_full.size(0)
        n_src_eval = min(n_src, max_mmd_cells)
        n_gt_eval = min(n_gt, max_mmd_cells)
        src_eval = src_full[rng_mmd.permutation(n_src)[:n_src_eval]]
        gt_eval = gt_full[rng_mmd.permutation(n_gt)[:n_gt_eval]]

        ode_src_parts.append(src_eval)
        if use_pe:
            pert_ode_parts.append(_pert_for_eval_batch(dataset, ds_name, cond, n_src_eval))
        else:
            pert_ode_parts.append(None)
        if use_support_context and support_context_source_active:
            support_context_ode = make_trackc_support_context_batch(
                trackc_support_context_bank or {},
                dataset,
                ds_name,
                cond,
                n_src_eval,
                cfg,
                torch.device("cpu"),
            )
            support_context_ode_parts.append(
                _apply_trackc_support_context_eval_control(
                    support_context_ode,
                    cfg=cfg,
                    ds_name=ds_name,
                    cond=cond,
                    batch_size=n_src_eval,
                    device=torch.device("cpu"),
                    shuffle_targets=support_context_shuffle_targets,
                )
            )
        else:
            support_context_ode_parts.append(None)
        if use_support_set_task and support_set_task_source_active:
            support_set_task_ode, support_set_task_present_ode = make_trackc_support_set_task_batch(
                trackc_support_set_task_bank or {},
                ds_name,
                cond,
                n_src_eval,
                cfg,
                torch.device("cpu"),
            )
            support_set_task_ode, support_set_task_present_ode = _apply_trackc_support_set_task_eval_control(
                support_set_task_ode,
                support_set_task_present_ode,
                cfg=cfg,
                ds_name=ds_name,
                cond=cond,
                batch_size=n_src_eval,
                device=torch.device("cpu"),
                shuffle_targets=support_set_task_shuffle_targets,
            )
            support_set_task_ode_parts.append(support_set_task_ode)
            support_set_task_present_ode_parts.append(support_set_task_present_ode)
        else:
            support_set_task_ode_parts.append(None)
            support_set_task_present_ode_parts.append(None)

        cond_meta.append((
            ds_name, cond, ctrl_mean, pert_mean,
            gt_eval,
            n_src_eval, n_gt_eval,
        ))
        count += 1

    # ── Phase 2: ODE integration (per condition; pert may differ per segment) ─
    emb_dim = ode_src_parts[0].size(1) if ode_src_parts else int(cfg.emb_dim)
    all_pred_blocks = []
    for src_block, pb_cpu, support_context_cpu, support_set_task_cpu, support_set_task_present_cpu in zip(
        ode_src_parts,
        pert_ode_parts,
        support_context_ode_parts,
        support_set_task_ode_parts,
        support_set_task_present_ode_parts,
    ):
        nloc = int(src_block.size(0))
        pb_dev_full = None if pb_cpu is None else _pert_to_device(pb_cpu, device)
        support_context_full = None if support_context_cpu is None else support_context_cpu.to(device)
        support_set_task_full = None if support_set_task_cpu is None else support_set_task_cpu.to(device)
        support_set_task_present_full = (
            None if support_set_task_present_cpu is None else support_set_task_present_cpu.to(device)
        )
        piece = torch.empty(nloc, emb_dim, device=device)
        for st in range(0, nloc, max_chunk):
            en = min(st + max_chunk, nloc)
            s = src_block[st:en].to(device)
            pb_use = None if pb_dev_full is None else _pert_chunk(pb_dev_full, st, en)
            support_context_use = None if support_context_full is None else support_context_full[st:en]
            support_set_task_use = None if support_set_task_full is None else support_set_task_full[st:en]
            support_set_task_present_use = (
                None if support_set_task_present_full is None else support_set_task_present_full[st:en]
            )
            piece[st:en] = ode_integrate(
                model, s, s, cfg,
                n_steps=ode_steps,
                perturbation_batch=pb_use if use_pe else None,
                support_context=support_context_use,
                support_set_task=support_set_task_use,
                support_set_task_present=support_set_task_present_use,
            )
        all_pred_blocks.append(piece)

    all_pred = torch.cat(all_pred_blocks, dim=0) if all_pred_blocks else torch.empty(0, emb_dim, device=device)
    del all_pred_blocks

    # ── Phase 3: per-condition MMD + Pearson ──────────────────────
    offset = 0
    for (ds_name, cond, ctrl_mean, pert_mean,
         gt_eval, n_src_eval, n_gt_eval) in cond_meta:
        x1_hat = all_pred[offset:offset + n_src_eval]
        gt_dev = gt_eval.to(device)

        sigmas, Dyy = median_sigmas(gt_dev, return_D2=True)
        mmd_val = mmd2_unbiased(x1_hat, gt_dev, sigmas, Dyy=Dyy).item()
        mmd_biased_val = mmd2_biased(x1_hat, gt_dev, sigmas, Dyy=Dyy).item()
        mmd_clamped_val = max(float(mmd_val), 0.0)
        per_ds_mmd[ds_name].append(mmd_val)
        per_ds_mmd_biased[ds_name].append(mmd_biased_val)
        per_ds_mmd_clamped[ds_name].append(mmd_clamped_val)

        pred_mean = x1_hat.mean(dim=0).cpu().numpy()
        gt_mean = gt_dev.mean(dim=0).cpu().numpy()

        direct_val = _pearson_np(pred_mean, gt_mean)
        per_ds_direct[ds_name].append(direct_val)

        p_ctrl_val = None
        if ctrl_mean is not None:
            p_ctrl_val = _pearson_np(pred_mean - ctrl_mean, gt_mean - ctrl_mean)
            per_ds_p_ctrl[ds_name].append(p_ctrl_val)

        p_pert_val = None
        if pert_mean is not None:
            p_pert_val = _pearson_np(pred_mean - pert_mean, gt_mean - pert_mean)
            per_ds_p_pert[ds_name].append(p_pert_val)

        cond_record = {
            "dataset": str(ds_name),
            "condition": str(cond),
            "test_mmd": float(mmd_val),
            "test_mmd_biased": float(mmd_biased_val),
            "test_mmd_clamped": float(mmd_clamped_val),
            "direct_pearson": float(direct_val),
            "pearson_ctrl": None if p_ctrl_val is None else float(p_ctrl_val),
            "pearson_pert": None if p_pert_val is None else float(p_pert_val),
            "n_src_eval": int(n_src_eval),
            "n_gt_eval": int(n_gt_eval),
        }
        if bool(getattr(cfg, "eval_save_condition_means", False)):
            cond_record["pred_mean"] = pred_mean.astype(np.float32)
            cond_record["gt_mean"] = gt_mean.astype(np.float32)
            cond_record["ctrl_mean"] = None if ctrl_mean is None else np.asarray(ctrl_mean, dtype=np.float32)
            cond_record["pert_mean"] = None if pert_mean is None else np.asarray(pert_mean, dtype=np.float32)
        condition_metrics.append(cond_record)

        del gt_dev
        offset += n_src_eval
    del all_pred

    model.train()

    all_mse_sum = _mse_acc.item()
    all_mae_sum = _mae_acc.item()
    test_mse = all_mse_sum / max(all_mse_n, 1)
    test_mae = all_mae_sum / max(all_mse_n, 1)

    def _agg(d):
        return {k: float(np.mean(v)) for k, v in sorted(d.items())}

    def _ds_mean(d):
        return float(np.mean(list(d.values()))) if d else float("nan")

    ds_mmd = _agg(per_ds_mmd)
    ds_mmd_biased = _agg(per_ds_mmd_biased)
    ds_mmd_clamped = _agg(per_ds_mmd_clamped)
    ds_direct = _agg(per_ds_direct)
    ds_p_ctrl = _agg(per_ds_p_ctrl)
    ds_p_pert = _agg(per_ds_p_pert)

    return {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_mmd": _ds_mean(ds_mmd),
        "test_mmd_biased": _ds_mean(ds_mmd_biased),
        "test_mmd_clamped": _ds_mean(ds_mmd_clamped),
        "direct_pearson": _ds_mean(ds_direct),
        "pearson_ctrl": _ds_mean(ds_p_ctrl),
        "pearson_pert": _ds_mean(ds_p_pert),
        "n_conds": count,
        "n_available_conditions": int(n_available_conditions),
        "eval_caps": {
            "eval_max_conditions": int(getattr(cfg, "eval_max_conditions", 0) or 0),
            "eval_max_conditions_per_dataset": int(
                getattr(cfg, "eval_max_conditions_per_dataset", 0) or 0
            ),
            "eval_max_mse_cells": int(getattr(cfg, "eval_max_mse_cells", 0) or 0),
            "eval_max_mmd_cells": int(max_mmd_cells),
            "eval_save_condition_means": bool(getattr(cfg, "eval_save_condition_means", False)),
            "condition_selection": "stable_hash_dataset_condition",
            "cell_selection": "stable_hash_dataset_condition_metric",
            "aggregation": "condition_mean_then_dataset_equal_mean",
        },
        "selected_conditions": [
            {"dataset": ds_name, "condition": cond}
            for ds_name, cond in cond_pairs
        ],
        "per_ds_mmd": ds_mmd,
        "per_ds_mmd_biased": ds_mmd_biased,
        "per_ds_mmd_clamped": ds_mmd_clamped,
        "per_ds_direct": ds_direct,
        "per_ds_p_ctrl": ds_p_ctrl,
        "per_ds_p_pert": ds_p_pert,
        "condition_metrics": condition_metrics,
    }


def _cross_dataset_kw(cfg: Config) -> dict:
    if getattr(cfg, "use_pert_condition", False):
        cd = str(getattr(cfg, "pert_gene_emb_cache_dir", "") or "").strip()
        if not cd:
            raise ValueError(
                "use_pert_condition=True requires non-empty pert_gene_emb_cache_dir (GeneEmbeddingCache root)"
            )
        return dict(
            use_pert_condition=True,
            max_pert_genes=int(getattr(cfg, "max_pert_genes", 16)),
            gene_embedding_cache_dir=cd,
            biflow_dir=getattr(cfg, "biflow_dir", None),
            latent_backbone=str(getattr(cfg, "latent_backbone", "state") or "state"),
            use_h5ad_pert_metadata=bool(getattr(cfg, "use_h5ad_pert_metadata", False)),
            pert_metainfo_path=str(getattr(cfg, "pert_metainfo_path", "") or ""),
            chem_emb_source_dir=str(getattr(cfg, "chem_emb_source_dir", "") or ""),
            chem_obs_column=str(getattr(cfg, "chem_obs_column", "") or ""),
            drug_emb_cache_dir=str(getattr(cfg, "drug_emb_cache_dir", "") or ""),
            max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
            chemical_metainfo_path=str(getattr(cfg, "chemical_metainfo_path", "") or ""),
            chem_fallback_embed_dim=max(
                8, int(getattr(cfg, "pert_chem_emb_dim", getattr(cfg, "chem_fallback_embed_dim", 512)) or 512),
            ),
            pert_chem_enabled=bool(getattr(cfg, "pert_chem_enabled", False)),
        )
    return {}


def maybe_load_response_normalizer(
    cfg: Config,
    *,
    split_path: Optional[Path],
    device: torch.device,
    log,
) -> Optional[ResponseNormalizer]:
    mode = str(getattr(cfg, "response_normalization_mode", "off") or "off").strip().lower()
    weight = float(getattr(cfg, "response_geometry_loss_weight", 0.0) or 0.0)
    if weight <= 0 and not ResponseNormalizer.is_enabled_mode(mode):
        return None
    if weight > 0 and not ResponseNormalizer.is_enabled_mode(mode):
        raise ValueError("response_geometry_loss_weight > 0 requires response_normalization_mode != off")
    artifact = str(getattr(cfg, "response_normalization_artifact", "") or "").strip()
    if weight > 0 and not artifact:
        raise ValueError("response_geometry_loss_weight > 0 requires response_normalization_artifact")
    if not artifact:
        return None
    if split_path is None and bool(getattr(cfg, "response_normalization_strict_split", True)):
        raise ValueError("response_normalization_strict_split requires a resolved split_path")
    rn = ResponseNormalizer.from_npz(
        artifact,
        mode=mode,
        device=device,
        strict_split_file=split_path if bool(getattr(cfg, "response_normalization_strict_split", True)) else None,
        strict_emb_dim=int(getattr(cfg, "emb_dim", 0) or 0),
    )
    meta = rn.metadata or {}
    if str(meta.get("fit_scope", "")) != "train_only":
        raise ValueError(f"response normalizer artifact fit_scope must be train_only, got {meta.get('fit_scope')!r}")
    log(
        "Loaded response normalizer: "
        f"mode={rn.mode} artifact={rn.artifact_path} "
        f"n_train_residuals={meta.get('n_train_residuals')} "
        f"pca_components={meta.get('pca_components')}"
    )
    return rn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = tyro.cli(Config)
    fill_condition_embedding_source(cfg)
    validate_support_context_config(cfg)
    validate_support_set_task_config(cfg)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank_raw = os.environ.get("LOCAL_RANK", "")
    if local_rank_raw not in (None, ""):
        local_rank = int(local_rank_raw)
    else:
        local_rank = max(0, rank % max(world_size, 1))
    ddp_active = world_size > 1 and torch.cuda.is_available()
    is_rank0 = rank == 0

    if ddp_active:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    save_dir = cfg.make_save_dir()
    log_path = save_dir / cfg.log_file

    # ── data ──────────────────────────────────────────────────────
    manifest_path = Path(cfg.data_dir) / cfg.manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load split: prefer cfg.split_file (--split-file), else SPLIT_FILE env (from patch), else default
    split_path: Optional[Path] = None
    if cfg.split_file:
        split_path = Path(cfg.split_file)
        if not split_path.is_absolute():
            # Resolve relative to expriement root (save_dir is .../expriement/exp0_.../runs/solo_xxx)
            base = Path(cfg.save_dir).resolve().parent.parent.parent
            split_path = base / cfg.split_file
        with open(split_path) as f:
            split = json.load(f)
        if is_rank0:
            print(
                f"[train] Loaded split from {split_path}: {len(split)} datasets, "
                f"{sum(len(v['train']) for v in split.values())} train, "
                f"{sum(len(v['test']) for v in split.values())} test"
            )
    else:
        split_path = Path(getattr(cfg, "biflow_dir", "")) / f"split_seed{cfg.split_seed}.json"
        split = load_or_create_split(
            cfg.data_dir, manifest, cfg.test_ratio, cfg.split_seed,
            biflow_dir=getattr(cfg, "biflow_dir", None),
        )

    iid_split = {ds: sp for ds, sp in split.items() if len(sp.get("train", [])) > 0}
    ood_split = {ds: sp for ds, sp in split.items() if len(sp.get("train", [])) == 0}
    has_ood = len(ood_split) > 0

    _ds_kw = _cross_dataset_kw(cfg)
    train_ds = CrossDatasetFMDataset(
        cfg.data_dir,
        iid_split,
        cfg.batch_size,
        cfg.seed,
        mode="train",
        min_cells=cfg.min_cells,
        ds_alpha=cfg.ds_alpha,
        scale_noise=cfg.scale_noise,
        min_selected_conditions_per_dataset=int(
            getattr(cfg, "min_selected_conditions_per_dataset", 0) or 0
        ),
        condition_visit_power=float(getattr(cfg, "condition_visit_power", 1.0) or 1.0),
        condition_visit_cap=int(getattr(cfg, "condition_visit_cap", 0) or 0),
        perturbation_family_filter=str(getattr(cfg, "perturbation_family_filter", "all") or "all"),
        ddp_rank=rank,
        ddp_world_size=world_size,
        ddp_sync_min_len=True,
        silent=not is_rank0,
        **_ds_kw,
    )
    train_eval_enabled = bool(getattr(cfg, "train_eval_enabled", True))
    test_ds = None
    if is_rank0 and train_eval_enabled:
        test_ds = CrossDatasetFMDataset(
            cfg.data_dir,
            iid_split,
            cfg.batch_size,
            cfg.seed,
            mode="test",
            min_cells=16,
            ds_alpha=1.0,
            perturbation_family_filter=str(getattr(cfg, "perturbation_family_filter", "all") or "all"),
            ddp_rank=0,
            ddp_world_size=1,
            silent=False,
            **_ds_kw,
        )

    if is_rank0:
        print(f"Train conditions: {train_ds.total_conditions}  epoch_steps: ~{train_ds.epoch_steps}")
        if train_eval_enabled:
            assert test_ds is not None
            print(f"Test  conditions (IID): {test_ds.total_conditions}")
        else:
            print("Test  conditions (IID): skipped because train_eval_enabled=False")
        if has_ood and train_eval_enabled:
            ood_cond_count = sum(len(sp["test"]) for sp in ood_split.values())
            print(f"OOD  datasets: {sorted(ood_split.keys())}  total conditions: {ood_cond_count}")
            print(f"  -> OOD evaluation will run ONCE after training completes")

    # ── model ─────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── resume / finetune from external checkpoint ───────────────
    start_step = 0
    metric_mode = selection_metric_direction(cfg.selection_metric)
    best_score = float("inf") if metric_mode == "min" else float("-inf")
    ckpt_path = save_dir / "latest.pt"
    init_ckpt = (getattr(cfg, "init_checkpoint", None) or "").strip()
    if init_ckpt:
        init_path = Path(init_ckpt).expanduser()
        if not init_path.is_absolute():
            init_path = init_path.resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        missing_keys, unexpected_keys, skipped_shape_mismatch = load_model_weights_only(
            init_path,
            model,
            device,
            strict=False,
            prefer_ema=bool(getattr(cfg, "init_checkpoint_use_ema", False)),
        )
        if is_rank0:
            print(
                "[train] Finetune: loaded "
                f"{'EMA' if bool(getattr(cfg, 'init_checkpoint_use_ema', False)) else 'model'} "
                f"weights from {init_path}; "
                f"fresh optimizer, step=0; not loading {ckpt_path}"
            )
            if missing_keys:
                print(f"[train] Finetune: missing newly initialized keys: {missing_keys}")
            if unexpected_keys:
                print(f"[train] Finetune: unexpected checkpoint keys ignored: {unexpected_keys}")
            if skipped_shape_mismatch:
                print(
                    "[train] Finetune: skipped shape-mismatched checkpoint keys: "
                    f"{skipped_shape_mismatch}"
                )
        apply_finetune_freeze(model, cfg)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters left after finetune freeze; check config")
        optimizer = _build_latent_optimizer(model, cfg, rank=rank, verbose=is_rank0)
    elif ckpt_path.exists():
        optimizer = _build_latent_optimizer(model, cfg, rank=rank, verbose=is_rank0)
        start_step, best_score = load_checkpoint(ckpt_path, model, optimizer, device)
        best_score = recover_best_score_from_best_checkpoint(
            latest_score=best_score,
            best_path=save_dir / "best.pt",
            metric_name=cfg.selection_metric,
            device=device,
        )
        if is_rank0:
            print(f"Resumed from step {start_step}, best_score={best_score:.6f}")
    else:
        apply_finetune_freeze(model, cfg)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters left after finetune freeze; check config")
        optimizer = _build_latent_optimizer(model, cfg, rank=rank, verbose=is_rank0)

    if ddp_active:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    anchor_replay_model = None
    if anchor_replay_loss_schedule(10**12, cfg) > 0:
        replay_ckpt_s = str(getattr(cfg, "anchor_replay_checkpoint", "") or init_ckpt or "").strip()
        if not replay_ckpt_s:
            raise ValueError(
                "anchor_replay_loss_weight > 0 requires anchor_replay_checkpoint "
                "or init_checkpoint"
            )
        replay_ckpt = Path(replay_ckpt_s).expanduser()
        if not replay_ckpt.is_absolute():
            replay_ckpt = replay_ckpt.resolve()
        if not replay_ckpt.is_file():
            raise FileNotFoundError(f"anchor_replay_checkpoint not found: {replay_ckpt}")
        anchor_cfg = dataclasses.replace(cfg, pert_pairwise_mode="off", anchor_replay_loss_weight=0.0)
        anchor_replay_model = build_model(anchor_cfg, device)
        missing_keys, unexpected_keys, skipped_shape_mismatch = load_model_weights_only(
            replay_ckpt,
            anchor_replay_model,
            device,
            strict=False,
            prefer_ema=bool(getattr(cfg, "anchor_replay_checkpoint_use_ema", False)),
        )
        anchor_replay_model.eval()
        for p in anchor_replay_model.parameters():
            p.requires_grad = False
        if is_rank0:
            print(
                "[train] Anchor replay: loaded frozen "
                f"{'EMA' if bool(getattr(cfg, 'anchor_replay_checkpoint_use_ema', False)) else 'model'} "
                f"anchor from {replay_ckpt}"
            )
            if missing_keys:
                print(f"[train] Anchor replay: missing newly initialized keys: {missing_keys}")
            if unexpected_keys:
                print(f"[train] Anchor replay: unexpected checkpoint keys ignored: {unexpected_keys}")
            if skipped_shape_mismatch:
                print(
                    "[train] Anchor replay: skipped shape-mismatched checkpoint keys: "
                    f"{skipped_shape_mismatch}"
                )

    # ── EMA (decay=0.999 by default) ──────────────────────────────
    ema = None
    if getattr(cfg, "use_ema", False):
        ema = ModelEMA(
            model,
            decay=cfg.ema_decay,
            update_after=cfg.ema_update_after,
            update_every=cfg.ema_update_every,
            device=device,
        )
        # resume 路径：load_checkpoint 已在上方处理 init_ckpt 的 finetune；
        # 这里在 resume (ckpt_path.exists) 后再尝试 load EMA state（若此前没 load）。
        if ckpt_path.exists() and not init_ckpt:
            try:
                _resume = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                if "ema" in _resume:
                    ema.load_state_dict(_resume["ema"], strict=False)
                    if is_rank0:
                        print(f"[train] Resumed EMA state from {ckpt_path}")
                del _resume
            except Exception as e:  # pragma: no cover
                if is_rank0:
                    print(f"[train] WARNING: failed to resume EMA state ({e})")

    ot_sampler = OTPlanSampler(
        method=cfg.ot_method,
        reg=getattr(cfg, "ot_sinkhorn_reg", 0.05),
        num_threads=cfg.ot_threads,
        n_iter=getattr(cfg, "ot_sinkhorn_iter", 50),
    )
    fm_path = CondOTPath()

    if is_rank0:
        print(f"Model: {cfg.model_type}  params: {count_params(_unwrap_model(model))}")
        if ddp_active:
            print(f"[train] DDP world_size={world_size} rank={rank} local_rank={local_rank}")

    # ── logging (rank 0 writes train.log) ─────────────────────────
    if is_rank0:
        log_f = open(log_path, "a")

        def log(msg):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            log_f.write(line + "\n")
            log_f.flush()
    else:
        log_f = None

        def log(_msg):  # noqa: ARG001
            pass

    log(f"Config: {cfg}")
    log(f"Device: {device}")
    log(f"Save dir: {save_dir}")
    log(f"Train conditions: {train_ds.total_conditions}  epoch_steps: ~{train_ds.epoch_steps}")
    log(
        f"OT microbatch={cfg.batch_size}  grad_accum_steps={cfg.grad_accum_steps}  "
        f"effective_update_batch={cfg.batch_size * max(1, int(cfg.grad_accum_steps))}"
    )
    log(
        f"Train sampling: ds_alpha={cfg.ds_alpha}  "
        f"min_selected_conditions_per_dataset={getattr(cfg, 'min_selected_conditions_per_dataset', 0)}  "
        f"condition_visit_power={getattr(cfg, 'condition_visit_power', 1.0)}  "
        f"condition_visit_cap={getattr(cfg, 'condition_visit_cap', 0)}"
    )
    log(
        f"Test  conditions (IID): {test_ds.total_conditions if test_ds else 0}  "
        f"datasets: {sorted(iid_split.keys())}"
    )
    if has_ood:
        log(f"OOD  datasets: {sorted(ood_split.keys())}  total_test: {sum(len(sp['test']) for sp in ood_split.values())}")
        log(f"  -> OOD evaluated ONCE after training; early stopping uses IID only")
    log(f"Model: {cfg.model_type}  params: {count_params(_unwrap_model(model))}")
    log(f"Selection metric: {cfg.selection_metric} ({metric_mode})")
    if getattr(cfg, "mmd_dataset_filter", ""):
        log(f"MMD dataset filter: {cfg.mmd_dataset_filter}")
    if getattr(cfg, "anchor_replay_dataset_filter", ""):
        log(f"Anchor replay dataset filter: {cfg.anchor_replay_dataset_filter}")
    if init_ckpt:
        log(f"Finetune init_checkpoint: {Path(init_ckpt).expanduser().resolve()}")

    # ── save config ───────────────────────────────────────────────
    if is_rank0:
        with open(save_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)

    # ── precomputed means for Pearson metrics ──────────────────────
    ctrl_means_path = Path(cfg.data_dir) / "ctrl_means.npz"
    ctrl_means = None
    if ctrl_means_path.exists():
        ctrl_means = {k: v for k, v in np.load(str(ctrl_means_path)).items()}
        log(f"Loaded ctrl_means for {len(ctrl_means)} datasets from {ctrl_means_path}")
    else:
        log(f"WARNING: {ctrl_means_path} not found, pearson_ctrl disabled")

    pert_means_override = str(getattr(cfg, "pert_means_file", "") or "").strip()
    pert_means_path = Path(pert_means_override).expanduser() if pert_means_override else Path(cfg.data_dir) / "pert_means.npz"
    if pert_means_override and not pert_means_path.is_absolute():
        pert_means_path = pert_means_path.resolve()
    pert_means = None
    if pert_means_path.exists():
        pert_means = {k: v for k, v in np.load(str(pert_means_path)).items()}
        log(f"Loaded pert_means for {len(pert_means)} datasets from {pert_means_path}")
    else:
        log(f"WARNING: {pert_means_path} not found, pearson_pert disabled")
    pert_means_t = {}
    if pert_means:
        pert_means_t = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in pert_means.items()
        }

    response_normalizer = maybe_load_response_normalizer(
        cfg,
        split_path=split_path,
        device=device,
        log=log,
    )

    # ── per-dataset loss weights ──────────────────────────────────
    ds_weights = {}
    if cfg.ds_loss_alpha > 0:
        log(
            "Dataset loss weighting enabled: "
            f"ds_loss_alpha={cfg.ds_loss_alpha} "
            f"ds_loss_warmup_start={getattr(cfg, 'ds_loss_warmup_start', 0)}"
        )
        batch_counts = {}
        for ds in train_ds.ds_names:
            sizes = train_ds._cond_sizes[ds]
            n_eff = train_ds._n_eff(len(sizes))
            avg_visits = sum(
                train_ds._condition_visits(s[1]) for s in sizes.values()
            ) / max(len(sizes), 1)
            batch_counts[ds] = max(1, int(n_eff * avg_visits))
        inv_freq = {ds: (1.0 / c) ** cfg.ds_loss_alpha for ds, c in batch_counts.items()}
        mean_w = sum(inv_freq.values()) / len(inv_freq)
        ds_weights = {ds: w / mean_w for ds, w in inv_freq.items()}
        for ds in sorted(ds_weights):
            log(f"  ds_weight[{ds}] = {ds_weights[ds]:.4f}  (batches/epoch ~ {batch_counts[ds]})")

    condition_loss_weights: dict[tuple[str, str], float] = {}
    condition_loss_weight_path = str(getattr(cfg, "condition_loss_weight_file", "") or "").strip()
    if condition_loss_weight_path:
        raw_condition_loss_weights = load_condition_loss_weight_table(
            condition_loss_weight_path,
            weight_column=str(getattr(cfg, "condition_loss_weight_column", "weight") or "weight"),
        )
        matched_weights = []
        missing_weight_rows = 0
        for ds in train_ds.ds_names:
            for cond in train_ds.ds_conds[ds]:
                w = float(raw_condition_loss_weights.get((ds, cond), 1.0))
                condition_loss_weights[(ds, cond)] = w
                matched_weights.append(w)
                if (ds, cond) not in raw_condition_loss_weights:
                    missing_weight_rows += 1
        if matched_weights and bool(getattr(cfg, "condition_loss_weight_normalize_mean", True)):
            mean_w = float(sum(matched_weights) / len(matched_weights))
            if mean_w <= 0 or not math.isfinite(mean_w):
                raise ValueError("condition loss weights have non-positive/non-finite train mean")
            condition_loss_weights = {
                key: float(value) / mean_w
                for key, value in condition_loss_weights.items()
            }
            matched_weights = [float(value) / mean_w for value in matched_weights]
        arr = np.asarray(matched_weights, dtype=np.float64)
        log(
            "Condition loss weighting enabled: "
            f"file={condition_loss_weight_path} "
            f"column={getattr(cfg, 'condition_loss_weight_column', 'weight')} "
            f"matched_train={len(matched_weights)} missing_default1={missing_weight_rows} "
            f"mean={float(arr.mean()):.4f} min={float(arr.min()):.4f} "
            f"median={float(np.median(arr)):.4f} max={float(arr.max()):.4f}"
        )

    # ── training loop ─────────────────────────────────────────────
    model.train()
    global_step = start_step
    epoch = 0
    no_improve = 0
    grad_accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    residual_bank_size = max(0, int(getattr(cfg, "pert_residual_contrastive_bank_size", 256) or 0))
    residual_bank = deque(maxlen=residual_bank_size)
    residual_min_norm = float(getattr(cfg, "pert_residual_contrastive_min_norm", 1e-6) or 1e-6)
    composition_bank_size = max(0, int(getattr(cfg, "composition_delta_bank_size", 512) or 0))
    composition_bank = defaultdict(lambda: deque(maxlen=composition_bank_size))
    composition_min_norm = float(getattr(cfg, "composition_delta_min_norm", 1e-6) or 1e-6)
    risk_row_cvar_state = RiskRowCvarTailState(
        history_size=int(getattr(cfg, "risk_row_cvar_history_size", 256) or 256),
        min_history=int(getattr(cfg, "risk_row_cvar_min_history", 8) or 8),
        top_frac=float(getattr(cfg, "risk_row_cvar_top_frac", 0.20) or 0.20),
        threshold=float(getattr(cfg, "risk_row_cvar_mmd_threshold", 0.005) or 0.005),
    )
    condition_prior_delta_bank = build_condition_prior_delta_bank(train_ds, cfg, log=log if is_rank0 else None)
    trackc_routed_distill_bank = build_trackc_routed_distill_bank(train_ds, cfg, log=log if is_rank0 else None)
    trackc_support_context_bank = trackc_routed_distill_bank if _support_context_source_active(cfg) else {}
    trackc_support_set_task_bank = build_trackc_support_set_task_bank(cfg, log=log if is_rank0 else None)
    configure_condition_delta_prior_gate(
        model,
        train_ds,
        condition_prior_delta_bank,
        cfg,
        log=log if is_rank0 else None,
    )
    if is_rank0 and LAST_CONDITION_PRIOR_BANK_SUMMARY:
        summary_path = save_dir / "condition_prior_bank_summary.json"
        summary_path.write_text(
            json.dumps(LAST_CONDITION_PRIOR_BANK_SUMMARY, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log(f"Condition-prior bank summary saved to {summary_path}")
    if is_rank0 and LAST_TRACKC_ROUTED_DISTILL_SUMMARY:
        summary_path = save_dir / "trackc_routed_distill_bank_summary.json"
        summary_path.write_text(
            json.dumps(LAST_TRACKC_ROUTED_DISTILL_SUMMARY, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log(f"Track C routed distill bank summary saved to {summary_path}")
    if is_rank0 and LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY:
        summary_path = save_dir / "trackc_support_set_task_bank_summary.json"
        summary_path.write_text(
            json.dumps(LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log(f"Track C support-set task bank summary saved to {summary_path}")
    if _support_context_source_active(cfg):
        log(
            "Track C support-context source active: "
            f"{getattr(cfg, 'trackc_support_context_source', '')}; "
            f"context_dim={getattr(cfg, 'trackc_support_context_dim', 0)}"
        )
    if _support_set_task_source_active(cfg):
        log(
            "Track C support-set task source active: "
            f"{getattr(cfg, 'trackc_support_set_task_source', '')}; "
            f"task_dim={getattr(cfg, 'trackc_support_set_task_dim', 0)} "
            f"eval_control={getattr(cfg, 'trackc_support_set_task_eval_control', 'actual')}"
        )

    stopped_early = False
    try:
        while global_step < cfg.total_steps and not stopped_early:
            epoch += 1
            run_total = torch.zeros((), device=device, dtype=torch.float64)
            run_mse = torch.zeros((), device=device, dtype=torch.float64)
            run_mmd = torch.zeros((), device=device, dtype=torch.float64)
            run_direction = torch.zeros((), device=device, dtype=torch.float64)
            run_endpoint_delta = torch.zeros((), device=device, dtype=torch.float64)
            run_response_geometry = torch.zeros((), device=device, dtype=torch.float64)
            run_pert_residual_direction = torch.zeros((), device=device, dtype=torch.float64)
            run_pert_residual_contrastive = torch.zeros((), device=device, dtype=torch.float64)
            run_pert_residual_relational = torch.zeros((), device=device, dtype=torch.float64)
            run_composition_delta = torch.zeros((), device=device, dtype=torch.float64)
            run_condition_prior_delta = torch.zeros((), device=device, dtype=torch.float64)
            run_condition_prior_additive_delta = torch.zeros((), device=device, dtype=torch.float64)
            run_trackc_routed_distill = torch.zeros((), device=device, dtype=torch.float64)
            run_trackc_routed_endpoint = torch.zeros((), device=device, dtype=torch.float64)
            run_anchor_replay = torch.zeros((), device=device, dtype=torch.float64)
            run_condition_delta_head = torch.zeros((), device=device, dtype=torch.float64)
            run_additive_condition_delta = torch.zeros((), device=device, dtype=torch.float64)
            run_risk_row_cvar_weight = torch.zeros((), device=device, dtype=torch.float64)
            risk_row_cvar_observe_count = 0
            risk_row_cvar_apply_count = 0
            epoch_steps = 0
            t0 = time.time()
            optimizer.zero_grad()
            accum_count = 0

            prefetch_iter = OTPrefetchIter(train_ds, ot_sampler,
                                           prefetch_n=cfg.prefetch,
                                           n_ot_workers=cfg.n_ot_workers,
                                           device=device,
                                           pair_mode=getattr(cfg, "ot_pair_mode", "multinomial"))
            for src_paired, gt_paired, ds_name, cond, perturbation_batch in prefetch_iter:
                if global_step >= cfg.total_steps:
                    break

                cur_lr = lr_warmup_cosine_to_eta_min(global_step, cfg.warmup_steps, cfg.lr_decay_steps, cfg.lr, cfg.eta_min)
                set_lr(optimizer, cur_lr, cfg)

                cur_gamma = gamma_schedule(global_step, cfg)
                if cfg.mmd_every > 1 and global_step % cfg.mmd_every != 0:
                    cur_gamma = 0.0
                cur_direction_weight = direction_loss_schedule(global_step, cfg)
                cur_endpoint_delta_weight = endpoint_delta_loss_schedule(global_step, cfg)
                cur_response_geometry_weight = response_geometry_loss_schedule(global_step, cfg)
                cur_pert_residual_direction_weight = pert_residual_direction_loss_schedule(global_step, cfg)
                cur_pert_residual_contrastive_weight = pert_residual_contrastive_loss_schedule(global_step, cfg)
                cur_pert_residual_relational_weight = pert_residual_relational_loss_schedule(global_step, cfg)
                cur_composition_delta_weight = composition_delta_loss_schedule(global_step, cfg)
                cur_condition_prior_delta_weight = condition_prior_delta_loss_schedule(global_step, cfg)
                cur_condition_prior_additive_delta_weight = condition_prior_additive_delta_loss_schedule(global_step, cfg)
                cur_trackc_routed_distill_weight = trackc_routed_distill_loss_schedule(global_step, cfg)
                cur_trackc_routed_endpoint_weight = trackc_routed_endpoint_loss_schedule(global_step, cfg)
                cur_anchor_replay_weight = anchor_replay_loss_schedule(global_step, cfg)
                cur_condition_delta_head_weight = condition_delta_head_loss_schedule(global_step, cfg)
                cur_additive_condition_delta_weight = additive_condition_delta_loss_schedule(global_step, cfg)
                cur_risk_row_cvar_observe, cur_risk_row_cvar_weight = (
                    risk_row_cvar_batch_control(
                        global_step,
                        cfg,
                        risk_row_cvar_state,
                        ds_name,
                        cond,
                    )
                )
                need_pert_mean = (
                    cur_pert_residual_direction_weight > 0
                    or cur_pert_residual_contrastive_weight > 0
                    or cur_pert_residual_relational_weight > 0
                    or (
                        cur_condition_delta_head_weight > 0
                        and str(getattr(cfg, "condition_delta_head_target", "endpoint_delta") or "endpoint_delta")
                        .strip()
                        .lower()
                        == "pert_residual"
                    )
                )
                pert_mean_ref = pert_means_t.get(ds_name) if need_pert_mean else None
                residual_bank_t = None
                if (
                    (cur_pert_residual_contrastive_weight > 0 or cur_pert_residual_relational_weight > 0)
                    and len(residual_bank) > 0
                ):
                    residual_bank_t = torch.stack(list(residual_bank), dim=0)
                composition_delta_target = None
                composition_perturbation_batch = None
                composition_key = None
                current_delta = None
                condition_prior_delta_target = None
                condition_prior_perturbation_batch = None
                trackc_routed_target = None
                support_context = None
                support_set_task = None
                support_set_task_present = None
                if (
                    (cur_composition_delta_weight > 0 or cur_additive_condition_delta_weight > 0)
                    and composition_bank_size > 0
                    and train_ds.gene_embedding_cache is not None
                ):
                    meta = train_ds.metadata_for_condition(ds_name, cond)
                    composition_key = _single_gene_composition_key(meta)
                    bank = composition_bank.get(ds_name)
                    if composition_key is not None and bank:
                        candidates = [
                            rec for rec in bank
                            if rec[0] != composition_key[0]
                        ]
                        if candidates:
                            pick = _stable_int_hash(f"{global_step}:{ds_name}:{cond}") % len(candidates)
                            gene_b, ptype_b, delta_b = candidates[pick]
                            current_delta = (gt_paired.float() - src_paired.float()).mean(dim=0).detach()
                            composition_delta_target = current_delta + delta_b.to(device=device)
                            ptype = composition_key[1] if composition_key[1] is not None else ptype_b
                            composition_perturbation_batch = _make_gene_combo_perturbation_batch(
                                genes=(composition_key[0], gene_b),
                                perturbation_type_raw=ptype,
                                batch_size=int(src_paired.size(0)),
                                cache=train_ds.gene_embedding_cache,
                                max_genes=int(getattr(cfg, "max_pert_genes", 16)),
                                max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
                            )
                if (
                    (cur_condition_prior_delta_weight > 0 or cur_condition_prior_additive_delta_weight > 0)
                    and condition_prior_delta_bank
                    and train_ds.gene_embedding_cache is not None
                ):
                    condition_prior_delta_target, condition_prior_perturbation_batch = sample_condition_prior_teacher(
                        bank=condition_prior_delta_bank,
                        ds_name=ds_name,
                        step=global_step,
                        cond=cond,
                        batch_size=int(src_paired.size(0)),
                        cache=train_ds.gene_embedding_cache,
                        max_genes=int(getattr(cfg, "max_pert_genes", 16)),
                        max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
                        num_genes=int(getattr(cfg, "condition_prior_num_genes", 2)),
                    )
                if (
                    (cur_trackc_routed_distill_weight > 0 or cur_trackc_routed_endpoint_weight > 0)
                    and trackc_routed_distill_bank
                ):
                    meta = train_ds.metadata_for_condition(ds_name, cond)
                    trackc_routed_target = get_trackc_routed_distill_target(
                        trackc_routed_distill_bank,
                        ds_name,
                        meta,
                    )
                if _support_context_source_active(cfg):
                    support_context = make_trackc_support_context_batch(
                        trackc_support_context_bank,
                        train_ds,
                        ds_name,
                        cond,
                        int(src_paired.size(0)),
                        cfg,
                        device,
                    )
                if _support_set_task_source_active(cfg):
                    support_set_task, support_set_task_present = make_trackc_support_set_task_batch(
                        trackc_support_set_task_bank,
                        ds_name,
                        cond,
                        int(src_paired.size(0)),
                        cfg,
                        device,
                    )

                out = train_step(
                    src_paired,
                    gt_paired,
                    model,
                    fm_path,
                    cfg,
                    device,
                    ds_name=ds_name,
                    gamma_t=cur_gamma,
                    direction_weight_t=cur_direction_weight,
                    endpoint_delta_weight_t=cur_endpoint_delta_weight,
                    response_geometry_weight_t=cur_response_geometry_weight,
                    response_normalizer=response_normalizer,
                    pert_residual_direction_weight_t=cur_pert_residual_direction_weight,
                    pert_residual_contrastive_weight_t=cur_pert_residual_contrastive_weight,
                    pert_residual_relational_weight_t=cur_pert_residual_relational_weight,
                    pert_residual_contrastive_bank=residual_bank_t,
                    pert_mean_ref=pert_mean_ref,
                    composition_delta_weight_t=cur_composition_delta_weight,
                    composition_delta_target=composition_delta_target,
                    composition_perturbation_batch=composition_perturbation_batch,
                    condition_prior_delta_weight_t=cur_condition_prior_delta_weight,
                    condition_prior_delta_target=condition_prior_delta_target,
                    condition_prior_perturbation_batch=condition_prior_perturbation_batch,
                    condition_prior_additive_delta_weight_t=cur_condition_prior_additive_delta_weight,
                    trackc_routed_distill_weight_t=cur_trackc_routed_distill_weight,
                    trackc_routed_distill_target=trackc_routed_target,
                    trackc_routed_endpoint_weight_t=cur_trackc_routed_endpoint_weight,
                    anchor_replay_weight_t=cur_anchor_replay_weight,
                    anchor_model=anchor_replay_model,
                    condition_delta_head_weight_t=cur_condition_delta_head_weight,
                    additive_condition_delta_weight_t=cur_additive_condition_delta_weight,
                    risk_row_cvar_weight_t=cur_risk_row_cvar_weight,
                    risk_row_cvar_observe=cur_risk_row_cvar_observe,
                    perturbation_batch=perturbation_batch,
                    support_context=support_context,
                    support_set_task=support_set_task,
                    support_set_task_present=support_set_task_present,
                )
                loss = out["loss"]
                if ds_weights and dataset_loss_weights_active(global_step, cfg):
                    loss = loss * ds_weights.get(ds_name, 1.0)
                if condition_loss_weights:
                    loss = loss * float(condition_loss_weights.get((ds_name, cond), 1.0))
                loss = loss / float(grad_accum_steps)
                loss.backward()
                accum_count += 1
                epoch_steps += 1
                run_total += out["total"].detach().to(torch.float64)
                run_mse += out["mse"].detach().to(torch.float64)
                run_mmd += out["mmd"].detach().to(torch.float64)
                run_direction += out["direction"].detach().to(torch.float64)
                run_endpoint_delta += out["endpoint_delta"].detach().to(torch.float64)
                run_response_geometry += out["response_geometry"].detach().to(torch.float64)
                run_pert_residual_direction += out["pert_residual_direction"].detach().to(torch.float64)
                run_pert_residual_contrastive += out["pert_residual_contrastive"].detach().to(torch.float64)
                run_pert_residual_relational += out["pert_residual_relational"].detach().to(torch.float64)
                run_composition_delta += out["composition_delta"].detach().to(torch.float64)
                run_condition_prior_delta += out["condition_prior_delta"].detach().to(torch.float64)
                run_condition_prior_additive_delta += out["condition_prior_additive_delta"].detach().to(torch.float64)
                run_trackc_routed_distill += out["trackc_routed_distill"].detach().to(torch.float64)
                run_trackc_routed_endpoint += out["trackc_routed_endpoint"].detach().to(torch.float64)
                run_anchor_replay += out["anchor_replay"].detach().to(torch.float64)
                run_condition_delta_head += out["condition_delta_head"].detach().to(torch.float64)
                run_additive_condition_delta += out["additive_condition_delta"].detach().to(torch.float64)
                run_risk_row_cvar_weight += out["risk_row_cvar_weight"].detach().to(torch.float64)
                if cur_risk_row_cvar_observe:
                    risk_row_cvar_observe_count += 1
                    if cur_risk_row_cvar_weight > 0:
                        risk_row_cvar_apply_count += 1
                    risk_row_cvar_state.update(ds_name, cond, float(out["mmd"].detach().float().cpu().item()))
                target_residual = out.get("target_residual")
                if target_residual is not None and residual_bank_size > 0:
                    target_cpu = target_residual.detach().float().cpu()
                    if torch.isfinite(target_cpu).all() and target_cpu.norm().item() > residual_min_norm:
                        residual_bank.append(target_cpu)
                if composition_key is not None and composition_bank_size > 0:
                    if current_delta is None:
                        current_delta = (gt_paired.float() - src_paired.float()).mean(dim=0).detach()
                    current_delta_cpu = current_delta.float().cpu()
                    if torch.isfinite(current_delta_cpu).all() and current_delta_cpu.norm().item() > composition_min_norm:
                        composition_bank[ds_name].append((
                            composition_key[0],
                            composition_key[1],
                            current_delta_cpu,
                        ))

                if accum_count >= grad_accum_steps:
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                    optimizer.step()

                    # ── EMA update (after optimizer.step) ─────────────
                    if ema is not None:
                        ema.update(model, step=global_step)

                    global_step += 1
                    accum_count = 0
                    optimizer.zero_grad()

                    if is_rank0 and global_step % cfg.print_every == 0:
                        n = epoch_steps
                        gamma_sched = gamma_schedule(global_step, cfg)
                        direction_sched = direction_loss_schedule(global_step, cfg)
                        endpoint_delta_sched = endpoint_delta_loss_schedule(global_step, cfg)
                        response_geometry_sched = response_geometry_loss_schedule(global_step, cfg)
                        pert_residual_direction_sched = pert_residual_direction_loss_schedule(global_step, cfg)
                        pert_residual_contrastive_sched = pert_residual_contrastive_loss_schedule(global_step, cfg)
                        pert_residual_relational_sched = pert_residual_relational_loss_schedule(global_step, cfg)
                        composition_delta_sched = composition_delta_loss_schedule(global_step, cfg)
                        condition_prior_delta_sched = condition_prior_delta_loss_schedule(global_step, cfg)
                        condition_prior_additive_delta_sched = condition_prior_additive_delta_loss_schedule(global_step, cfg)
                        trackc_routed_distill_sched = trackc_routed_distill_loss_schedule(global_step, cfg)
                        trackc_routed_endpoint_sched = trackc_routed_endpoint_loss_schedule(global_step, cfg)
                        anchor_replay_sched = anchor_replay_loss_schedule(global_step, cfg)
                        condition_delta_head_sched = condition_delta_head_loss_schedule(global_step, cfg)
                        additive_condition_delta_sched = additive_condition_delta_loss_schedule(global_step, cfg)
                        log(
                            f"step={global_step}  epoch={epoch}  microsteps={epoch_steps}  "
                            f"avg_loss={(run_total/n).item():.6f}  "
                            f"avg_mse={(run_mse/n).item():.6f}  "
                            f"avg_mmd={(run_mmd/n).item():.6f}  "
                            f"avg_dir={(run_direction/n).item():.6f}  "
                            f"avg_delta={(run_endpoint_delta/n).item():.6f}  "
                            f"avg_resp_geom={(run_response_geometry/n).item():.6f}  "
                            f"avg_pert_resid={(run_pert_residual_direction/n).item():.6f}  "
                            f"avg_pert_ctr={(run_pert_residual_contrastive/n).item():.6f}  "
                            f"avg_pert_rel={(run_pert_residual_relational/n).item():.6f}  "
                            f"avg_comp={(run_composition_delta/n).item():.6f}  "
                            f"avg_prior_delta={(run_condition_prior_delta/n).item():.6f}  "
                            f"avg_prior_add_delta={(run_condition_prior_additive_delta/n).item():.6f}  "
                            f"avg_trackc_route={(run_trackc_routed_distill/n).item():.6f}  "
                            f"avg_trackc_endpoint={(run_trackc_routed_endpoint/n).item():.6f}  "
                            f"avg_anchor_replay={(run_anchor_replay/n).item():.6f}  "
                            f"avg_cond_delta={(run_condition_delta_head/n).item():.6f}  "
                            f"avg_add_cond_delta={(run_additive_condition_delta/n).item():.6f}  "
                            f"avg_risk_row_cvar_w={(run_risk_row_cvar_weight/n).item():.6f}  "
                            f"risk_row_obs={risk_row_cvar_observe_count}  "
                            f"risk_row_apply={risk_row_cvar_apply_count}  "
                            f"γ={gamma_sched:.4f}  "
                            f"λdir={direction_sched:.4f}  "
                            f"λdelta={endpoint_delta_sched:.4f}  "
                            f"λresp_geom={response_geometry_sched:.4f}  "
                            f"λpert_resid={pert_residual_direction_sched:.4f}  "
                            f"λpert_ctr={pert_residual_contrastive_sched:.4f}  "
                            f"λpert_rel={pert_residual_relational_sched:.4f}  "
                            f"λcomp={composition_delta_sched:.4f}  "
                            f"λprior_delta={condition_prior_delta_sched:.4f}  "
                            f"λprior_add_delta={condition_prior_additive_delta_sched:.4f}  "
                            f"λtrackc_route={trackc_routed_distill_sched:.4f}  "
                            f"λtrackc_endpoint={trackc_routed_endpoint_sched:.4f}  "
                            f"λanchor_replay={anchor_replay_sched:.4f}  "
                            f"λcond_delta={condition_delta_head_sched:.4f}  "
                            f"λadd_cond_delta={additive_condition_delta_sched:.4f}  "
                            f"ctr_bank={len(residual_bank)}  "
                            f"comp_bank={sum(len(v) for v in composition_bank.values())}"
                        )

                    ckpt_every = int(getattr(cfg, "eval_every", 0) or 0)
                    if ckpt_every > 0 and global_step % ckpt_every == 0:
                        save_checkpoint(
                            str(ckpt_path),
                            model,
                            optimizer,
                            global_step,
                            best_score,
                            ema=ema,
                            config_dict=dataclasses.asdict(cfg),
                        )
                        log(
                            f"  [checkpoint] saved latest.pt at step {global_step} "
                            f"(epoch-end eval/best selection unchanged)"
                        )

                # Mid-epoch eval was removed; full evaluation happens at epoch end only.

            if accum_count > 0 and global_step < cfg.total_steps:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model, step=global_step)
                global_step += 1
                optimizer.zero_grad()

            if ddp_active:
                dist.barrier()

            # ── end of epoch: full eval + best / early stop (rank 0 only) ─
            stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
            if is_rank0:
                dt = time.time() - t0
                n = max(epoch_steps, 1)
                log(
                    f"Epoch {epoch} done: {epoch_steps} steps  "
                    f"avg_loss={(run_total/n).item():.6f}  "
                    f"avg_mse={(run_mse/n).item():.6f}  "
                    f"avg_mmd={(run_mmd/n).item():.6f}  "
                    f"avg_dir={(run_direction/n).item():.6f}  "
                    f"avg_delta={(run_endpoint_delta/n).item():.6f}  "
                    f"avg_resp_geom={(run_response_geometry/n).item():.6f}  "
                    f"avg_pert_resid={(run_pert_residual_direction/n).item():.6f}  "
                    f"avg_pert_ctr={(run_pert_residual_contrastive/n).item():.6f}  "
                    f"avg_pert_rel={(run_pert_residual_relational/n).item():.6f}  "
                    f"avg_comp={(run_composition_delta/n).item():.6f}  "
                    f"avg_prior_delta={(run_condition_prior_delta/n).item():.6f}  "
                    f"avg_prior_add_delta={(run_condition_prior_additive_delta/n).item():.6f}  "
                    f"avg_trackc_route={(run_trackc_routed_distill/n).item():.6f}  "
                    f"avg_trackc_endpoint={(run_trackc_routed_endpoint/n).item():.6f}  "
                    f"avg_anchor_replay={(run_anchor_replay/n).item():.6f}  "
                    f"avg_cond_delta={(run_condition_delta_head/n).item():.6f}  "
                    f"avg_add_cond_delta={(run_additive_condition_delta/n).item():.6f}  "
                    f"avg_risk_row_cvar_w={(run_risk_row_cvar_weight/n).item():.6f}  "
                    f"risk_row_obs={risk_row_cvar_observe_count}  "
                    f"risk_row_apply={risk_row_cvar_apply_count}  "
                    f"{dt:.1f}s"
                )

                if not bool(getattr(cfg, "train_eval_enabled", True)):
                    save_checkpoint(
                        str(ckpt_path),
                        model,
                        optimizer,
                        global_step,
                        best_score,
                        ema=ema,
                        config_dict=dataclasses.asdict(cfg),
                    )
                    log(
                        "  [train-only] train_eval_enabled=False; skipped epoch IID eval, "
                        "best checkpoint selection, and patience update"
                    )
                    stop_flag[0] = 0
                else:
                    assert test_ds is not None
                # 使用 EMA 权重做评估（FM/diffusion 最佳 practice）
                    if ema is not None and global_step >= cfg.ema_update_after:
                        with ema.apply_to(model):
                            epoch_eval = evaluate(
                                model, test_ds, fm_path, cfg, device,
                                ctrl_means=ctrl_means, pert_means=pert_means,
                                trackc_support_context_bank=trackc_support_context_bank,
                                trackc_support_set_task_bank=trackc_support_set_task_bank,
                            )
                    else:
                        epoch_eval = evaluate(
                            model, test_ds, fm_path, cfg, device,
                            ctrl_means=ctrl_means, pert_means=pert_means,
                            trackc_support_context_bank=trackc_support_context_bank,
                            trackc_support_set_task_bank=trackc_support_set_task_bank,
                        )
                    log(
                        f"  [IID eval] epoch={epoch}  "
                        f"test_mse={epoch_eval['test_mse']:.6f}  test_mae={epoch_eval['test_mae']:.6f}  test_mmd={epoch_eval['test_mmd']:.6f}  "
                        f"dp={epoch_eval['direct_pearson']:.4f}  pc={epoch_eval['pearson_ctrl']:.4f}  "
                        f"pp={epoch_eval['pearson_pert']:.4f}  "
                        f"n_conds={epoch_eval['n_conds']}"
                    )
                    for ds_k in sorted(epoch_eval["per_ds_mmd"].keys()):
                        log(
                            f"    {ds_k}: mmd={epoch_eval['per_ds_mmd'][ds_k]:.6f}  "
                            f"dp={epoch_eval['per_ds_direct'].get(ds_k, float('nan')):.4f}  "
                            f"pc={epoch_eval['per_ds_p_ctrl'].get(ds_k, float('nan')):.4f}  "
                            f"pp={epoch_eval['per_ds_p_pert'].get(ds_k, float('nan')):.4f}"
                        )

                    save_checkpoint(str(ckpt_path), model, optimizer, global_step, best_score, ema=ema,
                                    config_dict=dataclasses.asdict(cfg))

                    in_warmup = cfg.use_mmd and global_step < cfg.gamma_warmup_end
                    metric_value = selection_metric_value(
                        cfg.selection_metric,
                        epoch_eval,
                        mmd_lambda=float(getattr(cfg, "selection_mmd_lambda", 1.0) or 0.0),
                    )

                    if is_better_score(cfg.selection_metric, metric_value, best_score):
                        best_score = metric_value
                        no_improve = 0
                        save_checkpoint(str(save_dir / "best.pt"), model, optimizer, global_step, best_score, ema=ema,
                                        config_dict=dataclasses.asdict(cfg))
                        log(f"  [best] epoch {epoch}  new best_{cfg.selection_metric}={best_score:.6f}")
                    elif in_warmup:
                        log(f"  [warmup] γ still ramping (step {global_step} < {cfg.gamma_warmup_end}), patience frozen")
                    else:
                        no_improve += 1
                        log(f"  [patience] {no_improve}/{cfg.patience} epochs without improvement")
                        if cfg.patience > 0 and no_improve >= cfg.patience:
                            log(f"  [early stop] no improvement for {cfg.patience} epochs, stopping")
                            stopped_early = True

                    stop_flag[0] = 1 if stopped_early else 0

            if ddp_active:
                dist.broadcast(stop_flag, src=0)
                stopped_early = bool(int(stop_flag.item()))

        if is_rank0:
            save_checkpoint(str(ckpt_path), model, optimizer, global_step, best_score, ema=ema,
                            config_dict=dataclasses.asdict(cfg))
            reason = "early stop" if stopped_early else "max steps"
            log(f"Training finished at step {global_step} ({reason}), best_{cfg.selection_metric}={best_score:.6f}")

        # ── Final IID evaluation on best checkpoint ──────────────────
        best_path = save_dir / "best.pt"
        if is_rank0 and not bool(getattr(cfg, "train_eval_enabled", True)):
            log("Final IID/OOD evaluation skipped because train_eval_enabled=False")
        elif is_rank0 and best_path.exists():
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ckpt_best = torch.load(str(best_path), map_location=device, weights_only=False)
            _unwrap_model(model).load_state_dict(ckpt_best["model"])
            log(f"Loaded best.pt (step={ckpt_best.get('step', '?')}) for final evaluation")
            # best.pt 里保存的 ema 直接用作最终评估权重
            final_ema = None
            if checkpoint_ema_is_active(ckpt_best, cfg) and ema is not None:
                try:
                    final_ema = ModelEMA(
                        model, decay=cfg.ema_decay,
                        update_after=cfg.ema_update_after, device=device,
                    )
                    final_ema.load_state_dict(ckpt_best["ema"], strict=False)
                    log("  [final] using EMA weights for IID / OOD evaluation")
                except Exception as e:  # pragma: no cover
                    log(f"  [final] EMA load failed ({e}); falling back to raw weights")
                    final_ema = None
            elif "ema" in ckpt_best and bool(getattr(cfg, "use_ema", False)):
                log(
                    "  [final] EMA state present but inactive for best checkpoint "
                    f"step={ckpt_best.get('step')} < ema_update_after={getattr(cfg, 'ema_update_after', 0)}; "
                    "using raw weights"
                )

            def _eval(ds):
                if final_ema is not None:
                    with final_ema.apply_to(model):
                        return evaluate(
                            model, ds, fm_path, cfg, device,
                            ctrl_means=ctrl_means, pert_means=pert_means,
                            max_chunk=int(getattr(cfg, "eval_max_chunk", 256) or 256),
                            trackc_support_context_bank=trackc_support_context_bank,
                            trackc_support_set_task_bank=trackc_support_set_task_bank,
                        )
                return evaluate(
                    model, ds, fm_path, cfg, device,
                    ctrl_means=ctrl_means, pert_means=pert_means,
                    max_chunk=int(getattr(cfg, "eval_max_chunk", 256) or 256),
                    trackc_support_context_bank=trackc_support_context_bank,
                    trackc_support_set_task_bank=trackc_support_set_task_bank,
                )

            iid_final = _eval(test_ds)
            log(
                f"  [IID final] "
                f"test_mse={iid_final['test_mse']:.6f}  test_mmd={iid_final['test_mmd']:.6f}  "
                f"dp={iid_final['direct_pearson']:.4f}  pc={iid_final['pearson_ctrl']:.4f}  "
                f"pp={iid_final['pearson_pert']:.4f}  "
                f"n_conds={iid_final['n_conds']}"
            )
            with open(save_dir / "iid_eval_results.json", "w") as f:
                json.dump(iid_final, f, indent=2)

            # ── OOD evaluation (once, only if OOD datasets exist) ────
            if has_ood:
                log("=" * 60)
                log(f"OOD evaluation — datasets: {sorted(ood_split.keys())}")
                test_ds_ood = CrossDatasetFMDataset(
                    cfg.data_dir,
                    ood_split,
                    cfg.batch_size,
                    cfg.seed,
                    mode="test",
                    min_cells=16,
                    ds_alpha=1.0,
                    perturbation_family_filter=str(getattr(cfg, "perturbation_family_filter", "all") or "all"),
                    **_cross_dataset_kw(cfg),
                )
                ood_eval = _eval(test_ds_ood)
                log(
                    f"  [OOD eval] "
                    f"test_mse={ood_eval['test_mse']:.6f}  test_mmd={ood_eval['test_mmd']:.6f}  "
                    f"dp={ood_eval['direct_pearson']:.4f}  pc={ood_eval['pearson_ctrl']:.4f}  "
                    f"pp={ood_eval['pearson_pert']:.4f}  "
                    f"n_conds={ood_eval['n_conds']}"
                )
                for ds_k in sorted(ood_eval["per_ds_mmd"].keys()):
                    log(
                        f"    [OOD] {ds_k}: mmd={ood_eval['per_ds_mmd'][ds_k]:.6f}  "
                        f"dp={ood_eval['per_ds_direct'].get(ds_k, float('nan')):.4f}  "
                        f"pc={ood_eval['per_ds_p_ctrl'].get(ds_k, float('nan')):.4f}  "
                        f"pp={ood_eval['per_ds_p_pert'].get(ds_k, float('nan')):.4f}"
                    )
                with open(save_dir / "ood_eval_results.json", "w") as f:
                    json.dump(ood_eval, f, indent=2)
                log(f"OOD results saved to {save_dir / 'ood_eval_results.json'}")
                test_ds_ood.close()

    finally:
        if log_f is not None:
            log_f.close()
        if ddp_active and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
