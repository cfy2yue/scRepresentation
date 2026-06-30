#!/usr/bin/env python3
"""Read-only xverse LatentFM condition-mechanism audit.

This script inspects the current xverse anchor checkpoint and condition encoder
algebra. It uses checkpoint weights, split metadata, condition metadata, and
pretrained condition embeddings only. It does not read held-out GT embeddings,
perturbed means, posthoc predictions, or outcome metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.train import build_model
from model.utils.conditioning.perturbation import ConditionMetadata, PerturbationBatch


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_RUN_DIR = ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_condition_mechanism_audit_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONDITION_MECHANISM_AUDIT_20260621.md"

GROUPS = ("train", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")
FOCUS = ("Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_dict(obj: dict[str, Any]) -> Config:
    fields = {f.name for f in dataclasses.fields(Config)}
    kwargs = {k: v for k, v in obj.items() if k in fields}
    return Config(**kwargs)


def safe_norm(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t.detach().float()).cpu().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    na = torch.linalg.vector_norm(aa)
    nb = torch.linalg.vector_norm(bb)
    if float(na) <= 1e-8 or float(nb) <= 1e-8:
        return None
    return float(torch.dot(aa, bb).div(na * nb).cpu().item())


def tensor_stats(t: torch.Tensor | None) -> dict[str, float] | None:
    if t is None:
        return None
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "norm": float(torch.linalg.vector_norm(x).item()),
        "abs_mean": float(x.abs().mean().item()),
        "abs_max": float(x.abs().max().item()),
    }


def condition_pb(
    dataset: CrossDatasetFMDataset,
    ds_name: str,
    cond: str,
    *,
    batch_size: int,
) -> tuple[tuple[torch.Tensor, ...], ConditionMetadata]:
    cache = dataset.gene_embedding_cache
    if cache is None:
        raise RuntimeError("dataset gene cache is missing")
    meta = dataset.metadata_for_condition(ds_name, cond)
    meta = dataset.enrich_metadata_with_chem(meta)
    rows = [meta] * int(batch_size)
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=int(dataset.max_pert_genes),
        max_chem_slots=int(dataset.max_chem_keys),
        device=torch.device("cpu"),
    )
    return pb.as_tuple_full(), meta


def single_gene_pb(
    dataset: CrossDatasetFMDataset,
    gene: str,
    meta: ConditionMetadata,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    cache = dataset.gene_embedding_cache
    if cache is None:
        raise RuntimeError("dataset gene cache is missing")
    one = ConditionMetadata(
        genes=(str(gene).strip().upper(),),
        perturbation_type_raw=getattr(meta, "perturbation_type_raw", None),
        combo_id=0,
        nperts_obs=1,
    )
    rows = [one] * int(batch_size)
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=int(dataset.max_pert_genes),
        max_chem_slots=int(dataset.max_chem_keys),
        device=torch.device("cpu"),
    )
    return pb.as_tuple_full()


def encoder_outputs(model: torch.nn.Module, pb: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    gid, mk, tid, npt, cid, ce, cm = pb
    inner = model.module if hasattr(model, "module") else model
    enc = inner.pert_encoder(
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=ce,
        chem_mask=cm,
    )
    proj = inner.pert_to_c(enc)
    return enc.detach(), proj.detach()


def row_for_condition(
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    ds_name: str,
    cond: str,
    group: str,
) -> dict[str, Any] | None:
    pb, meta = condition_pb(dataset, ds_name, cond, batch_size=1)
    genes = [str(g).strip().upper() for g in getattr(meta, "genes", ()) if str(g).strip()]
    gid, mk, tid, npt, _cid, _ce, cm = pb
    has_chem = bool(cm is not None and (cm > 0).any().item())
    enc, proj = encoder_outputs(model, pb)
    out: dict[str, Any] = {
        "group": group,
        "dataset": ds_name,
        "condition": cond,
        "genes": genes,
        "n_genes": len(genes),
        "nperts_tensor": int(npt.reshape(-1)[0].item()) if npt.numel() else 0,
        "pert_type_id": int(tid.reshape(-1)[0].item()) if tid.numel() else 0,
        "has_chem": has_chem,
        "active_gene_slots": int((mk > 0).sum().item()),
        "encoder_norm": safe_norm(enc),
        "projected_norm": safe_norm(proj),
    }
    if len(genes) < 2 or has_chem:
        return out
    single_encs = []
    single_projs = []
    for gene in genes:
        spb = single_gene_pb(dataset, gene, meta, batch_size=1)
        senc, sproj = encoder_outputs(model, spb)
        single_encs.append(senc)
        single_projs.append(sproj)
    enc_sum = torch.stack(single_encs, dim=0).sum(dim=0)
    enc_mean = torch.stack(single_encs, dim=0).mean(dim=0)
    proj_sum = torch.stack(single_projs, dim=0).sum(dim=0)
    proj_mean = torch.stack(single_projs, dim=0).mean(dim=0)
    out.update(
        {
            "encoder_cos_vs_single_sum": cosine(enc, enc_sum),
            "encoder_cos_vs_single_mean": cosine(enc, enc_mean),
            "projected_cos_vs_single_sum": cosine(proj, proj_sum),
            "projected_cos_vs_single_mean": cosine(proj, proj_mean),
            "encoder_norm_over_single_sum": safe_norm(enc) / max(safe_norm(enc_sum), 1e-8),
            "encoder_norm_over_single_mean": safe_norm(enc) / max(safe_norm(enc_mean), 1e-8),
            "projected_norm_over_single_sum": safe_norm(proj) / max(safe_norm(proj_sum), 1e-8),
            "projected_norm_over_single_mean": safe_norm(proj) / max(safe_norm(proj_mean), 1e-8),
        }
    )
    return out


def avg(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def q(vals: list[float | None], pct: float) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.percentile(xs, pct)) if xs else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
        by_dataset[str(row["dataset"])].append(row)

    def one(rs: list[dict[str, Any]]) -> dict[str, Any]:
        multi = [r for r in rs if int(r.get("n_genes") or 0) >= 2 and not bool(r.get("has_chem"))]
        return {
            "n": len(rs),
            "n_gene_multi_no_chem": len(multi),
            "mean_projected_cos_vs_single_sum": avg([r.get("projected_cos_vs_single_sum") for r in multi]),
            "p10_projected_cos_vs_single_sum": q([r.get("projected_cos_vs_single_sum") for r in multi], 10),
            "mean_projected_norm_over_single_sum": avg([r.get("projected_norm_over_single_sum") for r in multi]),
            "mean_projected_norm_over_single_mean": avg([r.get("projected_norm_over_single_mean") for r in multi]),
            "mean_encoder_cos_vs_single_sum": avg([r.get("encoder_cos_vs_single_sum") for r in multi]),
            "mean_encoder_norm_over_single_sum": avg([r.get("encoder_norm_over_single_sum") for r in multi]),
        }

    return {
        "by_group": {k: one(v) for k, v in sorted(by_group.items())},
        "focus_datasets": {k: one(by_dataset[k]) for k in FOCUS if k in by_dataset},
    }


def render_md(payload: dict[str, Any]) -> str:
    pstats = payload["parameter_stats"]
    lines = [
        "# LatentFM xverse Condition Mechanism Audit 2026-06-21",
        "",
        "Status: `mechanism_audit_no_gpu_gate`",
        "",
        "This audit uses checkpoint/config/condition metadata and pretrained condition embeddings only; it does not read held-out GT, perturbed means, posthoc predictions, or outcome metrics.",
        "",
        "## Checkpoint",
        "",
        f"- run_dir: `{payload['run_dir']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- checkpoint_step: `{payload['checkpoint_step']}`",
        f"- use_pert_condition: `{payload['config_subset']['use_pert_condition']}`",
        f"- pert_pairwise_mode: `{payload['config_subset']['pert_pairwise_mode']}`",
        f"- pert_pool_aggregations: `{payload['config_subset']['pert_pool_aggregations']}`",
        f"- pert_pool_fusion_mode: `{payload['config_subset']['pert_pool_fusion_mode']}`",
        f"- pert_to_c_init_mode: `{payload['config_subset']['pert_to_c_init_mode']}`",
        "",
        "## Parameter Stats",
        "",
        f"- pool_scale: `{pstats.get('pool_scale')}`",
        f"- type_scale: `{pstats.get('type_scale')}`",
        f"- pair_to_out: `{pstats.get('pair_to_out')}`",
        f"- pert_to_c.weight: `{pstats.get('pert_to_c_weight')}`",
        f"- condition_delta_head_present: `{pstats.get('condition_delta_head_present')}`",
        f"- condition_delta_to_c_present: `{pstats.get('condition_delta_to_c_present')}`",
        "",
        "## Multi-Condition Algebra",
        "",
        "| slice | n | gene multi no chem | mean cos(proj multi, sum singles) | p10 cos | mean norm/sum | mean norm/mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in payload["summary"]["by_group"].items():
        lines.append(
            "| {group} | {n} | {nm} | {cos} | {p10} | {ns} | {nmf} |".format(
                group=group,
                n=row["n"],
                nm=row["n_gene_multi_no_chem"],
                cos=fmt(row.get("mean_projected_cos_vs_single_sum")),
                p10=fmt(row.get("p10_projected_cos_vs_single_sum")),
                ns=fmt(row.get("mean_projected_norm_over_single_sum")),
                nmf=fmt(row.get("mean_projected_norm_over_single_mean")),
            )
        )
    lines.extend(["", "## Focus Datasets", "", "| dataset | n | multi | mean cos | mean norm/sum |", "|---|---:|---:|---:|---:|"])
    for dataset, row in payload["summary"]["focus_datasets"].items():
        lines.append(
            f"| {dataset} | {row['n']} | {row['n_gene_multi_no_chem']} | "
            f"{fmt(row.get('mean_projected_cos_vs_single_sum'))} | "
            f"{fmt(row.get('mean_projected_norm_over_single_sum'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current xverse anchor has no explicit pairwise interaction branch when `pert_pairwise_mode=off`; multi perturbations are encoded through pooled single-gene/chem content plus learned type and projection bridges.",
            "- High cosine between a multi embedding and summed single embeddings indicates a mostly additive condition representation; this is expected without pairwise mode and true train-multi supervision.",
            "- This audit does not justify GPU by itself. It should guide the next predeclared architecture/loss gate.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--max-conditions-per-group", type=int, default=256)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    cfg_payload = load_json(args.run_dir / "config.json")
    cfg = config_from_dict(cfg_payload)
    if not str(getattr(cfg, "split_file", "") or "").strip():
        cfg.split_file = str(args.split_file)

    ckpt_path = args.run_dir / args.checkpoint_name
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(cfg, torch.device("cpu"))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    inner = model.module if hasattr(model, "module") else model

    split = load_json(args.split_file)
    dataset = CrossDatasetFMDataset(
        data_dir=cfg.data_dir,
        split=split,
        batch_size=1,
        seed=int(cfg.seed),
        mode="test",
        min_cells=1,
        ds_alpha=1.0,
        scale_noise=0.0,
        use_pert_condition=bool(cfg.use_pert_condition),
        max_pert_genes=int(cfg.max_pert_genes),
        gene_embedding_cache_dir=str(cfg.pert_gene_emb_cache_dir),
        biflow_dir=str(cfg.biflow_dir),
        use_h5ad_pert_metadata=bool(cfg.use_h5ad_pert_metadata),
        pert_metainfo_path=str(cfg.pert_metainfo_path),
        chem_emb_source_dir=str(cfg.chem_emb_source_dir),
        chem_obs_column=str(cfg.chem_obs_column),
        drug_emb_cache_dir=str(getattr(cfg, "drug_emb_cache_dir", "")),
        max_chem_keys=int(cfg.max_chem_keys),
        chemical_metainfo_path=str(cfg.chemical_metainfo_path),
        chem_fallback_embed_dim=int(cfg.chem_fallback_embed_dim),
        latent_backbone=str(cfg.latent_backbone),
        pert_chem_enabled=bool(cfg.pert_chem_enabled),
        perturbation_family_filter="all",
        silent=True,
    )

    rows = []
    with torch.no_grad():
        for group in GROUPS:
            for ds_name, obj in sorted(split.items()):
                conds = list(obj.get(group) or [])
                if args.max_conditions_per_group > 0:
                    conds = conds[: int(args.max_conditions_per_group)]
                for cond in conds:
                    row = row_for_condition(model, dataset, str(ds_name), str(cond), group)
                    if row is not None:
                        rows.append(row)

    pstats = {
        "pool_scale": (
            [float(x) for x in inner.pert_encoder.pool_scale.detach().cpu().tolist()]
            if getattr(getattr(inner, "pert_encoder", None), "pool_scale", None) is not None
            else None
        ),
        "type_scale": (
            [float(x) for x in inner.pert_encoder.type_scale.detach().cpu().tolist()]
            if getattr(getattr(inner, "pert_encoder", None), "type_scale", None) is not None
            else None
        ),
        "pair_to_out": tensor_stats(
            getattr(getattr(getattr(inner, "pert_encoder", None), "pair_to_out", None), "weight", None)
        ),
        "pert_to_c_weight": tensor_stats(getattr(getattr(inner, "pert_to_c", None), "weight", None)),
        "condition_delta_head_present": getattr(inner, "condition_delta_head", None) is not None,
        "condition_delta_to_c_present": getattr(inner, "condition_delta_to_c", None) is not None,
    }

    payload = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "split_file": str(args.split_file),
        "leakage_status": "checkpoint_config_condition_metadata_pretrained_condition_embeddings_only",
        "load_state_dict": {"missing": list(missing), "unexpected": list(unexpected)},
        "config_subset": {
            "use_pert_condition": bool(cfg.use_pert_condition),
            "pert_pairwise_mode": str(getattr(cfg, "pert_pairwise_mode", "off")),
            "pert_pool_aggregations": list(getattr(cfg, "pert_pool_aggregations", ())),
            "pert_pool_scale_init": list(getattr(cfg, "pert_pool_scale_init", ())),
            "pert_pool_fusion_mode": str(getattr(cfg, "pert_pool_fusion_mode", "")),
            "pert_type_adapter_mode": str(getattr(cfg, "pert_type_adapter_mode", "")),
            "pert_type_scale_init": list(getattr(cfg, "pert_type_scale_init", ())),
            "pert_to_c_init_mode": str(getattr(cfg, "pert_to_c_init_mode", "")),
            "use_pert_in_fusion": bool(getattr(cfg, "use_pert_in_fusion", False)),
            "condition_delta_head_use_in_model": bool(getattr(cfg, "condition_delta_head_use_in_model", False)),
        },
        "parameter_stats": pstats,
        "n_rows": len(rows),
        "rows": rows,
        "summary": summarize(rows),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
