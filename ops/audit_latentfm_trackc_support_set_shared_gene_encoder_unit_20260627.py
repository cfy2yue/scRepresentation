#!/usr/bin/env python3
"""CPU unit gate for shared-gene support-set token construction.

This checks source-level invariants before any training launcher exists:
permutation-invariant support-token construction, zero-token control, and
adapter-only gradient flow into the default-off support_set_task bridge.

It reads only safe trainselect condition-mean artifacts and does not train,
infer, use GPU, read canonical multi for selection, or read Track C query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

from model.latent.config import Config  # noqa: E402
from model.latent.fm_ot import CondOTPath  # noqa: E402
from model.latent.train import build_model, load_model_weights_only, train_step  # noqa: E402


PRE_JSON = ROOT / "reports/latentfm_trackc_support_set_shared_gene_source_preflight_20260627.json"
ANCHOR_PATH = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means/"
    "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json"
)
CANDIDATE_PATH = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means/"
    "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json"
)
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_shared_gene_encoder_unit_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_SHARED_GENE_ENCODER_UNIT_20260627.md"
ANCHOR_CONFIG = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/config.json"
)
ANCHOR_CKPT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair(cond: str) -> tuple[str, str] | None:
    parts = [p.strip().upper() for p in str(cond).split("+") if p.strip()]
    return (parts[0], parts[1]) if len(parts) == 2 else None


def vec(row: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(row[key], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError(f"bad {key} vector")
    return arr


def rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def paired_rows(anchor: dict[str, Any], candidate: dict[str, Any], group: str) -> list[dict[str, Any]]:
    a = {(str(r["dataset"]), str(r["condition"])): r for r in rows(anchor, group)}
    c = {(str(r["dataset"]), str(r["condition"])): r for r in rows(candidate, group)}
    out = []
    for key in sorted(set(a) & set(c)):
        p = pair(key[1])
        if p is None:
            continue
        ar = a[key]
        cr = c[key]
        out.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "pair": p,
                "src": vec(ar, "ctrl_mean"),
                "gt": vec(ar, "gt_mean"),
                "residual": vec(cr, "pred_mean") - vec(ar, "pred_mean"),
            }
        )
    return out


def support_for(query: dict[str, Any], train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qgenes = set(query["pair"])
    return [
        row
        for row in train
        if row["dataset"] == query["dataset"] and bool(qgenes & set(row["pair"]))
    ]


def token(support: list[dict[str, Any]]) -> np.ndarray:
    if not support:
        raise ValueError("support set is empty")
    return np.stack([row["residual"] for row in support], axis=0).mean(axis=0).astype(np.float32)


def cfg_for_dim(dim: int) -> Config:
    raw = load_json(ANCHOR_CONFIG)
    cfg = Config()
    for key in ("model_type", "emb_dim", "mlp_d_model", "mlp_n_layers", "mlp_ratio", "dropout"):
        if key in raw:
            setattr(cfg, key, raw[key])
    cfg.emb_dim = int(dim)
    cfg.use_pert_condition = False
    cfg.use_mmd = False
    cfg.trackc_support_set_task_use_in_model = True
    cfg.trackc_support_set_task_dim = int(dim)
    cfg.finetune_trainable_scope = "support_set_task_adapter"
    cfg.use_amp = False
    return cfg


def grad_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(prefix) or p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(torch.sum(g * g).item())
    return float(total ** 0.5)


def main() -> None:
    torch.set_num_threads(1)
    pre = load_json(PRE_JSON)
    pre_status = str(pre.get("status"))
    if not pre_status.endswith("encoder_unit_next_no_gpu"):
        raise RuntimeError(f"source preflight does not authorize encoder unit: {pre_status}")
    anchor = load_json(ANCHOR_PATH)
    candidate = load_json(CANDIDATE_PATH)
    train = paired_rows(anchor, candidate, "train_multi")
    support_val = paired_rows(anchor, candidate, "support_val_multi")
    query = next(row for row in support_val if support_for(row, train))
    support = support_for(query, train)
    tok = token(support)
    shuffled = list(reversed(support))
    tok_shuffled = token(shuffled)
    perm_diff = float(np.linalg.norm(tok - tok_shuffled))
    tok_norm = float(np.linalg.norm(tok))

    cfg = cfg_for_dim(int(tok.shape[0]))
    model = build_model(cfg, torch.device("cpu"))
    missing, unexpected, skipped = load_model_weights_only(
        ANCHOR_CKPT,
        model,
        torch.device("cpu"),
        strict=False,
        prefer_ema=True,
    )
    src = torch.from_numpy(query["src"]).float().unsqueeze(0).repeat(4, 1)
    gt = torch.from_numpy(query["gt"]).float().unsqueeze(0).repeat(4, 1)
    task = torch.from_numpy(tok).float().unsqueeze(0).repeat(4, 1)
    present = torch.ones((4, 1), dtype=torch.float32)

    out = train_step(
        src,
        gt,
        model,
        CondOTPath(),
        cfg,
        torch.device("cpu"),
        support_set_task=task,
        support_set_task_present=present,
    )
    out["loss"].backward()
    adapter_grad = grad_norm(model, "support_set_task_to_c.")

    default_cfg = cfg_for_dim(int(tok.shape[0]))
    default_cfg.trackc_support_set_task_use_in_model = False
    default_cfg.trackc_support_set_task_dim = 0
    default_model = build_model(default_cfg, torch.device("cpu"))
    default_has_params = any("support_set_task" in name for name, _ in default_model.named_parameters())

    status = (
        "shared_gene_support_encoder_unit_pass_source_plumbing_next_no_gpu"
        if perm_diff <= 1e-8 and tok_norm > 1e-8 and adapter_grad > 1e-10 and not default_has_params
        else "shared_gene_support_encoder_unit_fail_no_gpu"
    )
    reasons = []
    if perm_diff > 1e-8:
        reasons.append("token_not_permutation_invariant")
    if tok_norm <= 1e-8:
        reasons.append("token_near_zero")
    if adapter_grad <= 1e-10:
        reasons.append("adapter_gradient_near_zero")
    if default_has_params:
        reasons.append("default_off_model_has_support_set_params")
    if not reasons:
        reasons.append("source_token_unit_checks_pass_but_gpu_still_forbidden")

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training": False,
            "inference": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "source_preflight_status": pre_status,
        "selected_query": {
            "dataset": query["dataset"],
            "condition": query["condition"],
            "support_count": len(support),
        },
        "checks": {
            "token_dim": int(tok.shape[0]),
            "token_norm": tok_norm,
            "permutation_diff": perm_diff,
            "adapter_grad_norm": adapter_grad,
            "default_off_has_support_set_params": default_has_params,
            "anchor_missing_count": len(missing),
            "anchor_unexpected_count": len(unexpected),
            "anchor_skipped_shape_mismatch": skipped,
        },
        "reasons": reasons,
        "next_action": (
            "implement train/eval support-set source plumbing and controls before GPU"
            if status.endswith("next_no_gpu")
            else "close or redesign shared-gene support token"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = f"""# Track C Shared-Gene Support-Set Encoder Unit

## Status

`{status}`

GPU authorized: `False`

## Boundary

CPU-only unit gate. No training loop, inference, canonical multi selection,
Track C query, or GPU.

## Selected Query

* Dataset: `{query['dataset']}`
* Condition: `{query['condition']}`
* Support rows: `{len(support)}`

## Checks

* token dim: `{int(tok.shape[0])}`
* token norm: `{tok_norm:.6e}`
* permutation diff: `{perm_diff:.6e}`
* adapter grad norm: `{adapter_grad:.6e}`
* default-off has support-set params: `{default_has_params}`
* anchor load missing/unexpected/skipped: `{len(missing)}` / `{len(unexpected)}` / `{len(skipped)}`

## Decision Reasons

{chr(10).join(f'- `{r}`' for r in reasons)}

## Decision

This can only authorize train/eval source plumbing and controls. It cannot
authorize GPU training.

## Outputs

* JSON: `{OUT_JSON}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
