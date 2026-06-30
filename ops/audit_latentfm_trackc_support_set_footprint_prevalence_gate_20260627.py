#!/usr/bin/env python3
"""CPU gate for Track C support-set task prevalence and footprint.

This is query-free and canonical-free. It checks whether the failed min-support
support-set smoke was likely diluted by too few token-present train rows, and
whether the adapter path has a nonzero one-step footprint when token-present
rows are isolated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.latent.config import Config  # noqa: E402
from model.latent.train import (  # noqa: E402
    build_model,
    build_trackc_support_set_task_bank,
    load_model_weights_only,
    make_trackc_support_set_task_batch,
)


SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
ART_DIR = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623"
    / "xverse_support_film_retry1_trainmulti_condition_means"
    / "condition_means"
)
ANCHOR_MEANS = ART_DIR / "trainselect_anchor_train_support_multi_condition_means_ode20.json"
CANDIDATE_MEANS = ART_DIR / "trainselect_candidate_train_support_multi_condition_means_ode20.json"
ANCHOR_CKPT = (
    COUPLED
    / "output/latentfm_runs/xverse_8k_full_eval_20260620"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_footprint_prevalence_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_FOOTPRINT_PREVALENCE_GATE_20260627.md"


def pair_genes(cond: str) -> tuple[str, str] | None:
    parts = [part.strip().upper() for part in str(cond).split("+") if part.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def build_cfg(min_support: int = 2) -> Config:
    return Config(
        emb_dim=384,
        model_type="control_mlp",
        latent_backbone="xverse",
        mlp_d_model=512,
        mlp_n_layers=8,
        mlp_ratio=4.0,
        dropout=0.0,
        trackc_support_set_task_use_in_model=True,
        trackc_support_set_task_dim=384,
        trackc_support_set_task_source="shared_gene_condition_means",
        trackc_support_set_task_safe_split_file=str(SAFE_SPLIT),
        trackc_support_set_task_anchor_condition_means=str(ANCHOR_MEANS),
        trackc_support_set_task_candidate_condition_means=str(CANDIDATE_MEANS),
        trackc_support_set_task_scale=1.0,
        trackc_support_set_task_min_support_count=int(min_support),
        trackc_support_set_task_eval_control="actual",
        use_pert_condition=False,
        use_ema=False,
        gpu=0,
        seed=42,
    )


def support_counts(split: dict[str, Any], cfg: Config, bank: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, dict[str, int]] = {}
    for ds_name, groups in sorted(split.items()):
        train = [str(x) for x in (groups.get("train") or [])]
        train_multi = set(str(x) for x in (groups.get("train_multi") or []))
        support_val_multi = [str(x) for x in (groups.get("support_val_multi") or groups.get("test_multi") or [])]
        stats = {"train": len(train), "train_multi": len(train_multi), "train_present": 0, "train_multi_present": 0, "support_val_multi": len(support_val_multi), "support_val_present": 0}
        for cond in train:
            task, present = make_trackc_support_set_task_batch(bank, ds_name, cond, 1, cfg, torch.device("cpu"))
            is_present = bool(present is not None and float(present.sum().item()) > 0.5)
            if is_present:
                stats["train_present"] += 1
                if cond in train_multi:
                    stats["train_multi_present"] += 1
                rows.append({"dataset": ds_name, "condition": cond, "group": "train", "is_train_multi": cond in train_multi})
        for cond in support_val_multi:
            _task, present = make_trackc_support_set_task_batch(bank, ds_name, cond, 1, cfg, torch.device("cpu"))
            if present is not None and float(present.sum().item()) > 0.5:
                stats["support_val_present"] += 1
        by_dataset[ds_name] = stats
    total_train = sum(v["train"] for v in by_dataset.values())
    total_present = sum(v["train_present"] for v in by_dataset.values())
    total_train_multi = sum(v["train_multi"] for v in by_dataset.values())
    total_train_multi_present = sum(v["train_multi_present"] for v in by_dataset.values())
    support_val_total = sum(v["support_val_multi"] for v in by_dataset.values())
    support_val_present = sum(v["support_val_present"] for v in by_dataset.values())
    return {
        "by_dataset": by_dataset,
        "present_rows": rows,
        "totals": {
            "train": total_train,
            "train_present": total_present,
            "train_present_fraction": total_present / max(total_train, 1),
            "train_multi": total_train_multi,
            "train_multi_present": total_train_multi_present,
            "train_multi_present_fraction": total_train_multi_present / max(total_train_multi, 1),
            "support_val_multi": support_val_total,
            "support_val_present": support_val_present,
            "support_val_present_fraction": support_val_present / max(support_val_total, 1),
        },
    }


def footprint_probe(cfg: Config, bank: dict[str, list[dict[str, Any]]], present_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not present_rows:
        return {"status": "no_present_rows"}
    row = next((r for r in present_rows if r["dataset"] == "Wessels"), present_rows[0])
    ds_name = str(row["dataset"])
    cond = str(row["condition"])
    device = torch.device("cpu")
    torch.manual_seed(7)
    model = build_model(cfg, device)
    missing, unexpected, skipped = load_model_weights_only(
        ANCHOR_CKPT,
        model,
        device,
        strict=False,
        prefer_ema=True,
    )
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    inner = getattr(model, "module", model)
    if getattr(inner, "support_set_task_to_c", None) is None:
        return {"status": "missing_support_set_task_to_c"}
    inner.support_set_task_to_c.weight.requires_grad = True
    batch_size = 4
    x_0 = torch.randn(batch_size, cfg.emb_dim, device=device) * 0.05
    x_1 = x_0 + torch.randn(batch_size, cfg.emb_dim, device=device) * 0.02
    t = torch.full((batch_size,), 0.5, device=device)
    x_t = (1.0 - t[:, None]) * x_0 + t[:, None] * x_1
    support_task, support_present = make_trackc_support_set_task_batch(bank, ds_name, cond, batch_size, cfg, device)
    absent_task = torch.zeros_like(support_task)
    absent_present = torch.zeros_like(support_present)
    with torch.no_grad():
        pred_absent = model(x_t, t, x_0, support_set_task=absent_task, support_set_task_present=absent_present)
    pred_present = model(x_t, t, x_0, support_set_task=support_task, support_set_task_present=support_present)
    target = x_1 - x_0
    loss = torch.nn.functional.mse_loss(pred_present, target)
    loss.backward()
    grad = inner.support_set_task_to_c.weight.grad
    movement = (pred_present.detach() - pred_absent).norm(dim=1).mean().item()
    support_norm = support_task.norm(dim=1).mean().item()
    return {
        "status": "ok",
        "dataset": ds_name,
        "condition": cond,
        "support_norm_mean": float(support_norm),
        "present_mask_sum": float(support_present.sum().item()),
        "present_vs_absent_velocity_l2": float(movement),
        "support_adapter_grad_norm": 0.0 if grad is None else float(grad.norm().item()),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "skipped_shape_mismatch_count": len(skipped),
    }


def main() -> int:
    reasons: list[str] = []
    for path in [SAFE_SPLIT, ANCHOR_MEANS, CANDIDATE_MEANS, ANCHOR_CKPT]:
        if not path.is_file():
            reasons.append(f"missing_required:{path}")
    split = json.loads(SAFE_SPLIT.read_text(encoding="utf-8")) if SAFE_SPLIT.is_file() else {}
    cfg = build_cfg(min_support=2)
    bank = build_trackc_support_set_task_bank(cfg) if not reasons else {}
    counts = support_counts(split, cfg, bank) if bank else {}
    footprint = footprint_probe(cfg, bank, counts.get("present_rows") or []) if bank else {"status": "missing_bank"}
    totals = counts.get("totals") or {}
    present_fraction = float(totals.get("train_present_fraction") or 0.0)
    support_val_fraction = float(totals.get("support_val_present_fraction") or 0.0)
    if present_fraction >= 0.05:
        reasons.append("train_token_present_fraction_not_sparse")
    if support_val_fraction < 0.50:
        reasons.append("support_val_token_present_fraction_below_0p50")
    if footprint.get("status") != "ok":
        reasons.append(f"footprint_status_{footprint.get('status')}")
    else:
        if float(footprint.get("support_adapter_grad_norm") or 0.0) <= 1e-8:
            reasons.append("support_adapter_grad_norm_too_small")
        if float(footprint.get("support_norm_mean") or 0.0) <= 1e-8:
            reasons.append("support_token_norm_too_small")
    status = (
        "trackc_support_set_footprint_prevalence_gate_pass_focused_split_next_no_gpu"
        if not reasons
        else "trackc_support_set_footprint_prevalence_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "boundary": {
            "safe_trainselect_split": str(SAFE_SPLIT),
            "heldout_trackc_query_used": False,
            "canonical_multi_selection_used": False,
            "canonical_metrics_used": False,
            "training_or_inference_used": False,
        },
        "counts": counts,
        "footprint": footprint,
        "decision": (
            "Pass means the next legal step is a focused split/launcher preflight, not direct promotion."
            if not reasons
            else "Do not launch another support-set GPU smoke from this source without a materially new mechanism."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Track C Support-Set Footprint / Prevalence Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Reads safe trainselect split and train/support condition-mean artifacts only.",
        "- Does not use held-out Track C query, canonical multi, or canonical metrics.",
        "- Does not train or run evaluation inference.",
        "",
        "## Prevalence",
        "",
        f"- train token-present: `{totals.get('train_present')}/{totals.get('train')}` = `{present_fraction:.6f}`",
        f"- train_multi token-present: `{totals.get('train_multi_present')}/{totals.get('train_multi')}` = `{float(totals.get('train_multi_present_fraction') or 0.0):.6f}`",
        f"- support_val_multi token-present: `{totals.get('support_val_present')}/{totals.get('support_val_multi')}` = `{support_val_fraction:.6f}`",
        "",
        "## Footprint",
        "",
        f"- probe: `{footprint}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
