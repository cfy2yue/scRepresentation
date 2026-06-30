#!/usr/bin/env python3
"""CPU-only code-boundary audit for Track C support-set task adapter.

This verifies whether the default-off support-set task adapter is only a model
forward hook or is plumbed through the LatentFM velocity wrappers. It does not
claim that a safe trainselect support-set source has been implemented.
It does not load datasets, train, infer, use GPU, or read any Track C query.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

from model.latent.config import Config  # noqa: E402
from model.latent.train import _model_latent_velocity, build_model  # noqa: E402


JSON_PATH = ROOT / "reports/latentfm_trackc_support_set_task_code_boundary_20260627.json"
MD_PATH = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_TASK_CODE_BOUNDARY_20260627.md"


def tiny_cfg() -> Config:
    cfg = Config()
    cfg.model_type = "control_mlp"
    cfg.emb_dim = 16
    cfg.mlp_d_model = 32
    cfg.mlp_n_layers = 1
    cfg.mlp_ratio = 2.0
    cfg.dropout = 0.0
    cfg.use_pert_condition = False
    cfg.trackc_support_set_task_use_in_model = True
    cfg.trackc_support_set_task_dim = 8
    return cfg


def exception_name(fn) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - audit wants exact boundary behavior
        return f"{type(exc).__name__}: {exc}"
    return ""


def main() -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    cfg = tiny_cfg()
    model = build_model(cfg, device)
    model.eval()
    batch = 3
    x_t = torch.randn(batch, cfg.emb_dim)
    x_0 = torch.randn(batch, cfg.emb_dim)
    t = torch.rand(batch)
    task = torch.randn(batch, cfg.trackc_support_set_task_dim)
    present = torch.ones(batch, 1)

    with torch.no_grad():
        direct_out = model(
            x_t,
            t,
            x_0,
            support_set_task=task,
            support_set_task_present=present,
        )
    direct_forward_ok = tuple(direct_out.shape) == (batch, cfg.emb_dim)
    missing_task_error = exception_name(lambda: model(x_t, t, x_0))
    train_wrapper_missing_error = exception_name(lambda: _model_latent_velocity(model, x_t, t, x_0, None))
    with torch.no_grad():
        wrapper_out = _model_latent_velocity(
            model,
            x_t,
            t,
            x_0,
            None,
            support_set_task=task,
            support_set_task_present=present,
        )

    default_cfg = Config()
    default_cfg.model_type = "control_mlp"
    default_cfg.emb_dim = 16
    default_cfg.mlp_d_model = 32
    default_cfg.mlp_n_layers = 1
    default_cfg.mlp_ratio = 2.0
    default_model = build_model(default_cfg, device)
    default_has_support_set_params = any("support_set_task" in name for name, _ in default_model.named_parameters())

    wrapper_forward_ok = tuple(wrapper_out.shape) == (batch, cfg.emb_dim)
    status = (
        "code_boundary_wrapper_plumbed_source_missing_no_gpu"
        if direct_forward_ok and wrapper_forward_ok and missing_task_error and not default_has_support_set_params
        else "unexpected_boundary_state_review_required_no_gpu"
    )
    reasons = []
    if direct_forward_ok:
        reasons.append("direct_model_forward_accepts_support_set_task")
    else:
        reasons.append("direct_model_forward_failed")
    if missing_task_error:
        reasons.append("enabled_adapter_requires_explicit_support_set_task")
    if wrapper_forward_ok:
        reasons.append("train_eval_velocity_wrapper_accepts_explicit_support_set_task")
    if default_has_support_set_params:
        reasons.append("default_off_model_unexpectedly_has_support_set_params")
    else:
        reasons.append("default_off_model_has_no_support_set_params")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "dataset_loaded": False,
            "training": False,
            "inference_or_posthoc_eval": False,
            "canonical_multi_used": False,
            "trackc_query_used": False,
        },
        "checks": {
            "direct_forward_ok": direct_forward_ok,
            "direct_forward_shape": list(direct_out.shape),
            "missing_task_error": missing_task_error,
            "train_wrapper_missing_task_error": train_wrapper_missing_error,
            "train_wrapper_forward_ok_with_task": wrapper_forward_ok,
            "train_wrapper_forward_shape": list(wrapper_out.shape),
            "default_has_support_set_params": default_has_support_set_params,
        },
        "reasons": reasons,
        "next_action": (
            "Implement safe trainselect support-set source construction and controls before any Track C support-set GPU smoke."
            if status == "code_boundary_wrapper_plumbed_source_missing_no_gpu"
            else "Manual review required before proceeding."
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = f"""# Track C Support-Set Task Code Boundary Audit

## Status

`{status}`

GPU authorized: `False`

## Boundary

CPU-only unit/code-boundary audit. No dataset loading, training, inference,
canonical multi selection, Track C query, or GPU use.

## Checks

* Direct model forward with explicit `support_set_task`: `{direct_forward_ok}`
* Direct output shape: `{tuple(direct_out.shape)}`
* Missing task error: `{missing_task_error}`
* Train/eval velocity wrapper with explicit task: `{wrapper_forward_ok}`
* Train/eval velocity wrapper output shape: `{tuple(wrapper_out.shape)}`
* Train/eval velocity wrapper missing-task error: `{train_wrapper_missing_error}`
* Default-off model has support-set params: `{default_has_support_set_params}`

## Interpretation

The default-off adapter exists in the model forward path, remains absent from
default models, and the shared velocity wrapper can accept an explicit
`support_set_task`. A safe trainselect support-set source and train/eval control
plumbing are still missing, so this is not a launchable Track C training route.

## Decision

Before any support-set GPU smoke, implement support-set source construction from
the safe trainselect split, default-off unit tests, permutation invariance,
shuffle/zero controls, and a support-val-only promotion gate.

## Outputs

* JSON: `{JSON_PATH}`
"""
    MD_PATH.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "md": str(MD_PATH), "json": str(JSON_PATH)}, indent=2))


if __name__ == "__main__":
    main()
