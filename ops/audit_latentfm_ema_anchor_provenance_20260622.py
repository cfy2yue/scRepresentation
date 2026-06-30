#!/usr/bin/env python3
"""Audit raw-vs-EMA checkpoint provenance for anchor-preserving LatentFM finetunes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ANCHOR = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
DEFAULT_CANDIDATE = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_trackc_routed_distill_20260622/"
    "xverse_trackc_route_condprior_w05_replay1_2k_seed42/best.pt"
)
DEFAULT_CANDIDATE_CONFIG = DEFAULT_CANDIDATE.parent / "config.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_ema_anchor_provenance_audit_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_EMA_ANCHOR_PROVENANCE_AUDIT_20260622.md"

TRAINABLE_PREFIXES_BY_SCOPE = {
    "condition_prior_adapter": (
        "condition_delta_head.",
        "condition_delta_to_c.",
        "module.condition_delta_head.",
        "module.condition_delta_to_c.",
    ),
    "pairwise_adapter": (
        "pert_encoder.pair_to_out.",
        "module.pert_encoder.pair_to_out.",
    ),
    "pairwise_condition_adapter": (
        "pert_encoder.pair_to_out.",
        "pert_to_c.",
        "condition_delta_to_c.",
        "module.pert_encoder.pair_to_out.",
        "module.pert_to_c.",
        "module.condition_delta_to_c.",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint must be a dict with a model key: {path}")
    return ckpt


def ema_meta(ckpt: dict[str, Any]) -> dict[str, Any]:
    ema = ckpt.get("ema")
    if not isinstance(ema, dict):
        return {"present": False, "active": False, "n_shadow": 0}
    meta = ema.get("__meta__")
    values: list[float] = []
    if torch.is_tensor(meta):
        values = [float(x) for x in meta.detach().cpu().flatten().tolist()]
    n_shadow = sum(1 for key in ema if isinstance(key, str) and key.startswith("shadow."))
    updates = int(values[3]) if len(values) >= 4 else 0
    step = int(ckpt.get("step") or 0)
    update_after = int(values[1]) if len(values) >= 2 else 0
    active = bool(n_shadow > 0 and (updates > 0 or step >= update_after))
    return {
        "present": True,
        "active": active,
        "n_shadow": int(n_shadow),
        "step": step,
        "update_after": update_after,
        "updates": updates,
        "meta": values,
    }


def ema_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    ema = ckpt.get("ema")
    if not isinstance(ema, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key, value in ema.items():
        if isinstance(key, str) and key.startswith("shadow.") and torch.is_tensor(value):
            out[key[len("shadow.") :]] = value.detach().cpu()
    return out


def tensor_state(obj: Any) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        return {}
    return {str(k): v.detach().cpu() for k, v in obj.items() if torch.is_tensor(v)}


def shape(v: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in v.shape)


def base_keys(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    trainable_scope: str,
) -> list[str]:
    prefixes = TRAINABLE_PREFIXES_BY_SCOPE.get(trainable_scope, ())
    keys = []
    for key in sorted(set(left) & set(right)):
        if prefixes and key.startswith(prefixes):
            continue
        if shape(left[key]) == shape(right[key]):
            keys.append(key)
    return keys


def compare_states(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    keys: list[str],
    *,
    sample_limit: int,
) -> dict[str, Any]:
    max_abs = 0.0
    mean_abs_values: list[float] = []
    nonzero_keys = []
    for key in keys:
        diff = (left[key].float() - right[key].float()).abs()
        cur_max = float(diff.max().item()) if diff.numel() else 0.0
        cur_mean = float(diff.mean().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, cur_max)
        mean_abs_values.append(cur_mean)
        if cur_max > 0.0 and len(nonzero_keys) < sample_limit:
            nonzero_keys.append({"key": key, "max_abs": cur_max, "mean_abs": cur_mean})
    return {
        "n_keys": len(keys),
        "max_abs": max_abs,
        "mean_abs_over_keys": float(sum(mean_abs_values) / len(mean_abs_values)) if mean_abs_values else None,
        "nonzero_key_samples": nonzero_keys,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM EMA Anchor Provenance Audit",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "## Inputs",
        "",
        f"- anchor_checkpoint: `{payload['anchor_checkpoint']}`",
        f"- candidate_checkpoint: `{payload['candidate_checkpoint']}`",
        f"- candidate_config: `{payload['candidate_config']}`",
        f"- finetune_trainable_scope: `{payload['candidate_config_fields'].get('finetune_trainable_scope')}`",
        f"- init_checkpoint_use_ema: `{payload['candidate_config_fields'].get('init_checkpoint_use_ema')}`",
        f"- anchor_replay_checkpoint_use_ema: `{payload['candidate_config_fields'].get('anchor_replay_checkpoint_use_ema')}`",
        "",
        "## EMA State",
        "",
        f"- anchor EMA active: `{payload['anchor_ema']['active']}`; shadow tensors: `{payload['anchor_ema']['n_shadow']}`",
        f"- candidate EMA active: `{payload['candidate_ema']['active']}`; shadow tensors: `{payload['candidate_ema']['n_shadow']}`",
        "",
        "## Base Tensor Comparisons",
        "",
        "| comparison | n keys | max abs | mean abs over keys |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["comparisons"].items():
        mean = row.get("mean_abs_over_keys")
        lines.append(
            f"| `{name}` | {row['n_keys']} | {row['max_abs']:.9f} | "
            f"{'NA' if mean is None else f'{mean:.9f}'} |"
        )
    lines += ["", "## Interpretation", ""]
    lines.extend(f"- {reason}" for reason in payload["decision"]["reasons"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- This audit is CPU-only and reads checkpoints/config only.",
        "- It does not evaluate Track C held-out query or canonical posthoc outcomes.",
        "- Future anchor-preserving finetunes should pass this audit before no-harm claims.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    cfg = load_json(args.candidate_config)
    trainable_scope = str(cfg.get("finetune_trainable_scope") or "all")
    anchor = load_checkpoint(args.anchor_checkpoint)
    candidate = load_checkpoint(args.candidate_checkpoint)
    anchor_model = tensor_state(anchor["model"])
    candidate_model = tensor_state(candidate["model"])
    anchor_ema_state = ema_state(anchor)
    candidate_ema_state = ema_state(candidate)

    raw_keys = base_keys(candidate_model, anchor_model, trainable_scope=trainable_scope)
    ema_keys = base_keys(candidate_model, anchor_ema_state, trainable_scope=trainable_scope)
    candidate_ema_keys = base_keys(candidate_ema_state, candidate_model, trainable_scope=trainable_scope)

    comparisons = {
        "candidate_model_vs_anchor_raw_base": compare_states(
            candidate_model,
            anchor_model,
            raw_keys,
            sample_limit=args.sample_limit,
        ),
        "candidate_model_vs_anchor_ema_base": compare_states(
            candidate_model,
            anchor_ema_state,
            ema_keys,
            sample_limit=args.sample_limit,
        ),
        "candidate_ema_vs_candidate_model_base": compare_states(
            candidate_ema_state,
            candidate_model,
            candidate_ema_keys,
            sample_limit=args.sample_limit,
        ),
    }

    reasons = []
    init_use_ema = bool(cfg.get("init_checkpoint_use_ema") or False)
    replay_use_ema = bool(cfg.get("anchor_replay_checkpoint_use_ema") or False)
    anchor_ema = ema_meta(anchor)
    if anchor_ema["active"] and not init_use_ema:
        reasons.append("candidate warm-start did not request active anchor EMA")
    if anchor_ema["active"] and cfg.get("anchor_replay_loss_weight", 0.0) and not replay_use_ema:
        reasons.append("anchor replay did not request active anchor EMA")
    if comparisons["candidate_model_vs_anchor_raw_base"]["max_abs"] == 0.0:
        reasons.append("candidate frozen base matches raw anchor model exactly")
    if comparisons["candidate_model_vs_anchor_ema_base"]["max_abs"] > 0.0:
        reasons.append("candidate frozen base differs from anchor EMA baseline")
    if candidate_ema_state and comparisons["candidate_ema_vs_candidate_model_base"]["n_keys"] == 0:
        reasons.append("candidate EMA shadows do not cover frozen base keys")

    mismatch = any("did not request" in r or "differs from anchor EMA" in r for r in reasons)
    status = "ema_anchor_mismatch_risk_confirmed" if mismatch else "ema_anchor_provenance_consistent"
    payload = {
        "anchor_checkpoint": str(args.anchor_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_config": str(args.candidate_config),
        "candidate_config_fields": {
            "finetune_trainable_scope": cfg.get("finetune_trainable_scope"),
            "init_checkpoint": cfg.get("init_checkpoint"),
            "init_checkpoint_use_ema": cfg.get("init_checkpoint_use_ema"),
            "anchor_replay_checkpoint": cfg.get("anchor_replay_checkpoint"),
            "anchor_replay_checkpoint_use_ema": cfg.get("anchor_replay_checkpoint_use_ema"),
            "anchor_replay_loss_weight": cfg.get("anchor_replay_loss_weight"),
        },
        "anchor_ema": anchor_ema,
        "candidate_ema": ema_meta(candidate),
        "n_anchor_model_tensors": len(anchor_model),
        "n_candidate_model_tensors": len(candidate_model),
        "n_anchor_ema_tensors": len(anchor_ema_state),
        "n_candidate_ema_tensors": len(candidate_ema_state),
        "comparisons": comparisons,
        "decision": {
            "status": status,
            "reasons": reasons or ["checkpoint provenance is internally consistent"],
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
