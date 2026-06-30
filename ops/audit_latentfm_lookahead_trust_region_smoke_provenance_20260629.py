#!/usr/bin/env python3
"""Posthoc provenance audit for a lookahead trust-region adapter smoke.

CPU/read-only. This script inspects the run summary and smoke checkpoint
metadata after a detached smoke has finished. It does not train, infer, read
canonical multi, read Track C query, or select a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data/cyx/1030/scLatent")
ANCHOR_CKPT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_allowed_trainable(name: str, adapter_kind: str) -> bool:
    if adapter_kind == "lowrank_residual":
        return name.startswith("condition_lowrank_residual_down.") or name.startswith(
            "condition_lowrank_residual_up."
        )
    return name.startswith("condition_delta_head.") or name.startswith("condition_delta_to_c.")


def is_allowed_missing(name: str, adapter_kind: str) -> bool:
    if name == "condition_delta_prior_gene_allowlist":
        return True
    return is_allowed_trainable(name, adapter_kind)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Lookahead Trust-Region Smoke Provenance Audit",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU/read-only audit over a completed smoke run directory.",
        "- No training, inference, canonical multi selection, Track C query, or checkpoint selection.",
        "",
        "## Checks",
        "",
        "| check | value | pass |",
        "|---|---|---:|",
    ]
    for row in payload["checks"]:
        lines.append(f"| `{row['name']}` | `{row['value']}` | `{row['pass']}` |")
    if payload["reasons"]:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- `{reason}`" for reason in payload["reasons"])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{payload['json_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--noop-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    ckpt_path = run_dir / "latest.pt"
    out_dir = run_dir / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "lookahead_trust_region_smoke_provenance_audit.json"
    out_md = out_dir / "LATENTFM_LOOKAHEAD_TRUST_REGION_PROVENANCE_AUDIT.md"

    reasons: list[str] = []
    checks: list[dict[str, Any]] = []

    if not summary_path.is_file():
        reasons.append(f"missing_summary_json:{summary_path}")
        summary: dict[str, Any] = {}
    else:
        summary = load_json(summary_path)
    if not ckpt_path.is_file():
        reasons.append(f"missing_checkpoint:{ckpt_path}")
        ckpt: dict[str, Any] = {}
    else:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    meta = ckpt.get("smoke_metadata", {}) if isinstance(ckpt, dict) else {}
    adapter_kind = str(summary.get("adapter_kind") or meta.get("adapter_kind") or "condition_delta")
    trainable_names = list(summary.get("trainable_names") or meta.get("trainable_names") or [])
    candidate_load = meta.get("candidate_load") or {}
    raw_noop = summary.get("max_noop_drift", meta.get("max_noop_drift", 999.0))
    max_noop_drift = 999.0 if raw_noop is None else float(raw_noop)
    raw_accepted = summary.get("accepted", meta.get("accepted", 0))
    accepted = 0 if raw_accepted is None else int(raw_accepted)
    anchor_checkpoint = str(summary.get("anchor_checkpoint") or meta.get("anchor_checkpoint") or "")
    safe_split = str(summary.get("safe_split") or meta.get("safe_split") or "")

    def add(name: str, value: Any, ok: bool, reason: str) -> None:
        checks.append({"name": name, "value": value, "pass": bool(ok)})
        if not ok:
            reasons.append(reason)

    add("accepted_updates", accepted, accepted > 0, "no_accepted_updates")
    add(
        "max_noop_drift",
        f"{max_noop_drift:.6g}",
        max_noop_drift <= float(args.noop_threshold),
        "max_noop_drift_above_threshold",
    )
    add(
        "anchor_checkpoint",
        anchor_checkpoint,
        anchor_checkpoint == str(ANCHOR_CKPT),
        "anchor_checkpoint_mismatch",
    )
    add("safe_split", safe_split, safe_split == str(SAFE_SPLIT), "safe_split_mismatch")
    add(
        "trainable_names",
        ";".join(trainable_names),
        bool(trainable_names) and all(is_allowed_trainable(name, adapter_kind) for name in trainable_names),
        "unexpected_trainable_scope",
    )
    missing = list(candidate_load.get("missing") or [])
    unexpected = list(candidate_load.get("unexpected") or [])
    skipped = list(candidate_load.get("skipped_shape_mismatch") or [])
    add(
        "candidate_missing_keys",
        ";".join(missing),
        all(is_allowed_missing(str(name), adapter_kind) for name in missing),
        "unexpected_missing_keys",
    )
    add("candidate_unexpected_keys", ";".join(unexpected), not unexpected, "unexpected_checkpoint_keys")
    add("candidate_skipped_shape_mismatch", ";".join(skipped), not skipped, "shape_mismatch_keys")
    add(
        "ema_applied",
        candidate_load.get("ema_applied"),
        candidate_load.get("ema_applied") is True,
        "candidate_ema_not_applied",
    )

    status = (
        "lookahead_trust_region_smoke_provenance_pass"
        if not reasons
        else "lookahead_trust_region_smoke_provenance_fail_close_or_debug"
    )
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "checkpoint": str(ckpt_path),
        "checks": checks,
        "reasons": reasons,
        "json_path": str(out_json),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "report": str(out_md)}, indent=2))
    return 0 if status.endswith("_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
