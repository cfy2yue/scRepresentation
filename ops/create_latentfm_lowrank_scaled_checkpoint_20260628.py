#!/usr/bin/env python3
"""Create a scaled-output low-rank residual checkpoint for diagnostics.

The low-rank residual adapter computes ``up(down(condition))``.  Scaling only
the ``up`` layer scales the adapter output linearly, so negative alpha is a
true sign flip.  The shared anchor/backbone weights are left untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch


LOWRANK_UP_PREFIX = "condition_lowrank_residual_up."
LOWRANK_DOWN_PREFIX = "condition_lowrank_residual_down."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--source-summary", type=Path, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = args.out_dir / "latest.pt"
    if out_ckpt.exists():
        raise FileExistsError(out_ckpt)

    ckpt = torch.load(str(args.source_checkpoint), map_location="cpu", weights_only=False)
    scaled = deepcopy(ckpt)
    model = scaled.get("model")
    if not isinstance(model, dict):
        raise KeyError("checkpoint missing model state dict")

    up_keys = [key for key in model if key.startswith(LOWRANK_UP_PREFIX)]
    down_keys = [key for key in model if key.startswith(LOWRANK_DOWN_PREFIX)]
    if not up_keys or not down_keys:
        raise KeyError("checkpoint missing low-rank residual keys")
    for key in up_keys:
        model[key] = model[key] * float(args.alpha)

    metadata = dict(scaled.get("smoke_metadata", {}))
    metadata.update(
        {
            "scaled_lowrank_checkpoint": True,
            "scaled_from": str(args.source_checkpoint),
            "scale_strategy": "up_only_output_scale",
            "lowrank_output_alpha": float(args.alpha),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        }
    )
    scaled["smoke_metadata"] = metadata
    torch.save(scaled, out_ckpt)

    if args.source_summary and args.source_summary.is_file():
        shutil.copy2(args.source_summary, args.out_dir / "source_summary.json")
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": "scaled_checkpoint_created",
        "source_checkpoint": str(args.source_checkpoint),
        "checkpoint": str(out_ckpt),
        "alpha": float(args.alpha),
        "scale_strategy": "up_only_output_scale",
        "scaled_keys": up_keys,
        "preserved_down_keys": down_keys,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
