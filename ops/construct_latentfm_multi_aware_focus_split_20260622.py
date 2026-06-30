#!/usr/bin/env python3
"""Derive a focused Wessels/Norman multi-aware fine-tune split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_IN = ROOT / "dataset/biFlow_data/split_seed42_multi_aware_v1.json"
DEFAULT_OUT = ROOT / "dataset/biFlow_data/split_seed42_multi_aware_v1_finetune_wn.json"
DEFAULT_AUDIT = ROOT / "reports/LATENTFM_MULTI_AWARE_V1_FINETUNE_WN_SPLIT_AUDIT_20260622.md"
FOCUS = ("Wessels", "NormanWeissman2019_filtered")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Multi-Aware v1 Wessels/Norman Fine-Tune Split Audit 2026-06-22",
        "",
        "Status: `focused_training_split_created`",
        "",
        "This split is a training-efficiency derivative of `multi_aware_v1`. It is not a replacement for canonical or full multi-aware evaluation.",
        "",
        f"- input_split: `{payload['input_split']}`",
        f"- output_split: `{payload['output_split']}`",
        f"- focus_datasets: `{payload['focus_datasets']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        "",
        "| dataset | train single | train multi | val multi | test single | heldout multi |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['train_single']} | {row['train_multi']} | "
            f"{row['val_multi']} | {row['test_single']} | {row['test_multi']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Use this split only for warm-started Wessels/Norman multi-aware fine-tune.",
            "- Evaluate candidate checkpoints back on full `multi_aware_v1` and canonical single/background reports before any claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-split", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-split", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    src = load_json(args.input_split)
    out = {}
    rows = []
    for ds in FOCUS:
        obj = dict(src[ds])
        obj["train"] = list(obj.get("train_single") or []) + list(obj.get("train_multi") or [])
        obj["test"] = list(obj.get("test_single") or []) + list(obj.get("test_multi") or [])
        out[ds] = obj
        rows.append(
            {
                "dataset": ds,
                "train_single": len(obj.get("train_single") or []),
                "train_multi": len(obj.get("train_multi") or []),
                "val_multi": len(obj.get("val_multi") or []),
                "test_single": len(obj.get("test_single") or []),
                "test_multi": len(obj.get("test_multi") or []),
            }
        )

    payload = {
        "input_split": str(args.input_split),
        "output_split": str(args.out_split),
        "focus_datasets": list(FOCUS),
        "leakage_status": "derived from multi_aware_v1 condition split only; no outcome metrics",
        "rows": rows,
    }
    args.out_split.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_split": str(args.out_split), "out_md": str(args.out_md), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
