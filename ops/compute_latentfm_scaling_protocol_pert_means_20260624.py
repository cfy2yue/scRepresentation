#!/usr/bin/env python3
"""Compute train-only pert-mean artifacts for scaling protocol splits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_MANIFEST = ROOT / "reports/latentfm_scaling_protocol_splits_20260624.json"
OUT_DIR = ROOT / "runs/latentfm_scaling_protocol_splits_20260624/artifacts"
OUT_JSON = ROOT / "reports/latentfm_scaling_protocol_pert_means_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_PROTOCOL_PERT_MEANS_20260624.md"
HELPER = ROOT / "ops/build_latentfm_xverse_scaling_splits_20260624.py"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper():
    spec = importlib.util.spec_from_file_location("xverse_scaling_split_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    manifest = load_json(SPLIT_MANIFEST)
    helper = load_helper()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in manifest["arms"]:
        name = str(arm["arm"])
        split_file = Path(str(arm["split_file"]))
        split = load_json(split_file)
        means, audit = helper.compute_train_pert_means(DATA_DIR, split)
        out_file = OUT_DIR / f"{name}_trainonly_pert_means.npz"
        np.savez_compressed(out_file, **means)
        rows.append(
            {
                "arm": name,
                "split_file": str(split_file),
                "pert_means_file": str(out_file),
                "n_datasets_with_means": len(means),
                "audit": audit,
                "status": "ok" if all(r.get("status") in {"ok", "empty_train_dataset"} for r in audit) else "check",
            }
        )
    payload = {
        "status": "pass_cpu_pert_means_ready_no_gpu",
        "boundary": {
            "read_protocol_split_json": True,
            "read_train_h5_gt_embeddings": True,
            "read_canonical_metrics": False,
            "read_trackc_query": False,
            "launched_gpu": False,
        },
        "data_dir": str(DATA_DIR),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Scaling Protocol Pert Means",
        "",
        "Status: `pass_cpu_pert_means_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact generation.",
        "- Reads train-only protocol splits and train H5 GT embeddings only.",
        "- Does not read canonical metrics, Track C query, or use GPU.",
        "",
        "## Rows",
        "",
        "| arm | datasets with means | pert means |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['arm']}` | {row['n_datasets_with_means']} | `{row['pert_means_file']}` |")
    lines.extend(["", "## Output", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
