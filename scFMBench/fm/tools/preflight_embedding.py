#!/usr/bin/env python3
"""Scan data roots, check weights + adapter import per model; write preflight.json and optional manifest.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_registry import (
    DEFAULT_EMBEDDING_RUNS_DIR,
    LATENT_BENCH_ROOT,
    MODEL_QUEUE_ORDER,
    OUTPUT_ROOT,
    PRETRAINED_ROOT,
    SCFM_ROOT,
    check_weights,
    import_smoke_cmd,
    python_for_model,
    subprocess_env,
)
import paths


def _is_canonical_h5ad(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    if os.environ.get("SCFM_INCLUDE_NONCANONICAL_H5AD", "").strip() == "1":
        return True
    stem = path.stem
    return not (".tmp" in stem or ".before_" in stem or stem.endswith(".bak"))


def discover_h5ad(roots: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        cat = root.name
        for p in sorted(root.rglob("*.h5ad")):
            if not _is_canonical_h5ad(p):
                continue
            stem = p.stem
            rel = str(p)
            rows.append(
                {
                    "path": rel,
                    "dataset_id": stem,
                    "category": cat,
                }
            )
    return rows


def _h5ad_has_materialized_x(path: Path) -> bool:
    """Cheap backed check for a materialized AnnData X matrix."""
    try:
        import h5py
    except Exception:
        return True
    try:
        with h5py.File(path, "r") as h5:
            return "X" in h5
    except Exception:
        return False


def run_import_test(model: str) -> Dict[str, Any]:
    py = python_for_model(model)
    env = subprocess_env(model)
    cmd = [py, *import_smoke_cmd(model)]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            cwd=str(LATENT_BENCH_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        ok = r.returncode == 0
        return {
            "import_ok": ok,
            "returncode": r.returncode,
            "stdout": (r.stdout or "")[-2000:],
            "stderr": (r.stderr or "")[-2000:],
            "seconds": round(time.time() - t0, 2),
            "python": py,
        }
    except Exception as e:
        return {
            "import_ok": False,
            "error": str(e),
            "python": py,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--roots",
        type=Path,
        nargs="*",
        default=paths.default_h5ad_roots(),
        help="Directories to scan recursively for *.h5ad",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR / "preflight.json",
        help="preflight.json (default: output/embedding_runs/)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR / "manifest.jsonl",
        help="manifest.jsonl for submit_embedding_queue.py (default: output/embedding_runs/)",
    )
    ap.add_argument(
        "--manifest-with-x",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR / "manifest_with_X.jsonl",
        help="Manifest filtered to h5ad files with a materialized X group/dataset.",
    )
    ap.add_argument(
        "--require-materialized-x",
        action="store_true",
        help="Write --manifest from the X-filtered dataset list as well.",
    )
    ap.add_argument("--no-manifest", action="store_true", help="Do not write manifest.jsonl")
    ap.add_argument("--skip-import-test", action="store_true")
    ap.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Only check these models (import + weights); default: MODEL_QUEUE_ORDER",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    datasets = discover_h5ad(args.roots)
    datasets_with_x = [
        row for row in datasets if _h5ad_has_materialized_x(Path(str(row["path"])))
    ]
    model_list = args.models if args.models else list(MODEL_QUEUE_ORDER)
    models_report: Dict[str, Any] = {}
    for m in model_list:
        wst, wdetail = check_weights(m)
        entry: Dict[str, Any] = {
            "weights_status": wst,
            "weights_detail": wdetail,
            "python": python_for_model(m),
        }
        if not args.skip_import_test:
            entry["import"] = run_import_test(m)
            entry["ready"] = entry["weights_status"] == "ready" and entry["import"].get("import_ok", False)
        else:
            entry["import"] = {"skipped": True}
            entry["ready"] = entry["weights_status"] == "ready"
        models_report[m] = entry

    out_obj = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fm_root": str(LATENT_BENCH_ROOT),
        "scfm_root": str(SCFM_ROOT),
        "latent_bench_root": str(LATENT_BENCH_ROOT),
        "pretrained_root": str(PRETRAINED_ROOT),
        "default_output_root": str(OUTPUT_ROOT),
        "roots_scanned": [str(r) for r in args.roots],
        "n_h5ad": len(datasets),
        "n_h5ad_with_x": len(datasets_with_x),
        "datasets": datasets,
        "models": models_report,
        "queue_order": [m for m in model_list if models_report[m].get("ready")],
    }
    args.out.write_text(json.dumps(out_obj, indent=2, default=str))

    if not args.no_manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_rows = datasets_with_x if args.require_materialized_x else datasets
        with open(args.manifest, "w") as mf:
            for row in manifest_rows:
                mf.write(json.dumps(row, default=str) + "\n")
        args.manifest_with_x.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_with_x, "w") as mf:
            for row in datasets_with_x:
                mf.write(json.dumps(row, default=str) + "\n")

    print(
        json.dumps(
            {
                "preflight": str(args.out),
                "manifest": None if args.no_manifest else str(args.manifest),
                "manifest_with_x": None if args.no_manifest else str(args.manifest_with_x),
                "n_h5ad": len(datasets),
                "n_h5ad_with_x": len(datasets_with_x),
                "ready_models": out_obj["queue_order"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
