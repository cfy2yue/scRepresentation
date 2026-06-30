#!/usr/bin/env python3
"""Validate scFM external resource layout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

_ROOT = Path(__file__).resolve().parent
_FM_ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_FM_ROOT) not in sys.path:
    sys.path.insert(0, str(_FM_ROOT))

import paths
from model_registry import MODEL_QUEUE_ORDER, check_weights, import_smoke_cmd, python_for_model, subprocess_env


def _existing_h5ad(roots: Iterable[Path]) -> List[Path]:
    rows: List[Path] = []
    include_noncanonical = os.environ.get("SCFM_INCLUDE_NONCANONICAL_H5AD", "").strip() == "1"
    for root in roots:
        if root.is_dir():
            for p in sorted(root.rglob("*.h5ad")):
                if p.name.startswith("."):
                    continue
                if (
                    ".tmp" in p.stem
                    or ".before_" in p.stem
                    or p.stem.endswith(".bak")
                ) and not include_noncanonical:
                    continue
                rows.append(p)
    return rows


def _run_import_test(model: str, timeout: int = 240) -> Dict[str, Any]:
    py = python_for_model(model)
    cmd = [py, *import_smoke_cmd(model)]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(paths.fm_root()),
            env=subprocess_env(model),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "python": py,
            "seconds": round(time.time() - t0, 2),
            "stdout": (result.stdout or "")[-1200:],
            "stderr": (result.stderr or "")[-1200:],
        }
    except Exception as exc:
        return {"ok": False, "python": py, "error": str(exc)}


def _required_third_party() -> Dict[str, Path]:
    root = paths.third_party_root()
    return {
        "Geneformer": root / "Geneformer",
        "uce": root / "uce",
        "state": root / "state",
        "stack": root / "stack",
        "scGPT-main": root / "scGPT-main",
        "scFoundation": root / "scFoundation",
        "scldm": root / "scldm",
        "xVERSE_code": root / "xVERSE_code",
        "CellNavi": root / "CellNavi",
        "dataset_fitted_baseline": root / "dataset_fitted_baseline",
        "nicheformer": root / "nicheformer",
        "transcriptformer": root / "transcriptformer",
    }


def _third_party_for_models(models: Iterable[str]) -> Dict[str, Path]:
    all_dirs = _required_third_party()
    need = {
        "geneformer": ["Geneformer"],
        "uce": ["uce"],
        "state": ["state"],
        "stack": ["stack"],
        "scgpt": ["scGPT-main"],
        "scfoundation": ["scFoundation"],
        "scldm": ["scldm"],
        "xverse": ["xVERSE_code"],
        "cellnavi": ["CellNavi"],
        "nicheformer": ["nicheformer"],
        "transcriptformer": ["transcriptformer"],
        "pca": ["dataset_fitted_baseline"],
        "pca_baseline": ["dataset_fitted_baseline"],
    }
    out: Dict[str, Path] = {}
    for model in models:
        for name in need.get(model.lower().strip(), []):
            out[name] = all_dirs[name]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of foundation models to validate")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset IDs expected under data roots")
    parser.add_argument("--print-only", action="store_true", help="Print resolved layout/tree without enforcing resources")
    parser.add_argument("--skip-import-test", action="store_true", help="Only check files and directories")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    model_list = [m.lower().strip() for m in (args.models or MODEL_QUEUE_ORDER)]
    report: Dict[str, Any] = {
        "layout": {
            "scfm_root": str(paths.scfm_root()),
            "delivery_root": str(paths.delivery_root()),
            "data_root": str(paths.data_root()),
            "pretrained_root": str(paths.pretrained_root()),
            "output_root": str(paths.output_root()),
            "third_party_root": str(paths.third_party_root()),
            "envs_root": str(paths.envs_root()),
            "cache_root": str(paths.cache_root()),
        },
        "models": {},
        "datasets": {},
        "third_party": {},
    }

    for name, p in _third_party_for_models(model_list).items():
        report["third_party"][name] = {"path": str(p), "exists": p.exists()}

    h5ads = _existing_h5ad(paths.default_h5ad_roots())
    report["datasets"]["n_h5ad"] = len(h5ads)
    report["datasets"]["roots"] = [str(p) for p in paths.default_h5ad_roots()]
    if args.datasets:
        found = {p.stem for p in h5ads}
        report["datasets"]["requested"] = {
            ds: any(ds == x or x.startswith(ds) for x in found) for ds in args.datasets
        }

    if not args.print_only:
        for model in model_list:
            status, detail = check_weights(model)
            entry: Dict[str, Any] = {
                "weights_status": status,
                "weights_detail": detail,
                "python": python_for_model(model),
            }
            if not args.skip_import_test:
                entry["import"] = _run_import_test(model)
            report["models"][model] = entry

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(paths.describe_layout())
        print(paths.expected_tree())
        print(f"n_h5ad={report['datasets']['n_h5ad']}")
        if args.datasets:
            for ds, ok in report["datasets"]["requested"].items():
                print(f"dataset {ds}: {'OK' if ok else 'MISSING'}")
        if not args.print_only:
            for model, entry in report["models"].items():
                imp = entry.get("import")
                imp_s = " skipped" if imp is None else f" import={'OK' if imp.get('ok') else 'FAIL'}"
                print(f"{model}: weights={entry['weights_status']} {entry['weights_detail']}{imp_s}")

    if args.print_only:
        return 0

    missing: List[str] = []
    for name, entry in report["third_party"].items():
        if not entry["exists"]:
            missing.append(f"third_party/{name}: {entry['path']}")
    for model, entry in report["models"].items():
        if entry["weights_status"] != "ready":
            missing.append(f"{model}: {entry['weights_detail']}")
        imp = entry.get("import")
        if imp is not None and not imp.get("ok"):
            missing.append(f"{model} import failed: {imp.get('error') or imp.get('stderr') or imp.get('returncode')}")
    if args.datasets:
        for ds, ok in report["datasets"].get("requested", {}).items():
            if not ok:
                missing.append(f"dataset {ds} not found under data roots")

    if missing:
        print("Missing or failing resources:", file=sys.stderr)
        for row in missing:
            print(f"  - {row}", file=sys.stderr)
        return 2
    print("OK: scFM resources validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
