#!/usr/bin/env python3
"""
Merge per-run metric JSON files into one table (JSON lines or CSV).

Usage:
  python -m cli.aggregate_report --inputs 'output/metrics/**/*.json' --out-csv summary.csv
  python -m cli.aggregate_report --inputs 'output/dataset_fitted/**/metrics.json' --out-jsonl summary.jsonl
  python -m cli.aggregate_report --write-scfm-benchmark-csvs [--scfm-root <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

BENCH_ROOT = Path(__file__).resolve().parents[1]
SCFM_ROOT = BENCH_ROOT.parent
FM_ROOT = SCFM_ROOT / "fm"
sys.path.insert(0, str(FM_ROOT))
import paths


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p) as f:
        return json.load(f)


def aggregate(
    paths: List[Path],
    *,
    path_key: str = "source_path",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        row = dict(_load_json(p))
        row[path_key] = str(p.resolve())
        rows.append(row)
    return rows


def _collect_scfm_summaries(metrics_root: Path) -> List[Path]:
    out: List[Path] = []
    if not metrics_root.is_dir():
        return out
    for p in sorted(metrics_root.glob("*/*/*/summary.json")):
        if p.parent.name in ("raw", "pca128"):
            out.append(p)
    return out


def _rows_from_summaries(paths: List[Path], metrics_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = metrics_root.resolve()
    for p in paths:
        rel = p.resolve().relative_to(root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        model, dataset_id, latent_space = parts[0], parts[1], parts[2]
        data = dict(_load_json(p))
        data["model"] = model
        data["dataset_id"] = dataset_id
        data["latent_space"] = latent_space
        data["summary_path"] = str(p)
        rows.append(data)
    return rows


def _write_scfm_benchmark_csvs(scfm: Path) -> None:
    import pandas as pd

    metrics_root = Path(
        paths.output_root()
        if scfm == SCFM_ROOT.resolve()
        else scfm.parent / "scFM_output"
    ) / "metrics"
    summary_paths = _collect_scfm_summaries(metrics_root)
    rows = _rows_from_summaries(summary_paths, metrics_root)
    if not rows:
        print(json.dumps({"warn": "no summaries found", "metrics_root": str(metrics_root)}))
        return
    df = pd.json_normalize(rows)
    metrics_root.mkdir(parents=True, exist_ok=True)
    raw_df = df[df["latent_space"] == "raw"]
    pca_df = df[df["latent_space"] == "pca128"]
    raw_df.to_csv(metrics_root / "summary_all_raw.csv", index=False)
    pca_df.to_csv(metrics_root / "summary_all_pca128.csv", index=False)
    df.to_csv(metrics_root / "summary_all.csv", index=False)
    print(
        json.dumps(
            {
                "summary_all_raw": str(metrics_root / "summary_all_raw.csv"),
                "summary_all_pca128": str(metrics_root / "summary_all_pca128.csv"),
                "summary_all": str(metrics_root / "summary_all.csv"),
                "n_raw": len(raw_df),
                "n_pca128": len(pca_df),
                "n_total": len(df),
            }
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="*", default=None, help="JSON files or glob patterns")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--path-key", type=str, default="source_path")
    ap.add_argument(
        "--write-scfm-benchmark-csvs",
        action="store_true",
        help="Scan output/metrics/*/*/{raw,pca128}/summary.json and write summary_all*.csv",
    )
    ap.add_argument("--scfm-root", type=Path, default=SCFM_ROOT)
    args = ap.parse_args()

    if args.write_scfm_benchmark_csvs:
        _write_scfm_benchmark_csvs(args.scfm_root.resolve())
        return 0

    if not args.inputs:
        raise SystemExit("Provide --inputs or --write-scfm-benchmark-csvs")

    files: List[Path] = []
    for pat in args.inputs:
        if any(ch in pat for ch in "*?[]"):
            files.extend(Path(p) for p in sorted(glob(pat, recursive=True)))
        else:
            files.append(Path(pat))
    files = [p for p in files if p.is_file()]

    rows = aggregate(files, path_key=args.path_key)
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
    if args.out_csv:
        try:
            import pandas as pd
        except ImportError as e:
            raise SystemExit("CSV output needs pandas") from e
        df = pd.json_normalize(rows)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    if not args.out_csv and not args.out_jsonl:
        print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
