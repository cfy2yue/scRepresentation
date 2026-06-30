#!/usr/bin/env python3
"""Run paired LatentFM bootstrap summaries for every row in a posthoc manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BOOTSTRAP = ROOT / "ops/bootstrap_latentfm_paired_posthoc_20260621.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def manifest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("launched_runs")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    if {"baseline_split_json", "run_split_json"}.issubset(payload):
        return [payload]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split-groups", nargs="+", default=["test", "test_multi_unseen2"])
    parser.add_argument("--family-groups", nargs="+", default=["family_gene"])
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pearson_pert", "pearson_ctrl", "direct_pearson", "test_mmd_clamped"],
    )
    args = parser.parse_args()

    payload = load_json(args.manifest)
    rows = manifest_rows(payload)
    if not rows:
        raise ValueError(f"No manifest rows found in {args.manifest}")
    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(BOOTSTRAP)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        run_name = str(row.get("run_name") or row.get("arm") or f"row{i}")
        stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
        for kind, base_key, cand_key, groups in (
            ("split", "baseline_split_json", "run_split_json", args.split_groups),
            ("family", "baseline_family_json", "run_family_json", args.family_groups),
        ):
            base = str(row.get(base_key) or "").strip()
            cand = str(row.get(cand_key) or "").strip()
            if not base or not cand:
                continue
            base_path = Path(base)
            cand_path = Path(cand)
            if not base_path.is_file() or not cand_path.is_file():
                continue
            out_json = args.out_dir / f"{stem}.{kind}.bootstrap.json"
            out_md = args.out_dir / f"{stem}.{kind}.bootstrap.md"
            cmd = [
                args.python,
                str(BOOTSTRAP),
                "--baseline-json",
                str(base_path),
                "--candidate-json",
                str(cand_path),
                "--groups",
                *groups,
                "--metrics",
                *args.metrics,
                "--n-boot",
                str(int(args.n_boot)),
                "--seed",
                str(int(args.seed)),
                "--title",
                f"LatentFM Paired Bootstrap: {run_name} ({kind})",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
            run(cmd)
            written.append({"run_name": run_name, "kind": kind, "json": str(out_json), "md": str(out_md)})

    index = {
        "manifest": str(args.manifest),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "outputs": written,
    }
    index_path = args.out_dir / "bootstrap_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"index": str(index_path), "n_outputs": len(written)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
