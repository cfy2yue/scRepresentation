#!/usr/bin/env python3
"""Run uncapped LatentFM split/family posthoc from a manifest.

This is a promotion-stage tool. It intentionally sets all eval caps to ``0`` so
the checkpoint config's capped smoke settings do not leak into the final
posthoc.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
DEFAULT_ANCHOR = COUPLED / "output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/scfoundation"
DEFAULT_BIFLOW_DIR = ROOT / "dataset/biFlow_data"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def manifest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("launched_runs")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    if isinstance(payload, dict) and ("run_name" in payload or "candidate_checkpoint" in payload):
        return [payload]
    return []


def row_candidate_checkpoint(row: dict[str, Any]) -> Path:
    ckpt = str(row.get("candidate_checkpoint") or "").strip()
    if ckpt:
        return Path(ckpt)
    out_dir = str(row.get("out_dir") or "").strip()
    if out_dir:
        return Path(out_dir) / "best.pt"
    raise ValueError(f"Cannot infer candidate checkpoint for row {row}")


def row_anchor_checkpoint(payload: dict[str, Any], row: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    for obj in (row, payload):
        ckpt = str(obj.get("anchor_checkpoint") or "").strip()
        if ckpt:
            return Path(ckpt)
    return DEFAULT_ANCHOR


def run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.check_call(cmd)


def eval_pair(
    *,
    python: str,
    anchor_ckpt: Path,
    candidate_ckpt: Path,
    split_file: Path,
    data_dir: Path,
    biflow_dir: Path,
    out_dir: Path,
    gpu: int,
    ode_steps: int,
    max_chunk: int,
    eval_max_mse_cells: int,
    eval_max_mmd_cells: int,
    force_support_context_absent: bool,
    split_groups: list[str] | None,
    family_groups: list[str] | None,
    dry_run: bool,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_split = out_dir / "split_group_eval_anchor_uncapped_ode20.json"
    base_family = out_dir / "condition_family_eval_anchor_uncapped_ode20.json"
    cand_split = out_dir / "split_group_eval_candidate_uncapped_ode20.json"
    cand_family = out_dir / "condition_family_eval_candidate_uncapped_ode20.json"

    common = [
        "--data-dir", str(data_dir),
        "--biflow-dir", str(biflow_dir),
        "--split-file", str(split_file),
        "--gpu", str(gpu),
        "--ode-steps", str(ode_steps),
        "--max-chunk", str(max_chunk),
        "--eval-max-conditions", "0",
        "--eval-max-conditions-per-dataset", "0",
        "--eval-max-mse-cells", str(int(eval_max_mse_cells)),
        "--eval-max-mmd-cells", str(int(eval_max_mmd_cells)),
    ]
    support_absent_args = ["--force-support-context-absent"] if force_support_context_absent else []
    split_groups = split_groups or [
        "test", "test_single", "test_multi", "test_multi_seen",
        "test_multi_unseen1", "test_multi_unseen2",
    ]
    family_groups = family_groups or [
        "test_all", "family_gene", "family_drug", "structure_single",
        "structure_multi", "test_single", "test_multi", "test_multi_seen",
        "test_multi_unseen1", "test_multi_unseen2",
    ]
    for ckpt, out in ((anchor_ckpt, base_split), (candidate_ckpt, cand_split)):
        run([
            python, "-m", "model.latent.eval_split_groups",
            "--checkpoint", str(ckpt),
            "--groups", *split_groups,
            "--out", str(out),
            *support_absent_args,
            *common,
        ], dry_run=dry_run)
    for ckpt, out in ((anchor_ckpt, base_family), (candidate_ckpt, cand_family)):
        run([
            python, "-m", "model.latent.eval_condition_families",
            "--checkpoint", str(ckpt),
            "--groups", *family_groups,
            "--out", str(out),
            *support_absent_args,
            *common,
        ], dry_run=dry_run)
    return {
        "baseline_split_json": str(base_split),
        "baseline_family_json": str(base_family),
        "run_split_json": str(cand_split),
        "run_family_json": str(cand_family),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--biflow-dir", type=Path, default=DEFAULT_BIFLOW_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--max-chunk", type=int, default=512)
    parser.add_argument("--eval-max-mse-cells", type=int, default=0)
    parser.add_argument("--eval-max-mmd-cells", type=int, default=0)
    parser.add_argument(
        "--split-groups",
        nargs="+",
        default=None,
        help="Optional split groups override. Default preserves historical uncapped groups.",
    )
    parser.add_argument(
        "--family-groups",
        nargs="+",
        default=None,
        help="Optional family groups override. Default preserves historical uncapped groups.",
    )
    parser.add_argument("--only-run-name", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.manifest)
    rows = manifest_rows(payload)
    if not rows:
        raise ValueError(f"No manifest rows found in {args.manifest}")
    wanted = set(args.only_run_name or [])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        run_name = str(row.get("run_name") or row.get("arm") or f"row{i}")
        if wanted and run_name not in wanted and str(row.get("arm") or "") not in wanted:
            continue
        split_file = Path(
            str(
                row.get("split_file")
                or row.get("eval_split_file")
                or payload.get("split_file")
                or payload.get("eval_split_file")
                or ""
            )
        )
        if not split_file:
            raise ValueError(f"Missing split_file for row {run_name}")
        anchor_ckpt = row_anchor_checkpoint(payload, row, args.anchor_checkpoint)
        candidate_ckpt = row_candidate_checkpoint(row)
        data_dir = Path(str(row.get("data_dir") or payload.get("data_dir") or args.data_dir))
        biflow_dir = Path(str(row.get("biflow_dir") or payload.get("biflow_dir") or args.biflow_dir))
        force_support_context_absent = bool(
            row.get("force_support_context_absent")
            if "force_support_context_absent" in row
            else payload.get("force_support_context_absent", False)
        )
        if not anchor_ckpt.is_file():
            raise FileNotFoundError(anchor_ckpt)
        if not candidate_ckpt.is_file():
            raise FileNotFoundError(candidate_ckpt)
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        if not biflow_dir.exists():
            raise FileNotFoundError(biflow_dir)
        row_out = args.out_dir / sanitize(run_name)
        paths = eval_pair(
            python=args.python,
            anchor_ckpt=anchor_ckpt,
            candidate_ckpt=candidate_ckpt,
            split_file=split_file,
            data_dir=data_dir,
            biflow_dir=biflow_dir,
            out_dir=row_out,
            gpu=int(args.gpu),
            ode_steps=int(args.ode_steps),
            max_chunk=int(args.max_chunk),
            eval_max_mse_cells=int(args.eval_max_mse_cells),
            eval_max_mmd_cells=int(args.eval_max_mmd_cells),
            force_support_context_absent=force_support_context_absent,
            split_groups=args.split_groups,
            family_groups=args.family_groups,
            dry_run=bool(args.dry_run),
        )
        outputs.append({
            "run_name": run_name,
            "arm": row.get("arm"),
            "split_file": str(split_file),
            "data_dir": str(data_dir),
            "biflow_dir": str(biflow_dir),
            "anchor_checkpoint": str(anchor_ckpt),
            "candidate_checkpoint": str(candidate_ckpt),
            "force_support_context_absent": force_support_context_absent,
            **paths,
        })

    index = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "uncapped": True,
        "eval_caps": {
            "eval_max_conditions": 0,
            "eval_max_conditions_per_dataset": 0,
            "eval_max_mse_cells": int(args.eval_max_mse_cells),
            "eval_max_mmd_cells": int(args.eval_max_mmd_cells),
            "split_groups": args.split_groups,
            "family_groups": args.family_groups,
        },
        "outputs": outputs,
    }
    index_path = args.out_dir / "uncapped_posthoc_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"index": str(index_path), "n_outputs": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
