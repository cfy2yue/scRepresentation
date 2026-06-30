#!/usr/bin/env python3
"""Lightweight repo checks (no torch import; uses ast + pathlib).

NOTE (delivery scope): this module is NOT exercised by the two delivered
flows (raw flow pretrain & CoupledFM sweep / CellNavi-vs-scGPT compare).
It scans sibling ``scFM/`` / ``data/`` directories if present and is kept
for repo-level lint only. Use ``model.tools.validate_resources`` for the
runtime resource layout check instead.
"""
import argparse
import ast
import sys
from pathlib import Path

# Default: CoupledFM repo root (.../CoupledFM) inferred from this file under model/tools/
ROOT = Path(__file__).resolve().parents[2]

BLACKLIST = [
    "/data2/cfy/coupledFM",
    "/data2/cfy/FM/FM/CoupledFM",
    "/coupledFM/CoupledFM",
    "coupledfm_core",
    "latentFM/",
    "rawexprFM/",
]


def check_py(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"))


def scan_blacklist() -> list[str]:
    err: list[str] = []
    skip_dirs = {"__pycache__", ".git", "envs"}
    for subdir in [
        ROOT / "model",
        ROOT / "scFM",
        ROOT / "data",
    ]:
        if not subdir.exists():
            continue
        for p in sorted(subdir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix not in {".py", ".sh", ".toml"}:
                continue
            if any(part in skip_dirs for part in p.parts):
                continue
            if p.name == "validate_repo.py":
                continue
            text = p.read_text(encoding="utf-8")
            for pat in BLACKLIST:
                if pat in text:
                    err.append(f"blacklist '{pat}' in {p}")
    return err


def main() -> int:
    ap = argparse.ArgumentParser(description="Lightweight CoupledFM tree checks (no torch).")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="CoupledFM repository root (default: infer from this script)",
    )
    args = ap.parse_args()
    global ROOT
    root = args.root.expanduser().resolve() if args.root is not None else ROOT
    ROOT = root

    errors: list[str] = []
    model_dir = ROOT / "model"
    if not model_dir.is_dir():
        errors.append(f"missing directory {model_dir}")
    else:
        for p in sorted(model_dir.rglob("*.py")):
            try:
                check_py(p)
            except SyntaxError as e:
                errors.append(f"{p}: {e}")

    for p in [
        ROOT / "model" / "latent" / "train.py",
        ROOT / "model" / "latent" / "prepare_fm_data.py",
    ]:
        if p.exists():
            try:
                check_py(p)
            except SyntaxError as e:
                errors.append(f"{p}: {e}")

    for req in [
        ROOT / "README.md",
        ROOT / "model" / "utils" / "models" / "attention.py",
        ROOT / "dataset" / "biFlow_data" / "README.md",
        ROOT / "model" / "data" / "pert_split.py",
    ]:
        if not req.exists():
            errors.append(f"missing {req}")

    pert_split = ROOT / "model" / "data" / "pert_split.py"
    if pert_split.exists():
        text = pert_split.read_text(encoding="utf-8")
        if "ctrl_cluster" in text or "perturbed" in text:
            errors.append(
                "model/data/pert_split.py still references legacy ctrl_cluster/perturbed layout"
            )

    errors.extend(scan_blacklist())

    if errors:
        print("FAILED:\n" + "\n".join(errors))
        return 1
    print("OK: ast-parse model + key scripts; required files; blacklist clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
