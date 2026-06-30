#!/usr/bin/env python3
"""
生成 / 覆盖 canonical train-test split JSON。

产出文件：``{biflow_dir}/split_seed{seed}.json``。
latent / raw / CoupledFM 训练均从此文件读取，保证三端同一 split，无泄露、可对比。

用法：
    python tools/build_split.py                          # 默认 biflow_dir + seed=42
    python tools/build_split.py --seed 7 --force
    python tools/build_split.py --datasets Adamson Frangieh

在 CoupledFM 根目录执行：会把仓库根加入 PYTHONPATH，以便 ``import model``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_root_to_path() -> Path:
    # .../CoupledFM/model/tools/build_split.py -> parents[2] == CoupledFM repo root
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    return root


def main():
    root = _add_repo_root_to_path()

    from model.utils.data.split import (
        build_canonical_split,
        canonical_split_path,
        save_split,
    )
    from model import paths
    from model.utils.data.vocab import GeneVocab

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--biflow-dir",
        default=str(paths.biflow_dir()),
        help="biFlow 数据根目录（control_{backbone}/gt_{backbone}；stack/scLDM/scFoundation 可走 data/latent_data/{backbone}；兼容 control_center/与 gt/）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument(
        "--coupling-mode",
        default="coupled",
        choices=["baseline", "ot", "coupled"],
        help="coupled 与 ot 训练模式要求数据集带 latent（用于 OT），baseline 不需要",
    )
    parser.add_argument(
        "--gene-name-path",
        default=str(paths.gene_name_path()),
    )
    parser.add_argument(
        "--nichenet-node2idx-path",
        default=str(paths.nichenet_node2idx_path()),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="仅扫描这些数据集；缺省扫全部",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 canonical 已存在也重新生成并覆盖",
    )
    parser.add_argument(
        "--ot-feature",
        default="latent",
        choices=["latent", "de", "raw"],
        help="ot 模式下的特征空间：latent / de(需 --de-dir) / raw。coupled 默认 latent。",
    )
    parser.add_argument(
        "--de-dir",
        default=None,
        help="DE JSON 目录（ot-feature=de 时使用）：含 {ds}.json",
    )
    parser.add_argument(
        "--latent-backbone",
        default="stack",
        choices=["state", "uce", "stack", "scldm", "scfoundation"],
        help="扫描时使用的 latent 布局；stack/scLDM/scFoundation 优先 data/latent_data/{backbone}",
    )
    args = parser.parse_args()

    out_path = canonical_split_path(args.biflow_dir, args.seed)
    if out_path.exists() and not args.force:
        print(f"[build_split] 已存在 {out_path}；加 --force 可重建。")
        return

    print(f"[build_split] loading GeneVocab ...", flush=True)
    vocab = GeneVocab(args.gene_name_path, args.nichenet_node2idx_path)

    print(
        f"[build_split] scanning {args.biflow_dir}  seed={args.seed}  "
        f"min_cells={args.min_cells}  coupling={args.coupling_mode}  "
        f"ot_feature={args.ot_feature}  latent_backbone={args.latent_backbone} ...",
        flush=True,
    )
    split = build_canonical_split(
        biflow_dir=args.biflow_dir,
        vocab=vocab,
        seed=args.seed,
        min_cells=args.min_cells,
        coupling_mode=args.coupling_mode,
        dataset_names=args.datasets,
        verbose=True,
        ot_feature=args.ot_feature,
        de_dir=args.de_dir,
        latent_backbone=args.latent_backbone,
    )

    save_split(out_path, split)
    n_ds = len(split)
    n_train = sum(len(sp["train"]) for sp in split.values())
    n_test = sum(len(sp["test"]) for sp in split.values())
    print(
        f"[build_split] ✅ 写入 {out_path}\n"
        f"             datasets={n_ds}  train_conds={n_train}  test_conds={n_test}",
        flush=True,
    )


if __name__ == "__main__":
    main()
