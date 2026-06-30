"""
统一的 train / test 划分（latent / raw / coupled 三端共享）。

Canonical JSON 存放位置：
    {biflow_dir}/split_seed{seed}.json

Schema：
    {
        "<ds_name>": {"train": [cond, ...], "test": [cond, ...]},
        ...
    }

本模块是 **单一真相源**：``tools/build_split.py``、``train.py``、``pert_split`` 均由此构建/读取。

``_DatasetHandle`` 来自 ``model.data.dataset``（训练入口统一为 ``model`` 包）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from model.utils.data.biflow_paths import (
    iter_biflow_dataset_stems,
    resolve_biflow_control_gt_h5ad,
)


# ---------------------------------------------------------------------------
# Policy parameters
# ---------------------------------------------------------------------------

_CAP_SINGLE_ONLY_TEST = 30
_CAP_SINGLE_WHEN_MULTI_DS = 25


def is_multi_pert(cond: str) -> bool:
    """多基因扰动：biFlow 中以 ``+`` 连接（如 ``A+B``）。"""
    return "+" in str(cond)


def pert_components(cond: str) -> List[str]:
    """Return normalized perturbation components from a condition string."""
    return [p.strip() for p in str(cond).split("+") if p.strip()]


def n_single_pert_holdout(n_single: int, cap: int) -> int:
    """单基因 held-out test 条数：min(cap, n, max(ceil(0.4*n), 1))。"""
    if n_single <= 0:
        return 0
    return min(cap, n_single, max(1, int(math.ceil(0.4 * n_single))))


def split_condition_lists(
    single: List[str],
    multi: List[str],
    rng: np.random.RandomState,
) -> tuple[List[str], List[str], List[str], List[str]]:
    """Return train/test condition lists plus single/multi test components.

    Single perturbations are split into train/test. Multi-gene perturbations are
    kept entirely in test as OOD combinations; evaluation-time caps, not the
    split file, should control how many are scored in one run.
    """
    n_s, n_m = len(single), len(multi)
    if n_m == 0:
        k_s = n_single_pert_holdout(n_s, cap=_CAP_SINGLE_ONLY_TEST)
    else:
        k_s = n_single_pert_holdout(n_s, cap=_CAP_SINGLE_WHEN_MULTI_DS)

    if n_s > 0 and k_s > 0:
        idx = rng.choice(n_s, size=k_s, replace=False)
        single_test = [single[i] for i in idx]
    else:
        single_test = []

    st = set(single_test)
    single_train = [c for c in single if c not in st]
    multi_test = list(multi)
    return single_train, single_test + multi_test, single_test, multi_test


def classify_multi_perturbation_tests(
    multi_test: List[str],
    train_single: List[str],
) -> Dict[str, List[str]]:
    """Split multi-perturbation tests by component visibility.

    Definitions used for formal OOD evaluation:

    - ``seen``: all component single perturbations are present in train.
    - ``unseen1``: exactly one component is absent from train.
    - ``unseen2``: two or more components are absent from train.

    The labels describe component-level visibility, not whether the exact
    combination was observed. Current policy keeps all multi perturbations out
    of train, so every multi condition is a held-out combination.
    """
    seen_train = set(map(str, train_single))
    groups = {"seen": [], "unseen1": [], "unseen2": []}
    for cond in multi_test:
        comps = pert_components(cond)
        n_unseen = sum(1 for c in comps if c not in seen_train)
        if n_unseen == 0:
            groups["seen"].append(cond)
        elif n_unseen == 1:
            groups["unseen1"].append(cond)
        else:
            groups["unseen2"].append(cond)
    return groups


# ---------------------------------------------------------------------------
# Canonical path helpers
# ---------------------------------------------------------------------------

def canonical_split_path(biflow_dir: str | Path, seed: int = 42) -> Path:
    """返回 canonical split JSON 路径：``{biflow_dir}/split_seed{seed}.json``。"""
    return Path(biflow_dir) / f"split_seed{seed}.json"


def _import_dataset_handle():
    """Load raw biFlow dataset handle from the unified ``model`` package."""
    from model.data.dataset import _DatasetHandle

    return _DatasetHandle


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _valid_conds_for_handle(h, min_cells: int) -> List[str]:
    """筛出控制池与 gt 池 cell 数都 >= min_cells 的 condition。"""
    out: List[str] = []
    keys = set(h.pert_cond2idx.keys()) & set(h.gt_cond2idx.keys())
    for cond in sorted(keys):
        if len(h.pert_cond2idx[cond]) < min_cells:
            continue
        if len(h.gt_cond2idx[cond]) < min_cells:
            continue
        out.append(cond)
    return out


def build_canonical_split(
    biflow_dir: str | Path,
    vocab,
    seed: int = 42,
    min_cells: int = 16,
    coupling_mode: str = "coupled",
    dataset_names: Optional[List[str]] = None,
    verbose: bool = True,
    ot_feature: str = "latent",
    de_dir: Optional[str] = None,
    latent_backbone: str = "state",
) -> Dict[str, Dict[str, List[str]]]:
    """扫描 biFlow AnnData（优先 ``control_{backbone}/`` + ``gt_{backbone}/``，兼容 ``control_center/`` + ``gt/``），
    按策略生成 canonical split 字典。

    - ``coupling_mode == "coupled"``：要求 handle 有 latent。
    - ``coupling_mode == "ot"`` 且 ``ot_feature == "latent"``：要求 latent。
    - ``coupling_mode == "ot"`` 且 ``ot_feature == "de"``：要求 ``de_dir/{ds}.json``
      与 h5ad var 交集非空（见 ``_DatasetHandle.has_de``）。
    - ``coupling_mode == "ot"`` 且 ``ot_feature == "raw"``：不要求 latent。
    """
    _DatasetHandle = _import_dataset_handle()
    of = (ot_feature or "latent").lower()
    need_latent = (coupling_mode == "coupled") or (
        coupling_mode == "ot" and of == "latent"
    )
    de_root = Path(de_dir) if de_dir else None
    rng = np.random.RandomState(seed)
    biflow_dir = Path(biflow_dir)
    split: Dict[str, Dict[str, List[str]]] = {}

    ds_names = iter_biflow_dataset_stems(biflow_dir, latent_backbone=latent_backbone)
    n_added = 0
    for ds_name in ds_names:
        if dataset_names and ds_name not in dataset_names:
            continue
        pair = resolve_biflow_control_gt_h5ad(
            biflow_dir,
            ds_name,
            latent_backbone=latent_backbone,
        )
        if pair is None:
            continue
        ctrl_p, gt_p = pair

        de_list = None
        if coupling_mode == "ot" and of == "de" and de_root is not None:
            dj = de_root / f"{ds_name}.json"
            if dj.exists():
                de_list = json.loads(dj.read_text(encoding="utf-8"))
            else:
                de_list = []

        h = _DatasetHandle(
            ds_name,
            str(ctrl_p),
            str(gt_p),
            vocab,
            load_latent=need_latent,
            de_gene_list=de_list,
        )
        if need_latent and not h.has_latent:
            h.close()
            continue
        if coupling_mode == "ot" and of == "de" and not h.has_de:
            h.close()
            continue

        valid = _valid_conds_for_handle(h, min_cells)
        single = [c for c in valid if not is_multi_pert(c)]
        multi = [c for c in valid if is_multi_pert(c)]
        train_conds, test_conds, single_test, multi_test = split_condition_lists(
            single, multi, rng,
        )
        multi_groups = classify_multi_perturbation_tests(
            multi_test=multi_test,
            train_single=train_conds,
        )

        h.close()

        if not test_conds and not train_conds:
            continue

        split[ds_name] = {
            "train": train_conds,
            "test": test_conds,
            "test_single": single_test,
            "test_multi": multi_test,
            "test_multi_seen": multi_groups["seen"],
            "test_multi_unseen1": multi_groups["unseen1"],
            "test_multi_unseen2": multi_groups["unseen2"],
        }
        n_added += 1
        if verbose:
            kind = "single-only" if len(multi) == 0 else "single+multi"
            print(
                f"[unified_split] [{n_added}] {ds_name}  ({kind})  "
                f"train={len(train_conds)}  test={len(test_conds)}  "
                f"[test: single={len(single_test)} multi={len(multi_test)} "
                f"seen={len(multi_groups['seen'])} "
                f"unseen1={len(multi_groups['unseen1'])} "
                f"unseen2={len(multi_groups['unseen2'])}]",
                flush=True,
            )

    return split


def save_split(path: str | Path, split: Dict[str, Dict[str, List[str]]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)


def load_split_json(path: str | Path) -> Dict[str, Dict[str, List[str]]]:
    with open(path) as f:
        return json.load(f)


def load_or_build_unified_split(
    biflow_dir: str | Path,
    vocab=None,
    seed: int = 42,
    min_cells: int = 16,
    coupling_mode: str = "coupled",
    dataset_names: Optional[List[str]] = None,
    write: bool = True,
    verbose: bool = True,
    ot_feature: str = "latent",
    de_dir: Optional[str] = None,
    latent_backbone: str = "state",
) -> Dict[str, Dict[str, List[str]]]:
    """
    读取 canonical split；不存在则按策略构建（需要 vocab）。

    ``dataset_names`` 只用于在构建时限定扫描范围，不用于对已存在 canonical 做过滤，
    避免被动缩小全局 split（保证全局唯一性）。
    """
    p = canonical_split_path(biflow_dir, seed)
    if p.exists():
        return load_split_json(p)

    if vocab is None:
        raise FileNotFoundError(
            f"Canonical split JSON not found at {p}. "
            f"Build it once via:\n"
            f"  python tools/build_split.py --biflow-dir {biflow_dir} --seed {seed}"
        )

    split = build_canonical_split(
        biflow_dir,
        vocab,
        seed=seed,
        min_cells=min_cells,
        coupling_mode=coupling_mode,
        dataset_names=dataset_names,
        verbose=verbose,
        ot_feature=ot_feature,
        de_dir=de_dir,
        latent_backbone=latent_backbone,
    )
    if write:
        save_split(p, split)
        if verbose:
            print(f"[unified_split] wrote canonical split → {p}", flush=True)
    return split


__all__ = [
    "is_multi_pert",
    "pert_components",
    "split_condition_lists",
    "classify_multi_perturbation_tests",
    "n_single_pert_holdout",
    "canonical_split_path",
    "build_canonical_split",
    "save_split",
    "load_split_json",
    "load_or_build_unified_split",
]
