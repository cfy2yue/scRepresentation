"""Shared filesystem layout for the standalone scdfm handoff package.

The delivery root is the parent of this ``model`` package and is expected to
contain sibling ``dataset`` and ``pretrainckpt`` directories:

    delivery_root/
      model/
      dataset/
      pretrainckpt/

Environment variables are intentionally narrow and explicit so the handoff can
run without a sibling ``scFM`` checkout.
"""

from __future__ import annotations

import os
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent


def delivery_root() -> Path:
    return Path(os.environ.get("SCDFM_DELIVERY_ROOT", MODEL_DIR.parent)).expanduser()


def dataset_root() -> Path:
    return Path(os.environ.get("SCDFM_DATASET_ROOT", delivery_root() / "dataset")).expanduser()


def pretrain_root() -> Path:
    return Path(os.environ.get("SCDFM_PRETRAIN_ROOT", delivery_root() / "pretrainckpt")).expanduser()


def gene_cache_root() -> Path:
    return Path(
        os.environ.get("SCDFM_GENE_CACHE_ROOT", pretrain_root() / "genepert_cache")
    ).expanduser()


def biflow_dir() -> Path:
    return dataset_root() / "biFlow_data"


def de_dir() -> Path:
    return biflow_dir() / "de1024"


def cellgene_processed_dir() -> Path:
    return dataset_root() / "cellgene_census" / "processed"


def cellnavi_data_dir() -> Path:
    """Return the CellNavi resource dir.

    ``COUPLEDFM_SCFM_ROOT`` is supported only as an explicit compatibility
    override. It is never used as an implicit default.
    """

    if "SCDFM_PRETRAIN_ROOT" in os.environ:
        return pretrain_root() / "cellnavi" / "data"
    if "COUPLEDFM_SCFM_ROOT" in os.environ:
        legacy = (
            Path(os.environ["COUPLEDFM_SCFM_ROOT"]).expanduser()
            / "pretrained"
            / "cellnavi"
            / "data"
        )
        if legacy.exists():
            return legacy
    return pretrain_root() / "cellnavi" / "data"


def gene_name_path() -> Path:
    return cellnavi_data_dir() / "gene_name.txt"


def nichenet_node2idx_path() -> Path:
    return cellnavi_data_dir() / "Nichenet" / "node2idx.json"


def nichenet_graph_pkl_path() -> Path:
    return cellnavi_data_dir() / "Nichenet" / "graph.pkl"


def cellnavi_pretrain_ckpt_path() -> Path:
    return cellnavi_data_dir() / "pretrain" / "pretrain_weights.pth"


def cellnavi_cache_dir() -> Path:
    return gene_cache_root() / "cellnavi_embed_gene"


def scgpt_cache_dir() -> Path:
    return gene_cache_root() / "scgpt_embed_gene"


def describe_layout() -> str:
    return "\n".join(
        [
            f"delivery_root={delivery_root()}",
            f"dataset_root={dataset_root()}",
            f"pretrain_root={pretrain_root()}",
            f"gene_cache_root={gene_cache_root()}",
            f"biflow_dir={biflow_dir()}",
            f"cellgene_processed_dir={cellgene_processed_dir()}",
            f"cellnavi_data_dir={cellnavi_data_dir()}",
        ]
    )
