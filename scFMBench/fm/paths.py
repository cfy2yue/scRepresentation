"""Path resolution for the lightweight scFM benchmark checkout.

The code repository is ``scFM/``. Large resources live beside it by default:
``scFM_data/``, ``scFM_pretrained/``, ``scFM_output/``, ``scFM_third_party/``,
and ``scFM_envs/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def _expand(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def fm_root() -> Path:
    return Path(__file__).resolve().parent


def scfm_root() -> Path:
    return fm_root().parent


def delivery_root() -> Path:
    return _expand(os.environ.get("SCFM_DELIVERY_ROOT", str(scfm_root().parent)))


def data_root() -> Path:
    return _expand(os.environ.get("SCFM_DATA_ROOT", str(delivery_root() / "scFM_data")))


def pretrained_root() -> Path:
    return _expand(
        os.environ.get(
            "SCFM_PRETRAINED_ROOT",
            os.environ.get("COUPLEDFM_PRETRAINED_ROOT", str(delivery_root() / "scFM_pretrained")),
        )
    )


def output_root() -> Path:
    return _expand(
        os.environ.get(
            "SCFM_OUTPUT_ROOT",
            os.environ.get("LATENT_BENCH_OUTPUT_ROOT", str(delivery_root() / "scFM_output")),
        )
    )


def third_party_root() -> Path:
    return _expand(os.environ.get("SCFM_THIRD_PARTY_ROOT", str(delivery_root() / "scFM_third_party")))


def envs_root() -> Path:
    return _expand(os.environ.get("SCFM_ENVS_ROOT", str(delivery_root() / "scFM_envs")))


def cache_root() -> Path:
    return _expand(os.environ.get("SCFM_CACHE_ROOT", str(delivery_root() / "scFM_cache")))


def staging_root() -> Path:
    return data_root() / "staging"


def raw_roots() -> List[Path]:
    raw_env = os.environ.get("SCFM_RAW_ROOTS", "").strip()
    if raw_env:
        return [_expand(p) for p in raw_env.split(os.pathsep) if p.strip()]
    raw = data_root() / "raw"
    return [
        raw / "atlas_TS",
        raw / "chemicalpert_bench",
        raw / "genepert_bench",
    ]


def default_h5ad_roots() -> List[Path]:
    return [
        *raw_roots(),
        staging_root() / "atlas",
        staging_root() / "genepert",
        staging_root() / "chempert",
    ]


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def describe_layout() -> str:
    lines = [
        f"scfm_root={scfm_root()}",
        f"delivery_root={delivery_root()}",
        f"data_root={data_root()}",
        f"pretrained_root={pretrained_root()}",
        f"output_root={output_root()}",
        f"third_party_root={third_party_root()}",
        f"envs_root={envs_root()}",
        f"cache_root={cache_root()}",
    ]
    return "\n".join(lines)


def expected_tree() -> str:
    return """Expected layout:
<delivery_root>/
  scFM/
  scFM_data/
    staging/{atlas,chempert,genepert}/*.h5ad
    raw/{atlas_TS,chemicalpert_bench,genepert_bench}/*.h5ad  # optional
  scFM_pretrained/
    geneformer/Geneformer-V2-316M/
    uce/model_files/
    state/SE-600M/
    stack/{bc_large.ckpt,basecount_1000per_15000max.pkl}
    scdlm/vae_census/
    xVerse/xVERSE_384.pth
    scgpt/{best_model.pt,vocab.json,args.json}
    cellnavi/data/{gene_name.txt,Nichenet,node2idx/pretrain}
    scFoundation/models.ckpt
    nicheformer/{nicheformer.ckpt or theislab_Nicheformer/{config.json,model.safetensors}}
    transcriptformer/tf_sapiens/{config.json,model_weights.pt,vocabs/}
    nichenet/{node2idx.json,idx2node.json,graph.pkl,graph.pt}
  scFM_third_party/
    Geneformer/ uce/ state/ stack/ scGPT-main/ scFoundation/ scldm/ xVERSE_code/ CellNavi/ nicheformer/ transcriptformer/
  scFM_output/
    embeddings/ embedding_runs/ metrics/ logs/ figures/ dataset_fitted/ tmp/"""
