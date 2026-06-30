"""TranscriptFormer expression-only adapter.

The official package exposes a CLI that writes cell embeddings to
``obsm["embeddings"]``.  This adapter writes the incoming AnnData to a temporary
h5ad, delegates inference to that CLI, then loads the resulting embeddings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

import paths

GENE_ID_CANDIDATES = ("ensembl_id", "ensemblid", "Ensembl_ID", "ENSEMBL", "gene_id", "feature_id", "gene_ids")
COUNT_LAYER_CANDIDATES = ("counts", "raw_counts", "count")


def _checkpoint_dir() -> Path:
    value = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_CKPT", "").strip()
    model = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_MODEL", "tf_sapiens").strip()
    return Path(value).expanduser().resolve() if value else paths.pretrained_root() / "transcriptformer" / model


def _ensure_ensembl_column(adata: ad.AnnData, gene_col_name: str) -> tuple[ad.AnnData, str]:
    if gene_col_name in adata.var.columns:
        return adata, gene_col_name
    out = adata.copy()
    for candidate in GENE_ID_CANDIDATES:
        if candidate in out.var.columns:
            out.var[gene_col_name] = out.var[candidate].astype(str).values
            return out, candidate
    if all(str(x).startswith(("ENS", "ENSG", "ENSMUSG")) for x in out.var_names[: min(50, out.n_vars)]):
        out.var[gene_col_name] = out.var_names.astype(str)
        return out, "var_names"
    raise ValueError(
        "TranscriptFormer requires Ensembl gene IDs. Add adata.var['ensembl_id'] "
        "or set LATENT_BENCH_TRANSCRIPTFORMER_GENE_COL to an existing var column."
    )


def _counts_layer_name(adata: ad.AnnData) -> str | None:
    configured = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_COUNTS_LAYER", "").strip()
    candidates = (configured,) if configured else COUNT_LAYER_CANDIDATES
    for candidate in candidates:
        if candidate and candidate in adata.layers:
            return candidate
    return None


def _select_count_input(adata: ad.AnnData, input_is_log1p: bool) -> tuple[ad.AnnData, str, str]:
    if not input_is_log1p:
        return adata, "false", "X"
    if adata.raw is not None:
        return adata, "true", "raw.X"
    layer = _counts_layer_name(adata)
    if layer is not None:
        out = adata.copy()
        out.X = adata.layers[layer].copy()
        return out, "false", f"layers[{layer!r}]"
    raise ValueError(
        "TranscriptFormer needs raw counts, but benchmark X is marked log1p and no count source was found. "
        "This adapter will not apply a second log1p and will not silently expm1(log1p X) into pseudo-counts. "
        "Provide adata.raw.X or a raw-count layer such as layers['counts']; only pass --no-input-is-log1p when "
        "X is genuinely count-like."
    )


def _ensure_aux_obs_columns(adata: ad.AnnData, ckpt: Path) -> tuple[ad.AnnData, list[str]]:
    vocab_dir = ckpt / "vocabs"
    if not vocab_dir.is_dir():
        return adata, []
    aux_cols: list[str] = []
    for vocab_path in sorted(vocab_dir.glob("*_vocab.json")):
        aux_col = vocab_path.name.removesuffix("_vocab.json")
        if aux_col:
            aux_cols.append(aux_col)
    missing = [col for col in aux_cols if col not in adata.obs.columns]
    if not missing:
        return adata, []
    out = adata.copy()
    for col in missing:
        default = os.environ.get(f"LATENT_BENCH_TRANSCRIPTFORMER_DEFAULT_{col.upper()}", "unknown")
        # Confirm the configured fallback exists; otherwise use the first key so the official tokenizer
        # gets a legal category rather than raising KeyError before it can map to unknown.
        vocab_path = vocab_dir / f"{col}_vocab.json"
        try:
            vocab = json.load(open(vocab_path))
        except OSError:
            vocab = {}
        if vocab and default not in vocab:
            default = "unknown" if "unknown" in vocab else next(iter(vocab))
        out.obs[col] = default
    return out, missing


def _run_cli(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-20:] + (proc.stderr or "").splitlines()[-40:])
        raise RuntimeError(f"TranscriptFormer inference failed with code {proc.returncode}:\n{tail}")


def encode(
    adata: ad.AnnData,
    *,
    device: str = "cuda",
    batch_size: int = 2,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return TranscriptFormer mean-pooled cell embeddings.

    TranscriptFormer expects raw counts. Benchmark ``adata.X`` is commonly
    already log1p-normalized, so this adapter never applies another log1p and
    never silently ``expm1``-inverts values. If ``input_is_log1p`` is true, it
    uses an explicit count source only: first ``AnnData.raw.X``, then a raw-count
    layer such as ``layers["counts"]``. Otherwise it uses ``adata.X`` directly.
    """
    del force_pert, show_progress
    ckpt = _checkpoint_dir()
    if not (ckpt / "config.json").is_file() or not (ckpt / "model_weights.pt").is_file():
        raise FileNotFoundError(
            f"TranscriptFormer checkpoint missing under {ckpt}. Download with: "
            f"transcriptformer download tf-sapiens --checkpoint-dir {ckpt.parent}"
        )

    gene_col = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_GENE_COL", "ensembl_id").strip()
    count_adata, use_raw, counts_source = _select_count_input(adata, input_is_log1p)
    work_adata, gene_col_source = _ensure_ensembl_column(count_adata, gene_col)
    work_adata, filled_aux_cols = _ensure_aux_obs_columns(work_adata, ckpt)

    precision = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_PRECISION", "16-mixed")
    emb_type = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_EMB_TYPE", "cell")
    clip_counts = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_CLIP_COUNTS", "30")
    num_gpus = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_NUM_GPUS", "1")
    n_workers = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_N_DATA_WORKERS", "0")
    oom_loader = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_OOM_DATALOADER", "1") != "0"
    py = sys.executable

    third_party = paths.third_party_root() / "transcriptformer" / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{third_party}:{env.get('PYTHONPATH', '')}" if third_party.is_dir() else env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory(prefix="scfm_transcriptformer_") as td:
        tmp = Path(td)
        in_h5ad = tmp / "input.h5ad"
        out_dir = tmp / "out"
        out_name = "embeddings.h5ad"
        work_adata.write_h5ad(in_h5ad)
        cmd = [
            py,
            "-c",
            "import transcriptformer.cli as c; c.main()",
            "inference",
            "--checkpoint-path",
            str(ckpt),
            "--data-file",
            str(in_h5ad),
            "--output-path",
            str(out_dir),
            "--output-filename",
            out_name,
            "--batch-size",
            str(batch_size),
            "--gene-col-name",
            gene_col,
            "--precision",
            precision,
            "--use-raw",
            use_raw,
            "--emb-type",
            emb_type,
            "--num-gpus",
            num_gpus,
            "--device",
            "cuda" if device.startswith("cuda") else device,
            "--clip-counts",
            clip_counts,
            "--n-data-workers",
            n_workers,
        ]
        if oom_loader:
            cmd.append("--oom-dataloader")
        _run_cli(cmd, env)
        result = ad.read_h5ad(out_dir / out_name)

    if "embeddings" not in result.obsm:
        raise RuntimeError("TranscriptFormer output missing obsm['embeddings']")
    z = np.asarray(result.obsm["embeddings"], dtype=np.float32)
    if z.ndim != 2:
        raise RuntimeError(f"TranscriptFormer embeddings must be 2D, got shape {z.shape}")
    meta: dict[str, Any] = {
        "encoder_role": "ExpressionOnlyEncoder",
        "model_family": "TranscriptFormer",
        "official_repo": "https://github.com/czi-ai/transcriptformer",
        "checkpoint_path": str(ckpt),
        "checkpoint_model": ckpt.name,
        "pooling": "official cell mean-pooled embedding",
        "embedding_layer_index": "official_cli_default",
        "note": "README documents --embedding-layer-index, but current official argparse does not expose it.",
        "emb_type": emb_type,
        "gene_col_name": gene_col,
        "gene_col_source": gene_col_source,
        "use_raw": use_raw,
        "counts_source": counts_source,
        "filled_aux_obs_columns": filled_aux_cols,
        "precision": precision,
        "batch_size": int(batch_size),
        "num_gpus": int(num_gpus),
        "oom_dataloader": bool(oom_loader),
        "n_data_workers": int(n_workers),
        "input_is_log1p": bool(input_is_log1p),
        "third_party_src": str(third_party),
        "force_pert_effective": False,
        "pert_source": None,
    }
    return z, meta
