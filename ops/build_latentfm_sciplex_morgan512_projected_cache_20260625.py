#!/usr/bin/env python3
"""Build a 512-d projected SciPlex Morgan cache for LatentFM compatibility.

The current xverse anchor checkpoint has a 512-dimensional chemical projector.
The raw SciPlex Morgan cache is 2048-dimensional, so direct use would create a
different architecture and make anchor/candidate comparison unsafe. This script
creates a deterministic random-projection cache that keeps the existing drug
keys and metadata while matching the anchor's chemical input dimension.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SRC = ROOT / "dataset/drug_cache/sciplex_smiles_morgan2048_20260624"
OUT = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625"
REPORT_JSON = ROOT / "reports/latentfm_sciplex_morgan512_projected_cache_20260625.json"
REPORT_MD = ROOT / "reports/LATENTFM_SCIPLEX_MORGAN512_PROJECTED_CACHE_20260625.md"
SEED = 20260625
OUT_DIM = 512


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src_emb = SRC / "drug_embeddings.npy"
    src_index_json = SRC / "drug_index.json"
    src_index_tsv = SRC / "drug_index.tsv"
    src_meta = SRC / "drug_metadata.tsv"
    for p in [src_emb, src_index_json, src_index_tsv, src_meta, SRC / "manifest.json"]:
        if not p.exists():
            raise SystemExit(f"missing source artifact: {p}")
    emb = np.load(src_emb).astype(np.float32)
    if emb.ndim != 2 or emb.shape[1] != 2048:
        raise SystemExit(f"expected source shape (*, 2048), got {emb.shape}")
    rng = np.random.default_rng(SEED)
    proj = rng.standard_normal((emb.shape[1], OUT_DIM), dtype=np.float32) / np.sqrt(float(OUT_DIM))
    out = emb @ proj
    out[0, :] = 0.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    safe = norms[:, 0] > 0
    out[safe] = out[safe] / norms[safe]
    out = out.astype(np.float32, copy=False)
    np.save(OUT / "drug_embeddings.npy", out)
    shutil.copy2(src_index_json, OUT / "drug_index.json")
    shutil.copy2(src_index_tsv, OUT / "drug_index.tsv")
    shutil.copy2(src_meta, OUT / "drug_metadata.tsv")
    manifest = {
        "source": "sciplex_smiles_morgan512_projected_20260625",
        "source_cache": str(SRC),
        "source_embeddings_sha256": sha256_file(src_emb),
        "projection": "deterministic_gaussian_random_projection_row_l2_normalized",
        "projection_seed": SEED,
        "source_dim": int(emb.shape[1]),
        "embed_dim": OUT_DIM,
        "num_rows": int(out.shape[0]),
        "pad_index": 0,
        "unk_index": 1,
        "notes": [
            "Projection exists to match the 512-d chemical branch of xverse_8k_anchor.",
            "Use this cache for compatible LatentFM all-modality smokes; keep the 2048-d cache as provenance.",
            "This cache does not by itself authorize GPU training.",
        ],
        "artifact_files": {
            "embeddings": str(OUT / "drug_embeddings.npy"),
            "index_json": str(OUT / "drug_index.json"),
            "index_tsv": str(OUT / "drug_index.tsv"),
            "metadata_tsv": str(OUT / "drug_metadata.tsv"),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "sciplex_morgan512_projected_cache_ready_no_gpu",
        "cache_dir": str(OUT),
        "manifest": str(OUT / "manifest.json"),
        "source_cache": str(SRC),
        "source_dim": int(emb.shape[1]),
        "embed_dim": OUT_DIM,
        "num_rows": int(out.shape[0]),
        "projection_seed": SEED,
        "gpu_authorized": False,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# LatentFM SciPlex Morgan512 Projected Cache",
                "",
                f"Status: `{payload['status']}`",
                "",
                "## Boundary",
                "",
                "- CPU-only descriptor compatibility artifact.",
                "- Projects the 2048-d Morgan cache to 512 dimensions for anchor-compatible LatentFM chemical conditioning.",
                "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
                "",
                "## Artifact",
                "",
                f"- cache dir: `{OUT}`",
                f"- source cache: `{SRC}`",
                f"- source dim: `{emb.shape[1]}`",
                f"- projected dim: `{OUT_DIM}`",
                f"- projection seed: `{SEED}`",
                "",
                "## Decision",
                "",
                "- GPU authorized: `False`",
                "- next action: use only after all-modality artifact/metadata/dryload/design gates pass.",
                "",
                "## JSON",
                "",
                f"`{REPORT_JSON}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
