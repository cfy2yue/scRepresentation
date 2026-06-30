#!/usr/bin/env python3
"""Run one model on one h5ad; write latent.npy, obs.parquet, meta.json."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] [export_embedding_one] {msg}", flush=True)


def _sanitize_obs_for_parquet(obs: pd.DataFrame) -> pd.DataFrame:
    df = obs.copy()
    for c in df.columns:
        s = df[c]
        if hasattr(s, "cat"):
            df[c] = s.astype(str)
        elif s.dtype == object:
            df[c] = s.astype(str)
    return df


def _install_anndata_null_reader() -> None:
    """Allow older anndata envs to read h5ad fields encoded as null.

    Some source h5ads contain metadata like /uns/log1p/base with
    encoding-type=null. anndata>=0.12 reads this, but several model-specific
    environments pin older anndata and fail before we reach model encoding.
    Registering the reader maps those metadata leaves to None without mutating
    the input file.
    """
    try:
        import h5py
        from anndata._io.specs.registry import IOSpec, _REGISTRY
    except Exception:
        return

    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, *, _reader=None):  # type: ignore[unused-ignore]
            return None
    except Exception:
        # Newer anndata may already register this spec, or private APIs may differ.
        return


def _write_obs_table(df: pd.DataFrame, out_obs_parquet: Path) -> str:
    """
    Write obs sidecar. Prefer parquet; fall back to gzip CSV when pyarrow/fastparquet
    are missing in the model venv (e.g. stack uv env).
    Returns relative artifact name for meta.json: 'obs.parquet' or 'obs.csv.gz'.
    """
    df = _sanitize_obs_for_parquet(df)
    try:
        df.to_parquet(out_obs_parquet, index=True)
        return "obs.parquet"
    except (ImportError, OSError, ValueError):
        out_gz = out_obs_parquet.with_name("obs.csv.gz")
        df.to_csv(out_gz, index=True, compression="gzip")
        return "obs.csv.gz"


def _call_encode(
    model: str,
    adata: ad.AnnData,
    *,
    device: str,
    batch_size: int,
    force_pert: bool,
    input_is_log1p: bool,
) -> Tuple[np.ndarray, dict]:
    m = model.lower().strip()
    if m == "scgpt":
        from adapters.scgpt.encoder import encode

        return encode(adata, device=device, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p)
    if m == "xverse":
        from adapters.xverse.encoder import encode

        return encode(adata, device=device, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p)
    if m == "geneformer":
        from adapters.geneformer.encoder import encode

        return encode(adata, device=device, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p)
    if m == "stack":
        from adapters.stack.encoder import encode

        return encode(
            adata,
            device=device,
            batch_size=batch_size,
            num_workers=0,
            force_pert=force_pert,
            input_is_log1p=input_is_log1p,
            show_progress=False,
        )
    if m == "scldm":
        from adapters.scldm.encoder import encode

        return encode(adata, device=device, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p)
    if m == "cellnavi":
        from adapters.cellnavi.encoder import encode

        return encode(adata, device=device, force_pert=force_pert, input_is_log1p=input_is_log1p, show_progress=False)
    if m == "scfoundation":
        from adapters.scfoundation.encoder import encode

        return encode(adata, device=device, force_pert=force_pert, input_is_log1p=input_is_log1p, show_progress=False)
    if m == "uce":
        from adapters.uce.encoder import encode

        return encode(adata, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p, n_collate_workers=0)
    if m == "state":
        from adapters.state.encoder import encode

        return encode(
            adata,
            batch_size=batch_size,
            force_pert=force_pert,
            input_is_log1p=input_is_log1p,
            n_collate_workers=0,
            dataloader_num_workers=0,
        )
    if m == "nicheformer":
        from adapters.nicheformer.encoder import encode

        return encode(adata, device=device, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p)
    if m == "transcriptformer":
        from adapters.transcriptformer.encoder import encode

        return encode(adata, device=device, batch_size=batch_size, force_pert=force_pert, input_is_log1p=input_is_log1p)
    raise ValueError(f"Unknown model: {model}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="scgpt|xverse|geneformer|uce|state|stack|scldm|cellnavi|scfoundation|nicheformer|transcriptformer",
    )
    ap.add_argument("--adata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=8, help="Conservative default for GPU RAM")
    ap.add_argument("--force-pert", action="store_true", default=True)
    ap.add_argument("--no-force-pert", dest="force_pert", action="store_false")
    ap.add_argument("--input-is-log1p", action="store_true", default=True)
    ap.add_argument("--no-input-is-log1p", dest="input_is_log1p", action="store_false")
    ap.add_argument("--max-cells", type=int, default=0, help="0 = all; else first N cells (dry-run)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if latent.npy, obs table, meta.json exist")
    args = ap.parse_args()

    out_latent = args.out_dir / "latent.npy"
    out_obs = args.out_dir / "obs.parquet"
    out_meta = args.out_dir / "meta.json"
    if args.skip_existing and out_latent.is_file() and out_meta.is_file():
        obs_ok = out_obs.is_file() or (args.out_dir / "obs.csv.gz").is_file()
        if obs_ok:
            _log(f"skip-existing {args.out_dir}")
            return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(f"load_adata {args.adata} (max_cells={args.max_cells or 'all'})")
    _install_anndata_null_reader()
    adata = ad.read_h5ad(args.adata)
    if args.max_cells and adata.n_obs > args.max_cells:
        adata = adata[: args.max_cells].copy()

    if adata.X is None:
        raise RuntimeError(
            f"{args.adata}: adata.X is None and no layers — benchmark-only h5ad without expression. "
            "Use an h5ad with materialized X (e.g. atlas raw) or attach expression via "
            "adapters.dataset_fitted_io.attach_expression_from_h5ad before export."
        )

    _log(f"encode start model={args.model} n_obs={adata.n_obs} device={args.device} batch_size={args.batch_size}")
    z, meta = _call_encode(
        args.model,
        adata,
        device=args.device,
        batch_size=args.batch_size,
        force_pert=args.force_pert,
        input_is_log1p=args.input_is_log1p,
    )
    if z.shape[0] != adata.n_obs:
        raise RuntimeError(f"latent rows {z.shape[0]} != adata {adata.n_obs}")
    if np.isnan(z).any() or np.isinf(z).any():
        raise RuntimeError("latent contains NaN/Inf")

    _log(f"encode done latent_shape={getattr(z, 'shape', None)}")
    np.save(out_latent, z.astype(np.float32, copy=False))
    obs_artifact = _write_obs_table(adata.obs, out_obs)
    _log(f"wrote {out_latent.name}, {obs_artifact}, {out_meta.name}")

    run_meta: Dict[str, Any] = {
        **meta,
        "obs_artifact": obs_artifact,
        "export_tool": "export_embedding_one.py",
        "model": args.model.lower(),
        "source_adata": str(args.adata.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "n_obs": int(adata.n_obs),
        "latent_dim": int(z.shape[1]) if z.ndim == 2 else 0,
        "device": args.device,
        "batch_size": args.batch_size,
        "force_pert": args.force_pert,
        "input_is_log1p": args.input_is_log1p,
        "max_cells_applied": int(args.max_cells) if args.max_cells else None,
        "wall_time_s": round(time.time() - t0, 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
    }
    with open(out_meta, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    _log(f"finished wall_s={run_meta['wall_time_s']}")
    del adata, z
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
