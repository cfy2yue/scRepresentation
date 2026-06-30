#!/usr/bin/env python3
"""
Run PCA + scVI fitted baselines for every benchmark dataset_id into output/embeddings/{pca,scvi}/.

Uses ``run_dataset_fitted_baseline.py`` then writes ``obs.csv.gz`` and patches ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FM_ROOT = Path(__file__).resolve().parents[1]
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths

SCFM_ROOT = FM_ROOT.parent
SCVI_PYTHON_DEFAULT = Path(os.environ.get("SCFM_SCLDM_PYTHON", "python3"))
SCDFM_PYTHON_DEFAULT = Path(os.environ.get("SCFM_SCDFM_PYTHON", "python3"))

ATLAS_STEM_TO_TS = {
    "Blood": "TS_Blood_filtered",
    "BoneMarrow": "TS_Bone_Marrow_filtered",
    "Heart": "TS_Heart_filtered",
    "Lung": "TS_Lung_filtered",
    "LymphNode": "TS_Lymph_Node_filtered",
    "Skin": "TS_Skin_filtered",
}


def _setup_log(tag: str, scfm: Path) -> Path:
    log_dir = paths.output_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = log_dir / f"baseline_{tag}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(p), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return p


def _discover_source_adatas(scfm: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    emb_root = paths.output_root() / "embeddings"
    for meta_path in sorted(emb_root.glob("*/*/raw/meta.json")):
        if meta_path.parent.parent.parent.name in ("pca", "scvi"):
            continue
        ds = meta_path.parent.parent.name
        if ds in out:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        src = meta.get("source_adata")
        if src:
            out[ds] = Path(src)
    return out


def _atlas_ts_root(scfm: Path) -> Path:
    cand = paths.data_root() / "raw" / "atlas_TS"
    if cand.is_dir():
        return cand.resolve()
    return cand.resolve()


def _counts_h5ad(dataset_id: str, source_adata: Path, atlas_ts: Path) -> Tuple[Path | None, str]:
    if dataset_id.startswith("TS_") and dataset_id.endswith("_filtered"):
        p = atlas_ts / f"{dataset_id}.h5ad"
        return (p if p.is_file() else None, "atlas_TS_same_stem")
    if dataset_id in ATLAS_STEM_TO_TS:
        stem = ATLAS_STEM_TO_TS[dataset_id]
        p = atlas_ts / f"{stem}.h5ad"
        return (p if p.is_file() else None, "atlas_TS_from_staging_map")
    if dataset_id == "TS_Immune_xtissue":
        # No paired raw counts h5ad; staging X is log1p, will train scVI Gaussian on X.
        return source_adata.resolve(), "atlas_TS_no_counterpart_log1p"
    if "chemicalpert_bench" in str(source_adata):
        return source_adata.resolve(), "chempert_raw"
    if "staging/chempert" in str(source_adata) or "/chempert/" in str(source_adata):
        return source_adata.resolve(), "chempert_staging_same_file"
    if "/staging/genepert/" in str(source_adata):
        # Benchmark h5ad stores log1p-normalized X (no raw counts); Gaussian scVI on X.
        return source_adata.resolve(), "genepert_staging_log1p"
    return source_adata.resolve(), "fallback_source"


def _sanitize_h5ad_for_scvi(src: Path, out: Path, scdfm_python: Path) -> Path:
    """Rewrite h5ad with scdfm (new anndata): drop uns['log1p'] so scldm/old anndata can read it."""
    out = out.resolve()
    src = src.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    code = (
        "import scanpy as sc\n"
        "import pandas as pd\n"
        f"ad = sc.read_h5ad({repr(str(src))})\n"
        "ad.uns.pop('log1p', None)\n"
        "ad.obs.index = pd.Index([str(x) for x in ad.obs_names], dtype=object)\n"
        "ad.var.index = pd.Index([str(x) for x in ad.var_names], dtype=object)\n"
        "for _df in (ad.obs, ad.var):\n"
        "    for _col in list(_df.columns):\n"
        "        _s = _df[_col]\n"
        "        if pd.api.types.is_numeric_dtype(_s) or pd.api.types.is_bool_dtype(_s):\n"
        "            continue\n"
        "        if pd.api.types.is_datetime64_any_dtype(_s):\n"
        "            continue\n"
        "        _vals = ['' if pd.isna(_v) else str(_v) for _v in _s.tolist()]\n"
        "        _df[_col] = pd.Series(_vals, index=_df.index, dtype=object)\n"
        f"ad.write_h5ad({repr(str(out))}, convert_strings_to_categoricals=False)\n"
    )
    cp = subprocess.run(
        [str(scdfm_python), "-c", code],
        cwd=str(FM_ROOT),
    )
    if cp.returncode != 0:
        raise RuntimeError(f"sanitize exited {cp.returncode} for {src} -> {out}")
    if not out.is_file():
        raise RuntimeError(f"sanitize wrote no output: {out}")
    return out


def _smoke_subset_pair(
    source_p: Path,
    counts_p: Path,
    out_src: Path,
    out_counts: Path,
    scdfm_python: Path,
    *,
    max_obs: int = 5000,
    rng_seed: int = 0,
) -> None:
    """Subset aligned cells to at most ``max_obs`` using scdfm (random)."""
    out_src.parent.mkdir(parents=True, exist_ok=True)
    code = """
import numpy as np
import pandas as pd
import scanpy as sc

def _compact_for_h5write(ad):
    ad.obs.index = pd.Index([str(x) for x in ad.obs_names], dtype=object)
    ad.var.index = pd.Index([str(x) for x in ad.var_names], dtype=object)
    ad.uns.pop('log1p', None)
    for _df in (ad.obs, ad.var):
        for _col in list(_df.columns):
            _s = _df[_col]
            if pd.api.types.is_numeric_dtype(_s) or pd.api.types.is_bool_dtype(_s):
                continue
            if pd.api.types.is_datetime64_any_dtype(_s):
                continue
            _vals = ['' if pd.isna(_v) else str(_v) for _v in _s.tolist()]
            _df[_col] = pd.Series(_vals, index=_df.index, dtype=object)

src_path, counts_path = %r, %r
out_s, out_c = %r, %r
mx, seed = %d, %d
s = sc.read_h5ad(src_path)
c = sc.read_h5ad(counts_path)
common = sorted(set(s.obs_names) & set(c.obs_names))
if not common:
    raise SystemExit('smoke subset: zero obs intersection')
s = s[common].copy()
c = c[common].copy()
rng = np.random.default_rng(seed)
n = min(mx, s.n_obs)
if n < s.n_obs:
    ix = np.sort(rng.choice(s.n_obs, size=n, replace=False))
    s = s[ix].copy()
    c = c[ix].copy()
_compact_for_h5write(s)
_compact_for_h5write(c)
s.write_h5ad(out_s, convert_strings_to_categoricals=False)
c.write_h5ad(out_c, convert_strings_to_categoricals=False)
""" % (
        str(Path(source_p).resolve()),
        str(Path(counts_p).resolve()),
        str(Path(out_src).resolve()),
        str(Path(out_counts).resolve()),
        int(max_obs),
        int(rng_seed),
    )
    cp = subprocess.run([str(scdfm_python), "-c", code], cwd=str(FM_ROOT))
    if cp.returncode != 0:
        raise RuntimeError(f"smoke subset exited {cp.returncode}")


def _smoke_subset_single(
    in_p: Path,
    out_p: Path,
    scdfm_python: Path,
    *,
    max_obs: int = 5000,
    rng_seed: int = 0,
) -> None:
    """Randomly subsample one h5ad to at most ``max_obs`` rows (aligned index order preserved)."""
    out_p.parent.mkdir(parents=True, exist_ok=True)
    code = """
import numpy as np
import pandas as pd
import scanpy as sc

def _compact_for_h5write(ad):
    ad.obs.index = pd.Index([str(x) for x in ad.obs_names], dtype=object)
    ad.var.index = pd.Index([str(x) for x in ad.var_names], dtype=object)
    ad.uns.pop('log1p', None)
    for _df in (ad.obs, ad.var):
        for _col in list(_df.columns):
            _s = _df[_col]
            if pd.api.types.is_numeric_dtype(_s) or pd.api.types.is_bool_dtype(_s):
                continue
            if pd.api.types.is_datetime64_any_dtype(_s):
                continue
            _vals = ['' if pd.isna(_v) else str(_v) for _v in _s.tolist()]
            _df[_col] = pd.Series(_vals, index=_df.index, dtype=object)

inp, outp = %r, %r
mx, seed = %d, %d
s = sc.read_h5ad(inp)
rng = np.random.default_rng(seed)
n = min(mx, s.n_obs)
if n < s.n_obs:
    ix = np.sort(rng.choice(s.n_obs, size=n, replace=False))
    s = s[ix].copy()
_compact_for_h5write(s)
s.write_h5ad(outp, convert_strings_to_categoricals=False)
""" % (
        str(Path(in_p).resolve()),
        str(Path(out_p).resolve()),
        int(max_obs),
        int(rng_seed),
    )
    cp = subprocess.run([str(scdfm_python), "-c", code], cwd=str(FM_ROOT))
    if cp.returncode != 0:
        raise RuntimeError(f"smoke subset (single) exited {cp.returncode}")


def _run_cmd(cmd: List[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    r = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(r.returncode)


def _h5ad_has_layer(path: Path, layer: str) -> bool:
    """True if ``layers[layer]`` exists (cheap backed read)."""
    import anndata as ad

    adata = ad.read_h5ad(str(path), backed="r")
    try:
        return layer in adata.layers
    finally:
        adata.file.close()


def _run_scvi_one(
    *,
    dataset_id: str,
    source_adata: Path,
    counts_p: Path,
    counts_reason: str,
    atlas_use_log1p: bool,
    scfm: Path,
    baseline_root: Path,
    scdfm_py: Path,
    scvi_python: Path,
    sc_epochs: int,
    batch_size: int | None,
    smoke: bool,
    dry_run: bool,
    gpu_id: int | None,
) -> Dict[str, Any]:
    """Run scVI baseline for one dataset; return {ok, dataset_id, missing_line?}.

    If gpu_id is not None, sets CUDA_VISIBLE_DEVICES on the spawned process.
    """
    out_scvi = paths.output_root() / "embeddings" / "scvi" / dataset_id / "raw"
    tmp_dir = paths.output_root() / "tmp" / "scvi_sanitized" / dataset_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sanitized_source = tmp_dir / "source.h5ad"
    sanitized_counts = tmp_dir / "counts.h5ad"
    try:
        _sanitize_h5ad_for_scvi(source_adata, sanitized_source, scdfm_py)
        if not atlas_use_log1p:
            _sanitize_h5ad_for_scvi(counts_p, sanitized_counts, scdfm_py)
    except Exception:
        logging.exception("[gpu=%s] scVI sanitize failed for %s", gpu_id, dataset_id)
        return {
            "ok": False,
            "dataset_id": dataset_id,
            "missing_line": (
                f"- **{dataset_id}**: sanitize_failed; source={source_adata}; counts={counts_p}\n"
            ),
        }

    train_src_path = sanitized_source
    train_counts_path = sanitized_counts if not atlas_use_log1p else Path()
    extra_scvi_meta: Dict[str, Any] = {
        "scvi_counts_reason": counts_reason,
        "canonical_source_adata": str(source_adata.resolve()),
        "canonical_counts_h5ad": str(counts_p.resolve()),
        "scvi_sanitized_source": str(sanitized_source.resolve()),
        "scvi_gpu_id": gpu_id,
    }
    if not atlas_use_log1p:
        extra_scvi_meta["scvi_sanitized_counts"] = str(sanitized_counts.resolve())
    if atlas_use_log1p:
        extra_scvi_meta["scvi_atlas_log1p_path"] = True
        if counts_reason.startswith("genepert"):
            extra_scvi_meta["scvi_log1p_reason"] = "genepert staging h5ad X is log1p-normalized"
        else:
            extra_scvi_meta["scvi_log1p_reason"] = "TS_*_filtered X is log1p, no counts available"
    if batch_size is not None:
        extra_scvi_meta["scvi_batch_size"] = int(batch_size)

    if smoke:
        sub_src = tmp_dir / "smoke_subset_source.h5ad"
        try:
            if atlas_use_log1p:
                _smoke_subset_single(sanitized_source, sub_src, scdfm_py, max_obs=5000)
                train_counts_path = Path()
            else:
                sub_counts = tmp_dir / "smoke_subset_counts.h5ad"
                _smoke_subset_pair(
                    sanitized_source,
                    sanitized_counts,
                    sub_src,
                    sub_counts,
                    scdfm_py,
                    max_obs=5000,
                )
                train_counts_path = sub_counts
            train_src_path = sub_src
            extra_scvi_meta["smoke_subset"] = True
            extra_scvi_meta["smoke_max_obs"] = 5000
        except Exception:
            logging.exception("[gpu=%s] scVI smoke subset failed for %s", gpu_id, dataset_id)
            return {
                "ok": False,
                "dataset_id": dataset_id,
                "missing_line": f"- **{dataset_id}**: smoke_subset_failed; counts={counts_p}\n",
            }

    out_scvi.mkdir(parents=True, exist_ok=True)
    py = str(scvi_python) if Path(scvi_python).is_file() else sys.executable
    scldm_proj = Path(scvi_python).resolve().parent.parent
    scvi_cwd = scldm_proj if (scldm_proj / "pyproject.toml").is_file() else scfm
    cmd = [
        py,
        str(baseline_root),
        "--baseline",
        "scvi",
        "--adata",
        str(train_src_path),
        "--scvi-n-latent",
        "128",
        "--scvi-max-epochs",
        str(sc_epochs),
        "--out-dir",
        str(out_scvi),
    ]
    if batch_size is not None:
        cmd.extend(["--scvi-batch-size", str(int(batch_size))])
    if atlas_use_log1p:
        cmd.extend(["--scvi-input-is-log1p", "--scvi-log1p-use-x"])
    else:
        cmd.extend(["--scvi-counts-from-h5ad", str(train_counts_path)])
        try:
            same_file = counts_p.resolve() == source_adata.resolve()
        except Exception:
            same_file = False
        # Genepert (and similar) keep counts in X with no ``layers['counts']``; use X via omitted flag.
        if same_file and _h5ad_has_layer(train_counts_path, "counts"):
            cmd.extend(["--scvi-counts-source-layer", "counts"])
    env = None
    if gpu_id is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info("[gpu=%s] scVI %s: %s", gpu_id, dataset_id, " ".join(cmd))
    if not dry_run and _run_cmd(cmd, cwd=scvi_cwd, env=env) != 0:
        logging.error("[gpu=%s] scVI failed for %s", gpu_id, dataset_id)
        return {
            "ok": False,
            "dataset_id": dataset_id,
            "missing_line": (
                f"- **{dataset_id}**: scvi_run_failed cmd source={train_src_path} counts={train_counts_path}\n"
            ),
        }
    if not dry_run:
        _write_obs_and_meta(
            out_dir=out_scvi,
            source_adata=train_src_path,
            dataset_id=dataset_id,
            baseline="scvi",
            extra_meta=extra_scvi_meta,
        )
    return {"ok": True, "dataset_id": dataset_id, "missing_line": None}


def _scvi_dispatch(
    tasks: List[Dict[str, Any]],
    gpu_ids: List[int],
    max_workers: int,
    *,
    scfm: Path,
    baseline_root: Path,
    scdfm_py: Path,
    scvi_python: Path,
    sc_epochs: int,
    batch_size: int | None,
    smoke: bool,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    """Run scVI tasks in parallel, one per GPU.

    Each worker thread holds one GPU id from a Queue, runs scVI subprocess pinned
    to that GPU via CUDA_VISIBLE_DEVICES, then releases the GPU back.
    """
    if not tasks:
        return []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import queue as _q

    gpu_q: "_q.Queue[int]" = _q.Queue()
    for g in gpu_ids:
        gpu_q.put(g)

    def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
        g = gpu_q.get()
        try:
            return _run_scvi_one(
                dataset_id=task["dataset_id"],
                source_adata=task["source_adata"],
                counts_p=task["counts_p"],
                counts_reason=task["counts_reason"],
                atlas_use_log1p=task["atlas_use_log1p"],
                scfm=scfm,
                baseline_root=baseline_root,
                scdfm_py=scdfm_py,
                scvi_python=scvi_python,
                sc_epochs=sc_epochs,
                batch_size=batch_size,
                smoke=smoke,
                dry_run=dry_run,
                gpu_id=g,
            )
        except Exception as e:  # pragma: no cover - keep the pool alive
            logging.exception("[gpu=%s] worker crashed on %s", g, task.get("dataset_id"))
            return {
                "ok": False,
                "dataset_id": task.get("dataset_id"),
                "missing_line": f"- **{task.get('dataset_id')}**: worker_exception {type(e).__name__}\n",
            }
        finally:
            gpu_q.put(g)

    results: List[Dict[str, Any]] = []
    workers = max(1, min(int(max_workers), len(gpu_ids)))
    logging.info(
        "scVI dispatch: %d tasks across %d GPU(s) %s with batch_size=%s, epochs=%d",
        len(tasks),
        workers,
        gpu_ids,
        batch_size,
        sc_epochs,
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            logging.info(
                "scVI %s -> %s",
                r.get("dataset_id"),
                "ok" if r.get("ok") else "FAILED",
            )
    return results


def _write_obs_and_meta(
    *,
    out_dir: Path,
    source_adata: Path,
    dataset_id: str,
    baseline: str,
    extra_meta: Dict[str, Any],
) -> None:
    import scanpy as sc

    adata = sc.read_h5ad(str(source_adata))
    latent = np.load(out_dir / "latent.npy")
    if latent.shape[0] != adata.n_obs:
        raise SystemExit(
            f"{out_dir}: latent rows {latent.shape[0]} != adata.obs {adata.n_obs} for {source_adata}"
        )
    obs = adata.obs
    obs.to_csv(out_dir / "obs.csv.gz", compression="gzip")
    meta_path = out_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update(extra_meta)
    meta["latent_dim"] = int(latent.shape[1])
    meta["obs_artifact"] = "obs.csv.gz"
    meta["source_adata"] = str(source_adata.resolve())
    meta["model"] = baseline
    meta["dataset_id"] = dataset_id
    meta["fit_scope"] = "all_cells"
    meta["baseline"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=SCFM_ROOT)
    ap.add_argument("--only", choices=("pca", "scvi", "both"), default="both")
    ap.add_argument("--dataset", type=str, default=None, help="Single dataset_id; default: all discovered")
    ap.add_argument(
        "--scvi-python",
        type=Path,
        default=SCVI_PYTHON_DEFAULT,
        help="Python with scvi-tools for scVI baseline",
    )
    ap.add_argument(
        "--scdfm-python",
        type=Path,
        default=SCDFM_PYTHON_DEFAULT,
        help="Python (new scanpy/anndata) used only to sanitize h5ad for scldm scVI",
    )
    ap.add_argument(
        "--scvi-max-epochs",
        type=int,
        default=None,
        help="scVI epochs (default 400; with --smoke default 5 if omitted)",
    )
    ap.add_argument(
        "--scvi-batch-size",
        type=int,
        default=None,
        help="scvi-tools train batch_size (default 128). Set 1024+ to fully use modern GPU.",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids for parallel scVI dispatch, e.g. '0,1,2,3,4,5'. "
        "Each dataset runs on one pinned GPU; PCA still runs serially on CPU.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Quick scVI smoke: subset to ≤5000 cells/w max_epochs default 5 (testing only)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scfm: Path = args.scfm_root.resolve()
    baseline_root = FM_ROOT / "tools" / "run_dataset_fitted_baseline.py"
    atlas_ts = _atlas_ts_root(scfm)
    sources = _discover_source_adatas(scfm)
    if args.dataset:
        if args.dataset not in sources:
            raise SystemExit(f"unknown dataset_id {args.dataset!r}; not found in embeddings metadata")
        sources = {args.dataset: sources[args.dataset]}

    missing_path = paths.output_root() / "benchmark_inventory" / "scvi_missing_counts.md"
    missing_lines: List[str] = ["# scVI baseline skipped (counts / sanitize / failures)\n"]

    tag = args.only
    _setup_log(tag, scfm)

    sc_epochs = args.scvi_max_epochs
    if sc_epochs is None:
        sc_epochs = 5 if args.smoke else 400

    scdfm_py = args.scdfm_python if Path(args.scdfm_python).is_file() else Path(sys.executable)

    for dataset_id in sorted(sources.keys()):
        source_adata = sources[dataset_id]
        counts_p, counts_reason = _counts_h5ad(dataset_id, source_adata, atlas_ts)
        # Atlas TS_*_filtered.h5ad have no counts (X is log1p, layers empty);
        # Genepert staging h5ad is log1p X as well — same Gaussian scVI path.
        atlas_use_log1p = bool(
            counts_reason.startswith("atlas_TS") or counts_reason.startswith("genepert")
        )
        atlas_smoke_log1p = atlas_use_log1p
        out_pca = paths.output_root() / "embeddings" / "pca" / dataset_id / "raw"
        out_scvi = paths.output_root() / "embeddings" / "scvi" / dataset_id / "raw"

        if args.only in ("pca", "both"):
            out_pca.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(baseline_root),
                "--baseline",
                "pca",
                "--adata",
                str(source_adata),
                "--pca-n-components",
                "128",
                "--out-dir",
                str(out_pca),
            ]
            logging.info("PCA %s: %s", dataset_id, " ".join(cmd))
            if not args.dry_run and _run_cmd(cmd, cwd=FM_ROOT) != 0:
                logging.error("PCA failed for %s", dataset_id)
                continue
            if not args.dry_run:
                with open(out_pca / "meta.json") as f:
                    pmeta = json.load(f)
                n_act = int(pmeta.get("n_components_actual", 128))
                _write_obs_and_meta(
                    out_dir=out_pca,
                    source_adata=source_adata,
                    dataset_id=dataset_id,
                    baseline="pca",
                    extra_meta={"n_components_requested": 128, "n_components_actual": n_act},
                )

    if args.only in ("scvi", "both"):
        scvi_tasks: List[Dict[str, Any]] = []
        for dataset_id in sorted(sources.keys()):
            source_adata = sources[dataset_id]
            counts_p, counts_reason = _counts_h5ad(dataset_id, source_adata, atlas_ts)
            atlas_use_log1p_t = bool(
                counts_reason.startswith("atlas_TS") or counts_reason.startswith("genepert")
            )
            if counts_p is None:
                logging.warning("scVI skip %s: %s", dataset_id, counts_reason)
                missing_lines.append(f"- **{dataset_id}**: {counts_reason}; source={source_adata}\n")
                continue
            scvi_tasks.append({
                "dataset_id": dataset_id,
                "source_adata": source_adata,
                "counts_p": counts_p,
                "counts_reason": counts_reason,
                "atlas_use_log1p": atlas_use_log1p_t,
            })

        gpu_ids: List[int] = []
        if args.gpus:
            gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        if not gpu_ids:
            gpu_ids = [0]
        max_workers = min(len(gpu_ids), len(scvi_tasks)) if scvi_tasks else 1

        results = _scvi_dispatch(
            scvi_tasks,
            gpu_ids,
            max_workers,
            scfm=scfm,
            baseline_root=baseline_root,
            scdfm_py=scdfm_py,
            scvi_python=args.scvi_python,
            sc_epochs=sc_epochs,
            batch_size=args.scvi_batch_size,
            smoke=args.smoke,
            dry_run=args.dry_run,
        )
        for r in results:
            if r.get("missing_line"):
                missing_lines.append(r["missing_line"])

    if missing_lines and len(missing_lines) > 1:
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_path, "w") as f:
            f.writelines(missing_lines)
        logging.info("Wrote %s", missing_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
