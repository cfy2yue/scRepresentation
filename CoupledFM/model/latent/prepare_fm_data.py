#!/usr/bin/env python3
"""
Prepare latent FM HDF5 from biFlow_data (ctrl pool as flow source; no legacy ir/ group in new files).

For each GT condition, samples matching counts from the **control** embedding pool
(random with replacement, seeded per dataset). Writes:

  ctrl/emb, ctrl/offsets  (same layout as gt; total_ctrl == total_gt)
  gt/emb, gt/offsets
  conditions

Legacy HDF5 with ir/emb is still readable in dataset.py via fallback.
"""

import os
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()

for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[var] = '1'

import gc
import json
import time
import argparse
import numpy as np
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

from model.latent.perturb_helpers import condition_metadata_from_cond_string

_DEFAULT_REPO = _THIS_FILE.parents[2]
ROOT = Path(
    os.environ.get("COUPLEDFM_ROOT", str(_DEFAULT_REPO))
).expanduser().resolve()
CTRL_DIR = Path(os.environ.get('COUPLEDFM_BIFLOW_CTRL', str(ROOT / 'data' / 'biFlow_data' / 'control')))
GT_DIR = Path(os.environ.get('COUPLEDFM_BIFLOW_GT', str(ROOT / 'data' / 'biFlow_data' / 'gt')))
OUT_DIR = Path(os.environ.get('COUPLEDFM_FM_DATA', str(ROOT / 'model' / 'latent' / 'fm_data')))

CHUNK_ROWS = 256
GZIP_LEVEL = 1
IO_CHUNK = 20000
PROGRESS_EVERY = 500


def ts():
    return time.strftime('%H:%M:%S')


def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1


def read_condition_labels(h5ad_path: Path) -> np.ndarray:
    with h5py.File(str(h5ad_path), 'r') as f:
        grp = f['obs/perturbation']
        if isinstance(grp, h5py.Group) and 'codes' in grp:
            codes = grp['codes'][:]
            cats = grp['categories'].asstr()[:]
            return cats[codes]
        return grp.asstr()[:]


def _emb_dataset(gt_f):
    g = gt_f['obsm']
    if 'emb' in g:
        return g['emb']
    if 'exp_emb1' in g:
        return g['exp_emb1']
    raise KeyError('obsm must contain emb or exp_emb1')


def _ctrl_pool(ctrl_f):
    g = ctrl_f['obsm']
    if 'emb' in g:
        return np.asarray(g['emb'][:], dtype=np.float32)
    if 'exp_emb' in g:
        return np.asarray(g['exp_emb'][:], dtype=np.float32)
    raise KeyError('control obsm must contain emb or exp_emb')


def read_existing_meta(out_path: Path) -> dict:
    with h5py.File(str(out_path), 'r') as f:
        conds = f['conditions'].asstr()[:].tolist()
        if 'ctrl/offsets' in f:
            c_off = f['ctrl/offsets'][:]
            key_ctrl = 'ctrl'
        else:
            c_off = f['ir/offsets'][:]
            key_ctrl = 'ir'
        gt_offsets = f['gt/offsets'][:]
        D = f[f'{key_ctrl}/emb'].shape[1]

    n_conds = len(conds)
    n_c = int(c_off[-1])
    n_gt = int(gt_offsets[-1])
    return {
        'n_conds': n_conds,
        'n_src': n_c,
        'n_gt': n_gt,
        'src_per_cond': int(c_off[1] - c_off[0]) if n_conds > 0 else 0,
        'emb_dim': D,
        'conditions': conds,
    }


def _write_gt_sequential(gt_f, out_ds, gt_labels, common, gt_offsets, total_gt, D):
    gt_src = _emb_dataset(gt_f)
    n_total = gt_src.shape[0]
    cond_map = {c: i for i, c in enumerate(common)}
    cond_idx = np.full(n_total, -1, dtype=np.int32)
    for i, lbl in enumerate(gt_labels):
        ci = cond_map.get(lbl, -1)
        if ci >= 0:
            cond_idx[i] = ci

    valid_src = np.flatnonzero(cond_idx >= 0).astype(np.int64)
    order = np.argsort(cond_idx[valid_src], kind="stable")
    ordered_src = valid_src[order]
    if len(ordered_src) != total_gt:
        raise RuntimeError(f"ordered GT rows {len(ordered_src)} != expected {total_gt}")

    for start in range(0, total_gt, IO_CHUNK):
        end = min(start + IO_CHUNK, total_gt)
        idx = ordered_src[start:end]
        read_order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[read_order]
        inv = np.empty_like(read_order)
        inv[read_order] = np.arange(len(read_order))
        block = np.asarray(gt_src[sorted_idx], dtype=np.float32)[inv]
        out_ds[start:end] = block


def process_dataset(ds_name: str, force: bool = False, seed: int = 42):
    ctrl_path = CTRL_DIR / f'{ds_name}.h5ad'
    gt_path = GT_DIR / f'{ds_name}.h5ad'
    out_path = OUT_DIR / f'{ds_name}.h5'

    if not ctrl_path.exists():
        return None, f'control not found: {ctrl_path}'
    if not gt_path.exists():
        return None, f'GT not found: {gt_path}'

    if out_path.exists() and not force:
        try:
            info = read_existing_meta(out_path)
            return info, 'already_done'
        except Exception:
            pass

    t0 = time.time()
    gt_labels = read_condition_labels(gt_path)
    common = sorted([c for c in np.unique(gt_labels) if c != 'control'])
    n_conds = len(common)
    if n_conds == 0:
        return None, 'no conditions'

    gt_cond_counts = {c: int(np.sum(gt_labels == c)) for c in common}

    gt_offsets = np.zeros(n_conds + 1, dtype=np.int64)
    ctrl_offsets = np.zeros(n_conds + 1, dtype=np.int64)
    for i, c in enumerate(common):
        n = gt_cond_counts[c]
        gt_offsets[i + 1] = gt_offsets[i] + n
        ctrl_offsets[i + 1] = ctrl_offsets[i] + n

    total_gt = int(gt_offsets[-1])
    total_ctrl = int(ctrl_offsets[-1])
    assert total_ctrl == total_gt

    tmp_path = out_path.with_suffix('.h5.tmp')

    try:
        with h5py.File(str(ctrl_path), 'r') as cf, \
             h5py.File(str(gt_path), 'r') as gf, \
             h5py.File(str(tmp_path), 'w') as out_f:

            pool = _ctrl_pool(cf)
            gt_src = _emb_dataset(gf)
            D = int(gt_src.shape[1])
            if pool.shape[1] != D:
                return None, f'emb dim mismatch ctrl={pool.shape[1]} gt={D}'

            rng = np.random.RandomState(seed + hash(ds_name) % (2**31))

            ctrl_ds = out_f.create_dataset(
                'ctrl/emb', shape=(total_ctrl, D), dtype='float32',
                chunks=(min(CHUNK_ROWS, max(1, total_ctrl)), D),
                compression='gzip', compression_opts=GZIP_LEVEL,
            )
            pos = 0
            for i, c in enumerate(common):
                n = gt_cond_counts[c]
                idx = rng.choice(pool.shape[0], size=n, replace=True)
                ctrl_ds[pos:pos + n] = pool[idx]
                pos += n
                if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == n_conds:
                    print(f'    [{ts()}] ctrl {i+1}/{n_conds} pos={pos:,}', flush=True)

            gt_ds = out_f.create_dataset(
                'gt/emb', shape=(total_gt, D), dtype='float32',
                chunks=(min(CHUNK_ROWS, max(1, total_gt)), D),
                compression='gzip', compression_opts=GZIP_LEVEL,
            )
            print(f'    [{ts()}] Writing GT ...', flush=True)
            _write_gt_sequential(gf, gt_ds, gt_labels, common, gt_offsets, total_gt, D)

            out_f.create_dataset('ctrl/offsets', data=ctrl_offsets)
            out_f.create_dataset('gt/offsets', data=gt_offsets)
            dt = h5py.string_dtype()
            out_f.create_dataset('conditions', data=np.array(common, dtype=object), dtype=dt)

        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    gc.collect()
    elapsed = time.time() - t0
    mb = os.path.getsize(str(out_path)) / 1e6
    info = {
        'n_conds': n_conds,
        'n_src': total_ctrl,
        'n_gt': total_gt,
        'src_per_cond': int(ctrl_offsets[1] - ctrl_offsets[0]) if n_conds else 0,
        'emb_dim': D,
        'conditions': common,
    }
    print(f'    {n_conds} conds  ctrl={total_ctrl:,}  GT={total_gt:,}  {mb:.0f}MB  {elapsed:.1f}s', flush=True)
    return info, 'ok'


def _worker(args_tuple):
    ds_name, force = args_tuple
    try:
        info, status = process_dataset(ds_name, force=force)
        return ds_name, info, status, None
    except Exception:
        import traceback
        return ds_name, None, 'error', traceback.format_exc()


def _dataset_mean(h5_path: Path, key: str, chunk_rows: int = IO_CHUNK) -> np.ndarray:
    with h5py.File(str(h5_path), 'r') as f:
        if key in f:
            ds = f[key]
        elif key == 'ctrl/emb' and 'ir/emb' in f:
            ds = f['ir/emb']
        else:
            raise KeyError(f'{key} not found in {h5_path}')
        n = int(ds.shape[0])
        if n == 0:
            return np.zeros((int(ds.shape[1]),), dtype=np.float32)
        acc = np.zeros((int(ds.shape[1]),), dtype=np.float64)
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            acc += np.asarray(ds[start:end], dtype=np.float32).sum(axis=0, dtype=np.float64)
    return (acc / max(1, n)).astype(np.float32)


def _looks_like_drug_dataset(ds_name: str) -> bool:
    d = str(ds_name).lower()
    return any(tok in d for tok in ("sciplex", "chempert", "chemical", "drug"))


def _condition_metadata_for_export(ds_name: str, cond: str) -> dict:
    meta = condition_metadata_from_cond_string(str(cond))
    is_drug = _looks_like_drug_dataset(ds_name)
    chem_value = str(cond) if is_drug else None
    return {
        "perturbation_type_raw": "drug" if is_drug else meta.perturbation_type_raw,
        "genes": list(meta.genes),
        "chem_obs_value": chem_value,
        "chem_source": f"drug={chem_value}" if chem_value else None,
        "condition_col": "perturbation",
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare latent FM data (no IR)')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--n-workers', type=int, default=1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted([f.stem for f in GT_DIR.glob('*.h5ad')])

    print('=' * 70)
    print(f'  Prepare FM Data (ctrl pool → GT conditions)')
    print(f'  CTRL: {CTRL_DIR}')
    print(f'  GT:   {GT_DIR}')
    print(f'  OUT:  {OUT_DIR}')
    print(f'  N:    {len(datasets)}')
    print('=' * 70, flush=True)

    manifest = {'emb_dim': None, 'datasets': {}, 'total_conditions': 0, 'total_src_cells': 0, 'total_gt_cells': 0}
    condition_metadata = {}
    n_ok = n_skip = n_fail = 0
    tasks = [(ds, args.force) for ds in datasets]

    if args.n_workers <= 1:
        results = []
        for t in tasks:
            ds_name, info, status, err = _worker(t)
            results.append((ds_name, info, status, err))
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            results = list(ex.map(_worker, tasks))

    for ds_name, info, status, err in results:
        if status == 'already_done' and info:
            n_skip += 1
        elif status == 'ok' and info:
            n_ok += 1
        elif status == 'error':
            print(f'[{ds_name}] {err}', flush=True)
            n_fail += 1
            continue
        else:
            print(f'[{ds_name}] skip: {status}', flush=True)
            n_skip += 1
            continue
        if info:
            manifest['emb_dim'] = info['emb_dim']
            manifest['datasets'][ds_name] = {
                'n_conds': info['n_conds'],
                'n_src': info['n_src'],
                'n_gt': info['n_gt'],
                'src_per_cond': info['src_per_cond'],
                'conditions': info['conditions'],
            }
            condition_metadata[ds_name] = {
                str(cond): _condition_metadata_for_export(ds_name, str(cond))
                for cond in info['conditions']
            }
            manifest['total_conditions'] += info['n_conds']
            manifest['total_src_cells'] += info['n_src']
            manifest['total_gt_cells'] += info['n_gt']

    ctrl_means = {}
    pert_means = {}
    for ds_name in sorted(manifest['datasets'].keys()):
        h5_path = OUT_DIR / f'{ds_name}.h5'
        if not h5_path.exists():
            continue
        ctrl_means[ds_name] = _dataset_mean(h5_path, 'ctrl/emb')
        pert_means[ds_name] = _dataset_mean(h5_path, 'gt/emb')

    if ctrl_means:
        np.savez_compressed(OUT_DIR / 'ctrl_means.npz', **ctrl_means)
    if pert_means:
        np.savez_compressed(OUT_DIR / 'pert_means.npz', **pert_means)

    metadata_path = OUT_DIR / 'condition_metadata.json'
    metadata_path.write_text(
        json.dumps(condition_metadata, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    manifest['condition_metadata_file'] = str(metadata_path)

    mp = OUT_DIR / 'manifest.json'
    with open(mp, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(
        f'Done ok={n_ok} skip={n_skip} fail={n_fail}  '
        f'means={len(pert_means)}  manifest={mp}'
    )


if __name__ == '__main__':
    main()
