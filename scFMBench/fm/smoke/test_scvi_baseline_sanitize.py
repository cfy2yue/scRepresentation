#!/usr/bin/env python3
"""
Smoke (scdfm only for sanitize + routing): sanitized atlas h5ad readable by scldm; _counts_h5ad 17 routes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCDFM_PYTHON = Path(os.environ.get("SCDFM_PYTHON", os.environ.get("SCFM_SCDFM_PYTHON", "python3")))
SCLDM_PYTHON = Path(os.environ.get("SCLDM_PYTHON", os.environ.get("SCFM_SCLDM_PYTHON", "python3")))


def main() -> int:
    fm = Path(__file__).resolve().parents[1]
    if str(fm) not in sys.path:
        sys.path.insert(0, str(fm))
    import paths

    from tools.run_baseline_for_benchmark import (
        _atlas_ts_root,
        _counts_h5ad,
        _discover_source_adatas,
        _sanitize_h5ad_for_scvi,
    )

    scfm = fm.parent
    atlas_skin = (
        Path(
            os.environ.get(
                "SCFM_SKIN_ADATA",
                str(paths.staging_root() / "atlas" / "Skin.h5ad"),
            )
        ).resolve()
    )
    if not atlas_skin.is_file():
        print(f"SKIP: missing {atlas_skin}", file=sys.stderr)
        return 0

    import scanpy as sc

    ad0 = sc.read_h5ad(str(atlas_skin))
    n_obs0 = int(ad0.n_obs)
    n_vars0 = int(ad0.n_vars)

    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "sanitized_skin.h5ad"
        got = _sanitize_h5ad_for_scvi(atlas_skin, out_p, SCDFM_PYTHON)
        assert got.resolve() == out_p.resolve()

        read_py = rf"""
import scanpy as sc
ad = sc.read_h5ad({repr(str(out_p))})
assert ad.n_obs == {n_obs0} and ad.n_vars == {n_vars0}
print('scldm read sanitized OK', ad.n_obs, ad.n_vars)
"""
        cp = subprocess.run([str(SCLDM_PYTHON), "-c", read_py], capture_output=True, text=True)
        if cp.returncode != 0:
            print(cp.stdout, cp.stderr, file=sys.stderr)
            raise SystemExit(cp.returncode)

    persist = paths.output_root() / "tmp" / "sanitize_smoke_cache" / "skin.h5ad"
    persist.parent.mkdir(parents=True, exist_ok=True)
    r1 = _sanitize_h5ad_for_scvi(atlas_skin, persist, SCDFM_PYTHON)
    r2 = _sanitize_h5ad_for_scvi(atlas_skin, persist, SCDFM_PYTHON)
    assert r1 == r2 == persist

    atlas_ts = _atlas_ts_root(scfm)
    sources = _discover_source_adatas(scfm)
    expected = sorted(sources.keys())
    if len(expected) != 17:
        raise SystemExit(f"expected 17 discovered datasets; got {len(expected)}")

    for ds in expected:
        src = sources[ds]
        cnt, reason = _counts_h5ad(ds, src, atlas_ts)
        if ds == "TS_Immune_xtissue":
            assert cnt is None and "no_TS" in reason
        else:
            if cnt is None:
                raise SystemExit(f"{ds}: counts None unexpectedly (reason={reason})")
            if not Path(cnt).is_file():
                raise SystemExit(f"{ds}: counts path not a file: {cnt}")

    print("scvi sanitize smoke OK", n_obs0, n_vars0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
