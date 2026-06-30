"""Resolve control / GT AnnData paths under ``biFlow_data``.

Preferred layout:

* ``state`` / ``uce``: ``{biflow}/control_{backbone}/{ds}.h5ad`` +
  ``{biflow}/gt_{backbone}/{ds}.h5ad``
* ``stack`` / ``scldm`` / ``scfoundation`` / ``xverse``: preferred canonical root is
  ``data/latent_data/{backbone}`` with ``control_{backbone}/{ds}.h5ad`` +
  ``gt_{backbone}/{ds}.h5ad``. For compatibility, callers may still pass
  ``data/biFlow_data``; the resolver checks sibling canonical latent roots first.

Legacy fallbacks::

    ``control_center/{ds}.h5ad`` + ``gt/{ds}.h5ad``
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

_VALID_BACKBONES = frozenset({"state", "uce", "stack", "scldm", "scfoundation", "xverse"})


def _canonical_latent_roots(root: Path, backbone: str) -> list[Path]:
    """Candidate roots for canonical latent embeddings, preferred first."""
    roots: list[Path] = []

    # If callers pass the historical ``.../data/biFlow_data`` root, prefer
    # canonical model-specific latent locations under ``.../data/latent_data``.
    if root.name == "biFlow_data":
        roots.append(root.parent / "latent_data" / backbone)

    roots.append(root)

    # If callers pass ``.../data/latent_data`` instead of ``.../{backbone}``.
    if root.name == "latent_data":
        roots.insert(0, root / backbone)

    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        rr = r.expanduser()
        if rr not in seen:
            seen.add(rr)
            out.append(rr)
    return out


def normalize_latent_backbone(latent_backbone: str) -> str:
    """Return normalized backbone name; raises if invalid."""
    b = str(latent_backbone or "state").lower().strip()
    if b not in _VALID_BACKBONES:
        raise ValueError(
            f"latent_backbone must be one of {sorted(_VALID_BACKBONES)}, got {latent_backbone!r}",
        )
    return b


def resolve_biflow_control_gt_h5ad(
    biflow_dir: str | Path,
    ds_name: str,
    *,
    latent_backbone: str = "state",
) -> Optional[Tuple[Path, Path]]:
    """Return ``(control_h5ad, gt_h5ad)`` if both exist; else ``None``."""
    root = Path(biflow_dir).expanduser()
    bb = normalize_latent_backbone(latent_backbone)
    fn = f"{ds_name}.h5ad"

    if bb in {"stack", "scldm", "scfoundation", "xverse"}:
        for sr in _canonical_latent_roots(root, bb):
            pairs = [
                (sr / f"control_{bb}" / fn, sr / f"gt_{bb}" / fn),
            ]
            if bb == "stack":
                pairs.insert(0, (sr / "control_center_stack" / fn, sr / "gt_stack" / fn))
            for cc, gt in pairs:
                if cc.is_file() and gt.is_file():
                    return (cc.resolve(), gt.resolve())
    else:
        new_cc = root / f"control_{bb}" / fn
        new_gt = root / f"gt_{bb}" / fn
        if new_cc.is_file() and new_gt.is_file():
            return (new_cc.resolve(), new_gt.resolve())

    leg_cc = root / "control_center" / fn
    leg_gt = root / "gt" / fn
    if leg_cc.is_file() and leg_gt.is_file():
        return (leg_cc.resolve(), leg_gt.resolve())

    return None


def iter_biflow_dataset_stems(
    biflow_dir: str | Path,
    *,
    latent_backbone: str = "state",
) -> List[str]:
    """Dataset stems found under preferred control dirs and/or legacy ``control_center/``."""
    root = Path(biflow_dir).expanduser()
    bb = normalize_latent_backbone(latent_backbone)
    stems: set[str] = set()

    if bb in {"stack", "scldm", "scfoundation", "xverse"}:
        for sr in _canonical_latent_roots(root, bb):
            subs = [f"control_{bb}", f"gt_{bb}"]
            if bb == "stack":
                subs.insert(1, "control_center_stack")
            for sub in subs:
                d = sr / sub
                if d.is_dir():
                    stems.update(p.stem for p in d.glob("*.h5ad"))
    else:
        d_new = root / f"control_{bb}"
        if d_new.is_dir():
            stems.update(p.stem for p in d_new.glob("*.h5ad"))

    d_legacy = root / "control_center"
    if d_legacy.is_dir():
        stems.update(p.stem for p in d_legacy.glob("*.h5ad"))

    return sorted(stems)


def resolve_gt_h5ad_for_pert_metadata(
    biflow_dir: str | Path,
    ds_name: str,
    *,
    latent_backbone: str = "state",
) -> Optional[Path]:
    """GT h5ad path for obs / perturbation metadata (same resolution as pair loading)."""
    pair = resolve_biflow_control_gt_h5ad(
        biflow_dir, ds_name, latent_backbone=latent_backbone,
    )
    if pair is None:
        return None
    return pair[1]


__all__ = [
    "normalize_latent_backbone",
    "resolve_biflow_control_gt_h5ad",
    "resolve_gt_h5ad_for_pert_metadata",
    "iter_biflow_dataset_stems",
]
