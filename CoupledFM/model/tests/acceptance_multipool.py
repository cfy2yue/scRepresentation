"""Independent acceptance for the multi-pool (mean/sum/max/min + per-pool scale) extension
of UnifiedConditionEncoder.

Run after implementation:
    cd /path/to/CoupledFM && python -m model.tests.acceptance_multipool

Exits with non-zero on any failure; prints a final 'ACCEPT MULTIPOOL OK' on success.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Ensure local imports take precedence (mirror smoke_test approach)
ROOT = Path(__file__).resolve().parents[2]
for extra in (str(ROOT),):
    if extra in sys.path:
        sys.path.remove(extra)
    sys.path.insert(0, extra)

import numpy as np
import torch

from model.condition_emb.genepert.perturbation_encoder import UnifiedConditionEncoder


_FAIL = 0


def expect(cond: bool, msg: str) -> None:
    global _FAIL
    if cond:
        print(f"  ok  {msg}")
    else:
        print(f"  FAIL {msg}")
        _FAIL += 1


def _new_encoder(pool_aggregations=None, pool_scale_init=None, *, seed: int = 0):
    torch.manual_seed(seed)
    kw = {}
    if pool_aggregations is not None:
        kw["pool_aggregations"] = pool_aggregations
    if pool_scale_init is not None:
        kw["pool_scale_init"] = pool_scale_init
    return UnifiedConditionEncoder(
        mode="random_learned",
        out_dim=16,
        num_embeddings_random=64,
        embed_dim_random=8,
        chem_emb_dim=None,
        pert_type_scale_init=(0.0, -1.0, -1.0, -1.0, 1.0, 1.0),
        **kw,
    )


def section(label: str) -> None:
    print(f"\n=== {label} ===")


def main() -> None:
    section("1. BC: default ('mean',)/(1.0,) ~ pre-multipool behaviour")
    enc_default = _new_encoder()
    enc_explicit = _new_encoder(("mean",), (1.0,))
    B, K = 3, 4
    gene_ids = torch.randint(low=1, high=64, size=(B, K))
    pert_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.long)
    nperts = pert_mask.sum(dim=1)
    pert_type = torch.tensor([1, 4, 0], dtype=torch.long)

    out_default = enc_default(
        pert_gene_ids=gene_ids, pert_mask=pert_mask, pert_type_id=pert_type, nperts=nperts
    )
    out_explicit = enc_explicit(
        pert_gene_ids=gene_ids, pert_mask=pert_mask, pert_type_id=pert_type, nperts=nperts
    )
    expect(out_default.shape == (B, 16), f"default out shape {tuple(out_default.shape)}==(3,16)")
    # different seeds for the two encoders may diverge; we re-init both with seed=0 → equal weights
    diff = (out_default - out_explicit).abs().max().item()
    expect(diff < 1e-6, f"default vs explicit ('mean',)/(1.0,) max-abs diff={diff:.2e} < 1e-6")

    section("2. Single-pert: mean=sum=max=min when K_active=1")
    enc3 = _new_encoder(("mean", "sum", "max", "min"), (1.0, 0.5, 0.5, 0.5))
    # row with single active slot
    gene_ids2 = torch.zeros(1, K, dtype=torch.long)
    gene_ids2[0, 0] = 5
    mask2 = torch.zeros(1, K, dtype=torch.long)
    mask2[0, 0] = 1
    nperts2 = torch.tensor([1], dtype=torch.long)
    type2 = torch.tensor([1], dtype=torch.long)  # KO
    out3 = enc3(pert_gene_ids=gene_ids2, pert_mask=mask2, pert_type_id=type2, nperts=nperts2)
    expect(out3.shape == (1, 16), f"mean+max+min single-pert out shape {tuple(out3.shape)}==(1,16)")
    # Compare against same encoder run with only ('mean',) and pool_scale_init=(1+0.5+0.5+0.5,) = (2.5,)
    enc_eq = _new_encoder(("mean",), (2.5,))
    # IMPORTANT: enc3 and enc_eq differ in seeded init for pool_scale only.  We force-load
    # gene_table + projector + chem_proj + type_scale + layer_norm states from enc3 into enc_eq
    # so the only difference is pool aggregation scheme.
    sd_src = enc3.state_dict()
    sd_dst = enc_eq.state_dict()
    for k, v in sd_src.items():
        if k in sd_dst and sd_dst[k].shape == v.shape:
            sd_dst[k] = v.clone()
    enc_eq.load_state_dict(sd_dst, strict=False)
    out_eq = enc_eq(pert_gene_ids=gene_ids2, pert_mask=mask2, pert_type_id=type2, nperts=nperts2)
    diff2 = (out3 - out_eq).abs().max().item()
    expect(
        diff2 < 1e-5,
        f"single-pert: 4-pool sum (1+0.5+0.5+0.5) ~ 1-pool*(2.5); max-abs diff={diff2:.2e} < 1e-5",
    )

    section("3. Multi-pert: mean / sum / max / min produce different signals")
    enc3b = _new_encoder(("mean", "sum", "max", "min"), (1.0, 1.0, 1.0, 1.0))
    # Disable type_scale modulation: pick CRISPRA (id=4 -> +1) and let scale=+1
    # Set distinct gene rows whose post-projection values are deliberately asymmetric.
    gid = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mk = torch.ones(1, 4, dtype=torch.long)
    nperts3 = torch.tensor([4], dtype=torch.long)
    tid = torch.tensor([4], dtype=torch.long)

    # Probe internals: derive the three pooled gene outputs directly to check mean/max/min differ.
    # We patch each encoder to use a single op for the run.
    def _pool_only(op):
        e = _new_encoder((op,), (1.0,))
        # copy weights from enc3b so projections are identical across runs
        sd_src = enc3b.state_dict()
        sd_dst = e.state_dict()
        for k, v in sd_src.items():
            if k in sd_dst and sd_dst[k].shape == v.shape:
                sd_dst[k] = v.clone()
        e.load_state_dict(sd_dst, strict=False)
        with torch.no_grad():
            out = e(pert_gene_ids=gid, pert_mask=mk, pert_type_id=tid, nperts=nperts3)
        return out

    o_mean = _pool_only("mean")
    o_sum = _pool_only("sum")
    o_max = _pool_only("max")
    o_min = _pool_only("min")
    expect(
        not torch.allclose(o_mean, o_sum, atol=1e-5),
        "multi-pert: mean != sum",
    )
    expect(
        not torch.allclose(o_mean, o_max, atol=1e-5),
        "multi-pert: mean ≠ max",
    )
    expect(
        not torch.allclose(o_mean, o_min, atol=1e-5),
        "multi-pert: mean ≠ min",
    )
    expect(
        not torch.allclose(o_max, o_min, atol=1e-5),
        "multi-pert: max ≠ min",
    )

    section("4. All-zero mask row: no NaN/Inf and zero output (zero-row clamp)")
    enc_safe = _new_encoder(("mean", "sum", "max", "min"), (1.0, 0.5, 0.5, 0.5))
    gid0 = torch.zeros(1, 4, dtype=torch.long)
    mk0 = torch.zeros(1, 4, dtype=torch.long)
    n0 = torch.zeros(1, dtype=torch.long)
    t0 = torch.zeros(1, dtype=torch.long)
    o0 = enc_safe(pert_gene_ids=gid0, pert_mask=mk0, pert_type_id=t0, nperts=n0)
    expect(torch.isfinite(o0).all().item(), "all-zero-mask row -> finite output (no NaN/Inf)")
    expect(o0.abs().max().item() < 1e-6, f"all-zero-mask row -> exact zero (max-abs={o0.abs().max().item():.2e})")

    section("5. state_dict contains pool_scale of correct length")
    sd = enc3.state_dict()
    expect("pool_scale" in sd, "'pool_scale' in state_dict")
    if "pool_scale" in sd:
        expect(
            sd["pool_scale"].shape == (4,),
            f"pool_scale shape {tuple(sd['pool_scale'].shape)}==(4,)",
        )

    section("6. Old ckpt with ('mean',) loads into ('mean','sum','max','min') via strict=False")
    enc_old = _new_encoder(("mean",), (1.0,))
    # 旧时代 ckpt 无 pool_scale；若保留 shape (1,) 的 pool_scale 会与 (4,) 触发 size mismatch
    old_sd = {k: v for k, v in enc_old.state_dict().items() if k != "pool_scale"}
    enc_new = _new_encoder(("mean", "sum", "max", "min"), (1.0, 0.5, 0.5, 0.5))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc_new.load_state_dict(old_sd, strict=False)
        loaded_ok = True
    except RuntimeError as e:
        print(f"  load failed: {e}")
        loaded_ok = False
    expect(loaded_ok, "load old ('mean',) ckpt into multi-pool encoder via strict=False")
    expect(enc_new.pool_scale.numel() == 4, "new encoder pool_scale length stays 4 after load")

    section("7. ValueError when len(pool_aggregations) != len(pool_scale_init)")
    raised = False
    try:
        _new_encoder(("mean", "max"), (1.0,))
    except ValueError:
        raised = True
    expect(raised, "len mismatch -> ValueError")

    section("8. Unknown pool op rejected")
    raised2 = False
    try:
        _new_encoder(("median",), (1.0,))
    except ValueError:
        raised2 = True
    expect(raised2, "unknown pool op 'median' -> ValueError")

    if _FAIL:
        print(f"\n=== {_FAIL} FAILURES ===")
        sys.exit(1)
    print("\n=========== ACCEPT MULTIPOOL OK ===========")


if __name__ == "__main__":
    main()
