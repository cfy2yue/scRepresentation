#!/usr/bin/env python3
"""CPU gate for budget-aware train-time MMD before RawFM MMD smokes."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
OUT_DIR = ROOT / "reports/rawfm_mask_aware_mmd_gate_20260628"

sys.path.insert(0, str(COUPLEDFM))

from model.mmd_utils import median_sigmas, mmd2_unbiased  # noqa: E402
from model.train import _project_mmd_to_visible_genes  # noqa: E402


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def _mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    sigmas = median_sigmas(y)
    return float(mmd2_unbiased(x, y, sigmas).item())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    x = torch.randn(8, 10)
    y = x + 0.05 * torch.randn(8, 10)

    budget_mask = torch.zeros_like(x)
    budget_mask[:, 4:] = 1.0
    x_noisy = x.clone()
    y_noisy = y.clone()
    x_noisy[:, 4:] = 100.0 * torch.randn(8, 6)
    y_noisy[:, 4:] = -100.0 * torch.randn(8, 6)

    x_proj, y_proj, n_genes = _project_mmd_to_visible_genes(x_noisy, y_noisy, budget_mask)
    direct_subset_mmd = _mmd(x_noisy[:, :4], y_noisy[:, :4])
    projected_mmd = _mmd(x_proj, y_proj)
    unprojected_mmd = _mmd(x_noisy, y_noisy)

    row_mask = torch.zeros_like(x)
    row_mask[:, 5:] = 1.0
    row_mask[0, 2] = 1.0
    row_mask[3, 3] = 1.0
    xr, yr, n_row = _project_mmd_to_visible_genes(x, y, row_mask)
    expected_keep = (row_mask < 0.5).all(dim=0)
    row_mask_shape_ok = xr.shape[1] == int(expected_keep.sum().item()) == n_row
    row_mask_exact_ok = torch.equal(xr, x[:, expected_keep]) and torch.equal(yr, y[:, expected_keep])

    train_py = COUPLEDFM / "model/train.py"
    launcher_py = COUPLEDFM / "model/tools/launch_stack_train.py"
    train_text = train_py.read_text(encoding="utf-8")
    launcher_text = launcher_py.read_text(encoding="utf-8")
    guard_removed = "gene_budget_manifest_path with train.use_mmd=True is not supported yet" not in train_text
    helper_used = "_project_mmd_to_visible_genes(" in train_text
    launcher_has_mask_args = "--gene-mask-prob" in launcher_text and "--gene-mask-all-prob" in launcher_text

    subset_close = math.isclose(projected_mmd, direct_subset_mmd, rel_tol=1e-6, abs_tol=1e-6)
    unprojected_differs = abs(unprojected_mmd - projected_mmd) > 1e-3
    pass_gate = all(
        [
            n_genes == 4,
            subset_close,
            unprojected_differs,
            row_mask_shape_ok,
            row_mask_exact_ok,
            guard_removed,
            helper_used,
            launcher_has_mask_args,
        ]
    )

    status = (
        "rawfm_mask_aware_mmd_gate_pass_gpu_packet_possible"
        if pass_gate
        else "rawfm_mask_aware_mmd_gate_fail_no_gpu"
    )
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_packet_possible": bool(pass_gate),
        "n_projected_budget_genes": int(n_genes),
        "direct_subset_mmd": direct_subset_mmd,
        "projected_mmd": projected_mmd,
        "unprojected_mmd": unprojected_mmd,
        "subset_close": subset_close,
        "unprojected_differs": unprojected_differs,
        "row_mask_shape_ok": row_mask_shape_ok,
        "row_mask_exact_ok": row_mask_exact_ok,
        "guard_removed": guard_removed,
        "helper_used": helper_used,
        "launcher_has_mask_args": launcher_has_mask_args,
    }
    json_path = OUT_DIR / "rawfm_mask_aware_mmd_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = OUT_DIR / "LATENTFM_RAWFM_MASK_AWARE_MMD_GATE_20260628.md"
    lines = [
        "# RawFM Mask-Aware Budget MMD Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU packet possible: `{payload['gpu_packet_possible']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthetic and static code gate.",
        "- No training, no inference, no GPU, no checkpoint selection.",
        "- Verifies that train-time MMD ignores masked genes before bandwidth and kernel computation.",
        "",
        "## Checks",
        "",
        f"- projected budget genes: `{n_genes}`",
        f"- projected MMD: `{projected_mmd:.8f}`",
        f"- direct kept-gene subset MMD: `{direct_subset_mmd:.8f}`",
        f"- unprojected noisy MMD: `{unprojected_mmd:.8f}`",
        f"- projected equals direct subset: `{subset_close}`",
        f"- unprojected differs after masked-gene noise: `{unprojected_differs}`",
        f"- row-specific mask exact common-visible projection: `{row_mask_exact_ok}`",
        f"- old gene-budget+MMD guard removed: `{guard_removed}`",
        f"- helper used in train.py: `{helper_used}`",
        f"- launcher can set gene mask probs: `{launcher_has_mask_args}`",
        "",
        "## Gate",
        "",
        "- If pass: launch fixed-step/no-selection RawFM MMD smoke with `--gene-mask-prob 0 --gene-mask-all-prob 0`.",
        "- If fail: do not run budgeted MMD GPU; fix mask projection first.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{json_path}`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
