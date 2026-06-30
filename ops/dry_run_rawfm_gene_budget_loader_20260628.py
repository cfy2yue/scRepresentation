#!/usr/bin/env python3
"""CPU dry-run for RawFM gene-budget manifests and loader masking."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

from model.data.dataset import CoupledFMDataset  # noqa: E402
from model.data.vocab import GeneVocab  # noqa: E402
from model.paths import gene_name_path, nichenet_node2idx_path  # noqa: E402
from model.utils.data.split import load_split_json  # noqa: E402


OUT_DIR = ROOT / "reports/rawfm_gene_budget_loader_dryrun_20260628"
MANIFEST = (
    ROOT
    / "reports/rawfm_gene_budget_manifest_dryrun_20260628"
    / "response_topk_k256_seed42.json"
)
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
BIFLOW = ROOT / "dataset/biFlow_data"
DATASET = "Wessels"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = load_split_json(SPLIT)
    vocab = GeneVocab(str(gene_name_path()), str(nichenet_node2idx_path()))
    ds = CoupledFMDataset(
        str(BIFLOW),
        vocab,
        split,
        mode="train",
        coupling_mode="ot",
        batch_size=4,
        min_cells=16,
        ds_alpha=1.0,
        ot_method="torch_sinkhorn",
        ot_feature="raw",
        ot_emb_cap_src=8,
        ot_emb_cap_gt=8,
        gene_mask_prob=0.0,
        gene_mask_all_prob=0.0,
        gene_budget_manifest_path=str(MANIFEST),
        gene_budget_label="response_topk_k256_seed42_dryrun",
        dataset_names=[DATASET],
        latent_backbone="stack",
        seed=42,
    )
    test_ds = None
    try:
        batch = next(iter(ds))
        x_t, x_ctrl, _t, _gene_ids, dx_t, gene_mask = batch[:6]
        mask = ds.budget_mask_for_eval(DATASET)
        assert mask is not None
        masked = mask > 0.5
        kept = ~masked
        checks = {
            "manifest_exists": MANIFEST.is_file(),
            "loaded_datasets": ds.ds_names,
            "kept_genes": int(kept.sum()),
            "masked_genes": int(masked.sum()),
            "x_t_masked_zero": bool(np.allclose(x_t.numpy()[:, masked], 0.0)),
            "x_ctrl_masked_zero": bool(np.allclose(x_ctrl.numpy()[:, masked], 0.0)),
            "dx_t_masked_zero": bool(np.allclose(dx_t.numpy()[:, masked], 0.0)),
            "gene_mask_masked_one": bool(np.all(gene_mask.numpy()[:, masked] > 0.5)),
            "gene_mask_kept_zero": bool(np.all(gene_mask.numpy()[:, kept] < 0.5)),
        }
        test_ds = CoupledFMDataset(
            str(BIFLOW),
            vocab,
            split,
            mode="test",
            coupling_mode="ot",
            batch_size=4,
            min_cells=16,
            ds_alpha=1.0,
            ot_method="torch_sinkhorn",
            ot_feature="raw",
            ot_emb_cap_src=8,
            ot_emb_cap_gt=8,
            gene_mask_prob=0.0,
            gene_mask_all_prob=0.0,
            gene_budget_manifest_path=str(MANIFEST),
            gene_budget_label="response_topk_k256_seed42_dryrun",
            dataset_names=[DATASET],
            shared_handles=ds.handles,
            latent_backbone="stack",
            seed=1042,
        )
        eval_mask = test_ds.budget_mask_for_eval(DATASET)
        checks["eval_mask_matches_train"] = bool(np.array_equal(mask, eval_mask))
        pass_all = all(v for k, v in checks.items() if k not in {"loaded_datasets", "kept_genes", "masked_genes"})
        status = "rawfm_gene_budget_loader_dry_run_pass" if pass_all else "rawfm_gene_budget_loader_dry_run_fail"
    finally:
        if test_ds is not None:
            test_ds.close()
        ds.close()

    obj = {
        "timestamp": now_cst(),
        "status": status,
        "split": str(SPLIT),
        "manifest": str(MANIFEST),
        "dataset": DATASET,
        "checks": checks,
    }
    (OUT_DIR / "rawfm_gene_budget_loader_dryrun_20260628.json").write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# RawFM Gene-Budget Loader Dry-Run",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU loader dry-run only.",
        "- No training, no inference, no GPU, no canonical multi, and no Track C query.",
        "- Dataset: `Wessels`; budget: `response_topk_k256_seed42`.",
        "",
        "## Checks",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for k, v in checks.items():
        lines.append(f"| {k} | `{v}` |")
    lines.extend([
        "",
        "## Decision",
        "",
        "- Loader/provenance gate passes for the Wessels k=256 response-topk dry-run." if status.endswith("_pass")
        else "- Loader/provenance gate failed; do not launch GPU.",
    ])
    (OUT_DIR / "LATENTFM_RAWFM_GENE_BUDGET_LOADER_DRYRUN_20260628.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
