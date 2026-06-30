#!/usr/bin/env python3
"""CPU feasibility audit for train-only masked perturbation-token denoising.

The proposed pretext is only useful for Track A if the masked target token has
non-leaky context. For single-gene conditions, masking the only gene leaves no
perturbation-token context, so a denoiser would either learn dataset/count
proxies or leak the target through an unmasked representation. This script
quantifies that boundary over the canonical/train-only splits and exact Track A
rows. It does not train, infer, select checkpoints, read canonical multi for
selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
CROSSBG_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
SINGLEVAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_single_val_v1.json"
EXACT_CROSS_ROWS = ROOT / "reports/tracka_cross_background_seen_gene_exact_20260627/cross_background_seen_gene_rows.csv"
EXACT_SIMPLE_ROWS = ROOT / "reports/tracka_simple_single_unseen_exact_20260627/condition_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_masked_denoise_feasibility_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_MASKED_DENOISE_FEASIBILITY_20260627.md"


GENE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
NON_GENE_MARKERS = ("DMSO", "CTRL", "CONTROL", "VEHICLE", "UNTREATED", "NTC", "NON-TARGET", "SAFE-TARGET")


def canonical_gene(raw: str) -> str:
    return str(raw).strip().upper()


def parse_gene_tokens(condition: str) -> list[str]:
    cond = str(condition).strip()
    if not cond:
        return []
    parts = [p.strip() for p in re.split(r"[+;,|]", cond) if p.strip()]
    out = []
    for part in parts:
        gene = canonical_gene(part)
        if any(marker in gene for marker in NON_GENE_MARKERS):
            continue
        if not GENE_RE.match(gene):
            continue
        # Drug strings often contain dose/time punctuation or SMILES-like tokens;
        # keep this conservative for a feasibility audit.
        if len(gene) > 24:
            continue
        out.append(gene)
    return out


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_rows(path: Path, groups: tuple[str, ...]) -> list[dict[str, Any]]:
    split = load_split(path)
    rows = []
    for dataset, payload in split.items():
        if not isinstance(payload, dict):
            continue
        for group in groups:
            for cond in payload.get(group, []) or []:
                genes = parse_gene_tokens(str(cond))
                rows.append(
                    {
                        "split": path.name,
                        "dataset": str(dataset),
                        "group": group,
                        "condition": str(cond),
                        "n_gene_tokens": len(genes),
                        "genes": genes,
                    }
                )
    return rows


def csv_rows(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("seed") not in {"", None, "seed42"}:
                continue
            condition = str(row.get("condition", ""))
            gene = str(row.get("gene", "")).strip()
            genes = [canonical_gene(gene)] if gene else parse_gene_tokens(condition)
            rows.append(
                {
                    "split": label,
                    "dataset": str(row.get("dataset", "")),
                    "group": str(row.get("group", label)),
                    "condition": condition,
                    "n_gene_tokens": len([g for g in genes if g]),
                    "genes": [g for g in genes if g],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, Counter] = defaultdict(Counter)
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        n = int(row["n_gene_tokens"])
        bucket = "zero_gene_or_non_gene" if n == 0 else ("single_gene_no_mask_context" if n == 1 else "multi_gene_has_partner_context")
        key = f"{row['split']}::{row['group']}"
        by_group[key][bucket] += 1
        by_dataset[str(row["dataset"])][bucket] += 1
    total = Counter()
    for counter in by_group.values():
        total.update(counter)
    return {
        "total": dict(total),
        "by_group": {k: dict(v) for k, v in sorted(by_group.items())},
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
    }


def main() -> None:
    train_rows = []
    train_rows.extend(split_rows(CANONICAL_SPLIT, ("train",)))
    train_rows.extend(split_rows(CROSSBG_SPLIT, ("train", "internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")))
    train_rows.extend(split_rows(SINGLEVAL_SPLIT, ("train", "internal_val_from_canonical_train_single")))
    exact_rows = []
    exact_rows.extend(csv_rows(EXACT_CROSS_ROWS, "exact_cross_background_seen_gene"))
    exact_rows.extend(csv_rows(EXACT_SIMPLE_ROWS, "exact_simple_single_unseen"))

    train_summary = summarize(train_rows)
    exact_summary = summarize(exact_rows)
    total_exact = Counter(exact_summary["total"])
    total_train = Counter(train_summary["total"])

    exact_single = int(total_exact.get("single_gene_no_mask_context", 0))
    exact_multi = int(total_exact.get("multi_gene_has_partner_context", 0))
    exact_total = sum(total_exact.values())
    train_multi = int(total_train.get("multi_gene_has_partner_context", 0))

    reasons = []
    if exact_total == 0:
        reasons.append("no_exact_tracka_rows_found")
    if exact_single / max(exact_total, 1) > 0.80:
        reasons.append("tracka_exact_rows_are_mostly_single_gene_no_mask_context")
    if exact_multi < 20:
        reasons.append("too_few_exact_multigene_rows_for_tracka_pretext_validation")
    if train_multi < 50:
        reasons.append("too_few_train_multigene_partner_context_rows")
    reasons.append("no_trainonly_metric_probe_or_mmd_noharm_yet_no_gpu")

    status = "tracka_masked_denoise_feasibility_fail_no_gpu"
    if not any(r != "no_trainonly_metric_probe_or_mmd_noharm_yet_no_gpu" for r in reasons):
        status = "tracka_masked_denoise_feasibility_source_ok_needs_metric_probe_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_feasibility_only": True,
            "training": False,
            "inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "inputs": {
            "canonical_split": str(CANONICAL_SPLIT),
            "crossbg_split": str(CROSSBG_SPLIT),
            "singleval_split": str(SINGLEVAL_SPLIT),
            "exact_cross_rows": str(EXACT_CROSS_ROWS),
            "exact_simple_rows": str(EXACT_SIMPLE_ROWS),
        },
        "train_summary": train_summary,
        "exact_tracka_summary": exact_summary,
        "key_counts": {
            "exact_total": exact_total,
            "exact_single_gene_no_mask_context": exact_single,
            "exact_multi_gene_has_partner_context": exact_multi,
            "train_multi_gene_has_partner_context": train_multi,
        },
        "decision_reasons": reasons,
        "next_action": (
            "close masked-denoise as immediate Track A GPU route"
            if status.endswith("fail_no_gpu")
            else "design tiny train-only metric/no-harm probe before implementation"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Masked-Denoise Feasibility",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/source feasibility only. The audit counts whether masked perturbation-token denoising has non-leaky partner-token context in train/internal and exact Track A rows. It does not train, infer, select checkpoints, use canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Key Counts",
        "",
        "| item | count |",
        "|---|---:|",
    ]
    for key, val in payload["key_counts"].items():
        lines.append(f"| `{key}` | {val} |")
    lines.extend(["", "## Exact Track A Context Buckets", ""])
    lines.append("| bucket | count |")
    lines.append("|---|---:|")
    for key, val in sorted(total_exact.items()):
        lines.append(f"| `{key}` | {val} |")
    lines.extend(["", "## Train/Internal Context Buckets", ""])
    lines.append("| bucket | count |")
    lines.append("|---|---:|")
    for key, val in sorted(total_train.items()):
        lines.append(f"| `{key}` | {val} |")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not launch a GPU smoke from masked-denoising unless a later design supplies non-leaky context for single-gene Track A rows and then passes a train-only metric/no-harm probe.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
