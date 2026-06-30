#!/usr/bin/env python3
"""CPU-only feasibility gate for chemical unseen-drug/scaffold scaling splits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
S0_TSV = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
DRUG_META = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625/drug_metadata.tsv"
REPORT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_DRUG_SCAFFOLD_PROTOCOL_GATE_20260625.md"
REPORT_JSON = ROOT / "reports/latentfm_chemical_unseen_drug_scaffold_protocol_gate_20260625.json"


def stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def drug_from_row(row: dict[str, str]) -> str:
    perturbation = row["perturbation"]
    bg = row["cell_background_source"]
    prefix = f"{bg}_"
    if perturbation.startswith(prefix):
        return perturbation[len(prefix) :]
    return perturbation.split("_", 1)[-1]


def summarize_split(rows: list[dict[str, str]], drug_meta: dict[str, dict[str, str]], mode: str) -> dict:
    by_drug = defaultdict(list)
    for row in rows:
        drug = drug_from_row(row)
        if drug in drug_meta:
            by_drug[drug].append(row)

    drug_scaffold = {drug: drug_meta[drug]["scaffold"] for drug in by_drug}
    if mode == "unseen_drug":
        eval_drugs = {d for d in by_drug if stable_fraction(f"drug:{d}") < 0.20}
    elif mode == "unseen_scaffold":
        scaffolds = sorted({drug_scaffold[d] for d in by_drug})
        eval_scaffolds = {s for s in scaffolds if stable_fraction(f"scaffold:{s}") < 0.20}
        eval_drugs = {d for d, s in drug_scaffold.items() if s in eval_scaffolds}
    else:
        raise ValueError(mode)

    train_drugs = set(by_drug) - eval_drugs
    train_rows = [r for d in sorted(train_drugs) for r in by_drug[d]]
    eval_rows = [r for d in sorted(eval_drugs) for r in by_drug[d]]
    train_scaffolds = {drug_scaffold[d] for d in train_drugs}
    eval_scaffolds = {drug_scaffold[d] for d in eval_drugs}

    def row_summary(split_rows: list[dict[str, str]]) -> dict:
        return {
            "conditions": len(split_rows),
            "backgrounds": dict(sorted(Counter(r["cell_background_source"] for r in split_rows).items())),
            "doses": dict(sorted(Counter(r["dose"] for r in split_rows).items())),
            "pathways": dict(Counter(r["pathway"] for r in split_rows).most_common()),
            "loader_membership": dict(sorted(Counter(r["allmod_doseaware_budget32_seed42_loader_membership"] for r in split_rows).items())),
            "n_cells_total": sum(int(float(r["n_cells"])) for r in split_rows if r["n_cells"]),
        }

    train_eval_drug_overlap = sorted(train_drugs & eval_drugs)
    scaffold_overlap = sorted(train_scaffolds & eval_scaffolds)
    if mode == "unseen_drug":
        leakage_ok = not train_eval_drug_overlap
    else:
        leakage_ok = not scaffold_overlap

    eval_bgs = {r["cell_background_source"] for r in eval_rows}
    train_bgs = {r["cell_background_source"] for r in train_rows}
    pass_minima = {
        "train_drugs_ge_100": len(train_drugs) >= 100,
        "eval_drugs_ge_20": len(eval_drugs) >= 20,
        "train_conditions_ge_1000": len(train_rows) >= 1000,
        "eval_conditions_ge_200": len(eval_rows) >= 200,
        "train_all_3_backgrounds": len(train_bgs) == 3,
        "eval_all_3_backgrounds": len(eval_bgs) == 3,
        "leakage_boundary_ok": leakage_ok,
    }
    if mode == "unseen_scaffold":
        pass_minima["eval_scaffolds_ge_20"] = len(eval_scaffolds) >= 20
        pass_minima["train_scaffolds_ge_100"] = len(train_scaffolds) >= 100

    return {
        "mode": mode,
        "status": "feasible_cpu_materializer_next" if all(pass_minima.values()) else "not_feasible_or_needs_relaxed_design",
        "train": row_summary(train_rows),
        "eval": row_summary(eval_rows),
        "n_train_drugs": len(train_drugs),
        "n_eval_drugs": len(eval_drugs),
        "n_train_scaffolds": len(train_scaffolds),
        "n_eval_scaffolds": len(eval_scaffolds),
        "train_eval_drug_overlap": train_eval_drug_overlap[:20],
        "train_eval_scaffold_overlap_count": len(scaffold_overlap),
        "train_eval_scaffold_overlap_examples": scaffold_overlap[:20],
        "pass_minima": pass_minima,
        "example_eval_drugs": sorted(eval_drugs)[:20],
    }


def main() -> None:
    s0_rows = read_tsv(S0_TSV)
    meta_rows = read_tsv(DRUG_META)
    drug_meta = {r["drug"]: r for r in meta_rows}
    chemical_rows = [
        r
        for r in s0_rows
        if r["modality"] == "chemical"
        and r["source_quality"] == "source_verified"
        and r["perturbation_type"] == "drug"
    ]
    resolved = [r for r in chemical_rows if drug_from_row(r) in drug_meta]
    unresolved = [r for r in chemical_rows if drug_from_row(r) not in drug_meta]

    by_drug = defaultdict(list)
    for row in resolved:
        by_drug[drug_from_row(row)].append(row)
    complete_3bg = [
        d
        for d, rows in by_drug.items()
        if len({r["cell_background_source"] for r in rows}) == 3
    ]

    splits = [
        summarize_split(resolved, drug_meta, "unseen_drug"),
        summarize_split(resolved, drug_meta, "unseen_scaffold"),
    ]
    gpu_authorized = False
    cpu_materializer_authorized = any(s["status"] == "feasible_cpu_materializer_next" for s in splits)
    status = (
        "chemical_unseen_drug_scaffold_cpu_materializer_authorized_no_gpu"
        if cpu_materializer_authorized
        else "chemical_unseen_drug_scaffold_protocol_gate_fail_no_gpu"
    )
    result = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "cpu_materializer_authorized": cpu_materializer_authorized,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": {
            "task": "CPU-only feasibility gate",
            "uses_training": False,
            "uses_model_outputs": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "split_selection_inputs": ["S0 provenance rows", "SciPlex drug metadata", "stable hashes of drug/scaffold labels"],
        },
        "inputs": {"s0_tsv": str(S0_TSV), "drug_metadata": str(DRUG_META)},
        "summary": {
            "s0_chemical_rows": len(chemical_rows),
            "resolved_rows": len(resolved),
            "unresolved_rows": len(unresolved),
            "resolved_drugs": len(by_drug),
            "resolved_scaffolds": len({drug_meta[d]["scaffold"] for d in by_drug}),
            "complete_3_background_drugs": len(complete_3bg),
            "background_counts": dict(sorted(Counter(r["cell_background_source"] for r in resolved).items())),
            "dose_counts": dict(sorted(Counter(r["dose"] for r in resolved).items())),
            "loader_membership_counts": dict(sorted(Counter(r["allmod_doseaware_budget32_seed42_loader_membership"] for r in resolved).items())),
        },
        "splits": splits,
        "next_action": (
            "materialize deterministic unseen-drug/unseen-scaffold split artifacts and dry-load before any GPU smoke"
            if cpu_materializer_authorized
            else "do not launch chemical scaling GPU; improve metadata/protocol first"
        ),
    }
    REPORT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Chemical Unseen-Drug/Scaffold Protocol Gate 20260625",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only feasibility gate; no training, model outputs, canonical multi, or Track C query.",
        "- Deterministic split candidates use stable hashes of drug/scaffold labels.",
        "- This gate can authorize only a CPU split materializer/dry-load step, not GPU training.",
        "",
        "## Summary",
        "",
        f"- S0 chemical rows: `{len(chemical_rows)}`",
        f"- Resolved rows with Morgan512 metadata: `{len(resolved)}`",
        f"- Resolved drugs/scaffolds: `{len(by_drug)}` / `{len({drug_meta[d]['scaffold'] for d in by_drug})}`",
        f"- Drugs observed in all 3 backgrounds: `{len(complete_3bg)}`",
        f"- Loader membership: `{result['summary']['loader_membership_counts']}`",
        "",
        "## Candidate Splits",
        "",
        "| mode | status | train drugs | eval drugs | train scaffolds | eval scaffolds | train conds | eval conds | leakage issue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in splits:
        leakage_issue = (
            f"scaffold overlap {split['train_eval_scaffold_overlap_count']}"
            if split["mode"] == "unseen_scaffold"
            else f"drug overlap {len(split['train_eval_drug_overlap'])}"
        )
        lines.append(
            f"| `{split['mode']}` | `{split['status']}` | {split['n_train_drugs']} | {split['n_eval_drugs']} | "
            f"{split['n_train_scaffolds']} | {split['n_eval_scaffolds']} | {split['train']['conditions']} | "
            f"{split['eval']['conditions']} | {leakage_issue} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- `cpu_materializer_authorized`: `{cpu_materializer_authorized}`",
        "- `gpu_authorized`: `False`",
        f"- next action: {result['next_action']}",
        "",
        "## JSON",
        "",
        f"`{REPORT_JSON}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(REPORT_MD), "json": str(REPORT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
