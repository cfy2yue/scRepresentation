#!/usr/bin/env python3
"""CPU-only gate for confound-aware chemical pathway/dose sampling.

This audit only reads train-only split artifacts and condition metadata. It
does not read canonical/Track C query data, model outputs, or launch GPU jobs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
BIOFLOW = ROOT / "dataset/biFlow_data"
SCALING_SPLITS = BIOFLOW / "xverse_scaling_splits_v2_20260624"

SIDE_SLATE = REPORTS / "LATENTFM_SCALING_TRAINING_DATA_SIDE_SLATE_20260624.md"
INVENTORY = REPORTS / "latentfm_condition_level_inventory_20260624.json"
BASE_SPLIT = BIOFLOW / "split_seed42_xverse_trainonly_crossbg_val_v2.json"
CAP120_SPLIT = SCALING_SPLITS / "split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
TYPE_BALANCED_SPLIT = SCALING_SPLITS / "split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json"
PROTOCOL_SPLIT_BUILDER = ROOT / "ops/build_latentfm_scaling_protocol_splits_20260624.py"
PROTOCOL_MEANS_BUILDER = ROOT / "ops/compute_latentfm_scaling_protocol_pert_means_20260624.py"

OUT_JSON = REPORTS / "latentfm_modality_pathway_sampling_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_MODALITY_PATHWAY_SAMPLING_GATE_20260624.md"
OUT_SPLIT = REPORTS / "latentfm_modality_pathway_sampling_candidate_split_20260624.json"

CHEMICAL_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}
PATHWAY_QUOTA_PER_BACKGROUND = 12
SEED = 20260624


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_score(*parts: str) -> str:
    return hashlib.sha256("\t".join(parts).encode("utf-8")).hexdigest()


def norm_drug(row: dict[str, Any]) -> str:
    """Map condition-level sciplex rows back to the drug-level split key."""
    backgrounds = row.get("backgrounds") or []
    background = str(backgrounds[0]) if backgrounds else ""
    perturbation = str(row.get("perturbation") or "")
    if background and perturbation.startswith(f"{background}_"):
        return perturbation[len(background) + 1 :]
    condition = str(row.get("condition") or "")
    if background and condition.startswith(f"{background}_"):
        return condition[len(background) + 1 :].rsplit("_", 1)[0]
    return perturbation


def split_train_set(split: dict[str, Any], *, chemical_only: bool = False) -> set[tuple[str, str]]:
    out = set()
    for ds, groups in split.items():
        if chemical_only and ds not in CHEMICAL_DATASETS:
            continue
        for cond in groups.get("train") or []:
            out.add((str(ds), str(cond)))
    return out


def jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    return float(len(a & b) / max(1, len(a | b)))


def copy_groups_with_train(groups: dict[str, Any], train: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in groups.items():
        out[key] = [str(x) for x in val] if isinstance(val, list) else val
    out["train"] = sorted(str(x) for x in train)
    return out


def chemical_metadata(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset") or "")
        if ds not in CHEMICAL_DATASETS:
            continue
        drug = norm_drug(row)
        if not drug:
            continue
        item = meta.setdefault(
            (ds, drug),
            {
                "pathways": set(),
                "doses": set(),
                "backgrounds": set(),
                "condition_rows": 0,
            },
        )
        pathway = str(row.get("pathway") or "")
        dose = str(row.get("dose") or "")
        if pathway:
            item["pathways"].add(pathway)
        if dose:
            item["doses"].add(dose)
        for background in row.get("backgrounds") or []:
            item["backgrounds"].add(str(background))
        item["condition_rows"] += 1
    return meta


def public_meta(meta: dict[tuple[str, str], dict[str, Any]], key: tuple[str, str]) -> dict[str, Any]:
    item = meta.get(key) or {}
    return {
        "pathways": sorted(item.get("pathways") or []),
        "doses": sorted(item.get("doses") or []),
        "backgrounds": sorted(item.get("backgrounds") or []),
        "condition_rows": int(item.get("condition_rows") or 0),
    }


def summarize_split(split: dict[str, Any], meta: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    pathway_counts: Counter[str] = Counter()
    dose_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    dataset_counts: dict[str, int] = {}
    missing = []
    multi_pathway = []
    for ds in sorted(CHEMICAL_DATASETS):
        train = [str(x) for x in (split.get(ds) or {}).get("train") or []]
        dataset_counts[ds] = len(train)
        for cond in train:
            item = meta.get((ds, cond))
            if item is None:
                missing.append(f"{ds}:{cond}")
                continue
            pathways = sorted(item["pathways"])
            if len(pathways) != 1:
                multi_pathway.append(f"{ds}:{cond}:{pathways}")
            for pathway in pathways:
                pathway_counts[pathway] += 1
            for dose in item["doses"]:
                dose_counts[dose] += 1
            for background in item["backgrounds"]:
                background_counts[background] += 1
    chemical_total = sum(dataset_counts.values())
    return {
        "chemical_train_conditions": chemical_total,
        "chemical_dataset_counts": dataset_counts,
        "unique_pathways": len(pathway_counts),
        "pathway_counts": dict(sorted(pathway_counts.items())),
        "top_pathway_share": float(max(pathway_counts.values(), default=0) / max(1, chemical_total)),
        "unique_doses": len(dose_counts),
        "dose_counts": dict(sorted(dose_counts.items())),
        "dose_count_range": [min(dose_counts.values(), default=0), max(dose_counts.values(), default=0)],
        "background_counts": dict(sorted(background_counts.items())),
        "missing_chemical_metadata": missing[:20],
        "multi_or_missing_pathway_examples": multi_pathway[:20],
    }


def copy_cap120_with_pathway_quota(
    cap120: dict[str, Any],
    meta: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Keep gene/non-chemical coverage fixed and quota only sciplex drugs."""
    candidate: dict[str, Any] = {}
    for ds, groups in sorted(cap120.items()):
        ds_s = str(ds)
        train = [str(x) for x in groups.get("train") or []]
        if ds_s not in CHEMICAL_DATASETS:
            candidate[ds_s] = copy_groups_with_train(groups, train)
            continue
        by_pathway: dict[str, list[str]] = defaultdict(list)
        for cond in train:
            pathways = sorted((meta.get((ds_s, cond)) or {}).get("pathways") or [])
            pathway = pathways[0] if len(pathways) == 1 else "__missing_or_multi_pathway__"
            by_pathway[pathway].append(cond)
        selected: list[str] = []
        for pathway, conds in sorted(by_pathway.items()):
            ranked = sorted(
                conds,
                key=lambda cond: stable_score(str(SEED), ds_s, pathway, cond),
            )
            selected.extend(ranked[:PATHWAY_QUOTA_PER_BACKGROUND])
        candidate[ds_s] = copy_groups_with_train(groups, selected)
    return candidate


def split_safety_checks(candidate: dict[str, Any], base: dict[str, Any], cap120: dict[str, Any]) -> list[str]:
    reasons = []
    for ds, groups in sorted(candidate.items()):
        train = {str(x) for x in groups.get("train") or []}
        base_groups = base.get(ds) or {}
        cap_groups = cap120.get(ds) or {}
        base_train = {str(x) for x in base_groups.get("train") or []}
        cap_train = {str(x) for x in cap_groups.get("train") or []}
        eval_set = set()
        for key, val in base_groups.items():
            if key != "train" and isinstance(val, list):
                eval_set.update(str(x) for x in val)
        if not train.issubset(base_train):
            reasons.append(f"{ds}:train_not_subset_base_train")
        if not train.issubset(cap_train):
            reasons.append(f"{ds}:train_not_subset_cap120_parent")
        if train & eval_set:
            reasons.append(f"{ds}:train_eval_overlap")
        for key, val in base_groups.items():
            if key == "train" or not isinstance(val, list):
                continue
            cand_val = [str(x) for x in (groups.get(key) or [])]
            base_val = [str(x) for x in val]
            if cand_val != base_val:
                reasons.append(f"{ds}:{key}_validation_changed")
                break
    return reasons


def nonchemical_unchanged(candidate: dict[str, Any], cap120: dict[str, Any]) -> bool:
    for ds, groups in cap120.items():
        ds_s = str(ds)
        if ds_s in CHEMICAL_DATASETS:
            continue
        cap_train = sorted(str(x) for x in groups.get("train") or [])
        cand_train = sorted(str(x) for x in (candidate.get(ds_s) or {}).get("train") or [])
        if cap_train != cand_train:
            return False
    return True


def axis_separability(meta: dict[tuple[str, str], dict[str, Any]], cap120: dict[str, Any]) -> dict[str, Any]:
    pathway_backgrounds: dict[str, set[str]] = defaultdict(set)
    dose_backgrounds: dict[str, set[str]] = defaultdict(set)
    dose_sets: Counter[tuple[str, ...]] = Counter()
    matched = 0
    total = 0
    for ds in sorted(CHEMICAL_DATASETS):
        for cond in (cap120.get(ds) or {}).get("train") or []:
            total += 1
            item = meta.get((ds, str(cond)))
            if not item:
                continue
            matched += 1
            for pathway in item["pathways"]:
                pathway_backgrounds[pathway].update(item["backgrounds"])
            for dose in item["doses"]:
                dose_backgrounds[dose].update(item["backgrounds"])
            dose_sets[tuple(sorted(item["doses"]))] += 1
    shared_pathways = {k: sorted(v) for k, v in pathway_backgrounds.items() if len(v) >= 2}
    shared_doses = {k: sorted(v) for k, v in dose_backgrounds.items() if len(v) >= 2}
    return {
        "cap120_chemical_conditions": total,
        "matched_chemical_metadata": matched,
        "metadata_match_fraction": float(matched / max(1, total)),
        "pathways_seen": len(pathway_backgrounds),
        "pathways_shared_by_at_least_2_backgrounds": len(shared_pathways),
        "pathway_background_examples": dict(sorted(shared_pathways.items())[:8]),
        "doses_seen": len(dose_backgrounds),
        "doses_shared_by_at_least_2_backgrounds": len(shared_doses),
        "dose_backgrounds": dict(sorted(shared_doses.items())),
        "dose_sets_per_drug": {"|".join(k): v for k, v in sorted(dose_sets.items())},
        "split_granularity": "drug_level",
        "dose_selectable_by_current_split": False,
        "dose_balance_interpretation": "dose rows are balanced structurally for selected drugs, but the current split key cannot select individual doses without launcher/loader support",
    }


def fail_reasons(
    *,
    cap120_summary: dict[str, Any],
    type_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    separability: dict[str, Any],
    safety_reasons: list[str],
    same_nonchemical: bool,
    all_jaccard_type_balanced: float,
    chemical_jaccard_type_balanced: float,
) -> list[str]:
    reasons: list[str] = []
    if safety_reasons:
        reasons.extend(safety_reasons)
    if separability["metadata_match_fraction"] < 0.95:
        reasons.append("chemical_metadata_match_fraction_lt_0p95")
    if separability["pathways_shared_by_at_least_2_backgrounds"] < 10:
        reasons.append("pathway_background_crossing_too_sparse")
    if separability["doses_shared_by_at_least_2_backgrounds"] < 4:
        reasons.append("dose_background_crossing_too_sparse")
    if not same_nonchemical:
        reasons.append("gene_or_nonchemical_coverage_changed")
    if candidate_summary["chemical_train_conditions"] <= 0:
        reasons.append("candidate_removed_all_drugs")
    if candidate_summary["chemical_train_conditions"] < 0.65 * cap120_summary["chemical_train_conditions"]:
        reasons.append("candidate_chemical_count_lt_65pct_cap120")
    if candidate_summary["unique_pathways"] < 15:
        reasons.append("candidate_pathway_coverage_lt_15")
    if candidate_summary["top_pathway_share"] > 0.15:
        reasons.append("candidate_top_pathway_share_gt_0p15")
    if cap120_summary["top_pathway_share"] - candidate_summary["top_pathway_share"] < 0.05:
        reasons.append("pathway_top_share_not_reduced_by_0p05")
    if candidate_summary["dose_count_range"][0] != candidate_summary["dose_count_range"][1]:
        reasons.append("candidate_dose_rows_not_balanced")
    if all_jaccard_type_balanced > 0.90 or chemical_jaccard_type_balanced > 0.90:
        reasons.append("candidate_reproduces_type_balanced_cap120")
    if type_summary["chemical_train_conditions"] == candidate_summary["chemical_train_conditions"] and not same_nonchemical:
        reasons.append("candidate_degenerated_to_hard_type_balanced_cap120")
    return reasons


def render_md(payload: dict[str, Any]) -> str:
    status = payload["status"]
    lines = [
        "# LatentFM Modality/Pathway Sampling Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only audit.",
        "- Reads the side slate, condition-level inventory, and existing train-only scaling split artifacts.",
        "- Does not read canonical metrics, Track C query, model outputs, expression matrices, or active logs.",
        "- Does not launch GPU.",
        "",
        "## Gate Decision",
        "",
        f"- Immediate GPU authorization: `{payload['gpu_authorization']['immediate_gpu']}`",
        f"- Conditional candidate: `{payload['gpu_authorization']['conditional_candidate']}`",
        f"- Reason: {payload['gpu_authorization']['reason']}",
        "",
        "## Chemical Split Summaries",
        "",
        "| arm | chemical train | pathways | top pathway share | doses | dose range | background counts |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name in ["cap120_all", "type_balanced_cap120", "candidate_pathway_quota12"]:
        row = payload["summaries"][name]
        bgs = ", ".join(f"{k}:{v}" for k, v in row["background_counts"].items())
        lines.append(
            f"| `{name}` | {row['chemical_train_conditions']} | {row['unique_pathways']} | "
            f"{row['top_pathway_share']:.3f} | {row['unique_doses']} | "
            f"{row['dose_count_range'][0]}-{row['dose_count_range'][1]} | {bgs} |"
        )
    lines.extend(
        [
            "",
            "## Confound Checks",
            "",
            f"- Metadata match fraction for cap120 chemical drugs: `{payload['separability']['metadata_match_fraction']:.3f}`",
            f"- Pathways shared by at least two sciplex backgrounds: `{payload['separability']['pathways_shared_by_at_least_2_backgrounds']}`",
            f"- Doses shared by at least two sciplex backgrounds: `{payload['separability']['doses_shared_by_at_least_2_backgrounds']}`",
            f"- Current split granularity: `{payload['separability']['split_granularity']}`",
            f"- Dose selectable by current split: `{payload['separability']['dose_selectable_by_current_split']}`",
            f"- Dose note: {payload['separability']['dose_balance_interpretation']}",
            "",
            "## Degeneracy Checks",
            "",
            f"- Gene/non-chemical train coverage unchanged from cap120: `{payload['degeneracy_checks']['nonchemical_unchanged_from_cap120']}`",
            f"- Candidate vs type-balanced all-train Jaccard: `{payload['degeneracy_checks']['all_train_jaccard_vs_type_balanced']:.3f}`",
            f"- Candidate vs type-balanced chemical Jaccard: `{payload['degeneracy_checks']['chemical_train_jaccard_vs_type_balanced']:.3f}`",
            f"- Candidate split safety reasons: `{payload['degeneracy_checks']['split_safety_reasons']}`",
            "",
            "## Candidate Design",
            "",
            f"- Candidate split JSON: `{payload['candidate_design']['candidate_split_json']}`",
            f"- Hypothesis: {payload['candidate_design']['hypothesis']}",
            f"- Boundary: {payload['candidate_design']['boundary']}",
            f"- Resource plan: {payload['candidate_design']['resource_plan']}",
            f"- Promotion gate: {payload['candidate_design']['promotion_gate']}",
            f"- Stop rule: {payload['candidate_design']['stop_rule']}",
            f"- Launcher minimum change: {payload['candidate_design']['launcher_minimum_change']}",
            "",
            "## Fail Reasons",
            "",
        ]
    )
    if payload["fail_reasons"]:
        for reason in payload["fail_reasons"]:
            lines.append(f"- `{reason}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- MD: `{OUT_MD}`"])
    return "\n".join(lines)


def main() -> int:
    for path in [SIDE_SLATE, INVENTORY, BASE_SPLIT, CAP120_SPLIT, TYPE_BALANCED_SPLIT, PROTOCOL_SPLIT_BUILDER, PROTOCOL_MEANS_BUILDER]:
        if not path.exists():
            raise FileNotFoundError(path)

    # The side slate is read to make the audit provenance explicit; no model
    # outcomes are parsed or used from it.
    side_slate_bytes = SIDE_SLATE.read_bytes()
    inventory = load_json(INVENTORY)
    rows = inventory["rows"]
    base = load_json(BASE_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    type_balanced = load_json(TYPE_BALANCED_SPLIT)

    meta = chemical_metadata(rows)
    candidate = copy_cap120_with_pathway_quota(cap120, meta)
    OUT_SPLIT.write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")

    cap120_summary = summarize_split(cap120, meta)
    type_summary = summarize_split(type_balanced, meta)
    candidate_summary = summarize_split(candidate, meta)
    separability = axis_separability(meta, cap120)
    safety_reasons = split_safety_checks(candidate, base, cap120)
    same_nonchemical = nonchemical_unchanged(candidate, cap120)
    all_jaccard = jaccard(split_train_set(candidate), split_train_set(type_balanced))
    chemical_jaccard = jaccard(
        split_train_set(candidate, chemical_only=True),
        split_train_set(type_balanced, chemical_only=True),
    )
    reasons = fail_reasons(
        cap120_summary=cap120_summary,
        type_summary=type_summary,
        candidate_summary=candidate_summary,
        separability=separability,
        safety_reasons=safety_reasons,
        same_nonchemical=same_nonchemical,
        all_jaccard_type_balanced=all_jaccard,
        chemical_jaccard_type_balanced=chemical_jaccard,
    )

    status = (
        "modality_pathway_sampling_gate_fail_no_gpu"
        if reasons
        else "modality_pathway_sampling_gate_pass_candidate_design_no_immediate_gpu"
    )
    payload = {
        "status": status,
        "boundary": {
            "read_side_slate": str(SIDE_SLATE),
            "read_condition_inventory": str(INVENTORY),
            "read_base_split": str(BASE_SPLIT),
            "read_cap120_split": str(CAP120_SPLIT),
            "read_type_balanced_split": str(TYPE_BALANCED_SPLIT),
            "read_existing_protocol_scripts": [str(PROTOCOL_SPLIT_BUILDER), str(PROTOCOL_MEANS_BUILDER)],
            "read_canonical_metrics": False,
            "read_trackc_query": False,
            "read_model_results": False,
            "launched_gpu": False,
            "side_slate_sha256": hashlib.sha256(side_slate_bytes).hexdigest(),
        },
        "parameters": {
            "seed": SEED,
            "chemical_datasets": sorted(CHEMICAL_DATASETS),
            "pathway_quota_per_background": PATHWAY_QUOTA_PER_BACKGROUND,
            "parent_split": "cap120_all",
            "nonchemical_policy": "unchanged_from_cap120_all",
        },
        "separability": separability,
        "summaries": {
            "cap120_all": cap120_summary,
            "type_balanced_cap120": type_summary,
            "candidate_pathway_quota12": candidate_summary,
        },
        "degeneracy_checks": {
            "nonchemical_unchanged_from_cap120": same_nonchemical,
            "all_train_jaccard_vs_type_balanced": all_jaccard,
            "chemical_train_jaccard_vs_type_balanced": chemical_jaccard,
            "split_safety_reasons": safety_reasons,
        },
        "fail_reasons": reasons,
        "gpu_authorization": {
            "immediate_gpu": False,
            "conditional_candidate": not reasons,
            "reason": (
                "CPU design passes, but this agent only wrote reports; a real smoke still needs an accepted split path, pert-mean artifact, launcher/RUN_STATUS, and resource audit."
                if not reasons
                else "Fail-closed gate found degeneracy or inseparable confounding."
            ),
        },
        "candidate_design": {
            "candidate_split_json": str(OUT_SPLIT),
            "hypothesis": (
                "A within-chemical pathway-quota arm can reduce sciplex pathway overrepresentation "
                "while preserving gene/non-chemical cap120 coverage and avoiding the closed broad type-balanced cap120 design."
            ),
            "boundary": (
                "Train-only xverse cap120 parent; validation groups fixed; no canonical/Track C query/model-result selection; "
                "gene and non-chemical train conditions unchanged; sciplex drugs selected only from cap120 train."
            ),
            "resource_plan": (
                "CPU first: materialize accepted split plus train-only pert means. If promoted by coordinator, at most one 2k-3k warm-start smoke on one physical GPU after fresh resource audit."
            ),
            "promotion_gate": (
                "Internal cross-background pp >= anchor +0.010, family pp no regression, family MMD no material harm, "
                "dataset-min >= -0.020, and shuffled pathway labels/type-count-only controls collapse."
            ),
            "stop_rule": (
                "Stop if it matches type-balanced failure signs, harms family MMD, loses gene coverage, or pathway shuffle/control performs similarly."
            ),
            "launcher_minimum_change": (
                "Add this split to the scaling split manifest or pass it as a split-file override, compute a matching train-only pert-means NPZ, "
                "and record RUN_STATUS. Dose-specific sampling would require an additional dose-aware loader/condition filter; current split keys are drug-level with structurally balanced dose rows."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "immediate_gpu": False, "conditional_candidate": not reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
