#!/usr/bin/env python3
"""Control gate for nested true cell-count matrix.

Reads completed train-only/internal posthoc JSONs from the nested matrix. It
does not read canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_RUN_ROOT", ROOT / "runs/latentfm_true_cell_count_nested_smokes_20260624"))
OUT_JSON = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_OUT_JSON", ROOT / "reports/latentfm_true_cell_count_nested_controls_gate_20260624.json"))
OUT_MD = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_OUT_MD", ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_GATE_20260624.md"))
EXPECTED_RUNS = int(os.environ.get("LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_EXPECTED_RUNS", "9"))

GROUPS = {
    "cross_background": ("split_group", "internal_val_cross_background_seen_gene_proxy"),
    "family_gene": ("condition_family", "family_gene"),
    "test_single": ("condition_family", "test_single"),
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def parse_budget_seed(name: str) -> tuple[int | None, int | None]:
    seed_match = re.search(r"_seed(\d+)", name)
    budget_matches = re.findall(r"_budget(\d+)", name[: seed_match.start()] if seed_match else name)
    if not seed_match or not budget_matches:
        return None, None
    return int(budget_matches[-1]), int(seed_match.group(1))


def group_payload(run_dir: Path, family: str, group: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    eval_dir = run_dir / "posthoc_eval_internal"
    if family == "split_group":
        anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        candidate = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    else:
        anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
        candidate = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    return ((anchor or {}).get("groups") or {}).get(group), ((candidate or {}).get("groups") or {}).get(group)


def condition_map(payload: dict[str, Any] | None, metric: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    if not payload:
        return out
    for row in payload.get("condition_metrics") or []:
        value = row.get(metric)
        if value is None:
            continue
        try:
            out[(str(row.get("dataset")), str(row.get("condition")))] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def collect_records(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows = []
    records = []
    for run_dir in run_dirs:
        budget, seed = parse_budget_seed(run_dir.name)
        train_exit = read_exit(run_dir / "EXIT_CODE")
        posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
        complete = train_exit == 0 and posthoc_exit == 0
        run_rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "budget": budget,
                "seed": seed,
                "train_exit": train_exit,
                "posthoc_exit": posthoc_exit,
                "complete": complete,
            }
        )
        if not complete:
            continue
        for label, (family, group) in GROUPS.items():
            anchor, candidate = group_payload(run_dir, family, group)
            for metric in ["pearson_pert", "test_mmd"]:
                amap = condition_map(anchor, metric)
                cmap = condition_map(candidate, metric)
                for key in sorted(set(amap) & set(cmap)):
                    records.append(
                        {
                            "run_name": run_dir.name,
                            "budget": int(budget),
                            "seed": int(seed),
                            "group": label,
                            "metric": metric,
                            "dataset": key[0],
                            "condition": key[1],
                            "delta": float(cmap[key] - amap[key]),
                        }
                    )
    return run_rows, records


def finite_mean(values: list[float]) -> float | None:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def condition_identity(records: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    checks = []
    for group in sorted(GROUPS):
        subset = [r for r in records if r["group"] == group and r["metric"] == "pearson_pert"]
        by_seed_budget: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
        for row in subset:
            by_seed_budget[(int(row["seed"]), int(row["budget"]))].add((str(row["dataset"]), str(row["condition"])))
        budgets = sorted({k[1] for k in by_seed_budget})
        seeds = sorted({k[0] for k in by_seed_budget})
        for seed in seeds:
            sets = {budget: by_seed_budget.get((seed, budget), set()) for budget in budgets}
            sizes = {budget: len(vals) for budget, vals in sets.items()}
            ok = len({frozenset(v) for v in sets.values()}) == 1 and all(sizes.values())
            if not ok:
                reasons.append(f"{group}:seed{seed}:condition_identity_mismatch:{sizes}")
            checks.append({"group": group, "seed": seed, "status": "ok" if ok else "fail", "sizes": sizes})
        if len(budgets) == 1 and len(seeds) > 1:
            budget = budgets[0]
            seed_sets = {seed: by_seed_budget.get((seed, budget), set()) for seed in seeds}
            seed_sizes = {seed: len(vals) for seed, vals in seed_sets.items()}
            ok = len({frozenset(v) for v in seed_sets.values()}) == 1 and all(seed_sizes.values())
            if not ok:
                reasons.append(f"{group}:budget{budget}:cross_seed_condition_identity_mismatch:{seed_sizes}")
            checks.append(
                {
                    "group": group,
                    "budget": budget,
                    "status": "ok" if ok else "fail",
                    "cross_seed_sizes": seed_sizes,
                    "mode": "single_budget_cross_seed",
                }
            )
    return {"status": "ok" if not reasons else "fail", "reasons": reasons, "checks": checks}


def budget_dataset_control(records: list[dict[str, Any]], *, group: str, metric: str) -> dict[str, Any]:
    subset = [r for r in records if r["group"] == group and r["metric"] == metric]
    by_budget: dict[int, list[float]] = defaultdict(list)
    by_dataset_budget: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in subset:
        by_budget[int(row["budget"])].append(float(row["delta"]))
        by_dataset_budget[(str(row["dataset"]), int(row["budget"]))].append(float(row["delta"]))
    raw = {b: finite_mean(vals) for b, vals in sorted(by_budget.items())}
    ds_rows = []
    for ds in sorted({k[0] for k in by_dataset_budget}):
        means = {b: finite_mean(by_dataset_budget.get((ds, b), [])) for b in [64, 128, 256]}
        if all(means[b] is not None for b in [64, 128, 256]):
            ds_mean = float(np.mean([means[b] for b in [64, 128, 256]]))
            ds_rows.append({"dataset": ds, "means": means, "demeaned": {b: means[b] - ds_mean for b in [64, 128, 256]}})
    controlled = {b: finite_mean([row["demeaned"][b] for row in ds_rows]) for b in [64, 128, 256]}
    return {"raw_budget_means": raw, "dataset_demeaned_budget_means": controlled, "n_datasets": len(ds_rows)}


def budget_label_shuffle(records: list[dict[str, Any]], *, group: str, metric: str, n_perm: int, seed: int) -> dict[str, Any]:
    subset = [r for r in records if r["group"] == group and r["metric"] == metric]
    keyed: dict[tuple[int, str, str], dict[int, float]] = defaultdict(dict)
    for row in subset:
        keyed[(int(row["seed"]), str(row["dataset"]), str(row["condition"]))][int(row["budget"])] = float(row["delta"])
    triples = [v for v in keyed.values() if all(b in v for b in [64, 128, 256])]
    if not triples:
        return {"n_complete_triples": 0, "observed_range": None, "perm_p_ge_range": None}
    obs = {b: float(np.mean([v[b] for v in triples])) for b in [64, 128, 256]}
    obs_range = max(obs.values()) - min(obs.values())
    rng = np.random.default_rng(seed)
    budgets = np.asarray([64, 128, 256], dtype=np.int64)
    vals = np.asarray([[v[64], v[128], v[256]] for v in triples], dtype=np.float64)
    perm_ranges = []
    for _ in range(int(n_perm)):
        accum = {64: [], 128: [], 256: []}
        for row in vals:
            perm = rng.permutation(3)
            for out_budget, src_idx in zip(budgets, perm):
                accum[int(out_budget)].append(float(row[src_idx]))
        means = [float(np.mean(accum[int(b)])) for b in budgets]
        perm_ranges.append(max(means) - min(means))
    return {
        "n_complete_triples": len(triples),
        "observed_budget_means": obs,
        "observed_range": float(obs_range),
        "perm_p_ge_range": float((np.asarray(perm_ranges) >= obs_range).mean()),
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM True Cell-Count Nested Controls Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only control gate for nested true cell-count matrix.",
        "- Reads train-only/internal posthoc JSONs only.",
        "- Does not read canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Completeness",
        "",
        f"- complete runs: `{payload['complete_runs']}/{payload['expected_runs']}`",
        "",
        "## Condition Identity",
        "",
        f"- status: `{payload['condition_identity']['status']}`",
        f"- reasons: `{payload['condition_identity']['reasons'] or 'none'}`",
        "",
        "## Controls",
        "",
        "| group | metric | raw budget means | dataset-demeaned means | shuffle p(range>=obs) | triples |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in payload["control_rows"]:
        lines.append(
            f"| `{row['group']}` | `{row['metric']}` | `{row['dataset_control']['raw_budget_means']}` | "
            f"`{row['dataset_control']['dataset_demeaned_budget_means']}` | "
            f"{row['budget_shuffle']['perm_p_ge_range']} | {row['budget_shuffle']['n_complete_triples']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{payload['reasons'] or 'none'}`",
            f"- next action: `{payload['next_action']}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    run_dirs = sorted(p for p in RUN_ROOT.iterdir() if p.is_dir() and (p / "RUN_STATUS.md").is_file()) if RUN_ROOT.exists() else []
    run_rows, records = collect_records(run_dirs)
    complete_runs = sum(1 for r in run_rows if r["complete"])
    identity = condition_identity(records)
    control_rows = []
    for group in GROUPS:
        for metric in ["pearson_pert", "test_mmd"]:
            control_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "dataset_control": budget_dataset_control(records, group=group, metric=metric),
                    "budget_shuffle": budget_label_shuffle(records, group=group, metric=metric, n_perm=args.n_perm, seed=20260624 + len(control_rows)),
                }
            )
    reasons = []
    if complete_runs < EXPECTED_RUNS:
        reasons.append(f"matrix_incomplete:{complete_runs}_of_{EXPECTED_RUNS}")
    if identity["status"] != "ok":
        reasons.append("condition_identity_not_fixed")
    status = "nested_controls_pending" if complete_runs < EXPECTED_RUNS else ("nested_controls_pass_no_gpu" if not reasons else "nested_controls_fail_no_gpu")
    payload = {
        "status": status,
        "run_root": str(RUN_ROOT),
        "run_rows": run_rows,
        "complete_runs": complete_runs,
        "expected_runs": EXPECTED_RUNS,
        "condition_identity": identity,
        "control_rows": control_rows,
        "reasons": reasons,
        "gpu_authorized": False,
        "next_action": "wait_for_nested_matrix_completion" if complete_runs < 9 else ("interpret controls with nested matrix decision before no-harm" if not reasons else "fix or close nested matrix control failure"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
