#!/usr/bin/env python3
"""CPU gate for routed additive-prior strategies from prior-correction rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_IN = ROOT / "reports/latentfm_prior_correction_shifted_mmd_gate_scf_inject_20260621.csv"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_routed_prior_cpu_gate_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_ROUTED_PRIOR_CPU_GATE_20260621.md"


@dataclass(frozen=True)
class Router:
    name: str
    description: str
    k: int
    choose_alpha: Callable[[pd.Series], float]


def fnum(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def routers() -> list[Router]:
    return [
        Router(
            name="off_k5",
            description="No correction, alpha=0; baseline.",
            k=5,
            choose_alpha=lambda r: 0.0,
        ),
        Router(
            name="norman_alpha1_k5",
            description="Use full additive prior only for Norman rows with available prior; all other datasets off.",
            k=5,
            choose_alpha=lambda r: 1.0
            if r["dataset"] == "NormanWeissman2019_filtered" and float(r["prior_available"]) > 0
            else 0.0,
        ),
        Router(
            name="norman_alpha075_k5",
            description="Use cautious additive prior alpha=0.75 for Norman rows with available prior; all other datasets off.",
            k=5,
            choose_alpha=lambda r: 0.75
            if r["dataset"] == "NormanWeissman2019_filtered" and float(r["prior_available"]) > 0
            else 0.0,
        ),
        Router(
            name="low_missing_alpha05_k5",
            description="Use alpha=0.5 only when all components have direct/KNN prior coverage.",
            k=5,
            choose_alpha=lambda r: 0.5
            if float(r["prior_available"]) > 0 and float(r["n_missing"]) <= 0
            else 0.0,
        ),
        Router(
            name="norman_low_missing_alpha1_k5",
            description="Use alpha=1 only for Norman rows with complete direct/KNN prior coverage.",
            k=5,
            choose_alpha=lambda r: 1.0
            if (
                r["dataset"] == "NormanWeissman2019_filtered"
                and float(r["prior_available"]) > 0
                and float(r["n_missing"]) <= 0
            )
            else 0.0,
        ),
        Router(
            name="norman_alpha1_k10",
            description="Same as norman_alpha1_k5 but using KNN k=10.",
            k=10,
            choose_alpha=lambda r: 1.0
            if r["dataset"] == "NormanWeissman2019_filtered" and float(r["prior_available"]) > 0
            else 0.0,
        ),
    ]


def select_router_rows(raw: pd.DataFrame, router: Router) -> pd.DataFrame:
    base = raw[(raw["k"].astype(int) == int(router.k)) & (raw["alpha"].astype(float) == 0.0)].copy()
    rows: list[pd.Series] = []
    lookup = raw[raw["k"].astype(int) == int(router.k)].copy()
    lookup["_row_key"] = list(zip(lookup["dataset"], lookup["group"], lookup["condition"], lookup["alpha"].astype(float)))
    by_key = {key: row for key, row in lookup.set_index("_row_key").iterrows()}
    for _, row in base.iterrows():
        alpha = float(router.choose_alpha(row))
        key = (row["dataset"], row["group"], row["condition"], alpha)
        chosen = by_key.get(key)
        if chosen is None:
            chosen = row
            alpha = 0.0
        out = chosen.copy()
        out["router"] = router.name
        out["router_description"] = router.description
        out["router_requested_alpha"] = float(router.choose_alpha(row))
        out["router_effective_alpha"] = float(alpha)
        out["router_used_prior"] = float(alpha) > 0
        out["base_pp"] = float(row["pp"])
        out["base_pc"] = float(row["pc"])
        out["base_direct"] = float(row["direct"])
        out["base_mmd_clamped"] = float(row["mmd_clamped"])
        out["delta_pp"] = float(out["pp"] - row["pp"])
        out["delta_pc"] = float(out["pc"] - row["pc"])
        out["delta_direct"] = float(out["direct"] - row["direct"])
        out["delta_mmd_clamped"] = float(out["mmd_clamped"] - row["mmd_clamped"])
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_conditions": int(len(g)),
                "n_datasets": int(g["dataset"].nunique()),
                "used_prior_rate": float(g["router_used_prior"].mean()),
                "base_pp": float(g["base_pp"].mean()),
                "routed_pp": float(g["pp"].mean()),
                "delta_pp": float(g["delta_pp"].mean()),
                "base_pc": float(g["base_pc"].mean()),
                "routed_pc": float(g["pc"].mean()),
                "delta_pc": float(g["delta_pc"].mean()),
                "base_mmd_clamped": float(g["base_mmd_clamped"].mean()),
                "routed_mmd_clamped": float(g["mmd_clamped"].mean()),
                "delta_mmd_clamped": float(g["delta_mmd_clamped"].mean()),
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def equal_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    per_ds = summarize(df, ["router", "group", "dataset"])
    rows: list[dict[str, Any]] = []
    for (router, group), g in per_ds.groupby(["router", "group"], dropna=False):
        rows.append(
            {
                "router": router,
                "group": group,
                "n_datasets": int(g["dataset"].nunique()),
                "n_conditions": int(g["n_conditions"].sum()),
                "used_prior_rate": float(g["used_prior_rate"].mean()),
                "base_pp": float(g["base_pp"].mean()),
                "routed_pp": float(g["routed_pp"].mean()),
                "delta_pp": float(g["delta_pp"].mean()),
                "base_pc": float(g["base_pc"].mean()),
                "routed_pc": float(g["routed_pc"].mean()),
                "delta_pc": float(g["delta_pc"].mean()),
                "base_mmd_clamped": float(g["base_mmd_clamped"].mean()),
                "routed_mmd_clamped": float(g["routed_mmd_clamped"].mean()),
                "delta_mmd_clamped": float(g["delta_mmd_clamped"].mean()),
            }
        )
    return pd.DataFrame(rows)


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def render_md(
    *,
    router_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    routers_used: list[Router],
    out_json: Path,
) -> str:
    lines = [
        "# LatentFM Routed Additive Prior CPU Gate",
        "",
        "Status: `complete_cpu_gate`",
        "",
        "This is a CPU-only evaluation over existing shifted-MMD prior-correction rows.",
        "Routers are predeclared feature rules; no router is trained on held-out outcomes.",
        "",
        f"JSON: `{out_json}`",
        "",
        "## Router Definitions",
        "",
        "| router | k | definition |",
        "|---|---:|---|",
    ]
    for r in routers_used:
        lines.append(f"| `{r.name}` | {r.k} | {r.description} |")

    lines += [
        "",
        "## Equal-Dataset Summary",
        "",
        "| router | group | n ds | used prior | pp delta | pc delta | MMD delta | routed pp | routed MMD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in router_summary.sort_values(["group", "router"]).to_dict(orient="records"):
        lines.append(
            f"| `{row['router']}` | `{row['group']}` | {int(row['n_datasets'])} | "
            f"{fnum(row['used_prior_rate'])} | {fnum(row['delta_pp'])} | "
            f"{fnum(row['delta_pc'])} | {fnum(row['delta_mmd_clamped'])} | "
            f"{fnum(row['routed_pp'])} | {fnum(row['routed_mmd_clamped'])} |"
        )

    lines += [
        "",
        "## Focus Dataset Summary",
        "",
        "| router | dataset | group | n | used prior | pp delta | pc delta | MMD delta | routed pp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    focus = {"NormanWeissman2019_filtered", "Wessels", "GasperiniShendure2019_lowMOI"}
    for row in dataset_summary[dataset_summary["dataset"].isin(focus)].sort_values(
        ["group", "dataset", "router"]
    ).to_dict(orient="records"):
        ds = str(row["dataset"]).replace("NormanWeissman2019_filtered", "Norman").replace(
            "GasperiniShendure2019_lowMOI", "Gasperini"
        )
        lines.append(
            f"| `{row['router']}` | `{ds}` | `{row['group']}` | {int(row['n_conditions'])} | "
            f"{fnum(row['used_prior_rate'])} | {fnum(row['delta_pp'])} | "
            f"{fnum(row['delta_pc'])} | {fnum(row['delta_mmd_clamped'])} | "
            f"{fnum(row['routed_pp'])} |"
        )

    lines += [
        "",
        "## Gate Interpretation",
        "",
        "- Passing this CPU gate would require a deployable router that improves",
        "  Norman/unseen2 while keeping Wessels/Gasperini off or non-harmed and",
        "  avoiding shifted-MMD harm.",
        "- If the best deployable router still only recovers Norman while leaving",
        "  Wessels/Gasperini unchanged, it is a mechanism signal but not a broad",
        "  LatentFM promotion path.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    all_rows = pd.concat([select_router_rows(raw, r) for r in routers()], ignore_index=True)
    router_summary = equal_dataset_summary(all_rows)
    dataset_summary = summarize(all_rows, ["router", "dataset", "group"])
    payload = {
        "input": str(args.input),
        "router_definitions": [
            {"name": r.name, "k": r.k, "description": r.description} for r in routers()
        ],
        "n_condition_router_rows": int(len(all_rows)),
        "equal_dataset_summary": to_records(router_summary),
        "dataset_summary": to_records(dataset_summary),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(
        render_md(
            router_summary=router_summary,
            dataset_summary=dataset_summary,
            routers_used=routers(),
            out_json=args.out_json,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
