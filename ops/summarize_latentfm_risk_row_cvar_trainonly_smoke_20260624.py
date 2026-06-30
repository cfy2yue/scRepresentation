#!/usr/bin/env python3
"""Summarize the capped train-only risk-row CVaR smoke without reading held-out metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_risk_row_cvar_allrisk_w020_2k_seed42"
RUN_DIR = ROOT / "runs/latentfm_risk_row_cvar_trainonly_20260624" / RUN_NAME
LOG = ROOT / "logs/latentfm_risk_row_cvar_trainonly_20260624" / RUN_NAME / "launcher.log"
OUT_DIR = ROOT / "CoupledFM/output/latentfm_runs/risk_row_cvar_trainonly_20260624" / RUN_NAME
OUT_JSON = ROOT / "reports/latentfm_risk_row_cvar_trainonly_smoke_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_ROW_CVAR_TRAINONLY_SMOKE_DECISION_20260624.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_log(text: str) -> dict[str, Any]:
    obs_values = [int(x) for x in re.findall(r"risk_row_obs=(\d+)", text)]
    apply_values = [int(x) for x in re.findall(r"risk_row_apply=(\d+)", text)]
    weights = [float(x) for x in re.findall(r"avg_risk_row_cvar_w=([0-9.eE+-]+)", text)]
    steps = [int(x) for x in re.findall(r"step=(\d+)\s+epoch=", text)]
    return {
        "max_step_logged": max(steps) if steps else 0,
        "max_risk_row_obs": max(obs_values) if obs_values else 0,
        "max_risk_row_apply": max(apply_values) if apply_values else 0,
        "max_avg_risk_row_cvar_w": max(weights) if weights else 0.0,
        "contains_iid_eval": "[IID eval]" in text,
        "contains_ood_eval": "[OOD eval]" in text or "OOD evaluation —" in text,
        "contains_epoch_eval_skip": "train_eval_enabled=False; skipped epoch IID eval" in text,
        "contains_final_eval_skip": "Final IID/OOD evaluation skipped because train_eval_enabled=False" in text,
    }


def main() -> int:
    log_text = read_text(LOG)
    config = load_json(OUT_DIR / "config.json")
    exit_path = RUN_DIR / f"{RUN_NAME}.EXIT_CODE"
    exit_code = read_text(exit_path).strip() if exit_path.exists() else None
    parsed = parse_log(log_text)

    checks = {
        "exit_code_zero": exit_code == "0",
        "config_train_eval_disabled": config.get("train_eval_enabled") is False,
        "config_total_steps_capped": int(config.get("total_steps") or 0) <= 2000,
        "config_risk_row_weight_nonzero": float(config.get("risk_row_cvar_loss_weight") or 0.0) > 0,
        "no_iid_or_ood_eval_logged": not parsed["contains_iid_eval"] and not parsed["contains_ood_eval"],
        "epoch_eval_skip_logged": bool(parsed["contains_epoch_eval_skip"]),
        "final_eval_skip_logged": bool(parsed["contains_final_eval_skip"]),
        "risk_row_observed": int(parsed["max_risk_row_obs"]) > 0,
        "risk_row_applied": int(parsed["max_risk_row_apply"]) > 0
        or float(parsed["max_avg_risk_row_cvar_w"]) > 0,
        "latest_checkpoint_exists": (OUT_DIR / "latest.pt").exists(),
    }

    if exit_code is None:
        status = "risk_row_cvar_trainonly_smoke_running_no_decision"
    elif all(checks.values()):
        status = "risk_row_cvar_trainonly_smoke_mechanism_activated_no_promotion"
    else:
        status = "risk_row_cvar_trainonly_smoke_fail_or_no_mechanism_no_promotion"

    payload = {
        "status": status,
        "run_name": RUN_NAME,
        "exit_code": exit_code,
        "paths": {
            "run_dir": str(RUN_DIR),
            "log": str(LOG),
            "out_dir": str(OUT_DIR),
            "config": str(OUT_DIR / "config.json"),
            "latest": str(OUT_DIR / "latest.pt"),
        },
        "boundary": {
            "train_only": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "posthoc_read": False,
        },
        "checks": checks,
        "log_signals": parsed,
        "decision": {
            "gpu_extension_authorized": False,
            "canonical_noharm_authorized": False,
            "promotion_authorized": False,
            "next_if_mechanism_activated": (
                "Design a separate internal train-only posthoc/no-harm gate; do not use "
                "canonical metrics or multi/query without a frozen route decision."
            ),
            "next_if_fail": "Close or mutate the risk-row mechanism; record negative evidence.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Row CVaR Train-Only Smoke Decision",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Reads only run markers, train log, config, and checkpoint existence.",
        "- Does not read canonical metrics, canonical multi, Track C query, or posthoc metrics.",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, value in checks.items():
        lines.append(f"| `{name}` | `{bool(value)}` |")
    lines.extend(
        [
            "",
            "## Log Signals",
            "",
            f"- max step logged: `{parsed['max_step_logged']}`",
            f"- max risk_row_obs: `{parsed['max_risk_row_obs']}`",
            f"- max risk_row_apply: `{parsed['max_risk_row_apply']}`",
            f"- max avg_risk_row_cvar_w: `{parsed['max_avg_risk_row_cvar_w']}`",
            "",
            "## Decision",
            "",
            "- No promotion, canonical no-harm, or extension is authorized by this summary alone.",
            "- A pass means only that the bounded train-only mechanism activated cleanly.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
