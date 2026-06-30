"""CPU-only regression tests for Track C residual-operator route-gap gating."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = PROJECT_ROOT / "ops" / "evaluate_latentfm_trackc_residual_operator_route_gap_gate_20260623.py"
SUPPORT_FILM_GATE_PATH = PROJECT_ROOT / "ops" / "evaluate_latentfm_trackc_support_film_route_gap_gate_20260623.py"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("trackc_residual_route_gap_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_support_film_gate_module():
    spec = importlib.util.spec_from_file_location("trackc_support_film_route_gap_gate", SUPPORT_FILM_GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUPPORT_FILM_GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def support_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"groups": {"test": {"condition_metrics": rows}}}


def row(dataset: str, condition: str, pp: float) -> dict[str, object]:
    return {"dataset": dataset, "condition": condition, "pearson_pert": pp}


def cpu_gate_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "decision": {"status": "residual_operator_cpu_gate_pass_authorize_one_capped_gpu_smoke"},
        "eval_rows": rows,
    }


def support_film_cpu_gate_payload() -> dict[str, object]:
    return {
        "status": "trackc_alternative_support_conditioning_cpu_gate_pass_authorize_one_capped_gpu_smoke",
        "real": {
            "decision": {"gpu_authorization": "one_capped_trackc_support_only_smoke"},
            "dataset_breakdown": [
                {
                    "dataset": "NormanWeissman2019_filtered",
                    "support_selected_route": 0.50,
                    "candidate": 0.60,
                    "n_conditions": 1,
                },
                {
                    "dataset": "Wessels",
                    "support_selected_route": 0.20,
                    "candidate": 0.70,
                    "n_conditions": 2,
                },
            ],
        },
    }


class TestTrackCResidualOperatorRouteGapGate(unittest.TestCase):
    def setUp(self):
        self.gate = load_gate_module()

    def test_missing_support_conditions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            cpu_gate = root / "cpu_gate.json"
            anchor.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.5),
                            row("Wessels", "W1+W2", 0.0),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.5),
                            row("Wessels", "W1+W2", 0.1),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            cpu_gate.write_text(
                json.dumps(
                    cpu_gate_payload(
                        [
                            {"dataset": "NormanWeissman2019_filtered", "condition": "N1+N2", "candidate": 0.6},
                            {"dataset": "Wessels", "condition": "W1+W2", "candidate": 1.0},
                            {"dataset": "Wessels", "condition": "W3+W4", "candidate": 1.0},
                        ]
                    )
                ),
                encoding="utf-8",
            )
            _, rows, reasons = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                cpu_gate_json=cpu_gate,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows), reasons)
            self.assertEqual(decision["status"], "residual_route_gap_gate_missing_required_metrics")
            self.assertIn("missing_anchor_support_conditions_1", decision["reasons"])
            self.assertIn("missing_candidate_support_conditions_1", decision["reasons"])

    def test_full_coverage_material_wessels_closure_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            cpu_gate = root / "cpu_gate.json"
            anchor.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.5),
                            row("Wessels", "W1+W2", 0.0),
                            row("Wessels", "W3+W4", 0.2),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.5),
                            row("Wessels", "W1+W2", 0.1),
                            row("Wessels", "W3+W4", 0.3),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            cpu_gate.write_text(
                json.dumps(
                    cpu_gate_payload(
                        [
                            {"dataset": "NormanWeissman2019_filtered", "condition": "N1+N2", "candidate": 0.6},
                            {"dataset": "Wessels", "condition": "W1+W2", "candidate": 1.0},
                            {"dataset": "Wessels", "condition": "W3+W4", "candidate": 1.0},
                        ]
                    )
                ),
                encoding="utf-8",
            )
            _, rows, reasons = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                cpu_gate_json=cpu_gate,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows), reasons)
            self.assertEqual(decision["status"], "residual_route_gap_gate_pass")


class TestTrackCSupportFilmRouteGapGate(unittest.TestCase):
    def setUp(self):
        self.gate = load_support_film_gate_module()

    def test_missing_support_conditions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            cpu_gate = root / "cpu_gate.json"
            anchor.write_text(
                json.dumps(support_payload([row("NormanWeissman2019_filtered", "N1+N2", 0.50)])),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(support_payload([row("NormanWeissman2019_filtered", "N1+N2", 0.50)])),
                encoding="utf-8",
            )
            cpu_gate.write_text(json.dumps(support_film_cpu_gate_payload()), encoding="utf-8")
            _, rows, reasons, targets = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                cpu_gate_json=cpu_gate,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows, targets), reasons)
            self.assertEqual(decision["status"], "support_film_route_gap_gate_missing_required_metrics")
            self.assertIn("missing_wessels_support_film_route_rows", decision["reasons"])

    def test_full_coverage_material_wessels_closure_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            cpu_gate = root / "cpu_gate.json"
            anchor.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.50),
                            row("Wessels", "W1+W2", 0.20),
                            row("Wessels", "W3+W4", 0.30),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.50),
                            row("Wessels", "W1+W2", 0.24),
                            row("Wessels", "W3+W4", 0.34),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            cpu_gate.write_text(json.dumps(support_film_cpu_gate_payload()), encoding="utf-8")
            _, rows, reasons, targets = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                cpu_gate_json=cpu_gate,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows, targets), reasons)
            self.assertEqual(decision["status"], "support_film_route_gap_gate_pass")

    def test_full_coverage_low_wessels_closure_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            cpu_gate = root / "cpu_gate.json"
            anchor.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.50),
                            row("Wessels", "W1+W2", 0.20),
                            row("Wessels", "W3+W4", 0.30),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    support_payload(
                        [
                            row("NormanWeissman2019_filtered", "N1+N2", 0.50),
                            row("Wessels", "W1+W2", 0.21),
                            row("Wessels", "W3+W4", 0.31),
                        ]
                    )
                ),
                encoding="utf-8",
            )
            cpu_gate.write_text(json.dumps(support_film_cpu_gate_payload()), encoding="utf-8")
            _, rows, reasons, targets = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                cpu_gate_json=cpu_gate,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows, targets), reasons)
            self.assertEqual(decision["status"], "support_film_route_gap_gate_fail_close_branch")
            self.assertIn("wessels_support_pp_delta_below_0p02", decision["reasons"])
            self.assertIn("wessels_support_film_route_gap_closure_below_0p05", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
