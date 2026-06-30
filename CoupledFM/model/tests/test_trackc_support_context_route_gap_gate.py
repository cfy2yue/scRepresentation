"""CPU-only regression tests for Track C support-context route-gap gating."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = PROJECT_ROOT / "ops" / "evaluate_latentfm_trackc_support_context_route_gap_gate_20260622.py"


def load_gate_module():
    spec = importlib.util.spec_from_file_location("trackc_support_context_route_gap_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def support_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"groups": {"test": {"condition_metrics": rows}}}


def row(dataset: str, condition: str, pp: float) -> dict[str, object]:
    return {"dataset": dataset, "condition": condition, "pearson_pert": pp}


class TestTrackCSupportContextRouteGapGate(unittest.TestCase):
    def setUp(self):
        self.gate = load_gate_module()

    def test_missing_support_conditions_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            readout = root / "readout.json"
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
            readout.write_text(
                json.dumps(
                    {
                        "condition_rows": [
                            {"dataset": "NormanWeissman2019_filtered", "condition": "N1+N2", "support_selected_route": 0.6},
                            {"dataset": "Wessels", "condition": "W1+W2", "support_selected_route": 1.0},
                            {"dataset": "Wessels", "condition": "W3+W4", "support_selected_route": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            _, rows, reasons = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                readout_json=readout,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows), reasons)
            self.assertEqual(decision["status"], "route_gap_gate_missing_required_metrics")
            self.assertTrue(any("missing_anchor_support_conditions_1" in reason for reason in decision["reasons"]))
            self.assertTrue(any("missing_candidate_support_conditions_1" in reason for reason in decision["reasons"]))

    def test_full_coverage_material_wessels_closure_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.json"
            candidate = root / "candidate.json"
            readout = root / "readout.json"
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
            readout.write_text(
                json.dumps(
                    {
                        "condition_rows": [
                            {"dataset": "NormanWeissman2019_filtered", "condition": "N1+N2", "support_selected_route": 0.6},
                            {"dataset": "Wessels", "condition": "W1+W2", "support_selected_route": 1.0},
                            {"dataset": "Wessels", "condition": "W3+W4", "support_selected_route": 1.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            _, rows, reasons = self.gate.paired_condition_rows(
                anchor_json=anchor,
                candidate_json=candidate,
                readout_json=readout,
            )
            decision = self.gate.evaluate_gate(self.gate.summarize(rows), reasons)
            self.assertEqual(decision["status"], "route_gap_gate_pass")


if __name__ == "__main__":
    unittest.main()
