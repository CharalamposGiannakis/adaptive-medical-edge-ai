"""F007 Phase A1 checks; synthetic requests never open held-out images."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from experiments.F006a_2.run_analysis import quality_from_counts
from experiments.F006b import controller
from experiments.F007 import phase_a1 as f007
from src.f002_protocol import recording_units
from src.f007_protocol import (load_protocol, require_test_authorization,
                               validate_structure, verify_inputs)


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "experiments/F006a_2/quality_screen.csv"
BASELINE = quality_from_counts(168, 10, 19, 57)


def points_from_screen() -> tuple[controller.OperatingPoint, ...]:
    with SCREEN.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(controller.OperatingPoint(
        row["configuration"], float(row["width"]), int(row["k"]),
        int(row["flops"]), quality_from_counts(*(int(row[name]) for name in
        ("tn", "fp", "fn", "tp"))), float(row["macro_f1"])) for row in rows)


def synthetic_row(exam: str, target: int, *, zone="z1", group="G",
                  partition="validation") -> dict:
    return {"exam_id": exam, "zone_id": zone, "binary_target": str(target),
            "split_group": group, "partition": partition, "device": "synthetic"}


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.points = points_from_screen()

    def test_exact_headroom_and_five_state_six_system_table(self):
        expected = {
            "A": [None, "mobilenet_v2_k16", "mobilenet_v2_k16", "mobilenet_v2_k16", "mobilenet_v2_k16"],
            "B": [None, None, None, None, "original_tinyvit_k16"],
            "C": [None, None, "w0.5_k8", "w0.5_k8", "w0.5_k8"],
            "D": [None, "w0.5_k2", "w0.5_k2", "w0.5_k2", "w0.5_k2"],
            "E": [None, "w0.5_k2", "w0.75_k8", "w0.75_k8", "w1.0_k16"],
            "F": [None, "w0.5_k2", "w1.0_k8", "w1.0_k12", "w1.0_k16"],
        }
        states = self.protocol["resource_states"]
        self.assertEqual(tuple(expected), tuple(self.protocol["systems"]))
        for system in expected:
            plans = [f007.route(system, b, self.protocol, self.points, BASELINE)
                     for b in states]
            self.assertEqual([plan.configuration for plan in plans], expected[system])
            for b, plan in zip(states, plans):
                self.assertEqual(plan.permitted_flops,
                                 Fraction(b) * Fraction(226556754193, 10))
                self.assertLessEqual(plan.full_flops, plan.permitted_flops)
        self.assertEqual(f007.route("F", "0.50", self.protocol, self.points,
                                    BASELINE).configuration, "w1.0_k8")
        self.assertEqual(controller.select(self.points, BASELINE, "0.50").point.name,
                         "w0.5_k2")

    def test_mixed_schedule_is_label_and_order_independent_with_aliases(self):
        rows = [synthetic_row("A", 1), synthetic_row("B", 1),
                synthetic_row("C", 0, group="H"),
                synthetic_row("D", 1, group="I"),
                synthetic_row("E", 0, group="J"),
                synthetic_row("F", 1, group="K")]
        edges = [{"exam_a": "A", "zone_a": "z1", "exam_b": "B", "zone_b": "z1"},
                 {"exam_a": "B", "zone_a": "z1", "exam_b": "Z", "zone_b": "z1"}]
        unavailable = [{"exam_id": "Z", "zone_id": "z1",
                        "exclusion_reason": "t004:unavailable_label"}]
        unit_of = recording_units({"validation": rows}, edges, unavailable)
        by_unit, assignments = f007.mixed_assignments(rows, unit_of, self.protocol)
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(sorted(by_unit.values()), sorted(map(Fraction,
                         self.protocol["resource_states"])))
        changed_labels = [dict(row, binary_target=str(1 - int(row["binary_target"])))
                          for row in rows]
        self.assertEqual(f007.mixed_assignments(changed_labels, unit_of,
                         self.protocol)[0], by_unit)
        self.assertEqual(f007.mixed_assignments(list(reversed(rows)), unit_of,
                         self.protocol)[0], by_unit)
        self.assertEqual([by_unit[unit_of[(row["exam_id"], row["zone_id"])]]
                          for row in rows], assignments)

    def test_protocol_and_frozen_identity_fail_closed(self):
        bad = copy.deepcopy(self.protocol)
        bad["resource_states"][2] = "0.49"
        with self.assertRaises(ValueError):
            validate_structure(bad)
        bad = copy.deepcopy(self.protocol)
        bad["systems"]["A"]["flops"] += 1
        with self.assertRaisesRegex(ValueError, "static F007 system A"):
            validate_structure(bad)
        bad = copy.deepcopy(self.protocol)
        bad["frozen_t006"]["canonical_protocol_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_inputs(bad)
        bad = copy.deepcopy(self.protocol)
        bad["checkpoints"]["f002b_mobilenetv2"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            verify_inputs(bad, check_checkpoints=True)
        bad = copy.deepcopy(self.protocol)
        bad["checkpoints"]["f002c_original_tinyvit"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            verify_inputs(bad, check_checkpoints=True)
        bad = copy.deepcopy(self.protocol)
        bad["checkpoints"]["f005b_nested_tinyvit"]["importance_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "importance.pt hash mismatch"):
            verify_inputs(bad, check_checkpoints=True)


class RequestAndGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.points = points_from_screen()

    def _evaluate(self, rows, unit_of, systems, states, predictor, partition="validation"):
        with patch.object(f007.controller, "load_policy_points", return_value=self.points), \
                patch.object(f007.f006a_2, "read_baseline", return_value=BASELINE):
            return f007.evaluate_requests(rows, unit_of, systems, states,
                                          self.protocol, predictor, partition=partition)

    def test_unauthorized_test_rejected_before_predictor_or_image_loader(self):
        row = synthetic_row("T", 1, partition="test")
        called = []
        def predictor(*args):
            called.append(True)
            raise AssertionError("Predictor must not be called")
        with self.assertRaises(PermissionError):
            self._evaluate([row], {("T", "z1"): "singleton:T/z1"},
                           ("A",), [Fraction(1)], predictor, partition="test")
        self.assertEqual(called, [])
        with self.assertRaises(PermissionError):
            f007.CanonicalModels(self.protocol, partition="test")
        forged = copy.deepcopy(self.protocol)
        forged["status"] = "accepted_final_protocol"
        with self.assertRaises(PermissionError):
            require_test_authorization(forged)
        forged["authorizes_test_inference"] = True
        with self.assertRaises(PermissionError):
            require_test_authorization(forged)  # no committed approval provenance
        forged["acceptance_provenance"] = {"protocol_commit": "0" * 40}
        with self.assertRaises(PermissionError):
            require_test_authorization(forged)  # an in-memory claim is not a committed receipt

    def test_mixed_request_order_and_state_match_across_six_systems(self):
        rows = [synthetic_row("A", 1), synthetic_row("B", 1),
                synthetic_row("C", 0, group="H"),
                synthetic_row("D", 1, group="I")]
        unit_of = {("A", "z1"): "shared:synthetic",
                   ("B", "z1"): "shared:synthetic",
                   ("C", "z1"): "singleton:C/z1",
                   ("D", "z1"): "singleton:D/z1"}
        _, states = f007.mixed_assignments(rows, unit_of, self.protocol)
        records = self._evaluate(rows, unit_of, tuple(self.protocol["systems"]),
                                 states, lambda *args: (0.5, {}))
        requested = [(row["exam_id"], row["zone_id"]) for row in rows]
        for system in self.protocol["systems"]:
            selected = [record for record in records if record["system_id"] == system]
            self.assertEqual([(record["exam_id"], record["zone_id"])
                              for record in selected], requested)
            self.assertEqual([record["resource_state"] for record in selected],
                             [f007.state_string(state) for state in states])
        self.assertEqual(states[0], states[1])

    def test_duplicate_probability_aggregation_full_cost_and_missing_class(self):
        rows = [synthetic_row("A", 1), synthetic_row("B", 1),
                synthetic_row("C", 0, group="H")]
        unit_of = {("A", "z1"): "shared:synthetic",
                   ("B", "z1"): "shared:synthetic",
                   ("C", "z1"): "singleton:C/z1"}
        probabilities = {"A": 0.9, "B": 0.3, "C": 0.2}
        records = self._evaluate(rows, unit_of, ("A",), [Fraction(1)] * 3,
                                 lambda row, plan: (probabilities[row["exam_id"]],
                                                    {"frame_indices": "synthetic"}))
        result = f007.aggregate_requests(rows, unit_of, records, "A")
        self.assertEqual(len(records), 3)
        self.assertEqual(result["summary"]["full_flops"], 3 * 5107788033)
        self.assertEqual(result["summary"]["completed_units"], 2)
        self.assertEqual(result["summary"]["accuracy"], 1.0)
        self.assertEqual(result["summary"]["auroc"], 1.0)
        shared = next(unit for unit in result["units"] if unit["unit_id"].startswith("shared"))
        self.assertAlmostEqual(shared["probability"], 0.6)
        self.assertEqual(shared["full_flops"], 2 * 5107788033)
        self.assertEqual(len(shared["aliases"]), 2)
        inconsistent = copy.deepcopy(records)
        inconsistent[1]["full_flops"] += 1
        with self.assertRaisesRegex(ValueError, "conflicting alias evidence"):
            f007.aggregate_requests(rows, unit_of, inconsistent, "A")
        one_class = f007.aggregate_requests(rows[:2], unit_of, records[:2], "A")
        self.assertIsNone(one_class["summary"]["normal_recall"])
        self.assertIsNone(one_class["summary"]["macro_f1"])
        self.assertIsNone(one_class["summary"]["auroc"])
        self.assertEqual(one_class["summary"]["abnormal_recall"], 1.0)

    def test_zero_coverage_defers_without_inference_or_metrics(self):
        rows = [synthetic_row("A", 1)]
        unit_of = {("A", "z1"): "singleton:A/z1"}
        records = self._evaluate(rows, unit_of, tuple(self.protocol["systems"]),
                                 [Fraction(0)], lambda *args: (_ for _ in ()).throw(
                                     AssertionError("DEFER must not infer")))
        self.assertEqual(len(records), 6)
        for system in self.protocol["systems"]:
            result = f007.aggregate_requests(rows, unit_of, records, system)
            self.assertEqual(result["summary"]["coverage"], 0)
            self.assertEqual(result["summary"]["full_flops"], 0)
            self.assertIsNone(result["summary"]["accuracy"])
            self.assertIsNone(result["summary"]["macro_f1"])

    def test_partial_shared_unit_rejected_before_inference(self):
        row = synthetic_row("A", 1)
        unit_of = {("A", "z1"): "shared:synthetic",
                   ("B", "z1"): "shared:synthetic"}
        with self.assertRaisesRegex(ValueError, "omits a retained alias"):
            self._evaluate([row], unit_of, ("A",), [Fraction(1)],
                           lambda *args: (_ for _ in ()).throw(
                               AssertionError("Must not infer a partial unit")))

    def test_technical_inference_error_propagates_and_evidence_never_overwrites(self):
        rows = [synthetic_row("A", 1)]
        unit_of = {("A", "z1"): "singleton:A/z1"}
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self._evaluate(rows, unit_of, ("A",), [Fraction(1)],
                           lambda *args: (_ for _ in ()).throw(RuntimeError("inference failed")))
        with self.assertRaisesRegex(ValueError, "frame diagnostic"):
            self._evaluate(rows, unit_of, ("A",), [Fraction(1)],
                           lambda *args: (0.5, {"full_flops": 0}))
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "attempt"
            f007.write_evidence(path, [], [], {"partition": "validation"})
            with self.assertRaises(FileExistsError):
                f007.write_evidence(path, [], [], {"partition": "validation"})


if __name__ == "__main__":
    unittest.main()
