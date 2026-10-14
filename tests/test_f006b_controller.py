"""Exact-budget and deterministic-trace checks for the provisional T012b policy.

The integration fixture reads F006a_2's tracked 24-cell table rather than
re-running fvcore or doing image inference. F006b's regeneration command
separately checks the canonical source hashes and output bytes.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest

from experiments.F005.run_f005b import (CANONICAL_NESTED,
                                        CANONICAL_NESTED_CHECKPOINT_SHA256,
                                        CANONICAL_NESTED_IMPORTANCE_SHA256)
from experiments.F006a_2 import run_analysis as f006a_2
from experiments.F006a_2.run_analysis import quality_from_counts
from experiments.F006b import controller, simulate


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "experiments" / "F006a_2" / "quality_screen.csv"
BASELINE = quality_from_counts(168, 10, 19, 57)
EXPECTED = ("w0.5_k2", "w1.0_k8", "w1.0_k12", "w1.0_k16")


def tracked_points() -> tuple[controller.OperatingPoint, ...]:
    with SCREEN.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(controller.OperatingPoint(
        name=row["configuration"], width=float(row["width"]), k=int(row["k"]),
        flops=int(row["flops"]),
        quality=quality_from_counts(*(int(row[key]) for key in ("tn", "fp", "fn", "tp"))),
        recorded_macro_f1=float(row["macro_f1"]),
    ) for row in rows)


def selected_name(points: tuple[controller.OperatingPoint, ...], b) -> str:
    result = controller.select(points, BASELINE, b)
    return result.point.name if result.point else "DEFER"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ControllerPureTests(unittest.TestCase):
    def test_capacity_default_is_exact_historical_cmodel(self):
        points = tracked_points()
        for policy in controller.POLICIES:
            for resource in ("0", "0.25", "0.50", "0.75", "1.00"):
                old = controller.select(points, BASELINE, resource, policy)
                explicit = controller.select(points, BASELINE, resource, policy,
                                             capacity_flops=controller.C_MAX)
                self.assertEqual(old, explicit)
                self.assertEqual(old.permitted_flops,
                                 Fraction(resource) * controller.C_MAX)
        self.assertEqual(controller.capacity_fraction(Fraction(11, 10) * controller.C_MAX),
                         Fraction(226556754193, 10))
        for bad in (0, -1, 1.1, True, "bad"):
            with self.subTest(capacity=bad), self.assertRaises(ValueError):
                controller.capacity_fraction(bad)

    def test_f007_capacity_moves_exact_entry_boundary_without_changing_rule(self):
        points = tracked_points()
        capacity = Fraction(11, 10) * controller.C_MAX
        point = next(point for point in points if point.name == "w0.5_k2")
        at_boundary = Fraction(point.flops, 1) / capacity
        below_boundary = Fraction(point.flops - 1, 1) / capacity
        self.assertEqual(controller.select(points, BASELINE, at_boundary,
                         capacity_flops=capacity).point.name, point.name)
        deferred = controller.select(points, BASELINE, below_boundary,
                                     capacity_flops=capacity)
        self.assertTrue(deferred.deferred)
        self.assertEqual(deferred.permitted_flops, point.flops - 1)
        self.assertEqual(tuple(point.name for point in controller.selected_configurations(
            points, BASELINE, capacity_flops=capacity)), EXPECTED)

    def test_quality_qualification_is_exact_and_inclusive(self):
        point = controller.OperatingPoint("equal", 0.5, 2, 100, BASELINE, float(BASELINE.macro_f1))
        self.assertTrue(controller.quality_qualified(point, BASELINE))
        one_correct = Fraction(1, 254)
        one_abnormal = Fraction(1, 76)
        for changed in (
            replace(BASELINE, accuracy=BASELINE.accuracy - one_correct),
            replace(BASELINE, macro_f1=BASELINE.macro_f1 - Fraction(1, 10**12)),
            replace(BASELINE, abnormal_recall=BASELINE.abnormal_recall - one_abnormal),
        ):
            self.assertFalse(controller.quality_qualified(replace(point, quality=changed), BASELINE))

    def test_recall_then_flops_then_name_breaks_ties(self):
        high_recall = quality_from_counts(168, 10, 16, 60)
        low_recall = quality_from_counts(168, 10, 17, 59)
        points = (
            controller.OperatingPoint("cheap_low_recall", 0.5, 2, 100, low_recall, 0.90),
            controller.OperatingPoint("costly_high_recall", 1.0, 8, 300, high_recall, 0.88),
            controller.OperatingPoint("z_equal", 0.75, 4, 200, high_recall, 0.87),
            controller.OperatingPoint("a_equal", 0.5, 4, 200, high_recall, 0.86),
        )
        self.assertEqual(selected_name(points, 1), "a_equal")
        self.assertEqual(selected_name(points, Fraction(199, controller.C_MAX)), "cheap_low_recall")
        self.assertEqual(selected_name(points, Fraction(99, controller.C_MAX)), "DEFER")
        self.assertEqual(selected_name(tuple(reversed(points)), 1), "a_equal")

    def test_resource_parsing_rejects_invalid_values(self):
        for invalid in (-1, 1.0001, "not-a-number", "1/0", float("nan"),
                        float("inf"), Decimal("NaN"), Decimal("Infinity"), True, None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                controller.resource_fraction(invalid)
        self.assertEqual(controller.resource_fraction(Decimal("0.5001")), Fraction(5001, 10000))
        self.assertEqual(controller.resource_fraction(0.5001), Fraction(5001, 10000))


class TrackedGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = tracked_points()
        cls.by_name = {point.name: point for point in cls.points}

    def test_all_24_cells_and_four_selected_three_dimensional_points(self):
        self.assertEqual(len(self.points), 24)
        self.assertEqual(len(self.by_name), 24)
        self.assertEqual(sum(controller.quality_qualified(point, BASELINE) for point in self.points), 10)
        selected = controller.selected_configurations(self.points, BASELINE)
        self.assertEqual(tuple(point.name for point in selected), EXPECTED)
        self.assertTrue(all(not controller.three_dimensional_dominators(point, self.points)
                            for point in selected))
        self.assertEqual(simulate.validate_envelope(self.points, BASELINE), selected)

    def test_exact_five_state_and_eleven_value_continuous_sweep(self):
        expected = {
            "0": "DEFER", "0.05": "DEFER", "0.25": "w0.5_k2",
            "0.50": "w0.5_k2", "0.5001": "w1.0_k8", "0.67": "w1.0_k8",
            "0.75": "w1.0_k8", "0.7501": "w1.0_k12", "0.83": "w1.0_k12",
            "0.99": "w1.0_k12", "1.00": "w1.0_k16",
        }
        self.assertEqual(tuple(expected), simulate.SWEEP)
        self.assertEqual({b: selected_name(self.points, b) for b in simulate.SWEEP}, expected)
        self.assertEqual([selected_name(self.points, b) for b in
                          ("0", "0.25", "0.50", "0.75", "1.00")],
                         ["DEFER", "w0.5_k2", "w0.5_k2", "w1.0_k8", "w1.0_k16"])
        self.assertEqual(controller.select(self.points, BASELINE, "0").reason,
                         "resource_unavailable")
        self.assertEqual(controller.select(self.points, BASELINE, "0.05").reason,
                         "no_quality_qualified_point_within_budget")
        for policy in controller.POLICIES:
            self.assertTrue(controller.select(self.points, BASELINE, "0", policy).deferred)
        self.assertEqual(controller.select(self.points, BASELINE, "0.25", "fixed_full_k16").reason,
                         "fixed_point_over_budget")
        self.assertEqual(controller.select(self.points, BASELINE, "0.05", "minimum_quality").reason,
                         "minimum_quality_point_over_budget")
        for point in self.points:
            choice = controller.select(self.points, BASELINE,
                                       Fraction(point.flops, controller.C_MAX))
            self.assertLessEqual(choice.point.flops if choice.point else 0,
                                 choice.permitted_flops)

    def test_each_selected_threshold_has_no_floating_flop_tolerance(self):
        predecessors = ("DEFER", "w0.5_k2", "w1.0_k8", "w1.0_k12")
        for name, prior in zip(EXPECTED, predecessors):
            flops = self.by_name[name].flops
            with self.subTest(point=name):
                self.assertEqual(selected_name(self.points,
                                 Fraction(flops - 1, controller.C_MAX)), prior)
                self.assertEqual(selected_name(self.points,
                                 Fraction(flops, controller.C_MAX)), name)
                if flops < controller.C_MAX:
                    self.assertEqual(selected_name(self.points,
                                     Fraction(flops + 1, controller.C_MAX)), name)
        self.assertGreater(self.by_name["w1.0_k8"].flops,
                           Fraction(1, 2) * controller.C_MAX)
        self.assertGreater(self.by_name["w1.0_k12"].flops,
                           Fraction(3, 4) * controller.C_MAX)
        self.assertEqual(controller.select(self.points, BASELINE, "0.50").permitted_flops,
                         Fraction(controller.C_MAX, 2))

    def test_selected_abnormal_recall_is_monotone_at_every_measured_cost(self):
        resources = sorted({Fraction(point.flops, controller.C_MAX) for point in self.points})
        selected = [controller.select(self.points, BASELINE, budget).point
                    for budget in resources]
        recalls = [point.quality.abnormal_recall for point in selected if point]
        self.assertEqual(recalls, sorted(recalls))
        self.assertEqual([self.by_name[name].quality.tp for name in EXPECTED],
                         [59, 60, 62, 64])
        self.assertLess(self.by_name["w1.0_k12"].quality.macro_f1,
                        self.by_name["w1.0_k8"].quality.macro_f1)

    def test_traces_and_comparison_are_deterministic_and_cache_aware(self):
        cases = simulate.validation_cases()
        self.assertEqual(len({case["split_group"] for case in cases}), 3)
        first = simulate.trace_rows(self.points, BASELINE, cases)
        self.assertEqual(first, simulate.trace_rows(self.points, BASELINE, cases))
        self.assertEqual(len(first), 3 * len(controller.POLICIES) * 5)
        observed = {
            "decreasing": (["w1.0_k16", "w1.0_k12", "w1.0_k8", "w0.5_k2", "DEFER"],
                           [16, 0, 0, 2, 0]),
            "increasing": (["DEFER", "w0.5_k2", "w1.0_k8", "w1.0_k12", "w1.0_k16"],
                           [0, 2, 8, 4, 4]),
            "fluctuating": (["w1.0_k12", "w0.5_k2", "w1.0_k8", "w1.0_k16", "w0.5_k2"],
                            [12, 2, 0, 4, 0]),
        }
        for scenario, (names, new_frames) in observed.items():
            rows = [row for row in first if row["scenario"] == scenario
                    and row["policy"] == "recall_priority"]
            self.assertEqual([row["configuration"] for row in rows], names)
            self.assertEqual([row["new_encoded_frames"] for row in rows], new_frames)
            self.assertTrue(all(row["predicted_class"] == "" and row["observed_local_ms"] == ""
                                for row in rows))
            self.assertTrue(all(row["reused_frames"] + row["new_encoded_frames"] ==
                                (row["k"] if row["deferred"] == "false" else 0)
                                for row in rows))
        comparisons = simulate.comparison_rows(first)
        self.assertEqual(comparisons, simulate.comparison_rows(first))
        self.assertEqual(len(comparisons), 3 * len(controller.POLICIES))
        for row in comparisons:
            self.assertEqual(row["requests"], 5)
            self.assertEqual(row["inferences"] + row["deferrals"], 5)
            self.assertEqual(row["budget_violations"], 0)
            self.assertLessEqual(row["total_incremental_modeled_flops"],
                                 row["total_full_configuration_flops"])
            self.assertEqual(row["measured_mixed_policy_macro_f1"], "")
        self.assertTrue(all(row["deferrals"] == 4 for row in comparisons
                            if row["policy"] == "fixed_full_k16"))
        rendered, trace = simulate.products(self.points, BASELINE, cases)
        rendered_again, trace_again = simulate.products(self.points, BASELINE, cases)
        self.assertEqual(trace, trace_again)
        self.assertEqual(rendered, rendered_again)

    def test_z_frozen_checkpoints_and_importance_are_unchanged(self):
        """Recheck canonical artifacts after policy/runtime tests have exercised them."""
        self.assertEqual(file_sha256(f006a_2.F002_CHECKPOINT),
                         f006a_2.F002_CHECKPOINT_SHA256)
        self.assertEqual(file_sha256(CANONICAL_NESTED / "checkpoint.pt"),
                         CANONICAL_NESTED_CHECKPOINT_SHA256)
        self.assertEqual(file_sha256(CANONICAL_NESTED / "importance.pt"),
                         CANONICAL_NESTED_IMPORTANCE_SHA256)
        run = json.loads((CANONICAL_NESTED / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["status"], run["mode"], run["seed"]),
                         ("completed", "nested", 2026))
        self.assertEqual(run["checkpoint_sha256"], CANONICAL_NESTED_CHECKPOINT_SHA256)
        self.assertEqual((run["importance"], run["orders_criterion"]),
                         ("taylor", "taylor"))


if __name__ == "__main__":
    unittest.main()
