"""Focused F006a_2 quality-screening checks; no model or image inference."""
import csv
import unittest
from fractions import Fraction

from experiments.F006a.run_analysis import Point, budget_flops, dominance_witnesses, read_selected_cells
from experiments.F006a_2 import run_analysis as analysis


class ExactQualityTests(unittest.TestCase):
    def test_frozen_f002b_reference_is_reconstructed_from_confusion_counts(self):
        baseline = analysis.read_baseline()
        self.assertEqual((baseline.tn, baseline.fp, baseline.fn, baseline.tp), (168, 10, 19, 57))
        self.assertEqual(baseline.accuracy, Fraction(225, 254))
        self.assertEqual(baseline.macro_f1, Fraction(44829, 52195))
        self.assertEqual(baseline.normal_recall, Fraction(168, 178))
        self.assertEqual(baseline.abnormal_recall, Fraction(57, 76))
        # The README's three-decimal display values are not screening thresholds.
        self.assertNotEqual(baseline.accuracy, Fraction(886, 1000))
        self.assertNotEqual(baseline.macro_f1, Fraction(859, 1000))

    def test_confusion_counts_must_cover_all_duplicate_aware_units(self):
        with self.assertRaises(ValueError):
            analysis.quality_from_counts(167, 10, 19, 57)
        with self.assertRaises(ValueError):
            analysis.quality_from_counts(168, 10, 19, 58)
        with self.assertRaises(ValueError):
            analysis.quality_from_counts(-1, 179, 19, 57)
        with self.assertRaises(ValueError):
            analysis.quality_from_counts(168.0, 10, 19, 57)

    def test_inclusive_screening_at_exact_equality_and_each_strict_boundary(self):
        baseline = analysis.read_baseline()
        profile = analysis.Profile(Point("equal", 0.5, 12, 100, float(baseline.macro_f1)), baseline)
        thresholds = analysis.baseline_thresholds(baseline)
        self.assertEqual(analysis.screen(profile, thresholds),
                         {"accuracy": True, "macro_f1": True, "abnormal_recall": True, "all": True})

        for metric, increment in (("accuracy", Fraction(1, 254)),
                                  ("macro_f1", Fraction(1, 100000)),
                                  ("abnormal_recall", Fraction(1, 76))):
            raised = analysis.Thresholds(
                thresholds.accuracy + (increment if metric == "accuracy" else 0),
                thresholds.macro_f1 + (increment if metric == "macro_f1" else 0),
                thresholds.abnormal_recall + (increment if metric == "abnormal_recall" else 0),
            )
            result = analysis.screen(profile, raised)
            self.assertFalse(result[metric])
            self.assertFalse(result["all"])
            self.assertTrue(all(result[other] for other in
                                ("accuracy", "macro_f1", "abnormal_recall") if other != metric))


class CanonicalProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = analysis.read_baseline()
        cells = read_selected_cells()  # Hash-guarded F005b selected epoch.
        with (analysis.ROOT / "experiments" / "F006a" / "configurations.csv").open(
                newline="", encoding="utf-8") as stream:
            configurations = list(csv.DictReader(stream))
        cls.profiles = []
        for row in configurations:
            cell = cells[row["configuration"]]
            quality = analysis.quality_from_counts(*(cell[field] for field in ("tn", "fp", "fn", "tp")))
            point = Point(row["configuration"], float(row["width"]), int(row["k"]),
                          int(row["flops"]), cell["macro_f1"], cell["accuracy"],
                          float(row["auroc_rounded_3dp"]))
            cls.profiles.append(analysis.Profile(point, quality))
        cls.by_name = {profile.point.name: profile for profile in cls.profiles}

    def test_all_24_cells_are_screened_without_pareto_pruning(self):
        self.assertEqual(len(self.profiles), 24)
        self.assertEqual(len(self.by_name), 24)
        thresholds = analysis.baseline_thresholds(self.baseline)
        self.assertEqual(analysis.screening_summary(self.profiles, thresholds),
                         {"accuracy": 12, "macro_f1": 10, "abnormal_recall": 14, "all": 10})
        passing = {name for name, profile in self.by_name.items()
                   if analysis.screen(profile, thresholds)["all"]}
        self.assertEqual(passing, {
            "w0.5_k2", "w0.5_k8", "w0.75_k8", "w1.0_k8", "w0.5_k12",
            "w0.75_k12", "w0.5_k16", "w1.0_k12", "w0.75_k16", "w1.0_k16",
        })
        # A dominated configuration may still be worth considering for abnormal recall.
        self.assertIsNotNone(dominance_witnesses([p.point for p in self.profiles])["w1.0_k12"])
        self.assertIn("w1.0_k12", passing)

    def test_exact_equal_counts_pass_and_cheapest_survivor_is_distinct_from_best_recall(self):
        match = self.by_name["w0.5_k12"]
        self.assertEqual(match.quality, self.baseline)
        self.assertTrue(analysis.screen(match, analysis.baseline_thresholds(self.baseline))["all"])
        qualified = [profile for profile in self.profiles
                     if analysis.screen(profile, analysis.baseline_thresholds(self.baseline))["all"]]
        cheapest = analysis.cheapest(qualified)
        self.assertEqual(cheapest.point.name, "w0.5_k2")
        self.assertEqual(cheapest.point.flops, 1_816_575_299)
        self.assertEqual(cheapest.quality.abnormal_recall, Fraction(59, 76))
        self.assertGreater(self.by_name["w1.0_k16"].quality.abnormal_recall,
                           cheapest.quality.abnormal_recall)

    def test_budget_uses_exact_integer_flops_and_defers_without_qualified_profile(self):
        cap = analysis.EXPECTED_CMAX
        self.assertEqual(self.by_name["w1.0_k16"].point.flops, cap)
        self.assertGreater(self.by_name["w1.0_k4"].point.flops, budget_flops(Fraction(1, 4), cap))
        self.assertGreater(self.by_name["w1.0_k8"].point.flops, budget_flops(Fraction(1, 2), cap))
        self.assertGreater(self.by_name["w1.0_k12"].point.flops, budget_flops(Fraction(3, 4), cap))

        rows = analysis.budget_rows(self.profiles, self.baseline)
        self.assertEqual([row["budget_feasible_count"] for row in rows], [0, 11, 16, 21, 24])
        self.assertEqual([row["quality_and_budget_count"] for row in rows], [0, 1, 3, 7, 10])
        self.assertEqual([row["cheapest_qualified"] for row in rows],
                         ["DEFER", "w0.5_k2", "w0.5_k2", "w0.5_k2", "w0.5_k2"])
        self.assertEqual([row["status"] for row in rows],
                         ["DEFER", "FEASIBLE", "FEASIBLE", "FEASIBLE", "FEASIBLE"])

    def test_all_output_tables_are_byte_identical_for_reversed_profiles(self):
        forward = analysis.render_products(self.profiles, self.baseline)
        backward = analysis.render_products(list(reversed(self.profiles)), self.baseline)
        self.assertEqual(set(forward), {"quality_screen.csv", "screening_summary.csv",
                                        "budget_feasibility.csv", "sensitivity.csv"})
        self.assertEqual(forward, backward)
        self.assertEqual(analysis.render_figure(self.profiles, self.baseline),
                         analysis.render_figure(list(reversed(self.profiles)), self.baseline))

    def test_one_factor_sensitivity_can_make_active_budgets_defer(self):
        scenarios = {row["scenario"]: row for row in
                     analysis.sensitivity_rows(self.profiles, self.baseline)}
        self.assertEqual(scenarios["baseline"]["qualified_count"], 10)
        self.assertEqual(scenarios["accuracy_+3_unit"]["qualified_count"], 5)
        self.assertEqual(scenarios["accuracy_+3_unit"]["b0.25_cheapest"], "DEFER")
        self.assertEqual(scenarios["abnormal_recall_+3_unit"]["qualified_count"], 4)
        self.assertEqual(scenarios["abnormal_recall_+3_unit"]["b0.50_cheapest"], "DEFER")
        self.assertTrue(all(row["b0.00_cheapest"] == "DEFER" for row in scenarios.values()))


if __name__ == "__main__":
    unittest.main()
