"""Focused T012a policy-analysis checks; no model or image inference."""
import unittest
from fractions import Fraction

from experiments.F006a.run_analysis import (Point, budget_csv, budget_flops,
                                            configuration_csv, dominates,
                                            dominance_witnesses, pareto_front,
                                            read_rounded_auroc, read_selected_cells,
                                            select)


class ParetoTests(unittest.TestCase):
    def setUp(self):
        self.points = [Point("cheap", 0.25, 1, 10, 0.60),
                       Point("same_quality_more_cost", 0.5, 1, 20, 0.60),
                       Point("better", 0.5, 2, 20, 0.70),
                       Point("expensive_worse", 1.0, 2, 30, 0.55)]

    def test_strict_dominance_and_witness(self):
        a, b, c, d = self.points
        self.assertTrue(dominates(a, b))
        self.assertTrue(dominates(c, b))
        self.assertTrue(dominates(c, d))
        self.assertFalse(dominates(b, c))
        self.assertEqual([p.name for p in pareto_front(self.points)], ["cheap", "better"])
        witnesses = dominance_witnesses(self.points)
        self.assertEqual(witnesses["same_quality_more_cost"], "better")
        for p in self.points:
            if witnesses[p.name]:
                self.assertTrue(dominates(next(q for q in self.points if q.name == witnesses[p.name]), p))

    def test_equal_cost_and_quality_are_not_strict_dominance(self):
        left = Point("a", 0.25, 1, 10, 0.60)
        right = Point("b", 0.5, 1, 10, 0.60)
        self.assertFalse(dominates(left, right))
        self.assertFalse(dominates(right, left))
        self.assertEqual({p.name for p in pareto_front([right, left])}, {"a", "b"})
        self.assertEqual(select([right, left], Fraction(10), "max_f1").name, "a")


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.points = [Point("small", 0.25, 1, 10, 0.60),
                       Point("middle", 0.5, 2, 20, 0.70),
                       Point("large", 1.0, 16, 40, 0.71)]

    def test_exact_budget_feasibility_and_defer(self):
        self.assertEqual(budget_flops(Fraction(1, 4), 41), Fraction(41, 4))
        self.assertIsNone(select(self.points, budget_flops(Fraction(0), 40), "max_f1"))
        self.assertEqual(select(self.points, budget_flops(Fraction(1, 4), 40), "max_f1").name, "small")
        self.assertEqual(select(self.points, budget_flops(Fraction(1, 2), 40), "max_f1").name, "middle")
        for b in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
            limit = budget_flops(b, 40)
            for rule, parameter in (("max_f1", 0), ("near_best", 0.02), ("cost_aware", 0.02)):
                self.assertLessEqual(select(self.points, limit, rule, parameter, 40).flops, limit)

    def test_candidate_rule_sensitivity(self):
        self.assertEqual(select(self.points, Fraction(40), "max_f1").name, "large")
        self.assertEqual(select(self.points, Fraction(40), "near_best", 0.02).name, "middle")
        self.assertEqual(select(self.points, Fraction(40), "cost_aware", 0.03, 40).name, "middle")
        with self.assertRaises(ValueError):
            select(self.points, Fraction(40), "near_best", -0.01)

    def test_output_is_independent_of_input_order(self):
        self.assertEqual(configuration_csv(self.points), configuration_csv(list(reversed(self.points))))
        self.assertEqual(budget_csv(self.points), budget_csv(list(reversed(self.points))))


class CanonicalEvidenceTests(unittest.TestCase):
    def test_selected_epoch_is_the_deployable_run(self):
        cells = read_selected_cells()
        self.assertEqual(len(cells), 24)
        self.assertAlmostEqual(cells["w0.5_k2"]["macro_f1"], 0.8697064690795098)
        self.assertEqual(sum(cells["w1.0_k16"][name] for name in ("tn", "fp", "fn", "tp")), 254)

    def test_auroc_provenance_is_seed_2026_rounded_table(self):
        values = read_rounded_auroc()
        self.assertEqual(len(values), 24)
        self.assertEqual(values[(0.5, 2)], 0.925)
        self.assertEqual(values[(1.0, 16)], 0.960)


if __name__ == "__main__":
    unittest.main()
