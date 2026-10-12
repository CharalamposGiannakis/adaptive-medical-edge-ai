"""Validation-only F007 A2 statistics and runtime protocol checks."""
from __future__ import annotations

import copy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.F007 import analysis, benchmark_cpu, cache_diagnostic, phase_a1
from src.f007_protocol import ROOT, load_protocol


def unit(identity, group, target, probability, *, system="A", schedule="factorial_b0.50",
         aliases=None, flops=10, configuration="point"):
    aliases = aliases or [[identity, "z1"]]
    return {"unit_id": identity, "split_group": group, "binary_target": target,
            "aliases": aliases, "system_id": system, "schedule_id": schedule,
            "resource_state": "0.50", "completed": probability is not None,
            "probability": probability, "full_flops": flops if probability is not None else 0,
            "configuration": configuration if probability is not None else "DEFER",
            "deferral_reason": None if probability is not None else "resource_unavailable",
            "device": "synthetic"}


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_pcg64_indices_once_sorted_groups_and_group_multiplicity(self):
        groups = ("G1", "G2")
        first = analysis.bootstrap_indices(groups, self.protocol)
        second = analysis.bootstrap_indices(groups, self.protocol)
        self.assertEqual(first.shape, (5000, 2))
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            analysis.bootstrap_indices(("G2", "G1"), self.protocol)
        records = {"one": unit("one", "G1", 0, .2, aliases=[["a", "z1"], ["b", "z1"]]),
                   "two": unit("two", "G1", 1, .8),
                   "three": unit("three", "G2", 0, .1)}
        sampled = analysis._resample(records, groups, np.array([0, 0]))
        self.assertEqual([row["unit_id"] for row in sampled],
                         ["one", "two", "one", "two"])
        self.assertEqual(analysis._numeric_metrics(sampled)["requested_rows"], 6)

    def test_missing_class_draws_and_valid_draw_suppression(self):
        records = {"a": unit("a", "A", 0, .1), "b": unit("b", "B", 1, .9)}
        groups = ("A", "B")
        draws = analysis.bootstrap_indices(groups, self.protocol)
        result = analysis.cell_analysis(records, groups, draws, self.protocol,
                                        factorial=False)
        expected_valid = int(np.sum(draws[:, 0] != draws[:, 1]))
        self.assertEqual(result["intervals"]["macro_f1"]["valid_draws"], expected_valid)
        self.assertIsNone(result["intervals"]["macro_f1"]["ci95"])
        self.assertEqual(result["intervals"]["accuracy"]["valid_draws"], 5000)
        self.assertIsNone(analysis._numeric_metrics([records["a"]])["abnormal_recall"])
        self.assertIsNone(analysis._numeric_metrics([records["a"]])["macro_f1"])
        self.assertIsNone(analysis._numeric_metrics([records["a"]])["auroc"])

    def test_rank_auc_matches_frozen_binary_auc_with_ties(self):
        from sklearn.metrics import roc_auc_score
        targets = [0, 1, 0, 1, 1, 0, 1]
        probabilities = [.1, .5, .5, .5, .8, .8, .9]
        self.assertEqual(analysis._auc(targets, probabilities),
                         float(roc_auc_score(targets, probabilities)))

    def test_paired_common_completed_denominators_all_request_coverage_and_cost(self):
        left = {x["unit_id"]: x for x in (
            unit("a", "G1", 0, .1, aliases=[["a", "z1"], ["a2", "z1"]], flops=20),
            unit("b", "G2", 1, .8, flops=10), unit("c", "G3", 1, None))}
        right = {x["unit_id"]: x for x in (
            unit("a", "G1", 0, .2, aliases=[["a", "z1"], ["a2", "z1"]], flops=40),
            unit("b", "G2", 1, None), unit("c", "G3", 1, .7, flops=10))}
        result = analysis._paired_values(list(left.values()), list(right.values()))
        self.assertEqual(result["common_completed_units"], 1)
        self.assertEqual(result["common_completed_rows"], 2)
        self.assertEqual(result["common_full_flops_difference"], -20)
        self.assertEqual(result["left_full_flops_all"], 30)
        self.assertEqual(result["right_full_flops_all"], 50)
        self.assertEqual(result["coverage_difference"], 0)
        self.assertIsNone(result["macro_f1"])
        self.assertIsNone(result["auroc"])

    def test_preselected_comparisons_and_identity_skip(self):
        specs = analysis.comparison_specs(self.protocol)
        self.assertEqual(len(specs), 24)
        self.assertIn(("factorial_b0.25", "F", "D", "adaptive_policy", True), specs)
        self.assertIn(("factorial_b1.00", "F", "E", "adaptive_policy", True), specs)
        self.assertIn(("heterogeneous", "F", "B", "adaptive_vs_static", False), specs)
        records = {"a": unit("a", "G1", 0, .2)}
        draws = analysis.bootstrap_indices(("G1",), self.protocol)
        same = analysis.paired_analysis(records, copy.deepcopy(records), ("G1",), draws,
                                        self.protocol, identity_expected=True)
        self.assertTrue(same["bootstrap_tests_skipped"])
        changed = copy.deepcopy(records)
        changed["a"]["probability"] = .3
        with self.assertRaisesRegex(ValueError, "identity-by-construction"):
            analysis.paired_analysis(records, changed, ("G1",), draws,
                                     self.protocol, identity_expected=True)


class BenchmarkAndCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_nine_frozen_points_and_exact_receipts(self):
        from tests.test_f007_phase_a1 import BASELINE, points_from_screen
        grid = points_from_screen()
        points = benchmark_cpu.nine_points(self.protocol, grid)
        self.assertEqual(tuple(p.name for p in points), benchmark_cpu.POINT_IDS)
        self.assertEqual(points[1].flops, points[-1].flops)
        for point in points:
            benchmark_cpu._receipt(point, self.protocol, grid, BASELINE)
        changed = copy.deepcopy(points[2])
        object.__setattr__(changed, "flops", 999999999999)
        with self.assertRaises(ValueError):
            benchmark_cpu._receipt(changed, self.protocol, grid, BASELINE)

    def test_chunked_direct_agreement_and_fail_closed(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
                self.head = torch.nn.Linear(2, 1)
            def forward(self, frames):
                batch, k = frames.shape[:2]
                features = self.encoder(frames.reshape(batch * k, 3, 2, 2))
                return self.head(features.reshape(batch, k, 2).mean(1)).squeeze(-1)
        torch.manual_seed(7)
        model = Tiny().eval()
        point = benchmark_cpu.Point("synthetic", "original", 16, None, 1, "B", "1.00")
        frames = torch.rand(16, 3, 2, 2)
        direct, _ = benchmark_cpu._infer_frames(model, point, frames, chunk=None)
        chunked, _ = benchmark_cpu._infer_frames(model, point, frames, chunk=4)
        self.assertLessEqual(abs(direct - chunked), 1e-4)
        with self.assertRaises(ValueError):
            benchmark_cpu._infer_frames(model, point, frames, chunk=5)
        fake = benchmark_cpu.Inputs(("a/z1",), ({},), (frames,), (frames,),
                                    ({16: torch.arange(16)},), None, None)
        self.assertIn("synthetic", benchmark_cpu.verify_equivalence(
            {"original": model}, (point,), fake))
        with patch.object(benchmark_cpu, "infer_preloaded", side_effect=[(0.1, .52), (-0.1, .48)]):
            with self.assertRaises(RuntimeError):
                benchmark_cpu.verify_equivalence({"original": model}, (point,), fake)

    def test_cache_validation_only_and_unit_selection_preserves_aliases(self):
        rows = [{"exam_id": "a", "zone_id": "z1", "partition": "validation"},
                {"exam_id": "b", "zone_id": "z1", "partition": "validation"},
                {"exam_id": "c", "zone_id": "z1", "partition": "validation"}]
        mapping = {("a", "z1"): "shared", ("b", "z1"): "shared",
                   ("c", "z1"): "single"}
        self.assertEqual(len(cache_diagnostic.select_validation_units(rows, mapping, 1)), 2)
        with self.assertRaises(PermissionError):
            cache_diagnostic.diagnose([dict(rows[0], partition="test")], mapping,
                                      self.protocol, None, (), None)

    def test_cache_direct_parity_and_exact_incremental_work(self):
        import torch
        from tests.test_f007_phase_a1 import BASELINE, points_from_screen
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, frames):
                return frames.mean((2, 3)).mean(1, keepdim=True) * self.weight
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.head = torch.nn.Linear(1, 1)
        class Sequences:
            def __init__(self, rows, size):
                pass
            def frames_at(self, index, numbers):
                return torch.stack([torch.full((3, 224, 224), float(number) / 100)
                                    for number in numbers])
        row = {"exam_id": "a", "zone_id": "z1", "partition": "validation",
               "frame_count": "16"}
        from experiments.F005 import prune
        with patch.object(cache_diagnostic, "F003Sequences", Sequences), \
                patch.object(prune, "set_width", return_value=None):
            records = cache_diagnostic.diagnose([row], {("a", "z1"): "singleton:a/z1"},
                self.protocol, Model().eval(), points_from_screen(), BASELINE)
        self.assertEqual(len(records), 30)
        served = [record for record in records if record["configuration"] != "DEFER"]
        self.assertTrue(served)
        self.assertTrue(all(record["abs_logit_difference"] <= 1e-4 for record in served))
        self.assertTrue(all(record["incremental_cached_flops"] <=
                            record["full_independent_flops"] for record in records))
        self.assertTrue(any(record["cached_frames"] > 0 for record in served))


class ReportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        (ROOT / "artifacts/F007").mkdir(parents=True, exist_ok=True)

    def test_a1_requests_to_a2_mixed_and_multiple_states(self):
        from tests.test_f007_phase_a1 import BASELINE, points_from_screen, synthetic_row
        rows = [synthetic_row("a", 0, group="G1"),
                synthetic_row("b", 1, group="G2"),
                synthetic_row("c", 1, group="G2")]
        mapping = {("a", "z1"): "singleton:a/z1",
                   ("b", "z1"): "shared:bc", ("c", "z1"): "shared:bc"}
        cells, requests = {}, []
        predictor = lambda row, plan: (.2 if row["binary_target"] == "0" else .8, {})
        with patch.object(phase_a1.controller, "load_policy_points", return_value=points_from_screen()), \
                patch.object(phase_a1.f006a_2, "read_baseline", return_value=BASELINE):
            schedules = {"factorial_b0": [Fraction(0)] * 3,
                         "factorial_b0.25": [Fraction("0.25")] * 3,
                         "factorial_b0.50": [Fraction("0.50")] * 3,
                         "heterogeneous": phase_a1.mixed_assignments(rows, mapping,
                                                                       self.protocol)[1]}
            for schedule, states in schedules.items():
                records = phase_a1.evaluate_requests(rows, mapping, ("A", "D", "E", "F"),
                    states, self.protocol, predictor, schedule_id=schedule)
                requests.extend(records)
                cells[schedule] = {system: phase_a1.aggregate_requests(
                    rows, mapping, records, system, schedule_id=schedule)["units"]
                    for system in ("A", "D", "E", "F")}
        report, draws = analysis.analyze(cells, requests, self.protocol,
                                         require_complete=False)
        self.assertEqual(report["bootstrap"]["draws"], 5000)
        self.assertEqual(report["cells"]["factorial_b0:F"]["point"]["coverage"], 0)
        self.assertEqual(report["cells"]["factorial_b0.25:F"]["point"]["requested_rows"], 3)
        self.assertIn("heterogeneous:F", report["cells"])
        self.assertTrue(report["contrasts"]["factorial_b0.25:F-D"]["bootstrap_tests_skipped"])
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/F007") as folder:
            output = Path(folder) / "report"
            analysis.write_report(output, report, draws, source=Path(folder))
            self.assertTrue((output / "quality_vs_flops.svg").exists())
            self.assertTrue((output / "paired_contrasts.csv").exists())
            first = (output / "report.json").read_bytes()
            report2, draws2 = analysis.analyze(cells, requests, self.protocol,
                                               require_complete=False)
            np.testing.assert_array_equal(draws, draws2)
            self.assertEqual(report, report2)
            second_output = Path(folder) / "report_again"
            analysis.write_report(second_output, report2, draws2, source=Path(folder))
            self.assertEqual(first, (second_output / "report.json").read_bytes())
            for name in ("bootstrap_indices.npy", "quality_coverage_cost.csv",
                         "paired_contrasts.csv", "quality_vs_flops.svg",
                         "quality_vs_flops.png"):
                self.assertEqual((output / name).read_bytes(),
                                 (second_output / name).read_bytes(), name)
        with self.assertRaises(PermissionError):
            analysis.analyze(cells, [dict(requests[0], partition="test")], self.protocol,
                             require_complete=False)

    def test_evidence_loader_rejects_tampered_unit_aggregation(self):
        import json
        from tests.test_f007_phase_a1 import BASELINE, points_from_screen, synthetic_row
        from src.f007_protocol import sha256_file
        rows = [synthetic_row("a", 0, group="G1"), synthetic_row("b", 1, group="G2")]
        mapping = {("a", "z1"): "singleton:a/z1", ("b", "z1"): "singleton:b/z1"}
        with patch.object(phase_a1.controller, "load_policy_points", return_value=points_from_screen()), \
                patch.object(phase_a1.f006a_2, "read_baseline", return_value=BASELINE):
            records = phase_a1.evaluate_requests(rows, mapping, ("A",),
                [Fraction("0.25")] * 2, self.protocol,
                lambda row, plan: (.2 if row["binary_target"] == "0" else .8, {}),
                schedule_id="factorial_b0.25")
        result = phase_a1.aggregate_requests(rows, mapping, records, "A",
                                              schedule_id="factorial_b0.25")
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/F007") as folder:
            output = Path(folder) / "evidence"
            phase_a1.write_evidence(output, records, [result], {
                "experiment": "F007", "partition": "validation",
                "request_ids": [[r["exam_id"], r["zone_id"]] for r in rows],
                "protocol_sha256": sha256_file(ROOT / "configs/f007_protocol.json"),
                "checkpoints": self.protocol["checkpoints"]})
            with patch.object(phase_a1, "validation_context", return_value=(rows, mapping, {})):
                loaded, _, _ = analysis.load_validation_evidence(output)
                self.assertEqual(len(loaded["factorial_b0.25"]["A"]), 2)
                units_file = output / "units.jsonl"
                units = [json.loads(line) for line in units_file.read_text().splitlines()]
                units[0]["probability"] = .9
                units_file.write_text("\n".join(json.dumps(unit) for unit in units) + "\n")
                with self.assertRaisesRegex(ValueError, "canonical alias aggregation"):
                    analysis.load_validation_evidence(output)


if __name__ == "__main__":
    unittest.main()
