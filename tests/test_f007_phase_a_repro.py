"""Focused Phase A reproducibility and report corrections; no test images."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from experiments.F007 import analysis, provenance, validation_run
from src.f007_protocol import load_protocol
from tests.test_f007_phase_a2 import unit


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_git_clean_and_dirty_detection_excludes_local_agent_folders(self):
        commit = b"a" * 40 + b"\n"
        with patch.object(provenance, "_git", side_effect=[commit, b""]):
            self.assertEqual(provenance.git_state(),
                             {"commit_sha": "a" * 40, "working_tree_dirty": False})
        with patch.object(provenance, "_git", side_effect=[commit, b" M src/f007_protocol.py\n"]):
            self.assertTrue(provenance.git_state()["working_tree_dirty"])
        with patch.object(provenance, "_git", side_effect=[commit, b""]) as called:
            provenance.git_state()
            status_args = called.call_args_list[1].args
            self.assertIn(":(exclude)CLAUDE", status_args)
            self.assertIn(":(exclude)CODEX", status_args)

    def test_provenance_complete_and_official_dirty_gate(self):
        record = provenance.capture(self.protocol, {"verified": True},
                                    official=False, execution_device="cpu",
                                    parameters={"mode": "diagnostic"})
        self.assertEqual(set(record["source_sha256"]), set(provenance.SOURCE_MODULES))
        self.assertEqual(len(record["commit_sha"]), 40)
        self.assertEqual(record["frozen_checkpoints"], self.protocol["checkpoints"])
        self.assertEqual(record["frozen_inputs"], {"verified": True})
        self.assertIn("torch", record["packages"])
        self.assertIn("cpu_model", record["system"])
        self.assertIn("torch_intraop", record["threads"])
        self.assertEqual(record["parameters"]["mode"], "diagnostic")
        self.assertTrue(record["command"]["argv"])
        self.assertTrue(record["started_utc"])
        with patch.object(provenance, "git_state", return_value={
                "commit_sha": "a" * 40, "working_tree_dirty": True}), \
                patch.object(provenance, "source_hashes") as hashes:
            with self.assertRaisesRegex(RuntimeError, "clean committed"):
                provenance.capture(self.protocol, {}, official=True,
                                   execution_device="cpu", parameters={})
            hashes.assert_not_called()

    def test_execution_detects_changed_source(self):
        record = provenance.capture(self.protocol, {}, official=False,
                                    execution_device="cpu", parameters={})
        changed = copy.deepcopy(record["source_sha256"])
        changed["experiments/F007/analysis.py"]["working_tree_sha256"] = "0" * 64
        with patch.object(provenance, "source_hashes", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "changed during execution"):
                provenance.assert_unchanged(record)


class ReportingCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_realized_heterogeneous_counts_use_units_not_aliases(self):
        shared = unit("u1", "G1", 0, .2, system="A", schedule="heterogeneous",
                      aliases=[["a", "z1"], ["b", "z1"]])
        shared["resource_state"] = "0.25"
        deferred = unit("u2", "G2", 1, None, system="A", schedule="heterogeneous")
        deferred["resource_state"] = "0.50"
        systems = {"A": [shared, deferred],
                   "B": [dict(shared, system_id="B", completed=False,
                              probability=None, full_flops=0, configuration="DEFER"),
                         dict(deferred, system_id="B")]}
        _, indexed = analysis._index({"heterogeneous": systems})
        counts = analysis.heterogeneous_state_counts(indexed, self.protocol)
        self.assertEqual(counts["A"]["0.25"]["requested_units"], 1)
        self.assertEqual(counts["A"]["0.25"]["requested_alias_rows"], 2)
        self.assertEqual(counts["A"]["0.25"]["completed_units"], 1)
        self.assertEqual(counts["A"]["0.50"]["deferred_units"], 1)
        self.assertEqual(counts["B"]["0.25"]["coverage"], 0)
        self.assertIsNone(counts["A"]["0"]["coverage"])
        self.assertEqual(sum(v["requested_units"] for v in counts["A"].values()), 2)
        changed = copy.deepcopy(systems)
        changed["B"][0]["resource_state"] = "0.75"
        _, indexed = analysis._index({"heterogeneous": changed})
        with self.assertRaisesRegex(ValueError, "differ across systems"):
            analysis.heterogeneous_state_counts(indexed, self.protocol)

    def test_factorial_paired_coverage_has_no_interval(self):
        left = {x["unit_id"]: x for x in (
            unit("a", "G1", 0, .2), unit("b", "G2", 1, .8), unit("c", "G3", 0, .1))}
        right = {x["unit_id"]: x for x in (
            unit("a", "G1", 0, .2), unit("b", "G2", 1, None), unit("c", "G3", 0, .1))}
        draws = analysis.bootstrap_indices(("G1", "G2", "G3"), self.protocol)
        factorial = analysis.paired_analysis(left, right, ("G1", "G2", "G3"),
                                             draws, self.protocol, factorial=True)
        coverage = factorial["intervals"]["coverage_difference"]
        self.assertAlmostEqual(coverage["estimate"], 1 / 3)
        self.assertIsNone(coverage["ci95"])
        self.assertTrue(coverage["structural_no_interval"])
        self.assertIsNotNone(factorial["intervals"]["total_full_flops_difference"]["ci95"])
        mixed = analysis.paired_analysis(left, right, ("G1", "G2", "G3"),
                                         draws, self.protocol, factorial=False)
        self.assertIsNotNone(mixed["intervals"]["coverage_difference"]["ci95"])

    def test_fresh_same_configuration_predictions_are_exact(self):
        first = {"exam_id": "a", "zone_id": "z1", "checkpoint_key": "nested",
                 "checkpoint_sha256": "frozen", "preprocessing": "bicubic",
                 "configuration": "w0.5_k2", "width": .5, "k": 2,
                 "frame_indices": "1 2", "probability": .7, "full_flops": 10,
                 "deferred": False}
        self.assertEqual(validation_run.assert_identical_predictions(
            [first, dict(first, system_id="E")]), 1)
        with self.assertRaisesRegex(RuntimeError, "unequal fresh predictions"):
            validation_run.assert_identical_predictions(
                [first, dict(first, probability=.7000000000000001)])
        with self.assertRaisesRegex(RuntimeError, "unequal fresh predictions"):
            validation_run.assert_identical_predictions(
                [first, dict(first, full_flops=11)])
        self.assertEqual(validation_run.assert_identical_predictions(
            [first, dict(first, exam_id="b")]), 0)


if __name__ == "__main__":
    unittest.main()
