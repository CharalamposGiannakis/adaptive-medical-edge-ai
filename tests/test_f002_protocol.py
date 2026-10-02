import csv
import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

from src.f002_protocol import (aggregate_validation, nested_frame_indices,
                               recording_units, training_weights, verify_t006)
from src.f002_flops import MODEL_IDS, WEIGHT_IDS


ROOT = Path(__file__).resolve().parents[1]
T006 = ROOT / "data" / "openpocus" / "t006"
PROTOCOL = ROOT / "configs" / "f002_protocol.json"


def rows(name):
    with (T006 / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SamplerTests(unittest.TestCase):
    def test_nested_unique_spread_deterministic(self):
        for n in (17, 18, 63, 1090):
            selections = [nested_frame_indices(n, k) for k in (4, 8, 16)]
            for k, selected in zip((4, 8, 16), selections):
                self.assertEqual(len(selected), k)
                self.assertEqual(len(set(selected)), k)
                self.assertEqual(selected[0], 0)
                self.assertEqual(selected[-1], n - 1)
                self.assertEqual(selected, nested_frame_indices(n, k))
            self.assertLess(set(selections[0]), set(selections[1]))
            self.assertLess(set(selections[1]), set(selections[2]))
            self.assertEqual(selections[0], (0, (n-1)//3, 2*(n-1)//3, n-1))

    def test_invalid_size_or_k(self):
        for n, k in ((15, 4), (17, 5), (2, 16)):
            with self.assertRaises(ValueError):
                nested_frame_indices(n, k)


class DuplicateTests(unittest.TestCase):
    def test_actual_frozen_graph_and_weights(self):
        partitions = verify_t006(PROTOCOL, T006)
        units = recording_units(partitions, rows("shared_recording_edges.csv"),
                                rows("exclusions.csv"))
        self.assertEqual(len(units), 1762)
        self.assertEqual(len({units[(r["exam_id"], r["zone_id"])]
                              for r in partitions["validation"]}), 254)
        for partition_rows in partitions.values():
            weights = training_weights(partition_rows, units)
            by_unit = defaultdict(float)
            for row in partition_rows:
                node = row["exam_id"], row["zone_id"]
                by_unit[units[node]] += weights[node]
            self.assertTrue(all(abs(value - 1.0) < 1e-12 for value in by_unit.values()))
        self.assertEqual(training_weights(partitions["train"], units)[("ED1", "z7")], 1.0)
        conflicts = {(r["exam_id"], r["zone_id"]) for r in rows("exclusions.csv")
                     if r["exclusion_reason"].startswith("t006:")}
        self.assertEqual(conflicts, {("Pt147", "z6"), ("Pt177", "z8"), ("Pt81", "z8")})
        self.assertFalse(conflicts & units.keys())

    def test_missing_alias_abstains_and_aggregation_preserves_rows(self):
        a = {"exam_id": "A", "zone_id": "z1", "partition": "validation",
             "binary_target": "1", "split_group": "G"}
        b = {**a, "exam_id": "B"}
        c = {**a, "exam_id": "C"}
        missing = {"exam_id": "M", "zone_id": "z1",
                   "exclusion_reason": "t004:unavailable_label"}
        edges = [{"exam_a": "A", "zone_a": "z1", "exam_b": "B", "zone_b": "z1"},
                 {"exam_a": "B", "zone_a": "z1", "exam_b": "M", "zone_b": "z1"}]
        mapping = recording_units({"validation": [a, b, c]}, edges, [missing])
        weights = training_weights([a, b, c], mapping)
        self.assertEqual(weights[("A", "z1")], 0.5)
        self.assertEqual(weights[("B", "z1")], 0.5)
        self.assertEqual(weights[("C", "z1")], 1)
        predictions = [{**a, "probability": 0.6, "logit": 0.405},
                       {**b, "probability": 0.8, "logit": 1.386},
                       {**c, "probability": 0.9, "logit": 2.197}]
        result = aggregate_validation([a, b, c], predictions, mapping)
        self.assertIs(result["rows"], predictions)
        self.assertEqual(len(result["units"]), 2)
        self.assertIn(0.7, [round(row["probability"], 10) for row in result["units"]])
        self.assertEqual(result["duplicate_aware"]["tp"], 2)
        self.assertEqual(result["row_level"]["tp"], 3)
        with self.assertRaises(ValueError):
            aggregate_validation([a, b, c], [predictions[0], {**predictions[1],
                                  "binary_target": "0"}, predictions[2]], mapping)

    def test_conflict_component_must_not_enter_retained_units(self):
        a = {"exam_id": "A", "zone_id": "z1", "partition": "train",
             "binary_target": "1", "split_group": "G"}
        b = {**a, "exam_id": "B", "binary_target": "0"}
        edge = {"exam_a": "A", "zone_a": "z1", "exam_b": "B", "zone_b": "z1"}
        with self.assertRaises(ValueError):
            recording_units({"train": [a, b]}, [edge], [])
        with self.assertRaises(ValueError):
            recording_units({"train": [a]}, [edge], [{**b, "exclusion_reason":
                             "t006:shared_recording_conflicting_t004_semantics"}])


class ConfigTests(unittest.TestCase):
    def test_locked_values_and_fingerprints(self):
        config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(config["sampling"]["response_k"], [4, 8, 16])
        self.assertEqual(config["sampling"]["initial_k"], 16)
        self.assertEqual(config["preprocessing"]["target_size"], [224, 224])
        self.assertEqual(config["training"]["seed"], 42)
        self.assertEqual(config["f002b"]["seed"], 2026)
        self.assertEqual(config["model"]["binary_threshold"], 0.5)
        self.assertFalse(config["f002b"]["test_inference_during_t008"])
        self.assertEqual({name: item["model"] for name, item in config["candidates"].items()}, MODEL_IDS)
        self.assertEqual({name: item["weights"] for name, item in config["candidates"].items()}, WEIGHT_IDS)
        self.assertEqual(hashlib.sha256((ROOT / "docs" / "adaptivity_project_notes.md").read_bytes()).hexdigest(),
                         config["course_notes_sha256"])
        verify_t006(PROTOCOL, T006)


if __name__ == "__main__":
    unittest.main()
