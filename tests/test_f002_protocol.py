import csv
import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

from src.f002_protocol import (aggregate_validation, nested_frame_indices,
                               recording_units, training_weights, verify_t006,
                               optimization_loss, reported_weighted_train_loss,
                               benchmark_request_ids, reduce_k16_candidates,
                               seed43_confirmation_required, PRIMARY_BENCHMARK_FLAGS,
                               binary_metrics)
from src.f002_flops import MODEL_IDS, WEIGHT_IDS, BOOKKEEPING_ALLOWLIST


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

    def test_exhaustive_retained_frame_length_range(self):
        for n in range(17, 1091):
            four, eight, sixteen = (nested_frame_indices(n, k) for k in (4, 8, 16))
            self.assertEqual((len(four), len(eight), len(sixteen)), (4, 8, 16))
            self.assertLess(set(four), set(eight))
            self.assertLess(set(eight), set(sixteen))
            self.assertEqual((sixteen[0], sixteen[-1]), (0, n - 1))


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

    def test_fixed_denominator_coefficients_ignore_shuffle_composition(self):
        import torch

        weights = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for order in ((0, 1, 2, 3, 4, 5, 6, 7, 8),
                      (0, 2, 3, 4, 5, 6, 7, 8, 1)):
            losses = torch.arange(1.0, 10.0, requires_grad=True)
            step_totals = []
            for start in (0, 8):
                indices = order[start:start + 8]
                # Microbatch 2, accumulated with the same fixed denominator.
                micros = [optimization_loss([losses[i] for i in indices[j:j+2]],
                                            [weights[i] for i in indices[j:j+2]])
                          for j in range(0, len(indices), 2)]
                step_totals.append(sum(micros))
            sum(step_totals).backward()
            self.assertEqual(losses.grad.tolist(), [weight / 8 for weight in weights])
        self.assertAlmostEqual(reported_weighted_train_loss(list(range(1, 10)), weights),
                               sum(weight * loss for weight, loss in zip(weights, range(1, 10))) / sum(weights))
        self.assertAlmostEqual(optimization_loss([2.0], [0.5]), 0.125)


def evidence(name, f1, flops, latency, throughput, normal=0.5, abnormal=0.5):
    return {"id": name, "macro_f1": f1, "flops": flops,
            "model_only_median_latency_ms": latency,
            "model_only_sequence_throughput_s": throughput,
            "normal_recall": normal, "abnormal_recall": abnormal,
            "benchmark_platform": "DAS-5", "execution_device": "cuda:0",
            "gpu_model": "test_GPU", "node_class": "test_node", "hostname": "test_host",
            "benchmark_flags": PRIMARY_BENCHMARK_FLAGS.copy()}


class SelectionTests(unittest.TestCase):
    def test_runtime_advantage_blocks_k16_dominance_removal(self):
        a = evidence("a", .8, 100, 10, 100)
        b = evidence("b", .7, 120, 8, 120)
        result = reduce_k16_candidates([a, b], .4)
        self.assertEqual(set(result["retained"]), {"a", "b"})
        self.assertEqual(result["dominated"], [])
        b["model_only_median_latency_ms"] = 9.5
        b["model_only_sequence_throughput_s"] = 105
        result = reduce_k16_candidates([a, b], .4)
        self.assertEqual(result["retained"], ["a"])
        self.assertEqual(result["dominated"], ["b"])

    def test_second_slot_runtime_challenger_and_validity(self):
        a = evidence("a", .80, 100, 10, 100)
        b = evidence("b", .75, 50, 20, 50)
        c = evidence("c", .77, 80, 15, 70)
        result = reduce_k16_candidates([a, b, c], .4)
        self.assertEqual(result["retained"], ["a", "c"])
        c["abnormal_recall"] = 0
        result = reduce_k16_candidates([a, b, c], .4)
        self.assertEqual(set(result["retained"]), {"a", "b"})
        self.assertEqual(result["invalid"], ["c"])

    def test_seed43_trigger_is_bounded(self):
        finalists = [{"macro_f1": .70}, {"macro_f1": .719}]
        self.assertTrue(seed43_confirmation_required(finalists, None))
        self.assertFalse(seed43_confirmation_required(finalists, "a"))
        self.assertFalse(seed43_confirmation_required(finalists[:1], None))
        self.assertFalse(seed43_confirmation_required([finalists[0], {"macro_f1": .72}], None))

    def test_reduction_rejects_unmatched_or_cpu_runtime(self):
        a = evidence("a", .8, 100, 10, 100)
        b = evidence("b", .7, 120, 8, 120)
        b["execution_device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "DAS-5 CUDA"):
            reduce_k16_candidates([a, b], .4)
        b["execution_device"] = "cuda:0"
        b["gpu_model"] = "different_GPU"
        with self.assertRaisesRegex(ValueError, "one GPU model"):
            reduce_k16_candidates([a, b], .4)


class ConfigTests(unittest.TestCase):
    def test_locked_values_and_fingerprints(self):
        config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(config["sampling"]["response_k"], [4, 8, 16])
        self.assertEqual(config["sampling"]["initial_k"], 16)
        self.assertEqual(config["preprocessing"]["target_size"], [224, 224])
        self.assertEqual(config["training"]["seed"], 42)
        self.assertEqual(config["f002b"]["seed"], 2026)
        self.assertEqual(config["model"]["binary_threshold"], 0.5)
        self.assertEqual(binary_metrics([1], [0.5])["tp"], 1)
        self.assertTrue(config["model"]["mobilenet_original_classifier_and_dropout_removed"])
        self.assertEqual(config["training"]["gradient_clipping"], "none")
        self.assertEqual(config["training"]["batchnorm_training"],
                         "normal_train_mode_update_running_statistics")
        self.assertEqual(config["t006_expected"]["minimum_frames"], 17)
        self.assertFalse(config["f002b"]["test_inference_during_t008"])
        self.assertIn("no_seed_average", config["selection"]["confirm_role"])
        self.assertIn("stop", config["f002b"]["failure"])
        self.assertEqual(config["duplicates"]["optimization_loss"],
                         "sum_effective_batch_weight_times_per_sequence_BCE_divided_by_fixed_8_including_last_partial_window")
        self.assertEqual(config["training"]["effective_batch_sequences"], 8)
        self.assertEqual(config["benchmark"]["flags"], PRIMARY_BENCHMARK_FLAGS)
        self.assertEqual(config["benchmark"]["primary_platform"].split("_")[0], "DAS-5")
        self.assertEqual(config["benchmark"]["deit_runtime"]["equivalence_max_abs_logit_difference"], 1e-5)
        self.assertEqual(set(config["flops"]["bookkeeping_allowlist"]), BOOKKEEPING_ALLOWLIST)
        self.assertEqual({name: item["model"] for name, item in config["candidates"].items()}, MODEL_IDS)
        self.assertEqual({name: item["weights"] for name, item in config["candidates"].items()}, WEIGHT_IDS)
        self.assertEqual(hashlib.sha256((ROOT / "docs" / "adaptivity_project_notes.md").read_bytes()).hexdigest(),
                         config["course_notes_sha256"])
        verify_t006(PROTOCOL, T006)

    def test_benchmark_request_list_is_frozen_manifest_prefix(self):
        validation = rows("validation.csv")
        requests = benchmark_request_ids(validation)
        self.assertEqual(len(requests["timed"]), 200)
        self.assertEqual(requests["warmup"], requests["timed"][:50])
        self.assertEqual(requests["timed"][0], (validation[0]["exam_id"], validation[0]["zone_id"]))
        self.assertEqual(requests["timed"][-1], (validation[199]["exam_id"], validation[199]["zone_id"]))
        self.assertEqual(requests, benchmark_request_ids(validation))


if __name__ == "__main__":
    unittest.main()
