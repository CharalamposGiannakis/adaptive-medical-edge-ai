import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import tempfile
import unittest
from pathlib import Path

from src import f002_pipeline as pipe


def evidence(f1, flops, latency=10.0, throughput=100.0):
    return {"macro_f1": f1, "flops": flops, "model_only_median_latency_ms": latency,
            "model_only_sequence_throughput_s": throughput}


class SelectionRuleTests(unittest.TestCase):
    def test_checkpoint_prefers_f1_then_loss_then_earlier(self):
        best = {"macro_f1": 0.7, "loss": 0.5}
        self.assertTrue(pipe.checkpoint_better({"macro_f1": 0.71, "loss": 0.9}, best))
        self.assertTrue(pipe.checkpoint_better({"macro_f1": 0.7, "loss": 0.4}, best))
        self.assertFalse(pipe.checkpoint_better({"macro_f1": 0.7, "loss": 0.5}, best))
        self.assertFalse(pipe.checkpoint_better({"macro_f1": 0.69, "loss": 0.1}, best))
        self.assertTrue(pipe.checkpoint_better(best, None))

    def test_rescue_needs_one_percent_drop_by_epoch4(self):
        losses = lambda *values: [{"train_loss": v} for v in values]  # noqa: E731
        self.assertTrue(pipe.rescue_triggered(losses(1.0, 0.995, 0.999, 1.2)))
        self.assertFalse(pipe.rescue_triggered(losses(1.0, 1.1, 0.99, 1.2)))
        self.assertFalse(pipe.rescue_triggered(losses(1.0, 1.0, 1.0)))  # before epoch 4

    def test_validity_gate(self):
        metrics = {"macro_f1": 0.5, "normal_recall": 0.9, "abnormal_recall": 0.1}
        self.assertTrue(pipe.is_valid(metrics, 0.41))
        self.assertFalse(pipe.is_valid(metrics, 0.5))
        self.assertFalse(pipe.is_valid(metrics | {"abnormal_recall": 0.0}, 0.41))

    def test_final_dominance_every_k_and_runtime_contradiction(self):
        a = {k: evidence(0.80, 100 * k) for k in (4, 8, 16)}
        b = {k: evidence(0.75, 200 * k) for k in (4, 8, 16)}
        self.assertEqual(pipe.final_dominance({"a": a, "b": b}), "a")
        b_fast = {k: evidence(0.75, 200 * k, latency=8.9) for k in (4, 8, 16)}
        self.assertIsNone(pipe.final_dominance({"a": a, "b": b_fast}))
        b_mixed = dict(b) | {4: evidence(0.85, 800)}
        self.assertIsNone(pipe.final_dominance({"a": a, "b": b_mixed}))
        self.assertIsNone(pipe.final_dominance({"a": a, "b": a}))
        self.assertEqual(pipe.final_dominance({"only": a}), "only")


class PreprocessingTests(unittest.TestCase):
    def test_full_square_bilinear_imagenet_normalization(self):
        import torch
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            Image.new("L", (512, 512), color=128).save(path)
            tensor = pipe.preprocess_frame(path)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        expected = torch.tensor([(128 / 255 - m) / s for m, s in zip(pipe.MEAN, pipe.STD)])
        self.assertTrue(torch.allclose(tensor[:, 100, 100], expected, atol=1e-6))


class ReferenceTests(unittest.TestCase):
    def test_all_normal_and_metadata_use_component_units(self):
        inputs = pipe.load_inputs()
        all_normal = pipe.all_normal_reference(inputs)["duplicate_aware"]
        self.assertEqual(all_normal["abnormal_recall"], 0.0)
        self.assertEqual(all_normal["tp"] + all_normal["fp"], 0)
        self.assertEqual(sum(all_normal[c] for c in ("tn", "fp", "fn", "tp")), 254)
        metadata = pipe.metadata_reference(inputs)
        self.assertEqual(len(metadata["units"]), 254)
        self.assertEqual(len(metadata["rows"]), 265)


@unittest.skipUnless((pipe.RAW / "Pt121").is_dir(), "OpenPOCUS raw frames not downloaded")
class SmokeTrainingTests(unittest.TestCase):
    def test_cpu_training_loop_writes_checkpoint_and_predictions(self):
        import torch

        inputs = pipe.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            run = pipe.train_candidate("mobilenet_v2", 42, Path(tmp), torch.device("cpu"), 0.4,
                                       inputs=inputs, train_rows=inputs["train"][:4],
                                       val_rows=inputs["validation"][:3], epochs_max=1,
                                       pretrained=False)
            attempt = Path(tmp) / "lr1e-4"
            self.assertTrue((attempt / "checkpoint.pt").exists())
            self.assertTrue((attempt / "validation_selected.csv").exists())
        self.assertEqual(run["attempts"], ["lr1e-4"])
        self.assertEqual(run["selected"]["epoch"], 1)
        self.assertTrue(run["environment"]["source_sha256"])


if __name__ == "__main__":
    unittest.main()
