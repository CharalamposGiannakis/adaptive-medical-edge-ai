"""F002c safety and transfer-learning checks; no image or weight downloads."""
import os
import math
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from experiments.F002c import run_f002c as runner
from experiments.F002c import model as model_module


def manifest_row(exam="A", partition="train", target="0"):
    return {"exam_id": exam, "zone_id": "z1", "partition": partition,
            "binary_target": target, "target_name": "normal" if target == "0" else "abnormal",
            "split_group": "G", "device": "0", "frame_count": "37",
            "frame_paths_sha256": "verified-inventory"}


class ArtifactIsolationTests(unittest.TestCase):
    def test_run_name_cannot_escape_f002c(self):
        for name in ("", ".", "..", "../F002", "seed/1", "seed\\1", "C:\\F002", "é"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.output_dir(name)
        path = runner.output_dir("tinyvit_seed2026-primary")
        self.assertEqual(path, runner.ART / "tinyvit_seed2026-primary")
        self.assertEqual(path.parent.resolve(), runner.ART.resolve())

    def test_selected_checkpoint_requires_matching_configuration_and_frozen_inputs(self):
        checkpoint = b"synthetic checkpoint: never deserialize"
        valid = {"status": "completed", "config": runner.CONFIG.copy(),
                 "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
                 "frozen_inputs_sha256": {"frozen_source": "accepted-hash"},
                 "environment": {"f002c_source_sha256": {"model.py": "accepted-implementation"}}}
        mutations = (
            {"status": "running"},
            {"config": valid["config"] | {"threshold": 0.7}},
            {"checkpoint_sha256": "wrong-checkpoint"},
            {"frozen_inputs_sha256": {"frozen_source": "changed-hash"}},
            {"environment": {"f002c_source_sha256": {"model.py": "changed-implementation"}}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "checkpoint.pt").write_bytes(checkpoint)
            for change in mutations:
                (out / "run.json").write_text(json.dumps(valid | change), encoding="utf-8")
                with self.subTest(change=change), \
                        patch.object(runner, "frozen_snapshot", return_value=valid["frozen_inputs_sha256"]), \
                        patch.object(runner, "environment", return_value=valid["environment"]), \
                        patch.object(runner, "TinyViTSequence") as construct, \
                        patch.object(runner.torch, "load") as deserialize, \
                        self.assertRaises(ValueError):
                    runner.load_selected(out, SimpleNamespace(type="cpu"))
                construct.assert_not_called()
                deserialize.assert_not_called()


class PartitionSafetyTests(unittest.TestCase):
    def test_loader_rejects_test_missing_and_mixed_partition_before_inventory(self):
        for rows in ([manifest_row(partition="test")],
                     [manifest_row(partition=None)], [],
                     [manifest_row(), manifest_row("B", "validation")]):
            with self.subTest(rows=rows), \
                    patch.object(model_module.pipe, "retained_frames") as inventory, \
                    patch.object(model_module.Image, "open") as open_image, \
                    self.assertRaises(ValueError):
                model_module.F002cSequences(rows)
            inventory.assert_not_called()
            open_image.assert_not_called()

    def test_forged_or_changed_manifest_rows_fail_before_inventory(self):
        original = manifest_row()
        changes = ({"exam_id": "test-id-forged-as-train"},
                   {"binary_target": "1"}, {"frame_paths_sha256": "changed"},
                   {"frame_count": "38"}, {"device": "4"})
        with patch.object(model_module.pipe, "load_inputs", return_value={"train": [original]}):
            for change in changes:
                with self.subTest(change=change), \
                        patch.object(model_module.pipe, "retained_frames") as inventory, \
                        self.assertRaises(ValueError):
                    model_module.F002cSequences([original | change])
                inventory.assert_not_called()
            with patch.object(model_module.pipe, "retained_frames") as inventory, \
                    self.assertRaises(ValueError):
                model_module.F002cSequences([original, original])
            inventory.assert_not_called()

    def test_validation_cannot_enable_training_augmentation(self):
        with patch.object(model_module.pipe, "retained_frames") as inventory, \
                self.assertRaises(ValueError):
            model_module.F002cSequences([manifest_row(partition="validation")], augment=True)
        inventory.assert_not_called()

    def test_evaluation_refuses_nonvalidation_before_model_or_frames(self):
        for partitions in (("test",), ("train",), (None,), ("validation", "test")):
            with self.subTest(partitions=partitions):
                model = MagicMock()
                sequences = SimpleNamespace(
                    rows=[{"partition": partition} for partition in partitions],
                    frames=MagicMock(side_effect=AssertionError("Images must not be read")),
                )
                with self.assertRaises(ValueError):
                    runner.evaluate(model, sequences, {}, SimpleNamespace(type="cpu"), workers=0)
                model.assert_not_called()
                sequences.frames.assert_not_called()


class FrozenDataContractTests(unittest.TestCase):
    def test_f002c_loader_keeps_nested_frame_selection_and_path_provenance(self):
        row = manifest_row()
        raw = Path("synthetic-raw")
        frames = tuple(raw / "A" / f"frame-{i}.jpg" for i in range(37))
        selections = []
        with patch.object(model_module.pipe, "load_inputs", return_value={"train": [row]}), \
                patch.object(model_module.pipe, "retained_frames", return_value=frames), \
                patch.object(model_module.Image, "open") as open_image:
            for k in (4, 8, 16):
                sequences = model_module.F002cSequences([row], k=k, raw=raw)
                indices, paths = sequences.selected[0]
                selections.append(set(indices))
                self.assertEqual(len(indices), k)
                self.assertEqual((indices[0], indices[-1]), (0, 36))
                self.assertEqual(paths, tuple(frames[index] for index in indices))
                identity = sequences.identity(0)
                self.assertEqual(identity["frame_indices"], " ".join(map(str, indices)))
                self.assertEqual(identity["frame_files"].split(";"),
                                 [path.relative_to(raw).as_posix() for path in paths])
            open_image.assert_not_called()
        self.assertLess(selections[0], selections[1])
        self.assertLess(selections[1], selections[2])

    def test_validation_averages_alias_probabilities_before_thresholding(self):
        import torch

        class SyntheticSequences:
            k = 16
            augment = False
            rows = [manifest_row("A", "validation", "1"),
                    manifest_row("B", "validation", "1"),
                    manifest_row("C", "validation", "0")]

            def __len__(self):
                return len(self.rows)

            def frames(self, index):
                # Two aliases straddle the threshold. Their recording receives
                # one abnormal vote at probability 0.5, instead of two votes.
                return torch.tensor([-math.log(4), math.log(4), math.log(2 / 3)][index]).reshape(1, 1, 1, 1)

            def identity(self, index):
                return {"frame_indices": str(index), "frame_files": f"synthetic-{index}.jpg"}

        class LogitModel(torch.nn.Module):
            def forward(self, frames):
                return frames.flatten(start_dim=1)[:, 0]

        unit_of = {("A", "z1"): "shared", ("B", "z1"): "shared", ("C", "z1"): "unique"}
        result = runner.evaluate(LogitModel(), SyntheticSequences(), unit_of, torch.device("cpu"), workers=0)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(len(result["units"]), 2)
        self.assertEqual(result["duplicate_aware"]["tp"], 1)
        self.assertEqual(result["duplicate_aware"]["tn"], 1)
        self.assertEqual(result["duplicate_aware"]["macro_f1"], 1.0)
        self.assertEqual(result["row_level"]["fn"], 1)
        shared = next(unit for unit in result["units"] if len(unit["aliases"]) == 2)
        self.assertAlmostEqual(shared["probability"], 0.5, places=6)


class TinyViTModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.old_threads))
        torch.manual_seed(2026)
        cls.model = model_module.TinyViTSequence(
            pretrained=False, drop_path_rate=0.0, head_dropout=0.0)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def test_real_encoder_supports_sequence_batches_and_all_nested_k(self):
        torch = self.torch
        self.model.eval()
        with torch.inference_mode():
            for batch, k in ((2, 4), (1, 8), (1, 16)):
                with self.subTest(batch=batch, k=k):
                    logits = self.model(torch.zeros(batch, k, 3, 224, 224))
                    self.assertEqual(tuple(logits.shape), (batch,))
                    self.assertTrue(torch.isfinite(logits).all().item())

    def test_backprop_reaches_encoder_and_head_without_changing_bn_statistics(self):
        torch = self.torch
        self.model.eval().train()
        self.model.zero_grad(set_to_none=True)
        bn_modules = [module for module in self.model.encoder.modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        self.assertTrue(bn_modules)
        before = [(module.running_mean.clone(), module.running_var.clone(),
                   module.num_batches_tracked.clone()) for module in bn_modules]
        logits = self.model(torch.randn(2, 4, 3, 224, 224))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        loss.backward()
        for module, (mean, variance, batches) in zip(bn_modules, before):
            self.assertFalse(module.training)
            self.assertTrue(module.weight.requires_grad)
            self.assertTrue(module.bias.requires_grad)
            self.assertTrue(torch.equal(module.running_mean, mean))
            self.assertTrue(torch.equal(module.running_var, variance))
            self.assertTrue(torch.equal(module.num_batches_tracked, batches))
        for part in (self.model.encoder, self.model.head):
            gradients = [parameter.grad for parameter in part.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.)
        self.assertTrue(any(module.weight.grad is not None and
                            module.weight.grad.abs().sum().item() > 0 for module in bn_modules))

    def test_optimizer_covers_each_trainable_parameter_once_and_preserves_no_decay(self):
        optimizer = model_module.build_optimizer(
            self.model, backbone_lr=5e-5, head_lr=5e-4, weight_decay=0.01)
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        expected = {id(parameter) for parameter in self.model.parameters() if parameter.requires_grad}
        self.assertEqual({id(parameter) for parameter in parameters}, expected)
        group_of = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        attention_biases = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = group_of[id(parameter)]
            self.assertEqual(group["lr"], 5e-5 if name.startswith("encoder.") else 5e-4)
            if parameter.ndim <= 1 or name.endswith(".bias") or "attention_biases" in name:
                self.assertEqual(group["weight_decay"], 0.0, name)
            if "attention_biases" in name:
                attention_biases.append(name)
        self.assertTrue(attention_biases)
        self.assertTrue(any(group["weight_decay"] == 0.01 for group in optimizer.param_groups))

    def test_invalid_sequence_shapes_fail_clearly(self):
        torch = self.torch
        for shape in ((1, 3, 224, 224), (0, 4, 3, 224, 224),
                      (1, 5, 3, 224, 224), (1, 4, 3, 128, 128)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.model(torch.empty(shape))


if __name__ == "__main__":
    unittest.main()
