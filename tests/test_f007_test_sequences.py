"""Authorized F007 test-loader checks using validation images or mock tensors only."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from experiments.F002c.model import F002cSequences
from experiments.F003.adaptive import F003Sequences, frame_indices
from experiments.F007 import phase_a1, test_sequences
from src import f002_pipeline as pipe
from src.f007_protocol import load_protocol
from tests.test_f007_phase_a1 import BASELINE, points_from_screen


class _Static(torch.nn.Module):
    def forward(self, frames):
        return torch.zeros((frames.shape[0], 1))


class _Encoder(torch.nn.Module):
    def forward(self, frames):
        return torch.zeros((frames.shape[0], 1))


class _Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.head = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.head.weight)


class TestSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.validation_row = pipe.load_inputs()["validation"][0]
        cls.test_row = dict(cls.validation_row, partition="test")

    def _authorized(self):
        test_partition = {"test": [self.test_row]}
        return (patch.object(test_sequences, "require_test_authorization"),
                patch.object(test_sequences, "verify_t006", return_value=test_partition))

    def test_validation_tensor_equivalence_and_f002c_guard(self):
        with self.assertRaises(ValueError):
            F002cSequences([self.test_row])
        original = F003Sequences([self.validation_row], 224)
        auth, manifest = self._authorized()
        with auth, manifest:
            authorized = test_sequences.AuthorizedTestTinyViTSequences([self.test_row],
                                                                        self.protocol)
        self.assertEqual(original.selected, authorized.selected)
        for k in (1, 4, 8, 16):
            self.assertEqual(original.positions(0, k), authorized.positions(0, k))
            self.assertTrue(torch.equal(original.frames(0, k), authorized.frames(0, k)))
        indices = frame_indices(int(self.test_row["frame_count"]), 16)
        self.assertTrue(torch.equal(original.frames_at(0, (indices[0], indices[7])),
                                    authorized.frames_at(0, (indices[0], indices[7]))))

    def test_preflight_resolves_both_inventories_without_decoding(self):
        with patch.object(phase_a1, "require_test_authorization"), \
                patch.object(phase_a1, "verify_t006", return_value={"test": [self.test_row]}), \
                patch.object(test_sequences, "require_test_authorization"), \
                patch.object(test_sequences, "verify_t006", return_value={"test": [self.test_row]}), \
                patch("PIL.Image.open", side_effect=AssertionError("image decoded")):
            result = phase_a1.preflight_test_inventories([self.test_row], self.protocol)
        self.assertEqual(result["mobilenet_k16_inventories"], 1)
        self.assertEqual(result["tinyvit_k16_inventories"], 1)
        self.assertEqual(result["decoded_images"], 0)
        altered = copy.deepcopy(self.test_row)
        altered["frame_paths_sha256"] = "0" * 64
        with patch.object(phase_a1, "require_test_authorization"), \
                patch.object(phase_a1, "verify_t006", return_value={"test": [altered]}), \
                patch("PIL.Image.open", side_effect=AssertionError("image decoded")):
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                phase_a1.preflight_test_inventories([altered], self.protocol)

    def test_real_canonical_dispatch_all_systems_and_supplements_with_mock_models(self):
        from experiments.F004 import family, run_f004
        from experiments.F005 import prune, run_f005b
        from experiments.F003 import adaptive

        points = points_from_screen()
        plans = [phase_a1.route(system, "1.00", self.protocol, points, BASELINE)
                 for system in self.protocol["systems"]]
        plans.extend((phase_a1.Plan("f002c_original_k4", 1, 1_000_000_000_000,
                                    "f002c_original_tinyvit", "original_tinyvit_k4",
                                    None, 4, 5149017395, None),
                      phase_a1.Plan("f005b_nested_w1.0_k16", 1, 1_000_000_000_000,
                                    "f005b_nested_tinyvit", "w1.0_k16",
                                    1.0, 16, 20596068563, None)))
        with patch.object(phase_a1, "require_test_authorization"), \
                patch.object(phase_a1, "verify_t006", return_value={"test": [self.test_row]}), \
                patch.object(test_sequences, "require_test_authorization"), \
                patch.object(test_sequences, "verify_t006", return_value={"test": [self.test_row]}), \
                patch.object(run_f004, "load_static_f002b", return_value=_Static()), \
                patch.object(family, "load_family", return_value=_Static()), \
                patch.object(run_f005b, "load_deployable", return_value=_Nested()), \
                patch.object(prune, "set_width"), \
                patch.object(pipe.Sequences, "frames", return_value=torch.zeros(16, 3, 224, 224)), \
                patch.object(test_sequences.AuthorizedTestTinyViTSequences, "frames",
                             side_effect=lambda _index, k=16: torch.zeros(k, 3, 224, 224)), \
                patch.object(test_sequences.AuthorizedTestTinyViTSequences, "frames_at",
                             side_effect=lambda _index, numbers: torch.zeros(len(numbers), 3, 224, 224)), \
                patch.object(adaptive, "F003Sequences",
                             side_effect=AssertionError("validation loader used for test")):
            predictor = phase_a1.CanonicalModels(self.protocol, partition="test")
            for plan in plans:
                self.assertFalse(plan.deferred)
                probability, details = predictor(self.test_row, plan)
                self.assertEqual(probability, 0.5)
                self.assertEqual(details["new_encoded_frames"], plan.k)
            self.assertEqual(set(predictor.models),
                             {"f002b_mobilenetv2", "f002c_original_tinyvit",
                              "f005b_nested_tinyvit"})


if __name__ == "__main__":
    unittest.main()
