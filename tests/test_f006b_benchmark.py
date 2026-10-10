"""Focused fail-closed checks for T012b's model-only CPU timing helper."""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch import nn

from experiments.F006b import benchmark_cpu as bench


@dataclass(frozen=True)
class Point:
    name: str
    width: float
    k: int
    flops: int


class ToyEncoder(nn.Module):
    def __init__(self, batch_sensitive: bool = False):
        super().__init__()
        self.active_width = 1.0
        self.batch_sensitive = batch_sensitive

    def forward(self, frames):
        features = frames.mean(dim=(2, 3))[:, :1] * self.active_width
        if self.batch_sensitive:
            features = features + frames.shape[0] / 10
        return features


class ToyModel(nn.Module):
    def __init__(self, batch_sensitive: bool = False):
        super().__init__()
        self.encoder = ToyEncoder(batch_sensitive)
        self.head = nn.Identity()


def toy_inputs() -> bench.Inputs:
    frames = tuple(torch.arange(16, dtype=torch.float32).view(16, 1, 1, 1) + i
                   for i in range(2))
    indices = tuple({8: torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])} for _ in frames)
    return bench.Inputs(("a", "b"), frames, indices, "toy")


class BenchmarkTests(unittest.TestCase):
    def test_direct_chunked_agreement_for_width_and_input_changes(self):
        model = ToyModel()
        points = (Point("w0.5_k8", 0.5, 8, 1), Point("w1.0_k8", 1.0, 8, 2))
        with patch.object(bench, "set_width", lambda encoder, width: setattr(encoder, "active_width", width)):
            differences = bench.verify_chunk_agreement(model, points, toy_inputs(), 4)
        self.assertEqual(differences, {point.name: 0.0 for point in points})

    def test_batch_dependent_chunking_fails_closed(self):
        model = ToyModel(batch_sensitive=True)
        point = Point("w0.5_k8", 0.5, 8, 1)
        with patch.object(bench, "set_width", lambda encoder, width: setattr(encoder, "active_width", width)):
            with self.assertRaisesRegex(RuntimeError, "Direct/chunked inference differs"):
                bench.verify_chunk_agreement(model, (point,), toy_inputs(), 4)

    def test_switching_uses_identical_request_multiset(self):
        points = (Point("a", 0.5, 2, 1), Point("b", 1.0, 8, 2),
                  Point("c", 1.0, 16, 3))
        interleaved, grouped = bench.switching_requests(points, 20)
        self.assertEqual(sorted(interleaved), sorted(grouped))
        self.assertEqual((interleaved, grouped), bench.switching_requests(points, 20))
        self.assertGreater(sum(a[1] != b[1] for a, b in zip(interleaved, interleaved[1:])),
                           sum(a[1] != b[1] for a, b in zip(grouped, grouped[1:])))

    def test_validation_request_count_is_bounded_before_manifest_access(self):
        with self.assertRaises(ValueError):
            bench.validation_inputs((2, 8), 201)


if __name__ == "__main__":
    unittest.main()
