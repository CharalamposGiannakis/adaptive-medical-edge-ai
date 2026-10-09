"""Focused F006b checks for the frozen nested model's serving path."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from experiments.F003.adaptive import frame_indices
from experiments.F005 import prune
from experiments.F005.run_f005b import CANONICAL_NESTED
from experiments.F006b.runtime import NestedClipRuntime, direct_predict, load_runtime


class _ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.width = 1.0
        self.batches: list[tuple[float, int]] = []

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        self.batches.append((self.width, len(frames)))
        return (frames.mean(dim=(1, 2, 3)) * self.scale * self.width).unsqueeze(1)


class _ToyNestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ToyEncoder()
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.head.weight.fill_(1)
        self.eval()


def _toy_set_width(encoder: nn.Module, width: float) -> None:
    if width not in prune.WIDTHS:
        raise ValueError("Invalid width")
    getattr(encoder, "base", encoder).width = width
    for bn in getattr(encoder, "_width_bns", ()):
        bn.key = str(width)


def _clip(offset: float = 0.0) -> torch.Tensor:
    levels = (torch.arange(40, dtype=torch.float32) + offset) / 40
    return levels[:, None, None, None].expand(40, 3, 224, 224)


class NestedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.width_patch = patch.object(prune, "set_width", _toy_set_width)
        self.width_patch.start()
        self.addCleanup(self.width_patch.stop)
        self.model = _ToyNestedModel()
        self.frames = _clip()
        self.load = lambda numbers: self.frames[numbers]

    def test_same_width_upgrade_reuses_only_previously_encoded_frames(self):
        runtime = NestedClipRuntime(self.model, 40, self.load)
        first = runtime.predict(0.5, 2)
        larger = runtime.predict(0.5, 8)
        smaller = runtime.predict(0.5, 2)
        self.assertEqual([r.new_frames for r in (first, larger, smaller)], [2, 6, 0])
        self.assertEqual([r.cached_frames for r in (first, larger, smaller)], [0, 2, 2])
        self.assertEqual([r.cache_entries for r in (first, larger, smaller)], [2, 8, 8])
        self.assertEqual(self.model.encoder.batches, [(0.5, 2), (0.5, 6)])
        self.assertEqual(smaller.logit, first.logit)
        self.assertGreaterEqual(first.elapsed_ms, 0)

    def test_width_changes_do_not_reuse_the_wrong_features(self):
        runtime = NestedClipRuntime(self.model, 40, self.load)
        narrow = runtime.predict(0.5, 2)
        wide = runtime.predict(1.0, 2)
        back = runtime.predict(0.5, 2)
        self.assertEqual([r.new_frames for r in (narrow, wide, back)], [2, 2, 0])
        self.assertEqual([r.cache_entries for r in (narrow, wide, back)], [2, 4, 4])
        self.assertNotEqual(narrow.logit, wide.logit)
        self.assertEqual(back.logit, narrow.logit)
        self.assertEqual(self.model.encoder.batches, [(0.5, 2), (1.0, 2)])

    def test_every_cached_prediction_matches_direct_inference(self):
        runtime = NestedClipRuntime(self.model, 40, self.load)
        steps = [(1.0, 4), (0.5, 8), (0.5, 16), (1.0, 8), (0.25, 1)]
        actual = [runtime.predict(width, k) for width, k in steps]
        self.assertEqual([r.new_frames for r in actual], [4, 8, 8, 4, 1])
        for (width, k), result in zip(steps, actual):
            self.assertEqual(result.frame_indices, frame_indices(40, k))
            self.assertAlmostEqual(result.logit, direct_predict(self.model, 40, self.load, width, k), places=6)
            self.assertEqual(result.predicted_class, int(result.probability_abnormal >= 0.5))

    def test_reset_clip_clears_cache_and_changes_prediction(self):
        runtime = NestedClipRuntime(self.model, 40, self.load)
        first = runtime.predict(0.5, 2)
        other = _clip(100)
        runtime.reset_clip(40, lambda numbers: other[numbers])
        second = runtime.predict(0.5, 2)
        self.assertEqual((first.new_frames, second.new_frames), (2, 2))
        self.assertEqual(second.cache_entries, 2)
        self.assertNotEqual(first.logit, second.logit)

    def test_chunking_changes_batching_without_changing_prediction(self):
        self.model.encoder._width_bns = [SimpleNamespace(key="unset")]
        runtime = NestedClipRuntime(self.model, 40, self.load, chunk_size=4)
        result = runtime.predict(0.75, 8)
        self.assertEqual(self.model.encoder.batches, [(0.75, 4), (0.75, 4)])
        self.assertEqual(self.model.encoder._width_bns[0].key, "0.75")
        direct = direct_predict(self.model, 40, self.load, 0.75, 8, chunk_size=4)
        self.assertAlmostEqual(result.logit, direct, places=6)
        self.assertEqual(self.model.encoder.batches[-2:], [(0.75, 4), (0.75, 4)])

    def test_invalid_requests_and_inputs_fail_closed(self):
        runtime = NestedClipRuntime(self.model, 40, self.load)
        with self.assertRaises(ValueError):
            runtime.predict(0.6, 2)
        with self.assertRaises(ValueError):
            runtime.predict(0.5, 3)
        with self.assertRaises(ValueError):
            runtime.reset_clip(15, self.load)
        with self.assertRaises(ValueError):
            NestedClipRuntime(self.model, 40, self.load, chunk_size=0)
        with self.assertRaises(ValueError):
            NestedClipRuntime(self.model, 40, lambda numbers: torch.zeros(len(numbers), 3, 16, 16)).predict(0.5, 2)
        self.model.train()
        with self.assertRaises(ValueError):
            NestedClipRuntime(self.model, 40, self.load)

    def test_runtime_loader_uses_the_canonical_bundle(self):
        sentinel = object()
        with patch("experiments.F006b.runtime.load_deployable", return_value=sentinel) as loader:
            self.assertIs(load_runtime(device="cpu"), sentinel)
        loader.assert_called_once_with(device="cpu")


@unittest.skipUnless((CANONICAL_NESTED / "checkpoint.pt").is_file(), "Frozen F005b bundle unavailable")
class CanonicalRuntimeParityTests(unittest.TestCase):
    def test_canonical_model_k_upgrade_and_chunked_direct_match(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = load_runtime()
            frames = torch.linspace(-1, 1, 16).view(16, 1, 1, 1).expand(16, 3, 224, 224)
            lookup = {index: position for position, index in enumerate(frame_indices(40, 16))}
            load = lambda numbers: frames[[lookup[index] for index in numbers]]
            runtime = NestedClipRuntime(model, 40, load, chunk_size=4)
            first = runtime.predict(1.0, 1)
            second = runtime.predict(1.0, 2)
            direct = direct_predict(model, 40, load, 1.0, 2, chunk_size=4)
            changed_width = runtime.predict(0.5, 2)
            changed_direct = direct_predict(model, 40, load, 0.5, 2, chunk_size=4)
            return_to_first_width = runtime.predict(1.0, 2)
            self.assertEqual((first.new_frames, second.new_frames), (1, 1))
            self.assertLess(abs(second.logit - direct), 1e-5)
            self.assertEqual(changed_width.new_frames, 2)
            self.assertLess(abs(changed_width.logit - changed_direct), 1e-5)
            self.assertEqual(return_to_first_width.new_frames, 0)
            self.assertLess(abs(return_to_first_width.logit - direct), 1e-5)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
