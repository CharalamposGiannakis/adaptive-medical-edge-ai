import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import unittest

import torch

from experiments.F003.adaptive import frame_indices
from experiments.F004.family import OPERATING_POINTS, AnytimeClip
from experiments.F005.compress import build, quantized_linear_count, serialized_bytes
from experiments.F005.run_f005 import encode_per_frame


class QuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(2026)
        cls.fp32, cls.int8 = build("fp32"), build("int8_dynamic")
        cls.clip = torch.randn(40, 3, 224, 224)

    def test_every_linear_is_int8_and_the_file_shrinks(self):
        self.assertEqual(quantized_linear_count(self.int8), (41, 0))
        self.assertEqual(quantized_linear_count(self.fp32), (0, 41))
        self.assertLess(serialized_bytes(self.int8), serialized_bytes(self.fp32) / 3)

    def test_fp32_scheme_is_the_f004_family(self):
        frames = self.clip[:4].unsqueeze(0)
        with torch.no_grad():
            per_frame = self.fp32.classify(encode_per_frame(self.fp32, self.clip[:4]).unsqueeze(0))
            self.assertLess(float((self.fp32(frames) - per_frame).abs().max()), 1e-4)

    def test_per_frame_int8_keeps_the_anytime_cache_exact(self):
        """A frame encoded alone gives the same INT8 features whatever else is in the request."""
        clip = AnytimeClip(self.int8, 40, lambda numbers: self.clip[numbers],
                           encode=lambda frames: encode_per_frame(self.int8, frames))
        with torch.no_grad():
            rising = [clip.predict(k) for k in OPERATING_POINTS]
            for k, (logit, _) in zip(OPERATING_POINTS, rising):
                chosen = list(frame_indices(40, k))
                direct = self.int8.classify(encode_per_frame(self.int8, self.clip[chosen]).unsqueeze(0))[0]
                self.assertTrue(torch.equal(logit, direct))
        self.assertEqual([n for _, n in rising], [1, 1, 2, 4, 4, 4])


class NestedPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torch import nn
        from experiments.F004.family import AdaptiveTinyViT, load_family
        from experiments.F005.prune import groups, l1_scores, make_nested, recalibrate

        class Holder(nn.Module):
            def __init__(self, encoder, head):
                super().__init__()
                self.encoder, self.head = encoder, head
        torch.manual_seed(2026)
        cls.Holder, cls.Adaptive = Holder, AdaptiveTinyViT
        cls.reference = load_family()
        cls.model = load_family()
        orders = {n: torch.argsort(l1_scores(m), descending=True) for n, m in groups(cls.model.encoder)}
        make_nested(cls.model.encoder, orders)
        cls.frames = torch.randn(1, 4, 3, 224, 224)
        recalibrate(cls.model.encoder, [torch.randn(6, 3, 224, 224)])

    def test_reordering_and_recalibration_keep_full_width_exact(self):
        with torch.no_grad():
            difference = (self.model(self.frames) - self.reference(self.frames)).abs().max()
        self.assertLess(float(difference), 1e-5)

    def test_physical_copies_compute_what_the_sliced_runtime_computes(self):
        from experiments.F005.prune import WIDTHS, physical, set_width
        for width in WIDTHS:
            set_width(self.model.encoder, width)
            with torch.no_grad():
                sliced = self.model(self.frames)
                copy = self.Adaptive(self.Holder(physical(self.model.encoder, width), self.model.head)).eval()
                self.assertLess(float((copy(self.frames) - sliced).abs().max()), 1e-5)
        set_width(self.model.encoder, 1.0)

    def test_smaller_widths_use_a_prefix_of_the_larger_widths_weights(self):
        from experiments.F005.prune import channels, groups, physical
        wide, narrow = physical(self.model.encoder, 0.75), physical(self.model.encoder, 0.25)
        for (_, w), (_, n), (_, source) in zip(groups(wide), groups(narrow), groups(self.model.encoder)):
            k = channels(source, 0.25)
            first = (w.fc1.weight if hasattr(w, "fc1") else w.conv1.conv.weight)[:k]
            small = n.fc1.weight if hasattr(n, "fc1") else n.conv1.conv.weight
            self.assertTrue(torch.equal(first, small))


class FullBNAndNestedAnytimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torch import nn
        from experiments.F004.family import AdaptiveTinyViT, load_family
        from experiments.F005.prune import (groups, l1_scores, make_all_bn_width_aware, make_nested,
                                            recalibrate, recalibrate_all)

        class Holder(nn.Module):
            def __init__(self, encoder, head):
                super().__init__()
                self.encoder, self.head = encoder, head
        torch.manual_seed(7)
        cls.Holder, cls.Adaptive = Holder, AdaptiveTinyViT
        cls.reference = load_family()
        cls.model = load_family()
        make_nested(cls.model.encoder, {n: torch.argsort(l1_scores(m), descending=True)
                                        for n, m in groups(cls.model.encoder)})
        calibration = [torch.randn(4, 3, 224, 224)]
        recalibrate(cls.model.encoder, calibration)
        make_all_bn_width_aware(cls.model.encoder)
        cls.report = recalibrate_all(cls.model.encoder, calibration)
        cls.clip = torch.randn(40, 3, 224, 224)

    def test_full_recalibration_covers_downstream_bns_and_keeps_full_width_exact(self):
        self.assertEqual(len(self.report), 19)  # 10 local-conv BNs + 9 patch-merging BNs
        frames = self.clip[:4].unsqueeze(0)
        with torch.no_grad():
            self.assertLess(float((self.model(frames) - self.reference(frames)).abs().max()), 1e-5)

    def test_physical_copies_match_the_sliced_runtime_after_full_recalibration(self):
        from experiments.F005.prune import WIDTHS, physical, set_width
        frames = self.clip[:4].unsqueeze(0)
        for width in WIDTHS:
            set_width(self.model.encoder, width)
            with torch.no_grad():
                sliced = self.model(frames)
                copy = self.Adaptive(self.Holder(physical(self.model.encoder, width), self.model.head)).eval()
                self.assertLess(float((copy(frames) - sliced).abs().max()), 1e-5)
        set_width(self.model.encoder, 1.0)

    def test_width_change_mid_clip_matches_direct_inference(self):
        from experiments.F003.adaptive import frame_indices
        from experiments.F005.prune import NestedAnytimeClip, set_width
        clip = NestedAnytimeClip(self.model, 40, lambda numbers: self.clip[numbers])
        steps = [(1.0, 4), (0.5, 8), (0.5, 16), (1.0, 8), (0.25, 1)]
        with torch.no_grad():
            served = [clip.predict(w, k) for w, k in steps]
            for (width, k), (logit, _) in zip(steps, served):
                set_width(self.model.encoder, width)
                direct = self.model(self.clip[list(frame_indices(40, k))].unsqueeze(0))[0]
                self.assertLess(float((logit - direct).abs()), 1e-5)
        set_width(self.model.encoder, 1.0)
        self.assertEqual([n for _, n in served], [4, 8, 8, 4, 1])


class CacheTests(unittest.TestCase):
    def test_cached_frames_reproduce_f002c_preprocessing_exactly(self):
        from src import f002_pipeline as pipe
        from experiments.F002c.model import F002cSequences
        from experiments.F005 import cache
        rows = pipe.load_inputs()["validation"][:2]
        built = cache.build(rows)
        sequences = F002cSequences(rows, 16)
        for i in range(len(rows)):
            self.assertTrue(torch.equal(cache.normalize(built["frames"][i]), sequences.frames(i)))


class TrainingPieceTests(unittest.TestCase):
    def test_train_subsets_are_disjoint(self):
        from experiments.F005.run_f005b import train_subsets
        subsets = train_subsets([{}] * 1232)
        seen = [i for indices in subsets.values() for i in indices]
        self.assertEqual(len(seen), len(set(seen)))

    def test_distillation_loss_is_smallest_when_student_matches_teacher(self):
        from experiments.F005.run_f005b import kd_loss
        teacher, target = torch.tensor([2.0]), torch.tensor([1.0])
        values = {s: float(kd_loss(torch.tensor([s]), teacher, target)) for s in (0.0, 2.0, 6.0)}
        self.assertLess(values[2.0], values[0.0])


if __name__ == "__main__":
    unittest.main()
