import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import unittest

import torch

from src.f002_protocol import nested_frame_indices
from experiments.F003.adaptive import (ExitSequenceModel, count_flops, frame_indices, load_backbone,
                                       positions_in_k16)
from experiments.F004.family import (OPERATING_POINTS, AdaptiveTinyViT, AnytimeClip, Upgrade,
                                     read_parameters)
from experiments.F004.run_f004 import (ADOPTION_RULE, TRAIN_KS, adoption, better, k16_loader,
                                       per_k_logits, summarize_epoch)


class FamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backbone = load_backbone()
        cls.model = AdaptiveTinyViT(cls.backbone).eval()

    def test_every_point_computes_what_f002c_and_f003_compute(self):
        frames = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            for k in OPERATING_POINTS:
                expected = ExitSequenceModel(self.backbone)(frames[:, :k])
                self.assertTrue(torch.equal(self.model(frames[:, :k]), expected))
                if k in (4, 8, 16):
                    self.assertTrue(torch.equal(self.model(frames[:, :k]), self.backbone(frames[:, :k])))

    def test_rejects_k_outside_the_operating_points(self):
        for k in (3, 6):
            with self.assertRaises(ValueError):
                self.model(torch.zeros(1, k, 3, 224, 224))

    def test_anytime_reuses_frames_and_matches_direct_inference(self):
        frame_count = 40
        clip_frames = torch.randn(frame_count, 3, 224, 224)
        clip = AnytimeClip(self.model, frame_count, lambda numbers: clip_frames[numbers])
        with torch.no_grad():
            direct = {k: self.model(clip_frames[list(frame_indices(frame_count, k))].unsqueeze(0))[0]
                      for k in OPERATING_POINTS}
            rising = [clip.predict(k) for k in OPERATING_POINTS]
            falling = [clip.predict(k) for k in reversed(OPERATING_POINTS)]
        self.assertEqual([n for _, n in rising], [1, 1, 2, 4, 4, 4])
        self.assertEqual([n for _, n in falling], [0] * len(OPERATING_POINTS))
        for k, (logit, _) in zip(OPERATING_POINTS, rising):
            self.assertLess(float((logit - direct[k]).abs()), 1e-5)
        for k, (logit, _) in zip(reversed(OPERATING_POINTS), falling):
            self.assertLess(float((logit - direct[k]).abs()), 1e-5)

    def test_evaluate_loader_maps_clip_frames_to_preloaded_k16_frames(self):
        frame_count = 57
        selection = list(nested_frame_indices(frame_count, 16))
        frames16 = torch.randn(16, 3, 224, 224)
        clip = AnytimeClip(self.model, frame_count, k16_loader(frames16, selection))
        with torch.no_grad():
            for k in OPERATING_POINTS:
                logit, _ = clip.predict(k)
                direct = self.model(frames16[positions_in_k16(frame_count, k)].unsqueeze(0))[0]
                self.assertLess(float((logit - direct).abs()), 1e-5)

    def test_every_point_reads_every_parameter(self):
        everything = dict(self.model.named_parameters())
        for k in OPERATING_POINTS:
            used = read_parameters(self.model, k)
            self.assertEqual(used.keys(), everything.keys())
            self.assertTrue(all(used[n] is everything[n] for n in used))

    def test_upgrade_costs_only_the_added_frames(self):
        flops = {k: count_flops(self.model, k)["flops"] for k in (1, 2, 4)}
        for small, large in ((1, 2), (2, 4)):
            upgrade = count_flops(Upgrade(self.model, small), large - small)["flops"]
            pool_and_head = upgrade - (flops[large] - flops[small])
            # Pool over the K features and the 320-input head; the cached frames are not re-encoded.
            self.assertGreater(pool_and_head, 0)
            self.assertLess(pool_and_head, large * 320 + 2 * 320)

    def test_training_path_matches_the_family_on_real_positions(self):
        counts = (40, 143)
        frames = torch.randn(2, 16, 3, 224, 224)
        positions = [{k: positions_in_k16(n, k) for k in OPERATING_POINTS} for n in counts]
        with torch.no_grad():
            logits = per_k_logits(self.backbone.eval(), frames, positions, OPERATING_POINTS)
            for k in OPERATING_POINTS:
                for b, n in enumerate(counts):
                    direct = self.model(frames[b, positions_in_k16(n, k)].unsqueeze(0))[0]
                    self.assertLess(float((logits[k][b] - direct).abs()), 1e-5)


class SelectionTests(unittest.TestCase):
    def test_mean_f1_then_loss_then_earlier_epoch(self):
        best = {"mean_macro_f1": 0.8, "mean_loss": 0.3}
        self.assertTrue(better({"mean_macro_f1": 0.81, "mean_loss": 0.9}, best))
        self.assertTrue(better({"mean_macro_f1": 0.8, "mean_loss": 0.2}, best))
        self.assertFalse(better({"mean_macro_f1": 0.8, "mean_loss": 0.3}, best))
        self.assertFalse(better({"mean_macro_f1": 0.79, "mean_loss": 0.1}, best))

    def test_selection_averages_only_the_prespecified_k(self):
        self.assertEqual(TRAIN_KS, (1, 2, 4, 8, 16))
        evaluations = {k: {"duplicate_aware": {"macro_f1": 0.0 if k == 12 else 0.8, "loss": 0.3}}
                       for k in OPERATING_POINTS}
        summary = summarize_epoch(evaluations)
        self.assertAlmostEqual(summary["mean_macro_f1"], 0.8)
        self.assertIn("12", summary["per_k"])

    def test_adoption_rule(self):
        gain = ADOPTION_RULE["min_mean_f1_gain_over_d026"]
        base = {"d026": 0.85, "k16": 0.85}
        self.assertTrue(adoption(base | {"multik": 0.85 + gain}, {"d026": 0.87, "multik": 0.865})["propose_multik"])
        self.assertFalse(adoption(base | {"multik": 0.855}, {"d026": 0.87, "multik": 0.87})["propose_multik"])
        self.assertFalse(adoption(base | {"multik": 0.87}, {"d026": 0.87, "multik": 0.85})["propose_multik"])
        self.assertFalse(adoption({"d026": 0.85, "k16": 0.865, "multik": 0.87},
                                  {"d026": 0.87, "multik": 0.87})["propose_multik"])


if __name__ == "__main__":
    unittest.main()
