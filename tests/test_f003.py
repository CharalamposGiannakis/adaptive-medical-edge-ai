import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import unittest

import torch

from src import f002_pipeline as pipe
from src.f002_protocol import binary_metrics, nested_frame_indices
from experiments.F003.adaptive import (EXITS, EXIT_DIMS, K_VALUES, ExitSequenceModel,
                                       F003Sequences, Trunk, _scalar_only, count_flops,
                                       dense_indices, fit_exit_head, frame_indices,
                                       load_backbone, positions_in_k16, window_indices)
from experiments.F003.run_f003 import (auroc, cascade, count_sequence_flops_any_k, cross_validated_cascade, device_table,
                                       group_folds, mcnemar, paired_auroc_delta, pareto_front,
                                       paired_delta, temporal_configs)


class SamplerTests(unittest.TestCase):
    def test_nested_and_equal_to_frozen_sampler(self):
        for n in range(16, 400):
            chosen = {k: frame_indices(n, k) for k in K_VALUES}
            for k in (4, 8, 16):
                self.assertEqual(chosen[k], nested_frame_indices(n, k))
            for small, large in zip(K_VALUES, K_VALUES[1:]):
                self.assertLess(set(chosen[small]), set(chosen[large]))
            for k in K_VALUES:
                self.assertEqual(len(set(chosen[k])), k)
                k16 = nested_frame_indices(n, 16)
                self.assertEqual([k16[p] for p in positions_in_k16(n, k)], list(chosen[k]))

    def test_temporal_windows_stay_inside_their_share_of_the_clip(self):
        for n in (16, 60, 143, 572, 1090):
            for k in (4, 8, 16):
                self.assertEqual(window_indices(n, 1.0, k), frame_indices(n, k))
                self.assertEqual(dense_indices(n, k), tuple(range(k)))
                for w in (0.25, 0.5, 0.75):
                    chosen = window_indices(n, w, k)
                    self.assertEqual(len(set(chosen)), k)
                    self.assertLess(max(chosen), max(16, -(-int(w * n * 1000) // 1000)))
            self.assertEqual(len(temporal_configs(n)), 15)

    def test_rejects_unknown_k(self):
        with self.assertRaises(ValueError):
            frame_indices(100, 3)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backbone = load_backbone()

    def test_trunk_and_final_exit_reproduce_f002c(self):
        frames = torch.randn(2, 4, 3, 224, 224)
        with torch.no_grad():
            expected = self.backbone(frames)
            got = ExitSequenceModel(self.backbone)(frames)
            trunk = Trunk(self.backbone.encoder)(frames[0])
        self.assertTrue(torch.equal(expected, got))
        self.assertEqual({k: v.shape[1] for k, v in trunk.items()}, EXIT_DIMS)

    def test_exits_share_backbone_weights(self):
        head = torch.nn.Linear(EXIT_DIMS["stage2"], 1)
        model = ExitSequenceModel(self.backbone, "stage2", head)
        self.assertIs(model.trunk.encoder.stages[1].blocks[0].attn.qkv.weight,
                      self.backbone.encoder.stages[1].blocks[0].attn.qkv.weight)

    def test_flops_linear_in_k_and_cheaper_exits(self):
        final = ExitSequenceModel(self.backbone)
        f1, f4 = count_flops(final, 1)["flops"], count_flops(final, 4)["flops"]
        self.assertAlmostEqual(f4 / f1, 4, delta=0.01)
        early = ExitSequenceModel(self.backbone, "stage1", torch.nn.Linear(128, 1))
        self.assertLess(count_flops(early, 1)["flops"], f1)
        self.assertLess(count_flops(final, 1, 160)["flops"], f1)

    def test_scalar_handler_fails_closed(self):
        class Fake:
            def __init__(self, sizes): self._sizes = sizes
            def type(self): return self
            def sizes(self): return self._sizes
        handler = _scalar_only("aten::rsub")
        self.assertEqual(sum(handler([], [Fake([])]).values()), 0)
        with self.assertRaises(RuntimeError):
            handler([], [Fake([4, 4])])


class ExitHeadTests(unittest.TestCase):
    def test_folded_head_matches_sklearn(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(0)
        x = rng.normal(3, 2, size=(80, 6))
        y = (x[:, 0] + rng.normal(size=80) > 3).astype(int)
        w = rng.uniform(0.5, 1, size=80)
        head, info = fit_exit_head(x, y, w)
        scaler = StandardScaler().fit(x, sample_weight=w)
        reference = LogisticRegression(C=1.0, max_iter=5000, random_state=2026).fit(
            scaler.transform(x), y, sample_weight=w)
        with torch.no_grad():
            got = torch.sigmoid(head(torch.tensor(x, dtype=torch.float32))).squeeze(-1).numpy()
        np.testing.assert_allclose(got, reference.predict_proba(scaler.transform(x))[:, 1], atol=1e-5)
        self.assertEqual(info["dim"], 6)


class AnalysisTests(unittest.TestCase):
    def test_cascade_endpoints(self):
        cheap = [{"unit_id": i, "binary_target": i % 2, "probability": 0.45} for i in range(10)]
        full = [{"unit_id": i, "binary_target": i % 2, "probability": 0.9 if i % 2 else 0.1}
                for i in range(10)]
        points = cascade(cheap, full, 100, 400, 100)
        self.assertEqual(points[0]["escalation_rate"], 0)
        self.assertEqual(points[0]["expected_flops"], 100)
        self.assertEqual(points[-1]["escalation_rate"], 1)
        self.assertEqual(points[-1]["expected_flops"], 400)
        self.assertEqual(points[-1]["macro_f1"], 1.0)

    def test_pareto_front_and_paired_delta(self):
        points = [{"config": "a", "macro_f1": 0.8, "flops": 1}, {"config": "b", "macro_f1": 0.7, "flops": 2},
                  {"config": "c", "macro_f1": 0.9, "flops": 3}]
        self.assertEqual(pareto_front(points), ["a", "c"])
        units = [{"unit_id": i, "binary_target": i % 2, "probability": 0.8 if i % 2 else 0.2}
                 for i in range(20)]
        result = paired_delta(units, units, reps=50)
        self.assertEqual(result["delta"], 0)
        self.assertEqual(binary_metrics([u["binary_target"] for u in units],
                                        [u["probability"] for u in units])["macro_f1"], 1.0)


class CrossValidationTests(unittest.TestCase):
    def units(self):
        units = {f"u{i}": {"groups": {f"g{i // 2}"}, "devices": {str(i % 3)}} for i in range(40)}
        cheap = [{"unit_id": f"u{i}", "binary_target": i % 2,
                  "probability": (0.45 if i % 4 == 1 else (0.9 if i % 2 else 0.1))} for i in range(40)]
        full = [{"unit_id": f"u{i}", "binary_target": i % 2,
                 "probability": 0.9 if i % 2 else 0.1} for i in range(40)]
        return units, cheap, full

    def test_folds_keep_groups_together(self):
        units, _, _ = self.units()
        folds = group_folds(units)
        for i in range(0, 40, 2):
            self.assertEqual(folds[f"u{i}"], folds[f"u{i + 1}"])
        self.assertEqual(set(folds.values()), set(range(5)))

    def test_cv_cascade_escalates_only_uncertain_units(self):
        units, cheap, full = self.units()
        score = {u["unit_id"]: -abs(u["probability"] - 0.5) for u in cheap}
        result = cross_validated_cascade(cheap, full, score, group_folds(units), 100, 400, 100)
        self.assertEqual(result["held_out"]["macro_f1"], 1.0)
        self.assertAlmostEqual(result["escalation_rate"], 10 / 40)
        self.assertAlmostEqual(result["expected_flops"], 100 + 0.25 * 300)

    def test_device_table_splits_mixed_units(self):
        units, cheap, _ = self.units()
        units["u0"]["devices"] = {"0", "4"}
        table = device_table(cheap, units)
        self.assertEqual(table["mixed"]["units"], 1)
        self.assertEqual(sum(v["units"] for v in table.values()), 40)


class RobustnessTests(unittest.TestCase):
    def test_auroc_and_mcnemar_on_known_cases(self):
        perfect = [{"unit_id": i, "binary_target": i % 2, "probability": 0.9 if i % 2 else 0.1}
                   for i in range(20)]
        flipped = [dict(u, probability=1 - u["probability"]) if u["unit_id"] < 4 else u
                   for u in perfect]
        self.assertEqual(auroc(perfect), 1.0)
        self.assertLess(auroc(flipped), 1.0)
        self.assertGreater(paired_auroc_delta(perfect, flipped, reps=50)["delta"], 0)
        test = mcnemar(perfect, flipped)
        self.assertEqual((test["only_a_correct"], test["only_b_correct"]), (4, 0))
        self.assertAlmostEqual(test["exact_p"], 0.125)


class MobileNetFlopsTests(unittest.TestCase):
    def test_unrestricted_counter_matches_frozen_counter(self):
        from src.f002_flops import build_sequence_model, count_sequence_flops
        model = build_sequence_model("mobilenet_v2", pretrained=False)
        for k in (4, 8, 16):
            self.assertEqual(count_sequence_flops_any_k(model, k)["flops"],
                             count_sequence_flops(model, k)["flops"])
        self.assertAlmostEqual(count_sequence_flops_any_k(model, 4)["flops"] /
                               count_sequence_flops_any_k(model, 1)["flops"], 4, delta=0.01)


class LeakageTests(unittest.TestCase):
    def test_loader_refuses_test_rows(self):
        test_row = {"partition": "test", "exam_id": "x", "zone_id": "z1"}
        with self.assertRaises(ValueError):
            F003Sequences([test_row])


if __name__ == "__main__":
    unittest.main()
