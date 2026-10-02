import unittest

from src.f002_flops import (BOOKKEEPING_ALLOWLIST, build_sequence_model,
                            count_sequence_flops, verify_deit_fused_equivalence)


class FlopsCoverageTests(unittest.TestCase):
    def test_all_candidates_and_k_values_have_complete_count(self):
        for candidate in ("mobilenet_v2", "resnet18", "deit_tiny"):
            model = build_sequence_model(candidate, pretrained=False)
            previous = 0
            for k in (4, 8, 16):
                result = count_sequence_flops(model, k)
                self.assertEqual(result["unsupported"], {})
                self.assertEqual(result["unapproved_ignored"], {})
                self.assertLessEqual(set(result["allowed_bookkeeping"]), BOOKKEEPING_ALLOWLIST)
                self.assertGreater(result["flops"], previous)
                self.assertGreater(result["by_operator"]["mean_pool"], 0)
                self.assertEqual(result["binary_head_flops"], model.head.in_features)
                if candidate == "deit_tiny":
                    self.assertGreater(result["by_operator"]["matmul"], 0)
                    self.assertGreater(result["by_operator"]["softmax"], 0)
                if candidate == "resnet18":
                    self.assertGreater(result["by_operator"]["relu"], 0)
                previous = result["flops"]

    def test_bad_k_fails_closed(self):
        model = build_sequence_model("mobilenet_v2", pretrained=False)
        with self.assertRaises(ValueError):
            count_sequence_flops(model, 3)

    def test_unaccounted_operator_fails_closed(self):
        import torch

        class Unsupported(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(1, 1)

            def forward(self, frames):
                pooled = frames.flatten(1).mean(dim=1, keepdim=True)
                return self.head(torch.sin(pooled))

        with self.assertRaisesRegex(RuntimeError, "Uncounted F002 operators"):
            count_sequence_flops(Unsupported().eval(), 4)

    def test_fvcore_ignored_arithmetic_fails_closed(self):
        import torch

        class IgnoredArithmetic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(1, 1)

            def forward(self, frames):
                pooled = frames.flatten(1).mean(dim=1, keepdim=True)
                return self.head(torch.floor(pooled))

        with self.assertRaisesRegex(RuntimeError, "Unapproved fvcore-ignored F002 operator: aten::floor"):
            count_sequence_flops(IgnoredArithmetic().eval(), 4)

    def test_untrained_deit_fused_unfused_probe(self):
        import torch

        torch.manual_seed(42)
        model = build_sequence_model("deit_tiny", pretrained=False).eval()
        for k in (4, 8, 16):
            result = verify_deit_fused_equivalence(
                model, [torch.rand(1, k, 3, 224, 224)], expected_requests=1)
            self.assertTrue(result["equivalent"])
            self.assertLessEqual(result["max_abs_logit_difference"], 1e-5)
            self.assertTrue(all(not module.fused_attn for module in model.modules()
                                if hasattr(module, "fused_attn")))

if __name__ == "__main__":
    unittest.main()
