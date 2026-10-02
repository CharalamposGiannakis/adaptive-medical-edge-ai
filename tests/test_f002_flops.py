import unittest

from src.f002_flops import build_sequence_model, count_sequence_flops


class FlopsCoverageTests(unittest.TestCase):
    def test_all_candidates_and_k_values_have_complete_count(self):
        for candidate in ("mobilenet_v2", "resnet18", "deit_tiny"):
            model = build_sequence_model(candidate, pretrained=False)
            previous = 0
            for k in (4, 8, 16):
                result = count_sequence_flops(model, k)
                self.assertEqual(result["unsupported"], {})
                self.assertGreater(result["flops"], previous)
                self.assertGreater(result["by_operator"]["mean_pool"], 0)
                if candidate == "deit_tiny":
                    self.assertGreater(result["by_operator"]["matmul"], 0)
                    self.assertGreater(result["by_operator"]["softmax"], 0)
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
                pooled = frames.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
                return self.head(torch.sin(pooled))

        with self.assertRaisesRegex(RuntimeError, "Uncounted F002 operators"):
            count_sequence_flops(Unsupported().eval(), 4)

if __name__ == "__main__":
    unittest.main()
