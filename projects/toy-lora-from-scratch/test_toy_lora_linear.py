import unittest

import torch

from toy_lora_linear import LoRALinear, count_parameters


class LoRALinearTest(unittest.TestCase):
    def test_counts_full_linear_parameters(self):
        layer = torch.nn.Linear(128, 384, bias=True)

        total, trainable = count_parameters(layer)

        self.assertEqual(total, 49_536)
        self.assertEqual(trainable, 49_536)

    def test_counts_lora_trainable_parameters(self):
        layer = LoRALinear(128, 384, r=8, alpha=16, bias=True)

        total, trainable = count_parameters(layer)

        self.assertEqual(total, 53_632)
        self.assertEqual(trainable, 4_096)

    def test_freezes_only_base_linear(self):
        layer = LoRALinear(128, 384, r=8, alpha=16, bias=True)

        self.assertFalse(layer.linear.weight.requires_grad)
        self.assertFalse(layer.linear.bias.requires_grad)
        self.assertTrue(layer.lora_A.weight.requires_grad)
        self.assertTrue(layer.lora_B.weight.requires_grad)

    def test_forward_shape_matches_base_linear(self):
        layer = LoRALinear(128, 384, r=8, alpha=16, bias=True)
        x = torch.randn(32, 128)

        out = layer(x)

        self.assertEqual(tuple(out.shape), (32, 384))

    def test_zero_initialized_lora_b_starts_as_base_linear(self):
        torch.manual_seed(0)
        layer = LoRALinear(128, 384, r=8, alpha=16, bias=True)
        x = torch.randn(32, 128)

        max_diff = (layer(x) - layer.linear(x)).abs().max().item()

        self.assertEqual(max_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
