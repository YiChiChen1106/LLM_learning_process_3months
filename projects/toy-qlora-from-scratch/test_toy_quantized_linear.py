import unittest

import torch
import torch.nn as nn

from toy_quantized_linear import QLoRALinear, QuantizedLinear, merge_qlora_linear


class QuantizedLinearTest(unittest.TestCase):
    def test_quantizes_weights_into_int8_buffers(self):
        base = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            base.weight.copy_(
                torch.tensor(
                    [[1.0, -0.5], [0.25, -1.0]],
                    dtype=torch.float32,
                )
            )
            base.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))

        layer = QuantizedLinear(base)

        self.assertEqual(layer.qweight.dtype, torch.int8)
        self.assertEqual(layer.scale.dtype, torch.float32)
        self.assertAlmostEqual(layer.scale.item(), 1.0 / 127.0, places=7)
        self.assertTrue(
            torch.equal(
                layer.qweight,
                torch.tensor([[127, -64], [32, -127]], dtype=torch.int8),
            )
        )

    def test_keeps_no_trainable_parameters(self):
        base = nn.Linear(2, 2, bias=True)

        layer = QuantizedLinear(base)

        self.assertEqual(sum(p.numel() for p in layer.parameters()), 0)

    def test_state_dict_keeps_quantized_weight_scale_and_bias(self):
        base = nn.Linear(2, 2, bias=True)

        layer = QuantizedLinear(base)
        state_dict = layer.state_dict()

        self.assertEqual(set(state_dict.keys()), {"qweight", "scale", "bias"})

    def test_forward_matches_base_shape_and_is_close(self):
        base = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            base.weight.copy_(
                torch.tensor(
                    [[1.0, -0.5], [0.25, -1.0]],
                    dtype=torch.float32,
                )
            )
            base.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))

        layer = QuantizedLinear(base)
        x = torch.tensor([[1.5, -2.0], [0.0, 1.0]], dtype=torch.float32)

        quant_out = layer(x)
        base_out = base(x)

        self.assertEqual(tuple(quant_out.shape), (2, 2))
        self.assertTrue(torch.allclose(quant_out, base_out, atol=0.02, rtol=0.0))


class QLoRALinearTest(unittest.TestCase):
    def test_keeps_quantized_base_as_buffers_and_lora_as_parameters(self):
        base = nn.Linear(128, 384, bias=True)

        layer = QLoRALinear(base, r=8, alpha=16)

        self.assertEqual(layer.base.qweight.dtype, torch.int8)
        self.assertIn("base.qweight", layer.state_dict())
        self.assertIn("base.scale", layer.state_dict())
        self.assertIn("base.bias", layer.state_dict())
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 4_096)
        self.assertTrue(layer.lora_A.weight.requires_grad)
        self.assertTrue(layer.lora_B.weight.requires_grad)

    def test_zero_initialized_lora_b_starts_as_quantized_base(self):
        torch.manual_seed(0)
        base = nn.Linear(128, 384, bias=True)
        layer = QLoRALinear(base, r=8, alpha=16)
        x = torch.randn(32, 128)

        max_diff = (layer(x) - layer.base(x)).abs().max().item()

        self.assertEqual(max_diff, 0.0)

    def test_training_step_updates_lora_b_not_quantized_base(self):
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=True)
        layer = QLoRALinear(base, r=2, alpha=4)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(5, 4)

        qweight_before = layer.base.qweight.clone()
        lora_b_before = layer.lora_B.weight.detach().clone()

        loss = layer(x).sum()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.equal(layer.base.qweight, qweight_before))
        self.assertFalse(torch.equal(layer.lora_B.weight, lora_b_before))

    def test_merge_returns_plain_linear_with_equivalent_output(self):
        torch.manual_seed(0)
        base = nn.Linear(4, 3, bias=True)
        layer = QLoRALinear(base, r=2, alpha=4)
        with torch.no_grad():
            layer.lora_B.weight.normal_(mean=0.0, std=0.02)
        x = torch.randn(5, 4)

        merged = merge_qlora_linear(layer)

        self.assertIsInstance(merged, nn.Linear)
        self.assertNotIsInstance(merged, QLoRALinear)
        self.assertTrue(torch.allclose(layer(x), merged(x), atol=1e-6, rtol=0.0))

    def test_merged_linear_state_dict_has_only_plain_linear_weights(self):
        base = nn.Linear(4, 3, bias=True)
        layer = QLoRALinear(base, r=2, alpha=4)

        merged = merge_qlora_linear(layer)

        self.assertEqual(set(merged.state_dict().keys()), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main()
