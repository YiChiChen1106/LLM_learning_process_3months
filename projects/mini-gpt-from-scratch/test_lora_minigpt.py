import unittest

import torch
import torch.nn.functional as F
from torch import nn

from lora_minigpt import LoRALinear, count_parameters, replace_qkv_with_lora
from model import MiniGPT


class MiniGPTLoRATest(unittest.TestCase):
    def make_model(self) -> MiniGPT:
        torch.manual_seed(0)
        return MiniGPT(
            vocab_size=50,
            block_size=16,
            dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )

    def test_replaces_only_attention_qkv_layers(self):
        model = replace_qkv_with_lora(self.make_model(), r=8, alpha=16)

        for block in model.blocks:
            self.assertIsInstance(block.attn.qkv, LoRALinear)
            self.assertIsInstance(block.attn.proj, nn.Linear)
            self.assertIsInstance(block.ffn[0], nn.Linear)
            self.assertIsInstance(block.ffn[2], nn.Linear)

    def test_trainable_parameters_are_only_lora_adapters(self):
        model = replace_qkv_with_lora(self.make_model(), r=8, alpha=16)

        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertEqual(len(trainable), 8)
        self.assertTrue(all("lora_A" in name or "lora_B" in name for name in trainable))

    def test_qkv_lora_parameter_counts_match_hand_calculation(self):
        model = replace_qkv_with_lora(self.make_model(), r=8, alpha=16)

        total, trainable = count_parameters(model)

        self.assertEqual(trainable, 16_384)
        self.assertEqual(total, count_parameters(self.make_model())[0] + 16_384)

    def test_initial_lora_model_matches_base_model_logits(self):
        base_model = self.make_model()
        lora_model = replace_qkv_with_lora(self.make_model(), r=8, alpha=16)
        input_ids = torch.randint(0, 50, (2, 8))

        base_model.eval()
        lora_model.eval()
        with torch.no_grad():
            max_diff = (
                base_model(input_ids) - lora_model(input_ids)
            ).abs().max().item()

        self.assertEqual(max_diff, 0.0)

    def test_forward_and_backward_smoke(self):
        model = replace_qkv_with_lora(self.make_model(), r=8, alpha=16)
        input_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        self.assertEqual(tuple(logits.shape), (2, 8, 50))
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if "lora_" in name
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if "lora_" not in name
            )
        )


if __name__ == "__main__":
    unittest.main()
