import unittest

import torch
import torch.nn.functional as F

from lora_minigpt import replace_qkv_with_lora
from model import MiniGPT


class MiniGPTLoRATrainingSmokeTest(unittest.TestCase):
    def make_model(self) -> MiniGPT:
        torch.manual_seed(0)
        model = MiniGPT(
            vocab_size=50,
            block_size=16,
            dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )
        return replace_qkv_with_lora(model, r=8, alpha=16)

    def make_loss(self, model: MiniGPT) -> torch.Tensor:
        input_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))
        logits = model(input_ids)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

    def make_optimizer(self, model: MiniGPT) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=3e-4,
        )

    def test_optimizer_contains_only_trainable_lora_parameters(self):
        model = self.make_model()
        optimizer = self.make_optimizer(model)

        optimizer_param_count = sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        trainable_param_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        self.assertEqual(optimizer_param_count, trainable_param_count)
        self.assertEqual(trainable_param_count, 16_384)

    def test_backward_creates_gradients_only_for_lora_parameters(self):
        model = self.make_model()
        loss = self.make_loss(model)

        loss.backward()

        for name, parameter in model.named_parameters():
            if "lora_" in name:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_training_step_keeps_base_qkv_frozen_and_updates_lora_b(self):
        model = self.make_model()
        optimizer = self.make_optimizer(model)
        base_qkv_before = model.blocks[0].attn.qkv.linear.weight.detach().clone()
        lora_b_before = model.blocks[0].attn.qkv.lora_B.weight.detach().clone()

        loss = self.make_loss(model)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        base_qkv_after = model.blocks[0].attn.qkv.linear.weight.detach()
        lora_b_after = model.blocks[0].attn.qkv.lora_B.weight.detach()

        self.assertTrue(torch.equal(base_qkv_before, base_qkv_after))
        self.assertFalse(torch.equal(lora_b_before, lora_b_after))


if __name__ == "__main__":
    unittest.main()
