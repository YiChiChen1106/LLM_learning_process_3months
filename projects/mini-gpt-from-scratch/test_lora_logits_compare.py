import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from compare_lora_logits import compare_lora_logits
from lora_checkpoint import save_lora_adapter, save_merged_checkpoint
from lora_minigpt import merge_qkv_lora, replace_qkv_with_lora
from model import MiniGPT


class MiniGPTLoRALogitsCompareTest(unittest.TestCase):
    def make_base_model(self) -> MiniGPT:
        torch.manual_seed(0)
        return MiniGPT(
            vocab_size=50,
            block_size=16,
            dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )

    def make_base_checkpoint(self, model: MiniGPT) -> dict:
        return {
            "model": model.state_dict(),
            "stoi": {"l": 1, "a": 2, "r": 3, "g": 4, "e": 5},
            "itos": {1: "l", 2: "a", 3: "r", 4: "g", 5: "e"},
            "config": {
                "vocab_size": 50,
                "block_size": 16,
                "dim": 128,
                "num_heads": 4,
                "num_layers": 4,
                "dropout": 0.0,
            },
        }

    def train_lora_one_step(self, model: MiniGPT) -> None:
        torch.manual_seed(1)
        input_ids = torch.randint(0, 50, (2, 8))
        targets = torch.randint(0, 50, (2, 8))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=3e-4,
        )

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def test_base_plus_adapter_and_merged_checkpoint_have_same_logits(self):
        base_model = self.make_base_model()
        lora_model = self.make_base_model()
        lora_model.load_state_dict(base_model.state_dict())
        lora_model = replace_qkv_with_lora(lora_model, r=8, alpha=16)
        self.train_lora_one_step(lora_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "base.pt"
            adapter_path = Path(tmpdir) / "adapter.pt"
            merged_path = Path(tmpdir) / "merged.pt"
            base_checkpoint = self.make_base_checkpoint(base_model)
            torch.save(base_checkpoint, checkpoint_path)
            save_lora_adapter(lora_model, adapter_path)
            save_merged_checkpoint(
                merge_qkv_lora(lora_model),
                base_checkpoint,
                merged_path,
            )

            result = compare_lora_logits(
                checkpoint_path=checkpoint_path,
                lora_adapter_path=adapter_path,
                merged_checkpoint_path=merged_path,
                prompt="large",
                lora_rank=8,
                lora_alpha=16,
                atol=1e-6,
                device="cpu",
            )

        self.assertEqual(result.adapter_logits_shape, result.merged_logits_shape)
        self.assertLessEqual(result.max_abs_diff, 1e-6)
        self.assertTrue(result.allclose)


if __name__ == "__main__":
    unittest.main()
